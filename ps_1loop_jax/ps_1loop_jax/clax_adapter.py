from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .ps_1loop import PowerSpectrum1Loop


@dataclass(frozen=True)
class _ClaxLinearPowerResult:
    bg: Any
    th: Any
    pt: Any
    k_mpc: Any
    pk_mpc: Any


def _import_clax():
    try:
        import clax
        from clax import CosmoParams, PrecisionParams, background_solve, thermodynamics_solve
        from clax.perturbations import perturbations_solve
        from clax.primordial import primordial_scalar_pk
        from clax.transfer import compute_pk_from_perturbations
    except ImportError as exc:
        raise ImportError(
            "clax is required for ps_1loop_jax.clax_adapter but is not installed."
        ) from exc

    return {
        "clax": clax,
        "CosmoParams": CosmoParams,
        "PrecisionParams": PrecisionParams,
        "background_solve": background_solve,
        "thermodynamics_solve": thermodynamics_solve,
        "perturbations_solve": perturbations_solve,
        "primordial_scalar_pk": primordial_scalar_pk,
        "compute_pk_from_perturbations": compute_pk_from_perturbations,
    }


def _as_positive_1d(name, values):
    arr = jnp.atleast_1d(jnp.asarray(values, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if jnp.any(arr <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return arr


def _default_k_eval_h(k_eval=None, kmin=None, kmax=None, num=None):
    if k_eval is not None:
        return _as_positive_1d("k_eval", k_eval)

    if kmin is None:
        kmin = 1e-4
    if kmax is None:
        kmax = 1.0
    if num is None:
        num = 128
    if kmin <= 0 or kmax <= 0:
        raise ValueError("kmin and kmax must be positive.")
    if kmax <= kmin:
        raise ValueError("kmax must be strictly larger than kmin.")
    return jnp.geomspace(float(kmin), float(kmax), int(num))


def _default_pk_eval_grid_from_query(k_query_h):
    k_query_h = _as_positive_1d("k", k_query_h)
    kmin = max(1e-4, 0.5 * float(jnp.min(k_query_h)))
    kmax = max(1.0, 2.0 * float(jnp.max(k_query_h)))
    return jnp.geomspace(kmin, kmax, 128)


def _default_precision(k_eval_h, *, prec=None):
    if prec is not None:
        return prec

    clax_api = _import_clax()
    PrecisionParams = clax_api["PrecisionParams"]
    base = PrecisionParams.fast_cl()
    k_eval_h = _as_positive_1d("k_eval", k_eval_h)
    kmax_h = float(jnp.max(k_eval_h))

    # Keep the precision setup independent of traced cosmological parameters.
    # The solved k-range is in Mpc^-1, while the public interface uses h/Mpc.
    # Using max(..., 1.0) safely covers realistic h < 1 without baking params.h
    # into static precision arguments.
    return dataclass_replace(
        base,
        pt_k_min=1e-5,
        pt_k_max_cl=max(1.0, 1.25 * kmax_h),
    )


def _z_to_loga(z):
    z_arr = jnp.asarray(z, dtype=float)
    if jnp.any(z_arr < 0):
        raise ValueError("z must be non-negative.")
    return jnp.log(1.0 / (1.0 + z_arr))


def _solve_clax_background(clax_params, *, prec=None):
    clax_api = _import_clax()
    if prec is None:
        prec = _default_precision(jnp.asarray([1.0]))
    bg = clax_api["background_solve"](clax_params, prec)
    return bg


def _solve_clax_linear_matter(clax_params, z, *, prec=None, k_eval_h=None, method="perturbations"):
    k_eval_h = _default_k_eval_h(k_eval=k_eval_h)
    prec = _default_precision(k_eval_h, prec=prec)

    clax_api = _import_clax()
    k_eval_mpc = jnp.asarray(clax_params.h, dtype=float) * k_eval_h

    try:
        bg = clax_api["background_solve"](clax_params, prec)
    except Exception as exc:
        if getattr(clax_params, "N_ncdm", None) == 0 and "maximum number of solver steps" in str(exc):
            raise RuntimeError(
                "clax background_solve failed with N_ncdm=0 in this environment. "
                "Use clax.CosmoParams() or a configuration with N_ncdm>0 instead."
            ) from exc
        raise

    if method == "perturbations":
        th = clax_api["thermodynamics_solve"](clax_params, prec, bg)
        pt = clax_api["perturbations_solve"](clax_params, prec, bg, th)

        delta_m = clax_api["compute_pk_from_perturbations"](pt, bg, k_eval_mpc, z=z)
        primordial = clax_api["primordial_scalar_pk"](k_eval_mpc, clax_params)
        pk_mpc = 2.0 * jnp.pi**2 / k_eval_mpc**3 * primordial * delta_m**2
    elif method == "compute_pk":
        if float(z) != 0.0:
            raise ValueError("method='compute_pk' only supports z=0.0.")
        th = None
        pt = None
        pk_mpc = jax.vmap(
            lambda kval: clax_api["clax"].compute_pk(clax_params, prec, k=kval)
        )(k_eval_mpc)
    else:
        raise ValueError("method must be 'perturbations' or 'compute_pk'.")

    return _ClaxLinearPowerResult(
        bg=bg,
        th=th,
        pt=pt,
        k_mpc=k_eval_mpc,
        pk_mpc=pk_mpc,
    )


def make_clax_pk_data(
    clax_params,
    z,
    *,
    prec=None,
    k_eval=None,
    kmin=None,
    kmax=None,
    num=None,
    method="perturbations",
):
    """Build a ps_1loop_jax-compatible linear P(k) table from clax.

    Parameters
    ----------
    clax_params
        ``clax.CosmoParams`` instance.
    z : float
        Redshift at which to evaluate the linear matter power spectrum.
    prec
        Optional ``clax.PrecisionParams``. If omitted, a light-weight default
        is built from the requested ``k`` range.
    k_eval, kmin, kmax, num
        Output grid in units of ``h/Mpc``. Provide ``k_eval`` directly or let
        the helper build a logarithmic grid from ``kmin``/``kmax``.

    Returns
    -------
    dict
        Dictionary with keys ``"k"`` and ``"pk"`` in the units expected by
        ``PowerSpectrum1Loop``: ``k`` in ``h/Mpc`` and ``P(k)`` in ``(Mpc/h)^3``.
    """
    k_eval_h = _default_k_eval_h(k_eval=k_eval, kmin=kmin, kmax=kmax, num=num)
    result = _solve_clax_linear_matter(
        clax_params,
        z,
        prec=prec,
        k_eval_h=k_eval_h,
        method=method,
    )

    h = jnp.asarray(clax_params.h, dtype=float)
    return {
        "k": result.k_mpc / h,
        "pk": result.pk_mpc * h**3,
    }


def make_clax_background_data(
    clax_params,
    z,
    *,
    prec=None,
    growth_mode="clax",
    f_override=None,
):
    """Return growth quantities needed by ps_1loop_jax from clax background data."""
    if growth_mode != "clax":
        raise ValueError("growth_mode must be 'clax'.")

    bg = _solve_clax_background(clax_params, prec=prec)
    loga = _z_to_loga(z)

    growth = {
        "h": jnp.asarray(clax_params.h, dtype=float),
        "f": bg.f_of_loga.evaluate(loga),
        "D": bg.D_of_loga.evaluate(loga),
    }
    if f_override is not None:
        growth["f"] = jnp.asarray(f_override, dtype=float)
    return growth


def get_pk_ell_from_clax(
    clax_params,
    bias_survey_dict,
    k,
    ells,
    /,
    *,
    z,
    alpha_perp=1.0,
    alpha_para=1.0,
    clax_prec=None,
    ps1loop_model=None,
    growth_mode="clax",
    f_override=None,
    k_eval=None,
    pk_method="perturbations",
):
    """Evaluate 1-loop multipoles using clax linear power as input."""
    k = _as_positive_1d("k", k)
    if k_eval is None:
        k_eval = _default_pk_eval_grid_from_query(k)
    else:
        k_eval = _as_positive_1d("k_eval", k_eval)

    result = _solve_clax_linear_matter(
        clax_params,
        z,
        prec=clax_prec,
        k_eval_h=k_eval,
        method=pk_method,
    )
    h = jnp.asarray(clax_params.h, dtype=float)
    pk_data = {
        "k": result.k_mpc / h,
        "pk": result.pk_mpc * h**3,
    }

    if growth_mode != "clax":
        raise ValueError("growth_mode must be 'clax'.")
    loga = _z_to_loga(z)
    f_growth = result.bg.f_of_loga.evaluate(loga)
    if f_override is not None:
        f_growth = jnp.asarray(f_override, dtype=float)

    if ps1loop_model is None:
        ps1loop_model = PowerSpectrum1Loop(do_irres=True)

    params = {
        "h": h,
        "f": f_growth,
        **bias_survey_dict,
    }

    if alpha_perp == 1.0 and alpha_para == 1.0:
        pk_ell_list = [
            ps1loop_model.get_pk_ell(k, ell, pk_data, params)
            for ell in ells
        ]
    else:
        pk_ell_list = [
            ps1loop_model.get_pk_ell_ref(k, ell, alpha_perp, alpha_para, pk_data, params)
            for ell in ells
        ]

    return jnp.concatenate(pk_ell_list)
