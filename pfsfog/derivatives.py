"""Derivatives of power spectrum multipoles w.r.t. EFT parameters.

Primary: JAX autodiff (exact, fast).
Validation: adaptive finite difference via numdifftools (optimal step selection).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numdifftools as nd

from ps_1loop_jax import PowerSpectrum1Loop

from .eft_params import NUISANCE_NAMES

# ---------------------------------------------------------------------------
# Mapping: Fisher param name → location in ps_1loop_jax params dict
# ---------------------------------------------------------------------------

# Each entry: (nested_key_path, sigma8_power)
# sigma8_power: 0 = no σ8 scaling, 1 = divide by σ8, 2 = divide by σ8²
_PARAM_MAP = {
    "b1_sigma8":    (("bias", "b1"),     1),
    "b2_sigma8sq":  (("bias", "b2"),     2),
    "bG2_sigma8sq": (("bias", "bG2"),    2),
    "bGamma3":      (("bias", "bGamma3"), 0),
    "c0":           (("ctr", "c0"),      0),
    "c2":           (("ctr", "c2"),      0),
    "c4":           (("ctr", "c4"),      0),
    "c_tilde":      (("ctr", "cfog"),    0),
    "c1":           None,   # c1 not in ps_1loop_jax (sub-leading; ignored)
    "Pshot":        (("stoch", "P_shot"), 0),
    "a0":           (("stoch", "a0"),    0),
    "a2":           (("stoch", "a2"),    0),
}


def _set_nested(d: dict, path: tuple[str, ...], value):
    """Set a value in a nested dict, making mutable copies along the path."""
    if len(path) == 1:
        d[path[0]] = value
        return
    if path[0] not in d:
        d[path[0]] = {}
    else:
        d[path[0]] = dict(d[path[0]])
    _set_nested(d[path[0]], path[1:], value)


def _get_nested(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


def _make_mutable(params: dict) -> dict:
    """Deep-copy the nested dict so JAX tracing can replace leaves."""
    out = dict(params)
    for k in ("bias", "ctr", "stoch"):
        if k in out:
            out[k] = dict(out[k])
    if "bias2" in out:
        out["bias2"] = dict(out["bias2"])
    if "ctr2" in out:
        out["ctr2"] = dict(out["ctr2"])
    return out


# ---------------------------------------------------------------------------
# JAX autodiff derivatives (primary)
# ---------------------------------------------------------------------------


def dPell_dtheta_autodiff(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    fiducial_params: dict,
    param_name: str,
    sigma8_z: float,
    ell: int,
) -> jnp.ndarray:
    """∂P_ℓ(k)/∂θ via JAX forward-mode autodiff.

    Parameters
    ----------
    ps_model : PowerSpectrum1Loop
    k : array, shape (Nk,)
    pk_data : dict with 'k', 'pk'
    fiducial_params : ps_1loop_jax params dict at fiducial
    param_name : Fisher parameter name (e.g. 'c_tilde')
    sigma8_z : σ8(z) at the target redshift
    ell : multipole order (0, 2, or 4)

    Returns
    -------
    dPl_dtheta : array, shape (Nk,)
    """
    info = _PARAM_MAP.get(param_name)
    if info is None:
        # c1 or unknown — return zeros
        return jnp.zeros_like(k)

    path, s8_power = info
    s8_factor = sigma8_z ** s8_power if s8_power > 0 else 1.0

    fid_val = _get_nested(fiducial_params, path)

    def _pk_ell_of_param(val):
        p = _make_mutable(fiducial_params)
        _set_nested(p, path, val)
        return ps_model.get_pk_ell(k, ell, pk_data, p)

    # dP_ell / d(raw_param) via forward-mode
    dpd_raw = jax.jacfwd(_pk_ell_of_param)(fid_val)

    # Chain rule: d(raw)/d(fisher) = 1/σ8^power
    return dpd_raw / s8_factor


def dPell_dtheta_autodiff_all(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    fiducial_params: dict,
    param_names: list[str],
    sigma8_z: float,
    ells: tuple[int, ...] = (0, 2, 4),
) -> dict[str, dict[int, jnp.ndarray]]:
    """Compute all derivatives for auto-power multipoles.

    Returns
    -------
    derivs : {param_name: {ell: dP_ell/dtheta(k)}}
    """
    derivs = {}
    for pn in param_names:
        derivs[pn] = {}
        for ell in ells:
            derivs[pn][ell] = dPell_dtheta_autodiff(
                ps_model, k, pk_data, fiducial_params, pn, sigma8_z, ell,
            )
    return derivs


# ---------------------------------------------------------------------------
# Cross-power derivatives
# ---------------------------------------------------------------------------

# For cross-spectra, the parameter may belong to tracer A (bias/ctr),
# tracer B (bias2/ctr2), or be shared (f, h — cosmo).

_CROSS_PATH_A = {
    "b1_sigma8":    (("bias", "b1"),     1),
    "b2_sigma8sq":  (("bias", "b2"),     2),
    "bG2_sigma8sq": (("bias", "bG2"),    2),
    "bGamma3":      (("bias", "bGamma3"), 0),
    "c0":           (("ctr", "c0"),      0),
    "c2":           (("ctr", "c2"),      0),
    "c4":           (("ctr", "c4"),      0),
    "c_tilde":      (("ctr", "cfog"),    0),
    "Pshot":        None,  # no stoch in cross
    "a0":           None,
    "a2":           None,
    "c1":           None,
}

_CROSS_PATH_B = {
    "b1_sigma8":    (("bias2", "b1"),     1),
    "b2_sigma8sq":  (("bias2", "b2"),     2),
    "bG2_sigma8sq": (("bias2", "bG2"),    2),
    "bGamma3":      (("bias2", "bGamma3"), 0),
    "c0":           (("ctr2", "c0"),      0),
    "c2":           (("ctr2", "c2"),      0),
    "c4":           (("ctr2", "c4"),      0),
    "c_tilde":      (("ctr2", "cfog"),    0),
    "Pshot":        None,
    "a0":           None,
    "a2":           None,
    "c1":           None,
}


def dPcross_dtheta_autodiff(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    fiducial_params: dict,
    param_name: str,
    sigma8_z: float,
    which_tracer: str,
    ell: int,
) -> jnp.ndarray:
    """∂P^{AB}_ℓ(k)/∂θ for cross-power multipoles.

    Parameters
    ----------
    which_tracer : 'A', 'B', or 'shared'
    """
    if which_tracer == "A":
        path_map = _CROSS_PATH_A
    elif which_tracer == "B":
        path_map = _CROSS_PATH_B
    else:
        raise NotImplementedError("Shared (cosmo) cross-derivatives not yet wired")

    info = path_map.get(param_name)
    if info is None:
        return jnp.zeros_like(k)

    path, s8_power = info
    s8_factor = sigma8_z ** s8_power if s8_power > 0 else 1.0
    fid_val = _get_nested(fiducial_params, path)

    def _pk_ell_of_param(val):
        p = _make_mutable(fiducial_params)
        _set_nested(p, path, val)
        return ps_model.get_pk_ell_pair(k, ell, pk_data, p, add_stochasticity=False)

    dpd_raw = jax.jacfwd(_pk_ell_of_param)(fid_val)
    return dpd_raw / s8_factor


# ---------------------------------------------------------------------------
# Five-point stencil (validation)
# ---------------------------------------------------------------------------


def _numdiff(func, x0: float) -> float:
    """Adaptive central-difference derivative via numdifftools."""
    return nd.Derivative(func, method="central", order=4)(x0)


# ---------------------------------------------------------------------------
# Cosmological parameter derivatives
# ---------------------------------------------------------------------------

# fσ8: f enters the params dict directly; chain rule gives
#   ∂P/∂(fσ8) = ∂P/∂f × 1/σ8
# Ωm, Mν: change P_lin shape → need to recompute P_lin.
#   Autodiff version: traces through cosmopower-jax emulator + ps_1loop_jax.
#   Finite-difference version: 5-point stencil (kept for validation).


def dPell_d_fsigma8(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    fiducial_params: dict,
    sigma8_z: float,
    ell: int,
) -> jnp.ndarray:
    """∂P_ℓ/∂(fσ8) via autodiff w.r.t. f, then chain rule."""
    fid_f = fiducial_params["f"]

    def _pk_of_f(f_val):
        p = _make_mutable(fiducial_params)
        p["f"] = f_val
        return ps_model.get_pk_ell(k, ell, pk_data, p)

    dpd_f = jax.jacfwd(_pk_of_f)(fid_f)  # ∂P/∂f
    return dpd_f / sigma8_z  # ∂f/∂(fσ8) = 1/σ8


def dPell_d_cosmo_autodiff(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pkdata_fn,        # from cosmo.make_plin_func() — returns pk_data dict
    f_fn,             # from cosmo.make_growth_rate_func()
    cosmo_dict: dict,  # {omega_b, omega_cdm, h, n_s, ln10_10_As, mnu}
    fiducial_params: dict,
    cosmo_param: str,
    z: float,
    sigma8_z: float,
    ell: int,
) -> jnp.ndarray:
    r"""∂P_ℓ/∂θ_cosmo via JAX autodiff through cosmopower-jax.

    Traces through: cosmo_dict → pkdata_fn → pk_data → ps_model.get_pk_ell.
    pk_data is on the emulator's native k-grid, matching the stencil version.
    """
    def _pk_ell_of_delta(delta):
        p = {kk: vv for kk, vv in cosmo_dict.items()}
        if cosmo_param == "Omegam":
            p["omega_cdm"] = p["omega_cdm"] + delta * p["h"] ** 2
        elif cosmo_param == "Mnu":
            p["mnu"] = p["mnu"] + delta
        else:
            return jnp.zeros_like(k)

        pk_data = pkdata_fn(z, p)  # on emulator's native k-grid
        f_new = f_fn(z, p)
        par = _make_mutable(fiducial_params)
        par["f"] = f_new
        return jnp.array(ps_model.get_pk_ell(k, ell, pk_data, par))

    return jax.jacfwd(_pk_ell_of_delta)(0.0)


def dPell_d_cosmo_stencil(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    cosmo,  # FiducialCosmology
    fiducial_params: dict,
    cosmo_param: str,
    z: float,
    sigma8_z: float,
    ell: int,
) -> jnp.ndarray:
    """∂P_ℓ/∂θ_cosmo via adaptive finite difference (numdifftools).

    For Ωm: perturb omega_cdm (holding omega_b fixed) → recompute P_lin, f.
    For Mν: perturb mnu → recompute P_lin, f.
    """
    from .cosmo import FiducialCosmology

    if cosmo_param == "Omegam":
        h = cosmo.params["h"]

        def _eval(delta):
            p = dict(cosmo.params)
            p["omega_cdm"] = p["omega_cdm"] + delta * h**2
            c2 = FiducialCosmology(p, backend=cosmo.backend)
            pd = c2.pk_data(z)
            f_new = float(c2.f(z))
            par = _make_mutable(fiducial_params)
            par["f"] = f_new
            return np.asarray(ps_model.get_pk_ell(k, ell, pd, par))

    elif cosmo_param == "Mnu":
        def _eval(delta):
            p = dict(cosmo.params)
            p["mnu"] = p["mnu"] + delta
            c2 = FiducialCosmology(p, backend=cosmo.backend)
            pd = c2.pk_data(z)
            f_new = float(c2.f(z))
            par = _make_mutable(fiducial_params)
            par["f"] = f_new
            return np.asarray(ps_model.get_pk_ell(k, ell, pd, par))
    else:
        return jnp.zeros_like(k)

    deriv_func = nd.Derivative(_eval, method="central", order=4)
    return jnp.array(deriv_func(0.0))


def dPell_d_cosmo_all(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    cosmo,
    fiducial_params: dict,
    z: float,
    sigma8_z: float,
    ells: tuple[int, ...] = (0, 2, 4),
    use_autodiff: bool = True,
) -> dict[str, dict[int, jnp.ndarray]]:
    """Compute derivatives w.r.t. all cosmological parameters.

    Parameters
    ----------
    use_autodiff : bool
        If True, use JAX autodiff through cosmopower-jax for Ωm and Mν.
        If False, fall back to 5-point finite difference.
    """
    derivs = {}

    derivs["fsigma8"] = {}
    for ell in ells:
        derivs["fsigma8"][ell] = dPell_d_fsigma8(
            ps_model, k, pk_data, fiducial_params, sigma8_z, ell,
        )

    if use_autodiff:
        from .cosmo import make_plin_func, make_growth_rate_func
        pkdata_fn = make_plin_func(cosmo.backend)
        f_fn = make_growth_rate_func()
        cosmo_dict = dict(cosmo.params)
        for cparam in ("Omegam", "Mnu"):
            derivs[cparam] = {}
            for ell in ells:
                derivs[cparam][ell] = dPell_d_cosmo_autodiff(
                    ps_model, k, pkdata_fn, f_fn, cosmo_dict,
                    fiducial_params, cparam, z, sigma8_z, ell,
                )
    else:
        for cparam in ("Omegam", "Mnu"):
            derivs[cparam] = {}
            for ell in ells:
                derivs[cparam][ell] = dPell_d_cosmo_stencil(
                    ps_model, k, cosmo, fiducial_params, cparam,
                    z, sigma8_z, ell,
                )

    return derivs


# ---------------------------------------------------------------------------
# Adaptive finite difference (validation via numdifftools)
# ---------------------------------------------------------------------------


def dPell_dtheta_stencil(
    ps_model: PowerSpectrum1Loop,
    k: jnp.ndarray,
    pk_data: dict,
    fiducial_params: dict,
    param_name: str,
    sigma8_z: float,
    ell: int,
) -> jnp.ndarray:
    """∂P_ℓ(k)/∂θ via adaptive finite difference (numdifftools)."""
    info = _PARAM_MAP.get(param_name)
    if info is None:
        return jnp.zeros_like(k)

    path, s8_power = info
    s8_factor = sigma8_z ** s8_power if s8_power > 0 else 1.0
    fid_val = float(_get_nested(fiducial_params, path))

    def _eval(val):
        p = _make_mutable(fiducial_params)
        _set_nested(p, path, float(val))
        return np.asarray(ps_model.get_pk_ell(k, ell, pk_data, p))

    # Use a relative step of 1% of the fiducial value, with a floor of 1e-3.
    # This handles parameters like c_tilde (fid=400) where the derivative
    # is small relative to the function value.
    step_hint = max(abs(fid_val) * 0.01, 1e-3)
    deriv_func = nd.Derivative(_eval, method="central", order=4, step=step_hint)
    dp = deriv_func(fid_val)

    return jnp.array(dp) / s8_factor
