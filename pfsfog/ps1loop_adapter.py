"""Adapter between Fisher parameter vectors and theory backends.

Maps the σ8-scaled EFT parameterization (Chudaykin+ 2025) to the
``params`` dict expected by ``ps_1loop_jax.PowerSpectrum1Loop`` and
to the flat arguments of ``clax.ept.pk_gg_l0/l2/l4``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .cosmo import FiducialCosmology
from .eft_params import EFTFiducials, NUISANCE_NAMES


# ---------------------------------------------------------------------------
# Fisher → ps_1loop_jax
# ---------------------------------------------------------------------------


def fisher_to_ps1loop_auto(
    fiducials: EFTFiducials,
    sigma8_z: float,
    f_z: float,
    h: float,
    nbar: float,
    k_nl: float = 0.7,
) -> dict:
    """Build a ps_1loop_jax ``params`` dict for an auto-spectrum.

    All bias quantities are un-scaled from σ8.
    """
    s8 = sigma8_z
    s82 = s8**2

    fid = fiducials.as_dict()

    bias = {
        "b1": fid["b1_sigma8"] / s8,
        "b2": fid["b2_sigma8sq"] / s82,
        "bG2": fid["bG2_sigma8sq"] / s82,
        "bGamma3": fid["bGamma3"],
    }
    ctr = {
        "c0": fid["c0"],
        "c2": fid["c2"],
        "c4": fid["c4"],
        "cfog": fid["c_tilde"],
    }
    stoch = {
        "P_shot": fid["Pshot"],
        "a0": fid["a0"],
        "a2": fid["a2"],
    }
    return {
        "h": h,
        "f": f_z,
        "bias": bias,
        "ctr": ctr,
        "stoch": stoch,
        "k_nl": k_nl,
        "ndens": nbar,
    }


def fisher_to_ps1loop_cross(
    fiducials_A: EFTFiducials,
    fiducials_B: EFTFiducials,
    sigma8_z: float,
    f_z: float,
    h: float,
    nbar_A: float,
    nbar_B: float,
    k_nl: float = 0.7,
) -> dict:
    """Build a ps_1loop_jax ``params`` dict for a cross-spectrum.

    Tracer A parameters go into ``bias``/``ctr``, tracer B into
    ``bias2``/``ctr2``.  No stochastic terms (cross-shot = 0).
    """
    s8 = sigma8_z
    s82 = s8**2

    fid_a = fiducials_A.as_dict()
    fid_b = fiducials_B.as_dict()

    bias = {
        "b1": fid_a["b1_sigma8"] / s8,
        "b2": fid_a["b2_sigma8sq"] / s82,
        "bG2": fid_a["bG2_sigma8sq"] / s82,
        "bGamma3": fid_a["bGamma3"],
    }
    ctr = {
        "c0": fid_a["c0"],
        "c2": fid_a["c2"],
        "c4": fid_a["c4"],
        "cfog": fid_a["c_tilde"],
    }
    bias2 = {
        "b1": fid_b["b1_sigma8"] / s8,
        "b2": fid_b["b2_sigma8sq"] / s82,
        "bG2": fid_b["bG2_sigma8sq"] / s82,
        "bGamma3": fid_b["bGamma3"],
    }
    ctr2 = {
        "c0": fid_b["c0"],
        "c2": fid_b["c2"],
        "c4": fid_b["c4"],
        "cfog": fid_b["c_tilde"],
    }
    return {
        "h": h,
        "f": f_z,
        "bias": bias,
        "ctr": ctr,
        "bias2": bias2,
        "ctr2": ctr2,
        "k_nl": k_nl,
        "ndens": nbar_A,  # not used for cross (stoch disabled)
    }


# ---------------------------------------------------------------------------
# Fisher → clax.ept
# ---------------------------------------------------------------------------


def fisher_to_ept(
    fiducials: EFTFiducials,
    sigma8_z: float,
    nbar: float,
) -> dict:
    """Convert Fisher-level parameters to clax.ept wrapper arguments.

    Returns a dict with keys matching ``pk_gg_l0/l2/l4`` signatures:
    ``{b1, b2, bG2, bGamma3, cs0, cs2, cs4, Pshot, b4}``.
    """
    s8 = sigma8_z
    s82 = s8**2
    fid = fiducials.as_dict()

    return {
        "b1": fid["b1_sigma8"] / s8,
        "b2": fid["b2_sigma8sq"] / s82,
        "bG2": fid["bG2_sigma8sq"] / s82,
        "bGamma3": fid["bGamma3"],
        "cs0": fid["c0"],
        "cs2": fid["c2"],
        "cs4": fid["c4"],
        "Pshot": fid["Pshot"] / nbar if nbar > 0 else 0.0,
        "b4": fid["a2"],  # clax.ept naming convention
    }


# ---------------------------------------------------------------------------
# P(k,μ) wrappers for covariance computation (Phase B)
# ---------------------------------------------------------------------------


def make_ps1loop_pkmu_func(ps_model, pk_data, params):
    """Return a P(k, μ) callable using ps_1loop_jax for covariance.

    The returned function evaluates the full 1-loop auto-power P(k,μ)
    including stochastic terms, consistent with the derivatives.

    Parameters
    ----------
    ps_model : PowerSpectrum1Loop
    pk_data : dict with 'k', 'pk'
    params : ps_1loop_jax params dict (from fisher_to_ps1loop_auto)

    Returns
    -------
    callable : (k_array, mu_array) → ndarray (Nk, Nmu)
    """
    import numpy as np

    def pkmu_func(k, mu):
        k_jnp = jnp.atleast_1d(jnp.asarray(k, dtype=float))
        # ps_1loop_jax uses μ ∈ [0,1]; P(k,μ) = P(k,−μ) by symmetry
        mu_jnp = jnp.abs(jnp.atleast_1d(jnp.asarray(mu, dtype=float)))
        return np.asarray(ps_model.get_pkmu(k_jnp, mu_jnp, pk_data, params))

    return pkmu_func


def make_ps1loop_pkmu_cross_func(ps_model, pk_data, params):
    """Return a P^{AB}(k, μ) callable for cross-spectrum covariance.

    No stochastic terms (cross-shot = 0).

    Parameters
    ----------
    ps_model : PowerSpectrum1Loop
    pk_data : dict with 'k', 'pk'
    params : ps_1loop_jax params dict with bias2/ctr2

    Returns
    -------
    callable : (k_array, mu_array) → ndarray (Nk, Nmu)
    """
    import numpy as np

    def pkmu_func(k, mu):
        k_jnp = jnp.atleast_1d(jnp.asarray(k, dtype=float))
        mu_jnp = jnp.abs(jnp.atleast_1d(jnp.asarray(mu, dtype=float)))
        return np.asarray(
            ps_model.get_pkmu_pair(k_jnp, mu_jnp, pk_data, params,
                                   add_stochasticity=False)
        )

    return pkmu_func


# ---------------------------------------------------------------------------
# Perturbed parameters for derivative computation
# ---------------------------------------------------------------------------


def perturb_fiducials(
    fiducials: EFTFiducials,
    param_name: str,
    delta: float,
) -> EFTFiducials:
    """Return a copy of *fiducials* with *param_name* shifted by *delta*."""
    d = fiducials.as_dict()
    if param_name not in d:
        raise KeyError(f"Unknown nuisance parameter: {param_name}")
    d[param_name] = d[param_name] + delta
    return EFTFiducials(**d)
