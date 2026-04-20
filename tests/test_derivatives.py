"""Tests for pfsfog.derivatives — autodiff vs stencil agreement."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ps_1loop_jax import PowerSpectrum1Loop

from pfsfog.eft_params import desi_elg_fiducials
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto
from pfsfog.derivatives import dPell_dtheta_autodiff, dPell_dtheta_stencil


@pytest.fixture(scope="module")
def setup():
    """Shared setup: ps model, pk_data, fiducial params."""
    ps = PowerSpectrum1Loop(do_irres=False)
    k_lin = jnp.geomspace(1e-4, 10.0, 256)
    pk_lin = 1e4 * (k_lin / 0.05) ** (-2.5)
    pk_data = {"k": k_lin, "pk": pk_lin}
    k = jnp.geomspace(0.01, 0.25, 20)

    b1 = 1.3
    s8 = 0.49
    fid_eft = desi_elg_fiducials(b1=b1, sigma8_z=s8)
    params = fisher_to_ps1loop_auto(fid_eft, s8, f_z=0.87, h=0.6736, nbar=4e-4)

    return ps, pk_data, k, params, s8


@pytest.mark.parametrize("param_name,ell", [
    ("b1_sigma8", 0), ("b1_sigma8", 2),
    ("c2", 0), ("c2", 2),
    ("c_tilde", 0),
])
def test_autodiff_vs_stencil(setup, param_name, ell):
    """Autodiff and 5-point stencil should agree to <10% on most k.

    Tests only parameters with non-degenerate derivatives.  Stochastic
    params (Pshot, a0, a2) enter as additive constants that don't flow
    through the 1-loop evaluation — tested separately.
    """
    ps, pk_data, k, params, s8 = setup

    step = None  # uses DEFAULT_STEPS from derivatives.py

    dp_ad = np.asarray(dPell_dtheta_autodiff(
        ps, k, pk_data, params, param_name, s8, ell,
    ))
    dp_st = np.asarray(dPell_dtheta_stencil(
        ps, k, pk_data, params, param_name, s8, ell, step=step,
    ))

    # Skip bins where both are near zero
    scale = np.maximum(np.abs(dp_ad), np.abs(dp_st))
    mask = scale > 1e-8 * np.max(scale)

    if not np.any(mask):
        pytest.skip("derivative is zero everywhere")

    rel_err = np.abs(dp_ad[mask] - dp_st[mask]) / scale[mask]
    median_err = np.median(rel_err)
    assert median_err < 0.10, (
        f"median rel error {median_err:.3f} for {param_name} ell={ell}"
    )


def test_stochastic_derivative_nonzero(setup):
    """Stochastic params should have non-zero monopole derivatives."""
    ps, pk_data, k, params, s8 = setup
    for pn in ("Pshot", "a0"):
        dp = np.asarray(dPell_dtheta_autodiff(
            ps, k, pk_data, params, pn, s8, 0,
        ))
        assert np.any(np.abs(dp) > 0), f"{pn} derivative is all zeros"
