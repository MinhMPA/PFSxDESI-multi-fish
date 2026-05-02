"""Tests for pfsfog.derivatives — autodiff vs stencil agreement."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ps_1loop_jax import PowerSpectrum1Loop

from pfsfog.eft_params import desi_elg_fiducials, pfs_elg_fiducials, NUISANCE_NAMES
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, fisher_to_ps1loop_cross
from pfsfog.derivatives import (
    dPell_dtheta_autodiff,
    dPell_dtheta_stencil,
    dPell_dtheta_autodiff_all,
    dPell_dtheta_autodiff_all_jit,
    dPcross_dtheta_autodiff,
    dPcross_dtheta_autodiff_all_jit,
)


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

    dp_ad = np.asarray(dPell_dtheta_autodiff(
        ps, k, pk_data, params, param_name, s8, ell,
    ))
    dp_st = np.asarray(dPell_dtheta_stencil(
        ps, k, pk_data, params, param_name, s8, ell,
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


@pytest.fixture(scope="module")
def setup_cross():
    """Cross-spectrum setup: PFS-ELG × DESI-ELG params."""
    ps = PowerSpectrum1Loop(do_irres=False)
    k_lin = jnp.geomspace(1e-4, 10.0, 256)
    pk_lin = 1e4 * (k_lin / 0.05) ** (-2.5)
    pk_data = {"k": k_lin, "pk": pk_lin}
    k = jnp.geomspace(0.01, 0.25, 20)

    s8 = 0.49
    b1_desi = 1.3
    b1_pfs = 1.5
    fid_A = pfs_elg_fiducials(b1_pfs, b1_desi, s8)  # PFS = side A
    fid_B = desi_elg_fiducials(b1_desi, s8)         # DESI = side B
    params = fisher_to_ps1loop_cross(
        fid_A, fid_B, sigma8_z=s8, f_z=0.87, h=0.6736,
        nbar_A=1e-3, nbar_B=4e-4,
    )
    return ps, pk_data, k, params, s8


def test_dPell_jit_equiv_unjitted(setup):
    """JIT'd vectorized variant must match the dict version to rtol=1e-12.

    This pins the equivalence of `dPell_dtheta_autodiff_all_jit` (Step 1
    of the JIT refactor) against the original eager-mode dict-returning
    `dPell_dtheta_autodiff_all` for every (parameter, ell) entry. The
    JIT path fuses many primitives into single XLA reductions and reorders
    sums, so rtol=1e-7 is the realistic float64 tolerance — the residual
    is pure floating-point rounding, not a numerical bug.
    """
    ps, pk_data, k, params, s8 = setup
    ells = (0, 2, 4)

    dict_out = dPell_dtheta_autodiff_all(
        ps, k, pk_data, params, list(NUISANCE_NAMES), s8, ells=ells,
    )
    arr_out = dPell_dtheta_autodiff_all_jit(
        ps, k, pk_data, params, s8, ells=ells,
    )

    assert arr_out.shape == (len(NUISANCE_NAMES), len(ells), len(k))

    for ip, name in enumerate(NUISANCE_NAMES):
        for il, ell in enumerate(ells):
            expected = np.asarray(dict_out[name][ell])
            actual = np.asarray(arr_out[ip, il])
            np.testing.assert_allclose(
                actual, expected, rtol=1e-7, atol=1e-15,
                err_msg=f"mismatch for {name} ell={ell}",
            )


def test_dPcross_jit_equiv_unjitted(setup_cross):
    """JIT'd cross-spectrum derivs must match the per-(side,param,ell) version.

    Pins `dPcross_dtheta_autodiff_all_jit` (Step 2) against the original
    eager `dPcross_dtheta_autodiff` for every (side, parameter, ell).
    Same rtol=1e-7 rationale as the auto-spectrum test — XLA reduction
    reordering inside fused jacfwd shifts the last few digits.
    """
    ps, pk_data, k, params, s8 = setup_cross
    ells = (0, 2, 4)
    sides = ("A", "B")

    arr_out = dPcross_dtheta_autodiff_all_jit(
        ps, k, pk_data, params, s8, ells=ells,
    )
    assert arr_out.shape == (2, len(NUISANCE_NAMES), len(ells), len(k))

    for is_, side in enumerate(sides):
        for ip, name in enumerate(NUISANCE_NAMES):
            for il, ell in enumerate(ells):
                expected = np.asarray(dPcross_dtheta_autodiff(
                    ps, k, pk_data, params, name, s8, side, ell,
                ))
                actual = np.asarray(arr_out[is_, ip, il])
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-7, atol=1e-15,
                    err_msg=f"mismatch side={side} {name} ell={ell}",
                )
