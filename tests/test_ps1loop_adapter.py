"""Tests for pfsfog.ps1loop_adapter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.eft_params import desi_elg_fiducials, pfs_elg_fiducials
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto,
    fisher_to_ps1loop_cross,
    fisher_to_ept,
    perturb_fiducials,
)


B1 = 1.3
S8 = 0.49
F_Z = 0.87
H = 0.6736
NBAR = 4e-4


@pytest.fixture
def fid_desi():
    return desi_elg_fiducials(b1=B1, sigma8_z=S8)


@pytest.fixture
def fid_pfs():
    return pfs_elg_fiducials(b1_pfs=1.34, b1_desi=B1, sigma8_z=S8)


class TestAutoParams:
    def test_b1_roundtrip(self, fid_desi):
        p = fisher_to_ps1loop_auto(fid_desi, S8, F_Z, H, NBAR)
        assert abs(p["bias"]["b1"] - B1) < 1e-10

    def test_has_stoch(self, fid_desi):
        p = fisher_to_ps1loop_auto(fid_desi, S8, F_Z, H, NBAR)
        assert "stoch" in p
        assert "P_shot" in p["stoch"]

    def test_f_and_h(self, fid_desi):
        p = fisher_to_ps1loop_auto(fid_desi, S8, F_Z, H, NBAR)
        assert p["f"] == F_Z
        assert p["h"] == H


class TestCrossParams:
    def test_has_bias2(self, fid_pfs, fid_desi):
        p = fisher_to_ps1loop_cross(fid_pfs, fid_desi, S8, F_Z, H, 9e-4, NBAR)
        assert "bias2" in p and "ctr2" in p

    def test_no_stoch(self, fid_pfs, fid_desi):
        p = fisher_to_ps1loop_cross(fid_pfs, fid_desi, S8, F_Z, H, 9e-4, NBAR)
        assert "stoch" not in p


class TestEPT:
    def test_keys(self, fid_desi):
        e = fisher_to_ept(fid_desi, S8, NBAR)
        for key in ["b1", "b2", "bG2", "bGamma3", "cs0", "Pshot", "b4"]:
            assert key in e

    def test_b1_matches(self, fid_desi):
        e = fisher_to_ept(fid_desi, S8, NBAR)
        assert abs(e["b1"] - B1) < 1e-10


class TestPerturb:
    def test_perturb_c_tilde(self, fid_desi):
        fid2 = perturb_fiducials(fid_desi, "c_tilde", 10.0)
        assert abs(fid2.c_tilde - fid_desi.c_tilde - 10.0) < 1e-10

    def test_perturb_unknown_raises(self, fid_desi):
        with pytest.raises(KeyError):
            perturb_fiducials(fid_desi, "nonexistent", 1.0)


class TestAutodiffSmoke:
    """Verify that JAX can differentiate through the adapter + ps_1loop_jax."""

    def test_grad_b1(self, fid_desi):
        from ps_1loop_jax import PowerSpectrum1Loop

        ps = PowerSpectrum1Loop(do_irres=False)
        k_lin = jnp.geomspace(1e-4, 10.0, 256)
        pk_lin = 1e4 * (k_lin / 0.05) ** (-2.5)
        pk_data = {"k": k_lin, "pk": pk_lin}
        k = jnp.geomspace(0.01, 0.25, 15)

        def p0_of_b1(b1_val):
            p = fisher_to_ps1loop_auto(fid_desi, S8, F_Z, H, NBAR)
            p["bias"] = dict(p["bias"])
            p["bias"]["b1"] = b1_val
            return jnp.sum(ps.get_pk_ell(k, 0, pk_data, p))

        grad = jax.grad(p0_of_b1)(jnp.float64(B1))
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
