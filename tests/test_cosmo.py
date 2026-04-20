"""Tests for pfsfog.cosmo — FiducialCosmology."""

import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FiducialCosmology, FIDUCIAL


@pytest.fixture(scope="module")
def cosmo():
    return FiducialCosmology()


class TestFiducial:
    def test_fiducial_dict(self):
        assert abs(FIDUCIAL["h"] - 0.6736) < 1e-6
        assert abs(FIDUCIAL["omega_b"] - 0.02237) < 1e-6
        assert abs(FIDUCIAL["Omega_m"] - 0.3153) < 0.002


class TestBackground:
    def test_H_z0(self, cosmo):
        H0 = float(cosmo.H(0.0))
        assert abs(H0 - 67.36) < 0.5  # H0 = 100 h

    def test_H_increases(self, cosmo):
        assert float(cosmo.H(1.0)) > float(cosmo.H(0.0))

    def test_growth_factor_z0(self, cosmo):
        assert abs(float(cosmo.D(0.0)) - 1.0) < 1e-4

    def test_growth_factor_decreases(self, cosmo):
        assert float(cosmo.D(1.0)) < 1.0

    def test_growth_rate_range(self, cosmo):
        f = float(cosmo.f(1.0))
        assert 0.5 < f < 1.0  # f ≈ 0.87 for Planck at z=1

    def test_chi_positive(self, cosmo):
        assert float(cosmo.chi(1.0)) > 0

    def test_D_A_positive(self, cosmo):
        assert float(cosmo.D_A(1.0)) > 0


class TestSigma8:
    def test_sigma8_z0(self, cosmo):
        s8 = cosmo.sigma8(0.0)
        assert abs(s8 - 0.811) < 0.01  # Planck 2018 value

    def test_sigma8_decreases(self, cosmo):
        assert cosmo.sigma8(1.0) < cosmo.sigma8(0.0)

    def test_sigma8_z1(self, cosmo):
        s8 = cosmo.sigma8(1.0)
        assert 0.45 < s8 < 0.55  # ~ 0.49


class TestPlin:
    def test_plin_positive(self, cosmo):
        k = jnp.geomspace(0.01, 0.3, 20)
        pk = cosmo.Plin(k, 1.0)
        assert jnp.all(pk > 0)

    def test_plin_shape(self, cosmo):
        k = jnp.geomspace(0.01, 0.3, 20)
        pk = cosmo.Plin(k, 1.0)
        assert pk.shape == (20,)

    def test_pk_data_keys(self, cosmo):
        pd = cosmo.pk_data(1.0)
        assert "k" in pd and "pk" in pd
        assert len(pd["k"]) == len(pd["pk"])


class TestVolumeElement:
    def test_positive(self, cosmo):
        dv = cosmo.comoving_volume_element(1.0)
        assert dv > 0
