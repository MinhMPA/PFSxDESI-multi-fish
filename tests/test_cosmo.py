"""Tests for pfsfog.cosmo — FiducialCosmology."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import simpson

from pfsfog.cosmo import FiducialCosmology, FIDUCIAL, make_plin_func


def sigma8_from_power(k_h, pk_h):
    """Integrate P in (Mpc/h)^3 at k in h/Mpc with a radius of 8 Mpc/h."""
    k_h = np.asarray(k_h, dtype=np.float64)
    pk_h = np.asarray(pk_h, dtype=np.float64)
    x = 8.0 * k_h
    window = np.empty_like(x)
    small = x < 1.0e-3
    y = x[small]
    window[small] = 1.0 - y**2 / 10.0 + y**4 / 280.0
    y = x[~small]
    window[~small] = 3.0 * (np.sin(y) - y * np.cos(y)) / y**3
    integrand = k_h**3 * pk_h * window**2 / (2.0 * np.pi**2)
    return float(np.sqrt(simpson(integrand, x=np.log(k_h))))


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

    @pytest.mark.parametrize("h", [0.55, 0.6736, 0.8])
    @pytest.mark.parametrize("z", [0.0, 0.9, 2.05])
    def test_native_power_integrates_to_independent_sigma8(self, h, z):
        cosmo = FiducialCosmology(params=dict(FIDUCIAL, h=h))
        power = cosmo.pk_data(z)
        assert sigma8_from_power(power["k"], power["pk"]) == pytest.approx(
            cosmo.sigma8(z), rel=0.002,
        )

    @pytest.mark.parametrize("h,z", [(0.55, 0.9), (0.8, 2.05)])
    def test_interpolated_power_keeps_native_power_units(self, h, z):
        cosmo = FiducialCosmology(params=dict(FIDUCIAL, h=h))
        native_k = np.asarray(cosmo.pk_data(z)["k"])
        assert sigma8_from_power(native_k, cosmo.Plin(native_k, z)) == pytest.approx(
            cosmo.sigma8(z), rel=0.002,
        )

    @pytest.mark.parametrize("h,z", [(0.55, 0.9), (0.8, 2.05)])
    def test_jitted_power_factory_keeps_native_power_units(self, h, z):
        params = dict(FIDUCIAL, h=h)
        cosmo = FiducialCosmology(params=params)
        power = jax.jit(make_plin_func())(z, params)
        assert sigma8_from_power(power["k"], power["pk"]) == pytest.approx(
            cosmo.sigma8(z), rel=0.002,
        )


class TestVolumeElement:
    def test_positive(self, cosmo):
        dv = cosmo.comoving_volume_element(1.0)
        assert dv > 0
