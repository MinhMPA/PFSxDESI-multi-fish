"""Physical coordinates for the revision mean model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL, FiducialCosmology, make_plin_func
from pfsfog.eft_params import desi_elg_fiducials
from pfsfog.revision_theory import (
    COSMO_NAMES,
    NUISANCE_NAMES,
    cosmology_from_theta,
    fiducial_nuisance,
    make_sigma8_func,
    nuisance_to_backend,
)


def _theta():
    return jnp.array([FIDUCIAL["ln10_10_As"], FIDUCIAL["mnu"], FIDUCIAL["Omega_m"]])


def _q(**changes):
    values = dict.fromkeys(NUISANCE_NAMES, 0.0)
    values.update(changes)
    return jnp.array([values[name] for name in NUISANCE_NAMES])


def _project_ctr(params, f, *, k=0.13, power=3100.0):
    mu, weights = np.polynomial.legendre.leggauss(64)
    mu = jnp.asarray(mu)
    weights = jnp.asarray(weights)
    ctr = params["ctr"]
    angular = ctr["c0"] + ctr["c2"] * f * mu**2 + ctr["c4"] * f**2 * mu**4
    pkmu = -2.0 * k**2 * power * angular
    legendre = (jnp.ones_like(mu), (3 * mu**2 - 1) / 2, (35 * mu**4 - 30 * mu**2 + 3) / 8)
    return jnp.array([
        (2 * ell + 1) / 2 * jnp.sum(weights * pkmu * polynomial)
        for ell, polynomial in zip((0, 2, 4), legendre)
    ])


def test_coordinate_order_and_fixed_total_matter_mapping():
    assert COSMO_NAMES == ("ln10_10_As", "Mnu", "Omegam")
    assert isinstance(NUISANCE_NAMES, tuple)
    assert len(NUISANCE_NAMES) == 11
    assert "c1" not in NUISANCE_NAMES
    reference = dict(FIDUCIAL)
    changed = _theta().at[1].add(0.08)
    cosmo = cosmology_from_theta(changed, reference)
    assert float(cosmo["omega_cdm"]) == pytest.approx(
        FIDUCIAL["omega_cdm"] - 0.08 / 93.14, abs=1e-12,
    )
    assert float(cosmo["Omega_m"]) == pytest.approx(FIDUCIAL["Omega_m"])
    assert reference == FIDUCIAL
    jac = jax.jacfwd(lambda theta: cosmology_from_theta(theta, reference)["omega_cdm"])(_theta())
    np.testing.assert_allclose(jac, [0.0, -1.0 / 93.14, FIDUCIAL["h"] ** 2], rtol=1e-12, atol=1e-12)
    assert float(cosmology_from_theta(_theta(), FiducialCosmology())["omega_cdm"]) == pytest.approx(FIDUCIAL["omega_cdm"])


def test_sigma8_factory_matches_object_and_tracks_power_amplitude():
    reference = dict(FIDUCIAL)
    sigma8 = make_sigma8_func()
    plin = make_plin_func()
    z = 0.9
    cosmo = cosmology_from_theta(_theta(), reference)
    assert float(jax.jit(sigma8)(z, cosmo)) == pytest.approx(FiducialCosmology().sigma8(z), rel=1e-10)
    high = cosmology_from_theta(_theta().at[0].add(0.04), reference)
    s8_ratio = float(sigma8(z, high) / sigma8(z, cosmo))
    power_ratio = float(plin(z, high)["pk"][350] / plin(z, cosmo)["pk"][350])
    assert s8_ratio == pytest.approx(np.exp(0.02), rel=0.003)
    assert power_ratio == pytest.approx(np.exp(0.04), rel=0.003)
    assert power_ratio == pytest.approx(s8_ratio**2, rel=0.003)
    gradient = jax.jacfwd(lambda theta: sigma8(z, cosmology_from_theta(theta, reference)))(_theta())
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(gradient[0]) > 0


@pytest.mark.parametrize("f", [0.56, 0.91])
@pytest.mark.parametrize("amplitude", [1.0, 1.37])
def test_counterterm_projection_gives_physical_multipoles(f, amplitude):
    q = _q(c0=7.0, c2=23.0, c4=-9.0)
    params = nuisance_to_backend(q, np.sqrt(amplitude) * 0.8, 0.8, f, 0.6736, 3e-4, 0.7)
    projected = _project_ctr(params, f)
    # Q2/Q4 label the f-weighted counterterm basis. Ordinary Legendre
    # coefficients carry 2f/3 and 8f²/35 respectively.
    expected = -2 * 0.13**2 * 3100 * np.array([
        7.0, (2 * f / 3) * 23.0, (8 * f**2 / 35) * -9.0,
    ]) / amplitude
    np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("amplitude", [1.0, 1.37])
def test_c2_only_has_zero_counterterm_monopole(amplitude):
    f = 0.82
    params = nuisance_to_backend(_q(c2=30.0), np.sqrt(amplitude) * 0.8, 0.8, f, 0.6736, 3e-4, 0.7)
    assert abs(float(_project_ctr(params, f)[0])) < 1e-10


def test_backend_bias_fog_and_stochastic_conventions():
    fid = desi_elg_fiducials(b1=1.5, sigma8_z=0.6)
    q = fiducial_nuisance(fid)
    assert q.shape == (11,)
    np.testing.assert_allclose(q, np.array([getattr(fid, name) for name in NUISANCE_NAMES]))
    q = q.at[NUISANCE_NAMES.index("Pshot")].set(0.7)
    q = q.at[NUISANCE_NAMES.index("a2")].set(-0.3)
    backend = nuisance_to_backend(q, 0.72, 0.6, 0.8, 0.6736, 3e-4, 0.7)
    assert float(backend["bias"]["b1"]) == pytest.approx(float(q[0]) / 0.72)
    assert float(backend["bias"]["bGamma3"]) == pytest.approx(fid.bGamma3 / 1.44)
    assert float(backend["ctr"]["cfog"]) == pytest.approx(fid.c_tilde / 1.44**2)
    assert backend["stoch"]["P_shot"] == pytest.approx(0.7)
    assert backend["stoch"]["a2"] == pytest.approx(-0.3)
    assert backend["ndens"] == pytest.approx(3e-4)


def test_projected_counterterm_jacobian_matches_finite_difference():
    q = _q(c0=8.0, c2=19.0, c4=-6.0)

    def projected(x):
        sigma8, f = x
        params = nuisance_to_backend(q, sigma8, 0.8, f, 0.6736, 3e-4, 0.7)
        return _project_ctr(params, f)

    x = jnp.array([0.87, 0.73])
    ad = np.asarray(jax.jacfwd(projected)(x))
    step = 1e-5
    fd = np.column_stack([
        (np.asarray(projected(x.at[i].add(step))) - np.asarray(projected(x.at[i].add(-step)))) / (2 * step)
        for i in range(2)
    ])
    np.testing.assert_allclose(ad, fd, rtol=1e-8, atol=1e-7)
    assert np.linalg.norm(ad[:, 0]) > 0
    assert abs(ad[0, 1]) < 1e-8
