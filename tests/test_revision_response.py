"""The physical pair mean at fixed observed coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.revision_theory import APInterpolationBackend, NUISANCE_NAMES, ap_coordinates, make_pair_response
from ps_1loop_jax import PowerSpectrum1Loop


def _theta():
    return jnp.array([FIDUCIAL["ln10_10_As"], FIDUCIAL["mnu"], FIDUCIAL["Omega_m"]])


def _q(**changes):
    values = dict.fromkeys(NUISANCE_NAMES, 0.0)
    values.update(changes)
    return jnp.array([values[name] for name in NUISANCE_NAMES])


class PowerLawPair:
    """Exact Kaiser and stochastic AP limit with the backend method contract."""

    def __init__(self, slope=-1.2):
        self.slope = slope
        self.calls = []

    def _get_pkmu_pair_ref_interp(self, k, mu, alpha_perp, alpha_para, pk_data,
                                  params, add_stochasticity=False):
        self.calls.append(add_stochasticity)
        kt, mt, jac = ap_coordinates(k, mu, alpha_perp, alpha_para)
        b1 = params["bias"]["b1"]
        b2 = params.get("bias2", params["bias"])["b1"]
        f = params["f"]
        result = kt**self.slope * (b1 + f * mt**2) * (b2 + f * mt**2)
        if add_stochasticity:
            stoch = params["stoch"]
            result = result + (stoch["P_shot"] + stoch["a0"] * (kt / params["k_nl"])**2
                               + stoch["a2"] * (kt * mt / params["k_nl"])**2) / params["ndens"]
        return result / jac


class InterpolatedPowerLaw:
    """Analytic signal through the actual backend AP interpolation routine."""

    _get_pkmu_pair_ref_interp = PowerSpectrum1Loop._get_pkmu_pair_ref_interp

    def __init__(self, slope=-1.2):
        self.slope = slope
        self._kn = jnp.geomspace(0.002, 0.4, 400)
        self._mu = jnp.linspace(0.0, 1.0, 65)

    def get_pkmu_pair(self, k, mu, pk_data, params, add_stochasticity=False):
        b1 = params['bias']['b1']
        b2 = params.get('bias2', params['bias'])['b1']
        f = params['f']
        power = k[:, None]**self.slope * (b1 + f*mu[None, :]**2) * (b2 + f*mu[None, :]**2)
        if add_stochasticity:
            s = params['stoch']
            power += (s['P_shot'] + s['a0']*(k[:, None]/params['k_nl'])**2
                      + s['a2']*(k[:, None]*mu[None, :]/params['k_nl'])**2) / params['ndens']
        return power


@pytest.fixture
def toy(monkeypatch):
    import pfsfog.revision_theory as rt
    monkeypatch.setattr(rt, "make_plin_func", lambda: lambda z, c: {"k": jnp.array([0.01, 1.]), "pk": jnp.ones(2)})
    monkeypatch.setattr(rt, "make_sigma8_func", lambda: lambda z, c: jnp.exp(0.5 * (c["ln10_10_As"] - FIDUCIAL["ln10_10_As"])))
    monkeypatch.setattr(rt, "make_growth_rate_func", lambda: lambda z, c: jnp.array(0.8))
    return PowerLawPair()


def test_ap_coordinates_isotropic_and_anisotropic():
    k = jnp.array([0.1, 0.2])
    mu = jnp.array([0.0, 0.6, 1.0])
    kt, mt, jac = ap_coordinates(k, mu, 1.2, 0.8)
    np.testing.assert_allclose(kt[:, 0], k / 1.2)
    np.testing.assert_allclose(kt[:, -1], k / 0.8)
    np.testing.assert_allclose(mt[0], [0.0, 0.747409, 1.0], rtol=1e-5)
    assert jac == pytest.approx(1.2**2 * 0.8)


def test_reference_is_frozen_and_identity_auto_includes_fitted_noise(toy):
    reference = dict(FIDUCIAL)
    response = make_pair_response(toy, 0.9, reference, 2e-4, ap_nk=None)
    reference["Omega_m"] = 0.5
    reference["h"] = 0.9
    reference["n_s"] = 1.02
    k = jnp.array([0.08, 0.14])
    mu = jnp.array([0.0, 0.5, 1.0])
    q = _q(b1_sigma8=1.5, Pshot=0.4, a0=0.2, a2=-0.1)
    result = response.pkmu(k, mu, _theta(), q)
    b = 1.5
    expected = k[:, None]**toy.slope * (b + 0.8 * mu[None, :]**2)**2
    expected += (0.4 + 0.2 * (k[:, None] / 0.7)**2
                 - 0.1 * (k[:, None] * mu[None, :] / 0.7)**2) / 2e-4
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    assert toy.calls == [True]
    assert response.reference['h'] == FIDUCIAL['h']
    assert response.reference['n_s'] == FIDUCIAL['n_s']
    with pytest.raises(TypeError):
        response.reference['h'] = 0.9


def test_cross_ap_remaps_noise_once_and_both_biases(toy):
    response = make_pair_response(toy, 0.9, FIDUCIAL, 2e-4, 4e-4, ap_nk=None)
    theta = _theta().at[2].add(0.01)
    k = jnp.array([0.1, 0.2])
    mu = jnp.array([0.0, 0.5, 1.0])
    qa, qb = _q(b1_sigma8=1.3), _q(b1_sigma8=1.8)
    qc = jnp.array([0.3, -0.2])
    result = response.pkmu(k, mu, theta, qa, qb, qc)
    alpha_perp, alpha_para = response.ap_alphas(theta)
    kt, mt, jac = ap_coordinates(k, mu, alpha_perp, alpha_para)
    expected = (kt**toy.slope * (1.3 + 0.8 * mt**2) * (1.8 + 0.8 * mt**2)
                + (0.3 - 0.2 * (kt * mt / 0.7)**2) / np.sqrt(2e-4 * 4e-4)) / jac
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    assert toy.calls == [False]
    assert np.linalg.norm(np.asarray(jax.jacfwd(lambda x: response.pkmu(k, mu, x, qa, qb, qc))(theta)[..., 2])) > 0
    assert np.linalg.norm(np.asarray(jax.jacfwd(lambda x: response.pkmu(k, mu, theta, x, qb, qc))(qa))) > 0
    assert np.linalg.norm(np.asarray(jax.jacfwd(lambda x: response.pkmu(k, mu, theta, qa, x, qc))(qb))) > 0


def test_multipoles_gauss_legendre_kaiser(toy):
    response = make_pair_response(toy, 0.9, FIDUCIAL, 2e-4, ap_nk=None)
    k = jnp.array([0.1])
    result = response.multipoles(k, _theta(), _q(b1_sigma8=1.5), n_mu=32)
    b, f = 1.5, 0.8
    expected = k[0]**toy.slope * np.array([
        b*b + 2*b*f/3 + f*f/5,
        4*b*f/3 + 4*f*f/7,
        8*f*f/35,
    ])
    np.testing.assert_allclose(result[0], expected, rtol=1e-12)


def test_signed_mu_parity_through_backend_interpolation(toy):
    response = make_pair_response(InterpolatedPowerLaw(), 0.9, FIDUCIAL, 2e-4)
    k = jnp.array([0.035, 0.075, 0.15])
    mu = jnp.array([0.21, 0.63, 0.87])
    q = _q(b1_sigma8=1.5, a2=0.3)
    theta = _theta().at[2].add(0.01)
    positive = response.pkmu(k, mu, theta, q)
    negative = response.pkmu(k, -mu, theta, q)
    np.testing.assert_allclose(negative, positive, rtol=1e-12, atol=1e-12)


def test_isotropic_power_law_through_actual_backend_ap():
    model = InterpolatedPowerLaw()
    k = jnp.array([0.03, 0.08, 0.16])
    mu = jnp.array([0.0, 0.3, 0.7, 1.0])
    params = {'bias': {'b1': 1.6}, 'f': 0.8}
    alpha = 1.04
    actual = model._get_pkmu_pair_ref_interp(k, mu, alpha, alpha, {}, params)
    expected = k[:, None]**model.slope * (1.6+0.8*mu[None, :]**2)**2 * alpha**(-3-model.slope)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_ap_grid_is_separate_from_fft_grid_and_converges(toy):
    backend = InterpolatedPowerLaw()
    original_k = np.asarray(backend._kn).copy()
    response = make_pair_response(backend, 0.9, FIDUCIAL, 2e-4)
    assert len(response.ap_backend._kn) == 2048
    assert len(response.ap_backend._mu) == 101
    k, mu = jnp.array([.0125,.0775,.1975]), jnp.array([.1,.5,.9])
    q = _q(b1_sigma8=1.5, Pshot=.4, a2=.2)
    actual = response.pkmu(k,mu,_theta(),q)
    expected = k[:,None]**backend.slope*(1.5+.8*mu[None,:]**2)**2
    expected += (.4+.2*(k[:,None]*mu[None,:]/.7)**2)/2e-4
    np.testing.assert_allclose(actual,expected,rtol=1e-7)
    np.testing.assert_array_equal(backend._kn,original_k)
    assert np.isnan(np.asarray(response.pkmu(jnp.array([.7]),mu,_theta(),q))).all()


def test_ap_endpoint_survives_jit_roundoff_but_rejects_invalid_observed_mu():
    backend = APInterpolationBackend(InterpolatedPowerLaw())
    k = jnp.array([.0125,.1975])
    params = {'bias': {'b1': 1.5}, 'f': .8}
    evaluate = jax.jit(lambda alpha,mu: backend._get_pkmu_pair_ref_interp(
        k,mu,alpha[0],alpha[1],{},params))
    alpha = jnp.array([1.,np.nextafter(1.,0.)])
    endpoint = evaluate(alpha,jnp.array([0.,1.]))
    expected = k[:,None]**(-1.2)*(1.5+.8*jnp.array([0.,1.])[None,:]**2)**2
    assert np.all(np.isfinite(endpoint))
    np.testing.assert_allclose(endpoint,expected,rtol=1e-7)
    assert np.isnan(np.asarray(evaluate(alpha,jnp.array([1.001])))).all()


def test_real_response_endpoint_at_fixed_omega_m_neutrino_step():
    response = make_pair_response(PowerSpectrum1Loop(do_irres=False),.7,FIDUCIAL,1e-4)
    evaluate = jax.jit(lambda t: response.pkmu(
        jnp.array([.0125,.1975]),jnp.array([0.,1.]),t,_q(b1_sigma8=1.5)))
    value = evaluate(_theta().at[1].add(.005))
    assert np.all(np.isfinite(value))
