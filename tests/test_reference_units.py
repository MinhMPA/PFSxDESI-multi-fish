"""Fixed-reference power units, independent AP algebra and network radius."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL, make_plin_func
from pfsfog.cosmology_space import cosmology_from_coordinates, make_cosmology_spec
from pfsfog.reference_units import (
    make_reference_plin_func, native_power_to_reference, sigma8_radius_reference,
)
from pfsfog.revision_theory import make_sigma8_func
from test_cosmology_space import theta6


def test_exact_native_conversion_and_radius():
    kn = jnp.array([.01, .1, 1.])
    pn = jnp.array([12., 30., 9.])
    k, p = native_power_to_reference(kn, pn, h_trial=.8, h_ref=.64)
    np.testing.assert_allclose(k, kn/.64, rtol=1e-12)
    np.testing.assert_allclose(p, pn*(.64/.8)**3, rtol=1e-12)
    assert float(sigma8_radius_reference(.8, .64)) == pytest.approx(6.4)
    np.testing.assert_allclose(native_power_to_reference(kn, pn, .64, .64)[1], pn, rtol=1e-12)
    jac = jax.jacfwd(lambda h: native_power_to_reference(kn, pn, h, .64)[1])(.8)
    np.testing.assert_allclose(jac, -3*np.array(p)/.8, rtol=1e-12)


def test_reference_wrapper_recovers_old_wrapper_at_reference():
    old = make_plin_func()(1.1, FIDUCIAL)
    new = make_reference_plin_func(FIDUCIAL["h"])(1.1, FIDUCIAL)
    for key in ("k", "pk"):
        np.testing.assert_allclose(new[key], old[key], rtol=1e-10, atol=0)


def _sigma_integral(data, radius):
    k, p = np.asarray(data["k"]), np.asarray(data["pk"])
    x = k * radius
    window = 3 * (np.sin(x) - x*np.cos(x)) / x**3
    return np.sqrt(np.trapezoid(k**3*p*window**2, x=np.log(k)) / (2*np.pi**2))


def _validated_cosmo(theta, space):
    from pfsfog.cosmology_space import validate_coordinate_neighborhood
    validate_coordinate_neighborhood(np.array(theta), FIDUCIAL, space)
    return cosmology_from_coordinates(theta, FIDUCIAL, space)


def test_real_emulator_sigma8_radius_at_declared_neighborhood():
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    plin, sigma8 = make_reference_plin_func(space.h_ref), make_sigma8_func()
    points = [("fiducial", theta6())]
    points += [(f"{name}{sign:+}", theta6().at[i].add(sign*space.fd_steps[name][0]))
               for i, name in enumerate(space.names) if name in ("h", "omega_b", "n_s")
               for sign in (-1, 1)]
    for label, theta in points:
        cosmo = _validated_cosmo(theta, space)
        for z in (0., 1.1):
            integrated = _sigma_integral(plin(z, cosmo), float(sigma8_radius_reference(cosmo["h"], space.h_ref)))
            network = float(sigma8(z, cosmo))
            print(f"radius-check {label} z={z} integrated={integrated:.12g} network={network:.12g} relative={integrated/network-1:.12g}")
            assert integrated == pytest.approx(network, rel=.002), (label, z)


@pytest.mark.parametrize("h", [.64, .70])
def test_fixed_eight_radius_counterexample_in_mapped_domain(h):
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    cosmo = _validated_cosmo(theta6().at[3].set(h), space)
    plin, sigma8 = make_reference_plin_func(space.h_ref), make_sigma8_func()
    for z in (0., 1.1):
        data, expected = plin(z, cosmo), float(sigma8(z, cosmo))
        correct = _sigma_integral(data, 8*space.h_ref/h)
        wrong = _sigma_integral(data, 8.)
        print(f"radius-control h={h} z={z} correct_relative={correct/expected-1:.12g} fixed8_relative={wrong/expected-1:.12g}")
        assert correct == pytest.approx(expected, rel=.002)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(wrong, expected, rtol=.002, atol=0)


def _analytic_factories(monkeypatch):
    """Replace expensive physical networks only for exact unit-change fixtures."""
    from pfsfog import revision_theory as theory

    def power_factory(h_ref=None):
        def power(z, cosmo):
            h_units = cosmo["h"] if h_ref is None else h_ref
            # Physical P(k)=12000*k**(-.7) Mpc^3, independent of units.
            return {"amplitude": 12000*h_units**2.3, "slope": -.7}
        return power

    monkeypatch.setattr(theory, "make_plin_func", power_factory)
    monkeypatch.setattr(theory, "make_reference_plin_func", power_factory, raising=False)
    monkeypatch.setattr(theory, "make_sigma8_func", lambda: lambda z, c: jnp.asarray(.8))
    monkeypatch.setattr(theory, "make_growth_rate_func", lambda: lambda z, c: jnp.asarray(.7))
    return theory


class _AnalyticPair:
    do_irres = False

    def _get_pkmu_pair_ref_interp(self, k, mu, alpha_perp, alpha_para,
                                  pk_data, params, add_stochasticity=False):
        # Direct angular formula, independent of production AP interpolation.
        transverse = k[:, None]*jnp.sqrt(1-mu[None, :]**2)/alpha_perp
        parallel = k[:, None]*mu[None, :]/alpha_para
        kt = jnp.sqrt(transverse**2+parallel**2)
        mt = parallel/kt
        power = pk_data["amplitude"]*kt**pk_data["slope"]
        f = params["f"]
        b1 = params["bias"]["b1"]
        b2 = params.get("bias2", params["bias"])["b1"]
        ca, cb = params["ctr"], params.get("ctr2", params["ctr"])
        c = {key: (ca[key]+cb[key])/2 for key in ca}
        angular = c["c0"]+c["c2"]*f*mt**2+c["c4"]*f**2*mt**4
        rsd = (b1+f*mt**2)*(b2+f*mt**2)
        result = power*(rsd-2*kt**2*angular-c["cfog"]*kt**4*f**4*mt**4*rsd)
        if add_stochasticity:
            noise = params["stoch"]
            result += (noise["P_shot"] + noise["a0"]*(kt/params["k_nl"])**2
                       + noise["a2"]*(parallel/params["k_nl"])**2)/params["ndens"]
        return result/(alpha_perp**2*alpha_para)


def _analytic_q(theory, second=False):
    values = dict.fromkeys(theory.NUISANCE_NAMES, 0.)
    values.update(b1_sigma8=1.36 if second else 1.12,
                  c0=4. if second else 3., c2=8. if second else 5.,
                  c4=-1. if second else 2., c_tilde=330. if second else 220.,
                  Pshot=.9, a0=.4, a2=1.3)
    return jnp.array([values[name] for name in theory.NUISANCE_NAMES])


def _trial_h_expected(k, mu, h, h_ref, alphas, *, cross, omit=None):
    """Independent trial-h equation, constructed without production converters.

    s=h/h_ref: alpha_eff=s*alpha_phys, P_trial=s^3*P_ref(s*k),
    Q_trial=s^2 Q_ref, T_trial=s^4 T_ref, n_trial=n_ref/s^3,
    k_nl,trial=k_nl,ref/s. All q amplitudes and biases are written explicitly.
    """
    s, f = h/h_ref, .7
    ap, ar = alphas
    if omit != "ap":
        ap, ar = ap*s, ar*s
    parallel = np.asarray(k)[:, None]*np.asarray(mu)[None, :]/ar
    perpendicular = np.asarray(k)[:, None]*np.sqrt(1-np.asarray(mu)[None, :]**2)/ap
    kt = np.sqrt(parallel**2+perpendicular**2)
    mt = parallel/kt
    power_scale = h_ref if omit == "power" else h
    power = 12000*power_scale**2.3*kt**(-.7)
    qscale = 1 if omit == "q" else s**2
    tscale = 1 if omit == "t" else s**4
    q0, q2, q4, t = (3.5, 6.5, .5, 275.) if cross else (3., 5., 2., 220.)
    c0 = qscale*(q0-f*q2/3+3*f*f*q4/35)
    c2 = qscale*(q2-6*f*q4/7)
    c4 = qscale*q4
    rsd = (1.4+f*mt**2)*((1.7 if cross else 1.4)+f*mt**2)
    result = power*(rsd-2*kt**2*(c0+c2*f*mt**2+c4*f*f*mt**4)
                    -tscale*t*kt**4*f**4*mt**4*rsd)
    nscale = 1 if omit == "density" else s**(-3)
    knl = .7 if omit == "knl" else .7/s
    if cross:
        result += (.6+1.1*(parallel/knl)**2)/(np.sqrt(2e-4*4e-4)*nscale)
    else:
        result += (.9+.4*(kt/knl)**2+1.3*(parallel/knl)**2)/(2e-4*nscale)
    return result/(ap**2*ar)


@pytest.mark.parametrize("h", [.6, .8])
@pytest.mark.parametrize("cross", [False, True])
def test_independent_auto_cross_ap_units_with_dimensional_templates(monkeypatch, h, cross):
    theory = _analytic_factories(monkeypatch)
    reference = dict(FIDUCIAL)
    space = make_cosmology_spec(reference, extended=True)
    response = theory.make_pair_response(_AnalyticPair(), 1.1, reference, 2e-4,
                                        4e-4 if cross else None, ap_nk=None, space=space)
    alphas = (1.07, .93)
    response.ap_alphas = lambda theta: alphas
    k, mu = jnp.array([.03, .09, .17]), jnp.array([.1, .45, .9])
    theta = theta6().at[3].set(h)
    actual = response.pkmu(k, mu, theta, _analytic_q(theory),
                          _analytic_q(theory, True) if cross else None,
                          jnp.array([.6, 1.1]) if cross else None)
    expected = _trial_h_expected(k, mu, h, space.h_ref, alphas, cross=cross)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    for omitted in ("ap", "power", "q", "t", "density", "knl"):
        wrong = _trial_h_expected(k, mu, h, space.h_ref, alphas, cross=cross, omit=omitted)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(actual, wrong, rtol=1e-10, atol=1e-10)


def test_explicit_space_response_maps_all_columns_freezes_reference_and_physical_ap(monkeypatch):
    theory = _analytic_factories(monkeypatch)
    reference = dict(FIDUCIAL)
    space = make_cosmology_spec(reference, extended=True)
    response = theory.make_pair_response(_AnalyticPair(), 1.1, reference, 2e-4,
                                        ap_nk=None, space=space)
    reference["h"] = .8
    assert response.reference["h"] == space.h_ref
    theta = theta6().at[3].add(.001).at[4].add(.0001).at[5].add(.001)
    # Rebuild physical background arguments explicitly, not through the map.
    from ps_1loop_jax import background as bg
    omc = theta[2]*theta[3]**2-theta[4]-theta[1]/93.14
    args = (theta[4], omc, theta[3], 1.1, theta[1])
    refargs = (FIDUCIAL["omega_b"], FIDUCIAL["omega_cdm"], FIDUCIAL["h"], 1.1, FIDUCIAL["mnu"])
    expected = (bg.angular_diameter_distance(*args)/bg.angular_diameter_distance(*refargs),
                bg.Hz(*refargs)/bg.Hz(*args))
    np.testing.assert_allclose(response.ap_alphas(theta), expected, rtol=1e-12)
    derivative = jax.jacfwd(lambda t: jnp.stack(response.ap_alphas(t)))(theta)
    assert derivative.shape == (2, 6)
    assert np.isfinite(derivative).all()
    assert np.linalg.norm(np.asarray(derivative[:, 3])) > 0
    # Check full mapping routing by observing each physical network input.
    response.plin = lambda z, c: {"amplitude": (
        c["ln10_10_As"]+2*c["mnu"]+3*c["Omega_m"]+4*c["h"]
        +5*c["omega_b"]+6*c["n_s"]+7*c["omega_cdm"]), "slope": -.7}
    result = jax.jacfwd(lambda t: response.pkmu(jnp.array([.1]), jnp.array([.4]),
                       t, _analytic_q(theory), ap=False))(theta)
    assert result.shape == (1, 1, 6)
    assert np.all(np.abs(result) > 0)
    expected_gradient = np.array([1., 2-7/93.14, 3+7*float(theta[3])**2,
                                  4+14*float(theta[2]*theta[3]), -2., 6.])
    np.testing.assert_allclose(np.asarray(result).ravel()/float(result[0, 0, 0]),
                               expected_gradient, rtol=1e-12)
    assert theory.COSMO_NAMES == ("ln10_10_As", "Mnu", "Omegam")


@pytest.mark.parametrize("z", [-.01, 5.01, np.nan])
def test_explicit_space_validates_actual_redshift_before_network_loading(monkeypatch, z):
    theory = _analytic_factories(monkeypatch)
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    def forbidden():
        pytest.fail("invalid redshift reached network factory")
    monkeypatch.setattr(theory, "make_sigma8_func", forbidden)
    with pytest.raises(ValueError, match="redshift"):
        theory.make_pair_response(_AnalyticPair(), z, FIDUCIAL, 2e-4, space=space)


def test_none_space_preserves_old_response_and_explicit_baseline_at_fiducial(monkeypatch):
    theory = _analytic_factories(monkeypatch)
    old = theory.make_pair_response(_AnalyticPair(), 1.1, FIDUCIAL, 2e-4, ap_nk=None)
    explicit_none = theory.make_pair_response(_AnalyticPair(), 1.1, FIDUCIAL, 2e-4,
                                             ap_nk=None, space=None)
    explicit_three = theory.make_pair_response(_AnalyticPair(), 1.1, FIDUCIAL, 2e-4,
        ap_nk=None, space=make_cosmology_spec(FIDUCIAL, extended=False))
    args = (jnp.array([.1, .15]), jnp.array([.2, .8]), theta6()[:3], _analytic_q(theory))
    expected = old.pkmu(*args)
    np.testing.assert_array_equal(explicit_none.pkmu(*args), expected)
    np.testing.assert_allclose(explicit_three.pkmu(*args), expected, rtol=1e-12)
