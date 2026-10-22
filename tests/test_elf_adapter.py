"""Independent physical identities for the ELF translation.

These catch changed FoG angular dependence, cross-noise leakage, lost AP
Jacobian, endpoint derivative singularities and silent extrapolation.
"""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_counterterm_fields import field_from_projection


@pytest.fixture(scope="module")
def engine():
    from pfsfog.elf_adapter import ELFPowerSpectrum
    return ELFPowerSpectrum()


@pytest.fixture(scope="module")
def spectrum():
    k = jnp.geomspace(1.e-5, 100., 512)
    p = 2.e4 * (k / .05) ** .96 / (1 + (k / .05) ** 2) ** 2
    rows = jnp.array([230, 265, 300])
    return {"k": k, "pk": p}, k[rows], p[rows]


def parameters():
    return dict(h=.6736, f=.81,
                bias=dict(b1=1.8, b2=.35, bG2=-.22, bGamma3=.17),
                ctr=dict(c0=0., c2=0., c4=0., cfog=0.),
                stoch=dict(P_shot=0., a0=0., a2=0.), ndens=3.e-4, k_nl=.7)


@pytest.mark.parametrize("cross", [False, True])
def test_counterterms_and_fog_preserve_full_angular_form(engine, spectrum, cross):
    pd, k, plin = spectrum
    mu = jnp.array([0., .29, .73, 1.])
    base = parameters()
    if cross:
        base["bias2"] = dict(b1=1.3, b2=-.2, bG2=-.1, bGamma3=.05)
        base["ctr2"] = dict(c0=0., c2=0., c4=0., cfog=0.)
    varied = deepcopy(base)
    varied["ctr"] = dict(c0=4., c2=-7., c4=9., cfog=110.)
    if cross:
        varied["ctr2"] = dict(c0=-2., c2=3., c4=5., cfog=250.)
    actual = engine.get_pkmu_pair(k, mu, pd, varied) - engine.get_pkmu_pair(k, mu, pd, base)
    # Recover each tracer field by independent Legendre quadrature.
    b_b = 1.3 if cross else 1.8
    cb = varied["ctr2"] if cross else varied["ctr"]
    da = field_from_projection(varied["ctr"], 1.8, .81)
    db = field_from_projection(cb, b_b, .81)
    fog = (varied["ctr"]["cfog"]+cb["cfog"])/2
    kk, mm, pp = k[:, None], mu[None, :], plin[:, None]
    fa = da[0]+da[1]*mm**2+da[2]*mm**4
    fb = db[0]+db[1]*mm**2+db[2]*mm**4
    expected = kk**2*pp*((1.8+.81*mm**2)*fb+(b_b+.81*mm**2)*fa)
    expected -= fog * kk**4 * pp * .81**4 * mm**4 * (1.8+.81*mm**2) * (b_b+.81*mm**2)
    np.testing.assert_allclose(actual, expected, rtol=2.e-10, atol=2.e-9)


def test_auto_noise_and_cross_signal_are_separate(engine, spectrum):
    pd, k, _ = spectrum
    mu = jnp.array([0., .4, 1.])
    p = parameters()
    p["stoch"] = dict(P_shot=.6, a0=-.3, a2=.7)
    signal = engine.get_pkmu_pair(k, mu, pd, p)
    auto = engine.get_pkmu(k, mu, pd, p)
    noise = (.6 + (k[:, None]/.7)**2 * (-.3+.7*mu[None, :]**2)) / 3.e-4
    np.testing.assert_allclose(auto-signal, noise, rtol=2.e-12, atol=2.e-10)
    p["bias2"], p["ctr2"] = deepcopy(p["bias"]), deepcopy(p["ctr"])
    np.testing.assert_allclose(engine.get_pkmu_pair(k, mu, pd, p), signal, rtol=2.e-12, atol=2.e-10)
    with pytest.raises(ValueError, match="cross.*stochastic"):
        engine.get_pkmu_pair(k, mu, pd, p, add_stochasticity=True)


def test_cross_is_symmetric_and_even(engine, spectrum):
    pd, k, _ = spectrum
    mu = jnp.array([-.9, -.3, 0., .3, .9])
    p = parameters()
    p["ctr"]["cfog"] = 120.
    p["bias2"] = dict(b1=1.1, b2=-.2, bG2=.12, bGamma3=-.08)
    p["ctr2"] = dict(c0=3., c2=5., c4=-2., cfog=320.)
    swapped = {**p, "bias": p["bias2"], "bias2": p["bias"],
               "ctr": p["ctr2"], "ctr2": p["ctr"]}
    a = engine.get_pkmu_pair_ref(k, mu, 1.02, .97, pd, p)
    b = engine.get_pkmu_pair_ref(k, mu, 1.02, .97, pd, swapped)
    np.testing.assert_allclose(a, b, rtol=2.e-12, atol=2.e-10)
    np.testing.assert_allclose(a, a[:, ::-1], rtol=2.e-12, atol=2.e-10)


def test_ap_maps_noise_and_isotropic_full_signal_once(engine, spectrum):
    pd, k, _ = spectrum
    mu = jnp.array([0., .45, 1.])
    p = parameters()
    p["ctr"] = dict(c0=4., c2=-7., c4=9., cfog=110.)
    p["stoch"] = dict(P_shot=.6, a0=-.3, a2=.7)
    scale = 1.03
    actual = engine.get_pkmu_pair_ref(k, mu, scale, scale, pd, p, add_stochasticity=True)
    expected = engine.get_pkmu(k/scale, mu, pd, p) / scale**3
    np.testing.assert_allclose(actual, expected, rtol=3.e-12, atol=3.e-9)
    ap, al = 1.02, .97
    signal = engine.get_pkmu_pair_ref(k, mu, ap, al, pd, p)
    total = engine.get_pkmu_pair_ref(k, mu, ap, al, pd, p, add_stochasticity=True)
    k2true = k[:, None]**2*((1-mu[None, :]**2)/ap**2+mu[None, :]**2/al**2)
    k2mu2true = (k[:, None]*mu[None, :]/al)**2
    expected_noise = (.6 + (-.3*k2true+.7*k2mu2true)/.7**2) / (3.e-4*ap**2*al)
    np.testing.assert_allclose(total-signal, expected_noise, rtol=3.e-12, atol=3.e-9)


def test_ap_derivatives_are_finite_at_endpoints(engine, spectrum):
    pd, k, _ = spectrum
    p = parameters()
    p["ctr"]["cfog"] = 200.
    def call(scales):
        return engine.get_pkmu_pair_ref(k, jnp.array([0., 1.]), *scales, pd, p)
    ad = jax.jacfwd(call)(jnp.ones(2))
    assert np.isfinite(ad).all()
    step = 2.e-5
    for i in range(2):
        d = jnp.eye(2)[i]*step
        fd = (call(jnp.ones(2)+d)-call(jnp.ones(2)-d))/(2*step)
        np.testing.assert_allclose(ad[..., i], fd, rtol=2.e-5, atol=2.e-4)


def test_invalid_ap_or_true_domain_is_not_extrapolated(engine, spectrum):
    pd, _, _ = spectrum
    p = parameters()
    assert np.isnan(engine.get_pkmu_pair_ref(jnp.array([1.e-6]), jnp.array([.5]), 1., 1., pd, p)).all()
    assert np.isnan(engine.get_pkmu_pair_ref(jnp.array([.1]), jnp.array([1.1]), 1., 1., pd, p)).all()
    assert np.isnan(engine.get_pkmu_pair_ref(jnp.array([.1]), jnp.array([.5]), -1., 1., pd, p)).all()


def test_shot_noise_has_only_monopole_at_identity(engine, spectrum):
    pd, k, _ = spectrum
    p = parameters()
    p["stoch"]["P_shot"] = .6
    for ell, expected in [(0, 2000.), (2, 0.), (4, 0.)]:
        delta = engine.get_pk_ell(k, ell, pd, p)-engine.get_pk_ell_pair(k, ell, pd, p)
        np.testing.assert_allclose(delta, expected, rtol=2.e-12, atol=2.e-8)


@pytest.mark.parametrize("ell,num", [(6, 7), (8, 7), (8, 8)])
def test_underresolved_high_multipoles_are_rejected(engine, spectrum, ell, num):
    pd, k, _ = spectrum
    with pytest.raises(ValueError, match="angular nodes"):
        engine.get_pk_ell(k, ell, pd, parameters(), num=num)


@pytest.mark.parametrize("ell,num", [(6, 8), (8, 9)])
def test_high_multipole_fog_matches_analytic_projection(engine, spectrum, ell, num):
    pd, k, plin = spectrum
    p = parameters()
    def power(fog):
        varied = {**p, "ctr": {**p["ctr"], "cfog": fog}}
        return engine.get_pk_ell(k, ell, pd, varied, num=num)
    derivative = jax.jacfwd(power)(jnp.asarray(0.))
    # Exact Legendre coefficients of mu^6 and mu^8, independent of quadrature.
    f, b = p["f"], p["bias"]["b1"]
    coefficient = (2*b*f*16/231 + f**2*64/495) if ell == 6 else f**2*128/6435
    expected = -k**4 * plin * f**4 * coefficient
    np.testing.assert_allclose(derivative, expected, rtol=2.e-11, atol=1.e-12)


def test_compiled_multipole_accepts_dynamic_order(engine, spectrum):
    pd, k, _ = spectrum
    p = parameters()
    compiled = jax.jit(engine.get_pk_ell)
    for ell in (0, 2, 4, 6, 8):
        np.testing.assert_allclose(compiled(k, ell, pd, p),
                                   engine.get_pk_ell(k, ell, pd, p),
                                   rtol=2.e-11, atol=1.e-9)
    for ell in (-2, 1, 10):
        assert np.isnan(compiled(k, ell, pd, p)).all()
    low_nodes = jax.jit(lambda ell: engine.get_pk_ell(k, ell, pd, p, num=7))
    assert np.isfinite(low_nodes(4)).all()
    assert np.isnan(low_nodes(6)).all()
    assert np.isnan(low_nodes(8)).all()


@pytest.mark.parametrize("warm", [False, True])
def test_engine_configuration_is_immutable_before_and_after_compilation(spectrum, warm):
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from elf.fftlog import FFTlogSettings
    engine = ELFPowerSpectrum()
    pd, k, _ = spectrum
    compiled = jax.jit(engine.get_pk_ell)
    if warm:
        before = np.asarray(compiled(k, 0, pd, parameters()))
    settings = engine.numerical_settings()
    changes = [(engine, "_nfft", 1024), (engine, "query_bounds", (.01, .2)),
               (engine, "method", "matrix"), (engine, "model", object()),
               (engine, "do_irres", True), (engine.model, "_nfft", 1024),
               (engine.model, "subtract_k0_const", False),
               (engine.model, "do_irres", True), (engine.model, "pad_mode", "power-law"),
               (engine.model, "fftlog_settings", FFTlogSettings(nu=1.2)),
               (engine.model.fftlog_settings, "nu", 1.2),
               (engine.model._counterterms, "base", "linear_ir_resum"),
               (engine.model._hankel, "pad_mode", "power-law")]
    for target, name, value in changes:
        with pytest.raises(AttributeError, match="immutable|frozen|cannot assign"):
            setattr(target, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        del engine.model
    assert engine.numerical_settings() == settings
    assert settings["fft_n"] == engine.model._nfft
    assert settings["do_irres"] == engine.model.do_irres
    assert settings["subtract_k0_const"] == engine.model.subtract_k0_const
    assert settings["counterterm_base"] == engine.model.counterterm_base
    if warm:
        np.testing.assert_array_equal(compiled(k, 0, pd, parameters()), before)


def test_nonidentity_ap_counterterm_contraction_at_true_coordinates(engine, spectrum):
    pd, k, _ = spectrum
    mu = jnp.array([-.9, -.3, 0., .3, .9])
    base = parameters()
    base['bias2'] = dict(b1=1.3, b2=-.2, bG2=-.1, bGamma3=.05)
    base['ctr2'] = dict(c0=0., c2=0., c4=0., cfog=0.)
    varied = deepcopy(base)
    varied['ctr'] = dict(c0=4., c2=-7., c4=9., cfog=110.)
    varied['ctr2'] = dict(c0=-2., c2=3., c4=5., cfog=250.)
    da = field_from_projection(varied['ctr'], 1.8, .81)
    db = field_from_projection(varied['ctr2'], 1.3, .81)
    def direct(scales):
        ap, al = scales
        kt = k[:, None]*jnp.sqrt((1-mu[None, :]**2)/ap**2+mu[None, :]**2/al**2)
        mt = k[:, None]*mu[None, :]/al/kt
        plin = engine.model._pk_at(kt, jnp.stack((pd['k'], pd['pk'])))
        za, zb = 1.8+.81*mt**2, 1.3+.81*mt**2
        fa = da[0]+da[1]*mt**2+da[2]*mt**4
        fb = db[0]+db[1]*mt**2+db[2]*mt**4
        return (kt**2*plin*(za*fb+zb*fa)-180*kt**4*plin*.81**4*mt**4*za*zb)/(ap**2*al)
    def delta(scales):
        return engine.get_pkmu_pair_ref(k, mu, *scales, pd, varied)-engine.get_pkmu_pair_ref(k, mu, *scales, pd, base)
    point = jnp.array([1.04, .97])
    np.testing.assert_allclose(delta(point), direct(point), rtol=2.e-10, atol=2.e-9)
    np.testing.assert_allclose(jax.jacfwd(delta)(point), jax.jacfwd(direct)(point), rtol=2.e-10, atol=2.e-8)
