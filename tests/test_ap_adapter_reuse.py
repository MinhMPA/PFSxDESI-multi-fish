"""Exact AP adapter identity without caching a response's physical inputs."""

import gc
import weakref
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog import revision_theory as rt
from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec, validate_coordinate_neighborhood
from ps_1loop_jax import PowerSpectrum1Loop


class AnalyticBackend:
    """Polynomial power through the real backend's JIT and AP interpolation."""

    __hash__ = None  # Reuse must not require a hashable physical backend.

    def __eq__(self, other):
        raise AssertionError("backend structural equality is not a reuse key")

    def get_pkmu_pair(self, k, mu, pk_data, params, add_stochasticity=False):
        def coefficient(bias, ctr):
            return (sum(w * bias[name] for w, name in enumerate(
                ('b1', 'b2', 'bG2', 'bGamma3'), 1))
                + sum(w * ctr[name] for w, name in enumerate(
                    ('c0', 'c2', 'c4', 'cfog'), 5)) * mu[None, :])

        signal = coefficient(params['bias'], params['ctr'])
        if 'bias2' in params:
            signal += 2 * coefficient(params['bias2'], params['ctr2'])
        result = pk_data['amplitude'] * k[:, None] * (
            signal + params['f'] * mu[None, :]**2)
        if add_stochasticity:
            s = params['stoch']
            result += (s['P_shot'] + s['a0'] * (k[:, None] / params['k_nl'])**2
                       + s['a2'] * (k[:, None] * mu[None, :] / params['k_nl'])**2) / params['ndens']
        return result


@pytest.fixture
def response_factory(monkeypatch):
    # Only replace the external emulator factories. PairResponse construction,
    # cosmology/background, nuisance mapping, JIT and interpolation stay real.
    monkeypatch.setattr(rt, 'make_plin_func', lambda: lambda z, c: {
        'amplitude': jnp.exp(.2 * c['ln10_10_As']) + .4 * c['mnu'] + .5 * c['Omega_m'] + .1 * z})
    monkeypatch.setattr(rt, 'make_sigma8_func', lambda: lambda z, c:
                        jnp.exp(.1 * c['ln10_10_As'] + .2 * c['mnu'] + .3 * c['Omega_m'] + .05 * z))
    monkeypatch.setattr(rt, 'make_growth_rate_func', lambda: lambda z, c:
                        .5 + .4 * c['Omega_m'] + .1 * c['mnu'] + .01 * z)

    def make(backend, **kwargs):
        settings = dict(z=.9, reference=dict(FIDUCIAL), nbar_a=2e-4,
                        nbar_b=4e-4, k_nl=.7, ap_nk=32, ap_nmu=17)
        settings.update(kwargs)
        return rt.make_pair_response(backend, **settings)

    return make


def test_fresh_physical_contexts_reuse_the_same_live_static_adapter(response_factory):
    backend = AnalyticBackend()
    first = response_factory(backend)
    changed_reference = {**FIDUCIAL, 'ln10_10_As': FIDUCIAL['ln10_10_As'] + .03}
    second = response_factory(backend, z=1.3, reference=changed_reference,
                              nbar_a=8e-4, nbar_b=None, k_nl=.8)
    assert second.ap_backend is first.ap_backend
    assert second.z != first.z
    assert second.reference != first.reference
    assert second.nbar_a != first.nbar_a


def test_backend_identity_and_both_grid_dimensions_separate_adapters(response_factory):
    backend = AnalyticBackend()
    first = response_factory(backend)
    contexts = [response_factory(AnalyticBackend()),
                response_factory(backend, ap_nk=64),
                response_factory(backend, ap_nmu=33)]
    assert all(response.ap_backend is not first.ap_backend for response in contexts)
    assert len({id(response.ap_backend) for response in contexts}) == 3
    assert first.ap_backend.backend is backend


def test_x64_context_cannot_reuse_stale_grid_constants(response_factory):
    backend = AnalyticBackend()
    with jax.enable_x64(True):
        full = response_factory(backend)
    with jax.enable_x64(False):
        narrow = response_factory(backend)
    with jax.enable_x64(True):
        full_again = response_factory(backend)
    assert full.ap_backend._kn.dtype == jnp.float64
    assert narrow.ap_backend._kn.dtype == jnp.float32
    assert full.ap_backend is not narrow.ap_backend
    assert full_again.ap_backend is full.ap_backend


@pytest.mark.parametrize('name,value', [
    ('ap_nk', 32.), ('ap_nmu', 17.), ('ap_nk', True),
    ('ap_nmu', False), ('ap_nk', 3), ('ap_nmu', 0),
])
def test_invalid_dimensions_cannot_hit_an_existing_valid_context(response_factory, name, value):
    backend = AnalyticBackend()
    live = response_factory(backend)
    with pytest.raises((TypeError, ValueError)):
        response_factory(backend, **{name: value})
    assert live.ap_backend.backend is backend


def test_numpy_integer_dimensions_use_the_same_exact_grid(response_factory):
    backend = AnalyticBackend()
    first = response_factory(backend)
    second = response_factory(backend, ap_nk=np.int64(32), ap_nmu=np.int32(17))
    assert first.ap_backend is second.ap_backend


@pytest.mark.parametrize('attribute', ['backend', '_kn', '_mu'])
@pytest.mark.parametrize('operation', ['assign', 'delete'])
def test_shared_backend_and_grid_associations_are_immutable(attribute, operation):
    adapter = rt.APInterpolationBackend(AnalyticBackend(), 32, 17)
    original = getattr(adapter, attribute)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        if operation == 'assign':
            setattr(adapter, attribute, object())
        else:
            delattr(adapter, attribute)
    assert getattr(adapter, attribute) is original


def test_registry_does_not_own_unused_adapters_or_backends(response_factory):
    backend = AnalyticBackend()
    response = response_factory(backend)
    adapter_ref = weakref.ref(response.ap_backend)
    backend_ref = weakref.ref(backend)
    del response, backend
    gc.collect()
    assert adapter_ref() is None
    assert backend_ref() is None


def test_none_preserves_original_backend_without_grid_validation(response_factory):
    backend = AnalyticBackend()
    response = response_factory(backend, ap_nk=None, ap_nmu='unused')
    assert response.ap_backend is backend


def test_repeated_fresh_responses_add_only_one_static_jit_context(response_factory):
    backend = AnalyticBackend()
    live = response_factory(backend)
    interpolator = PowerSpectrum1Loop._get_pkmu_pair_ref_interp
    before = interpolator._cache_size()
    k, mu = jnp.asarray([.07, .13]), jnp.asarray([.2, .8])
    theta = jnp.asarray([FIDUCIAL['ln10_10_As'], FIDUCIAL['mnu'], FIDUCIAL['Omega_m']])
    q = jnp.arange(1., 12.) / 10
    original = np.asarray(live.pkmu(k, mu, theta, q))
    compiled = interpolator._cache_size()
    assert compiled == before + 1
    for _ in range(4):
        fresh = response_factory(backend)
        np.testing.assert_array_equal(fresh.pkmu(k, mu, theta, q), original)
        assert fresh.ap_backend is live.ap_backend
        assert interpolator._cache_size() == compiled


def test_identity_token_collision_cannot_return_another_backend(response_factory, monkeypatch):
    # Force an id-token collision while both backend objects remain alive;
    # correctness must also validate the actual association on a cache hit.
    monkeypatch.setattr(rt, 'id', lambda obj: 7, raising=False)
    first_backend, second_backend = AnalyticBackend(), AnalyticBackend()
    first = response_factory(first_backend)
    second = response_factory(second_backend)
    again = response_factory(first_backend)
    assert second.ap_backend is not first.ap_backend
    assert second.ap_backend.backend is second_backend
    assert again.ap_backend.backend is first_backend


@pytest.mark.parametrize('n_k', [8192, 16384, 32768])
def test_required_table_nodes_keep_exact_original_grid_construction(response_factory, n_k):
    backend = AnalyticBackend()
    cold_adapters = []

    def construct_inside_jit(scale):
        response = response_factory(backend, ap_nk=n_k, ap_nmu=101)
        cold_adapters.append(response.ap_backend)
        return scale * response.ap_backend._kn[-1]

    np.asarray(jax.jit(construct_inside_jit)(2.))
    response = response_factory(backend, ap_nk=n_k, ap_nmu=101)
    assert response.ap_backend is cold_adapters[0]
    np.testing.assert_array_equal(response.ap_backend._kn, jnp.geomspace(.002, .5, n_k))
    np.testing.assert_array_equal(response.ap_backend._mu, jnp.linspace(0., 1., 101))


def _analytic_mean(packed, k, mu, z, reference, nbar_a, nbar_b, k_nl, cross):
    """Independent closed polynomial, with no adapter or nuisance helper."""
    ln_as, mnu, omega_m = packed[:3]
    h = reference['h']
    omega_b = reference['omega_b']
    omega_cdm = omega_m * h**2 - omega_b - mnu / 93.14
    da = rt.bg.angular_diameter_distance(omega_b, omega_cdm, h, z, mnu)
    da_ref = rt.bg.angular_diameter_distance(
        omega_b, reference['omega_cdm'], h, z, reference['mnu'])
    hz = rt.bg.Hz(omega_b, omega_cdm, h, z, mnu)
    hz_ref = rt.bg.Hz(omega_b, reference['omega_cdm'], h, z, reference['mnu'])
    perpendicular, parallel = da / da_ref, hz_ref / hz
    factor = jnp.sqrt(1 + mu**2 * ((perpendicular / parallel)**2 - 1))
    kt = k[:, None] * factor[None, :] / perpendicular
    mt = mu[None, :] * perpendicular / parallel / factor[None, :]
    volume = perpendicular**2 * parallel
    sigma = jnp.exp(.1 * ln_as + .2 * mnu + .3 * omega_m + .05 * z)
    sigma_ref = jnp.exp(.1 * reference['ln10_10_As'] + .2 * reference['mnu']
                        + .3 * reference['Omega_m'] + .05 * z)
    amplitude_ratio = (sigma / sigma_ref)**2
    f = .5 + .4 * omega_m + .1 * mnu + .01 * z
    power = jnp.exp(.2 * ln_as) + .4 * mnu + .5 * omega_m + .1 * z

    def signal(q):
        b1, b2, bg2, gamma, q0, q2, q4, fog = q[:8]
        bias = b1 / sigma + 2 * b2 / sigma**2 + 3 * bg2 / sigma**2 + 4 * gamma / amplitude_ratio
        counterterm = (5 * (q0 - f * q2 / 3 + 3 * f**2 * q4 / 35)
                       + 6 * (q2 - 6 * f * q4 / 7) + 7 * q4) / amplitude_ratio
        counterterm += 8 * fog / amplitude_ratio**2
        return bias + counterterm * mt

    qa = packed[3:14]
    coefficient = signal(qa)
    if cross:
        coefficient += 2 * signal(packed[14:25])
    value = power * kt * (coefficient + f * mt**2)
    if cross:
        noise = (packed[25] + packed[26] * (kt * mt / k_nl)**2) / jnp.sqrt(nbar_a * nbar_b)
    else:
        noise = (qa[8] + qa[9] * (kt / k_nl)**2 + qa[10] * (kt * mt / k_nl)**2) / nbar_a
    return (value + noise) / volume


@pytest.mark.parametrize('cross', [False, True], ids=['auto14', 'cross27'])
def test_shared_adapter_keeps_means_and_all_tangents_dynamic(response_factory, cross):
    backend = AnalyticBackend()
    settings = [dict(z=.9, reference=dict(FIDUCIAL), nbar_a=2e-4, nbar_b=4e-4, k_nl=.7),
                dict(z=1.3, reference={**FIDUCIAL, 'ln10_10_As': FIDUCIAL['ln10_10_As'] + .03},
                     nbar_a=6e-4, nbar_b=9e-4, k_nl=.8)]
    # The stochastic k^2 term is approximate under local cubic interpolation
    # on a geometric grid. Use the unchanged default resolution here; the
    # smaller grid suffices for the separate adapter identity regressions.
    responses = [response_factory(backend, ap_nk=2048, **context) for context in settings]
    assert responses[0].ap_backend is responses[1].ap_backend
    k, mu = jnp.asarray([.07, .13]), jnp.asarray([0., .4, 1.])
    means, derivatives = [], []
    for index, (response, context) in enumerate(zip(responses, settings)):
        ref = context['reference']
        theta = jnp.asarray([ref['ln10_10_As'] + .07, ref['mnu'] + .004, ref['Omega_m'] + .01])
        qa = jnp.arange(1., 12.) / 10 + .02 * index
        packed = jnp.concatenate((theta, qa, *((qa + .4, jnp.asarray([.3, -.2])) if cross else ())))

        def evaluate(p):
            return response.pkmu(k, mu, p[:3], p[3:14],
                                 p[14:25] if cross else None, p[25:] if cross else None)

        def expected(p):
            return _analytic_mean(p, k, mu, **context, cross=cross)

        actual = np.asarray(evaluate(packed))
        ad = np.asarray(jax.jacfwd(evaluate)(packed))
        np.testing.assert_allclose(actual, expected(packed), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ad, jax.jacfwd(expected)(packed), rtol=1e-7, atol=1e-7)
        assert ad.shape == (2, 3, 27 if cross else 14)
        # Cross spectra intentionally ignore each tracer's auto-noise inputs;
        # their full tangent slots remain present and exactly zero.
        inactive = [11, 12, 13, 22, 23, 24] if cross else []
        active = [i for i in range(len(packed)) if i not in inactive]
        assert np.all(np.linalg.norm(ad[:, :, active], axis=(0, 1)) > 0)
        if inactive:
            np.testing.assert_array_equal(ad[:, :, inactive], 0.)
        step = 1e-5
        fd = np.stack([(np.asarray(evaluate(packed.at[i].add(step)))
                        - np.asarray(evaluate(packed.at[i].add(-step)))) / (2 * step)
                       for i in range(len(packed))], axis=-1)
        np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=2e-5)
        # Fresh construction must preserve both primal and full AD bitwise.
        fresh = response_factory(backend, ap_nk=2048, **context)
        assert fresh.ap_backend is response.ap_backend
        fresh_function = lambda p: fresh.pkmu(
            k, mu, p[:3], p[3:14], p[14:25] if cross else None, p[25:] if cross else None)
        np.testing.assert_array_equal(fresh_function(packed), actual)
        np.testing.assert_array_equal(jax.jacfwd(fresh_function)(packed), ad)
        means.append(actual)
        derivatives.append(ad)
    assert not np.array_equal(means[0], means[1])
    assert not np.array_equal(derivatives[0], derivatives[1])


@pytest.mark.parametrize('cross', [False, True], ids=['auto', 'cross'])
def test_shared_adapter_keeps_domain_rejections_live(response_factory, cross):
    backend = AnalyticBackend()
    first, second = response_factory(backend), response_factory(backend, z=1.3)
    assert first.ap_backend is second.ap_backend
    theta = jnp.asarray([FIDUCIAL['ln10_10_As'], FIDUCIAL['mnu'], FIDUCIAL['Omega_m']])
    q = jnp.arange(1., 12.) / 10
    for response in (first, second):
        def evaluate(k, mu):
            return response.pkmu(jnp.asarray(k), jnp.asarray(mu), theta, q,
                                 q + .4 if cross else None,
                                 jnp.asarray([.3, -.2]) if cross else None)
        assert np.isfinite(evaluate([.07], [-1., 0., 1.])).all()
        assert np.isnan(evaluate([.0005, .7], [.4])).all()
        assert np.isnan(evaluate([.07], [1.001])).all()


def test_backend_failure_propagates_through_reused_adapter(response_factory):
    class FailingBackend(AnalyticBackend):
        def get_pkmu_pair(self, *args, **kwargs):
            raise RuntimeError('analytic backend failed')

    backend = FailingBackend()
    first, second = response_factory(backend), response_factory(backend)
    assert first.ap_backend is second.ap_backend
    theta = jnp.asarray([FIDUCIAL['ln10_10_As'], FIDUCIAL['mnu'], FIDUCIAL['Omega_m']])
    for response in (first, second):
        with pytest.raises(RuntimeError, match='analytic backend failed'):
            response.pkmu(jnp.asarray([.07]), jnp.asarray([.4]), theta, jnp.arange(1., 12.) / 10)


@pytest.mark.parametrize('cross', [False, True], ids=['auto14', 'cross27'])
def test_cold_jitted_public_response_keeps_reusable_grids_concrete(response_factory, cross):
    backend = AnalyticBackend()  # Never warm this backend before the first JIT.
    context = dict(z=.9, reference=dict(FIDUCIAL), nbar_a=2e-4, nbar_b=4e-4, k_nl=.7)
    k, mu = jnp.asarray([.07, .13]), jnp.asarray([0., .4, 1.])
    theta = jnp.asarray([FIDUCIAL['ln10_10_As'] + .07, FIDUCIAL['mnu'] + .004,
                         FIDUCIAL['Omega_m'] + .01])
    qa = jnp.arange(1., 12.) / 10
    packed = jnp.concatenate((theta, qa, *((qa + .4, jnp.asarray([.3, -.2])) if cross else ())))
    adapters_seen = []

    def evaluate(p):
        response = response_factory(backend, ap_nk=2048, **context)
        # Observe only the static adapter, never retain traced response state.
        adapters_seen.append(weakref.ref(response.ap_backend))
        return response.pkmu(k, mu, p[:3], p[3:14],
                             p[14:25] if cross else None, p[25:] if cross else None)

    def expected(p):
        return _analytic_mean(p, k, mu, **context, cross=cross)

    compiled = jax.jit(evaluate)
    cold_mean = np.asarray(compiled(packed))
    changed = packed.at[0].add(.01).at[3].add(.02)
    eager_mean = np.asarray(evaluate(changed))
    np.testing.assert_allclose(cold_mean, expected(packed), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(eager_mean, expected(changed), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(compiled(changed), eager_mean, rtol=1e-12, atol=1e-8)
    assert not np.array_equal(cold_mean, eager_mean)

    # A separate full-width transform must also reuse the cold-created adapter
    # without referencing tracers from the completed primal trace.
    ad = np.asarray(jax.jit(jax.jacfwd(evaluate))(changed))
    np.testing.assert_allclose(ad, jax.jacfwd(expected)(changed), rtol=1e-7, atol=1e-7)
    assert ad.shape == (2, 3, 27 if cross else 14)
    live = response_factory(backend, ap_nk=2048, **context).ap_backend
    assert all(observed() is live for observed in adapters_seen)
    for grid in (live._kn, live._mu):
        assert isinstance(grid, jax.Array)
        assert not isinstance(grid, jax.core.Tracer)
    np.testing.assert_array_equal(live._kn, jnp.geomspace(.002, .5, 2048))
    np.testing.assert_array_equal(live._mu, jnp.linspace(0., 1., 17))


@pytest.fixture
def extended_response_factory(response_factory, monkeypatch):
    # P_phys(k_phys)=A*k_phys implies P_ref(k_ref)=A*h_ref**4*k_ref.
    # Patch the explicit-space leaf, keeping mapping, backgrounds, nuisance
    # equations and real backend interpolation/JIT in the production path.
    def reference_power(h_ref):
        return lambda z, c: {'amplitude': h_ref**4 * (
            jnp.exp(.2*c['ln10_10_As']) + .4*c['mnu'] + .5*c['Omega_m']
            + .1*z + .6*c['h'] + 7*c['omega_b'] + .8*c['n_s'])}

    monkeypatch.setattr(rt, 'make_reference_plin_func', reference_power)
    monkeypatch.setattr(rt, 'make_sigma8_func', lambda: lambda z, c:
        jnp.exp(.1*c['ln10_10_As'] + .2*c['mnu'] + .3*c['Omega_m']
                + .05*z + .4*c['h'] + 5*c['omega_b'] + .6*c['n_s']))
    monkeypatch.setattr(rt, 'make_growth_rate_func', lambda: lambda z, c:
        .5 + .4*c['Omega_m'] + .1*c['mnu'] + .01*z
        + .2*c['h'] + 3*c['omega_b'] + .4*c['n_s'])
    return response_factory


def _theta_for_space(reference, space):
    keys = {'Mnu': 'mnu', 'Omegam': 'Omega_m'}
    return jnp.asarray([reference[keys.get(name, name)] for name in space.names])


def _sampled_square(query, knots):
    """Closed error of quadratic Hermite interpolation with secant slopes.

    Linear/constant polynomials are exact. For x^2, endpoint secants differ
    from2x by the adjacent interval; the average of interior secants differs
    by half the second knot difference. Multiply just these slope errors by
    the two Hermite derivative basis polynomials, without an interpolator.
    """
    error = jnp.concatenate((knots[1:2]-knots[:1],
        (knots[:-2]+knots[2:])/2-knots[1:-1], knots[-2:-1]-knots[-1:]))
    right = jnp.clip(jnp.searchsorted(knots, query, side='right'), 1, len(knots)-1)
    width = knots[right]-knots[right-1]
    t = (query-knots[right-1])/width
    return query**2 + width*(error[right-1]*t*(1-t)**2-error[right]*t**2*(1-t))


def _extended_analytic_mean(packed, k, mu, *, z, reference, nbar_a, nbar_b,
                            k_nl, space, cross, sampled=False):
    """Independent physical polynomial in fixed h_ref units, all17/30 slots."""
    nc = len(space.names)
    ln_as, mnu, omega_m = packed[:3]
    h, omega_b, n_s = (packed[3:6] if nc == 6 else
                       (reference['h'], reference['omega_b'], reference['n_s']))
    omega_cdm = omega_m*h**2 - omega_b - mnu/93.14
    trial = (omega_b, omega_cdm, h, z, mnu)
    frozen = (reference['omega_b'], reference['omega_cdm'], reference['h'],
              z, reference['mnu'])
    perpendicular = rt.bg.angular_diameter_distance(*trial)/rt.bg.angular_diameter_distance(*frozen)
    parallel = rt.bg.Hz(*frozen)/rt.bg.Hz(*trial)
    mu = jnp.abs(mu)
    factor = jnp.sqrt(1+mu**2*((perpendicular/parallel)**2-1))
    kt = k[:, None]*factor[None, :]/perpendicular
    mt = mu[None, :]*perpendicular/parallel/factor[None, :]
    # The analytic backend contains only1,k,k^2 and1,mu,mu^2. Tensor-product
    # interpolation acts separably, so the closed sampled-square corrections
    # account for the entire finite-grid error without calling any adapter.
    ksq = _sampled_square(kt, jnp.geomspace(.002, .5, 2048)) if sampled else kt**2
    musq = _sampled_square(mt, jnp.linspace(0., 1., 17)) if sampled else mt**2
    volume = perpendicular**2*parallel
    sigma = jnp.exp(.1*ln_as+.2*mnu+.3*omega_m+.05*z+.4*h+5*omega_b+.6*n_s)
    sigma_ref = jnp.exp(.1*reference['ln10_10_As']+.2*reference['mnu']
        +.3*reference['Omega_m']+.05*z+.4*reference['h']
        +5*reference['omega_b']+.6*reference['n_s'])
    amplitude = (sigma/sigma_ref)**2
    f = .5+.4*omega_m+.1*mnu+.01*z+.2*h+3*omega_b+.4*n_s
    power = space.h_ref**4*(jnp.exp(.2*ln_as)+.4*mnu+.5*omega_m+.1*z
                             +.6*h+7*omega_b+.8*n_s)

    def signal(q):
        b1, b2, bg2, gamma, q0, q2, q4, fog = q[:8]
        bias = b1/sigma+2*b2/sigma**2+3*bg2/sigma**2+4*gamma/amplitude
        counterterm = (5*(q0-f*q2/3+3*f**2*q4/35)
                       +6*(q2-6*f*q4/7)+7*q4)/amplitude
        counterterm += 8*fog/amplitude**2
        return bias+counterterm*mt

    qa = packed[nc:nc+11]
    coefficient = signal(qa)
    if cross:
        coefficient += 2*signal(packed[nc+11:nc+22])
        noise = (packed[nc+22]+packed[nc+23]*(kt*mt/k_nl)**2)/jnp.sqrt(nbar_a*nbar_b)
    else:
        noise = (qa[8]+qa[9]*ksq/k_nl**2+qa[10]*ksq*musq/k_nl**2)/nbar_a
    return (power*kt*(coefficient+f*musq)+noise)/volume


def _extended_evaluate(response, packed, k, mu, cross, *, multipoles=False):
    nc = len(response.space.names)
    qa = packed[nc:nc+11]
    qb = packed[nc+11:nc+22] if cross else None
    qc = packed[nc+22:nc+24] if cross else None
    if multipoles:
        return response.multipoles(k, packed[:nc], qa, qb, qc, n_mu=32)
    return response.pkmu(k, mu, packed[:nc], qa, qb, qc)


def _extended_packed(reference, space, cross, offset=0.):
    theta = _theta_for_space(reference, space) + jnp.asarray(
        [.007, .004, .001, .0005, .00005, .0005])
    validate_coordinate_neighborhood(np.asarray(theta), reference, space)
    qa = jnp.arange(1., 12.)/10 + offset
    return jnp.concatenate((theta, qa, *((qa+.4, jnp.asarray([.3, -.2])) if cross else ())))


@pytest.mark.parametrize('cross', [False, True], ids=['auto17', 'cross30'])
@pytest.mark.parametrize('multipoles', [False, True], ids=['pkmu', 'multipoles'])
def test_extended_shared_adapter_keeps_every_native_tangent_dynamic(
        extended_response_factory, cross, multipoles):
    backend = AnalyticBackend()
    changed_reference = dict(FIDUCIAL, h=FIDUCIAL['h']+.001,
        omega_b=FIDUCIAL['omega_b']+.0001, n_s=FIDUCIAL['n_s']+.001)
    changed_reference['omega_cdm'] = (changed_reference['Omega_m']*changed_reference['h']**2
        -changed_reference['omega_b']-changed_reference['mnu']/93.14)
    contexts = [dict(z=.9, reference=dict(FIDUCIAL), nbar_a=2e-4, nbar_b=4e-4, k_nl=.7),
                dict(z=1.3, reference=changed_reference, nbar_a=6e-4, nbar_b=9e-4, k_nl=.8)]
    for context in contexts:
        context['space'] = make_cosmology_spec(context['reference'], extended=True)
    responses = [extended_response_factory(backend, ap_nk=2048, **c) for c in contexts]
    assert responses[0].ap_backend is responses[1].ap_backend
    assert responses[0].space is not responses[1].space
    assert responses[0].space.h_ref != responses[1].space.h_ref
    k, mu = jnp.asarray([.07, .13]), jnp.asarray([0., .4, 1.])
    nodes, weights = np.polynomial.legendre.leggauss(32)
    projection_mu, weights = jnp.asarray((nodes+1)/2), jnp.asarray(weights/2)
    legendre = jnp.stack((jnp.ones_like(projection_mu), (3*projection_mu**2-1)/2,
                          (35*projection_mu**4-30*projection_mu**2+3)/8))
    means, tangents = [], []
    for index, (response, context) in enumerate(zip(responses, contexts)):
        packed = _extended_packed(context['reference'], context['space'], cross, .02*index)
        assert len(packed) == (30 if cross else 17)

        def evaluate(p):
            return _extended_evaluate(response, p, k, mu, cross, multipoles=multipoles)

        def expected(p):
            if multipoles:
                # The continuous physical oracle is checked by the pkmu cases;
                # near table endpoints use its independently derived finite-grid
                # polynomial, with the identical unchanged comparison tolerances.
                field = _extended_analytic_mean(p, k, projection_mu,
                                                **context, cross=cross, sampled=True)
                return jnp.einsum('km,lm,m,l->kl', field, legendre, weights,
                                  jnp.asarray([1., 5., 9.]))
            return _extended_analytic_mean(p, k, mu, **context, cross=cross)

        actual = np.asarray(evaluate(packed))
        ad = np.asarray(jax.jacfwd(evaluate)(packed))
        assert ad.shape == (2, 3, 30 if cross else 17)
        np.testing.assert_allclose(actual, expected(packed), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ad, jax.jacfwd(expected)(packed), rtol=1e-7, atol=1e-7)
        inactive = [14, 15, 16, 25, 26, 27] if cross else []
        active = [i for i in range(len(packed)) if i not in inactive]
        assert np.all(np.linalg.norm(ad[:, :, active], axis=(0, 1)) > 0)
        if inactive:
            np.testing.assert_array_equal(ad[:, :, inactive], 0.)
        step = 1e-5
        fd = np.stack([(np.asarray(evaluate(packed.at[i].add(step)))
                        -np.asarray(evaluate(packed.at[i].add(-step))))/(2*step)
                       for i in range(len(packed))], axis=-1)
        np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=2e-5)
        fresh = extended_response_factory(backend, ap_nk=2048, **context)
        assert fresh.ap_backend is response.ap_backend
        fresh_function = lambda p: _extended_evaluate(fresh, p, k, mu, cross, multipoles=multipoles)
        np.testing.assert_array_equal(fresh_function(packed), actual)
        np.testing.assert_array_equal(jax.jacfwd(fresh_function)(packed), ad)
        means.append(actual)
        tangents.append(ad)
    assert not np.array_equal(means[0], means[1])
    assert not np.array_equal(tangents[0], tangents[1])


@pytest.mark.parametrize('cross', [False, True], ids=['auto17', 'cross30'])
def test_extended_cold_public_jit_keeps_full_ad_and_eager_use_live(extended_response_factory, cross):
    backend = AnalyticBackend()  # First construction happens inside the primal JIT.
    context = dict(z=.9, reference=dict(FIDUCIAL), nbar_a=2e-4, nbar_b=4e-4, k_nl=.7,
                   space=make_cosmology_spec(FIDUCIAL, extended=True))
    packed = _extended_packed(context['reference'], context['space'], cross)
    k, mu = jnp.asarray([.07, .13]), jnp.asarray([0., .4, 1.])
    observed = []

    def evaluate(p):
        response = extended_response_factory(backend, ap_nk=2048, **context)
        observed.append(weakref.ref(response.ap_backend))
        return _extended_evaluate(response, p, k, mu, cross)

    expected = lambda p: _extended_analytic_mean(p, k, mu, **context, cross=cross)
    compiled = jax.jit(evaluate)
    cold = np.asarray(compiled(packed))
    changed = packed.at[3].add(.0001).at[4].add(.00002).at[5].add(.0002).at[6].add(.02)
    validate_coordinate_neighborhood(np.asarray(changed[:6]), context['reference'], context['space'])
    eager = np.asarray(evaluate(changed))
    np.testing.assert_allclose(cold, expected(packed), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(eager, expected(changed), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(compiled(changed), eager, rtol=1e-12, atol=1e-8)
    assert not np.array_equal(cold, eager)
    ad = np.asarray(jax.jit(jax.jacfwd(evaluate))(changed))
    assert ad.shape == (2, 3, 30 if cross else 17)
    np.testing.assert_allclose(ad, jax.jacfwd(expected)(changed), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(ad, jax.jacfwd(evaluate)(changed), rtol=1e-12, atol=1e-8)
    live = extended_response_factory(backend, ap_nk=2048, **context).ap_backend
    assert all(reference() is live for reference in observed)
    for grid in (live._kn, live._mu):
        assert isinstance(grid, jax.Array) and not isinstance(grid, jax.core.Tracer)
    np.testing.assert_array_equal(live._kn, jnp.geomspace(.002, .5, 2048))
    np.testing.assert_array_equal(live._mu, jnp.linspace(0., 1., 17))


def test_declared_spaces_share_only_static_adapter_and_keep_own_mapping(extended_response_factory):
    backend = AnalyticBackend()
    baseline = make_cosmology_spec(FIDUCIAL, extended=False)
    extended = make_cosmology_spec(FIDUCIAL, extended=True)
    other_reference = dict(FIDUCIAL, h=FIDUCIAL['h']+.001)
    other_reference['omega_cdm'] = (other_reference['Omega_m']*other_reference['h']**2
        -other_reference['omega_b']-other_reference['mnu']/93.14)
    other = make_cosmology_spec(other_reference, extended=True)
    # Identity-only fixtures use32 k nodes; numeric oracle comparisons need
    # the same2048-node resolution as the independent primal/AD cases.
    responses = [extended_response_factory(backend, space=space, reference=ref, ap_nk=2048)
        for space, ref in ((None, FIDUCIAL), (baseline, FIDUCIAL),
                           (extended, FIDUCIAL), (other, other_reference))]
    assert all(r.ap_backend is responses[0].ap_backend for r in responses)
    assert [r.space for r in responses] == [None, baseline, extended, other]
    theta = _theta_for_space(FIDUCIAL, extended).at[3].add(.0005).at[4].add(.00005).at[5].add(.0005)
    k, mu, qa = jnp.asarray([.07]), jnp.asarray([.4]), jnp.arange(1., 12.)/10
    for response in responses[1:]:
        nc = len(response.space.names)
        packed = jnp.concatenate((theta[:nc], qa))
        context = dict(z=response.z, reference=response.reference, nbar_a=response.nbar_a,
                       nbar_b=response.nbar_b, k_nl=response.k_nl, space=response.space)
        np.testing.assert_allclose(_extended_evaluate(response, packed, k, mu, False),
            _extended_analytic_mean(packed, k, mu, **context, cross=False), rtol=1e-8, atol=1e-8)
    assert responses[1]._cosmology(theta[:3])['h'] == FIDUCIAL['h']
    assert responses[2]._cosmology(theta)['h'] == theta[3]
    assert not np.array_equal(responses[1].pkmu(k, mu, theta[:3], qa), responses[2].pkmu(k, mu, theta, qa))
    assert not np.array_equal(responses[2].pkmu(k, mu, theta, qa), responses[3].pkmu(k, mu, theta, qa))
    diagnostic = extended_response_factory(backend, space=extended, ap_nk=None, ap_nmu='unused')
    assert diagnostic.ap_backend is backend
    with pytest.raises(ValueError, match='redshift'):
        extended_response_factory(backend, space=extended, z=5.01)


@pytest.mark.parametrize('cross', [False, True], ids=['auto17', 'cross30'])
def test_extended_shared_adapter_preserves_live_domain_rejection(extended_response_factory, cross):
    backend = AnalyticBackend()
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    first = extended_response_factory(backend, space=space)
    second = extended_response_factory(backend, space=space, z=1.3)
    assert first.ap_backend is second.ap_backend
    packed = jnp.concatenate((_theta_for_space(FIDUCIAL, space), jnp.arange(1., 12.)/10,
        *((jnp.arange(1., 12.)/10+.4, jnp.asarray([.3, -.2])) if cross else ())))
    for response in (first, second):
        evaluate = lambda k, mu: _extended_evaluate(response, packed, jnp.asarray(k), jnp.asarray(mu), cross)
        assert np.isfinite(evaluate([.07], [-1., 0., 1.])).all()
        assert np.isnan(evaluate([.0005, .7], [.4])).all()
        assert np.isnan(evaluate([.07], [1.001])).all()
