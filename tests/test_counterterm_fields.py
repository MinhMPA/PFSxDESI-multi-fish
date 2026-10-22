"""Shared tracer fields, independently projected before contraction."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.elf_adapter import parameters_to_elf


def raw(c0=0., c2=0., c4=0., cfog=0.):
    return dict(c0=c0, c2=c2, c4=c4, cfog=cfog)


def field_from_projection(ctr, b, f):
    """Infer physical d from 64-node integrals, without the production matrix."""
    mu, weights = np.polynomial.legendre.leggauss(64)
    basis = np.array([np.ones_like(mu), mu**2, mu**4])
    legs = np.array([np.polynomial.legendre.Legendre.basis(l)(mu) for l in (0, 2, 4)])
    project = legs * weights * np.array([1, 5, 9])[:, None] / 2
    matrix = project @ ((b+f*mu**2)*basis).T
    old = -(ctr['c0']+f*ctr['c2']*mu**2+f*f*ctr['c4']*mu**4)
    return np.linalg.solve(matrix, project @ old)


def angular(ctr, k, mu, plin, f):
    c = jnp.asarray(ctr)
    return -2*k**2*plin*(c[0]+f*c[1]*mu**2+f**2*c[2]*mu**4+f**3*c[3]*mu**6) - 2*k**4*plin*f**4*mu**4*(c[4]+f*c[5]*mu**2+f**2*c[6]*mu**4)


def translated(ca, cb, ba, bb, f):
    bias = lambda b: dict(b1=b, b2=0., bG2=0., bGamma3=0.)
    return parameters_to_elf(dict(h=.67, f=f, bias=bias(ba), bias2=bias(bb), ctr=ca, ctr2=cb)).ctr


def test_unequal_bias_real_space_is_field_contraction_not_power_average():
    actual = angular(translated(raw(c0=1.), raw(), 1., 2., 0.), .1, .4, 100., 0.)
    np.testing.assert_allclose(actual, -2., rtol=1.e-14)


@pytest.mark.parametrize('ba,bb', [(1.2, 2.3), (1.7, 1.7)])
def test_full_angular_counterterm_is_contraction_of_projected_fields(ba, bb):
    ca, cb, f = raw(4., -7., 9., 110.), raw(-2., 3., 5., 250.), .81
    da, db = field_from_projection(ca, ba, f), field_from_projection(cb, bb, f)
    mu = np.linspace(-1., 1., 37)
    k, plin = .13, 850.
    za, zb = ba+f*mu**2, bb+f*mu**2
    expected = k*k*plin*(za*np.polynomial.polynomial.polyval(mu**2, db)+zb*np.polynomial.polynomial.polyval(mu**2, da))
    expected -= .5*(ca['cfog']+cb['cfog'])*k**4*plin*f**4*mu**4*za*zb
    got = angular(translated(ca, cb, ba, bb, f), k, mu, plin, f)
    np.testing.assert_allclose(got, expected, rtol=1.e-11, atol=1.e-11*np.max(np.abs(expected)))


@pytest.mark.parametrize('b,f', [(1., 0.), (1.2, .4), (2.4, .9)])
def test_intrinsic_auto_fitted_multipoles_preserved_and_l6_retained(b, f):
    ca = raw(4., -7., 9.)
    mu, w = np.polynomial.legendre.leggauss(64)
    c = translated(ca, ca, b, b, f)
    got = np.asarray(angular(c, .1, mu, 100., f))
    old = -2*(ca['c0']+f*ca['c2']*mu**2+f*f*ca['c4']*mu**4)
    scale = max(np.max(np.abs(old)), np.max(np.abs(got)))
    for ell in (0, 2, 4):
        proj = (2*ell+1)/2*w*np.polynomial.legendre.Legendre.basis(ell)(mu)
        np.testing.assert_allclose(proj @ got, proj @ old, rtol=1.e-11, atol=1.e-11*scale)
    d = field_from_projection(ca, b, f)
    proj6 = 13/2*w*np.polynomial.legendre.Legendre.basis(6)(mu)
    np.testing.assert_allclose(proj6 @ got, 2*f*d[2]*16/231, rtol=1.e-11, atol=1.e-11*scale)
    if f:
        assert abs(proj6 @ got) > 1.e-4
        assert np.max(abs(got-old)) > 1.e-4


def test_regular_zero_growth_coefficients_and_derivatives():
    from pfsfog.counterterm_fields import leading_field_coefficients, pair_counterterms
    ca, b = raw(4., -7., 9., 110.), 2.
    np.testing.assert_allclose(leading_field_coefficients(ca, b, 0.), [-2., 4.5, -6.75], rtol=1.e-14)
    call = lambda f: pair_counterterms(ca, raw(-2., 3., 5., 250.), b, 1.3, f)
    ad = jax.jacfwd(call)(jnp.asarray(0.))
    assert np.isfinite(ad).all()
    fd = (call(1.e-5)-call(-1.e-5))/2.e-5
    np.testing.assert_allclose(ad, fd, rtol=1.e-8, atol=1.e-8)


def test_exchange_evenness_equal_population_limit_and_dimensional_scaling():
    from pfsfog.counterterm_fields import pair_counterterms
    ca, cb, f = raw(4., -7., 9., 110.), raw(-2., 3., 5., 250.), .81
    c = pair_counterterms(ca, cb, 1.8, 1.3, f)
    np.testing.assert_allclose(c, pair_counterterms(cb, ca, 1.3, 1.8, f), rtol=1.e-14)
    np.testing.assert_allclose(c, translated(ca, cb, 1.8, 1.3, f), rtol=1.e-14)
    auto = parameters_to_elf(dict(h=.67, f=f, bias=dict(b1=1.8, b2=0., bG2=0., bGamma3=0.), ctr=ca)).ctr
    np.testing.assert_allclose(auto, pair_counterterms(ca, ca, 1.8, 1.8, f), rtol=1.e-14)
    np.testing.assert_allclose(angular(c, .1, -.3, 100., f), angular(c, .1, .3, 100., f), rtol=1.e-14)
    # Length coordinates: leading coefficients carry L² and FoG carries L⁴.
    lam = 1.7
    scaled = lambda a: {key: val*lam**(4 if key == 'cfog' else 2) for key, val in a.items()}
    cc = pair_counterterms(scaled(ca), scaled(cb), 1.8, 1.3, f)
    np.testing.assert_allclose(angular(cc, .1/lam, .6, 100., f), angular(c, .1, .6, 100., f), rtol=1.e-13)
