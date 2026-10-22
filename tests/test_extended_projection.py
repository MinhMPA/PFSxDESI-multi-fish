"""Analytic precision conditioning and exact supplement prior contracts."""

from dataclasses import replace
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.revision_forecast import (
    RevisionConfig, RevisionLikelihood, make_shell_grid, apply_revision_priors,
)

NAMES = make_cosmology_spec(FIDUCIAL, extended=True).names
TREATMENTS = ("conditional3", "release_h", "release_omega_b", "release_n_s",
              "marginal6", "marginal6_weak10", "marginal6_no_extra")


def projection_api():
    # A missing implementation is an assertion failure during the first RED.
    spec = importlib.util.find_spec("pfsfog.cosmology_projection")
    assert spec is not None, "six-coordinate projection implementation missing"
    return importlib.import_module("pfsfog.cosmology_projection")


def fixture(monkeypatch, fisher=None, *, case="B", nuisance_width=2.):
    module = projection_api()
    monkeypatch.setattr(module, "make_growth_rate_func", lambda: lambda z, c: (
        c["ln10_10_As"] + c["mnu"]*z + c["Omega_m"]*z**2
        + c["h"] + c["omega_b"]*z**3 + c["n_s"]*z**4))
    monkeypatch.setattr(module, "make_sigma8_func", lambda: lambda z, c: 1.)
    config = RevisionConfig(case, cosmology_basis="extended6", nuisance_factor=3.)
    names = list(NAMES) + ["bin00:DESI-ELG:c2"]
    records = [{"name": name, "role": "cosmo" if i < 6 else "tracer",
                "prior_sigma": nuisance_width if i == 6 else None}
               for i, name in enumerate(names)]
    theta = [FIDUCIAL[key] for key in
             ("ln10_10_As", "mnu", "Omega_m", "h", "omega_b", "n_s")]
    if fisher is None:
        rng = np.random.default_rng(123)
        a = rng.normal(size=(12, 7))
        fisher = a.T @ a + 2*np.eye(7)
    return RevisionLikelihood(config, fisher, names, np.array(theta + [0.]),
                              records, [], [fisher/2, fisher/2],
                              make_shell_grid(.01, .02, .005), FIDUCIAL, {}, {})


def test_precision_selection_conditions_while_covariance_subblock_marginalizes(monkeypatch):
    f = np.eye(7)*2
    f[0, 3] = f[3, 0] = 1.
    like = fixture(monkeypatch, f, nuisance_width=None)
    api = projection_api()
    conditional = api.apply_cosmology_treatment(like, "conditional3")
    released = api.apply_cosmology_treatment(like, "release_h")
    assert conditional.cov_cosmo_marg[0, 0] == pytest.approx(.5)
    assert released.cov_cosmo_marg[0, 0] == pytest.approx(2/3)
    assert np.linalg.inv(f)[0, 0] == pytest.approx(2/3)
    gradient = np.array([1., 1.])
    assert gradient @ released.cov_cosmo_marg[np.ix_([0, 3], [0, 3])] @ gradient == pytest.approx(2/3)
    assert gradient @ conditional.cov_cosmo_marg[np.ix_([0, 3], [0, 3])] @ gradient == pytest.approx(.5)
    assert conditional.growth_cov_marg[0, 0] < released.growth_cov_marg[0, 0]
    np.testing.assert_allclose(released.growth_cov_marg[0, 0]
                              - conditional.growth_cov_marg[0, 0], 1/6)
    # The same 2x2 example with a retained nuisance: fixing h still marginalizes q.
    f[0, 6] = f[6, 0] = .5
    like.F_like = f
    result = api.apply_cosmology_treatment(like, "conditional3")
    assert result.cov_cosmo_marg[0, 0] == pytest.approx(1/(2-.5**2/2))
    assert result.cov_cosmo_nuisance_fixed[0, 0] == pytest.approx(.5)
    assert result.active_parameter_names[-1] == like.names[-1]


def test_two_likelihoods_have_seven_exact_once_only_treatments(monkeypatch):
    api = projection_api()
    results = [api.apply_cosmology_treatment(fixture(monkeypatch, case=case), treatment)
               for case in ("B", "C") for treatment in TREATMENTS]
    assert len(results) == 14
    extras = {"release_omega_b": {"omega_b": .00055}, "release_n_s": {"n_s": .042},
              "marginal6": {"omega_b": .00055, "n_s": .042},
              "marginal6_weak10": {"omega_b": .0055, "n_s": .42}}
    for result in results:
        assert result.growth_jacobian.shape == (8, 6)
        assert result.cov_cosmo_marg.shape == result.cov_cosmo_nuisance_fixed.shape == (6, 6)
        assert result.growth_cov_marg.shape == result.growth_cov_nuisance_fixed.shape == (8, 8)
        for i, name in enumerate(result.active_parameter_names):
            expected = {"Mnu": .04, "Omegam": 1.}.get(name, 0.)
            if name in extras.get(result.treatment, {}):
                expected = extras[result.treatment][name]**-2
            assert result.F_prior_cosmo[i, i] == pytest.approx(expected)
        np.testing.assert_allclose(result.F_prior_nuisance[-1, -1], 1/6**2)
        np.testing.assert_allclose(result.F_total, result.F_like + result.F_prior_cosmo
                                   + result.F_prior_nuisance)
        fixed = [NAMES.index(name) for name in NAMES if name not in result.free_cosmology]
        np.testing.assert_array_equal(result.cov_cosmo_marg[fixed], 0.)
        np.testing.assert_array_equal(result.cov_cosmo_nuisance_fixed[fixed], 0.)
        np.testing.assert_allclose(result.growth_cov_marg, result.growth_jacobian
                                   @ result.cov_cosmo_marg @ result.growth_jacobian.T)
        free = [NAMES.index(name) for name in result.free_cosmology]
        expected_fixed = np.linalg.inv(result.F_total[:len(free), :len(free)])
        np.testing.assert_allclose(result.cov_cosmo_nuisance_fixed[np.ix_(free, free)],
                                   expected_fixed, rtol=1e-12)


def test_release_and_same_treatment_survey_monotonicity(monkeypatch):
    api = projection_api()
    b = fixture(monkeypatch)
    c = replace(b, config=replace(b.config, case="C"), F_like=b.F_like + np.eye(7))
    rb = {t: api.apply_cosmology_treatment(b, t) for t in TREATMENTS}
    rc = {t: api.apply_cosmology_treatment(c, t) for t in TREATMENTS}
    for treatment in ("release_h", "release_omega_b", "release_n_s"):
        for field in ("cov_cosmo_marg", "cov_cosmo_nuisance_fixed"):
            lo, mid, hi = [np.diag(getattr(rb[t], field))[:3]
                           for t in ("conditional3", treatment, "marginal6")]
            assert np.all(lo <= mid + 1e-12)
            assert np.all(mid <= hi + 1e-12)
        for field in ("growth_cov_marg", "growth_cov_nuisance_fixed"):
            lo, mid, hi = [np.diag(getattr(rb[t], field))
                           for t in ("conditional3", treatment, "marginal6")]
            assert np.all(lo <= mid + 1e-12)
            assert np.all(mid <= hi + 1e-12)
    for treatment in TREATMENTS:
        for field in ("cov_cosmo_marg", "growth_cov_marg"):
            assert np.all(np.diag(getattr(rc[treatment], field))
                          <= np.diag(getattr(rb[treatment], field)) + 1e-12)


def test_projection_uses_named_coordinates_after_parameter_reordering(monkeypatch):
    api = projection_api()
    like = fixture(monkeypatch)
    order = [6, 4, 2, 0, 5, 3, 1]
    shuffled = replace(like, F_like=like.F_like[np.ix_(order, order)],
                       names=[like.names[i] for i in order],
                       fiducial=like.fiducial[order], parameters=[like.parameters[i] for i in order])
    for treatment in TREATMENTS:
        original = api.apply_cosmology_treatment(like, treatment)
        result = api.apply_cosmology_treatment(shuffled, treatment)
        np.testing.assert_allclose(result.cov_cosmo_marg, original.cov_cosmo_marg, rtol=1e-12)
        np.testing.assert_allclose(result.growth_cov_marg, original.growth_cov_marg, rtol=1e-12)
        assert tuple(shuffled.names[i] for i in result.active_indices) == result.active_parameter_names


def test_singular_treatment_unknown_treatment_and_legacy_extended_calls_fail(monkeypatch):
    api = projection_api()
    like = fixture(monkeypatch, np.zeros((7, 7)))
    with pytest.raises(ValueError, match="unconstrained directions"):
        api.apply_cosmology_treatment(like, "marginal6_no_extra")
    with pytest.raises(ValueError, match="treatment"):
        api.apply_cosmology_treatment(like, "flat")
    with pytest.raises(ValueError, match="extended6"):
        apply_revision_priors(like, 1.)
    with pytest.raises(ValueError, match="extended6"):
        api.apply_cosmology_treatment(replace(like, config=RevisionConfig("B")), "conditional3")


def test_pure_projection_never_initializes_growth_factories(monkeypatch):
    api=projection_api(); like=fixture(monkeypatch)
    expected=api.apply_cosmology_treatment(like,'marginal6')
    assert hasattr(api,'project_cosmology_precision'), 'pure precision projection missing'
    def forbidden(): raise AssertionError('growth initialized by pure matrix projection')
    monkeypatch.setattr(api,'make_growth_rate_func',forbidden)
    monkeypatch.setattr(api,'make_sigma8_func',forbidden)
    actual=api.project_cosmology_precision(like,'marginal6')
    for field in ('F_like','F_prior_cosmo','F_prior_nuisance','F_total','covariance',
                  'cov_cosmo_marg','cov_cosmo_nuisance_fixed'):
        np.testing.assert_array_equal(actual[field],getattr(expected,field))
