"""Analytic Q-multipole zeros must not become ratios of cancellation noise."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest


def orthogonal_cell(*, ad_leak=0., fd_leak=0., wrong_nonzero=False):
    name = "bin00:DESI-ELG:c4"
    pair = ("DESI-ELG", "DESI-ELG")
    derivative = np.zeros((2, 3, 1))
    derivative[:, 0, 0] = ad_leak
    derivative[:, 2, 0] = 1.1 if wrong_nonzero else 1.
    cell = SimpleNamespace(names=[name], derivatives=derivative,
        covariance=np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
        metadata={"cell_id": "orthogonal-Q4", "retained_pairs": [pair]})
    def mean(values):
        q = values[name]
        return np.tile([1.+fd_leak*q+4.e-15*np.sign(q), 1., 1.+q], (2, 1))
    return cell, {pair: mean}, {name: 0.}, {name: (.001, .0005, .00025)}


def test_analytic_null_support_requires_identity_ap_and_isotropic_counterterm_base():
    from pfsfog.elf_validation_zeros import counterterm_zero_support
    names = ["h", "bin00:DESI-ELG:c4", "bin00:DESI-LRG:c2", "bin00:DESI-QSO:c0"]
    pairs = [("DESI-ELG", "DESI-LRG"), ("DESI-ELG", "DESI-ELG"), ("DESI-LRG", "DESI-LRG")]
    settings = {"engine": "elf.EPT", "counterterm_base": "linear", "do_irres": False}
    rows = counterterm_zero_support(names, pairs, alphas=(1., 1.), settings=settings)
    assert set(rows) == {
        (names[1], pairs[1], 0), (names[1], pairs[1], 2),
        (names[2], pairs[2], 0), (names[2], pairs[2], 4)}
    assert counterterm_zero_support(names, pairs, alphas=(1.001, 1.), settings=settings) == ()
    assert counterterm_zero_support(names, pairs, alphas=(1., .999), settings=settings) == ()
    assert counterterm_zero_support(names, pairs, alphas=(1., 1.),
        settings={**settings, "counterterm_base": "linear_ir_resum"}) == ()
    assert counterterm_zero_support(names, pairs, alphas=(1., 1.),
        settings={**settings, "do_irres": True}) == ()


def test_exact_angular_zero_uses_unchanged_absolute_criterion_for_all_steps():
    from pfsfog.revision_validation import compare_cell_derivatives
    cell, means, values, steps = orthogonal_cell()
    common = dict(configuration="test", physical_scale=1.)
    original = compare_cell_derivatives(cell, means, values, steps, **common)
    assert original["status"] == "failed"  # Cancellation alone crosses the generic classifier.
    declared = [(cell.names[0], ("DESI-ELG", "DESI-ELG"), ell) for ell in (0, 2)]
    result = compare_cell_derivatives(cell, means, values, steps,
                                      analytic_zero_support=declared, **common)
    assert result["status"] == "passed"
    null = [r for r in result["errors"] if r["ell"] == 0]
    assert len(null) == 3 and all(r["limit"] == 1.e-6 and r["error"] < 1.e-10 for r in null)
    assert all(r["reason"] == "analytic angular zero" for r in null)
    assert all(r["limit"] == 1.e-3 for r in result["errors"] if r["ell"] == 4)


@pytest.mark.parametrize("ad_leak,fd_leak", [(1.e-5, 0.), (0., 1.e-5), (1.e-5, 1.e-5)])
def test_declared_zero_cannot_hide_leakage_even_when_ad_and_fd_agree(ad_leak, fd_leak):
    from pfsfog.revision_validation import compare_cell_derivatives
    cell, means, values, steps = orthogonal_cell(ad_leak=ad_leak, fd_leak=fd_leak)
    result = compare_cell_derivatives(cell, means, values, steps, configuration="test",
        physical_scale=1., analytic_zero_support=[(cell.names[0], ("DESI-ELG", "DESI-ELG"), 0)])
    assert result["status"] == "failed" and result["failures"]


def test_counterterm_nonzero_multipole_keeps_relative_derivative_test():
    from pfsfog.revision_validation import compare_cell_derivatives
    cell, means, values, steps = orthogonal_cell(wrong_nonzero=True)
    result = compare_cell_derivatives(cell, means, values, steps, configuration="test",
        physical_scale=1., analytic_zero_support=[(cell.names[0], ("DESI-ELG", "DESI-ELG"), ell) for ell in (0, 2)])
    assert result["status"] == "failed"
    assert any(r.get("ell") == 4 for r in result["failures"])


def test_saved_report_must_match_derived_angular_support():
    from pfsfog.revision_validation import compare_cell_derivatives
    from pfsfog.elf_survey_evidence import _check_derivative_report
    cell, means, values, steps = orthogonal_cell()
    pair = ("DESI-ELG", "DESI-ELG")
    support = [(cell.names[0], pair, ell) for ell in (0, 2)]
    report = compare_cell_derivatives(cell, means, values, steps, configuration="test",
                                      physical_scale=1., analytic_zero_support=support)
    saved_steps = {n: list(v) for n, v in steps.items()}
    # JSON roundtrip mirrors persisted lists in all coverage records.
    import json
    report = json.loads(json.dumps(report))
    _check_derivative_report(report, cell.names, [list(pair)], saved_steps, "test",
                             "orthogonal-Q4", analytic_zero_support=support)
    corrupt = copy.deepcopy(report)
    corrupt["analytic_zero_support"].append([cell.names[0], list(pair), 4])
    with pytest.raises(ValueError, match="analytic"):
        _check_derivative_report(corrupt, cell.names, [list(pair)], saved_steps, "test",
                                 "orthogonal-Q4", analytic_zero_support=support)
    corrupt = copy.deepcopy(report)
    zero = next(row for row in corrupt["errors"] if row["ell"] == 0)
    zero["limit"] = 1.e-3
    with pytest.raises(ValueError, match="analytic"):
        _check_derivative_report(corrupt, cell.names, [list(pair)], saved_steps, "test",
                                 "orthogonal-Q4", analytic_zero_support=support)


@pytest.mark.parametrize("cross", [False, True])
def test_physical_Q_counterterms_have_the_derived_legendre_support(cross):
    import jax
    import jax.numpy as jnp
    from pfsfog.cosmo import FIDUCIAL, FiducialCosmology
    from pfsfog.eft_params import tracer_fiducials
    from pfsfog.elf_survey import elf_cosmology_spec
    from pfsfog.one_loop import make_one_loop_engine
    from pfsfog.revision_theory import make_pair_response, fiducial_nuisance
    space = elf_cosmology_spec(FIDUCIAL)
    theta = jnp.array([FIDUCIAL[n] for n in ("ln10_10_As", "mnu", "Omega_m", "h", "omega_b", "n_s")])
    cosmo = FiducialCosmology()
    qa = fiducial_nuisance(tracer_fiducials("DESI-ELG", 1.3, cosmo.sigma8(.9)))
    qb = fiducial_nuisance(tracer_fiducials("DESI-LRG", 2., cosmo.sigma8(.9))) if cross else None
    response = make_pair_response(make_one_loop_engine(), .9, FIDUCIAL, 3.e-4,
                                  1.e-4 if cross else None, space=space)
    k = jnp.array([.04, .11, .19])
    mu = jnp.array([.02, .25, .5, .8, .98])
    derivative = np.asarray(jax.jacfwd(lambda q: response.pkmu(k, mu, theta, q, qb))(qa))
    f = float(response.growth_rate(.9, response._cosmology(theta)))
    # Each Q direction fixes fitted intrinsic auto multipoles, but its
    # completed angular field also supplies ell=6. Cross multipoles mix Q.
    from test_counterterm_fields import field_from_projection
    from pfsfog.revision_theory import nuisance_to_backend
    cosmology = response._cosmology(theta)
    s8 = response.sigma8(.9, cosmology)
    ba = float(qa[0]/s8)
    bb = float(qb[0]/s8) if cross else ba
    for coordinate in (4, 5, 6):
        unit = jnp.zeros_like(qa).at[coordinate].set(1.)
        ctr = nuisance_to_backend(unit, s8, response.sigma8_ref, f,
                                  cosmology["h"], 3.e-4, .7)["ctr"]
        d = field_from_projection(ctr, ba, f)
        shape = (bb+f*np.asarray(mu)**2)*np.polynomial.polynomial.polyval(np.asarray(mu)**2, d)
        if not cross:
            shape *= 2
        # The common k² P_lin cancels in this independently predicted ratio.
        if coordinate == 4:
            reference = shape
        else:
            np.testing.assert_allclose(derivative[..., coordinate],
                derivative[..., 4]*(shape/reference), rtol=2.e-11, atol=2.e-11)
    multipoles = np.asarray(jax.jacfwd(lambda q: response.multipoles(k, theta, q, qb, n_mu=32))(qa))
    for coordinate, ell_index in ((4, 0), (5, 1), (6, 2)):
        others = [i for i in range(3) if i != ell_index]
        ratio = np.linalg.norm(multipoles[:, others, coordinate])/np.linalg.norm(multipoles[:, ell_index, coordinate])
        if cross:
            assert ratio > 1.e-4
        else:
            assert ratio < 2.e-12
    # A fixed anisotropic AP distortion physically mixes these multipoles.
    response.ap_alphas = lambda _: (1.04, .97)
    distorted = np.asarray(jax.jacfwd(lambda q: response.multipoles(k, theta, q, qb, n_mu=32))(qa))
    assert np.linalg.norm(distorted[:, 1, 6])/np.linalg.norm(distorted[:, 2, 6]) > 1.e-4
