"""Scientific failure tests for the revision gate harness."""

import json
import sys

import numpy as np
import pytest

from pfsfog.revision_forecast import CellLikelihood
from pfsfog.revision_validation import (compare_cell_derivatives,
                                        compare_marginal_errors,
                                        compare_nested_likelihoods,
                                        evidence_gate_record,
                                        require_complete_coverage)


def _cell(*, swap=False):
    pairs = (("DESI-ELG", "DESI-ELG"), ("DESI-LRG", "DESI-LRG"))
    k = np.array([.05, .10])
    names = ["ln10_10_As", "bin00:DESI-ELG:b1_sigma8"]
    derivatives = np.zeros((2, 6, 2))
    derivatives[:, 0:3, 0] = 2. * k[:, None]
    derivatives[:, 3:6, 0] = 3. * k[:, None]
    derivatives[:, 0:3, 1] = k[:, None]
    if swap:
        derivatives = np.concatenate((derivatives[:, 3:], derivatives[:, :3]), axis=1)
    means = {
        pairs[0]: lambda p: np.tile((2*p[names[0]]+p[names[1]])*k[:, None], (1, 3)),
        pairs[1]: lambda p: np.tile(3*p[names[0]]*k[:, None], (1, 3)),
    }
    cell = CellLikelihood(np.eye(2), names,
                          {"cell_id":"bin00-radial00-overlap", "retained_pairs":pairs,
                           "retained_observations":[(pair, ell) for pair in pairs
                                                    for ell in (0, 2, 4)]},
                          {"expected_shells":2, "processed_shells":2},
                          derivatives, np.tile(np.eye(6), (2, 1, 1)))
    return cell, means


def test_independent_pair_means_detect_derivative_misrouting():
    cell, means = _cell(swap=True)
    report = compare_cell_derivatives(cell, means,
                                      {name:1. for name in cell.names},
                                      {name:(1e-3, 5e-4, 2.5e-4) for name in cell.names},
                                      configuration="B-run", physical_scale=1.)
    assert report["status"] == "failed"
    assert report["failures"]
    assert any(entry["pair"] == ["DESI-ELG", "DESI-ELG"]
               for entry in report["failures"])


def test_coverage_manifest_rejects_a_missing_pair_or_fd_step():
    cell, means = _cell()
    report = compare_cell_derivatives(cell, means,
                                      {name:1. for name in cell.names},
                                      {name:(1e-3, 5e-4, 2.5e-4) for name in cell.names},
                                      configuration="B-run", physical_scale=1.)
    assert report["status"] == "passed"
    assert len(report["expected_coverage"]) == 2*3*2*3
    assert all(len(key) == 6 and key[0] == "B-run" for key in report["measured_coverage"])
    require_complete_coverage(report["expected_coverage"], report["measured_coverage"])
    with pytest.raises(ValueError, match="incomplete coverage"):
        require_complete_coverage(report["expected_coverage"], report["measured_coverage"][:-1])


def test_inactive_pair_parameter_block_must_be_zero():
    cell, means = _cell()
    cell.derivatives[:, 3:, 1] = .01
    report = compare_cell_derivatives(cell, means,
                                      {name:1. for name in cell.names},
                                      {name:(1e-3, 5e-4, 2.5e-4) for name in cell.names},
                                      configuration="B-run", physical_scale=1.)
    assert report["status"] == "failed"
    assert any(item["column"] == cell.names[1] and item["reason"] == "zero block"
               for item in report["failures"])


def test_full_vector_whitening_catches_correlated_low_variance_mode():
    name = "ln10_10_As"
    pair = ("DESI-ELG", "DESI-ELG")
    covariance = np.array([[1., .9999, 0.], [.9999, 1., 0.], [0., 0., 1.]])
    cell = CellLikelihood(np.eye(1), [name],
                          {"cell_id":"correlated", "retained_pairs":[pair]},
                          {}, np.array([[[1.0001], [.9999], [0.]]]), covariance[None])
    means = {pair:lambda p: np.array([[p[name], p[name], 0.]])}
    report = compare_cell_derivatives(cell, means, {name:1.},
                                      {name:(1e-3, 5e-4, 2.5e-4)},
                                      configuration="B-run", physical_scale=1.)
    assert report["status"] == "failed"
    assert report["column_results"][0]["step_errors"][0] > 1e-3
    assert report["max_error"] >= report["column_results"][0]["step_errors"][0]


def test_small_observed_pair_error_cannot_hide_behind_large_pair():
    name = "ln10_10_As"
    large, small = ("DESI-ELG", "DESI-ELG"), ("DESI-LRG", "DESI-LRG")
    derivative = np.zeros((1, 6, 1))
    derivative[0, 0, 0] = 1000.
    derivative[0, 3, 0] = 1.1
    cell = CellLikelihood(np.eye(1), [name],
                          {"cell_id":"diluted", "retained_pairs":[large, small]},
                          {}, derivative, np.eye(6)[None])
    means = {large:lambda p: np.array([[1000*p[name], 0., 0.]]),
             small:lambda p: np.array([[p[name], 0., 0.]])}
    report = compare_cell_derivatives(cell, means, {name:1.},
                                      {name:(1e-3, 5e-4, 2.5e-4)},
                                      configuration="B-run", physical_scale=1.)
    assert max(report["column_results"][0]["step_errors"]) < 1e-3
    assert report["status"] == "failed"
    assert any(item.get("pair") == list(small) and item.get("ell") == 0
               for item in report["failures"])


def test_coverage_rejects_duplicate_expected_and_cross_configuration_alias():
    key = ["B-run", "c", ["A", "A"], 0, "Mnu", .005]
    with pytest.raises(ValueError, match="duplicate expected"):
        require_complete_coverage([key, key], [key])
    other = ["C-run", *key[1:]]
    with pytest.raises(ValueError, match="incomplete coverage"):
        require_complete_coverage([key], [other])


def test_gate_evidence_rejects_stale_identity_and_keeps_failure(tmp_path):
    evidence = tmp_path / "measure.json"
    evidence.write_text(json.dumps({"run_id":"run-a", "configuration":{"dk":.0025},
                                    "provenance":{"source_content_hash":"abc"},
                                    "gates":{"ad_fd":{"status":"failed",
                                                       "measured":{"max_error":.2},
                                                       "tolerance":.001,
                                                       "reason":"misrouted pair"}}}))
    record = evidence_gate_record(evidence, "run-a", {"dk":.0025},
                                  {"source_content_hash":"abc"}, "ad_fd")
    assert record["status"] == "failed"
    assert record["measured"] == {"max_error":.2}
    assert record["reason"] == "misrouted pair"
    with pytest.raises(ValueError, match="identity"):
        evidence_gate_record(evidence, "run-a", {"dk":.00125},
                             {"source_content_hash":"abc"}, "ad_fd")


def test_ad_fd_gate_cannot_pass_with_partial_or_absent_coverage(tmp_path):
    evidence = tmp_path / "measure.json"
    document = {"run_id":"run-a", "configuration":{"dk":.0025},
                "provenance":{"source_content_hash":"abc"},
                "gates":{"ad_fd":{"status":"passed", "measured":{},
                                   "tolerance":.001, "reason":None}}}
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="coverage"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd")
    document["gates"]["ad_fd"]["measured"] = {
        "expected_coverage":[["run-a", "cell", ["A", "A"], 0, "Mnu", .005]],
        "measured_coverage":[]}
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="incomplete coverage"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd")
    one = ["run-a", "cell", ["A", "A"], 0, "Mnu", .005]
    two = ["run-a", "cell", ["A", "A"], 0, "Mnu", .0025]
    three = ["run-a", "cell", ["A", "A"], 0, "Mnu", .00125]
    four = ["run-a", "other", ["A", "A"], 0, "Mnu", .005]
    result = document["gates"]["ad_fd"]
    document["gates"]["ad_fd"]["measured"] = {
        "status":"passed", "failures":[],
        "expected_coverage":[one, two, three], "measured_coverage":[one, two, three],
        "errors":[{"configuration":"run-a", "cell_id":"cell", "pair":["A", "A"],
                   "ell":0, "column":"Mnu", "step":step, "error":1e-4,
                   "limit":1e-3, "reason":"relative"}
                  for step in (.005,.0025,.00125)],
        "column_results":[{"configuration":"run-a", "cell_id":"cell",
                           "column":"Mnu", "steps":[.005,.0025,.00125],
                           "step_errors":[1e-4,1e-4,1e-4],
                           "step_tolerances":[1e-3,1e-3,1e-3]}]}
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="declared coverage"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd")
    with pytest.raises(ValueError, match="incomplete coverage"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=[one, two, three, four])
    record = evidence_gate_record(evidence, "run-a", document["configuration"],
                                  document["provenance"], "ad_fd",
                                  declared_coverage=[one, two, three])
    assert record["status"] == "passed"
    result["measured"]["status"] = "failed"
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="measured status"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=[one, two, three])
    result["measured"]["status"] = "passed"
    result["measured"]["failures"] = [{"reason":"bad FD"}]
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="failures"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=[one, two, three])
    result["measured"]["failures"] = []
    result["measured"]["column_results"][0]["step_errors"] = [.01,.01,.01]
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="tolerance"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=[one, two, three])
    result["measured"]["column_results"][0]["step_errors"] = [1e-4]*3
    result["measured"]["errors"][0]["error"] = .1
    result["measured"]["errors"][1]["error"] = .1
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="row"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=[one, two, three])


@pytest.mark.parametrize("small_pair_steps", [(.005,), (.005, .0025)])
def test_ad_fd_gate_rejects_incomplete_steps_for_one_pair(tmp_path, small_pair_steps):
    steps = (.005, .0025, .00125)
    coverage = [["run-a", "cell", ["A", "A"], 0, "Mnu", step] for step in steps]
    coverage += [["run-a", "cell", ["B", "B"], 0, "Mnu", step]
                 for step in small_pair_steps]
    document = {"run_id":"run-a", "configuration":{"dk":.0025},
                "provenance":{"source_content_hash":"abc"},
                "gates":{"ad_fd":{"status":"passed", "tolerance":.001,
                                   "reason":None, "measured":{
                                       "status":"passed", "failures":[],
                                       "expected_coverage":coverage,
                                       "measured_coverage":coverage,
                                       "column_results":[{
                                           "configuration":"run-a", "cell_id":"cell",
                                           "column":"Mnu", "steps":list(steps),
                                           "step_errors":[1e-4]*3,
                                           "step_tolerances":[1e-3]*3}],
                                       "errors":[{
                                           "configuration":key[0], "cell_id":key[1],
                                           "pair":key[2], "ell":key[3], "column":key[4],
                                           "step":key[5], "error":1e-4, "limit":1e-3,
                                           "reason":"relative"} for key in coverage]}}}}
    evidence = tmp_path / "measure.json"
    evidence.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="row steps"):
        evidence_gate_record(evidence, "run-a", document["configuration"],
                             document["provenance"], "ad_fd",
                             declared_coverage=coverage)


def _nested(old_f, new_f, *, corrupt=False):
    names = ["ln10_10_As", "Mnu", "Omegam"]
    old = {"names":names, "fiducial":[3., .06, .31], "prior_sigma":[None, 5., 1.],
           "F_like":np.diag(old_f), "k":[.05], "dk":[.01],
           "cells":[{"cell_id":"c", "pairs":[["A", "A"]],
                     "geometry":{"radial_bin":[.8, 1.], "volume":100., "nbar":{"A":.001}},
                     "derivatives":np.array([[[1., 0., 0.], [0., 0., 0.],
                                               [0., 0., 0.]]]),
                     "covariance":np.eye(3)[None]}]}
    new = {"names":names, "fiducial":[3., .06, .31], "prior_sigma":[None, 5., 1.],
           "F_like":np.diag(new_f), "k":[.05], "dk":[.01],
           "cells":[{"cell_id":"c", "pairs":[["A", "A"], ["A", "B"]],
                     "geometry":{"radial_bin":[.8, 1.], "volume":100., "nbar":{"A":.001}},
                     "derivatives":np.array([[[1., 0., 0.], [0., 0., 0.],
                                               [0., 0., 0.], [0., 1., 0.],
                                               [0., 0., 0.], [0., 0., 0.]]]),
                     "covariance":np.eye(6)[None]}]}
    if corrupt:
        new["cells"][0]["derivatives"][0, 0, 0] = 1.2
    return old, new


def test_matched_basis_psd_and_broken_nesting():
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    report = compare_nested_likelihoods(old, new)
    assert report["status"] == "passed"
    assert report["near_null_eigenvalues"] >= 2
    old, new = _nested([1., 1., 1.], [1., .5, 1.])
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.], corrupt=True)
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["cells"][0]["geometry"]["volume"] = 101.
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["dk"] = [.011]
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["names"] = new["names"].copy()
    new["names"][2] = "different"
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["F_like"][1, 1] = np.nan
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["cells"][0]["names"] = ["ln10_10_As", "Mnu"]
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["cells"][0]["derivatives"][0, 1, 0] = 1e-11
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    new["cells"][0]["covariance"][0, 0, 0] += 1e-13
    assert compare_nested_likelihoods(old, new)["status"] == "failed"
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    old["cells"][0]["covariance"][0, 0, 0] = np.inf
    new["cells"][0]["covariance"][0, 0, 0] = np.inf
    report = compare_nested_likelihoods(old, new)
    assert report["status"] == "failed"
    assert "nonfinite" in " ".join(report["reasons"])
    old, new = _nested([1., 1., 1.], [1., 2., 1.])
    old["cells"][0]["derivatives"][0, 0, 0] = np.inf
    new["cells"][0]["derivatives"][0, 0, 0] = np.inf
    assert compare_nested_likelihoods(old, new)["status"] == "failed"


def test_native_and_growth_errors_use_each_valid_active_solve():
    old = {"cov_cosmo_marg":np.diag([4., 9., 16.]),
           "growth_z":np.arange(8.)/10+.5,
           "growth_jacobian":np.tile([1., .5, .2], (8, 1))}
    new = {"cov_cosmo_marg":np.diag([3., 8., 15.]),
           "growth_z":old["growth_z"].copy(),
           "growth_jacobian":old["growth_jacobian"].copy()}
    assert compare_marginal_errors(old, new)["status"] == "passed"
    new["cov_cosmo_marg"][0, 0] = 5.
    report = compare_marginal_errors(old, new)
    assert report["status"] == "failed"
    assert report["native_ratio"][0] > 1.
    new["growth_z"][0] += .01
    assert compare_marginal_errors(old, new)["status"] == "failed"


def test_validation_import_does_not_change_production_run_identity():
    from scripts.run_revision import collect_provenance
    from pfsfog.revision_artifacts import _configuration, run_identity
    from pfsfog.revision_forecast import RevisionConfig

    class Backend:
        do_irres = False
        subtract_k0_limit = True
        _kmin, _kmax, _nfft = 1e-5, 1e3, 256

    module = sys.modules.pop("pfsfog.revision_validation")
    try:
        before = collect_provenance(Backend(), requested_dk=.005, command="identity-test")
    finally:
        sys.modules["pfsfog.revision_validation"] = module
    after = collect_provenance(Backend(), requested_dk=.005, command="identity-test")
    config = _configuration(RevisionConfig("B", dk=.0025), 1.,
                            requested_dk=.005, model_flags=before["model_flags"])
    assert run_identity(config, before) == run_identity(config, after)
