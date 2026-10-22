"""Physical adapter failure tests on the real likelihood data structures."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.revision_forecast import CellLikelihood, RevisionConfig, retained_pairs
from pfsfog.revision_theory import COSMO_NAMES, NUISANCE_NAMES
from pfsfog.revision_validation_physical import (
    PhysicalFDCache, physical_cell_means, declare_fd_manifest,
    assembled_fd_certificate, audit_cell_covariance_neighborhood,
    likelihood_snapshot, CovarianceAuditFailure, validate_support_schedule,
    expected_support_schedule,
    _audited_spd,
)
from pfsfog.revision_validation import compare_nested_likelihoods, evidence_gate_record
from pfsfog.surveys import default_survey_group


def _cross_fixture():
    pair = ("DESI-ELG", "DESI-LRG")
    names = list(COSMO_NAMES)
    names += [f"bin00:{tracer}:{coordinate}" for tracer in pair
              for coordinate in NUISANCE_NAMES]
    names += [f"bin00:{pair[0]}|{pair[1]}:{coordinate}"
              for coordinate in ("Pshot_cross", "a2_cross")]
    fiducial = {name: float(i + 1) for i, name in enumerate(names)}
    cells = [CellLikelihood(
        np.eye(len(names)), names,
        {"cell_id":"bin00-radial00-overlap", "nominal_bin_index":0,
         "nominal_bin":(.4,.6), "radial_bin":(.4,.6), "model_redshift":.5,
         "region":"overlap", "volume":1.,
         "nbar":{pair[0]:1e-4, pair[1]:2e-4},
         "candidate_populations":pair,
         "active_populations":pair,
         "theoretical_pairs":retained_pairs("B", pair),
         "local_names":names,
         "retained_pairs":retained_pairs("B", pair),
         "retained_observations":[(p, ell) for p in retained_pairs("B", pair)
                                  for ell in (0,2,4)]},
        {"expected_shells":1, "processed_shells":1},
        np.zeros((1, 9, len(names))), np.eye(9)[None])]
    likelihood = SimpleNamespace(
        config=RevisionConfig("B", kmin=.01, kmax=.02, dk=.01, nominal_bins=((.4,.6),)),
        cells=cells, names=names, fiducial=np.array(list(fiducial.values())),
        grid=SimpleNamespace(centers=np.array([.015]), widths=np.array([.01])),
        reference_cosmology={"h":.7})
    return likelihood, cells[0], pair, fiducial


def _support(cell):
    metadata = cell.metadata
    return {"cell_id":metadata["cell_id"], "nominal_bin_index":metadata["nominal_bin_index"],
            "nominal_bin":metadata["nominal_bin"], "radial_bin":metadata["radial_bin"],
            "region":metadata["region"], "model_redshift":metadata["model_redshift"],
            "volume":metadata["volume"], "nbar":metadata["nbar"],
            "candidate_populations":metadata["active_populations"],
            "active_populations":metadata["active_populations"],
            "retained_pairs":metadata["retained_pairs"],
            "local_names":cell.names, "empty_observation":False,
            "absent_observations":[]}


def test_physical_cross_mean_routes_both_tracers_and_measured_stochastic(monkeypatch):
    likelihood, cell, pair, fiducial = _cross_fixture()
    calls = []

    class Response:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu):
            calls.append((self.a, self.b, np.asarray(q_a),
                          None if q_b is None else np.asarray(q_b),
                          None if q_cross is None else np.asarray(q_cross), n_mu))
            return np.full((len(k), 3), float(q_a[0]) +
                           (0 if q_b is None else 10*float(q_b[0])) +
                           (0 if q_cross is None else 100*float(q_cross[0])))

    def fake_response(ps, z, reference, nbar_a, nbar_b, k_nl, **kwargs):
        return Response(nbar_a, nbar_b)

    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response", fake_response)
    means = physical_cell_means(likelihood, cell, object())
    base = means[pair](fiducial)
    changed = dict(fiducial)
    changed[f"bin00:{pair[1]}:{NUISANCE_NAMES[0]}"] += 1
    assert np.all(means[pair](changed) - base == 10)
    changed = dict(fiducial)
    changed[f"bin00:{pair[0]}|{pair[1]}:Pshot_cross"] += 1
    assert np.all(means[pair](changed) - base == 100)
    assert any(record[0:2] == (1e-4, 2e-4) and record[-1] == 32 for record in calls)
    assert np.allclose(calls[-1][3],
                       [fiducial[f"bin00:{pair[1]}:{name}"] for name in NUISANCE_NAMES])


def test_nonfinite_pkmu_and_eigensolver_failure_keep_exact_point(monkeypatch):
    likelihood, cell, pair, _ = _cross_fixture()

    class Response:
        def pkmu(self, k, mu, theta, q_a, q_b, q_cross):
            result = np.ones((len(k), len(mu)))
            result[0, 5] = np.nan
            return result

    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response",
                        lambda *args, **kwargs: Response())
    with pytest.raises(CovarianceAuditFailure) as caught:
        audit_cell_covariance_neighborhood(likelihood, cell, object())
    failure = caught.value.diagnostics
    assert failure["cell_id"] == cell.metadata["cell_id"]
    assert failure["k"] == .015
    assert failure["mu"] == pytest.approx(np.polynomial.legendre.leggauss(32)[0][5])
    assert failure["pair"] == list(pair) or failure["pair"] == [pair[0], pair[0]]
    assert failure["eigenvalue_unavailable_reason"]

    def fail_eigh(_):
        raise np.linalg.LinAlgError("injected eigensolver failure")

    monkeypatch.setattr(np.linalg, "eigh", fail_eigh)
    with pytest.raises(CovarianceAuditFailure) as caught:
        _audited_spd(np.eye(2), "field cell=c column=Mnu sign=+ step=0.005 k=0.015 mu=0.2")
    failure = caught.value.diagnostics
    assert failure["k"] == .015 and failure["mu"] == .2
    assert failure["column"] == "Mnu" and failure["sign"] == "+"
    assert failure["eigenvalue_unavailable_reason"] == "eigensolver failed: injected eigensolver failure"


def test_candidate_ap_table_reaches_both_physical_response_factories(monkeypatch):
    likelihood, cell, _, _ = _cross_fixture()
    likelihood.config = replace(likelihood.config, ap_interpolation_nk=8192,
                                ap_refinement_nk=(16384, 32768))
    calls = []

    class Response:
        def pkmu(self, k, mu, theta, q_a, q_b, q_cross):
            return np.full((len(k), len(mu)), np.nan)

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response", factory)
    physical_cell_means(likelihood, cell, object())
    with pytest.raises(CovarianceAuditFailure, match="invalid field response"):
        audit_cell_covariance_neighborhood(likelihood, cell, object())
    assert len(calls) == 6
    assert all(row == {"ap_nk":8192, "ap_nmu":101} for row in calls)


def test_physical_cache_key_uses_actual_adopted_ap_table(monkeypatch, tmp_path):
    likelihood, cell, pair, fiducial = _cross_fixture()
    likelihood.config = replace(likelihood.config, ap_interpolation_nk=8192,
                                ap_refinement_nk=(16384, 32768))
    likelihood.model = {"ap_interpolation_nk":2048, "ap_interpolation_nmu":101,
                        "fft_kmin":1e-5, "fft_kmax":1e3, "fft_n":256}
    class Response:
        def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu):
            return np.ones((len(k), 3))
    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response",
                        lambda *args, **kwargs: Response())
    provenance = {field:field for field in (
        "project_head", "source_content_hash", "executed_module_hashes",
        "backend_snapshot", "dependency_hash", "network_hash", "survey_input_hash")}
    means = physical_cell_means(likelihood, cell, object(),
                                cache=PhysicalFDCache(tmp_path), provenance=provenance)
    means[pair](fiducial)
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["key"]["ap_fft"]["ap_interpolation_nk"] == 8192


def test_physical_adapter_rejects_omitted_retained_pair():
    likelihood, cell, _, _ = _cross_fixture()
    cell.metadata["retained_pairs"] = cell.metadata["retained_pairs"][:-1]
    with pytest.raises(ValueError, match="retained pair"):
        physical_cell_means(likelihood, cell, object())


def test_fd_manifest_declares_every_pair_ell_column_step_before_measurement():
    likelihood, cell, _, _ = _cross_fixture()
    support = [_support(cell)]
    manifest = declare_fd_manifest(likelihood, [cell.metadata["cell_id"]], "B-run",
                                   expected_support=support)
    assert len(manifest) == 3 * 3 * len(cell.names) * 3
    assert ["B-run", cell.metadata["cell_id"], ["DESI-ELG", "DESI-LRG"], 4,
            "Mnu", .00125] in manifest
    with pytest.raises(ValueError, match="manifest"):
        assembled_fd_certificate(likelihood, object(), [cell.metadata["cell_id"]],
                                 manifest[:-1], expected_support=support)


def test_fd_manifest_rejects_unknown_or_omitted_shell_cell():
    likelihood, cell, _, _ = _cross_fixture()
    with pytest.raises(ValueError, match="selected cell"):
        declare_fd_manifest(likelihood, ["absent"], "B-run", expected_support=[_support(cell)])
    cell.validation["processed_shells"] = 0
    with pytest.raises(ValueError, match="shell"):
        declare_fd_manifest(likelihood, [cell.metadata["cell_id"]], "B-run",
                            expected_support=[_support(cell)])


def test_cache_key_includes_unvaried_local_fiducial(monkeypatch, tmp_path):
    likelihood, cell, pair, fiducial = _cross_fixture()
    likelihood.model = {"ap_interpolation_nk":2048, "ap_interpolation_nmu":101,
                        "fft_kmin":1e-5, "fft_kmax":1e3, "fft_n":256}
    provenance = {field:field for field in (
        "project_head", "source_content_hash", "executed_module_hashes",
        "backend_snapshot", "dependency_hash", "network_hash", "survey_input_hash")}
    calls = []

    class Response:
        def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu):
            calls.append(None)
            return np.full((len(k), 3), float(q_a[0]) + 10*float(q_b[0]))

    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response",
                        lambda *args, **kwargs: Response())
    cache = PhysicalFDCache(tmp_path)
    first = physical_cell_means(likelihood, cell, object(), cache=cache,
                                provenance=provenance)
    varied = dict(fiducial)
    varied["ln10_10_As"] += .001
    old = first[pair](varied)
    local = f"bin00:{pair[1]}:{NUISANCE_NAMES[0]}"
    likelihood.fiducial[likelihood.names.index(local)] += 1
    varied[local] += 1
    second = physical_cell_means(likelihood, cell, object(), cache=cache,
                                 provenance=provenance)
    new = second[pair](varied)
    assert np.all(new - old == 10)
    assert len(calls) == 2


def test_independent_support_rejects_coordinated_population_omission():
    likelihood, cell, pair, _ = _cross_fixture()
    support = [_support(cell)]
    cell.metadata["active_populations"] = (pair[0],)
    cell.metadata["retained_pairs"] = [(pair[0], pair[0])]
    cell.metadata["retained_observations"] = [((pair[0], pair[0]), ell) for ell in (0,2,4)]
    cell.names = list(COSMO_NAMES) + [f"bin00:{pair[0]}:{name}" for name in NUISANCE_NAMES]
    cell.metadata["local_names"] = cell.names
    cell.derivatives = cell.derivatives[:, :3, :len(cell.names)]
    cell.covariance = cell.covariance[:, :3, :3]
    with pytest.raises(ValueError, match="independent support"):
        declare_fd_manifest(likelihood, [cell.metadata["cell_id"]], "B-run",
                            expected_support=support)


def test_independent_support_rejects_omitted_empty_cell():
    likelihood, cell, _, _ = _cross_fixture()
    support = [_support(cell), {**_support(cell),
                "cell_id":"bin00-radial01-nonoverlap", "region":"nonoverlap",
                "active_populations":(), "retained_pairs":[], "local_names":[],
                "empty_observation":True}]
    with pytest.raises(ValueError, match="independent support"):
        validate_support_schedule(likelihood, support)


def test_survey_derived_support_records_empty_observations():
    config = RevisionConfig("A")
    support = expected_support_schedule(config, default_survey_group())
    assert support
    assert len({row["cell_id"] for row in support}) == len(support)
    assert any(row["absent_observations"] and all(pair[0][0] != pair[0][1]
               for pair in row["absent_observations"]) for row in support)
    assert any(row["absent_observations"] for row in expected_support_schedule(
        RevisionConfig("D"), default_survey_group()))
    assert all(not row["absent_observations"] for row in expected_support_schedule(
        RevisionConfig("C"), default_survey_group()))


def test_direct_gate_binder_rejects_representative_physical_certificate(tmp_path):
    steps = [.005, .0025, .00125]
    coverage = [["B-run", "cell", ["DESI-ELG", "DESI-ELG"], 0, "Mnu", step]
                for step in steps]
    measured = {"status":"passed", "scope":"representative",
                "complete_run_coverage":False,
                "selected_cell_ids":["cell"], "all_cell_ids":["cell", "empty"],
                "expected_support_observations":[
                    {"cell_id":"cell", "empty_observation":False,
                     "retained_observations":[[["DESI-ELG", "DESI-ELG"], 0]]},
                    {"cell_id":"empty", "empty_observation":True,
                     "retained_observations":[]}],
                "expected_coverage":coverage, "measured_coverage":coverage,
                "failures":[],
                "column_results":[{"configuration":"B-run", "cell_id":"cell",
                                   "column":"Mnu", "steps":steps,
                                   "step_errors":[1e-4]*3,
                                   "step_tolerances":[1e-3]*3,
                                   "two_successive_pass":True}],
                "errors":[{"configuration":key[0], "cell_id":key[1],
                           "pair":key[2], "ell":key[3], "column":key[4],
                           "step":key[5], "error":1e-4, "limit":1e-3,
                           "reason":"relative"} for key in coverage]}
    doc = {"run_id":"B-run", "configuration":{"case":"B"},
           "provenance":{"source":"a"},
           "gates":{"ad_fd":{"status":"passed", "measured":measured,
                               "tolerance":1e-3, "reason":None}}}
    path = tmp_path / "representative.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="complete.*support|representative"):
        evidence_gate_record(path, "B-run", doc["configuration"], doc["provenance"],
                             "ad_fd", declared_coverage=coverage)
    for field in ("scope", "complete_run_coverage", "selected_cell_ids",
                  "all_cell_ids", "expected_support_observations"):
        measured.pop(field)
    measured["physical_scales"] = {"cell":1.}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="complete.*support|representative"):
        evidence_gate_record(path, "B-run", doc["configuration"], doc["provenance"],
                             "ad_fd", declared_coverage=coverage)


def test_direct_gate_binder_checks_full_independent_support_ledger(tmp_path):
    pair = ["DESI-ELG", "DESI-ELG"]
    steps = [.005, .0025, .00125]
    coverage = [["B-run", "cell", pair, ell, "Mnu", step]
                for ell in (0, 2, 4) for step in steps]
    support = [
        {"cell_id":"cell", "retained_pairs":[pair],
         "absent_observations":[], "local_names":["Mnu"]},
        {"cell_id":"empty", "retained_pairs":[],
         "absent_observations":[], "local_names":[]},
    ]
    observed = [
        {"cell_id":"cell", "retained_observations":[[pair, ell] for ell in (0, 2, 4)],
         "absent_observations":[], "columns":[{"name":"Mnu", "steps":steps}],
         "empty_observation":False},
        {"cell_id":"empty", "retained_observations":[],
         "absent_observations":[], "columns":[], "empty_observation":True},
    ]
    measured = {"status":"passed", "scope":"all_cells", "complete_run_coverage":True,
                "selected_cell_ids":["cell", "empty"], "all_cell_ids":["cell", "empty"],
                "expected_support_observations":observed,
                "expected_coverage":coverage, "measured_coverage":coverage,
                "failures":[], "column_results":[{
                    "configuration":"B-run", "cell_id":"cell", "column":"Mnu",
                    "steps":steps, "step_errors":[1e-4]*3,
                    "step_tolerances":[1e-3]*3, "two_successive_pass":True}],
                "errors":[{"configuration":key[0], "cell_id":key[1],
                           "pair":key[2], "ell":key[3], "column":key[4],
                           "step":key[5], "error":1e-4, "limit":1e-3,
                           "reason":"relative"} for key in coverage]}
    doc = {"run_id":"B-run", "configuration":{"case":"B"},
           "provenance":{"source":"a"},
           "gates":{"ad_fd":{"status":"passed", "measured":measured,
                               "tolerance":1e-3, "reason":None}}}
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(doc))
    record = evidence_gate_record(path, "B-run", doc["configuration"],
                                  doc["provenance"], "ad_fd",
                                  declared_coverage=coverage, declared_support=support)
    assert record["status"] == "passed"
    observed.pop()
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="support cell ledger"):
        evidence_gate_record(path, "B-run", doc["configuration"],
                             doc["provenance"], "ad_fd",
                             declared_coverage=coverage, declared_support=support)


def test_cache_rejects_changed_physical_identity_and_preserves_failure(tmp_path):
    cache = PhysicalFDCache(tmp_path)
    context = {"source":"abc", "shell_centers":[.015], "nbar":1e-4,
               "pair":("DESI-ELG", "DESI-LRG"), "steps":(.001,.0005,.00025)}
    assert cache.get_or_compute(context, lambda: np.ones((1, 3))).shape == (1, 3)
    assert np.all(cache.get_or_compute(context, lambda: np.zeros((1, 3))) == 1)
    changed = {**context, "nbar":2e-4}
    assert np.all(cache.get_or_compute(changed, lambda: np.zeros((1, 3))) == 0)
    failed = {**context, "source":"changed"}
    with pytest.raises(RuntimeError, match="physical failure"):
        cache.get_or_compute(failed, lambda: (_ for _ in ()).throw(RuntimeError("physical failure")))
    with pytest.raises(ValueError, match="cached physical FD failure"):
        cache.get_or_compute(failed, lambda: np.ones((1, 3)))


def test_one_coordinate_covariance_audit_preserves_failing_context(monkeypatch):
    likelihood, cell, _, _ = _cross_fixture()
    cell.metadata["theoretical_pairs"] = cell.metadata["retained_pairs"]
    likelihood.model = {"mean_n_mu":32, "covariance_n_mu":32}

    class Response:
        def __init__(self, cross):
            self.cross = cross

        def pkmu(self, k, mu, theta, q_a, q_b=None, q_cross=None):
            bad = float(theta[1]) > 2.
            return np.full((len(k), len(mu)), 2e4 if bad and self.cross else 0.)

    monkeypatch.setattr("pfsfog.revision_validation_physical.make_pair_response",
                        lambda *args, **kwargs: Response(args[4] is not None))
    with pytest.raises(CovarianceAuditFailure,
                       match=r"cell=bin00-radial00-overlap.*column=Mnu.*sign=\+.*k=0.015") as error:
        audit_cell_covariance_neighborhood(likelihood, cell, object())
    metrics = error.value.diagnostics
    assert metrics["min_scaled_eigenvalue"] < 0
    assert metrics["scaled_condition"] > 1
    assert metrics["scaled_residual"] < 1e-12
    assert metrics["context"].startswith("field cell=bin00-radial00-overlap")


def test_live_snapshot_exposes_geometry_covariance_and_derivative_counterexamples():
    likelihood, cell, _, _ = _cross_fixture()
    likelihood.F_like = np.eye(len(likelihood.names))
    likelihood.parameters = [{"name":name, "prior_sigma":None}
                             for name in likelihood.names]
    cell.metadata["local_names"] = cell.names
    old = likelihood_snapshot(likelihood)
    new = likelihood_snapshot(likelihood)
    assert compare_nested_likelihoods(old, new)["status"] == "passed"
    new["cells"][0]["geometry"]["volume"] = 2.
    assert "geometry mismatch" in " ".join(compare_nested_likelihoods(old, new)["reasons"])
    new = likelihood_snapshot(likelihood)
    new["cells"][0]["covariance"][0, 0, 0] *= 2
    assert "covariance submatrix mismatch" in " ".join(compare_nested_likelihoods(old, new)["reasons"])
    new = likelihood_snapshot(likelihood)
    new["cells"][0]["derivatives"][0, 0, 0] = 1.
    assert "derivative definition mismatch" in " ".join(compare_nested_likelihoods(old, new)["reasons"])


def test_live_snapshot_preserves_shared_prior_centers_and_widths():
    likelihood, _, _, _ = _cross_fixture()
    likelihood.F_like = np.eye(len(likelihood.names))
    likelihood.parameters = [{"name":name, "prior_sigma":1.}
                             for name in likelihood.names]
    old = likelihood_snapshot(likelihood)
    new = likelihood_snapshot(likelihood)
    new["prior_sigma"][0] = 2.
    assert "shared prior_sigma mismatch" in compare_nested_likelihoods(old, new)["reasons"]
    new = likelihood_snapshot(likelihood)
    new["fiducial"][0] += .1
    assert "shared fiducial mismatch" in compare_nested_likelihoods(old, new)["reasons"]
