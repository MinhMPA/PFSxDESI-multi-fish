"""Frozen revision artifact contract, using an analytic eight-bin fixture."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pfsfog.revision_forecast import (CellLikelihood, PriorResult, RevisionConfig,
                                      RevisionLikelihood, make_shell_grid)
from pfsfog.revision_artifacts import (MANDATORY_GATES, finalize_revision_artifacts,
                                       _configuration, likelihood_identity, load_cached_revision_likelihood,
                                       load_revision_artifacts, write_revision_artifacts)
from pfsfog.revision_validation_suite import EvidenceJournal


def fixture_result():
    config = RevisionConfig("B", kmax=.15)
    names = ["ln10_10_As", "Mnu", "Omegam", "bin00:DESI-ELG:c2"]
    fisher = np.diag([4., 9., 16., 2.])
    fid = np.array([3., .06, .31, 0.])
    parameters = [dict(name=name, role="cosmo" if i < 3 else "tracer",
                       nominal_bin=None if i < 3 else 0,
                       tracers=[] if i < 3 else ["DESI-ELG"],
                       coordinate=name.split(":")[-1],
                       unit="(Mpc/h)^2" if i == 3 else ("eV" if i == 1 else ""),
                       fiducial=float(fid[i]),
                       prior_sigma=[None, 5., 1., 30.][i])
                  for i, name in enumerate(names)]
    cells = [CellLikelihood(fisher, names,
                            dict(cell_id="bin00-radial00-overlap", nominal_bin_index=0,
                                 nominal_bin=(.4, .6), radial_bin=(.4, .5),
                                 model_redshift=.5, region="overlap", volume=100.,
                                 nbar={"DESI-ELG":.0002}, candidate_populations=("DESI-ELG",),
                                 active_populations=("DESI-ELG",),
                                 theoretical_pairs=[("DESI-ELG", "DESI-ELG")],
                                 retained_pairs=[("DESI-ELG", "DESI-ELG")],
                                 retained_observations=[(("DESI-ELG", "DESI-ELG"), 0)],
                                 local_names=names),
                            dict(expected_shells=28, processed_shells=28,
                                 field_min_scaled_eigenvalue=.5,
                                 covariance_min_scaled_eigenvalue=.4),
                            np.zeros((1, 1, 4)), np.eye(1)[None],
                            np.full((28, 3), 7.))]
    empty = CellLikelihood(np.zeros((0, 0)), [],
                           dict(cell_id="bin00-radial00-nonoverlap", nominal_bin_index=0,
                                nominal_bin=(.4, .6), radial_bin=(.4, .5),
                                model_redshift=.5, region="nonoverlap", volume=0.,
                                nbar={}, candidate_populations=(), active_populations=(),
                                theoretical_pairs=[], retained_pairs=[], retained_observations=[],
                                local_names=[]),
                           dict(reason="zero_volume", expected_shells=0, processed_shells=0,
                                field_min_scaled_eigenvalue=None,
                                covariance_min_scaled_eigenvalue=None),
                           np.zeros((0, 0, 0)), np.zeros((0, 0, 0)))
    cells.append(empty)
    bins = [fisher] + [np.zeros_like(fisher) for _ in range(7)]
    like = RevisionLikelihood(config, fisher, names, fid, parameters, cells, bins,
                              make_shell_grid(.01, .15, .005),
                              {"h":.6736, "omega_b":.02237, "omega_cdm":.12,
                               "n_s":.9649, "ln10_10_As":3., "mnu":.06, "Omega_m":.31},
                              {"backend":"PowerSpectrum1Loop", "do_irres":False,
                               "subtract_k0_limit":True, "fft_kmin":1e-5,
                               "fft_kmax":1e3, "fft_n":256,
                               "mean_n_mu":32, "covariance_n_mu":32,
                               "ap_interpolation_nk":2048,
                               "ap_refinement_nk":[4096, 8192],
                               "ap_interpolation_nmu":101,
                               "fixed_observed_geometry":True, "fixed_covariance":True,
                               "covariance_derivative_trace":False, "serial":True},
                              {"expected_shells":28, "processed_shells":28})
    prior_c = np.diag([0., 1/25., 1., 0.])
    prior_n = np.diag([0., 0., 0., 1/900.])
    total = fisher + prior_c + prior_n
    covariance = np.linalg.inv(total)
    fixed = np.linalg.inv(total[:3, :3])
    z = np.array([(a+b)/2 for a,b in config.nominal_bins])
    jac = np.stack([np.ones(8), z, z*z], axis=1)
    return PriorResult(like, 1., parameters, prior_c, prior_n, total, covariance,
                       covariance[:3, :3], fixed, z, 2+z, jac,
                       jac @ covariance[:3, :3] @ jac.T, jac @ fixed @ jac.T,
                       {"global_solve":{"scaled_inverse_residual":0.},
                        "cosmology_block_solve":{"scaled_inverse_residual":0.},
                        "q13_penalty_min":1.})


def provenance():
    return {"source_content_hash":"source", "dependency_hash":"deps",
            "network_hash":"network", "survey_input_hash":"survey",
            "backend_snapshot":"backend", "command":"test", "precision":"float64",
            "model_flags":{"backend_class":"PowerSpectrum1Loop", "do_irres":False,
                           "subtract_k0_limit":True, "fft_kmin":1e-5,
                           "fft_kmax":1e3, "fft_n":256,
                           "ap_interpolation_nk":2048,
                           "ap_refinement_nk":[4096, 8192],
                           "ap_interpolation_nmu":101,
                           "fixed_observed_geometry":True, "fixed_covariance":True,
                           "covariance_derivative_trace":False}}


def gates(status="pending", *, root=None, result=None, source=None):
    if status == "passed":
        source = provenance() if source is None else source
        config = result.likelihood.config
        configuration = _configuration(config, result.nuisance_factor,
                                       requested_dk=source.get("requested_dk", config.dk),
                                       model_flags=source.get("model_flags"))
        from pfsfog.revision_artifacts import run_identity
        journal = EvidenceJournal(root, run_identity(configuration, source),
                                  configuration, source)
        for gate in MANDATORY_GATES:
            if gate != "save_reload":
                measured = {"scope":"all_cells", "complete_run_coverage":True} if gate == "ad_fd" else {"value":1.}
                journal.set_gate(gate, "passed", measured, .01, None)
        digest = hashlib.sha256(journal.path.read_bytes()).hexdigest()
        return {gate:{**record, "evidence_path":str(journal.path),
                      "evidence_hash":digest}
                for gate, record in journal.document["gates"].items()}
    return {gate:{"status":status, "measured":None, "tolerance":None,
                  "evidence_path":None, "evidence_hash":None,
                  "reason":"measurement not completed"}
            for gate in MANDATORY_GATES}


def complete_artifact(result, root, source=None):
    source = provenance() if source is None else source
    path = write_revision_artifacts(result, root, source, gates("pending"))
    return finalize_revision_artifacts(path, {"runs":{path.name:gates("passed", root=root,
                                                                 result=result, source=source)}})


def _rehash_journal(records, change):
    evidence = Path(records["domain"]["evidence_path"])
    document = json.loads(evidence.read_text())
    change(document)
    evidence.write_text(json.dumps(document))
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    for record in records.values():
        record["evidence_hash"] = digest


def test_finalizer_rejects_other_run_pending_journal_with_passed_outer_gates(tmp_path):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates())
    records = gates("passed", root=tmp_path, result=result)
    _rehash_journal(records, lambda document: document.update(
        run_id="C-other-source-identity",
        gates={gate:{"status":"pending", "measured":None, "tolerance":None,
                     "reason":"not measured"} for gate in records}))
    bundle = {"runs":{path.name:records}}
    with pytest.raises(ValueError, match="identity|run ID|run_id"):
        finalize_revision_artifacts(path, bundle)
    assert json.loads((path / "manifest.json").read_text())["status"] == "building"


@pytest.mark.parametrize("field,value", [
    ("measured", {"value":2.}), ("tolerance", .02), ("reason", "changed")])
def test_finalizer_rejects_changed_outer_gate_field(tmp_path, field, value):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates())
    records = gates("passed", root=tmp_path, result=result)
    records["domain"][field] = value
    with pytest.raises(ValueError, match="domain.*evidence|domain.*journal"):
        finalize_revision_artifacts(path, {"runs":{path.name:records}})


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_finalizer_rejects_inner_nonpass_behind_outer_pass(tmp_path, status):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates())
    records = gates("passed", root=tmp_path, result=result)
    _rehash_journal(records, lambda document: document["gates"]["domain"].update(status=status))
    with pytest.raises(ValueError, match="domain.*evidence|domain.*journal"):
        finalize_revision_artifacts(path, {"runs":{path.name:records}})


@pytest.mark.parametrize("field", ["configuration", "provenance"])
def test_finalizer_rejects_wrong_journal_configuration_or_physical_source(tmp_path, field):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates())
    records = gates("passed", root=tmp_path, result=result)
    def change(document):
        if field == "configuration":
            document[field]["kmax"] = .20
        else:
            document[field]["source_content_hash"] = "other-source"
    _rehash_journal(records, change)
    with pytest.raises(ValueError, match="configuration|provenance|identity"):
        finalize_revision_artifacts(path, {"runs":{path.name:records}})


def test_command_and_runtime_only_provenance_changes_keep_evidence_identity(tmp_path):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates())
    records = gates("passed", root=tmp_path, result=result)
    _rehash_journal(records, lambda document: document["provenance"].update(
        command="another invocation", runtime={"elapsed_seconds":9.}))
    finalize_revision_artifacts(path, {"runs":{path.name:records}})
    assert load_revision_artifacts(path)[0]["status"] == "complete"


def test_reload_rejects_rehashed_outer_measurement_that_disagrees_with_journal(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    validation_path = path / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation["domain"]["measured"] = {"value":2.}
    validation_path.write_text(json.dumps(validation))
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["validation.json"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="domain.*evidence|domain.*journal"):
        load_revision_artifacts(path)


def test_cli_identity_survives_producer_execution_and_finalization(
        tmp_path, monkeypatch, capsys):
    from scripts import run_revision, validate_revision

    source = {**provenance(), "requested_dk":.005}
    monkeypatch.setattr(run_revision, "_load_backend", lambda:object())
    monkeypatch.setattr(run_revision, "collect_provenance",
                        lambda *args, **kwargs:source)
    monkeypatch.setattr(run_revision, "default_survey_group", lambda:object())
    monkeypatch.setattr(run_revision, "FiducialCosmology", lambda:object())
    monkeypatch.setattr(run_revision, "build_revision_likelihood",
                        lambda *args:fixture_result().likelihood)
    monkeypatch.setattr(run_revision, "apply_revision_priors",
                        lambda *args:fixture_result())
    options = ["--case", "B", "--kmax", ".15", "--dk", ".005",
               "--requested-dk", ".005"]
    assert run_revision.main([*options, "--identities"]) == 0
    producer_id = json.loads(capsys.readouterr().out)["run_ids"][0]
    assert validate_revision.main([*options, "--identities"]) == 0
    assert json.loads(capsys.readouterr().out)["run_ids"][0] == producer_id

    assert run_revision.main([*options, "--execute", "--output-root",
                              str(tmp_path / "runs")]) == 0
    path = Path(json.loads(capsys.readouterr().out)["paths"][0])
    assert path.name == producer_id
    assert json.loads((path / "manifest.json").read_text())["status"] == "building"

    bundle_path = tmp_path / "gates.json"
    bundle_path.write_text(json.dumps({"runs":{producer_id:gates("passed", root=tmp_path,
                                                                 result=fixture_result(), source=source)}}))
    assert run_revision.main(["--finalize", str(path), "--gates",
                              str(bundle_path)]) == 0
    assert json.loads(capsys.readouterr().out)["path"] == str(path)
    manifest, _, _, _ = load_revision_artifacts(path)
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == producer_id


def test_bare_passed_gates_never_complete_direct_writer(tmp_path):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(),
                                    gates("passed", root=tmp_path, result=result))
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "building"
    with pytest.raises(ValueError, match="complete"):
        load_revision_artifacts(path)
    other = {"runs":{"B-k015-p1-all-other":gates()}}
    with pytest.raises(ValueError, match="run identity"):
        finalize_revision_artifacts(path, other)
    assert json.loads((path / "manifest.json").read_text())["status"] == "building"


@pytest.mark.parametrize("key,value", [("fft_n", 512), ("fft_kmin", 2e-5),
                                          ("ap_refinement_nk", [16384, 32768]),
                                          ("fixed_observed_geometry", False),
                                          ("covariance_derivative_trace", True)])
def test_unexecuted_model_claim_is_rejected_before_persistence(tmp_path, key, value):
    source = provenance()
    source["model_flags"][key] = value
    with pytest.raises(ValueError, match=key):
        write_revision_artifacts(fixture_result(), tmp_path, source, gates("pending"))
    assert not list(tmp_path.iterdir())


def test_unrecognized_model_claim_is_not_copied_into_manifest(tmp_path):
    source = provenance()
    source["model_flags"]["unverified_flag"] = True
    with pytest.raises(ValueError, match="model_flags"):
        write_revision_artifacts(fixture_result(), tmp_path, source, gates("pending"))
    assert not list(tmp_path.iterdir())


def test_nuisance_convention_records_physical_coordinates_and_assumptions(tmp_path):
    path = write_revision_artifacts(fixture_result(), tmp_path, provenance(), gates("pending"))
    manifest = json.loads((path / "manifest.json").read_text())
    convention = manifest["nuisance_convention"]
    assert len(convention["tracer_definitions"]) == 11
    assert convention["amplitude_factor"] == "A = (sigma8(theta,z) / sigma8(reference,z))^2"
    assert convention["tracer_definitions"]["c2"]["definition"] == "Q2 = A * C2"
    assert convention["tracer_definitions"]["c2"]["unit"] == "(Mpc/h)^2"
    assert convention["tracer_definitions"]["c_tilde"]["definition"] == "T = A^2 * c_tilde_backend"
    assert convention["tracer_definitions"]["bGamma3"]["definition"] == "G = A * bGamma3_backend"
    assert convention["cross_stochastic"]["base_prior_sigma"] == {"Pshot_cross":1., "a2_cross":1.}
    assert convention["fiducial_ctilde"]["DESI"] == 400.
    assert convention["fiducial_ctilde"]["PFS"] == 126.5625
    assert convention["active_c1"] is False


@pytest.mark.parametrize("key,value", [("role", "cosmo"), ("nominal_bin", 1),
                                          ("tracers", ["PFS-ELG"]),
                                          ("coordinate", "c0"), ("unit", "")])
def test_manifest_only_parameter_semantic_tampering_is_rejected(tmp_path, key, value):
    path = complete_artifact(fixture_result(), tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"][3][key] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parameter metadata"):
        load_revision_artifacts(path)


def test_configuration_fft_is_actual_and_part_of_run_and_cache_identity():
    from pfsfog.revision_artifacts import _configuration, run_identity
    source = provenance()
    config = RevisionConfig("B")
    actual = _configuration(config, 1., model_flags=source["model_flags"])
    assert actual["fft_kmin"] == 1e-5
    assert actual["fft_kmax"] == 1e3
    assert actual["fft_n"] == 256
    changed = {**actual, "fft_n":512}
    assert run_identity(actual, source) != run_identity(changed, source)
    assert likelihood_identity(actual, source) != likelihood_identity(changed, source)


def test_reload_gate_reports_method_without_invented_zero_residual(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    validation = json.loads((path / "validation.json").read_text())
    measured = validation["save_reload"]["measured"]
    assert measured["passed"] is True
    assert "method" in measured
    assert "relative_error" not in measured


def test_complete_round_trip_recomputes_all_results(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    manifest, arrays, rows, validation = load_revision_artifacts(path)
    assert manifest["status"] == "complete"
    assert len(manifest["nominal_bins"]) == 8
    assert manifest["cells"][1]["validation"]["reason"] == "zero_volume"
    assert manifest["model"]["fft_n"] == 256
    assert manifest["model"]["subtract_k0_limit"] is True
    assert arrays["cell_F_like__001"].shape == (0, 0)
    cached = load_cached_revision_likelihood(
        path, likelihood_identity(manifest["configuration"], provenance()), provenance())
    np.testing.assert_array_equal(cached.cells[0].fiducial_mean,
                                  fixture_result().likelihood.cells[0].fiducial_mean)
    assert len(rows) == 11
    assert next(row for row in rows if row["quantity"] == "fsigma8" and row["z"] == "1.1")
    assert validation["save_reload"]["status"] == "passed"


def test_candidate_ap_table_round_trips_through_artifact_identity(tmp_path):
    result = fixture_result()
    result.likelihood.config = replace(result.likelihood.config,
                                       ap_interpolation_nk=8192,
                                       ap_refinement_nk=(16384, 32768))
    result.likelihood.model["ap_interpolation_nk"] = 8192
    result.likelihood.model["ap_refinement_nk"] = [16384, 32768]
    source = provenance()
    source["model_flags"]["ap_interpolation_nk"] = 8192
    source["model_flags"]["ap_refinement_nk"] = [16384, 32768]
    path = complete_artifact(result, tmp_path, source)
    manifest, _, _, _ = load_revision_artifacts(path)
    assert manifest["configuration"]["ap_refinement_nk"] == [16384, 32768]
    cached = load_cached_revision_likelihood(
        path, likelihood_identity(manifest["configuration"], source), source)
    assert cached.config.ap_interpolation_nk == 8192
    assert cached.config.ap_refinement_nk == (16384, 32768)


def test_incomplete_run_and_wrong_labels_or_hash_are_rejected(tmp_path):
    path = write_revision_artifacts(fixture_result(), tmp_path, provenance(), gates("pending"))
    with pytest.raises(ValueError, match="complete"):
        load_revision_artifacts(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gate"):
        load_revision_artifacts(path)
    path2 = complete_artifact(fixture_result(), tmp_path / "other")
    with np.load(path2 / "matrices.npz", allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    arrays["parameter_names"][0] = "wrong"
    np.savez(path2 / "matrices.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_revision_artifacts(path2)
    manifest = json.loads((path2 / "manifest.json").read_text())
    manifest["artifacts"]["matrices.npz"] = hashlib.sha256((path2 / "matrices.npz").read_bytes()).hexdigest()
    (path2 / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="label"):
        load_revision_artifacts(path2)


def test_rehashed_inconsistent_matrix_is_rejected(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    with np.load(path / "matrices.npz", allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    arrays["growth_cov_marg"][0, 0] += .1
    np.savez(path / "matrices.npz", **arrays)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["artifacts"]["matrices.npz"] = hashlib.sha256((path / "matrices.npz").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="growth_cov_marg"):
        load_revision_artifacts(path)


def test_failed_construction_preserves_diagnostics_and_collision(tmp_path):
    config = RevisionConfig("C")
    validation = {"status":"failed", "failure":{"cell_id":"bin03-radial01-overlap",
                                               "k":.1975, "exception":"nonpositive covariance"}}
    path = write_revision_artifacts(None, tmp_path, {**provenance(), "configuration":config}, validation)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (path / "summary.csv").exists()
    assert "bin03-radial01-overlap" in (path / "run.log").read_text()
    with pytest.raises(FileExistsError):
        write_revision_artifacts(None, tmp_path, {**provenance(), "configuration":config}, validation)


def test_wrong_backend_flag_is_rejected_instead_of_recorded_as_false(tmp_path):
    result = fixture_result()
    result.likelihood.model["do_irres"] = True
    with pytest.raises(ValueError, match="do_irres"):
        write_revision_artifacts(result, tmp_path, provenance(), gates())


def test_one_bin_result_cannot_be_published_as_complete(tmp_path):
    result = fixture_result()
    result.likelihood.config = RevisionConfig("B", nominal_bins=((1., 1.2),))
    with pytest.raises(ValueError, match="eight nominal bins"):
        write_revision_artifacts(result, tmp_path, provenance(), gates())


def test_nonstandard_eight_bin_schedule_cannot_be_published(tmp_path):
    result = fixture_result()
    bins = ((.4, .55),) + tuple(result.likelihood.config.nominal_bins[1:])
    result.likelihood.config = RevisionConfig("B", kmax=.15, nominal_bins=bins)
    with pytest.raises(ValueError, match="frozen nominal bins"):
        write_revision_artifacts(result, tmp_path, provenance(), gates())


def test_missing_or_conflicting_actual_model_flags_cannot_publish(tmp_path):
    source = provenance()
    del source["model_flags"]
    with pytest.raises(ValueError, match="model_flags"):
        write_revision_artifacts(fixture_result(), tmp_path, source, gates())
    source = provenance()
    source["model_flags"]["do_irres"] = True
    with pytest.raises(ValueError, match="do_irres"):
        write_revision_artifacts(fixture_result(), tmp_path, source, gates())


def test_rehashed_bad_cell_positivity_and_shell_inventory_are_rejected(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["validation"]["field_min_scaled_eigenvalue"] = -.1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positivity"):
        load_revision_artifacts(path)
    manifest["cells"][0]["validation"]["field_min_scaled_eigenvalue"] = .5
    manifest["cells"][0]["validation"]["expected_shells"] = 1
    manifest["cells"][0]["validation"]["processed_shells"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shell"):
        load_revision_artifacts(path)


def test_building_artifact_finalizes_same_matrices_after_exact_gate_evidence(tmp_path):
    result = fixture_result()
    path = write_revision_artifacts(result, tmp_path, provenance(), gates("pending"))
    before = json.loads((path / "manifest.json").read_text())
    assert before["status"] == "building"
    bundle = {"runs":{before["run_id"]:gates("passed", root=tmp_path, result=result)}}
    assert finalize_revision_artifacts(path, bundle) == path
    after, arrays, rows, validation = load_revision_artifacts(path)
    assert after["status"] == "complete"
    assert before["artifacts"]["matrices.npz"] == after["artifacts"]["matrices.npz"]
    assert before["artifacts"]["summary.csv"] == after["artifacts"]["summary.csv"]
    assert len(rows) == 11
    with pytest.raises(ValueError, match="building"):
        finalize_revision_artifacts(path, bundle)


def test_finalize_rejects_wrong_identity_and_corrupted_building_payload(tmp_path):
    path = write_revision_artifacts(fixture_result(), tmp_path, provenance(), gates("pending"))
    manifest = json.loads((path / "manifest.json").read_text())
    with pytest.raises(ValueError, match="run identity"):
        finalize_revision_artifacts(path, {"runs":{"other":gates()}})
    assert json.loads((path / "manifest.json").read_text())["status"] == "building"
    (path / "matrices.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        finalize_revision_artifacts(path, {"runs":{manifest["run_id"]:gates()}})
    assert json.loads((path / "manifest.json").read_text())["status"] == "building"


def test_failed_external_gate_preserves_failure_without_uncertainty_rows(tmp_path):
    validation = gates()
    validation["domain"]["status"] = "failed"
    validation["domain"]["reason"] = "reference cosmology outside released domain"
    path = write_revision_artifacts(fixture_result(), tmp_path, provenance(), validation)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (path / "summary.csv").exists()
    assert "domain" in (path / "run.log").read_text()
    with pytest.raises(ValueError, match="complete"):
        load_revision_artifacts(path)


def test_building_factor_one_artifact_is_exact_identity_likelihood_cache(tmp_path):
    source = provenance()
    path = write_revision_artifacts(fixture_result(), tmp_path, source, gates("pending"))
    manifest = json.loads((path / "manifest.json").read_text())
    identity = likelihood_identity(manifest["configuration"], source)
    cached = load_cached_revision_likelihood(path, identity, source)
    assert cached.names == fixture_result().names
    np.testing.assert_allclose(cached.F_like, fixture_result().F_like)
    assert cached.cells[1].validation["reason"] == "zero_volume"
    with pytest.raises(ValueError, match="identity"):
        load_cached_revision_likelihood(path, identity, {**source, "network_hash":"changed"})


def test_nonfinite_json_metadata_is_rejected(tmp_path):
    path = complete_artifact(fixture_result(), tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["elapsed_seconds"] = float("nan")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonfinite JSON"):
        load_revision_artifacts(path)


def test_nonfinite_gate_measurement_is_rejected_before_directory_creation(tmp_path):
    validation = gates("pending")
    validation["ad_fd"]["measured"] = {"relative_error":float("nan")}
    with pytest.raises(ValueError, match="JSON"):
        write_revision_artifacts(fixture_result(), tmp_path, provenance(), validation)
    assert not list(tmp_path.iterdir())


def test_reused_factor_one_likelihood_persists_effective_prior_factor(tmp_path):
    result = fixture_result()
    result.nuisance_factor = 3.
    result.parameters[-1]["prior_sigma"] = 90.
    result.F_prior_nuisance[3,3] = 1/90.**2
    result.F_total = result.F_like + result.F_prior_cosmo + result.F_prior_nuisance
    result.covariance = np.linalg.inv(result.F_total)
    result.cov_cosmo_marg = result.covariance[:3,:3]
    result.cov_cosmo_nuisance_fixed = np.linalg.inv(result.F_total[:3,:3])
    jac = result.growth_jacobian
    result.growth_cov_marg = jac @ result.cov_cosmo_marg @ jac.T
    result.growth_cov_nuisance_fixed = jac @ result.cov_cosmo_nuisance_fixed @ jac.T
    path = complete_artifact(result, tmp_path)
    manifest, _, _, _ = load_revision_artifacts(path)
    assert "-p3-" in path.name
    assert manifest["configuration"]["nuisance_factor"] == 3.
    assert manifest["parameters"][-1]["prior_sigma"] == 90.


def test_scaled_solve_uses_declared_absolute_residual_gate():
    from pfsfog.revision_artifacts import _inverse
    fisher = np.array([[1., .999999], [.999999, 1.]])
    inverse = _inverse(fisher, "near_degenerate_valid")
    assert np.max(np.abs(fisher @ inverse - np.eye(2))) < 1e-10
