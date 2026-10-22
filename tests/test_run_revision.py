"""Revision driver inventory, execution gates, and likelihood reuse."""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pfsfog.revision_artifacts import (MANDATORY_GATES, _configuration,
                                       likelihood_identity, run_identity)
from pfsfog.revision_forecast import RevisionConfig
from scripts import run_revision


def test_inventory_is_exactly_thirteen_independent_configurations_and_nine_likelihoods():
    configs = run_revision.inventory()
    keys = [(c.case, c.kmax, c.nuisance_factor, c.exclude_elg1) for c in configs]
    assert keys == [(case, .20, 1., False) for case in ("I", "A", "B", "D", "C")] + [
        (case, .15, 1., False) for case in ("B", "C")
    ] + [(case, .20, factor, False) for factor in (3., 10.)
         for case in ("B", "C")] + [
        (case, .20, 1., True) for case in ("B", "C")]
    assert len(keys) == len(set(keys)) == 13
    provenance = {"source_content_hash":"source", "network_hash":"network"}
    likelihoods = [likelihood_identity(_configuration(c, c.nuisance_factor), provenance)
                   for c in configs]
    assert len(set(likelihoods)) == 9
    assert likelihoods[2] == likelihoods[7] == likelihoods[9]
    assert likelihoods[4] == likelihoods[8] == likelihoods[10]


def test_likelihood_reuse_key_omits_only_prior_factor():
    p = {"source_content_hash":"one", "backend_snapshot":"two",
         "dependency_hash":"three", "network_hash":"four", "survey_input_hash":"five"}
    one = _configuration(RevisionConfig("B"), 1)
    three = _configuration(RevisionConfig("B", nuisance_factor=3), 3)
    assert likelihood_identity(one, p) == likelihood_identity(three, p)
    for changed in (RevisionConfig("C"), RevisionConfig("B", kmax=.18),
                    RevisionConfig("B", exclude_elg1=True), RevisionConfig("B", n_mu=64)):
        assert likelihood_identity(one, p) != likelihood_identity(_configuration(changed, 1), p)
    assert likelihood_identity(one, p) != likelihood_identity(one, {**p, "network_hash":"changed"})


def test_default_single_and_grid_are_inventory_only(capsys, tmp_path):
    assert run_revision.main(["--case", "B", "--kmax", ".20", "--nuisance-factor", "1",
                              "--output-root", str(tmp_path)]) == 0
    single = json.loads(capsys.readouterr().out)
    assert single["scope"] == "validation_inventory"
    assert len(single["configurations"]) == 1
    assert single["configurations"][0]["case"] == "B"
    assert not list(tmp_path.iterdir())
    assert run_revision.main(["--grid", "--output-root", str(tmp_path)]) == 0
    grid = json.loads(capsys.readouterr().out)
    assert len(grid["configurations"]) == 13
    assert not list(tmp_path.iterdir())


def test_identity_inventory_exposes_exact_gate_bundle_keys_without_running(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(run_revision, "_load_backend", lambda: object())
    monkeypatch.setattr(run_revision, "collect_provenance", lambda *args, **kwargs: {"source_content_hash":"source",
                                                                             "requested_dk":.005})
    assert run_revision.main(["--case", "B", "--dk", ".0025", "--identities",
                              "--output-root", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scope"] == "validation_inventory"
    assert plan["configurations"][0]["adopted_dk"] == .0025
    assert plan["configurations"][0]["requested_dk"] == .005
    assert plan["run_ids"][0].startswith("B-k020-p1-all-")
    assert not list(tmp_path.iterdir())


def test_validation_execution_can_start_without_gates_but_grid_requires_bundle(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_revision, "execute", lambda configs, root, bundle, **kwargs:
                        calls.append((configs, bundle)) or [])
    assert run_revision.main(["--case", "B", "--execute", "--output-root", str(tmp_path)]) == 0
    assert len(calls) == 1 and len(calls[0][0]) == 1 and calls[0][1] is None
    with pytest.raises(ValueError, match="gate file"):
        run_revision.main(["--grid", "--execute", "--output-root", str(tmp_path)])
    gate_path = tmp_path / "gates.json"
    gate_path.write_text(json.dumps({gate:{"status":"passed"} for gate in MANDATORY_GATES}))
    with pytest.raises(ValueError, match="evidence"):
        run_revision.load_gates(gate_path)
    assert not list(tmp_path.glob("B-*"))


def test_pending_gate_records_allow_validation_but_do_not_claim_passed(tmp_path):
    config = run_revision.inventory()[0]
    source = {"source_content_hash":"source"}
    run_id = run_identity(_configuration(config, config.nuisance_factor), source)
    pending = {"status":"pending", "measured":None, "tolerance":None,
               "evidence_path":None, "evidence_hash":None, "reason":"awaiting FD audit"}
    bundle = tmp_path / "pending.json"
    bundle.write_text(json.dumps({"runs":{run_id:{gate:pending for gate in MANDATORY_GATES
                                                  if gate != "save_reload"}}}))
    loaded = run_revision.load_gates(bundle)
    assert run_revision.gates_for_run(config, source, loaded)["domain"]["status"] == "pending"


def test_nonfinite_gate_file_is_rejected_before_execution(tmp_path):
    bundle = tmp_path / "nonfinite.json"
    bundle.write_text('{"runs":{"x":{"domain":{"measured":NaN}}}}')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        run_revision.load_gates(bundle)


def test_gate_bundle_is_bound_to_exact_run_and_hashed_evidence(tmp_path):
    from pfsfog.revision_validation_suite import EvidenceJournal
    config = run_revision.inventory()[0]
    source = {"source_content_hash":"source"}
    configuration = _configuration(config, config.nuisance_factor)
    run_id = run_identity(configuration, source)
    journal = EvidenceJournal(tmp_path, run_id, configuration, source)
    journal.set_gate("domain", "passed", {"error":.001}, .01, None)
    evidence = journal.path
    bundle = tmp_path / "gates-valid.json"
    bundle.write_text(journal.write_bundle().read_text())
    loaded = run_revision.load_gates(bundle)
    assert run_revision.gates_for_run(config, source, loaded)["domain"]["status"] == "passed"
    with pytest.raises(ValueError, match="run identity"):
        run_revision.gates_for_run(run_revision.inventory()[1], source, loaded)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        run_revision.load_gates(bundle)


def test_loader_rejects_generated_journal_rekeyed_to_other_run(tmp_path):
    from pfsfog.revision_validation_suite import EvidenceJournal
    source = {"source_content_hash":"source"}
    config_c = run_revision.inventory()[4]
    config_b = run_revision.inventory()[2]
    configuration = _configuration(config_c, config_c.nuisance_factor)
    journal = EvidenceJournal(tmp_path, run_identity(configuration, source),
                              configuration, source)
    journal.set_gate("domain", "passed", {"value":1.}, .01, None)
    bundle = json.loads(journal.write_bundle().read_text())
    records = bundle["runs"].pop(journal.document["run_id"])
    bundle["runs"][run_identity(_configuration(config_b, config_b.nuisance_factor), source)] = records
    path = tmp_path / "wrong-run-gates.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="identity|run ID|run_id"):
        run_revision.load_gates(path)


@pytest.mark.parametrize("field", ["configuration", "provenance", "measured"])
def test_loader_rejects_rehashed_journal_that_contradicts_run_or_outer_record(tmp_path, field):
    from pfsfog.revision_validation_suite import EvidenceJournal
    config = run_revision.inventory()[2]
    source = {"source_content_hash":"source"}
    configuration = _configuration(config, config.nuisance_factor)
    run_id = run_identity(configuration, source)
    journal = EvidenceJournal(tmp_path, run_id, configuration, source)
    journal.set_gate("domain", "passed", {"value":1.}, .01, None)
    bundle = json.loads(journal.write_bundle().read_text())
    if field == "measured":
        bundle["runs"][run_id]["domain"]["measured"] = {"value":2.}
    else:
        document = json.loads(journal.path.read_text())
        if field == "configuration":
            document[field]["kmax"] = .15
        else:
            document[field]["source_content_hash"] = "other-source"
        journal.path.write_text(json.dumps(document))
        digest = hashlib.sha256(journal.path.read_bytes()).hexdigest()
        bundle["runs"][run_id]["domain"]["evidence_hash"] = digest
    path = tmp_path / "changed-gates.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="identity|journal"):
        run_revision.load_gates(path)


def test_grid_reuses_factor_one_likelihood_only_for_matching_inputs(monkeypatch, tmp_path):
    builds = []
    writes = []
    finalizations = []
    def build(config, survey, cosmo, backend):
        item = SimpleNamespace(config=config)
        builds.append(item)
        return item
    def priors(likelihood, factor):
        return SimpleNamespace(likelihood=likelihood, nuisance_factor=factor)
    def write(result, root, provenance, gates):
        writes.append((result.likelihood, result.nuisance_factor))
        return Path(root) / str(len(writes))
    monkeypatch.setattr(run_revision, "build_revision_likelihood", build)
    monkeypatch.setattr(run_revision, "apply_revision_priors", priors)
    monkeypatch.setattr(run_revision, "write_revision_artifacts", write)
    monkeypatch.setattr(run_revision, "finalize_revision_artifacts", lambda path, bundle:
                        finalizations.append((path, bundle)) or path)
    monkeypatch.setattr(run_revision, "_load_backend", lambda: object())
    monkeypatch.setattr(run_revision, "default_survey_group", lambda: object())
    monkeypatch.setattr(run_revision, "FiducialCosmology", lambda: object())
    monkeypatch.setattr(run_revision, "collect_provenance", lambda *args, **kwargs: {"source_content_hash":"one"})
    configs = run_revision.inventory()
    source = {"source_content_hash":"one"}
    gate = {"status":"passed", "measured":{}, "tolerance":1.,
            "evidence_path":"dummy", "evidence_hash":"digest", "reason":None}
    bundle = {"runs":{run_identity(_configuration(c, c.nuisance_factor), source):
                      {name:gate for name in MANDATORY_GATES if name != "save_reload"}
                      for c in configs}}
    run_revision.execute(configs, tmp_path, bundle)
    assert len(writes) == 13
    assert len(finalizations) == 13
    assert all(item[1] is bundle for item in finalizations)
    assert len(builds) == 9
    baseline = {item.config.case:item for item, factor in writes
                if factor == 1 and item.config.kmax == .20
                and not item.config.exclude_elg1}
    assert all(item is baseline[item.config.case] for item, factor in writes
               if factor in (3, 10))


def test_separate_prior_sweep_uses_only_matching_disk_factor_one_cache(monkeypatch, tmp_path):
    config = RevisionConfig("B", nuisance_factor=3.)
    source = {"source_content_hash":"one"}
    baseline = RevisionConfig("B", nuisance_factor=1.)
    base_config = _configuration(baseline, 1.)
    base_path = tmp_path / run_identity(base_config, source)
    base_path.mkdir()
    sentinel = SimpleNamespace(config=baseline)
    loaded = []
    monkeypatch.setattr(run_revision, "_load_backend", lambda: object())
    monkeypatch.setattr(run_revision, "collect_provenance", lambda *args, **kwargs: source)
    monkeypatch.setattr(run_revision, "default_survey_group", lambda: object())
    monkeypatch.setattr(run_revision, "FiducialCosmology", lambda: object())
    monkeypatch.setattr(run_revision, "load_cached_revision_likelihood", lambda path, identity, provenance:
                        loaded.append((path, identity, provenance)) or sentinel)
    monkeypatch.setattr(run_revision, "build_revision_likelihood", lambda *args:
                        pytest.fail("likelihood recomputed despite exact cache"))
    monkeypatch.setattr(run_revision, "apply_revision_priors", lambda like, factor:
                        SimpleNamespace(likelihood=like, nuisance_factor=factor))
    monkeypatch.setattr(run_revision, "write_revision_artifacts", lambda *args:tmp_path / "sweep")
    assert run_revision.execute([config], tmp_path, None) == [tmp_path / "sweep"]
    assert loaded == [(base_path, likelihood_identity(base_config, source), source)]


def test_backend_is_selected_local_elf_and_float64():
    backend = run_revision._load_backend()
    assert backend.do_irres is False
    assert run_revision.backend_module_path().is_relative_to(Path("/Users/nguyenmn/elf/src/elf"))
    assert run_revision.jax.config.jax_enable_x64


def test_provenance_rehashes_local_sources_networks_survey_and_reference():
    backend = run_revision._load_backend()
    record = run_revision.collect_provenance(backend)
    source = run_revision.PROJECT_ROOT / "pfsfog/revision_forecast.py"
    assert record["source_file_hashes"][str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["emulator_input_hashes"]
    for path, digest in record["emulator_input_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert record["survey_input_hashes"]
    assert record["reference_cosmology"]["mnu"] == .06
    assert record["model_flags"]["fft_n"] == backend._nfft


def test_physical_construction_failure_writes_cell_and_k_without_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(run_revision, "_load_backend", lambda: object())
    monkeypatch.setattr(run_revision, "collect_provenance", lambda *args, **kwargs: {"source_content_hash":"source"})
    monkeypatch.setattr(run_revision, "default_survey_group", lambda: object())
    monkeypatch.setattr(run_revision, "FiducialCosmology", lambda: object())
    def broken(*args):
        raise ValueError("nonpositive field cell=(1.0, 1.1) k=0.1975 mu=0.0")
    monkeypatch.setattr(run_revision, "build_revision_likelihood", broken)
    with pytest.raises(ValueError, match="nonpositive field"):
        run_revision.execute([RevisionConfig("B")], tmp_path, None)
    (path,) = tuple(tmp_path.iterdir())
    manifest = json.loads((path / "manifest.json").read_text())
    details = json.loads((path / "validation.json").read_text())
    assert manifest["status"] == "failed"
    assert details["failure"]["cell_context"] == "(1.0, 1.1)"
    assert details["failure"]["k"] == .1975
    assert not (path / "summary.csv").exists()


def test_cli_finalize_uses_existing_artifact_without_backend(monkeypatch, tmp_path, capsys):
    path = tmp_path / "B-k020-p1-all-fake"
    bundle_file = tmp_path / "gates.json"
    gate = {"status":"pending", "measured":None, "tolerance":None,
            "evidence_path":None, "evidence_hash":None, "reason":"pending"}
    bundle_file.write_text(json.dumps({"runs":{path.name:{name:gate for name in MANDATORY_GATES
                                                       if name != "save_reload"}}}))
    calls = []
    monkeypatch.setattr(run_revision, "finalize_revision_artifacts", lambda p, b:
                        calls.append((p, b)) or p)
    monkeypatch.setattr(run_revision, "_load_backend", lambda: pytest.fail("backend initialized during finalize"))
    assert run_revision.main(["--finalize", str(path), "--gates", str(bundle_file)]) == 0
    assert calls[0][0] == path
    assert json.loads(capsys.readouterr().out)["scope"] == "complete"
