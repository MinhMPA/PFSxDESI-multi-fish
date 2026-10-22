"""Failure-mode tests for the run-bound validation orchestrator."""

from copy import deepcopy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.revision_artifacts import _configuration, run_identity
from pfsfog.revision_artifacts import MANDATORY_GATES
from pfsfog.revision_forecast import RevisionConfig
from scripts.run_revision import collect_provenance


class Backend:
    do_irres = False
    subtract_k0_limit = True
    _kmin, _kmax, _nfft = 1e-5, 1e3, 256


@pytest.mark.parametrize("ap_nk", [2048, 8192])
def test_physical_validator_import_cannot_rename_production_run(ap_nk):
    from scripts import run_revision
    from scripts import validate_revision
    import pfsfog.revision_validation_physical as physical
    before = collect_provenance(Backend(), requested_dk=.005, command="producer")
    assert str(physical.__file__) in before["source_file_hashes"]
    assert str(validate_revision.__file__) in before["source_file_hashes"]
    args = ["--case", "B", "--dk", ".0025", "--requested-dk", ".005",
            "--ap-nk", str(ap_nk),
            "--identities"]
    ids = []
    for script in ("run_revision.py", "validate_revision.py"):
        output = subprocess.run([sys.executable, str(run_revision.PROJECT_ROOT / "scripts" / script),
                                 *args], cwd=run_revision.PROJECT_ROOT,
                                check=True, capture_output=True, text=True)
        identity = json.loads(output.stdout)
        assert identity["configurations"][0]["ap_method"] == "elf-full-mean-cubic-logk-v1"
        assert "ap_refinement_nk" not in identity["configurations"][0]
        ids.append(identity["run_ids"][0])
    assert ids[0] == ids[1]


def _case(fisher):
    names = ["ln10_10_As", "Mnu", "Omegam"]
    snapshot = {"names":names, "fiducial":[3., .06, .31],
                "prior_sigma":[None, 5., 1.], "F_like":np.diag(fisher),
                "k":np.array([.05]), "dk":np.array([.01]),
                "reference_cosmology":{"Omega_m":.31},
                "cells":[{"cell_id":"c", "pairs":[["A", "A"]],
                          "geometry":{"volume":100., "nbar":{"A":.001}},
                          "names":names,
                          "derivatives":np.array([[[1., 0., 0.], [0., 0., 0.],
                                                    [0., 0., 0.]]]),
                          "covariance":np.eye(3)[None]}]}
    marginal = {"cov_cosmo_marg":np.diag(1/np.asarray(fisher)),
                "growth_z":np.arange(8.)/10+.5,
                "growth_jacobian":np.tile([1., .5, .2], (8, 1)),
                "growth_fiducial":np.arange(8.)/10+1.}
    return snapshot, marginal


def _bind_cases(cases, *, kmax=.20, nuisance_factor=1., exclude_elg1=False):
    from pfsfog.revision_validation_suite import physical_dependencies
    provenance = {"project_head":"head", "source_content_hash":"source",
                  "backend_snapshot":"backend", "network_hash":"network",
                  "survey_input_hash":"survey", "requested_dk":.005,
                  "model_flags":{"do_irres":False, "fft_n":256},
                  "source_file_hashes":{"source.py":"digest"},
                  "backend_file_hashes":{"backend.py":"digest"},
                  "executed_module_hashes":{"pfsfog.revision_forecast":{"sha256":"digest"}},
                  "emulator_input_hashes":{"network.npz":"digest"},
                  "survey_input_hashes":{"survey.txt":"digest"},
                  "dependency_hash":"dependency", "package_versions":{"numpy":"1"},
                  "interpreter":{"path":"python", "version":"3.10"},
                  "precision":"float64", "reference_cosmology":{"Omega_m":.31},
                  "thread_settings":{}, "process_settings":{"mode":"serial", "workers":1},
                  "backend_path":"backend.py"}
    for case, (likelihood, solve) in cases.items():
        configuration = _configuration(RevisionConfig(
            case, kmax=kmax, dk=.01, nuisance_factor=nuisance_factor,
            exclude_elg1=exclude_elg1), nuisance_factor,
                                       requested_dk=.005,
                                       model_flags=provenance["model_flags"])
        identity = run_identity(configuration, provenance)
        for record in (likelihood, solve):
            record.update({"configuration":deepcopy(configuration),
                           "physical_dependencies":physical_dependencies(provenance),
                           "run_id":identity})
    return cases


def test_suite_identity_rejects_missing_stale_and_swapped_solve():
    from pfsfog.revision_validation_suite import compare_suite_cases
    cases = {case:_case([1., 1., 1.]) for case in ("I", "A", "B", "D", "C")}
    assert compare_suite_cases(cases)["status"] == "failed"
    cases = _bind_cases(cases)
    assert compare_suite_cases(cases)["status"] == "passed"
    stale = deepcopy(cases)
    stale["B"][0]["physical_dependencies"]["source_content_hash"] = "stale"
    stale_report = compare_suite_cases(stale)
    assert stale_report["status"] == "failed"
    assert stale_report["checked_transitions"] == []
    changed = deepcopy(cases)
    changed["B"][0]["configuration"]["k_nl"] = .9
    assert compare_suite_cases(changed)["status"] == "failed"
    swapped = deepcopy(cases)
    swapped["B"] = (swapped["B"][0], swapped["C"][1])
    assert compare_suite_cases(swapped)["status"] == "failed"
    swapped_i = deepcopy(cases)
    swapped_i["I"] = (swapped_i["I"][0], swapped_i["A"][1])
    assert compare_suite_cases(swapped_i)["status"] == "failed"
    incomplete = deepcopy(cases)
    for likelihood, solve in incomplete.values():
        for record in (likelihood, solve):
            record["physical_dependencies"].pop("backend_file_hashes")
            record["run_id"] = run_identity(record["configuration"],
                                            record["physical_dependencies"])
    assert compare_suite_cases(incomplete)["status"] == "failed"


def test_suite_requires_matched_physics_and_orders_native_growth_errors():
    from pfsfog.revision_validation_suite import compare_suite_cases
    cases = _bind_cases({case:_case(fisher) for case, fisher in
             (("I", [5., 5., 5.]), ("A", [1., 1., 1.]), ("B", [1., 2., 1.]),
              ("D", [2., 2., 1.]), ("C", [2., 3., 2.]))})
    report = compare_suite_cases(cases)
    assert report["status"] == "passed"
    assert report["checked_cases"] == ["I", "A", "B", "D", "C"]
    assert report["checked_transitions"] == ["A->B", "B->D", "D->C", "B->C"]
    assert report["unsigned_comparisons"] == ["I->A"]
    broken = deepcopy(cases)
    broken["C"][0]["cells"][0]["geometry"]["volume"] = 101.
    assert compare_suite_cases(broken)["status"] == "failed"
    broken = deepcopy(cases)
    broken["C"][1]["growth_fiducial"][0] += .01
    assert compare_suite_cases(broken)["status"] == "failed"
    broken = deepcopy(cases)
    broken["C"][1]["cov_cosmo_marg"][0, 0] = 2.
    assert compare_suite_cases(broken)["status"] == "failed"


def test_suite_refuses_missing_case_and_mismatched_identity():
    from pfsfog.revision_validation_suite import compare_suite_cases
    cases = _bind_cases({case:_case([1., 1., 1.]) for case in ("I", "A", "B", "D", "C")})
    with pytest.raises(ValueError, match="I, A, B, D, C"):
        compare_suite_cases({key:cases[key] for key in ("I", "A", "B", "C")})
    with pytest.raises(ValueError, match="I, A, B, D, C"):
        compare_suite_cases({key:cases[key] for key in ("A", "B", "D", "C")})
    cases["B"][0]["reference_cosmology"]["Omega_m"] = .32
    assert compare_suite_cases(cases)["status"] == "failed"


@pytest.mark.parametrize("setting", [
    {"kmax":.15}, {"nuisance_factor":3.},
    {"nuisance_factor":10.}, {"exclude_elg1":True},
])
def test_nonbaseline_suite_checks_only_matched_b_to_c(setting):
    from pfsfog.revision_validation_suite import compare_suite_cases
    cases = _bind_cases({case:_case(fisher) for case, fisher in
                        (("B", [1., 1., 1.]), ("C", [2., 2., 2.]))}, **setting)
    report = compare_suite_cases(cases, **setting)
    assert report["status"] == "passed"
    assert report["checked_cases"] == ["B", "C"]
    assert report["checked_transitions"] == ["B->C"]
    assert report["unsigned_comparisons"] == []
    assert list(report["edges"]) == ["B->C"]
    swapped = deepcopy(cases)
    swapped["B"] = (swapped["B"][0], swapped["C"][1])
    assert compare_suite_cases(swapped, **setting)["status"] == "failed"
    stale = deepcopy(cases)
    stale["C"][0]["physical_dependencies"]["network_hash"] = "stale"
    assert compare_suite_cases(stale, **setting)["status"] == "failed"
    mismatched = deepcopy(cases)
    mismatched["C"][0]["configuration"]["nuisance_factor"] = 7.
    assert compare_suite_cases(mismatched, **setting)["status"] == "failed"
    changed_prior = deepcopy(cases)
    changed_prior["C"][0]["prior_sigma"][1] *= 2
    assert compare_suite_cases(changed_prior, **setting)["status"] == "failed"
    changed_jacobian = deepcopy(cases)
    changed_jacobian["C"][1]["growth_jacobian"][0, 0] += .01
    assert compare_suite_cases(changed_jacobian, **setting)["status"] == "failed"


def test_suite_declares_expected_group_without_trusting_results():
    from pfsfog.revision_validation_suite import compare_suite_cases
    cases = _bind_cases({case:_case([1., 1., 1.]) for case in ("B", "C")})
    with pytest.raises(ValueError, match="I, A, B, D, C"):
        compare_suite_cases(cases)
    with pytest.raises(ValueError, match="unsupported suite setting"):
        compare_suite_cases(cases, kmax=.18)
    varied = _bind_cases({case:_case([1., 1., 1.]) for case in ("B", "C")},
                         kmax=.15)
    assert compare_suite_cases(varied, kmax=.15)["status"] == "passed"
    assert compare_suite_cases(varied, nuisance_factor=3.)["status"] == "failed"


def test_prior_gate_rejects_duplicate_prior_and_false_q13():
    from pfsfog.revision_validation_suite import validate_prior_result
    params = [{"name":"ln10_10_As", "role":"cosmo", "prior_sigma":None},
              {"name":"Mnu", "role":"cosmo", "prior_sigma":5.},
              {"name":"Omegam", "role":"cosmo", "prior_sigma":1.},
              {"name":"bias", "role":"tracer", "prior_sigma":2.}]
    f_like = np.diag([2., 3., 4., 5.])
    f_cosmo = np.diag([0., 1/25, 1., 0.])
    f_nuisance = np.diag([0., 0., 0., 1/4])
    total = f_like + f_cosmo + f_nuisance
    covariance = np.linalg.inv(total)
    result = SimpleNamespace(names=[p["name"] for p in params], parameters=params,
                             F_like=f_like, F_prior_cosmo=f_cosmo,
                             F_prior_nuisance=f_nuisance, F_total=total,
                             covariance=covariance,
                             cov_cosmo_marg=covariance[:3, :3],
                             cov_cosmo_nuisance_fixed=np.linalg.inv(total[:3, :3]))
    assert validate_prior_result(result)["status"] == "passed"
    bad = deepcopy(result)
    bad.F_prior_nuisance[3, 3] = 1/2
    assert validate_prior_result(bad)["status"] == "failed"
    bad = deepcopy(result)
    bad.cov_cosmo_nuisance_fixed[0, 0] *= 2
    assert validate_prior_result(bad)["status"] == "failed"
    bad = deepcopy(result)
    bad.validation = {"global_solve":{"scaled_inverse_residual":1e-5},
                      "cosmology_block_solve":{"scaled_inverse_residual":0.}}
    assert validate_prior_result(bad)["status"] == "failed"
    bad = deepcopy(result)
    bad.validation = {"global_solve":{"scaled_inverse_residual":0.},
                      "cosmology_block_solve":{"scaled_inverse_residual":0.}}
    bad.covariance[3, 3] *= 2
    assert validate_prior_result(bad)["status"] == "failed"
    bad = deepcopy(result)
    bad.validation = {"global_solve":{"scaled_inverse_residual":0.},
                      "cosmology_block_solve":{"scaled_inverse_residual":0.}}
    bad.covariance[3, 0] = .1
    assert validate_prior_result(bad)["status"] == "failed"


def test_journal_persists_failure_and_keeps_representative_fd_pending(tmp_path):
    import json
    from pfsfog.revision_validation_suite import EvidenceJournal
    from scripts.run_revision import load_gates

    journal = EvidenceJournal(tmp_path, "run-x", {"case":"B"}, {"source":"abc"})
    journal.event("build", "started", {"cell_id":"cell-1"})
    journal.set_gate("ad_fd", "passed", {"status":"passed",
                     "scope":"representative", "complete_run_coverage":False,
                     "selected_cell_ids":["cell-1"]}, 1e-3, "subset measured")
    journal.failure("covariance", ValueError("bad matrix"),
                    {"cell_id":"cell-1", "k":.02125, "mu":.2,
                     "column":"Mnu", "sign":"+", "step":.005,
                     "eigenvalue_unavailable_reason":"eigensolver failed"})
    evidence = json.loads(journal.path.read_text())
    assert evidence["started_unix_seconds"] > 0
    assert evidence["events"][0]["elapsed_seconds"] >= 0
    assert evidence["gates"]["ad_fd"]["status"] == "pending"
    assert evidence["events"][-1]["context"]["k"] == .02125
    assert evidence["events"][-1]["context"]["eigenvalue_unavailable_reason"]
    bundle_path = journal.write_bundle()
    bundle = load_gates(bundle_path)
    assert set(bundle["runs"]["run-x"]) == set(MANDATORY_GATES) - {"save_reload"}
    assert all(record["status"] != "passed" for record in bundle["runs"]["run-x"].values())


def test_validation_cli_identity_matches_producer_fresh_and_warm(capsys):
    import json
    from scripts import run_revision, validate_revision
    options = ["--case", "B", "--kmax", ".2", "--dk", ".0025",
               "--requested-dk", ".005", "--identities"]
    assert run_revision.main(options) == 0
    producer = json.loads(capsys.readouterr().out)["run_ids"][0]
    assert validate_revision.main(options) == 0
    gate = json.loads(capsys.readouterr().out)["run_ids"][0]
    assert gate == producer
    assert validate_revision.main(options) == 0
    assert json.loads(capsys.readouterr().out)["run_ids"][0] == producer


def test_resolution_requires_all_eleven_errors_and_cb_ratios():
    from pfsfog.revision_validation_suite import compare_resolution_errors
    base = np.linspace(.1, 1.1, 11)
    values = {f"{case}_{tag}":base * (1.05 if case == "C" else 1.) *
              (1.002 if tag == "angular" else 1.001 if tag == "shell" else 1.)
              for case in ("B", "C") for tag in ("base", "angular", "shell")}
    report = compare_resolution_errors(values)
    assert report["status"] == "passed"
    assert len(report["comparisons"]) == 6
    assert all(len(row["relative_change"]) == 11 for row in report["comparisons"])
    bad = deepcopy(values)
    bad["C_angular"][8] *= 1.02
    assert compare_resolution_errors(bad)["status"] == "failed"
    with pytest.raises(ValueError, match="eleven"):
        compare_resolution_errors({**values, "B_base":base[:4]})


def test_resolution_reuse_rejects_stale_physical_dependencies():
    from pfsfog.revision_validation_suite import validate_resolution_evidence
    provenance = {"project_head":"h", "source_content_hash":"s",
                  "backend_snapshot":"b", "network_hash":"n",
                  "survey_input_hash":"u", "dependency_hash":"d",
                  "model_flags":{"do_irres":False}}
    config = RevisionConfig("B", dk=.0025)
    values = {f"{case}_{tag}":[1.]*11 for case in ("B", "C")
              for tag in ("base", "angular", "shell")}
    report = {"status":"passed", "scope":"representative_BC_full_redshift",
              "physical_dependencies":provenance.copy(),
              "configuration":{"kmin":.01, "kmax":.20, "dk":.0025,
                               "n_mu":32, "nuisance_factor":1.,
                               "exclude_elg1":False,
                               "nominal_bins":[list(x) for x in config.nominal_bins]},
              "values":values}
    with pytest.raises(ValueError, match="not attachable"):
        validate_resolution_evidence(report, config, provenance)
    report["changed_shell_domain_ap"] = {"status":"passed",
        "scope":"all changed .00125 shell points and support cells",
        "expected_shells":608, "processed_shells":608,
        "shape_checked":True, "domain_checked":True, "ap_identity_checked":True}
    with pytest.raises(ValueError, match="not attachable"):
        validate_resolution_evidence(report, config, provenance)
    changed = {**provenance, "network_hash":"different"}
    with pytest.raises(ValueError, match="physical dependencies"):
        validate_resolution_evidence(report, config, changed)
    bad = deepcopy(report)
    bad["values"]["C_shell"][5] = 1.02
    from pfsfog.revision_validation_suite import compare_resolution_errors
    assert compare_resolution_errors(bad["values"])["status"] == "failed"


@pytest.mark.parametrize("companion", [
    {"status":"passed", "scope":"all changed .00125 shell points and support cells",
     "expected_shells":1, "processed_shells":1,
     "shape_checked":True, "domain_checked":True, "ap_identity_checked":True},
    {"status":"passed", "scope":"all changed .00125 shell points and support cells",
     "expected_shells":3040, "processed_shells":3040,
     "shape_checked":True, "domain_checked":True, "ap_identity_checked":True,
     "cells":["one-of-twenty"]},
    {"status":"passed", "scope":"all changed .00125 shell points and support cells",
     "expected_shells":3040, "processed_shells":3040,
     "shape_checked":True, "domain_checked":True, "ap_identity_checked":True,
     "runs":[{"key":"B_shell"}], "ap_chain":None},
])
def test_resolution_attachment_rejects_self_declared_or_missing_companion(companion):
    from pfsfog.revision_validation_suite import validate_resolution_evidence
    config = RevisionConfig("B", dk=.0025)
    provenance = {"project_head":"head", "source_content_hash":"source"}
    report = {"status":"passed", "scope":"representative_BC_full_redshift",
              "configuration":{"kmin":.01, "kmax":.20, "dk":.0025,
                               "n_mu":32, "nuisance_factor":1.,
                               "exclude_elg1":False,
                               "nominal_bins":[list(x) for x in config.nominal_bins]},
              "physical_dependencies":provenance,
              "values":{f"{case}_{tag}":[1.]*11 for case in ("B", "C")
                        for tag in ("base", "angular", "shell")},
              "changed_shell_domain_ap":companion}
    with pytest.raises(ValueError, match="companion.*not attachable"):
        validate_resolution_evidence(report, config, provenance)


def test_resolution_attachment_requires_live_changed_shell_likelihoods():
    from pfsfog.revision_validation_suite import validate_resolution_evidence
    config = RevisionConfig("B", dk=.0025)
    provenance = {"project_head":"head", "source_content_hash":"source"}
    report = {"status":"passed", "scope":"representative_BC_full_redshift",
              "configuration":{"kmin":.01, "kmax":.20, "dk":.0025,
                               "n_mu":32, "nuisance_factor":1.,
                               "exclude_elg1":False,
                               "nominal_bins":[list(x) for x in config.nominal_bins]},
              "physical_dependencies":provenance,
              "changed_shell_domain_ap":{"status":"passed", "reports":{"B":{}, "C":{}}}}
    with pytest.raises(ValueError, match="live changed-shell likelihoods"):
        validate_resolution_evidence(report, config, provenance)


def test_changed_shell_rebuild_keeps_candidate_ap_table(monkeypatch):
    from scripts import validate_revision
    config = RevisionConfig("B", dk=.0025, ap_interpolation_nk=8192,
                            ap_refinement_nk=(16384, 32768))
    calls = []
    def build(variant, survey, cosmo, backend):
        calls.append(variant)
        return SimpleNamespace(config=variant)
    monkeypatch.setattr(validate_revision, "build_revision_likelihood", build)
    lives = validate_revision._changed_shell_likelihoods(
        config, object(), object(), object())
    assert set(lives) == {"B", "C"}
    assert [row.case for row in calls] == ["B", "C"]
    assert all(row.dk == .00125 and row.n_mu == 32 for row in calls)
    assert all(row.ap_interpolation_nk == 8192 and
               row.ap_refinement_nk == (16384, 32768) for row in calls)


def test_smoke_bin_is_explicitly_partial_and_cannot_be_suite():
    from scripts.validate_revision import _arguments, _configs
    config = _configs(_arguments(["--case", "B", "--smoke-bin"]))[0]
    assert config.nominal_bins == ((1., 1.2),)
    assert _arguments(["--cache-root", "/tmp/exact-cache"]).cache_root.as_posix() == "/tmp/exact-cache"
    with pytest.raises(ValueError, match="suite"):
        _configs(_arguments(["--suite", "--smoke-bin"]))


def test_suite_cli_schedules_approved_cohorts_only():
    from scripts.validate_revision import _arguments, _configs
    settings = [([], ["I", "A", "B", "D", "C"]),
                (["--kmax", ".15"], ["B", "C"]),
                (["--nuisance-factor", "3"], ["B", "C"]),
                (["--nuisance-factor", "10"], ["B", "C"]),
                (["--exclude-elg1"], ["B", "C"])]
    for options, expected in settings:
        assert [config.case for config in _configs(_arguments(["--suite", *options]))] == expected
    with pytest.raises(ValueError, match="unsupported suite setting"):
        _configs(_arguments(["--suite", "--kmax", ".18"]))


def test_producer_and_validator_identity_cli_cover_exact_retained_cohort():
    from scripts import run_revision
    prefix = [sys.executable]
    root = run_revision.PROJECT_ROOT
    driver = subprocess.run([*prefix, str(root / "scripts/run_revision.py"),
                             "--grid", "--dk", ".0025", "--requested-dk", ".005",
                             "--identities"], cwd=root, check=True,
                            capture_output=True, text=True)
    plan = json.loads(driver.stdout)
    assert len(plan["run_ids"]) == 13
    assert len(set(plan["run_ids"])) == 13
    assert len(set(plan["likelihood_identities"])) == 9
    expected = dict(zip(plan["run_ids"], plan["configurations"]))
    scheduled = {}
    for options in ([], ["--kmax", ".15"],
                    ["--nuisance-factor", "3"], ["--nuisance-factor", "10"],
                    ["--exclude-elg1"]):
        output = subprocess.run([*prefix, str(root / "scripts/validate_revision.py"),
                                 "--suite", *options, "--dk", ".0025",
                                 "--requested-dk", ".005", "--identities"],
                                cwd=root, check=True, capture_output=True, text=True)
        cohort = json.loads(output.stdout)
        assert cohort["scope"] == "suite"
        scheduled.update(zip(cohort["run_ids"], cohort["configurations"]))
    assert scheduled == expected


@pytest.mark.parametrize("options, expected_cases, expected_edges", [
    ([], ["I", "A", "B", "D", "C"], ["A->B", "B->D", "D->C", "B->C"]),
    (["--kmax", ".15"], ["B", "C"], ["B->C"]),
])
def test_suite_cli_binds_applicable_evidence_to_each_run(
        tmp_path, monkeypatch, capsys, options, expected_cases, expected_edges):
    from scripts import validate_revision
    monkeypatch.setattr(validate_revision.producer, "_load_backend", lambda:Backend())
    fishers = {"I":[5., 5., 5.], "A":[1., 1., 1.], "B":[1., 2., 1.],
               "D":[2., 2., 1.], "C":[2., 3., 2.]}

    def measure(config, journal, *args):
        snapshot, solve = _case(fishers[config.case])
        result = SimpleNamespace(cov_cosmo_marg=solve["cov_cosmo_marg"],
                                 growth_z=solve["growth_z"],
                                 growth_jacobian=solve["growth_jacobian"],
                                 growth_fiducial=solve["growth_fiducial"])
        return result, snapshot, [], []

    monkeypatch.setattr(validate_revision, "_measure", measure)
    assert validate_revision.main(["--suite", *options, "--evidence-root",
                                   str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [run_id.split("-")[0] for run_id in output["run_ids"]] == expected_cases
    for run_id in output["run_ids"]:
        document = json.loads((tmp_path / f"{run_id}.validation-evidence.json").read_text())
        gate = document["gates"]["suite_monotonicity"]
        assert gate["status"] == "passed"
        assert gate["measured"]["checked_cases"] == expected_cases
        assert gate["measured"]["checked_transitions"] == expected_edges
        assert set(gate["measured"]["edges"]) == set(expected_edges)
        assert gate["measured"]["unsigned_comparisons"] == (["I->A"] if not options else [])
        bundle = validate_revision.producer.load_gates(tmp_path / f"{run_id}.gates.json")
        assert bundle["runs"][run_id]["suite_monotonicity"]["status"] == "passed"


def test_cli_ap_attachment_passes_assembled_live_likelihood(tmp_path, monkeypatch):
    from scripts import validate_revision
    monkeypatch.setattr(validate_revision.producer, "_load_backend", lambda:Backend())
    live = object()
    snapshot, solve = _case([1., 1., 1.])

    def measure(*args):
        result = SimpleNamespace(likelihood=live, cov_cosmo_marg=solve["cov_cosmo_marg"],
                                 growth_z=solve["growth_z"],
                                 growth_jacobian=solve["growth_jacobian"],
                                 growth_fiducial=solve["growth_fiducial"])
        return result, snapshot, [], []

    observed = []
    def bind(report, config, provenance, survey, run_id, *, likelihood):
        observed.append(likelihood)
        return {"status":"passed"}

    monkeypatch.setattr(validate_revision, "_measure", measure)
    monkeypatch.setattr(validate_revision, "validate_ap_evidence", bind)
    evidence = tmp_path / "ap.json"
    evidence.write_text(json.dumps({"scope":"all_cells"}))
    assert validate_revision.main(["--case", "B", "--ap-evidence", str(evidence),
                                   "--evidence-root", str(tmp_path / "out")]) == 0
    assert observed == [live]


def _ap_manifest(tmp_path, run_ids):
    reports = []
    for run_id in run_ids:
        path = tmp_path / f"{run_id}.ap.json"
        path.write_text(json.dumps({"scope":"all_cells", "run_id":run_id}))
        reports.append({"run_id":run_id, "path":path.name,
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    path = tmp_path / "ap-manifest.json"
    path.write_text(json.dumps({"schema_version":1, "reports":reports}))
    return path, reports


@pytest.mark.parametrize("setting", [[], ["--kmax", ".15"]])
def test_cli_ap_manifest_routes_each_report_to_its_live_suite_member(
        tmp_path, monkeypatch, setting):
    from scripts import validate_revision as cli
    monkeypatch.setattr(cli.producer, "_load_backend", lambda:Backend())
    provenance = {"requested_dk":.005, "model_flags":{"fft_n":256},
                  "source_content_hash":"source"}
    monkeypatch.setattr(cli.producer, "collect_provenance", lambda *a, **k:provenance)
    monkeypatch.setattr(cli, "default_survey_group", lambda:object())
    monkeypatch.setattr(cli, "FiducialCosmology", lambda:object())
    args = cli._arguments(["--suite", *setting])
    run_ids = [cli._identity(c, provenance)[1] for c in cli._configs(args)]
    manifest, reports = _ap_manifest(tmp_path, run_ids)
    seen = []
    def measure(config, journal, *args):
        snapshot, solve = _case({"I":[5.,5.,5.], "A":[1.,1.,1.],
                                 "B":[1.,2.,1.], "D":[2.,2.,1.],
                                 "C":[2.,3.,2.]}[config.case])
        result = SimpleNamespace(likelihood=config.case,
                                 cov_cosmo_marg=solve["cov_cosmo_marg"],
                                 growth_z=solve["growth_z"],
                                 growth_jacobian=solve["growth_jacobian"],
                                 growth_fiducial=solve["growth_fiducial"])
        return result, snapshot, [], []
    def bind(report, config, provenance, survey, run_id, *, likelihood):
        seen.append((report["run_id"], config.case, run_id, likelihood))
        assert report["run_id"] == run_id
        assert likelihood == config.case
        return {"status":"passed"}
    monkeypatch.setattr(cli, "_measure", measure)
    monkeypatch.setattr(cli, "validate_ap_evidence", bind)
    assert cli.main(["--suite", *setting, "--ap-evidence-manifest", str(manifest),
                     "--evidence-root", str(tmp_path / "out")]) == 0
    assert [row[0] for row in seen] == run_ids
    for entry in reports:
        document = json.loads((tmp_path / "out" / f"{entry['run_id']}.validation-evidence.json").read_text())
        for gate in ("domain", "ap_identity_chain_resolution"):
            source = document["gates"][gate]["measured"]
            assert source["source_path"] == str((tmp_path / entry["path"]).resolve())
            assert source["source_sha256"] == entry["sha256"]
            assert document["gates"][gate]["status"] == "passed"


@pytest.mark.parametrize("change, match", [
    (lambda rows: rows.pop(), "missing"),
    (lambda rows: rows.append({**rows[0], "run_id":"extra"}), "extra"),
    (lambda rows: rows.append(rows[0].copy()), "duplicate"),
    (lambda rows: rows[0].update(sha256="0"*64), "SHA-256"),
    (lambda rows: rows[0].update(path=""), "path"),
])
def test_ap_manifest_rejects_invalid_members_before_measurement(tmp_path, monkeypatch,
                                                                change, match):
    from scripts import validate_revision as cli
    monkeypatch.setattr(cli.producer, "_load_backend", lambda:Backend())
    monkeypatch.setattr(cli.producer, "collect_provenance", lambda *a, **k:
                        {"requested_dk":.005, "model_flags":{"fft_n":256}})
    args = cli._arguments(["--suite", "--kmax", ".15"])
    provenance = {"requested_dk":.005, "model_flags":{"fft_n":256}}
    ids = [cli._identity(c, provenance)[1] for c in cli._configs(args)]
    manifest, rows = _ap_manifest(tmp_path, ids)
    change(rows)
    manifest.write_text(json.dumps({"schema_version":1, "reports":rows}))
    monkeypatch.setattr(cli, "_measure", lambda *a:pytest.fail("measurement reached"))
    with pytest.raises(ValueError, match=match):
        cli.main(["--suite", "--kmax", ".15", "--ap-evidence-manifest", str(manifest),
                  "--evidence-root", str(tmp_path / "out")])


def test_ap_manifest_rejects_swapped_report_identity_and_bad_schema(tmp_path):
    from scripts.validate_revision import _load_ap_manifest
    manifest, rows = _ap_manifest(tmp_path, ["B-run", "C-run"])
    source = tmp_path / rows[0]["path"]
    source.write_text(json.dumps({"scope":"all_cells", "run_id":"C-run"}))
    rows[0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema_version":1, "reports":rows}))
    with pytest.raises(ValueError, match="run ID"):
        _load_ap_manifest(manifest, ["B-run", "C-run"])
    manifest.write_text(json.dumps({"schema_version":2, "reports":rows}))
    with pytest.raises(ValueError, match="schema"):
        _load_ap_manifest(manifest, ["B-run", "C-run"])


def test_ap_manifest_preflight_keeps_only_sources_and_rechecks_at_attachment(tmp_path):
    from scripts.validate_revision import _load_ap_manifest, _read_ap_report
    manifest, rows = _ap_manifest(tmp_path, ["B-run", "C-run"])
    sources = _load_ap_manifest(manifest, ["B-run", "C-run"])
    assert sources["B-run"] == ((tmp_path / rows[0]["path"]).resolve(),
                                 rows[0]["sha256"])
    path, digest = sources["C-run"]
    assert _read_ap_report(path, digest, "C-run")["run_id"] == "C-run"
    path.write_text(json.dumps({"scope":"all_cells", "run_id":"C-run", "changed":True}))
    with pytest.raises(ValueError, match="SHA-256"):
        _read_ap_report(path, digest, "C-run")


def test_ap_manifest_delegates_rejection_to_existing_live_validator(tmp_path, monkeypatch):
    from scripts import validate_revision as cli
    provenance = {"requested_dk":.005, "model_flags":{"fft_n":256}}
    monkeypatch.setattr(cli.producer, "_load_backend", lambda:Backend())
    monkeypatch.setattr(cli.producer, "collect_provenance", lambda *a, **k:provenance)
    run_id = cli._identity(cli._configs(cli._arguments(["--case", "B"]))[0],
                           provenance)[1]
    manifest, _ = _ap_manifest(tmp_path, [run_id])
    snapshot, solve = _case([1.,2.,1.])
    def measure(*args):
        return SimpleNamespace(likelihood=object(),
                               cov_cosmo_marg=solve["cov_cosmo_marg"],
                               growth_z=solve["growth_z"],
                               growth_jacobian=solve["growth_jacobian"],
                               growth_fiducial=solve["growth_fiducial"]), snapshot, [], []
    monkeypatch.setattr(cli, "_measure", measure)
    with pytest.raises(ValueError, match="AP evidence is incomplete"):
        cli.main(["--case", "B", "--ap-evidence-manifest", str(manifest),
                  "--evidence-root", str(tmp_path / "out")])
    document = json.loads((tmp_path / "out" / f"{run_id}.validation-evidence.json").read_text())
    assert document["events"][-1]["status"] == "failed"
    assert document["gates"]["ap_identity_chain_resolution"]["status"] != "passed"


def test_ap_manifest_flag_rules_and_single_file_suite_rejection(tmp_path, monkeypatch):
    from scripts import validate_revision as cli
    manifest, _ = _ap_manifest(tmp_path, ["B-run"])
    for flags, match in [
        (["--ap-evidence", str(tmp_path / "ap.json"),
          "--ap-evidence-manifest", str(manifest)], "mutually exclusive"),
        (["--suite", "--ap-evidence", str(tmp_path / "ap.json")], "manifest"),
        (["--ap-probe", "--ap-evidence-manifest", str(manifest)], "AP probe"),
        (["--resolution-probe", "--ap-evidence-manifest", str(manifest)], "resolution probe"),
        (["--resolution-complete", str(tmp_path / "base.json"),
          "--ap-evidence-manifest", str(manifest)], "resolution completion"),
    ]:
        monkeypatch.setattr(cli.producer, "_load_backend", lambda:Backend())
        monkeypatch.setattr(cli.producer, "collect_provenance", lambda *a, **k:
                            {"requested_dk":.005, "model_flags":{"fft_n":256}})
        with pytest.raises(ValueError, match=match):
            cli.main([*flags, "--evidence-root", str(tmp_path / "out")])


def test_partial_smoke_demotes_every_outer_gate(tmp_path):
    from pfsfog.revision_validation_suite import EvidenceJournal
    journal = EvidenceJournal(tmp_path, "smoke", {"case":"B"}, {"source":"x"})
    journal.set_gate("domain", "passed", {"checked":True}, None, "subset domain")
    journal.mark_partial("one nominal bin")
    assert all(row["status"] == "pending" for row in journal.document["gates"].values())


def test_cli_failed_smoke_cannot_persist_early_passed_outer_gate(tmp_path, monkeypatch):
    from scripts import validate_revision
    monkeypatch.setattr(validate_revision.producer, "_load_backend", lambda:Backend())
    monkeypatch.setattr(validate_revision.producer, "collect_provenance",
                        lambda *args, **kwargs:{"requested_dk":.005,
                                                "model_flags":{"fft_n":256},
                                                "source_content_hash":"source"})
    monkeypatch.setattr(validate_revision, "default_survey_group", lambda:object())
    monkeypatch.setattr(validate_revision, "FiducialCosmology", lambda:object())

    def fail_after_early_pass(config, journal, *args):
        journal.set_gate("domain", "passed", {"checked":True}, None,
                         "representative support inspected")
        raise RuntimeError("failed after first gate")

    monkeypatch.setattr(validate_revision, "_measure", fail_after_early_pass)
    with pytest.raises(RuntimeError, match="failed after first gate"):
        validate_revision.main(["--case", "B", "--smoke-bin", "--max-cells", "1",
                                "--evidence-root", str(tmp_path)])
    document = json.loads(next(tmp_path.glob("*.validation-evidence.json")).read_text())
    assert document["partial_scope_reason"]
    assert document["gates"]["domain"]["status"] == "pending"
    assert document["events"][-1]["status"] == "failed"
    bundle = json.loads(next(tmp_path.glob("*.gates.json")).read_text())
    assert all(row["status"] != "passed" for row in
               next(iter(bundle["runs"].values())).values())


def test_physical_cache_counts_exact_warm_reuse(tmp_path):
    from pfsfog.revision_validation_physical import PhysicalFDCache
    cache = PhysicalFDCache(tmp_path)
    value = cache.get_or_compute({"source":"s", "point":1},
                                 lambda:np.ones((2, 3)))
    assert value.shape == (2, 3)
    cache.get_or_compute({"source":"s", "point":1},
                         lambda:pytest.fail("warm cache recomputed"))
    assert cache.misses == 1 and cache.hits == 1
