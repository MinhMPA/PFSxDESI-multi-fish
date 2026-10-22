#!/usr/bin/env python
"""Measure bounded, run-bound numerical revision evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from scripts import run_revision as producer

from pfsfog.cosmo import FiducialCosmology
from pfsfog.revision_artifacts import _configuration, _strict_json, run_identity
from pfsfog.revision_forecast import (NOMINAL_BINS, RevisionConfig,
                                      apply_revision_priors, build_revision_likelihood)
from pfsfog.revision_validation_physical import (
    PhysicalFDCache, assembled_fd_certificate, audit_cell_covariance_neighborhood,
    declare_fd_manifest, expected_support_schedule, likelihood_snapshot,
    validate_support_schedule)
from pfsfog.revision_validation_ap import (
    REFINEMENT_TOLERANCE, run_ap_probe, validate_ap_evidence)
from pfsfog.revision_validation_suite import (
    EvidenceJournal, compare_resolution_errors, compare_suite_cases,
    physical_dependencies, suite_order, validate_prior_result,
    validate_resolution_evidence)
from pfsfog.surveys import default_survey_group


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("I", "A", "B", "D", "C"), default="B")
    parser.add_argument("--suite", action="store_true",
                        help="Baseline I/A/B/D/C or approved nonbaseline B/C at one setting")
    parser.add_argument("--smoke-bin", action="store_true",
                        help="Measure only the representative z=1.1 nominal bin; never complete")
    parser.add_argument("--kmin", type=float, default=.01)
    parser.add_argument("--kmax", type=float, default=.20)
    parser.add_argument("--dk", type=float, default=.0025)
    parser.add_argument("--requested-dk", type=float, default=.005)
    parser.add_argument("--n-mu", type=int, default=32)
    parser.add_argument("--ap-nk", type=int, choices=(2048, 8192), default=2048)
    parser.add_argument("--nuisance-factor", type=float, default=1.)
    parser.add_argument("--exclude-elg1", action="store_true")
    parser.add_argument("--max-cells", type=int, default=1,
                        help="Independent support cells per case; 0 requests all cells")
    parser.add_argument("--identities", action="store_true")
    parser.add_argument("--resolution-probe", action="store_true",
                        help="Measure full-redshift B/C .0025 angular and shell refinements")
    parser.add_argument("--ap-probe", action="store_true",
                        help="Measure selected AP/domain support cells with incremental evidence")
    parser.add_argument("--ap-evidence", type=Path,
                        help="Attach complete exact-run quantitative adopted-grid AP/domain evidence")
    parser.add_argument("--ap-evidence-manifest", type=Path,
                        help="Attach one SHA-256-bound AP report per requested run ID")
    parser.add_argument("--resolution-evidence", type=Path,
                        help="Attach exact-dependency B/C resolution evidence")
    parser.add_argument("--resolution-complete", type=Path,
                        help="Bind saved six-build evidence to both changed-shell AP reports")
    parser.add_argument("--changed-shell-ap-B", dest="changed_shell_ap_b", type=Path)
    parser.add_argument("--changed-shell-ap-C", dest="changed_shell_ap_c", type=Path)
    parser.add_argument("--evidence-root", type=Path,
                        default=producer.PROJECT_ROOT / "results/revision-validation")
    parser.add_argument("--cache-root", type=Path,
                        help="Exact physical FD cache shared across serial validation runs")
    return parser.parse_args(argv)


def _configs(args):
    if args.smoke_bin and args.suite:
        raise ValueError("--smoke-bin cannot be a complete suite")
    cases = (suite_order(kmax=args.kmax, nuisance_factor=args.nuisance_factor,
                         exclude_elg1=args.exclude_elg1)
             if args.suite else (args.case,))
    return [RevisionConfig(case, kmin=args.kmin, kmax=args.kmax,
                           dk=args.dk, n_mu=args.n_mu,
                           nuisance_factor=args.nuisance_factor,
                           exclude_elg1=args.exclude_elg1,
                           ap_interpolation_nk=args.ap_nk,
                           ap_refinement_nk=(2*args.ap_nk, 4*args.ap_nk),
                           nominal_bins=((1., 1.2),) if args.smoke_bin else
                                        NOMINAL_BINS)
            for case in cases]


def _identity(config, provenance):
    configuration = _configuration(
        config, config.nuisance_factor,
        requested_dk=provenance["requested_dk"],
        model_flags=provenance["model_flags"])
    return configuration, run_identity(configuration, provenance)


def _read_ap_report(path, digest, run_id):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"AP evidence SHA-256 mismatch for {run_id}")
    report = _strict_json(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("run_id") != run_id:
        raise ValueError(f"AP evidence report run ID mismatch for {run_id}")
    return report


def _load_ap_manifest(path, run_ids):
    """Verify the inventory, retaining only immutable source coordinates."""
    manifest = _strict_json(path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "reports"}
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or not isinstance(manifest["reports"], list)):
        raise ValueError("AP evidence manifest schema must be version 1 with a reports list")
    entries = {}
    for entry in manifest["reports"]:
        if (not isinstance(entry, dict) or set(entry) != {"run_id", "path", "sha256"}
                or not isinstance(entry["run_id"], str) or not entry["run_id"]
                or not isinstance(entry["path"], str) or not entry["path"]
                or not isinstance(entry["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
            raise ValueError("AP evidence manifest entry needs run_id, path and SHA-256")
        run_id = entry["run_id"]
        if run_id in entries:
            raise ValueError(f"duplicate AP evidence manifest run ID: {run_id}")
        entries[run_id] = entry
    requested = set(run_ids)
    missing, extra = requested - entries.keys(), entries.keys() - requested
    if missing or extra:
        raise ValueError(f"AP evidence manifest membership: missing {sorted(missing)}, extra {sorted(extra)}")
    sources = {}
    for run_id, entry in entries.items():
        report_path = (path.parent / entry["path"]).resolve()
        _read_ap_report(report_path, entry["sha256"], run_id)
        sources[run_id] = (report_path, entry["sha256"])
    return sources


def _combine_fd(certificates, selected_ids, expected_support):
    """Combine disjoint per-cell certificates after independent declaration."""
    all_ids = [row["cell_id"] for row in expected_support]
    first = certificates[0]
    failures = [failure for c in certificates for failure in c["failures"]]
    complete = set(selected_ids) == set(all_ids)
    return {"status":"failed" if failures else "passed",
            "scope":"all_cells" if complete else "representative",
            "complete_run_coverage":complete,
            "selected_cell_ids":list(selected_ids), "all_cell_ids":all_ids,
            "expected_support_observations":first["expected_support_observations"],
            "expected_coverage":[row for c in certificates for row in c["expected_coverage"]],
            "measured_coverage":[row for c in certificates for row in c["measured_coverage"]],
            "column_results":[row for c in certificates for row in c["column_results"]],
            "errors":[row for c in certificates for row in c["errors"]],
            "physical_scales":{k:v for c in certificates for k,v in c["physical_scales"].items()},
            "relative_tolerance":first["relative_tolerance"],
            "zero_tolerance":first["zero_tolerance"],
            "max_error":max(c["max_error"] for c in certificates),
            "failures":failures}


def _measure(config, journal, survey, cosmo, backend, cache, provenance, max_cells):
    support = expected_support_schedule(config, survey)
    selected = [row["cell_id"] for row in support]
    if max_cells:
        selected = selected[:max_cells]
    journal.event("assemble", "started", {"case":config.case, "support_cells":len(support)})
    likelihood = build_revision_likelihood(config, survey, cosmo, backend)
    result = apply_revision_priors(likelihood, config.nuisance_factor)
    validate_support_schedule(likelihood, support)
    journal.event("assemble", "passed", {"case":config.case,
                  "processed_shells":likelihood.validation["processed_shells"],
                  "expected_shells":likelihood.validation["expected_shells"],
                  "active_parameters":len(likelihood.names)})
    journal.set_gate("domain", "pending", {"reference_cosmology":likelihood.reference_cosmology,
                     "nominal_bins":list(config.nominal_bins)}, None,
                     "assembled reference domain only; complete mapped response domain pending")
    shell_good = (likelihood.validation["expected_shells"] ==
                  likelihood.validation["processed_shells"] and
                  all(cell.validation["expected_shells"] == cell.validation["processed_shells"]
                      for cell in likelihood.cells))
    journal.set_gate("no_skipped_shells", "passed" if shell_good else "failed",
                     likelihood.validation, 0, "exact assembled shell inventory")
    prior = validate_prior_result(result)
    journal.set_gate("once_only_priors", prior["status"], prior, 1e-10,
                     "declared once-only prior matrices and active solve")
    journal.set_gate("q13_penalty", prior["status"],
                     {"native_penalties":prior["q13_penalty"], "reasons":prior["reasons"]},
                     1e-10, "complete 3x3 cosmology block inverse")
    solve = result.validation
    journal.set_gate("symmetry_solve", "passed" if shell_good and prior["status"] == "passed" else "failed",
                     solve, 1e-10, "assembler and active solve residual checks")

    by_id = {cell.metadata["cell_id"]:cell for cell in likelihood.cells}
    audits, certificates = [], []
    for cell_id in selected:
        cell = by_id[cell_id]
        journal.event("cell", "started", {"cell_id":cell_id,
                      "expected_shells":cell.validation["expected_shells"]})
        audit = audit_cell_covariance_neighborhood(likelihood, cell, backend)
        audits.append(audit)
        journal.event("covariance_neighborhood", "passed", audit)
        manifest = declare_fd_manifest(likelihood, [cell_id], journal.document["run_id"],
                                       expected_support=support)
        if manifest:
            journal.event("ad_fd", "started", {"cell_id":cell_id,
                          "declared_entries":len(manifest)})
            certificate = assembled_fd_certificate(
                likelihood, backend, [cell_id], manifest,
                expected_support=support, cache=cache, provenance=provenance)
            certificates.append(certificate)
            journal.event("ad_fd", certificate["status"],
                          {"cell_id":cell_id, "measured_entries":len(certificate["measured_coverage"]),
                           "max_error":certificate["max_error"],
                           "failures":certificate["failures"]})
        journal.event("cell", "passed", {"cell_id":cell_id})
    complete = set(selected) == set(by_id)
    for gate, field in (("field_positivity", "field"),
                        ("retained_covariance_positivity", "retained_covariance")):
        journal.set_gate(gate, "passed" if complete else "pending",
                         {"scope":"all_cells" if complete else "representative",
                          "audits":audits,
                          "minimum_scaled_eigenvalue":min(
                              (a.get("min_scaled_eigenvalue", {}).get(field, float("inf"))
                               for a in audits), default=None)},
                         0., "all declared support cells audited" if complete else
                         "representative covariance neighborhood only")
    declaration = declare_fd_manifest(likelihood, selected,
                                      journal.document["run_id"], expected_support=support)
    if certificates:
        combined = _combine_fd(certificates, selected, support)
        journal.set_gate("ad_fd", combined["status"], combined,
                         combined["relative_tolerance"],
                         "independent six-coordinate physical FD comparison")
    return result, likelihood_snapshot(likelihood, prior_parameters=result.parameters), support, declaration


def _resolution_probe(config, provenance, survey, cosmo, backend, root):
    suite_order(kmax=config.kmax, nuisance_factor=config.nuisance_factor,
                exclude_elg1=config.exclude_elg1)
    if (config.dk != .0025 or config.n_mu != 32 or config.kmin != .01
            or config.nominal_bins != NOMINAL_BINS):
        raise ValueError("resolution probe requires adopted .0025, 32 nodes and full fixed-edge eight-bin schedule")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bc-resolution-evidence.json"
    if path.exists():
        raise FileExistsError(f"preserving existing resolution evidence: {path}")
    configuration = {"kmin":config.kmin, "kmax":config.kmax,
                     "dk":config.dk, "n_mu":config.n_mu,
                     "nuisance_factor":config.nuisance_factor,
                     "exclude_elg1":config.exclude_elg1,
                     "nominal_bins":[list(x) for x in config.nominal_bins]}
    report = {"status":"running", "scope":"representative_BC_full_redshift",
              "physical_dependencies":physical_dependencies(provenance),
              "configuration":configuration, "values":{}, "runs":[],
              "comparisons":[], "production_certificate":False,
              "changed_shell_domain_ap":{"status":"pending",
                  "reason":"awaiting exact changed-shell B/C AP reports and live attachment"}}

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)

    save()
    start = time.monotonic()
    try:
        for case in ("B", "C"):
            for tag, dk, n_mu in (("base", .0025, 32),
                                  ("angular", .0025, 64),
                                  ("shell", .00125, 32)):
                key = f"{case}_{tag}"
                report["active_run"] = {"key":key, "dk":dk, "n_mu":n_mu}
                save()
                then = time.monotonic()
                variant = replace(config, case=case, dk=dk, n_mu=n_mu)
                variant_configuration = _configuration(
                    variant, variant.nuisance_factor,
                    requested_dk=provenance["requested_dk"],
                    model_flags=provenance["model_flags"])
                variant_run_id = run_identity(variant_configuration, provenance)
                likelihood = build_revision_likelihood(variant, survey, cosmo, backend)
                result = apply_revision_priors(likelihood, variant.nuisance_factor)
                errors = np.r_[np.sqrt(np.diag(result.cov_cosmo_marg)),
                               np.sqrt(np.diag(result.growth_cov_marg))]
                if (errors.shape != (11,) or not np.all(np.isfinite(errors))
                        or np.any(errors <= 0)
                        or likelihood.validation["expected_shells"] !=
                           likelihood.validation["processed_shells"]):
                    raise ValueError(f"invalid eleven errors or shell inventory in {key}")
                report["values"][key] = errors.tolist()
                report["runs"].append({"key":key, "dk":dk, "n_mu":n_mu,
                    "configuration":variant_configuration, "run_id":variant_run_id,
                    "elapsed_seconds":time.monotonic()-then,
                    "runtime_qualification":"first build in process" if len(report["runs"]) == 0
                    else "same-process subsequent build; changed case/shape can retrace",
                    "expected_shells":likelihood.validation["expected_shells"],
                    "processed_shells":likelihood.validation["processed_shells"],
                    "active_parameters":len(result.names),
                    "growth_redshifts":result.growth_z.tolist(),
                    "likelihood_validation":likelihood.validation,
                    "prior_validation":result.validation})
                save()
        comparison = compare_resolution_errors(report["values"])
        report["resolution_error_status"] = comparison["status"]
        report["status"] = "pending" if comparison["status"] == "passed" else "failed"
        report["comparisons"] = comparison["comparisons"]
    except Exception as exc:
        report.update(status="failed", failure={"exception_type":type(exc).__name__,
                      "exception":str(exc), "traceback":traceback.format_exc()})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        report.pop("active_run", None)
        save()
    if report["status"] == "failed":
        raise ValueError("B/C resolution tolerance failed; evidence preserved")
    return path


def _changed_shell_likelihoods(config, survey, cosmo, backend):
    """Rebuild the two exact changed-shell means/covariances for AP binding."""
    return {case:build_revision_likelihood(
        replace(config, case=case, dk=.00125, n_mu=32), survey, cosmo, backend)
        for case in ("B", "C")}


def _resolution_complete(base_path, b_path, c_path, config, provenance,
                         survey, root, *, cosmo=None, backend=None):
    """Attach two full changed-shell reports without rerunning six builds."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bc-resolution-evidence.complete.json"
    if path.exists():
        raise FileExistsError(f"preserving existing completed resolution evidence: {path}")
    report = _strict_json(base_path.read_text())
    if report.get("status") not in ("pending", "passed"):
        raise ValueError("base six-build resolution evidence is not pending/passed")
    reports = {"B":_strict_json(b_path.read_text()),
               "C":_strict_json(c_path.read_text())}
    report["changed_shell_domain_ap"] = {
        "status":"passed", "reports":reports,
        "sources":{case:{"path":str(source.resolve()),
                         "sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
                   for case, source in (("B",b_path),("C",c_path))}}
    report["status"] = "running"

    def save():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)

    save()
    try:
        if cosmo is None or backend is None:
            raise ValueError("resolution completion requires the live physical backend")
        live = _changed_shell_likelihoods(config, survey, cosmo, backend)
        report["status"] = "passed"
        report["validation"] = validate_resolution_evidence(
            report, config, provenance, survey,
            changed_shell_likelihoods=live)
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"exception_type":type(exc).__name__,
                             "exception":str(exc),
                             "traceback":traceback.format_exc()}
        raise
    finally:
        save()
    return path


def main(argv=None):
    started = time.monotonic()
    args = _arguments(argv)
    if args.max_cells < 0:
        raise ValueError("--max-cells must be nonnegative")
    configs = _configs(args)
    if args.ap_evidence and args.ap_evidence_manifest:
        raise ValueError("--ap-evidence and --ap-evidence-manifest are mutually exclusive")
    if args.ap_evidence and len(configs) > 1:
        raise ValueError("multi-run suite AP attachment requires --ap-evidence-manifest")
    if args.ap_evidence_manifest and args.ap_probe:
        raise ValueError("AP probe measures one selected case without evidence attachment")
    if args.ap_evidence_manifest and args.resolution_probe:
        raise ValueError("resolution probe does not accept AP evidence attachment")
    if args.ap_evidence_manifest and args.resolution_complete:
        raise ValueError("resolution completion uses changed-shell AP reports, not adopted-grid attachment")
    backend = producer._load_backend()
    provenance = producer.collect_provenance(backend, requested_dk=args.requested_dk,
                                             ap_nk=args.ap_nk)
    identities = [_identity(config, provenance) for config in configs]
    ap_manifest = (_load_ap_manifest(args.ap_evidence_manifest,
                                    [run_id for _, run_id in identities])
                   if args.ap_evidence_manifest else None)
    if args.identities:
        print(json.dumps({"scope":"suite" if args.suite else "selected",
                          "configurations":[c for c, _ in identities],
                          "run_ids":[run_id for _, run_id in identities]}, indent=2))
        return 0
    from pfsfog.one_loop import reject_legacy_ap_qualification
    reject_legacy_ap_qualification(backend)
    survey = default_survey_group()
    cosmo = FiducialCosmology()
    if args.resolution_complete:
        if (args.suite or args.smoke_bin or args.resolution_probe or args.ap_probe
                or args.changed_shell_ap_b is None or args.changed_shell_ap_c is None):
            raise ValueError("resolution completion requires B/C AP reports and one full B config")
        path = _resolution_complete(args.resolution_complete,
                                    args.changed_shell_ap_b,
                                    args.changed_shell_ap_c,
                                    configs[0], provenance, survey,
                                    args.evidence_root, cosmo=cosmo, backend=backend)
        print(json.dumps({"scope":"resolution_complete", "path":str(path)}))
        return 0
    if args.resolution_probe:
        path = _resolution_probe(configs[0], provenance, survey, cosmo,
                                 backend, args.evidence_root)
        print(json.dumps({"scope":"resolution_probe", "path":str(path)}))
        return 0
    if args.ap_probe:
        if args.suite or args.resolution_evidence or args.ap_evidence:
            raise ValueError("AP probe measures one selected case without evidence attachment")
        path = run_ap_probe(configs[0], provenance, survey, cosmo, backend,
                            args.evidence_root, identities[0][1],
                            max_cells=args.max_cells)
        print(json.dumps({"scope":"ap_probe", "path":str(path)}))
        return 0
    resolution = (_strict_json(args.resolution_evidence.read_text())
                  if args.resolution_evidence else None)
    ap_evidence = (_strict_json(args.ap_evidence.read_text())
                   if args.ap_evidence else None)
    cache = PhysicalFDCache(args.cache_root or args.evidence_root / "fd-cache")
    journals = []
    cases = {}
    changed_shell_likelihoods = None
    for config, (configuration, run_id) in zip(configs, identities):
        journal = EvidenceJournal(
            args.evidence_root, run_id, configuration, provenance,
            partial_reason=("one nominal bin; full eight-bin support unmeasured"
                            if args.smoke_bin else None))
        journals.append(journal)
        try:
            result, snapshot, support, declaration = _measure(
                config, journal, survey, cosmo, backend, cache, provenance,
                args.max_cells)
            binding = {"configuration":configuration,
                       "physical_dependencies":physical_dependencies(provenance),
                       "run_id":run_id}
            snapshot.update(binding)
            cases[config.case] = (snapshot, {**binding,
                 "cov_cosmo_marg":result.cov_cosmo_marg,
                 "growth_z":result.growth_z, "growth_jacobian":result.growth_jacobian,
                 "growth_fiducial":result.growth_fiducial})
            journal.declared_support = support
            journal.declared_coverage = declaration
            if ap_manifest is not None or ap_evidence is not None:
                if ap_manifest is not None:
                    report_path, digest = ap_manifest[run_id]
                    report = _read_ap_report(report_path, digest, run_id)
                else:
                    report = ap_evidence
                    report_path = args.ap_evidence.resolve()
                    digest = hashlib.sha256(args.ap_evidence.read_bytes()).hexdigest()
                ap_result = validate_ap_evidence(report, config, provenance,
                                                 survey, run_id,
                                                 likelihood=result.likelihood)
                ap_source = {"scope":report["scope"],
                             "source_path":str(report_path),
                             "source_sha256":digest,
                             "validation":ap_result}
                journal.set_gate("domain", "passed", ap_source, 0.,
                                 "all mapped AP/domain points and physical response shapes")
                journal.set_gate("ap_identity_chain_resolution", "passed", ap_source,
                                 REFINEMENT_TOLERANCE,
                                 "complete AP identity, chain, derivative and refinement diagnostics")
                del report
            if resolution is not None:
                if changed_shell_likelihoods is None:
                    changed_shell_likelihoods = _changed_shell_likelihoods(
                        config, survey, cosmo, backend)
                comparison = validate_resolution_evidence(
                    resolution, config, provenance, survey,
                    changed_shell_likelihoods=changed_shell_likelihoods)
                journal.set_gate("dk_angular_convergence", comparison["status"],
                                 {"scope":resolution["scope"],
                                  "source_path":str(args.resolution_evidence.resolve()),
                                  "source_sha256":hashlib.sha256(
                                      args.resolution_evidence.read_bytes()).hexdigest(),
                                  "comparisons":comparison["comparisons"]},
                                 .01, "adopted B/C full-redshift fixed-edge refinements")
            journal.event("selected_config", "finished",
                          {"cache_hits":cache.hits, "cache_misses":cache.misses,
                           "cache_root":str(cache.root.resolve())})
        except Exception as exc:
            journal.failure("selected_config", exc, {"case":config.case,
                            "run_id":run_id})
            journal.write_bundle()
            raise
    if args.suite:
        report = compare_suite_cases(cases, kmax=args.kmax,
                                     nuisance_factor=args.nuisance_factor,
                                     exclude_elg1=args.exclude_elg1)
        for journal in journals:
            journal.set_gate("suite_monotonicity", report["status"], report,
                             1e-10, "matched likelihood and error checks: " +
                             ", ".join(report["checked_transitions"]) +
                             ("; I->A has no sign requirement" if report["unsigned_comparisons"]
                              else ""))
    bundle_paths = []
    for journal in journals:
        path = journal.write_bundle(declared_coverage=journal.declared_coverage,
                                    declared_support=journal.declared_support)
        producer.load_gates(path)
        bundle_paths.append(str(path))
    print(json.dumps({"scope":"suite" if args.suite else "selected",
                      "run_ids":[run_id for _, run_id in identities],
                      "bundles":bundle_paths,
                      "elapsed_seconds":time.monotonic()-started,
                      "cache_hits":cache.hits, "cache_misses":cache.misses}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
