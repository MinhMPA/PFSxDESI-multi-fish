"""Matched-configuration statistical checks for the revision case suite."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import time
import traceback
import numpy as np

from .revision_artifacts import MANDATORY_GATES, _canonical, run_identity
from .revision_validation import compare_marginal_errors, compare_nested_likelihoods
from .revision_validation import evidence_gate_record


BASELINE_ORDER = ("I", "A", "B", "D", "C")
BASELINE_EDGES = (("A", "B"), ("B", "D"), ("D", "C"), ("B", "C"))
NONBASELINE_ORDER = ("B", "C")
NONBASELINE_EDGES = (("B", "C"),)
REQUIRED_CONFIGURATION = frozenset((
    "case", "kmin", "kmax", "dk", "nuisance_factor", "exclude_elg1",
    "k_nl", "r_sigma_v", "n_mu", "nominal_bins", "selection",
    "requested_dk", "adopted_dk", "multipoles", "mean_n_mu",
    "covariance_n_mu", "ap_interpolation_nk", "ap_refinement_nk",
    "ap_interpolation_nmu",
    "fft_kmin", "fft_kmax", "fft_n"))
REQUIRED_PHYSICAL_DEPENDENCIES = frozenset((
    "project_head", "source_content_hash", "source_file_hashes",
    "backend_snapshot", "backend_file_hashes", "executed_module_hashes",
    "network_hash", "emulator_input_hashes", "survey_input_hash",
    "survey_input_hashes", "dependency_hash", "package_versions",
    "interpreter", "precision", "reference_cosmology", "requested_dk",
    "thread_settings", "process_settings", "backend_path", "model_flags"))


def physical_dependencies(provenance):
    """All producer identity fields except invocation text, as in run_identity."""
    return {key:value for key, value in provenance.items()
            if key not in ("command", "runtime", "status", "log")}


def validate_resolution_evidence(report, config, provenance, survey=None, *,
                                 changed_shell_likelihoods=None):
    """Recompute the quantitative result only after exact dependency checks."""
    if report.get("scope") != "representative_BC_full_redshift":
        raise ValueError("resolution scope is not the declared B/C eight-redshift probe")
    if _canonical(report.get("physical_dependencies")) != _canonical(physical_dependencies(provenance)):
        raise ValueError("resolution physical dependencies differ from this run")
    expected = {"kmin":config.kmin, "kmax":config.kmax,
                "dk":config.dk, "n_mu":config.n_mu,
                "nuisance_factor":config.nuisance_factor,
                "exclude_elg1":config.exclude_elg1,
                "nominal_bins":[list(x) for x in config.nominal_bins]}
    if _canonical(report.get("configuration")) != _canonical(expected):
        raise ValueError("resolution physical configuration differs from this run")
    if report.get("status") != "passed":
        raise ValueError("resolution evidence has not passed")
    companion = report.get("changed_shell_domain_ap")
    if (not isinstance(companion, dict) or companion.get("status") != "passed"
            or not isinstance(companion.get("reports"), dict)
            or set(companion["reports"]) != {"B", "C"}):
        raise ValueError("resolution companion quantitative contract is not attachable yet")
    if (not isinstance(changed_shell_likelihoods, dict)
            or set(changed_shell_likelihoods) != {"B", "C"}):
        raise ValueError("live changed-shell likelihoods for B and C are required")
    from .revision_artifacts import _configuration
    from .revision_forecast import make_shell_grid
    from .revision_validation_ap import validate_ap_evidence
    from .revision_validation_physical import expected_support_schedule
    if survey is None:
        from .surveys import default_survey_group
        survey = default_survey_group()
    expected_runs = {}
    for case in ("B", "C"):
        for tag, dk, n_mu in (("base", .0025, 32),
                              ("angular", .0025, 64),
                              ("shell", .00125, 32)):
            variant = replace(config, case=case, dk=dk, n_mu=n_mu)
            key = f"{case}_{tag}"
            configuration = _configuration(
                variant, variant.nuisance_factor,
                requested_dk=provenance["requested_dk"],
                model_flags=provenance["model_flags"])
            support = expected_support_schedule(variant, survey)
            shells = len(make_shell_grid(variant.kmin, variant.kmax, variant.dk).centers)
            expected_runs[key] = (variant, configuration,
                                  run_identity(configuration, provenance),
                                  shells * sum(bool(row["retained_pairs"]) for row in support))
    runs = report.get("runs")
    if (not isinstance(runs, list) or len(runs) != 6
            or len({row.get("key") for row in runs}) != 6
            or {row.get("key") for row in runs} != set(expected_runs)):
        raise ValueError("resolution six-run inventory omitted or duplicated")
    for row in runs:
        key = row["key"]
        variant, configuration, run_id, shells = expected_runs[key]
        if (row.get("run_id") != run_id
                or _canonical(row.get("configuration")) != _canonical(configuration)
                or row.get("dk") != variant.dk or row.get("n_mu") != variant.n_mu
                or row.get("expected_shells") != shells
                or row.get("processed_shells") != shells):
            raise ValueError(f"resolution run identity/configuration/shell mismatch: {key}")
    comparison = compare_resolution_errors(report.get("values", {}))
    if comparison["status"] != "passed":
        raise ValueError("resolution quantitative error comparison failed")
    if (report.get("comparisons") is not None
            and _canonical(report["comparisons"]) != _canonical(comparison["comparisons"])):
        raise ValueError("resolution comparison rows differ from measured values")
    for case in ("B", "C"):
        variant, _, run_id, _ = expected_runs[f"{case}_shell"]
        likelihood = changed_shell_likelihoods[case]
        if likelihood.config != variant:
            raise ValueError(f"live changed-shell likelihood configuration mismatch: {case}")
        validate_ap_evidence(companion["reports"][case], variant,
                             provenance, survey, run_id, likelihood=likelihood)
    return comparison


def compare_resolution_errors(values, *, tolerance=.01):
    """Require all three native and eight growth errors plus C/B ratios."""
    keys = [f"{case}_{tag}" for case in ("B", "C")
            for tag in ("base", "angular", "shell")]
    arrays = {}
    for key in keys:
        if key not in values:
            raise ValueError(f"missing resolution result {key}")
        array = np.asarray(values[key], dtype=float)
        if array.shape != (11,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{key} requires eleven finite positive errors")
        arrays[key] = array
    rows = []
    for tag in ("angular", "shell"):
        for case in ("B", "C"):
            change = np.abs(arrays[f"{case}_{tag}"] / arrays[f"{case}_base"] - 1)
            rows.append({"quantity":f"{case} native/growth errors", "refinement":tag,
                         "relative_change":change.tolist(),
                         "passed":bool(np.all(change < tolerance))})
        base_ratio = arrays["C_base"] / arrays["B_base"]
        refined_ratio = arrays[f"C_{tag}"] / arrays[f"B_{tag}"]
        change = np.abs(refined_ratio / base_ratio - 1)
        rows.append({"quantity":"C/B native/growth error ratios", "refinement":tag,
                     "relative_change":change.tolist(),
                     "passed":bool(np.all(change < tolerance))})
    return {"status":"passed" if all(row["passed"] for row in rows) else "failed",
            "comparisons":rows, "relative_tolerance":float(tolerance),
            "quantity_names":["ln10_10_As", "Mnu", "Omegam"] +
                             [f"fsigma8_bin{i:02d}" for i in range(8)]}


class EvidenceJournal:
    """Serial, atomically persisted measurements and Task 2 gate bindings."""

    def __init__(self, root, run_id, configuration, provenance, *, partial_reason=None):
        self.started_monotonic = time.monotonic()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{run_id}.validation-evidence.json"
        if self.path.exists():
            raise FileExistsError(f"preserving existing validation evidence: {self.path}")
        self.document = {"run_id":run_id, "configuration":configuration,
                         "provenance":provenance,
                         "partial_scope_reason":partial_reason,
                         "started_unix_seconds":time.time(), "events":[],
                         "gates":{gate:{"status":"pending", "measured":None,
                                        "tolerance":None,
                                        "reason":"measurement not completed"}
                                  for gate in MANDATORY_GATES if gate != "save_reload"}}
        self._persist()

    def _persist(self):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(_canonical(self.document) + b"\n")
        temporary.replace(self.path)

    def event(self, operation, status, context):
        self.document["events"].append({"operation":operation, "status":status,
                                        "context":context,
                                        "elapsed_seconds":time.monotonic()-self.started_monotonic})
        self._persist()

    def failure(self, operation, error, context):
        diagnostic = dict(context)
        diagnostic.update({"exception_type":type(error).__name__,
                           "exception":str(error),
                           "diagnostics":getattr(error, "diagnostics", None),
                           "traceback":traceback.format_exc()})
        self.event(operation, "failed", diagnostic)

    def set_gate(self, gate, status, measured, tolerance, reason):
        if gate not in self.document["gates"]:
            raise ValueError(f"unknown external gate {gate}")
        if status not in ("pending", "passed", "failed"):
            raise ValueError(f"invalid gate status {status}")
        partial_reason = self.document["partial_scope_reason"]
        if partial_reason and status == "passed":
            status = "pending"
            reason = f"partial smoke: {partial_reason}; {reason}"
        if gate == "ad_fd" and status == "passed" and (
                measured.get("scope") != "all_cells"
                or measured.get("complete_run_coverage") is not True):
            status = "pending"
            reason = "representative FD measurement; complete support pending"
        self.document["gates"][gate] = {"status":status, "measured":measured,
                                         "tolerance":tolerance, "reason":reason}
        self._persist()

    def mark_partial(self, reason):
        """Preserve measurements while barring a subset from outer passes."""
        self.document["partial_scope_reason"] = reason
        self._demote_partial()
        self.event("scope", "partial", {"reason":reason})

    def _demote_partial(self):
        reason = self.document["partial_scope_reason"]
        if not reason:
            return
        for record in self.document["gates"].values():
            if record["status"] == "passed":
                record["status"] = "pending"
                record["reason"] = f"partial smoke: {reason}; {record['reason']}"
        self._persist()

    def write_bundle(self, *, declared_coverage=None, declared_support=None):
        """Bind each gate from the final document with the canonical A1 binder."""
        self._demote_partial()
        run_id = self.document["run_id"]
        records = {}
        for gate in self.document["gates"]:
            records[gate] = evidence_gate_record(
                self.path, run_id, self.document["configuration"],
                self.document["provenance"], gate,
                declared_coverage=declared_coverage if gate == "ad_fd" else None,
                declared_support=declared_support if gate == "ad_fd" else None)
        path = self.root / f"{run_id}.gates.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(_canonical({"runs":{run_id:records}}) + b"\n")
        temporary.replace(path)
        return path


def validate_prior_result(result, *, tolerance=1e-10):
    """Check once-only declared precisions and the complete 3x3 Q13 solve."""
    names = list(result.names)
    parameters = list(result.parameters)
    reasons = []
    if len(names) != len(parameters) or [p["name"] for p in parameters] != names:
        raise ValueError("prior parameter labels do not match the active basis")
    if names[:3] != ["ln10_10_As", "Mnu", "Omegam"]:
        raise ValueError("native cosmology basis mismatch")
    expected_cosmo = np.zeros((len(names), len(names)))
    expected_cosmo[1, 1], expected_cosmo[2, 2] = 1/25, 1.
    expected_nuisance = np.zeros_like(expected_cosmo)
    for i, parameter in enumerate(parameters):
        width = parameter["prior_sigma"]
        if i >= 3 and width is not None:
            expected_nuisance[i, i] = 1/float(width)**2
    covariance = np.asarray(result.covariance, dtype=float)
    expected_shape = (len(names), len(names))
    if covariance.shape != expected_shape:
        reasons.append("active full covariance shape mismatch")
    elif not np.all(np.isfinite(covariance)):
        reasons.append("nonfinite active covariance")
    elif not np.allclose(covariance, covariance.T, rtol=tolerance, atol=tolerance):
        reasons.append("active full covariance symmetry mismatch")
    covariance_block = covariance[:3, :3] if covariance.ndim == 2 else np.full((3, 3), np.nan)
    checks = {
        "cosmology_prior": (result.F_prior_cosmo, expected_cosmo),
        "nuisance_prior": (result.F_prior_nuisance, expected_nuisance),
        "total": (result.F_total, result.F_like + expected_cosmo + expected_nuisance),
        "marginal_block": (result.cov_cosmo_marg, covariance_block),
    }
    for label, (actual, expected) in checks.items():
        if (np.shape(actual) != np.shape(expected)
                or not np.all(np.isfinite(actual))
                or not np.allclose(actual, expected, rtol=tolerance, atol=tolerance)):
            reasons.append(f"{label} mismatch")
    block = np.asarray(result.F_total[:3, :3], dtype=float)
    fixed = np.asarray(result.cov_cosmo_nuisance_fixed, dtype=float)
    block_scale = np.sqrt(np.diag(block))
    scaled_block = block / block_scale[:, None] / block_scale[None, :]
    scaled_fixed = fixed * block_scale[:, None] * block_scale[None, :]
    if (fixed.shape != (3, 3) or not np.all(np.isfinite(fixed))
            or not np.allclose(scaled_block @ scaled_fixed, np.eye(3),
                               rtol=tolerance, atol=tolerance)):
        reasons.append("complete cosmology block inverse mismatch")
    # Recompute against the saved active matrices. Recorded producer diagnostics
    # are a separate check and cannot substitute for this solve.
    if (covariance.shape == expected_shape and np.all(np.isfinite(covariance))
            and np.shape(result.F_total) == expected_shape
            and np.all(np.isfinite(result.F_total))
            and np.all(np.diag(result.F_total) > 0)):
        total_scale = np.sqrt(np.diag(result.F_total))
        scaled_total = result.F_total / total_scale[:, None] / total_scale[None, :]
        scaled_covariance = covariance * total_scale[:, None] * total_scale[None, :]
        solved = np.einsum("ik,kj->ij", scaled_total, scaled_covariance,
                           optimize=False)
        if (not np.all(np.isfinite(solved))
                or not np.allclose(solved, np.eye(len(names)),
                                   rtol=tolerance, atol=tolerance)):
            reasons.append("active full solve mismatch")
    else:
        reasons.append("active full solve unavailable")
    validation = getattr(result, "validation", None)
    if validation is not None:
        for label, key in (("active full solve", "global_solve"),
                           ("complete cosmology block", "cosmology_block_solve")):
            residual = validation.get(key, {}).get("scaled_inverse_residual")
            if residual is None or not np.isfinite(residual) or residual > tolerance:
                reasons.append(f"{label} scaled inverse residual exceeds tolerance")
    marginal_diag = np.diag(result.cov_cosmo_marg)
    fixed_diag = np.diag(fixed)
    if np.any(marginal_diag <= 0) or np.any(fixed_diag <= 0):
        reasons.append("nonpositive cosmology variances")
        penalty = []
    else:
        penalty = np.sqrt(marginal_diag / fixed_diag).tolist()
        if min(penalty) < 1-tolerance:
            reasons.append("Q13 nuisance penalty below unity")
    return {"status":"failed" if reasons else "passed", "reasons":reasons,
            "q13_penalty":penalty, "relative_tolerance":float(tolerance),
            "basis":names}


def suite_order(*, kmax=.20, nuisance_factor=1., exclude_elg1=False):
    """Declare the approved cohort from the requested setting, never its results."""
    setting = (float(kmax), float(nuisance_factor), bool(exclude_elg1))
    if setting == (.20, 1., False):
        return BASELINE_ORDER
    if setting in ((.15, 1., False), (.20, 3., False),
                   (.20, 10., False), (.20, 1., True)):
        return NONBASELINE_ORDER
    raise ValueError(f"unsupported suite setting {setting}")


def compare_suite_cases(cases, *, kmax=.20, nuisance_factor=1., exclude_elg1=False):
    """Compare live case snapshots and active marginal solves for one approved cohort.

    ``cases`` maps each case to ``(likelihood_snapshot, marginal_snapshot)``.
    Each likelihood and paired solve carries its exact canonical configuration,
    physical producer dependencies and run identity. All non-case settings and
    physical dependencies must agree across the declared members. Baseline I
    receives identity evidence only; no I→A information or error sign is asserted.
    """
    order = suite_order(kmax=kmax, nuisance_factor=nuisance_factor,
                        exclude_elg1=exclude_elg1)
    edges = BASELINE_EDGES if order == BASELINE_ORDER else NONBASELINE_EDGES
    if set(cases) != set(order):
        raise ValueError(f"suite requires exactly {', '.join(order)}")
    checked_transitions = [f"{old}->{new}" for old, new in edges]
    unsigned = ["I->A"] if order == BASELINE_ORDER else []
    identity_reasons = []
    member_ids = {}
    shared_configuration = None
    shared_dependencies = None
    for case in order:
        likelihood, solve = cases[case]
        configuration = likelihood.get("configuration")
        dependencies = likelihood.get("physical_dependencies")
        identity = likelihood.get("run_id")
        if (not isinstance(configuration, dict) or not isinstance(dependencies, dict)
                or not isinstance(identity, str) or not identity):
            identity_reasons.append(f"{case}: missing canonical likelihood identity")
            continue
        missing_configuration = REQUIRED_CONFIGURATION - configuration.keys()
        missing_dependencies = REQUIRED_PHYSICAL_DEPENDENCIES - dependencies.keys()
        if missing_configuration:
            identity_reasons.append(
                f"{case}: incomplete effective configuration: {sorted(missing_configuration)}")
        if missing_dependencies:
            identity_reasons.append(
                f"{case}: incomplete physical dependencies: {sorted(missing_dependencies)}")
        if configuration.get("case") != case:
            identity_reasons.append(f"{case}: configuration case mismatch")
        if (configuration.get("kmax") != kmax
                or configuration.get("nuisance_factor") != nuisance_factor
                or configuration.get("exclude_elg1") != exclude_elg1):
            identity_reasons.append(f"{case}: suite setting differs from declaration")
        try:
            if run_identity(configuration, dependencies) != identity:
                identity_reasons.append(f"{case}: likelihood run identity mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            identity_reasons.append(f"{case}: invalid canonical identity: {exc}")
        for field, value in (("configuration", configuration),
                             ("physical_dependencies", dependencies),
                             ("run_id", identity)):
            if _canonical(solve.get(field)) != _canonical(value):
                identity_reasons.append(f"{case}: solve {field} differs from likelihood")
        non_case = {key:value for key, value in configuration.items() if key != "case"}
        if shared_configuration is None:
            shared_configuration = non_case
            shared_dependencies = dependencies
        elif (_canonical(non_case) != _canonical(shared_configuration)
              or _canonical(dependencies) != _canonical(shared_dependencies)):
            identity_reasons.append(f"{case}: suite physical configuration/dependencies mismatch")
        member_ids[case] = identity
    if identity_reasons:
        return {"status":"failed", "order":list(order), "expected_cases":list(order),
                "expected_transitions":checked_transitions,
                "checked_cases":[], "checked_transitions":[],
                "unsigned_comparisons":unsigned,
                "member_run_ids":member_ids, "edges":{}, "reasons":identity_reasons,
                "i_to_a_sign_requirement":None}
    reports = {}
    reasons = []
    for old_name, new_name in edges:
        old, old_solve = cases[old_name]
        new, new_solve = cases[new_name]
        edge = f"{old_name}->{new_name}"
        edge_reasons = []
        if old.get("case", old_name) != old_name or new.get("case", new_name) != new_name:
            edge_reasons.append("case snapshot label mismatch")
        if old.get("reference_cosmology") != new.get("reference_cosmology"):
            edge_reasons.append("reference cosmology mismatch")
        for key in ("growth_z", "growth_jacobian", "growth_fiducial"):
            a, b = np.asarray(old_solve.get(key)), np.asarray(new_solve.get(key))
            if a.shape != b.shape or not np.array_equal(a, b):
                edge_reasons.append(f"shared {key} mismatch")
        nesting = compare_nested_likelihoods(old, new)
        errors = compare_marginal_errors(old_solve, new_solve)
        edge_reasons += nesting["reasons"] + errors["reasons"]
        reports[edge] = {"status":"failed" if edge_reasons else "passed",
                         "reasons":edge_reasons, "likelihood":nesting,
                         "marginal_errors":errors}
        reasons += [f"{edge}: {reason}" for reason in edge_reasons]
    return {"status":"failed" if reasons else "passed", "order":list(order),
            "expected_cases":list(order), "expected_transitions":checked_transitions,
            "checked_cases":list(order), "checked_transitions":checked_transitions,
            "unsigned_comparisons":unsigned, "member_run_ids":member_ids,
            "edges":reports, "reasons":reasons,
            "i_to_a_sign_requirement":None}
