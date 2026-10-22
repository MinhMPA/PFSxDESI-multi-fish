"""Numerical acceptance checks for the frozen DESI/PFS revision model.

These functions produce explicit measurements. They never promote an artifact;
Task 2's exact-run gate binding and independent reload do that separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .revision_artifacts import _canonical, _strict_json


COSMO_STEPS = {"ln10_10_As": (.001, .0005, .00025),
               "Mnu": (.005, .0025, .00125),
               "Omegam": (.001, .0005, .00025)}
FD_RELATIVE_TOLERANCE = 1e-3
FD_ZERO_TOLERANCE = 1e-6
PSD_RELATIVE_TOLERANCE = 1e-9


def require_complete_coverage(expected, measured):
    """Compare canonical (configuration, cell, pair, ell, column, FD step) rows."""
    def encode(entries):
        result = []
        for entry in entries:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 6
                    or not isinstance(entry[0], str) or not entry[0]
                    or not isinstance(entry[1], str) or not entry[1]
                    or not isinstance(entry[2], (list, tuple)) or len(entry[2]) != 2
                    or entry[3] not in (0, 2, 4)
                    or not isinstance(entry[4], str) or not entry[4]
                    or not isinstance(entry[5], (int, float))
                    or not np.isfinite(entry[5]) or entry[5] <= 0):
                raise ValueError(f"invalid six-coordinate coverage key: {entry}")
            result.append(_canonical(entry).decode())
        return result
    want, got = encode(expected), encode(measured)
    if len(want) != len(set(want)):
        raise ValueError("duplicate expected coverage entry")
    if len(got) != len(set(got)) or set(want) != set(got):
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise ValueError(f"incomplete coverage: missing={missing[:3]}, extra={extra[:3]}, "
                         f"duplicates={len(got)-len(set(got))}")


def _active_for_pair(column, pair, *, space=None):
    if column in (("ln10_10_As", "Mnu", "Omegam") if space is None else space.names):
        return True
    fields = column.split(":")
    if len(fields) != 3:
        raise ValueError(f"unrecognized local parameter label {column}")
    population = fields[1]
    if "|" in population:
        return tuple(population.split("|")) == tuple(pair)
    return population in pair


def compare_cell_derivatives(cell, means, fiducials, steps, *, configuration, physical_scale,
                             relative_tolerance=FD_RELATIVE_TOLERANCE,
                             zero_tolerance=FD_ZERO_TOLERANCE, space=None,
                             analytic_zero_support=()):
    """Compare the retained pair-major/ell-minor tensor with independent means.

    `means[pair](coordinates)` returns (shell, ell) means for the actual pair.
    Whitening and the pass decision use each complete retained column at every
    shell. Pair/ell diagnostics retain routing and zero-block detail. The
    nonzero `physical_scale` expresses an
    absolute zero-response tolerance in dimensionless whitened units.
    """
    if not isinstance(configuration, str) or not configuration:
        raise ValueError("exact configuration identity is required")
    if not np.isfinite(physical_scale) or physical_scale <= 0:
        raise ValueError("physical_scale must be positive and finite")
    pairs = [tuple(pair) for pair in cell.metadata["retained_pairs"]]
    D = np.asarray(cell.derivatives, dtype=float)
    C = np.asarray(cell.covariance, dtype=float)
    names = list(cell.names)
    analytic_zeros = {(name, tuple(pair), ell) for name, pair, ell in analytic_zero_support}
    if (len(analytic_zeros) != len(analytic_zero_support) or any(
            name not in names or pair not in pairs or ell not in (0, 2, 4)
            for name, pair, ell in analytic_zeros)):
        raise ValueError("analytic zero support is outside the measured observations")
    if (D.ndim != 3 or D.shape[1:] != (3*len(pairs), len(names))
            or C.shape != (D.shape[0], D.shape[1], D.shape[1])
            or set(means) != set(pairs) or set(fiducials) != set(names)
            or set(steps) != set(names)):
        raise ValueError("cell derivative/covariance/pair/column layout mismatch")
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(C)):
        raise ValueError("nonfinite derivative or covariance")
    chol = [np.linalg.cholesky(cov) for cov in C]
    expected, measured, failures, errors, column_results = [], [], [], [], []
    for icol, name in enumerate(names):
        if len(steps[name]) != 3 or any(not np.isfinite(x) or x <= 0 for x in steps[name]):
            raise ValueError(f"invalid three-step schedule for {name}")
        global_errors, global_tolerances = [], []
        for step in steps[name]:
            fd_full = np.zeros((D.shape[0], D.shape[1]))
            for ipair, pair in enumerate(pairs):
                shape = (D.shape[0], 3)
                plus, minus = dict(fiducials), dict(fiducials)
                plus[name] += step
                minus[name] -= step
                f_plus = np.asarray(means[pair](plus), dtype=float)
                f_minus = np.asarray(means[pair](minus), dtype=float)
                if f_plus.shape != shape or f_minus.shape != shape or not np.all(np.isfinite(f_plus)) or not np.all(np.isfinite(f_minus)):
                    raise ValueError(f"invalid independent mean for pair {pair}, column {name}, step {step}")
                fd = (f_plus - f_minus)/(2*step)
                fd_full[:, 3*ipair:3*ipair+3] = fd
                for iell, ell in enumerate((0, 2, 4)):
                    key = [configuration, cell.metadata["cell_id"], list(pair), ell,
                           name, float(step)]
                    expected.append(key)
                    ad_white, fd_white = [], []
                    for ik in range(D.shape[0]):
                        row_ad = np.zeros(D.shape[1])
                        row_fd = np.zeros(D.shape[1])
                        row = 3*ipair+iell
                        row_ad[row] = D[ik, row, icol]
                        row_fd[row] = fd[ik, iell]
                        ad_white.append(np.linalg.solve(chol[ik], row_ad))
                        fd_white.append(np.linalg.solve(chol[ik], row_fd))
                    ad_white, fd_white = np.asarray(ad_white), np.asarray(fd_white)
                    difference = float(np.linalg.norm(ad_white-fd_white))
                    denominator = float(np.linalg.norm(fd_white))
                    inactive = not _active_for_pair(name, pair, space=space)
                    angular_zero = (name, pair, ell) in analytic_zeros
                    is_zero = inactive or angular_zero or denominator <= 1e-12*physical_scale
                    error = difference/physical_scale if is_zero else difference/denominator
                    reason = ("zero block" if inactive else "analytic angular zero" if angular_zero
                              else "normalized absolute" if is_zero else "relative")
                    limit = zero_tolerance if is_zero else relative_tolerance
                    entry = {"cell_id":cell.metadata["cell_id"], "pair":list(pair),
                             "ell":ell, "configuration":configuration, "column":name,
                             "step":float(step),
                             "error":float(error), "limit":float(limit), "reason":reason}
                    measured.append(key)
                    errors.append(entry)
                    if (inactive or angular_zero) and (np.linalg.norm(ad_white) > zero_tolerance*physical_scale
                                                       or np.linalg.norm(fd_white) > zero_tolerance*physical_scale):
                        failures.append(entry)
            ad_white = np.asarray([np.linalg.solve(chol[ik], D[ik, :, icol])
                                   for ik in range(D.shape[0])])
            fd_white = np.asarray([np.linalg.solve(chol[ik], fd_full[ik])
                                   for ik in range(D.shape[0])])
            difference = float(np.linalg.norm(ad_white-fd_white))
            denominator = float(np.linalg.norm(fd_white))
            is_zero = denominator <= 1e-12*physical_scale
            global_errors.append(difference/physical_scale if is_zero else difference/denominator)
            global_tolerances.append(zero_tolerance if is_zero else relative_tolerance)
        good = [error < limit for error, limit in zip(global_errors, global_tolerances)]
        consecutive = (good[0] and good[1]) or (good[1] and good[2])
        column_results.append({"configuration":configuration,
                               "cell_id":cell.metadata["cell_id"], "column":name,
                               "steps":[float(step) for step in steps[name]],
                               "step_errors":[float(error) for error in global_errors],
                               "step_tolerances":[float(limit) for limit in global_tolerances],
                               "two_successive_pass":bool(consecutive)})
        if not consecutive:
            failures.append({"configuration":configuration,
                             "cell_id":cell.metadata["cell_id"], "column":name,
                             "pair":max((entry for entry in errors if entry["column"] == name),
                                        key=lambda entry:entry["error"])["pair"],
                             "reason":"full-vector whitened FD", "step_errors":global_errors,
                             "step_tolerances":global_tolerances})
        for pair in pairs:
            for ell in (0, 2, 4):
                group = [entry for entry in errors if entry["column"] == name
                         and entry["pair"] == list(pair) and entry["ell"] == ell]
                row_good = [entry["error"] < entry["limit"] for entry in group]
                inactive = all(entry["reason"] in ("zero block", "analytic angular zero") for entry in group)
                row_pass = all(row_good) if inactive else (
                    (row_good[0] and row_good[1]) or (row_good[1] and row_good[2]))
                if not row_pass:
                    failures.append({"configuration":configuration,
                                     "cell_id":cell.metadata["cell_id"],
                                     "pair":list(pair), "ell":ell, "column":name,
                                     "reason":"pair/ell whitened FD",
                                     "step_errors":[entry["error"] for entry in group],
                                     "step_tolerances":[entry["limit"] for entry in group]})
    require_complete_coverage(expected, measured)
    return {"status":"failed" if failures else "passed", "cell_id":cell.metadata["cell_id"],
            "configuration":configuration, "column_results":column_results,
            "analytic_zero_support":[[name, list(pair), ell] for name, pair, ell in analytic_zero_support],
            "expected_coverage":expected, "measured_coverage":measured,
            "physical_scale":float(physical_scale),
            "relative_tolerance":float(relative_tolerance),
            "zero_tolerance":float(zero_tolerance),
            "max_error":max([entry["error"] for entry in errors]
                            + [error for column in column_results
                               for error in column["step_errors"]], default=0.),
            "errors":errors, "failures":failures}


def evidence_gate_record(path, run_id, configuration, provenance, gate, *,
                         declared_coverage=None, declared_support=None):
    """Bind measured evidence to Task 2's six-field external gate record."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = _strict_json(raw.decode())
    if (document.get("run_id") != run_id
            or _canonical(document.get("configuration")) != _canonical(configuration)
            or _canonical(document.get("provenance")) != _canonical(provenance)):
        raise ValueError("gate evidence identity does not match the exact run")
    result = document.get("gates", {}).get(gate)
    if not isinstance(result, dict) or result.get("status") not in ("passed", "failed", "pending"):
        raise ValueError(f"gate {gate} has no explicit status")
    if not {"measured", "tolerance", "reason"}.issubset(result):
        raise ValueError(f"gate {gate} lacks measured/tolerance/reason")
    if gate == "ad_fd" and result["status"] == "passed":
        measured = result["measured"]
        if (not isinstance(measured, dict)
                or not measured.get("expected_coverage")
                or "measured_coverage" not in measured):
            raise ValueError("passed ad_fd gate lacks declared nonempty coverage")
        require_complete_coverage(measured["expected_coverage"], measured["measured_coverage"])
        if declared_coverage is None:
            raise ValueError("passed ad_fd gate requires independent declared coverage")
        require_complete_coverage(declared_coverage, measured["expected_coverage"])
        if any(key[0] != run_id for key in declared_coverage):
            raise ValueError("ad_fd coverage configuration differs from exact run identity")
        scope_fields = ("scope", "complete_run_coverage", "selected_cell_ids",
                        "all_cell_ids", "expected_support_observations", "physical_scales")
        if any(field in measured for field in scope_fields):
            if (measured.get("scope") != "all_cells"
                    or measured.get("complete_run_coverage") is not True):
                raise ValueError("representative ad_fd measurement cannot bind a complete support gate")
            selected = measured.get("selected_cell_ids")
            all_ids = measured.get("all_cell_ids")
            observations = measured.get("expected_support_observations")
            if (not isinstance(selected, list) or not isinstance(all_ids, list)
                    or not all_ids or len(selected) != len(set(selected))
                    or len(all_ids) != len(set(all_ids)) or set(selected) != set(all_ids)
                    or not isinstance(observations, list)
                    or [entry.get("cell_id") for entry in observations] != all_ids):
                raise ValueError("ad_fd complete support cell ledger mismatch")
            if declared_support is None or [entry.get("cell_id") for entry in declared_support] != all_ids:
                raise ValueError("ad_fd requires independently declared complete support")
            support_keys = []
            for row, independent in zip(observations, declared_support):
                pairs = [tuple(pair) for pair in independent["retained_pairs"]]
                retained = [[list(pair), ell] for pair in pairs for ell in (0, 2, 4)]
                if (row.get("retained_observations") != retained
                        or row.get("absent_observations") != independent["absent_observations"]
                        or row.get("empty_observation") is not (not bool(pairs))):
                    raise ValueError("ad_fd independent support observation mismatch")
                columns = row.get("columns")
                if not isinstance(columns, list) or [record.get("name") for record in columns] != independent["local_names"]:
                    raise ValueError("ad_fd independent support column mismatch")
                for pair in pairs:
                    for ell in (0, 2, 4):
                        for column in columns:
                            for step in column["steps"]:
                                support_keys.append([run_id, row["cell_id"], list(pair),
                                                     ell, column["name"], step])
            require_complete_coverage(support_keys, declared_coverage)
        if measured.get("status") != "passed":
            raise ValueError("passed ad_fd gate contradicts measured status")
        if measured.get("failures") != []:
            raise ValueError("passed ad_fd gate contains measured failures")
        columns = measured.get("column_results")
        if not isinstance(columns, list) or not columns:
            raise ValueError("passed ad_fd gate lacks quantitative column results")
        groups = {}
        for config, cell_id, _pair, _ell, column, step in declared_coverage:
            groups.setdefault((config, cell_id, column), set()).add(step)
        seen = set()
        column_steps = {}
        declared_tolerance = result["tolerance"]
        if isinstance(declared_tolerance, dict):
            declared_tolerance = max(declared_tolerance.values())
        if not isinstance(declared_tolerance, (int, float)) or not np.isfinite(declared_tolerance) or declared_tolerance <= 0:
            raise ValueError("invalid ad_fd tolerance")
        for record in columns:
            key = (record.get("configuration"), record.get("cell_id"), record.get("column"))
            steps = record.get("steps")
            errors = record.get("step_errors")
            limits = record.get("step_tolerances")
            if (key not in groups or key in seen or not isinstance(steps, list)
                    or not isinstance(errors, list) or not isinstance(limits, list)
                    or len(steps) != len(errors) or len(steps) != len(limits)
                    or len(steps) != 3 or set(steps) != groups[key]
                    or steps != sorted(steps, reverse=True)):
                raise ValueError("ad_fd column result/coverage mismatch")
            seen.add(key)
            column_steps[key] = tuple(steps)
            if (not all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in errors)
                    or not all(isinstance(x, (int, float)) and np.isfinite(x)
                               and 0 < x <= declared_tolerance for x in limits)):
                raise ValueError("ad_fd measured error or tolerance is invalid")
            good = [error < limit for error, limit in zip(errors, limits)]
            consecutive = (good[0] and good[1]) or (good[1] and good[2])
            if record.get("two_successive_pass", consecutive) is not consecutive or not consecutive:
                raise ValueError("ad_fd measured error exceeds tolerance at two successive steps")
        if seen != set(groups):
            raise ValueError("ad_fd column results omit declared coverage")
        row_errors = measured.get("errors")
        if not isinstance(row_errors, list):
            raise ValueError("passed ad_fd gate lacks quantitative row results")
        expected_rows = {(config, cell_id, tuple(pair), ell, column, step)
                         for config, cell_id, pair, ell, column, step in declared_coverage}
        rows = {}
        row_groups = {}
        for entry in row_errors:
            key = (entry.get("configuration"), entry.get("cell_id"),
                   tuple(entry.get("pair", ())), entry.get("ell"),
                   entry.get("column"), entry.get("step"))
            if key not in expected_rows or key in rows:
                raise ValueError("ad_fd row result/coverage mismatch")
            error, limit = entry.get("error"), entry.get("limit")
            if (not isinstance(error, (int, float)) or not np.isfinite(error) or error < 0
                    or not isinstance(limit, (int, float)) or not np.isfinite(limit)
                    or not 0 < limit <= declared_tolerance):
                raise ValueError("ad_fd row error or tolerance is invalid")
            rows[key] = entry
            row_groups.setdefault(key[:5], []).append(entry)
        if set(rows) != expected_rows:
            raise ValueError("ad_fd row results omit declared coverage")
        for key, group in row_groups.items():
            prescribed = column_steps[(key[0], key[1], key[4])]
            if len(group) != 3 or {entry["step"] for entry in group} != set(prescribed):
                raise ValueError("ad_fd row steps differ from prescribed column steps")
            group.sort(key=lambda entry:entry["step"], reverse=True)
            good = [entry["error"] < entry["limit"] for entry in group]
            inactive = all(entry.get("reason") == "zero block" for entry in group)
            accepted = all(good) if inactive else ((good[0] and good[1]) or (good[1] and good[2]))
            if not accepted:
                raise ValueError("ad_fd row error exceeds tolerance")
    return {"status":result["status"], "measured":result["measured"],
            "tolerance":result["tolerance"], "reason":result["reason"],
            "evidence_path":str(path), "evidence_hash":hashlib.sha256(raw).hexdigest()}


def _scale_fisher(matrix, scales):
    return np.asarray(matrix, dtype=float)*scales[:, None]*scales[None, :]


def _old_geometry_is_nested(old, new):
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    for key, value in old.items():
        if key not in new:
            return False
        if isinstance(value, dict):
            if not _old_geometry_is_nested(value, new[key]):
                return False
        elif _canonical(value) != _canonical(new[key]):
            return False
    return True


def compare_nested_likelihoods(old, new, *, relative_tolerance=PSD_RELATIVE_TOLERANCE):
    """Check exact old-observation nesting before a union-basis likelihood PSD test.

    Inputs are compact live snapshots with names/fiducials/prior widths, F_like,
    shell centers and each cell's retained pairs, D and covariance. Smaller
    union extensions are never inverted.
    """
    reasons = []
    old_names, new_names = list(old["names"]), list(new["names"])
    if not set(old_names).issubset(new_names):
        reasons.append("old parameter labels are not nested")
    union = list(dict.fromkeys(old_names + new_names))
    for field in ("fiducial", "prior_sigma"):
        a = dict(zip(old_names, old[field]))
        b = dict(zip(new_names, new[field]))
        if any(a[name] != b[name] for name in set(a) & set(b)):
            reasons.append(f"shared {field} mismatch")
    if not np.array_equal(old["k"], new["k"]) or not np.array_equal(old["dk"], new["dk"]):
        reasons.append("shell grid mismatch")
    old_cells = {cell["cell_id"]:cell for cell in old["cells"]}
    new_cells = {cell["cell_id"]:cell for cell in new["cells"]}
    for cell_id, a in old_cells.items():
        b = new_cells.get(cell_id)
        if b is None:
            reasons.append(f"missing old cell {cell_id}")
            continue
        if not _old_geometry_is_nested(a.get("geometry"), b.get("geometry")):
            reasons.append(f"geometry mismatch in {cell_id}")
        old_pairs = [tuple(pair) for pair in a["pairs"]]
        new_pairs = [tuple(pair) for pair in b["pairs"]]
        if (not set(old_pairs).issubset(new_pairs)
                or any(new_pairs.count(pair) != 1 for pair in old_pairs)):
            reasons.append(f"retained pair not nested in {cell_id}")
            continue
        rows = [3*new_pairs.index(pair)+i for pair in old_pairs for i in range(3)]
        da, db = np.asarray(a["derivatives"]), np.asarray(b["derivatives"])
        ca, cb = np.asarray(a["covariance"]), np.asarray(b["covariance"])
        if da.shape[1] != len(rows) or ca.shape[1:] != (len(rows), len(rows)):
            reasons.append(f"old cell layout mismatch in {cell_id}")
            continue
        selected_cov = cb[:, rows][:, :, rows]
        if ca.shape != selected_cov.shape:
            reasons.append(f"covariance submatrix shape mismatch in {cell_id}")
        elif not np.all(np.isfinite(ca)) or not np.all(np.isfinite(selected_cov)):
            reasons.append(f"nonfinite covariance submatrix in {cell_id}")
        elif not np.array_equal(ca, selected_cov):
            first = tuple(np.argwhere(ca != selected_cov)[0])
            reasons.append(f"covariance submatrix mismatch in {cell_id} at {first}")
        old_local = list(a.get("names", old_names))
        new_local = list(b.get("names", new_names))
        if (len(old_local) != da.shape[2] or len(new_local) != db.shape[2]
                or len(set(old_local)) != len(old_local)
                or any(new_local.count(name) != 1 for name in old_local)):
            reasons.append(f"old local label missing or duplicated in {cell_id}")
            continue
        bi = [new_local.index(name) for name in old_local]
        selected_derivatives = db[:, rows][:, :, bi]
        if da.shape != selected_derivatives.shape:
            reasons.append(f"derivative definition shape mismatch in {cell_id}")
        elif not np.all(np.isfinite(da)) or not np.all(np.isfinite(selected_derivatives)):
            reasons.append(f"nonfinite derivative definition in {cell_id}")
        elif not np.array_equal(da, selected_derivatives):
            first = tuple(np.argwhere(da != selected_derivatives)[0])
            reasons.append(f"derivative definition mismatch in {cell_id} at "
                           f"row/column {first} {old_local[first[2]]}")
    def embed(record):
        output = np.zeros((len(union), len(union)))
        indices = [union.index(name) for name in record["names"]]
        output[np.ix_(indices, indices)] = record["F_like"]
        return output
    difference = embed(new)-embed(old)
    scale_by_name = {}
    for record in (old, new):
        for name, fid, prior in zip(record["names"], record["fiducial"], record["prior_sigma"]):
            scale_by_name[name] = max(1., abs(fid), 0. if prior is None else prior)
    scales = np.array([scale_by_name[name] for name in union], dtype=float)
    scaled = (_scale_fisher(difference, scales)+_scale_fisher(difference.T, scales))/2
    if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(scales)):
        return {"status":"failed", "reasons":reasons+["nonfinite likelihood information"],
                "basis":union,
                "scales":[float(x) if np.isfinite(x) else None for x in scales],
                "eigenvalues":[],
                "minimum_eigenvalue":None, "near_null_eigenvalues":0,
                "relative_eigenvalue_tolerance":float(relative_tolerance),
                "absolute_eigenvalue_tolerance":None}
    eigenvalues = np.linalg.eigvalsh(scaled)
    spectral_scale = max(1., float(np.max(np.abs(eigenvalues))))
    threshold = relative_tolerance*spectral_scale
    if eigenvalues[0] < -threshold:
        reasons.append("likelihood information difference is not PSD")
    return {"status":"failed" if reasons else "passed", "reasons":reasons,
            "basis":union, "scales":scales.tolist(),
            "eigenvalues":eigenvalues.tolist(),
            "minimum_eigenvalue":float(eigenvalues[0]),
            "near_null_eigenvalues":int(np.count_nonzero(np.abs(eigenvalues) <= threshold)),
            "relative_eigenvalue_tolerance":float(relative_tolerance),
            "absolute_eigenvalue_tolerance":float(threshold)}


def compare_marginal_errors(old, new, *, relative_tolerance=1e-10, space=None):
    """Compare native and eight derived errors from each case's active solve."""
    reasons = []
    ncosmo = 3 if space is None else len(space.names)
    if space is not None:
        for record in (old,new):
            if tuple(record.get("cosmology_columns", ())) != space.names:
                raise ValueError("marginal comparison cosmology column mismatch")
        if (old.get("treatment") != new.get("treatment") or not old.get("treatment")
                or old.get("free_cosmology") != new.get("free_cosmology")):
            raise ValueError("marginal comparison treatment mismatch")
    z_old, z_new = np.asarray(old["growth_z"]), np.asarray(new["growth_z"])
    j_old, j_new = np.asarray(old["growth_jacobian"]), np.asarray(new["growth_jacobian"])
    if (z_old.shape != (8,) or j_old.shape != (8, ncosmo)
            or not np.array_equal(z_old, z_new)
            or not np.allclose(j_old, j_new, rtol=1e-12, atol=1e-14)):
        reasons.append("growth redshift or fiducial Jacobian mismatch")
    ratios = {}
    for label, record in (("old", old), ("new", new)):
        covariance = np.asarray(record["cov_cosmo_marg"], dtype=float)
        jacobian = np.asarray(record["growth_jacobian"], dtype=float)
        if covariance.shape != (ncosmo, ncosmo) or jacobian.shape != (8, ncosmo):
            raise ValueError("invalid active cosmology solve shape")
        native_var = np.diag(covariance)
        if space is not None:
            free = tuple(record.get("free_cosmology", ()))
            if (not free or len(set(free)) != len(free)
                    or tuple(name for name in space.names if name in free) != free):
                raise ValueError("invalid ordered free cosmology names")
            fixed = [i for i,name in enumerate(space.names) if name not in free]
            if fixed and (np.any(covariance[fixed] != 0) or np.any(covariance[:,fixed] != 0)):
                raise ValueError("fixed covariance columns must be zero embedded")
            native_var = native_var[[space.names.index(name) for name in free]]
        growth_var = np.einsum("zi,ij,zj->z", jacobian, covariance, jacobian)
        if (not np.all(np.isfinite(native_var)) or not np.all(np.isfinite(growth_var))
                or np.any(native_var <= 0) or np.any(growth_var <= 0)):
            raise ValueError(f"invalid {label} marginal variances")
        ratios[label] = (np.sqrt(native_var), np.sqrt(growth_var))
    native = ratios["new"][0]/ratios["old"][0]
    growth = ratios["new"][1]/ratios["old"][1]
    if np.any(native > 1+relative_tolerance):
        reasons.append("native cosmology error worsened")
    if np.any(growth > 1+relative_tolerance):
        reasons.append("derived growth error worsened")
    return {"status":"failed" if reasons else "passed", "reasons":reasons,
            "native_ratio":native.tolist(), "growth_ratio":growth.tolist(),
            "relative_tolerance":float(relative_tolerance),
            **({} if space is None else {
                "native_columns":list(old["free_cosmology"]),
                "old_native_sigma":ratios["old"][0].tolist(),
                "new_native_sigma":ratios["new"][0].tolist(),
                "old_growth_sigma":ratios["old"][1].tolist(),
                "new_growth_sigma":ratios["new"][1].tolist()})}
