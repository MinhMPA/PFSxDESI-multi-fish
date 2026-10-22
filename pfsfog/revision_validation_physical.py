"""Live physical measurements for the revision's run-bound numerical gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .revision_artifacts import _canonical
from .revision_forecast import _scaled_spd, retained_pairs
from .survey_geometry import ShellGeometry, support_cells
from .revision_theory import COSMO_NAMES, NUISANCE_NAMES, make_pair_response
from .cosmology_space import (cosmology_from_coordinates, TRAINING_BOUNDS,
                              validate_coordinate_neighborhood)
from .eft_params import CROSS_STOCH_NAMES
from .covariance_mt_general import multi_tracer_cov_general
from .revision_validation import (COSMO_STEPS, FD_RELATIVE_TOLERANCE,
                                  FD_ZERO_TOLERANCE, compare_cell_derivatives,
                                  require_complete_coverage)


def likelihood_snapshot(likelihood, *, prior_parameters=None):
    """Copy the live shared basis and observed cell matrices for nesting tests."""
    parameters = likelihood.parameters if prior_parameters is None else prior_parameters
    if ([record["name"] for record in parameters] != list(likelihood.names)
            or len(likelihood.fiducial) != len(likelihood.names)):
        raise ValueError("likelihood snapshot parameter layout mismatch")
    cells = []
    empty = []
    for cell in likelihood.cells:
        metadata = cell.metadata
        _checked_pairs(likelihood, cell)
        _check_shells(likelihood, cell)
        if not metadata["retained_pairs"]:
            empty.append(metadata["cell_id"])
            continue
        cells.append({
            "cell_id":metadata["cell_id"],
            "geometry":{"nominal_bin":list(metadata["nominal_bin"]),
                        "radial_bin":list(metadata["radial_bin"]),
                        "model_redshift":float(metadata["model_redshift"]),
                        "region":metadata["region"], "volume":float(metadata["volume"]),
                        "nbar":dict(metadata["nbar"]),
                        "k_nl":float(likelihood.config.k_nl),
                        "n_mu":int(likelihood.config.n_mu)},
            "pairs":[list(pair) for pair in metadata["retained_pairs"]],
            "names":list(cell.names),
            "derivatives":np.array(cell.derivatives, dtype=float, copy=True),
            "covariance":np.array(cell.covariance, dtype=float, copy=True),
        })
    return {"case":likelihood.config.case,
            "names":list(likelihood.names),
            "fiducial":np.asarray(likelihood.fiducial, dtype=float).tolist(),
            "prior_sigma":[record["prior_sigma"] for record in parameters],
            "F_like":np.array(likelihood.F_like, dtype=float, copy=True),
            "k":np.asarray(likelihood.grid.centers, dtype=float).copy(),
            "dk":np.asarray(likelihood.grid.widths, dtype=float).copy(),
            "reference_cosmology":dict(likelihood.reference_cosmology),
            "empty_cell_ids":empty, "cells":cells}


class CovarianceAuditFailure(ValueError):
    """A rejected physical neighborhood with its scaled spectral diagnostics."""

    def __init__(self, diagnostics, reason):
        self.diagnostics = diagnostics
        super().__init__(f"{reason}: {diagnostics['context']}; "
                         f"min_scaled_eigenvalue={diagnostics['min_scaled_eigenvalue']}, "
                         f"scaled_condition={diagnostics['scaled_condition']}, "
                         f"scaled_residual={diagnostics['scaled_residual']}")


def _point_details(context):
    details = {"context":context}
    for field in ("cell", "column", "sign", "step", "k", "mu"):
        match = re.search(rf"(?:^| )({field})=([^ ]+)", context)
        if match:
            value = match.group(2)
            details["cell_id" if field == "cell" else field] = (
                None if value == "unavailable" else
                float(value) if field in ("step", "k", "mu") else value)
    return details


def _audited_spd(matrix, context):
    """Measure even a rejected matrix; never repair or clip it."""
    m = np.asarray(matrix, dtype=float)
    diagnostics = {**_point_details(context), "min_scaled_eigenvalue":None,
                   "scaled_condition":None, "scaled_residual":None}
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
        diagnostics["eigenvalue_unavailable_reason"] = "nonfinite or nonsquare covariance"
        raise CovarianceAuditFailure(diagnostics, "nonfinite or nonsquare covariance")
    diagonal = np.diag(m)
    floor = max(float(np.max(np.abs(diagonal))) * 1e-15, np.finfo(float).tiny)
    scale = np.sqrt(np.maximum(np.abs(diagonal), floor))
    scaled = m / scale[:, None] / scale[None, :]
    sym = (scaled + scaled.T) / 2
    try:
        eigen, vectors = np.linalg.eigh(sym)
    except np.linalg.LinAlgError as exc:
        diagnostics["eigenvalue_unavailable_reason"] = f"eigensolver failed: {exc}"
        raise CovarianceAuditFailure(diagnostics, "covariance eigensolver failed") from exc
    absolute = np.abs(eigen)
    diagnostics.update({
        "min_scaled_eigenvalue":float(eigen[0]),
        "scaled_condition":float(np.max(absolute) / np.min(absolute))
        if np.min(absolute) > 0 else None,
        "scaled_residual":float(np.max(np.abs(
            np.einsum("ik,k,jk->ij", vectors, eigen, vectors, optimize=False) - sym))),
        "scaled_asymmetry":float(np.max(np.abs(scaled - scaled.T))),
    })
    if (np.any(diagonal <= 0) or eigen[0] <= 0
            or diagnostics["scaled_asymmetry"] > 1e-10
            or diagnostics["scaled_residual"] > 1e-10):
        raise CovarianceAuditFailure(diagnostics, "nonpositive or asymmetric covariance")
    try:
        return _scaled_spd(m, context)
    except ValueError as exc:
        raise CovarianceAuditFailure(diagnostics, str(exc)) from exc


def _cell_fiducials(likelihood, cell):
    global_values = dict(zip(likelihood.names, likelihood.fiducial))
    if len(global_values) != len(likelihood.names):
        raise ValueError("duplicate global fiducial label")
    if len(set(cell.names)) != len(cell.names) or any(name not in global_values for name in cell.names):
        raise ValueError("cell local/global fiducial label mismatch")
    values = {name:float(global_values[name]) for name in cell.names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite local fiducial")
    return values


def _checked_pairs(likelihood, cell):
    metadata = cell.metadata
    active = tuple(metadata["active_populations"])
    want = retained_pairs(likelihood.config.case, active)
    got = [tuple(pair) for pair in metadata["retained_pairs"]]
    if got != want or len(got) != len(set(got)):
        raise ValueError(f"retained pair metadata mismatch in {metadata['cell_id']}")
    if metadata.get("retained_observations") != [(pair, ell) for pair in want for ell in (0, 2, 4)]:
        raise ValueError(f"retained pair observation layout mismatch in {metadata['cell_id']}")
    return got


def validate_physical_point(theta, reference, space, redshift):
    """Check one mapped input outside AD, before any external prediction."""
    if not np.isfinite(redshift) or not 0 <= redshift <= 5:
        raise ValueError("redshift outside [0, 5]")
    mapped = {key:float(value) for key,value in
              cosmology_from_coordinates(theta, reference, space).items()}
    for name, (lower,upper) in TRAINING_BOUNDS.items():
        if name == "redshift":
            continue
        if not np.isfinite(mapped[name]) or not lower <= mapped[name] <= upper:
            raise ValueError(f"mapped {name} outside [{lower}, {upper}]")
    return mapped


def _steps(name, fiducial, *, space=None):
    if space is not None and name in space.names:
        return tuple(space.fd_steps[name])
    if name in COSMO_STEPS:
        return tuple(COSMO_STEPS[name])
    scale = max(1., abs(fiducial))
    return tuple(scale * x for x in (1e-3, 5e-4, 2.5e-4))


def _support_row(config, cell, ibin, icell, region, *, space=None):
    candidates = tuple(cell.overlap_tracers if region == "overlap"
                       else cell.nonoverlap_tracers)
    volume = cell.overlap_volume if region == "overlap" else cell.nonoverlap_volume
    active = (tuple(sorted(name for name in candidates
                           if config.case in ("C", "D") or name.startswith("DESI-")))
              if volume > 0 and retained_pairs(config.case, candidates) else ())
    pairs = retained_pairs(config.case, active)
    theoretical = [(name, name) for name in active] + [
        (a, b) for i, a in enumerate(active) for b in active[i + 1:]]
    absent = [[list(pair), ell] for pair in theoretical if pair not in pairs
              for ell in (0, 2, 4)]
    names = list(COSMO_NAMES if space is None else space.names) if pairs else []
    names += [f"bin{ibin:02d}:{tracer}:{coord}" for tracer in active
              for coord in NUISANCE_NAMES]
    names += [f"bin{ibin:02d}:{a}|{b}:{coord}" for a, b in pairs if a != b
              for coord in CROSS_STOCH_NAMES]
    return {"cell_id":f"bin{ibin:02d}-radial{icell:02d}-{region}",
            "nominal_bin_index":ibin, "nominal_bin":tuple(cell.nominal_bin),
            "radial_bin":tuple(cell.radial_bin), "region":region,
            "model_redshift":float(cell.model_redshift), "volume":float(volume),
            "nbar":{name:cell.nbar[name] for name in (active if pairs else candidates)},
            "candidate_populations":candidates,
            "active_populations":active, "retained_pairs":pairs,
            "local_names":names, "empty_observation":not bool(pairs),
            "absent_observations":absent}


def expected_support_schedule(config, survey_group, *, space=None):
    """Derive cell and observation support directly from survey geometry."""
    geometry = ShellGeometry.from_group(survey_group)
    rows = []
    for ibin, nominal in enumerate(config.nominal_bins):
        selected = support_cells(survey_group, nominal, include_pfs=True,
                                 exclude_elg1=config.exclude_elg1, geometry=geometry)
        for icell, cell in enumerate(selected):
            for region in ("overlap", "nonoverlap"):
                rows.append(_support_row(config, cell, ibin, icell, region, space=space))
    return rows


def validate_support_schedule(likelihood, expected_support, *, space=None):
    """Check an independently supplied complete support and empty-cell ledger."""
    if not expected_support or not isinstance(expected_support, (list, tuple)):
        raise ValueError("independent support schedule is required")
    want_ids = [row["cell_id"] for row in expected_support]
    actual_ids = [cell.metadata["cell_id"] for cell in likelihood.cells]
    if (len(want_ids) != len(set(want_ids)) or len(actual_ids) != len(set(actual_ids))
            or want_ids != actual_ids):
        raise ValueError("independent support cell IDs differ from assembled cells")
    checked = []
    for row, cell in zip(expected_support, likelihood.cells):
        metadata = cell.metadata
        candidate = tuple(row["candidate_populations"])
        volume = float(row["volume"])
        active = (tuple(sorted(name for name in candidate
                               if likelihood.config.case in ("C", "D")
                               or name.startswith("DESI-")))
                  if volume > 0 and retained_pairs(likelihood.config.case, candidate) else ())
        pairs = retained_pairs(likelihood.config.case, active)
        theoretical = [(name, name) for name in active] + [
            (a, b) for i, a in enumerate(active) for b in active[i + 1:]]
        absent = [[list(pair), ell] for pair in theoretical if pair not in pairs
                  for ell in (0, 2, 4)]
        ibin = row["nominal_bin_index"]
        names = list(COSMO_NAMES if space is None else space.names) if pairs else []
        names += [f"bin{ibin:02d}:{tracer}:{coord}" for tracer in active
                  for coord in NUISANCE_NAMES]
        names += [f"bin{ibin:02d}:{a}|{b}:{coord}" for a, b in pairs if a != b
                  for coord in CROSS_STOCH_NAMES]
        expected = {"nominal_bin_index":ibin, "nominal_bin":row["nominal_bin"],
                    "radial_bin":row["radial_bin"], "region":row["region"],
                    "model_redshift":row["model_redshift"], "volume":volume,
                    "nbar":row["nbar"], "candidate_populations":candidate,
                    "active_populations":active, "retained_pairs":pairs,
                    "local_names":names, "theoretical_pairs":theoretical}
        if (tuple(row["active_populations"]) != active
                or [tuple(pair) for pair in row["retained_pairs"]] != pairs
                or list(row["local_names"]) != names
                or bool(row["empty_observation"]) != (not bool(pairs))
                or row["absent_observations"] != absent
                or any(_canonical(metadata.get(field)) != _canonical(value)
                       for field, value in expected.items())
                or list(cell.names) != names):
            raise ValueError(f"independent support mismatch in {row['cell_id']}")
        _checked_pairs(likelihood, cell)
        _check_shells(likelihood, cell)
        fiducial = _cell_fiducials(likelihood, cell) if names else {}
        checked.append({"cell_id":row["cell_id"],
                        "retained_observations":[[list(pair), ell] for pair in pairs
                                                 for ell in (0, 2, 4)],
                        "absent_observations":absent,
                        "columns":[{"name":name,
                                    "steps":list(_steps(name, fiducial[name], space=space))}
                                   for name in names],
                        "empty_observation":not bool(pairs)})
    return checked


def _selected(likelihood, selected_cell_ids):
    ids = list(selected_cell_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selected cell IDs must be nonempty and unique")
    by_id = {cell.metadata["cell_id"]:cell for cell in likelihood.cells}
    if len(by_id) != len(likelihood.cells) or any(cell_id not in by_id for cell_id in ids):
        raise ValueError("selected cell ID absent or duplicated in likelihood")
    return [by_id[cell_id] for cell_id in ids]


def _check_shells(likelihood, cell):
    expected = len(likelihood.grid.centers) if cell.metadata["retained_pairs"] else 0
    if (cell.validation["expected_shells"] != expected
            or cell.validation["processed_shells"] != expected
            or (expected and (cell.derivatives.shape[0] != expected
                              or cell.covariance.shape[0] != expected))):
        raise ValueError(f"missing or skipped shell in {cell.metadata['cell_id']}")
    if not np.all(np.isfinite(cell.derivatives)):
        raise ValueError(f"nonfinite derivative in {cell.metadata['cell_id']}")


def physical_cell_means(likelihood, cell, ps, *, cache=None, provenance=None, space=None):
    """Build independent response means with the assembler's exact local basis."""
    pairs = _checked_pairs(likelihood, cell)
    _check_shells(likelihood, cell)
    fiducial = _cell_fiducials(likelihood, cell)
    ibin = cell.metadata["nominal_bin_index"]
    active = tuple(cell.metadata["active_populations"])
    expected_names = list(COSMO_NAMES if space is None else space.names)
    expected_names += [f"bin{ibin:02d}:{tracer}:{coord}" for tracer in active
                       for coord in NUISANCE_NAMES]
    expected_names += [f"bin{ibin:02d}:{a}|{b}:{coord}" for a, b in pairs if a != b
                       for coord in CROSS_STOCH_NAMES]
    if cell.names != expected_names or list(cell.metadata["local_names"]) != expected_names:
        raise ValueError(f"cell local label layout mismatch in {cell.metadata['cell_id']}")
    if cache is not None and provenance is None:
        raise ValueError("physical FD cache requires provenance")
    reference = likelihood.reference_cosmology
    z = cell.metadata["model_redshift"]
    nbar = cell.metadata["nbar"]
    k_nl = likelihood.config.k_nl
    k = np.asarray(likelihood.grid.centers, dtype=float)
    if space is not None:
        validate_coordinate_neighborhood(np.array([fiducial[n] for n in space.names]),
                                         {**reference,"z":z},space)
    means = {}
    for pair in pairs:
        a, b = pair
        response = make_pair_response(ps, z, reference, nbar[a],
                                      None if a == b else nbar[b], k_nl,
                                      ap_nk=likelihood.config.ap_interpolation_nk,
                                      ap_nmu=likelihood.config.ap_interpolation_nmu,
                                      **({} if space is None else {"space":space}))

        def evaluate(coordinates, pair=pair, response=response):
            if set(coordinates) != set(fiducial) or any(not np.isfinite(value) for value in coordinates.values()):
                raise ValueError("physical mean coordinate mismatch")
            a, b = pair
            theta = np.asarray([coordinates[name] for name in (COSMO_NAMES if space is None else space.names)])
            q_a = np.asarray([coordinates[f"bin{ibin:02d}:{a}:{name}"]
                              for name in NUISANCE_NAMES])
            q_b = (None if a == b else np.asarray([
                coordinates[f"bin{ibin:02d}:{b}:{name}"] for name in NUISANCE_NAMES]))
            q_cross = (None if a == b else np.asarray([
                coordinates[f"bin{ibin:02d}:{a}|{b}:{name}"] for name in CROSS_STOCH_NAMES]))

            if space is not None:
                validate_physical_point(theta, reference, space, z)

            def compute():
                value = np.asarray(response.multipoles(
                    k, theta, q_a, q_b, q_cross, n_mu=likelihood.config.n_mu), dtype=float)
                if value.shape != (len(k), 3) or not np.all(np.isfinite(value)):
                    raise ValueError(f"nonfinite physical mean for {cell.metadata['cell_id']} {pair}")
                return value

            if cache is None:
                return compute()
            changed = [(name, float(value)) for name, value in coordinates.items()
                       if value != fiducial[name]]
            if len(changed) > 1:
                raise ValueError("physical cache accepts only one-coordinate FD points")
            label, value = changed[0] if changed else ("fiducial", None)
            context = {"provenance":{field:provenance[field] for field in (
                "project_head", "source_content_hash", "executed_module_hashes",
                "backend_snapshot", "dependency_hash", "network_hash", "survey_input_hash")},
                "reference_theta":likelihood.reference_cosmology,
                **({} if space is None else {"cosmology_space":{
                    "names":space.names,"h_ref":space.h_ref,
                    "unit_convention":space.unit_convention}}),
                "ordered_local_fiducials":[[name, fiducial[name]] for name in cell.names],
                "cell":{"z":z, "nbar":nbar, "k_nl":k_nl},
                "ap_fft":{"ap_interpolation_nk":likelihood.config.ap_interpolation_nk,
                          "ap_refinement_nk":list(likelihood.config.ap_refinement_nk),
                          "ap_interpolation_nmu":likelihood.config.ap_interpolation_nmu,
                          "fft_kmin":likelihood.model["fft_kmin"],
                          "fft_kmax":likelihood.model["fft_kmax"],
                          "fft_n":likelihood.model["fft_n"]},
                "shell_centers":k.tolist(), "pair":pair,
                "coordinate":label, "value":value,
                "three_fd_steps":None if label == "fiducial" else _steps(label, fiducial[label], space=space),
                "response_angular_order":likelihood.config.n_mu}
            return cache.get_or_compute(context, compute)

        means[pair] = evaluate
    return means


def declare_fd_manifest(likelihood, selected_cell_ids, run_id, *, expected_support, space=None):
    """Predeclare the independent six-coordinate measurement universe."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("exact run ID is required")
    validate_support_schedule(likelihood, expected_support, space=space)
    support_by_id = {row["cell_id"]:row for row in expected_support}
    rows = []
    for cell in _selected(likelihood, selected_cell_ids):
        support = support_by_id[cell.metadata["cell_id"]]
        pairs = [tuple(pair) for pair in support["retained_pairs"]]
        fiducial = _cell_fiducials(likelihood, cell)
        for pair in pairs:
            for ell in (0, 2, 4):
                for name in support["local_names"]:
                    for step in _steps(name, fiducial[name], space=space):
                        rows.append([run_id, cell.metadata["cell_id"], list(pair), ell,
                                     name, float(step)])
    if not rows:
        raise ValueError("FD manifest has no observed columns")
    require_complete_coverage(rows, rows)
    return rows


def _physical_scale(cell, means, fiducials):
    """Nonzero norm of the observed fiducial mean in each actual C metric."""
    pairs = [tuple(pair) for pair in cell.metadata["retained_pairs"]]
    full = np.concatenate([means[pair](fiducials) for pair in pairs], axis=1)
    white = [np.linalg.solve(np.linalg.cholesky(cov), row)
             for cov, row in zip(cell.covariance, full)]
    scale = float(np.linalg.norm(white))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"zero or nonfinite whitened physical mean scale in {cell.metadata['cell_id']}")
    return scale


def assembled_fd_certificate(likelihood, ps, selected_cell_ids, expected_manifest,
                             *, expected_support, cache=None, provenance=None, space=None):
    """Measure selected actual D columns against independently built FD means."""
    selected = _selected(likelihood, selected_cell_ids)
    if not expected_manifest:
        raise ValueError("independent FD manifest is required")
    run_ids = {row[0] for row in expected_manifest}
    if len(run_ids) != 1:
        raise ValueError("FD manifest must use one exact run ID")
    run_id = next(iter(run_ids))
    observations = validate_support_schedule(likelihood, expected_support, space=space)
    declared = declare_fd_manifest(likelihood, selected_cell_ids, run_id,
                                   expected_support=expected_support, space=space)
    try:
        require_complete_coverage(declared, expected_manifest)
    except ValueError as exc:
        raise ValueError(f"independent FD manifest mismatch: {exc}") from exc
    results = []
    for cell in selected:
        if not cell.metadata["retained_pairs"]:
            continue
        fiducial = _cell_fiducials(likelihood, cell)
        means = physical_cell_means(likelihood, cell, ps, cache=cache, provenance=provenance, space=space)
        scale = _physical_scale(cell, means, fiducial)
        steps = {name:_steps(name, value, space=space) for name, value in fiducial.items()}
        results.append(compare_cell_derivatives(
            cell, means, fiducial, steps, configuration=run_id, physical_scale=scale, space=space))
    measured = [row for result in results for row in result["measured_coverage"]]
    require_complete_coverage(expected_manifest, measured)
    all_ids = [row["cell_id"] for row in expected_support]
    complete = set(selected_cell_ids) == set(all_ids)
    failures = [failure for result in results for failure in result["failures"]]
    return {"status":"failed" if failures else "passed",
            "scope":"all_cells" if complete else "representative",
            "complete_run_coverage":bool(complete),
            "selected_cell_ids":list(selected_cell_ids),
            "all_cell_ids":all_ids,
            "expected_support_observations":observations,
            "expected_coverage":expected_manifest,
            "measured_coverage":measured,
            "column_results":[row for result in results for row in result["column_results"]],
            "errors":[row for result in results for row in result["errors"]],
            "physical_scales":{result["cell_id"]:result["physical_scale"] for result in results},
            "relative_tolerance":FD_RELATIVE_TOLERANCE,
            "zero_tolerance":FD_ZERO_TOLERANCE,
            "max_error":max((result["max_error"] for result in results), default=0.),
            "failures":failures}


class PhysicalFDCache:
    """Exact-input persistent mean cache; failures remain visible and sticky."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, context, compute):
        key = hashlib.sha256(_canonical(context)).hexdigest()
        path = self.root / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if _canonical(record.get("key")) != _canonical(context):
                raise ValueError(f"physical FD cache context mismatch: {path}")
            if record["status"] == "failed":
                raise ValueError(f"cached physical FD failure: {record['error']}")
            value = np.asarray(record["value"], dtype=float)
            if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"corrupt physical FD cache: {path}")
            self.hits += 1
            return value
        self.misses += 1
        try:
            value = np.asarray(compute(), dtype=float)
            record = {"key":context, "status":"passed", "value":value.tolist()}
        except Exception as exc:
            record = {"key":context, "status":"failed", "error":repr(exc)}
            path.write_bytes(_canonical(record) + b"\n")
            raise
        path.write_bytes(_canonical(record) + b"\n")
        return value


def audit_cell_covariance_neighborhood(likelihood, cell, ps, *, space=None):
    """Check field and retained covariance on all adopted shells and FD points."""
    retained = _checked_pairs(likelihood, cell)
    _check_shells(likelihood, cell)
    if not retained:
        return {"status":"passed", "cell_id":cell.metadata["cell_id"],
                "expected_shells":0, "processed_shells":0, "points":0}
    fiducial = _cell_fiducials(likelihood, cell)
    ibin = cell.metadata["nominal_bin_index"]
    active = tuple(cell.metadata["active_populations"])
    theoretical = [(name, name) for name in active] + [
        (a, b) for i, a in enumerate(active) for b in active[i + 1:]]
    if [tuple(pair) for pair in cell.metadata["theoretical_pairs"]] != theoretical:
        raise ValueError(f"theoretical pair layout mismatch in {cell.metadata['cell_id']}")
    k = np.asarray(likelihood.grid.centers, dtype=float)
    widths = np.asarray(likelihood.grid.widths, dtype=float)
    mu, _ = leggauss(likelihood.config.n_mu)
    nbar = cell.metadata["nbar"]
    if space is not None:
        validate_coordinate_neighborhood(np.array([fiducial[n] for n in space.names]),
            {**likelihood.reference_cosmology,"z":cell.metadata["model_redshift"]},space)
    responses = {pair:make_pair_response(ps, cell.metadata["model_redshift"],
                                         likelihood.reference_cosmology, nbar[pair[0]],
                                         None if pair[0] == pair[1] else nbar[pair[1]],
                                         likelihood.config.k_nl,
                                         ap_nk=likelihood.config.ap_interpolation_nk,
                                         ap_nmu=likelihood.config.ap_interpolation_nmu,
                                      **({} if space is None else {"space":space}))
                 for pair in theoretical}
    all_index = {pair:i for i, pair in enumerate(theoretical)}
    observed = [3*all_index[pair]+i for pair in retained for i in range(3)]
    minima = {"field":float("inf"), "retained_covariance":float("inf")}
    conditions = {"field":0., "retained_covariance":0.}
    residuals = {"field":0., "retained_covariance":0.}
    worst = {"field":None, "retained_covariance":None}
    points = 0

    def record(kind, matrix, context):
        check = _audited_spd(matrix, context)
        if check["min_scaled_eigenvalue"] < minima[kind]:
            minima[kind] = check["min_scaled_eigenvalue"]
            worst[kind] = context
        conditions[kind] = max(conditions[kind], check["scaled_condition"])
        residuals[kind] = max(residuals[kind], check["scaled_residual"])

    def evaluate(coordinates, description):
        nonlocal points
        theta = np.asarray([coordinates[name] for name in (COSMO_NAMES if space is None else space.names)])
        if space is not None:
            validate_physical_point(theta, likelihood.reference_cosmology, space, cell.metadata["model_redshift"])
        signal = {}
        for pair, response in responses.items():
            a, b = pair
            q_a = np.asarray([coordinates[f"bin{ibin:02d}:{a}:{name}"]
                              for name in NUISANCE_NAMES])
            q_b = None if a == b else np.asarray([
                coordinates[f"bin{ibin:02d}:{b}:{name}"] for name in NUISANCE_NAMES])
            cross_names = [f"bin{ibin:02d}:{a}|{b}:{name}" for name in CROSS_STOCH_NAMES]
            q_cross = (None if a == b else np.asarray([
                coordinates[name] if name in coordinates else 0. for name in cross_names]))
            signal[pair] = np.asarray(response.pkmu(k, mu, theta, q_a, q_b, q_cross), dtype=float)
            if signal[pair].shape != (len(k), len(mu)) or not np.all(np.isfinite(signal[pair])):
                shape_good = signal[pair].shape == (len(k), len(mu))
                first = tuple(np.argwhere(~np.isfinite(signal[pair]))[0]) if shape_good else None
                point = (f"k={float(k[first[0]])} mu={float(mu[first[1]])}"
                         if first is not None else "k=unavailable mu=unavailable")
                context = (f"field cell={cell.metadata['cell_id']} {description} "
                           f"pair={pair} {point}")
                diagnostics = {**_point_details(context), "pair":list(pair),
                               "min_scaled_eigenvalue":None,
                               "scaled_condition":None, "scaled_residual":None,
                               "eigenvalue_unavailable_reason":
                               "nonfinite response before field covariance" if shape_good
                               else f"field response shape {signal[pair].shape} differs from {(len(k), len(mu))}"}
                raise CovarianceAuditFailure(diagnostics, "invalid field response")
        for ik, kval in enumerate(k):
            for im, angle in enumerate(mu):
                field = np.array([[signal[tuple(sorted((a, b)))][ik, im] +
                                   (1./nbar[a] if a == b else 0.) for b in active]
                                  for a in active])
                record("field", field,
                       f"field cell={cell.metadata['cell_id']} {description} k={kval} mu={angle}")
        functions = {pair:(lambda _k, _mu, value=value: value)
                     for pair, value in signal.items()}
        covariance = multi_tracer_cov_general(
            list(active), functions, {name:nbar[name] for name in active},
            k, cell.metadata["volume"], widths, n_mu=likelihood.config.n_mu)
        retained_cov = covariance[:, observed][:, :, observed].copy()
        if likelihood.config.case == "I":
            for ia in range(len(retained)):
                for ib in range(len(retained)):
                    if ia != ib:
                        retained_cov[:, 3*ia:3*ia+3, 3*ib:3*ib+3] = 0.
        for ik, kval in enumerate(k):
            record("retained_covariance", retained_cov[ik],
                   f"multipole cell={cell.metadata['cell_id']} {description} k={kval}")
        points += 1

    evaluate(fiducial, "fiducial")
    for name, value in fiducial.items():
        for step in _steps(name, value, space=space):
            for sign in (+1, -1):
                coordinates = dict(fiducial)
                coordinates[name] += sign*step
                evaluate(coordinates, f"column={name} sign={'+' if sign > 0 else '-'} step={step}")
    return {"status":"passed", "cell_id":cell.metadata["cell_id"],
            "expected_shells":len(k), "processed_shells":len(k),
            "response_angular_order":likelihood.config.n_mu,
            "points":points, "min_scaled_eigenvalue":minima,
            "max_scaled_condition":conditions, "max_scaled_residual":residuals,
            "worst_context":worst}
