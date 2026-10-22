"""Exact scope, source identity and retained-evidence checks for ELF cells."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .cosmo import FIDUCIAL
from .elf_survey import elf_cosmology_spec
from .elf_validation_zeros import counterterm_zero_support
from .revision_forecast import RevisionConfig, make_shell_grid
from .revision_validation_physical import expected_support_schedule, _steps
from .surveys import default_survey_group

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "elf-survey-physical-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    """Replace only this run's status after a complete write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode()+b"\n")
    temporary.replace(path)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_digest(request):
    return hashlib.sha256(canonical(request)).hexdigest()


def source_identity(settings, engine=None):
    from .one_loop import make_one_loop_engine
    from scripts.run_revision import collect_provenance
    engine = engine if engine is not None else make_one_loop_engine(nfft=settings["nfft"])
    provenance = collect_provenance(engine, requested_dk=settings["dk"])
    # Import inventories and git HEAD can change without changing a calculation.
    # Every source byte and the executed engine/settings remain bound below.
    keys = ("source_file_hashes", "backend_file_hashes", "background_file_hashes",
            "emulator_input_hashes", "active_emulator_provenance", "survey_input_hashes",
            "package_versions", "interpreter", "precision", "reference_cosmology",
            "thread_settings", "model_flags")
    details = {key: provenance[key] for key in keys}
    details["measurement_programs"] = {str(ROOT/p): file_digest(ROOT/p) for p in (
        "scripts/validate_elf_engine.py", "scripts/validate_elf_survey.py",
        "requirements-elf.lock")}
    details["settings"] = dict(settings)
    details["engine_settings"] = engine.numerical_settings()
    return {"identity_hash": request_digest(details), "details": details}


def make_request(settings, *, cases=("B", "C"), cell_ids=None, source=None):
    """Derive the schedule independently of the worker's assembled spectra."""
    if (set(settings) != {"nfft", "dk", "n_mu"} or not np.isfinite(settings["dk"])
            or settings["dk"] <= 0 or settings["n_mu"] < 8
            or int(settings["n_mu"]) != settings["n_mu"]
            or int(settings["nfft"]) != settings["nfft"] or settings["nfft"] < 16):
        raise ValueError("invalid physical measurement settings")
    cases = list(cases)
    if not cases or len(set(cases)) != len(cases) or set(cases)-{"B", "C"}:
        raise ValueError("expected unique B/C cases")
    cases = sorted(cases)
    cell_ids = list(cell_ids) if cell_ids is not None else None
    if cell_ids is not None and (not cell_ids or len(set(cell_ids)) != len(cell_ids)):
        raise ValueError("expected nonempty unique selected cell IDs")
    configurations, expected = {}, []
    for case in cases:
        config = RevisionConfig(case, cosmology_basis="extended6", dk=settings["dk"],
                                n_mu=settings["n_mu"])
        configurations[case] = asdict(config)
        rows = expected_support_schedule(config, default_survey_group(),
                                         space=elf_cosmology_spec(FIDUCIAL))
        if cell_ids is not None and set(cell_ids)-{row["cell_id"] for row in rows}:
            raise ValueError("selected cell is outside baseline survey support")
        for row in rows:
            if cell_ids is None or row["cell_id"] in cell_ids:
                expected.append({"key": case+":"+row["cell_id"], "case": case, "support": row})
    complete = cases == ["B", "C"] and cell_ids is None
    result = {"schema": SCHEMA, "settings": dict(settings), "cases": cases,
              "cell_ids": cell_ids, "scope": "full_B6_C6" if complete else "selected_cells",
              "configurations": configurations, "expected": expected,
              "source": source if source is not None else source_identity(settings)}
    return json.loads(canonical(result))


def validate_ledger(request, summary, live_source):
    problems = []
    try:
        rebuilt = make_request(request["settings"], cases=request["cases"],
                               cell_ids=request["cell_ids"], source=request["source"])
        if canonical(request) != canonical(rebuilt):
            problems.append("request differs from independent baseline schedule/configuration")
        if canonical(request["source"]) != canonical(live_source):
            problems.append("current sources/settings differ from measurement")
        wanted = [row["key"] for row in rebuilt["expected"]]
        got = [row["key"] for row in summary["workers"]]
        if got != wanted or len(got) != len(set(got)):
            problems.append("missing, duplicate, foreign or reordered worker")
        digest = request_digest(request)
        identity = request["source"]["identity_hash"]
        if (summary.get("status") != "passed" or summary.get("source_changed") is not False
                or summary.get("scope") != request["scope"]
                or summary.get("request_sha256") != digest
                or summary.get("source_identity_hash") != identity
                or summary.get("complete_survey") != (request["scope"] == "full_B6_C6")
                or summary.get("full_forecast_qualification") != "pending"):
            problems.append("incomplete, failed or overstated survey summary")
        for row in summary["workers"]:
            if (row.get("exit_code") != 0 or row.get("status") != "passed"
                    or row.get("source_identity_hash") != identity
                    or row.get("request_sha256") != digest):
                problems.append("failed or unrelated worker: "+row["key"])
    except (KeyError, TypeError, ValueError) as error:
        problems.append("invalid survey ledger: "+str(error))
    return problems


def artifact_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): file_digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_files(directory, recorded):
    directory = Path(directory)
    problems = []
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if not recorded or set(recorded) != actual:
        problems.append("missing or unlisted cell artifact")
    for relative, digest in recorded.items():
        path = directory/relative
        if not path.resolve().is_relative_to(directory.resolve()):
            problems.append("artifact escapes its cell directory: "+relative)
        elif not path.is_file() or file_digest(path) != digest:
            problems.append("missing or changed cell artifact: "+relative)
    return problems


def cell_directory(root, key):
    # The key comes only from the independently reconstructed survey schedule.
    return Path(root)/"cells"/key.replace(":", "__")


def config_from_request(request, case):
    saved = request["configurations"][case]
    return RevisionConfig(**{**saved, "nominal_bins": tuple(tuple(b) for b in saved["nominal_bins"])})


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _check_derivative_report(check, names, pairs, steps, label, cell_id, *, analytic_zero_support=()):
    declared = [[n, list(p), ell] for n, p, ell in analytic_zero_support]
    _require(check.get("analytic_zero_support", []) == declared, "analytic angular support differs")
    analytic = {(n, tuple(p), ell) for n, p, ell in analytic_zero_support}
    coverage = [[label, cell_id, pair, ell, name, step] for name in names
                for step in steps[name] for pair in pairs for ell in (0, 2, 4)]
    _require(check["status"] == "passed" and check["failures"] == []
             and check["expected_coverage"] == check["measured_coverage"] == coverage
             and check["relative_tolerance"] == 1.e-3 and check["zero_tolerance"] == 1.e-6,
             "incomplete or failed derivative coverage")
    columns = check["column_results"]
    _require([row["column"] for row in columns] == names, "missing derivative column")
    for row in columns:
        passes = np.asarray(row["step_errors"]) <= np.asarray(row["step_tolerances"])
        _require(row["steps"] == steps[row["column"]] and row["two_successive_pass"]
                 and len(passes) == 3 and np.any(passes[:-1] & passes[1:]),
                 "column lacks two successive finite-difference steps")
    _require(len(check["errors"]) == len(coverage), "missing pair/multipole comparison")
    errors = {}
    required_all_steps = set()
    for row, expected in zip(check["errors"], coverage):
        got = [row["configuration"], row["cell_id"], row["pair"], row["ell"], row["column"], row["step"]]
        _require(got == expected and np.isfinite(row["error"])
                 and row["limit"] in (1.e-3, 1.e-6), "invalid pair/multipole comparison")
        key = (tuple(row["pair"]), row["ell"], row["column"])
        angular = (row["column"], tuple(row["pair"]), row["ell"]) in analytic
        _require((row["reason"] == "analytic angular zero") == angular
                 and (not angular or row["limit"] == 1.e-6), "analytic zero criterion differs")
        if angular or row["reason"] == "zero block":
            required_all_steps.add(key)
        errors.setdefault(key, []).append(row["error"] <= row["limit"])
    _require(all(len(p) == 3 and ((p[0] and p[1]) or (p[1] and p[2]))
                 for p in errors.values()), "pair/multipole lacks successive passing steps")
    _require(all(all(errors[key]) for key in required_all_steps), "analytic/inactive zero requires all steps")


def verify_cell(directory, expected, request):
    """Check saved scope/coverage and reconstruct elementary array identities."""
    directory = Path(directory)
    read = lambda name: json.loads((directory/name).read_text())
    inputs, measurement, neighborhood, worker = map(read,
        ("inputs.json", "measurement.json", "neighborhood.json", "worker.json"))
    covariance_inputs = read("covariance_inputs.json")
    row, key = expected["support"], expected["key"]
    names, pairs = row["local_names"], row["retained_pairs"]
    config = config_from_request(request, expected["case"])
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    metadata = inputs["metadata"]
    _require(covariance_inputs["configuration"] == inputs["configuration"]
             and covariance_inputs["fiducials"] == inputs["fiducials"]
             and covariance_inputs["engine_settings"] == inputs["engine_settings"]
             and all(metadata.get(k) == v for k, v in covariance_inputs["metadata"].items()),
             "mean and covariance input records differ")
    _require(inputs["configuration"] == request["configurations"][expected["case"]],
             "cell configuration differs from requested calculation")
    for name in ("cell_id", "nominal_bin_index", "nominal_bin", "radial_bin", "region",
                 "model_redshift", "volume", "nbar", "candidate_populations",
                 "active_populations", "retained_pairs", "local_names"):
        _require(metadata[name] == row[name], "cell support differs: "+name)
    active = row["active_populations"]
    theoretical = [[a, a] for a in active]+[[a, b] for i, a in enumerate(active) for b in active[i+1:]]
    _require(metadata["theoretical_pairs"] == theoretical and metadata["retained_observations"] ==
             [[p, ell] for p in pairs for ell in (0, 2, 4)], "incorrect observation layout")
    _require(measurement["metadata"] == metadata and measurement["status"] == "passed"
             and inputs["scope"] == measurement["scope"] == "baseline_survey_physical",
             "failed or unrelated mean measurement")
    _require(worker["status"] == "passed" and worker["key"] == key
             and worker["request_sha256"] == request_digest(request)
             and worker["source_before"] == worker["source_after"] == request["source"]["identity_hash"],
             "failed or unrelated worker receipt")
    _require(inputs["engine_settings"] == request["source"]["details"]["engine_settings"],
             "cell engine resolution or physical model differs")
    _require(set(inputs["fiducials"]) == set(names), "missing local fiducial")
    space = elf_cosmology_spec(FIDUCIAL)
    steps = {n: list(_steps(n, inputs["fiducials"][n], space=space)) for n in names}
    _require(inputs["steps"] == steps, "finite-difference steps changed")
    witness = inputs["analytic_zero_response"]
    zeros = counterterm_zero_support(names, pairs, alphas=witness["alphas"], settings=inputs["engine_settings"])
    _require(witness["support"] == [[n, list(p), ell] for n, p, ell in zeros], "analytic zero witness differs")
    _check_derivative_report(measurement["ad_fd"], names, pairs, steps, key, row["cell_id"],
                             analytic_zero_support=zeros)
    for component in ("intrinsic", "geometry"):
        _check_derivative_report(measurement["h_split_fd"][component], ["h"], pairs,
                                 {"h": steps["h"]}, key+"-"+component+"-h", row["cell_id"])
    _require(measurement["chain_tolerance"] == 1.e-10 and
             measurement["chain_relative_error"] < 1.e-10, "AP chain rule failed")
    nk, nobs, npar = len(grid.centers), 3*len(pairs), len(names)
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        shapes = {"D": (nk, nobs, npar), "C": (nk, nobs, nobs), "mean": (nk, nobs),
                  "F_like": (npar, npar), "D_intrinsic": (nk, nobs, npar),
                  "D_geometry": (nk, nobs, npar), "fiducial": (npar,),
                  "fd_plus": (npar, 3, nk, nobs), "fd_minus": (npar, 3, nk, nobs),
                  "fd_steps": (npar, 3), "fd_derivatives": (npar, 3, nk, nobs)}
        for component in ("intrinsic", "geometry"):
            shapes.update({f"h_{component}_{n}": shape for n, shape in
                (("plus", (1, 3, nk, nobs)), ("minus", (1, 3, nk, nobs)),
                 ("steps", (1, 3)), ("derivatives", (1, 3, nk, nobs)))})
        for name, shape in shapes.items():
            _require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(),
                     "invalid saved array: "+name)
        _require(np.array_equal(arrays["k"], grid.centers) and np.array_equal(arrays["dk"], grid.widths)
                 and np.array_equal(arrays["signed_mu"], np.polynomial.legendre.leggauss(config.n_mu)[0]),
                 "saved Fourier/quadrature grid differs")
        fiducial = np.array([inputs["fiducials"][n] for n in names])
        _require(np.array_equal(arrays["fiducial"], fiducial), "saved fiducial differs")
        _require(np.array_equal(arrays["fd_steps"], [steps[n] for n in names]), "saved steps differ")
        for prefix in ("fd", "h_intrinsic", "h_geometry"):
            direct = (arrays[prefix+"_plus"]-arrays[prefix+"_minus"])/(2*arrays[prefix+"_steps"][:, :, None, None])
            _require(np.array_equal(direct, arrays[prefix+"_derivatives"]), "saved finite difference differs")
        white = np.linalg.solve(np.linalg.cholesky(arrays["C"]), arrays["D"])
        inverse = np.linalg.solve(arrays["C"], np.broadcast_to(np.eye(nobs), arrays["C"].shape))
        scale = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(arrays["C"]), arrays["mean"][..., None]))
        for name, pair, ell in zeros:
            column = names.index(name)
            position = 3*pairs.index(list(pair))+(0, 2, 4).index(ell)
            weight = inverse[:, position, position]
            ad_norm = np.sqrt(np.sum(arrays["D"][:, position, column]**2*weight))
            fd_norm = np.sqrt(np.sum(arrays["fd_derivatives"][column, :, :, position]**2*weight[None, :], axis=1))
            _require(ad_norm <= 1.e-6*scale and np.all(fd_norm <= 1.e-6*scale), "analytic angular zero has physical leakage")
        fisher = np.einsum("kop,koq->pq", white, white, optimize=False)
        fisher_error = float(np.linalg.norm(fisher-arrays["F_like"])/np.linalg.norm(fisher))
        _require(fisher_error < 1.e-10, "saved Fisher sum differs")
    _require(measurement["arrays_sha256"] == file_digest(directory/"arrays.npz"), "mean arrays changed")
    points = [{"column": None, "sign": 0, "step": 0.}]
    coordinates = [fiducial]
    for i, name in enumerate(names):
        for step in steps[name]:
            for sign in (1, -1):
                point = fiducial.copy(); point[i] += sign*step
                coordinates.append(point); points.append({"column": name, "sign": sign, "step": step})
    _require(neighborhood["status"] == "passed" and neighborhood["cell_id"] == row["cell_id"]
             and neighborhood["expected_points"] == neighborhood["processed_points"] == len(points)
             and neighborhood["parameter_names"] == names and neighborhood["theoretical_pairs"] == theoretical
             and neighborhood["shell_count"] == nk and neighborhood["angular_order"] == config.n_mu
             and neighborhood["full_forecast_qualification"] == "pending"
             and len(neighborhood["points"]) == len(points), "incomplete covariance neighborhood")
    _require(np.array_equal(np.load(directory/"neighborhood_coordinates.npy", allow_pickle=False), coordinates),
             "saved covariance displacements differ")
    spectra = np.load(directory/"neighborhood_spectra.npy", mmap_mode="r", allow_pickle=False)
    _require(spectra.shape == (len(points), len(theoretical), nk, config.n_mu), "neighborhood spectra missing")
    for index, (record, description) in enumerate(zip(neighborhood["points"], points)):
        _require(all(record[k] == v for k, v in description.items()) and record["status"] == "passed"
                 and np.isfinite(spectra[index]).all(), "missing/nonfinite covariance point")
        for kind, count in (("field", nk*config.n_mu), ("covariance", nk)):
            audit = record[kind]
            _require(audit["matrix_count"] == count and audit["min_scaled_eigenvalue"] > 0
                     and audit["max_scaled_residual"] <= 1.e-10 and audit["max_scaled_asymmetry"] <= 1.e-10,
                     "failed field/covariance point")
    for name in ("spectra", "coordinates"):
        _require(neighborhood[name+"_sha256"] == file_digest(directory/f"neighborhood_{name}.npy"),
                 "neighborhood arrays changed")
    return {"key": key, "status": "passed", "points": len(points), "fisher_relative_error": fisher_error}


def verify_run(root):
    root = Path(root)
    result = {"status": "failed", "complete_survey": False,
              "full_forecast_qualification": "pending", "problems": [], "cells": []}
    try:
        request = json.loads((root/"request.json").read_text())
        summary = json.loads((root/"summary.json").read_text())
        live = source_identity(request["settings"])
        result["problems"] = validate_ledger(request, summary, live)
        result["scope"] = request["scope"]
        by_key = {row["key"]: row for row in summary["workers"]}
        for expected in request["expected"]:
            key = expected["key"]
            if key not in by_key:
                continue
            try:
                directory = cell_directory(root, key)
                problems = verify_files(directory, by_key[key]["files"])
                _require(not problems, "; ".join(problems))
                result["cells"].append(verify_cell(directory, expected, request))
            except Exception as error:
                result["problems"].append(key+": "+str(error))
        if not result["problems"]:
            result.update(status="passed", complete_survey=request["scope"] == "full_B6_C6")
    except Exception as error:
        result["problems"].append(str(error))
    return result
