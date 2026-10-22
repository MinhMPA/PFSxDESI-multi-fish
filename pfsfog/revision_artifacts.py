"""Versioned, independently checked persistence for revision Fisher results."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from .revision_forecast import (NOMINAL_BINS, CellLikelihood, RevisionConfig,
                                RevisionLikelihood, ShellGrid)
from .revision_theory import COSMO_NAMES, NUISANCE_NAMES
from .eft_params import CROSS_STOCH_NAMES, CROSS_STOCH_PRIOR_SIGMA, get_ctilde


SCHEMA_VERSION = 2
MANDATORY_GATES = (
    "domain", "field_positivity", "retained_covariance_positivity",
    "no_skipped_shells", "symmetry_solve", "ad_fd",
    "ap_identity_chain_resolution", "dk_angular_convergence",
    "once_only_priors", "q13_penalty", "suite_monotonicity", "save_reload",
)
SUMMARY_COLUMNS = ("run_id", "case", "quantity", "z", "fiducial", "sigma_marg",
                   "sigma_nuisance_fixed", "nuisance_penalty", "unit")
CORE_ARRAYS = ("parameter_names", "fiducial", "F_like", "F_prior_cosmo",
               "F_prior_nuisance", "F_total", "covariance", "cov_cosmo_marg",
               "cov_cosmo_nuisance_fixed", "growth_z", "growth_fiducial",
               "growth_jacobian", "growth_cov_marg", "growth_cov_nuisance_fixed",
               "k_edges", "k", "dk")
GATE_FIELDS = {"status", "measured", "tolerance", "evidence_path", "evidence_hash", "reason"}
TRACER_DEFINITIONS = {
    "b1_sigma8":{"definition":"b1 * sigma8(theta,z)", "unit":""},
    "b2_sigma8sq":{"definition":"b2 * sigma8(theta,z)^2", "unit":""},
    "bG2_sigma8sq":{"definition":"bG2 * sigma8(theta,z)^2", "unit":""},
    "bGamma3":{"definition":"G = A * bGamma3_backend", "unit":""},
    "c0":{"definition":"Q0 = A * C0", "unit":"(Mpc/h)^2"},
    "c2":{"definition":"Q2 = A * C2", "unit":"(Mpc/h)^2"},
    "c4":{"definition":"Q4 = A * C4", "unit":"(Mpc/h)^2"},
    "c_tilde":{"definition":"T = A^2 * c_tilde_backend", "unit":"(Mpc/h)^4"},
    "Pshot":{"definition":"auto shot-noise amplitude normalized by 1/nbar", "unit":""},
    "a0":{"definition":"auto isotropic (k/k_nl)^2 stochastic amplitude normalized by 1/nbar", "unit":""},
    "a2":{"definition":"auto mu^2 (k/k_nl)^2 stochastic amplitude normalized by 1/nbar", "unit":""},
}


def _nuisance_convention(result):
    factor = float(result.nuisance_factor)
    config = result.likelihood.config
    desi_ctilde = float(get_ctilde("DESI-ELG"))
    pfs_ctilde = desi_ctilde * config.r_sigma_v**4
    for record in result.parameters:
        if record["role"] == "tracer" and record["coordinate"] == "c_tilde":
            tracer = record["tracers"][0]
            expected = pfs_ctilde if tracer == "PFS-ELG" else float(get_ctilde(tracer))
            if not np.isclose(record["fiducial"], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"fiducial c_tilde assumption differs for {tracer}")
        if record["role"] == "cross":
            expected = CROSS_STOCH_PRIOR_SIGMA[record["coordinate"]] * factor
            if record["fiducial"] != 0. or not np.isclose(record["prior_sigma"], expected):
                raise ValueError(f"cross stochastic fiducial/prior differs for {record['name']}")
    return {
        "tracer_coordinates":list(NUISANCE_NAMES),
        "tracer_definitions":TRACER_DEFINITIONS,
        "amplitude_factor":"A = (sigma8(theta,z) / sigma8(reference,z))^2",
        "backend_counterterms":{
            "c0":"C0 - f*C2/3 + 3*f^2*C4/35",
            "c2":"C2 - 6*f*C4/7", "c4":"C4",
            "cfog":"T/A^2"},
        "coevolution_centers":{
            "b2":"0.412 - 2.143*b1 + 0.929*b1^2 + 0.008*b1^3",
            "bG2":"-2*(b1-1)/7", "bGamma3":"23*(b1-1)/42",
            "role":"fiducial centers; these coordinates are fitted, not fixed"},
        "fixed_coevolution_centers":True,
        "stochastic_ap_normalization":"auto and cross terms use true k,mu and "
                                      "the AP volume Jacobian; auto is divided by nbar and "
                                      "cross by sqrt(nbar_a*nbar_b)",
        "cross_coordinates":list(CROSS_STOCH_NAMES),
        "cross_stochastic":{
            "names":list(CROSS_STOCH_NAMES), "unit":"dimensionless",
            "definition":"[Pshot_cross + a2_cross*(k_true*mu_true/k_nl)^2] "
                         "/ [sqrt(nbar_a*nbar_b)*AP_jacobian]",
            "fiducial":{"Pshot_cross":0., "a2_cross":0.},
            "base_prior_sigma":dict(CROSS_STOCH_PRIOR_SIGMA),
            "effective_prior_sigma":{name:width*factor for name,width in
                                     CROSS_STOCH_PRIOR_SIGMA.items()},
            "nuisance_factor":factor},
        "active_c1":False,
        "pfs_sigma_v_ratio":config.r_sigma_v,
        "fiducial_ctilde":{"DESI":desi_ctilde, "PFS":pfs_ctilde,
                           "unit":"(Mpc/h)^4",
                           "qualification":"forecast assumption: PFS=DESI*r_sigma_v^4; "
                                           "not a PFS calibration"},
        "fiducial_cross_stochasticity":0.,
    }


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _strict_json(data):
    def reject(token):
        raise ValueError(f"nonfinite JSON number: {token}")
    return json.loads(data, parse_constant=reject)


def _reload_measurement():
    return {"passed":True,
            "method":"independent reload of checksums, labels, cell/bin and prior sums, "
                     "scaled solves, growth covariance, and summary at the declared gates"}


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity_provenance(provenance):
    return {key:value for key, value in provenance.items()
            if key not in ("command", "runtime", "status", "log")}


def _check_external_gates(validation, *, run_id, require_passed,
                          configuration=None, provenance=None):
    """Bind passed outer records to one exact-run, hashed evidence journal."""
    journals = {}
    for gate in MANDATORY_GATES:
        if gate == "save_reload":
            continue
        record = validation.get(gate)
        if record is None:
            if require_passed:
                raise ValueError(f"mandatory gate {gate} is missing")
            continue
        if not isinstance(record, dict) or not GATE_FIELDS.issubset(record):
            raise ValueError(f"mandatory gate {gate} is malformed")
        status = record["status"]
        if status not in ("pending", "failed", "passed"):
            raise ValueError(f"mandatory gate {gate} status is invalid")
        if require_passed and status != "passed":
            raise ValueError(f"mandatory gate {gate} is incomplete")
        if status == "passed":
            path = record["evidence_path"]
            if not path or not record["evidence_hash"]:
                raise ValueError(f"mandatory gate {gate} lacks evidence path/hash")
            key = (str(Path(path).resolve()), record["evidence_hash"])
            if key not in journals:
                if not Path(path).is_file():
                    raise ValueError(f"mandatory gate {gate} evidence hash mismatch")
                raw = Path(path).read_bytes()
                if hashlib.sha256(raw).hexdigest() != record["evidence_hash"]:
                    raise ValueError(f"mandatory gate {gate} evidence hash mismatch")
                document = _strict_json(raw.decode())
                if not isinstance(document, dict):
                    raise ValueError(f"mandatory gate {gate} evidence journal is malformed")
                if document.get("run_id") != run_id:
                    raise ValueError(f"mandatory gate {gate} evidence run identity mismatch")
                journal_config = document.get("configuration")
                journal_provenance = document.get("provenance")
                if not isinstance(journal_config, dict) or not isinstance(journal_provenance, dict):
                    raise ValueError(f"mandatory gate {gate} evidence identity is malformed")
                if {"case", "kmax", "nuisance_factor", "selection"}.issubset(journal_config):
                    if run_identity(journal_config, journal_provenance) != run_id:
                        raise ValueError(f"mandatory gate {gate} evidence run identity mismatch")
                if configuration is not None and _canonical(journal_config) != _canonical(configuration):
                    raise ValueError(f"mandatory gate {gate} evidence configuration mismatch")
                if (provenance is not None
                        and _canonical(_identity_provenance(journal_provenance))
                        != _canonical(_identity_provenance(provenance))):
                    raise ValueError(f"mandatory gate {gate} evidence provenance mismatch")
                journals[key] = document
            inner = journals[key].get("gates")
            inner = inner.get(gate) if isinstance(inner, dict) else None
            if not isinstance(inner, dict) or any(
                    field not in inner or _canonical(inner[field]) != _canonical(record[field])
                    for field in ("status", "measured", "tolerance", "reason")):
                raise ValueError(f"mandatory gate {gate} evidence journal disagrees with outer record")


def _configuration(config, factor, *, requested_dk=None, model_flags=None):
    data = _jsonable(config)
    flags = model_flags or {}
    data.update({"nuisance_factor": float(factor),
                 "selection": "exclude_elg1" if config.exclude_elg1 else "all",
                 "requested_dk": float(config.dk if requested_dk is None else requested_dk),
                 "adopted_dk": float(config.dk),
                 "multipoles": [0, 2, 4], "mean_n_mu": config.n_mu,
                 "covariance_n_mu": config.n_mu,
                 "ap_interpolation_nk": config.ap_interpolation_nk,
                 "ap_refinement_nk": list(config.ap_refinement_nk),
                 "ap_interpolation_nmu": config.ap_interpolation_nmu,
                 "fft_kmin": flags.get("fft_kmin"),
                 "fft_kmax": flags.get("fft_kmax"),
                 "fft_n": flags.get("fft_n")})
    if flags.get("engine") == "elf.EPT":
        # These ps-only controls are not part of an ELF likelihood. Omitting
        # them also lets old dataclass readers use their harmless defaults.
        for key in ("ap_interpolation_nk", "ap_refinement_nk", "ap_interpolation_nmu"):
            data.pop(key, None)
        data["ap_method"] = flags["ap_method"]
    return data


def run_identity(configuration, provenance):
    """Content identity; omit only run bookkeeping and command text."""
    config = _jsonable(configuration)
    identity_provenance = _jsonable(_identity_provenance(provenance))
    content = {"configuration": config, "provenance": identity_provenance,
               "parameter_convention": {"cosmo": list(COSMO_NAMES),
                                        "tracer": list(NUISANCE_NAMES)}}
    digest = hashlib.sha256(_canonical(content)).hexdigest()[:12]
    cut = f"{float(config['kmax']):.2f}".replace(".", "")
    factor = f"{float(config['nuisance_factor']):g}".replace(".", "d")
    return f"{config['case']}-k{cut}-p{factor}-{config['selection']}-{digest}"


def likelihood_identity(configuration, provenance):
    config = dict(_jsonable(configuration))
    config.pop("nuisance_factor", None)
    return hashlib.sha256(_canonical({"configuration": config,
                                     "provenance": _jsonable(_identity_provenance(provenance)),
                                     "parameter_convention": {"cosmo": list(COSMO_NAMES),
                                                              "tracer": list(NUISANCE_NAMES)}})).hexdigest()


def _check_close(actual, expected, context):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError(f"{context} shape or finiteness mismatch")
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{context} mismatch")


def _inverse(matrix, context):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context} shape mismatch")
    _check_close(matrix, matrix.T, f"{context} symmetry")
    if np.any(np.diag(matrix) <= 0):
        raise ValueError(f"{context} nonpositive diagonal")
    scale = np.sqrt(np.diag(matrix))
    normalized = matrix / scale[:, None] / scale[None, :]
    try:
        chol = np.linalg.cholesky(normalized)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context} is not positive definite") from exc
    inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(matrix))))
    residual = np.max(np.abs(normalized @ inverse - np.eye(len(matrix))))
    if not np.isfinite(residual) or residual > 1e-10:
        raise ValueError(f"{context} scaled solve residual {residual} exceeds 1e-10")
    return inverse / scale[:, None] / scale[None, :]


def _summary_rows(run_id, case, parameters, arrays):
    rows = []
    for i in range(3):
        marginal = float(np.sqrt(arrays["cov_cosmo_marg"][i, i]))
        fixed = float(np.sqrt(arrays["cov_cosmo_nuisance_fixed"][i, i]))
        rows.append(dict(run_id=run_id, case=case, quantity=parameters[i]["name"], z="",
                         fiducial=float(arrays["fiducial"][i]), sigma_marg=marginal,
                         sigma_nuisance_fixed=fixed, nuisance_penalty=marginal/fixed,
                         unit=parameters[i]["unit"]))
    for i, z in enumerate(arrays["growth_z"]):
        marginal = float(np.sqrt(arrays["growth_cov_marg"][i, i]))
        fixed = float(np.sqrt(arrays["growth_cov_nuisance_fixed"][i, i]))
        rows.append(dict(run_id=run_id, case=case, quantity="fsigma8", z=float(z),
                         fiducial=float(arrays["growth_fiducial"][i]), sigma_marg=marginal,
                         sigma_nuisance_fixed=fixed, nuisance_penalty=marginal/fixed,
                         unit=""))
    return rows


def _arrays(result):
    like = result.likelihood
    arrays = {"parameter_names": np.asarray(result.names, dtype=str),
              "fiducial": np.asarray(result.fiducial, dtype=np.float64)}
    for key in ("F_like", "F_prior_cosmo", "F_prior_nuisance", "F_total",
                "covariance", "cov_cosmo_marg", "cov_cosmo_nuisance_fixed",
                "growth_z", "growth_fiducial", "growth_jacobian",
                "growth_cov_marg", "growth_cov_nuisance_fixed"):
        arrays[key] = np.asarray(getattr(result, key), dtype=np.float64)
    arrays.update(k_edges=np.asarray(result.grid.edges, dtype=np.float64),
                  k=np.asarray(result.grid.centers, dtype=np.float64),
                  dk=np.asarray(result.grid.widths, dtype=np.float64))
    for i, matrix in enumerate(like.bin_F_like):
        arrays[f"bin_F_like__{i:02d}"] = np.asarray(matrix, dtype=np.float64)
    for i, cell in enumerate(like.cells):
        arrays[f"cell_F_like__{i:03d}"] = np.asarray(cell.F_like, dtype=np.float64)
        arrays[f"cell_parameter_names__{i:03d}"] = np.asarray(cell.names, dtype=str)
        arrays[f"cell_fiducial_mean__{i:03d}"] = np.asarray(cell.fiducial_mean,
                                                            dtype=np.float64)
    return arrays


def _validate_parameter_metadata(parameters, cells):
    """Bind manifest-only annotations to labels and measured cell membership."""
    for record in parameters:
        name = record["name"]
        if name in COSMO_NAMES:
            expected = {"role":"cosmo", "nominal_bin":None, "tracers":[],
                        "coordinate":name, "unit":"eV" if name == "Mnu" else ""}
        else:
            match = re.fullmatch(r"bin(\d{2}):([^:]+):([^:]+)", name)
            if match is None:
                raise ValueError(f"parameter metadata label invalid: {name}")
            ibin, population, coordinate = match.groups()
            ibin = int(ibin)
            if ibin >= len(NOMINAL_BINS):
                raise ValueError(f"parameter metadata bin invalid: {name}")
            if "|" in population:
                tracers = population.split("|")
                if len(tracers) != 2 or tracers[0] >= tracers[1] or coordinate not in CROSS_STOCH_NAMES:
                    raise ValueError(f"parameter metadata cross label invalid: {name}")
                role, unit = "cross", ""
                present = any(cell["nominal_bin_index"] == ibin
                              and name in cell["local_labels"]
                              and tracers in cell["retained_pairs"] for cell in cells)
            else:
                tracers = [population]
                if coordinate not in NUISANCE_NAMES:
                    raise ValueError(f"parameter metadata tracer label invalid: {name}")
                role, unit = "tracer", TRACER_DEFINITIONS[coordinate]["unit"]
                present = any(cell["nominal_bin_index"] == ibin
                              and name in cell["local_labels"]
                              and population in cell["active_populations"] for cell in cells)
            if not present:
                raise ValueError(f"parameter metadata has no active cell: {name}")
            expected = {"role":role, "nominal_bin":ibin, "tracers":tracers,
                        "coordinate":coordinate, "unit":unit}
        if any(record.get(key, object()) != value for key, value in expected.items()):
            raise ValueError(f"parameter metadata mismatch: {name}")


def _validate_payload(manifest, arrays, rows):
    parameters = manifest["parameters"]
    names = [record["name"] for record in parameters]
    p = len(names)
    zcount = len(manifest["nominal_bins"])
    if zcount != 8 or names[:3] != list(COSMO_NAMES) or len(set(names)) != p:
        raise ValueError("parameter or nominal-bin labels invalid")
    if manifest["model"]["do_irres"] is not False:
        raise ValueError("do_irres model flag mismatch")
    for key, value in manifest["provenance"]["model_flags"].items():
        actual = manifest["model"].get("backend" if key == "backend_class" else key)
        if actual != value:
            raise ValueError(f"model {key} flag mismatch")
    for key in ("ap_interpolation_nk", "ap_refinement_nk",
                "ap_interpolation_nmu", "fft_kmin",
                "fft_kmax", "fft_n"):
        if manifest["configuration"].get(key) != manifest["model"].get(key):
            raise ValueError(f"configuration {key} resolution mismatch")
    if len(manifest["configuration"]["nominal_bins"]) != 8:
        raise ValueError("configuration nominal-bin count mismatch")
    if tuple(tuple(pair) for pair in manifest["configuration"]["nominal_bins"]) != NOMINAL_BINS:
        raise ValueError("configuration differs from frozen nominal bins")
    for i, nominal in enumerate(manifest["nominal_bins"]):
        interval = manifest["configuration"]["nominal_bins"][i]
        if nominal["index"] != i or nominal["interval"] != interval:
            raise ValueError("nominal-bin interval mismatch")
        _check_close(nominal["model_redshift"], sum(interval)/2, "nominal-bin redshift")
    required = set(CORE_ARRAYS)
    required.update(f"bin_F_like__{i:02d}" for i in range(8))
    required.update(f"cell_F_like__{i:03d}" for i in range(len(manifest["cells"])))
    required.update(f"cell_parameter_names__{i:03d}" for i in range(len(manifest["cells"])))
    required.update(f"cell_fiducial_mean__{i:03d}" for i in range(len(manifest["cells"])))
    if set(arrays) != required or arrays["parameter_names"].tolist() != names:
        raise ValueError("array key or parameter label mismatch")
    for key, array in arrays.items():
        if key.startswith("cell_parameter_names") or key == "parameter_names":
            if array.dtype.kind != "U":
                raise ValueError(f"{key} labels must be Unicode")
        elif array.dtype != np.float64 or not np.all(np.isfinite(array)):
            raise ValueError(f"{key} must be finite float64")
    expected_shapes = {"fiducial":(p,), "F_like":(p,p), "F_prior_cosmo":(p,p),
                       "F_prior_nuisance":(p,p), "F_total":(p,p),
                       "covariance":(p,p), "cov_cosmo_marg":(3,3),
                       "cov_cosmo_nuisance_fixed":(3,3), "growth_z":(8,),
                       "growth_fiducial":(8,), "growth_jacobian":(8,3),
                       "growth_cov_marg":(8,8), "growth_cov_nuisance_fixed":(8,8)}
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} shape mismatch")
    _check_close(arrays["fiducial"], [record["fiducial"] for record in parameters], "fiducial")
    edges, centers, widths = (arrays[key] for key in ("k_edges", "k", "dk"))
    if edges.ndim != 1 or centers.shape != (len(edges)-1,) or widths.shape != centers.shape:
        raise ValueError("k grid shape mismatch")
    _check_close(widths, np.diff(edges), "dk")
    _check_close(centers, (edges[:-1]+edges[1:])/2, "k centers")
    _check_close(edges[[0,-1]], [manifest["configuration"]["kmin"],
                                  manifest["configuration"]["kmax"]], "k edges")
    _check_close(arrays["growth_z"], [item["model_redshift"] for item in manifest["nominal_bins"]], "growth_z")
    bins = np.zeros((p,p))
    for i in range(8):
        matrix = arrays[f"bin_F_like__{i:02d}"]
        if matrix.shape != (p,p):
            raise ValueError("bin matrix shape mismatch")
        bins += matrix
    _check_close(bins, arrays["F_like"], "bin Fisher sum")
    embedded = [np.zeros((p,p)) for _ in range(8)]
    index = {name:i for i,name in enumerate(names)}
    seen_cells = set()
    expected_shells = processed_shells = 0
    for i, cell in enumerate(manifest["cells"]):
        local = arrays[f"cell_parameter_names__{i:03d}"].tolist()
        matrix = arrays[f"cell_F_like__{i:03d}"]
        mean = arrays[f"cell_fiducial_mean__{i:03d}"]
        if cell["array_contribution_key"] != f"cell_F_like__{i:03d}" or local != cell["local_labels"]:
            raise ValueError("cell label or contribution key mismatch")
        if matrix.shape != (len(local),len(local)) or len(local) != len(set(local)) or any(n not in index for n in local):
            raise ValueError("cell matrix shape or label mismatch")
        if cell["cell_id"] in seen_cells or not 0 <= cell["nominal_bin_index"] < 8:
            raise ValueError("cell identity or nominal-bin index mismatch")
        seen_cells.add(cell["cell_id"])
        if list(cell["nominal_bin"]) != manifest["nominal_bins"][cell["nominal_bin_index"]]["interval"]:
            raise ValueError("cell nominal-bin interval mismatch")
        v = cell["validation"]
        if cell["expected_shells"] != v["expected_shells"] or cell["processed_shells"] != v["processed_shells"]:
            raise ValueError("cell shell count metadata mismatch")
        expected_shells += v["expected_shells"]
        processed_shells += v["processed_shells"]
        if v["expected_shells"] != v["processed_shells"]:
            raise ValueError("skipped cell shells")
        if not local and (matrix.shape != (0,0) or not v.get("reason") or v["expected_shells"] != 0):
            raise ValueError("empty cell metadata mismatch")
        if local:
            if mean.shape != (len(centers), 3*len(cell["retained_pairs"])):
                raise ValueError("active cell fiducial mean shape mismatch")
            if v["expected_shells"] != len(centers):
                raise ValueError("active cell shell inventory mismatch")
            for key in ("field_min_scaled_eigenvalue", "covariance_min_scaled_eigenvalue"):
                value = v.get(key)
                if value is None or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"cell {key} positivity mismatch")
        elif any(v.get(key) is not None for key in
                 ("field_min_scaled_eigenvalue", "covariance_min_scaled_eigenvalue")):
            raise ValueError("empty cell has invented positivity evidence")
        elif mean.shape != (0, 0):
            raise ValueError("empty cell has a fiducial mean")
        dest = [index[n] for n in local]
        embedded[cell["nominal_bin_index"]][np.ix_(dest,dest)] += matrix
    assembly = manifest.get("assembly_validation")
    if assembly is not None and (assembly["expected_shells"] != expected_shells
                                 or assembly["processed_shells"] != processed_shells):
        raise ValueError("global shell inventory mismatch")
    _validate_parameter_metadata(parameters, manifest["cells"])
    for i, matrix in enumerate(embedded):
        _check_close(matrix, arrays[f"bin_F_like__{i:02d}"], f"cell embedding bin {i}")
    _check_close(arrays["F_total"], arrays["F_like"]+arrays["F_prior_cosmo"]+arrays["F_prior_nuisance"], "F_total")
    expected_c = np.zeros((p,p))
    expected_c[1,1], expected_c[2,2] = 1/25., 1.
    _check_close(arrays["F_prior_cosmo"], expected_c, "cosmology prior")
    expected_n = np.zeros((p,p))
    for i, record in enumerate(parameters[3:], 3):
        sigma = record["prior_sigma"]
        if sigma is not None:
            expected_n[i,i] = 1/float(sigma)**2
    _check_close(arrays["F_prior_nuisance"], expected_n, "nuisance prior")
    inverse = _inverse(arrays["F_total"], "F_total")
    _check_close(arrays["covariance"], inverse, "covariance")
    _check_close(arrays["cov_cosmo_marg"], inverse[:3,:3], "cov_cosmo_marg")
    _check_close(arrays["cov_cosmo_nuisance_fixed"], _inverse(arrays["F_total"][:3,:3], "cosmology block"),
                 "cov_cosmo_nuisance_fixed")
    jac = arrays["growth_jacobian"]
    _check_close(arrays["growth_cov_marg"], jac @ arrays["cov_cosmo_marg"] @ jac.T, "growth_cov_marg")
    _check_close(arrays["growth_cov_nuisance_fixed"], jac @ arrays["cov_cosmo_nuisance_fixed"] @ jac.T,
                 "growth_cov_nuisance_fixed")
    if len(rows) != 11:
        raise ValueError("summary row count mismatch")
    expected_rows = _summary_rows(manifest["run_id"], manifest["configuration"]["case"], parameters, arrays)
    for actual, expected in zip(rows, expected_rows):
        for key in SUMMARY_COLUMNS:
            if key in ("z", "fiducial", "sigma_marg", "sigma_nuisance_fixed", "nuisance_penalty") and expected[key] != "":
                _check_close(float(actual[key]), expected[key], f"summary {key}")
            elif str(actual[key]) != str(expected[key]):
                raise ValueError(f"summary {key} mismatch")


def _read_payload(path, *, require_complete):
    path = Path(path)
    manifest = _strict_json((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema version mismatch")
    if manifest["run_id"] != path.name or manifest["run_id"] != run_identity(manifest["configuration"], manifest["provenance"]):
        raise ValueError("run identity mismatch")
    if require_complete and manifest["status"] != "complete":
        raise ValueError("run is not complete")
    if manifest["status"] == "failed":
        raise ValueError("failed run has no uncertainty payload")
    for filename in ("matrices.npz", "summary.csv", "validation.json", "run.log"):
        if filename not in manifest["artifacts"] or _hash_file(path / filename) != manifest["artifacts"][filename]:
            raise ValueError(f"{filename} checksum mismatch")
    validation = _strict_json((path / "validation.json").read_text())
    _check_external_gates(validation, run_id=manifest["run_id"],
                          configuration=manifest["configuration"],
                          provenance=manifest["provenance"], require_passed=require_complete)
    for gate in MANDATORY_GATES:
        record = validation.get(gate)
        if not isinstance(record, dict) or not GATE_FIELDS.issubset(record):
            raise ValueError(f"mandatory gate {gate} is malformed")
        if require_complete and record["status"] != "passed":
            raise ValueError(f"mandatory gate {gate} is incomplete")
    with np.load(path / "matrices.npz", allow_pickle=False) as npz:
        arrays = {key:npz[key] for key in npz.files}
    with (path / "summary.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
            raise ValueError("summary columns mismatch")
        rows = list(reader)
    _validate_payload(manifest, arrays, rows)
    return manifest, arrays, rows, validation


def load_revision_artifacts(path):
    """Reject incomplete, corrupted, or scientifically inconsistent runs."""
    return _read_payload(path, require_complete=True)


def load_cached_revision_likelihood(path, expected_identity, provenance):
    """Recover a verified factor-one likelihood for an exact prior sweep."""
    manifest, arrays, _, _ = _read_payload(path, require_complete=False)
    if manifest["status"] != "building" and manifest["status"] != "complete":
        raise ValueError("likelihood cache must be building or complete")
    if manifest["configuration"]["nuisance_factor"] != 1.:
        raise ValueError("likelihood cache must use factor one")
    stored = likelihood_identity(manifest["configuration"], manifest["provenance"])
    requested = likelihood_identity(manifest["configuration"], provenance)
    if stored != expected_identity or requested != expected_identity:
        raise ValueError("likelihood cache identity mismatch")
    data = {key:value for key,value in manifest["configuration"].items()
            if key in RevisionConfig.__dataclass_fields__}
    data["nominal_bins"] = tuple(tuple(pair) for pair in data["nominal_bins"])
    config = RevisionConfig(**data)
    cells = []
    for i, record in enumerate(manifest["cells"]):
        metadata = {key:value for key,value in record.items()
                    if key not in ("array_contribution_key", "local_labels", "validation",
                                   "expected_shells", "processed_shells")}
        cells.append(CellLikelihood(arrays[f"cell_F_like__{i:03d}"],
                                    arrays[f"cell_parameter_names__{i:03d}"].tolist(),
                                    metadata, record["validation"],
                                    np.zeros((0,0,0)), np.zeros((0,0,0)),
                                    arrays[f"cell_fiducial_mean__{i:03d}"]))
    grid = ShellGrid(arrays["k_edges"], arrays["k"], arrays["dk"])
    return RevisionLikelihood(config, arrays["F_like"],
                              arrays["parameter_names"].tolist(), arrays["fiducial"],
                              manifest["parameters"], cells,
                              [arrays[f"bin_F_like__{i:02d}"] for i in range(8)],
                              grid, manifest["cosmology"]["reference_cosmology"],
                              manifest["model"], manifest["assembly_validation"])


def finalize_revision_artifacts(path, gate_bundle):
    """Attach exact-run evidence to a building artifact without recomputing it."""
    path = Path(path)
    original_manifest = (path / "manifest.json").read_bytes()
    original_validation = (path / "validation.json").read_bytes()
    manifest, _, _, validation = _read_payload(path, require_complete=False)
    if manifest["status"] != "building":
        raise ValueError("only a building artifact can be finalized")
    external = gate_bundle.get("runs", {}).get(manifest["run_id"])
    if external is None:
        raise ValueError("gate bundle has no exact run identity")
    _check_external_gates(external, run_id=manifest["run_id"],
                          configuration=manifest["configuration"],
                          provenance=manifest["provenance"], require_passed=True)
    updated = dict(validation)
    updated.update({gate:external[gate] for gate in MANDATORY_GATES if gate != "save_reload"})
    updated["save_reload"] = {"status":"passed", "measured":_reload_measurement(),
                              "tolerance":1e-10, "evidence_path":str(path),
                              "evidence_hash":manifest["artifacts"]["matrices.npz"],
                              "reason":None}
    next_validation = path / ".validation.next.json"
    next_validation.write_text(json.dumps(_jsonable(updated), indent=2, allow_nan=False)+"\n")
    try:
        os.replace(next_validation, path / "validation.json")
        manifest["artifacts"]["validation.json"] = _hash_file(path / "validation.json")
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        _read_payload(path, require_complete=False)
        manifest["status"] = "complete"
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        load_revision_artifacts(path)
    except Exception:
        (path / "validation.json").write_bytes(original_validation)
        (path / "manifest.json").write_bytes(original_manifest)
        raise
    finally:
        next_validation.unlink(missing_ok=True)
    return path


def write_revision_artifacts(result, root, provenance, validation):
    """Persist a result or failed construction without overwriting any run."""
    start = time.time()
    provenance = dict(provenance)
    config = result.likelihood.config if result is not None else provenance.pop("configuration")
    if isinstance(config, dict):
        config = RevisionConfig(**{k:v for k,v in config.items()
                                   if k in RevisionConfig.__dataclass_fields__})
    convention = None
    if result is not None:
        _canonical(validation)  # Reject nonfinite gate measurements before creating a directory.
        if len(config.nominal_bins) != 8:
            raise ValueError("production artifacts require eight nominal bins")
        if tuple(tuple(pair) for pair in config.nominal_bins) != NOMINAL_BINS:
            raise ValueError("production artifacts require frozen nominal bins")
        if result.likelihood.model.get("do_irres") is not False:
            raise ValueError("revision backend requires do_irres=False")
        flags = provenance.get("model_flags")
        required_flags = {"backend_class", "do_irres", "subtract_k0_limit",
                          "fft_kmin", "fft_kmax", "fft_n", "ap_interpolation_nk",
                          "ap_refinement_nk",
                          "ap_interpolation_nmu", "fixed_observed_geometry",
                          "fixed_covariance", "covariance_derivative_trace"}
        if isinstance(flags, dict) and flags.get("engine") == "elf.EPT":
            required_flags |= {"engine", "method", "counterterm_base", "subtract_k0_const",
                               "pad_mode", "fftlog_settings", "ap_method", "ap_query_bounds",
                               "multipole_default_nmu"}
        if not isinstance(flags, dict) or set(flags) != required_flags:
            raise ValueError("provenance requires actual model_flags including FFT/AP settings")
        if flags["do_irres"] is not False:
            raise ValueError("provenance do_irres does not match revision backend")
        for source in required_flags:
            target = "backend" if source == "backend_class" else source
            actual = result.likelihood.model.get(target)
            if target not in result.likelihood.model or actual != flags[source]:
                raise ValueError(f"provenance model flag {source} differs from likelihood")
        convention = _nuisance_convention(result)
    factor = result.nuisance_factor if result is not None else config.nuisance_factor
    configuration = _configuration(config, factor,
                                   requested_dk=provenance.get("requested_dk", config.dk),
                                   model_flags=provenance.get("model_flags"))
    run_id = run_identity(configuration, provenance)
    if result is not None:
        _check_external_gates(validation, run_id=run_id, configuration=configuration,
                              provenance=provenance, require_passed=False)
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {"schema_version":SCHEMA_VERSION, "run_id":run_id,
                "status":"failed" if result is None else "building",
                "configuration":configuration, "provenance":_jsonable(provenance),
                "runtime":{"start_utc":now, "end_utc":None, "elapsed_seconds":None,
                           "qualification":provenance.get("qualification", "unknown")},
                "artifacts":{}}
    if result is None:
        manifest.update(cosmology=None, nuisance_convention=None, model=None,
                        parameters=[], nominal_bins=[], cells=[])
        details = _jsonable(validation)
        (path / "validation.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
        (path / "run.log").write_text(json.dumps(details.get("failure", details), indent=2)+"\n")
    else:
        like = result.likelihood
        manifest.update(
            cosmology={"native_names":list(COSMO_NAMES), "fiducial":_jsonable(result.fiducial[:3]),
                       "fixed_emulator_inputs":{k:v for k,v in like.reference_cosmology.items()
                                                if k not in ("ln10_10_As", "mnu", "Omega_m")},
                       "reference_cosmology":_jsonable(like.reference_cosmology),
                       "matter_field":"total matter including massive neutrinos",
                       "background_growth":"ps_1loop_jax background approximation",
                       "derivative_convention":"Mnu at fixed total Omegam",
                       "external_amplitude_prior":False, "headline_z":1.1},
            nuisance_convention=convention,
            model=_jsonable(like.model),
            assembly_validation=_jsonable(like.validation),
            parameters=_jsonable(result.parameters),
            nominal_bins=[{"index":i, "interval":list(pair),
                           "model_redshift":(pair[0]+pair[1])/2}
                          for i,pair in enumerate(config.nominal_bins)],
            cells=[{**_jsonable(cell.metadata), "expected_shells":cell.validation["expected_shells"],
                    "processed_shells":cell.validation["processed_shells"],
                    "array_contribution_key":f"cell_F_like__{i:03d}",
                    "local_labels":list(cell.names), "validation":_jsonable(cell.validation)}
                   for i,cell in enumerate(like.cells)])
        arrays = _arrays(result)
        np.savez(path / "matrices.npz", **arrays)
        with (path / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(_summary_rows(run_id, config.case, result.parameters, arrays))
        details = {key:_jsonable(value) for key,value in validation.items()}
        for gate in MANDATORY_GATES:
            details.setdefault(gate, {"status":"pending", "measured":None,
                                      "tolerance":None, "evidence_path":None,
                                      "evidence_hash":None, "reason":"not supplied"})
        details["assembly"] = _jsonable({"likelihood":like.validation, "prior":result.validation})
        (path / "validation.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
        (path / "run.log").write_text(f"{now} created {run_id}; scope={'production' if len(config.nominal_bins)==8 else 'smoke'}\n")
    manifest["runtime"]["end_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime"]["elapsed_seconds"] = time.time()-start
    for filename in ("matrices.npz", "summary.csv", "validation.json", "run.log"):
        if (path / filename).exists():
            manifest["artifacts"][filename] = _hash_file(path / filename)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    if result is not None:
        try:
            _read_payload(path, require_complete=False)
            details = json.loads((path / "validation.json").read_text())
            details["save_reload"] = {"status":"passed", "measured":_reload_measurement(),
                                      "tolerance":1e-10, "evidence_path":str(path),
                                      "evidence_hash":manifest["artifacts"]["matrices.npz"],
                                      "reason":None}
            (path / "validation.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
            manifest["artifacts"]["validation.json"] = _hash_file(path / "validation.json")
            failed_gates = [gate for gate in MANDATORY_GATES
                            if details[gate]["status"] == "failed"]
            if failed_gates:
                manifest["status"] = "failed"
                (path / "summary.csv").unlink()
                manifest["artifacts"].pop("summary.csv")
                with (path / "run.log").open("a") as stream:
                    stream.write("failed gates: "+json.dumps(
                        {gate:details[gate] for gate in failed_gates}, allow_nan=False)+"\n")
                manifest["artifacts"]["run.log"] = _hash_file(path / "run.log")
            # Bare gate records never publish a complete run. Exact-run
            # evidence is attached only through finalize_revision_artifacts.
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            if manifest["status"] == "complete":
                load_revision_artifacts(path)
        except Exception as exc:
            manifest["status"] = "failed"
            (path / "run.log").write_text((path / "run.log").read_text()+f"validation failure: {exc}\n")
            manifest["artifacts"]["run.log"] = _hash_file(path / "run.log")
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            raise
    return path
