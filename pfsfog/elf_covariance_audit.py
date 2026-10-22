"""Field and multipole covariance checks at every physical FD displacement."""
from pathlib import Path
import hashlib
import json

import numpy as np

from .covariance_mt_general import multi_tracer_cov_general
from .eft_params import CROSS_STOCH_NAMES
from .elf_survey import elf_cosmology_spec
from .revision_forecast import make_shell_grid
from .revision_theory import NUISANCE_NAMES, make_pair_response
from .revision_validation_physical import (
    CovarianceAuditFailure, _audited_spd, _steps, validate_physical_point)


class BatchCovarianceFailure(ValueError):
    def __init__(self, matrix, index, context, reason):
        self.matrix = np.array(matrix, copy=True)
        self.index = tuple(int(i) for i in index)
        try:
            self.diagnostics = _audited_spd(matrix, context)
        except CovarianceAuditFailure as error:
            self.diagnostics = error.diagnostics
        super().__init__(f"{reason}: {context}, matrix_index={self.index}")


def audit_spd_batch(matrices, context):
    """Check every unchanged matrix; scaling is only used for diagnostics."""
    m = np.asarray(matrices, dtype=float)
    if m.ndim < 3 or m.shape[-1] != m.shape[-2] or m.size == 0:
        raise ValueError("expected a nonempty batch of square covariance matrices")
    leading = m.shape[:-2]
    flat = m.reshape(-1, m.shape[-1], m.shape[-1])

    def reject(mask, reason):
        if np.any(mask):
            i = int(np.flatnonzero(mask)[0])
            raise BatchCovarianceFailure(flat[i], np.unravel_index(i, leading), context, reason)

    reject(~np.isfinite(flat).all(axis=(-2, -1)), "nonfinite covariance")
    diagonal = np.diagonal(flat, axis1=-2, axis2=-1)
    reject(np.any(diagonal <= 0, axis=-1), "nonpositive covariance diagonal")
    scale = np.sqrt(diagonal)
    scaled = flat / scale[:, :, None] / scale[:, None, :]
    asymmetry = np.max(np.abs(scaled-scaled.swapaxes(-1, -2)), axis=(-2, -1))
    reject(asymmetry > 1.e-10, "asymmetric covariance")
    symmetric = (scaled+scaled.swapaxes(-1, -2))/2

    def factor(operation, name):
        try:
            return operation(symmetric)
        except np.linalg.LinAlgError as original:
            # A rejected batch must still identify and retain the unchanged
            # physical matrix. Scalar attempts diagnose the failure only.
            for i, matrix in enumerate(symmetric):
                try:
                    operation(matrix)
                except np.linalg.LinAlgError as error:
                    raise BatchCovarianceFailure(flat[i], np.unravel_index(i, leading),
                                                 context, name+" failed: "+str(error)) from original
            # Do not invent a shell if the batch failure cannot be localized.
            # Preserve the entire original batch and explicitly leave index empty.
            raise BatchCovarianceFailure(m, (), context,
                name+" batch failed but scalar localization did not reproduce it") from original

    eigen, vectors = factor(np.linalg.eigh, "eigensolver")
    reject(eigen[:, 0] <= 0, "nonpositive covariance eigenvalue")
    eigen_residual = np.max(np.abs(np.einsum("...ik,...k,...jk->...ij", vectors,
                            eigen, vectors, optimize=False)-symmetric), axis=(-2, -1))
    reject(eigen_residual > 1.e-10, "covariance eigen residual")
    chol = factor(np.linalg.cholesky, "Cholesky")
    chol_residual = np.max(np.abs(np.einsum("...ik,...jk->...ij", chol, chol,
                              optimize=False)-symmetric), axis=(-2, -1))
    reject(chol_residual > 1.e-10, "covariance Cholesky residual")
    worst = int(np.argmin(eigen[:, 0]))
    return {"matrix_count": len(flat), "min_scaled_eigenvalue": float(eigen[worst, 0]),
            "max_scaled_condition": float(np.max(eigen[:, -1]/eigen[:, 0])),
            "max_scaled_residual": float(max(eigen_residual.max(), chol_residual.max())),
            "max_scaled_asymmetry": float(asymmetry.max()),
            "worst_index": [int(i) for i in np.unravel_index(worst, leading)]}


def covariance_from_spectra(signals, metadata, k, widths, *, n_mu):
    """Retain every theoretical cross and count Poisson/modes exactly once."""
    active = tuple(metadata["active_populations"])
    pairs = [tuple(p) for p in metadata["theoretical_pairs"]]
    expected = [(n, n) for n in active] + [
        (a, b) for i, a in enumerate(active) for b in active[i+1:]]
    if pairs != expected or np.shape(signals) != (len(pairs), len(k), n_mu):
        raise ValueError("theoretical angular spectra do not match the full field")
    indexes = {p: i for i, p in enumerate(pairs)}
    field = np.empty((len(k), n_mu, len(active), len(active)))
    for ia, a in enumerate(active):
        for ib, b in enumerate(active):
            field[:, :, ia, ib] = signals[indexes[tuple(sorted((a, b)))]]
            if a == b:
                field[:, :, ia, ib] += 1./metadata["nbar"][a]
    functions = {pair: (lambda _k, _mu, value=signals[i]: value) for i, pair in enumerate(pairs)}
    covariance = multi_tracer_cov_general(list(active), functions, metadata["nbar"],
        np.asarray(k), metadata["volume"], np.asarray(widths), n_mu=n_mu)
    observed = [3*indexes[tuple(pair)]+ell for pair in metadata["retained_pairs"] for ell in range(3)]
    return field, covariance[:, observed][:, :, observed]


def _write_json(path, data):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False)+"\n")


def _digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_neighborhood(cell, config, reference, engine, values, output):
    """Stream every point's spectra to disk before checking its covariance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"neighborhood_spectra.npy").exists():
        raise FileExistsError("a covariance neighborhood already exists in this directory")
    if set(values) != set(cell.names) or len(cell.names) != len(values):
        raise ValueError("neighborhood fiducials do not cover the exact local coordinates")
    space = elf_cosmology_spec(reference)
    metadata = cell.metadata
    ibin = metadata["nominal_bin_index"]
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    mu = np.polynomial.legendre.leggauss(config.n_mu)[0]
    descriptions = [{"column": None, "sign": 0, "step": 0.}]
    coordinates = [np.array([values[n] for n in cell.names])]
    for i, name in enumerate(cell.names):
        for step in _steps(name, values[name], space=space):
            for sign in (1, -1):
                point = coordinates[0].copy()
                point[i] += sign*step
                descriptions.append({"column": name, "sign": sign, "step": step})
                coordinates.append(point)
    coordinates = np.asarray(coordinates)
    np.save(output/"neighborhood_coordinates.npy", coordinates)
    pairs = [tuple(pair) for pair in metadata["theoretical_pairs"]]
    functions, caches = {}, {}
    for a, b in pairs:
        functions[(a, b)] = make_pair_response(engine, metadata["model_redshift"], reference,
            metadata["nbar"][a], metadata["nbar"][b] if a != b else None, config.k_nl, space=space)
        caches[(a, b)] = {}
    stored = np.lib.format.open_memmap(output/"neighborhood_spectra.npy", mode="w+",
        dtype=np.float64, shape=(len(coordinates), len(pairs), len(grid.centers), len(mu)))
    stored[:] = np.nan
    report = {"status": "running", "cell_id": metadata["cell_id"],
              "expected_points": len(coordinates), "processed_points": 0,
              "shell_count": len(grid.centers), "angular_order": config.n_mu,
              "parameter_names": list(cell.names), "theoretical_pairs": pairs,
              "points": [], "full_forecast_qualification": "pending"}
    for ipoint, (point, description) in enumerate(zip(coordinates, descriptions)):
        record = {**description, "status": "running"}
        report["points"].append(record)
        context = (f"cell={metadata['cell_id']} column={description['column']} "
                   f"sign={description['sign']} step={description['step']}")
        try:
            current = dict(zip(cell.names, point))
            theta = np.array([current[n] for n in space.names])
            validate_physical_point(theta, reference, space, metadata["model_redshift"])
            for ipair, (a, b) in enumerate(pairs):
                qa = np.array([current[f"bin{ibin:02d}:{a}:{n}"] for n in NUISANCE_NAMES])
                qb = (np.array([current[f"bin{ibin:02d}:{b}:{n}"] for n in NUISANCE_NAMES])
                      if a != b else None)
                qc = (np.array([current.get(f"bin{ibin:02d}:{a}|{b}:{n}", 0.)
                                for n in CROSS_STOCH_NAMES]) if a != b else None)
                key = tuple(np.concatenate((theta, qa, qb if qb is not None else [],
                                             qc if qc is not None else [])))
                cache = caches[(a, b)]
                if key not in cache:
                    cache[key] = np.asarray(functions[(a, b)].pkmu(grid.centers, mu,
                                                theta, qa, qb, qc), dtype=float)
                stored[ipoint, ipair] = cache[key]
            stored.flush()
            field, covariance = covariance_from_spectra(stored[ipoint], metadata,
                grid.centers, grid.widths, n_mu=config.n_mu)
            record["field"] = audit_spd_batch(field, "field "+context)
            record["covariance"] = audit_spd_batch(covariance, "multipole "+context)
            record["status"] = "passed"
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, reason=str(error))
            if isinstance(error, BatchCovarianceFailure):
                np.save(output/"failed_matrix.npy", error.matrix)
                record.update(matrix_index=error.index, diagnostics=error.diagnostics)
                if error.index:
                    record["k"] = float(grid.centers[error.index[0]])
                else:
                    record["matrix_index_unavailable_reason"] = "batch failure not reproduced by scalar localization"
                if len(error.index) == 2:
                    record["mu"] = float(mu[error.index[1]])
            report["status"] = "failed"
        report["processed_points"] = ipoint+1
        if report["status"] == "failed" or (ipoint+1) % 32 == 0:
            stored.flush()
            _write_json(output/"neighborhood.json", report)
        if (ipoint+1) % 64 == 0:
            print(f"{metadata['cell_id']}: covariance points {ipoint+1}/{len(coordinates)}", flush=True)
        if report["status"] == "failed":
            break
    stored.flush()
    if report["status"] != "failed":
        report["status"] = "passed" if report["processed_points"] == len(coordinates) else "failed"
    report["spectra_sha256"] = _digest(output/"neighborhood_spectra.npy")
    report["coordinates_sha256"] = _digest(output/"neighborhood_coordinates.npy")
    _write_json(output/"neighborhood.json", report)
    return report
