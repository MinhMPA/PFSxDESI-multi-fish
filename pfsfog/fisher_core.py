"""Information from independent Fourier shells with finite-shell covariance."""

from __future__ import annotations

import numpy as np


def fisher_from_shells(
    derivatives: np.ndarray,
    covariance: np.ndarray,
    k: np.ndarray,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Sum D.T C^-1 D; covariance already includes volume and shell width.

    D has axes (shell, observable, parameter), C (shell, observable,
    observable). Select retained observations before factorization. Scale
    C to a correlation matrix and whiten D using Cholesky solves; no
    pseudoinverse, eigenvalue clipping or skipped invalid shells is allowed.
    """
    D = np.asarray(derivatives, dtype=float)
    C = np.asarray(covariance, dtype=float)
    k = np.asarray(k, dtype=float)
    if D.ndim != 3 or k.ndim != 1:
        raise ValueError("derivatives must be (Nk,Nobs,Nparam); k must be 1D")
    nk, nobs, nparam = D.shape
    if len(k) != nk or C.shape != (nk, nobs, nobs):
        raise ValueError("incompatible shell covariance/derivative shapes")
    if not np.all(np.isfinite(k)):
        raise ValueError("k must be finite")
    mask = np.ones((nk, nobs), dtype=bool) if active is None else np.asarray(active)
    if mask.shape != (nk, nobs) or mask.dtype != np.dtype(bool):
        raise ValueError("active must be a boolean (Nk,Nobs) mask")
    F = np.zeros((nparam, nparam))
    for ik in range(nk):
        use = mask[ik]
        if not np.any(use):
            continue
        c = C[ik][np.ix_(use, use)]
        d = D[ik][use]
        context = f"shell {ik} at k={k[ik]:.12g}"
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
            raise ValueError(f"nonfinite retained input in {context}")
        scale = np.max(np.abs(c))
        if scale == 0 or np.max(np.abs(c - c.T)) > 1e-10 * scale:
            raise ValueError(f"invalid covariance symmetry/scale in {context}")
        if np.any(np.diag(c) <= 0):
            raise ValueError(f"nonpositive covariance diagonal in {context}")
        s = np.sqrt(np.diag(c))
        c = (c + c.T) / 2 / s[:, None] / s[None, :]
        try:
            L = np.linalg.cholesky(c)
            w = np.linalg.solve(L, d / s[:, None])
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"covariance factorization failed in {context}") from exc
        # Direct contraction avoids Accelerate's spurious floating-point
        # status flags on this host without suppressing arithmetic warnings.
        # The shell matrices are small; retain explicit finite-output checks.
        F += np.einsum("oi,oj->ij", w, w, optimize=False)
    if not np.all(np.isfinite(F)):
        raise ValueError("nonfinite summed Fisher")
    return (F + F.T) / 2
