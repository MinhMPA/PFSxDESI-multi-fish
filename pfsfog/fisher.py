"""Single-tracer Fisher matrix assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FisherResult:
    """Result of a Fisher matrix computation."""

    F: np.ndarray                       # (Nparam, Nparam)
    param_names: list[str]
    z_bin: tuple[float, float]
    survey_name: str
    kmax: float

    @property
    def Nparam(self) -> int:
        return len(self.param_names)

    def _idx(self, name: str) -> int:
        return self.param_names.index(name)

    def marginalized_sigma(self, name: str) -> float:
        """σ from (F⁻¹)_{ii}^{1/2}, marginalized over all other params."""
        Finv = np.linalg.inv(self.F)
        i = self._idx(name)
        return float(np.sqrt(Finv[i, i]))

    def conditional_sigma(self, name: str, fixed: list[str] | None = None) -> float:
        """σ fixing a subset of other params.

        If *fixed* is ``None``, returns the unmarginalized (conditional)
        σ = 1/√F_{ii}.
        """
        if fixed is None:
            i = self._idx(name)
            return float(1.0 / np.sqrt(self.F[i, i]))

        # Keep only the rows/cols NOT in *fixed*
        keep = [n for n in self.param_names if n not in fixed]
        idx = [self._idx(n) for n in keep]
        Fsub = self.F[np.ix_(idx, idx)]
        Finv_sub = np.linalg.inv(Fsub)
        i_sub = keep.index(name)
        return float(np.sqrt(Finv_sub[i_sub, i_sub]))


# ---------------------------------------------------------------------------
# Fisher matrix from derivatives + covariance
# ---------------------------------------------------------------------------


def fisher_matrix(
    derivs: dict[str, dict[int, np.ndarray]],
    cov_inv: np.ndarray,
    k: np.ndarray,
    volume: float,
    dk: float,
    param_names: list[str],
    ells: tuple[int, ...] = (0, 2, 4),
) -> np.ndarray:
    """Assemble a Fisher matrix from pre-computed derivatives and covariance.

    Parameters
    ----------
    derivs : {param_name: {ell: dP_ell/dtheta(k)}}
        Derivative arrays, each shape ``(Nk,)``.
    cov_inv : array, shape (Nk, Nell, Nell)
        Inverse of the multipole covariance at each k.
    k : array, shape (Nk,)
    volume : survey volume (Mpc/h)³  [used only if not folded into cov_inv]
    dk : k-bin width
    param_names : ordered list of parameter names
    ells : multipole orders matching the covariance ordering

    Returns
    -------
    F : array, shape (Nparam, Nparam)
    """
    Nk = len(k)
    Nell = len(ells)
    Np = len(param_names)

    # Build derivative matrix: (Nk, Nell, Nparam)
    D = np.zeros((Nk, Nell, Np))
    for ip, pn in enumerate(param_names):
        for il, ell in enumerate(ells):
            if pn in derivs and ell in derivs[pn]:
                D[:, il, ip] = np.asarray(derivs[pn][ell])

    # F_ab = Σ_k  Dᵀ(k) C⁻¹(k) D(k)  × dk
    # (trapezoidal weights are approximately dk for uniform grid)
    F = np.zeros((Np, Np))
    for ik in range(Nk):
        DtCinv = D[ik].T @ cov_inv[ik]  # (Np, Nell)
        F += DtCinv @ D[ik] * dk

    return F


def add_gaussian_prior(F: np.ndarray, prior_diag: np.ndarray) -> np.ndarray:
    """Add Gaussian prior: F_total = F + diag(1/σ²)."""
    return F + np.diag(prior_diag)


# ---------------------------------------------------------------------------
# Convenience: build single-tracer Fisher for one z-bin
# ---------------------------------------------------------------------------


def single_tracer_fisher(
    derivs: dict[str, dict[int, np.ndarray]],
    cov: np.ndarray,
    k: np.ndarray,
    dk: float,
    param_names: list[str],
    z_bin: tuple[float, float],
    survey_name: str,
    kmax: float,
    prior_diag: np.ndarray | None = None,
    ells: tuple[int, ...] = (0, 2, 4),
) -> FisherResult:
    """End-to-end single-tracer Fisher for one z-bin.

    Parameters
    ----------
    derivs : derivative dict from derivatives.py
    cov : shape (Nk, Nell, Nell) covariance
    k : k-grid
    dk : bin width
    param_names : parameter names
    z_bin : (zlo, zhi)
    survey_name : label
    kmax : maximum wavenumber used
    prior_diag : optional Gaussian prior 1/σ² for each param
    ells : multipole orders

    Returns
    -------
    FisherResult
    """
    # Invert covariance at each k
    Nk = len(k)
    Nell = len(ells)
    cov_inv = np.zeros_like(cov)
    for ik in range(Nk):
        cov_inv[ik] = np.linalg.inv(cov[ik])

    F = fisher_matrix(derivs, cov_inv, k, volume=1.0, dk=dk,
                      param_names=param_names, ells=ells)

    if prior_diag is not None:
        F = add_gaussian_prior(F, prior_diag)

    return FisherResult(
        F=F,
        param_names=list(param_names),
        z_bin=z_bin,
        survey_name=survey_name,
        kmax=kmax,
    )
