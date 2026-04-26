"""Multi-tracer Fisher matrix in the PFS × DESI overlap volume.

Parameter vector per z-bin (27 params):
    θ = [θ_cosmo(3), θ_PFS(12), θ_DESI(12)]

Observable vector at each k (9 elements):
    d = [P_0^AA, P_2^AA, P_4^AA,
         P_0^BB, P_2^BB, P_4^BB,
         P_0^AB, P_2^AB, P_4^AB]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariance import multi_tracer_cov
from .eft_params import NUISANCE_NAMES, COSMO_NAMES
from .fisher import FisherResult


# ---------------------------------------------------------------------------
# Parameter vector layout
# ---------------------------------------------------------------------------

def mt_param_names() -> list[str]:
    """Full 27-parameter vector for the multi-tracer Fisher."""
    cosmo = list(COSMO_NAMES)
    pfs = [f"{n}_PFS" for n in NUISANCE_NAMES]
    desi = [f"{n}_DESI" for n in NUISANCE_NAMES]
    return cosmo + pfs + desi


_N_COSMO = len(COSMO_NAMES)
_N_NUIS = len(NUISANCE_NAMES)
_N_TOT = _N_COSMO + 2 * _N_NUIS  # 27


# ---------------------------------------------------------------------------
# Multi-tracer Fisher assembly
# ---------------------------------------------------------------------------


def multi_tracer_fisher(
    derivs_AA: dict[str, dict[int, np.ndarray]],
    derivs_BB: dict[str, dict[int, np.ndarray]],
    derivs_AB_A: dict[str, dict[int, np.ndarray]],
    derivs_AB_B: dict[str, dict[int, np.ndarray]],
    cov_mt: np.ndarray,
    k: np.ndarray,
    dk: float,
    z_bin: tuple[float, float],
    ells: tuple[int, ...] = (0, 2, 4),
) -> FisherResult:
    """Assemble the multi-tracer Fisher matrix.

    Parameters
    ----------
    derivs_AA : {nuisance_name: {ell: dP^AA_ell/dtheta}} for PFS params
    derivs_BB : {nuisance_name: {ell: dP^BB_ell/dtheta}} for DESI params
    derivs_AB_A : {nuisance_name: {ell: dP^AB_ell/dtheta_A}} PFS side of cross
    derivs_AB_B : {nuisance_name: {ell: dP^AB_ell/dtheta_B}} DESI side of cross
    cov_mt : array (Nk, 9, 9) — multi-tracer covariance
    k : array (Nk,)
    dk : k-bin width
    z_bin : (zlo, zhi)
    ells : multipole orders

    Returns
    -------
    FisherResult with 27×27 Fisher matrix
    """
    Nk = len(k)
    Nell = len(ells)
    Nobs = 3 * Nell  # 9
    param_names = mt_param_names()
    Np = len(param_names)

    # Build derivative matrix: (Nk, Nobs, Np)
    # Observable ordering: [AA_l0, AA_l2, AA_l4, BB_l0, BB_l2, BB_l4, AB_l0, AB_l2, AB_l4]
    D = np.zeros((Nk, Nobs, Np))

    for ip, nuis_name in enumerate(NUISANCE_NAMES):
        # PFS nuisance index in full param vector
        idx_pfs = _N_COSMO + ip

        # AA (auto PFS) — rows 0,1,2
        for il, ell in enumerate(ells):
            if nuis_name in derivs_AA and ell in derivs_AA[nuis_name]:
                D[:, il, idx_pfs] = np.asarray(derivs_AA[nuis_name][ell])

        # AB cross — PFS side — rows 6,7,8
        for il, ell in enumerate(ells):
            if nuis_name in derivs_AB_A and ell in derivs_AB_A[nuis_name]:
                D[:, 2 * Nell + il, idx_pfs] = np.asarray(derivs_AB_A[nuis_name][ell])

    for ip, nuis_name in enumerate(NUISANCE_NAMES):
        # DESI nuisance index in full param vector
        idx_desi = _N_COSMO + _N_NUIS + ip

        # BB (auto DESI) — rows 3,4,5
        for il, ell in enumerate(ells):
            if nuis_name in derivs_BB and ell in derivs_BB[nuis_name]:
                D[:, Nell + il, idx_desi] = np.asarray(derivs_BB[nuis_name][ell])

        # AB cross — DESI side — rows 6,7,8
        for il, ell in enumerate(ells):
            if nuis_name in derivs_AB_B and ell in derivs_AB_B[nuis_name]:
                D[:, 2 * Nell + il, idx_desi] += np.asarray(derivs_AB_B[nuis_name][ell])

    # Note: cosmo derivatives (columns 0,1,2) would appear in ALL
    # observables (AA, BB, AB). These are left at zero for now since
    # the overlap Fisher is used for nuisance calibration, not cosmo
    # constraints. The cosmo params are regularized by broad priors.

    # Invert covariance at each k
    cov_inv = np.zeros_like(cov_mt)
    for ik in range(Nk):
        cov_inv[ik] = np.linalg.inv(cov_mt[ik])

    # F_ab = Σ_k D^T(k) C^{-1}(k) D(k) × dk
    F = np.zeros((Np, Np))
    for ik in range(Nk):
        DtCinv = D[ik].T @ cov_inv[ik]  # (Np, Nobs)
        F += DtCinv @ D[ik] * dk

    return FisherResult(
        F=F,
        param_names=param_names,
        z_bin=z_bin,
        survey_name="PFS×DESI overlap",
        kmax=float(k[-1]),
    )


# ---------------------------------------------------------------------------
# Asymmetric kmax multi-tracer Fisher (Phase A)
# ---------------------------------------------------------------------------

# Observable indices for the reduced 6-element vector (AA + AB only)
_IDX_REDUCED = [0, 1, 2, 6, 7, 8]  # AA_l0, AA_l2, AA_l4, AB_l0, AB_l2, AB_l4


def multi_tracer_fisher_asymmetric(
    derivs_AA: dict[str, dict[int, np.ndarray]],
    derivs_BB: dict[str, dict[int, np.ndarray]],
    derivs_AB_A: dict[str, dict[int, np.ndarray]],
    derivs_AB_B: dict[str, dict[int, np.ndarray]],
    cov_mt_low: np.ndarray,
    cov_mt_high: np.ndarray | None,
    k_low: np.ndarray,
    k_high: np.ndarray | None,
    dk: float,
    z_bin: tuple[float, float],
    ells: tuple[int, ...] = (0, 2, 4),
) -> FisherResult:
    """Multi-tracer Fisher with asymmetric kmax per spectrum.

    k ∈ [kmin, kmax_DESI]: full 9 observables (AA + BB + AB).
    k ∈ (kmax_DESI, kmax_PFS]: 6 observables (AA + AB only).

    Parameters
    ----------
    derivs_AA : derivatives on k_all = concat(k_low, k_high)
    derivs_BB : derivatives on k_low only
    derivs_AB_A, derivs_AB_B : derivatives on k_all
    cov_mt_low : (Nk_low, 9, 9) — full multi-tracer covariance
    cov_mt_high : (Nk_high, 6, 6) — AA+AB subblock, or None
    k_low : k-grid for the low range
    k_high : k-grid for the high range, or None
    dk : bin width
    z_bin : (zlo, zhi)
    ells : multipole orders
    """
    Nell = len(ells)
    param_names = mt_param_names()
    Np = len(param_names)
    Nk_low = len(k_low)
    Nk_high = len(k_high) if k_high is not None else 0

    # --- Low range: full 9 observables ---
    F = np.zeros((Np, Np))

    # Build D_low: (Nk_low, 9, Np)
    D_low = np.zeros((Nk_low, 3 * Nell, Np))
    for ip, nn in enumerate(NUISANCE_NAMES):
        idx_pfs = _N_COSMO + ip
        idx_desi = _N_COSMO + _N_NUIS + ip
        for il, ell in enumerate(ells):
            if nn in derivs_AA and ell in derivs_AA[nn]:
                D_low[:, il, idx_pfs] = np.asarray(derivs_AA[nn][ell])[:Nk_low]
            if nn in derivs_BB and ell in derivs_BB[nn]:
                D_low[:, Nell + il, idx_desi] = np.asarray(derivs_BB[nn][ell])
            if nn in derivs_AB_A and ell in derivs_AB_A[nn]:
                D_low[:, 2 * Nell + il, idx_pfs] = np.asarray(derivs_AB_A[nn][ell])[:Nk_low]
            if nn in derivs_AB_B and ell in derivs_AB_B[nn]:
                D_low[:, 2 * Nell + il, idx_desi] += np.asarray(derivs_AB_B[nn][ell])[:Nk_low]

    cov_inv_low = np.zeros_like(cov_mt_low)
    for ik in range(Nk_low):
        cov_inv_low[ik] = np.linalg.inv(cov_mt_low[ik])

    for ik in range(Nk_low):
        DtCinv = D_low[ik].T @ cov_inv_low[ik]
        F += DtCinv @ D_low[ik] * dk

    # --- High range: 6 observables (AA + AB only) ---
    if Nk_high > 0 and k_high is not None and cov_mt_high is not None:
        D_high = np.zeros((Nk_high, 2 * Nell, Np))
        for ip, nn in enumerate(NUISANCE_NAMES):
            idx_pfs = _N_COSMO + ip
            idx_desi = _N_COSMO + _N_NUIS + ip
            for il, ell in enumerate(ells):
                # AA — rows 0,1,2 in the reduced vector
                if nn in derivs_AA and ell in derivs_AA[nn]:
                    arr = np.asarray(derivs_AA[nn][ell])
                    D_high[:, il, idx_pfs] = arr[Nk_low:Nk_low + Nk_high]
                # AB — rows 3,4,5 in the reduced vector
                if nn in derivs_AB_A and ell in derivs_AB_A[nn]:
                    arr = np.asarray(derivs_AB_A[nn][ell])
                    D_high[:, Nell + il, idx_pfs] = arr[Nk_low:Nk_low + Nk_high]
                if nn in derivs_AB_B and ell in derivs_AB_B[nn]:
                    arr = np.asarray(derivs_AB_B[nn][ell])
                    D_high[:, Nell + il, idx_desi] += arr[Nk_low:Nk_low + Nk_high]

        cov_inv_high = np.zeros_like(cov_mt_high)
        for ik in range(Nk_high):
            cov_inv_high[ik] = np.linalg.inv(cov_mt_high[ik])

        for ik in range(Nk_high):
            DtCinv = D_high[ik].T @ cov_inv_high[ik]
            F += DtCinv @ D_high[ik] * dk

    kmax = float(k_low[-1])
    if k_high is not None and len(k_high) > 0:
        kmax = float(k_high[-1])

    return FisherResult(
        F=F,
        param_names=param_names,
        z_bin=z_bin,
        survey_name="PFS×DESI overlap (asymmetric kmax)",
        kmax=kmax,
    )
