"""Full-area DESI Fisher with imported priors.

F_total(z) = F_DESI_ST(z; V_full, kmax) + F_ext_prior(z)

Combines z-bins for shared cosmo params; per-z nuisance params
are independent across z-bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eft_params import NUISANCE_NAMES, COSMO_NAMES, COSMO_PRIOR_SIGMA
from .fisher import FisherResult, add_gaussian_prior


# Combined param names: cosmo + (nuisance × Nz)
def full_area_param_names(
    z_bins: list[tuple[float, float]],
    sample_labels: list[str] | None = None,
) -> list[str]:
    """Parameter names for the combined full-area Fisher.

    Cosmo params are shared; nuisance params are per-z-bin (or per-sample).
    If sample_labels is provided, uses those instead of z-bin ranges
    (avoids collisions when two samples share the same z-range, e.g.
    LRG3 and ELG1 at z=0.8–1.1).
    """
    names = list(COSMO_NAMES)
    labels = sample_labels or [f"z{zlo:.1f}_{zhi:.1f}" for zlo, zhi in z_bins]
    for label in labels:
        for n in NUISANCE_NAMES:
            names.append(f"{n}_{label}")
    return names


def full_area_fisher_per_zbin(
    derivs: dict[str, dict[int, np.ndarray]],
    cov: np.ndarray,
    k: np.ndarray,
    dk: float,
    nuisance_prior_diag: np.ndarray,
    z_bin: tuple[float, float],
    kmax: float,
    ells: tuple[int, ...] = (0, 2, 4),
    survey_name: str = "DESI full",
) -> FisherResult:
    """Single z-bin DESI Fisher + nuisance prior.

    Parameters
    ----------
    derivs : {param_name: {ell: dP_ell/dtheta}} for NUISANCE_NAMES only
        (cosmo derivatives are omitted in this implementation — the
        overlap calibration is about nuisance params).
    cov : (Nk, Nell, Nell)
    k : (Nk,)
    dk : bin width
    nuisance_prior_diag : (N_NUIS,) — 1/σ² per nuisance param
    z_bin : (zlo, zhi)
    kmax : maximum k used
    ells : multipole orders

    Returns
    -------
    FisherResult with N_COSMO + N_NUIS params
    """
    Nk = len(k)
    Nell = len(ells)
    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)
    Np = N_COSMO + N_NUIS

    param_names = list(COSMO_NAMES) + list(NUISANCE_NAMES)

    # Build derivative matrix (Nk, Nell, Np)
    D = np.zeros((Nk, Nell, Np))

    # Cosmo derivatives (columns 0..N_COSMO-1)
    for ic, cosmo_name in enumerate(COSMO_NAMES):
        for il, ell in enumerate(ells):
            if cosmo_name in derivs and ell in derivs[cosmo_name]:
                D[:, il, ic] = np.asarray(derivs[cosmo_name][ell])

    # Nuisance derivatives
    for ip, nuis_name in enumerate(NUISANCE_NAMES):
        idx = N_COSMO + ip
        for il, ell in enumerate(ells):
            if nuis_name in derivs and ell in derivs[nuis_name]:
                D[:, il, idx] = np.asarray(derivs[nuis_name][ell])

    # Invert covariance
    cov_inv = np.zeros_like(cov)
    for ik in range(Nk):
        cov_inv[ik] = np.linalg.inv(cov[ik])

    # Fisher from data
    F = np.zeros((Np, Np))
    for ik in range(Nk):
        DtCinv = D[ik].T @ cov_inv[ik]
        F += DtCinv @ D[ik] * dk

    # Add nuisance prior
    prior_full = np.zeros(Np)
    prior_full[N_COSMO:] = nuisance_prior_diag
    F = add_gaussian_prior(F, prior_full)

    # Add weak cosmo prior for regularisation
    for i, cn in enumerate(COSMO_NAMES):
        F[i, i] += 1.0 / COSMO_PRIOR_SIGMA[cn] ** 2

    return FisherResult(
        F=F, param_names=param_names,
        z_bin=z_bin, survey_name=survey_name, kmax=kmax,
    )


def combine_zbins(
    fisher_per_z: list[FisherResult],
    z_bins: list[tuple[float, float]],
    sample_labels: list[str] | None = None,
    survey_name: str = "DESI full combined",
) -> FisherResult:
    """Combine per-z (or per-sample) Fisher matrices into a joint Fisher.

    Cosmo params are shared across z-bins/samples (their Fisher
    contributions add). Per-z/sample nuisance params are independent.

    Parameters
    ----------
    sample_labels : optional list of unique labels per Fisher entry.
        When provided, used for parameter naming instead of z-bin ranges.
        Required when two samples share the same z-range (e.g. LRG3 and
        ELG1 at z=0.8–1.1).

    Returns
    -------
    FisherResult with N_COSMO + N_NUIS * Nz parameters
    """
    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)
    Nz = len(z_bins)
    Np = N_COSMO + N_NUIS * Nz

    param_names = full_area_param_names(z_bins, sample_labels)
    F = np.zeros((Np, Np))

    for iz, fr in enumerate(fisher_per_z):
        # Cosmo-cosmo block: adds directly
        F[:N_COSMO, :N_COSMO] += fr.F[:N_COSMO, :N_COSMO]

        # Cosmo-nuisance cross: slot into correct z-block
        nuis_start = N_COSMO + iz * N_NUIS
        nuis_end = nuis_start + N_NUIS

        F[:N_COSMO, nuis_start:nuis_end] += fr.F[:N_COSMO, N_COSMO:]
        F[nuis_start:nuis_end, :N_COSMO] += fr.F[N_COSMO:, :N_COSMO]

        # Nuisance-nuisance block
        F[nuis_start:nuis_end, nuis_start:nuis_end] += fr.F[N_COSMO:, N_COSMO:]

    kmax = max(fr.kmax for fr in fisher_per_z)
    return FisherResult(
        F=F, param_names=param_names,
        z_bin=(z_bins[0][0], z_bins[-1][1]),
        survey_name=survey_name,
        kmax=kmax,
    )
