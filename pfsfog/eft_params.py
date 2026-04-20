"""EFT parameter fiducials and priors.

Parameterisation follows Chudaykin, Ivanov & Philcox (2025,
arXiv:2511.20757, Table I).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Parameter names (order matters for Fisher indexing)
# ---------------------------------------------------------------------------

# Per-tracer nuisance parameters in the Fisher matrix
NUISANCE_NAMES: list[str] = [
    "b1_sigma8",
    "b2_sigma8sq",
    "bG2_sigma8sq",
    "bGamma3",
    "c0",
    "c2",
    "c4",
    "c_tilde",
    "c1",
    "Pshot",
    "a0",
    "a2",
]

# Shared cosmological parameters
COSMO_NAMES: list[str] = ["fsigma8", "Mnu", "Omegam"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EFTFiducials:
    """Fiducial values for one tracer at one redshift."""

    b1_sigma8: float
    b2_sigma8sq: float
    bG2_sigma8sq: float
    bGamma3: float
    c0: float          # [Mpc/h]²
    c2: float          # [Mpc/h]²
    c4: float          # [Mpc/h]²
    c_tilde: float     # [Mpc/h]⁴  (FoG counterterm)
    c1: float          # [Mpc/h]²
    Pshot: float       # dimensionless, relative to 1/nbar
    a0: float          # dimensionless
    a2: float          # dimensionless

    def as_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in NUISANCE_NAMES}

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in NUISANCE_NAMES])


@dataclass
class EFTPriors:
    """Gaussian prior widths (σ) for one tracer at one redshift.

    ``None`` means flat / no informative prior (e.g. b1_sigma8).
    """

    b1_sigma8: float | None = None   # flat U[0,3]
    b2_sigma8sq: float = 5.0
    bG2_sigma8sq: float = 5.0
    bGamma3: float = 1.0
    c0: float = 30.0
    c2: float = 30.0
    c4: float = 30.0
    c_tilde: float = 400.0
    c1: float = 5.0
    Pshot: float = 1.0
    a0: float = 1.0
    a2: float = 1.0

    def sigma_dict(self) -> dict[str, float | None]:
        return {n: getattr(self, n) for n in NUISANCE_NAMES}

    def prior_fisher_diag(self) -> np.ndarray:
        """Diagonal of F_prior = 1/σ² for each nuisance parameter.

        Flat priors contribute 0.
        """
        out = []
        for n in NUISANCE_NAMES:
            s = getattr(self, n)
            out.append(0.0 if s is None else 1.0 / s**2)
        return np.array(out)


# ---------------------------------------------------------------------------
# Lazeyras+ 2016 co-evolution relations
# ---------------------------------------------------------------------------


def lazeyras_b2(b1: float) -> float:
    """b2 from b1 via Lazeyras+ (2016) fitting formula."""
    return 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3


def lazeyras_bG2(b1: float) -> float:
    """bG2 from b1 via local Lagrangian: bG2 = -2/7 (b1 - 1)."""
    return -2.0 / 7.0 * (b1 - 1.0)


# ---------------------------------------------------------------------------
# DESI-ELG fiducials (§3.2)
# ---------------------------------------------------------------------------


def desi_elg_fiducials(b1: float, sigma8_z: float) -> EFTFiducials:
    """DESI-ELG EFT fiducials at a given b1 and σ8(z).

    Prior means from Chudaykin+ 2025 Table I.
    """
    b2 = lazeyras_b2(b1)
    bG2 = lazeyras_bG2(b1)
    bGamma3 = 23.0 / 42.0 * (b1 - 1.0)

    return EFTFiducials(
        b1_sigma8=b1 * sigma8_z,
        b2_sigma8sq=b2 * sigma8_z**2,
        bG2_sigma8sq=bG2 * sigma8_z**2,
        bGamma3=bGamma3,
        c0=0.0,
        c2=30.0,
        c4=0.0,
        c_tilde=400.0,
        c1=0.0,
        Pshot=0.0,
        a0=0.0,
        a2=0.0,
    )


# ---------------------------------------------------------------------------
# DESI-LRG and DESI-QSO fiducials
# ---------------------------------------------------------------------------


def desi_lrg_fiducials(b1: float, sigma8_z: float) -> EFTFiducials:
    """DESI-LRG EFT fiducials.

    LRGs live in massive halos → larger virial velocities → larger c̃.
    c̃_LRG ≈ 800 (2× ELG fiducial, reflecting higher σ_v).
    Counterterms scale with bias.
    """
    b2 = lazeyras_b2(b1)
    bG2 = lazeyras_bG2(b1)
    bGamma3 = 23.0 / 42.0 * (b1 - 1.0)

    return EFTFiducials(
        b1_sigma8=b1 * sigma8_z,
        b2_sigma8sq=b2 * sigma8_z**2,
        bG2_sigma8sq=bG2 * sigma8_z**2,
        bGamma3=bGamma3,
        c0=0.0,
        c2=30.0 * b1 / 1.3,       # scale from ELG fiducial by bias ratio
        c4=0.0,
        c_tilde=800.0,              # larger FoG for LRGs
        c1=0.0,
        Pshot=0.0,
        a0=0.0,
        a2=0.0,
    )


def desi_qso_fiducials(b1: float, sigma8_z: float) -> EFTFiducials:
    """DESI-QSO EFT fiducials.

    QSOs have the highest FoG (massive host halos, broad-line kinematics).
    c̃_QSO ≈ 1200 (3× ELG fiducial).
    """
    b2 = lazeyras_b2(b1)
    bG2 = lazeyras_bG2(b1)
    bGamma3 = 23.0 / 42.0 * (b1 - 1.0)

    return EFTFiducials(
        b1_sigma8=b1 * sigma8_z,
        b2_sigma8sq=b2 * sigma8_z**2,
        bG2_sigma8sq=bG2 * sigma8_z**2,
        bGamma3=bGamma3,
        c0=0.0,
        c2=30.0 * b1 / 1.3,
        c4=0.0,
        c_tilde=1200.0,             # largest FoG
        c1=0.0,
        Pshot=0.0,
        a0=0.0,
        a2=0.0,
    )


# ---------------------------------------------------------------------------
# Generic fiducial builder for any tracer
# ---------------------------------------------------------------------------


def tracer_fiducials(
    tracer_name: str, b1: float, sigma8_z: float,
    b1_ref: float = 1.3, r_sigma_v: float = 0.75,
) -> EFTFiducials:
    """Build EFT fiducials for any named tracer."""
    if tracer_name == "DESI-ELG":
        return desi_elg_fiducials(b1, sigma8_z)
    if tracer_name == "DESI-LRG":
        return desi_lrg_fiducials(b1, sigma8_z)
    if tracer_name == "DESI-QSO":
        return desi_qso_fiducials(b1, sigma8_z)
    if tracer_name == "PFS-ELG":
        return pfs_elg_fiducials(b1, b1_ref, sigma8_z, r_sigma_v)
    raise ValueError(f"Unknown tracer: {tracer_name}")


# ---------------------------------------------------------------------------
# PFS-ELG fiducials (§3.3 — scaled from DESI)
# ---------------------------------------------------------------------------


def pfs_elg_fiducials(
    b1_pfs: float,
    b1_desi: float,
    sigma8_z: float,
    r_sigma_v: float = 0.75,
) -> EFTFiducials:
    """PFS-ELG EFT fiducials scaled from DESI fiducials.

    Parameters
    ----------
    b1_pfs, b1_desi : float
        Linear biases at the same redshift.
    sigma8_z : float
        σ8 at the target redshift.
    r_sigma_v : float
        σ_v,PFS / σ_v,DESI ratio.  Default 0.75.
    """
    b2 = lazeyras_b2(b1_pfs)
    bG2 = lazeyras_bG2(b1_pfs)
    bGamma3 = 23.0 / 42.0 * (b1_pfs - 1.0)

    # Scale counterterms by bias ratio
    bias_ratio = b1_pfs / b1_desi if b1_desi != 0 else 1.0

    # FoG counterterm scales with σ_v²
    c_tilde_desi = 400.0
    c_tilde_pfs = c_tilde_desi * r_sigma_v**2

    return EFTFiducials(
        b1_sigma8=b1_pfs * sigma8_z,
        b2_sigma8sq=b2 * sigma8_z**2,
        bG2_sigma8sq=bG2 * sigma8_z**2,
        bGamma3=bGamma3,
        c0=0.0 * bias_ratio,       # fiducial c0 = 0
        c2=30.0 * bias_ratio,
        c4=0.0 * bias_ratio,       # fiducial c4 = 0
        c_tilde=c_tilde_pfs,
        c1=0.0 * bias_ratio,       # fiducial c1 = 0
        Pshot=0.0,
        a0=0.0,
        a2=0.0,
    )


# ---------------------------------------------------------------------------
# Broad priors (§3.4)
# ---------------------------------------------------------------------------


def broad_priors() -> EFTPriors:
    """Conservative Gaussian prior widths from Chudaykin+ 2025 Table I."""
    return EFTPriors()


# ---------------------------------------------------------------------------
# HOD-prior benchmark reference lines (§3.5)
# ---------------------------------------------------------------------------

HOD_BENCHMARK = {
    "source": "Zhang+ 2025 (arXiv:2504.10407)",
    "sigma8_improvement": 0.23,
    "Omegam_improvement": 0.04,
}

FIELD_LEVEL_BENCHMARK = {
    "source": "Chudaykin+ 2026 (arXiv:2602.18554)",
    "H0_improvement": 0.40,
    "sigma8_improvement": 0.50,
}


# ---------------------------------------------------------------------------
# Cosmological parameter priors (weak, for Fisher regularisation)
# ---------------------------------------------------------------------------

COSMO_PRIOR_SIGMA = {
    "fsigma8": 10.0,
    "Mnu": 5.0,        # eV
    "Omegam": 1.0,
}
