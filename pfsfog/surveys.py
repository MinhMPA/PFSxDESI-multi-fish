"""Survey specifications: n(z) loading, volume computation, z-binning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .cosmo import FiducialCosmology, FIDUCIAL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SPEC_DIR = Path(__file__).resolve().parent.parent / "survey_specs"

# ---------------------------------------------------------------------------
# n(z) file loader
# ---------------------------------------------------------------------------


def load_nz_table(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a survey_specs ``*.txt`` file.

    Returns
    -------
    z_min, z_max, nz, Vz : ndarray
        Bin edges, number density [(h⁻¹Mpc)⁻³], comoving volume [(h⁻¹Mpc)³].
    """
    data = np.loadtxt(path, comments="#")
    z_min, z_max, nz, Vz = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return z_min, z_max, nz, Vz


def _interp_nz(
    z_min: np.ndarray, z_max: np.ndarray, nz: np.ndarray,
) -> Callable[[float], float]:
    """Return a callable nbar(z) interpolating the fine-bin table."""
    z_mid = 0.5 * (z_min + z_max)

    def nbar_of_z(z: float) -> float:
        return float(np.interp(z, z_mid, nz, left=0.0, right=0.0))

    return nbar_of_z


# ---------------------------------------------------------------------------
# Survey dataclass
# ---------------------------------------------------------------------------


@dataclass
class Survey:
    """A spectroscopic galaxy survey."""

    name: str
    area_deg2: float
    z_min_all: np.ndarray        # fine-bin lower edges
    z_max_all: np.ndarray        # fine-bin upper edges
    nz_all: np.ndarray           # n(z) per fine bin [(h⁻¹Mpc)⁻³]
    Vz_all: np.ndarray           # comoving volume per fine bin [(h⁻¹Mpc)³]
    b1_of_z: Callable[[float], float]

    @property
    def nbar_of_z(self) -> Callable[[float], float]:
        return _interp_nz(self.z_min_all, self.z_max_all, self.nz_all)

    def nbar_eff(self, zlo: float, zhi: float) -> float:
        """Volume-weighted effective nbar in [zlo, zhi]."""
        mask = (self.z_min_all >= zlo - 1e-6) & (self.z_max_all <= zhi + 1e-6)
        nz_sel = self.nz_all[mask]
        Vz_sel = self.Vz_all[mask]
        if Vz_sel.sum() == 0:
            return 0.0
        return float(np.average(nz_sel, weights=Vz_sel))

    def z_eff(self, zlo: float, zhi: float) -> float:
        """Volume-weighted effective redshift in [zlo, zhi]."""
        mask = (self.z_min_all >= zlo - 1e-6) & (self.z_max_all <= zhi + 1e-6)
        z_mid = 0.5 * (self.z_min_all[mask] + self.z_max_all[mask])
        Vz_sel = self.Vz_all[mask]
        if Vz_sel.sum() == 0:
            return 0.5 * (zlo + zhi)
        return float(np.average(z_mid, weights=Vz_sel))

    def volume(self, zlo: float, zhi: float) -> float:
        """Total comoving volume in [zlo, zhi] in (Mpc/h)³.

        The stored Vz are for the file's native survey area.
        """
        mask = (self.z_min_all >= zlo - 1e-6) & (self.z_max_all <= zhi + 1e-6)
        return float(self.Vz_all[mask].sum())

    def volume_rescaled(self, zlo: float, zhi: float, area_deg2: float) -> float:
        """Volume rescaled to a different sky area."""
        return self.volume(zlo, zhi) * area_deg2 / self.area_deg2


# ---------------------------------------------------------------------------
# Built-in surveys
# ---------------------------------------------------------------------------


def pfs_elg() -> Survey:
    """PFS-ELG survey (1,200 deg²)."""
    z_min, z_max, nz, Vz = load_nz_table(_SPEC_DIR / "PFS_nz_pfs_fine.txt")
    # Rescale volumes from the file's native area to 1200 deg²
    # PFS file volumes correspond to 1200 deg²
    return Survey(
        name="PFS-ELG",
        area_deg2=1200.0,
        z_min_all=z_min,
        z_max_all=z_max,
        nz_all=nz,
        Vz_all=Vz,
        b1_of_z=lambda z: 0.9 + 0.4 * z,
    )


def desi_elg() -> Survey:
    """DESI-ELG survey (14,000 deg²)."""
    z_min, z_max, nz, Vz = load_nz_table(_SPEC_DIR / "DESI_nz_elg_fine.txt")

    def _b1_desi(z: float, cosmo: FiducialCosmology | None = None) -> float:
        if cosmo is None:
            cosmo = FiducialCosmology()
        return 0.84 / float(cosmo.D(z))

    return Survey(
        name="DESI-ELG",
        area_deg2=14000.0,
        z_min_all=z_min,
        z_max_all=z_max,
        nz_all=nz,
        Vz_all=Vz,
        b1_of_z=_b1_desi,
    )


# ---------------------------------------------------------------------------
# SurveyPair
# ---------------------------------------------------------------------------

# Matched z-bins for the overlap (§2.3)
OVERLAP_ZBINS: list[tuple[float, float]] = [
    (0.8, 1.0),
    (1.0, 1.2),
    (1.2, 1.4),
    (1.4, 1.6),
]


@dataclass
class SurveyPair:
    """Two surveys with an overlapping footprint."""

    A: Survey           # PFS
    B: Survey           # DESI
    overlap_area_deg2: float = 1200.0

    def V_overlap(self, zlo: float, zhi: float) -> float:
        """Comoving volume in the overlap region, (Mpc/h)³."""
        # Use survey B volumes rescaled to the overlap area
        return self.B.volume_rescaled(zlo, zhi, self.overlap_area_deg2)

    def V_full_B(self, zlo: float, zhi: float) -> float:
        """Comoving volume of the full B (DESI) footprint, (Mpc/h)³."""
        return self.B.volume(zlo, zhi)

    def lever_arm(self, zlo: float, zhi: float) -> float:
        """V_full_B / V_overlap ≈ 14000/1200 ≈ 11.7."""
        v_ov = self.V_overlap(zlo, zhi)
        if v_ov == 0:
            return float("inf")
        return self.V_full_B(zlo, zhi) / v_ov


def default_survey_pair() -> SurveyPair:
    """PFS × DESI with 1,200 deg² overlap."""
    return SurveyPair(A=pfs_elg(), B=desi_elg(), overlap_area_deg2=1200.0)
