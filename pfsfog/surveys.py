"""Survey specifications: n(z) loading, volume computation, z-binning."""

from __future__ import annotations

import functools
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
# Bias model callables (module-level so Survey objects are picklable)
# ---------------------------------------------------------------------------


def _pfs_elg_bias(z: float) -> float:
    """PFS-ELG linear bias model: b1(z) = 0.9 + 0.4*z (Orsi+ 2010)."""
    return 0.9 + 0.4 * z


def _desi_bias(b0: float, z: float,
               cosmo: FiducialCosmology | None = None) -> float:
    """DESI-style linear bias model: b1(z) = b0 / D(z).

    Module-level (rather than nested) so ``functools.partial(_desi_bias, b0)``
    is picklable and can cross multiprocessing boundaries.
    """
    if cosmo is None:
        cosmo = FiducialCosmology()
    return b0 / float(cosmo.D(z))


def _make_desi_bias(b0: float):
    """Create a DESI-style bias function b(z) = b0 / D(z).

    Returns a ``functools.partial`` rather than a nested closure so the
    resulting callable can be pickled (required for multiprocessing).
    """
    return functools.partial(_desi_bias, b0)


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
        b1_of_z=_pfs_elg_bias,
    )


def desi_elg() -> Survey:
    """DESI-ELG survey (14,000 deg²). b(z) = 0.84/D(z)."""
    z_min, z_max, nz, Vz = load_nz_table(_SPEC_DIR / "DESI_nz_elg_fine.txt")
    return Survey(
        name="DESI-ELG",
        area_deg2=14000.0,
        z_min_all=z_min, z_max_all=z_max, nz_all=nz, Vz_all=Vz,
        b1_of_z=_make_desi_bias(0.84),
    )


def desi_lrg() -> Survey:
    """DESI-LRG survey (14,000 deg²). b(z) = 1.7/D(z)."""
    z_min, z_max, nz, Vz = load_nz_table(_SPEC_DIR / "DESI_nz_lrg_fine.txt")
    return Survey(
        name="DESI-LRG",
        area_deg2=14000.0,
        z_min_all=z_min, z_max_all=z_max, nz_all=nz, Vz_all=Vz,
        b1_of_z=_make_desi_bias(1.7),
    )


def desi_qso() -> Survey:
    """DESI-QSO survey (14,000 deg²). b(z) = 1.2/D(z)."""
    z_min, z_max, nz, Vz = load_nz_table(_SPEC_DIR / "DESI_nz_qso_fine.txt")
    return Survey(
        name="DESI-QSO",
        area_deg2=14000.0,
        z_min_all=z_min, z_max_all=z_max, nz_all=nz, Vz_all=Vz,
        b1_of_z=_make_desi_bias(1.2),
    )


# ---------------------------------------------------------------------------
# SurveyPair (original, kept for backward compatibility)
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
        return self.B.volume_rescaled(zlo, zhi, self.overlap_area_deg2)

    def V_full_B(self, zlo: float, zhi: float) -> float:
        """Comoving volume of full DESI footprint, (Mpc/h)³."""
        return self.B.volume(zlo, zhi)

    def lever_arm(self, zlo: float, zhi: float) -> float:
        v_ov = self.V_overlap(zlo, zhi)
        if v_ov == 0:
            return float("inf")
        return self.V_full_B(zlo, zhi) / v_ov


# ---------------------------------------------------------------------------
# Multi-tracer survey group
# ---------------------------------------------------------------------------


@dataclass
class SurveyGroup:
    """PFS + multiple DESI tracers in the overlap.

    Tracers are identified by name. At each z-bin, only tracers
    with non-zero nbar are included.
    """

    pfs: Survey
    desi_tracers: dict[str, Survey]    # {"DESI-ELG": ..., "DESI-LRG": ..., ...}
    overlap_area_deg2: float = 1200.0
    desi_full_area_deg2: float = 14000.0
    pfs_zmax: float = 1.6              # PFS truncation for joint analysis

    @property
    def all_surveys(self) -> dict[str, Survey]:
        return {"PFS-ELG": self.pfs, **self.desi_tracers}

    def active_tracers(self, zlo: float, zhi: float,
                       nbar_min: float = 1e-6) -> dict[str, Survey]:
        """Return tracers with non-negligible nbar in [zlo, zhi]."""
        active = {}
        for name, s in self.all_surveys.items():
            if s.nbar_eff(zlo, zhi) > nbar_min:
                active[name] = s
        return active

    def tracer_pairs(self, zlo: float, zhi: float,
                     nbar_min: float = 1e-6) -> list[tuple[str, str]]:
        """All unique tracer pairs (including auto) active in [zlo, zhi].

        Returns list of (name_A, name_B) with A <= B lexicographically.
        """
        names = sorted(self.active_tracers(zlo, zhi, nbar_min).keys())
        pairs = []
        for i, a in enumerate(names):
            for b in names[i:]:
                pairs.append((a, b))
        return pairs

    def V_overlap(self, zlo: float, zhi: float) -> float:
        """Overlap volume using any DESI tracer's volume rescaled."""
        # All DESI tracers share the same sky; pick any for the volume
        for s in self.desi_tracers.values():
            v = s.volume_rescaled(zlo, zhi, self.overlap_area_deg2)
            if v > 0:
                return v
        return 0.0

    def V_nonoverlap(self, zlo: float, zhi: float) -> float:
        """DESI footprint volume outside the PFS overlap."""
        area = self.desi_full_area_deg2 - self.overlap_area_deg2
        for s in self.desi_tracers.values():
            v = s.volume_rescaled(zlo, zhi, area)
            if v > 0:
                return v
        return 0.0

    def V_desi_full(self, zlo: float, zhi: float) -> float:
        """Full DESI volume (overlap + nonoverlap)."""
        for s in self.desi_tracers.values():
            v = s.volume_rescaled(zlo, zhi, self.desi_full_area_deg2)
            if v > 0:
                return v
        return 0.0

    def active_desi(self, zlo: float, zhi: float,
                    nbar_min: float = 1e-6) -> dict[str, Survey]:
        """DESI-only active tracers (excludes PFS)."""
        return {n: s for n, s in self.desi_tracers.items()
                if s.nbar_eff(zlo, zhi) > nbar_min}

    def active_with_pfs_truncation(self, zlo: float, zhi: float,
                                    nbar_min: float = 1e-6) -> dict[str, Survey]:
        """All active tracers, dropping PFS above pfs_zmax."""
        active = self.active_tracers(zlo, zhi, nbar_min)
        if zlo >= self.pfs_zmax and "PFS-ELG" in active:
            active.pop("PFS-ELG")
        return active


def default_survey_pair() -> SurveyPair:
    """PFS × DESI-ELG with 1,200 deg² overlap (backward compat)."""
    return SurveyPair(A=pfs_elg(), B=desi_elg(), overlap_area_deg2=1200.0)


def default_survey_group() -> SurveyGroup:
    """PFS × {DESI-ELG, DESI-LRG, DESI-QSO} with 1,200 deg² overlap."""
    return SurveyGroup(
        pfs=pfs_elg(),
        desi_tracers={
            "DESI-ELG": desi_elg(),
            "DESI-LRG": desi_lrg(),
            "DESI-QSO": desi_qso(),
        },
        overlap_area_deg2=1200.0,
    )
