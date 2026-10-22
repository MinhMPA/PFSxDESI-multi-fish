"""Fixed survey geometry and population support within nominal nuisance bins.

Native volumes are integrated using constant volume density inside each native
row for fractional cuts. The revision's nominal/support boundaries align with
native rows. Splitting support never introduces a new model redshift or nuisance
identity, and never changes the underlying reference cosmology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surveys import Survey, SurveyGroup


_TOL = 1e-10


def _validate(survey: Survey) -> None:
    arrays = [np.asarray(getattr(survey, name)) for name in
              ("z_min_all", "z_max_all", "nz_all", "Vz_all")]
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1 or not len(arrays[0]):
        raise ValueError(f"{survey.name}: inconsistent native table arrays")
    lo, hi, density, volume = arrays
    if (not all(np.all(np.isfinite(a)) for a in arrays)
            or not np.isfinite(survey.area_deg2) or survey.area_deg2 <= 0
            or np.any(hi <= lo) or np.any(volume <= 0) or np.any(density < 0)
            or np.any(lo[1:] < hi[:-1] - _TOL)):
        raise ValueError(f"{survey.name}: invalid native geometry/density table")


def _row_weights(survey, zlo, zhi):
    overlap = np.maximum(0., np.minimum(zhi, survey.z_max_all) - np.maximum(zlo, survey.z_min_all))
    return survey.Vz_all * overlap / (survey.z_max_all - survey.z_min_all)


def cell_number_density(survey, zlo, zhi):
    """Native-volume-weighted density, including fractional native rows."""
    weights = _row_weights(survey, zlo, zhi)
    return float(np.average(survey.nz_all, weights=weights)) if weights.sum() > 0 else 0.


@dataclass(frozen=True)
class ShellGeometry:
    """A common shell table, independent of tracer selection/dictionary order."""
    edges: np.ndarray
    volume_per_deg2: np.ndarray

    @classmethod
    def from_group(cls, group: SurveyGroup, rtol=1e-6) -> ShellGeometry:
        if not group.desi_tracers:
            raise ValueError("common geometry requires a DESI shell table")
        for s in group.all_surveys.values():
            _validate(s)
        rows = []
        for name, s in sorted(group.desi_tracers.items()):
            for lo, hi, v in zip(s.z_min_all, s.z_max_all, s.Vz_all):
                rows.append((round(float(lo), 12), round(float(hi), 12), float(v / s.area_deg2), name))
        edges = np.array(sorted({r[i] for r in rows for i in (0, 1)}))
        volumes = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            covering = [r for r in rows if r[0] <= lo + _TOL and r[1] >= hi - _TOL]
            if not covering:
                raise ValueError(f"gap in common geometric support at [{lo}, {hi}]")
            r = min(covering, key=lambda r: (round(r[1] - r[0], 12), r[3]))
            volumes.append(r[2] * (hi - lo) / (r[1] - r[0]))
        geometry = cls(edges, np.array(volumes))
        for name, s in sorted(group.all_surveys.items()):
            native_edges = np.unique(np.concatenate([s.z_min_all, s.z_max_all]))
            common = np.array([e for e in edges if np.any(np.abs(native_edges - e) <= _TOL)])
            for lo, hi in zip(common[:-1], common[1:]):
                # A population's table may have gaps; compare covered intervals only.
                overlap = np.maximum(0., np.minimum(hi, s.z_max_all) - np.maximum(lo, s.z_min_all))
                if not np.isclose(overlap.sum(), hi - lo, rtol=0, atol=_TOL):
                    continue
                native = float(_row_weights(s, lo, hi).sum() / s.area_deg2)
                expected = geometry.volume(lo, hi, 1.)
                if not np.isclose(native, expected, rtol=rtol, atol=0):
                    raise ValueError(f"{name} volume mismatch on [{lo}, {hi}]: {native} vs {expected} per deg2")
        return geometry

    def volume(self, zlo, zhi, area_deg2):
        if (not np.all(np.isfinite([zlo, zhi, area_deg2])) or zhi <= zlo
                or area_deg2 < 0 or zlo < self.edges[0] - _TOL or zhi > self.edges[-1] + _TOL):
            raise ValueError(f"unsupported geometric interval/area: [{zlo}, {zhi}], {area_deg2}")
        overlap = np.maximum(0., np.minimum(zhi, self.edges[1:]) - np.maximum(zlo, self.edges[:-1]))
        return float(np.sum(self.volume_per_deg2 * overlap / np.diff(self.edges)) * area_deg2)


@dataclass(frozen=True)
class SupportCell:
    nominal_bin: tuple[float, float]
    radial_bin: tuple[float, float]
    model_redshift: float
    overlap_tracers: tuple[str, ...]
    nonoverlap_tracers: tuple[str, ...]
    nbar: dict[str, float]
    overlap_volume: float
    nonoverlap_volume: float


def _support_intervals(survey, nbar_min):
    intervals = []
    for lo, hi, n in zip(survey.z_min_all, survey.z_max_all, survey.nz_all):
        if n <= nbar_min:
            continue
        if intervals and abs(intervals[-1][1] - lo) <= _TOL:
            intervals[-1] = (intervals[-1][0], float(hi))
        else:
            intervals.append((float(lo), float(hi)))
    return intervals


def support_cells(group, nominal_bin, *, include_pfs=True, exclude_elg1=False,
                  nbar_min=1e-6, geometry=None) -> list[SupportCell]:
    """Partition only at population-support and selection boundaries.

    Model redshift is always the nominal midpoint, including partial cells.
    The density in a cell is a native-volume-weighted mean for its population.
    """
    if (not np.all(np.isfinite([group.overlap_area_deg2, group.desi_full_area_deg2]))
            or group.desi_full_area_deg2 <= 0 or group.overlap_area_deg2 < 0
            or group.overlap_area_deg2 > min(group.desi_full_area_deg2, group.pfs.area_deg2)):
        raise ValueError("invalid overlap/full survey areas")
    if not np.isfinite(nbar_min) or nbar_min < 0:
        raise ValueError("invalid minimum density")
    geometry = ShellGeometry.from_group(group) if geometry is None else geometry
    zlo, zhi = nominal_bin
    geometry.volume(zlo, zhi, 0.)  # validate interval even for an empty cell
    support = {name: _support_intervals(s, nbar_min) for name, s in group.all_surveys.items()}
    boundaries = {float(zlo), float(zhi)}
    for intervals in support.values():
        boundaries.update(e for interval in intervals for e in interval if zlo < e < zhi)
    if zlo < group.pfs_zmax < zhi:
        boundaries.add(group.pfs_zmax)
    if exclude_elg1:
        boundaries.update(e for e in (.8, 1.1) if zlo < e < zhi)
    edges = sorted(boundaries)
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        names, densities = [], {}
        for name, s in sorted(group.all_surveys.items()):
            if name == "PFS-ELG" and (not include_pfs or lo >= group.pfs_zmax - _TOL):
                continue
            if name == "DESI-ELG" and exclude_elg1 and lo >= .8 - _TOL and hi <= 1.1 + _TOL:
                continue
            if not any(a <= lo + _TOL and b >= hi - _TOL for a, b in support[name]):
                continue
            densities[name] = cell_number_density(s, lo, hi)
            names.append(name)
        desi = tuple(n for n in names if n in group.desi_tracers)
        result.append(SupportCell(
            nominal_bin=tuple(nominal_bin), radial_bin=(lo, hi), model_redshift=(zlo + zhi) / 2,
            overlap_tracers=tuple(names), nonoverlap_tracers=desi, nbar=densities,
            overlap_volume=geometry.volume(lo, hi, group.overlap_area_deg2),
            nonoverlap_volume=geometry.volume(lo, hi, group.desi_full_area_deg2 - group.overlap_area_deg2),
        ))
    return result
