#!/usr/bin/env python
"""Stress test: extreme n(z) configs to see if Step 1 actually responds.

Configs:
  A. Baseline (Takada 3,3,4,4,4 ×10⁻⁴)
  B. Hypothetical (3,6,8,8,6 ×10⁻⁴)
  C. EXTREME (3, 60, 80, 80, 60 ×10⁻⁴) — 20× hypothetical
  D. STARVED (3, 0.3, 0.4, 0.4, 0.4 ×10⁻⁴) — 0.1× baseline

Reports σ_cal for b1_sigma8 (which lacks broad prior) plus other params.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from pfsfog import surveys as _surv
from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import Survey, load_nz_table

import scripts.run_desi_multisample as _rdm
from scripts.run_desi_multisample import run_overlap_step1


def make_pfs_with_nbar(zedges_nbar):
    """Build a fine-binned PFS Survey with piecewise nbar."""
    data = np.loadtxt("survey_specs/PFS_nz_pfs_fine.txt", comments="#")
    zmin, zmax, _, Vz = data.T
    mask = zmax <= 1.6 + 1e-9
    zmin_t, zmax_t, Vz_t = zmin[mask], zmax[mask], Vz[mask]
    zmid_t = 0.5 * (zmin_t + zmax_t)

    nz = np.zeros_like(zmid_t)
    for zlo, zhi, n in zedges_nbar:
        m = (zmid_t >= zlo) & (zmid_t < zhi)
        nz[m] = n

    return Survey(
        name="PFS-ELG",
        area_deg2=1200.0,
        z_min_all=zmin_t, z_max_all=zmax_t, nz_all=nz, Vz_all=Vz_t,
        b1_of_z=lambda z: 0.9 + 0.4 * z,
    )


def report(label: str, pfs_survey: Survey, area: float, cosmo, ps):
    print(f"\n{'#'*78}")
    print(f"## {label}  (area={area} deg²)")
    print(f"## PFS n̄ at fine bins z=0.85, 1.05, 1.25, 1.45 (×10⁻⁴):", end=" ")
    for ztest in [0.85, 1.05, 1.25, 1.45]:
        n = pfs_survey.nbar_of_z(ztest) * 1e4
        print(f"{n:.2f}", end=" ")
    print()
    print('#'*78)

    _surv.pfs_elg = lambda: pfs_survey
    _rdm.pfs_elg = lambda: pfs_survey

    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = area
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]

    cal = run_overlap_step1(cfg, cosmo, ps, verbose=False)

    nuis_show = ["b1_sigma8", "b2_sigma8sq", "bG2_sigma8sq",
                 "c0", "c2", "c_tilde", "Pshot"]
    print(f"\n  σ_cal absolute values (smaller = better calibrated):")
    header = f"  {'tracer':<10s}{'z-bin':<14s}"
    for n in nuis_show:
        header += f"{n:>14s}"
    print(header)
    for tr in ["DESI-ELG", "DESI-LRG", "DESI-QSO"]:
        if tr not in cal:
            continue
        for zb in sorted(cal[tr].keys()):
            row = f"  {tr:<10s}{str(zb):<14s}"
            for n in nuis_show:
                s = cal[tr][zb].params.get(n)
                if s is None:
                    row += f"{'---':>14s}"
                else:
                    row += f"{s:>14.4g}"
            print(row)


def main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    configs = [
        ("BASELINE (Takada 3,3,4,4,4)",
         [(0.6, 1.0, 3.0e-4), (1.0, 1.6, 4.0e-4)], 1200.0),
        ("HYPOTHETICAL (3,6,8,8,6)",
         [(0.6,0.8,3e-4),(0.8,1.0,6e-4),(1.0,1.2,8e-4),(1.2,1.4,8e-4),(1.4,1.6,6e-4)], 1200.0),
        ("EXTREME 20× (3,60,80,80,60)",
         [(0.6,0.8,3e-4),(0.8,1.0,60e-4),(1.0,1.2,80e-4),(1.2,1.4,80e-4),(1.4,1.6,60e-4)], 1200.0),
        ("STARVED 0.1× (0.3,0.3,0.4,0.4,0.4)",
         [(0.6,1.0,0.3e-4),(1.0,1.6,0.4e-4)], 1200.0),
    ]
    for label, edges, area in configs:
        pfs = make_pfs_with_nbar(edges)
        report(label, pfs, area, cosmo, ps)


if __name__ == "__main__":
    main()
