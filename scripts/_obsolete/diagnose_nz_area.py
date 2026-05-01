#!/usr/bin/env python
"""Diagnostic dump for n(z) × area comparison.

Prints intermediate quantities to verify the inputs to Step 1 calibration
actually differ across configs:
  - nbar_eff per (tracer, z-bin)
  - V_overlap per z-bin
  - kmax_PFS / kmax_DESI
  - cross-shot P_shot^cross
  - calibrated σ per nuisance per z-bin (Step 1 output)
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
from pfsfog.surveys import Survey, load_nz_table, SurveyGroup, desi_elg, desi_lrg, desi_qso

import scripts.run_desi_multisample as _rdm
from scripts.run_desi_multisample import run_overlap_step1


def make_pfs_survey(nz_file: str) -> Survey:
    z_min, z_max, nz, Vz = load_nz_table(Path(__file__).resolve().parent.parent
                                         / "survey_specs" / nz_file)
    return Survey(
        name="PFS-ELG",
        area_deg2=1200.0,
        z_min_all=z_min, z_max_all=z_max, nz_all=nz, Vz_all=Vz,
        b1_of_z=lambda z: 0.9 + 0.4 * z,
    )


def report(nz_file: str, area: float, cosmo, ps):
    print(f"\n{'#'*78}")
    print(f"## CONFIG: {nz_file} @ {area} deg²")
    print(f"{'#'*78}")

    pfs = make_pfs_survey(nz_file)
    _surv.pfs_elg = lambda: pfs
    _rdm.pfs_elg = lambda: pfs

    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = area
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]

    sg = SurveyGroup(
        pfs=pfs,
        desi_tracers={"DESI-ELG": desi_elg(),
                      "DESI-LRG": desi_lrg(),
                      "DESI-QSO": desi_qso()},
        overlap_area_deg2=area,
    )

    # Inputs per z-bin
    print(f"\n  kmax: DESI={cfg.kmax_desi_overlap}, PFS={cfg.compute_kmax_pfs():.3f}, "
          f"cross={cfg.compute_kmax_cross():.3f}")
    print(f"  f_shared = {cfg.f_shared_elg}")
    print(f"\n  Per-z-bin inputs:")
    print(f"  {'z-bin':<14s}{'V_ov [Gpc^3]':>14s}  "
          f"{'PFS n̄':>9s}{'ELG n̄':>9s}{'LRG n̄':>9s}{'QSO n̄':>9s}  "
          f"{'P_shot_cross':>14s}")
    for zlo, zhi in cfg.z_bins:
        active = sg.active_tracers(zlo, zhi)
        nbars = {n: s.nbar_eff(zlo, zhi) for n, s in active.items()}
        Vov = sg.V_overlap(zlo, zhi)
        cross_shot = (cfg.f_shared_elg / nbars["DESI-ELG"]
                      if "DESI-ELG" in nbars and "PFS-ELG" in nbars else 0.0)
        row = f"  z=[{zlo},{zhi}]  {Vov/1e9:>12.3f}  "
        for tr in ["PFS-ELG", "DESI-ELG", "DESI-LRG", "DESI-QSO"]:
            n = nbars.get(tr, 0.0) * 1e4
            row += f"{n:>9.2f}"
        row += f"  {cross_shot:>14.2f}"
        print(row)

    # Run Step 1
    cal = run_overlap_step1(cfg, cosmo, ps, verbose=False)

    # Calibrated prior per nuisance per z-bin per tracer
    from pfsfog.eft_params import NUISANCE_NAMES, broad_priors
    bp = broad_priors().sigma_dict()
    print(f"\n  Calibrated σ per nuisance (Step 1 output, σ_cal/σ_broad):")
    nuis_show = ["b1_sigma8", "b2_sigma8sq", "bG2_sigma8sq",
                 "c0", "c2", "c_tilde", "Pshot"]
    header = f"  {'tracer':<10s}{'z-bin':<12s}"
    for n in nuis_show:
        header += f"{n:>14s}"
    print(header)
    for tr in ["DESI-ELG", "DESI-LRG", "DESI-QSO"]:
        if tr not in cal:
            continue
        for zb in sorted(cal[tr].keys()):
            row = f"  {tr:<10s}{str(zb):<12s}"
            for n in nuis_show:
                s = cal[tr][zb].params.get(n, np.nan)
                bps = bp.get(n)
                if bps is None or s != s:
                    row += f"{'---':>14s}"
                else:
                    row += f"  σ={s:.2f} ({s/bps:.2f})".rjust(14)
            print(row)


def main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    configs = [
        ("PFS_nz_pfs_baseline_fine.txt",     1200.0),
        ("PFS_nz_pfs_baseline_fine.txt",     1400.0),
        ("PFS_nz_pfs_hypothetical_fine.txt", 1200.0),
        ("PFS_nz_pfs_hypothetical_fine.txt", 1400.0),
    ]
    for nz_file, area in configs:
        report(nz_file, area, cosmo, ps)


if __name__ == "__main__":
    main()
