#!/usr/bin/env python
"""Two diagnostics:

1) Does PFS×DESI overlap calibration add anything ON TOP of DESI's own
   multi-tracer self-calibration over the full 14,000 deg²?

   Compare:
   E2.  F_calib = DESI multi-tracer over 14,000 deg²   (no PFS anywhere)
   F.   F_calib = DESI multi-tracer over 12,800 deg² (non-overlap)
                 + PFS×DESI multi-tracer over 1,200 deg² (overlap)

   F adds the unique cross-survey info from PFS×DESI without double-counting.
   If σ(Mν, F) ≈ σ(Mν, E2), PFS adds nothing on top of DESI's self-cal.

2) Does the PFS-DESI cross-shot noise term itself matter?
   Run the standard PFS×DESI overlap pipeline (D) with f_shared = 0, 0.045, 1.0.
   The cross-shot is a *noise* term in the cross-power covariance; varying it
   tells us how much it affects the calibration.
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

import scripts.run_desi_multisample as _rdm
from scripts.run_desi_multisample import run_overlap_step1
from scripts.diagnose_no_pfs import run_overlap_no_pfs, make_pfs_with_nbar, step2_cosmo
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, broad_priors


def combine_calibrations(cal_pfs_overlap, cal_desi_nonoverlap):
    """Combine two independent calibrations: 1/σ²_combined = 1/σ²_A + 1/σ²_B per (tracer, z-bin, param).

    For tracers/z-bins present in only one calibration, use that one.
    """
    out = {}
    all_tracers = set(cal_pfs_overlap) | set(cal_desi_nonoverlap)
    for tr in all_tracers:
        out[tr] = {}
        zbins = set(cal_pfs_overlap.get(tr, {})) | set(cal_desi_nonoverlap.get(tr, {}))
        for zb in zbins:
            cA = cal_pfs_overlap.get(tr, {}).get(zb)
            cB = cal_desi_nonoverlap.get(tr, {}).get(zb)
            params = {}
            all_params = set()
            if cA: all_params |= set(cA.params)
            if cB: all_params |= set(cB.params)
            for n in all_params:
                sA = cA.params.get(n) if cA else None
                sB = cB.params.get(n) if cB else None
                if sA is not None and sB is not None and sA > 0 and sB > 0:
                    sigma_sq = 1.0 / (1.0/sA**2 + 1.0/sB**2)
                    params[n] = float(np.sqrt(sigma_sq))
                elif sA is not None and sA > 0:
                    params[n] = sA
                elif sB is not None and sB > 0:
                    params[n] = sB
            from pfsfog.prior_export import CalibratedPriors
            out[tr][zb] = CalibratedPriors(params=params, z_bin=zb, source="combined")
    return out


def run_main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    # ============================================================
    # Diagnostic 1: PFS unique value on top of DESI-fullarea
    # ============================================================
    print(f"\n{'='*100}")
    print("DIAGNOSTIC 1: Does PFS×DESI overlap add anything on top of DESI 14,000 deg² self-cal?")
    print('='*100)
    print(f"{'Config':<60s}  {'Prior':>10s}  {'σ(fσ8)':>10s}{'Δ%':>7s}  "
          f"{'σ(Mν)':>10s}{'Δ%':>7s}  {'σ(Ωm)':>10s}{'Δ%':>7s}")
    print('='*100)

    # E2: DESI multi-tracer at 14,000 deg² (no PFS)
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = 14000.0
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_E2 = run_overlap_no_pfs(cfg, cosmo, ps, verbose=False)
    res_E2 = step2_cosmo(cfg, cosmo, ps, cal_E2)

    # F: DESI at 12,800 deg² (non-overlap) + PFS×DESI at 1,200 deg² (overlap)
    cfg_no = ForecastConfig.from_yaml("configs/default.yaml")
    cfg_no.overlap_area_deg2 = 14000.0 - 1200.0  # 12,800 deg² DESI non-overlap
    cfg_no.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_DESI_nonoverlap = run_overlap_no_pfs(cfg_no, cosmo, ps, verbose=False)

    pfs = make_pfs_with_nbar([(0.6, 1.0, 3e-4), (1.0, 1.6, 4e-4)])
    _surv.pfs_elg = lambda p=pfs: p
    _rdm.pfs_elg = lambda p=pfs: p
    cfg_ov = ForecastConfig.from_yaml("configs/default.yaml")
    cfg_ov.overlap_area_deg2 = 1200.0
    cfg_ov.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_PFS_overlap = run_overlap_step1(cfg_ov, cosmo, ps, verbose=False)

    cal_F = combine_calibrations(cal_PFS_overlap, cal_DESI_nonoverlap)
    res_F = step2_cosmo(cfg_ov, cosmo, ps, cal_F)

    # Reference: D = PFS×DESI at 1,200 deg² alone
    res_D = step2_cosmo(cfg_ov, cosmo, ps, cal_PFS_overlap)

    rows = [
        ("D.  PFS×DESI overlap only (1,200 deg²)",                 res_D),
        ("E2. DESI-only fullarea (14,000 deg²)",                   res_E2),
        ("F.  DESI nonoverlap (12,800) + PFS×DESI overlap (1,200)", res_F),
    ]
    for label, res in rows:
        broad = res["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"  {label:<60s}  {prior:>10s}"
            for cp in COSMO_NAMES:
                s = res[prior][cp]
                imp = 100 * (broad[cp] - s) / broad[cp]
                row += f"  {s:>9.4g}  {imp:+5.1f}%"
            print(row)
        print('-'*100)

    # ============================================================
    # Diagnostic 2: f_shared sensitivity (cross-shot)
    # ============================================================
    print(f"\n\n{'='*100}")
    print("DIAGNOSTIC 2: Does the f_shared cross-shot value matter? (Test at PFS×DESI 1,200 deg²)")
    print('='*100)
    print(f"{'Config':<55s}  {'Prior':>10s}  {'σ(fσ8)':>10s}{'Δ%':>7s}  "
          f"{'σ(Mν)':>10s}{'Δ%':>7s}  {'σ(Ωm)':>10s}{'Δ%':>7s}")
    print('='*100)

    for fs in [0.0, 0.045, 1.0]:
        cfg_fs = ForecastConfig.from_yaml("configs/default.yaml")
        cfg_fs.overlap_area_deg2 = 1200.0
        cfg_fs.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
        cfg_fs.f_shared_elg = fs
        cal_fs = run_overlap_step1(cfg_fs, cosmo, ps, verbose=False)
        res_fs = step2_cosmo(cfg_fs, cosmo, ps, cal_fs)
        broad = res_fs["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"  PFS×DESI 1,200 deg², f_shared={fs:.3f}".ljust(60) + f"{prior:>10s}"
            for cp in COSMO_NAMES:
                s = res_fs[prior][cp]
                imp = 100 * (broad[cp] - s) / broad[cp]
                row += f"  {s:>9.4g}  {imp:+5.1f}%"
            print(row)
        print('-'*100)


if __name__ == "__main__":
    run_main()
