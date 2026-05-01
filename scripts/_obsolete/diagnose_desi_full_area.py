#!/usr/bin/env python
"""Rigorous test: DESI-only multi-tracer over the FULL 14,000 deg² footprint.

If DESI-only-fullarea calibration already saturates σ(Mν), then PFS×DESI
overlap has near-zero unique value-add and the paper's premise collapses.

Configs:
  E1. DESI-only Step 1 over 1,200 deg² (overlap area)
  E2. DESI-only Step 1 over 14,000 deg² (DESI full footprint)
  D.  PFS×DESI overlap Step 1 (Takada baseline, 1,200 deg²) — for reference
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog import surveys as _surv
from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import Survey

import scripts.run_desi_multisample as _rdm
from scripts.run_desi_multisample import (
    DR2_SAMPLES, SURVEY_CONSTRUCTORS,
    run_overlap_step1, average_calibrated_prior_diag,
)
from scripts.diagnose_no_pfs import run_overlap_no_pfs, make_pfs_with_nbar, step2_cosmo
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, broad_priors, tracer_fiducials,
)


def main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    print(f"\n{'='*100}")
    print(f"{'Config':<55s}  {'Prior':>10s}  {'σ(fσ8)':>10s}{'Δ%':>7s}  "
          f"{'σ(Mν)':>10s}{'Δ%':>7s}  {'σ(Ωm)':>10s}{'Δ%':>7s}")
    print('='*100)

    # === E1: DESI-only Step 1 at 1200 deg² (overlap area) ===
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = 1200.0
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_E1 = run_overlap_no_pfs(cfg, cosmo, ps, verbose=False)
    res_E1 = step2_cosmo(cfg, cosmo, ps, cal_E1)

    # === E2: DESI-only Step 1 at 14,000 deg² (full DESI) ===
    cfg2 = ForecastConfig.from_yaml("configs/default.yaml")
    cfg2.overlap_area_deg2 = 14000.0
    cfg2.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_E2 = run_overlap_no_pfs(cfg2, cosmo, ps, verbose=False)
    res_E2 = step2_cosmo(cfg2, cosmo, ps, cal_E2)

    # === D: PFS×DESI overlap Step 1 (Takada baseline) — for comparison ===
    pfs = make_pfs_with_nbar([(0.6, 1.0, 3e-4), (1.0, 1.6, 4e-4)])
    _surv.pfs_elg = lambda p=pfs: p
    _rdm.pfs_elg = lambda p=pfs: p
    cfg3 = ForecastConfig.from_yaml("configs/default.yaml")
    cfg3.overlap_area_deg2 = 1200.0
    cfg3.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal_D = run_overlap_step1(cfg3, cosmo, ps, verbose=False)
    res_D = step2_cosmo(cfg3, cosmo, ps, cal_D)

    rows = [
        ("E1. DESI-only Step 1 @ 1,200 deg²",          res_E1),
        ("E2. DESI-only Step 1 @ 14,000 deg² (FULL)",  res_E2),
        ("D.  PFS×DESI Step 1 @ 1,200 deg² (TAKADA)",  res_D),
    ]
    for label, res in rows:
        broad = res["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"  {label:<55s}  {prior:>10s}"
            for cp in COSMO_NAMES:
                s = res[prior][cp]
                imp = 100 * (broad[cp] - s) / broad[cp]
                row += f"  {s:>9.4g}  {imp:+5.1f}%"
            print(row)
        print('-'*100)

    # Show calibrated σ for b1_sigma8 across configs
    print(f"\n\n{'='*70}")
    print("Calibrated σ(b1_sigma8) per (tracer, z-bin) — DESI-ELG focus")
    print('='*70)
    print(f"  {'z-bin':<14s}{'E1: 1200 deg²':>16s}{'E2: 14,000 deg²':>18s}{'D: PFSxDESI':>16s}")
    for zb in sorted(set(cal_E1.get("DESI-ELG", {}).keys())
                     | set(cal_E2.get("DESI-ELG", {}).keys())
                     | set(cal_D.get("DESI-ELG", {}).keys())):
        e1 = cal_E1.get("DESI-ELG", {}).get(zb)
        e2 = cal_E2.get("DESI-ELG", {}).get(zb)
        d  = cal_D.get("DESI-ELG", {}).get(zb)
        s1 = e1.params.get("b1_sigma8") if e1 else None
        s2 = e2.params.get("b1_sigma8") if e2 else None
        sd = d.params.get("b1_sigma8")  if d  else None
        row = f"  {str(zb):<14s}"
        for s in [s1, s2, sd]:
            row += f"{s:>16.5g}" if s is not None else f"{'---':>16s}"
        print(row)


if __name__ == "__main__":
    main()
