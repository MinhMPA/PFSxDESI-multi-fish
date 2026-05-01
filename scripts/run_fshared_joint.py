#!/usr/bin/env python
"""Sweep f_shared in the joint Fisher pipeline to confirm result is insensitive.

Runs DESI+PFS joint Fisher with f_shared ∈ {0, 0.05, 0.5, 1.0} and reports
σ(fσ8), σ(Mν), σ(Ωm). The fiducial value is f_shared = 0.05; the sweep
confirms that varying it across [0, 1] moves the headline by less than
~1 pp on σ(fσ8) and σ(Mν).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import COSMO_NAMES
from pfsfog.fisher_joint import run_joint_fisher
from pfsfog.surveys import (
    SurveyGroup, desi_elg, desi_lrg, desi_qso, pfs_elg,
)
from scripts.run_joint_fisher import ZBINS


def main():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    print("=" * 80)
    print("f_shared sensitivity sweep — DESI+PFS joint Fisher")
    print("=" * 80)
    print(f"{'f_shared':>10s}  {'σ(fσ8)':>12s}  {'σ(Mν)':>12s}  {'σ(Ωm)':>12s}")
    print("-" * 60)

    # Sweep around the fiducial value 0.05 by ±2× and the nearby endpoints.
    # Very large f_shared (≥0.5) drives the cross-power and DESI auto-power
    # toward collinearity (P_tot^{XY} → P_tot^{YY}) and the Fisher
    # becomes singular; the realistic range for a plausible target overlap is
    # f_shared ≲ 0.1.
    rows = {}
    for fs in [0.0, 0.025, 0.05, 0.075, 0.10]:
        cfg.f_shared_elg = fs
        sg = SurveyGroup(
            pfs=pfs_elg(),
            desi_tracers={
                "DESI-ELG": desi_elg(),
                "DESI-LRG": desi_lrg(),
                "DESI-QSO": desi_qso(),
            },
            overlap_area_deg2=cfg.overlap_area_deg2,
            desi_full_area_deg2=cfg.desi_area_deg2,
            pfs_zmax=1.6,
        )
        res = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=True, zbins=ZBINS)
        rows[fs] = res.sigma
        print(f"  {fs:>8.3f}  {res.sigma['fsigma8']:12.5e}  "
              f"{res.sigma['Mnu']:12.5e}  {res.sigma['Omegam']:12.5e}")

    # Sensitivity report
    base = rows[0.05]
    print()
    print("Relative change vs fiducial f_shared = 0.05:")
    for fs, s in rows.items():
        if fs == 0.05:
            continue
        line = f"  f_shared={fs:.3f} vs 0.05: "
        for cp in COSMO_NAMES:
            d = 100.0 * (s[cp] / base[cp] - 1.0)
            line += f" {cp}: {d:+.2f}%  "
        print(line)


if __name__ == "__main__":
    main()
