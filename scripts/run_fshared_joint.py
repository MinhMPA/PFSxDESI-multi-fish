#!/usr/bin/env python
"""Legacy sweep of f_shared (cross-shot fraction) in the joint Fisher.

This sweep exercises the LEGACY cross-stochasticity treatment where
the PFS×DESI-ELG cross-shot is fixed at f_shared/nbar_DESI-ELG and
no cross-stoch parameter is marginalized. The current default
(cfg.marginalize_cross_stoch=True) makes f_shared inert because the
fiducial cross-shot in the covariance is set to zero and the
cross-stochastic Lagrangian is instead marginalized via free
Pshot_cross / a2_cross parameters per cross-pair (Ebina & White
2024). This script forces marginalize_cross_stoch=False so the
sweep produces a meaningful sensitivity diagnostic.

Reports σ(fσ8), σ(Mν), σ(Ωm) at f_shared ∈ {0, 0.025, 0.05, 0.075, 0.10}.
With cross-stoch marginalization disabled, varying f_shared across
[0, 0.1] moves the headline σ by less than ~1 pp on σ(fσ8) and
σ(Mν) — confirming that the catalog-overlap fraction itself is not
the driver of the multi-tracer information.
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
    # Legacy diagnostic: disable the cross-stoch marginalization so
    # f_shared has a measurable effect on the cov.
    cfg.marginalize_cross_stoch = False
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
