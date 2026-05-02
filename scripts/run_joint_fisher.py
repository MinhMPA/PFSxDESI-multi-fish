#!/usr/bin/env python
"""Run the joint multi-tracer Fisher analysis: DESI-only vs DESI+PFS.

Replaces the legacy two-stage pipeline (`scripts/run_desi_multisample.py`),
which required a calibration step that exported nuisance σ as Gaussian
priors, then a single-tracer cosmology Fisher with those priors. That
architecture had two methodological issues — auto-spectra double-counting,
and an unfairly weak baseline that ignored DESI's own multi-tracer
self-calibration. The joint Fisher fixes both: one matrix over all data,
marginalize once.

Usage:
    python scripts/run_joint_fisher.py
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


# Z-bin schedule covering the full DESI footprint.
# Each bin is matched to the per-tracer redshift coverage:
#   LRG: 0.4-1.1, ELG: 0.8-1.6, QSO: 0.8-2.1, PFS: 0.6-1.6 (truncated)
ZBINS = [
    (0.4, 0.6),   # LRG only
    (0.6, 0.8),   # LRG + PFS overlap
    (0.8, 1.0),   # LRG + ELG + QSO + PFS overlap
    (1.0, 1.2),   # ditto (LRG drops above z=1.1, picked up partially by nbar_eff)
    (1.2, 1.4),   # ELG + QSO + PFS overlap
    (1.4, 1.6),   # ELG + QSO + PFS overlap
    (1.6, 2.0),   # QSO only (PFS truncated at z=1.6)
    (2.0, 2.1),   # QSO only
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--parallel", action="store_true",
        help="Multi-process the per-z-bin loop (≈1.5–2× wall-clock speedup; "
             "warm JAX cache after first run helps the most).",
    )
    ap.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of worker processes (default: min(len(zbins), cpu_count, 8)).",
    )
    args = ap.parse_args()

    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

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

    print("=" * 80)
    print("Joint multi-tracer Fisher: DESI alone vs DESI+PFS")
    print("=" * 80)
    print(f"Footprint: DESI {sg.desi_full_area_deg2:.0f} deg², "
          f"overlap {sg.overlap_area_deg2:.0f} deg²")
    print(f"PFS truncation: z < {sg.pfs_zmax}")
    if args.parallel:
        print(f"Parallel: True (n_workers={args.n_workers or 'auto'})")
    print()

    kwargs = dict(parallel=args.parallel, n_workers=args.n_workers)

    print("Running DESI-only joint Fisher ...")
    res_desi = run_joint_fisher(cfg, cosmo, ps, sg,
                                include_pfs=False, zbins=ZBINS, **kwargs)

    print("Running DESI+PFS joint Fisher ...")
    res_joint = run_joint_fisher(cfg, cosmo, ps, sg,
                                 include_pfs=True, zbins=ZBINS, **kwargs)

    # --- Reporting ---
    print()
    print(f"{'z-bin':<14s} {'DESI-only active':<35s} {'DESI+PFS active':<40s}")
    print("-" * 90)
    for zb, a_desi, a_joint in zip(
        ZBINS, res_desi.per_zbin_active, res_joint.per_zbin_active
    ):
        print(f"  z={zb}".ljust(14)
              + f"  {','.join(a_desi):<33s}"
              + f"  {','.join(a_joint):<38s}")

    print()
    print(f"{'Cosmo param':<10s}  {'σ DESI-only':>12s}  {'σ DESI+PFS':>12s}  "
          f"{'Δ% (PFS unique)':>18s}")
    print("-" * 60)
    for cp in COSMO_NAMES:
        s_desi = res_desi.sigma[cp]
        s_joint = res_joint.sigma[cp]
        delta = 100.0 * (s_desi - s_joint) / s_desi if s_desi > 0 else 0.0
        print(f"  {cp:<10s}  {s_desi:12.5e}  {s_joint:12.5e}  {delta:+15.2f}%")

    print()
    print("PFS unique contribution = relative tightening of σ when adding "
          "PFS×DESI overlap to the DESI-only joint Fisher.")


if __name__ == "__main__":
    main()
