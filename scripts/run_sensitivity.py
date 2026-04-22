#!/usr/bin/env python
"""Sensitivity sweep: σ(fσ8) vs r_σv with and without asymmetric kmax.

Two curves:
  1. Asymmetric kmax: kmax_PFS = kmax_DESI / r_σv  (default)
  2. Symmetric kmax:  kmax_PFS = kmax_DESI = 0.20   (fixed)

The gap between curve 2 and the broad baseline isolates the gain from
counterterm rescaling; the gap between curves 1 and 2 isolates the
additional gain from the asymmetric scale cut.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

from pfsfog.config import ForecastConfig
from pfsfog.cli import run_pipeline


def sweep_r_sigma_v(base_cfg: ForecastConfig, fix_kmax: bool = False) -> dict:
    """Sweep r_σv ∈ {0.5, 0.6, 0.75, 0.9, 1.0}.

    If fix_kmax=True, force kmax_PFS = kmax_DESI (symmetric).
    """
    r_values = [0.5, 0.6, 0.75, 0.9, 1.0]
    results = {}
    label = "symmetric kmax" if fix_kmax else "asymmetric kmax"

    for r in r_values:
        print(f"  r_sigma_v = {r}  ({label})")
        cfg = ForecastConfig(
            r_sigma_v=r,
            output_dir="results/sensitivity_rsv",
            cosmo_backend=base_cfg.cosmo_backend,
        )
        if fix_kmax:
            cfg.kmax_pfs_overlap = cfg.kmax_desi_overlap  # force symmetric
            cfg.kmax_cross_overlap = cfg.kmax_desi_overlap
        res = run_pipeline(cfg, verbose=False)

        xcal = res.scenario_results.get("cross-cal")
        if xcal:
            results[r] = xcal.sigmas_combined["fsigma8"]
            print(f"    σ(fσ8) = {results[r]:.4e}")

    broad = res.scenario_results.get("broad")
    if broad:
        results["broad_baseline"] = broad.sigmas_combined["fsigma8"]

    return results


def main():
    base_cfg = ForecastConfig.from_yaml("configs/default.yaml")
    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)

    print("=== Sweep 1: asymmetric kmax (kmax_PFS = kmax_DESI / r_σv) ===")
    asym = sweep_r_sigma_v(base_cfg, fix_kmax=False)

    print("\n=== Sweep 2: symmetric kmax (kmax_PFS = kmax_DESI = 0.20) ===")
    sym = sweep_r_sigma_v(base_cfg, fix_kmax=True)

    combined = {"asymmetric": asym, "symmetric": sym}
    with open(out / "rsigmav_sweep.json", "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   for k, v in combined.items()}, f, indent=2)
    print(f"\nSaved to {out / 'rsigmav_sweep.json'}")

    # Generate figure
    from pfsfog.plots import fig5_sensitivity_rsigmav
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig5_sensitivity_rsigmav(asym, fig_dir, symmetric_data=sym)
    print(f"Figure saved to {fig_dir}")

    bl = asym.get("broad_baseline")
    if bl:
        for label, data in [("asymmetric", asym), ("symmetric", sym)]:
            best = min(v for k, v in data.items() if isinstance(k, float))
            print(f"  {label}: best σ(fσ8) = {best:.4e}, "
                  f"improvement = {(bl - best) / bl * 100:.1f}%")


if __name__ == "__main__":
    main()
