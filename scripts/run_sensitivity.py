#!/usr/bin/env python
"""Sensitivity sweeps: vary r_σv, kmax, overlap area.

Outputs a JSON file with σ(fσ8)_combined for each parameter point.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

from pfsfog.config import ForecastConfig
from pfsfog.cli import run_pipeline


def sweep_r_sigma_v(base_cfg: ForecastConfig) -> dict:
    """Sweep r_σv ∈ {0.5, 0.6, 0.75, 0.9, 1.0}."""
    r_values = [0.5, 0.6, 0.75, 0.9, 1.0]
    results = {}

    for r in r_values:
        print(f"\n{'='*60}")
        print(f"  r_sigma_v = {r}")
        print(f"{'='*60}")
        cfg = ForecastConfig(
            r_sigma_v=r,
            output_dir=f"results/sensitivity_rsv",
            cosmo_backend=base_cfg.cosmo_backend,
        )
        res = run_pipeline(cfg, verbose=False)

        xcal_ext = res.scenario_results.get("cross-cal")
        if xcal_ext:
            results[r] = xcal_ext.sigmas_combined["fsigma8"]
            print(f"  σ(fσ8) cross-cal = {results[r]:.4e}")

    # Add broad baseline
    broad = res.scenario_results.get("broad")
    if broad:
        results["broad_baseline"] = broad.sigmas_combined["fsigma8"]

    return results


def main():
    base_cfg = ForecastConfig.from_yaml("configs/default.yaml")

    print("=== Sensitivity sweep: r_sigma_v ===")
    rsv_results = sweep_r_sigma_v(base_cfg)

    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "rsigmav_sweep.json", "w") as f:
        json.dump({str(k): v for k, v in rsv_results.items()}, f, indent=2)

    print(f"\nSaved to {out / 'rsigmav_sweep.json'}")

    # Generate Fig 5
    from pfsfog.plots import fig5_sensitivity_rsigmav
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    # rsv_results already has float keys + "broad_baseline" string key
    fig5_sensitivity_rsigmav(rsv_results, fig_dir)
    print(f"Fig 5 saved to {fig_dir}")

    # Also show the broad baseline for context
    bl = rsv_results.get("broad_baseline")
    if bl is not None:
        print(f"\nBroad baseline σ(fσ8) = {bl:.4e}")
        best = min(v for k, v in rsv_results.items() if isinstance(k, float))
        print(f"Best cross-cal σ(fσ8) = {best:.4e}")
        print(f"Improvement: {(bl - best) / bl * 100:.1f}%")


if __name__ == "__main__":
    main()
