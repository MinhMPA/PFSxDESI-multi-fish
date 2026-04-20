#!/usr/bin/env python
"""Generate fig1–fig5 from the most recent pipeline run.

Usage:
    python scripts/make_all_figures.py [results_dir]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

from pfsfog.config import ForecastConfig
from pfsfog.cli import run_pipeline
from pfsfog.plots import make_all_figures


def main():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    results = run_pipeline(cfg, verbose=True)

    # Try to load sensitivity data for Fig 5
    sens_path = Path("results/sensitivity/rsigmav_sweep.json")
    sensitivity_data = None
    if sens_path.exists():
        import json
        with open(sens_path) as f:
            raw = json.load(f)
        sensitivity_data = {}
        for k, v in raw.items():
            try:
                sensitivity_data[float(k)] = v
            except ValueError:
                sensitivity_data[k] = v

    make_all_figures(results, sensitivity_data)


if __name__ == "__main__":
    main()
