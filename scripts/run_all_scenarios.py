#!/usr/bin/env python
"""End-to-end: overlap → export → full area → summary + figures."""

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
    make_all_figures(results)


if __name__ == "__main__":
    main()
