#!/usr/bin/env python
"""LEGACY single-ELG end-to-end driver: overlap → priors → full-area
Fisher → all main-text figures (legacy two-stage pipeline only).

The recommended pipeline for new work is the joint Fisher
(``scripts/run_joint_fisher.py``), which spans all four tracers in the
overlap and the full DESI footprint in a single Fisher matrix. This
script is retained for reproducibility of the original two-stage
results.
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
    make_all_figures(results)


if __name__ == "__main__":
    main()
