#!/usr/bin/env python
"""Sweep f_shared from 0 to 1 for the full multi-tracer pipeline."""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cli_multitrace import run_multitrace_pipeline

f_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
results = {}

# Broad baseline (same for all f_shared)
cfg0 = ForecastConfig.from_yaml("configs/default.yaml")
cfg0.f_shared_elg = 0.0
res0 = run_multitrace_pipeline(cfg0, verbose=False)
broad = res0.scenario_results.get("broad", {})
print(f"Broad: fsigma8={broad['fsigma8']:.4e}, Mnu={broad['Mnu']:.4e}")

for f_sh in f_values:
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.f_shared_elg = f_sh
    cfg.output_dir = "results/_fshared_sweep"

    res = run_multitrace_pipeline(cfg, verbose=False)

    xcal = res.scenario_results.get("cross-cal-ext", {})
    results[f_sh] = {
        "fsigma8": xcal.get("fsigma8", float("nan")),
        "Mnu": xcal.get("Mnu", float("nan")),
        "Omegam": xcal.get("Omegam", float("nan")),
    }

    imp_fs = (broad["fsigma8"] - results[f_sh]["fsigma8"]) / broad["fsigma8"] * 100
    imp_mn = (broad["Mnu"] - results[f_sh]["Mnu"]) / broad["Mnu"] * 100
    print(f"  f_shared={f_sh:.1f}: sigma(fsig8)={results[f_sh]['fsigma8']:.4e} ({imp_fs:+.1f}%)  "
          f"sigma(Mnu)={results[f_sh]['Mnu']:.4e} ({imp_mn:+.1f}%)")

# Save
out = Path("results/sensitivity")
out.mkdir(parents=True, exist_ok=True)
with open(out / "fshared_sweep.json", "w") as f:
    json.dump({"broad": {k: float(v) for k, v in broad.items()},
               "sweep": {str(k): v for k, v in results.items()}}, f, indent=2)

# Plot
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                      "font.size": 14, "axes.labelsize": 16, "figure.dpi": 150,
                      "xtick.direction": "in", "ytick.direction": "in"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

f_arr = sorted(results.keys())
fs_arr = [results[f]["fsigma8"] for f in f_arr]
mn_arr = [results[f]["Mnu"] for f in f_arr]

ax1.plot(f_arr, fs_arr, "o-", color="#55A868", lw=2, ms=7)
ax1.axhline(broad["fsigma8"], ls="--", color="#4C72B0", lw=1, label="Broad")
ax1.set_xlabel(r"$f_{\rm shared}$")
ax1.set_ylabel(r"$\sigma(f\sigma_8)$")
# No title
ax1.legend(frameon=False)

ax2.plot(f_arr, mn_arr, "o-", color="#DD8452", lw=2, ms=7)
ax2.axhline(broad["Mnu"], ls="--", color="#4C72B0", lw=1, label="Broad")
ax2.set_xlabel(r"$f_{\rm shared}$")
ax2.set_ylabel(r"$\sigma(M_\nu)$ [eV]")
# No title
ax2.legend(frameon=False)

# No suptitle
fig.tight_layout()
(out / "figures").mkdir(exist_ok=True)
fig.savefig(out / "figures" / "fig_fshared_sensitivity.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {out / 'figures' / 'fig_fshared_sensitivity.pdf'}")
