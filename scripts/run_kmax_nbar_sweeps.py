#!/usr/bin/env python
"""Sweep improvement vs PFS kmax and PFS nbar.

1. kmax sweep: vary kmax_PFS in the overlap (step 1) while keeping
   full-area kmax_DESI = 0.20 fixed.
2. nbar sweep: scale PFS number density from 0.01x to 2x nominal.

Both should show smooth, monotonically increasing improvement until
saturation.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.surveys import Survey, SurveyGroup, pfs_elg, desi_elg, desi_lrg, desi_qso
from pfsfog.cli_multitrace import run_multitrace_pipeline
import pfsfog.cli_multitrace as cmt


def run_and_extract(cfg, label=""):
    """Run multi-tracer pipeline, return broad and cross-cal sigmas."""
    try:
        res = run_multitrace_pipeline(cfg, verbose=False)
        b = res.scenario_results.get("broad", {})
        x = res.scenario_results.get("cross-cal", {})
        return b, x
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return {}, {}


def main():
    base_cfg = ForecastConfig.from_yaml("configs/default.yaml")

    # =====================================================================
    # Sweep 1: PFS kmax in the overlap
    # =====================================================================
    print("=" * 60)
    print("SWEEP 1: PFS kmax in overlap (full-area kmax_DESI = 0.20)")
    print("=" * 60)

    kmax_values = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    kmax_results = {}

    for kmax_pfs in kmax_values:
        cfg = ForecastConfig.from_yaml("configs/default.yaml")
        cfg.kmax_pfs_overlap = kmax_pfs
        cfg.output_dir = "results/_sweep_kmax"

        b, x = run_and_extract(cfg, f"kmax_PFS={kmax_pfs}")
        if b and x:
            imp_f = (b["fsigma8"] - x["fsigma8"]) / b["fsigma8"] * 100
            imp_m = (b["Mnu"] - x["Mnu"]) / b["Mnu"] * 100
            kmax_results[kmax_pfs] = {
                "fsigma8": x["fsigma8"], "Mnu": x["Mnu"],
                "imp_fsigma8": imp_f, "imp_Mnu": imp_m,
            }
            print(f"  kmax_PFS={kmax_pfs:.2f}: sigma(fsig8)={x['fsigma8']:.4e} ({imp_f:+.1f}%)  "
                  f"sigma(Mnu)={x['Mnu']:.4e} ({imp_m:+.1f}%)")

    # =====================================================================
    # Sweep 2: PFS number density
    # =====================================================================
    print()
    print("=" * 60)
    print("SWEEP 2: PFS nbar (overlap kmax_PFS from r_sigma_v formula)")
    print("=" * 60)

    nbar_scales = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    nbar_results = {}

    pfs_nom = pfs_elg()

    for scale in nbar_scales:
        pfs_scaled = Survey(
            name="PFS-ELG", area_deg2=1200.0,
            z_min_all=pfs_nom.z_min_all, z_max_all=pfs_nom.z_max_all,
            nz_all=pfs_nom.nz_all * scale, Vz_all=pfs_nom.Vz_all,
            b1_of_z=pfs_nom.b1_of_z,
        )

        def make_patched(pfs):
            def p():
                return SurveyGroup(pfs=pfs,
                    desi_tracers={"DESI-ELG": desi_elg(), "DESI-LRG": desi_lrg(),
                                  "DESI-QSO": desi_qso()},
                    overlap_area_deg2=1200.0)
            return p

        cmt.default_survey_group = make_patched(pfs_scaled)

        cfg = ForecastConfig.from_yaml("configs/default.yaml")
        cfg.output_dir = "results/_sweep_nbar"

        b, x = run_and_extract(cfg, f"nbar_scale={scale}")
        if b and x:
            imp_f = (b["fsigma8"] - x["fsigma8"]) / b["fsigma8"] * 100
            imp_m = (b["Mnu"] - x["Mnu"]) / b["Mnu"] * 100
            nbar_results[scale] = {
                "nbar_eff": float(pfs_scaled.nbar_eff(0.8, 1.0)),
                "fsigma8": x["fsigma8"], "Mnu": x["Mnu"],
                "imp_fsigma8": imp_f, "imp_Mnu": imp_m,
            }
            print(f"  nbar_scale={scale:.2f} (nbar~{pfs_scaled.nbar_eff(0.8,1.0):.1e}): "
                  f"sigma(fsig8)={x['fsigma8']:.4e} ({imp_f:+.1f}%)  "
                  f"sigma(Mnu)={x['Mnu']:.4e} ({imp_m:+.1f}%)")

    # =====================================================================
    # Save and plot
    # =====================================================================
    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "kmax_nbar_sweeps.json", "w") as f:
        json.dump({"kmax_sweep": {str(k): v for k, v in kmax_results.items()},
                    "nbar_sweep": {str(k): v for k, v in nbar_results.items()}}, f, indent=2)

    # Plot
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                          "font.size": 14, "axes.labelsize": 16, "figure.dpi": 150,
                          "xtick.direction": "in", "ytick.direction": "in"})

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # kmax sweep
    k_arr = sorted(kmax_results.keys())
    ax = axes[0, 0]
    ax.plot(k_arr, [-kmax_results[k]["imp_fsigma8"] for k in k_arr], "o-", color="#55A868", lw=2, ms=7)
    ax.set_xlabel(r"$k_{\rm max}^{\rm PFS}$ [$h\,{\rm Mpc}^{-1}$]")
    ax.set_ylabel(r"Improvement on $\sigma(f\sigma_8)$ [\%]")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.plot(k_arr, [-kmax_results[k]["imp_Mnu"] for k in k_arr], "o-", color="#DD8452", lw=2, ms=7)
    ax.set_xlabel(r"$k_{\rm max}^{\rm PFS}$ [$h\,{\rm Mpc}^{-1}$]")
    ax.set_ylabel(r"Improvement on $\sigma(M_\nu)$ [\%]")
    ax.invert_yaxis()

    # nbar sweep
    n_arr = sorted(nbar_results.keys())
    nbar_vals = [nbar_results[n]["nbar_eff"] for n in n_arr]

    ax = axes[1, 0]
    ax.plot(nbar_vals, [-nbar_results[n]["imp_fsigma8"] for n in n_arr], "o-", color="#55A868", lw=2, ms=7)
    ax.set_xlabel(r"$\bar{n}_{\rm PFS}$ [$h^3\,{\rm Mpc}^{-3}$]")
    ax.set_ylabel(r"Improvement on $\sigma(f\sigma_8)$ [\%]")
    ax.set_xscale("log")
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.plot(nbar_vals, [-nbar_results[n]["imp_Mnu"] for n in n_arr], "o-", color="#DD8452", lw=2, ms=7)
    ax.set_xlabel(r"$\bar{n}_{\rm PFS}$ [$h^3\,{\rm Mpc}^{-3}$]")
    ax.set_ylabel(r"Improvement on $\sigma(M_\nu)$ [\%]")
    ax.set_xscale("log")
    ax.invert_yaxis()

    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "kmax_nbar_sweeps.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out / 'figures' / 'kmax_nbar_sweeps.pdf'}")


if __name__ == "__main__":
    main()
