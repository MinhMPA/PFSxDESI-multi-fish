"""Publication figures (serif, Computer Modern, ≥14pt)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .eft_params import HOD_BENCHMARK, FIELD_LEVEL_BENCHMARK, broad_priors, NUISANCE_NAMES
from .scenarios import SCENARIOS, compute_calibration_efficiency

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": False,
    })


SCENARIO_COLORS = {
    "broad": "#4C72B0",
    "cross-cal": "#DD8452",
    "cross-cal": "#55A868",
    "oracle": "#C44E52",
}

SCENARIO_LABELS = {
    "broad": "Broad",
    "cross-cal": "Cross-cal",
    "cross-cal": "Cross-cal",
    "oracle": "Fixed nuis.",
}


# ---------------------------------------------------------------------------
# Fig 1: Overlap calibration
# ---------------------------------------------------------------------------

def fig1_overlap_calibration(overlap_results, out_dir: Path):
    """Per-z-bin bar chart: σ / σ_broad for DESI-only, PFS-only, MT.

    Normalised by broad prior width so all parameters are comparable.
    """
    set_style()
    bp = broad_priors().sigma_dict()

    params_to_plot = ["c_tilde", "c0", "Pshot", "a0"]
    param_labels = [r"$\tilde{c}$", r"$c_0$", r"$P_{\rm shot}$", r"$a_0$"]

    z_bins = sorted(overlap_results.keys())
    n_z = len(z_bins)
    n_p = len(params_to_plot)

    fig, axes = plt.subplots(1, n_z, figsize=(3.8 * n_z, 4), sharey=True)
    if n_z == 1:
        axes = [axes]

    for iz, zb in enumerate(z_bins):
        ax = axes[iz]
        ov = overlap_results[zb]

        x = np.arange(n_p)
        w = 0.25
        norms = [bp[p] for p in params_to_plot]
        vals_desi = [ov.sigma_desi_only[p] / n for p, n in zip(params_to_plot, norms)]
        vals_pfs = [ov.sigma_pfs_only[p] / n for p, n in zip(params_to_plot, norms)]
        vals_mt = [ov.sigma_mt[p] / n for p, n in zip(params_to_plot, norms)]

        ax.bar(x - w, vals_desi, w, label="DESI-only", color="#4C72B0")
        ax.bar(x, vals_pfs, w, label="PFS-only", color="#DD8452")
        ax.bar(x + w, vals_mt, w, label=r"PFS$\times$DESI", color="#55A868")

        ax.axhline(1.0, ls="--", color="gray", lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels)
        ax.set_title(f"$z \\in [{zb[0]:.1f}, {zb[1]:.1f}]$")
        if iz == 0:
            ax.set_ylabel(r"$\sigma\,/\,\sigma_{\rm broad}$")
            ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_overlap_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Calibrated vs broad priors
# ---------------------------------------------------------------------------

def fig2_calibrated_vs_broad(overlap_results, out_dir: Path):
    """Error bars: broad vs cross-calibrated prior widths per nuisance param."""
    set_style()

    bp = broad_priors()
    bp_dict = bp.sigma_dict()
    # Include ALL nuisance params, including b1_sigma8 (flat prior)
    params_to_show = list(NUISANCE_NAMES)
    param_labels = {
        "b1_sigma8": r"$b_1\sigma_8$",
        "b2_sigma8sq": r"$b_2\sigma_8^2$",
        "bG2_sigma8sq": r"$b_{G_2}\sigma_8^2$",
        "bGamma3": r"$b_{\Gamma_3}$",
        "c0": r"$c_0$", "c2": r"$c_2$", "c4": r"$c_4$",
        "c_tilde": r"$\tilde{c}$", "c1": r"$c_1$",
        "Pshot": r"$P_{\rm shot}$", "a0": r"$a_0$", "a2": r"$a_2$",
    }

    z_bins = sorted(overlap_results.keys())
    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(params_to_show))
    # Broad priors as bars (skip b1_sigma8 which has flat prior)
    broad_vals = []
    for p in params_to_show:
        s = bp_dict[p]
        broad_vals.append(s if s is not None else np.nan)
    # Plot bars only where broad prior exists
    for i, (p, bv) in enumerate(zip(params_to_show, broad_vals)):
        if not np.isnan(bv):
            ax.bar(i, bv, 0.4, color="#4C72B0", alpha=0.4,
                   label="Broad" if i == 1 else None)

    # Overlay calibrated for each z-bin
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(z_bins)))
    offset_step = 0.08
    for iz, zb in enumerate(z_bins):
        cal = overlap_results[zb].calibrated_priors
        cal_vals = [cal.params.get(p, np.nan) for p in params_to_show]
        offset = (iz - len(z_bins) / 2 + 0.5) * offset_step
        ax.scatter(x + offset, cal_vals, color=colors[iz], s=30, zorder=5,
                   label=f"$z \\in [{zb[0]:.1f},{zb[1]:.1f}]$")

    ax.set_xticks(x)
    ax.set_xticklabels([param_labels.get(p, p) for p in params_to_show],
                        rotation=45, ha="right")
    ax.set_ylabel(r"Prior width $\sigma$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10, ncol=2)
    # No title — caption provides context
    # Annotate b1_sigma8 as "flat prior"
    b1_idx = params_to_show.index("b1_sigma8")
    ax.annotate("flat prior", xy=(b1_idx, 0.5), fontsize=8, ha="center",
                color="gray", style="italic")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_calibrated_vs_broad.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Full-area DESI constraints (money figure)
# ---------------------------------------------------------------------------

def fig3_full_area_constraints(scenario_results, out_dir: Path):
    """Grouped bars: σ(fσ8) and σ(Mν) per scenario + HOD benchmark lines."""
    set_style()

    cosmo_params = ["fsigma8", "Mnu", "Omegam"]
    param_titles = [r"$f\sigma_8$", r"$M_\nu$ [eV]", r"$\Omega_m$"]
    param_ylabels = [
        r"$\sigma(f\sigma_8)$",
        r"$\sigma(M_\nu)$ [eV]",
        r"$\sigma(\Omega_m)$",
    ]
    scenario_names = [s.name for s in SCENARIOS]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ip, (cp, title, ylabel) in enumerate(zip(cosmo_params, param_titles, param_ylabels)):
        ax = axes[ip]
        x = np.arange(len(scenario_names))
        vals = [scenario_results[sn].sigmas_combined[cp] for sn in scenario_names]
        colors = [SCENARIO_COLORS[sn] for sn in scenario_names]

        ax.bar(x, vals, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[sn] for sn in scenario_names],
                           rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # SBP benchmark lines (dashed = PS-level, dotted = field-level)
        sigma_broad = scenario_results["broad"].sigmas_combined[cp]

        if cp == "fsigma8":
            hod_imp = HOD_BENCHMARK["sigma8_improvement"]
            ax.axhline(sigma_broad * (1 - hod_imp), ls="--", color="gray",
                       lw=1.2, label=fr"SBP, PS ($-{hod_imp*100:.0f}$%)")
            fl_imp = FIELD_LEVEL_BENCHMARK["sigma8_improvement"]
            ax.axhline(sigma_broad * (1 - fl_imp), ls=":", color="gray",
                       lw=1.2, label=fr"SBP, FL ($-{fl_imp*100:.0f}$%)")

        if cp == "Omegam":
            hod_imp = HOD_BENCHMARK["Omegam_improvement"]
            ax.axhline(sigma_broad * (1 - hod_imp), ls="--", color="gray", lw=1.2)
            fl_imp = FIELD_LEVEL_BENCHMARK["Omegam_improvement"]
            ax.axhline(sigma_broad * (1 - fl_imp), ls=":", color="gray", lw=1.2)

        # Mnu: no SBP line — Chudaykin+ 2026 Table IV shows SBPs
        # worsen the LCDM Mnu bound (sigma8 shift effect).

    # Single legend from the fsigma8 panel (shared line styles)
    axes[0].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_full_area_constraints.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Calibration efficiency
# ---------------------------------------------------------------------------

def fig4_calibration_efficiency(scenario_results, z_bins, out_dir: Path):
    """Line plot: efficiency = (σ_broad - σ_xcal) / (σ_broad - σ_oracle) per z."""
    set_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    cosmo_params = ["fsigma8", "Mnu", "Omegam"]
    markers = {"fsigma8": "o", "Mnu": "s", "Omegam": "^"}
    labels = {"fsigma8": r"$f\sigma_8$", "Mnu": r"$M_\nu$", "Omegam": r"$\Omega_m$"}

    z_mids = [0.5 * (zb[0] + zb[1]) for zb in z_bins]

    for cp in cosmo_params:
        effs = []
        for zb in z_bins:
            sb = scenario_results["broad"].sigmas_per_z[zb][cp]
            so = scenario_results["oracle"].sigmas_per_z[zb][cp]
            sx = scenario_results["cross-cal"].sigmas_per_z[zb][cp]
            eff = compute_calibration_efficiency(sx, sb, so)
            effs.append(eff if eff is not None else 0.0)

        ax.plot(z_mids, effs, marker=markers[cp], label=labels[cp], lw=1.5)

    ax.set_xlabel(r"$z_\mathrm{eff}$")
    ax.set_ylabel("Calibration efficiency")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, ls=":", color="gray", lw=0.5)
    ax.axhline(0.0, ls=":", color="gray", lw=0.5)
    ax.legend(frameon=False)
    # No title

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_calibration_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Sensitivity to σ_v ratio
# ---------------------------------------------------------------------------

def fig5_sensitivity_rsigmav(
    sensitivity_data: dict, out_dir: Path,
    symmetric_data: dict | None = None,
):
    """σ(fσ8) combined for cross-cal vs r_σv.

    Parameters
    ----------
    sensitivity_data : asymmetric-kmax sweep {r_σv: σ(fσ8)}
    symmetric_data : optional symmetric-kmax sweep {r_σv: σ(fσ8)}
    """
    set_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    def _extract(data):
        r = sorted(k for k in data if isinstance(k, (int, float)))
        return r, [data[k] for k in r]

    # Asymmetric kmax curve
    r_vals, sigma_vals = _extract(sensitivity_data)
    ax.plot(r_vals, sigma_vals, "o-", color="#55A868", lw=2, ms=7,
            label=r"Asymmetric $k_{\max}$")

    # Symmetric kmax curve
    if symmetric_data is not None:
        r_sym, s_sym = _extract(symmetric_data)
        ax.plot(r_sym, s_sym, "s--", color="#C44E52", lw=2, ms=7,
                label=r"Symmetric $k_{\max}$")

    # Broad baseline and reference lines
    broad_baseline = sensitivity_data.get("broad_baseline")
    if broad_baseline is not None:
        ax.axhline(broad_baseline, ls="-", color="#4C72B0",
                   lw=1, label="Broad baseline")
        for pct in (10, 20):
            ax.axhline(broad_baseline * (1 - pct / 100), ls=":",
                       color="gray", lw=0.8,
                       label=fr"$-{pct}$%")

    ax.set_xlabel(r"$r_{\sigma_v} = \sigma_{v,\mathrm{PFS}} / \sigma_{v,\mathrm{DESI}}$")
    ax.set_ylabel(r"$\sigma(f\sigma_8)$ combined")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_sensitivity_rsigmav.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# All figures
# ---------------------------------------------------------------------------

def make_all_figures(results, sensitivity_data: dict | None = None):
    """Generate all 5 figures from pipeline results."""
    out_dir = results.output_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    fig1_overlap_calibration(results.overlap_results, out_dir)
    fig2_calibrated_vs_broad(results.overlap_results, out_dir)
    fig3_full_area_constraints(results.scenario_results, out_dir)
    fig4_calibration_efficiency(
        results.scenario_results, results.config.z_bins, out_dir,
    )
    if sensitivity_data is not None:
        fig5_sensitivity_rsigmav(sensitivity_data, out_dir)

    print(f"Figures saved to {out_dir}")
