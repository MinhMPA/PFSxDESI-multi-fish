#!/usr/bin/env python
"""Fisher contour figure: DESI-only joint vs DESI+PFS joint.

Three panels showing 1- and 2-σ ellipses for the cosmology pairs
(b₁σ₈ at z∼0.9, Mν), (b₁σ₈ at z∼0.9, fσ₈), and (fσ₈, Mν). The b₁σ₈
column is the DESI-ELG nuisance in the z=[0.8, 1.0] bin (chosen as
representative; other bins look similar). All three cosmology parameters
are shared globally in the joint Fisher.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": True})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.fisher_joint import run_joint_fisher
from pfsfog.surveys import (
    SurveyGroup, desi_elg, desi_lrg, desi_qso, pfs_elg,
)
from scripts.run_joint_fisher import ZBINS

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs"


def _build():
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
    res_d = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=False, zbins=ZBINS)
    res_j = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=True, zbins=ZBINS)
    return cosmo, res_d, res_j


def _ellipse_panel(ax, Cd, Cj, ix, iy, xc, yc, xlabel, ylabel):
    for C, color, label in [(Cd, "#4C72B0", "DESI-only"),
                            (Cj, "#55A868", "DESI+PFS")]:
        c2 = np.array([[C[ix, ix], C[ix, iy]],
                       [C[iy, ix], C[iy, iy]]])
        vals, vecs = np.linalg.eigh(c2)
        vals = np.maximum(vals, 1e-30)
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for ns, al in [(1, 0.35), (2, 0.12)]:
            w = 2 * ns * np.sqrt(vals[0])
            h = 2 * ns * np.sqrt(vals[1])
            e = Ellipse(xy=(xc, yc), width=w, height=h, angle=ang,
                        fill=True, facecolor=color, alpha=al,
                        edgecolor=color, linewidth=1.5)
            ax.add_patch(e)
        ax.plot([], [], color=color, lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.autoscale_view()
    x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
    dx = (x2 - x1) * 0.15; dy = (y2 - y1) * 0.15
    ax.set_xlim(x1 - dx, x2 + dx)
    ax.set_ylim(y1 - dy, y2 + dy)
    ax.plot(xc, yc, "+", color="k", ms=8, mew=1.5)


def main():
    cosmo, res_d, res_j = _build()

    Cd = np.linalg.inv(res_d.fisher.F)
    Cj = np.linalg.inv(res_j.fisher.F)

    pn_d = res_d.fisher.param_names
    pn_j = res_j.fisher.param_names

    # Picks: representative b1σ8 = DESI-ELG at z=[0.8, 1.0]
    label_b = "b1_sigma8_DESI-ELG_z0.80-1.00"
    ix_d_b = pn_d.index(label_b)
    ix_j_b = pn_j.index(label_b)
    ix_d_f = pn_d.index("fsigma8")
    ix_j_f = pn_j.index("fsigma8")
    ix_d_m = pn_d.index("Mnu")
    ix_j_m = pn_j.index("Mnu")

    # Fiducials at z=0.9
    xf = float(cosmo.f(0.9)) * cosmo.sigma8(0.9)
    xm = 0.06
    xb = desi_elg().b1_of_z(0.9) * cosmo.sigma8(0.9)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Build per-panel sub-covariances by index
    def sub(C, names, names_pair):
        i, j = names.index(names_pair[0]), names.index(names_pair[1])
        return np.array([[C[i, i], C[i, j]], [C[j, i], C[j, j]]])

    # Panel 1: (b1σ8, Mν)
    _ellipse_panel(axes[0], Cd, Cj,
                   ix_d_b, ix_d_m, xb, xm,
                   r"$b_1\sigma_8\;(z{\sim}0.9)$", r"$M_\nu$ [eV]")

    # For DESI+PFS, indices may differ — rebuild for each panel using sub-blocks
    def panel(ax, Cd_pair, Cj_pair, xc, yc, xl, yl):
        for C, color, label in [(Cd_pair, "#4C72B0", "DESI-only"),
                                (Cj_pair, "#55A868", "DESI+PFS")]:
            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-30)
            ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            for ns, al in [(1, 0.35), (2, 0.12)]:
                w = 2 * ns * np.sqrt(vals[0])
                h = 2 * ns * np.sqrt(vals[1])
                e = Ellipse(xy=(xc, yc), width=w, height=h, angle=ang,
                            fill=True, facecolor=color, alpha=al,
                            edgecolor=color, linewidth=1.5)
                ax.add_patch(e)
            ax.plot([], [], color=color, lw=2, label=label)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.autoscale_view()
        x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
        dx = (x2 - x1) * 0.15; dy = (y2 - y1) * 0.15
        ax.set_xlim(x1 - dx, x2 + dx)
        ax.set_ylim(y1 - dy, y2 + dy)
        ax.plot(xc, yc, "+", color="k", ms=8, mew=1.5)

    axes[0].clear()
    panel(axes[0],
          sub(Cd, pn_d, (label_b, "Mnu")),
          sub(Cj, pn_j, (label_b, "Mnu")),
          xb, xm, r"$b_1\sigma_8\;(z{\sim}0.9)$", r"$M_\nu$ [eV]")
    panel(axes[1],
          sub(Cd, pn_d, (label_b, "fsigma8")),
          sub(Cj, pn_j, (label_b, "fsigma8")),
          xb, xf, r"$b_1\sigma_8\;(z{\sim}0.9)$", r"$f\sigma_8$")
    panel(axes[2],
          sub(Cd, pn_d, ("fsigma8", "Mnu")),
          sub(Cj, pn_j, ("fsigma8", "Mnu")),
          xf, xm, r"$f\sigma_8$", r"$M_\nu$ [eV]")

    axes[0].legend(frameon=False, fontsize=12, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "fisher_contours.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT / 'fisher_contours.png'}")


if __name__ == "__main__":
    main()
