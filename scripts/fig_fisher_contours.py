#!/usr/bin/env python
"""Fisher contour figure: broad single-tracer vs DESI-only joint vs DESI+PFS joint.

Three panels showing 1- and 2-sigma ellipses for the cosmology pairs
(b1*sigma8 at z~0.9, M_nu), (b1*sigma8 at z~0.9, f*sigma8), and (f*sigma8, M_nu).

The b1*sigma8 column is the DESI-ELG nuisance in the z=[0.8, 1.0] bin
(representative; other bins look similar). All three cosmology
parameters are shared globally.

The three scenarios:
- DESI single-tracer broad: each (DESI tracer, z-bin) analyzed independently
  with broad nuisance priors. The b1*sigma8-Mnu degeneracy runs free.
- DESI-only joint: multi-tracer Fisher across LRG, ELG, QSO with internal
  cross-correlations. The McDonald-Seljak mechanism breaks the
  b1*sigma8 degeneracy from data.
- DESI+PFS joint: adds PFS-ELG in the 1,200 deg^2 overlap. Further
  tightens b1*sigma8 and Mnu via the 4-tracer cross-correlations.
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
from pfsfog.fisher_joint import (
    run_broad_baseline, run_joint_fisher,
)
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
    res_b = run_broad_baseline(cfg, cosmo, ps, sg, zbins=ZBINS)
    res_d = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=False, zbins=ZBINS)
    res_j = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=True, zbins=ZBINS)
    return cosmo, res_b, res_d, res_j


def main():
    cosmo, res_b, res_d, res_j = _build()

    Cb = np.linalg.inv(res_b.fisher.F)
    Cd = np.linalg.inv(res_d.fisher.F)
    Cj = np.linalg.inv(res_j.fisher.F)

    pn_b = res_b.fisher.param_names
    pn_d = res_d.fisher.param_names
    pn_j = res_j.fisher.param_names

    # Representative b1*sigma8: DESI-ELG at z=[0.8, 1.0]
    label_b = "b1_sigma8_DESI-ELG_z0.80-1.00"

    # Fiducials at z=0.9
    xf = float(cosmo.f(0.9)) * cosmo.sigma8(0.9)
    xm = 0.06
    xb = desi_elg().b1_of_z(0.9) * cosmo.sigma8(0.9)

    def sub(C, names, names_pair):
        i, j = names.index(names_pair[0]), names.index(names_pair[1])
        return np.array([[C[i, i], C[i, j]], [C[j, i], C[j, j]]])

    def panel(ax, sub_b, sub_d, sub_j, xc, yc, xl, yl):
        scenarios = [
            (sub_b, "#C44E52", "DESI broad (single-tracer)"),
            (sub_d, "#4C72B0", "DESI-only joint"),
            (sub_j, "#55A868", "DESI+PFS joint"),
        ]
        for Cpair, color, label in scenarios:
            vals, vecs = np.linalg.eigh(Cpair)
            vals = np.maximum(vals, 1e-30)
            ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            for ns, al in [(1, 0.30), (2, 0.10)]:
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panel(axes[0],
          sub(Cb, pn_b, (label_b, "Mnu")),
          sub(Cd, pn_d, (label_b, "Mnu")),
          sub(Cj, pn_j, (label_b, "Mnu")),
          xb, xm, r"$b_1\sigma_8\;(z{\sim}0.9)$", r"$M_\nu$ [eV]")
    panel(axes[1],
          sub(Cb, pn_b, (label_b, "fsigma8")),
          sub(Cd, pn_d, (label_b, "fsigma8")),
          sub(Cj, pn_j, (label_b, "fsigma8")),
          xb, xf, r"$b_1\sigma_8\;(z{\sim}0.9)$", r"$f\sigma_8$")
    panel(axes[2],
          sub(Cb, pn_b, ("fsigma8", "Mnu")),
          sub(Cd, pn_d, ("fsigma8", "Mnu")),
          sub(Cj, pn_j, ("fsigma8", "Mnu")),
          xf, xm, r"$f\sigma_8$", r"$M_\nu$ [eV]")

    axes[0].legend(frameon=False, fontsize=11, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "fisher_contours.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {OUT / 'fisher_contours.png'}")


if __name__ == "__main__":
    main()
