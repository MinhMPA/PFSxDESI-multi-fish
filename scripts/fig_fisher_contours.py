#!/usr/bin/env python
"""Generate Fisher contour figure: broad vs cross-cal ellipses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": True})
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, broad_priors, desi_elg_fiducials
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.surveys import desi_elg
from pfsfog.prior_export import calibrated_prior_fisher_diag
from pfsfog.cli import run_pipeline

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs"

cfg = ForecastConfig.from_yaml("configs/default.yaml")
results = run_pipeline(cfg, verbose=False)
cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
from ps_1loop_jax import PowerSpectrum1Loop
ps = PowerSpectrum1Loop(do_irres=False)
ells = (0, 2, 4); kmax = 0.20
k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)
desi_s = desi_elg()

def build(scenario):
    fishers = []
    for zlo, zhi in cfg.z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff); f_z = float(cosmo.f(z_eff)); h = cosmo.params["h"]
        nbar = desi_s.nbar_eff(zlo, zhi)
        if nbar == 0: continue
        b1 = desi_s.b1_of_z(z_eff); V = desi_s.volume(zlo, zhi)
        pk_data = cosmo.pk_data(z_eff)
        fid = desi_elg_fiducials(b1, s8)
        params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
        derivs = dPell_dtheta_autodiff_all(ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
        derivs.update(dPell_d_cosmo_all(ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells))
        pkmu = make_ps1loop_pkmu_func(ps, pk_data, params)
        cov = single_tracer_cov(pkmu, k, nbar, V, cfg.dk, ells)
        if scenario == "broad":
            npd = broad_priors().prior_fisher_diag()
        else:
            ov = results.overlap_results.get((zlo, zhi))
            npd = calibrated_prior_fisher_diag(ov.calibrated_priors) if ov else broad_priors().prior_fisher_diag()
        fishers.append(full_area_fisher_per_zbin(derivs, cov, k, cfg.dk, npd, (zlo, zhi), kmax, ells))
    return combine_zbins(fishers, cfg.z_bins)

print("Building Fisher matrices...")
fr_b = build("broad"); fr_c = build("cross-cal")
Cb = np.linalg.inv(fr_b.F); Cc = np.linalg.inv(fr_c.F)
pn = fr_b.param_names
i_f = pn.index("fsigma8"); i_m = pn.index("Mnu"); i_b = pn.index("b1_sigma8_z0.8_1.0")

xf = float(cosmo.f(0.9)) * cosmo.sigma8(0.9)
xm = 0.06
xb = desi_s.b1_of_z(0.9) * cosmo.sigma8(0.9)

panels = [
    (i_b, i_m, r"$b_1\sigma_8\;(z{=}0.9)$", r"$M_\nu$ [eV]", xb, xm),
    (i_b, i_f, r"$b_1\sigma_8\;(z{=}0.9)$", r"$f\sigma_8$", xb, xf),
    (i_f, i_m, r"$f\sigma_8$", r"$M_\nu$ [eV]", xf, xm),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

for ax, (ix, iy, xl, yl, xc, yc) in zip(axes, panels):
    for C, col, lab in [(Cb, "#4C72B0", "Broad"), (Cc, "#55A868", "Cross-cal")]:
        c2 = np.array([[C[ix, ix], C[ix, iy]], [C[iy, ix], C[iy, iy]]])
        vals, vecs = np.linalg.eigh(c2)
        vals = np.maximum(vals, 1e-30)
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for ns, al in [(1, 0.35), (2, 0.12)]:
            w = 2 * ns * np.sqrt(vals[0])
            h = 2 * ns * np.sqrt(vals[1])
            e = Ellipse(xy=(xc, yc), width=w, height=h, angle=ang,
                        fill=True, facecolor=col, alpha=al,
                        edgecolor=col, linewidth=1.5)
            ax.add_patch(e)
        ax.plot([], [], color=col, lw=2, label=lab)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.autoscale_view()
    x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
    dx = (x2 - x1) * 0.15; dy = (y2 - y1) * 0.15
    ax.set_xlim(x1 - dx, x2 + dx)
    ax.set_ylim(y1 - dy, y2 + dy)
    ax.plot(xc, yc, "+", color="k", ms=8, mew=1.5)

axes[0].legend(frameon=False, fontsize=12, loc="upper right")

fig.tight_layout()
fig.savefig(OUT / "fisher_contours.png", bbox_inches="tight", dpi=300)
fig.savefig(OUT / "fisher_contours.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved to {OUT / 'fisher_contours.png'}")
