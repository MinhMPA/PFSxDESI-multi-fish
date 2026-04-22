#!/usr/bin/env python
"""Generate Fig. A1: autodiff vs stencil derivative comparison.

Usage:  python scripts/fig_deriv_validation.py
Output: paper/figs/deriv_autodiff_vs_stencil.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": True})
import matplotlib.pyplot as plt
import numpy as np

from ps_1loop_jax import PowerSpectrum1Loop
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import desi_elg_fiducials
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto
from pfsfog.derivatives import dPell_dtheta_autodiff, dPell_dtheta_stencil
from pfsfog.plots import set_style

OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figs"

# --- setup ---
ps = PowerSpectrum1Loop(do_irres=False)
cosmo = FiducialCosmology(backend="cosmopower")
z_eff = 0.9
s8 = cosmo.sigma8(z_eff)
f_z = float(cosmo.f(z_eff))
h = cosmo.params["h"]
pk_data = cosmo.pk_data(z_eff)

b1 = 1.3
nbar = 4e-4
fid = desi_elg_fiducials(b1, s8)
params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)

k = jnp.arange(0.01, 0.26, 0.005)

# --- parameters and multipoles ---
param_specs = [
    ("b1_sigma8", r"$b_1\sigma_8$"),
    ("c2",        r"$c_2$"),
    ("c_tilde",   r"$\tilde{c}$"),
]
ells = [0, 2]
ell_labels = {0: r"$\ell=0$", 2: r"$\ell=2$"}

set_style()
fig, axes = plt.subplots(
    2 * len(ells), len(param_specs),
    figsize=(4.5 * len(param_specs), 2.2 * 2 * len(ells)),
    gridspec_kw={"height_ratios": [3, 1] * len(ells)},
    sharex="col",
)

k_np = np.asarray(k)

for j, (pname, plabel) in enumerate(param_specs):
    for ie, ell in enumerate(ells):
        ax_main = axes[2 * ie, j]
        ax_res  = axes[2 * ie + 1, j]

        dp_ad = np.asarray(dPell_dtheta_autodiff(
            ps, k, pk_data, params, pname, s8, ell))
        dp_st = np.asarray(dPell_dtheta_stencil(
            ps, k, pk_data, params, pname, s8, ell))

        ax_main.plot(k_np, dp_ad, "-", color="#4C72B0", lw=1.5, label="Autodiff")
        ax_main.plot(k_np, dp_st, "--", color="#DD8452", lw=1.5, label="Stencil")
        if ie == 0 and j == 0:
            ax_main.legend(frameon=False, fontsize=10)
        ax_main.set_ylabel(
            rf"$\partial P_{ell}/\partial${plabel}")
        ax_main.text(0.95, 0.92, ell_labels[ell], transform=ax_main.transAxes,
                     ha="right", va="top", fontsize=12)

        # Fractional difference
        scale = np.maximum(np.abs(dp_ad), np.abs(dp_st))
        mask = scale > 1e-8 * np.max(scale)
        frac = np.full_like(dp_ad, np.nan)
        frac[mask] = (dp_ad[mask] - dp_st[mask]) / scale[mask]

        ax_res.plot(k_np, frac, "-", color="k", lw=0.8)
        ax_res.axhline(0, ls="-", color="gray", lw=0.5)
        ax_res.axhspan(-0.1, 0.1, color="gray", alpha=0.15)
        ax_res.set_ylim(-0.5, 0.5)
        ax_res.set_ylabel("Frac. diff.")

        if ie == len(ells) - 1:
            ax_res.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
        if ie == 0:
            ax_main.set_title(plabel)

fig.tight_layout()
fig.savefig(OUT_DIR / "deriv_autodiff_vs_stencil.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "deriv_autodiff_vs_stencil.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved to {OUT_DIR / 'deriv_autodiff_vs_stencil.pdf'}")
