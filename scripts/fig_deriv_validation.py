#!/usr/bin/env python
"""Generate Fig. A1: derivative landscape + autodiff validation.

Left:  Normalized derivatives |dP_0/dtheta| / P_0 for all 15 parameters.
Right: Fractional autodiff-vs-stencil difference for all parameters.

Usage:  python scripts/fig_deriv_validation.py
Output: paper/figs/deriv_autodiff_vs_stencil.{pdf,png}
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
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, desi_elg_fiducials
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import (
    dPell_dtheta_autodiff, dPell_dtheta_stencil,
    dPell_d_cosmo_all,
)

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs"

# --- setup ---
ps = PowerSpectrum1Loop(do_irres=False)
cosmo = FiducialCosmology(backend="cosmopower")
z_eff = 0.9
s8 = cosmo.sigma8(z_eff)
f_z = float(cosmo.f(z_eff))
h = cosmo.params["h"]
pk_data = cosmo.pk_data(z_eff)

b1, nbar = 1.3, 4e-4
fid = desi_elg_fiducials(b1, s8)
params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
k = jnp.arange(0.01, 0.26, 0.005)
k_np = np.asarray(k)

# Fiducial P_0
P0_fid = np.asarray(ps.get_pk_ell(k, 0, pk_data, params))

# --- compute all derivatives (monopole) ---
ell = 0
all_params = list(COSMO_NAMES) + list(NUISANCE_NAMES)

# Cosmological derivatives (stencil-based)
cosmo_derivs = {}
from pfsfog.derivatives import dPell_d_fsigma8, dPell_d_cosmo_stencil
cosmo_derivs["fsigma8"] = np.asarray(dPell_d_fsigma8(ps, k, pk_data, params, s8, ell))
for cp in ("Omegam", "Mnu"):
    cosmo_derivs[cp] = np.asarray(dPell_d_cosmo_stencil(
        ps, k, cosmo, params, cp, z_eff, s8, ell))

# Nuisance derivatives (autodiff + stencil)
nuis_ad = {}
nuis_st = {}
for pname in NUISANCE_NAMES:
    nuis_ad[pname] = np.asarray(dPell_dtheta_autodiff(ps, k, pk_data, params, pname, s8, ell))
    nuis_st[pname] = np.asarray(dPell_dtheta_stencil(ps, k, pk_data, params, pname, s8, ell))

# --- colors and labels ---
PARAM_GROUPS = {
    "Cosmological": (["fsigma8", "Mnu", "Omegam"],
                     [r"$f\sigma_8$", r"$M_\nu$", r"$\Omega_m$"],
                     ["#E24A33", "#348ABD", "#988ED5"]),
    "Bias": (["b1_sigma8", "b2_sigma8sq", "bG2_sigma8sq", "bGamma3"],
             [r"$b_1\sigma_8$", r"$b_2\sigma_8^2$", r"$b_{G_2}\sigma_8^2$", r"$b_{\Gamma_3}$"],
             ["#2ca02c", "#98df8a", "#006400", "#90EE90"]),
    "Counterterm": (["c0", "c2", "c4", "c_tilde", "c1"],
                    [r"$c_0$", r"$c_2$", r"$c_4$", r"$\tilde{c}$", r"$c_1$"],
                    ["#ff7f0e", "#ffbb78", "#d62728", "#ff9896", "#bcbd22"]),
    "Stochastic": (["Pshot", "a0", "a2"],
                   [r"$P_{\rm shot}$", r"$a_0$", r"$a_2$"],
                   ["#17becf", "#9edae5", "#7f7f7f"]),
}

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# LEFT: |dP_0/dtheta| / P_0 for all params
for group_name, (pnames, labels, colors) in PARAM_GROUPS.items():
    for pname, label, color in zip(pnames, labels, colors):
        if pname in cosmo_derivs:
            deriv = cosmo_derivs[pname]
        elif pname in nuis_ad:
            deriv = nuis_ad[pname]
        else:
            continue
        # Normalize
        norm_deriv = np.abs(deriv) / np.abs(P0_fid)
        if np.max(norm_deriv) < 1e-10:
            continue  # skip c1 (zero derivative)
        ax1.semilogy(k_np, norm_deriv, lw=1.3, color=color, label=label)

ax1.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
ax1.set_ylabel(r"$|\partial P_0 / \partial \theta_i| \;/\; P_0$")
ax1.set_xlim(k_np[0], k_np[-1])
ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax1.set_title("Derivative landscape")

# RIGHT: fractional autodiff-stencil difference for nuisance params
for group_name, (pnames, labels, colors) in PARAM_GROUPS.items():
    if group_name == "Cosmological":
        continue  # cosmo derivs use stencil only
    for pname, label, color in zip(pnames, labels, colors):
        ad = nuis_ad.get(pname)
        st = nuis_st.get(pname)
        if ad is None or st is None:
            continue
        scale = np.maximum(np.abs(ad), np.abs(st))
        mask = scale > 1e-10 * np.max(scale)
        frac = np.full_like(ad, np.nan)
        frac[mask] = np.abs(ad[mask] - st[mask]) / scale[mask]
        if np.nanmax(frac) < 1e-15:
            continue
        ax2.semilogy(k_np, frac, lw=1.0, color=color, label=label, alpha=0.8)

ax2.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
ax2.set_ylabel(r"$|\mathrm{autodiff} - \mathrm{stencil}| \;/\; |\mathrm{max}|$")
ax2.set_xlim(k_np[0], k_np[-1])
ax2.set_ylim(1e-12, 1e-6)
ax2.axhline(1e-9, ls=":", color="gray", lw=0.8, label=r"$10^{-9}$")
ax2.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax2.set_title("Autodiff vs.\\ stencil agreement")

fig.tight_layout()
fig.savefig(OUT / "deriv_autodiff_vs_stencil.pdf", bbox_inches="tight")
fig.savefig(OUT / "deriv_autodiff_vs_stencil.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved to {OUT / 'deriv_autodiff_vs_stencil.png'}")
