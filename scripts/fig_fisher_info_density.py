#!/usr/bin/env python
"""Generate Fisher information density dF_ii/dk vs k for cosmo params.

Shows where in k-space the constraining power comes from for fσ₈, Mν, Ωm,
and how nuisance marginalization redistributes the information.

Usage:  python scripts/fig_fisher_info_density.py
Output: paper/figs/fisher_info_density.{pdf,png}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": True})
import matplotlib.pyplot as plt

from ps_1loop_jax import PowerSpectrum1Loop
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, desi_elg_fiducials, broad_priors
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.config import ForecastConfig

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs"

# --- setup ---
cfg = ForecastConfig.from_yaml("configs/default.yaml")
ps = PowerSpectrum1Loop(do_irres=False)
cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
from pfsfog.surveys import desi_elg
desi = desi_elg()

ells = (0, 2, 4)
kmax = 0.20
k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)

# Use z=0.9 as representative
zlo, zhi = 0.8, 1.0
z_eff = 0.9
s8 = cosmo.sigma8(z_eff)
f_z = float(cosmo.f(z_eff))
h = cosmo.params["h"]
nbar = desi.nbar_eff(zlo, zhi)
b1 = desi.b1_of_z(z_eff)
V = desi.volume(zlo, zhi)
pk_data = cosmo.pk_data(z_eff)

fid = desi_elg_fiducials(b1, s8)
params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)

# Derivatives
derivs = dPell_dtheta_autodiff_all(
    ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
cosmo_derivs = dPell_d_cosmo_all(
    ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells)
derivs.update(cosmo_derivs)

# Covariance
pkmu = make_ps1loop_pkmu_func(ps, pk_data, params)
cov = single_tracer_cov(pkmu, k, nbar, V, cfg.dk, ells)

# Build per-k Fisher and extract diagonal info density
N_COSMO = len(COSMO_NAMES)
N_NUIS = len(NUISANCE_NAMES)
Np = N_COSMO + N_NUIS
Nk = len(k)
Nell = len(ells)

# Fisher info density: dF_ii/dk at each k (CONDITIONAL = no marginalization)
info_cond = {cp: np.zeros(Nk) for cp in COSMO_NAMES}

# For marginalized info density, accumulate full Fisher up to each k
# then invert — this is expensive but gives the right answer
info_marg_broad = {cp: np.zeros(Nk) for cp in COSMO_NAMES}

# Build derivative matrix
D = np.zeros((Nk, Nell, Np))
for ic, cn in enumerate(COSMO_NAMES):
    for il, ell in enumerate(ells):
        if cn in derivs and ell in derivs[cn]:
            D[:, il, ic] = np.asarray(derivs[cn][ell])
for ip, nn in enumerate(NUISANCE_NAMES):
    idx = N_COSMO + ip
    for il, ell in enumerate(ells):
        if nn in derivs and ell in derivs[nn]:
            D[:, il, idx] = np.asarray(derivs[nn][ell])

# Conditional info density (no marginalization)
for ik in range(Nk):
    cov_inv_k = np.linalg.inv(cov[ik])
    DtCinv = D[ik].T @ cov_inv_k
    F_k = DtCinv @ D[ik]  # per-k Fisher contribution
    for ic, cn in enumerate(COSMO_NAMES):
        info_cond[cn][ik] = F_k[ic, ic]

# Cumulative marginalized info: F(k<k_max) = sum F_k + prior
# Then sigma_marg(k_max) = sqrt([F^{-1}]_ii)
# Info density = -d(sigma^{-2})/dk ≈ delta(sigma^{-2})/delta_k
bp_diag = broad_priors().prior_fisher_diag()
prior_full = np.zeros(Np)
prior_full[N_COSMO:] = bp_diag
from pfsfog.eft_params import COSMO_PRIOR_SIGMA
for i, cn in enumerate(COSMO_NAMES):
    prior_full[i] = 1.0 / COSMO_PRIOR_SIGMA[cn]**2

F_cum = np.diag(prior_full).copy()
prev_marg_fisher = {cn: prior_full[i] for i, cn in enumerate(COSMO_NAMES)}

for ik in range(Nk):
    cov_inv_k = np.linalg.inv(cov[ik])
    DtCinv = D[ik].T @ cov_inv_k
    F_cum += DtCinv @ D[ik] * cfg.dk

    try:
        C_cum = np.linalg.inv(F_cum)
        for ic, cn in enumerate(COSMO_NAMES):
            marg_fisher_now = 1.0 / C_cum[ic, ic]
            info_marg_broad[cn][ik] = (marg_fisher_now - prev_marg_fisher[cn]) / cfg.dk
            prev_marg_fisher[cn] = marg_fisher_now
    except np.linalg.LinAlgError:
        pass

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

cosmo_labels = {
    "fsigma8": r"$f\sigma_8$",
    "Mnu": r"$M_\nu$",
    "Omegam": r"$\Omega_m$",
}

for ic, (cn, ax) in enumerate(zip(COSMO_NAMES, axes)):
    ax.semilogy(k, info_cond[cn], "-", color="#4C72B0", lw=2,
                label="Conditional (nuis.\\ fixed)")
    # Clip negative marginalized values
    marg = np.maximum(info_marg_broad[cn], 1e-10)
    ax.semilogy(k, marg, "-", color="#DD8452", lw=2,
                label="Marginalized (broad)")
    ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    if ic == 0:
        ax.set_ylabel(r"$\mathrm{d}F_{ii}/\mathrm{d}k$")
    ax.set_title(cosmo_labels[cn])
    ax.set_xlim(k[0], k[-1])
    if ic == 0:
        ax.legend(frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "fisher_info_density.pdf", bbox_inches="tight")
fig.savefig(OUT / "fisher_info_density.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved to {OUT / 'fisher_info_density.png'}")
