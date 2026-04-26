#!/usr/bin/env python
"""Validate GL covariance against analytic Wigner 3j using the full one-loop EFT model.

Produces a single-panel figure with shaded bands showing the fractional
difference between GL quadrature and the Wigner 3j formula at
ell_max = 4, 6, 8, spanning all covariance elements.

Usage:
    python scripts/fig_cov_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
from numpy.polynomial.legendre import leggauss

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import desi_elg, pfs_elg
from pfsfog.eft_params import tracer_fiducials
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto, fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_func, make_ps1loop_pkmu_cross_func,
)
from ps_1loop_jax import PowerSpectrum1Loop


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "text.usetex": True,
})

# Colorblind-friendly sequential palette (Tol's muted: indigo, teal, sand)
COL = {4: "#882255", 6: "#117733", 8: "#332288"}


def legendre_poly(ell, mu):
    if ell == 0: return np.ones_like(mu)
    if ell == 2: return 0.5 * (3 * mu**2 - 1)
    if ell == 4: return (1.0 / 8) * (35 * mu**4 - 30 * mu**2 + 3)
    if ell == 6: return (1.0 / 16) * (231 * mu**6 - 315 * mu**4 + 105 * mu**2 - 5)
    if ell == 8: return (1.0 / 128) * (6435 * mu**8 - 12012 * mu**6 + 6930 * mu**4 - 1260 * mu**2 + 35)
    raise ValueError(f"ell={ell}")


_w3j_cache = {}

def wigner3j_sq(l1, l2, l3):
    key = (l1, l2, l3)
    if key not in _w3j_cache:
        val = float(wigner_3j(l1, l2, l3, 0, 0, 0))
        _w3j_cache[key] = val * val
    return _w3j_cache[key]


def decompose_multipoles(Ptot_kmu, mu, w, ells):
    result = {}
    for ell in ells:
        L = legendre_poly(ell, mu)
        result[ell] = (2 * ell + 1) / 2.0 * np.sum(
            L[None, :] * Ptot_kmu * w[None, :], axis=1)
    return result


def cov_wigner3j_array(Pell_XW, Pell_YZ, Pell_XZ, Pell_YW,
                       ell, ellp, Nmodes, decomp_ells):
    Nk = len(Nmodes)
    result = np.zeros(Nk)
    for l1 in decomp_ells:
        for l2 in decomp_ells:
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                w1 = wigner3j_sq(l1, l2, l3)
                if w1 == 0: continue
                w2 = wigner3j_sq(ell, ellp, l3)
                if w2 == 0: continue
                coeff = (2 * l3 + 1) * w1 * w2
                term1 = (-1)**ellp * Pell_XW[l1] * Pell_YZ[l2]
                term2 = Pell_XZ[l1] * Pell_YW[l2]
                result += coeff * (term1 + term2)
    return (2 * ell + 1) * (2 * ellp + 1) / Nmodes * result


def cov_gl_array(Ptot_XW, Ptot_YZ, Ptot_XZ, Ptot_YW,
                 ell, ellp, Nmodes, mu, w):
    L_ell = legendre_poly(ell, mu)
    L_ellp = legendre_poly(ellp, mu)
    integrand = (L_ell[None, :] * L_ellp[None, :]
                 * (Ptot_XW * Ptot_YZ + Ptot_XZ * Ptot_YW)
                 * w[None, :])
    return (2 * ell + 1) * (2 * ellp + 1) / (2.0 * Nmodes) * np.sum(integrand, axis=1)


def build_data():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    ps = PowerSpectrum1Loop(do_irres=False)

    z_eff = 0.9
    s8 = cosmo.sigma8(z_eff)
    f_z = float(cosmo.f(z_eff))
    h = cosmo.params["h"]
    pk_data = cosmo.pk_data(z_eff)

    surv_A, surv_B = pfs_elg(), desi_elg()
    nbar_A = surv_A.nbar_eff(0.8, 1.0)
    nbar_B = surv_B.nbar_eff(0.8, 1.0)
    b1_A, b1_B = surv_A.b1_of_z(z_eff), surv_B.b1_of_z(z_eff)

    fid_A = tracer_fiducials("PFS-ELG", b1_A, s8, b1_ref=b1_B,
                             r_sigma_v=cfg.r_sigma_v)
    fid_B = tracer_fiducials("DESI-ELG", b1_B, s8)
    params_A = fisher_to_ps1loop_auto(fid_A, s8, f_z, h, nbar_A)
    params_B = fisher_to_ps1loop_auto(fid_B, s8, f_z, h, nbar_B)
    cross_params = fisher_to_ps1loop_cross(fid_A, fid_B, s8, f_z, h,
                                           nbar_A, nbar_B)

    pkmu_AA = make_ps1loop_pkmu_func(ps, pk_data, params_A)
    pkmu_BB = make_ps1loop_pkmu_func(ps, pk_data, params_B)
    pkmu_AB = make_ps1loop_pkmu_cross_func(ps, pk_data, cross_params)

    k = np.arange(cfg.kmin, 0.25 + cfg.dk / 2, cfg.dk)
    Nleg = 20
    mu, w = leggauss(Nleg)
    V = 5.88e8
    Nmodes = k**2 * cfg.dk * V / (2.0 * np.pi**2)

    P_AA = np.asarray(pkmu_AA(k, mu)) + 1.0 / nbar_A
    P_BB = np.asarray(pkmu_BB(k, mu)) + 1.0 / nbar_B
    P_AB = np.asarray(pkmu_AB(k, mu))

    return k, mu, w, Nmodes, P_AA, P_BB, P_AB


def compute_all_frac_diffs(k, mu, w, Nmodes, P_AA, P_BB, P_AB):
    ell_pairs = [(0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 4)]
    spec_cases = [
        (P_AA, P_AA, P_AA, P_AA),  # AA-AA
        (P_AA, P_BB, P_AB, P_AB),  # AB-AB
        (P_AA, P_AB, P_AB, P_AA),  # AA-AB
        (P_BB, P_BB, P_BB, P_BB),  # BB-BB
        (P_AB, P_BB, P_BB, P_AB),  # BB-AB
        (P_AB, P_AB, P_AB, P_AB),  # AA-BB
    ]

    ell_max_list = [4, 6, 8]
    results = {}

    for lmax in ell_max_list:
        decomp_ells = tuple(range(0, lmax + 1, 2))
        Pell_AA = decompose_multipoles(P_AA, mu, w, decomp_ells)
        Pell_BB = decompose_multipoles(P_BB, mu, w, decomp_ells)
        Pell_AB = decompose_multipoles(P_AB, mu, w, decomp_ells)
        pell_map = {id(P_AA): Pell_AA, id(P_BB): Pell_BB, id(P_AB): Pell_AB}

        fracs = []
        for (ptXW, ptYZ, ptXZ, ptYW) in spec_cases:
            for (el, elp) in ell_pairs:
                cgl = cov_gl_array(ptXW, ptYZ, ptXZ, ptYW,
                                   el, elp, Nmodes, mu, w)
                cw3j = cov_wigner3j_array(
                    pell_map[id(ptXW)], pell_map[id(ptYZ)],
                    pell_map[id(ptXZ)], pell_map[id(ptYW)],
                    el, elp, Nmodes, decomp_ells)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.abs(cgl - cw3j) / np.abs(cgl)
                frac[np.abs(cgl) < 1e-30] = np.nan
                fracs.append(frac)
        results[lmax] = np.array(fracs)

    return results, ell_max_list


def make_figure(k, results, ell_max_list):
    fig, ax = plt.subplots(figsize=(7, 5))

    for lmax in ell_max_list:
        fracs = results[lmax]
        lo = np.nanmin(fracs, axis=0)
        hi = np.nanmax(fracs, axis=0)
        med = np.nanmedian(fracs, axis=0)

        ax.fill_between(k, lo, hi, color=COL[lmax], alpha=0.25)
        ax.semilogy(k, med, color=COL[lmax], lw=1.8,
                    label=rf"$\ell_{{\max}}={lmax}$")

    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(
        r"$|\mathrm{Cov}_{\mathrm{GL}} - \mathrm{Cov}_{\mathrm{W3}j}|"
        r"\;/\;|\mathrm{Cov}_{\mathrm{GL}}|$")
    ax.set_ylim(1e-14, 1e-3)
    ax.legend(loc="upper right", title="Wigner 3$j$ truncation",
              frameon=True, fancybox=False, edgecolor="0.8")

    fig.tight_layout()
    return fig


def main():
    print("Building one-loop P(k,mu)...")
    k, mu, w, Nmodes, P_AA, P_BB, P_AB = build_data()

    print("Computing fractional differences...")
    results, ell_max_list = compute_all_frac_diffs(
        k, mu, w, Nmodes, P_AA, P_BB, P_AB)

    fig = make_figure(k, results, ell_max_list)

    out = Path("paper/figs/cov_gl_vs_wigner3j.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved to {out}")

    for lmax in ell_max_list:
        hi = np.nanmax(results[lmax])
        med = np.nanmedian(results[lmax])
        print(f"  ell_max={lmax}: max={hi:.2e}, median={med:.2e}")


if __name__ == "__main__":
    main()
