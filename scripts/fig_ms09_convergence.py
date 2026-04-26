#!/usr/bin/env python
"""Generate Fig. A2: McDonald-Seljak cosmic-variance-free convergence.

Usage:  python scripts/fig_ms09_convergence.py
Output: paper/figs/ms09_convergence.{pdf,png}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figs"

# --- MS09 setup (mirrors test_mcdonald_seljak.py) ---
bA, bB, f, Pm = 1.0, 2.0, 0.8, 1e3  # Pm ~ P(k=0.1) in (Mpc/h)^3
k, dk, V = 0.1, 0.01, 1e9
Nmodes = k**2 * dk * V / (2 * np.pi**2)
mu_gl, w_gl = leggauss(100)  # high-order quadrature for smooth curves


def _sigma2_st(nbar):
    """Single-tracer sigma^2(Pm), marginalized over bA via Schur complement."""
    F = np.zeros((2, 2))
    for mi, wi in zip(mu_gl, w_gl):
        sA = bA + f * mi**2
        PAA = sA**2 * Pm + 1.0 / nbar
        D = np.array([sA**2, 2.0 * sA * Pm])
        F += wi * np.outer(D, D) / (2.0 * PAA**2)
    F *= Nmodes / 2.0
    schur = F[0, 0] - F[0, 1]**2 / F[1, 1]
    return 1.0 / schur


def _sigma2_mt(nbar):
    """Multi-tracer sigma^2(Pm), marginalized over bA and bB."""
    F = np.zeros((3, 3))
    for mi, wi in zip(mu_gl, w_gl):
        sA = bA + f * mi**2
        sB = bB + f * mi**2
        PAA = sA**2 * Pm + 1.0 / nbar
        PBB = sB**2 * Pm + 1.0 / nbar
        PAB = sA * sB * Pm
        C3 = np.array([
            [2 * PAA**2,      2 * PAB**2,         2 * PAA * PAB],
            [2 * PAB**2,      2 * PBB**2,         2 * PBB * PAB],
            [2 * PAA * PAB,   2 * PBB * PAB,      PAA * PBB + PAB**2],
        ])
        D = np.array([
            [sA**2,    2 * sA * Pm,   0.0],
            [sB**2,    0.0,           2 * sB * Pm],
            [sA * sB,  sB * Pm,       sA * Pm],
        ])
        F += wi * D.T @ np.linalg.inv(C3) @ D
    F *= Nmodes / 2.0
    F_bias = F[1:, 1:]
    F_cross = F[0, 1:]
    schur = F[0, 0] - F_cross @ np.linalg.solve(F_bias, F_cross)
    return 1.0 / schur


def _sigma2_mt_fixed_bB(nbar):
    """Multi-tracer sigma^2(Pm) with bB fixed, marginalized over bA only."""
    F = np.zeros((2, 2))
    for mi, wi in zip(mu_gl, w_gl):
        sA = bA + f * mi**2
        sB = bB + f * mi**2
        PAA = sA**2 * Pm + 1.0 / nbar
        PBB = sB**2 * Pm + 1.0 / nbar
        PAB = sA * sB * Pm
        C3 = np.array([
            [2 * PAA**2,      2 * PAB**2,         2 * PAA * PAB],
            [2 * PAB**2,      2 * PBB**2,         2 * PBB * PAB],
            [2 * PAA * PAB,   2 * PBB * PAB,      PAA * PBB + PAB**2],
        ])
        D = np.array([
            [sA**2,    2 * sA * Pm],
            [sB**2,    0.0],
            [sA * sB,  sB * Pm],
        ])
        F += wi * D.T @ np.linalg.inv(C3) @ D
    F *= Nmodes / 2.0
    schur = F[0, 0] - F[0, 1]**2 / F[1, 1]
    return 1.0 / schur


# --- sweep nbar (cap at 10^3 to stay numerically stable with Pm=1e3) ---
nbars = np.logspace(-4, 3, 200)  # dense sampling for smooth transition
s2_st = np.array([_sigma2_st(n) for n in nbars])
s2_mt = np.array([_sigma2_mt(n) for n in nbars])
s2_mt_fixB = np.array([_sigma2_mt_fixed_bB(n) for n in nbars])
cv_floor = 2.0 * Pm**2 / Nmodes

# --- plot (use usetex for clean minus signs) ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "text.usetex": True,
})

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.loglog(nbars, s2_st / Pm**2, "-", color="#4C72B0", lw=2,
          label=r"Single-tracer ($b_A$ marginalized)")
ax.loglog(nbars, s2_mt / Pm**2, "-", color="#DD8452", lw=2,
          label=r"Multi-tracer ($b_A, b_B$ marginalized)")
ax.loglog(nbars, s2_mt_fixB / Pm**2, "-.", color="#DD8452", lw=1.5, alpha=0.7,
          label=r"Multi-tracer ($b_B$ fixed)")
ax.axhline(cv_floor / Pm**2, ls="--", color="gray", lw=1.2,
           label=r"CV-free floor $2/N_{\rm modes}$")

# Typical survey nbar range
ax.axvspan(5e-4, 1e-3, color="green", alpha=0.10, zorder=0)
ax.text(7e-4, 5e-3, r"typical $\bar{n}$", color="forestgreen",
        fontsize=10, ha="center", rotation=90)

ax.set_xlabel(r"$\bar{n}$ [$(h^{-1}\,\mathrm{Mpc})^{-3}$]")
ax.set_ylabel(r"$\sigma^2(P_m)\,/\,P_m^2$")
ax.set_xlim(nbars[0], nbars[-1])
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "ms09_convergence.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "ms09_convergence.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved to {OUT_DIR / 'ms09_convergence.pdf'}")
