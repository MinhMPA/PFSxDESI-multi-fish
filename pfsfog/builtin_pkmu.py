"""Lightweight Kaiser + counterterm P(k,μ) model.

Used for:
- Generating fiducial P(k,μ) for covariance computation
- Rapid sanity checks
- Unit tests that don't require ps_1loop_jax initialization

Does NOT include one-loop bias terms — not used for production Fisher
derivatives.
"""

from __future__ import annotations

import numpy as np


def pkmu_auto(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    b1: float,
    f: float,
    c0: float = 0.0,
    c2: float = 0.0,
    c4: float = 0.0,
    cfog: float = 0.0,
    nbar: float = 1e-3,
    Pshot: float = 0.0,
    a0: float = 0.0,
    a2: float = 0.0,
    k_nl: float = 0.7,
) -> np.ndarray:
    """Kaiser + EFT counterterm auto-power P(k, μ).

    Parameters
    ----------
    k : array (Nk,)
    mu : array (Nmu,)
    plin : array (Nk,)  — linear P(k) at the same k grid
    b1, f : linear bias and growth rate
    c0, c2, c4 : counterterms [Mpc/h]²
    cfog : FoG counterterm [Mpc/h]⁴
    nbar : number density [(h⁻¹Mpc)⁻³]
    Pshot, a0, a2 : stochastic terms (dimensionless)
    k_nl : nonlinear scale for stochastic k-dependence

    Returns
    -------
    P(k, μ) : array (Nk, Nmu)
    """
    k = np.atleast_1d(k)
    mu = np.atleast_1d(mu)
    plin = np.atleast_1d(plin)

    mu2 = mu[None, :] ** 2
    mu4 = mu[None, :] ** 4
    k2 = k[:, None] ** 2
    k4 = k[:, None] ** 4
    pl = plin[:, None]

    # Kaiser term
    Z1 = b1 + f * mu2
    P_kaiser = Z1**2 * pl

    # Counterterms
    P_ctr = -2.0 * k2 * (c0 + c2 * f * mu2 + c4 * f**2 * mu4) * pl

    # FoG counterterm (perturbative, not Lorentzian)
    P_fog = -cfog * k4 * f**4 * mu4 * Z1**2 * pl

    # Stochastic
    k_over_knl2 = (k[:, None] / k_nl) ** 2
    P_stoch = (1.0 / nbar) * (Pshot + a0 * k_over_knl2 + a2 * k_over_knl2 * mu2)

    return P_kaiser + P_ctr + P_fog + P_stoch


def pkmu_cross(
    k: np.ndarray,
    mu: np.ndarray,
    plin: np.ndarray,
    b1_A: float,
    b1_B: float,
    f: float,
    c0_A: float = 0.0,
    c0_B: float = 0.0,
    c2_A: float = 0.0,
    c2_B: float = 0.0,
    c4_A: float = 0.0,
    c4_B: float = 0.0,
    cfog_A: float = 0.0,
    cfog_B: float = 0.0,
) -> np.ndarray:
    """Kaiser + counterterm cross-power P^{AB}(k, μ).

    Cross-stochastic = 0.  Counterterms averaged.
    """
    k = np.atleast_1d(k)
    mu = np.atleast_1d(mu)
    plin = np.atleast_1d(plin)

    mu2 = mu[None, :] ** 2
    mu4 = mu[None, :] ** 4
    k2 = k[:, None] ** 2
    k4 = k[:, None] ** 4
    pl = plin[:, None]

    Z1_A = b1_A + f * mu2
    Z1_B = b1_B + f * mu2

    P_kaiser = Z1_A * Z1_B * pl

    # Averaged counterterms
    c0_avg = 0.5 * (c0_A + c0_B)
    c2_avg = 0.5 * (c2_A + c2_B)
    c4_avg = 0.5 * (c4_A + c4_B)
    cfog_avg = 0.5 * (cfog_A + cfog_B)

    P_ctr = -2.0 * k2 * (c0_avg + c2_avg * f * mu2 + c4_avg * f**2 * mu4) * pl
    P_fog = -cfog_avg * k4 * f**4 * mu4 * Z1_A * Z1_B * pl

    return P_kaiser + P_ctr + P_fog
