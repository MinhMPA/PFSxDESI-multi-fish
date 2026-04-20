"""Gaussian multipole covariance matrices.

Single-tracer: 3×3 (ℓ=0,2,4) at each k.
Multi-tracer:  9×9 ({AA,BB,AB} × {0,2,4}) at each k.

μ-integrals use Gauss-Legendre quadrature (20 points).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


# 20-point Gauss-Legendre on [-1, 1]
_NLEG = 20
_MU_GL, _W_GL = leggauss(_NLEG)


def _legendre(ell: int, mu: np.ndarray) -> np.ndarray:
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3 * mu**2 - 1)
    if ell == 4:
        return (1.0 / 8.0) * (35 * mu**4 - 30 * mu**2 + 3)
    raise ValueError(f"ell={ell} not supported")


# ---------------------------------------------------------------------------
# Single-tracer Gaussian covariance
# ---------------------------------------------------------------------------


def single_tracer_cov(
    pkmu_func,
    k: np.ndarray,
    nbar: float,
    volume: float,
    dk: float,
    ells: tuple[int, ...] = (0, 2, 4),
) -> np.ndarray:
    """Gaussian multipole covariance for a single tracer.

    Parameters
    ----------
    pkmu_func : callable
        P(k, mu) → array of shape ``(Nk, Nmu)`` when called with
        ``pkmu_func(k, mu)``.  Must include signal only (no 1/nbar).
    k : array, shape (Nk,)
    nbar : float, number density [(h⁻¹Mpc)⁻³]
    volume : float, survey volume [(h⁻¹Mpc)³]  (= (Mpc/h)³)
    dk : float, k-bin width [h/Mpc]
    ells : tuple of multipole orders

    Returns
    -------
    C : array, shape (Nk, Nell, Nell)
        Covariance matrix at each k.
    """
    Nk = len(k)
    Nell = len(ells)
    mu = _MU_GL
    w = _W_GL

    # P(k, mu) on the GL grid — shape (Nk, Nmu)
    Pkmu = np.asarray(pkmu_func(k, mu))       # signal
    Ptot = Pkmu + 1.0 / nbar                   # signal + noise

    # N_modes(k) = k² Δk V / (2π²)
    Nmodes = k**2 * dk * volume / (2.0 * np.pi**2)

    # Precompute Legendre × weights
    Lw = np.zeros((Nell, _NLEG))
    for i, ell in enumerate(ells):
        Lw[i] = _legendre(ell, mu) * w  # (Nmu,)

    C = np.zeros((Nk, Nell, Nell))
    for i, ell_i in enumerate(ells):
        for j, ell_j in enumerate(ells):
            # ∫ dμ L_ℓ(μ) L_ℓ'(μ) [P_tot]²
            integrand = Lw[i] * _legendre(ell_j, mu) * Ptot**2  # (Nk, Nmu)
            integral = np.sum(integrand, axis=1)  # (Nk,)
            C[:, i, j] = (
                (2 * ell_i + 1) * (2 * ell_j + 1)
                / (2.0 * Nmodes)
                * integral
            )
    return C


# ---------------------------------------------------------------------------
# Multi-tracer Gaussian covariance
# ---------------------------------------------------------------------------


def multi_tracer_cov(
    pkmu_AA,
    pkmu_BB,
    pkmu_AB,
    k: np.ndarray,
    nbar_A: float,
    nbar_B: float,
    volume: float,
    dk: float,
    ells: tuple[int, ...] = (0, 2, 4),
) -> np.ndarray:
    """Gaussian covariance for the 9-element multi-tracer observable vector.

    Observable ordering at each k:
        d = [P_0^AA, P_2^AA, P_4^AA,
             P_0^BB, P_2^BB, P_4^BB,
             P_0^AB, P_2^AB, P_4^AB]

    Parameters
    ----------
    pkmu_AA, pkmu_BB, pkmu_AB : callable
        P^{XY}(k, mu) → array (Nk, Nmu).  Signal only.
    k : array (Nk,)
    nbar_A, nbar_B : number densities
    volume : survey volume (overlap)
    dk : k-bin width

    Returns
    -------
    C : array, shape (Nk, 9, 9)
    """
    Nk = len(k)
    Nell = len(ells)
    Nobs = 3 * Nell  # 9
    mu = _MU_GL
    w = _W_GL

    # Evaluate P(k,μ) on GL grid — shape (Nk, Nmu) each
    P_AA = np.asarray(pkmu_AA(k, mu))
    P_BB = np.asarray(pkmu_BB(k, mu))
    P_AB = np.asarray(pkmu_AB(k, mu))

    # P_tot^{XY} = P^{XY} + δ_{XY} / nbar_X
    Ptot = {
        ("A", "A"): P_AA + 1.0 / nbar_A,
        ("B", "B"): P_BB + 1.0 / nbar_B,
        ("A", "B"): P_AB,
        ("B", "A"): P_AB,
    }

    Nmodes = k**2 * dk * volume / (2.0 * np.pi**2)

    # Precompute Legendre values
    L = {}
    for ell in ells:
        L[ell] = _legendre(ell, mu)

    # Observable index → (tracer pair, ell)
    obs_spec = []
    for pair_label, pair in [("AA", ("A", "A")), ("BB", ("B", "B")), ("AB", ("A", "B"))]:
        for ell in ells:
            obs_spec.append((pair, ell))

    C = np.zeros((Nk, Nobs, Nobs))
    for i, ((X, Y), ell_i) in enumerate(obs_spec):
        for j, ((W, Z), ell_j) in enumerate(obs_spec):
            # Cov = (2l+1)(2l'+1)/(2 Nmodes) × ∫ dμ L_l L_l'
            #        × [P_tot^{XW} P_tot^{YZ} + P_tot^{XZ} P_tot^{YW}]
            term1 = Ptot[(X, W)] * Ptot[(Y, Z)]
            term2 = Ptot[(X, Z)] * Ptot[(Y, W)]
            integrand = L[ell_i] * L[ell_j] * (term1 + term2) * w
            integral = np.sum(integrand, axis=1)  # (Nk,)
            C[:, i, j] = (
                (2 * ell_i + 1) * (2 * ell_j + 1)
                / (2.0 * Nmodes)
                * integral
            )
    return C
