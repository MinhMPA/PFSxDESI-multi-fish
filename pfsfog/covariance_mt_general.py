"""Generalized N-tracer Gaussian multipole covariance.

For N tracers with Npairs = N(N+1)/2 spectrum pairs and Nell multipoles,
builds an (Npairs*Nell) × (Npairs*Nell) covariance matrix at each k.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


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


def multi_tracer_cov_general(
    tracer_names: list[str],
    pkmu_funcs: dict[tuple[str, str], callable],
    nbar: dict[str, float],
    k: np.ndarray,
    volume: float,
    dk: float,
    ells: tuple[int, ...] = (0, 2, 4),
    cross_shot: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """N-tracer Gaussian covariance.

    Parameters
    ----------
    tracer_names : ordered list of tracer names
    pkmu_funcs : {(name_A, name_B): callable(k, mu) -> (Nk, Nmu)}
        For auto: key = (name, name). For cross: key = sorted pair.
        Signal only (no 1/nbar noise).
    nbar : {tracer_name: number density}
    k : array (Nk,)
    volume : overlap volume (Mpc/h)³
    dk : bin width
    ells : multipole orders

    Returns
    -------
    C : array (Nk, Nobs, Nobs) where Nobs = Npairs * Nell
    """
    Nt = len(tracer_names)
    Nell = len(ells)
    Nk = len(k)
    mu = _MU_GL
    w = _W_GL

    # Build ordered pairs: auto first, then cross
    pairs = []
    for a in tracer_names:
        pairs.append((a, a))
    for i, a in enumerate(tracer_names):
        for j in range(i + 1, Nt):
            pairs.append((a, tracer_names[j]))
    Npairs = len(pairs)
    Nobs = Npairs * Nell

    # Evaluate all P(k,mu) on GL grid — shape (Nk, Nmu) each
    Pkmu = {}
    for pair in set(pairs):
        key = tuple(sorted(pair)) if pair[0] != pair[1] else pair
        if key not in Pkmu:
            func = pkmu_funcs.get(key) or pkmu_funcs.get(pair)
            if func is None:
                # Try reverse order
                func = pkmu_funcs.get((pair[1], pair[0]))
            if func is not None:
                Pkmu[key] = np.asarray(func(k, mu))
            else:
                Pkmu[key] = np.zeros((Nk, len(mu)))

    # P_tot^{XY} = P^{XY} + delta_{XY}/nbar_X + cross_shot^{XY}
    # cross_shot is non-zero when surveys share galaxies (e.g. PFS-ELG × DESI-ELG)
    _cross_shot = cross_shot or {}

    def get_ptot(X, Y):
        key = (X, Y) if X <= Y else (Y, X)
        if X == Y:
            key = (X, X)
        p = Pkmu.get(key, np.zeros((Nk, len(mu))))
        if X == Y:
            return p + 1.0 / nbar[X]
        # Non-zero cross-shot for partially shared catalogues
        cs_key = (X, Y) if X <= Y else (Y, X)
        cs = _cross_shot.get(cs_key, 0.0)
        return p + cs

    # Number of modes
    Nmodes = k**2 * dk * volume / (2.0 * np.pi**2)

    # Precompute Legendre values
    L = {ell: _legendre(ell, mu) for ell in ells}

    # Build covariance
    C = np.zeros((Nk, Nobs, Nobs))

    for ip, (X, Y) in enumerate(pairs):
        for jp, (W, Z) in enumerate(pairs):
            for il, ell_i in enumerate(ells):
                for jl, ell_j in enumerate(ells):
                    i_idx = ip * Nell + il
                    j_idx = jp * Nell + jl

                    # Cov = (2l+1)(2l'+1)/(2 Nmodes) ×
                    #   ∫ dμ L_l L_l' [P_tot^{XW} P_tot^{YZ} + P_tot^{XZ} P_tot^{YW}]
                    Ptot_XW = get_ptot(X, W)
                    Ptot_YZ = get_ptot(Y, Z)
                    Ptot_XZ = get_ptot(X, Z)
                    Ptot_YW = get_ptot(Y, W)

                    integrand = L[ell_i] * L[ell_j] * (
                        Ptot_XW * Ptot_YZ + Ptot_XZ * Ptot_YW
                    ) * w
                    integral = np.sum(integrand, axis=1)

                    C[:, i_idx, j_idx] = (
                        (2 * ell_i + 1) * (2 * ell_j + 1)
                        / (2.0 * Nmodes)
                        * integral
                    )

    return C
