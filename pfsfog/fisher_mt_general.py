"""Generalized multi-tracer Fisher matrix for N tracers.

Handles an arbitrary number of tracers in the overlap, each with its
own bias, counterterm, and stochastic parameters. At each k, the
observable vector is:

    d(k) = [P_0^{AA}, P_2^{AA}, P_4^{AA},   # tracer A auto
            P_0^{BB}, P_2^{BB}, P_4^{BB},   # tracer B auto
            P_0^{AB}, P_2^{AB}, P_4^{AB},   # cross A×B
            ...]

For N tracers: N auto-spectra + N(N-1)/2 cross-spectra = N(N+1)/2 pairs,
each with 3 multipoles → 3N(N+1)/2 observables.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .eft_params import NUISANCE_NAMES, COSMO_NAMES
from .fisher import FisherResult


def mt_general_param_names(tracer_names: list[str]) -> list[str]:
    """Parameter names: cosmo + nuisance per tracer."""
    names = list(COSMO_NAMES)
    for tn in tracer_names:
        for n in NUISANCE_NAMES:
            names.append(f"{n}_{tn}")
    return names


def multi_tracer_fisher_general(
    tracer_names: list[str],
    derivs_auto: dict[str, dict[str, dict[int, np.ndarray]]],
    derivs_cross: dict[tuple[str, str], dict[str, dict[int, np.ndarray]]],
    cov_mt: np.ndarray,
    k: np.ndarray,
    dk: float,
    z_bin: tuple[float, float],
    ells: tuple[int, ...] = (0, 2, 4),
) -> FisherResult:
    """Assemble the N-tracer Fisher matrix.

    Parameters
    ----------
    tracer_names : list of tracer names (e.g. ["PFS-ELG", "DESI-ELG", "DESI-LRG"])
    derivs_auto : {tracer_name: {nuisance_name: {ell: dP_ell/dtheta}}}
        Auto-spectrum derivatives for each tracer's own nuisance params.
    derivs_cross : {(name_A, name_B): {nuisance_name_with_side: {ell: dP_ell/dtheta}}}
        Cross-spectrum derivatives. Keys are sorted pairs (A < B).
        nuisance_name_with_side has format "param_name:A" or "param_name:B"
        indicating which tracer's param is being perturbed.
    cov_mt : array (Nk, Nobs, Nobs)
        Full multi-tracer covariance.
    k : array (Nk,)
    dk : float
    z_bin : (zlo, zhi)
    ells : multipole orders

    Returns
    -------
    FisherResult
    """
    Nt = len(tracer_names)
    Nell = len(ells)
    Nk = len(k)

    # Build the ordered list of spectrum pairs
    pairs = []
    for i, a in enumerate(tracer_names):
        pairs.append((a, a))  # auto
    for i, a in enumerate(tracer_names):
        for j in range(i + 1, Nt):
            pairs.append((a, tracer_names[j]))  # cross
    Npairs = len(pairs)
    Nobs = Npairs * Nell

    # Parameter layout
    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)
    param_names = mt_general_param_names(tracer_names)
    Np = len(param_names)

    # Map tracer name → index in the nuisance block
    tracer_idx = {name: i for i, name in enumerate(tracer_names)}

    # Build derivative matrix D: (Nk, Nobs, Np)
    D = np.zeros((Nk, Nobs, Np))

    for ip_pair, (pA, pB) in enumerate(pairs):
        obs_offset = ip_pair * Nell  # start row for this pair's multipoles

        if pA == pB:
            # Auto-spectrum: derivatives w.r.t. this tracer's nuisance params
            tracer_name = pA
            ti = tracer_idx[tracer_name]
            if tracer_name in derivs_auto:
                for ip_nuis, nuis_name in enumerate(NUISANCE_NAMES):
                    param_col = N_COSMO + ti * N_NUIS + ip_nuis
                    if nuis_name in derivs_auto[tracer_name]:
                        for il, ell in enumerate(ells):
                            if ell in derivs_auto[tracer_name][nuis_name]:
                                D[:, obs_offset + il, param_col] = np.asarray(
                                    derivs_auto[tracer_name][nuis_name][ell]
                                )
        else:
            # Cross-spectrum: derivatives w.r.t. both tracers' params
            pair_key = (pA, pB) if pA < pB else (pB, pA)
            if pair_key in derivs_cross:
                cross_derivs = derivs_cross[pair_key]
                for ip_nuis, nuis_name in enumerate(NUISANCE_NAMES):
                    # A-side
                    key_a = f"{nuis_name}:A"
                    if key_a in cross_derivs:
                        ti_a = tracer_idx[pair_key[0]]
                        param_col_a = N_COSMO + ti_a * N_NUIS + ip_nuis
                        for il, ell in enumerate(ells):
                            if ell in cross_derivs[key_a]:
                                D[:, obs_offset + il, param_col_a] += np.asarray(
                                    cross_derivs[key_a][ell]
                                )
                    # B-side
                    key_b = f"{nuis_name}:B"
                    if key_b in cross_derivs:
                        ti_b = tracer_idx[pair_key[1]]
                        param_col_b = N_COSMO + ti_b * N_NUIS + ip_nuis
                        for il, ell in enumerate(ells):
                            if ell in cross_derivs[key_b]:
                                D[:, obs_offset + il, param_col_b] += np.asarray(
                                    cross_derivs[key_b][ell]
                                )

    # Invert covariance and accumulate Fisher
    F = np.zeros((Np, Np))
    for ik in range(Nk):
        try:
            cov_inv = np.linalg.inv(cov_mt[ik])
        except np.linalg.LinAlgError:
            continue
        DtCinv = D[ik].T @ cov_inv
        F += DtCinv @ D[ik] * dk

    return FisherResult(
        F=F,
        param_names=param_names,
        z_bin=z_bin,
        survey_name="Multi-tracer overlap",
        kmax=float(k[-1]),
    )
