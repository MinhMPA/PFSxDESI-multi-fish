"""Joint multi-tracer Fisher analysis.

Replaces the two-stage (calibration → cosmology with priors) architecture
with a single joint Fisher over all tracers, all redshift bins, all
volumes — marginalizing over all nuisance parameters at the very end.

Key concepts
------------
**Volume partition.** At each z-bin we build the Fisher as
``F_zbin = F_overlap(V=overlap_area, all active incl. PFS)
         + F_nonoverlap(V=nonoverlap_area, DESI-only)``.
The two regions are statistically independent under the Gaussian
covariance approximation, so Fishers add in a common parameter space.

**Heterogeneous z-bin combination.** Each z-bin has its own active-tracer
set (e.g., LRG-only at z<0.6, four tracers at z=[0.8, 1.0]), so nuisance
parameter counts vary across z-bins. Cosmology (fσ8, Mν, Ωm) is shared
across z-bins; nuisance is unique per (tracer, z-bin).

**Cosmology in the Fisher.** Unlike ``multi_tracer_fisher_general`` (which
leaves the cosmo block of the derivative matrix at zero — the legacy
two-stage pipeline only used the multi-tracer Fisher to extract nuisance
priors), here we populate auto-spectra cosmology derivatives via
``dPell_d_cosmo_all`` so that cosmology constraints come out directly.
Cross-spectra cosmology derivatives are not yet wired (see
``derivatives.py:209``); this is a small underestimate of the cross
information but does not affect the dominant nuisance-calibration channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .covariance_mt_general import multi_tracer_cov_general
from .derivatives import (
    dPcross_dtheta_autodiff_all_jit,
    dPell_d_cosmo_all,
    dPell_dtheta_autodiff_all_jit,
)
from .eft_params import (
    COSMO_NAMES,
    COSMO_PRIOR_SIGMA,
    NUISANCE_NAMES,
    broad_priors,
    tracer_fiducials,
)
from .fisher import FisherResult
from .ps1loop_adapter import (
    fisher_to_ps1loop_auto,
    fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_cross_func,
    make_ps1loop_pkmu_func,
)
from .surveys import Survey, SurveyGroup


# ---------------------------------------------------------------------------
# Parameter naming
# ---------------------------------------------------------------------------


def _zbin_label(zbin: tuple[float, float]) -> str:
    return f"z{zbin[0]:.2f}-{zbin[1]:.2f}"


def zbin_param_names(
    tracer_names: list[str],
    zbin: tuple[float, float],
) -> list[str]:
    """Local per-z-bin parameter names: cosmo first, then nuisance per tracer."""
    label = _zbin_label(zbin)
    names = list(COSMO_NAMES)
    for tn in tracer_names:
        for n in NUISANCE_NAMES:
            names.append(f"{n}_{tn}_{label}")
    return names


def joint_param_names(
    zbins: list[tuple[float, float]],
    active_per_zbin: list[list[str]],
) -> list[str]:
    """Global joint-Fisher parameter names: cosmo + per-(tracer, z-bin) nuisance."""
    names = list(COSMO_NAMES)
    for zb, active in zip(zbins, active_per_zbin):
        label = _zbin_label(zb)
        for tn in active:
            for n in NUISANCE_NAMES:
                names.append(f"{n}_{tn}_{label}")
    return names


# ---------------------------------------------------------------------------
# Fisher embedding
# ---------------------------------------------------------------------------


def embed_fisher(
    F_small: np.ndarray,
    names_small: list[str],
    names_union: list[str],
) -> np.ndarray:
    """Lift a Fisher into a larger parameter space by name match.

    Entries of ``names_small`` are placed at their corresponding positions
    in ``names_union``. Parameters in ``names_union`` not in
    ``names_small`` get zero rows/columns (no information from this
    Fisher about those parameters).
    """
    Np = len(names_union)
    F_big = np.zeros((Np, Np))
    idx_in_big = [names_union.index(n) for n in names_small]
    for i_small, i_big in enumerate(idx_in_big):
        for j_small, j_big in enumerate(idx_in_big):
            F_big[i_big, j_big] = F_small[i_small, j_small]
    return F_big


# ---------------------------------------------------------------------------
# Per-z-bin Fisher with cosmology
# ---------------------------------------------------------------------------


def _build_pkmu_funcs(active, fids, ps1l_params, nbars, ps, pk_data,
                     s8, f_z, h, pairs):
    pkmu_funcs = {}
    for (a, b) in pairs:
        if a == b:
            pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(
                ps, pk_data, ps1l_params[a]
            )
        else:
            cross_params = fisher_to_ps1loop_cross(
                fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b]
            )
            pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(
                ps, pk_data, cross_params
            )
    return pkmu_funcs


def build_zbin_fisher(
    zbin: tuple[float, float],
    active_tracers: dict[str, Survey],
    volume: float,
    cosmo,
    ps,
    cfg,
    cross_shot: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the per-z-bin multi-tracer Fisher with cosmology in the matrix.

    Refactored from ``run_overlap_step1`` so the same assembly logic powers
    both the overlap and the non-overlap volume calls.

    Returns ``(F, param_names)`` where ``param_names`` follows
    ``zbin_param_names`` ordering: ``[fσ8, Mν, Ωm,
    nuis_tracer1_zbin..., nuis_tracer2_zbin..., ...]``.
    """
    zlo, zhi = zbin
    z_eff = 0.5 * (zlo + zhi)
    s8 = cosmo.sigma8(z_eff)
    f_z = float(cosmo.f(z_eff))
    h = cosmo.params["h"]
    pk_data = cosmo.pk_data(z_eff)
    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, cfg.kmax_desi_overlap + cfg.dk / 2, cfg.dk)

    tracer_names = sorted(active_tracers.keys())
    Nt = len(tracer_names)

    # Reference b1 for PFS counterterm scaling: prefer DESI-ELG if present.
    b1_ref = 1.3
    for tn in tracer_names:
        if "ELG" in tn and "PFS" not in tn:
            b1_ref = active_tracers[tn].b1_of_z(z_eff)
            break

    fids, ps1l_params, nbars = {}, {}, {}
    for tn in tracer_names:
        s = active_tracers[tn]
        b1 = s.b1_of_z(z_eff)
        nb = s.nbar_eff(zlo, zhi)
        nbars[tn] = nb
        fid = tracer_fiducials(tn, b1, s8, b1_ref=b1_ref,
                               r_sigma_v=cfg.r_sigma_v)
        par = fisher_to_ps1loop_auto(fid, s8, f_z, h, nb)
        fids[tn] = fid
        ps1l_params[tn] = par

    # Pairs ordering must match the covariance and the D matrix indexing
    pairs = [(a, a) for a in tracer_names]
    for i, a in enumerate(tracer_names):
        for j in range(i + 1, Nt):
            pairs.append((a, tracer_names[j]))

    pkmu_funcs = _build_pkmu_funcs(
        active_tracers, fids, ps1l_params, nbars, ps, pk_data,
        s8, f_z, h, pairs,
    )

    cov = multi_tracer_cov_general(
        tracer_names, pkmu_funcs, nbars, k, volume, cfg.dk, ells,
        cross_shot=cross_shot,
    )

    # Auto-spectrum derivatives: nuisance (JIT'd, vectorized) + cosmology (dict)
    # nuis_arr has shape (N_nuis, N_ell, Nk); the small cosmology dict
    # (3 params) is left as-is — it's not the dispatch bottleneck.
    derivs_auto = {}
    for tn in tracer_names:
        nuis_arr = dPell_dtheta_autodiff_all_jit(
            ps, jnp.array(k), pk_data, ps1l_params[tn], s8, ells=ells,
        )
        d_cosmo = dPell_d_cosmo_all(
            ps, jnp.array(k), pk_data, cosmo, ps1l_params[tn],
            z_eff, s8, ells,
        )
        derivs_auto[tn] = (nuis_arr, d_cosmo)

    # Cross-spectrum derivatives: nuisance only (cross-cosmo not wired).
    # cross_arr has shape (2, N_nuis, N_ell, Nk) — axis 0 is side ("A"=0, "B"=1).
    derivs_cross = {}
    for (a, b) in pairs:
        if a == b:
            continue
        cross_params = fisher_to_ps1loop_cross(
            fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b]
        )
        derivs_cross[(a, b)] = dPcross_dtheta_autodiff_all_jit(
            ps, jnp.array(k), pk_data, cross_params, s8, ells=ells,
        )

    F = _assemble_fisher_with_cosmo(
        tracer_names, pairs, derivs_auto, derivs_cross, cov, k,
        cfg.dk, zbin, ells,
    )
    param_names = zbin_param_names(tracer_names, zbin)
    return F, param_names


def _assemble_fisher_with_cosmo(
    tracer_names, pairs, derivs_auto, derivs_cross, cov_mt, k, dk,
    z_bin, ells,
):
    """Like ``multi_tracer_fisher_general`` but populates the cosmology block.

    Parameter ordering: ``[fσ8, Mν, Ωm,
    NUISANCE_NAMES × tracer_0, NUISANCE_NAMES × tracer_1, ...]`` —
    matches ``zbin_param_names``.

    ``derivs_auto[tn]`` is a ``(nuis_arr, cosmo_dict)`` tuple where
    ``nuis_arr`` has shape ``(N_nuis, N_ell, Nk)`` (NUISANCE_NAMES order,
    matches ``ells``). ``derivs_cross[pair_key]`` is a numpy array of
    shape ``(2, N_nuis, N_ell, Nk)``; axis 0 is side index (A=0, B=1).
    """
    Nt = len(tracer_names)
    Nell = len(ells)
    Nk = len(k)

    Npairs = len(pairs)
    Nobs = Npairs * Nell

    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)
    Np = N_COSMO + Nt * N_NUIS

    tracer_idx = {name: i for i, name in enumerate(tracer_names)}

    D = np.zeros((Nk, Nobs, Np))

    for ip_pair, (pA, pB) in enumerate(pairs):
        obs_offset = ip_pair * Nell

        if pA == pB:
            tracer_name = pA
            ti = tracer_idx[tracer_name]
            entry = derivs_auto.get(tracer_name)
            if entry is None:
                continue
            nuis_arr, d_cosmo = entry  # (N_nuis, N_ell, Nk), {cn: {ell: arr}}

            # Cosmology columns (auto only)
            for ic, cn in enumerate(COSMO_NAMES):
                if cn in d_cosmo:
                    for il, ell in enumerate(ells):
                        if ell in d_cosmo[cn]:
                            # Each tracer's auto-spectrum contributes its
                            # cosmology derivative on its own ell rows.
                            D[:, obs_offset + il, ic] = np.asarray(
                                d_cosmo[cn][ell]
                            )

            # Nuisance columns — straight slice from the JIT'd array
            col_lo = N_COSMO + ti * N_NUIS
            for il in range(Nell):
                D[:, obs_offset + il, col_lo:col_lo + N_NUIS] = (
                    nuis_arr[:, il, :].T
                )
        else:
            pair_key = (pA, pB) if pA < pB else (pB, pA)
            cross_arr = derivs_cross.get(pair_key)
            if cross_arr is None:
                continue
            # cross_arr shape: (2, N_nuis, N_ell, Nk); side 0=A, 1=B
            for is_, side_tracer in enumerate(pair_key):
                ti = tracer_idx[side_tracer]
                col_lo = N_COSMO + ti * N_NUIS
                for il in range(Nell):
                    D[:, obs_offset + il, col_lo:col_lo + N_NUIS] += (
                        cross_arr[is_, :, il, :].T
                    )

    F = np.zeros((Np, Np))
    for ik in range(Nk):
        eigvals = np.linalg.eigvalsh(cov_mt[ik])
        if np.any(eigvals <= 0):
            continue
        try:
            cov_inv = np.linalg.inv(cov_mt[ik])
        except np.linalg.LinAlgError:
            continue
        DtCinv = D[ik].T @ cov_inv
        F += DtCinv @ D[ik] * dk

    return F


# ---------------------------------------------------------------------------
# Volume partition
# ---------------------------------------------------------------------------


def volume_partitioned_zbin_fisher(
    zbin: tuple[float, float],
    sg: SurveyGroup,
    cosmo,
    ps,
    cfg,
    include_pfs: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Per-z-bin Fisher = F_overlap + F_nonoverlap, in a common param space.

    F_overlap uses all active tracers (PFS plus DESI tracers, subject to
    ``sg.pfs_zmax`` and ``include_pfs``) at ``sg.V_overlap``.
    F_nonoverlap uses the DESI-only active tracers at ``sg.V_nonoverlap``.

    The two Fishers live in different per-z-bin subspaces (PFS nuisance
    is only in F_overlap), so we lift both to the union before summing.
    """
    zlo, _ = zbin
    if include_pfs and zlo < sg.pfs_zmax:
        active_overlap = sg.active_with_pfs_truncation(zbin[0], zbin[1])
    else:
        active_overlap = sg.active_desi(zbin[0], zbin[1])
    active_nonoverlap = sg.active_desi(zbin[0], zbin[1])

    V_overlap = sg.V_overlap(zbin[0], zbin[1])
    V_nonoverlap = sg.V_nonoverlap(zbin[0], zbin[1])

    fishers, names_list = [], []

    if len(active_overlap) >= 1 and V_overlap > 0:
        cs = None
        if (cfg.f_shared_elg > 0
                and "PFS-ELG" in active_overlap
                and "DESI-ELG" in active_overlap):
            nbar_de = active_overlap["DESI-ELG"].nbar_eff(zbin[0], zbin[1])
            cs = {("DESI-ELG", "PFS-ELG"): cfg.f_shared_elg / nbar_de}
        F_o, names_o = build_zbin_fisher(
            zbin, active_overlap, V_overlap, cosmo, ps, cfg, cross_shot=cs,
        )
        fishers.append(F_o)
        names_list.append(names_o)

    if len(active_nonoverlap) >= 1 and V_nonoverlap > 0:
        F_n, names_n = build_zbin_fisher(
            zbin, active_nonoverlap, V_nonoverlap, cosmo, ps, cfg,
        )
        fishers.append(F_n)
        names_list.append(names_n)

    if not fishers:
        return np.zeros((len(COSMO_NAMES), len(COSMO_NAMES))), list(COSMO_NAMES)

    union: list[str] = list(COSMO_NAMES)
    for nl in names_list:
        for n in nl:
            if n not in union:
                union.append(n)

    F_total = np.zeros((len(union), len(union)))
    for F, nl in zip(fishers, names_list):
        F_total += embed_fisher(F, nl, union)

    return F_total, union


# ---------------------------------------------------------------------------
# Combine z-bins (heterogeneous)
# ---------------------------------------------------------------------------


def combine_zbins_heterogeneous(
    per_zbin_results: list[tuple[np.ndarray, list[str]]],
    survey_name: str = "joint",
) -> FisherResult:
    """Sum per-z-bin Fishers into a single global Fisher.

    Each input ``(F, names)`` has cosmology shared globally (first three
    entries of ``names``) plus z-bin-local nuisance parameters whose
    names already encode their (tracer, z-bin). Cosmology blocks add
    across z-bins; nuisance blocks land in their unique slots.
    """
    global_names: list[str] = list(COSMO_NAMES)
    for _, names in per_zbin_results:
        for n in names:
            if n not in global_names:
                global_names.append(n)

    Np = len(global_names)
    F = np.zeros((Np, Np))
    for F_zb, names_zb in per_zbin_results:
        F += embed_fisher(F_zb, names_zb, global_names)

    return FisherResult(
        F=F,
        param_names=global_names,
        z_bin=(0.0, 0.0),
        survey_name=survey_name,
        kmax=0.0,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class JointFisherResult:
    fisher: FisherResult
    sigma: dict[str, float]   # cosmology σ values
    per_zbin_active: list[list[str]]


def run_joint_fisher(
    cfg,
    cosmo,
    ps,
    sg: SurveyGroup,
    include_pfs: bool,
    zbins: list[tuple[float, float]],
    add_cosmo_priors: bool = True,
    parallel: bool = False,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
) -> JointFisherResult:
    """Build the joint multi-tracer Fisher across all z-bins.

    ``include_pfs=False`` runs the DESI-only joint baseline; ``True`` runs
    the full PFS+DESI joint analysis. The two scenarios are compared via
    their cosmology σ values.

    Parameters
    ----------
    parallel : bool, default False
        Opt-in flag. If True, the per-z-bin loop is dispatched across a
        ``multiprocessing.Pool`` (spawn context). The parallel path is
        numerically identical to the sequential path; default behavior
        is unchanged.
    n_workers : int, optional
        Number of worker processes when ``parallel=True``. Defaults to
        ``min(len(zbins), os.cpu_count(), 8)``.
    threads_per_worker : int, default 1
        BLAS thread budget per worker. Only used when ``parallel=True``.
        Tune to balance worker count against per-worker BLAS threading
        on your hardware (typical: ``n_workers × threads_per_worker``
        ≤ logical CPU count).
    """
    if parallel:
        from ._fisher_joint_parallel import _run_joint_parallel
        per_zbin, per_zbin_active = _run_joint_parallel(
            cfg, sg, include_pfs, zbins,
            n_workers=n_workers, threads_per_worker=threads_per_worker,
        )
    else:
        per_zbin = []
        per_zbin_active = []
        for zb in zbins:
            F, names = volume_partitioned_zbin_fisher(
                zb, sg, cosmo, ps, cfg, include_pfs=include_pfs,
            )
            per_zbin.append((F, names))

            # Track active tracers for reporting
            if include_pfs and zb[0] < sg.pfs_zmax:
                active = sorted(sg.active_with_pfs_truncation(zb[0], zb[1]).keys())
            else:
                active = sorted(sg.active_desi(zb[0], zb[1]).keys())
            per_zbin_active.append(active)

    fr = combine_zbins_heterogeneous(
        per_zbin,
        survey_name="DESI+PFS joint" if include_pfs else "DESI-only joint",
    )

    if add_cosmo_priors:
        prior_diag = np.zeros(len(fr.param_names))
        for i, pn in enumerate(fr.param_names):
            if pn in COSMO_PRIOR_SIGMA:
                prior_diag[i] = 1.0 / COSMO_PRIOR_SIGMA[pn] ** 2
            else:
                # Broad nuisance prior (None for b1_sigma8 → 0)
                for nuis in NUISANCE_NAMES:
                    if pn.startswith(nuis + "_"):
                        s = broad_priors().sigma_dict()[nuis]
                        if s is not None:
                            prior_diag[i] = 1.0 / s ** 2
                        break
        fr = FisherResult(
            F=fr.F + np.diag(prior_diag),
            param_names=fr.param_names,
            z_bin=fr.z_bin,
            survey_name=fr.survey_name,
            kmax=fr.kmax,
        )

    sigma = {cp: fr.marginalized_sigma(cp) for cp in COSMO_NAMES}
    return JointFisherResult(
        fisher=fr, sigma=sigma, per_zbin_active=per_zbin_active,
    )


def run_broad_baseline(
    cfg,
    cosmo,
    ps,
    sg: SurveyGroup,
    zbins: list[tuple[float, float]],
    add_cosmo_priors: bool = True,
    parallel: bool = False,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
) -> JointFisherResult:
    """DESI single-tracer broad-prior baseline (no multi-tracer info).

    Builds, per (DESI tracer, z-bin), an independent single-tracer
    auto-spectrum Fisher at the full DESI footprint volume, then combines
    them in the same (cosmo + per-(tracer, z-bin) nuisance) parameter
    space as the joint Fisher. Cross-spectra are excluded — this is the
    legacy "DESI alone, no multi-tracer" reference scenario in which the
    b1*sigma8-Mnu degeneracy runs free under the broad nuisance priors.

    Parameters
    ----------
    parallel : bool, default False
        Opt-in flag. If True, dispatches the per-z-bin loop across a
        multiprocessing pool (spawn context). Numerically identical to
        the sequential path.
    n_workers : int, optional
        Number of worker processes when ``parallel=True``.
    """
    if parallel:
        from ._fisher_joint_parallel import _run_broad_parallel
        per_zbin, per_zbin_active = _run_broad_parallel(
            cfg, sg, zbins,
            n_workers=n_workers, threads_per_worker=threads_per_worker,
        )
    else:
        per_zbin = []
        per_zbin_active = []
        for zb in zbins:
            active_desi = sg.active_desi(zb[0], zb[1])
            if not active_desi:
                continue
            V_full = sg.V_desi_full(zb[0], zb[1])

            zbin_fishers = []
            zbin_names_list = []
            for tn, surv in active_desi.items():
                F, names = build_zbin_fisher(
                    zb, {tn: surv}, V_full, cosmo, ps, cfg,
                )
                zbin_fishers.append(F)
                zbin_names_list.append(names)

            union: list[str] = list(COSMO_NAMES)
            for nl in zbin_names_list:
                for n in nl:
                    if n not in union:
                        union.append(n)
            F_total = np.zeros((len(union), len(union)))
            for F, nl in zip(zbin_fishers, zbin_names_list):
                F_total += embed_fisher(F, nl, union)

            per_zbin.append((F_total, union))
            per_zbin_active.append(sorted(active_desi.keys()))

    fr = combine_zbins_heterogeneous(
        per_zbin, survey_name="DESI single-tracer broad",
    )

    if add_cosmo_priors:
        prior_diag = np.zeros(len(fr.param_names))
        for i, pn in enumerate(fr.param_names):
            if pn in COSMO_PRIOR_SIGMA:
                prior_diag[i] = 1.0 / COSMO_PRIOR_SIGMA[pn] ** 2
            else:
                for nuis in NUISANCE_NAMES:
                    if pn.startswith(nuis + "_"):
                        s = broad_priors().sigma_dict()[nuis]
                        if s is not None:
                            prior_diag[i] = 1.0 / s ** 2
                        break
        fr = FisherResult(
            F=fr.F + np.diag(prior_diag),
            param_names=fr.param_names,
            z_bin=fr.z_bin,
            survey_name=fr.survey_name,
            kmax=fr.kmax,
        )

    sigma = {cp: fr.marginalized_sigma(cp) for cp in COSMO_NAMES}
    return JointFisherResult(
        fisher=fr, sigma=sigma, per_zbin_active=per_zbin_active,
    )
