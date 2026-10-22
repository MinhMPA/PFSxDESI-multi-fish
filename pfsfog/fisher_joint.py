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
    dPell_d_cosmo_all_jit,
    dPell_dtheta_autodiff_all_jit,
)
from .eft_params import (
    COSMO_NAMES,
    COSMO_PRIOR_SIGMA,
    CROSS_STOCH_NAMES,
    CROSS_STOCH_PRIOR_SIGMA,
    NUISANCE_NAMES,
    broad_priors,
    tracer_fiducials,
)
from .fisher import FisherResult
from .fisher_core import fisher_from_shells
from .ps1loop_adapter import (
    fisher_to_ps1loop_auto,
    fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_cross_func,
    make_ps1loop_pkmu_func,
)
from .surveys import Survey, SurveyGroup
from .survey_geometry import cell_number_density, support_cells


# ---------------------------------------------------------------------------
# Parameter naming
# ---------------------------------------------------------------------------


def _zbin_label(zbin: tuple[float, float]) -> str:
    return f"z{zbin[0]:.2f}-{zbin[1]:.2f}"


def _cross_pair_label(a: str, b: str) -> str:
    """Canonical pair label (sorted) so PFS×DESI matches DESI×PFS."""
    a_, b_ = (a, b) if a < b else (b, a)
    return f"{a_}x{b_}"


def zbin_param_names(
    tracer_names: list[str],
    zbin: tuple[float, float],
    marginalize_cross_stoch: bool = True,
) -> list[str]:
    """Local per-z-bin parameter names: cosmo, per-tracer nuisance, and
    (when ``marginalize_cross_stoch=True``) per-cross-pair stochastic
    parameters following Ebina & White (2024).
    """
    label = _zbin_label(zbin)
    names = list(COSMO_NAMES)
    for tn in tracer_names:
        for n in NUISANCE_NAMES:
            names.append(f"{n}_{tn}_{label}")
    if marginalize_cross_stoch:
        Nt = len(tracer_names)
        for i in range(Nt):
            for j in range(i + 1, Nt):
                pair_label = _cross_pair_label(tracer_names[i], tracer_names[j])
                for cs in CROSS_STOCH_NAMES:
                    names.append(f"{cs}_{pair_label}_{label}")
    return names


def joint_param_names(
    zbins: list[tuple[float, float]],
    active_per_zbin: list[list[str]],
    marginalize_cross_stoch: bool = True,
) -> list[str]:
    """Global joint-Fisher parameter names: cosmo + per-(tracer, z-bin) nuisance
    + (optional) per-(cross-pair, z-bin) stochastic.
    """
    names = list(COSMO_NAMES)
    for zb, active in zip(zbins, active_per_zbin):
        label = _zbin_label(zb)
        for tn in active:
            for n in NUISANCE_NAMES:
                names.append(f"{n}_{tn}_{label}")
        if marginalize_cross_stoch:
            Nt = len(active)
            for i in range(Nt):
                for j in range(i + 1, Nt):
                    pair_label = _cross_pair_label(active[i], active[j])
                    for cs in CROSS_STOCH_NAMES:
                        names.append(f"{cs}_{pair_label}_{label}")
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
                     s8, f_z, h, pairs, k_nl=0.7):
    pkmu_funcs = {}
    for (a, b) in pairs:
        if a == b:
            pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(
                ps, pk_data, ps1l_params[a]
            )
        else:
            cross_params = fisher_to_ps1loop_cross(
                fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b], k_nl=k_nl,
            )
            pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(
                ps, pk_data, cross_params
            )
    return pkmu_funcs


def _prepare_tracer_parameters(active, zbin, density_bin, reference, s8, f_z, h, cfg):
    """Freeze nominal-bin fiducials before applying observation selection.

    Only the number-density factors depend on the computational support cell.
    The supplied reference populations determine PFS's counterterm scaling.
    """
    z = sum(zbin) / 2
    reference = active if reference is None else reference
    ref_names = sorted(n for n in reference if "ELG" in n and "PFS" not in n)
    b1_ref = reference[ref_names[0]].b1_of_z(z) if ref_names else 1.3
    fids, params, nbars = {}, {}, {}
    for name, s in sorted(active.items()):
        nb = cell_number_density(s, *density_bin)
        if nb <= 0:
            raise ValueError(f"{name}: no density in support cell {density_bin}")
        fid = tracer_fiducials(name, s.b1_of_z(z), s8, b1_ref=b1_ref, r_sigma_v=cfg.r_sigma_v)
        fids[name] = fid
        params[name] = fisher_to_ps1loop_auto(fid, s8, f_z, h, nb, k_nl=cfg.k_nl)
        nbars[name] = nb
    return fids, params, nbars


def build_zbin_fisher(
    zbin: tuple[float, float],
    active_tracers: dict[str, Survey],
    volume: float,
    cosmo,
    ps,
    cfg,
    cross_shot: dict | None = None,
    marginalize_cross_stoch: bool | None = None,
    *,
    density_bin: tuple[float, float] | None = None,
    fiducial_tracers: dict[str, Survey] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the per-z-bin multi-tracer Fisher with cosmology in the matrix.

    Refactored from ``run_overlap_step1`` so the same assembly logic powers
    both the overlap and the non-overlap volume calls.

    When ``marginalize_cross_stoch=True`` (the conservative default,
    following Ebina & White 2024), each cross-pair (a,b) gains two
    free stochastic parameters — ``Pshot_cross`` (constant) and
    ``a2_cross`` ((kμ)² term) — with fiducial 0 and broad priors of
    width matching the auto-spectrum stochastic priors. The fiducial
    cross-shot in the covariance is therefore set to zero (any non-zero
    cross_shot dict from the caller is ignored when this flag is True).

    Returns ``(F, param_names)`` where ``param_names`` follows
    ``zbin_param_names`` ordering: ``[fσ8, Mν, Ωm,
    nuis_tracer1_zbin..., ..., Pshot_cross_axb_zbin, a2_cross_axb_zbin, ...]``.
    """
    # zbin is the nominal model/parameter bin; density_bin is support only.
    zlo, zhi = zbin
    density_bin = zbin if density_bin is None else density_bin
    z_eff = 0.5 * (zlo + zhi)
    s8 = cosmo.sigma8(z_eff)
    f_z = float(cosmo.f(z_eff))
    h = cosmo.params["h"]
    pk_data = cosmo.pk_data(z_eff)
    ells = (0, 2, 4)

    if marginalize_cross_stoch is None:
        marginalize_cross_stoch = getattr(cfg, "marginalize_cross_stoch", True)

    tracer_names = sorted(active_tracers.keys())
    Nt = len(tracer_names)

    # Asymmetric kmax. Each (tracer, tracer) pair has its own kmax — PFS
    # tolerates higher k than DESI because its FoG damping scale σ_v is
    # smaller (σ_v,PFS = r_sigma_v · σ_v,DESI). The k-grid is built up
    # to the largest kmax across all pairs; per-pair masking is then
    # applied inside `_assemble_fisher_with_cosmo` so that a (pair, ell)
    # observable contributes Fisher information only over modes where
    # both sides of the pair are trustworthy.
    has_pfs_in_zbin = any("PFS" in tn for tn in tracer_names)
    kmax_desi = cfg.kmax_desi_overlap
    kmax_pfs = cfg.compute_kmax_pfs() if has_pfs_in_zbin else kmax_desi
    kmax_cross = cfg.compute_kmax_cross() if has_pfs_in_zbin else kmax_desi

    def _pair_kmax(a: str, b: str) -> float:
        a_pfs = "PFS" in a
        b_pfs = "PFS" in b
        if a == b:
            return kmax_pfs if a_pfs else kmax_desi
        # cross: PFS×DESI uses kmax_cross; DESI×DESI uses kmax_desi
        if a_pfs ^ b_pfs:
            return kmax_cross
        return kmax_pfs if a_pfs else kmax_desi

    kmax_global = max(kmax_desi, kmax_pfs, kmax_cross)
    k = np.arange(cfg.kmin, kmax_global + cfg.dk / 2, cfg.dk)

    fids, ps1l_params, nbars = _prepare_tracer_parameters(
        active_tracers, zbin, density_bin, fiducial_tracers, s8, f_z, h, cfg,
    )

    # Pairs ordering must match the covariance and the D matrix indexing
    pairs = [(a, a) for a in tracer_names]
    for i, a in enumerate(tracer_names):
        for j in range(i + 1, Nt):
            pairs.append((a, tracer_names[j]))
    pair_kmax = np.array([_pair_kmax(a, b) for (a, b) in pairs])

    pkmu_funcs = _build_pkmu_funcs(
        active_tracers, fids, ps1l_params, nbars, ps, pk_data,
        s8, f_z, h, pairs, k_nl=cfg.k_nl,
    )

    # When the conservative cross-stochasticity prescription is enabled,
    # the cross-shot fiducial in the covariance is zero — the cross-stoch
    # parameters live in the Fisher D matrix instead.
    cov_cross_shot = None if marginalize_cross_stoch else cross_shot
    cov = multi_tracer_cov_general(
        tracer_names, pkmu_funcs, nbars, k, volume, cfg.dk, ells,
        cross_shot=cov_cross_shot,
    )

    # Auto-spectrum derivatives: nuisance + cosmology, both JIT'd & vectorized.
    # `nuis_arr` shape: (N_nuis, N_ell, Nk); `cosmo_arr` shape: (3, N_ell, Nk)
    # with axis 0 in COSMO_NAMES order.
    derivs_auto = {}
    for tn in tracer_names:
        nuis_arr = dPell_dtheta_autodiff_all_jit(
            ps, jnp.array(k), pk_data, ps1l_params[tn], s8, ells=ells,
        )
        cosmo_arr = dPell_d_cosmo_all_jit(
            ps, jnp.array(k), pk_data, cosmo, ps1l_params[tn],
            z_eff, s8, ells=ells,
        )
        derivs_auto[tn] = (nuis_arr, cosmo_arr)

    # Cross-spectrum derivatives: nuisance only (cross-cosmo not wired).
    # cross_arr has shape (2, N_nuis, N_ell, Nk) — axis 0 is side ("A"=0, "B"=1).
    derivs_cross = {}
    for (a, b) in pairs:
        if a == b:
            continue
        cross_params = fisher_to_ps1loop_cross(
            fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b], k_nl=cfg.k_nl,
        )
        derivs_cross[(a, b)] = dPcross_dtheta_autodiff_all_jit(
            ps, jnp.array(k), pk_data, cross_params, s8, ells=ells,
        )

    # Cross-stochastic derivatives (Ebina & White 2024 Eq. 3.7).
    # Per cross-pair (a,b) with a≠b, two new free parameters:
    #   Pshot_cross : constant cross shot noise → ∂P_ℓ/∂P = δ_{ℓ,0}/√(n_a n_b)
    #   a2_cross    : (kμ)² stochastic         → ∂P_ℓ/∂a₂ =
    #                 (k/k_NL)²/√(n_a n_b) · {1/3, 2/3, 0} for ℓ ∈ {0,2,4}
    # Fiducial 0 (independent populations); marginalized via broad priors.
    cross_stoch_derivs: dict[tuple[str, str], np.ndarray] | None
    if marginalize_cross_stoch:
        k_arr = np.asarray(k)
        cross_stoch_derivs = {}
        for (a, b) in pairs:
            if a == b:
                continue
            inv_nab = 1.0 / np.sqrt(nbars[a] * nbars[b])
            kkn2 = (k_arr / cfg.k_nl) ** 2
            # axis 0: cross-stoch param index (matches CROSS_STOCH_NAMES order)
            arr = np.zeros((len(CROSS_STOCH_NAMES), len(ells), len(k_arr)))
            # Pshot_cross: only the monopole picks it up
            i_P = CROSS_STOCH_NAMES.index("Pshot_cross")
            i_a2 = CROSS_STOCH_NAMES.index("a2_cross")
            for il, ell in enumerate(ells):
                if ell == 0:
                    arr[i_P, il, :] = inv_nab
                    arr[i_a2, il, :] = inv_nab * kkn2 * (1.0 / 3.0)
                elif ell == 2:
                    arr[i_a2, il, :] = inv_nab * kkn2 * (2.0 / 3.0)
                # ell == 4 → both stay 0 (only k⁰ and k²μ² contribute)
            cross_stoch_derivs[(a, b)] = arr
    else:
        cross_stoch_derivs = None

    F = _assemble_fisher_with_cosmo(
        tracer_names, pairs, derivs_auto, derivs_cross, cov, k,
        cfg.dk, zbin, ells, pair_kmax=pair_kmax,
        cross_stoch_derivs=cross_stoch_derivs,
    )
    param_names = zbin_param_names(
        tracer_names, zbin,
        marginalize_cross_stoch=marginalize_cross_stoch,
    )
    return F, param_names


def _assemble_fisher_with_cosmo(
    tracer_names, pairs, derivs_auto, derivs_cross, cov_mt, k, dk,
    z_bin, ells, pair_kmax=None, cross_stoch_derivs=None,
):
    """Like ``multi_tracer_fisher_general`` but populates the cosmology block.

    Parameter ordering: ``[fσ8, Mν, Ωm,
    NUISANCE_NAMES × tracer_0, NUISANCE_NAMES × tracer_1, ...,
    CROSS_STOCH_NAMES × cross_pair_0, CROSS_STOCH_NAMES × cross_pair_1, ...]``
    where the cross_pair block is appended only when
    ``cross_stoch_derivs`` is provided. Matches ``zbin_param_names``.

    ``derivs_auto[tn]`` is a ``(nuis_arr, cosmo_arr)`` tuple where
    ``nuis_arr`` has shape ``(N_nuis, N_ell, Nk)`` (NUISANCE_NAMES order)
    and ``cosmo_arr`` has shape ``(N_COSMO, N_ell, Nk)`` (COSMO_NAMES
    order). ``derivs_cross[pair_key]`` is a numpy array of shape
    ``(2, N_nuis, N_ell, Nk)`` with axis 0 the side index (A=0, B=1).

    ``pair_kmax`` (optional, shape ``(Npairs,)``) is the per-pair maximum
    k each pair contributes information up to. At each k-bin the active
    observable subset is selected and the covariance is inverted on that
    submatrix only. If ``pair_kmax`` is ``None``, all pairs are active at
    all k bins (symmetric-kmax fallback, matches the pre-asymmetric path).

    ``cross_stoch_derivs`` (optional dict[pair_key, (N_CS, N_ell, Nk)])
    introduces ``len(CROSS_STOCH_NAMES)`` extra free parameters per
    cross-pair following Ebina & White 2024 — fiducial 0, marginalized
    over via broad priors applied by the caller. If ``None``, no
    cross-stoch columns are added.
    """
    Nt = len(tracer_names)
    Nell = len(ells)
    Nk = len(k)

    Npairs = len(pairs)
    Nobs = Npairs * Nell

    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)
    Np_base = N_COSMO + Nt * N_NUIS

    # Cross-stoch parameter accounting (in pair iteration order).
    if cross_stoch_derivs is not None:
        N_CS = len(CROSS_STOCH_NAMES)
        cross_pair_order = [(a, b) for (a, b) in pairs if a != b]
        N_cs_extra = len(cross_pair_order) * N_CS
    else:
        cross_pair_order = []
        N_CS = 0
        N_cs_extra = 0
    Np = Np_base + N_cs_extra

    tracer_idx = {name: i for i, name in enumerate(tracer_names)}
    cs_pair_offset = {pair: Np_base + i * N_CS
                      for i, pair in enumerate(cross_pair_order)}

    D = np.zeros((Nk, Nobs, Np))

    for ip_pair, (pA, pB) in enumerate(pairs):
        obs_offset = ip_pair * Nell

        if pA == pB:
            tracer_name = pA
            ti = tracer_idx[tracer_name]
            entry = derivs_auto.get(tracer_name)
            if entry is None:
                continue
            nuis_arr, cosmo_arr = entry   # (N_nuis, N_ell, Nk), (N_COSMO, N_ell, Nk)

            # Cosmology columns: each tracer's auto contributes its own
            # cosmology derivative on its own ell rows.
            for il in range(Nell):
                D[:, obs_offset + il, :N_COSMO] = cosmo_arr[:, il, :].T

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

            # Cross-stoch columns: only this pair's observable carries them.
            if cross_stoch_derivs is not None:
                cs_arr = cross_stoch_derivs.get(pair_key)
                if cs_arr is not None:
                    cs_lo = cs_pair_offset[(pA, pB)]
                    for il in range(Nell):
                        D[:, obs_offset + il, cs_lo:cs_lo + N_CS] = (
                            cs_arr[:, il, :].T
                        )

    # Per-(observable, k) active mask. Each observable is a (pair, ell)
    # combo; an observable is active at k iff k <= pair_kmax[pair_idx].
    # When `pair_kmax` is None the mask is all True (symmetric-kmax mode).
    if pair_kmax is None:
        obs_active_per_k = np.ones((Nk, Nobs), dtype=bool)
    else:
        # Repeat each pair's kmax across its Nell observables.
        obs_kmax = np.repeat(np.asarray(pair_kmax, dtype=float), Nell)
        # +1e-12 tolerance so a k-bin exactly at kmax stays active.
        obs_active_per_k = k[:, None] <= (obs_kmax[None, :] + 1e-12)

    return fisher_from_shells(D, cov_mt, k, active=obs_active_per_k)


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
    """Sum radial/sky cells while sharing each nominal-bin nuisance vector."""
    cells = support_cells(sg, zbin, include_pfs=include_pfs,
                          exclude_elg1=cfg.exclude_elg1)
    baseline_populations = sg.active_with_pfs_truncation(*zbin)
    contributions = []
    for cell in cells:
        for tracers, volume in [(cell.overlap_tracers, cell.overlap_volume),
                                (cell.nonoverlap_tracers, cell.nonoverlap_volume)]:
            if not tracers or volume == 0:
                continue
            active = {n: sg.all_surveys[n] for n in tracers}
            cs = None
            if cfg.f_shared_elg > 0 and "PFS-ELG" in active and "DESI-ELG" in active:
                cs = {("DESI-ELG", "PFS-ELG"): cfg.f_shared_elg / cell.nbar["DESI-ELG"]}
            contributions.append(build_zbin_fisher(
                zbin, active, volume, cosmo, ps, cfg, cross_shot=cs,
                density_bin=cell.radial_bin, fiducial_tracers=baseline_populations,
            ))
    combined = combine_zbins_heterogeneous(contributions)
    return combined.F, combined.param_names


def independent_autos_zbin_fisher(zbin, sg, cosmo, ps, cfg):
    """Independent DESI autos on identical support cells, with no priors."""
    contributions = []
    baseline_populations = sg.active_with_pfs_truncation(*zbin)
    for cell in support_cells(sg, zbin, include_pfs=False, exclude_elg1=cfg.exclude_elg1):
        for name in cell.nonoverlap_tracers:
            contributions.append(build_zbin_fisher(
                zbin, {name: sg.desi_tracers[name]}, cell.overlap_volume + cell.nonoverlap_volume,
                cosmo, ps, cfg, density_bin=cell.radial_bin, fiducial_tracers=baseline_populations,
            ))
    combined = combine_zbins_heterogeneous(contributions)
    return combined.F, combined.param_names


def active_cell_tracers(sg, zbin, cfg, include_pfs):
    """Union of measured populations, respecting partial support/selection."""
    return sorted({name for cell in support_cells(
        sg, zbin, include_pfs=include_pfs, exclude_elg1=cfg.exclude_elg1,
    ) for name in cell.overlap_tracers + cell.nonoverlap_tracers})


# ---------------------------------------------------------------------------
# Combine z-bins (heterogeneous)
# ---------------------------------------------------------------------------


def _broad_prior_diag(param_names: list[str]) -> np.ndarray:
    """Build the broad-prior diagonal for a joint-Fisher parameter list.

    Recognizes three parameter categories by name prefix:
      - cosmology (matches COSMO_PRIOR_SIGMA)
      - per-tracer nuisance (12 NUISANCE_NAMES, broad_priors())
      - per-cross-pair stochastic (CROSS_STOCH_NAMES, CROSS_STOCH_PRIOR_SIGMA)

    Parameters with flat priors (e.g. b1_sigma8) get 0.0 in the diagonal.
    """
    nuis_widths = broad_priors().sigma_dict()
    out = np.zeros(len(param_names))
    for i, pn in enumerate(param_names):
        if pn in COSMO_PRIOR_SIGMA:
            out[i] = 1.0 / COSMO_PRIOR_SIGMA[pn] ** 2
            continue
        # Cross-stoch first (longer / more specific names) so they don't
        # accidentally match a NUISANCE_NAMES prefix.
        matched = False
        for cs in CROSS_STOCH_NAMES:
            if pn.startswith(cs + "_"):
                s = CROSS_STOCH_PRIOR_SIGMA.get(cs)
                if s is not None:
                    out[i] = 1.0 / s ** 2
                matched = True
                break
        if matched:
            continue
        for nuis in NUISANCE_NAMES:
            if pn.startswith(nuis + "_"):
                s = nuis_widths[nuis]
                if s is not None:
                    out[i] = 1.0 / s ** 2
                break
    return out


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

            per_zbin_active.append(active_cell_tracers(sg, zb, cfg, include_pfs))

    fr = combine_zbins_heterogeneous(
        per_zbin,
        survey_name="DESI+PFS joint" if include_pfs else "DESI-only joint",
    )

    if add_cosmo_priors:
        fr = FisherResult(
            F=fr.F + np.diag(_broad_prior_diag(fr.param_names)),
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
            active_desi = active_cell_tracers(sg, zb, cfg, False)
            if not active_desi:
                continue
            per_zbin.append(independent_autos_zbin_fisher(zb, sg, cosmo, ps, cfg))
            per_zbin_active.append(active_desi)

    fr = combine_zbins_heterogeneous(
        per_zbin, survey_name="DESI single-tracer broad",
    )

    if add_cosmo_priors:
        fr = FisherResult(
            F=fr.F + np.diag(_broad_prior_diag(fr.param_names)),
            param_names=fr.param_names,
            z_bin=fr.z_bin,
            survey_name=fr.survey_name,
            kmax=fr.kmax,
        )

    sigma = {cp: fr.marginalized_sigma(cp) for cp in COSMO_NAMES}
    return JointFisherResult(
        fisher=fr, sigma=sigma, per_zbin_active=per_zbin_active,
    )
