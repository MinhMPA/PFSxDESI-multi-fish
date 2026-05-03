"""Tests for the joint multi-tracer Fisher analysis.

Validates:
1. ``embed_fisher`` round-trip
2. Volume partition identity: F(V_full) ≈ F(V_overlap) + F(V_nonoverlap)
3. Heterogeneous = single-Fisher reduction when only one Fisher contributes
4. PFS-empty reduction: include_pfs=False reproduces DESI-only path
5. End-to-end smoke: run_joint_fisher returns finite sensible σ values
6. Survey picklability — required for multiprocessing the per-z-bin loop
"""
from __future__ import annotations

import pickle

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import COSMO_NAMES
from pfsfog.fisher_joint import (
    build_zbin_fisher,
    combine_zbins_heterogeneous,
    embed_fisher,
    run_joint_fisher,
    volume_partitioned_zbin_fisher,
    zbin_param_names,
)
from pfsfog.surveys import (
    SurveyGroup, desi_elg, desi_lrg, desi_qso, pfs_elg,
)


@pytest.fixture(scope="module")
def cosmo():
    return FiducialCosmology(backend="cosmopower")


@pytest.fixture(scope="module")
def ps():
    from ps_1loop_jax import PowerSpectrum1Loop
    return PowerSpectrum1Loop(do_irres=False)


@pytest.fixture(scope="module")
def cfg():
    c = ForecastConfig()
    c.kmin = 0.01
    c.kmax_desi_overlap = 0.20
    c.dk = 0.005
    return c


@pytest.fixture(scope="module")
def sg():
    return SurveyGroup(
        pfs=pfs_elg(),
        desi_tracers={
            "DESI-ELG": desi_elg(),
            "DESI-LRG": desi_lrg(),
            "DESI-QSO": desi_qso(),
        },
        overlap_area_deg2=1200.0,
        desi_full_area_deg2=14000.0,
        pfs_zmax=1.6,
    )


# ---------------------------------------------------------------------------
# Test 1: embed_fisher round-trip
# ---------------------------------------------------------------------------


def test_surveys_picklable():
    """Survey objects must round-trip through pickle for multiprocessing.

    The bias callable ``b1_of_z`` was historically a lambda / nested
    closure, which is not picklable. After replacing with a module-level
    function (PFS-ELG) and ``functools.partial`` (DESI tracers), all
    Survey constructors must produce picklable objects.
    """
    from pfsfog.surveys import pfs_elg, desi_elg, desi_lrg, desi_qso

    cases = [
        (pfs_elg,  0.85),
        (desi_elg, 0.95),
        (desi_lrg, 0.70),
        (desi_qso, 1.50),
    ]
    for ctor, ztest in cases:
        s = ctor()
        s_round = pickle.loads(pickle.dumps(s))
        # b1(z) must be unchanged after round-trip
        assert s.b1_of_z(ztest) == pytest.approx(s_round.b1_of_z(ztest), rel=1e-12)
        # nz_eff must be unchanged
        assert s.nbar_eff(ztest, ztest + 0.05) == pytest.approx(
            s_round.nbar_eff(ztest, ztest + 0.05), rel=1e-12
        )


def test_embed_fisher_roundtrip():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    F_small = A @ A.T  # PSD

    names_small = ["a", "b", "c"]
    names_union = ["a", "x", "b", "y", "c"]

    F_big = embed_fisher(F_small, names_small, names_union)

    # Shape
    assert F_big.shape == (5, 5)

    # Original block recovered at its name positions
    idx = [names_union.index(n) for n in names_small]
    sub = F_big[np.ix_(idx, idx)]
    np.testing.assert_allclose(sub, F_small, rtol=1e-15)

    # Other rows/cols are zero
    keep_mask = np.zeros(5, dtype=bool)
    keep_mask[idx] = True
    assert np.allclose(F_big[~keep_mask, :], 0.0)
    assert np.allclose(F_big[:, ~keep_mask], 0.0)


# ---------------------------------------------------------------------------
# Test 2: Volume partition identity
# ---------------------------------------------------------------------------


def test_volume_partition_identity(cosmo, ps, cfg, sg):
    """F at V_full = F at V_overlap + F at V_nonoverlap (DESI-only).

    The Gaussian Fisher scales linearly with volume, so this is exact algebra.
    """
    zbin = (1.0, 1.2)
    active_desi = sg.active_desi(*zbin)
    assert len(active_desi) >= 2  # need at least 2 tracers in this z-bin

    V_overlap = sg.V_overlap(*zbin)
    V_nonoverlap = sg.V_nonoverlap(*zbin)
    V_full = sg.V_desi_full(*zbin)
    np.testing.assert_allclose(V_overlap + V_nonoverlap, V_full, rtol=1e-12)

    F_full, names_full = build_zbin_fisher(
        zbin, active_desi, V_full, cosmo, ps, cfg,
    )
    F_ov, names_ov = build_zbin_fisher(
        zbin, active_desi, V_overlap, cosmo, ps, cfg,
    )
    F_no, names_no = build_zbin_fisher(
        zbin, active_desi, V_nonoverlap, cosmo, ps, cfg,
    )

    # Same parameter ordering — direct addition
    assert names_full == names_ov == names_no
    F_sum = F_ov + F_no

    # Allow 1e-9 relative; numerical noise from per-k inversions
    np.testing.assert_allclose(F_sum, F_full, rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------------------
# Test 3: Heterogeneous combine reduces to identity for a single Fisher
# ---------------------------------------------------------------------------


def test_combine_zbins_single_input(cosmo, ps, cfg, sg):
    zbin = (0.8, 1.0)
    active = sg.active_desi(*zbin)
    F, names = build_zbin_fisher(zbin, active, sg.V_overlap(*zbin),
                                 cosmo, ps, cfg)
    fr = combine_zbins_heterogeneous([(F, names)], survey_name="single")

    # Combined F equals the original (modulo possible param reordering)
    for i, n in enumerate(names):
        j = fr.param_names.index(n)
        for i2, n2 in enumerate(names):
            j2 = fr.param_names.index(n2)
            np.testing.assert_allclose(fr.F[j, j2], F[i, i2], rtol=1e-15)


# ---------------------------------------------------------------------------
# Test 4: PFS-empty reduction (include_pfs=False excludes PFS nuisance)
# ---------------------------------------------------------------------------


def test_pfs_empty_reduction(cosmo, ps, cfg, sg):
    zbin = (1.0, 1.2)
    F, names = volume_partitioned_zbin_fisher(
        zbin, sg, cosmo, ps, cfg, include_pfs=False,
    )

    # No PFS nuisance parameters in the joint param space
    pfs_params = [n for n in names if "PFS-ELG" in n]
    assert pfs_params == []


# ---------------------------------------------------------------------------
# Test 5: End-to-end smoke — joint Fisher returns sensible σ
# ---------------------------------------------------------------------------


def test_run_joint_fisher_smoke(cosmo, ps, cfg, sg):
    zbins = [(0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]

    res_desi = run_joint_fisher(cfg, cosmo, ps, sg,
                                include_pfs=False, zbins=zbins)
    res_joint = run_joint_fisher(cfg, cosmo, ps, sg,
                                 include_pfs=True, zbins=zbins)

    for cp in COSMO_NAMES:
        assert np.isfinite(res_desi.sigma[cp])
        assert res_desi.sigma[cp] > 0
        assert np.isfinite(res_joint.sigma[cp])
        assert res_joint.sigma[cp] > 0
        # Joint must be at least as tight as DESI-only (more data)
        assert res_joint.sigma[cp] <= res_desi.sigma[cp] + 1e-12

    # PFS-unique improvement should be small but positive on σ(fσ8)
    delta_fs8 = (res_desi.sigma["fsigma8"] - res_joint.sigma["fsigma8"]) \
                / res_desi.sigma["fsigma8"]
    assert 0.0 <= delta_fs8 < 0.5  # 0–50% range


# ---------------------------------------------------------------------------
# Test 7: parallel-vs-sequential equivalence (slow-marked)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_joint_fisher_parallel_matches_sequential(cosmo, ps, cfg, sg):
    """run_joint_fisher(parallel=True) must produce a Fisher matrix
    numerically identical (rtol=1e-10) to the sequential path."""
    from pfsfog.fisher_joint import run_joint_fisher
    zbins = [(0.8, 1.0), (1.0, 1.2)]  # 2 bins keeps test ~1 min

    res_seq = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=True,
                               zbins=zbins, parallel=False)
    res_par = run_joint_fisher(cfg, cosmo, ps, sg, include_pfs=True,
                               zbins=zbins, parallel=True, n_workers=2)

    assert res_seq.fisher.param_names == res_par.fisher.param_names
    np.testing.assert_allclose(res_seq.fisher.F, res_par.fisher.F,
                               rtol=1e-10, atol=0)
    for cp in COSMO_NAMES:
        np.testing.assert_allclose(
            res_seq.sigma[cp], res_par.sigma[cp], rtol=1e-10
        )


def test_assemble_kmax_mask_inert_when_uniform():
    """``_assemble_fisher_with_cosmo`` must produce identical Fisher when
    every pair shares the same kmax — i.e. the asymmetric-kmax masking
    is a no-op in the symmetric case (pair_kmax=None vs pair_kmax larger
    than k.max()).

    This pins that the masking machinery doesn't perturb the
    symmetric-kmax code path; only when at least one pair has a
    different kmax does the masked-submatrix Fisher differ.
    """
    from pfsfog.fisher_joint import _assemble_fisher_with_cosmo
    from pfsfog.eft_params import COSMO_NAMES, NUISANCE_NAMES

    rng = np.random.default_rng(0)

    # Two fake tracers, three pairs total (two autos + one cross), three ells.
    tracer_names = ["A", "B"]
    pairs = [("A", "A"), ("B", "B"), ("A", "B")]
    Nell = 3
    Nk = 12
    k = np.linspace(0.01, 0.20, Nk)
    dk = k[1] - k[0]
    ells = (0, 2, 4)

    N_COSMO = len(COSMO_NAMES)
    N_NUIS = len(NUISANCE_NAMES)

    # Random nonzero derivatives + an SPD covariance per k-bin.
    derivs_auto = {}
    for tn in tracer_names:
        nuis_arr = rng.standard_normal((N_NUIS, Nell, Nk))
        cosmo_arr = rng.standard_normal((N_COSMO, Nell, Nk))
        derivs_auto[tn] = (nuis_arr, cosmo_arr)
    derivs_cross = {("A", "B"): rng.standard_normal((2, N_NUIS, Nell, Nk))}

    Nobs = len(pairs) * Nell
    cov_mt = np.empty((Nk, Nobs, Nobs))
    for ik in range(Nk):
        a = rng.standard_normal((Nobs, Nobs))
        cov_mt[ik] = a @ a.T + np.eye(Nobs)   # SPD

    F_none = _assemble_fisher_with_cosmo(
        tracer_names, pairs, derivs_auto, derivs_cross, cov_mt, k, dk,
        z_bin=(0.8, 1.0), ells=ells, pair_kmax=None,
    )
    # Uniform pair_kmax above k.max() → all observables active at every k.
    F_uniform = _assemble_fisher_with_cosmo(
        tracer_names, pairs, derivs_auto, derivs_cross, cov_mt, k, dk,
        z_bin=(0.8, 1.0), ells=ells,
        pair_kmax=np.full(len(pairs), k.max() + 1.0),
    )
    np.testing.assert_array_equal(F_none, F_uniform)

    # And: a tighter pair_kmax that excludes the last few k-bins for
    # one pair should produce a *different* (smaller) Fisher.
    tight = np.full(len(pairs), k.max() + 1.0)
    tight[2] = k[Nk // 2]   # cross pair: drop top half of modes
    F_tight = _assemble_fisher_with_cosmo(
        tracer_names, pairs, derivs_auto, derivs_cross, cov_mt, k, dk,
        z_bin=(0.8, 1.0), ells=ells, pair_kmax=tight,
    )
    assert not np.allclose(F_tight, F_none)


@pytest.mark.slow
def test_run_broad_baseline_parallel_matches_sequential(cosmo, ps, cfg, sg):
    """run_broad_baseline(parallel=True) must match the sequential path."""
    from pfsfog.fisher_joint import run_broad_baseline
    zbins = [(0.8, 1.0), (1.0, 1.2)]

    res_seq = run_broad_baseline(cfg, cosmo, ps, sg,
                                 zbins=zbins, parallel=False)
    res_par = run_broad_baseline(cfg, cosmo, ps, sg,
                                 zbins=zbins, parallel=True, n_workers=2)

    assert res_seq.fisher.param_names == res_par.fisher.param_names
    np.testing.assert_allclose(res_seq.fisher.F, res_par.fisher.F,
                               rtol=1e-10, atol=0)
    for cp in COSMO_NAMES:
        np.testing.assert_allclose(
            res_seq.sigma[cp], res_par.sigma[cp], rtol=1e-10
        )
