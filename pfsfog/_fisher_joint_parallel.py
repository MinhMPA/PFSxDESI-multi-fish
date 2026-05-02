"""Parallel multi-process runner for the per-z-bin loop in
``pfsfog.fisher_joint.run_joint_fisher`` and ``run_broad_baseline``.

The per-z-bin Fisher build is embarrassingly parallel — different z-bins
are statistically independent under the Gaussian-covariance approximation,
so their Fisher matrices add. This module pools the z-bin work across
worker processes (spawn context, JAX-safe on macOS) and returns the same
list of ``(F, param_names)`` tuples that the sequential loop produces.

The module is *internal* (leading underscore) and only imported by
``fisher_joint.run_joint_fisher`` / ``run_broad_baseline`` when
``parallel=True``. The sequential code path is untouched.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .eft_params import COSMO_NAMES
from .surveys import SurveyGroup


# ---------------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------------


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


_JAX_CACHE_DIR = os.environ.get(
    "PFSFOG_JAX_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".cache" / "jax"),
)


def _init_worker() -> None:
    """Per-worker initialization run once when each spawn-process starts.

    By the time this runs, numpy/JAX may already be loaded by the spawn
    bootstrap (which imports this module to find the worker function),
    so setting BLAS env vars here is too late for the BLAS thread count.
    Throttling is therefore done in the parent (see ``_run_zbin_pool``)
    *before* the workers spawn — workers inherit the parent's env at
    spawn time.

    We additionally enable JAX's persistent compilation cache here so
    sibling workers and subsequent invocations can reuse JIT artifacts.
    The cost of JIT-compiling the one-loop EFT model is the dominant
    per-worker startup expense; sharing the cache amortizes it.
    """
    import jax  # noqa: F401
    jax.config.update("jax_enable_x64", True)
    Path(_JAX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


# Per-worker cache for FiducialCosmology / PowerSpectrum1Loop.
# Lazy-initialized on first task; reused across subsequent tasks so the
# ~30-60s setup cost is paid once per worker, not once per z-bin task.
_WORKER_COSMO = None
_WORKER_PS = None


def _get_worker_state(cfg):
    """Return (cosmo, ps), building them on first call inside this worker."""
    global _WORKER_COSMO, _WORKER_PS
    if _WORKER_COSMO is None:
        from .cosmo import FiducialCosmology
        from ps_1loop_jax import PowerSpectrum1Loop
        _WORKER_COSMO = FiducialCosmology(backend=cfg.cosmo_backend)
        _WORKER_PS = PowerSpectrum1Loop(do_irres=False)
    return _WORKER_COSMO, _WORKER_PS


# ---------------------------------------------------------------------------
# Worker functions (run inside spawned processes)
# ---------------------------------------------------------------------------


def _worker_zbin_joint(payload: tuple) -> tuple[np.ndarray, list[str]]:
    """Compute volume-partitioned per-z-bin Fisher for the joint analysis.

    Uses the per-worker cached ``cosmo``/``ps`` (lazy-initialized on first
    task) — these are JAX/cosmopower-stateful and rebuilding them per task
    would dominate wall time when each worker handles multiple z-bins.
    """
    zbin, sg, cfg, include_pfs = payload

    from .fisher_joint import volume_partitioned_zbin_fisher
    cosmo, ps = _get_worker_state(cfg)

    return volume_partitioned_zbin_fisher(
        zbin, sg, cosmo, ps, cfg, include_pfs=include_pfs,
    )


def _worker_zbin_broad(payload: tuple) -> tuple[np.ndarray, list[str], list[str]] | None:
    """Per-z-bin work for ``run_broad_baseline`` — single-tracer auto Fishers
    at full DESI volume, summed in the union parameter space.

    Returns ``(F_total, union_names, sorted_active_tracers)`` or ``None`` if
    the z-bin has no active DESI tracers (the parent skips it).
    """
    zbin, sg, cfg = payload

    from .fisher_joint import build_zbin_fisher, embed_fisher

    active_desi = sg.active_desi(zbin[0], zbin[1])
    if not active_desi:
        return None

    cosmo, ps = _get_worker_state(cfg)

    V_full = sg.V_desi_full(zbin[0], zbin[1])

    zbin_fishers = []
    zbin_names_list = []
    for tn, surv in active_desi.items():
        F, names = build_zbin_fisher(
            zbin, {tn: surv}, V_full, cosmo, ps, cfg,
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

    return (F_total, union, sorted(active_desi.keys()))


# ---------------------------------------------------------------------------
# Pool runner
# ---------------------------------------------------------------------------


def _run_zbin_pool(
    payloads: list[tuple],
    worker_fn: Any,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
) -> list:
    """Spawn a Pool of workers, execute ``worker_fn(payload)`` for each
    payload, return results in submission order.

    Uses spawn context (macOS-safe with JAX/XLA). BLAS thread count per
    worker is set in the parent's environment *before* spawning so that
    each worker inherits the configured thread count at the point of its
    numpy/JAX import. The default ``threads_per_worker=1`` avoids
    oversubscription for ``n_workers × threads`` ≤ ``cpu_count``; the
    user can override to balance worker count against per-worker BLAS
    threading.
    """
    n_total = len(payloads)
    if n_total == 0:
        return []

    n_default = min(n_total, os.cpu_count() or 1, 8)
    n = n_workers if n_workers is not None else n_default
    n = max(1, min(n, n_total))

    # Throttle BLAS threads per worker by setting env vars in the parent
    # before the workers spawn (spawn workers inherit parent env at start).
    # Restore parent env afterwards so the parent's numpy/JAX state is
    # unaffected.
    threads_str = str(max(1, int(threads_per_worker)))
    saved_env = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
    for var in _THREAD_ENV_VARS:
        os.environ[var] = threads_str
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n, initializer=_init_worker) as pool:
            # ``map`` preserves order — keeps reduction order deterministic.
            return pool.map(worker_fn, payloads)
    finally:
        for var, val in saved_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


# ---------------------------------------------------------------------------
# Pre-flight pickle check
# ---------------------------------------------------------------------------


def _check_picklable(name: str, obj) -> None:
    """Raise a helpful error if ``obj`` cannot be pickled for spawn dispatch."""
    try:
        pickle.dumps(obj)
    except Exception as e:
        raise TypeError(
            f"Object '{name}' is not picklable and cannot be sent to a "
            f"multiprocessing worker (spawn context): {type(e).__name__}: {e}\n"
            f"This usually means a closure or lambda is held somewhere; "
            f"replace with a module-level function or functools.partial."
        ) from e


# ---------------------------------------------------------------------------
# Top-level parallel dispatchers
# ---------------------------------------------------------------------------


def _run_joint_parallel(
    cfg,
    sg: SurveyGroup,
    include_pfs: bool,
    zbins: list[tuple[float, float]],
    n_workers: int | None = None,
    threads_per_worker: int = 1,
):
    """Pool the per-z-bin work for ``run_joint_fisher``.

    Returns ``(per_zbin_results, per_zbin_active)`` — the two lists that
    the sequential loop produces. The caller (``run_joint_fisher``) feeds
    them into the existing ``combine_zbins_heterogeneous`` and prior-add
    machinery, so post-processing remains identical to the sequential path.
    """
    _check_picklable("cfg", cfg)
    _check_picklable("sg", sg)

    payloads = [(zb, sg, cfg, include_pfs) for zb in zbins]
    per_zbin = _run_zbin_pool(
        payloads, _worker_zbin_joint,
        n_workers=n_workers, threads_per_worker=threads_per_worker,
    )

    # active-tracer list (cheap, do in parent for accuracy)
    per_zbin_active = []
    for zb in zbins:
        if include_pfs and zb[0] < sg.pfs_zmax:
            active = sorted(sg.active_with_pfs_truncation(zb[0], zb[1]).keys())
        else:
            active = sorted(sg.active_desi(zb[0], zb[1]).keys())
        per_zbin_active.append(active)

    return per_zbin, per_zbin_active


def _run_broad_parallel(
    cfg,
    sg: SurveyGroup,
    zbins: list[tuple[float, float]],
    n_workers: int | None = None,
    threads_per_worker: int = 1,
):
    """Pool the per-z-bin work for ``run_broad_baseline``.

    Returns ``(per_zbin_results, per_zbin_active)`` filtered to drop empty
    z-bins (those with no active DESI tracer), matching the sequential path.
    """
    _check_picklable("cfg", cfg)
    _check_picklable("sg", sg)

    payloads = [(zb, sg, cfg) for zb in zbins]
    raw = _run_zbin_pool(
        payloads, _worker_zbin_broad,
        n_workers=n_workers, threads_per_worker=threads_per_worker,
    )

    per_zbin = []
    per_zbin_active = []
    for r in raw:
        if r is None:
            continue
        F_total, union, active = r
        per_zbin.append((F_total, union))
        per_zbin_active.append(active)

    return per_zbin, per_zbin_active
