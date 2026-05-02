#!/usr/bin/env python
"""Phase 0 benchmark — does multi-threaded BLAS already saturate CPU?

Spawns two subprocesses with different threading env vars and times
`run_joint_fisher(include_pfs=False)` (the cheaper of the three Fishers,
which is enough to measure scaling). Reports the BLAS scaling
efficiency S = T_single_thread / T_multi_thread and emits a go/no-go
verdict per the plan in /Users/nguyenmn/.claude/plans/splendid-beaming-sutherland.md.

Usage:
    python scripts/_bench_joint_fisher.py
    python scripts/_bench_joint_fisher.py --inner --threads 1   # subprocess mode
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _inner(threads: int, parallel: bool = False,
           n_workers: int | None = None,
           threads_per_worker: int = 1) -> float:
    """Subprocess entry point — assumes thread env vars are already set."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # JAX must be imported here, after the parent set the env vars.
    import jax
    jax.config.update("jax_enable_x64", True)

    from pfsfog.config import ForecastConfig
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.fisher_joint import run_joint_fisher
    from pfsfog.surveys import (
        SurveyGroup, desi_elg, desi_lrg, desi_qso, pfs_elg,
    )
    from scripts.run_joint_fisher import ZBINS
    from ps_1loop_jax import PowerSpectrum1Loop

    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    ps = PowerSpectrum1Loop(do_irres=False)
    sg = SurveyGroup(
        pfs=pfs_elg(),
        desi_tracers={
            "DESI-ELG": desi_elg(),
            "DESI-LRG": desi_lrg(),
            "DESI-QSO": desi_qso(),
        },
        overlap_area_deg2=cfg.overlap_area_deg2,
        desi_full_area_deg2=cfg.desi_area_deg2,
        pfs_zmax=1.6,
    )

    t0 = time.perf_counter()
    res = run_joint_fisher(
        cfg, cosmo, ps, sg, include_pfs=False, zbins=ZBINS,
        parallel=parallel, n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )
    t = time.perf_counter() - t0

    label = (f"threads={threads}, parallel={parallel}, "
             f"n_workers={n_workers}, tpw={threads_per_worker}")
    print(f"[inner {label}] elapsed = {t:.2f} s   "
          f"sigma(fs8)={res.sigma['fsigma8']:.4e}")
    return t


def _run_subprocess(threads: int, parallel: bool = False,
                    n_workers: int | None = None) -> float:
    """Spawn this script as a subprocess with the given thread budget."""
    env = os.environ.copy()
    for var in THREAD_ENV_VARS:
        env[var] = str(threads)
    env["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen={'true' if threads > 1 else 'false'}"
    )
    cmd = [sys.executable, __file__, "--inner", "--threads", str(threads)]
    if parallel:
        cmd.append("--parallel")
    if n_workers:
        cmd += ["--n-workers", str(n_workers)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    print(out.stdout, end="")
    if out.stderr:
        print(out.stderr, end="", file=sys.stderr)
    import re
    m = re.search(r"elapsed\s*=\s*([0-9.]+)\s*s", out.stdout)
    if not m:
        raise RuntimeError(f"Could not parse elapsed time from output:\n{out.stdout}")
    return float(m.group(1))


def _run_subprocess_parallel(n_workers: int = 8) -> float:
    """Spawn a subprocess that runs the Fisher in parallel mode."""
    return _run_subprocess(threads=1, parallel=True, n_workers=n_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true",
                    help="Internal subprocess mode (do not invoke directly).")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    args = ap.parse_args()

    if args.inner:
        if args.threads is None:
            ap.error("--inner requires --threads")
        _inner(args.threads, parallel=args.parallel,
               n_workers=args.n_workers,
               threads_per_worker=args.threads_per_worker)
        return

    n_cores = os.cpu_count() or 1

    print(f"Benchmarking on {n_cores}-core CPU.")
    print(f"Workload: run_joint_fisher(include_pfs=False) over 8 z-bins.")
    print(f"Each timing is one cold run (compile + execute).\n")

    print(f"=== Run 1: multi-threaded BLAS ({n_cores} threads) ===")
    T_mt = _run_subprocess(n_cores)
    print()

    print(f"=== Run 2: single-threaded BLAS (1 thread) ===")
    T_st = _run_subprocess(1)
    print()

    print(f"=== Run 3: parallel (8 workers, 1 BLAS thread each) ===")
    T_par = _run_subprocess_parallel(n_workers=8)
    print()

    S = T_st / T_mt if T_mt > 0 else float("inf")
    speedup = T_mt / T_par if T_par > 0 else float("inf")
    print(f"=== Result ===")
    print(f"  T_seq_mt  ({n_cores} threads) = {T_mt:.2f} s")
    print(f"  T_seq_st  (1 thread)          = {T_st:.2f} s")
    print(f"  T_par     (8 workers × 1 t)   = {T_par:.2f} s")
    print(f"  S = T_st / T_mt               = {S:.2f}")
    print(f"  Speedup  = T_mt / T_par       = {speedup:.2f}×")
    print()
    if S >= 6:
        verdict = (f"BLAS scales nearly perfectly (S={S:.1f}). "
                   f"Multiprocessing adds little.")
    elif S >= 3:
        verdict = (f"Typical scaling (S={S:.1f}). "
                   f"Multiprocessing helps; tune n_workers vs OMP_NUM_THREADS.")
    else:
        verdict = (f"BLAS plateaus early (S={S:.1f} < 3) — "
                   f"multiprocessing should give large gains.")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
