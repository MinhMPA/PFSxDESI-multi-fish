"""JAX persistent compilation cache configuration.

The cache lives at ``$PFSFOG_JAX_CACHE_DIR`` (default
``<repo>/.cache/jax/``) and is shared across:
  - the parent kernel / driver process (configured by ``enable_cache()``)
  - all spawned workers from ``_fisher_joint_parallel`` (configured in
    ``_init_worker``)

Without a parent-side configuration, JIT'd derivative kernels invoked
directly in the kernel (sequential mode, notebook diagnostic cells, or
top-level scripts) recompile from scratch every kernel session.

Caveat — cosmology JIT is *not* persisted to disk. The
``_dPell_d_OmMnu_jit`` compile traces through cosmopower-jax, whose
predict path embeds a JAX host callback. JAX refuses to serialize any
compile that contains host callbacks, so this kernel recompiles on
every kernel restart (~24 s through cosmopower's neural net). Within
one kernel session the in-memory JIT cache amortizes the cost across
all subsequent calls (~0.5 s each thereafter, ~50× faster than the
eager pre-JIT path), so the trade-off is still a clear win once
~3-5 z-bin builds have run.

The auto/cross nuisance JIT kernels and the fσ8 kernel do NOT use
host callbacks and are persisted normally — those compiles are paid
once across all kernel sessions for the same `(ps_model, ells, shapes)`
signature.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "jax"
JAX_CACHE_DIR = os.environ.get("PFSFOG_JAX_CACHE_DIR", str(_DEFAULT_CACHE_DIR))


_ENABLED = False


def enable_cache() -> None:
    """Enable JAX persistent compilation cache for this process.

    Idempotent — safe to call multiple times. Sets the cache directory,
    lowers the min-compile-time and min-entry-size thresholds so even
    small JIT kernels are persisted, and creates the cache directory if
    it doesn't exist.

    Must be called *before* any JIT compile that should be cached. We
    call it from ``pfsfog/__init__.py`` so any ``import pfsfog`` path
    is covered, and from ``_init_worker`` for spawned workers.
    """
    global _ENABLED
    if _ENABLED:
        return
    import jax  # noqa: F401  (deferred to keep pfsfog import cheap if jax is unused)
    Path(JAX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    _ENABLED = True
