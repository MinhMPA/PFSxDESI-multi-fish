# PFSFOG — Pooled Full-shape Surveys: Forecasts Of Galaxies

> **Dependency**: requires [`ps_1loop_jax`](https://github.com/archaeo-pteryx/ps_1loop_jax) (Kobayashi & Akitsu, in prep.) for one-loop power spectrum evaluation.

Joint multi-tracer, multi-survey Fisher analysis. The framework pools
Fisher information across the disjoint regions of N overlapping
spectroscopic surveys' footprints, with per-region multi-tracer Fishers
combined in a common parameter space (cosmology shared globally;
nuisance unique per (tracer, redshift bin)). Marginalize once over all
nuisance to get cosmology σ.

**Headline driver:** `scripts/run_joint_fisher.py` — runs DESI-only
joint vs DESI+PFS joint and reports σ(fσ8), σ(Mν), σ(Ωm) plus the
PFS-unique relative tightening.

The implementation realizes N=2 (Subaru/PFS × DESI overlap). The
architecture (`pfsfog/fisher_joint.py`) generalizes to N ≥ 3 by
enumerating the 2^N − 1 disjoint footprint regions.

**Legacy two-stage pipeline** (deprecated): `scripts/run_desi_multisample.py`
plus `pfsfog/prior_export.py` — kept for reproducibility.

## Quick start

```bash
# Joint Fisher (headline driver): DESI-only vs DESI+PFS (~14 min sequential)
python scripts/run_joint_fisher.py
python scripts/run_joint_fisher.py --parallel --n-workers 8   # ~7 min with warm JAX cache

# Tests
pytest tests/ -q                  # 130 tests, ~12 min (default — slow tests skipped)
pytest tests/ -m slow -v          # 2 parallel-vs-sequential equivalence tests
```

Legacy two-stage pipeline (kept for reproducibility):

```bash
python -m pfsfog                                 # legacy single-ELG pipeline
python scripts/run_desi_multisample.py           # legacy 6-sample DR2 forecast
python scripts/run_parameter_importance.py       # legacy diagnostic
python scripts/run_sensitivity.py                # legacy r_sigma_v sweep
```

## Module guide

### Cosmology and surveys

| Module | Purpose |
|--------|---------|
| `cosmo.py` | `FiducialCosmology` — P_lin via cosmopower-jax; sigma8 from emulator; H, D, f from `ps_1loop_jax.background`. `make_plin_func()` and `make_growth_rate_func()` return pure JAX functions for end-to-end autodiff. Planck 2018 fiducial. |
| `surveys.py` | Load n(z) from `survey_specs/*.txt`. `Survey` dataclass with nbar_eff, z_eff, volume. `SurveyGroup` manages PFS + multiple DESI tracers. `b1_of_z` callables are module-level functions / `functools.partial` so `Survey` objects are picklable (required for multiprocessing). |
| `config.py` | `ForecastConfig` — all tuneable knobs: kmin/kmax/dk, z-bins, areas, r_sigma_v, f_shared_elg=0.05, backend choices. YAML loader. |

### EFT parameters

| Module | Purpose |
|--------|---------|
| `eft_params.py` | `EFTFiducials` / `EFTPriors` dataclasses. Fiducial builders for all 4 tracers. c_tilde templates (Zhang+ and Ivanov). Lazeyras+ co-evolution for b2. Broad priors from Chudaykin+ 2025 Table I. |
| `ps1loop_adapter.py` | Maps Fisher sigma8-scaled parameters <-> `ps_1loop_jax` params dict. Handles cross-spectra. `make_ps1loop_pkmu_func` wrappers for covariance. |

### Derivatives

| Module | Purpose |
|--------|---------|
| `derivatives.py` | **All 15 parameters use JAX autodiff** (`jax.jacfwd`). Nuisance params: through `ps_1loop_jax.get_pk_ell`. Cosmo params (Omega_m, M_nu): end-to-end through `cosmopower-jax` emulator + growth rate + one-loop model. Validation: adaptive finite difference via `numdifftools`. |

### Covariance

| Module | Purpose |
|--------|---------|
| `covariance.py` | Gaussian multipole covariance via 20-point Gauss-Legendre. Single-tracer (3x3 for ell=0,2,4). |
| `covariance_mt_general.py` | N-tracer generalization. Supports cross-shot noise for partially shared catalogs (f_shared). |

### Fisher matrices

| Module | Purpose |
|--------|---------|
| `fisher.py` | `FisherResult` dataclass with `marginalized_sigma`. Fisher assembly + `add_gaussian_prior`. |
| `fisher_mt_general.py` | N-tracer Fisher. Parameter vector: 3 cosmo + 12 nuisance per tracer. |
| `fisher_joint.py` | **Joint multi-tracer, multi-survey Fisher.** Volume-partitioned per z-bin, heterogeneous combine across z-bins, opt-in `parallel`/`n_workers`/`threads_per_worker` kwargs on `run_joint_fisher` and `run_broad_baseline`. |
| `_fisher_joint_parallel.py` | Internal — spawn-context multiprocessing pool for the per-z-bin loop, with parent-side BLAS throttling and JAX persistent compilation cache. |
| `prior_export.py` | **Legacy** — extract calibrated σ for DESI nuisance from the two-stage overlap Fisher. Superseded by `fisher_joint.py`. |
| `fisher_full_area.py` | **Legacy** — single-tracer DESI Fisher with imported priors. `combine_zbins` assumes uniform tracer counts; superseded by `fisher_joint.combine_zbins_heterogeneous` for the joint analysis. |

### Scenarios and output

| Module | Purpose |
|--------|---------|
| `scenarios.py` | **Legacy** — broad / cross-cal / oracle scenario labels and `nuisance_prior_diag` for the two-stage pipeline. Joint Fisher uses "DESI-only joint" / "DESI+PFS joint" labels directly and does not import this module. |
| `plots.py` | Publication figures (serif font, >=14pt). SBP benchmark lines from Zhang+ 2025 and Chudaykin+ 2026. |

### Pipeline runners

| Module | Purpose |
|--------|---------|
| `cli.py` | **Legacy** — single-ELG two-stage CLI. `run_pipeline` -> `PipelineResults`. Superseded by `scripts/run_joint_fisher.py`. |
| `cli_multitrace.py` | **Legacy** — multi-tracer two-stage CLI (PFS × DESI). Superseded by `scripts/run_joint_fisher.py`. |

## Key conventions

- **EFT parameterization**: Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757, Table I).
- **Cross-shot noise**: Zero for different populations. For PFS x DESI-ELG: `f_shared / nbar_DESI` where f_shared = n_shared / nbar_PFS = 0.05 (fiducial; results are insensitive to f_shared in [0, 1]).
- **Asymmetric kmax**: `kmax_PFS = kmax_DESI / r_sigma_v`. Default r_sigma_v = 0.75.
- **Units**: k in h/Mpc, P(k) in (Mpc/h)^3, volumes in (Mpc/h)^3.
- **Precision**: `jax.config.update('jax_enable_x64', True)` — float64 throughout.
- **Parallel mode**: `run_joint_fisher` and `run_broad_baseline` accept opt-in `parallel=True`, `n_workers`, `threads_per_worker` kwargs. Workers spawn (macOS-safe with JAX/XLA) and share a JAX persistent compilation cache at `.cache/jax/` (override with `PFSFOG_JAX_CACHE_DIR`). Numerically identical to sequential (rtol=1e-10, verified by slow-marked tests).
- **Test markers**: default `pytest tests/ -q` skips `@pytest.mark.slow`. The two parallel-vs-sequential equivalence tests (~12 min) run via `pytest -m slow -v`.
