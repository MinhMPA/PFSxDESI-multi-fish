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
# Single-ELG pipeline (~30s)
python -m pfsfog

# Multi-tracer pipeline (up to 4 tracers, ~2min)
python -c "
from pfsfog.config import ForecastConfig
from pfsfog.cli_multitrace import run_multitrace_pipeline
cfg = ForecastConfig.from_yaml('configs/default.yaml')
run_multitrace_pipeline(cfg)
"

# 6-sample DESI DR2 forecast (~5min)
python scripts/run_desi_multisample.py

# Parameter importance decomposition
python scripts/run_parameter_importance.py

# Sensitivity sweep (r_sigma_v)
python scripts/run_sensitivity.py

# Tests
pytest tests/ -q   # 96 tests, ~10s
```

## Module guide

### Cosmology and surveys

| Module | Purpose |
|--------|---------|
| `cosmo.py` | `FiducialCosmology` — P_lin via cosmopower-jax; sigma8 from emulator; H, D, f from `ps_1loop_jax.background`. `make_plin_func()` and `make_growth_rate_func()` return pure JAX functions for end-to-end autodiff. Planck 2018 fiducial. |
| `surveys.py` | Load n(z) from `survey_specs/*.txt`. `Survey` dataclass with nbar_eff, z_eff, volume. `SurveyGroup` manages PFS + multiple DESI tracers. |
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
| `prior_export.py` | Regularize overlap Fisher -> extract calibrated sigma for each DESI nuisance param. |
| `fisher_full_area.py` | Single-tracer DESI Fisher with imported priors. `combine_zbins` (or `combine_samples`) sums cosmo blocks. Supports sample-label disambiguation for DR2 samples. |

### Scenarios and output

| Module | Purpose |
|--------|---------|
| `scenarios.py` | Three scenarios: broad, cross-cal, oracle. `nuisance_prior_diag` selects prior source. |
| `plots.py` | Publication figures (serif font, >=14pt). SBP benchmark lines from Zhang+ 2025 and Chudaykin+ 2026. |

### Pipeline runners

| Module | Purpose |
|--------|---------|
| `cli.py` | Single-ELG pipeline. `run_pipeline` -> `PipelineResults`. |
| `cli_multitrace.py` | Multi-tracer pipeline. PFS x {DESI-ELG, DESI-LRG, DESI-QSO}. |

## Key conventions

- **EFT parameterization**: Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757, Table I).
- **Cross-shot noise**: Zero for different populations. For PFS x DESI-ELG: `f_shared / nbar_DESI` where f_shared = n_shared / nbar_PFS = 0.05 (fiducial; results are insensitive to f_shared in [0, 1]).
- **Asymmetric kmax**: `kmax_PFS = kmax_DESI / r_sigma_v`. Default r_sigma_v = 0.75.
- **Units**: k in h/Mpc, P(k) in (Mpc/h)^3, volumes in (Mpc/h)^3.
- **Precision**: `jax.config.update('jax_enable_x64', True)` — float64 throughout.
