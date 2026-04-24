# pfsfog — Fisher forecast for PFS x DESI EFT prior calibration

> **Dependency**: requires [`ps_1loop_jax`](https://github.com/archaeo-pteryx/ps_1loop_jax) (Kobayashi & Akitsu, in prep.) for one-loop power spectrum evaluation.

Two-step pipeline:
1. **Step 1 (overlap calibration)**: Multi-tracer Fisher in the ~1,200 deg² PFS--DESI overlap. Up to 4 tracers (PFS-ELG, DESI-ELG, DESI-LRG, DESI-QSO), 10 cross-spectra per z-bin. Extracts calibrated priors for each DESI tracer's nuisance parameters.
2. **Step 2 (full-area forecast)**: Single-tracer auto-spectrum Fisher for each DESI DR2 sample (LRG1--3, ELG1--2, QSO) in 14,000 deg², using calibrated or broad priors. Combines via shared cosmological parameters.

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
| `config.py` | `ForecastConfig` — all tuneable knobs: kmin/kmax/dk, z-bins, areas, r_sigma_v, f_shared_elg=0.045, backend choices. YAML loader. |

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
| `covariance_mt_general.py` | N-tracer generalisation. Supports cross-shot noise for partially shared catalogues (f_shared). |

### Fisher matrices

| Module | Purpose |
|--------|---------|
| `fisher.py` | `FisherResult` dataclass with `marginalized_sigma`. Fisher assembly + `add_gaussian_prior`. |
| `fisher_mt_general.py` | N-tracer Fisher. Parameter vector: 3 cosmo + 12 nuisance per tracer. |
| `prior_export.py` | Regularise overlap Fisher -> extract calibrated sigma for each DESI nuisance param. |
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

- **EFT parameterisation**: Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757, Table I).
- **Cross-shot noise**: Zero for different populations. For PFS x DESI-ELG: `f_shared / nbar_PFS` where f_shared = 0.045 (J. Shi, priv. comm.).
- **Asymmetric kmax**: `kmax_PFS = kmax_DESI / r_sigma_v`. Default r_sigma_v = 0.75.
- **Units**: k in h/Mpc, P(k) in (Mpc/h)^3, volumes in (Mpc/h)^3.
- **Precision**: `jax.config.update('jax_enable_x64', True)` — float64 throughout.
