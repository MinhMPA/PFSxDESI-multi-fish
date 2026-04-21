# pfsfog — Fisher forecast for PFS × DESI EFT prior calibration

Fisher forecast pipeline that uses the ~1,200 deg² overlap between PFS-ELG and DESI tracers (ELG, LRG, QSO) to cross-calibrate EFT nuisance parameters, then exports the calibrated priors to DESI's full 14,000 deg² footprint.

## Quick start

```bash
conda activate sbi_pytorch_osx-arm64_py310forge

# Run the ELG-only pipeline (2-tracer, ~30s)
python -m pfsfog.cli --config configs/default.yaml

# Run the multi-tracer pipeline (up to 4 tracers, ~2min)
python -c "
from pfsfog.config import ForecastConfig
from pfsfog.cli_multitrace import run_multitrace_pipeline
cfg = ForecastConfig.from_yaml('configs/default.yaml')
run_multitrace_pipeline(cfg)
"

# Run pair comparison (PFS×ELG vs PFS×LRG vs all combined)
python scripts/run_pair_comparison.py

# Run sensitivity sweep (r_σv = 0.5–1.0)
python scripts/run_sensitivity.py

# Tests
pytest tests/ -q   # 90 tests, ~10s
```

## Output

Each pipeline run creates a timestamped directory under `results/`:

```
results/YYYYMMDD_HHMMSS/
├── summary.csv                          # σ per scenario × parameter
├── config_snapshot.yaml                 # reproduciblity
├── priors/
│   └── cross_calibrated_z0.8_1.0.json  # calibrated σ per nuisance param
└── figures/
    ├── fig1_overlap_calibration.pdf
    ├── fig2_calibrated_vs_broad.pdf
    ├── fig3_full_area_constraints.pdf
    ├── fig4_calibration_efficiency.pdf
    └── fig5_sensitivity_rsigmav.pdf
```

## Module guide

### Cosmology and surveys

| Module | Purpose |
|--------|---------|
| `cosmo.py` | `FiducialCosmology` — P_lin via cosmopower-jax (default) or clax (fallback); σ8 from emulator; H, D, f, χ from `ps_1loop_jax.background`. Planck 2018 fiducial. |
| `surveys.py` | Load n(z) from `survey_specs/*.txt`. `Survey` dataclass with nbar_eff, z_eff, volume. `SurveyGroup` manages PFS + multiple DESI tracers, determines active tracers per z-bin. |
| `config.py` | `ForecastConfig` — all tuneable knobs: kmin/kmax/dk, z-bins, areas, r_σv, asymmetric kmax formula, f_shared, backend choices. YAML loader. |

### EFT parameters

| Module | Purpose |
|--------|---------|
| `eft_params.py` | `EFTFiducials` / `EFTPriors` dataclasses. Fiducial builders for all 4 tracers: `desi_elg_fiducials`, `desi_lrg_fiducials` (c̃=800), `desi_qso_fiducials` (c̃=1200), `pfs_elg_fiducials` (scaled by r_σv). Lazeyras+ co-evolution for b2. Broad priors from Chudaykin+ 2025 Table I. |
| `ps1loop_adapter.py` | Maps Fisher σ8-scaled parameters ↔ `ps_1loop_jax` params dict and `clax.ept` flat args. Handles cross-spectra (bias2/ctr2). Provides `make_ps1loop_pkmu_func` wrappers for covariance evaluation. |

### Theory and derivatives

| Module | Purpose |
|--------|---------|
| `derivatives.py` | Primary: JAX autodiff (`jax.jacfwd`) through `ps_1loop_jax.get_pk_ell`. Validation: 5-point stencil. Handles nuisance params (auto + cross spectra) and cosmological params (fσ8 via autodiff; Ωm, Mν via stencil through cosmopower-jax). |
| `builtin_pkmu.py` | Lightweight Kaiser + counterterm + FoG P(k,μ) model. Used only for unit tests — not for production forecasts. |

### Covariance

| Module | Purpose |
|--------|---------|
| `covariance.py` | Gaussian multipole covariance Cov_{ℓℓ'}(k) via 20-point Gauss-Legendre μ-integration. Single-tracer (3×3) and 2-tracer (9×9). |
| `covariance_mt_general.py` | N-tracer generalisation. Handles arbitrary number of tracers, supports non-zero cross-shot noise for partially shared catalogues (f_shared). |

### Fisher matrices

| Module | Purpose |
|--------|---------|
| `fisher.py` | `FisherResult` dataclass with `marginalized_sigma` and `conditional_sigma`. Fisher assembly from derivative matrix + inverse covariance. `add_gaussian_prior`. |
| `fisher_mt.py` | 2-tracer (27-param) multi-tracer Fisher. Includes `multi_tracer_fisher_asymmetric` for split k-ranges (9 obs below kmax_DESI + 6 obs above). |
| `fisher_mt_general.py` | N-tracer Fisher for arbitrary tracer combinations. Parameter vector: 3 cosmo + 12 nuisance per tracer. |
| `prior_export.py` | Regularise F_MT with broad priors → invert → extract σ_cal for each DESI nuisance param. `CalibratedPriors` dataclass. |
| `fisher_full_area.py` | Single-tracer DESI-ELG Fisher over V_full with imported priors. `combine_zbins` sums cosmo blocks across z-bins. |

### Scenarios and output

| Module | Purpose |
|--------|---------|
| `scenarios.py` | Four scenarios: broad, cross-cal, cross-cal-ext, fixed-nuisance. `nuisance_prior_diag` selects the prior source. `SummaryRow` and CSV writer. |
| `plots.py` | Five figures with serif/Computer Modern/≥14pt style. Scenario labels: "Fixed nuis.", "SBP, PS (23%)", "SBP, FL (50%)". |

### Pipeline runners

| Module | Purpose |
|--------|---------|
| `cli.py` | ELG-only pipeline. `run_pipeline` → `PipelineResults` with overlap calibration (asymmetric kmax) + full-area scenarios. Entry point: `python -m pfsfog`. |
| `cli_multitrace.py` | Multi-tracer pipeline. PFS × {DESI-ELG, DESI-LRG, DESI-QSO} with up to 4 tracers and 10 cross-spectra per z-bin. |

## Key conventions

- **EFT parameterisation**: Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757, Table I). RSD model from Chudaykin et al. (2020, arXiv:2004.10607).
- **Counterterms**: `-2k² [c₀ + c₂ f μ² + c₄ f² μ⁴] P_lin`. FoG: `-k⁴ cfog f⁴ μ⁴ Z₁² P_lin`.
- **Stochastic**: `(P_shot + a₀(k/k_nl)² + a₂(k/k_nl)²μ²) / ndens`. P_shot is dimensionless departure from Poisson; base 1/nbar added in covariance.
- **Cross-shot noise**: Zero for different populations (PFS×LRG, PFS×QSO). Non-zero for same-type (PFS×DESI-ELG): `1/(f_shared × n̄_DESI-ELG)`, fiducial f_shared=0.5.
- **Asymmetric kmax**: `kmax_PFS = kmax_DESI × r_σv^{-1/2}`. Default r_σv=0.75 → kmax_PFS≈0.231 h/Mpc.
- **Units**: k in h/Mpc, P(k) in (Mpc/h)³, volumes in (Mpc/h)³.
- **Precision**: `jax.config.update('jax_enable_x64', True)` — float64 throughout.

## Dependencies

`jax>=0.4.26`, `ps_1loop_jax` (editable from `ps_1loop_jax/`), `cosmopower-jax`, `clax` (branch `clax-pt`), `numpy`, `scipy`, `quadax`, `interpax`, `matplotlib`, `pyyaml`.

## What the forecast shows

The cross-calibration tightens two parameters most:
- **P_shot** (shot-noise departure): 10–45% of broad prior — from the zero-stochastic cross-spectrum
- **bG₂σ₈²** (tidal bias): 16–36% of broad prior — from the one-loop μ-structure

The k-dependent stochastic terms (a₀, a₂) and counterterms (c₀, c₂, c₄, c̃) remain prior-dominated — the former are degenerate with the latter at the available k-scales.

Exporting the calibrated priors to DESI's full 14,000 deg² improves σ(fσ₈) by 32–34% and σ(Mν) by 52–55%.
