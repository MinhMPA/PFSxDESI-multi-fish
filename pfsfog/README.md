# PFSFOG — Pooled Full-shape Surveys: Forecasts Of Galaxies

**Current revision, October 8, 2026:** new one-loop spectra use the local ELF
Eulerian `EPT` calculation. Cosmopower supplies the same total-matter linear
power; `ps_1loop_jax.background` still supplies the retained background and
growth functions. The accepted cosmology, nuisance coordinates, priors and
survey geometry are unchanged. The current continuation record is
[ELF_CHECKPOINT.md](../docs/revision/ELF_CHECKPOINT.md).

ELF is pinned to `/Users/nguyenmn/elf` at
`17d131deff9d03064722b3f57f960aa0a1c1cdba`; the factory verifies both Python
source and coefficient-table bytes. The physical choices are EPT, IR off,
linear counterterms and zero-lag constant subtraction. The initial loop grid
is 512 points from 1e-5 to 100 in h_ref/Mpc. ELF applies AP to the complete
angular mean, including the tied mu^4, mu^6 and mu^8 FoG terms. The former
two-dimensional AP-table sizes are inactive and cannot qualify this engine.

In the `revision/counterterms-leading` comparison, CT-A replaces the leading
power-coefficient average with common tracer fields. The current raw dictionary
still carries `(c0,c2,c4,cfog)`; `leading_field_coefficients(ctr,b1,f)` returns
`(d0,d2/f,d4/f²)` through the regular N system, including at zero growth, and
`pair_counterterms(ctr_a,ctr_b,b_a,b_b,f)` returns ELF's seven coefficients.
The Q and T coordinates and their priors are unchanged. Intrinsic auto
multipoles 0,2,4 match the previous prescription; the induced multipole 6
changes the full angular auto spectrum, AP response and covariance. Cross
spectra contract the two tracer fields, with the existing single-coefficient
FoG contribution retained. Only intrinsic auto spectra retain the declared
Q-multipole zeros. Survey forecasts for this prescription require separate
physical and numerical qualification.

Use the existing isolated `.venv` in `.worktrees/elf-engine`. For a fresh
environment only, from that worktree:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-elf.lock
.venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig
root = Path.cwd().resolve()
sources = ['/Users/nguyenmn/elf/src', '/Users/nguyenmn/cosmopower-jax', str(root/'ps_1loop_jax')]
pth = Path(sysconfig.get_paths()['purelib'])/'pfs_elf_sources.pth'
pth.write_text('import sys; sys.path[:0] = '+repr(sources)+'\n')
PY
```

The lock records the tested Python package versions (JAX0.9.2, SciPy1.17.1).
The local Cosmopower model files remain at
`/Users/nguyenmn/cosmopower-jax/cosmology/jense2024`. Keep the original
Python3.10 recovery environment for authenticating historical results.

```sh
# New spectrum and source checks
.venv/bin/python -m pytest tests/test_elf_adapter.py tests/test_elf_integration.py tests/test_elf_validation.py -q
# Real-emulator derivatives, full cell covariance, loop/angular refinement
# Use a new output directory on every invocation.
.venv/bin/python scripts/validate_elf_engine.py --output results/elf-physical-new
# Inventory only: does not qualify or publish any forecast
.venv/bin/python scripts/run_revision.py --grid --identities
```

The physical command checks two representative overlap cells, including a
separately labelled high-z PFS diagnostic (not the complete high-z study).
It saves signed finite
differences, assembled derivatives, spectra and covariances. A failed check
returns a nonzero exit. Full survey/Fisher convergence and the thirteen
configuration results remain a separate qualification task. Historical
`validate_revision.py` and extended-cosmology qualification commands reject
ELF measurements explicitly; their read-only inventory remains available.

## Historical routines and original architecture

The descriptions and parameterization below concern the older forecast
routines. Their newly constructed spectra also select ELF, but their timing
and old numerical certificates do not apply to the revision. The revision
uses global ln(10^10 A_s), reports derived correlated f sigma8 constraints,
and uses the common accepted kmax rather than the legacy asymmetric cut.

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
| `config.py` | `ForecastConfig` — all tuneable knobs: kmin/kmax/dk, z-bins, areas, r_sigma_v=0.75, f_shared_elg=0.05 (legacy only), marginalize_cross_stoch=True, backend choices. YAML loader. |

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
- **Cross-stochasticity**: Following Ebina & White (2024, arXiv:2401.13166), every cross-pair carries two free stochastic parameters — `Pshot_cross` (constant) and `a2_cross` ((kμ)² term) — with fiducial 0 and broad Gaussian priors of width matching the auto-spectrum stochastic priors. Controlled by `cfg.marginalize_cross_stoch` (default `True`). When `False`, the legacy fixed-fiducial path is restored: cross-shot for PFS×DESI-ELG is `f_shared / nbar_DESI` with `f_shared = n_shared / nbar_PFS = 0.05`, no cross-stoch free parameters.
- **Asymmetric kmax**: `kmax_PFS = kmax_DESI / r_sigma_v`. Default r_sigma_v = 0.75.
- **Units**: k in h/Mpc, P(k) in (Mpc/h)^3, volumes in (Mpc/h)^3.
- **Precision**: `jax.config.update('jax_enable_x64', True)` — float64 throughout.
- **Parallel mode**: `run_joint_fisher` and `run_broad_baseline` accept opt-in `parallel=True`, `n_workers`, `threads_per_worker` kwargs. Workers spawn (macOS-safe with JAX/XLA) and share a JAX persistent compilation cache at `.cache/jax/` (override with `PFSFOG_JAX_CACHE_DIR`). Numerically identical to sequential (rtol=1e-10, verified by slow-marked tests).
- **Test markers**: default `pytest tests/ -q` skips `@pytest.mark.slow`. The two parallel-vs-sequential equivalence tests (~12 min) run via `pytest -m slow -v`.
