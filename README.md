# Multi-tracers, multi-surveys: a joint Fisher analysis of DESI+PFS

Joint multi-tracer Fisher analysis for full-shape galaxy power spectrum
forecasts. Across the 14,000 deg² DESI footprint plus the 1,200 deg²
PFS overlap at 0.6 < z < 1.6, the framework pools Fisher information
from all auto- and cross-spectra into a single joint Fisher with
shared cosmology and per-tracer, per-redshift-bin nuisance parameters,
marginalized in one pass.

This branch (`v3`) holds the revised version of the paper and the code
that produced it.

**Paper**: [Nhat-Minh Nguyen (2026), arXiv:2604.25171, *"Multi-tracers, multi-surveys: a joint Fisher analysis of DESI+PFS"*](https://arxiv.org/abs/2604.25171).

## Paper versions

| Branch | Version | Date |
|---|---|---|
| `v1` | arXiv:2604.25171v1 | 28 April 2026 |
| `v2` | arXiv:2604.25171v2, submitted to JCAP | 5 May 2026 |
| `v3` | Revised version | October 2026 |

The results of v3 supersede those of v1 and v2.

## Key results (v3)

At kmax = 0.20 h/Mpc, varying ln(10¹⁰A_s), Mν and Ωm with h, ω_b and n_s
fixed, and marginalizing over all EFT nuisance parameters:

- Adding DESI cross-spectra to the correlated DESI auto-spectra reduces the
  errors on the derived fσ₈(z=1.1), Mν and Ωm by 7.1%, 3.5% and 3.8%.
- Adding PFS auto- and cross-spectra over 1,200 deg² at 0.6 < z < 1.6
  reduces them by a further 3.8%, 2.6% and 3.2%.
- Extending PFS to z = 2.4 raises these PFS reductions to 6.7%, 6.7% and 7.4%.
- The multi-tracer constraints come from the clustering likelihood alone and
  complement simulation-based priors.

## Structure

```
pfsfog/              Fisher forecast pipeline (Python/JAX)
survey_specs/        PFS and DESI n(z) tables
configs/             YAML configuration files
scripts/             Analysis and figure-generation scripts (legacy in scripts/_obsolete/)
tests/               137 unit tests + 2 slow-marked parallel-equivalence tests
notebooks/           Figure notebooks; v3 figures: figures_author_revision.ipynb
paper/               LaTeX source, bibliography and figures (v3: figs/author-revision/)
paper/anc/           Supplementary per-bin correlation matrices (PDF, CSV, JSON)
scripts/provenance/  Drivers used for the v3 supplementary forecasts (see below)
```

## Installation

### 1. Environment

```bash
conda create -n pfsfog python=3.10
conda activate pfsfog
pip install jax[cpu]>=0.4.26    # or jax[cuda12] for GPU
```

### 2. One-loop power spectrum

v3 computes the one-loop spectra with [ELF](https://github.com/archaeo-pteryx/elf)
(Kobayashi and Akitsu). `pfsfog/elf_source.json` records the SHA-256 of every
ELF source file used for the paper, and `requirements-elf.lock` pins the
Python environment (Python ≥ 3.11):

```bash
git clone https://github.com/archaeo-pteryx/elf.git
pip install -e elf
pip install -r requirements-elf.lock
```

The code also imports `ps_1loop_jax` for background quantities (growth and
distances) and the v1/v2 spectra. Some v3 scripts expect a clone at
`./ps_1loop_jax`:

`ps_1loop_jax` (Kobayashi & Akitsu, in prep.) is required but not included in this repo.
It is not yet publicly available; once released, install it from
https://github.com/archaeo-pteryx/ps_1loop_jax:

```bash
git clone https://github.com/archaeo-pteryx/ps_1loop_jax.git
cd ps_1loop_jax && pip install -e . && cd ..
```

### 3. cosmopower-jax emulator

The pipeline uses [cosmopower-jax](https://github.com/dpiras/cosmopower-jax)
with the [Jense et al. (2024) nuLCDM trained networks](https://github.com/cosmopower-organization/jense_2024_emulators).

```bash
pip install cosmopower-jax
git clone https://github.com/cosmopower-organization/jense_2024_emulators.git
```

Then update the network path in `pfsfog/cosmo.py` to point to your local copy:

```python
_JENSE_DIR = Path("/path/to/jense_2024_emulators/jense_2023_camb_mnu")
```

The emulator expects two `.npz` files under `networks/`:
- `jense_2023_camb_mnu_Pk_lin.npz` (linear P(k) emulator)
- `jense_2023_camb_mnu_sigma8.npz` (sigma8 emulator)

### 4. This package

```bash
pip install -e .     # installs pfsfog + remaining deps (numpy, scipy, matplotlib, pyyaml, numdifftools)
```

### 5. Verify

```bash
pytest tests/ -q
```

Many validation tests compare against the author's saved forecast outputs and
intermediate results, which are not distributed. In a fresh clone these tests
fail with missing-file errors; the remaining unit tests run without them.

## Usage

### Run the forecast

```bash
# Joint Fisher: DESI-only vs DESI+PFS (~14 min sequential, ~7 min parallel with warm cache)
python scripts/run_joint_fisher.py
python scripts/run_joint_fisher.py --parallel --n-workers 8   # multi-process opt-in

# Parameter importance decomposition (legacy two-stage pipeline)
python scripts/run_parameter_importance.py

# Sensitivity sweep (r_sigma_v) — legacy diagnostic
python scripts/run_sensitivity.py
```

### Parallel mode

`run_joint_fisher` (and `run_broad_baseline`) accepts opt-in
`parallel=True`, `n_workers`, and `threads_per_worker` kwargs. The parallel
path dispatches the per-z-bin loop across a `multiprocessing` spawn pool,
using a JAX persistent compilation cache (`.cache/jax/`, set via
`PFSFOG_JAX_CACHE_DIR`) to amortize JIT-compile cost across worker
processes. After the first run populates the cache, subsequent runs see a
~1.9× wall-clock speedup on a typical 8–16 core machine.

The parallel path is numerically identical to sequential (rtol = 1e-10,
verified by `tests/test_fisher_joint.py::test_run_*_parallel_matches_sequential`).

### Configuration

All settings are in `configs/default.yaml`:

```yaml
kmin: 0.01           # h/Mpc
kmax: 0.20           # h/Mpc (full-area analysis)
dk: 0.005            # bin width
z_bins: [[0.8,1.0], [1.0,1.2], [1.2,1.4], [1.4,1.6]]   # legacy overlap pipeline only
r_sigma_v: 0.75      # sigma_v,PFS / sigma_v,DESI
overlap_area_deg2: 1200.0
desi_area_deg2: 14000.0
cosmo_backend: cosmopower   # or "clax" for Boltzmann solver
```

The 8-bin schedule covering 0.4 ≤ z ≤ 2.1 used by the joint-Fisher
driver is defined in `scripts/run_joint_fisher.py::ZBINS`, not in the
YAML.

Override from Python:

```python
from pfsfog.config import ForecastConfig
cfg = ForecastConfig(r_sigma_v=0.5, kmax=0.25)
```

### Reproduce paper figures

For v3, `notebooks/figures_author_revision.ipynb` regenerates Figures 1–9
from the exported forecast results in `notebooks/data/author_revision/`
(see its `README.md` and `manifest.json`), without rerunning the forecasts.

The v3 forecasts were computed with the ELF-based drivers in `scripts/`
(`run_counterterm_comparison.py` for the main comparison) and the drivers in
`scripts/provenance/` for the supplementary profiles and the density and bias
scans. The provenance drivers read the author's saved intermediate results,
which are not distributed, and are kept to document the calculation.
Some validation tests read the same local outputs and fail without them.

For v1/v2, the notebook `notebooks/figures.ipynb` is the canonical figure source —
each figure has a separate **compute** cell and **plot** cell, so plot
tweaks don't trigger expensive recomputation. The compute cells default
to `PARALLEL = True` for joint-Fisher figures (Fig. 1, 2, 5).

```bash
jupyter notebook notebooks/figures.ipynb
```

Standalone scripts for individual figures:

```bash
python scripts/fig_fisher_contours.py     # Fig. 5: 3-scenario contour plot
python scripts/fig_deriv_validation.py    # Fig. A1
python scripts/fig_ms09_convergence.py    # Fig. A2
python scripts/fig_fisher_info_density.py # Fig. A3
python scripts/fig_cov_validation.py      # Fig. A4
```

### Output

Each run creates a timestamped directory:

```
results/YYYYMMDD_HHMMSS/
├── summary.csv               # sigma per scenario x parameter
├── priors/                    # tightened nuisance-prior JSON files (legacy two-stage CLI only)
└── figures/                   # publication PDFs + PNGs
```

`run_joint_fisher.py` writes `summary.csv` and figure outputs only;
the `priors/` subdirectory is produced by the legacy two-stage
pipeline (`pfsfog.cli`/`cli_multitrace`).

## License

MIT
