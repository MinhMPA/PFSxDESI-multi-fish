# Joint multi-tracer Fisher forecast: PFS × DESI

Joint multi-tracer Fisher analysis for full-shape galaxy power spectrum
forecasts. Across the 14,000 deg² DESI footprint plus the 1,200 deg²
PFS overlap at 0.6 < z < 1.6, the framework pools Fisher information
from all auto- and cross-spectra into a single joint Fisher with
shared cosmology and per-(tracer, z-bin) nuisance parameters,
marginalized in one pass.

**Paper**: [Nhat-Minh Nguyen (2026), arXiv:2604.25171, *"Multi-tracers, multi-surveys: a joint Fisher analysis of PFS×DESI"*](https://arxiv.org/abs/2604.25171).

## Key results

- Compared to the **single-tracer analysis with broad priors** (the legacy DESI-DR1 baseline), the DESI+PFS joint Fisher tightens σ(fσ₈) by ~42%, σ(Mν) by ~85%, σ(Ωm) by ~55% at kmax = 0.20 h/Mpc.
- The bulk of this improvement comes from **DESI's own internal multi-tracer self-calibration** (LRG × ELG × QSO across the full footprint). Adding the PFS overlap on top contributes a unique +8% on σ(fσ₈), +22% on σ(Mν), and +9% on σ(Ωm).
- The dominant mechanism is b₁σ₈ calibration via the McDonald–Seljak cosmic-variance-free bias-ratio extraction, which breaks the b₁σ₈–Mν degeneracy intrinsic to single-tracer redshift-space analyses.
- The joint Fisher is not a replacement for simulation-based priors but a model-independent companion — particularly relevant as an independent cross-check on the S8 tension.

## Structure

```
pfsfog/              Fisher forecast pipeline (Python/JAX)
survey_specs/        PFS and DESI n(z) tables
configs/             YAML configuration files
scripts/             Analysis and figure-generation scripts (legacy in scripts/_obsolete/)
tests/               130 unit tests + 2 slow-marked parallel-equivalence tests
notebooks/           Jupyter notebook for interactive figure reproduction
paper/               LaTeX draft, bibliography, and figures
```

## Installation

### 1. Environment

```bash
conda create -n pfsfog python=3.10
conda activate pfsfog
pip install jax[cpu]>=0.4.26    # or jax[cuda12] for GPU
```

### 2. One-loop power spectrum (`ps_1loop_jax`)

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
pytest tests/ -q                          # 130 tests, ~12 min (sequential default)
pytest tests/ -q -m slow -v               # 2 parallel-vs-sequential equivalence tests, ~12 min
```

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
z_bins: [[0.8,1.0], [1.0,1.2], [1.2,1.4], [1.4,1.6]]
r_sigma_v: 0.75      # sigma_v,PFS / sigma_v,DESI
overlap_area_deg2: 1200.0
desi_area_deg2: 14000.0
cosmo_backend: cosmopower   # or "clax" for Boltzmann solver
```

Override from Python:

```python
from pfsfog.config import ForecastConfig
cfg = ForecastConfig(r_sigma_v=0.5, kmax=0.25)
```

### Reproduce paper figures

The notebook `notebooks/figures.ipynb` is the canonical figure source —
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
├── priors/                    # calibrated prior JSON files
└── figures/                   # publication PDFs + PNGs
```

## License

MIT
