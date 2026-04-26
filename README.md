# Multi-survey EFT prior calibration: PFS x DESI Fisher forecast

Fisher forecast for **multi-survey priors** — data-driven calibration of EFT nuisance parameters from the multi-tracer analysis of the PFS--DESI overlap volume, exported to DESI's full 14,000 deg² footprint.

**Paper**: N.-M. Nguyen, *"Multi-tracers, multi-surveys: data-driven EFT prior calibration from the PFS--DESI overlap"*.

## Key results

- The ~1,200 deg² PFS--DESI overlap at 0.6 < z < 1.6 calibrates EFT parameters for all DESI tracers (ELG, LRG, QSO) using up to 4 tracers and 10 cross-spectra per z-bin.
- Forecasting all 6 DESI DR2 samples (LRG1--3, ELG1--2, QSO), multi-survey priors improve sigma(f*sigma8) by 8% (7--25% per sample) and sigma(M_nu) by 53% (46--69% per sample).
- The dominant driver is b1*sigma8 calibration, which breaks the b1--f degeneracy (~70% of the f*sigma8 gain, ~97% of the M_nu gain).
- Multi-survey priors are not a replacement for simulation-based priors but a model-independent cross-check -- particularly relevant for the S8 tension.

## Structure

```
pfsfog/              Fisher forecast pipeline (Python/JAX)
survey_specs/        PFS and DESI n(z) tables
configs/             YAML configuration files
scripts/             Analysis and figure-generation scripts
tests/               96 unit tests
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
pytest tests/ -q     # 120 tests, ~10s
```

## Usage

### Run the forecast

```bash
# Single-ELG pipeline with default config (~30s)
python -m pfsfog

# 6-sample DESI DR2 forecast (~5min)
python scripts/run_desi_multisample.py

# Parameter importance decomposition
python scripts/run_parameter_importance.py

# Sensitivity sweep (r_sigma_v)
python scripts/run_sensitivity.py
```

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

```bash
# All main-text figures
python scripts/make_all_figures.py

# Appendix figures
python scripts/fig_deriv_validation.py
python scripts/fig_ms09_convergence.py
python scripts/fig_fisher_info_density.py
python scripts/fig_fisher_contours.py
```

Or interactively: `jupyter notebook notebooks/figures.ipynb`

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
