# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fisher forecast for a multi-tracer PFS×DESI setup that leverages the ~1,200 deg² PFS-DESI ELG overlap volume to data-drive, cross-survey calibrate EFT bias/nuisance parameters. Calibrated priors are exported to DESI's full 14,000 deg² footprint to tighten cosmological constraints (fσ8, Mν, Ωm).

The project has two main components:
1. **`ps_1loop_jax`** — A JAX-based one-loop galaxy power spectrum library (Kobayashi & Akitsu, in prep.; https://github.com/archaeo-pteryx/ps_1loop_jax). Not included in this repo; install separately.
2. **`pfsfog/`** — The Fisher forecast pipeline (implemented). Two-tier backends with automatic fallback:
   - **P_lin / σ8**: cosmopower-jax (default) → clax (fallback)
   - **1-loop P_ℓ(k)**: ps_1loop_jax (default) → clax.ept (fallback)
   - **Background (H, D, f, χ)**: ps_1loop_jax.background (always)
   
   Compares five analysis scenarios (`broad`, `cross-cal`, `cross-cal-ext`, `HOD-prior` benchmark, `oracle`).

## Installing ps_1loop_jax

```bash
cd ps_1loop_jax
python -m pip install -e .
# With test deps:
python -m pip install -e ".[tests]"
```

Dependencies: `jax>=0.4.26`, `jaxlib>=0.4.26`, `numpy`, `scipy`, `sympy>=1.11.0`, `quadax`, `interpax`.

## Running tests

This repo's test suite (run from the repository root):

```bash
pytest tests/ -q              # default suite — 130 tests, ~12 min, slow tests skipped
pytest tests/ -m slow -v      # 2 slow parallel-vs-sequential equivalence tests (~12 min)
pytest tests/test_derivatives.py -v   # one module
ruff check .                  # linting
```

The external `ps_1loop_jax` dependency has its own suite; only run it when debugging the
dependency itself, and only if you have a local checkout of that repo:

```bash
cd ps_1loop_jax && pytest -q   # dependency-only smoke test (optional, requires checkout)
```

## Architecture

### ps_1loop_jax

Core class: `PowerSpectrum1Loop` in `ps_1loop_jax/ps_1loop.py`.
- Computes 1-loop galaxy P(k,μ) via FFTLog decomposition of PT kernels (matrices in `pt_matrix/`).
- Key methods: `get_pk_ell(k, ℓ, pk_data, params)` for auto-power multipoles; `get_pk_ell_pair(...)` for cross-power; `get_pk_ell_ref(...)` for Alcock-Paczynski distorted spectra.
- `params` dict structure: `{'h': ..., 'f': ..., 'bias': {'b1':, 'b2':, 'bG2':, 'bGamma3':}, 'ctr': {'c0':, 'c2':, 'c4':, 'cfog':}, 'stoch': {'P_shot':, 'a0':, 'a2':}, 'k_nl':, 'ndens':}`.
- `clax_adapter.py` bridges `clax` (a JAX Boltzmann solver) to provide linear P(k) input (fallback). Use `make_clax_pk_data()` and `make_clax_background_data()`.
- IR resummation is on by default (`do_irres=True`); uses BAO wiggle/no-wiggle decomposition.
- All core methods are `@jit`-compiled; designed for JAX autodiff (e.g., `jax.jacfwd` for Fisher derivatives).

### pfsfog (Fisher pipeline)

Module structure (see `pfsfog/README.md` for full module guide):
- `cosmo.py` — P_lin via cosmopower-jax (default) or clax (fallback); background via ps_1loop_jax.background
- `surveys.py` — Load n(z) from `survey_specs/*.txt`; volume computation
- `eft_params.py` — EFT fiducials and priors (Chudaykin+ 2025 parameterization)
- `ps1loop_adapter.py` — Map Fisher σ8-scaled params ↔ ps_1loop_jax or clax.ept
- `derivatives.py` — JAX autodiff (primary) + 5-point stencil (validation)
- `fisher.py` / `fisher_mt.py` — Single-tracer and multi-tracer Fisher matrices
- `prior_export.py` — Extract calibrated priors from overlap Fisher
- `fisher_full_area.py` — DESI full-area Fisher with imported priors
- `scenarios.py` — Five analysis scenarios
- `plots.py` — Figures (serif/Computer Modern, ≥14pt)

### Survey specs

`survey_specs/` contains DESI and PFS n(z) tables in fine (Δz=0.01) and FDR binning. Columns: `z_min z_max nz[(h⁻¹Mpc)⁻³] Vz[(h⁻¹Mpc)⁻³]`. Fiducial cosmology: Planck 2018 (h=0.6736, Ωm=0.3153, Ωbh²=0.02237, ns=0.9649, ln(10¹⁰As)=3.044, Mν=0.06 eV).

## Key conventions

- EFT parameterization follows Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757, Table I) exactly.
- PFS-ELG EFT fiducials are scaled from DESI: FoG counterterm c̃ scales with σ_v² ratio (default r_σv=0.75), counterterms scale with b1 ratio.
- Units: k in h/Mpc, P(k) in (Mpc/h)³, distances in Mpc/h or Mpc (see per-module docstrings).
- `jax.config.update('jax_enable_x64', True)` is required — double precision throughout.
- Fiducial cosmology stored in a module-level `FIDUCIAL` dict; no magic numbers in function bodies.
- Timestamped output directories: `results/YYYYMMDD_HHMMSS/`.
- Figure style: serif font, Computer Modern, fontsize ≥14pt, hyphenated labels (PFS-ELG, DESI-ELG).
