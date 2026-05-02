# Scripts

All scripts should be run from the repo root: `python scripts/<name>.py`

## Headline driver

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_joint_fisher.py` | **Joint multi-tracer Fisher: DESI-only vs DESI+PFS.** Reports σ(fσ8), σ(Mν), σ(Ωm) and the PFS-unique relative tightening. Supports `--parallel` / `--n-workers`. | ~14 min sequential, ~7 min parallel (warm cache) |
| `_bench_joint_fisher.py` | Internal — BLAS scaling diagnostic + worker/threads sweep for parallel mode tuning. | ~10–20 min |

## Legacy two-stage pipeline (kept for reproducibility)

These scripts implement the original two-stage architecture (Step 1 overlap calibration → Step 2 single-tracer cosmology with calibrated priors). The joint Fisher above replaces them; we keep these for reproducibility of the older results.

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_all_scenarios.py` | Legacy single-ELG pipeline + figures | ~1 min |
| `run_desi_multisample.py` | Legacy 6-sample DESI DR2 forecast | ~5 min |
| `run_pair_comparison.py` | Per-tracer-pair decomposition | ~5 min |
| `run_parameter_importance.py` | Nuisance-parameter importance | ~3 min |
| `run_sensitivity.py` | r_sigma_v sensitivity | ~5 min |
| `run_kmax_nbar_sweeps.py` | kmax / nbar sensitivity sweeps | ~10 min |
| `run_fshared_sweep.py` | Shared-catalog fraction sensitivity | ~5 min |
| `run_coevolution_test.py` | Co-evolution relation test | ~1 min |
| `run_fshared_joint.py` | f_shared sweep on the joint Fisher | ~10 min |

## Figure generation

| Script | Output | Paper figure |
|--------|--------|-------------|
| `fig_fisher_contours.py` | `fisher_contours.png` (3-scenario) | Fig. 5 |
| `fig_deriv_validation.py` | `deriv_autodiff_vs_stencil.{pdf,png}` | Fig. A1 |
| `fig_ms09_convergence.py` | `ms09_convergence.{pdf,png}` | Fig. A2 |
| `fig_fisher_info_density.py` | `fisher_info_density.{pdf,png}` | Fig. A3 |
| `fig_cov_validation.py` | `cov_gl_vs_wigner3j.png` | Fig. A4 |

For Figs. 1, 2, 3 (per-z-bin nuisance, headline cosmology bars, calibration efficiency), use the notebook `notebooks/figures.ipynb`. Each figure has a separate compute and plot cell, so plot tweaks don't trigger expensive recomputation.

## Obsolete scripts

`scripts/_obsolete/` contains diagnostic scripts from the development cycle that produced the joint Fisher reframing — kept for reference but not part of any pipeline. They are not maintained.
