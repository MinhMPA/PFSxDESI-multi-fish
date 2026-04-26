# Scripts

All scripts should be run from the repo root: `python scripts/<name>.py`

## Pipeline runs

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_all_scenarios.py` | Single-ELG pipeline + all figures | ~1 min |
| `run_desi_multisample.py` | 6-sample DESI DR2 forecast (LRG1--3, ELG1--2, QSO) | ~5 min |
| `run_pair_comparison.py` | Per-tracer-pair decomposition (PFS x ELG vs LRG vs QSO) | ~5 min |
| `run_parameter_importance.py` | Which nuisance params drive the improvement | ~3 min |
| `run_sensitivity.py` | r_sigma_v sweep (asymmetric vs symmetric kmax) | ~5 min |
| `run_kmax_nbar_sweeps.py` | kmax and nbar sensitivity sweeps | ~10 min |
| `run_fshared_sweep.py` | Shared-catalog fraction sensitivity | ~5 min |
| `run_coevolution_test.py` | Co-evolution relation test (Appendix B) | ~1 min |

## Figure generation

| Script | Output | Paper figure |
|--------|--------|-------------|
| `make_all_figures.py` | All main-text figures | Fig. 1--4 |
| `fig_deriv_validation.py` | `deriv_autodiff_vs_stencil.{pdf,png}` | Fig. A1 |
| `fig_ms09_convergence.py` | `ms09_convergence.{pdf,png}` | Fig. A2 |
| `fig_fisher_info_density.py` | `fisher_info_density.{pdf,png}` | Fig. A3 |
| `fig_fisher_contours.py` | `fisher_contours.{pdf,png}` | Fig. 5 |
| `fig_cov_validation.py` | `cov_gl_vs_wigner3j.png` | Fig. A2 |

