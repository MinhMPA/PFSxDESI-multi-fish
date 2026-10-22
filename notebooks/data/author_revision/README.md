# Plotting data for the revised paper (v3)

Open [`../../figures_author_revision.ipynb`](../../figures_author_revision.ipynb)
and run all cells. Each figure has separate data and plotting cells; fonts,
colours and export settings are collected near the top. The notebook exports
PDF, PNG and SVG files to `paper/figs/author-revision/`, the directory used by
`paper/PFSxDESI-multi-fish.tex`. Set `EXPORT = False` for previews.

The nine figures use the saved forecast results in this directory, so no Fisher
forecast, spectrum or emulator calculation is needed. `manifest.json` gives the
SHA-256 checksum of every input, which the notebook checks before plotting.
Figure 8 is analytic.

| Figure | Input | Selection / physical quantity |
|---|---|---|
| 1 | `nuisance_constraints.csv` | Case D; tracer-nuisance errors, with finite-prior ratios where defined |
| 2 | `cosmological_constraints.csv` | Cases A–D; absolute cosmological errors |
| 3 | `nuisance_marginalization.csv` | Marginalized / nuisance-fixed error in each case |
| 4 | `pfs_gains.csv` | Matched B/D variations and incremental PFS gain |
| 5 | `selected_covariances.json` | Cases A, B, D; full marginalized joint covariances |
| 6 | `fig6_data.npz` | ELG-1 at z=0.9; 14 derivative responses and three finite-difference steps |
| 7 | `fig7_data.npz` | ELG-1/PFS-ELG-2 covariance in the overlap at z=0.9 |
| 8 | `fig8_data.npz` | Analytic Kaiser matter-amplitude limit with fixed f and free biases |
| 9 | `fig9_data.npz` | Accumulated ELG-1 inverse variances at z=0.9 over the full DESI area |

`appendix_verification.json` records the numerical checks of Appendix A.
The results use the adopted `shell304` grid; growth is reported at z=1.1 in
Figures 2–5 and at z=0.9 in Figure 9.

Cases: A, correlated DESI auto-spectra; B, A plus DESI cross-spectra; C, B plus
PFS auto-spectra; D, C plus DESI x PFS cross-spectra. Subsamples are numbered by
increasing nominal redshift within each population (LRG-1–4, ELG-1–4, QSO-1–6,
PFS-ELG-1–5; the high-redshift extension adds PFS-ELG-6–8). Saved files keep
their internal numerical identifiers; `case_labels.json` maps earlier case labels.

The plotting cells require Python ≥ 3.9 with NumPy, Matplotlib and IPython.
