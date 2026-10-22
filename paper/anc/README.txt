Supplementary material for "Multi-tracers, multi-surveys: a joint Fisher analysis
of DESI+PFS" (N.-M. Nguyen). Marginalized cosmology and linear-bias correlations
for cases A-D (Sec. 2.4), from the forecasts at k_max = 0.20 h_ref/Mpc.

per_bin_correlations.pdf
  Eight pages, one per nominal redshift bin. Each page shows the correlations of
  ln(10^10 A_s), M_nu, Omega_m and b1*sigma8 of every active subsample in A-D.
  Upper row: slice of the full-survey covariance. Lower row: separate forecast
  using only that bin's likelihood and nuisance parameters, with each prior
  inserted once.

per_bin_correlations.csv
  Full-precision values. Columns: grid (shell304 = adopted grid, final-loop =
  refined grid), case (A-D), bin (0-7), z_model, context (global or bin_only),
  first, second, covariance, rho.

per_bin_covariances.json
  The corresponding covariance matrices and parameter definitions.

Parameter names: ln10_10_As, Mnu, Omegam; binNN:POPULATION:b1_sigma8 is b1*sigma8
of that population in nominal bin NN. Bins 00-07 are [0.4,0.6], [0.6,0.8],
[0.8,1.0], [1.0,1.2], [1.2,1.4], [1.4,1.6], [1.6,2.0] and [2.0,2.1]. The
subsample labels of Sec. 3.1 number each population by increasing bin:
DESI-LRG bins 00-03 = LRG-1 to LRG-4; DESI-ELG bins 02-05 = ELG-1 to ELG-4;
DESI-QSO bins 02-07 = QSO-1 to QSO-6; PFS-ELG bins 01-05 = PFS-ELG-1 to PFS-ELG-5.
