#!/usr/bin/env python
"""Compare Fisher results: autodiff cosmo derivs (new) vs finite-diff (old)."""
import sys; sys.path.insert(0, ".")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology, make_plin_func, make_growth_rate_func
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, desi_elg_fiducials, broad_priors
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import (
    dPell_dtheta_autodiff_all, dPell_d_cosmo_all,
    dPell_d_cosmo_autodiff, dPell_d_cosmo_stencil,
)
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.surveys import desi_elg
from pfsfog.cli import run_pipeline

cfg = ForecastConfig.from_yaml("configs/default.yaml")
cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
from ps_1loop_jax import PowerSpectrum1Loop
ps = PowerSpectrum1Loop(do_irres=False)
ells = (0, 2, 4)
kmax = 0.20
k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)
desi_s = desi_elg()

# Old numbers from previous stencil runs
OLD = {
    "broad":     {"fsigma8": 9.3665e-02, "Mnu": 1.3325e+00, "Omegam": 6.4089e-02},
    "cross-cal": {"fsigma8": 7.5864e-02, "Mnu": 5.9614e-01, "Omegam": 5.1524e-02},
    "oracle":    {"fsigma8": 3.8470e-02, "Mnu": 2.0857e-01, "Omegam": 3.3417e-02},
}

# =====================================================================
print("=" * 70)
print("1. DERIVATIVE AGREEMENT (z=0.9)")
print("=" * 70)

z_t = 0.9; s8 = cosmo.sigma8(z_t); f_z = float(cosmo.f(z_t))
h = cosmo.params["h"]
fid = desi_elg_fiducials(1.3, s8)
par = fisher_to_ps1loop_auto(fid, s8, f_z, h, 4e-4)
pkdata_fn = make_plin_func("cosmopower")
f_fn = make_growth_rate_func()
cd = dict(cosmo.params)
k_t = jnp.arange(0.01, 0.21, 0.005)

print(f"\n{'Param':<10s} {'ell':>4s} {'med|frac|':>12s} {'max|frac|':>12s}")
print("-" * 42)
for cparam in ("Mnu", "Omegam"):
    for ell in (0, 2, 4):
        d_ad = np.asarray(dPell_d_cosmo_autodiff(
            ps, k_t, pkdata_fn, f_fn, cd, par, cparam, z_t, s8, ell))
        d_fd = np.asarray(dPell_d_cosmo_stencil(
            ps, k_t, cosmo, par, cparam, z_t, s8, ell))
        scale = np.maximum(np.abs(d_ad), np.abs(d_fd))
        mask = scale > 1e-10 * np.max(scale)
        frac = np.zeros_like(d_ad)
        frac[mask] = np.abs(d_ad[mask] - d_fd[mask]) / scale[mask]
        print(f"{cparam:<10s} {ell:4d} {np.median(frac[mask]):12.2e} {np.max(frac[mask]):12.2e}")

# =====================================================================
print("\n" + "=" * 70)
print("2. SINGLE-ELG PIPELINE: new (autodiff) vs old (stencil)")
print("=" * 70)

res_new = run_pipeline(cfg, verbose=False)

print(f"\n{'Scenario':<12s} {'Param':<10s} {'Old':>12s} {'New':>12s} {'Diff%':>8s}")
print("-" * 55)
for sn in ["broad", "cross-cal", "oracle"]:
    for cp in COSMO_NAMES:
        s_old = OLD[sn][cp]
        s_new = res_new.scenario_results[sn].sigmas_combined[cp]
        diff = (s_new - s_old) / s_old * 100
        print(f"{sn:<12s} {cp:<10s} {s_old:12.4e} {s_new:12.4e} {diff:+7.3f}%")

# =====================================================================
print("\n" + "=" * 70)
print("3. IMPROVEMENT % SHIFT")
print("=" * 70)

print(f"\n{'Param':<10s} {'Old':>8s} {'New':>8s} {'Shift':>8s}")
print("-" * 38)
for cp in COSMO_NAMES:
    imp_old = (OLD["broad"][cp] - OLD["cross-cal"][cp]) / OLD["broad"][cp] * 100
    sb = res_new.scenario_results["broad"].sigmas_combined[cp]
    sx = res_new.scenario_results["cross-cal"].sigmas_combined[cp]
    imp_new = (sb - sx) / sb * 100
    print(f"{cp:<10s} {imp_old:+7.1f}% {imp_new:+7.1f}% {imp_new-imp_old:+7.2f}pp")

print("\nDone.")
