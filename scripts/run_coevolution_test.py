#!/usr/bin/env python
"""Test: imposing co-evolution relations b2(b1) and bG2(b1) as constraints.

Reduces the nuisance space by 2 parameters per tracer. Compare sigma(fsigma8)
and sigma(Mnu) with and without the co-evolution constraint.

This implements the test discussed in Mergulhão et al. (2022, Appendix B).
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import desi_elg
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, desi_elg_fiducials, broad_priors,
)
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto, make_ps1loop_pkmu_func,
)
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.scenarios import SCENARIOS, nuisance_prior_diag

from ps_1loop_jax import PowerSpectrum1Loop

cfg = ForecastConfig.from_yaml("configs/default.yaml")
cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
ps = PowerSpectrum1Loop(do_irres=False)
desi = desi_elg()
ells = (0, 2, 4)

def run_full_area(fix_coevol=False, scenario_name="cross-cal-ext"):
    """Run full-area DESI Fisher with optional co-evolution constraints."""
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    k_full = np.arange(cfg.kmin, scenario.kmax + cfg.dk / 2, cfg.dk)

    bp = broad_priors()
    bp_diag = bp.prior_fisher_diag()

    if fix_coevol:
        # Fix b2_sigma8sq and bG2_sigma8sq by setting their prior to delta function
        # (sigma -> 0, i.e. 1/sigma^2 -> infinity)
        idx_b2 = NUISANCE_NAMES.index("b2_sigma8sq")
        idx_bG2 = NUISANCE_NAMES.index("bG2_sigma8sq")
        bp_diag[idx_b2] = 1.0 / 1e-20  # effectively fix b2
        bp_diag[idx_bG2] = 1.0 / 1e-20  # effectively fix bG2

    fishers = []
    for zlo, zhi in cfg.z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff)
        f_z = float(cosmo.f(z_eff))
        h = cosmo.params["h"]
        nbar = desi.nbar_eff(zlo, zhi)
        if nbar == 0:
            continue
        b1 = desi.b1_of_z(z_eff)
        V_full = desi.volume(zlo, zhi)
        fid = desi_elg_fiducials(b1, s8)
        params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
        pk_data = cosmo.pk_data(z_eff)

        import jax.numpy as jnp
        derivs = dPell_dtheta_autodiff_all(
            ps, jnp.array(k_full), pk_data, params,
            NUISANCE_NAMES, s8, ells)
        cosmo_derivs = dPell_d_cosmo_all(
            ps, jnp.array(k_full), pk_data, cosmo, params,
            z_eff, s8, ells)
        derivs.update(cosmo_derivs)

        pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params)
        cov_st = single_tracer_cov(pkmu_func, k_full, nbar, V_full, cfg.dk, ells)

        # Use the scenario's calibrated priors for non-bias params,
        # but override b2/bG2 if fix_coevol
        nuis_prior = nuisance_prior_diag(scenario, calibrated=None)
        if fix_coevol:
            nuis_prior[idx_b2] = 1.0 / 1e-20
            nuis_prior[idx_bG2] = 1.0 / 1e-20

        fr = full_area_fisher_per_zbin(
            derivs, cov_st, k_full, cfg.dk, nuis_prior,
            (zlo, zhi), scenario.kmax, ells)
        fishers.append(fr)

    if not fishers:
        return {}
    fr_combined = combine_zbins(fishers, cfg.z_bins)
    return {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}


print("=== Co-evolution prior test ===")
print()

for scenario_name in ["broad", "oracle"]:
    print(f"Scenario: {scenario_name}")
    sig_free = run_full_area(fix_coevol=False, scenario_name=scenario_name)
    sig_fixed = run_full_area(fix_coevol=True, scenario_name=scenario_name)

    print(f"  {'Parameter':<12s}  {'Free b2,bG2':>12s}  {'Fixed b2,bG2':>12s}  {'Change':>8s}")
    print(f"  {'-'*50}")
    for cp in COSMO_NAMES:
        s_free = sig_free.get(cp, float("nan"))
        s_fixed = sig_fixed.get(cp, float("nan"))
        change = (s_fixed - s_free) / s_free * 100
        print(f"  {cp:<12s}  {s_free:12.4e}  {s_fixed:12.4e}  {change:+7.1f}%")
    print()
