#!/usr/bin/env python
"""Parameter importance decomposition: which calibrated nuisance parameters
drive the cosmological improvement?

Runs the full-area DESI Fisher with the calibrated prior applied to
different subsets of nuisance parameters, keeping the rest at broad.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import desi_elg
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, desi_elg_fiducials,
    broad_priors, COSMO_PRIOR_SIGMA,
)
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.prior_export import CalibratedPriors, calibrated_prior_fisher_diag
from pfsfog.cli import run_pipeline


def hybrid_prior_diag(calibrated: CalibratedPriors, params_to_calibrate: list[str]):
    """Build a prior diagonal that uses calibrated values for `params_to_calibrate`
    and broad values for everything else."""
    broad_diag = broad_priors().prior_fisher_diag()
    cal_diag = calibrated_prior_fisher_diag(calibrated)

    hybrid = broad_diag.copy()
    for i, name in enumerate(NUISANCE_NAMES):
        if name in params_to_calibrate:
            hybrid[i] = cal_diag[i]
    return hybrid


# Parameter groups to test
# NOTE: b1_sigma8 has FLAT broad prior (Fisher = 0), so calibrating it
# switches from no prior info to sigma_cal ~ 0.17.  This is the biggest
# relative change and matters most for Mnu (corr = -0.994 with b1sigma8).
GROUPS = {
    "None (broad)":           [],
    "b1s8 only":              ["b1_sigma8"],
    "Pshot only":             ["Pshot"],
    "bG2 only":               ["bG2_sigma8sq"],
    "b1s8 + Pshot":           ["b1_sigma8", "Pshot"],
    "b1s8 + Pshot + bG2":     ["b1_sigma8", "Pshot", "bG2_sigma8sq"],
    "All biases":             ["b1_sigma8", "b2_sigma8sq", "bG2_sigma8sq", "bGamma3"],
    "Stochastic":             ["Pshot", "a0", "a2"],
    "Counterterms":           ["c0", "c2", "c4", "c_tilde", "c1"],
    "Biases + stochastic":    ["b1_sigma8", "b2_sigma8sq", "bG2_sigma8sq", "bGamma3",
                               "Pshot", "a0", "a2"],
    "All except b1s8":        [n for n in NUISANCE_NAMES if n != "b1_sigma8"],
    "All (= cross-cal)":      list(NUISANCE_NAMES),
}


def main():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")

    # Run the full pipeline once to get calibrated priors
    print("Running pipeline to get calibrated priors...")
    results = run_pipeline(cfg, verbose=False)
    cal_elg = results.overlap_results  # {(zlo,zhi): OverlapResult}

    # Setup
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)
    ells = (0, 2, 4)
    z_bins = cfg.z_bins
    kmax = 0.20
    k_full = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)
    desi = desi_elg()

    print(f"\n{'Group':<28s}  {'σ(fσ8)':>10s}  {'Δ%':>6s}  "
          f"{'σ(Mν)':>10s}  {'Δ%':>6s}  {'σ(Ωm)':>10s}  {'Δ%':>6s}")
    print("-" * 90)

    broad_sigmas = {}

    for group_name, params_to_cal in GROUPS.items():
        fishers_per_z = []

        for zlo, zhi in z_bins:
            z_eff = 0.5 * (zlo + zhi)
            s8 = cosmo.sigma8(z_eff)
            f_z = float(cosmo.f(z_eff))
            h = cosmo.params["h"]
            nbar_desi = desi.nbar_eff(zlo, zhi)
            if nbar_desi == 0:
                continue
            b1_desi = desi.b1_of_z(z_eff)
            V_full = desi.volume(zlo, zhi)
            pk_data = cosmo.pk_data(z_eff)

            fid_desi = desi_elg_fiducials(b1_desi, s8)
            params_desi = fisher_to_ps1loop_auto(fid_desi, s8, f_z, h, nbar_desi)

            derivs = dPell_dtheta_autodiff_all(
                ps, jnp.array(k_full), pk_data, params_desi,
                NUISANCE_NAMES, s8, ells)
            cosmo_derivs = dPell_d_cosmo_all(
                ps, jnp.array(k_full), pk_data, cosmo, params_desi,
                z_eff, s8, ells)
            derivs.update(cosmo_derivs)

            pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params_desi)
            cov_st = single_tracer_cov(pkmu_func, k_full, nbar_desi, V_full, cfg.dk, ells)

            # Get calibrated priors for this z-bin
            ov = cal_elg.get((zlo, zhi))
            if ov is not None and params_to_cal:
                cal = ov.calibrated_priors
                nuis_prior = hybrid_prior_diag(cal, params_to_cal)
            else:
                nuis_prior = broad_priors().prior_fisher_diag()

            fr = full_area_fisher_per_zbin(
                derivs, cov_st, k_full, cfg.dk, nuis_prior,
                (zlo, zhi), kmax, ells)
            fishers_per_z.append(fr)

        if not fishers_per_z:
            continue

        fr_combined = combine_zbins(fishers_per_z, z_bins)
        sigmas = {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}

        if group_name == "None (broad)":
            broad_sigmas = sigmas.copy()

        row = f"{group_name:<28s}"
        for cp in COSMO_NAMES:
            s = sigmas[cp]
            sb = broad_sigmas.get(cp, s)
            imp = (sb - s) / sb * 100 if sb > 0 else 0
            row += f"  {s:10.4e}  {imp:+5.1f}%"
        print(row)


if __name__ == "__main__":
    main()
