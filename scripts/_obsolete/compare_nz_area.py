#!/usr/bin/env python
"""Compare PFS n(z) baseline vs hypothetical, at 1200 vs 1400 deg² overlap.

Runs Step 1 calibration + Step 2 6-sample DESI Fisher for each (n(z), area)
config and reports σ(fσ8), σ(Mν), σ(Ωm) under broad and cross-cal scenarios.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog import surveys as _surv
from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import Survey, load_nz_table

from scripts.run_desi_multisample import (
    DR2_SAMPLES, SURVEY_CONSTRUCTORS,
    run_overlap_step1, average_calibrated_prior_diag,
)
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, broad_priors, tracer_fiducials,
)
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto, make_ps1loop_pkmu_func,
)
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.scenarios import compute_improvement


def make_pfs_survey(nz_file: str) -> Survey:
    z_min, z_max, nz, Vz = load_nz_table(Path(__file__).resolve().parent.parent
                                         / "survey_specs" / nz_file)
    return Survey(
        name="PFS-ELG",
        area_deg2=1200.0,
        z_min_all=z_min, z_max_all=z_max, nz_all=nz, Vz_all=Vz,
        b1_of_z=lambda z: 0.9 + 0.4 * z,
    )


def run_step2_full_area(cfg, cosmo, ps, cal_per_tracer, kmax=0.20):
    """Step 2 only: 6-sample DESI Fisher with cross-cal priors."""
    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)

    scenarios = {
        "broad":     lambda s: broad_priors().prior_fisher_diag(),
        "cross-cal": lambda s: average_calibrated_prior_diag(
            cal_per_tracer.get(s.tracer, {}), s.overlap_zbins),
    }
    results = {}
    for name, prior_fn in scenarios.items():
        fishers, labels, zranges = [], [], []
        for sample in DR2_SAMPLES:
            survey = SURVEY_CONSTRUCTORS[sample.tracer]()
            zlo, zhi = sample.z_range
            z_eff = 0.5 * (zlo + zhi)
            nbar = survey.nbar_eff(zlo, zhi)
            if nbar == 0:
                continue
            b1 = survey.b1_of_z(z_eff)
            V = survey.volume(zlo, zhi)
            s8 = cosmo.sigma8(z_eff)
            f_z = float(cosmo.f(z_eff))
            h = cosmo.params["h"]

            fid = tracer_fiducials(sample.tracer, b1, s8)
            params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
            pk_data = cosmo.pk_data(z_eff)

            derivs = dPell_dtheta_autodiff_all(
                ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
            cosmo_derivs = dPell_d_cosmo_all(
                ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells)
            derivs.update(cosmo_derivs)

            pkmu = make_ps1loop_pkmu_func(ps, pk_data, params)
            cov = single_tracer_cov(pkmu, k, nbar, V, cfg.dk, ells)
            nuis_prior = prior_fn(sample)

            fr = full_area_fisher_per_zbin(
                derivs, cov, k, cfg.dk, nuis_prior,
                sample.z_range, kmax, ells,
                survey_name=f"DESI-{sample.name}",
            )
            fishers.append(fr)
            labels.append(sample.name)
            zranges.append(sample.z_range)

        fr_combined = combine_zbins(fishers, zranges, sample_labels=labels,
                                    survey_name=f"DESI 6-sample {name}")
        results[name] = {cp: fr_combined.marginalized_sigma(cp)
                         for cp in COSMO_NAMES}
    return results


def run_one_config(nz_file: str, area: float, cosmo, ps):
    print(f"\n{'='*70}")
    print(f"  Config: nz={nz_file}, overlap_area={area} deg²")
    print(f"{'='*70}")

    pfs_survey = make_pfs_survey(nz_file)

    # run_overlap_step1 imports pfs_elg by name — patch in its module namespace
    import scripts.run_desi_multisample as _rdm
    _surv.pfs_elg = lambda: pfs_survey
    _rdm.pfs_elg = lambda: pfs_survey

    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = area
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]

    cal = run_overlap_step1(cfg, cosmo, ps, verbose=False)
    res = run_step2_full_area(cfg, cosmo, ps, cal)
    return res


def main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    configs = [
        ("baseline",     "PFS_nz_pfs_baseline_fine.txt",      1200.0),
        ("baseline",     "PFS_nz_pfs_baseline_fine.txt",      1400.0),
        ("hypothetical", "PFS_nz_pfs_hypothetical_fine.txt",  1200.0),
        ("hypothetical", "PFS_nz_pfs_hypothetical_fine.txt",  1400.0),
    ]

    all_results = []
    for tag, nz_file, area in configs:
        res = run_one_config(nz_file, area, cosmo, ps)
        all_results.append((tag, area, res))

    # --- Summary table ---
    print(f"\n\n{'='*100}")
    print("SUMMARY (Option B: PFS truncated at z=1.6)")
    print(f"{'='*100}")
    header = (f"{'Scenario':<14s}{'Area':>7s}{'Prior':>11s}"
              f"{'σ(fσ8)':>12s}{'Δ%':>7s}"
              f"{'σ(Mν) eV':>12s}{'Δ%':>7s}"
              f"{'σ(Ωm)':>12s}{'Δ%':>7s}")
    print(header)
    print("-" * 100)

    for tag, area, res in all_results:
        broad = res["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"{tag:<14s}{int(area):>7d}{prior:>11s}"
            for cp in COSMO_NAMES:
                s = res[prior][cp]
                imp = compute_improvement(s, broad[cp])
                row += f"  {s:10.4e} {imp:+5.1f}%"
            print(row)
        print("-" * 100)


if __name__ == "__main__":
    main()
