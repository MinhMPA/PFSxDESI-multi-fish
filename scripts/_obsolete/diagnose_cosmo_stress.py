#!/usr/bin/env python
"""Run full Step 1+2 with extreme PFS n(z) configs to verify cosmological
σ(fσ8), σ(Mν) move sensibly when n̄ changes drastically.
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
from pfsfog.surveys import Survey

import scripts.run_desi_multisample as _rdm
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


def make_pfs_with_nbar(zedges_nbar):
    data = np.loadtxt("survey_specs/PFS_nz_pfs_fine.txt", comments="#")
    zmin, zmax, _, Vz = data.T
    mask = zmax <= 1.6 + 1e-9
    zmin_t, zmax_t, Vz_t = zmin[mask], zmax[mask], Vz[mask]
    zmid_t = 0.5 * (zmin_t + zmax_t)
    nz = np.zeros_like(zmid_t)
    for zlo, zhi, n in zedges_nbar:
        m = (zmid_t >= zlo) & (zmid_t < zhi)
        nz[m] = n
    return Survey(name="PFS-ELG", area_deg2=1200.0,
                  z_min_all=zmin_t, z_max_all=zmax_t,
                  nz_all=nz, Vz_all=Vz_t,
                  b1_of_z=lambda z: 0.9 + 0.4 * z)


def step2_cosmo(cfg, cosmo, ps, cal_per_tracer):
    ells = (0, 2, 4)
    kmax = 0.20
    k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)

    scenarios = {
        "broad":     lambda s: broad_priors().prior_fisher_diag(),
        "cross-cal": lambda s: average_calibrated_prior_diag(
            cal_per_tracer.get(s.tracer, {}), s.overlap_zbins),
    }
    out = {}
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
            s8 = cosmo.sigma8(z_eff); f_z = float(cosmo.f(z_eff)); h = cosmo.params["h"]
            fid = tracer_fiducials(sample.tracer, b1, s8)
            params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
            pk_data = cosmo.pk_data(z_eff)
            derivs = dPell_dtheta_autodiff_all(ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
            cosmo_derivs = dPell_d_cosmo_all(ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells)
            derivs.update(cosmo_derivs)
            pkmu = make_ps1loop_pkmu_func(ps, pk_data, params)
            cov = single_tracer_cov(pkmu, k, nbar, V, cfg.dk, ells)
            fr = full_area_fisher_per_zbin(derivs, cov, k, cfg.dk, prior_fn(sample),
                                           sample.z_range, kmax, ells, survey_name=f"DESI-{sample.name}")
            fishers.append(fr); labels.append(sample.name); zranges.append(sample.z_range)
        fr_combined = combine_zbins(fishers, zranges, sample_labels=labels,
                                    survey_name=f"DESI 6-sample {name}")
        out[name] = {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}
    return out


def main():
    cosmo = FiducialCosmology(backend="cosmopower")
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    configs = [
        ("STARVED 0.1× (0.3,0.3,0.4,0.4,0.4)",
         [(0.6,1.0,0.3e-4),(1.0,1.6,0.4e-4)]),
        ("BASELINE Takada (3,3,4,4,4)",
         [(0.6,1.0,3e-4),(1.0,1.6,4e-4)]),
        ("HYPOTHETICAL (3,6,8,8,6)",
         [(0.6,0.8,3e-4),(0.8,1.0,6e-4),(1.0,1.2,8e-4),(1.2,1.4,8e-4),(1.4,1.6,6e-4)]),
        ("EXTREME 20× (3,60,80,80,60)",
         [(0.6,0.8,3e-4),(0.8,1.0,60e-4),(1.0,1.2,80e-4),(1.2,1.4,80e-4),(1.4,1.6,60e-4)]),
        ("ULTRA 100× (3,300,400,400,300)",
         [(0.6,0.8,3e-4),(0.8,1.0,300e-4),(1.0,1.2,400e-4),(1.2,1.4,400e-4),(1.4,1.6,300e-4)]),
    ]

    print(f"\n{'='*100}")
    print(f"{'Config':<40s}  {'Prior':>10s}  {'σ(fσ8)':>10s}{'Δ%':>7s}  "
          f"{'σ(Mν)':>10s}{'Δ%':>7s}  {'σ(Ωm)':>10s}{'Δ%':>7s}")
    print('='*100)

    for label, edges in configs:
        pfs = make_pfs_with_nbar(edges)
        _surv.pfs_elg = lambda p=pfs: p
        _rdm.pfs_elg = lambda p=pfs: p
        cfg = ForecastConfig.from_yaml("configs/default.yaml")
        cfg.overlap_area_deg2 = 1200.0
        cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
        cal = run_overlap_step1(cfg, cosmo, ps, verbose=False)
        res = step2_cosmo(cfg, cosmo, ps, cal)
        broad = res["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"  {label:<40s}  {prior:>10s}"
            for cp in COSMO_NAMES:
                s = res[prior][cp]
                imp = 100 * (broad[cp] - s) / broad[cp]
                row += f"  {s:>9.4g}  {imp:+5.1f}%"
            print(row)
        print('-'*100)


if __name__ == "__main__":
    main()
