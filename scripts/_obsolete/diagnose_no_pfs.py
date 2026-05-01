#!/usr/bin/env python
"""Test whether the σ(Mν) gain comes from PFS or from DESI-internal multi-tracer.

Compares:
  A. STARVED PFS (n̄ → 0) — PFS effectively absent
  B. NO PFS — PFS literally removed from SurveyGroup (DESI-only multi-tracer)
  C. NO STEP 1 — Cross-cal scenario uses broad priors (no calibration at all)
  D. BASELINE Takada — for reference

If (A) ≈ (B) ≈ (D) cross-cal Mν, then the gain comes from DESI-internal calibration,
not from PFS at all — the paper's PFS×DESI framing would be misleading.
If (C) ≈ no improvement, that confirms calibration is the only source of gain.
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
from pfsfog.surveys import Survey, SurveyGroup, desi_elg, desi_lrg, desi_qso

import scripts.run_desi_multisample as _rdm
from scripts.run_desi_multisample import (
    DR2_SAMPLES, SURVEY_CONSTRUCTORS,
    run_overlap_step1, average_calibrated_prior_diag,
)
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, COSMO_PRIOR_SIGMA,
    broad_priors, tracer_fiducials,
)
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto, fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_func, make_ps1loop_pkmu_cross_func,
)
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPcross_dtheta_autodiff, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.covariance_mt_general import multi_tracer_cov_general
from pfsfog.fisher_mt_general import multi_tracer_fisher_general, mt_general_param_names
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.prior_export import CalibratedPriors


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
                  z_min_all=zmin_t, z_max_all=zmax_t, nz_all=nz, Vz_all=Vz_t,
                  b1_of_z=lambda z: 0.9 + 0.4 * z)


def run_overlap_no_pfs(cfg, cosmo, ps, verbose=False):
    """Step 1 with DESI-only multi-tracer — PFS literally not in the group."""
    sg = SurveyGroup(
        pfs=desi_elg(),  # placeholder — never used because we override active_tracers
        desi_tracers={"DESI-ELG": desi_elg(),
                      "DESI-LRG": desi_lrg(),
                      "DESI-QSO": desi_qso()},
        overlap_area_deg2=cfg.overlap_area_deg2,
    )
    # Replace all_surveys to exclude PFS
    desi_only = {"DESI-ELG": desi_elg(), "DESI-LRG": desi_lrg(), "DESI-QSO": desi_qso()}

    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, cfg.kmax_desi_overlap + cfg.dk / 2, cfg.dk)
    calibrated = {}

    for zlo, zhi in cfg.z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff); f_z = float(cosmo.f(z_eff)); h = cosmo.params["h"]

        # Active tracers (DESI only)
        active = {n: s for n, s in desi_only.items() if s.nbar_eff(zlo, zhi) > 1e-6}
        tracer_names = sorted(active.keys())
        Nt = len(tracer_names)
        if Nt < 2:
            continue
        V_ov = sg.V_overlap(zlo, zhi)
        pairs = [(a, a) for a in tracer_names] + \
                [(tracer_names[i], tracer_names[j])
                 for i in range(Nt) for j in range(i+1, Nt)]
        pk_data = cosmo.pk_data(z_eff)

        b1_ref = active.get("DESI-ELG", list(active.values())[0]).b1_of_z(z_eff)
        fids, ps1l_params, nbars = {}, {}, {}
        for tn in tracer_names:
            s = active[tn]
            b1 = s.b1_of_z(z_eff); nb = s.nbar_eff(zlo, zhi)
            nbars[tn] = nb
            fid = tracer_fiducials(tn, b1, s8, b1_ref=b1_ref, r_sigma_v=cfg.r_sigma_v)
            par = fisher_to_ps1loop_auto(fid, s8, f_z, h, nb)
            fids[tn] = fid; ps1l_params[tn] = par

        pkmu_funcs = {}
        for (a, b) in pairs:
            if a == b:
                pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(ps, pk_data, ps1l_params[a])
            else:
                cp = fisher_to_ps1loop_cross(fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
                pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(ps, pk_data, cp)

        cov = multi_tracer_cov_general(tracer_names, pkmu_funcs, nbars, k, V_ov, cfg.dk, ells)

        derivs_auto = {tn: dPell_dtheta_autodiff_all(
                          ps, jnp.array(k), pk_data, ps1l_params[tn], NUISANCE_NAMES, s8, ells)
                       for tn in tracer_names}
        derivs_cross = {}
        for (a, b) in pairs:
            if a == b: continue
            cp = fisher_to_ps1loop_cross(fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
            cd = {}
            for nn in NUISANCE_NAMES:
                for side in ["A", "B"]:
                    dd = {ell: dPcross_dtheta_autodiff(
                              ps, jnp.array(k), pk_data, cp, nn, s8, side, ell)
                          for ell in ells}
                    cd[f"{nn}:{side}"] = dd
            derivs_cross[(a, b)] = cd

        fr = multi_tracer_fisher_general(tracer_names, derivs_auto, derivs_cross,
                                         cov, k, cfg.dk, (zlo, zhi), ells)
        all_pn = mt_general_param_names(tracer_names)
        bp_diag = broad_priors().prior_fisher_diag()
        prior_diag = np.zeros(len(all_pn))
        for i, pn in enumerate(all_pn):
            if pn in COSMO_PRIOR_SIGMA:
                prior_diag[i] = 1.0 / COSMO_PRIOR_SIGMA[pn]**2
            else:
                for nuis in NUISANCE_NAMES:
                    if pn.startswith(nuis + "_"):
                        prior_diag[i] = bp_diag[NUISANCE_NAMES.index(nuis)]
                        break
        F_reg = fr.F + np.diag(prior_diag)
        try:
            C_inv = np.linalg.inv(F_reg)
        except np.linalg.LinAlgError:
            continue
        for tn in tracer_names:
            cal_params = {}
            for ip, nuis in enumerate(NUISANCE_NAMES):
                fn = f"{nuis}_{tn}"
                if fn in all_pn:
                    idx = all_pn.index(fn)
                    cal_params[nuis] = float(np.sqrt(max(C_inv[idx, idx], 0)))
            calibrated.setdefault(tn, {})[(zlo, zhi)] = CalibratedPriors(
                params=cal_params, z_bin=(zlo, zhi), source="overlap-no-pfs")
    return calibrated


def step2_cosmo(cfg, cosmo, ps, cal_per_tracer):
    ells = (0, 2, 4); kmax = 0.20
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
            z_eff = 0.5 * (zlo + zhi); nbar = survey.nbar_eff(zlo, zhi)
            if nbar == 0: continue
            b1 = survey.b1_of_z(z_eff); V = survey.volume(zlo, zhi)
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

    print(f"\n{'='*100}")
    print(f"{'Config':<48s}  {'Prior':>10s}  {'σ(fσ8)':>10s}{'Δ%':>7s}  "
          f"{'σ(Mν) [eV]':>11s}{'Δ%':>7s}  {'σ(Ωm)':>10s}{'Δ%':>7s}")
    print('='*100)

    # === Config A: STARVED PFS ===
    pfs = make_pfs_with_nbar([(0.6,1.0,0.3e-4),(1.0,1.6,0.4e-4)])
    _surv.pfs_elg = lambda p=pfs: p
    _rdm.pfs_elg = lambda p=pfs: p
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cfg.overlap_area_deg2 = 1200.0
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    cal = run_overlap_step1(cfg, cosmo, ps, verbose=False)
    res_starved = step2_cosmo(cfg, cosmo, ps, cal)

    # === Config B: NO PFS (DESI-only multi-tracer in Step 1) ===
    cal_no_pfs = run_overlap_no_pfs(cfg, cosmo, ps, verbose=False)
    res_no_pfs = step2_cosmo(cfg, cosmo, ps, cal_no_pfs)

    # === Config C: NO STEP 1 (cross-cal == broad) ===
    res_no_step1 = step2_cosmo(cfg, cosmo, ps, {})  # empty calibration

    # === Config D: BASELINE Takada PFS ===
    pfs = make_pfs_with_nbar([(0.6,1.0,3e-4),(1.0,1.6,4e-4)])
    _surv.pfs_elg = lambda p=pfs: p
    _rdm.pfs_elg = lambda p=pfs: p
    cal_base = run_overlap_step1(cfg, cosmo, ps, verbose=False)
    res_base = step2_cosmo(cfg, cosmo, ps, cal_base)

    rows = [
        ("A. STARVED PFS (n̄=0.3,0.3,0.4,0.4,0.4)", res_starved),
        ("B. NO PFS (DESI-only Step 1)",              res_no_pfs),
        ("C. NO STEP 1 (cross-cal = broad)",         res_no_step1),
        ("D. BASELINE Takada (n̄=3,3,4,4,4)",         res_base),
    ]
    for label, res in rows:
        broad = res["broad"]
        for prior in ["broad", "cross-cal"]:
            row = f"  {label:<48s}  {prior:>10s}"
            for cp in COSMO_NAMES:
                s = res[prior][cp]
                imp = 100 * (broad[cp] - s) / broad[cp]
                row += f"  {s:>9.4g}  {imp:+5.1f}%"
            print(row)
        print('-'*100)


if __name__ == "__main__":
    main()
