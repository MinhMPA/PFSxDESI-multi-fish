#!/usr/bin/env python
"""Full-area DESI forecast using all 6 DR2 samples with calibrated priors.

Step 1: Multi-tracer overlap calibration (PFS × DESI) → calibrated priors
        for DESI-ELG, DESI-LRG, DESI-QSO per overlap z-bin.
Step 2: Single-tracer auto-spectrum Fisher for each DR2 sample
        (LRG1–3, ELG1–2, QSO), then combine via shared cosmology.

Usage:
    python scripts/run_desi_multisample.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.surveys import desi_elg, desi_lrg, desi_qso, pfs_elg, SurveyGroup
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
from pfsfog.scenarios import compute_improvement


# ---------------------------------------------------------------------------
# DR2 sample definitions
# ---------------------------------------------------------------------------

@dataclass
class DR2Sample:
    name: str
    tracer: str
    z_range: tuple[float, float]
    overlap_zbins: list[tuple[float, float]]  # Step 1 bins for prior averaging


DR2_SAMPLES = [
    DR2Sample("LRG1", "DESI-LRG", (0.4, 0.6),  []),
    DR2Sample("LRG2", "DESI-LRG", (0.6, 0.8),  [(0.6, 0.8)]),
    DR2Sample("LRG3", "DESI-LRG", (0.8, 1.1),  [(0.8, 1.0), (1.0, 1.2)]),
    DR2Sample("ELG1", "DESI-ELG", (0.8, 1.1),  [(0.8, 1.0), (1.0, 1.2)]),
    DR2Sample("ELG2", "DESI-ELG", (1.1, 1.6),  [(1.2, 1.4), (1.4, 1.6)]),
    DR2Sample("QSO",  "DESI-QSO", (0.8, 2.1),  [(0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]),
]

SURVEY_CONSTRUCTORS = {
    "DESI-LRG": desi_lrg,
    "DESI-ELG": desi_elg,
    "DESI-QSO": desi_qso,
}


# ---------------------------------------------------------------------------
# Step 1: Overlap calibration (extracted from cli_multitrace.py)
# ---------------------------------------------------------------------------

def run_overlap_step1(cfg, cosmo, ps, verbose=True):
    """Run Step 1 only: multi-tracer overlap → calibrated priors per tracer per z-bin."""
    sg = SurveyGroup(
        pfs=pfs_elg(),
        desi_tracers={"DESI-ELG": desi_elg(), "DESI-LRG": desi_lrg(), "DESI-QSO": desi_qso()},
        overlap_area_deg2=cfg.overlap_area_deg2,
    )
    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, cfg.kmax_desi_overlap + cfg.dk / 2, cfg.dk)

    calibrated = {}  # {tracer_name: {z_bin: CalibratedPriors}}

    for zlo, zhi in cfg.z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff)
        f_z = float(cosmo.f(z_eff))
        h = cosmo.params["h"]
        V_ov = sg.V_overlap(zlo, zhi)
        active = sg.active_tracers(zlo, zhi)
        tracer_names = sorted(active.keys())
        Nt = len(tracer_names)

        if Nt < 2:
            if verbose:
                print(f"  z=[{zlo},{zhi}]: only {Nt} tracer(s), skipping")
            continue

        pairs = sg.tracer_pairs(zlo, zhi)
        if verbose:
            print(f"  z=[{zlo},{zhi}]: {Nt} tracers {tracer_names}, "
                  f"{len(pairs)} spectra, V_ov={V_ov:.2e}")

        pk_data = cosmo.pk_data(z_eff)
        b1_ref = 1.3
        for tn in tracer_names:
            if "ELG" in tn and "PFS" not in tn:
                b1_ref = active[tn].b1_of_z(z_eff)
                break

        fids, ps1l_params, nbars = {}, {}, {}
        for tn in tracer_names:
            s = active[tn]
            b1 = s.b1_of_z(z_eff)
            nb = s.nbar_eff(zlo, zhi)
            nbars[tn] = nb
            fid = tracer_fiducials(tn, b1, s8, b1_ref=b1_ref, r_sigma_v=cfg.r_sigma_v)
            par = fisher_to_ps1loop_auto(fid, s8, f_z, h, nb)
            fids[tn] = fid
            ps1l_params[tn] = par

        # P(k,mu) callables
        pkmu_funcs = {}
        for (a, b) in pairs:
            if a == b:
                pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(ps, pk_data, ps1l_params[a])
            else:
                cross_params = fisher_to_ps1loop_cross(fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
                pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(ps, pk_data, cross_params)

        # Cross-shot noise
        cross_shot = None
        if cfg.f_shared_elg > 0 and "PFS-ELG" in active and "DESI-ELG" in active:
            cross_shot = {("DESI-ELG", "PFS-ELG"): cfg.f_shared_elg / nbars["DESI-ELG"]}

        cov = multi_tracer_cov_general(
            tracer_names, pkmu_funcs, nbars, k, V_ov, cfg.dk, ells,
            cross_shot=cross_shot)

        # Derivatives
        derivs_auto = {}
        for tn in tracer_names:
            derivs_auto[tn] = dPell_dtheta_autodiff_all(
                ps, jnp.array(k), pk_data, ps1l_params[tn], NUISANCE_NAMES, s8, ells)

        derivs_cross = {}
        for (a, b) in pairs:
            if a == b:
                continue
            cross_params = fisher_to_ps1loop_cross(fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
            cd = {}
            for nn in NUISANCE_NAMES:
                for side in ["A", "B"]:
                    dd = {}
                    for ell in ells:
                        dd[ell] = dPcross_dtheta_autodiff(
                            ps, jnp.array(k), pk_data, cross_params, nn, s8, side, ell)
                    cd[f"{nn}:{side}"] = dd
            derivs_cross[(a, b)] = cd

        fr = multi_tracer_fisher_general(
            tracer_names, derivs_auto, derivs_cross, cov, k, cfg.dk, (zlo, zhi), ells)

        # Extract calibrated priors for each DESI tracer
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
            if verbose:
                print(f"    WARNING: Fisher not invertible at z=[{zlo},{zhi}]")
            continue

        for tn in tracer_names:
            if "DESI" not in tn:
                continue
            cal_params = {}
            for ip, nuis in enumerate(NUISANCE_NAMES):
                fn = f"{nuis}_{tn}"
                if fn in all_pn:
                    idx = all_pn.index(fn)
                    cal_params[nuis] = float(np.sqrt(max(C_inv[idx, idx], 0)))

            if tn not in calibrated:
                calibrated[tn] = {}
            calibrated[tn][(zlo, zhi)] = CalibratedPriors(
                params=cal_params, z_bin=(zlo, zhi), source="overlap")

            if verbose:
                ct = cal_params.get("c_tilde", float("nan"))
                ps_val = cal_params.get("Pshot", float("nan"))
                print(f"    {tn}: σ_cal(c̃)={ct:.1f}  σ_cal(Pshot)={ps_val:.3f}")

    return calibrated


# ---------------------------------------------------------------------------
# Prior averaging
# ---------------------------------------------------------------------------

def average_calibrated_prior_diag(cal_per_z, overlap_zbins):
    """Average calibrated σ across overlap z-bins, return 1/σ_avg² diagonal."""
    if not overlap_zbins:
        return broad_priors().prior_fisher_diag()

    sigma_lists = {n: [] for n in NUISANCE_NAMES}
    for zb in overlap_zbins:
        cal = cal_per_z.get(zb)
        if cal is None:
            continue
        for n in NUISANCE_NAMES:
            s = cal.params.get(n)
            if s is not None and s > 0:
                sigma_lists[n].append(s)

    bp = broad_priors().sigma_dict()
    diag = np.zeros(len(NUISANCE_NAMES))
    for i, n in enumerate(NUISANCE_NAMES):
        if sigma_lists[n]:
            sigma_avg = np.mean(sigma_lists[n])
            diag[i] = 1.0 / sigma_avg**2
        else:
            s = bp[n]
            diag[i] = 0.0 if s is None else 1.0 / s**2
    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    # --- Step 1: Overlap calibration with extended z-bins ---
    cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    print("=== Step 1: Multi-tracer overlap calibration ===")
    cal_per_tracer = run_overlap_step1(cfg, cosmo, ps, verbose=True)

    # --- Step 2: Full-area multi-sample DESI forecast ---
    print("\n=== Step 2: Full-area 6-sample DESI forecast ===")
    ells = (0, 2, 4)
    kmax = 0.20
    k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)

    scenarios = {
        "broad":     lambda s: broad_priors().prior_fisher_diag(),
        "cross-cal": lambda s: average_calibrated_prior_diag(
            cal_per_tracer.get(s.tracer, {}), s.overlap_zbins),
        "oracle":    lambda s: np.full(len(NUISANCE_NAMES), 1e20),
    }

    results = {}

    for scenario_name, prior_fn in scenarios.items():
        fishers = []
        sample_labels = []
        z_bins_list = []

        for sample in DR2_SAMPLES:
            survey = SURVEY_CONSTRUCTORS[sample.tracer]()
            zlo, zhi = sample.z_range
            z_eff = 0.5 * (zlo + zhi)
            nbar = survey.nbar_eff(zlo, zhi)
            b1 = survey.b1_of_z(z_eff)
            V = survey.volume(zlo, zhi)
            s8 = cosmo.sigma8(z_eff)
            f_z = float(cosmo.f(z_eff))
            h = cosmo.params["h"]

            if nbar == 0:
                if scenario_name == "broad":
                    print(f"  SKIP {sample.name}: nbar=0 at z=[{zlo},{zhi}]")
                continue

            if scenario_name == "broad":
                has_cal = bool(sample.overlap_zbins)
                print(f"  {sample.name:<6s} {sample.tracer:<10s} "
                      f"z=[{zlo:.1f},{zhi:.1f}]  n̄={nbar:.2e}  b1={b1:.2f}  "
                      f"V={V/1e9:.1f} (Gpc/h)³  prior={'cal' if has_cal else 'broad'}")

            fid = tracer_fiducials(sample.tracer, b1, s8)
            params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
            pk_data = cosmo.pk_data(z_eff)

            derivs = dPell_dtheta_autodiff_all(
                ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
            cosmo_derivs = dPell_d_cosmo_all(
                ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells)
            derivs.update(cosmo_derivs)

            pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params)
            cov = single_tracer_cov(pkmu_func, k, nbar, V, cfg.dk, ells)

            nuis_prior = prior_fn(sample)

            fr = full_area_fisher_per_zbin(
                derivs, cov, k, cfg.dk, nuis_prior,
                sample.z_range, kmax, ells,
                survey_name=f"DESI-{sample.name}",
            )
            fishers.append(fr)
            sample_labels.append(sample.name)
            z_bins_list.append(sample.z_range)

        fr_combined = combine_zbins(
            fishers, z_bins_list,
            sample_labels=sample_labels,
            survey_name=f"DESI 6-sample {scenario_name}",
        )

        sigmas = {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}
        results[scenario_name] = sigmas

    # --- Print results ---
    print(f"\n{'='*80}")
    print(f"{'Scenario':<18s}  {'σ(fσ8)':>10s}  {'Δ%':>6s}  "
          f"{'σ(Mν) [eV]':>10s}  {'Δ%':>6s}  {'σ(Ωm)':>10s}  {'Δ%':>6s}")
    print(f"{'-'*80}")

    broad = results["broad"]
    for sn in ["broad", "cross-cal", "oracle"]:
        row = f"{sn:<18s}"
        for cp in COSMO_NAMES:
            s = results[sn][cp]
            imp = compute_improvement(s, broad[cp])
            row += f"  {s:10.4e}  {imp:+5.1f}%"
        print(row)

    print(f"\nDegradation (broad/oracle):")
    for cp in COSMO_NAMES:
        print(f"  {cp}: {broad[cp]/results['oracle'][cp]:.1f}×")


if __name__ == "__main__":
    main()
