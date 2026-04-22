#!/usr/bin/env python
"""Run PFS × DESI overlap calibration for each DESI tracer separately,
then combined. Shows which cross-survey combination drives the improvement.

Configurations:
  1. PFS-ELG × DESI-ELG  (with f_shared cross-shot correction)
  2. PFS-ELG × DESI-LRG  (zero cross-shot, guaranteed clean)
  3. PFS-ELG × DESI-QSO  (zero cross-shot, guaranteed clean)
  4. All combined         (with f_shared for ELG×ELG pair)
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
from pfsfog.surveys import pfs_elg, desi_elg, desi_lrg, desi_qso, SurveyGroup
from pfsfog.eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, tracer_fiducials,
    broad_priors, COSMO_PRIOR_SIGMA, desi_elg_fiducials,
)
from pfsfog.ps1loop_adapter import (
    fisher_to_ps1loop_auto, fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_func, make_ps1loop_pkmu_cross_func,
)
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPcross_dtheta_autodiff, dPell_d_cosmo_all
from pfsfog.covariance_mt_general import multi_tracer_cov_general
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_mt_general import multi_tracer_fisher_general, mt_general_param_names
from pfsfog.prior_export import CalibratedPriors
from pfsfog.fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from pfsfog.scenarios import SCENARIOS, nuisance_prior_diag, compute_improvement


def compute_cross_shot_noise(nbar_pfs, nbar_desi_elg, f_shared):
    """Cross-shot noise for partially shared ELG catalogues.

    P^{AB}_shot = f_shared / n_bar_PFS.
    Limits: f_shared=0 → 0 (independent); f_shared=1 → 1/n_PFS (fully shared).
    """
    if f_shared <= 0 or nbar_pfs <= 0:
        return 0.0
    return f_shared / nbar_pfs


def run_overlap_calibration(
    tracer_config: dict[str, object],
    cfg: ForecastConfig,
    cosmo: FiducialCosmology,
    ps,
    cross_shot_pairs: dict[tuple[str, str], float] | None = None,
    label: str = "",
):
    """Run overlap Fisher for a specific set of tracers.

    Parameters
    ----------
    tracer_config : {tracer_name: Survey}
    cfg : ForecastConfig
    cosmo : FiducialCosmology
    ps : PowerSpectrum1Loop
    cross_shot_pairs : {(nameA, nameB): 1/n_shared} for partially shared catalogues
    label : label for printing

    Returns
    -------
    calibrated : {(zlo,zhi): CalibratedPriors} for DESI-ELG
    """
    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, cfg.kmax_desi_overlap + cfg.dk / 2, cfg.dk)
    z_bins = cfg.z_bins

    calibrated = {}

    for zlo, zhi in z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff)
        f_z = float(cosmo.f(z_eff))
        h = cosmo.params["h"]

        # Filter to active tracers
        active = {}
        for tn, survey in tracer_config.items():
            nb = survey.nbar_eff(zlo, zhi)
            if nb > 1e-6:
                active[tn] = survey

        tracer_names = sorted(active.keys())
        if len(tracer_names) < 2:
            continue

        pk_data = cosmo.pk_data(z_eff)

        # Build fiducials and params
        b1_ref = 1.3
        for tn in tracer_names:
            if tn == "DESI-ELG":
                b1_ref = active[tn].b1_of_z(z_eff)
                break

        fids, ps1l_params, nbars = {}, {}, {}
        for tn in tracer_names:
            b1 = active[tn].b1_of_z(z_eff)
            nb = active[tn].nbar_eff(zlo, zhi)
            nbars[tn] = nb
            fid = tracer_fiducials(tn, b1, s8, b1_ref=b1_ref, r_sigma_v=cfg.r_sigma_v)
            par = fisher_to_ps1loop_auto(fid, s8, f_z, h, nb)
            fids[tn] = fid
            ps1l_params[tn] = par

        # Overlap volume (use any DESI tracer)
        V_ov = 0
        for tn in tracer_names:
            if "DESI" in tn:
                V_ov = active[tn].volume_rescaled(zlo, zhi, cfg.overlap_area_deg2)
                break
        if V_ov == 0:
            continue

        # Build pairs
        pairs = []
        for i, a in enumerate(tracer_names):
            pairs.append((a, a))
        for i, a in enumerate(tracer_names):
            for j in range(i + 1, len(tracer_names)):
                pairs.append((a, tracer_names[j]))

        # P(k,mu) callables
        pkmu_funcs = {}
        for (a, b) in pairs:
            if a == b:
                pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(ps, pk_data, ps1l_params[a])
            else:
                cross_params = fisher_to_ps1loop_cross(fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
                pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(ps, pk_data, cross_params)

        # Cross-shot noise for this z-bin
        cs_dict = {}
        if cross_shot_pairs:
            for pair_key, cs_func in cross_shot_pairs.items():
                if pair_key[0] in active and pair_key[1] in active:
                    cs_dict[pair_key] = cs_func(zlo, zhi)

        # Covariance
        cov = multi_tracer_cov_general(
            tracer_names, pkmu_funcs, nbars, k, V_ov, cfg.dk, ells,
            cross_shot=cs_dict if cs_dict else None,
        )

        # Derivatives
        derivs_auto = {}
        for tn in tracer_names:
            derivs_auto[tn] = dPell_dtheta_autodiff_all(
                ps, jnp.array(k), pk_data, ps1l_params[tn],
                NUISANCE_NAMES, s8, ells)

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

        # Fisher
        fr = multi_tracer_fisher_general(
            tracer_names, derivs_auto, derivs_cross, cov, k, cfg.dk, (zlo, zhi), ells)

        # Extract calibrated priors for DESI-ELG
        all_pn = mt_general_param_names(tracer_names)
        bp = broad_priors()
        bp_diag = bp.prior_fisher_diag()

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

        # Extract calibrated priors for each DESI tracer present
        # For the full-area forecast we need DESI-ELG priors.
        # If DESI-ELG is not in this pair, we still extract what we can
        # from the overlap (e.g. PFS×LRG constrains PFS params, which
        # indirectly helps — but for DESI-ELG we fall back to broad).
        for tn in tracer_names:
            if "DESI" not in tn:
                continue
            cal_params = {}
            for ip, nuis in enumerate(NUISANCE_NAMES):
                fn = f"{nuis}_{tn}"
                if fn in all_pn:
                    idx = all_pn.index(fn)
                    val = C_inv[idx, idx]
                    if val > 0:
                        cal_params[nuis] = float(np.sqrt(val))
                    else:
                        # Negative diagonal → regularization issue; use broad
                        cal_params[nuis] = broad_priors().sigma_dict().get(nuis) or 10.0

            if tn == "DESI-ELG":
                calibrated[(zlo, zhi)] = CalibratedPriors(
                    params=cal_params, z_bin=(zlo, zhi), source=label)

    return calibrated


def run_full_area_with_priors(calibrated, cfg, cosmo, ps, scenario_name="cross-cal"):
    """Run the full-area DESI-ELG Fisher with calibrated priors."""
    ells = (0, 2, 4)
    scenario = next(s for s in SCENARIOS if s.name == scenario_name)
    k_full = np.arange(cfg.kmin, scenario.kmax + cfg.dk / 2, cfg.dk)

    from pfsfog.surveys import desi_elg as make_desi_elg

    fishers = []
    for zlo, zhi in cfg.z_bins:
        z_eff = 0.5 * (zlo + zhi)
        s8 = cosmo.sigma8(z_eff)
        f_z = float(cosmo.f(z_eff))
        h = cosmo.params["h"]
        desi = make_desi_elg()
        nbar = desi.nbar_eff(zlo, zhi)
        if nbar == 0:
            continue
        b1 = desi.b1_of_z(z_eff)
        V_full = desi.volume(zlo, zhi)
        fid = desi_elg_fiducials(b1, s8)
        params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
        pk_data = cosmo.pk_data(z_eff)

        derivs = dPell_dtheta_autodiff_all(ps, jnp.array(k_full), pk_data, params, NUISANCE_NAMES, s8, ells)
        cosmo_derivs = dPell_d_cosmo_all(ps, jnp.array(k_full), pk_data, cosmo, params, z_eff, s8, ells)
        derivs.update(cosmo_derivs)

        pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params)
        cov_st = single_tracer_cov(pkmu_func, k_full, nbar, V_full, cfg.dk, ells)

        cal = calibrated.get((zlo, zhi))
        if scenario.prior_source == "oracle":
            nuis_prior = nuisance_prior_diag(scenario, None)
        elif scenario.prior_source == "broad":
            nuis_prior = nuisance_prior_diag(scenario, None)
        elif cal is not None:
            nuis_prior = nuisance_prior_diag(scenario, cal)
        else:
            nuis_prior = broad_priors().prior_fisher_diag()
        fr = full_area_fisher_per_zbin(derivs, cov_st, k_full, cfg.dk, nuis_prior, (zlo, zhi), scenario.kmax, ells)
        fishers.append(fr)

    if not fishers:
        return {}
    fr_combined = combine_zbins(fishers, cfg.z_bins)
    return {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}


def main():
    cfg = ForecastConfig.from_yaml("configs/default.yaml")
    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)

    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    pfs = pfs_elg()
    d_elg = desi_elg()
    d_lrg = desi_lrg()
    d_qso = desi_qso()

    f_shared = cfg.f_shared_elg

    # Cross-shot noise function for PFS-ELG × DESI-ELG
    def elg_cross_shot(zlo, zhi):
        nb_desi = d_elg.nbar_eff(zlo, zhi)
        return compute_cross_shot_noise(pfs.nbar_eff(zlo, zhi), nb_desi, f_shared)

    # --- Broad baseline ---
    print("=== Broad baseline (no cross-calibration) ===")
    broad_scenario = next(s for s in SCENARIOS if s.name == "broad")
    broad_sigmas = run_full_area_with_priors({}, cfg, cosmo, ps, "broad")
    for cp in COSMO_NAMES:
        print(f"  σ({cp}) = {broad_sigmas[cp]:.4e}")

    # --- Oracle ---
    print("\n=== Oracle (perfect nuisance knowledge) ===")
    oracle_sigmas = run_full_area_with_priors({}, cfg, cosmo, ps, "oracle")
    for cp in COSMO_NAMES:
        imp = compute_improvement(oracle_sigmas[cp], broad_sigmas[cp])
        print(f"  σ({cp}) = {oracle_sigmas[cp]:.4e}  ({imp:+.1f}%)")

    # --- Individual pair configurations ---
    configs = [
        ("PFS×DESI-ELG (f_shared=0)",
         {"PFS-ELG": pfs, "DESI-ELG": d_elg}, None),
        (f"PFS×DESI-ELG (f_shared={f_shared})",
         {"PFS-ELG": pfs, "DESI-ELG": d_elg},
         {("DESI-ELG", "PFS-ELG"): elg_cross_shot}),
        ("PFS×DESI-LRG",
         {"PFS-ELG": pfs, "DESI-LRG": d_lrg}, None),
        ("PFS×DESI-QSO",
         {"PFS-ELG": pfs, "DESI-QSO": d_qso}, None),
        (f"All combined (f_shared={f_shared})",
         {"PFS-ELG": pfs, "DESI-ELG": d_elg, "DESI-LRG": d_lrg, "DESI-QSO": d_qso},
         {("DESI-ELG", "PFS-ELG"): elg_cross_shot}),
    ]

    print(f"\n{'='*80}")
    print(f"{'Configuration':<35s} | {'σ(fσ8)':>10s} {'Δ%':>6s} | "
          f"{'σ(Mν)':>10s} {'Δ%':>6s} | {'σ(Ωm)':>10s} {'Δ%':>6s}")
    print(f"{'-'*80}")

    for label, tracers, cs_pairs in configs:
        print(f"\n--- {label} ---")
        cal = run_overlap_calibration(tracers, cfg, cosmo, ps, cs_pairs, label)

        # Show Pshot calibration
        for (zlo, zhi), c in sorted(cal.items()):
            ps_val = c.params.get("Pshot", float("nan"))
            ct_val = c.params.get("c_tilde", float("nan"))
            print(f"  z=[{zlo},{zhi}]: σ(Pshot)={ps_val:.3f}  σ(c̃)={ct_val:.1f}")

        # Full-area forecast
        sigmas = run_full_area_with_priors(cal, cfg, cosmo, ps, "cross-cal")
        row = f"{label:<35s}"
        for cp in COSMO_NAMES:
            s = sigmas.get(cp, float("nan"))
            imp = compute_improvement(s, broad_sigmas[cp])
            row += f" | {s:10.4e} {imp:+5.1f}%"
        print(f"\n  Result: {row}")

    print(f"\n{'='*80}")
    print("Broad baseline and oracle for reference:")
    for cp in COSMO_NAMES:
        print(f"  σ({cp}): broad={broad_sigmas[cp]:.4e}  oracle={oracle_sigmas[cp]:.4e}")


if __name__ == "__main__":
    main()
