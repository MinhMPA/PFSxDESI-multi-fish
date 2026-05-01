"""Multi-tracer pipeline: PFS × {DESI-ELG, DESI-LRG, DESI-QSO}.

DEPRECATED — legacy two-stage CLI. Use ``scripts/run_joint_fisher.py`` for
the proper joint Fisher analysis.

Generalises cli.py to N tracers in the overlap volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .config import ForecastConfig
from .cosmo import FiducialCosmology
from .surveys import default_survey_group, SurveyGroup
from .eft_params import (
    NUISANCE_NAMES, COSMO_NAMES, tracer_fiducials,
    broad_priors, COSMO_PRIOR_SIGMA,
)
from .ps1loop_adapter import (
    fisher_to_ps1loop_auto, fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_func, make_ps1loop_pkmu_cross_func,
)
from .derivatives import dPell_dtheta_autodiff_all, dPcross_dtheta_autodiff, dPell_d_cosmo_all
from .covariance_mt_general import multi_tracer_cov_general
from .fisher_mt_general import multi_tracer_fisher_general, mt_general_param_names
from .prior_export import CalibratedPriors
from .fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from .covariance import single_tracer_cov
from .scenarios import (
    SCENARIOS, Scenario, nuisance_prior_diag,
    SummaryRow, compute_improvement, compute_calibration_efficiency,
    write_summary_csv,
)


@dataclass
class MultiTraceResults:
    """Results from the multi-tracer pipeline."""
    config: ForecastConfig
    calibrated_per_tracer_z: dict[str, dict[tuple[float, float], CalibratedPriors]]
    scenario_results: dict[str, dict]  # {scenario_name: {cosmo_param: sigma}}
    summary_rows: list[SummaryRow]
    output_dir: Path


def _get_fiducials_and_params(tracer_name, b1, s8, f_z, h, nbar, r_sigma_v, b1_ref):
    """Build fiducials and ps_1loop_jax params for one tracer."""
    fid = tracer_fiducials(tracer_name, b1, s8, b1_ref=b1_ref, r_sigma_v=r_sigma_v)
    params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
    return fid, params


def run_multitrace_pipeline(cfg: ForecastConfig, verbose: bool = True) -> MultiTraceResults:
    """Run the full multi-tracer pipeline."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"mt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output → {out_dir}")

    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    sg = default_survey_group()
    sg.overlap_area_deg2 = cfg.overlap_area_deg2

    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    ells = (0, 2, 4)
    k = np.arange(cfg.kmin, cfg.kmax_desi_overlap + cfg.dk / 2, cfg.dk)

    # Storage
    calibrated_per_tracer_z: dict[str, dict[tuple[float, float], CalibratedPriors]] = {}

    # =====================================================================
    # STEP 1: Multi-tracer overlap Fisher per z-bin
    # =====================================================================
    if verbose:
        print("\n=== Step 1: Multi-tracer overlap calibration ===")

    z_bins = cfg.z_bins

    for zlo, zhi in z_bins:
        z_eff_ref = 0.5 * (zlo + zhi)
        # Use a DESI tracer for z_eff (any will do for overlap volume)
        s8 = cosmo.sigma8(z_eff_ref)
        f_z = float(cosmo.f(z_eff_ref))
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

        # Build fiducials, params, P(k,mu) for each tracer
        pk_data = cosmo.pk_data(z_eff_ref)
        fids = {}
        ps1l_params = {}
        nbars = {}

        # Get a reference b1 for PFS scaling
        b1_ref = 1.3
        for tn in tracer_names:
            if "ELG" in tn and "PFS" not in tn:
                b1_ref = active[tn].b1_of_z(z_eff_ref)
                break

        for tn in tracer_names:
            s = active[tn]
            b1 = s.b1_of_z(z_eff_ref)
            nb = s.nbar_eff(zlo, zhi)
            nbars[tn] = nb
            fid, par = _get_fiducials_and_params(
                tn, b1, s8, f_z, h, nb, cfg.r_sigma_v, b1_ref)
            fids[tn] = fid
            ps1l_params[tn] = par

        # Build P(k,mu) callables for covariance
        pkmu_funcs = {}
        for (a, b) in pairs:
            if a == b:
                pkmu_funcs[(a, a)] = make_ps1loop_pkmu_func(
                    ps, pk_data, ps1l_params[a])
            else:
                cross_params = fisher_to_ps1loop_cross(
                    fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
                pkmu_funcs[(a, b)] = make_ps1loop_pkmu_cross_func(
                    ps, pk_data, cross_params)

        # Cross-shot noise for PFS-ELG × DESI-ELG (shared catalog).
        # f_shared = n_shared / n_PFS, so P^{AB}_shot = f_shared / n_DESI.
        # Limits: f=0 → 0 (independent); f=1 → 1/n_DESI (all PFS-ELGs in DESI).
        cross_shot = None
        if cfg.f_shared_elg > 0 and "PFS-ELG" in active and "DESI-ELG" in active:
            cross_shot = {("DESI-ELG", "PFS-ELG"): cfg.f_shared_elg / nbars["DESI-ELG"]}

        # Multi-tracer covariance
        cov = multi_tracer_cov_general(
            tracer_names, pkmu_funcs, nbars, k, V_ov, cfg.dk, ells,
            cross_shot=cross_shot)

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
            cross_params = fisher_to_ps1loop_cross(
                fids[a], fids[b], s8, f_z, h, nbars[a], nbars[b])
            cross_d = {}
            for nn in NUISANCE_NAMES:
                for side, which in [("A", "A"), ("B", "B")]:
                    dd = {}
                    for ell in ells:
                        dd[ell] = dPcross_dtheta_autodiff(
                            ps, jnp.array(k), pk_data, cross_params,
                            nn, s8, which, ell)
                    cross_d[f"{nn}:{side}"] = dd
            derivs_cross[(a, b)] = cross_d

        # Assemble Fisher
        fr_mt = multi_tracer_fisher_general(
            tracer_names, derivs_auto, derivs_cross,
            cov, k, cfg.dk, z_bin=(zlo, zhi), ells=ells)

        # Extract calibrated priors for each DESI tracer
        all_param_names = mt_general_param_names(tracer_names)
        bp = broad_priors()
        bp_diag = bp.prior_fisher_diag()

        # Build regularization prior
        prior_diag = np.zeros(len(all_param_names))
        for i, pn in enumerate(all_param_names):
            if pn in COSMO_PRIOR_SIGMA:
                prior_diag[i] = 1.0 / COSMO_PRIOR_SIGMA[pn]**2
            else:
                # Find the nuisance param base name
                for nuis in NUISANCE_NAMES:
                    if pn.startswith(nuis + "_"):
                        idx = NUISANCE_NAMES.index(nuis)
                        prior_diag[i] = bp_diag[idx]
                        break

        F_reg = fr_mt.F + np.diag(prior_diag)
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
                full_name = f"{nuis}_{tn}"
                if full_name in all_param_names:
                    idx = all_param_names.index(full_name)
                    cal_params[nuis] = float(np.sqrt(C_inv[idx, idx]))

            cal = CalibratedPriors(params=cal_params, z_bin=(zlo, zhi),
                                   source=f"Multi-tracer overlap ({Nt} tracers)")

            if tn not in calibrated_per_tracer_z:
                calibrated_per_tracer_z[tn] = {}
            calibrated_per_tracer_z[tn][(zlo, zhi)] = cal

            if verbose:
                ct = cal.params.get("c_tilde", float("nan"))
                ps_val = cal.params.get("Pshot", float("nan"))
                print(f"    {tn}: σ_cal(c̃)={ct:.1f}  σ_cal(Pshot)={ps_val:.3f}")

    # Save calibrated priors
    priors_dir = out_dir / "priors"
    priors_dir.mkdir(exist_ok=True)
    for tn, z_dict in calibrated_per_tracer_z.items():
        for (zlo, zhi), cal in z_dict.items():
            jp = priors_dir / f"cross_calibrated_{tn}_z{zlo:.1f}_{zhi:.1f}.json"
            with open(jp, "w") as f:
                json.dump({"tracer": tn, "z_bin": [zlo, zhi], "source": cal.source,
                            "priors": {k: {"sigma": v} for k, v in cal.params.items()}},
                          f, indent=2)

    # =====================================================================
    # STEP 2: Full-area DESI Fisher per scenario (using DESI-ELG)
    # =====================================================================
    if verbose:
        print("\n=== Step 2: Full-area DESI-ELG Fisher ===")

    summary_rows: list[SummaryRow] = []
    scenario_results: dict[str, dict] = {}
    broad_sigmas: dict[str, float] = {}

    # Use DESI-ELG calibrated priors for the full-area forecast
    cal_elg = calibrated_per_tracer_z.get("DESI-ELG", {})

    for scenario in SCENARIOS:
        if verbose:
            print(f"\n  Scenario: {scenario.name} (kmax={scenario.kmax})")
        k_full = np.arange(cfg.kmin, scenario.kmax + cfg.dk / 2, cfg.dk)

        fishers_per_z = []
        for zlo, zhi in z_bins:
            z_eff = 0.5 * (zlo + zhi)
            s8 = cosmo.sigma8(z_eff)
            f_z = float(cosmo.f(z_eff))
            h = cosmo.params["h"]

            from .surveys import desi_elg
            desi = desi_elg()
            nbar_desi = desi.nbar_eff(zlo, zhi)
            if nbar_desi == 0:
                continue
            b1_desi = desi.b1_of_z(z_eff)
            V_full = desi.volume(zlo, zhi)

            from .eft_params import desi_elg_fiducials
            fid_desi = desi_elg_fiducials(b1_desi, s8)
            params_desi = fisher_to_ps1loop_auto(fid_desi, s8, f_z, h, nbar_desi)
            pk_data = cosmo.pk_data(z_eff)

            derivs = dPell_dtheta_autodiff_all(
                ps, jnp.array(k_full), pk_data, params_desi,
                NUISANCE_NAMES, s8, ells)
            cosmo_derivs = dPell_d_cosmo_all(
                ps, jnp.array(k_full), pk_data, cosmo, params_desi,
                z_eff, s8, ells)
            derivs.update(cosmo_derivs)

            pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params_desi)
            cov_st = single_tracer_cov(pkmu_func, k_full, nbar_desi, V_full, cfg.dk, ells)

            cal = cal_elg.get((zlo, zhi))
            if scenario.prior_source == "cross-cal" and cal is None:
                # No calibrated priors for this z-bin — fall back to broad
                nuis_prior = broad_priors().prior_fisher_diag()
            else:
                nuis_prior = nuisance_prior_diag(scenario, cal)

            fr = full_area_fisher_per_zbin(
                derivs, cov_st, k_full, cfg.dk, nuis_prior,
                (zlo, zhi), scenario.kmax, ells)
            fishers_per_z.append(fr)

        if not fishers_per_z:
            continue

        fr_combined = combine_zbins(fishers_per_z, z_bins)
        sigmas_combined = {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}
        scenario_results[scenario.name] = sigmas_combined

        for cp in COSMO_NAMES:
            sigma = sigmas_combined[cp]
            if scenario.name == "broad":
                broad_sigmas[cp] = sigma
            sigma_broad = broad_sigmas.get(cp, sigma)
            if scenario.name == "oracle":
                broad_sigmas[f"{cp}_oracle"] = sigma
            sigma_oracle = broad_sigmas.get(f"{cp}_oracle", sigma)

            improvement = compute_improvement(sigma, sigma_broad)
            efficiency = compute_calibration_efficiency(sigma, sigma_broad, sigma_oracle)

            summary_rows.append(SummaryRow(
                scenario=scenario.name, kmax=scenario.kmax,
                z_bin_min=z_bins[0][0], z_bin_max=z_bins[-1][1],
                param_name=cp, sigma_marginalized=sigma,
                sigma_broad_baseline=sigma_broad, improvement_pct=improvement,
                calibration_efficiency=efficiency,
            ))

            if verbose:
                print(f"    σ({cp}) = {sigma:.4e}  (improvement: {improvement:+.1f}%)")

    csv_path = out_dir / "summary.csv"
    write_summary_csv(summary_rows, str(csv_path))
    if verbose:
        print(f"\nSummary written to {csv_path}")

    return MultiTraceResults(
        config=cfg,
        calibrated_per_tracer_z=calibrated_per_tracer_z,
        scenario_results=scenario_results,
        summary_rows=summary_rows,
        output_dir=out_dir,
    )
