"""End-to-end pipeline: overlap → export → full-area Fisher → summary.

Usage:
    python -m pfsfog.cli [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .config import ForecastConfig
from .cosmo import FiducialCosmology
from .surveys import default_survey_pair, OVERLAP_ZBINS
from .eft_params import (
    NUISANCE_NAMES, COSMO_NAMES,
    desi_elg_fiducials, pfs_elg_fiducials, broad_priors,
)
from .ps1loop_adapter import (
    fisher_to_ps1loop_auto, fisher_to_ps1loop_cross,
    make_ps1loop_pkmu_func, make_ps1loop_pkmu_cross_func,
)
from .builtin_pkmu import pkmu_auto, pkmu_cross
from .derivatives import dPell_dtheta_autodiff_all, dPcross_dtheta_autodiff, dPell_d_cosmo_all
from .covariance import single_tracer_cov, multi_tracer_cov
from .fisher import single_tracer_fisher, FisherResult
from .fisher_mt import multi_tracer_fisher, multi_tracer_fisher_asymmetric, _IDX_REDUCED
from .prior_export import export_calibrated_priors, CalibratedPriors
from .fisher_full_area import full_area_fisher_per_zbin, combine_zbins
from .scenarios import (
    SCENARIOS, Scenario, nuisance_prior_diag,
    SummaryRow, compute_improvement, compute_calibration_efficiency,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    """Per-z-bin overlap calibration result."""
    z_bin: tuple[float, float]
    calibrated_priors: CalibratedPriors
    fisher_mt: FisherResult
    # Single-tracer sigmas in the overlap (for Fig 1)
    sigma_desi_only: dict[str, float] = field(default_factory=dict)
    sigma_pfs_only: dict[str, float] = field(default_factory=dict)
    sigma_mt: dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Per-scenario result (combined z-bins)."""
    scenario: Scenario
    fisher_combined: FisherResult
    fishers_per_z: list[FisherResult]
    sigmas_combined: dict[str, float]     # {cosmo_param: σ}
    sigmas_per_z: dict[tuple[float, float], dict[str, float]]


@dataclass
class PipelineResults:
    """All results from one pipeline run."""
    config: ForecastConfig
    overlap_results: dict[tuple[float, float], OverlapResult]
    scenario_results: dict[str, ScenarioResult]
    summary_rows: list[SummaryRow]
    output_dir: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pkmu_func(plin_k, b1, f, ctr_dict, nbar):
    def func(k, mu):
        pl = np.interp(k, plin_k["k"], plin_k["pk"])
        return pkmu_auto(
            np.asarray(k), np.asarray(mu), pl,
            b1=b1, f=f,
            c0=ctr_dict.get("c0", 0.0),
            c2=ctr_dict.get("c2", 0.0),
            c4=ctr_dict.get("c4", 0.0),
            cfog=ctr_dict.get("c_tilde", 0.0),
            nbar=nbar,
            Pshot=ctr_dict.get("Pshot", 0.0),
            a0=ctr_dict.get("a0", 0.0),
            a2=ctr_dict.get("a2", 0.0),
        )
    return func


def _make_pkmu_cross_func(plin_k, b1_A, b1_B, f, fid_A, fid_B):
    def func(k, mu):
        pl = np.interp(k, plin_k["k"], plin_k["pk"])
        return pkmu_cross(
            np.asarray(k), np.asarray(mu), pl,
            b1_A=b1_A, b1_B=b1_B, f=f,
            c0_A=fid_A.c0, c0_B=fid_B.c0,
            c2_A=fid_A.c2, c2_B=fid_B.c2,
            c4_A=fid_A.c4, c4_B=fid_B.c4,
            cfog_A=fid_A.c_tilde, cfog_B=fid_B.c_tilde,
        )
    return func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: ForecastConfig, verbose: bool = True) -> PipelineResults:
    """Execute the full forecast pipeline."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy2("configs/default.yaml", out_dir / "config_snapshot.yaml")

    if verbose:
        print(f"Output → {out_dir}")

    cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
    sp = default_survey_pair()
    sp.overlap_area_deg2 = cfg.overlap_area_deg2

    from ps_1loop_jax import PowerSpectrum1Loop
    ps = PowerSpectrum1Loop(do_irres=False)

    z_bins = cfg.z_bins
    ells = (0, 2, 4)

    # =================================================================
    # STEP 1: Overlap calibration
    # =================================================================
    if verbose:
        print("\n=== Step 1: Overlap calibration ===")

    # Asymmetric kmax: PFS can go to higher k than DESI
    kmax_desi = cfg.kmax_desi_overlap
    kmax_pfs = cfg.compute_kmax_pfs()
    kmax_cross = cfg.compute_kmax_cross()
    kmax_high = max(kmax_pfs, kmax_cross)

    k_low = np.arange(cfg.kmin, kmax_desi + cfg.dk / 2, cfg.dk)
    if kmax_high > kmax_desi + cfg.dk / 2:
        k_high = np.arange(kmax_desi + cfg.dk, kmax_high + cfg.dk / 2, cfg.dk)
    else:
        k_high = np.array([])
    k_all = np.concatenate([k_low, k_high]) if len(k_high) > 0 else k_low

    if verbose:
        if len(k_high) > 0:
            print(f"  Asymmetric kmax: DESI={kmax_desi:.3f}, PFS={kmax_pfs:.3f}, "
                  f"cross={kmax_cross:.3f} h/Mpc")
            print(f"  k_low: {len(k_low)} bins to {k_low[-1]:.3f}, "
                  f"k_high: {len(k_high)} bins to {k_high[-1]:.3f}")
        else:
            print(f"  Symmetric kmax={kmax_desi:.3f} h/Mpc")

    overlap_results: dict[tuple[float, float], OverlapResult] = {}
    fig1_params = ["c_tilde", "c0", "c2", "Pshot", "a0"]

    for zlo, zhi in z_bins:
        z_eff = sp.B.z_eff(zlo, zhi)
        s8_z = cosmo.sigma8(z_eff)
        f_z = float(cosmo.f(z_eff))
        h = cosmo.params["h"]

        nbar_pfs = sp.A.nbar_eff(zlo, zhi)
        nbar_desi = sp.B.nbar_eff(zlo, zhi)
        b1_pfs = sp.A.b1_of_z(z_eff)
        b1_desi = sp.B.b1_of_z(z_eff)
        V_ov = sp.V_overlap(zlo, zhi)

        fid_pfs = pfs_elg_fiducials(b1_pfs, b1_desi, s8_z, cfg.r_sigma_v)
        fid_desi = desi_elg_fiducials(b1_desi, s8_z)

        pk_data = cosmo.pk_data(z_eff)

        params_pfs = fisher_to_ps1loop_auto(fid_pfs, s8_z, f_z, h, nbar_pfs)
        params_desi = fisher_to_ps1loop_auto(fid_desi, s8_z, f_z, h, nbar_desi)
        params_cross = fisher_to_ps1loop_cross(fid_pfs, fid_desi, s8_z, f_z, h, nbar_pfs, nbar_desi)

        # Derivatives: AA and AB on k_all; BB on k_low only
        derivs_AA = dPell_dtheta_autodiff_all(ps, jnp.array(k_all), pk_data, params_pfs, NUISANCE_NAMES, s8_z, ells)
        derivs_BB = dPell_dtheta_autodiff_all(ps, jnp.array(k_low), pk_data, params_desi, NUISANCE_NAMES, s8_z, ells)
        derivs_AB_A, derivs_AB_B = {}, {}
        for nn in NUISANCE_NAMES:
            derivs_AB_A[nn], derivs_AB_B[nn] = {}, {}
            for ell in ells:
                derivs_AB_A[nn][ell] = dPcross_dtheta_autodiff(ps, jnp.array(k_all), pk_data, params_cross, nn, s8_z, "A", ell)
                derivs_AB_B[nn][ell] = dPcross_dtheta_autodiff(ps, jnp.array(k_all), pk_data, params_cross, nn, s8_z, "B", ell)

        # Covariances
        pkmu_AA = make_ps1loop_pkmu_func(ps, pk_data, params_pfs)
        pkmu_BB = make_ps1loop_pkmu_func(ps, pk_data, params_desi)
        pkmu_AB_f = make_ps1loop_pkmu_cross_func(ps, pk_data, params_cross)

        # Full 9×9 on k_low
        cov_mt_low = multi_tracer_cov(pkmu_AA, pkmu_BB, pkmu_AB_f, k_low, nbar_pfs, nbar_desi, V_ov, cfg.dk, ells)

        # 6×6 subblock (AA + AB) on k_high
        cov_mt_high_6x6 = None
        if len(k_high) > 0:
            cov_9x9_high = multi_tracer_cov(pkmu_AA, pkmu_BB, pkmu_AB_f, k_high, nbar_pfs, nbar_desi, V_ov, cfg.dk, ells)
            idx = np.array(_IDX_REDUCED)
            cov_mt_high_6x6 = cov_9x9_high[:, idx[:, None], idx[None, :]]

        # Multi-tracer Fisher (asymmetric kmax)
        fr_mt = multi_tracer_fisher_asymmetric(
            derivs_AA, derivs_BB, derivs_AB_A, derivs_AB_B,
            cov_mt_low, cov_mt_high_6x6, k_low,
            k_high if len(k_high) > 0 else None,
            cfg.dk, z_bin=(zlo, zhi), ells=ells,
        )
        cal = export_calibrated_priors(fr_mt, z_bin=(zlo, zhi))

        # Single-tracer Fishers in the overlap (for Fig 1 comparison)
        bp = broad_priors()
        bp_diag = bp.prior_fisher_diag()

        cov_pfs = single_tracer_cov(pkmu_AA, k_all, nbar_pfs, V_ov, cfg.dk, ells)
        fr_pfs = single_tracer_fisher(derivs_AA, cov_pfs, k_all, cfg.dk, NUISANCE_NAMES, (zlo, zhi), "PFS-only", kmax_pfs, bp_diag, ells)

        cov_desi_ov = single_tracer_cov(pkmu_BB, k_low, nbar_desi, V_ov, cfg.dk, ells)
        fr_desi_ov = single_tracer_fisher(derivs_BB, cov_desi_ov, k_low, cfg.dk, NUISANCE_NAMES, (zlo, zhi), "DESI-only", kmax_desi, bp_diag, ells)

        sigma_desi_only = {p: fr_desi_ov.marginalized_sigma(p) for p in fig1_params}
        sigma_pfs_only = {p: fr_pfs.marginalized_sigma(p) for p in fig1_params}
        sigma_mt_dict = {p: cal.params[p] for p in fig1_params}

        overlap_results[(zlo, zhi)] = OverlapResult(
            z_bin=(zlo, zhi), calibrated_priors=cal, fisher_mt=fr_mt,
            sigma_desi_only=sigma_desi_only, sigma_pfs_only=sigma_pfs_only, sigma_mt=sigma_mt_dict,
        )

        if verbose:
            print(f"  z=[{zlo},{zhi}]: σ_cal(c_tilde)={cal.params['c_tilde']:.1f}  σ_cal(Pshot)={cal.params['Pshot']:.3f}")

    # Save calibrated priors
    priors_dir = out_dir / "priors"
    priors_dir.mkdir(exist_ok=True)
    for (zlo, zhi), ov in overlap_results.items():
        jp = priors_dir / f"cross_calibrated_z{zlo:.1f}_{zhi:.1f}.json"
        with open(jp, "w") as f:
            json.dump({"z_bin": [zlo, zhi], "source": ov.calibrated_priors.source,
                        "priors": {k: {"sigma": v} for k, v in ov.calibrated_priors.params.items()}}, f, indent=2)

    # =================================================================
    # STEP 2: Full-area DESI Fisher per scenario
    # =================================================================
    if verbose:
        print("\n=== Step 2: Full-area DESI Fisher ===")

    summary_rows: list[SummaryRow] = []
    scenario_results: dict[str, ScenarioResult] = {}
    broad_sigmas: dict[str, float] = {}

    for scenario in SCENARIOS:
        if verbose:
            print(f"\n  Scenario: {scenario.name} (kmax={scenario.kmax})")
        k_full = np.arange(cfg.kmin, scenario.kmax + cfg.dk / 2, cfg.dk)

        fishers_per_z = []
        sigmas_per_z: dict[tuple[float, float], dict[str, float]] = {}

        for zlo, zhi in z_bins:
            z_eff = sp.B.z_eff(zlo, zhi)
            s8_z = cosmo.sigma8(z_eff)
            f_z = float(cosmo.f(z_eff))
            h = cosmo.params["h"]
            nbar_desi = sp.B.nbar_eff(zlo, zhi)
            b1_desi = sp.B.b1_of_z(z_eff)
            V_full = sp.V_full_B(zlo, zhi)
            fid_desi = desi_elg_fiducials(b1_desi, s8_z)

            pk_data = cosmo.pk_data(z_eff)
            pk_data_np = {"k": np.asarray(pk_data["k"]), "pk": np.asarray(pk_data["pk"])}
            params_desi = fisher_to_ps1loop_auto(fid_desi, s8_z, f_z, h, nbar_desi)

            derivs = dPell_dtheta_autodiff_all(ps, jnp.array(k_full), pk_data, params_desi, NUISANCE_NAMES, s8_z, ells)
            cosmo_derivs = dPell_d_cosmo_all(ps, jnp.array(k_full), pk_data, cosmo, params_desi, z_eff, s8_z, ells)
            derivs.update(cosmo_derivs)

            pkmu_func = make_ps1loop_pkmu_func(ps, pk_data, params_desi)
            cov_st = single_tracer_cov(pkmu_func, k_full, nbar_desi, V_full, cfg.dk, ells)

            cal = overlap_results[(zlo, zhi)].calibrated_priors
            nuis_prior = nuisance_prior_diag(scenario, cal)

            fr = full_area_fisher_per_zbin(derivs, cov_st, k_full, cfg.dk, nuis_prior, (zlo, zhi), scenario.kmax, ells)
            fishers_per_z.append(fr)

            # Per-z sigmas for cosmo params
            per_z_sig = {}
            for cp in COSMO_NAMES:
                per_z_sig[cp] = fr.marginalized_sigma(cp)
            sigmas_per_z[(zlo, zhi)] = per_z_sig

        fr_combined = combine_zbins(fishers_per_z, z_bins)
        sigmas_combined = {cp: fr_combined.marginalized_sigma(cp) for cp in COSMO_NAMES}

        scenario_results[scenario.name] = ScenarioResult(
            scenario=scenario, fisher_combined=fr_combined,
            fishers_per_z=fishers_per_z, sigmas_combined=sigmas_combined,
            sigmas_per_z=sigmas_per_z,
        )

        # Summary rows (combined)
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

        # Also per-z-bin rows
        for zlo, zhi in z_bins:
            for cp in COSMO_NAMES:
                sigma = sigmas_per_z[(zlo, zhi)][cp]
                sigma_broad = broad_sigmas.get(cp, sigma)
                sigma_oracle = broad_sigmas.get(f"{cp}_oracle", sigma)
                improvement = compute_improvement(sigma, sigma_broad)
                efficiency = compute_calibration_efficiency(sigma, sigma_broad, sigma_oracle)
                summary_rows.append(SummaryRow(
                    scenario=scenario.name, kmax=scenario.kmax,
                    z_bin_min=zlo, z_bin_max=zhi,
                    param_name=cp, sigma_marginalized=sigma,
                    sigma_broad_baseline=sigma_broad, improvement_pct=improvement,
                    calibration_efficiency=efficiency,
                ))

    csv_path = out_dir / "summary.csv"
    write_summary_csv(summary_rows, str(csv_path))
    if verbose:
        print(f"\nSummary written to {csv_path}")

    return PipelineResults(
        config=cfg, overlap_results=overlap_results,
        scenario_results=scenario_results, summary_rows=summary_rows,
        output_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="PFS×DESI Fisher forecast")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = ForecastConfig.from_yaml(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
