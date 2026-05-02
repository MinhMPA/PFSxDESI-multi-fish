"""Analysis scenarios and summary output.

LEGACY — defines the "broad / cross-cal / oracle" scenario labels used
by the two-stage pipeline. The joint Fisher
(``pfsfog.fisher_joint``) reports its results directly with
"DESI-only joint" / "DESI+PFS joint" labels and does not use this
module. Kept for the legacy ``run_desi_multisample.py`` driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .eft_params import (
    NUISANCE_NAMES, COSMO_NAMES,
    broad_priors, HOD_BENCHMARK, FIELD_LEVEL_BENCHMARK,
)
from .prior_export import CalibratedPriors, calibrated_prior_fisher_diag


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    prior_source: str      # "broad", "cross-cal", "oracle"
    kmax: float            # h/Mpc


SCENARIOS = [
    Scenario("broad",          "broad",     0.20),
    Scenario("cross-cal",      "cross-cal", 0.20),
    Scenario("oracle",         "oracle",    0.20),
]


def nuisance_prior_diag(
    scenario: Scenario,
    calibrated: CalibratedPriors | None = None,
) -> np.ndarray:
    """Return the nuisance-prior Fisher diagonal for a scenario.

    Parameters
    ----------
    scenario : Scenario
    calibrated : CalibratedPriors, required for cross-cal scenarios

    Returns
    -------
    diag : array (N_NUIS,) — 1/σ² per nuisance parameter
    """
    if scenario.prior_source == "broad":
        return broad_priors().prior_fisher_diag()

    if scenario.prior_source == "cross-cal":
        if calibrated is None:
            raise ValueError("cross-cal scenario requires CalibratedPriors")
        return calibrated_prior_fisher_diag(calibrated)

    if scenario.prior_source == "oracle":
        # Fix nuisance params: effectively infinite Fisher → 1/ε²
        eps = 1e-10
        return np.full(len(NUISANCE_NAMES), 1.0 / eps**2)

    raise ValueError(f"Unknown prior_source: {scenario.prior_source}")


# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------


@dataclass
class SummaryRow:
    scenario: str
    kmax: float
    z_bin_min: float
    z_bin_max: float
    param_name: str
    sigma_marginalized: float
    sigma_broad_baseline: float
    improvement_pct: float
    calibration_efficiency: float | None


def compute_improvement(sigma: float, sigma_broad: float) -> float:
    """Improvement percentage: (σ_broad − σ) / σ_broad × 100."""
    if sigma_broad <= 0:
        return 0.0
    return (sigma_broad - sigma) / sigma_broad * 100.0


def compute_calibration_efficiency(
    sigma: float,
    sigma_broad: float,
    sigma_oracle: float,
) -> float | None:
    """(σ_broad − σ) / (σ_broad − σ_oracle).  Returns None if denom ≈ 0."""
    denom = sigma_broad - sigma_oracle
    if abs(denom) < 1e-15:
        return None
    eff = (sigma_broad - sigma) / denom
    return float(eff)


# ---------------------------------------------------------------------------
# Summary CSV writer
# ---------------------------------------------------------------------------


def write_summary_csv(rows: list[SummaryRow], path: str) -> None:
    """Write summary.csv."""
    header = (
        "scenario,kmax,z_bin_min,z_bin_max,param_name,"
        "sigma_marginalized,sigma_broad_baseline,improvement_pct,"
        "calibration_efficiency\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for r in rows:
            eff_str = f"{r.calibration_efficiency:.4f}" if r.calibration_efficiency is not None else ""
            f.write(
                f"{r.scenario},{r.kmax:.2f},{r.z_bin_min:.1f},{r.z_bin_max:.1f},"
                f"{r.param_name},{r.sigma_marginalized:.6e},"
                f"{r.sigma_broad_baseline:.6e},{r.improvement_pct:.2f},"
                f"{eff_str}\n"
            )
