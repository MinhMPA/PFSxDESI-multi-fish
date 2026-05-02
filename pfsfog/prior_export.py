"""Export calibrated priors from the overlap multi-tracer Fisher.

LEGACY — used only by the two-stage pipeline (Step 1 overlap calibration
→ Step 2 single-tracer cosmology with calibrated Gaussian priors).
The joint Fisher in ``pfsfog/fisher_joint.py`` replaces this approach:
all data and nuisance parameters live in one Fisher matrix, marginalized
in a single pass, so no prior export step is needed. Kept for
reproducibility of the original two-stage results.

Steps:
1. Add broad Gaussian priors to regularize F_MT.
2. Invert → covariance.
3. Extract marginalized σ for each DESI nuisance parameter.
4. Package as CalibratedPriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .eft_params import NUISANCE_NAMES, COSMO_NAMES, COSMO_PRIOR_SIGMA, broad_priors
from .fisher import FisherResult, add_gaussian_prior
from .fisher_mt import _N_COSMO, _N_NUIS, mt_param_names


@dataclass
class CalibratedPriors:
    """Calibrated prior widths for DESI nuisance parameters from the overlap."""

    params: dict[str, float]          # {param_name: σ_calibrated}
    z_bin: tuple[float, float]
    source: str = "PFS×DESI overlap"


def _build_broad_prior_diag() -> np.ndarray:
    """27-element diagonal: 1/σ² for broad priors on all params."""
    names = mt_param_names()
    diag = np.zeros(len(names))

    bp = broad_priors()
    bp_diag = bp.prior_fisher_diag()  # (12,) for nuisance

    for i, name in enumerate(names):
        if name in COSMO_PRIOR_SIGMA:
            diag[i] = 1.0 / COSMO_PRIOR_SIGMA[name] ** 2
        elif name.endswith("_PFS"):
            base = name.removesuffix("_PFS")
            idx = NUISANCE_NAMES.index(base)
            diag[i] = bp_diag[idx]
        elif name.endswith("_DESI"):
            base = name.removesuffix("_DESI")
            idx = NUISANCE_NAMES.index(base)
            diag[i] = bp_diag[idx]

    return diag


def export_calibrated_priors(
    fisher_mt: FisherResult,
    z_bin: tuple[float, float],
) -> CalibratedPriors:
    """Extract calibrated DESI nuisance priors from the overlap Fisher.

    Parameters
    ----------
    fisher_mt : FisherResult
        27×27 multi-tracer Fisher from ``multi_tracer_fisher()``.
    z_bin : (zlo, zhi)

    Returns
    -------
    CalibratedPriors
    """
    # Step 1: regularize with broad priors
    prior_diag = _build_broad_prior_diag()
    F_reg = add_gaussian_prior(fisher_mt.F, prior_diag)

    # Step 2: invert
    C = np.linalg.inv(F_reg)

    # Step 3: extract σ_cal for DESI nuisance params
    names = mt_param_names()
    calibrated = {}
    for ip, nuis_name in enumerate(NUISANCE_NAMES):
        full_name = f"{nuis_name}_DESI"
        idx = names.index(full_name)
        sigma_cal = float(np.sqrt(C[idx, idx]))
        calibrated[nuis_name] = sigma_cal

    return CalibratedPriors(
        params=calibrated,
        z_bin=z_bin,
        source="PFS×DESI overlap",
    )


def calibrated_prior_fisher_diag(
    cal_priors: CalibratedPriors,
) -> np.ndarray:
    """Convert CalibratedPriors to a diagonal Fisher contribution.

    Returns array of shape ``(N_NUIS,)`` with ``1/σ_cal²`` per
    DESI nuisance parameter.
    """
    diag = np.zeros(len(NUISANCE_NAMES))
    for i, name in enumerate(NUISANCE_NAMES):
        sigma = cal_priors.params.get(name)
        if sigma is not None and sigma > 0:
            diag[i] = 1.0 / sigma**2
    return diag
