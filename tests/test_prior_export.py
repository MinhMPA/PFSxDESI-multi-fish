"""Tests for pfsfog.prior_export."""

import numpy as np
import pytest

from pfsfog.prior_export import (
    export_calibrated_priors,
    calibrated_prior_fisher_diag,
    _build_broad_prior_diag,
)
from pfsfog.fisher import FisherResult
from pfsfog.fisher_mt import mt_param_names
from pfsfog.eft_params import NUISANCE_NAMES, broad_priors


class TestBroadPriorDiag:
    def test_length(self):
        diag = _build_broad_prior_diag()
        assert len(diag) == 27

    def test_cosmo_entries_positive(self):
        diag = _build_broad_prior_diag()
        # First 3 are cosmo
        assert np.all(diag[:3] > 0)

    def test_flat_prior_is_zero(self):
        diag = _build_broad_prior_diag()
        names = mt_param_names()
        # b1_sigma8 has flat prior → 0
        idx_b1_pfs = names.index("b1_sigma8_PFS")
        assert diag[idx_b1_pfs] == 0.0


class TestExportCalibratedPriors:
    def test_basic(self):
        """With a non-trivial Fisher, exported σ should be positive and
        tighter than broad priors for at least some parameters."""
        names = mt_param_names()
        N = len(names)
        # Create a Fisher matrix with some structure
        np.random.seed(42)
        A = np.random.randn(N, N) * 0.1
        F = A.T @ A + np.eye(N) * 10  # positive definite

        fr = FisherResult(F=F, param_names=names,
                          z_bin=(0.8, 1.0), survey_name="test", kmax=0.2)

        cal = export_calibrated_priors(fr, z_bin=(0.8, 1.0))

        # All σ_cal should be positive
        for name, sigma in cal.params.items():
            assert sigma > 0, f"σ_cal({name}) = {sigma} ≤ 0"

    def test_tighter_than_broad(self):
        """Calibrated priors should be ≤ broad priors (data adds info)."""
        names = mt_param_names()
        N = len(names)
        np.random.seed(123)
        A = np.random.randn(N, N) * 0.5
        F = A.T @ A + np.eye(N) * 100

        fr = FisherResult(F=F, param_names=names,
                          z_bin=(0.8, 1.0), survey_name="test", kmax=0.2)
        cal = export_calibrated_priors(fr, z_bin=(0.8, 1.0))

        bp = broad_priors()
        bp_dict = bp.sigma_dict()

        # Check that calibrated σ ≤ broad σ for params with Gaussian priors
        for name in NUISANCE_NAMES:
            sigma_broad = bp_dict[name]
            if sigma_broad is None:
                continue  # flat prior, skip
            sigma_cal = cal.params[name]
            assert sigma_cal <= sigma_broad + 1e-10, (
                f"σ_cal({name})={sigma_cal:.4f} > σ_broad={sigma_broad:.4f}"
            )


class TestCalibratedPriorFisherDiag:
    def test_shape(self):
        from pfsfog.prior_export import CalibratedPriors
        cal = CalibratedPriors(
            params={n: 1.0 for n in NUISANCE_NAMES},
            z_bin=(0.8, 1.0),
        )
        diag = calibrated_prior_fisher_diag(cal)
        assert diag.shape == (len(NUISANCE_NAMES),)
        assert np.all(diag == 1.0)  # 1/1² = 1
