"""Tests for pfsfog.scenarios and end-to-end scenario ordering.

Key property: σ_oracle ≤ σ_cross-cal-ext ≤ σ_cross-cal ≤ σ_broad
"""

import numpy as np
import pytest

from pfsfog.scenarios import (
    Scenario, SCENARIOS, nuisance_prior_diag,
    compute_improvement, compute_calibration_efficiency,
    SummaryRow, write_summary_csv,
)
from pfsfog.eft_params import NUISANCE_NAMES, broad_priors
from pfsfog.prior_export import CalibratedPriors


class TestScenarioDefinitions:
    def test_four_scenarios(self):
        assert len(SCENARIOS) == 4

    def test_names(self):
        names = [s.name for s in SCENARIOS]
        assert "broad" in names
        assert "cross-cal" in names
        assert "cross-cal-ext" in names
        assert "oracle" in names


class TestNuisancePriorDiag:
    def test_broad(self):
        s = Scenario("broad", "broad", 0.20)
        diag = nuisance_prior_diag(s)
        assert diag.shape == (len(NUISANCE_NAMES),)
        bp = broad_priors().prior_fisher_diag()
        np.testing.assert_allclose(diag, bp)

    def test_oracle(self):
        s = Scenario("oracle", "oracle", 0.25)
        diag = nuisance_prior_diag(s)
        assert np.all(diag > 1e10)  # very tight

    def test_cross_cal_requires_calibrated(self):
        s = Scenario("cross-cal", "cross-cal", 0.20)
        with pytest.raises(ValueError):
            nuisance_prior_diag(s, calibrated=None)

    def test_cross_cal_tighter_than_broad(self):
        """Calibrated priors should be at least as tight as broad."""
        cal = CalibratedPriors(
            params={n: 0.5 for n in NUISANCE_NAMES},
            z_bin=(0.8, 1.0),
        )
        s = Scenario("cross-cal", "cross-cal", 0.20)
        diag_cal = nuisance_prior_diag(s, cal)
        diag_broad = nuisance_prior_diag(Scenario("broad", "broad", 0.20))
        # 1/0.5² = 4.0 for each param in cal
        # For params with broad σ < 0.5, broad is already tighter.
        # For c_tilde (σ_broad=400), cal σ=0.5 is much tighter.
        idx_ctilde = NUISANCE_NAMES.index("c_tilde")
        assert diag_cal[idx_ctilde] > diag_broad[idx_ctilde]


class TestImprovement:
    def test_no_improvement(self):
        assert compute_improvement(1.0, 1.0) == 0.0

    def test_full_improvement(self):
        assert abs(compute_improvement(0.0, 1.0) - 100.0) < 1e-10

    def test_half_improvement(self):
        assert abs(compute_improvement(0.5, 1.0) - 50.0) < 1e-10


class TestCalibrationEfficiency:
    def test_perfect(self):
        eff = compute_calibration_efficiency(0.0, 1.0, 0.0)
        assert abs(eff - 1.0) < 1e-10

    def test_no_improvement(self):
        eff = compute_calibration_efficiency(1.0, 1.0, 0.0)
        assert abs(eff - 0.0) < 1e-10

    def test_half(self):
        eff = compute_calibration_efficiency(0.5, 1.0, 0.0)
        assert abs(eff - 0.5) < 1e-10

    def test_degenerate_returns_none(self):
        eff = compute_calibration_efficiency(0.5, 1.0, 1.0)
        assert eff is None


class TestSummaryCSV:
    def test_write_read(self, tmp_path):
        rows = [
            SummaryRow("broad", 0.20, 0.8, 1.6, "fsigma8",
                       0.05, 0.05, 0.0, None),
            SummaryRow("cross-cal", 0.20, 0.8, 1.6, "fsigma8",
                       0.04, 0.05, 20.0, 0.5),
        ]
        path = str(tmp_path / "test.csv")
        write_summary_csv(rows, path)

        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
        assert data[0]["scenario"] == "broad"
        assert float(data[1]["improvement_pct"]) == 20.0
