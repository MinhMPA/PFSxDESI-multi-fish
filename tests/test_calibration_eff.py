"""Tests for calibration efficiency ∈ [0, 1] from pipeline results."""

import csv
from pathlib import Path

import pytest

from pfsfog.scenarios import compute_calibration_efficiency


class TestEfficiencyBounds:
    """Calibration efficiency should be in [0, 1] for well-behaved results."""

    def test_range(self):
        # Simulate results from the pipeline
        sigma_broad = 0.075
        sigma_oracle = 0.010
        for sigma in [0.075, 0.060, 0.050, 0.040, 0.020, 0.010]:
            eff = compute_calibration_efficiency(sigma, sigma_broad, sigma_oracle)
            if eff is not None:
                assert 0.0 - 1e-10 <= eff <= 1.0 + 1e-10, (
                    f"efficiency {eff} out of range for σ={sigma}"
                )

    def test_monotonicity(self):
        """Tighter σ → higher efficiency."""
        sigma_broad = 0.10
        sigma_oracle = 0.01
        eff1 = compute_calibration_efficiency(0.08, sigma_broad, sigma_oracle)
        eff2 = compute_calibration_efficiency(0.05, sigma_broad, sigma_oracle)
        assert eff2 > eff1
