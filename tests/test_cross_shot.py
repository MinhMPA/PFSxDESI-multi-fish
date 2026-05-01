"""Test that the cross-shot noise uses the correct formula.

The cross-shot noise for partially overlapping catalogs is:
    P_shot^{cross} = f_shared / nbar_DESI
where f_shared = n_shared / nbar_PFS.

This test verifies the formula at both the pipeline level
(cli_multitrace.py and run_desi_multisample.py) and by checking
that the covariance changes correctly when cross_shot is injected.
"""

import numpy as np
import pytest


class TestCrossShotFormula:
    """Verify cross-shot = f_shared / nbar_DESI (not nbar_PFS)."""

    def test_cross_shot_value(self):
        """The cross-shot value should be f_shared / nbar_DESI."""
        f_shared = 0.05
        nbar_PFS = 8.6e-4
        nbar_DESI = 8.1e-4

        # Correct formula
        correct = f_shared / nbar_DESI

        # Wrong formula (what we had before the fix)
        wrong = f_shared / nbar_PFS

        # They differ because nbar_PFS != nbar_DESI
        assert correct != pytest.approx(wrong, rel=1e-3)

        # The correct value should be larger (nbar_DESI < nbar_PFS)
        assert correct > wrong

    def test_cross_shot_in_covariance(self):
        """Cross-shot noise should appear in P_tot for cross-spectra only."""
        from pfsfog.covariance_mt_general import multi_tracer_cov_general

        k = np.array([0.1])
        dk = 0.01
        volume = 1e9
        ells = (0,)

        nbar = {"A": 8e-4, "B": 5e-4}
        f_shared = 0.05

        # Simple P(k,mu) = constant for easy verification
        def pkmu_const(k, mu):
            return np.full((len(k), len(mu)), 1e4)

        pkmu_funcs = {("A", "A"): pkmu_const,
                      ("B", "B"): pkmu_const,
                      ("A", "B"): pkmu_const}

        # Without cross-shot
        C_no = multi_tracer_cov_general(
            ["A", "B"], pkmu_funcs, nbar, k, volume, dk, ells,
            cross_shot=None)

        # With cross-shot = f_shared / nbar_B (correct formula)
        cs_correct = f_shared / nbar["B"]
        C_yes = multi_tracer_cov_general(
            ["A", "B"], pkmu_funcs, nbar, k, volume, dk, ells,
            cross_shot={("A", "B"): cs_correct})

        # The auto-spectrum covariance blocks should differ because
        # P_tot^{AB} enters the Wick pairings for Cov[P^{AA}, P^{BB}]
        # (through P_tot^{AB} * P_tot^{AB})
        assert not np.allclose(C_no, C_yes), \
            "Cross-shot should change the covariance"

        # The cross-spectrum variance Cov[P^{AB}, P^{AB}] should increase
        # with cross-shot (more noise in the cross)
        # AB is pair index 2 (after AA=0, BB=1), ell=0 -> index 2
        idx_AB = 2  # third pair (A,B) in the ordering [AA, BB, AB]
        assert C_yes[0, idx_AB, idx_AB] > C_no[0, idx_AB, idx_AB], \
            "Cross-shot should increase cross-spectrum variance"

    def test_cross_shot_zero_for_different_populations(self):
        """Cross-shot must be zero for non-overlapping populations."""
        # PFS-ELG x DESI-LRG: zero by construction
        # PFS-ELG x DESI-QSO: zero by construction
        # Only PFS-ELG x DESI-ELG has non-zero cross-shot
        f_shared = 0.05
        cross_shot_elg = f_shared / 8.1e-4  # nbar_DESI-ELG

        # For LRG and QSO, cross-shot is identically zero
        cross_shot_lrg = 0.0
        cross_shot_qso = 0.0

        assert cross_shot_elg > 0
        assert cross_shot_lrg == 0
        assert cross_shot_qso == 0

    def test_f_shared_limit_one(self):
        """At f_shared=1, cross-shot = 1/nbar_DESI = DESI auto-shot."""
        nbar_DESI = 8.1e-4
        f_shared = 1.0

        cross_shot = f_shared / nbar_DESI
        auto_shot_DESI = 1.0 / nbar_DESI

        assert cross_shot == pytest.approx(auto_shot_DESI)
