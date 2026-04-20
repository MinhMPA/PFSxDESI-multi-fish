"""Tests for pfsfog.covariance."""

import numpy as np
import pytest

from pfsfog.covariance import single_tracer_cov, multi_tracer_cov


def _toy_pkmu(k, mu):
    """Simple power-law P(k,μ) for testing."""
    k = np.atleast_1d(k)
    mu = np.atleast_1d(mu)
    return 1e4 * (k[:, None] / 0.1) ** (-1.5) * (1 + 0.5 * mu[None, :] ** 2)


class TestSingleTracerCov:
    def test_positive_definite(self):
        k = np.linspace(0.01, 0.25, 20)
        C = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e9, dk=0.01)
        for ik in range(len(k)):
            eigvals = np.linalg.eigvalsh(C[ik])
            assert np.all(eigvals > 0), f"not positive definite at k={k[ik]}"

    def test_diagonal_positive(self):
        k = np.linspace(0.01, 0.25, 20)
        C = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e9, dk=0.01)
        for i in range(3):
            assert np.all(C[:, i, i] > 0)

    def test_shape(self):
        k = np.linspace(0.01, 0.25, 20)
        C = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e9, dk=0.01)
        assert C.shape == (20, 3, 3)

    def test_symmetric(self):
        k = np.linspace(0.01, 0.25, 20)
        C = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e9, dk=0.01)
        for ik in range(len(k)):
            np.testing.assert_allclose(C[ik], C[ik].T, atol=1e-15)

    def test_larger_volume_smaller_cov(self):
        k = np.linspace(0.01, 0.25, 10)
        C1 = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e9, dk=0.01)
        C2 = single_tracer_cov(_toy_pkmu, k, nbar=1e-3, volume=1e10, dk=0.01)
        # larger volume → more modes → smaller variance
        assert np.all(C2[:, 0, 0] < C1[:, 0, 0])


class TestMultiTracerCov:
    def _pkmu_B(self, k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        return 8e3 * (k[:, None] / 0.1) ** (-1.5) * (1 + 0.3 * mu[None, :] ** 2)

    def _pkmu_AB(self, k, mu):
        """Cross-power must satisfy |P^{AB}|² ≤ P^{AA} × P^{BB}."""
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        # geometric mean of AA and BB, scaled down for safety
        return 0.5 * np.sqrt(
            _toy_pkmu(k, mu) * self._pkmu_B(k, mu)
        )

    def test_shape(self):
        k = np.linspace(0.01, 0.25, 15)
        C = multi_tracer_cov(
            _toy_pkmu, self._pkmu_B, self._pkmu_AB,
            k, nbar_A=1e-3, nbar_B=5e-4, volume=1e9, dk=0.01,
        )
        assert C.shape == (15, 9, 9)

    def test_positive_definite(self):
        k = np.linspace(0.01, 0.25, 15)
        C = multi_tracer_cov(
            _toy_pkmu, self._pkmu_B, self._pkmu_AB,
            k, nbar_A=1e-3, nbar_B=5e-4, volume=1e9, dk=0.01,
        )
        for ik in range(len(k)):
            eigvals = np.linalg.eigvalsh(C[ik])
            assert np.all(eigvals > 0), f"not positive definite at k={k[ik]}"

    def test_symmetric(self):
        k = np.linspace(0.01, 0.25, 15)
        C = multi_tracer_cov(
            _toy_pkmu, self._pkmu_B, self._pkmu_AB,
            k, nbar_A=1e-3, nbar_B=5e-4, volume=1e9, dk=0.01,
        )
        for ik in range(len(k)):
            np.testing.assert_allclose(C[ik], C[ik].T, atol=1e-12)
