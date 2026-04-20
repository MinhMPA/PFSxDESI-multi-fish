"""Tests for pfsfog.fisher — including analytic Gaussian regression."""

import numpy as np
import pytest

from pfsfog.fisher import fisher_matrix, add_gaussian_prior, single_tracer_fisher, FisherResult
from pfsfog.covariance import single_tracer_cov


class TestFisherResult:
    def test_marginalized_sigma(self):
        F = np.array([[4.0, 1.0], [1.0, 2.0]])
        fr = FisherResult(F=F, param_names=["a", "b"],
                          z_bin=(0.8, 1.0), survey_name="test", kmax=0.2)
        Finv = np.linalg.inv(F)
        assert abs(fr.marginalized_sigma("a") - np.sqrt(Finv[0, 0])) < 1e-10

    def test_conditional_sigma(self):
        F = np.array([[4.0, 1.0], [1.0, 2.0]])
        fr = FisherResult(F=F, param_names=["a", "b"],
                          z_bin=(0.8, 1.0), survey_name="test", kmax=0.2)
        # Conditional on b fixed: σ(a) = 1/√F_aa
        assert abs(fr.conditional_sigma("a") - 1.0 / np.sqrt(4.0)) < 1e-10


class TestAddPrior:
    def test_adds_diagonal(self):
        F = np.eye(3)
        prior = np.array([1.0, 2.0, 3.0])
        Fp = add_gaussian_prior(F, prior)
        np.testing.assert_allclose(np.diag(Fp), [2.0, 3.0, 4.0])


class TestAnalyticFisher:
    """Analytic test: monopole-only Fisher for P_0(k) = A × P_shape(k).

    Model: P_0(k) = A × k^n  (two parameters: A, n)
    Cov_00(k) = [P_0(k) + 1/nbar]² / N_modes(k)

    The Fisher matrix for {A, n} from the monopole alone is:

        F_AA = Σ_k [∂P_0/∂A]² / Var(P_0)
        F_An = Σ_k [∂P_0/∂A][∂P_0/∂n] / Var(P_0)
        F_nn = Σ_k [∂P_0/∂n]² / Var(P_0)

    where ∂P_0/∂A = k^n, ∂P_0/∂n = A k^n ln(k).
    """

    def test_monopole_only(self):
        A_fid = 1e4
        n_fid = -1.5
        nbar = 1e-3
        V = 1e9
        dk = 0.01
        k = np.arange(0.01, 0.25, dk)
        Nk = len(k)

        # Fiducial P_0(k)
        P0 = A_fid * k ** n_fid

        # Gaussian variance for monopole (ℓ=0 only)
        Nmodes = k**2 * dk * V / (2 * np.pi**2)
        Var_P0 = (P0 + 1.0 / nbar) ** 2 / Nmodes

        # Analytic derivatives
        dP_dA = k ** n_fid
        dP_dn = A_fid * k ** n_fid * np.log(k)

        # Analytic Fisher
        F_analytic = np.zeros((2, 2))
        F_analytic[0, 0] = np.sum(dP_dA**2 / Var_P0 * dk)
        F_analytic[0, 1] = np.sum(dP_dA * dP_dn / Var_P0 * dk)
        F_analytic[1, 0] = F_analytic[0, 1]
        F_analytic[1, 1] = np.sum(dP_dn**2 / Var_P0 * dk)

        # Now compute via our machinery
        # Derivatives dict
        derivs = {
            "A": {0: dP_dA},
            "n": {0: dP_dn},
        }

        # Covariance: monopole only → shape (Nk, 1, 1)
        def pkmu_func(kk, mu):
            kk = np.atleast_1d(kk)
            mu = np.atleast_1d(mu)
            return A_fid * kk[:, None] ** n_fid * np.ones_like(mu)[None, :]

        cov = single_tracer_cov(pkmu_func, k, nbar, V, dk, ells=(0,))
        assert cov.shape == (Nk, 1, 1)

        # Invert
        cov_inv = np.zeros_like(cov)
        for ik in range(Nk):
            cov_inv[ik, 0, 0] = 1.0 / cov[ik, 0, 0]

        F_numerical = fisher_matrix(
            derivs, cov_inv, k, volume=V, dk=dk,
            param_names=["A", "n"], ells=(0,),
        )

        # They should agree to ~1% (trapezoidal vs sum difference)
        np.testing.assert_allclose(F_numerical, F_analytic, rtol=0.01)

    def test_prior_tightens(self):
        """Adding a prior should decrease marginalized σ."""
        F = np.array([[10.0, 2.0], [2.0, 5.0]])
        fr1 = FisherResult(F=F, param_names=["a", "b"],
                           z_bin=(0.8, 1.0), survey_name="t", kmax=0.2)
        sigma_no_prior = fr1.marginalized_sigma("a")

        F2 = add_gaussian_prior(F, np.array([1.0, 1.0]))
        fr2 = FisherResult(F=F2, param_names=["a", "b"],
                           z_bin=(0.8, 1.0), survey_name="t", kmax=0.2)
        sigma_with_prior = fr2.marginalized_sigma("a")

        assert sigma_with_prior < sigma_no_prior
