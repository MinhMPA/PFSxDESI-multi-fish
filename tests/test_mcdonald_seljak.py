"""Test: McDonald & Seljak (2009, arXiv:0810.0323) cosmic-variance-free limit.

Verify that the multi-tracer Fisher for P_m (with biases marginalized)
achieves the cosmic-variance-free limit σ²(P_m)/P²_m = 2/N_modes in the
noiseless limit, reproducing the core result of their Eq. (14).

Setup: two tracers with linear Kaiser P(k,μ) = (b + fμ²)² P_m + 1/nbar.
No EFT counterterms, no stochastic departures — pure linear bias + shot noise.

Two test suites:
  1. μ-mode Fisher (analytic covariance per μ) — numerically stable to nbar ~ 1e6.
     Marginalizes over unknown biases using Schur complement.
  2. Multipole Fisher (using our actual covariance code) — cross-checks the
     pipeline at moderate nbar.
"""

import numpy as np
import pytest

from numpy.polynomial.legendre import leggauss

from pfsfog.covariance_mt_general import multi_tracer_cov_general
from pfsfog.covariance import single_tracer_cov, _legendre


# ---------------------------------------------------------------------------
# Kaiser model P(k,mu) callables (for the multipole tests)
# ---------------------------------------------------------------------------

def make_kaiser_pkmu(b, f, Pm):
    """Return a P(k,mu) callable for the Kaiser model (no counterterms)."""
    def pkmu(k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        return (b + f * mu[None, :] ** 2) ** 2 * Pm * np.ones_like(k)[:, None]
    return pkmu


def make_kaiser_cross_pkmu(bA, bB, f, Pm):
    """Return P^{AB}(k,mu) for the Kaiser cross-spectrum."""
    def pkmu(k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        return ((bA + f * mu[None, :] ** 2)
                * (bB + f * mu[None, :] ** 2)
                * Pm * np.ones_like(k)[:, None])
    return pkmu


# ---------------------------------------------------------------------------
# μ-mode Fisher with bias marginalization
# ---------------------------------------------------------------------------

def _mu_mode_fisher_st(bA, f, Pm, nbar, Nmodes, mu_gl, w_gl):
    """Single-tracer 2×2 Fisher for (P_m, b_A) in μ-mode basis.

    Returns marginalized σ²(P_m) via Schur complement.
    """
    F = np.zeros((2, 2))
    for mi, wi in zip(mu_gl, w_gl):
        sA = bA + f * mi**2
        PAA = sA**2 * Pm + 1.0 / nbar
        var_PAA = 2.0 * PAA**2  # Gaussian variance per mode
        D = np.array([sA**2, 2.0 * sA * Pm])
        F += wi * np.outer(D, D) / var_PAA
    F *= Nmodes / 2.0

    # Schur complement: σ²(Pm) = 1 / (F_00 - F_01² / F_11)
    schur = F[0, 0] - F[0, 1] ** 2 / F[1, 1]
    return 1.0 / schur


def _mu_mode_fisher_mt(bA, bB, f, Pm, nbar, Nmodes, mu_gl, w_gl):
    """Multi-tracer 3×3 Fisher for (P_m, b_A, b_B) in μ-mode basis.

    Returns marginalized σ²(P_m) via Schur complement.
    """
    F = np.zeros((3, 3))
    for mi, wi in zip(mu_gl, w_gl):
        sA = bA + f * mi**2
        sB = bB + f * mi**2
        PAA = sA**2 * Pm + 1.0 / nbar
        PBB = sB**2 * Pm + 1.0 / nbar
        PAB = sA * sB * Pm  # no shot noise for cross

        C3 = np.array([
            [2 * PAA**2,       2 * PAB**2,          2 * PAA * PAB],
            [2 * PAB**2,       2 * PBB**2,          2 * PBB * PAB],
            [2 * PAA * PAB,    2 * PBB * PAB,       PAA * PBB + PAB**2],
        ])

        D = np.array([
            [sA**2,    2 * sA * Pm,   0.0],
            [sB**2,    0.0,           2 * sB * Pm],
            [sA * sB,  sB * Pm,       sA * Pm],
        ])

        C3_inv = np.linalg.inv(C3)
        F += wi * D.T @ C3_inv @ D
    F *= Nmodes / 2.0

    # Schur complement: marginalize (bA, bB)
    F_bias = F[1:, 1:]
    F_cross = F[0, 1:]
    schur = F[0, 0] - F_cross @ np.linalg.solve(F_bias, F_cross)
    return 1.0 / schur


# ---------------------------------------------------------------------------
# Tests: μ-mode basis (numerically stable, direct MS09 comparison)
# ---------------------------------------------------------------------------

class TestMcDonaldSeljakMuMode:
    """Cosmic-variance-free limit in the μ-mode Fisher basis."""

    bA = 1.0
    bB = 2.0
    f = 0.8
    Pm = 1.0       # normalized amplitude
    beta = f / bA   # = 0.8
    k = 0.1
    dk = 0.01
    V = 1e9
    Nmodes = k**2 * dk * V / (2 * np.pi**2)
    mu_gl, w_gl = leggauss(40)

    def test_mt_always_beats_st(self):
        """σ²_MT < σ²_ST at all nbar values."""
        for nbar in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
            s2_st = _mu_mode_fisher_st(
                self.bA, self.f, self.Pm, nbar,
                self.Nmodes, self.mu_gl, self.w_gl)
            s2_mt = _mu_mode_fisher_mt(
                self.bA, self.bB, self.f, self.Pm, nbar,
                self.Nmodes, self.mu_gl, self.w_gl)
            assert s2_mt < s2_st * 1.01, (
                f"MT not better than ST at nbar={nbar:.0e}: "
                f"σ²_MT={s2_mt:.3e}, σ²_ST={s2_st:.3e}"
            )

    def test_mt_advantage_grows_with_nbar(self):
        """The MT/ST ratio should decrease as nbar increases (more gain)."""
        nbars = [1e0, 1e2, 1e4, 1e6]
        ratios = []
        for nbar in nbars:
            s2_st = _mu_mode_fisher_st(
                self.bA, self.f, self.Pm, nbar,
                self.Nmodes, self.mu_gl, self.w_gl)
            s2_mt = _mu_mode_fisher_mt(
                self.bA, self.bB, self.f, self.Pm, nbar,
                self.Nmodes, self.mu_gl, self.w_gl)
            ratios.append(s2_mt / s2_st)

        # Each ratio should be less than or equal to the previous one
        for i in range(1, len(ratios)):
            assert ratios[i] <= ratios[i - 1] * 1.01, (
                f"MT advantage did not grow: ratio[{i}]={ratios[i]:.4f} > "
                f"ratio[{i-1}]={ratios[i-1]:.4f}"
            )

    def test_mt_reaches_cosmic_variance_free_floor(self):
        """In the noiseless limit, σ²_MT(P_m)/P²_m → 2/N_modes (Eq. 14).

        This is the central result of McDonald & Seljak (2009): with two
        tracers of different bias, the error on P_m marginalized over biases
        equals the Poisson counting error, with no cosmic-variance contribution.
        """
        nbar = 1e6  # noiseless limit (stable up to ~1e6)
        s2_mt = _mu_mode_fisher_mt(
            self.bA, self.bB, self.f, self.Pm, nbar,
            self.Nmodes, self.mu_gl, self.w_gl)

        cv_floor = 2.0 * self.Pm**2 / self.Nmodes

        # MT should reach the cosmic-variance-free floor to <1%
        ratio = s2_mt / cv_floor
        assert 0.99 < ratio < 1.01, (
            f"MT did not reach CV-free floor: σ²_MT/σ²_floor = {ratio:.6f}"
        )

    def test_st_limited_by_cosmic_variance(self):
        """Single-tracer σ²(P_m) should be >> 2/N_modes due to b-f degeneracy."""
        nbar = 1e6
        s2_st = _mu_mode_fisher_st(
            self.bA, self.f, self.Pm, nbar,
            self.Nmodes, self.mu_gl, self.w_gl)

        cv_floor = 2.0 * self.Pm**2 / self.Nmodes

        # ST error should be much larger than CV floor
        assert s2_st > cv_floor * 5, (
            f"ST not CV-limited: σ²_ST={s2_st:.3e}, floor={cv_floor:.3e}"
        )


# ---------------------------------------------------------------------------
# Tests: multipole basis (cross-check of actual covariance code)
# ---------------------------------------------------------------------------

class TestMcDonaldSeljakMultipole:
    """Cross-check using our actual multipole covariance code."""

    bA = 1.0
    bB = 2.0
    f = 0.8
    Pm = 1000.0     # realistic amplitude
    k = np.array([0.1])
    V = 1e9
    dk = 0.01
    ells = (0, 2, 4)

    def _multipole_derivs(self, bX, bY):
        """dP_ell^{XY}/dP_m for ell=0,2,4."""
        mu_gl, w_gl = leggauss(20)
        derivs = []
        for ell in self.ells:
            Ll = _legendre(ell, mu_gl)
            integrand = (bX + self.f * mu_gl**2) * (bY + self.f * mu_gl**2) * Ll * w_gl
            derivs.append((2 * ell + 1) * np.sum(integrand))
        return np.array(derivs)

    def _mt_fisher_pm(self, nbar):
        """MT Fisher for P_m (known biases) using multipole covariance."""
        tracer_names = ["A", "B"]
        pkmu_funcs = {
            ("A", "A"): make_kaiser_pkmu(self.bA, self.f, self.Pm),
            ("A", "B"): make_kaiser_cross_pkmu(self.bA, self.bB, self.f, self.Pm),
            ("B", "B"): make_kaiser_pkmu(self.bB, self.f, self.Pm),
        }
        nbars = {"A": nbar, "B": nbar}
        cov = multi_tracer_cov_general(
            tracer_names, pkmu_funcs, nbars,
            self.k, self.V, self.dk, self.ells,
        )
        d_AA = self._multipole_derivs(self.bA, self.bA)
        d_BB = self._multipole_derivs(self.bB, self.bB)
        d_AB = self._multipole_derivs(self.bA, self.bB)
        derivs = np.concatenate([d_AA, d_BB, d_AB])

        cov_inv = np.linalg.inv(cov[0])
        return float(derivs @ cov_inv @ derivs) * self.dk

    def _st_fisher_pm(self, nbar):
        """ST Fisher for P_m (known bias) using multipole covariance."""
        pkmu_AA = make_kaiser_pkmu(self.bA, self.f, self.Pm)
        cov = single_tracer_cov(pkmu_AA, self.k, nbar, self.V, self.dk, self.ells)
        d_AA = self._multipole_derivs(self.bA, self.bA)
        cov_inv = np.linalg.inv(cov[0])
        return float(d_AA @ cov_inv @ d_AA) * self.dk

    def test_mt_beats_st_moderate_nbar(self):
        """MT Fisher ≥ ST Fisher at moderate nbar (well-conditioned regime)."""
        for nbar in [1e-1, 1e0, 1e2, 1e4]:
            F_MT = self._mt_fisher_pm(nbar)
            F_ST = self._st_fisher_pm(nbar)
            assert F_MT >= F_ST * 0.99, (
                f"MT Fisher ({F_MT:.2e}) < ST Fisher ({F_ST:.2e}) at nbar={nbar}"
            )

    def test_mt_fisher_positive_moderate_nbar(self):
        """MT Fisher should be positive at moderate nbar."""
        for nbar in [1e0, 1e2, 1e4]:
            F_MT = self._mt_fisher_pm(nbar)
            assert F_MT > 0, f"MT Fisher negative at nbar={nbar}: {F_MT}"
