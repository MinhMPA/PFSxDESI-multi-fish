"""Validate numerical GL covariance against analytic Wigner 3j formula.

Compares the multi-tracer Gaussian covariance computed via 20-point
Gauss-Legendre quadrature (our pipeline) against the analytic
multipole-decomposed formula of Rubira & Conteddu (2025, Eq. 2.20)
using Wigner 3j symbols.
"""

import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j


# --------------------------------------------------------------------------
# Analytic covariance via Wigner 3j (Rubira & Conteddu 2025, Eq. 2.20)
# --------------------------------------------------------------------------

def _wigner3j_sq(l1, l2, l3):
    """Compute (l1 l2 l3 / 0 0 0)^2."""
    val = float(wigner_3j(l1, l2, l3, 0, 0, 0))
    return val * val


def _cov_wigner3j(P_ell_XW, P_ell_YZ, P_ell_XZ, P_ell_YW,
                  ell, ellp, Nmodes, ells=(0, 2, 4)):
    """Analytic covariance via Wigner 3j (Eq. 2.20 of Rubira & Conteddu 2025).

    Parameters
    ----------
    P_ell_XW, P_ell_YZ, P_ell_XZ, P_ell_YW : dict
        {ell: float} multipole-decomposed total power spectra at a single k.
    ell, ellp : int
        Multipole orders of the covariance entry.
    Nmodes : float
        Number of modes in the k-bin.
    ells : tuple
        Multipole orders included in the decomposition.

    Returns
    -------
    float
        Cov^{ell,ell'}_{XY,WZ}(k).
    """
    result = 0.0
    for l1 in ells:
        for l2 in ells:
            # Triangle inequality for l3
            l3_min = abs(l1 - l2)
            l3_max = l1 + l2
            for l3 in range(l3_min, l3_max + 1):
                w1 = _wigner3j_sq(l1, l2, l3)
                if w1 == 0:
                    continue
                w2 = _wigner3j_sq(ell, ellp, l3)
                if w2 == 0:
                    continue
                term1 = (-1)**ellp * P_ell_XW[l1] * P_ell_YZ[l2]
                term2 = P_ell_XZ[l1] * P_ell_YW[l2]
                result += (2 * l3 + 1) * w1 * w2 * (term1 + term2)
    return (2 * ell + 1) * (2 * ellp + 1) / Nmodes * result


# --------------------------------------------------------------------------
# Numerical covariance via GL quadrature (our pipeline)
# --------------------------------------------------------------------------

def _cov_gl(Ptot_XW_mu, Ptot_YZ_mu, Ptot_XZ_mu, Ptot_YW_mu,
            ell, ellp, Nmodes, mu, w):
    """Numerical covariance via Gauss-Legendre quadrature."""
    from pfsfog.covariance_mt_general import _legendre
    L_ell = _legendre(ell, mu)
    L_ellp = _legendre(ellp, mu)
    integrand = L_ell * L_ellp * (Ptot_XW_mu * Ptot_YZ_mu
                                  + Ptot_XZ_mu * Ptot_YW_mu) * w
    return (2 * ell + 1) * (2 * ellp + 1) / (2.0 * Nmodes) * np.sum(integrand)


# --------------------------------------------------------------------------
# Helper: decompose P_tot(mu) into multipoles
# --------------------------------------------------------------------------

def _decompose_multipoles(Ptot_mu, mu, w, ells=(0, 2, 4)):
    """Project P_tot(mu) onto Legendre multipoles."""
    from pfsfog.covariance_mt_general import _legendre
    result = {}
    for ell in ells:
        L = _legendre(ell, mu)
        result[ell] = (2 * ell + 1) / 2.0 * np.sum(L * Ptot_mu * w)
    return result


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestCovarianceWigner3j:
    """Compare GL quadrature against analytic Wigner 3j for two tracers."""

    @pytest.fixture
    def setup(self):
        """Set up a toy two-tracer system with realistic mu-dependence."""
        from numpy.polynomial.legendre import leggauss
        Nleg = 20
        mu, w = leggauss(Nleg)

        # Kaiser model: P(k,mu) = (b + f*mu^2)^2 * Pm + 1/nbar
        Pm = 1e4  # signal power at k=0.1
        f = 0.8
        b_A, b_B = 1.3, 2.5
        nbar_A, nbar_B = 8e-4, 3e-4

        # P_tot(mu) for each pair
        P_AA_mu = (b_A + f * mu**2)**2 * Pm + 1.0 / nbar_A
        P_BB_mu = (b_B + f * mu**2)**2 * Pm + 1.0 / nbar_B
        P_AB_mu = (b_A + f * mu**2) * (b_B + f * mu**2) * Pm  # no shot noise

        Nmodes = 100.0

        return {
            'mu': mu, 'w': w, 'Nmodes': Nmodes,
            'P_AA_mu': P_AA_mu, 'P_BB_mu': P_BB_mu, 'P_AB_mu': P_AB_mu,
        }

    @pytest.fixture
    def multipoles(self, setup):
        """Decompose P_tot(mu) into multipoles for the Wigner 3j formula."""
        mu, w = setup['mu'], setup['w']
        return {
            'AA': _decompose_multipoles(setup['P_AA_mu'], mu, w),
            'BB': _decompose_multipoles(setup['P_BB_mu'], mu, w),
            'AB': _decompose_multipoles(setup['P_AB_mu'], mu, w),
        }

    @pytest.mark.parametrize("ell,ellp", [
        (0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 4),
    ])
    def test_auto_auto(self, setup, multipoles, ell, ellp):
        """Cov[P_ell^{AA}, P_ellp^{AA}]: GL vs Wigner 3j."""
        s = setup
        # GL
        cov_gl = _cov_gl(s['P_AA_mu'], s['P_AA_mu'],
                         s['P_AA_mu'], s['P_AA_mu'],
                         ell, ellp, s['Nmodes'], s['mu'], s['w'])
        # Wigner 3j
        m = multipoles
        cov_w3j = _cov_wigner3j(m['AA'], m['AA'], m['AA'], m['AA'],
                                ell, ellp, s['Nmodes'])
        np.testing.assert_allclose(cov_gl, cov_w3j, rtol=1e-10,
                                   err_msg=f"AA-AA (ell={ell}, ell'={ellp})")

    @pytest.mark.parametrize("ell,ellp", [
        (0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 4),
    ])
    def test_cross_cross(self, setup, multipoles, ell, ellp):
        """Cov[P_ell^{AB}, P_ellp^{AB}]: GL vs Wigner 3j."""
        s = setup
        # For (XY)=(AB), (WZ)=(AB): XW=AA, YZ=BB, XZ=AB, YW=BA=AB
        cov_gl = _cov_gl(s['P_AA_mu'], s['P_BB_mu'],
                         s['P_AB_mu'], s['P_AB_mu'],
                         ell, ellp, s['Nmodes'], s['mu'], s['w'])
        m = multipoles
        cov_w3j = _cov_wigner3j(m['AA'], m['BB'], m['AB'], m['AB'],
                                ell, ellp, s['Nmodes'])
        np.testing.assert_allclose(cov_gl, cov_w3j, rtol=1e-10,
                                   err_msg=f"AB-AB (ell={ell}, ell'={ellp})")

    @pytest.mark.parametrize("ell,ellp", [
        (0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 4),
    ])
    def test_auto_cross(self, setup, multipoles, ell, ellp):
        """Cov[P_ell^{AA}, P_ellp^{AB}]: GL vs Wigner 3j."""
        s = setup
        # For (XY)=(AA), (WZ)=(AB): XW=AA, YZ=AB, XZ=AB, YW=AA
        cov_gl = _cov_gl(s['P_AA_mu'], s['P_AB_mu'],
                         s['P_AB_mu'], s['P_AA_mu'],
                         ell, ellp, s['Nmodes'], s['mu'], s['w'])
        m = multipoles
        cov_w3j = _cov_wigner3j(m['AA'], m['AB'], m['AB'], m['AA'],
                                ell, ellp, s['Nmodes'])
        np.testing.assert_allclose(cov_gl, cov_w3j, rtol=1e-10,
                                   err_msg=f"AA-AB (ell={ell}, ell'={ellp})")

    @pytest.mark.parametrize("ell,ellp", [
        (0, 0), (0, 2), (2, 2),
    ])
    def test_auto_auto_bb(self, setup, multipoles, ell, ellp):
        """Cov[P_ell^{BB}, P_ellp^{BB}]: GL vs Wigner 3j."""
        s = setup
        cov_gl = _cov_gl(s['P_BB_mu'], s['P_BB_mu'],
                         s['P_BB_mu'], s['P_BB_mu'],
                         ell, ellp, s['Nmodes'], s['mu'], s['w'])
        m = multipoles
        cov_w3j = _cov_wigner3j(m['BB'], m['BB'], m['BB'], m['BB'],
                                ell, ellp, s['Nmodes'])
        np.testing.assert_allclose(cov_gl, cov_w3j, rtol=1e-10,
                                   err_msg=f"BB-BB (ell={ell}, ell'={ellp})")

    @pytest.mark.parametrize("ell,ellp", [
        (0, 0), (0, 2), (2, 2),
    ])
    def test_auto_bb_cross(self, setup, multipoles, ell, ellp):
        """Cov[P_ell^{BB}, P_ellp^{AB}]: GL vs Wigner 3j."""
        s = setup
        # For (XY)=(BB), (WZ)=(AB): XW=BA=AB, YZ=AB, XZ=BB, YW=BA=AB
        # Actually: XW = P_tot(B,A) = P_AB, YZ = P_tot(B,B) = P_BB
        #           XZ = P_tot(B,B) = P_BB, YW = P_tot(B,A) = P_AB
        # Wait, let me be careful:
        # (X,Y) = (B,B), (W,Z) = (A,B)
        # XW = P_tot(B,A) = P_AB (symmetric)
        # YZ = P_tot(B,B) = P_BB
        # XZ = P_tot(B,B) = P_BB
        # YW = P_tot(B,A) = P_AB
        cov_gl = _cov_gl(s['P_AB_mu'], s['P_BB_mu'],
                         s['P_BB_mu'], s['P_AB_mu'],
                         ell, ellp, s['Nmodes'], s['mu'], s['w'])
        m = multipoles
        cov_w3j = _cov_wigner3j(m['AB'], m['BB'], m['BB'], m['AB'],
                                ell, ellp, s['Nmodes'])
        np.testing.assert_allclose(cov_gl, cov_w3j, rtol=1e-10,
                                   err_msg=f"BB-AB (ell={ell}, ell'={ellp})")
