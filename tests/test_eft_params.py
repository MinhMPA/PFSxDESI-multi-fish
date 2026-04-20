"""Tests for pfsfog.eft_params."""

import numpy as np
import pytest

from pfsfog.eft_params import (
    desi_elg_fiducials, pfs_elg_fiducials, broad_priors,
    lazeyras_b2, NUISANCE_NAMES, COSMO_NAMES,
)


class TestDESIFiducials:
    def test_b1_sigma8(self):
        fid = desi_elg_fiducials(b1=1.3, sigma8_z=0.5)
        assert abs(fid.b1_sigma8 - 0.65) < 1e-10

    def test_bGamma3(self):
        fid = desi_elg_fiducials(b1=1.3, sigma8_z=0.5)
        expected = 23 / 42 * (1.3 - 1.0)
        assert abs(fid.bGamma3 - expected) < 1e-10

    def test_c_tilde_desi(self):
        fid = desi_elg_fiducials(b1=1.3, sigma8_z=0.5)
        assert fid.c_tilde == 400.0

    def test_c2_desi(self):
        fid = desi_elg_fiducials(b1=1.3, sigma8_z=0.5)
        assert fid.c2 == 30.0

    def test_as_array_length(self):
        fid = desi_elg_fiducials(b1=1.3, sigma8_z=0.5)
        assert len(fid.as_array()) == len(NUISANCE_NAMES)


class TestPFSFiducials:
    def test_c_tilde_scaled(self):
        fid = pfs_elg_fiducials(
            b1_pfs=1.34, b1_desi=1.3, sigma8_z=0.5, r_sigma_v=0.75,
        )
        assert abs(fid.c_tilde - 400 * 0.75**2) < 1e-10

    def test_c2_scaled_by_bias_ratio(self):
        fid = pfs_elg_fiducials(
            b1_pfs=1.34, b1_desi=1.3, sigma8_z=0.5,
        )
        assert abs(fid.c2 - 30.0 * 1.34 / 1.3) < 1e-10


class TestBroadPriors:
    def test_prior_widths(self):
        bp = broad_priors()
        assert bp.c_tilde == 400.0
        assert bp.c0 == 30.0
        assert bp.Pshot == 1.0
        assert bp.b1_sigma8 is None  # flat prior

    def test_prior_fisher_diag_shape(self):
        bp = broad_priors()
        diag = bp.prior_fisher_diag()
        assert diag.shape == (len(NUISANCE_NAMES),)

    def test_flat_prior_gives_zero(self):
        bp = broad_priors()
        diag = bp.prior_fisher_diag()
        assert diag[0] == 0.0  # b1_sigma8 is flat


class TestLazeyras:
    def test_b2_at_b1_1(self):
        # b2(b1=1) ≈ 0.412 - 2.143 + 0.929 + 0.008 = -0.794
        b2 = lazeyras_b2(1.0)
        assert abs(b2 - (-0.794)) < 0.01

    def test_b2_at_b1_2(self):
        # b2(b1=2) ≈ 0.412 - 4.286 + 3.716 + 0.064 = -0.094
        b2 = lazeyras_b2(2.0)
        assert abs(b2 - (-0.094)) < 0.01
