"""Tests for pfsfog.surveys."""

import numpy as np
import pytest

from pfsfog.surveys import (
    load_nz_table, pfs_elg, desi_elg, default_survey_pair, OVERLAP_ZBINS,
)


class TestLoadNz:
    def test_loads_desi(self):
        z_min, z_max, nz, Vz = load_nz_table(
            "survey_specs/DESI_nz_elg_fine.txt"
        )
        assert len(z_min) > 50
        assert np.all(nz > 0)
        assert np.all(Vz > 0)
        assert np.all(z_max > z_min)

    def test_loads_pfs(self):
        z_min, z_max, nz, Vz = load_nz_table(
            "survey_specs/PFS_nz_pfs_fine.txt"
        )
        assert len(z_min) > 40


class TestSurvey:
    def test_pfs_area(self):
        s = pfs_elg()
        assert s.area_deg2 == 1200.0

    def test_desi_area(self):
        s = desi_elg()
        assert s.area_deg2 == 14000.0

    def test_nbar_eff_positive(self):
        s = desi_elg()
        for zlo, zhi in OVERLAP_ZBINS:
            assert s.nbar_eff(zlo, zhi) > 0

    def test_z_eff_in_range(self):
        s = desi_elg()
        for zlo, zhi in OVERLAP_ZBINS:
            ze = s.z_eff(zlo, zhi)
            assert zlo <= ze <= zhi

    def test_volume_positive(self):
        s = desi_elg()
        for zlo, zhi in OVERLAP_ZBINS:
            assert s.volume(zlo, zhi) > 0

    def test_pfs_bias(self):
        s = pfs_elg()
        b1 = s.b1_of_z(1.0)
        assert abs(b1 - 1.3) < 0.05  # 0.9 + 0.4*1.0 = 1.3


class TestSurveyPair:
    def test_lever_arm(self):
        sp = default_survey_pair()
        for zlo, zhi in OVERLAP_ZBINS:
            la = sp.lever_arm(zlo, zhi)
            assert abs(la - 14000 / 1200) < 0.5

    def test_V_overlap_lt_V_full(self):
        sp = default_survey_pair()
        for zlo, zhi in OVERLAP_ZBINS:
            assert sp.V_overlap(zlo, zhi) < sp.V_full_B(zlo, zhi)
