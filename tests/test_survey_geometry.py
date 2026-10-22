"""Independent support/volume expectations for the accepted nominal bins."""
from dataclasses import replace

import numpy as np
import pytest

from pfsfog.surveys import default_survey_group, Survey, SurveyGroup
from pfsfog.survey_geometry import ShellGeometry, support_cells


def test_pfs_lower_support_and_nominal_redshift():
    sg = default_survey_group()
    cells = support_cells(sg, (0.6, 0.8))
    assert [c.radial_bin for c in cells] == [(0.6, 0.7), (0.7, 0.8)]
    assert "PFS-ELG" not in cells[0].overlap_tracers
    assert "PFS-ELG" in cells[1].overlap_tracers
    assert all(c.nominal_bin == (0.6, 0.8) and c.model_redshift == 0.7 for c in cells)
    assert all("PFS-ELG" not in c.nonoverlap_tracers for c in cells)


def test_lrg_only_occupies_supported_volume():
    sg = default_survey_group()
    cells = support_cells(sg, (1.0, 1.2))
    assert [c.radial_bin for c in cells] == [(1.0, 1.1), (1.1, 1.2)]
    assert "DESI-LRG" in cells[0].overlap_tracers
    assert "DESI-LRG" not in cells[1].overlap_tracers
    volume = sum(c.overlap_volume + c.nonoverlap_volume for c in cells if "DESI-LRG" in c.overlap_tracers)
    np.testing.assert_allclose(volume, sg.desi_tracers["DESI-LRG"].volume(1.0, 1.1), rtol=1e-12)
    assert cells[0].nbar["DESI-LRG"] == pytest.approx(sg.desi_tracers["DESI-LRG"].nbar_eff(1, 1.1))


def test_pfs_upper_truncation():
    cells = support_cells(default_survey_group(), (1.6, 2.0))
    assert len(cells) == 1
    assert cells[0].overlap_tracers == cells[0].nonoverlap_tracers == ("DESI-QSO",)


def test_tracer_order_invariance_and_volume_additivity():
    sg = default_survey_group()
    swapped = replace(sg, desi_tracers=dict(reversed(list(sg.desi_tracers.items()))))
    geometry = ShellGeometry.from_group(sg)
    other = ShellGeometry.from_group(swapped)
    np.testing.assert_array_equal(geometry.edges, other.edges)
    np.testing.assert_array_equal(geometry.volume_per_deg2, other.volume_per_deg2)
    cells = support_cells(sg, (1.0, 1.2))
    assert cells == support_cells(swapped, (1.0, 1.2))
    full = sum(c.overlap_volume + c.nonoverlap_volume for c in cells)
    np.testing.assert_allclose(full, sg.desi_tracers["DESI-ELG"].volume(1, 1.2), rtol=1e-12)
    no_pfs = support_cells(sg, (1, 1.2), include_pfs=False)
    assert [c.radial_bin for c in no_pfs] == [c.radial_bin for c in cells]
    assert sum(c.overlap_volume + c.nonoverlap_volume for c in no_pfs) == full


def test_elg1_selection_keeps_other_populations():
    sg = default_survey_group()
    full = support_cells(sg, (1, 1.2))
    masked = support_cells(sg, (1, 1.2), exclude_elg1=True)
    assert "DESI-ELG" not in masked[0].overlap_tracers
    assert "DESI-ELG" in masked[1].overlap_tracers
    for original, selected in zip(full, masked):
        for name in original.overlap_tracers:
            if name != "DESI-ELG":
                assert name in selected.overlap_tracers
                assert original.nbar[name] == selected.nbar[name]
    assert "DESI-ELG" not in support_cells(sg, (.8, 1), exclude_elg1=True)[0].overlap_tracers


def toy_group():
    s = Survey("DESI-X", 100, np.array([0., 1.]), np.array([1., 2.]),
               np.array([2., 4.]), np.array([10., 30.]), lambda z: 1.)
    pfs = replace(s, name="PFS-ELG", area_deg2=10., Vz_all=s.Vz_all / 10)
    return SurveyGroup(pfs, {s.name: s}, 10, 100, pfs_zmax=2.)


def test_analytic_volume_and_density_weights():
    sg = toy_group()
    cell = support_cells(sg, (0, 2))[0]
    assert cell.nbar["DESI-X"] == 3.5  # (2*10 + 4*30)/40
    assert cell.overlap_volume == 4
    assert cell.nonoverlap_volume == 36
    geom = ShellGeometry.from_group(sg)
    assert geom.volume(.5, 1.5, 100) == 20  # half of each native shell


def test_mismatched_geometric_input_is_an_error():
    sg = default_survey_group()
    sg.pfs = replace(sg.pfs, Vz_all=sg.pfs.Vz_all * 1.05)
    with pytest.raises(ValueError, match="PFS-ELG.*volume"):
        ShellGeometry.from_group(sg)
