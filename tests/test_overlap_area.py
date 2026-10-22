"""Change common volume at fixed DESI coverage and fixed population statistics."""
from copy import deepcopy

import numpy as np
import pytest

from overlap_fixtures import overlap_inputs_for_grid, overlap_inputs, overlap_selected


def discrepancy(actual, expected):
    scale = np.sqrt(np.diag(expected)).copy()
    scale[scale == 0] = 1.
    return np.max(np.abs(actual - expected) / scale[:, None] / scale[None, :], initial=0.)


@pytest.mark.parametrize('radial,region,area,expected', [
    ((.4, .6), 'overlap', 14000, 1.),
    ((.4, .6), 'nonoverlap', 14000, 1.),
    ((1.6, 2.), 'overlap', 14000, 14000/1200),
    ((1.6, 2.), 'nonoverlap', 14000, 0.),
    ((.6, 2.1), 'overlap', 1200, 1.),
    ((.6, 2.1), 'nonoverlap', 1200, 1.),
    ((2.1, 2.4), 'overlap', 14000, 1.),
    ((2.1, 2.4), 'nonoverlap', 14000, 1.),
])
def test_declared_common_volume_limits(radial, region, area, expected):
    from scripts.overlap_study import volume_weight
    assert volume_weight(radial, region, area) == pytest.approx(expected)


@pytest.mark.parametrize('radial,region,area', [
    ((.6, 1.), 'overlap', 1400), ((.6, 1.), 'overlap', np.nan),
    ((.6, 1.), 'overlap', 1200.), ((.6, 1.), 'other', 1200),
    ((.4, .7), 'overlap', 1200), ((2., 2.4), 'nonoverlap', 14000),
    ((.4, 2.4), 'overlap', 14000), ((np.nan, 1.), 'overlap', 1200),
    ((1., 1.), 'overlap', 1200), ((1.2, .8), 'overlap', 1200),
])
def test_undeclared_or_unsplit_volume_is_rejected(radial, region, area):
    from scripts.overlap_study import volume_weight
    with pytest.raises(ValueError):
        volume_weight(radial, region, area)


@pytest.fixture(scope='module')
def scaled(overlap_selected):
    from scripts.overlap_study import scale_area
    return {(case, area): scale_area(like, area)
            for case, like in overlap_selected.items() for area in (1200, 14000)}


@pytest.mark.parametrize('case', ['A', 'B', 'C', 'D'])
def test_original_area_reproduces_every_cell(overlap_selected, scaled, case):
    source, result = overlap_selected[case], scaled[case, 1200]
    assert result.names == source.names
    assert discrepancy(result.F_like, source.F_like) < 1e-10
    for a, b in zip(source.cells, result.cells):
        assert b.metadata['volume'] == a.metadata['volume']
        for field in ('fiducial_mean', 'derivatives', 'covariance', 'F_like'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize('case', ['A', 'B', 'C', 'D'])
def test_full_area_covariance_volume_and_empty_cells(overlap_selected, scaled, case):
    source, result = overlap_selected[case], scaled[case, 14000]
    nk = len(source.grid.centers)
    for original, cell in zip(source.cells, result.cells):
        shared = original.metadata['radial_bin'][0] >= .6
        w = (14000/1200 if original.metadata['region'] == 'overlap' else 0.) if shared else 1.
        assert cell.metadata['volume'] == pytest.approx(w * original.metadata['volume'])
        assert cell.metadata['area_scaling']['source_metadata'] == original.metadata
        assert cell.metadata['nbar'] == original.metadata['nbar']
        if w == 0:
            assert cell.names == [] and cell.metadata['empty_observation']
            assert cell.metadata['retained_observations'] == []
            assert cell.metadata['retained_pairs'] == []
            assert cell.metadata['local_names'] == []
            assert cell.derivatives.shape == (nk, 0, 0)
            assert cell.covariance.shape == (nk, 0, 0)
            assert cell.fiducial_mean.shape == (nk, 0)
            assert cell.F_like.shape == (0, 0)
            assert cell.validation['processed_shells'] == 0
        else:
            np.testing.assert_array_equal(cell.fiducial_mean, original.fiducial_mean)
            np.testing.assert_array_equal(cell.derivatives, original.derivatives)
            np.testing.assert_array_equal(cell.covariance, original.covariance / w)
            assert discrepancy(cell.F_like, w * original.F_like) < 1e-10
            # Independent contraction does not consume the saved cell Fisher.
            direct = np.einsum('koi,koj->ij', cell.derivatives,
                               np.linalg.solve(cell.covariance, cell.derivatives))
            assert discrepancy(cell.F_like, direct) < 1e-10


@pytest.mark.parametrize('case', ['A', 'B'])
def test_desi_reference_is_independent_of_common_area(scaled, case):
    a, b = scaled[case, 1200], scaled[case, 14000]
    assert a.names == b.names
    assert discrepancy(b.F_like, a.F_like) < 1e-10
    for actual, expected in zip(b.bin_F_like, a.bin_F_like):
        assert discrepancy(actual, expected) < 1e-10


@pytest.mark.parametrize('case', ['A', 'B', 'C', 'D'])
def test_low_redshift_information_grid_and_nuisance_sharing_are_preserved(overlap_selected, scaled, case):
    source, result = overlap_selected[case], scaled[case, 14000]
    assert discrepancy(result.bin_F_like[0], source.bin_F_like[0]) < 1e-10
    assert result.names == source.names
    assert len(set(result.names)) == len(result.names)
    assert result.parameters == source.parameters
    np.testing.assert_array_equal(result.fiducial, source.fiducial)
    for field in ('edges', 'centers', 'widths'):
        np.testing.assert_array_equal(getattr(result.grid, field), getattr(source.grid, field))
    # Every nuisance identity survives once globally, shared across all its cells.
    assert set(result.names) == {n for c in result.cells for n in c.names}
    assembled = np.zeros_like(result.F_like)
    for c in result.cells:
        index = [result.names.index(n) for n in c.names]
        assembled[np.ix_(index, index)] += c.F_like
    assert discrepancy(assembled, result.F_like) < 1e-10


def test_homogeneous_region_splitting_preserves_shared_parameter_information(overlap_selected):
    from scripts.overlap_study import scale_area
    source = deepcopy(overlap_selected['B'])
    original = source.cells[2]
    # One homogeneous region is split into two independent angular pieces,
    # with the same density, mean, responses and named nuisance parameters.
    halves = [deepcopy(original), deepcopy(original)]
    for cell, fraction, suffix in zip(halves, (.3, .7), ('a', 'b')):
        cell.metadata['cell_id'] += suffix
        cell.metadata['volume'] *= fraction
        cell.covariance /= fraction
        cell.F_like *= fraction
    source.cells[2:3] = halves
    split = scale_area(source, 14000)
    whole = scale_area(overlap_selected['B'], 14000)
    assert split.names == whole.names
    assert discrepancy(split.F_like, whole.F_like) < 1e-10


def test_scaling_does_not_modify_or_alias_source_arrays(overlap_selected):
    from scripts.overlap_study import scale_area
    source = overlap_selected['D']
    before = deepcopy(source)
    result = scale_area(source, 14000)
    result.F_like[0, 0] = -1.
    result.fiducial[0] = -1.
    result.grid.centers[0] = -1.
    result.cells[0].derivatives[0, 0, 0] = -1.
    result.cells[0].metadata['nbar']['DESI-LRG'] = -1.
    for field in ('F_like', 'fiducial'):
        np.testing.assert_array_equal(getattr(source, field), getattr(before, field))
    np.testing.assert_array_equal(source.grid.centers, before.grid.centers)
    for a, b in zip(source.cells, before.cells):
        assert a.metadata == b.metadata
        for field in ('F_like', 'derivatives', 'covariance', 'fiducial_mean'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_scale_area_rejects_unsupported_footprint(overlap_selected):
    from scripts.overlap_study import scale_area
    with pytest.raises(ValueError):
        scale_area(overlap_selected['A'], 1400)
