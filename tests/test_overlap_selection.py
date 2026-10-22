"""Select measured multipoles, retaining their Gaussian mutual covariance."""
from copy import deepcopy

import numpy as np
import pytest

from overlap_fixtures import overlap_inputs_for_grid, overlap_inputs, overlap_selected

@pytest.fixture(scope='module')
def inputs(overlap_inputs):
    return overlap_inputs


@pytest.fixture(scope='module')
def selected(overlap_selected):
    return overlap_selected


def in_basis(like, names):
    index = [like.names.index(n) for n in names]
    return like.F_like[np.ix_(index, index)]


def scaled_discrepancy(actual, expected):
    scale = np.sqrt(np.diag(expected))
    scale[scale == 0] = 1.
    return np.max(np.abs(actual - expected) / scale[:, None] / scale[None, :])


def test_current_cases_have_the_observable_parameter_counts(inputs, selected):
    assert [len(selected[c].names) for c in ('A', 'B', 'C', 'D')] == [157, 173, 250, 276]
    assert len(inputs.results['H21'].names) == 272
    assert not any('PFS-ELG' in p and p[0] != p[1]
                   for cell in selected['C'].cells for p in cell.metadata['retained_pairs'])
    assert {c.metadata['nominal_bin_index'] for c in selected['C'].cells
            if 'PFS-ELG' in c.metadata['active_populations']} == set(range(1, 8))


@pytest.mark.parametrize('case', ['A', 'B', 'C', 'D'])
def test_exact_source_means_responses_and_both_covariance_axes(inputs, selected, case):
    source = inputs.results['C21'].likelihood
    result = selected[case]
    assert [c.metadata['cell_id'] for c in result.cells] == [c.metadata['cell_id'] for c in source.cells]
    assert len(result.cells) == 20
    for original, cell in zip(source.cells, result.cells):
        obs = [(tuple(p), ell) for p, ell in original.metadata['retained_observations']]
        wanted = [(tuple(p), ell) for p, ell in cell.metadata['retained_observations']]
        rows = [obs.index(o) for o in wanted]
        cols = [original.names.index(n) for n in cell.names]
        np.testing.assert_array_equal(cell.fiducial_mean, original.fiducial_mean[:, rows])
        np.testing.assert_array_equal(cell.derivatives, original.derivatives[:, rows][:, :, cols])
        np.testing.assert_array_equal(cell.covariance, original.covariance[:, rows][:, :, rows])
        assert cell.metadata['selection']['source_rows'] == rows
        assert cell.metadata['selection']['source_columns'] == cols
        assert cell.metadata['local_names'] == cell.names
        active = cell.metadata['active_populations']
        theoretical = [(p, p) for p in active] + [(p, q) for i, p in enumerate(active) for q in active[i+1:]]
        assert list(map(tuple, cell.metadata['theoretical_pairs'])) == theoretical
        assert set(cell.metadata['nbar']) == set(active)
        for field in ('radial_bin', 'nominal_bin', 'volume', 'region', 'model_redshift'):
            assert cell.metadata[field] == original.metadata[field]
    assert result.config.case == {'A':'A', 'B':'B', 'C':'D', 'D':'C'}[case]
    assert result.validation['selection']['case'] == case
    assert result.validation['selection']['source_columns'] == [source.names.index(n) for n in result.names]
    np.testing.assert_array_equal(result.fiducial, [p['fiducial'] for p in result.parameters])
    assert [p['name'] for p in result.parameters] == result.names
    for p in result.parameters:
        assert p['prior_only'] == (result.F_like[result.names.index(p['name']), result.names.index(p['name'])] == 0
                                   and p['prior_sigma'] is not None)


@pytest.mark.parametrize('case,archive', [('A','A'), ('B','B'), ('D','C21')])
def test_selected_information_matches_saved_likelihood_in_named_basis(inputs, selected, case, archive):
    expected = inputs.results[archive].likelihood
    actual = selected[case]
    assert set(actual.names) == set(expected.names)
    assert scaled_discrepancy(in_basis(actual, expected.names), expected.F_like) <= 1e-10
    order = [actual.names.index(n) for n in expected.names]
    for a, b in zip(actual.bin_F_like, expected.bin_F_like):
        assert scaled_discrepancy(a[np.ix_(order, order)], b) <= 1e-10


def test_mutual_auto_covariance_is_retained(selected):
    for case, pair in [('A', ('DESI-ELG', 'DESI-LRG')), ('C', ('DESI-LRG', 'PFS-ELG'))]:
        cell = next(c for c in selected[case].cells if set(pair) <= set(c.metadata['active_populations']))
        observations = [(tuple(p), ell) for p, ell in cell.metadata['retained_observations']]
        a, b = [observations.index(((n, n), 0)) for n in pair]
        assert np.any(cell.covariance[:, a, b] != 0.)


def test_new_c_information_is_direct_response_covariance_contraction(inputs, selected):
    like = selected['C']
    bins = [np.zeros_like(like.F_like) for _ in like.config.nominal_bins]
    for cell in like.cells:
        direct = np.einsum('koi,koj->ij', cell.derivatives,
                           np.linalg.solve(cell.covariance, cell.derivatives))
        assert scaled_discrepancy(cell.F_like, direct) <= 1e-10
        index = [like.names.index(n) for n in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(index, index)] += direct
    assert scaled_discrepancy(like.F_like, np.sum(bins, axis=0)) <= 1e-10
    # Keeping the D Fisher columns still keeps information from removed spectra.
    wrong = in_basis(inputs.results['C21'].likelihood, like.names)
    assert scaled_discrepancy(like.F_like, wrong) > 1e-5


def test_joint_reordering_of_observations_and_parameters_preserves_information(inputs, selected):
    from scripts.overlap_study import select_observations
    source = deepcopy(inputs.results['C21'].likelihood)
    rng = np.random.default_rng(73)
    order = rng.permutation(len(source.names))
    source.names = [source.names[i] for i in order]
    source.parameters = [source.parameters[i] for i in order]
    source.fiducial = source.fiducial[order]
    source.F_like = source.F_like[np.ix_(order, order)]
    source.bin_F_like = [f[np.ix_(order, order)] for f in source.bin_F_like]
    for cell in source.cells:
        rows = rng.permutation(cell.derivatives.shape[1])
        cols = rng.permutation(len(cell.names))
        cell.names = [cell.names[i] for i in cols]
        cell.metadata['local_names'] = list(cell.names)
        cell.metadata['retained_observations'] = [cell.metadata['retained_observations'][i] for i in rows]
        cell.derivatives = cell.derivatives[:, rows][:, :, cols]
        cell.covariance = cell.covariance[:, rows][:, :, rows]
        cell.fiducial_mean = cell.fiducial_mean[:, rows]
        cell.F_like = cell.F_like[np.ix_(cols, cols)]
    for case in ('A', 'B', 'C', 'D'):
        actual = select_observations(source, case)
        assert scaled_discrepancy(in_basis(actual, selected[case].names), selected[case].F_like) <= 1e-10


def test_missing_global_cosmological_column_is_rejected(inputs):
    from scripts.overlap_study import select_observations
    source = deepcopy(inputs.results['C21'].likelihood)
    source.names.remove('Mnu')
    with pytest.raises(ValueError, match='cosmolog'):
        select_observations(source, 'A')


def test_missing_local_cosmological_column_is_rejected(inputs):
    from scripts.overlap_study import select_observations
    source = deepcopy(inputs.results['C21'].likelihood)
    cell = source.cells[0]
    col = cell.names.index('Mnu')
    cell.names.pop(col)
    cell.metadata['local_names'] = list(cell.names)
    cell.derivatives = np.delete(cell.derivatives, col, axis=2)
    with pytest.raises(ValueError, match='cosmolog'):
        select_observations(source, 'A')


def test_removed_nuisance_with_retained_response_is_rejected(inputs):
    from scripts.overlap_study import select_observations
    source = deepcopy(inputs.results['C21'].likelihood)
    cell = next(c for c in source.cells if 'PFS-ELG' in c.metadata['active_populations'])
    col = next(i for i, n in enumerate(cell.names) if 'PFS-ELG' in n)
    row = next(i for i, (p, _) in enumerate(cell.metadata['retained_observations']) if 'PFS-ELG' not in p)
    cell.derivatives[0, row, col] = 1.
    with pytest.raises(ValueError, match='omitted|removed|retained response'):
        select_observations(source, 'A')


def test_unknown_public_case_is_rejected(inputs):
    from scripts.overlap_study import select_observations
    with pytest.raises(ValueError, match='case'):
        select_observations(inputs.results['C21'].likelihood, 'H21')
