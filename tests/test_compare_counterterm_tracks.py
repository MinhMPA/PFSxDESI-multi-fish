"""Cross-track equality must detect physical changes and incomplete evidence."""
import hashlib
import numpy as np
import pytest


def test_small_response_change_is_detected_even_beside_large_entries():
    from scripts.compare_counterterm_tracks import compare_array
    with pytest.raises(ValueError, match='response'):
        compare_array(np.array([1e12, 1.]), np.array([1e12, 1.00001]), 'response')
    record = compare_array(np.array([0., 1.]), np.array([0., 1.+1e-12]), 'response')
    assert record['rtol'] == 1e-10 and record['max_abs'] < 2e-12


@pytest.mark.parametrize('changed', [np.nan, np.inf])
def test_nonfinite_physical_arrays_are_never_equal(changed):
    from scripts.compare_counterterm_tracks import compare_array
    with pytest.raises(ValueError):
        compare_array(np.array([changed]), np.array([changed]), 'spectrum')


def test_common_responses_are_matched_by_name_and_only_extra_U_is_removed():
    from scripts.compare_counterterm_tracks import common_indices
    assert common_indices(['A', 'Q'], ['A', 'x:c_fog_extra', 'Q']) == [0, 2]
    for names in (['A', 'Q', 'rogue'], ['A', 'A', 'Q'], ['A', 'x:c_fog_extra']):
        with pytest.raises(ValueError):
            common_indices(['A', 'Q'], names)


def test_engine_identity_allows_only_the_known_descriptive_track_label():
    from scripts.compare_counterterm_tracks import compare_engine
    compare_engine({'fft_n': 4096, 'counterterm_prescription': 'CT-A-shared-field-leading-v1'}, {'fft_n': 4096})
    for changed in ({'fft_n': 2048}, {'fft_n': 4096, 'counterterm_prescription': 'unknown'}):
        with pytest.raises(ValueError):
            compare_engine({'fft_n': 4096}, changed)


def test_grid_coverage_rejects_duplicate_or_missing_physical_cells():
    from scripts.compare_counterterm_tracks import require_coverage
    require_coverage(['a', 'b'], {'a', 'b'}, 'cells')
    with pytest.raises(ValueError):
        require_coverage(['a', 'a'], {'a', 'b'}, 'cells')


def test_changed_bound_result_is_rejected(tmp_path):
    from scripts.compare_counterterm_tracks import verify_map
    p = tmp_path/'result'; p.write_bytes(b'original')
    binding = {str(p): hashlib.sha256(b'original').hexdigest()}
    assert verify_map(binding) == 1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding'):
        verify_map(binding)


def test_historical_exception_applies_only_to_exact_runner_path_and_old_digest(tmp_path):
    from scripts.compare_counterterm_tracks import verify_map
    runner = tmp_path/'runner'; runner.write_bytes(b'acceptance rule')
    old_sha = hashlib.sha256(b'physical runner').hexdigest()
    assert verify_map({str(runner): old_sha}, historical=(str(runner), old_sha)) == 1
    physical = tmp_path/'physical'; physical.write_bytes(b'changed model')
    with pytest.raises(ValueError):
        verify_map({str(physical): old_sha}, historical=(str(runner), old_sha))
    with pytest.raises(ValueError):
        verify_map({str(runner): '0'*64}, historical=(str(runner), old_sha))


def test_gain_uses_matching_absolute_error_denominator():
    from scripts.compare_counterterm_tracks import error_gains
    rows = [{'public_case': k, 'sigma_marginalized': {'growth': v},
             'sigma_nuisance_fixed': {'growth': v/2}}
            for k, v in [('A', 10.), ('B', 8.), ('C', 6.), ('D', 4.)]]
    gains = error_gains(rows)
    assert gains['marginalized']['A_to_B']['growth'] == pytest.approx(20.)
    assert gains['marginalized']['B_to_D']['growth'] == 50.
    assert gains['nuisance_fixed']['C_to_D']['growth'] == pytest.approx(100/3)
