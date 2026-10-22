"""Regression on preserved producer bytes; no theory calculation or source mutation."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pfsfog import revision_artifacts as old

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / 'results/revision-qualification-d8dfcf4/production/I-k020-p1-all-ee4b092d0661'


def checker():
    assert importlib.util.find_spec('artifact_checker._checker') is not None, 'isolated corrected checker is missing'
    from artifact_checker import _checker
    return _checker


def test_real_covariance_old_rejects_corrected_full_reload_passes():
    manifest = json.loads((REAL / 'manifest.json').read_text())
    with np.load(REAL / 'matrices.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    with (REAL / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with pytest.raises(ValueError, match='^covariance mismatch$'):
        old._validate_payload(manifest, arrays, rows)
    fixed = checker()
    fixed._validate_payload(manifest, arrays, rows)
    np.testing.assert_array_equal(fixed._inverse(arrays['F_total'], 'F_total'), arrays['covariance'])


@pytest.mark.parametrize('matrix', [np.array([[np.nan]]), np.array([[np.inf]]),
    np.array([[0.]]), np.array([[-1.]]), np.array([[1., 2.], [2., 1.]]),
    np.array([[1., .2], [.3, 1.]]), np.ones((2, 3))])
def test_bad_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        checker()._inverse(matrix, 'invalid')
