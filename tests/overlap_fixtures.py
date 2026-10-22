"""Process-local immutable positive fixtures; corrupted records never enter here."""
from pathlib import Path

import numpy as np
import pytest

def freeze_overlap_arrays(value):
    """Protect authenticated source arrays shared by positive overlap tests."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            freeze_overlap_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze_overlap_arrays(item)
    elif hasattr(value, '__dict__'):
        freeze_overlap_arrays(vars(value))


_POSITIVE_INPUTS = {}


@pytest.fixture(scope='session')
def overlap_inputs_for_grid():
    from scripts.overlap_inputs import load_inputs
    cache = _POSITIVE_INPUTS

    def authenticated(grid):
        if grid not in cache:
            source = Path(__file__).resolve().parents[1] / 'results/takada-density-full-20261014-02'
            cache[grid] = load_inputs(source, grid)
            freeze_overlap_arrays(cache[grid])
        return cache[grid]

    return authenticated


@pytest.fixture(scope='session')
def overlap_inputs(overlap_inputs_for_grid):
    return overlap_inputs_for_grid('shell304')


@pytest.fixture(scope='session')
def overlap_selected(overlap_inputs):
    from scripts.overlap_study import select_observations
    selected = {case: select_observations(overlap_inputs.results['C21'].likelihood, case)
                for case in ('A', 'B', 'C', 'D')}
    freeze_overlap_arrays(selected)
    return selected
