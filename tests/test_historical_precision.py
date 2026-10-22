"""An accurate inverse of historical F is a derived reference, not a new oracle."""
import copy
import importlib

import numpy as np
import pytest

from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER, LEGACY_FISHER_SOLVER
from test_cosmology_resolution import sample, native3


def setup():
    like, results = sample(8192)
    like, result = native3(like, results['conditional3'])
    result.fisher_solver = LEGACY_FISHER_SOLVER
    parent = {'oracle_export':{'path':'/authenticated/oracle','sha256':'a'*64,'case':'B'},
              'worker_index_sha256':'b'*64, 'arrays_sha256':'c'*64, 'control_sha256':'d'*64}
    return like, result, parent


def derive(like, result, parent):
    return importlib.import_module('pfsfog.historical_precision').historical_precision_reference(
        like, result, fisher_solver=DECIMAL_FISHER_SOLVER, parent_identity=parent)


def test_historical_replay_and_derived_reference_are_separately_identified():
    like, old, parent = setup()
    before = old.covariance.copy()
    derived, receipt = derive(like, old, parent)
    assert receipt['legacy_replay']['status'] == 'passed'
    assert receipt['scientific_identity']['status'] == 'passed'
    assert receipt['parent_identity'] == parent
    assert receipt['fisher_solver']['name'] == DECIMAL_FISHER_SOLVER
    assert set(receipt['full_covariance_changes']) == {'same_resolution_allowance', 'reconstruction_allowance'}
    for change in receipt['full_covariance_changes'].values():
        assert 'failed_entries' in change and 'failed_values' in change
    np.testing.assert_array_equal(old.covariance, before)
    assert not np.shares_memory(old.covariance, derived.covariance)
    for suffix in ('marg','nuisance_fixed'):
        np.testing.assert_allclose(getattr(derived, 'growth_cov_'+suffix),
            old.growth_jacobian @ getattr(derived, 'cov_cosmo_'+suffix) @ old.growth_jacobian.T,
            rtol=1e-10, atol=1e-12)
    again, second = derive(like, old, parent)
    assert receipt == second


@pytest.mark.parametrize('mutation', ['covariance', 'prior', 'labels', 'growth_marg', 'growth_fixed', 'parent', 'solver'])
def test_derived_reference_cannot_hide_broken_historical_replay(mutation):
    like, old, parent = setup()
    if mutation == 'covariance': old.covariance[3,4] += .01
    elif mutation == 'prior': like.parameters[-1]['prior_sigma'] = 2.
    elif mutation == 'labels': like.names[3], like.names[4] = like.names[4], like.names[3]
    elif mutation == 'growth_marg': old.growth_cov_marg[0,1] += .01
    elif mutation == 'growth_fixed': old.growth_cov_nuisance_fixed[0,1] += .01
    elif mutation == 'parent': parent.pop('arrays_sha256')
    elif mutation == 'solver': old.fisher_solver = DECIMAL_FISHER_SOLVER
    with pytest.raises(ValueError): derive(like, old, parent)
