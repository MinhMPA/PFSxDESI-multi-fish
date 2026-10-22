"""Bin-only information is distinct from a slice of the global covariance."""
import numpy as np
from test_run_elf_studies import fake_result
from scripts.elf_study_diagnostics import bin_diagnostics


def test_bin_only_and_global_conditioning_are_explicit_and_independent():
    r=fake_result();rows=bin_diagnostics(r)
    assert len(rows)==8 and all(x['bin_only']['status']=='passed' for x in rows)
    for i,row in enumerate(rows):
        assert row['nominal_bin_index']==i
        assert row['labels'][:3]==['ln10_10_As','Mnu','Omegam']
        assert all(n.startswith(f'bin{i:02d}:') and n.endswith(':b1_sigma8') for n in row['labels'][3:])
        glob=np.asarray(row['global_marginal_covariance']);local=np.asarray(row['bin_only']['marginal_covariance'])
        assert local[0,0]>glob[0,0]
        ix=[r.names.index(n) for n in row['labels']]
        np.testing.assert_array_equal(glob,r.covariance[np.ix_(ix,ix)])
        assert row['bin_only']['nuisance_penalty_growth']>=1


def test_nonpositive_bin_information_is_reported_as_failed_not_regularized():
    r=fake_result();r.likelihood.bin_F_like[0][0,0]=-1
    rows=bin_diagnostics(r)
    assert rows[0]['bin_only']['status']=='failed'
    assert rows[0]['bin_only']['error']
    assert 'marginal_covariance' not in rows[0]['bin_only']
