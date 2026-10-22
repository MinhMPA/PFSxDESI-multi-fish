"""Fresh population information must preserve shared priors and covariance."""
from copy import deepcopy

import numpy as np
import pytest

from pfsfog.fisher_core import fisher_from_shells
from test_pfs_density_inference import toy_result
from scripts.pfs_bias_inference import (bias_bin_covariance, full_bias_covariance,
    reference_physical_check, check_cross_information)
from scripts.report_pfs_bias_study import percent_improvement, loop_response, check_point_identity


def changed_population(result):
    parts=[]
    for old in result.likelihood.cells[1:]:
        c=deepcopy(old)
        # Density unchanged. This represents a genuinely changed mean response
        # and correlated covariance, rather than a scalar Fisher rescaling.
        c.derivatives[...,0]*=1.3
        c.derivatives[...,-1]*=.9
        c.covariance[:,0,1]=c.covariance[:,1,0]=.2
        c.F_like=fisher_from_shells(c.derivatives,c.covariance,result.likelihood.grid.centers)
        parts.append({'cell':c})
    return parts


def test_population_replacement_matches_direct_full_inverse_with_priors_once():
    r=toy_result();parts=changed_population(r)
    calculated=bias_bin_covariance(r,parts)
    total=r.F_total.copy();indices={n:i for i,n in enumerate(r.likelihood.names)}
    for p,old in zip(parts,r.likelihood.cells[1:]):
        c=p['cell'];ix=[indices[n] for n in c.names]
        total[np.ix_(ix,ix)]+=c.F_like-old.F_like
    expected=np.linalg.inv(total)[:3,:3]
    np.testing.assert_allclose(calculated['cosmological_covariance'],expected,rtol=1e-11,atol=1e-12)
    full=full_bias_covariance(r,parts)
    np.testing.assert_allclose(full['cov_cosmo_marg'],expected,rtol=1e-11,atol=1e-12)
    np.testing.assert_array_equal(full['F_prior_nuisance'],r.F_prior_nuisance)
    np.testing.assert_array_equal(full['F_prior_cosmo'],r.F_prior_cosmo)
    assert calculated['changed_prior_widths']=={}
    assert 'factor' not in calculated and 'prior_mode' not in calculated
    np.testing.assert_allclose(calculated['growth_covariance'],r.growth_jacobian@expected@r.growth_jacobian.T,rtol=1e-11)


def test_missing_radial_population_or_changed_density_rejected():
    r=toy_result();parts=changed_population(r)
    with pytest.raises(ValueError,match='complete PFS'):bias_bin_covariance(r,parts[:1])
    parts[0]['cell'].metadata['nbar']['PFS-ELG']*=2
    with pytest.raises(ValueError,match='support|densit'):bias_bin_covariance(r,parts)


def test_unvaried_population_recovers_global_covariance():
    r=toy_result();parts=[{'cell':deepcopy(c)} for c in r.likelihood.cells[1:]]
    got=bias_bin_covariance(r,parts)
    np.testing.assert_allclose(got['cosmological_covariance'],r.cov_cosmo_marg,rtol=1e-11)


def test_gain_sign_and_loop_uncertainty_do_not_turn_a_tie_into_preference():
    assert percent_improvement(2.,1.8)==pytest.approx(10.)
    assert percent_improvement(2.,2.2)==pytest.approx(-10.)
    tied=loop_response(.1,-.02)
    assert tied['sign_resolved_by_loop_pair'] is False
    assert tied['half_width_percentage_points']==pytest.approx(.12)
    resolved=loop_response(1.,1.01)
    assert resolved['sign_resolved_by_loop_pair'] is True
    assert resolved['full_grid_qualification'] is False
    with pytest.raises(ValueError):percent_improvement(0.,1.)


def test_original_population_comparison_checks_arrays_not_just_global_errors():
    a={'arrays':{'mean':np.ones((2,3)),'D':np.ones((2,3,4))}}
    b=deepcopy(a)
    assert reference_physical_check(a,b)['status']=='passed'
    b['arrays']['D'][0,0,1]*=1.001
    with pytest.raises(ValueError):reference_physical_check(a,b)


def test_report_rejects_a_good_result_with_the_wrong_population_label():
    expected={'key':'w0-b1_only-r0.8','experiment':{'ratio':.8,'window':[.7,.8]}}
    point={'status':'passed','key':expected['key'],'grid':'shell304',
           'experiment':deepcopy(expected['experiment'])}
    check_point_identity(point,expected,'shell304')
    point['experiment']['ratio']=1.2
    with pytest.raises(ValueError,match='population'):check_point_identity(point,expected,'shell304')


def test_added_cross_spectra_cannot_lose_a_correlated_cosmological_combination():
    baseline=np.eye(3);improved=np.diag([.9,.8,.5])
    assert check_cross_information(baseline,improved)['minimum_scaled_eigenvalue']==pytest.approx(.1)
    # Each individual error shrinks, but one correlated combination worsens.
    invalid=np.diag([.9,.9,.5]);invalid[0,1]=invalid[1,0]=.5
    with pytest.raises(ValueError,match='cross spectra'):check_cross_information(baseline,invalid)
