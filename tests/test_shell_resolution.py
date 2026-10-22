"""Correlated Gaussian controls isolate quadrature errors from survey gains."""
import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest

from test_shell_numerics import grid_sample


def api():
    assert importlib.util.find_spec('pfsfog.shell_resolution'), 'shell/output comparisons missing'
    return importlib.import_module('pfsfog.shell_resolution')


def compare(a,b,role='convergence'):
    return api().compare_shell_resolution(a[0],b[0],a[1],b[1],role=role)


def test_constant_correlated_information_density_integrates_on_both_grids():
    a,b=grid_sample(),grid_sample(.000625)
    report=compare(a,b)
    assert report['status']=='passed' and len(report['treatments'])==7
    assert report['base_grid']['shells']==152 and report['refined_grid']['shells']==304
    assert set(report['treatments']['marginal6']['outputs'])=={
        'cosmology_marg','cosmology_nuisance_fixed','growth_marg','growth_nuisance_fixed'}
    for treatment in report['treatments'].values():
        for row in treatment['outputs'].values():
            np.testing.assert_allclose(row['fractional_changes'],0,atol=1e-13)
            np.testing.assert_allclose(row['reverse_fractional_changes'],0,atol=1e-13)
    assert np.asarray(report['treatments']['marginal6']['covariances']['growth_marg']['base']).shape==(8,8)
    np.testing.assert_allclose(report['treatments']['marginal6']['covariances']['growth_marg']['base'],
                               a[1]['marginal6'].growth_cov_marg,rtol=1e-13)


def test_old_to_new_grid_shift_is_assessment_not_a_candidate_convergence_pass():
    a,b=grid_sample(.0025,amplitude_error=1.03),grid_sample()
    report=compare(a,b,'historical_assessment')
    assert report['status']=='assessed' and report['within_tolerance'] is False
    assert report['role']=='historical_assessment' and report['reasons']
    with pytest.raises(ValueError):compare(a,b)
    with pytest.raises(ValueError):compare(b,grid_sample(.000625),'historical_assessment')


def test_exact_one_percent_is_rejected_with_adopted_denominator():
    row=api()._change(np.array([1.]),np.array([1.01]),['amplitude'])
    assert row['status']=='failed'
    np.testing.assert_allclose(row['reverse_fractional_changes'],[1-1/1.01],rtol=1e-13)


@pytest.mark.parametrize('change',['support','ap','prior','mask','growth','missing','model','grid'])
def test_unmatched_physics_or_incomplete_outputs_cannot_pass(change):
    a,b=grid_sample(),grid_sample(.000625)
    if change=='support':b[0].cells[0].metadata['nbar']['A']*=2
    elif change=='ap':b[0].config=replace(b[0].config,ap_interpolation_nk=32768,ap_refinement_nk=(65536,131072))
    elif change=='prior':b[0].parameters[-1]['prior_sigma']=2
    elif change=='mask':b[1]['marginal6'].free_cosmology=('ln10_10_As',)
    elif change=='growth':b[1]['marginal6'].growth_cov_marg[0,1]*=.9
    elif change=='missing':b[1].pop('marginal6_no_extra')
    elif change=='model':b[0].model['fixed_physics']='changed'
    elif change=='grid':b[0].grid.centers[0]+=.0001
    with pytest.raises(ValueError):compare(a,b)


def failed():return {'status':'failed','error':'retained weak-direction failure','raw':{'stage':'projection'}}


def test_matching_optional_failures_are_retained_and_changed_success_set_fails():
    a,b=grid_sample(),grid_sample(.000625)
    a[1]['marginal6_no_extra']=failed()
    assert compare(a,b)['status']=='failed'
    b[1]['marginal6_no_extra']=failed()
    report=compare(a,b)
    assert report['status']=='passed' and len(report['treatments'])==7
    assert report['treatments']['marginal6_no_extra']['status']=='unavailable'


def test_reciprocal_gain_can_fail_when_original_direction_and_all_absolute_errors_pass():
    b,c=grid_sample(),grid_sample(case='C')
    br,cr=grid_sample(.000625,amplitude_error=1.005),grid_sample(.000625,case='C',amplitude_error=.995)
    assert compare(b,br)['status']==compare(c,cr)['status']=='passed'
    report=api().compare_shell_ratios(b,c,br,cr,role='convergence')
    row=report['treatments']['conditional3']['cosmology_marg']
    assert row['C_over_B']['status']=='passed' and row['B_over_C']['status']=='failed'
    assert report['status']=='failed' and report['successful_ratio_treatments']==7


def test_fixed_nuisance_gain_failure_is_not_hidden_by_marginalized_agreement():
    b,c=grid_sample(total_amplitude=1.4),grid_sample(case='C',total_amplitude=1.4)
    br,cr=grid_sample(.000625,total_amplitude=1.41),grid_sample(.000625,case='C',total_amplitude=1.38)
    assert compare(b,br)['status']==compare(c,cr)['status']=='passed'
    report=api().compare_shell_ratios(b,c,br,cr,role='convergence')
    row=report['treatments']['conditional3']
    assert row['cosmology_marg']['B_over_C']['status']=='passed'
    assert row['cosmology_nuisance_fixed']['B_over_C']['status']=='failed'
    assert report['status']=='failed'


def test_only_one_survey_refined_rejects_the_gain_comparison():
    b,c=grid_sample(),grid_sample(case='C')
    with pytest.raises(ValueError):
        api().compare_shell_ratios(b,c,grid_sample(.000625),grid_sample(case='C'),role='convergence')


def test_angular_resolution_preserves_shells_and_checks_same_error_families():
    a,b=grid_sample(),grid_sample(amplitude_error=1.005)
    b[0].config=replace(b[0].config,n_mu=64)
    for like in (a[0],b[0]):
        like.model.update(mean_n_mu=like.config.n_mu,covariance_n_mu=like.config.n_mu)
    report=compare(a,b)
    assert report['comparison']=='angular output resolution' and report['status']=='passed'
    assert report['base_grid']['shells']==report['refined_grid']['shells']==152
    assert report['refined_grid']['n_mu']==64
