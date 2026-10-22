"""Exact residual cross-stochastic conditioning and independent Gaussian checks."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from tests.overlap_fixtures import overlap_inputs_for_grid, overlap_inputs, overlap_selected
from pfsfog.revision_forecast import CellLikelihood
from scripts import overlap_study as study
from scripts.overlap_inputs import G4, TREATMENTS
from scripts.elf_study_profiles import TAIL_BINS


@pytest.mark.parametrize('treatment,count', [('marginalized',276), ('desi_fixed',260),
                                           ('pfs_fixed',250), ('all_cross_fixed',234)])
def test_exact_fixed_columns_preserve_observations(overlap_inputs, treatment, count):
    source = overlap_inputs.results['C21'].likelihood
    reduced, fixed = study.fix_cross_coefficients(source, treatment)
    assert len(reduced.names) == count
    assert len(fixed) == 276-count
    assert all(p['name'] in reduced.names for p in source.parameters if p['role'] != 'cross')
    assert all(source.fiducial[source.names.index(n)] == 0 for n in fixed)
    assert reduced.validation['conditioning']['fixed_values'] == [0.]*len(fixed)
    for old, new in zip(source.cells, reduced.cells):
        np.testing.assert_array_equal(new.covariance, old.covariance)
        np.testing.assert_array_equal(new.fiducial_mean, old.fiducial_mean)
        assert new.metadata['retained_observations'] == old.metadata['retained_observations']
        keep = [old.names.index(n) for n in new.names]
        np.testing.assert_array_equal(new.derivatives, old.derivatives[:,:,keep])
    keep = [source.names.index(n) for n in reduced.names]
    np.testing.assert_allclose(reduced.F_like, source.F_like[np.ix_(keep,keep)], rtol=1e-13, atol=1e-10)


@pytest.mark.parametrize('case', ['A','B','C'])
def test_absent_families_give_identical_conditioning(overlap_selected, case):
    source = overlap_selected[case]
    a, fa = study.fix_cross_coefficients(source, 'marginalized')
    b, fb = study.fix_cross_coefficients(source, 'pfs_fixed')
    np.testing.assert_array_equal(a.F_like,b.F_like)
    assert fa == fb == ()
    a, fa = study.fix_cross_coefficients(source, 'desi_fixed')
    b, fb = study.fix_cross_coefficients(source, 'all_cross_fixed')
    np.testing.assert_array_equal(a.F_like,b.F_like)
    assert fa == fb


@pytest.mark.parametrize('damage', ['treatment','coordinate','fiducial','array_fiducial'])
def test_invalid_conditioning_fails(overlap_inputs, damage):
    source = deepcopy(overlap_inputs.results['C21'].likelihood)
    i = next(i for i,p in enumerate(source.parameters) if p['role']=='cross')
    treatment = 'all_cross_fixed'
    if damage == 'treatment': treatment = 'all_fixed'
    if damage == 'coordinate': source.parameters[i]['coordinate'] = 'unknown_cross'
    if damage == 'fiducial': source.parameters[i]['fiducial'] = 1.
    if damage == 'array_fiducial': source.fiducial[i] = 1.
    with pytest.raises(ValueError): study.fix_cross_coefficients(source, treatment)


@pytest.fixture(scope='module')
def conditioned_results(overlap_inputs, overlap_selected):
    out = {}
    for case, area, treatment in [('D',1200,t) for t in TREATMENTS] + [
            (c,a,t) for c in ('C','D') for a in (1200,14000) for t in ('marginalized','all_cross_fixed')]:
        key = (case,area,treatment)
        if key in out: continue
        like = study.scale_area(overlap_selected[case], area)
        like, _ = study.fix_cross_coefficients(like,treatment)
        out[key] = study.infer_saved_growth(like,overlap_inputs.results['C21'])
    return out


def test_saved_growth_and_independent_marginalization(conditioned_results):
    from scripts.overlap_checks import check_marginalization
    maxima = {}
    for key, result in conditioned_results.items():
        check = check_marginalization(result)
        for metric in ('full_schur_scaled_error','qr_covariance_error','assembly_scaled_error',
                       'growth_scaled_error','fixed_growth_scaled_error'):
            assert check[metric] < (1e-10 if metric == 'assembly_scaled_error' else 1e-8)
            maxima[metric] = max(maxima.get(metric,0),check[metric])
        assert np.all(np.diag(check['C_fixed']) > 1/np.diag(check['H']))
        np.testing.assert_allclose(check['H']-check['loss'], check['S'])
        np.testing.assert_allclose(result.growth_cov_marg,
                                   result.growth_jacobian @ check['C_marg'] @ result.growth_jacobian.T,
                                   rtol=1e-8,atol=0)
    print('INDEPENDENT_MAXIMA',json.dumps(maxima,sort_keys=True))


@pytest.mark.parametrize('damage', ['cosmology','fiducial','unit','redshift','growth_z','model','cell_redshift'])
def test_saved_growth_rejects_mismatched_physics(overlap_inputs,damage):
    ref = overlap_inputs.results['C21']
    like = deepcopy(ref.likelihood)
    if damage == 'cosmology': like.reference_cosmology['h'] += .001
    if damage == 'fiducial': like.fiducial[1] += .01
    if damage == 'unit': like.parameters[1]['unit'] = 'keV'
    if damage == 'redshift': like.config = replace(like.config,nominal_bins=TAIL_BINS)
    if damage == 'model': like.model['engine'] = 'different'
    if damage == 'cell_redshift': like.cells[0].metadata['model_redshift'] += .01
    if damage == 'growth_z':
        ref = deepcopy(ref)
        ref.growth_z[0] += .01
    with pytest.raises(ValueError): study.infer_saved_growth(like,ref)


def test_singular_amplitude_direction_fails(overlap_inputs):
    ref = overlap_inputs.results['C21']
    like = deepcopy(ref.likelihood)
    like.F_like[0,:] = like.F_like[:,0] = 0.
    for cell in like.cells:
        if cell.names: cell.derivatives[:,:,0] = 0.
    with pytest.raises(ValueError): study.infer_saved_growth(like,ref)


def _toy_cell(d,c,labels):
    d=np.asarray(d,dtype=float)[None,:,:]; c=np.asarray(c,dtype=float)[None,:,:]
    f=d[0].T@np.linalg.solve(c[0],d[0])
    return CellLikelihood(f,['theta'],{'cell_id':'toy','retained_observations':labels}, {},d,c,
                          np.zeros(d.shape[:2]))


def test_correlated_extra_row_with_zero_conditional_response():
    from scripts.overlap_checks import conditional_information
    old = _toy_cell([[1.]],[[1.]],[('auto',0)])
    new = _toy_cell([[1.],[.5]],[[1.,.5],[.5,1.]],[('auto',0),('cross',0)])
    check = conditional_information(old,new)
    np.testing.assert_array_equal(check['D'],np.zeros((1,1,1)))
    np.testing.assert_array_equal(check['C'],[[[.75]]])
    np.testing.assert_array_equal(check['F'],[[0.]])
    same = conditional_information(new,new)
    assert same['D'].shape == (1,0,1)
    np.testing.assert_array_equal(same['F'],[[0.]])


@pytest.mark.parametrize('treatment', TREATMENTS)
@pytest.mark.parametrize('before,after',[('A','B'),('B','C'),('C','D')])
def test_conditional_reconstruction_and_common_prior_ordering(overlap_selected,before,after,treatment):
    from scripts.overlap_checks import conditional_information
    old = study.fix_cross_coefficients(overlap_selected[before],treatment)[0]
    new = study.fix_cross_coefficients(overlap_selected[after],treatment)[0]
    f_old = np.zeros_like(new.F_like)
    ids = [new.names.index(n) for n in old.names]
    f_old[np.ix_(ids,ids)] = old.F_like
    increment = np.zeros_like(new.F_like)
    worst = 0.
    for a,b in zip(old.cells,new.cells):
        check = conditional_information(a,b)
        ids = [new.names.index(n) for n in b.names]
        increment[np.ix_(ids,ids)] += check['F']
        worst = max(worst,check['reconstruction_scaled_error'])
        old_rows = [(tuple(p),ell) for p,ell in a.metadata['retained_observations']]
        new_rows = [(tuple(p),ell) for p,ell in b.metadata['retained_observations']]
        kept = [new_rows.index(row) for row in old_rows]
        extra = [i for i in range(len(new_rows)) if i not in kept]
        if kept and extra:
            yx = b.covariance[:,extra][:,:,kept]
            response = check['D'] + yx @ np.linalg.solve(a.covariance,b.derivatives[:,kept])
            cov = check['C'] + yx @ np.linalg.solve(a.covariance,yx.swapaxes(-1,-2))
            np.testing.assert_allclose(response,b.derivatives[:,extra],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(cov,b.covariance[:,extra][:,:,extra],rtol=1e-10,atol=1e-10)
    scale = np.sqrt(np.diag(new.F_like)); scale[scale==0] = 1.
    assert np.max(abs(new.F_like-f_old-increment)/scale[:,None]/scale[None,:]) <= 1e-10
    # Add identical independent priors in the larger inventory, including newly
    # introduced stochastic coefficients, before comparing information.
    prior = np.diag([0 if p['prior_sigma'] is None else 1/p['prior_sigma']**2 for p in new.parameters])
    delta = (new.F_like+prior)-(f_old+prior)
    eig = np.linalg.eigvalsh(delta/scale[:,None]/scale[None,:])
    assert eig.min() >= -1e-10
    # Compare only shared coordinates; absent unconstrained new biases need not
    # have an inverse before their spectra have been included.
    print('CONDITIONAL',before,after,treatment,worst,float(eig.min()))


def test_fixing_and_area_do_not_increase_common_variances(conditioned_results):
    for case in ('C','D'):
        for area in (1200,14000):
            free = conditioned_results[case,area,'marginalized']
            fixed = conditioned_results[case,area,'all_cross_fixed']
            ids = [free.names.index(n) for n in fixed.names]
            assert np.max(np.diag(fixed.covariance)/np.diag(free.covariance)[ids]-1) <= 1e-8
        for treatment in ('marginalized','all_cross_fixed'):
            old = conditioned_results[case,1200,treatment]
            new = conditioned_results[case,14000,treatment]
            ids = [new.names.index(n) for n in old.names]
            assert np.max(np.diag(new.covariance)[ids]/np.diag(old.covariance)-1) <= 1e-8


def test_g4_historical_main_selection_errors_and_gains(overlap_inputs):
    saved = json.loads(G4.read_text())
    errors = {}
    discrepancies = []
    # G4 labels B,C,D,E use archived profiles A,B,D,C (current A,B,C,D).
    for label,profile in zip('BCDE',('A','B','D','C')):
        reference = overlap_inputs.results[profile]
        like,_ = study.fix_cross_coefficients(reference.likelihood,'all_cross_fixed')
        result = study.infer_saved_growth(like,reference)
        pivot = np.flatnonzero(np.isclose(result.growth_z,1.1,rtol=0,atol=1e-12))[0]
        errors[label] = np.sqrt([result.growth_cov_marg[pivot,pivot],result.covariance[1,1],result.covariance[2,2]])
        expected = np.asarray(saved[label]['treatments']['cross_stochastic_fixed']['errors'])
        np.testing.assert_allclose(errors[label],expected,rtol=1e-8,atol=0)
        discrepancies.append(float(np.max(abs(errors[label]/expected-1))))
    gain_errors = []
    for pair, treatments in saved['gains'].items():
        a,b = pair.split('->')
        gains = 100*(1-errors[b]/errors[a])
        expected = treatments['cross_stochastic_fixed']
        np.testing.assert_allclose(gains,expected,rtol=0,atol=1e-6)
        gain_errors.append(float(np.max(abs(gains-expected))))
    print('G4_MAXIMA',max(discrepancies),max(gain_errors))
