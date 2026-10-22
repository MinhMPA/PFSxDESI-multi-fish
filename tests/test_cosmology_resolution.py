"""AP resolution affects numerical accuracy, not physical survey information.

Small Gaussian likelihoods provide independent error/ratio expectations. These
are algebraic controls; native3 physical independence is tested by the runner.
"""
import copy
from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_projection import COSMOLOGY_TREATMENTS, project_cosmology_precision
from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.revision_forecast import RevisionConfig, RevisionLikelihood, CellLikelihood, make_shell_grid


def api():
    assert importlib.util.find_spec('pfsfog.cosmology_resolution'), 'AP output-resolution comparison is missing'
    return importlib.import_module('pfsfog.cosmology_resolution')


def sample(nk=16384, amplitude_error=1., case='B', *, total_amplitude=None):
    """Diagonal information with a controlled amplitude error; two nuisances."""
    config=RevisionConfig(case,kmax=.02,dk=.01,cosmology_basis='extended6',
                          ap_interpolation_nk=nk,ap_refinement_nk=(2*nk,4*nk))
    reference=dict(FIDUCIAL); space=make_cosmology_spec(reference,extended=True,ap_nk=nk)
    names=list(space.names)+['bias_a','bias_b']
    theta=np.array([reference['ln10_10_As'],reference['mnu'],reference['Omega_m'],
                    reference['h'],reference['omega_b'],reference['n_s'],1.,2.])
    params=[{'name':n,'role':'cosmo' if i<6 else 'nuisance','fiducial':float(v),
             'prior_sigma':None if i<6 else 1.,'unit':''} for i,(n,v) in enumerate(zip(names,theta))]
    fisher=np.eye(8); fisher[0,0]=amplitude_error**-2
    if total_amplitude is not None:
        # With nuisance prior precision1, this gives marginalized precision1
        # but nuisance-fixed precision total_amplitude, independently of the API.
        fisher[0,0]=total_amplitude
        fisher[0,6]=fisher[6,0]=np.sqrt(2*(total_amplitude-1))
    derivatives=np.vstack((np.linalg.cholesky(fisher).T,np.zeros(8)))[None]
    grid=make_shell_grid(.01,.02,.01)
    meta={'cell_id':'z09','nominal_bin_index':2,'model_redshift':.9,
          'retained_pairs':[['A','A'],['B','B'],['A','B']],'nbar':{'A':1.,'B':2.}}
    cell=CellLikelihood(fisher,names.copy(),meta,{},derivatives,np.eye(9)[None],np.ones((1,9)))
    bins=[np.zeros((8,8)) for _ in config.nominal_bins]; bins[2]=fisher.copy()
    like=RevisionLikelihood(config,fisher.copy(),names,theta,params,[cell],bins,grid,reference,{'fixed_physics':'toy'}, {})
    return like, projected(like)


def projected(like):
    z=np.array([(lo+hi)/2 for lo,hi in like.config.nominal_bins])
    jac=np.arange(1,49,dtype=float).reshape(8,6)/100
    output={}
    for name in COSMOLOGY_TREATMENTS:
        p=project_cosmology_precision(like,name)
        output[name]=SimpleNamespace(likelihood=like,treatment=name,**p,growth_z=z,
            growth_fiducial=np.full(8,.4),growth_jacobian=jac.copy(),
            growth_cov_marg=jac@p['cov_cosmo_marg']@jac.T,
            growth_cov_nuisance_fixed=jac@p['cov_cosmo_nuisance_fixed']@jac.T)
    return output


def failed(reason='controlled optional weak-direction failure'):
    return {'status':'failed','error':reason,'raw':{'exception_type':'ValueError'}}


def compare(a,b):
    return api().compare_ap_resolution(a[0],b[0],a[1],b[1])


def test_half_percent_error_shift_is_convergence_not_identity():
    a,b=sample(amplitude_error=1.005),sample(32768)
    report=compare(a,b)
    assert report['status']=='passed'
    row=report['treatments']['conditional3']['cosmology_marg']
    assert row['labels']==list(a[1]['conditional3'].free_cosmology)
    np.testing.assert_allclose(row['fractional_changes'],[.005,0.,0.],atol=1e-14)
    assert report['base_ap_nk']==16384 and report['refined_ap_nk']==32768
    assert len(report['treatments'])==7
    assert len(report['treatments']['marginal6']['growth_nuisance_fixed']['labels'])==8
    from pfsfog.extended_artifacts import require_close
    with pytest.raises(ValueError):
        require_close(a[1]['conditional3'].cov_cosmo_marg,b[1]['conditional3'].cov_cosmo_marg,
                      'strict same-grid identity',rtol=1e-8)


def test_one_percent_is_rejected_without_rounding_the_report():
    r=compare(sample(amplitude_error=1.01),sample(32768))
    assert r['status']=='failed'
    assert r['treatments']['conditional3']['cosmology_marg']['fractional_changes'][0]>=.01


def test_adjacent_passes_do_not_replace_direct_adopted_to_finest():
    a,b,c=sample(amplitude_error=1.018081),sample(32768,1.009),sample(65536)
    assert compare(a,b)['status']==compare(b,c)['status']=='passed'
    assert compare(a,c)['status']=='failed'


def test_nuisance_fixed_error_convergence_is_required_separately():
    a,b=sample(total_amplitude=1.04),sample(32768)
    report=compare(a,b)
    assert report['status']=='failed'
    assert report['treatments']['marginal6']['cosmology_marg']['status']=='passed'
    assert report['treatments']['marginal6']['cosmology_nuisance_fixed']['status']=='failed'


@pytest.mark.parametrize('mutation',['same_grid','reverse_grid','support','prior','fiducial',
                                    'shells','angular','model','covariance','local_fisher','missing_treatment'])
def test_resolution_rejects_changed_physics_or_inconsistent_saved_arrays(mutation):
    a,b=sample(),sample(32768); like,results=b
    if mutation=='same_grid': like.config=a[0].config
    elif mutation=='reverse_grid': a,b=b,a
    elif mutation=='support': like.cells[0].metadata['nbar']['A']*=2
    elif mutation=='prior': like.parameters[-1]['prior_sigma']=2.
    elif mutation=='fiducial': like.fiducial[0]+=.01
    elif mutation=='shells': like.grid.centers[0]+=.001
    elif mutation=='angular': like.config=replace(like.config,n_mu=64)
    elif mutation=='model': like.model['fixed_physics']='different'
    elif mutation=='covariance': like.cells[0].covariance[0,0,0]*=2
    elif mutation=='local_fisher': like.cells[0].F_like=like.cells[0].F_like*2
    elif mutation=='missing_treatment': results.pop('release_h')
    with pytest.raises(ValueError): compare(a,b)


@pytest.mark.parametrize('mutation',['jacobian','propagation','mask','prior_matrix','active_names','fixed_leak'])
def test_resolution_rejects_changed_growth_prior_or_coordinate_mask(mutation):
    a,b=sample(),sample(32768); r=b[1]['conditional3']
    if mutation=='jacobian':
        for result in b[1].values():
            result.growth_jacobian[0,0]+=.01
            for suffix in ('marg','nuisance_fixed'):
                setattr(result,'growth_cov_'+suffix,result.growth_jacobian@getattr(result,'cov_cosmo_'+suffix)@result.growth_jacobian.T)
    elif mutation=='propagation': r.growth_cov_marg=np.diag(np.diag(r.growth_cov_marg))
    elif mutation=='mask': r.free_cosmology=('ln10_10_As','Mnu','h')
    elif mutation=='prior_matrix': r.F_prior_cosmo[0,0]=1.
    elif mutation=='active_names': r.active_parameter_names=tuple(reversed(r.active_parameter_names))
    elif mutation=='fixed_leak': r.cov_cosmo_marg[3,3]=1e-3
    with pytest.raises(ValueError): compare(a,b)


def test_each_resolution_reconstructs_its_own_covariance_and_reports_differences():
    a,b=sample(),sample(32768); like,_=a
    for c in like.cells:
        c.covariance*=1.001; c.F_like=c.F_like/1.001
    like.bin_F_like=[f/1.001 for f in like.bin_F_like]; like.F_like=like.F_like/1.001
    a=(like,projected(like))
    r=compare(a,b)
    assert r['status']=='passed'
    assert r['spectra'][0]['covariance']['whitened_frobenius']>0
    assert r['fisher']['global']['absolute_frobenius']>0
    assert len(r['fisher']['bins'])==8 and len(r['fisher']['cells'])==1
    assert r['treatments']['conditional3']['solves']['base']['global_solve']['scaled_residual']<1e-10


def test_optional_failure_coverage_is_explicit_and_inconsistent_success_fails():
    a,b=sample(),sample(32768)
    a[1]['marginal6_no_extra']=failed()
    assert compare(a,b)['status']=='failed'
    b[1]['marginal6_no_extra']=failed()
    r=compare(a,b)
    assert r['status']=='passed'
    assert r['treatments']['marginal6_no_extra']['status']=='unavailable'
    assert 'controlled' in r['treatments']['marginal6_no_extra']['failures']['base']['error']


def test_required_treatment_failure_cannot_pass_convergence():
    a,b=sample(),sample(32768)
    for results in (a[1],b[1]): results['marginal6']=failed('required projection failed')
    assert compare(a,b)['status']=='failed'


def test_ratio_rejects_different_models_even_when_each_case_is_consistent():
    b0,c0,b1,c1=sample(),sample(case='C'),sample(32768),sample(32768,case='C')
    for like,_ in (c0,c1): like.model['fixed_physics']='different'
    with pytest.raises(ValueError,match='model'):
        api().compare_resolution_ratios(b0[1],c0[1],b1[1],c1[1])


def test_both_reciprocal_ratios_use_the_finer_reference():
    old_b,old_c=sample(amplitude_error=2.),sample(amplitude_error=1.5,case='C')
    new_b,new_c=sample(32768,2.005),sample(32768,1.502,case='C')
    report=api().compare_resolution_ratios(old_b[1],old_c[1],new_b[1],new_c[1])
    assert report['status']=='passed'
    row=report['treatments']['conditional3']['cosmology_marg']
    assert set(row)=={'C_over_B','B_over_C'}
    for direction,coarse,fine in [('C_over_B',1.5/2.,1.502/2.005),('B_over_C',2./1.5,2.005/1.502)]:
        np.testing.assert_allclose(row[direction]['coarse'][0],coarse,rtol=1e-14)
        np.testing.assert_allclose(row[direction]['fine'][0],fine,rtol=1e-14)
        np.testing.assert_allclose(row[direction]['fractional_changes'][0],abs(coarse/fine-1),atol=1e-14)


def test_ratio_failure_is_not_hidden_by_individual_error_convergence():
    a,c=sample(amplitude_error=1.009),sample(amplitude_error=.9915,case='C')
    b,d=sample(32768),sample(32768,case='C')
    assert compare(a,b)['status']==compare(c,d)['status']=='passed'
    r=api().compare_resolution_ratios(a[1],c[1],b[1],d[1])
    assert r['status']=='failed'
    assert r['treatments']['conditional3']['cosmology_marg']['C_over_B']['status']=='failed'


def test_optional_missing_ratio_is_unavailable_not_a_numerical_pass():
    a,c=sample(),sample(case='C'); b,d=sample(32768),sample(32768,case='C')
    for r in (a[1],b[1]): r['release_h']=failed()
    report=api().compare_resolution_ratios(a[1],c[1],b[1],d[1])
    assert report['status']=='passed'
    assert report['treatments']['release_h']['status']=='unavailable'
    assert set(report['treatments']['release_h']['failures'])=={'base_B','refined_B'}
    assert report['successful_ratio_treatments']==6


def control_provenance():
    import hashlib
    from pfsfog.extended_artifacts import IDENTITY_KEYS
    return {key:('a'*40 if key=='project_head' else hashlib.sha256(key.encode()).hexdigest()) for key in IDENTITY_KEYS}


def test_saved_output_controls_attach_reconstructed_likelihood(tmp_path):
    from scripts import run_extended_cosmology as driver
    like,results=sample(32768); provenance=control_provenance()
    record=driver.save_control(tmp_path/'control',like,results,provenance=provenance)
    restored,outputs=driver.load_control(record,provenance=provenance)
    assert all(getattr(r,'likelihood',None) is restored for r in outputs.values())
    base,projected=sample()
    assert api().compare_ap_resolution(base,restored,projected,outputs)['status']=='passed'


@pytest.mark.parametrize('mutation',['growth_covariance','growth_shape','extra_array','missing_array','dtype'])
def test_saved_output_control_rejects_rehashed_inconsistent_arrays(tmp_path,mutation):
    from scripts import run_extended_cosmology as driver
    like,results=sample(32768); provenance=control_provenance()
    record=driver.save_control(tmp_path/'control',like,results,provenance=provenance)
    path=tmp_path/'control/arrays.npz'
    with np.load(path) as f: arrays={name:f[name] for name in f.files}
    if mutation=='growth_covariance': arrays['conditional3/growth_cov_marg'][0,1]+=.01
    elif mutation=='growth_shape': arrays['conditional3/growth_jacobian']=arrays['conditional3/growth_jacobian'][:,:3]
    elif mutation=='extra_array': arrays['unclaimed/reference']=np.ones(1)
    elif mutation=='missing_array': arrays.pop('conditional3/growth_jacobian')
    elif mutation=='dtype': arrays['conditional3/growth_z']=arrays['conditional3/growth_z'].astype('float32')
    np.savez(path,**arrays);record['arrays_sha256']=driver.file_hash(path)
    with pytest.raises(ValueError): driver.load_control(record,provenance=provenance)


def native3(like, conditional):
    """Controlled test algebra only; physical native3 builds use their own response."""
    ids=[like.names.index(n) for n in conditional.active_parameter_names]
    base=replace(like,config=replace(like.config,cosmology_basis='baseline3'),
        names=[like.names[i] for i in ids],parameters=[like.parameters[i] for i in ids],
        fiducial=like.fiducial[ids],F_like=like.F_like[np.ix_(ids,ids)],
        bin_F_like=[f[np.ix_(ids,ids)] for f in like.bin_F_like],
        cells=[replace(c,names=[c.names[i] for i in ids],F_like=c.F_like[np.ix_(ids,ids)],
                       derivatives=c.derivatives[:,:,ids]) for c in like.cells])
    from pfsfog.extended_artifacts import PRECISION_FIELDS,GROWTH_FIELDS
    result=SimpleNamespace(**{name:getattr(conditional,name).copy() for name in (*PRECISION_FIELDS,*GROWTH_FIELDS)})
    for field in ('cov_cosmo_marg','cov_cosmo_nuisance_fixed'): setattr(result,field,getattr(result,field)[:3,:3])
    result.growth_jacobian=result.growth_jacobian[:,:3]
    result.fisher_solver=conditional.fisher_solver
    return base,result


def test_direct_native3_identity_uses_strict_equal_resolution_rules():
    from pfsfog import extended_validation as ev
    assert hasattr(ev,'validate_native_baseline_identity'), 'direct native3 historical identity comparison missing'
    like,results=sample(8192); base,result=native3(like,results['conditional3'])
    other,output=copy.deepcopy((base,result))
    assert ev.validate_native_baseline_identity(base,result,other,output)['status']=='passed'
    output.growth_cov_marg*=1.005**2
    with pytest.raises(ValueError,match='identity'):
        ev.validate_native_baseline_identity(base,result,other,output)
    output=copy.deepcopy(result);other.config=replace(other.config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    with pytest.raises(ValueError,match='configuration'):
        ev.validate_native_baseline_identity(base,result,other,output)


def test_new_native3_check_keeps_historical_oracle_a_separate_comparison(tmp_path,monkeypatch):
    from scripts import run_extended_cosmology as driver
    from pfsfog import revision_forecast as forecast
    like,results=sample();base,result=native3(like,results['conditional3'])
    monkeypatch.setattr(forecast,'apply_revision_priors',lambda *args:result)
    record=driver.check_baseline_control(like,results['conditional3'],base,None,tmp_path,control_provenance())
    assert set(record['reference_checks'])=={'conditional3_native3_same_resolution'}
    assert record['reference_checks']['conditional3_native3_same_resolution']['status']=='passed'
