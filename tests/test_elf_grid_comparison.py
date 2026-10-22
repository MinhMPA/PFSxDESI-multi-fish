"""Known spectra and errors distinguish quadrature changes from physical changes."""
import copy
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest

TREATMENTS=('conditional3','release_h','release_omega_b','release_n_s',
            'marginal6','marginal6_weak10','marginal6_no_extra')
COSMO=['ln10_10_As','Mnu','Omegam','h','omega_b','n_s']
KINDS=('marginalized','nuisance_fixed')


def api():
    assert importlib.util.find_spec('scripts.elf_grid_comparison') is not None, 'grid comparison mathematics missing'
    return importlib.import_module('scripts.elf_grid_comparison')


def local(n=4):
    dk=.8/n;k=(np.arange(n)+.5)*dk
    mean=np.column_stack((1+k,2-k))
    D=np.stack((mean,np.column_stack((k,1+k))),axis=-1)
    C=np.array([np.diag([1+x,2+x])/dk for x in k])
    return dict(names=['amplitude','tilt'],k=k,dk=np.full(n,dk),mean=mean,D=D,C=C)


def outcomes():
    out={}
    for case in ('B','C'):
        out[case]={}
        for t in TREATMENTS:
            free=COSMO if t.startswith('marginal6') else COSMO[:3]+([t[8:]] if t.startswith('release_') else [])
            out[case][t]={'status':'passed','free_cosmology':free.copy(),
                **{kind:{'cosmology':dict.fromkeys(free,100.),'growth':[100.]*8} for kind in KINDS}}
    return out


def summary(out=None):
    from scripts.infer_elf_survey import constraint_ratios
    out=outcomes() if out is None else out
    return {'outcomes':out,'ratios':constraint_ratios(out)}


def request(nfft=1024,n_mu=32,dk=.00125):
    settings=dict(nfft=nfft,n_mu=n_mu,dk=dk)
    details={'settings':settings.copy(),'model_flags':{'fft_n':nfft,'do_irres':False},
             'engine_settings':{'fft_n':nfft,'do_irres':False},'source_file_hashes':{'theory':'a'},
             'reference_cosmology':{'h':.67},'emulator_input_hashes':{'model':'b'}}
    result={'schema':'elf-survey-physical-v1','scope':'full_B6_C6','cases':['B','C'],'cell_ids':None,
        'settings':settings,'source':{'details':details},
        'configurations':{case:{'dk':dk,'n_mu':n_mu,'nuisance_factor':1.} for case in ('B','C')},
        'expected':[{'key':'B:a','support':{'volume':100.,'local_names':['h']}}]}
    rehash(result)
    return result


def rehash(r):
    r['source']['identity_hash']=hashlib.sha256(json.dumps(r['source']['details'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def test_linear_shell_functions_and_covariance_kernel_compare_exactly():
    a,b=local(),local(8)
    mean,D,C=api().align_shells(a['k'],a['dk'],b['k'],b['dk'],b['mean'],b['D'],b['C'])
    for x,y in [(mean,a['mean']),(D,a['D']),(C,a['C'])]:np.testing.assert_allclose(x,y,rtol=1e-15,atol=1e-15)
    result=api().compare_local(a,b,shell=True)
    assert result['status']=='passed'
    assert result['covariance_maximum_spectral_change']<1e-14
    assert result['mean_relative']<1e-14
    assert set(result['columns'])==set(a['names'])


def test_shell_interpolation_never_extrapolates():
    a,b=local(),local(8);a['k'][0]=0.
    with pytest.raises(ValueError,match='extrapolat'):api().align_shells(a['k'],a['dk'],b['k'],b['dk'],b['mean'],b['D'],b['C'])


def test_wrong_bandwidth_covariance_normalization_is_not_convergence():
    a,b=local(),local(8);b['C']/=2
    result=api().compare_local(a,b,shell=True)
    assert result['status']=='failed'
    assert result['covariance_maximum_spectral_change']==pytest.approx(.5)


def test_one_changed_weak_response_is_named_and_fails():
    a=local();a['D'][:,:,1]*=1e-8;b=copy.deepcopy(a);b['D'][:,:,1]*=1.001
    result=api().compare_local(a,b)
    assert result['status']=='failed'
    assert result['columns']['tilt']['change']==pytest.approx(.001)
    assert result['columns']['amplitude']['change']==0.


def test_both_zero_responses_use_full_physical_mean_scale():
    a=local();a['D'][:,:,1]=0.;b=copy.deepcopy(a);b['D'][:,:,1]=1e-14
    result=api().compare_local(a,b)
    assert result['status']=='passed'
    assert result['columns']['tilt']['normalization']=='full_mean_for_both_zero'
    assert result['columns']['tilt']['change']>0.


def test_new_finite_response_from_exact_zero_fails_without_infinity():
    a=local();a['D'][:,:,1]=0.;b=copy.deepcopy(a);b['D'][:,:,1]=1e-3
    result=api().compare_local(a,b)
    assert result['status']=='failed' and result['columns']['tilt']['status']=='failed'
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('damage',['nan','asymmetric','indefinite','labels','k','dk','shape'])
def test_invalid_local_comparison_fails(damage):
    a=local();b=copy.deepcopy(a)
    if damage=='nan':b['D'][0,0,0]=np.nan
    elif damage=='asymmetric':b['C'][0,0,1]=.01
    elif damage=='indefinite':b['C'][0,0,0]=-1.
    elif damage=='labels':b['names'][1]='foreign'
    elif damage=='k':b['k'][0]*=1.01
    elif damage=='dk':b['dk']*=2
    else:b['D']=b['D'][:,:,:1]
    with pytest.raises(ValueError):api().compare_local(a,b)


def test_all_constraints_and_both_ratio_directions_are_enumerated():
    result=api().compare_constraints(summary(),summary())
    assert result['status']=='passed' and len(result['metrics'])==712
    assert result['unqualified']==[]
    assert result['maximum_relative_change']==0.


@pytest.mark.parametrize('case',['B','C'])
@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('field',['growth','cosmology'])
def test_exact_one_percent_is_rejected_everywhere(case,kind,field):
    out=outcomes();row=out[case]['marginal6'][kind][field]
    row[7 if field=='growth' else 'h']=101.
    result=api().compare_constraints(summary(),summary(out))
    assert result['status']=='failed'
    assert any(m['status']=='failed' and m['label'].startswith(case+'/marginal6/'+kind+'/'+field) for m in result['metrics'])


def test_reciprocal_only_failure_cannot_hide_behind_ratio_direction():
    out=outcomes();out['B']['marginal6']['marginalized']['growth'][3]=99.005
    result=api().compare_constraints(summary(),summary(out))
    changed={m['label']:m for m in result['metrics']}
    prefix='ratios/marginal6/marginalized/'
    assert changed[prefix+'B_over_C/growth/3']['status']=='passed'
    assert changed[prefix+'C_over_B/growth/3']['status']=='failed'
    assert result['status']=='failed'


@pytest.mark.parametrize('damage',['missing','basis','zero','negative','nan','growth_count','stale_ratio'])
def test_invalid_global_inputs_rejected(damage):
    s=summary();o=s['outcomes']
    if damage=='missing':o['B'].pop('release_h')
    elif damage=='basis':o['B']['release_h']['free_cosmology']=['h']
    elif damage=='growth_count':o['C']['marginal6']['nuisance_fixed']['growth'].pop()
    elif damage=='stale_ratio':s['ratios']['marginal6']['marginalized']['C_over_B']['growth'][0]=.9
    else:o['B']['marginal6']['marginalized']['growth'][0]={'zero':0.,'negative':-1.,'nan':np.nan}[damage]
    with pytest.raises(ValueError):api().compare_constraints(summary(),s)


def test_matching_optional_failures_are_explicit_and_never_fabricated():
    out=outcomes();out['B']['marginal6_no_extra']={'status':'failed','error':'singular'}
    s=summary(out);r=api().compare_constraints(s,s)
    assert r['status']=='passed'
    assert 'B/marginal6_no_extra' in r['unqualified']
    assert not any(m['label'].startswith('B/marginal6_no_extra/') for m in r['metrics'])
    assert api().compare_constraints(summary(),s)['status']=='failed'


def test_required_treatment_failure_is_rejected():
    out=outcomes();out['B']['conditional3']={'status':'failed','error':'singular'}
    with pytest.raises(ValueError,match='required'):api().compare_constraints(summary(out),summary(out))


@pytest.mark.parametrize('axis,kwargs',[('loop',{'nfft':2048}),('angular',{'n_mu':64}),('shell',{'dk':.000625}),('combined',{'nfft':2048,'n_mu':64,'dk':.000625})])
def test_only_declared_grid_refinement_is_permitted(axis,kwargs):
    r=api().validate_pair_contract(request(),request(**kwargs),axis)
    assert r['status']=='passed'


@pytest.mark.parametrize('damage',['source','emulator','model','volume','parameter','fiducial','prior','multiple','wrong_factor','duplicated_setting','digest'])
def test_fixed_physical_content_cannot_change_with_grid(damage):
    a,b=request(),request(nfft=2048)
    if damage=='source':b['source']['details']['source_file_hashes']['theory']='changed'
    elif damage=='emulator':b['source']['details']['emulator_input_hashes']['model']='changed'
    elif damage=='model':b['source']['details']['model_flags']['do_irres']=True
    elif damage=='volume':b['expected'][0]['support']['volume']=101.
    elif damage=='parameter':b['expected'][0]['support']['local_names']=['Mnu']
    elif damage=='fiducial':b['source']['details']['reference_cosmology']['h']=.7
    elif damage=='prior':b['configurations']['B']['nuisance_factor']=2.
    elif damage=='multiple':b=request(nfft=2048,n_mu=64)
    elif damage=='wrong_factor':b=request(nfft=4096)
    elif damage=='duplicated_setting':b['source']['details']['engine_settings']['fft_n']=1024
    else:b['source']['identity_hash']='foreign'
    if damage!='digest':rehash(b)
    with pytest.raises(ValueError):api().validate_pair_contract(a,b,'loop')


@pytest.mark.parametrize('coarse,fine',[(3.,3.03),(10.,10.1),(3.,2.97),(10.,9.9)])
def test_common_growth_one_percent_boundary_rejects_binary64_roundoff(coarse,fine):
    a,b=outcomes(),outcomes()
    for case in ('B','C'):
        a[case]['marginal6']['marginalized']['growth'][0]=coarse
        b[case]['marginal6']['marginalized']['growth'][0]=fine
    result=api().compare_constraints(summary(a),summary(b))
    assert result['status']=='failed'
    assert not any(m['status']=='failed' for m in result['metrics'] if m['label'].startswith('ratios/'))


def test_ratio_at_one_percent_boundary_rejects_roundoff():
    a,b=outcomes(),outcomes()
    a['B']['marginal6']['marginalized']['growth'][0]=3.
    a['C']['marginal6']['marginalized']['growth'][0]=1.
    b['B']['marginal6']['marginalized']['growth'][0]=3.015
    b['C']['marginal6']['marginalized']['growth'][0]=3.015/3.03
    result=api().compare_constraints(summary(a),summary(b))
    metric=next(m for m in result['metrics'] if m['label']=='ratios/marginal6/marginalized/B_over_C/growth/0')
    assert metric['status']=='failed'


def test_resolved_change_below_one_percent_still_passes():
    a,b=outcomes(),outcomes()
    for case in ('B','C'):b[case]['marginal6']['marginalized']['growth'][0]=100.999999
    assert api().compare_constraints(summary(a),summary(b))['status']=='passed'
