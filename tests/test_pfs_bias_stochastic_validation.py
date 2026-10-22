"""Isotropic shot noise must be tested as zero without accepting real leakage."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.pfs_bias_stochastic_validation import (
    cross_shot_zero_support, replay_saved_derivatives, allow_original_failure,
)


NAME='bin01:DESI-LRG|PFS-ELG:Pshot_cross'
PAIR=['DESI-LRG','PFS-ELG']


def test_constant_cross_shot_noise_has_only_monopole_and_exact_pair_support():
    names=[NAME,'bin01:DESI-LRG|PFS-ELG:a2_cross','h']
    pairs=[['DESI-LRG','DESI-LRG'],PAIR,['PFS-ELG','PFS-ELG']]
    assert cross_shot_zero_support(names,pairs,alphas=[1.,1.])==(
        (NAME,tuple(PAIR),2),(NAME,tuple(PAIR),4))
    assert cross_shot_zero_support(names,pairs,alphas=[1.01,1.])==()
    assert cross_shot_zero_support(names,pairs,alphas=[np.nan,1.])==()


def saved_fixture():
    inputs={'metadata':{'local_names':[NAME],'retained_pairs':[PAIR],'cell_id':'one'},
            'fiducials':{NAME:0.},'steps':{NAME:[.001,.0005,.00025]},
            'analytic_zero_response':{'alphas':[1.,1.],'support':[]}}
    steps=np.array([[.001,.0005,.00025]])
    mean=np.array([[10.,4.,2.]])
    ad=np.array([[[1.],[0.],[0.]]])
    fd=np.broadcast_to(ad.transpose(2,0,1)[:,None],(1,3,1,3)).copy()
    fd[0,:,0,1]=[0.,3e-8,4e-8]
    plus=mean[None,None]+steps[:,:,None,None]*fd
    minus=mean[None,None]-steps[:,:,None,None]*fd
    a={'D':ad,'C':np.eye(3)[None],'mean':mean,'fd_steps':steps,
       'fd_plus':plus,'fd_minus':minus,
       'fd_derivatives':(plus-minus)/(2*steps[:,:,None,None])}
    return inputs,a


def test_replay_retains_relative_failures_but_accepts_guarded_exact_zero():
    inputs,a=saved_fixture()
    old=replay_saved_derivatives(inputs,a,configuration='case',add_cross_shot_zeros=False)
    assert old['status']=='failed'
    new=replay_saved_derivatives(inputs,a,configuration='case')
    assert new['status']=='passed'
    assert new['relative_tolerance']==1e-3 and new['zero_tolerance']==1e-6
    broken=deepcopy(a);broken['D'][0,0,0]*=1.01
    assert replay_saved_derivatives(inputs,broken,configuration='case')['status']=='failed'


def test_declared_zero_rejects_ad_and_fd_leakage_even_when_they_agree():
    inputs,a=saved_fixture();a['D'][0,1,0]=.01
    a['fd_plus'][0,:,0,1]=4.+.01*a['fd_steps'][0]
    a['fd_minus'][0,:,0,1]=4.-.01*a['fd_steps'][0]
    a['fd_derivatives']=(a['fd_plus']-a['fd_minus'])/(2*a['fd_steps'][:,:,None,None])
    assert replay_saved_derivatives(inputs,a,configuration='case')['status']=='failed'


def test_unrelated_failure_is_never_reclassified_as_shot_noise_roundoff():
    failure={'column':NAME,'pair':PAIR,'ell':2,'reason':'pair/ell whitened FD'}
    measurement={'status':'failed','ad_fd':{'failures':[failure],
        'column_results':[{'two_successive_pass':True}]},
        'h_split_fd':{'intrinsic':{'status':'passed'},'geometry':{'status':'passed'}},
        'chain_relative_error':1e-14,'chain_tolerance':1e-10}
    worker={'status':'failed','mean_status':'failed','covariance_status':'passed'}
    zeros=((NAME,tuple(PAIR),2),(NAME,tuple(PAIR),4))
    allow_original_failure(worker,measurement,1,zeros)
    for change in ({'ell':0},{'column':'h'},{'reason':'full-vector whitened FD'}):
        bad=deepcopy(measurement);bad['ad_fd']['failures'][0].update(change)
        with pytest.raises(ValueError):allow_original_failure(worker,bad,1,zeros)
    with pytest.raises(ValueError):allow_original_failure({**worker,'reason':'source changed'},measurement,1,zeros)
    with pytest.raises(ValueError):allow_original_failure(worker,measurement,137,zeros)


FAILED=Path(__file__).resolve().parents[1]/'results/pfs-bias-broad-20261010-01/physical/angular64/cells/w0-adopted_relations-equal-DESI-LRG'


@pytest.mark.skipif(not (FAILED/'arrays.npz').exists(),reason='requires retained failed physical measurement')
def test_real_failed_quadrupole_replays_without_changing_any_spectrum():
    import json
    inputs=json.loads((FAILED/'inputs.json').read_text())
    with np.load(FAILED/'arrays.npz') as saved:a={k:saved[k] for k in saved.files}
    old=replay_saved_derivatives(inputs,a,configuration='w0-adopted_relations-equal-DESI-LRG',add_cross_shot_zeros=False)
    assert old['status']=='failed'
    new=replay_saved_derivatives(inputs,a,configuration='w0-adopted_relations-equal-DESI-LRG')
    assert new['status']=='passed'
    assert new['column_results']==old['column_results']
    assert all(f['column']==NAME and f['ell']==2 for f in old['failures'])
