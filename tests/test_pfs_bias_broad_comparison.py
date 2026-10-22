"""Convergence must compare the same population and physical wavenumbers."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.pfs_bias_broad_comparison import (
    check_population_pair, query_reconstruction, project_local,
    compare_profile_constraints, validate_common_execution,
)
from test_pfs_bias_broad_inference import point_and_expected


def test_a_refinement_cannot_change_population_or_support():
    a={'key':'x','support':{'z':1.5},'experiment':{'ratio':2.},'fiducials':{'b1':3.}}
    check_population_pair(a,deepcopy(a))
    for field in ('support','experiment','fiducials'):
        b=deepcopy(a);b[field]['extra']=1
        with pytest.raises(ValueError):check_population_pair(a,b)


def local_data():
    k=np.array([.1,.2]);dk=np.array([.01,.01])
    c=np.array([[2.,.3,.2],[.3,1.,.1],[.2,.1,1.]])
    return {'k':k,'dk':dk,'names':['As','Mnu','bias'],
            'mean':np.ones((2,3)),'D':np.ones((2,3,3)),
            'C':np.broadcast_to(c,(2,3,3)).copy()}


def test_direct_shell_reconstruction_preserves_covariance_bandwidth():
    data=local_data();direct={'k':data['k'],'mean':data['mean'],'D':data['D'],
                             'covariance_kernel':data['C']*data['dk'][:,None,None]}
    assert query_reconstruction(data,direct)['status']=='passed'
    direct['covariance_kernel']*=2
    assert query_reconstruction(data,direct)['status']=='failed'


def test_direct_shell_check_rejects_missing_wavenumber():
    data=local_data();direct={'k':np.array([.11,.21]),'mean':data['mean'],'D':data['D'],
                             'covariance_kernel':data['C']*.01}
    with pytest.raises(ValueError,match='cover'):query_reconstruction(data,direct)


def test_auto_projection_keeps_mutual_covariance_and_selected_derivatives():
    data=local_data();mapping={'source_shells':[0,1],'source_rows':[0,1],
                              'source_columns':[0,2],'independent_autos':False}
    out=project_local(data,mapping)
    assert out['names']==['As','bias']
    np.testing.assert_array_equal(out['C'],data['C'][:,:2,:2])
    assert np.all(out['C'][:,0,1]==.3)
    np.testing.assert_array_equal(out['D'],data['D'][:,:2][:,:,[0,2]])
    mapping['source_rows']=[0,0]
    with pytest.raises(ValueError):project_local(data,mapping)


def test_all_profile_errors_and_ratios_include_strict_one_percent_boundary():
    a,_=point_and_expected();b=deepcopy(a)
    assert compare_profile_constraints(a,b)['status']=='passed'
    for out in b['profiles'].values():
        for key in ('cosmological_covariance','cosmological_covariance_nuisance_fixed',
                    'growth_covariance','growth_covariance_nuisance_fixed'):
            out[key]*=1.01**2
    report=compare_profile_constraints(a,b)
    assert report['status']=='failed'
    labels={r['label'] for r in report['metrics']}
    assert any('C22/nuisance_fixed/growth/' in label for label in labels)
    assert any('ratio/D_over_C/' in label for label in labels)


def test_missing_or_failed_common_worker_prevents_qualification():
    keys=['a','b']
    good={'status':'passed','workers':[{'key':k,'actual_exit':0} for k in keys]}
    validate_common_execution(good,keys)
    for bad in ({'status':'passed','workers':good['workers'][:1]},
                {'status':'passed','workers':[{'key':'a','actual_exit':1},{'key':'b','actual_exit':0}]},
                {'status':'passed','workers':[good['workers'][0]]*2}):
        with pytest.raises(ValueError):validate_common_execution(bad,keys)
