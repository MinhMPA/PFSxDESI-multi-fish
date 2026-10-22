"""Independent checks of the wider scan's cosmological controls and saved results."""
from copy import deepcopy

import numpy as np
import pytest

from pfsfog.extended_artifacts import write_json
from scripts.elf_study_report import all_profiles
from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.pfs_bias_broad_inference import (
    full_inverse_required, require_grid_exit, verify_point,
)
from scripts.run_pfs_density_scan import constraint_row
from scripts.run_pfs_bias_broad import GRIDS


def test_full_inverse_controls_cover_both_prescriptions_at_large_and_equal_bias():
    rows=broad_schedule()
    selected=[r for r in rows if full_inverse_required('shell304',r)]
    assert len(selected)==14
    assert len(selected)*3==42
    assert sum(r['ratio']==2. for r in selected)==6
    assert sum(r['equal_to'] is not None for r in selected)==8
    assert not any(full_inverse_required('loop2048',r) for r in rows)


def test_a_saved_success_without_an_actual_controller_exit_cannot_supply_inference(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):require_grid_exit(tmp_path,'physical','shell304')
    path=tmp_path/'physical-execution.json'
    for records in ([],[{'grid':'shell304','actual_exit':1}],
                    [{'grid':'other','actual_exit':0}],
                    [{'grid':'shell304','actual_exit':0}]*2):
        write_json(path,{'status':'running','controllers':records})
        with pytest.raises(ValueError):require_grid_exit(tmp_path,'physical','shell304')
    write_json(path,{'status':'running','controllers':[{'grid':'shell304','actual_exit':0}]})
    assert require_grid_exit(tmp_path,'physical','shell304')['actual_exit']==0


def test_repaired_inference_waits_for_all_validation_exits_separately_from_raw_physics(tmp_path):
    write_json(tmp_path/'physical-execution.json',{'status':'passed',
        'controllers':[{'grid':g,'actual_exit':0} for g in GRIDS]})
    with pytest.raises((ValueError,FileNotFoundError)):
        require_grid_exit(tmp_path,'validated_physical','shell304')
    folder=tmp_path/'stochastic-validation';folder.mkdir()
    rows=[{'grid':g,'actual_exit':0} for g in GRIDS]
    write_json(folder/'execution.json',{'status':'running','controllers':rows[:-1]})
    with pytest.raises((ValueError,FileNotFoundError)):
        require_grid_exit(tmp_path,'validated_physical','shell304')
    write_json(folder/'execution.json',{'status':'passed','controllers':rows})
    assert require_grid_exit(tmp_path,'validated_physical','shell304')['actual_exit']==0
    rows[0]['actual_exit']=1
    write_json(folder/'execution.json',{'status':'failed','controllers':rows})
    with pytest.raises(ValueError):require_grid_exit(tmp_path,'validated_physical','shell304')


def point_and_expected():
    row=next(r for r in broad_schedule() if r['ratio']==.8 and r['window']==[1.4,1.6])
    expected={'key':row['key'],'experiment':deepcopy(row)}
    point={'status':'passed','key':row['key'],'experiment':deepcopy(row),'grid':'final-loop','profiles':{}}
    for name in ('D','C','C22'):
        z=np.mean(all_profiles()[name].nominal_bins,axis=1)
        j=np.column_stack([np.ones(len(z)),z,z*z])
        cov=np.array([[2.,.1,0],[.1,1.,.1],[0,.1,.5]])
        fixed=.5*cov
        out={'cosmological_covariance':cov,'cosmological_covariance_nuisance_fixed':fixed,
             'global_information':np.linalg.inv(cov),'growth_z':z,'growth_jacobian':j,
             'growth_covariance':j@cov@j.T,'growth_covariance_nuisance_fixed':j@fixed@j.T,
             'changed_prior_widths':{},'density_and_prior_widths_fixed':True}
        out['sigma']=constraint_row(out);point['profiles'][name]=out
    return point,expected


def test_saved_growth_and_precision_are_independently_reconstructed():
    point,expected=point_and_expected()
    assert verify_point(point,expected,'final-loop')['status']=='passed'
    point['profiles']['C22']['growth_covariance'][0,1]+=.01
    with pytest.raises(ValueError):verify_point(point,expected,'final-loop')


@pytest.mark.parametrize('mutation',['grid','population','missing_profile','prior','precision','growth_labels','sigma'])
def test_mislabeled_or_inconsistent_inference_cannot_be_reported(mutation):
    point,expected=point_and_expected()
    if mutation=='grid':point['grid']='shell304'
    elif mutation=='population':point['experiment']['ratio']=2.
    elif mutation=='missing_profile':point['profiles'].pop('D')
    elif mutation=='prior':point['profiles']['C22']['changed_prior_widths']={'b1':.1}
    elif mutation=='precision':point['profiles']['C22']['global_information'][0,0]*=1.01
    elif mutation=='growth_labels':point['profiles']['C22']['growth_z'][0]+=.1
    else:point['profiles']['C22']['sigma']['Mnu']*=1.01
    with pytest.raises(ValueError):verify_point(point,expected,'final-loop')
