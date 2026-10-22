"""A broad-study certificate requires every promised physical/inference control."""
from copy import deepcopy

import pytest

from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.run_pfs_bias_broad import GRIDS
from scripts.check_pfs_bias_broad_checkpoint import summarize_controls, validate_stage_exits


def controls_fixture():
    grids={}
    for grid in GRIDS:
        points={}
        for row in broad_schedule():
            names=('D','C','C22') if row['window'][1]<=1.6 else ('H21','C21','C22')
            point={'key':row['key'],'experiment':row,'profiles':{n:{} for n in names},
                   'cross_information_check':{'status':'passed'}}
            if row['ratio']==1.:
                point['reference_physical_check']={'status':'passed'}
                for out in point['profiles'].values():out['original_bias_covariance_check']='passed'
            if grid=='shell304' and (row['ratio']==2. or row['equal_to']):
                for out in point['profiles'].values():out['full_inverse_control']={'status':'passed'}
            points[row['key']]=point
        grids[grid]={'points':points}
    return grids


def test_independent_control_inventory_cannot_be_replaced_by_report_constants():
    grids=controls_fixture()
    assert summarize_controls(grids)=={'physical_populations':175,'profile_calculations':525,
        'original_array_controls':15,'original_global_controls':45,
        'full_inverse_controls':42,'cross_information_controls':175}
    for kind in ('population','profile','original','full_inverse','cross'):
        bad=deepcopy(grids);points=bad['shell304']['points']
        base=next(p for p in points.values() if p['experiment']['ratio']==1.)
        full=next(p for p in points.values() if p['experiment']['ratio']==2.)
        if kind=='population':points.pop(base['key'])
        elif kind=='profile':full['profiles'].pop('C22')
        elif kind=='original':base.pop('reference_physical_check')
        elif kind=='full_inverse':full['profiles']['C22'].pop('full_inverse_control')
        else:base['cross_information_check']['status']='failed'
        with pytest.raises(ValueError):summarize_controls(bad)


def test_report_exit_and_complete_order_are_required_for_numerical_completion():
    rows=[{'stage':stage,'actual_exit':0} for stage in ('inference','common','comparison','report')]
    good={'status':'numerical_stages_finished_review_pending','after_stochastic_validation':True,'stages':rows}
    validate_stage_exits(good)
    for change in ({'status':'running_report'},{'stages':rows[:-1]},
                   {'stages':[rows[0]]*4},{'after_stochastic_validation':False}):
        with pytest.raises(ValueError):validate_stage_exits({**good,**change})
    bad=deepcopy(good);bad['stages'][2]['actual_exit']=1
    with pytest.raises(ValueError):validate_stage_exits(bad)
