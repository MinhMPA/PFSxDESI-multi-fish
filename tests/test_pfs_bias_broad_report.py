"""Numerically unresolved bias responses must not become a claimed preference."""
from copy import deepcopy

import pytest

from scripts.run_pfs_bias_broad import GRIDS, REFINEMENTS
from scripts.report_pfs_bias_broad import (
    numerical_interval, unresolved_groups, selected_curve, qualified_comparisons,
)


def test_resolution_uses_all_refinements_and_fine_grid_spread():
    values={'loop2048':.2,'loop4096':.201,'angular64':.25,'shell304':.251,'final-loop':.252}
    result=numerical_interval(values)
    assert result['value']==.252
    assert result['half_width']==pytest.approx(.052)
    assert result['lower']==pytest.approx(.2)
    assert result['upper']==pytest.approx(.304)
    assert numerical_interval(dict.fromkeys(GRIDS,0.))['half_width']==1e-5
    assert numerical_interval(dict.fromkeys(GRIDS,2.))['half_width']==.02
    with pytest.raises(ValueError):numerical_interval({'final-loop':.2})


def test_unresolved_groups_are_connected_even_if_a_wide_interval_bridges_later():
    rows=[{'key':'high','value':2.5,'half_width':.01},
          {'key':'low','value':-2.5,'half_width':.01},
          {'key':'bridge','value':0.,'half_width':3.},
          {'key':'separate','value':6.,'half_width':.1}]
    groups=unresolved_groups(rows)
    assert groups[0]['members']==['separate']
    assert set(groups[1]['members'])=={'high','low','bridge'}
    assert len(groups)==2


def test_original_population_is_reused_for_each_prescription_without_double_counting():
    rows=[{'key':'base','window':[.7,.8],'prescription':'b1_only','ratio':1.,'parameter':'Mnu'},
          {'key':'low','window':[.7,.8],'prescription':'b1_only','ratio':.8,'parameter':'Mnu'},
          {'key':'high','window':[.7,.8],'prescription':'adopted_relations','ratio':2.,'parameter':'Mnu'}]
    selected=selected_curve(rows,[.7,.8],'adopted_relations','Mnu')
    assert [r['key'] for r in selected]==['base','high']
    assert selected[0]['reused_original_population'] is True
    assert all(r['on_tested_boundary'] for r in selected)
    assert 'on_tested_boundary' not in rows[0]


def test_two_resolution_or_missing_comparisons_do_not_qualify_a_broad_report():
    summary={'status':'passed','full_bias_grid_qualification':True,
             'refinements':REFINEMENTS,'reports':{a+'-to-'+b+'.json':'hash' for a,b,_ in REFINEMENTS}}
    qualified_comparisons(summary)
    for field,value in [('status','failed'),('full_bias_grid_qualification',False),
                        ('refinements',REFINEMENTS[:1]),('reports',{})]:
        changed=deepcopy(summary);changed[field]=value
        with pytest.raises(ValueError):qualified_comparisons(changed)
