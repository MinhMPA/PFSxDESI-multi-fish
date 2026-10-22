"""Physical interpretation must follow the corrected common reference."""
import copy
import numpy as np
import pytest


def density_points():
    from scripts.run_pfs_density_scan import FACTORS
    from scripts.pfs_density_inference import PRIOR_MODES
    points={}
    for i in range(8):
        for name,q in FACTORS.items():
            sigma={p:2*q**(-.01*(i+1)) for p in ('ln10_10_As','Mnu','Omegam','fsigma8_z1.1')}
            points[f'w{i:02d}-{name}']={'status':'passed','window_index':i,'factor':q,
                'profiles':{'C24':{'count_at_q1':1000*(i+1),'prior_modes':{
                    m:{'sigma':sigma} for m in PRIOR_MODES}}}}
    return points


def test_density_elasticity_uses_full_table_reference_and_retains_failed_variation():
    from scripts.takada_report import density_responses
    from scripts.run_takada_density_scan import density_windows
    points=density_points();points['w07-times10']={'status':'failed','error':'nonpositive field'}
    rows=density_responses(points,density_windows(.6,2.4),'C24')
    first=rows[0];last=next(r for r in rows if r['window_index']==7)
    assert first['z_min']==.6 and last['z_max']==2.4
    assert first['comparison_profile']=='C24'
    assert first['elasticity']==pytest.approx(.01)
    assert first['per_added_galaxy']==pytest.approx(1e-5)
    assert last['finite_error_reduction_percent']['times10'] is None
    assert last['finite_failures']['times10']=='nonpositive field'
    assert last['step_check']=='passed'


def test_failed_small_step_cannot_produce_a_density_ranking():
    from scripts.takada_report import density_responses
    from scripts.run_takada_density_scan import density_windows
    points=density_points();points['w00-ln_m005']={'status':'failed','error':'bad derivative'}
    with pytest.raises(ValueError,match='local density response'):
        density_responses(points,density_windows(.6,2.4),'C24')


def test_resolution_comparison_rejects_missing_values_and_one_percent_boundary():
    from scripts.takada_report import compare_values
    assert compare_values({'x':2.},{'x':2.019})['status']=='passed'
    assert compare_values({'x':2.},{'x':2.02})['status']=='failed'
    with pytest.raises(ValueError,match='inventory'):
        compare_values({'x':2.},{'y':2.})


def test_complete_forecast_comparison_includes_new_tail_and_both_ratio_orders():
    from scripts.takada_report import main_values,compare_values
    from scripts.run_takada_revision import ROOT
    import json
    # Individually reconstructed profiles from the preserved diagnostic run.
    p=ROOT/'results/takada-density-full-20261014-01/inference/loop2048/summary.json'
    outcomes=json.loads(p.read_text())['outcomes']
    values=main_values(outcomes)
    assert values['C24/marginalized/cosmology/1']==outcomes['C24']['marginalized']['cosmology'][1]
    a=values['ratios/C21_over_C24/marginalized/cosmology/1']
    b=values['ratios/C24_over_C21/marginalized/cosmology/1']
    assert a*b==pytest.approx(1.)
    altered=copy.deepcopy(outcomes);altered['C24']['marginalized']['cosmology'][1]*=1.02
    assert compare_values(values,main_values(altered))['status']=='failed'
