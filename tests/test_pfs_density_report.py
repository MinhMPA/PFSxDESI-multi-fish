"""A density ranking compares one global survey and records numerical limits."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.run_pfs_density_scan import FACTORS


def synthetic_points():
    points={}
    for i in range(8):
        for label,q in FACTORS.items():
            modes={}
            for mode in ('dimensionless','fixed_physical'):
                exponent=np.array([.01,.02,.03,.04])*(i+1)*(1 if mode=='dimensionless' else .8)
                sigma=np.array([.1,.2,.3,.4])*q**(-exponent)
                modes[mode]={'sigma':dict(zip(('ln10_10_As','Mnu','Omegam','fsigma8_z1.1'),sigma))}
            points[f'w{i:02d}-{label}']={'status':'passed','factor':q,'window_index':i,
                'profiles':{'C22':{'count_at_q1':1000.*(i+1),'prior_modes':modes}}}
    return points


def test_density_elasticity_recovers_power_law_and_per_galaxy_scaling():
    from scripts.report_pfs_density_scan import response_rows
    rows=response_rows(synthetic_points())
    assert len(rows)==64
    for row in rows:
        j=['ln10_10_As','Mnu','Omegam','fsigma8_z1.1'].index(row['parameter'])+1
        expected=.01*j*(row['window_index']+1)*(1 if row['prior_mode']=='dimensionless' else .8)
        assert row['elasticity']==pytest.approx(expected,abs=1e-13)
        assert row['per_added_galaxy']==pytest.approx(expected/(1000*(row['window_index']+1)),abs=1e-15)
        assert row['step_check']=='passed'
        assert row['comparison_profile']=='C22'


@pytest.mark.parametrize('damage',['missing','failed','wrong_factor','wrong_count','nonpositive'])
def test_response_rejects_missing_unqualified_or_inconsistent_inputs(damage):
    from scripts.report_pfs_density_scan import response_rows
    points=synthetic_points();p=points['w00-ln_p005']
    if damage=='missing':points.pop('w00-ln_p005')
    if damage=='failed':p['status']='failed'
    if damage=='wrong_factor':p['factor']=2.
    if damage=='wrong_count':p['profiles']['C22']['count_at_q1']=2000.
    if damage=='nonpositive':p['profiles']['C22']['prior_modes']['dimensionless']['sigma']['Mnu']=0.
    with pytest.raises(ValueError):response_rows(points)


def test_step_dependence_is_reported_as_a_failure():
    from scripts.report_pfs_density_scan import response_rows
    points=synthetic_points()
    points['w00-ln_p010']['profiles']['C22']['prior_modes']['dimensionless']['sigma']['Mnu']*=.95
    row=next(r for r in response_rows(points) if r['window_index']==0 and r['prior_mode']=='dimensionless' and r['parameter']=='Mnu')
    assert row['step_check']=='failed'


def test_grid_comparison_checks_responses_as_well_as_absolute_errors():
    from scripts.report_pfs_density_scan import response_rows, compare_grids
    a=synthetic_points();b=deepcopy(a)
    good=compare_grids(a,b,response_rows(a),response_rows(b))
    assert good['status']=='passed'
    b['w00-ln_p005']['profiles']['C22']['prior_modes']['dimensionless']['sigma']['Mnu']*=.999
    assert compare_grids(a,b,response_rows(a),response_rows(b))['status']=='failed'


def test_total_pfs_gains_use_the_fixed_desi_reference():
    from scripts.report_pfs_density_scan import total_pfs_rows
    points=synthetic_points();baseline=dict(zip(('ln10_10_As','Mnu','Omegam','fsigma8_z1.1'),[.2,.4,.6,.8]))
    rows=total_pfs_rows(points,baseline)
    zero=[r for r in rows if r['factor']==1.]
    assert len(zero)==16
    assert all(v==pytest.approx(50.) for row in zero for v in row['error_reduction_percent'].values())
    doubled=next(r for r in rows if r['window_index']==0 and r['factor']==2. and r['prior_mode']=='dimensionless')
    assert doubled['error_reduction_percent']['Mnu']==pytest.approx(100*(1-.5*2**(-.02)))


def test_saved_density_covariance_must_be_symmetric_and_reconstruct_growth():
    from scripts.report_pfs_density_scan import verify_calculation
    cov=np.diag([1.,4.,9.]);j=np.array([[1.,2.,3.],[.5,1.,1.]])
    c={'cosmological_covariance':cov,'global_information':np.linalg.inv(cov),
       'growth_jacobian':j,'growth_covariance':j@cov@j.T,'growth_z':[.9,1.1],
       'sigma':dict(zip(('ln10_10_As','Mnu','Omegam','fsigma8_z1.1'),[1.,2.,3.,np.sqrt(13.25)]))}
    verify_calculation(c)
    broken=deepcopy(c);broken['growth_covariance'][1,1]+=1.
    with pytest.raises(ValueError,match='growth'):verify_calculation(broken)
    broken=deepcopy(c);broken['cosmological_covariance'][0,1]=.01
    broken['global_information']=np.linalg.inv(broken['cosmological_covariance'])
    broken['growth_covariance']=j@broken['cosmological_covariance']@j.T
    broken['sigma']['fsigma8_z1.1']=np.sqrt(broken['growth_covariance'][1,1])
    with pytest.raises(ValueError,match='symmetric'):verify_calculation(broken)


def synthetic_comparison_points():
    from scripts.run_pfs_density_scan import window_profiles
    points=synthetic_points()
    for p in points.values():
        for name in window_profiles(p['window_index']):
            p['profiles'][name]=deepcopy(p['profiles']['C22'])
    return points


def synthetic_references():
    from scripts.report_pfs_density_scan import QUANTITIES
    return {'B':dict(zip(QUANTITIES,[.2,.4,.6,.8])),
            'C21':dict(zip(QUANTITIES,[.15,.3,.45,.6]))}


def test_baseline_comparisons_keep_low_high_and_tail_denominators():
    from scripts.report_pfs_density_scan import baseline_gain_rows
    rows=baseline_gain_rows(synthetic_comparison_points(),synthetic_references())
    for i,names in [(0,{'B_to_C22','B_to_D','B_to_C'}),
                    (5,{'B_to_C22','B_to_H21','B_to_C21'}),
                    (7,{'B_to_C22','C21_to_C22'})]:
        own=[r for r in rows if r['window_index']==i and r['factor']==1.]
        assert {r['comparison'] for r in own}==names
        for r in own:
            expected=100/3 if r['comparison']=='C21_to_C22' else 50.
            assert all(v==pytest.approx(expected) for v in r['error_reduction_percent'].values())
            for p in r['error_reduction_percent']:
                assert r['ratio_before_over_after'][p]*r['ratio_after_over_before'][p]==pytest.approx(1.)


def test_grid_checks_catch_unstable_baseline_ratio_with_stable_individual_errors():
    from scripts.report_pfs_density_scan import response_rows,compare_grids
    a=synthetic_comparison_points();b=deepcopy(a)
    refa=synthetic_references();refb=deepcopy(refa)
    for point in b.values():
        for profile in point['profiles'].values():
            for mode in profile['prior_modes'].values():mode['sigma']['Mnu']*=1.009
    refb['B']['Mnu']*=.991
    result=compare_grids(a,b,response_rows(a),response_rows(b),refa,refb)
    assert result['status']=='failed'
    failed=[m for m in result['metrics'] if m['status']=='failed']
    assert failed and all('B_to_' in m['label'] for m in failed)
    assert any('before_over_after' in m['label'] for m in failed)
    assert any('after_over_before' in m['label'] for m in failed)


def test_equal_per_galaxy_responses_are_an_unresolved_group():
    from scripts.report_pfs_density_scan import response_rows,density_rankings,GRID_PROFILES
    rows={g:response_rows(synthetic_points()) for g in GRID_PROFILES}
    ranked=density_rankings(rows)
    for mode in ('dimensionless','fixed_physical'):
        for parameter in ('ln10_10_As','Mnu','Omegam','fsigma8_z1.1'):
            assert ranked[mode][parameter]['per_added_galaxy']==[list(range(8))]
            assert ranked[mode][parameter]['elasticity']==[[i] for i in reversed(range(8))]
    for r in rows['shell304']:
        assert r['per_added_galaxy_numerical_resolution']==pytest.approx(r['elasticity_numerical_resolution']/r['count_at_q1'])


def test_wide_resolution_interval_joins_earlier_disjoint_ranking_groups():
    from scripts.report_pfs_density_scan import response_rows,density_rankings,GRID_PROFILES
    rows={g:response_rows(synthetic_points()) for g in GRID_PROFILES}
    # The low-count window has the smallest central response but its interval
    # overlaps both earlier windows. All three belong to one connected group.
    values={0:(.009,1e6),1:(.007,1e6),2:(.000012,2000.)}
    for grid_rows in rows.values():
        for r in grid_rows:
            if r['prior_mode']=='dimensionless' and r['parameter']=='fsigma8_z1.1' and r['window_index'] in values:
                e,n=values[r['window_index']]
                r.update(elasticity=e,count_at_q1=n,per_added_galaxy=e/n,step_difference=0.)
    for reverse in (False,True):
        ordered={g:list(reversed(v)) if reverse else deepcopy(v) for g,v in rows.items()}
        result=density_rankings(ordered)
        assert result['dimensionless']['fsigma8_z1.1']['per_added_galaxy']==[[3,4,5,6,7],[0,1,2]]


@pytest.mark.parametrize('failure',['result_mutation','request_mutation','figure_error'])
def test_reporting_failure_never_publishes_a_passed_result(tmp_path,monkeypatch,failure):
    import json
    from scripts import report_pfs_density_scan as m
    (tmp_path/'request.json').write_text('{}')
    bound=tmp_path/'point.json';bound.write_text('original')
    original=m.digest(bound);points=synthetic_comparison_points()
    monkeypatch.setattr(m,'load_request',lambda output: ({'study':'unused'},None))
    monkeypatch.setattr(m,'read_grid',lambda *args:(deepcopy(points),{str(bound):original}))
    monkeypatch.setattr(m,'read_references',lambda *args:(synthetic_references(),{}),raising=False)
    def render(rows,dest):
        (dest/'density-response.png').write_text('provisional figure')
        if failure=='result_mutation':bound.write_text('changed after load')
        elif failure=='request_mutation':(tmp_path/'request.json').write_text('{"changed":true}')
        else:raise RuntimeError('figure creation failed')
    monkeypatch.setattr(m,'make_figure',render)
    with pytest.raises((ValueError,RuntimeError)):m.report(tmp_path)
    assert not (tmp_path/'report').exists()
    provisional=tmp_path/'report.incomplete'
    assert json.loads((provisional/'summary.json').read_text())['status']=='failed'
    assert 'Status: passed.' not in (provisional/'RESULTS.md').read_text()
    assert not (provisional/'files.json').exists()


def test_report_publishes_only_after_verification(tmp_path,monkeypatch):
    import json
    from scripts import report_pfs_density_scan as m
    (tmp_path/'request.json').write_text('{}');points=synthetic_comparison_points()
    monkeypatch.setattr(m,'load_request',lambda output: ({'study':'unused'},None))
    monkeypatch.setattr(m,'read_grid',lambda *args:(deepcopy(points),{}))
    monkeypatch.setattr(m,'read_references',lambda *args:(synthetic_references(),{}),raising=False)
    monkeypatch.setattr(m,'make_figure',lambda rows,dest:(dest/'density-response.png').write_text('figure'))
    assert m.report(tmp_path)['status']=='passed'
    assert not (tmp_path/'report.incomplete').exists()
    dest=tmp_path/'report';receipt=json.loads((dest/'files.json').read_text())
    assert all(m.digest(dest/name)==h for name,h in receipt.items())
    result=json.loads((dest/'summary.json').read_text())
    assert len(result['baseline_PFS_gains'])==368
    assert result['rankings']['dimensionless']['Mnu']['per_added_galaxy']==[list(range(8))]
