"""Failed prior choices and historical error ratios remain scientifically visible."""
import pytest

from test_cosmology_resolution import sample,failed


@pytest.mark.parametrize('change',['omitted','overlap','unknown','status','reason','raw','required'])
def test_refinement_treatment_inventory_cannot_drop_failed_solutions(change):
    from scripts import run_extended_cosmology as driver
    assert hasattr(driver,'treatment_outcomes'),'refinement failure coverage is not validated'
    _,success=sample()
    name='marginal6_no_extra';success.pop(name)
    failures={name:failed()}
    complete=driver.treatment_outcomes(success,failures)
    assert len(complete)==7 and complete[name]['error']=='controlled optional weak-direction failure'
    if change=='omitted': failures={}
    elif change=='overlap': failures['release_h']=failed()
    elif change=='unknown': failures['unrequested_prior']=failures.pop(name)
    elif change=='status': failures[name]['status']='passed'
    elif change=='reason': failures[name]['error']=''
    elif change=='raw': failures[name].pop('raw')
    elif change=='required': success.pop('marginal6');failures['marginal6']=failed()
    with pytest.raises(ValueError): driver.treatment_outcomes(success,failures)


def historical_comparisons():
    from pfsfog.cosmology_resolution import compare_ap_resolution,compare_resolution_ratios
    # Each historical absolute error changes by only 0.6%, but C/B changes
    # by 0.994/1.006-1 = -1.193%, and its reciprocal by +1.207%.
    b0,c0=sample(8192,1.006),sample(8192,.994,case='C')
    b1,c1=sample(),sample(case='C')
    cases={}
    for case,coarse,fine in (('B',b0,b1),('C',c0,c1)):
        row=compare_ap_resolution(coarse[0],fine[0],coarse[1],fine[1])
        assert row['status']=='passed'
        cases[case]={'ap_references':{'comparisons':{'8192_to_16384':row},
                                     'historical_change_requires_assessment':False}}
    ratios={'8192_to_16384':compare_resolution_ratios(b0[1],c0[1],b1[1],c1[1])}
    assert ratios['8192_to_16384']['status']=='failed'
    return cases,ratios


def test_historical_ratio_failure_triggers_assessment_when_absolute_errors_pass():
    from scripts import run_extended_cosmology as driver
    assert hasattr(driver,'historical_ap_assessment'),'cohort historical assessment is missing'
    cases,ratios=historical_comparisons()
    assessment=driver.historical_ap_assessment(cases,ratios)
    assert assessment['required'] is True
    assert assessment['case_errors']=={'B':'passed','C':'passed'}
    assert assessment['error_ratios']=='failed'
    assert assessment['reasons']


def test_candidate_shell_ap_assessment_keeps_fixed_shell_spacing_explicit():
    from scripts import run_extended_cosmology as driver
    cases,ratios=historical_comparisons()
    assessment=driver.historical_ap_assessment(cases,ratios,shell_profile='shell00125-v1')
    assert assessment['required'] is True
    assert assessment['shell_dk']==.00125
    assert assessment['error_ratios']=='failed'


def test_report_carries_historical_changes_and_raw_resolution_comparisons(monkeypatch,tmp_path):
    from scripts import run_extended_cosmology as driver
    cases,ratios=historical_comparisons()
    q={'inventory':driver.inventory(profile=driver.INTRINSIC_PROFILE),'cases':cases,
       'ap_resolution_ratios':ratios}
    # Selection authentication is covered by the full process test. Here the
    # selected measurements are controlled so the report cannot omit a failure.
    loaded={case:(None,None,[],{'failed_treatments':{}}) for case in ('B','C')}
    monkeypatch.setattr(driver,'read_selection',lambda *a,**kw:({'qualification_sha256':'a'*64},q,loaded))
    report=driver.report(tmp_path/'selection','b'*64,tmp_path/'report.json',profile=driver.INTRINSIC_PROFILE)
    assert report.get('historical_ap_assessment',{}).get('required') is True
    assert report['ap_resolution_ratios']==ratios
    assert report['ap_resolution_comparisons']=={
        case:cases[case]['ap_references']['comparisons'] for case in ('B','C')}
