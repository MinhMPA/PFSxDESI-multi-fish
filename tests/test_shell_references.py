"""An original-grid replay cannot replace same-grid native-basis equality."""
import copy
import importlib
from dataclasses import replace

import pytest
from test_shell_numerics import grid_sample, candidate_provenance, NEW


def reference_context(monkeypatch, *, case='B'):
    from scripts import run_extended_cosmology as driver
    from pfsfog import revision_forecast as forecast
    from test_cosmology_resolution import native3, projected
    from test_fisher_solver_identity import precise
    calls = []
    def build(config):
        calls.append((config.cosmology_basis,config.dk,config.ap_interpolation_nk))
        like, results = grid_sample(config.dk,config.case,config.ap_interpolation_nk)
        assert config == replace(like.config,cosmology_basis=config.cosmology_basis)
        return native3(like,results['conditional3'])[0] if config.cosmology_basis=='baseline3' else like
    def priors(like,*args,**kwargs):
        extended, results = grid_sample(like.config.dk,like.config.case,like.config.ap_interpolation_nk)
        return native3(extended,results['conditional3'])[1]
    monkeypatch.setattr(forecast,'apply_revision_priors',priors)
    monkeypatch.setattr(driver,'treatments',lambda like,**kwargs:(precise(like),{}))
    base,results=grid_sample(case=case)
    old,_=grid_sample(.0025,case,8192)
    oracle=native3(old,projected(old)['conditional3'])
    identity={'path':'/authenticated/oracle','sha256':'a'*64,'case':case}
    oracle[1].historical_parent_identity={'oracle_export':identity,'worker_index_sha256':'b'*64,
        'arrays_sha256':'c'*64,'control_sha256':'d'*64}
    return driver,base,results,oracle,identity,build,calls


def measure(monkeypatch,path):
    assert importlib.util.find_spec('pfsfog.shell_references'), 'separate shell references missing'
    d,base,results,oracle,identity,build,calls=reference_context(monkeypatch)
    record=d.measure_ap_references(base,results,oracle,path,candidate_provenance(),
        oracle_identity=identity,build_likelihood=build)
    return d,base,results,oracle,identity,calls,record


def test_native_comparisons_and_original_replay_have_separate_physical_builds(monkeypatch,tmp_path):
    d,base,results,oracle,identity,calls,record=measure(monkeypatch,tmp_path)
    assert record['definition']=='ap-independent-references-v2'
    assert record['shell_profile']['name']==NEW
    assert calls==[('baseline3',.00125,16384),('extended6',.00125,8192),('baseline3',.00125,8192),
                   ('extended6',.00125,32768),('extended6',.00125,65536),
                   ('extended6',.0025,8192),('baseline3',.0025,8192),('extended6',.0025,16384)]
    current=record['controls']['native3_8192']
    assert current['oracle_export'] is None
    assert set(current['reference_checks'])=={'conditional3_native3_same_resolution'}
    original=record['original_grid_replay']['controls']['original_grid_native3_8192']
    assert original['oracle_export']==identity and original['shell_grid_role']=='original_grid'
    assert original['historical_derived_reference']['receipt']['legacy_replay']['status']=='passed'
    loaded,recomputed=d.validate_ap_references(base,results,record,oracle,candidate_provenance(),oracle_identity=identity)
    assert recomputed==record['comparisons'] and 'original_grid' in loaded
    assert loaded['original_grid'][16384][0].config.dk==.0025


@pytest.mark.parametrize('mutation', ['v1','missing_original','original_as_fine','fine_as_original',
                                    'role','oracle','source','spacing','receipt'])
def test_reference_reload_rejects_cross_grid_or_relabelled_evidence(monkeypatch,tmp_path,mutation):
    d,base,results,oracle,identity,_,record=measure(monkeypatch,tmp_path)
    original=record['original_grid_replay']
    if mutation=='v1': record['definition']='ap-independent-references-v1'
    elif mutation=='missing_original':record.pop('original_grid_replay')
    elif mutation=='original_as_fine':record['controls']['native3_8192']=copy.deepcopy(original['controls']['original_grid_native3_8192'])
    elif mutation=='fine_as_original':original['controls']['original_grid_native3_8192']=copy.deepcopy(record['controls']['native3_8192'])
    elif mutation=='role':original['controls']['original_grid_extended6_16384']['reference_role']='qualified'
    elif mutation=='oracle':original['oracle_export']['sha256']='e'*64
    elif mutation=='source':original['controls']['original_grid_extended6_8192']['producer_identity']['source_content_hash']='f'*64
    elif mutation=='spacing':original['controls']['original_grid_extended6_16384']['shell_grid_role']='adopted'
    elif mutation=='receipt':original['controls']['original_grid_native3_8192']['historical_derived_reference']['receipt']['parent_identity']['arrays_sha256']='e'*64
    with pytest.raises(ValueError):
        d.validate_ap_references(base,results,record,oracle,candidate_provenance(),oracle_identity=identity)


def test_original_oracle_cannot_be_used_for_fine_native_equality(monkeypatch,tmp_path):
    assert importlib.util.find_spec('pfsfog.shell_references'), 'separate shell references missing'
    from test_cosmology_resolution import native3
    d,like,results,oracle,identity,_,_=reference_context(monkeypatch)
    native,_=native3(like,results['conditional3'])
    with pytest.raises(ValueError,match='configuration'):
        d.check_baseline_control(like,results['conditional3'],native,oracle,tmp_path,
                                candidate_provenance(),oracle_identity=identity)


def test_report_separates_candidate_ap_change_original_replay_and_arithmetic(monkeypatch,tmp_path):
    from test_shell_numerics import AP,SOLVER
    from test_ap_review_boundaries import historical_comparisons
    d,base,results,oracle,identity,_,record=measure(monkeypatch,tmp_path/'references')
    cases,ratios=historical_comparisons()
    for case in ('B','C'):
        cases[case]['ap_references']=copy.deepcopy(record)
    q={'inventory':d.inventory(profile=AP,fisher_solver=SOLVER,shell_profile=NEW),
       'cases':cases,'ap_resolution_ratios':ratios,
       'shell_grid_assessment':{'independently_reloaded':'shell assessment'},
       'resolution_comparisons':{'independently_reloaded':'shell/angular'}}
    loaded={case:(None,None,[],{'failed_treatments':{}}) for case in ('B','C')}
    monkeypatch.setattr(d,'read_selection',lambda *a,**kw:({'qualification_sha256':'a'*64},q,loaded))
    report=d.report(tmp_path/'selection','b'*64,tmp_path/'report.json',
                    profile=AP,fisher_solver=SOLVER,shell_profile=NEW)
    assert 'historical_ap_assessment' not in report
    assert report['candidate_shell_ap8192_assessment']['shell_dk']==.00125
    assert report['shell_grid_assessment']==q['shell_grid_assessment']
    assert report['resolution_comparisons']==q['resolution_comparisons']
    assert report['original_grid_replay']['B']==record['original_grid_replay']
    precision=report['fisher_precision_assessment']['cases']['B']
    assert precision==record['original_grid_replay']['controls']['original_grid_native3_8192']['historical_derived_reference']['receipt']
