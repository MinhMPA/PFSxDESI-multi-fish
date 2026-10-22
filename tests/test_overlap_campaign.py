"""Condition inventory, immutable evidence, and failed/resumable campaign records."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scripts import run_overlap_study as campaign
from scripts.overlap_inputs import conditions, canonical_condition


def test_complete_named_inventory():
    ids = [campaign.condition_id(c) for c in conditions()]
    assert len(ids) == len(set(ids)) == 32
    assert len({campaign.condition_id(canonical_condition(c)) for c in conditions()}) == 15


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Tiny authenticated files exercise bookkeeping without copying 38.8 GB."""
    source = tmp_path/'source'; source.mkdir()
    evidence = source/'physical.json'; evidence.write_text('{"qualified":true}')
    monkeypatch.setattr(campaign, '_collect_sources', lambda root: {
        g: {str(evidence): campaign.digest(evidence)} for g in campaign.GRID_PROFILES})
    output = tmp_path/'new'
    campaign.prepare(source, output)
    return output, evidence


def test_preparation_is_not_numerical_completion(prepared):
    output, _ = prepared
    result = campaign.status(output)
    assert result['status'] == 'prepared' and result['completed_records'] == 0
    assert result['expected_records'] == 160 and result['expected_distinct'] == 75
    with pytest.raises(FileExistsError):
        campaign.prepare(output.parent/'source', output)


@pytest.mark.parametrize('damage', ['request','source','code'])
def test_changed_physical_inputs_rejected(prepared, monkeypatch, damage):
    output, evidence = prepared
    if damage == 'request':
        request = json.loads((output/'request.json').read_text())
        request['grids']['shell304']['n_mu'] = 7
        (output/'request.json').write_text(json.dumps(request))
    elif damage == 'source':
        evidence.write_text('changed')
    else:
        monkeypatch.setattr(campaign, 'OWN_FILES', ('scripts/missing_numerical_module.py',))
    with pytest.raises(ValueError):
        campaign.status(output)
    with pytest.raises(ValueError):
        campaign.run_grid(output, 'shell304')


def fake_records(output, grid='shell304'):
    request, state = campaign._request(output)
    records = []
    for c in conditions():
        name = campaign.condition_id(c); target = campaign.condition_id(canonical_condition(c))
        path = output/'inference'/grid/name; path.mkdir(parents=True)
        checks = {'status':'passed', 'equivalence': {'status':'passed'}}
        campaign.write(path/'checks.json', checks)
        meta = {'schema':campaign.SCHEMA, 'status':'passed', 'consumable':True,
                'grid':grid, 'condition':asdict(c), 'condition_id':name,
                'canonical_condition':asdict(canonical_condition(c)),
                'request_sha256':state['request_sha256'],
                'source_hashes':request['source_hashes'][grid],
                'checks_sha256':campaign.digest(path/'checks.json')}
        if name != target:
            meta['alias_of'] = target
            meta['alias_manifest_sha256'] = campaign.digest(output/'inference'/grid/target/'manifest.json')
        else:
            (path/'arrays.npz').write_bytes(b'authenticated test array bytes')
            meta['arrays_sha256'] = campaign.digest(path/'arrays.npz')
        campaign.write(path/'manifest.json',meta)
        records.append({'condition_id':name,'manifest_sha256':campaign.digest(path/'manifest.json')})
    campaign.write(output/'inference'/grid/'checks.json', {'status':'passed','area_identities':{'status':'passed'}})
    state['grids'][grid] = {'status':'passed','records':records,
        'checks_sha256':campaign.digest(output/'inference'/grid/'checks.json')}
    campaign.write(output/'state.json',state)
    return state


def test_grid_inventory_counts_all_names_and_distinct_solutions(prepared):
    output,_=prepared; fake_records(output)
    state=campaign.status(output)
    assert state['completed_records']==32 and state['completed_distinct']==15
    assert state['status']!='passed'


@pytest.mark.parametrize('damage',['mixed_grid','duplicate','missing_checks','failed','alias','array','partial','grid_checks'])
def test_incompatible_or_incomplete_results_never_complete(prepared, damage):
    output,_=prepared; state=fake_records(output)
    row=state['grids']['shell304']; name=campaign.condition_id(conditions()[1])
    path=output/'inference'/'shell304'/name
    if damage=='duplicate': row['records'].append(row['records'][0])
    elif damage=='partial': row['records'].pop()
    elif damage=='missing_checks': (path/'checks.json').unlink()
    elif damage=='grid_checks': (output/'inference/shell304/checks.json').unlink()
    else:
        meta=json.loads((path/'manifest.json').read_text())
        if damage=='mixed_grid': meta['grid']='loop2048'
        elif damage=='failed': meta.update(status='failed',consumable=False)
        elif damage=='alias': meta['alias_of']='not-a-condition'
        elif damage=='array':
            canonical=output/'inference/shell304'/campaign.condition_id(canonical_condition(conditions()[1]))
            (canonical/'arrays.npz').write_bytes(b'changed')
        campaign.write(path/'manifest.json',meta)
        for r in row['records']:
            if r['condition_id']==name:r['manifest_sha256']=campaign.digest(path/'manifest.json')
    campaign.write(output/'state.json',state)
    with pytest.raises(ValueError): campaign.status(output)
    with pytest.raises(ValueError): campaign.run_grid(output,'shell304')


def test_failed_condition_has_error_without_consumable_result(prepared, monkeypatch):
    output,_=prepared
    monkeypatch.setattr(campaign,'load_inputs',lambda *args: (_ for _ in ()).throw(ValueError('failed physical source')))
    with pytest.raises(ValueError,match='failed physical source'):
        campaign.run_grid(output,'shell304')
    state=json.loads((output/'state.json').read_text())
    assert state['status']=='failed' and state['exit_status']==1
    assert state['grids']['shell304']['error']=='failed physical source'
    assert not list((output/'inference/shell304').glob('*/manifest.json'))
    assert list((output/'inference/shell304/attempts').glob('*/error.json'))


def test_resume_retains_checked_records_and_failed_attempts(prepared, monkeypatch):
    output,_=prepared
    state=fake_records(output)
    keep=state['grids']['shell304']['records'][0]
    # Retain one completed condition; remove other test-only directories.
    import shutil
    for path in (output/'inference/shell304').iterdir():
        if path.is_dir() and path.name!=keep['condition_id']:shutil.rmtree(path)
    state['grids']['shell304']={'status':'failed','records':[keep]}
    state['status']='failed'; campaign.write(output/'state.json',state)
    monkeypatch.setattr(campaign,'load_inputs',lambda *args: (_ for _ in ()).throw(ValueError('diagnosed source failure')))
    for _ in range(2):
        with pytest.raises(ValueError,match='diagnosed source failure'):campaign.run_grid(output,'shell304')
    state=campaign.status(output)
    assert state['completed_records']==1 and state['status']=='failed'
    assert len(list((output/'inference/shell304/attempts').glob('*/error.json')))==2


def test_unrecorded_partial_result_requires_diagnosis(prepared):
    output,_=prepared
    path=output/'inference/shell304/A-area01200-marginalized';path.mkdir(parents=True)
    campaign.write(path/'manifest.json',{'request_sha256':'different'})
    with pytest.raises(ValueError,match='partial output requires diagnosis'):
        campaign.run_grid(output,'shell304')


def test_full_campaign_bookkeeping_and_resume_without_repeated_solve(prepared,monkeypatch):
    """Numerical identities have separate real-likelihood tests; isolate receipts here."""
    from types import SimpleNamespace
    output,evidence=prepared
    like=SimpleNamespace(names=['ln10_10_As','Mnu','Omegam'],parameters=[],
                         F_like=__import__('numpy').eye(3),cells=[])
    calls=[]
    monkeypatch.setattr(campaign,'load_inputs',lambda root,g:SimpleNamespace(
        grid=g,hashes={str(evidence):campaign.digest(evidence)},results={'C21':SimpleNamespace(likelihood=like)}))
    monkeypatch.setattr(campaign,'select_observations',lambda *args:like)
    monkeypatch.setattr(campaign,'scale_area',lambda *args:like)
    monkeypatch.setattr(campaign,'fix_cross_coefficients',lambda *args:(like,()))
    monkeypatch.setattr(campaign,'_area_checks',lambda *args:{'status':'passed'})
    def infer(*args):calls.append(1);return like
    monkeypatch.setattr(campaign,'infer_saved_growth',infer)
    monkeypatch.setattr(campaign,'check_marginalization',lambda result:{})
    def save(path,*args):
        __import__('numpy').savez(path/'arrays.npz',test=like.F_like)
        return {'arrays_sha256':campaign.digest(path/'arrays.npz')},{}
    monkeypatch.setattr(campaign,'_save_result',save)
    for grid in campaign.GRID_PROFILES:
        row=campaign.run_grid(output,grid)
        assert row['status']=='passed' and len(row['records'])==32
        campaign.run_grid(output,grid)
    state=campaign.status(output)
    assert state['status']=='passed' and state['completed_records']==160
    assert state['completed_distinct']==len(calls)==75


from tests.overlap_fixtures import overlap_inputs_for_grid, overlap_inputs


@pytest.mark.parametrize('grid',list(campaign.GRID_PROFILES))
def test_area_identities_and_alias_physics_on_every_grid(overlap_inputs_for_grid,grid):
    source=overlap_inputs_for_grid(grid).results['C21'].likelihood
    selected={c:campaign.select_observations(source,c) for c in ('A','B','C','D')}
    areas={(c,a):campaign.scale_area(selected[c],a) for c in selected for a in (1200,14000)}
    assert campaign._area_checks(selected,areas)['status']=='passed'
    for c in conditions():
        canonical=canonical_condition(c)
        if c==canonical:continue
        like,fixed=campaign.fix_cross_coefficients(areas[c.case,c.area_deg2],c.stochastic)
        other,other_fixed=campaign.fix_cross_coefficients(areas[canonical.case,canonical.area_deg2],canonical.stochastic)
        assert fixed==other_fixed
        assert campaign._equivalence(like,other)['status']=='passed'


def test_real_saved_arrays_preserve_likelihood_precision_growth_and_diagnostics(tmp_path,overlap_inputs):
    import numpy as np
    source=overlap_inputs.results['C21']
    meta,metrics=campaign._save_result(tmp_path,source,{'H':source.F_total[:3,:3], 'assembly_scaled_error':0.})
    assert meta['arrays_sha256']==campaign.digest(tmp_path/'arrays.npz')
    assert meta['fisher_solver']==campaign.DECIMAL_FISHER_SOLVER
    assert metrics=={'assembly_scaled_error':0.}
    with np.load(tmp_path/'arrays.npz',allow_pickle=False) as data:
        for key in (*campaign.PRECISION_FIELDS,*campaign.GROWTH_FIELDS):
            np.testing.assert_array_equal(data['result/'+key],getattr(source,key))
        np.testing.assert_array_equal(data['diagnostics/H'],source.F_total[:3,:3])
        np.testing.assert_array_equal(data['likelihood/cell0/covariance'],source.likelihood.cells[0].covariance)


def test_interrupted_attempt_preserves_prior_pid_and_active_condition(prepared,monkeypatch):
    output,_=prepared
    request,state=campaign._request(output)
    attempt=output/'inference/shell304/attempts/001';attempt.mkdir(parents=True)
    state.update(status='running',pid=999999999,active_grid='shell304',active_condition='D-area14000-marginalized')
    state['grids']['shell304']={'status':'running','pid':999999999,'records':[],
        'start_utc':'2000-01-01T00:00:00+00:00','attempt':str(attempt),
        'active_condition':'D-area14000-marginalized'}
    campaign.write(output/'state.json',state)
    monkeypatch.setattr(campaign,'load_inputs',lambda *args: (_ for _ in ()).throw(ValueError('resume source failure')))
    with pytest.raises(ValueError,match='resume source failure'):campaign.run_grid(output,'shell304')
    old=json.loads((attempt/'interruption.json').read_text())
    assert old['pid']==999999999
    assert old['active_condition']=='D-area14000-marginalized'
    assert old['start_utc']=='2000-01-01T00:00:00+00:00'
    fresh=json.loads((output/'inference/shell304/attempts/002/start.json').read_text())
    assert fresh['request_sha256']==state['request_sha256'] and fresh['pid']>0


def test_interrupted_condition_remains_owned_by_its_grid_after_another_grid_attempt(prepared,monkeypatch):
    """A second grid must not relabel the original interrupted spectrum condition."""
    from types import SimpleNamespace
    output,evidence=prepared
    _,state=campaign._request(output)
    original='D-area14000-marginalized'
    attempt=output/'inference/shell304/attempts/001';attempt.mkdir(parents=True)
    state.update(status='running',pid=999999999,active_grid='shell304',active_condition=original)
    state['grids']['shell304']={'status':'running','pid':999999999,'records':[],
        'active_condition':original,'start_utc':'2000-01-01T00:00:00+00:00','attempt':str(attempt)}
    campaign.write(output/'state.json',state)
    like=SimpleNamespace()
    monkeypatch.setattr(campaign,'load_inputs',lambda root,g:SimpleNamespace(
        grid=g,hashes={str(evidence):campaign.digest(evidence)},results={'C21':SimpleNamespace(likelihood=like)}))
    monkeypatch.setattr(campaign,'select_observations',lambda *args:like)
    monkeypatch.setattr(campaign,'scale_area',lambda *args:like)
    monkeypatch.setattr(campaign,'_area_checks',lambda *args:{'status':'passed'})
    monkeypatch.setattr(campaign,'fix_cross_coefficients',lambda *args: (_ for _ in ()).throw(ValueError('controlled stop before solve')))
    with pytest.raises(ValueError,match='controlled stop before solve'):
        campaign.run_grid(output,'loop2048')
    between=json.loads((output/'state.json').read_text())
    assert between['active_condition']=='A-area01200-marginalized'
    with pytest.raises(ValueError,match='controlled stop before solve'):
        campaign.run_grid(output,'shell304')
    receipt=json.loads((attempt/'interruption.json').read_text())
    assert receipt['active_condition']==original
    assert receipt['pid']==999999999
    assert receipt['start_utc']=='2000-01-01T00:00:00+00:00'
