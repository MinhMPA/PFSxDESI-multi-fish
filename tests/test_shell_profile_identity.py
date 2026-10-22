"""Requested shell resolution is bound before spectra, reuse, or reporting."""
import copy
from pathlib import Path
import pytest
from test_shell_numerics import api, NEW, OLD, AP, SOLVER


def test_prepare_declares_common_grid_without_evaluating_spectra(tmp_path):
    from scripts import run_extended_cosmology as d
    api()
    legacy=d.prepare(tmp_path/'old')
    assert 'shell_profile' not in legacy and legacy['requested_dk'] == .005
    chosen=d.prepare(tmp_path/'new',profile=AP,fisher_solver=SOLVER,shell_profile=NEW)
    assert [x['dk'] for x in chosen['physical_likelihoods']] == [.00125,.00125]
    assert chosen['requested_dk'] == .00125 and chosen['shell_profile']['original_grid_dk'] == .0025
    assert d.main(['--phase','prepare','--output-root',str(tmp_path/'cli'),
                   '--numerical-profile',AP,'--fisher-solver',SOLVER,'--shell-profile',NEW]) == 0
    assert d.read_json(next((tmp_path/'cli').glob('prepare-*.json'))) == d.jsonable(chosen)


@pytest.mark.parametrize('profile,solver', [('legacy8192-v1',SOLVER),(AP,'scaled-cholesky-float64-v1')])
def test_candidate_shells_reject_undeclared_ap_or_inverse_combination(profile,solver):
    from scripts import run_extended_cosmology as d
    api()
    with pytest.raises(ValueError): d.inventory(profile=profile,fisher_solver=solver,shell_profile=NEW)


def test_qualification_and_reload_commands_carry_shell_choice():
    from scripts import run_extended_cosmology as d
    api(); inv=d.inventory(profile=AP,fisher_solver=SOLVER,shell_profile=NEW)
    q={'inventory':inv,'oracle_export':'/oracle','oracle_export_sha256':'a'*64}
    selected={'run_root':'/run','phase':'qualify','qualification':'/q','qualification_sha256':'b'*64}
    for role in ('worker','reload'):
        command=d._execution_command('/driver','/python',selected,q,role)
        assert command[command.index('--shell-profile')+1] == NEW
    changed=copy.deepcopy(inv);changed.pop('shell_profile')
    with pytest.raises(ValueError):d._inventory_profile(changed,shell_profile=NEW)


@pytest.mark.parametrize('shell,requested', [(OLD,.005),(NEW,.00125)])
def test_producer_records_requested_and_historical_spacing(monkeypatch,shell,requested):
    from scripts import run_extended_cosmology as d, run_revision as original
    from scripts import export_cosmology_oracle as exporter
    api()
    # The expensive backend/input scan is replaced; real extended provenance
    # construction still binds the grid request, source, and runtime hashes.
    monkeypatch.setattr(original,'collect_provenance',lambda backend,**kw:
                        {'source_file_hashes':{},**kw})
    monkeypatch.setattr(exporter,'runtime_record',lambda:{'test_runtime':'pinned'})
    record=d.collect_provenance(None,profile=AP,fisher_solver=SOLVER,shell_profile=shell)
    assert record['requested_dk']==requested
    if shell==NEW:
        assert record['original_grid_dk']==.0025 and record['shell_profile']['name']==NEW
    else:
        assert 'shell_profile' not in record and 'original_grid_dk' not in record


def test_resume_refuses_other_shell_grid_before_reusing_data(tmp_path):
    from scripts import run_extended_cosmology as d
    api()
    record={'schema_version':'pfs-desi-cosmology-case-checkpoint-v1','status':'local_cases_validated',
            'inventory':d.inventory(profile=AP,fisher_solver=SOLVER)}
    path=tmp_path/'checkpoint.json';d.write_json(path,record)
    with pytest.raises(ValueError,match='shell'):
        d.load_checkpoint(path,d.file_hash(path),provenance={},oracle_pin='a'*64,
                          profile=AP,fisher_solver=SOLVER,shell_profile=NEW)


@pytest.mark.parametrize('phase,mode', [('report',None),('produce','--reload'),
                                     ('produce','--worker'),('produce',None)])
def test_selected_legacy_shells_cannot_satisfy_candidate_request(tmp_path,monkeypatch,phase,mode):
    from scripts import run_extended_cosmology as d
    api(); q={'inventory':d.inventory(profile=AP,fisher_solver=SOLVER)}
    qp=tmp_path/'q.json';d.write_json(qp,q)
    selected={'qualification':str(qp),'qualification_sha256':d.file_hash(qp)}
    sp=tmp_path/'selected.json';d.write_json(sp,selected)
    monkeypatch.setattr(d,'read_selection',lambda *a,**k:(selected,q,{}))
    monkeypatch.setattr(d,'validate_qualification',lambda *a,**k:(q,{}))
    def forbidden(*a,**k): raise AssertionError('wrong grid reached execution')
    monkeypatch.setattr(d,'observe_process',forbidden);monkeypatch.setattr(d,'emit_completion',forbidden)
    path=qp if mode=='--reload' else sp
    args=['--phase',phase,'--output-root',str(tmp_path/'out'),'--numerical-profile',AP,
          '--fisher-solver',SOLVER,'--shell-profile',NEW,'--selection',str(path),
          '--selection-sha256',d.file_hash(path)]
    if mode:args.append(mode)
    with pytest.raises(ValueError,match='shell'): d.main(args)


def test_scientific_inventory_detects_replaced_shell_definition(tmp_path,monkeypatch):
    from scripts import export_cosmology_oracle as e
    api(); module=Path(__file__).resolve().parents[1]/'pfsfog/shell_numerics.py'
    # A minimal real source tree, with real inventory hashing, needs no spectra.
    (tmp_path/'pfsfog').mkdir();(tmp_path/'scripts').mkdir();(tmp_path/'ps_1loop_jax').mkdir()
    (tmp_path/'survey_specs').mkdir();(tmp_path/'configs').mkdir()
    (tmp_path/'pfsfog/shell_numerics.py').write_bytes(module.read_bytes())
    for p in ('scripts/run_revision.py','scripts/validate_revision.py','scripts/run_extended_cosmology.py',
              'scripts/export_cosmology_oracle.py','ps_1loop_jax/model.py','survey_specs/sample.txt','configs/default.yaml'):
        (tmp_path/p).write_text('input')
    first=e.scientific_inventory(tmp_path,extended=True)
    assert str(tmp_path/'pfsfog/shell_numerics.py') in first['source_file_hashes']
    (tmp_path/'pfsfog/shell_numerics.py').write_text('changed')
    with pytest.raises(ValueError,match='inventory'):
        e.verify_scientific_inventory(first,tmp_path,extended=True)
