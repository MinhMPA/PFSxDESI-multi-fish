"""A selected AP resolution must determine every cosmological FD schedule."""
from dataclasses import replace
import copy
import importlib

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec, cosmology_from_coordinates
from pfsfog.revision_forecast import RevisionConfig

NEW = 'ap16384-intrinsic-v1'
OLD = 'legacy8192-v1'
H_STEPS = (.000125, .0000625, .00003125)


def api():
    assert importlib.util.find_spec('pfsfog.ap_numerics'), 'explicit AP profile is missing'
    return importlib.import_module('pfsfog.ap_numerics')


def test_profile_fresh_records_preserve_physics_and_resolution():
    a = api(); old = a.profile_record(OLD); new = a.profile_record(NEW)
    assert new['ap_interpolation_nk'] == 16384
    assert new['ap_refinement_nk'] == [32768, 65536]
    assert new['ap_interpolation_nmu'] == 101
    assert new['fft_n'] == old['fft_n'] == 256
    assert new['decomposition'] == 'intrinsic-table-v1'
    assert old['decomposition'] == 'forward-table-v1'
    assert new['fd_steps']['h'] == list(H_STEPS)
    assert {k: v for k, v in new['fd_steps'].items() if k != 'h'} == {
        k: v for k, v in old['fd_steps'].items() if k != 'h'}
    new['fd_steps']['h'][0] = .5
    assert a.profile_record(NEW)['fd_steps']['h'] == list(H_STEPS)
    with pytest.raises(ValueError): a.profile_record('nearest-grid')


@pytest.mark.parametrize('nk', [16384, 32768, 65536])
def test_grid_selects_smaller_steps_without_changing_coordinate_map(nk):
    old = make_cosmology_spec(FIDUCIAL, extended=True)
    new = make_cosmology_spec(FIDUCIAL, extended=True, ap_nk=nk)
    assert old.fd_steps['h'] == (.001, .0005, .00025)
    assert new.fd_steps['h'] == H_STEPS
    theta = np.array([FIDUCIAL[k] for k in ('ln10_10_As','mnu','Omega_m','h','omega_b','n_s')])
    for key, value in cosmology_from_coordinates(theta, FIDUCIAL, old).items():
        assert np.array_equal(value, cosmology_from_coordinates(theta, FIDUCIAL, new)[key])
    config = RevisionConfig('B', cosmology_basis='extended6', ap_interpolation_nk=nk,
                            ap_refinement_nk=(2*nk,4*nk))
    record = api().configuration_profile(config)
    assert record['ap_interpolation_nk'] == nk
    if nk > 16384:
        assert record['purpose'] == 'output_resolution'
        assert record['derivative_grids_to_evaluate'] == []
    assert max([nk, *record['derivative_grids_to_evaluate']]) <= 65536


def test_reject_undeclared_ap_grid_and_changed_refinement():
    a = api()
    with pytest.raises(ValueError): a.cosmology_steps(12288)
    with pytest.raises(ValueError):
        RevisionConfig('B', ap_interpolation_nk=16384, ap_refinement_nk=(32768,32768))


def test_all_cell_schedule_uses_configured_h_steps():
    from test_extended_validation import _fixture
    from pfsfog.extended_validation import extended_expected_schedule
    from pfsfog.revision_forecast import _cosmology_coordinates
    _, like, _, _ = _fixture()
    like.config = replace(like.config, ap_interpolation_nk=16384,
                          ap_refinement_nk=(32768,65536))
    space, _ = _cosmology_coordinates(like.config, like.reference_cosmology)
    assert space.fd_steps['h'] == H_STEPS
    rows = extended_expected_schedule(like)
    assert {r['step'] for r in rows if r['column'] == 'h'} == set(H_STEPS)
    assert len(rows) == 6*3*3


def test_saved_likelihood_rejects_missing_or_substituted_profile(monkeypatch):
    from test_extended_artifacts import fixture
    from pfsfog.extended_artifacts import pack_likelihood, unpack_likelihood
    like, _, _, _ = fixture(monkeypatch)
    like.config = replace(like.config, ap_interpolation_nk=16384,
                          ap_refinement_nk=(32768,65536))
    arrays = {}; meta = pack_likelihood(like, arrays)
    assert meta['numerical_profile']['fd_steps']['h'] == list(H_STEPS)
    assert unpack_likelihood(meta, arrays).config == like.config
    for mutation in ('missing','old_steps','old_grid'):
        changed = copy.deepcopy(meta)
        if mutation == 'missing': changed.pop('numerical_profile')
        elif mutation == 'old_steps': changed['numerical_profile']['fd_steps']['h'] = [.001,.0005,.00025]
        else: changed['numerical_profile']['ap_interpolation_nk'] = 8192
        with pytest.raises(ValueError, match='numerical profile'):
            unpack_likelihood(changed, arrays)


def test_prepare_explicit_profile_and_legacy_default_without_emulator(tmp_path,monkeypatch):
    from scripts import run_extended_cosmology as d
    from pfsfog import cosmo
    def forbidden(): raise AssertionError('prepare must not load an emulator')
    monkeypatch.setattr(cosmo, '_load_pklin_emulator', forbidden)
    monkeypatch.setattr(cosmo, '_load_sigma8_emulator', forbidden)
    legacy = d.prepare(tmp_path/'legacy')
    assert all(c['ap_interpolation_nk'] == 8192 for c in legacy['physical_likelihoods'])
    assert 'numerical_profile' not in legacy
    chosen = d.prepare(tmp_path/'chosen', profile=NEW)
    assert chosen['numerical_profile'] == api().profile_record(NEW)
    assert all(c['ap_interpolation_nk'] == 16384 for c in chosen['physical_likelihoods'])
    assert d.main(['--phase','prepare','--output-root',str(tmp_path/'cli'),
                   '--numerical-profile',NEW]) == 0
    records = list((tmp_path/'cli').glob('prepare-*.json'))
    assert len(records) == 1 and d.read_json(records[0]) == d.jsonable(chosen)


def test_resume_rejects_different_requested_profile_before_numerical_use(tmp_path):
    from scripts import run_extended_cosmology as d
    record = {'schema_version':'pfs-desi-cosmology-case-checkpoint-v1',
              'status':'local_cases_validated','inventory':d.inventory()}
    path=tmp_path/'checkpoint.json'; d.write_json(path, record)
    with pytest.raises(ValueError,match='numerical profile'):
        d.load_checkpoint(path,d.file_hash(path),provenance={},oracle_pin='a'*64,profile=NEW)


def test_child_command_carries_selected_profile():
    from scripts import run_extended_cosmology as d
    q={'inventory':d.inventory(profile=NEW),'oracle_export':'/oracle','oracle_export_sha256':'a'*64}
    selected={'run_root':'/run','phase':'qualify','qualification':'/q','qualification_sha256':'b'*64}
    for role in ('worker','reload'):
        command=d._execution_command('/driver','/python',selected,q,role)
        assert command[command.index('--numerical-profile')+1] == NEW


@pytest.mark.parametrize('phase,mode', [('report',None),('produce','--reload'),
                                      ('produce','--worker'),('produce',None)])
def test_requested_profile_rejected_before_using_other_profile_selection(tmp_path,monkeypatch,phase,mode):
    from scripts import run_extended_cosmology as d
    q={'inventory':d.inventory()}
    qpath=tmp_path/'qualification.json'; d.write_json(qpath,q)
    selected={'qualification':str(qpath),'qualification_sha256':d.file_hash(qpath)}
    spath=tmp_path/'selection.json'; d.write_json(spath,selected)
    monkeypatch.setattr(d,'read_selection',lambda *a,**k:(selected,q,{}))
    monkeypatch.setattr(d,'validate_qualification',lambda *a,**k:(q,{}))
    def forbidden(*a,**k): raise AssertionError('wrong profile reached physical execution or completion')
    monkeypatch.setattr(d,'observe_process',forbidden)
    monkeypatch.setattr(d,'emit_completion',forbidden)
    argv=['--phase',phase,'--output-root',str(tmp_path/'output'),'--numerical-profile',NEW,
          '--selection',str(qpath if mode=='--reload' else spath),
          '--selection-sha256',d.file_hash(qpath if mode=='--reload' else spath)]
    if mode: argv.append(mode)
    with pytest.raises(ValueError,match='numerical profile'): d.main(argv)


@pytest.mark.parametrize('value',[None,[], 'ap16384-intrinsic-v1'])
def test_invalid_profile_declaration_rejected(value):
    from scripts import run_extended_cosmology as d
    record=d.inventory(profile=NEW); record['numerical_profile']=value
    with pytest.raises(ValueError,match='numerical profile'): d._inventory_profile(record)


def test_candidate_cannot_qualify_without_independent_ap_references():
    from scripts import run_extended_cosmology as d
    from pfsfog.fisher_precision import fisher_solver_record
    from test_cosmology_resolution import sample,control_provenance
    like,results=sample()
    record={'fisher_solver':fisher_solver_record(results['conditional3'].fisher_solver)}
    with pytest.raises(ValueError,match='AP reference'):
        d.validate_ap_references(like,results,record,None,control_provenance(),oracle_identity={})


def test_incomplete_candidate_cannot_be_certified_as_local_artifact(monkeypatch,tmp_path):
    from test_extended_artifacts import fixture
    from pfsfog.extended_artifacts import write_extended_artifacts
    like,results,provenance,evidence=fixture(monkeypatch)
    like.config=replace(like.config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    for record in (provenance,evidence): record.update(api().profile_fields(like.config))
    with pytest.raises(ValueError,match='intrinsic-table'):
        write_extended_artifacts(like,results,tmp_path,provenance,evidence)
    assert not list(tmp_path.glob('*/manifest.json'))
