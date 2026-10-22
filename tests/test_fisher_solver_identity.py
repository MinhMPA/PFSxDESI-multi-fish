"""A numerical resolution comparison must not silently change inverse arithmetic."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.fisher_precision import (DECIMAL_FISHER_SOLVER, LEGACY_FISHER_SOLVER,
                                     fisher_solver_record)
from test_cosmology_resolution import sample, projected, native3, control_provenance


def precise(like):
    from pfsfog.cosmology_projection import project_cosmology_precision
    out = projected(like)
    for name, result in out.items():
        for key, value in project_cosmology_precision(like, name, fisher_solver=DECIMAL_FISHER_SOLVER).items():
            setattr(result, key, value)
        for suffix in ('marg', 'nuisance_fixed'):
            setattr(result, 'growth_cov_'+suffix, result.growth_jacobian
                    @ getattr(result, 'cov_cosmo_'+suffix) @ result.growth_jacobian.T)
    return out


def provenance():
    return {**control_provenance(), 'fisher_solver': fisher_solver_record(DECIMAL_FISHER_SOLVER)}


@pytest.mark.parametrize('mutation', ['missing', 'digits', 'mixed_result'])
def test_controls_reject_missing_changed_or_mixed_arithmetic(tmp_path, mutation):
    from scripts import run_extended_cosmology as d
    like, _ = sample(8192)
    results = precise(like)
    if mutation == 'mixed_result':
        results['release_h'].fisher_solver = LEGACY_FISHER_SOLVER
        with pytest.raises(ValueError, match='solver'):
            d.save_control(tmp_path/'control', like, results, provenance=provenance())
        return
    saved = d.save_control(tmp_path/'control', like, results, provenance=provenance())
    assert saved['fisher_solver'] == fisher_solver_record(DECIMAL_FISHER_SOLVER)
    if mutation == 'missing': saved.pop('fisher_solver')
    else: saved['fisher_solver']['decimal_digits'] = 70
    with pytest.raises(ValueError, match='solver'):
        d.load_control(saved, provenance=provenance())


def test_native_reload_reconstructs_full_covariance_after_rehash(tmp_path):
    from scripts import run_extended_cosmology as d
    from pfsfog.revision_forecast import project_baseline_precision
    like, results = sample(8192)
    native, old = native3(like, results['conditional3'])
    p = project_baseline_precision(native, 1., fisher_solver=DECIMAL_FISHER_SOLVER)
    result = SimpleNamespace(**p, likelihood=native,
        growth_z=old.growth_z, growth_fiducial=old.growth_fiducial, growth_jacobian=old.growth_jacobian)
    for suffix in ('marg', 'nuisance_fixed'):
        setattr(result, 'growth_cov_'+suffix, result.growth_jacobian @ p['cov_cosmo_'+suffix] @ result.growth_jacobian.T)
    path = tmp_path/'control'
    saved = d.save_control(path, native, {'baseline3': result}, provenance=provenance())
    d.load_control(saved, provenance=provenance())
    with np.load(path/'arrays.npz') as z: arrays = {k:z[k] for k in z.files}
    arrays['baseline3/covariance'][3, 4] += .01
    np.savez(path/'arrays.npz', **arrays)
    saved['arrays_sha256'] = d.file_hash(path/'arrays.npz')
    with pytest.raises(ValueError, match='covariance'):
        d.load_control(saved, provenance=provenance())


def test_resolution_and_ratios_reject_mixed_inverse_arithmetic():
    from pfsfog.cosmology_resolution import compare_ap_resolution, compare_resolution_ratios
    a, ar = sample(16384)
    b, _ = sample(32768)
    br = precise(b)
    with pytest.raises(ValueError, match='solver'):
        compare_ap_resolution(a, b, ar, br)
    c, cr = sample(16384, case='C')
    e, er = sample(32768, case='C')
    with pytest.raises(ValueError, match='solver'):
        compare_resolution_ratios(ar, cr, br, er)


def test_inventory_and_execution_bind_explicit_solver():
    from scripts import run_extended_cosmology as d
    from pfsfog.ap_numerics import INTRINSIC_PROFILE
    inventory = d.inventory(profile=INTRINSIC_PROFILE, fisher_solver=DECIMAL_FISHER_SOLVER)
    assert inventory['fisher_solver'] == fisher_solver_record(DECIMAL_FISHER_SOLVER)
    q = {'inventory': inventory, 'oracle_export': '/oracle', 'oracle_export_sha256': 'a'*64}
    command = d._execution_command('/driver.py', '/python', {'run_root':'/run','phase':'qualify'}, q, 'worker')
    assert command[command.index('--fisher-solver')+1] == DECIMAL_FISHER_SOLVER
    inventory['fisher_solver']['decimal_digits'] = 70
    with pytest.raises(ValueError, match='solver'):
        d._inventory_profile(inventory)


def test_resume_cannot_select_another_solver(tmp_path,monkeypatch):
    from scripts import run_extended_cosmology as d
    from pfsfog.ap_numerics import INTRINSIC_PROFILE
    q={'schema_version':'pfs-desi-cosmology-case-checkpoint-v1','status':'local_cases_validated',
       'inventory':d.inventory(profile=INTRINSIC_PROFILE,fisher_solver=DECIMAL_FISHER_SOLVER),
       'provenance':provenance()}
    path=tmp_path/'checkpoint.json';d.write_json(path,q)
    with pytest.raises(ValueError,match='solver'):
        d.load_checkpoint(path,d.file_hash(path),provenance=control_provenance(),oracle_pin='b'*64,
                          profile=INTRINSIC_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER)


def test_artifact_roundtrip_declares_arithmetic_and_rejects_missing_identity(monkeypatch, tmp_path):
    from test_extended_artifacts import fixture
    from pfsfog import extended_artifacts as a
    from pfsfog.cosmology_projection import apply_cosmology_treatment
    like, old, prov, evidence = fixture(monkeypatch)
    prov['fisher_solver'] = fisher_solver_record(DECIMAL_FISHER_SOLVER)
    results = {name:apply_cosmology_treatment(like, name, fisher_solver=DECIMAL_FISHER_SOLVER) for name in old}
    path = a.write_extended_artifacts(like, results, tmp_path, prov, evidence)
    manifest, _, _, _ = a.load_extended_artifacts(path, manifest_sha256=a.file_hash(path/'manifest.json'))
    assert manifest['fisher_solver'] == prov['fisher_solver']
    assert all(row['fisher_solver'] == prov['fisher_solver'] for row in manifest['treatments'].values())
    manifest.pop('fisher_solver')
    a.write_json(path/'manifest.json', manifest)
    with pytest.raises(ValueError, match='solver'):
        a.load_extended_artifacts(path, manifest_sha256=a.file_hash(path/'manifest.json'))


def baseline_reference(monkeypatch,path):
    from scripts import run_extended_cosmology as d
    from pfsfog import revision_forecast as f
    from test_historical_precision import setup
    like,_=sample(8192);new=precise(like)
    baseline,result=native3(like,new['conditional3'])
    historical,old,parent=setup();old.historical_parent_identity=parent
    monkeypatch.setattr(f,'apply_revision_priors',lambda *args,**kwargs:result)
    record=d.check_baseline_control(like,new['conditional3'],baseline,(historical,old),path,
                                   provenance(),oracle_identity=parent['oracle_export'])
    return d,like,new['conditional3'],(historical,old),record,parent['oracle_export']


@pytest.mark.parametrize('mutation',['none','missing_receipt','forged_receipt','oracle_identity','derived_parent'])
def test_same_grid_reload_repeats_direct_native_identity_and_rejects_receipt_changes(monkeypatch,tmp_path,mutation):
    from pfsfog import extended_validation as ev
    d,like,conditional,oracle,record,identity=baseline_reference(monkeypatch,tmp_path)
    if mutation=='missing_receipt': record.pop('reference_checks')
    elif mutation=='forged_receipt': record['reference_checks']['historical_legacy_replay']['status']='invented'
    elif mutation=='oracle_identity': record['oracle_export']={'sha256':'0'*64}
    elif mutation=='derived_parent': record['historical_derived_reference']['receipt']['parent_identity']['arrays_sha256']='0'*64
    assert hasattr(d,'validate_baseline_control'),'common matched-resolution reference reload is missing'
    calls=[];original=ev.validate_native_baseline_identity
    def checked(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(ev,'validate_native_baseline_identity',checked)
    if mutation=='none':
        d.validate_baseline_control(like,conditional,record,oracle,provenance(),oracle_identity=identity)
        assert len(calls)==1
    else:
        with pytest.raises(ValueError):
            d.validate_baseline_control(like,conditional,record,oracle,provenance(),oracle_identity=identity)


def test_public_report_discloses_historical_covariance_roundoff(monkeypatch,tmp_path):
    from scripts import run_extended_cosmology as d
    from pfsfog.ap_numerics import INTRINSIC_PROFILE
    receipt={'legacy_replay':{'status':'passed'},'scientific_identity':{'status':'passed'},
             'full_covariance_changes':{'same_resolution_allowance':{'failed_entries':2},
                                        'reconstruction_allowance':{'failed_entries':104}}}
    comparison={'status':'passed','reasons':[]}
    q={'inventory':d.inventory(profile=INTRINSIC_PROFILE,fisher_solver=DECIMAL_FISHER_SOLVER),
       'cases':{case:{'ap_references':{'comparisons':{'8192_to_16384':comparison},
           'controls':{'native3_8192':{'historical_derived_reference':{'receipt':receipt}}}}} for case in ('B','C')},
       'ap_resolution_ratios':{'8192_to_16384':comparison}}
    row={'treatment':'conditional3','native_sigma':{'Mnu':1.},'growth_sigma':[1.]*8}
    loaded={case:(None,None,[row],{'failed_treatments':{}}) for case in ('B','C')}
    monkeypatch.setattr(d,'read_selection',lambda *args,**kwargs:({'qualification_sha256':'b'*64},q,loaded))
    report=d.report('/selection','a'*64,tmp_path/'report.json',profile=INTRINSIC_PROFILE,fisher_solver=DECIMAL_FISHER_SOLVER)
    assert report['historical_ap_assessment']['required'] is False
    assert report['fisher_precision_assessment']['cases']['B']==receipt
    assert report['fisher_precision_assessment']['historical_covariances_preserved'] is True
