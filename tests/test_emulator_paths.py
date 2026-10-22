"""Model relocation preserves bytes, original paths and package identity."""
from pathlib import Path
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
RELOCATION=ROOT/'docs/revision/evidence/extended-cosmology-study/ap-global-continuation-2026-10-05/RELOCATION_AUTHENTICATION.json'
MODEL_ROOT=Path('/Users/nguyenmn/cosmopower-jax/cosmology/jense2024')


def api():
    assert importlib.util.find_spec('pfsfog.emulator_paths'), 'explicit emulator locator is missing'
    return importlib.import_module('pfsfog.emulator_paths')


def record(): return json.loads(RELOCATION.read_text())


def _verify_in_historical_runtime(test_name):
    """Exercise positive old-export authentication in its actual pinned Python.

    The ELF interpreter must reject that export. It cannot be relabelled as
    the historical runtime merely to make a positive-replay test pass.
    """
    from scripts import export_cosmology_oracle as e
    env = {**os.environ, 'PYTHONPATH': f'{ROOT}:{ROOT}/ps_1loop_jax:/Users/nguyenmn/cosmopower-jax'}
    result = subprocess.run([str(e.RECOVERY/'bin/python'), '-B', '-m', 'pytest',
        f'{Path(__file__).resolve()}::{test_name}', '-q'], cwd=ROOT, env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


def test_active_models_match_reviewed_relocation_and_cosmology_locator():
    m=api(); expected={r['active_path']:r['sha256'] for r in record()['relocated_paths']
                      if '/cosmology/' in r['active_path']}
    actual=m.active_emulator_inputs()
    assert len(actual)==3 and actual==expected
    from pfsfog import cosmo
    assert cosmo._JENSE_DIR==MODEL_ROOT
    actual.clear(); assert m.active_emulator_inputs()==expected
    p=m.active_emulator_provenance()
    assert p['model_root']==str(MODEL_ROOT) and p['model']=='jense_2023_camb_mnu'
    assert p['package_version']=='0.5.7'
    assert p['package_files']==record()['active_package_python_files']


def test_historical_verification_preserves_original_keys_and_records_exact_exception():
    m=api(); r=record(); originals={row['historical_path']:row['sha256'] for row in r['relocated_paths']}
    before=copy.deepcopy(originals)
    result=m.verify_historical_relocation(originals,r)
    assert originals==before
    assert result['historical_exact_path_replay'] is False
    assert result['verified_file_count']==4
    assert sorted(result['relocated_paths'],key=lambda x:x['historical_path'])==sorted(r['relocated_paths'],key=lambda x:x['historical_path'])


@pytest.mark.parametrize('mutation',['fifth','expected_hash','active_origin','package_hash'])
def test_relocation_rejects_unapproved_mapping_or_package(mutation,tmp_path):
    m=api(); r=record(); original={row['historical_path']:row['sha256'] for row in r['relocated_paths']}
    if mutation=='fifth':
        p=tmp_path/'fifth';p.write_bytes(b'example')
        r['relocated_paths'].append({'historical_path':'/unlisted/old','active_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    elif mutation=='expected_hash': r['relocated_paths'][0]['sha256']='0'*64
    elif mutation=='active_origin': r['package_origins']['cosmopower_jax']['origin']=str(tmp_path/'elsewhere.py')
    else: r['active_package_python_files'][next(k for k in r['active_package_python_files'] if k.endswith('/cosmopower_jax.py'))]='0'*64
    with pytest.raises(ValueError): m.verify_historical_relocation(original,r)


def test_historical_changed_or_unlisted_file_is_not_a_path_exception(tmp_path):
    m=api(); p=tmp_path/'unchanged';p.write_bytes(b'approved')
    original={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
    assert m.verify_historical_relocation(original,record())['verified_file_count']==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes'): m.verify_historical_relocation(original,record())
    with pytest.raises(ValueError,match='missing|unlisted'):
        m.verify_historical_relocation({'/Users/nguyenmn/cosmopower-jax-for-pfs/unlisted.py':'a'*64},record())


def test_current_package_resolution_is_checked(monkeypatch,tmp_path):
    m=api()
    from importlib.machinery import PathFinder
    find=PathFinder.find_spec
    def altered(name,path=None,target=None):
        spec=find(name,path,target)
        if name=='cosmopower_jax': spec.origin=str(tmp_path/'elsewhere.py')
        return spec
    monkeypatch.setattr(PathFinder,'find_spec',altered)
    with pytest.raises(ValueError,match='origin'): m.active_emulator_provenance()


def test_current_package_bytes_are_checked(monkeypatch):
    m=api(); read=Path.read_bytes
    def altered(path,*a,**kw):
        if str(path).endswith('/cosmopower_jax/cosmopower_jax.py'): return b'changed package source'
        return read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_bytes',altered)
    with pytest.raises(ValueError,match='bytes'): m.active_emulator_provenance()


def test_active_inventory_and_explicit_historical_path_are_separate():
    api()
    from scripts import export_cosmology_oracle as e
    active=e.scientific_inventory(ROOT,extended=True)
    assert all(p.startswith(str(MODEL_ROOT)) for p in active['emulator_input_hashes'])
    assert active['active_emulator_provenance']['model_root']==str(MODEL_ROOT)
    historical=e.scientific_inventory(e.ORACLE,extended=False,input_role='historical',
                                      relocation_record=record())
    assert all('/cosmopower-jax-for-pfs/' in p for p in historical['emulator_input_hashes'])
    p={'project_head':e.HEAD,**historical}
    assert e.verify_inputs(p,relocation_record=record())['historical_exact_path_replay'] is False
    with pytest.raises((FileNotFoundError,ValueError)): e.verify_inputs(p)


def test_runtime_only_normalizes_reviewed_relocation(monkeypatch):
    api()
    from scripts import export_cosmology_oracle as e
    from scripts import run_extended_cosmology as d
    active=e.runtime_record(); old=copy.deepcopy(active)
    old['environment']['PYTHONPATH']=f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax'
    if Path(sys.prefix) != e.RECOVERY:
        with pytest.raises(ValueError, match='not the reviewed recovery interpreter'):
            e.verify_runtime(old, recorded_root=e.ORACLE, recorded_cwd=e.ORACLE,
                             active_root=d.ROOT, relocation_record=record())
        _verify_in_historical_runtime('test_runtime_only_normalizes_reviewed_relocation')
        return
    result=e.verify_runtime(old,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,
                            active_root=d.ROOT,relocation_record=record())
    assert result==active
    with pytest.raises(ValueError):
        e.verify_runtime(old,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,active_root=d.ROOT)
    old['environment']['OMP_NUM_THREADS']='2'
    with pytest.raises(ValueError,match='runtime'):
        e.verify_runtime(old,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,
                         active_root=d.ROOT,relocation_record=record())


def test_historical_export_is_authenticated_against_its_own_source_bytes():
    from scripts import export_cosmology_oracle as e
    # This is the immutable historical export, not a result rebuilt in whichever
    # worktree is currently verifying it.
    path=e.ORACLE.parent/'extended-cache-repair/results/supplement-cosmology-20260930/69e33b3-20261004T040618Z/oracle'
    selected=json.loads((path/'export-index.json').read_text())
    index=json.loads((path/'worker-index.json').read_text())
    assert hasattr(e,'authenticate_historical_export'), 'explicit historical exporter authentication is missing'
    if sys.executable != index['runtime']['interpreter']:
        with pytest.raises(ValueError, match='oracle worker interpreter differs'):
            e.authenticate_historical_export(path, selected, index)
        _verify_in_historical_runtime('test_historical_export_is_authenticated_against_its_own_source_bytes')
        return
    receipt=e.authenticate_historical_export(path,selected,index)
    assert receipt['historical_exporter']['sha256']==index['exporter_sha256']
    assert receipt['active_verifier']['sha256']!=index['exporter_sha256']
    assert receipt['historical_exact_path_replay'] is False
    changed=copy.deepcopy(index);changed['exporter_sha256']='0'*64
    with pytest.raises(ValueError,match='source|index'):
        e.authenticate_historical_export(path,selected,changed)
    changed=copy.deepcopy(index);changed['exporter_path']='/unapproved/export_cosmology_oracle.py'
    with pytest.raises(ValueError,match='source|index'):
        e.authenticate_historical_export(path,selected,changed)
