"""Isolated lifecycle tests; all writable artifacts are temporary fixtures."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('original_artifact_fixtures', ROOT / 'tests/test_revision_artifacts.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def api():
    assert importlib.util.find_spec('artifact_checker.certification') is not None, 'certification interface is missing'
    from artifact_checker import certification
    return certification


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False))


@pytest.fixture
def source(tmp_path):
    result = fixtures.fixture_result()
    provenance = fixtures.provenance()
    producer = tmp_path / 'producer'
    bundle = tmp_path / 'gates.json'
    argv = [sys.executable, 'scripts/run_revision.py', '--case', 'B', '--execute',
            '--output-root', str(producer), '--gates', str(bundle)]
    provenance.update(command=' '.join(argv[1:]), interpreter={'path':sys.executable},
        source_file_hashes={str(ROOT / 'pfsfog/revision_artifacts.py'):digest(ROOT / 'pfsfog/revision_artifacts.py')})
    path = fixtures.write_revision_artifacts(result, producer, provenance, fixtures.gates())
    gates = fixtures.gates('passed', root=tmp_path, result=result, source=provenance)
    write_json(bundle, {'runs':{path.name:gates}})
    job = tmp_path / 'job'
    job.mkdir()
    for name, data in {'argv.txt':'\n'.join(argv)+'\n', 'exit-code.txt':'0\n',
                       'stdout.log':'fixture producer exited normally\n', 'stderr.log':''}.items():
        (job / name).write_text(data)
    diagnosis = tmp_path / 'diagnosis.md'
    diagnosis.write_text('Test-only numerical checker normalization diagnosis.\n')
    return dict(source=path, destination_root=tmp_path / 'certified', gate_bundle=bundle,
                job_directory=job, diagnosis=diagnosis)


def certify(source):
    return api().certify_artifact(**source)


def load(result):
    return api().load_revision_artifacts(result.path, receipt_sha256=result.receipt_sha256)


def test_building_round_trip_preserves_bytes_and_science(source):
    before = {p.name:p.read_bytes() for p in source['source'].iterdir()}
    result = certify(source)
    manifest, arrays, rows, validation = load(result)
    original = json.loads(before['manifest.json'])
    assert manifest['status'] == 'complete'
    assert manifest['run_id'] == source['source'].name
    for key in set(original) - {'status', 'artifacts'}:
        assert manifest[key] == original[key]
    assert {p.name:p.read_bytes() for p in source['source'].iterdir()} == before
    for name in ('matrices.npz', 'summary.csv'):
        assert (result.path / name).read_bytes() == before[name]
        assert os.stat(result.path / name).st_ino != os.stat(source['source'] / name).st_ino
    assert len(rows) == 11
    assert validation['save_reload']['status'] == 'passed'
    receipt = json.loads((result.path / 'certification.json').read_text())
    assert receipt['original_status'] == 'building'
    assert receipt['execution']['result'] == 'passed'


def test_complete_original_still_requires_receipt(source):
    fixtures.finalize_revision_artifacts(source['source'], json.loads(source['gate_bundle'].read_text()))
    with pytest.raises(ValueError, match='receipt|certification'):
        api().load_revision_artifacts(source['source'], receipt_sha256='0'*64)
    assert load(certify(source))[0]['status'] == 'complete'


@pytest.mark.parametrize('status', ['pending', 'failed'])
def test_normal_exit_requires_original_save_reload_success(source, status):
    validation_path = source['source'] / 'validation.json'
    validation = json.loads(validation_path.read_text())
    validation['save_reload']['status'] = status
    write_json(validation_path, validation)
    manifest_path = source['source'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['validation.json'] = digest(validation_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='save/reload'):
        certify(source)
    assert not source['destination_root'].exists()


@pytest.mark.parametrize('change', ['wrong_run', 'pending', 'failed', 'measurement', 'provenance'])
def test_bad_gate_bundle_rejects_atomically(source, change):
    bundle = json.loads(source['gate_bundle'].read_text())
    records = bundle['runs'][source['source'].name]
    if change == 'wrong_run':
        bundle['runs'] = {'other':records}
    elif change in ('pending', 'failed'):
        records['domain']['status'] = change
    elif change == 'measurement':
        records['domain']['measured'] = {'value':99}
    else:
        p = Path(records['domain']['evidence_path'])
        journal = json.loads(p.read_text())
        journal['provenance']['network_hash'] = 'different'
        write_json(p, journal)
        for record in records.values():
            record['evidence_hash'] = digest(p)
    write_json(source['gate_bundle'], bundle)
    with pytest.raises(ValueError):
        certify(source)
    assert not source['destination_root'].exists()


@pytest.mark.parametrize('filename', ['matrices.npz', 'summary.csv', 'validation.json', 'run.log', 'manifest.json'])
def test_source_tampering_after_certification_rejects(source, filename):
    result = certify(source)
    path = source['source'] / filename
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='source|lineage|checksum'):
        load(result)


@pytest.mark.parametrize('filename', ['matrices.npz', 'summary.csv', 'validation.json', 'run.log', 'certification.json'])
def test_destination_tampering_rejects_even_rehashed_manifest(source, filename):
    result = certify(source)
    path = result.path / filename
    path.write_bytes(path.read_bytes()+b' ')
    manifest = json.loads((result.path / 'manifest.json').read_text())
    manifest['artifacts'][filename] = digest(path)
    write_json(result.path / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        load(result)


def test_missing_receipt_no_downgrade(source):
    result = certify(source)
    (result.path / 'certification.json').unlink()
    manifest = json.loads((result.path / 'manifest.json').read_text())
    manifest.pop('certification')
    manifest['artifacts'].pop('certification.json')
    write_json(result.path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='receipt|certification'):
        load(result)


@pytest.mark.parametrize('target', ['journal', 'job', 'diagnosis'])
def test_original_evidence_tampering_rejects(source, target):
    result = certify(source)
    if target == 'journal':
        gates = json.loads(source['gate_bundle'].read_text())['runs'][source['source'].name]
        path = Path(gates['domain']['evidence_path'])
    elif target == 'job':
        path = source['job_directory'] / 'exit-code.txt'
    else:
        path = source['diagnosis']
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        load(result)


def test_source_checksum_corruption_rejects(source):
    (source['source'] / 'matrices.npz').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        certify(source)
    assert not source['destination_root'].exists()


@pytest.mark.parametrize('field', ['covariance', 'F_prior_cosmo', 'cell_F_like__000', 'growth_cov_marg'])
def test_rehashed_numerical_corruption_rejects(source, field):
    p = source['source'] / 'matrices.npz'
    with np.load(p) as data:
        arrays = {k:data[k] for k in data.files}
    arrays[field][0, 0] += 1
    np.savez(p, **arrays)
    manifest = json.loads((source['source'] / 'manifest.json').read_text())
    manifest['artifacts']['matrices.npz'] = digest(p)
    write_json(source['source'] / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        certify(source)
    assert not source['destination_root'].exists()


def test_faked_failed_status_is_not_recoverable(source):
    p = source['source'] / 'manifest.json'
    m = json.loads(p.read_text())
    m['status'] = 'failed'
    write_json(p, m)
    (source['job_directory'] / 'exit-code.txt').write_text('1\n')
    with pytest.raises(ValueError):
        certify(source)
    assert not source['destination_root'].exists()


@pytest.mark.parametrize('change', ['exit', 'argv', 'missing'])
def test_invalid_job_records_reject(source, change):
    job = source['job_directory']
    if change == 'exit':
        (job / 'exit-code.txt').write_text('1\n')
    elif change == 'argv':
        (job / 'argv.txt').write_text('/other/python\nscripts/run_revision.py\n')
    else:
        (job / 'stdout.log').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        certify(source)
    assert not source['destination_root'].exists()


def test_existing_destination_refused(source):
    source['destination_root'].mkdir()
    sentinel = source['destination_root'] / 'sentinel'
    sentinel.write_text('preserve')
    with pytest.raises(FileExistsError):
        certify(source)
    assert sentinel.read_text() == 'preserve'


@pytest.mark.parametrize('kind', ['symlink', 'hardlink'])
def test_alias_source_payload_rejected(source, kind):
    p = source['source'] / 'matrices.npz'
    moved = source['source'].parent / 'other.npz'
    p.rename(moved)
    if kind == 'symlink':
        p.symlink_to(moved)
    else:
        os.link(moved, p)
    with pytest.raises(ValueError, match='link|alias'):
        certify(source)
    assert not source['destination_root'].exists()


def test_failure_during_destination_build_leaves_source_unchanged(source, monkeypatch):
    module = api()
    before = {p.name:p.read_bytes() for p in source['source'].iterdir()}
    original = Path.write_bytes
    def fail_receipt(path, data):
        if path.name == 'certification.json':
            raise OSError('injected disk failure')
        return original(path, data)
    monkeypatch.setattr(Path, 'write_bytes', fail_receipt)
    with pytest.raises(OSError, match='disk failure'):
        certify(source)
    assert not source['destination_root'].exists()
    assert {p.name:p.read_bytes() for p in source['source'].iterdir()} == before


def test_modified_checker_copy_fails_integrity(source, monkeypatch):
    module = api()
    original = Path.read_bytes
    checker_path = ROOT / 'artifact_checker/_checker.py'
    def altered(path):
        data = original(path)
        return data+b'\n# changed\n' if path == checker_path else data
    monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(ValueError, match='checker|integrity'):
        certify(source)


@pytest.mark.parametrize('status', ['pending','failed'])
def test_rehashed_inner_scientific_failure_cannot_pass_outer(source, status):
    bundle = json.loads(source['gate_bundle'].read_text())
    records = bundle['runs'][source['source'].name]
    p = Path(records['domain']['evidence_path'])
    journal = json.loads(p.read_text())
    journal['gates']['domain']['status'] = status
    write_json(p, journal)
    for record in records.values():
        record['evidence_hash'] = digest(p)
    write_json(source['gate_bundle'], bundle)
    with pytest.raises(ValueError, match='journal|evidence'):
        certify(source)


def test_wrong_source_run_identity_rejects(source):
    path = source['source']/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['configuration']['kmax'] = .2
    write_json(path, manifest)
    with pytest.raises(ValueError, match='identity'):
        certify(source)


def test_receipt_pin_is_required_argument(source):
    result = certify(source)
    with pytest.raises(TypeError, match='receipt_sha256'):
        api().load_revision_artifacts(result.path)


def test_manifest_scientific_metadata_tampering_rejects(source):
    result = certify(source)
    p = result.path/'manifest.json'
    manifest = json.loads(p.read_text())
    manifest['cosmology']['headline_z'] = 1.3
    write_json(p, manifest)
    with pytest.raises(ValueError, match='scientific metadata'):
        load(result)


def test_incomplete_payload_rejected(source):
    (source['source']/'summary.csv').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        certify(source)


def test_summary_corruption_rehashed_still_rejected(source):
    import csv
    p = source['source']/'summary.csv'
    with p.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['sigma_marg'] = '999'
    with p.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((source['source']/'manifest.json').read_text())
    manifest['artifacts']['summary.csv'] = digest(p)
    write_json(source['source']/'manifest.json',manifest)
    with pytest.raises(ValueError, match='summary'):
        certify(source)


def test_missing_producer_source_identity_rejected(source):
    p = source['source']/'manifest.json'
    manifest = json.loads(p.read_text())
    del manifest['provenance']['source_file_hashes']
    write_json(p,manifest)
    with pytest.raises(ValueError, match='identity|frozen'):
        certify(source)


def test_wrong_loaded_original_module_location_rejected(source, monkeypatch):
    module = api()
    monkeypatch.setattr(module.original, '__file__', '/other/checkout/pfsfog/revision_artifacts.py')
    with pytest.raises(ValueError, match='loaded|location'):
        certify(source)


def test_failure_at_final_public_load_removes_complete_destination(source, monkeypatch):
    module = api()
    before = {p.name:p.read_bytes() for p in source['source'].iterdir()}
    def fail(*args, **kwargs):
        raise ValueError('injected consumer reload failure')
    monkeypatch.setattr(module, 'load_revision_artifacts', fail)
    with pytest.raises(ValueError, match='consumer reload failure'):
        certify(source)
    assert not source['destination_root'].exists()
    assert {p.name:p.read_bytes() for p in source['source'].iterdir()} == before
