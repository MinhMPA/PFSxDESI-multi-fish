"""Controlled certification of unchanged frozen-producer artifacts.

The supported loader requires an externally retained receipt digest. The private
materialized checker is never a receipt-aware consumer or a producer replacement.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
import tempfile
import time

import numpy as np
from pfsfog import revision_artifacts as original
from . import _checker as checker

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
ORIGINAL = ROOT / 'pfsfog/revision_artifacts.py'
ORIGINAL_SHA256 = '16a8ebe76ddcaabcfb9fa19bf354c8912e03bcd6733e2633f9b6ab06e3c04f17'
VERSION = 1
FILES = ('manifest.json', 'matrices.npz', 'summary.csv', 'validation.json', 'run.log')
JOB_FILES = ('argv.txt', 'exit-code.txt', 'stdout.log', 'stderr.log')
EXTERNAL = tuple(g for g in checker.MANDATORY_GATES if g != 'save_reload')
REASON = 'Frozen producer/checker normalization order mismatch; unchanged numerical bytes independently revalidated.'


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    return checker._strict_json(data.decode('utf-8'))


def _dump(value):
    return checker._canonical(value) + b'\n'


def _canonical_path(path):
    path = Path(path).absolute()
    if path != path.resolve():
        raise ValueError(f'unsafe path alias or symlink: {path}')
    return path


def _read_regular(path):
    path = _canonical_path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f'unsafe nonregular file or hard link: {path}')
        data = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed during read: {path}')
    return data


def _entry(path, data):
    return {'path':str(path), 'sha256':_digest(data), 'bytes':len(data)}


def _snapshot(directory, names):
    directory = _canonical_path(directory)
    if not directory.is_dir():
        raise ValueError(f'missing directory: {directory}')
    if {p.name for p in directory.iterdir()} != set(names):
        raise ValueError(f'incomplete or unexpected files in {directory}')
    return {name:_read_regular(directory / name) for name in names}


def _expected_checker(source):
    text = source.decode()
    for name in ('revision_forecast', 'revision_theory', 'eft_params'):
        text = text.replace(f'from .{name} import', f'from pfsfog.{name} import')
    text = text.replace('normalized = matrix / scale[:, None] / scale[None, :]',
        'normalized = (matrix + matrix.T) / (2 * scale[:, None] * scale[None, :])')
    text = text.replace('residual = np.max(np.abs(normalized @ inverse - np.eye(len(matrix))))',
        'product = np.einsum("ik,kj->ij", normalized, inverse, optimize=False)\n'
        '    if not np.all(np.isfinite(product)):\n'
        '        raise ValueError(f"{context} nonfinite scaled solve product")\n'
        '    residual = np.max(np.abs(product - np.eye(len(matrix))))')
    return text.encode()


def _code_hashes():
    # Include every Python source in the checker package (including tests) and
    # every frozen pfsfog source supporting imports. This is intentionally strict.
    paths = sorted([*PACKAGE.rglob('*.py'), *(ROOT / 'pfsfog').glob('*.py')])
    return {str(p):_digest(p.read_bytes()) for p in paths}


_IMPORTED_CODE = _code_hashes()


def verify_checker_integrity():
    if Path(original.__file__).resolve() != ORIGINAL or Path(checker.__file__).resolve() != PACKAGE / '_checker.py':
        raise ValueError('loaded checker module location mismatch')
    for name, module in tuple(sys.modules.items()):
        if name == 'pfsfog' or name.startswith('pfsfog.'):
            location = getattr(module, '__file__', None)
            if location is not None and Path(location).resolve().parent != ROOT / 'pfsfog':
                raise ValueError(f'loaded dependency module location mismatch: {name}')
    source = ORIGINAL.read_bytes()
    if _digest(source) != ORIGINAL_SHA256:
        raise ValueError('frozen original checker integrity mismatch')
    if (PACKAGE / '_checker.py').read_bytes() != _expected_checker(source):
        raise ValueError('materialized checker differs beyond the reviewed arithmetic/import changes')
    current = _code_hashes()
    if current != _IMPORTED_CODE:
        raise ValueError('checker source changed after process import; restart with an identified build')
    return current


def _science(manifest):
    return {k:v for k,v in manifest.items() if k not in ('status', 'artifacts', 'certification')}


def _payload(data):
    with np.load(io.BytesIO(data['matrices.npz']), allow_pickle=False) as npz:
        arrays = {key:npz[key] for key in npz.files}
    reader = csv.DictReader(io.StringIO(data['summary.csv'].decode(), newline=''))
    if tuple(reader.fieldnames or ()) != checker.SUMMARY_COLUMNS:
        raise ValueError('summary columns mismatch')
    return arrays, list(reader)


def _inspect_source(source, data, bundle, job, bundle_path):
    manifest = _json(data['manifest.json'])
    if manifest.get('schema_version') != checker.SCHEMA_VERSION or 'certification' in manifest:
        raise ValueError('source must be an original producer schema-2 artifact')
    run_id = manifest.get('run_id')
    if run_id != source.name or run_id != checker.run_identity(manifest['configuration'], manifest['provenance']):
        raise ValueError('source run identity mismatch')
    if manifest.get('status') not in ('building', 'complete', 'failed'):
        raise ValueError('unsupported original status')
    if set(manifest['artifacts']) != set(FILES) - {'manifest.json'}:
        raise ValueError('source artifact inventory mismatch')
    for name, expected in manifest['artifacts'].items():
        if _digest(data[name]) != expected:
            raise ValueError(f'source {name} checksum mismatch')
    if manifest['provenance'].get('source_file_hashes', {}).get(str(ORIGINAL)) != ORIGINAL_SHA256:
        raise ValueError('source does not identify the frozen original checker')
    external = bundle.get('runs', {}).get(run_id)
    if not isinstance(external, dict):
        raise ValueError('gate bundle has no exact run identity')
    checker._check_external_gates(external, run_id=run_id, require_passed=True,
        configuration=manifest['configuration'], provenance=manifest['provenance'])
    validation = _json(data['validation.json'])
    for gate in checker.MANDATORY_GATES:
        record = validation.get(gate)
        if not isinstance(record, dict) or not checker.GATE_FIELDS.issubset(record):
            raise ValueError(f'source gate {gate} is malformed')
        if record['status'] not in ('pending', 'passed', 'failed'):
            raise ValueError(f'source gate {gate} status invalid')
        if gate != 'save_reload':
            if record['status'] == 'failed':
                raise ValueError(f'source scientific gate {gate} failed')
            if record['status'] == 'passed' and record != external[gate]:
                raise ValueError(f'source gate {gate} differs from exact gate bundle')
    checker._check_external_gates(validation, run_id=run_id,
        require_passed=manifest['status']=='complete', configuration=manifest['configuration'],
        provenance=manifest['provenance'])
    if manifest['status'] != 'failed' and validation['save_reload']['status'] != 'passed':
        raise ValueError('normally exited original artifact lacks passed save/reload')
    argv = job['argv.txt'].decode().splitlines()
    if (len(argv) < 3 or argv[0] != manifest['provenance'].get('interpreter', {}).get('path')
            or ' '.join(argv[1:]) != manifest['provenance'].get('command')):
        raise ValueError('producer job argv/provenance mismatch')
    for flag, expected in (('--output-root', str(source.parent)), ('--gates', str(bundle_path))):
        if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or argv[argv.index(flag)+1] != expected:
            raise ValueError(f'producer job {flag} binding mismatch')
    exit_text = job['exit-code.txt'].decode().strip()
    if not re.fullmatch(r'0|[1-9][0-9]*', exit_text):
        raise ValueError('producer job exit record malformed')
    exit_code = int(exit_text)
    failed = manifest['status'] == 'failed'
    if exit_code != (1 if failed else 0):
        raise ValueError('producer job exit does not match source lifecycle')
    arrays, rows = _payload(data)
    old_error = None
    try:
        original._validate_payload(manifest, arrays, rows)
    except ValueError as exc:
        old_error = str(exc)
    if failed:
        if old_error != 'covariance mismatch':
            raise ValueError(f'failed source is not the demonstrated old covariance reload failure: {old_error}')
        if (not data['run.log'].decode().endswith('validation failure: covariance mismatch\n')
                or not job['stderr.log'].decode().rstrip().endswith('ValueError: covariance mismatch')
                or 'write_revision_artifacts' not in job['stderr.log'].decode()):
            raise ValueError('producer failure/log records do not establish covariance reload failure')
        if validation['save_reload']['status'] != 'pending':
            raise ValueError('failed source save/reload checkpoint unexplained')
    elif old_error is not None:
        raise ValueError(f'normal original source unexpectedly fails old checker: {old_error}')
    # Full check includes all elementwise covariance, priors, cells and summary.
    checker._validate_payload(manifest, arrays, rows)
    return manifest, validation, external, old_error, exit_code


def _environment():
    return {'executable':sys.executable, 'interpreter_sha256':_digest(Path(sys.executable).read_bytes()),
        'python':sys.version, 'platform':platform.platform(),
        'packages':{name:importlib.metadata.version(name) for name in ('numpy', 'scipy', 'jax', 'jaxlib')},
        'settings':{k:os.environ.get(k) for k in ('PYTHONPATH', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS', 'JAX_ENABLE_X64')},
        'cwd':str(Path.cwd()), 'argv':list(sys.argv)}


def _check_entries(entries, label):
    for entry in entries.values():
        data = _read_regular(entry['path'])
        if _digest(data) != entry['sha256'] or len(data) != entry['bytes']:
            raise ValueError(f'{label} lineage checksum mismatch: {entry["path"]}')


@dataclass(frozen=True)
class Certification:
    path: Path
    receipt_sha256: str


def certify_artifact(*, source, destination_root, gate_bundle, job_directory, diagnosis):
    """Certify unchanged original bytes into a new root; never mutate originals.

    Run in a separate process from numerical production. The returned digest is
    an integrity anchor to retain in independently reviewed selection metadata.
    """
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    source, destination_root, gate_bundle, job_directory, diagnosis = map(_canonical_path,
        (source, destination_root, gate_bundle, job_directory, diagnosis))
    if destination_root.exists():
        raise FileExistsError(destination_root)
    if source == destination_root or source in destination_root.parents or destination_root in source.parents:
        raise ValueError('destination aliases or contains source')
    if not destination_root.parent.is_dir():
        raise ValueError('destination parent must already exist')
    # Snapshot and hash originals, payloads and execution records before checks.
    data = _snapshot(source, FILES)
    job = _snapshot(job_directory, JOB_FILES)
    bundle_raw = _read_regular(gate_bundle)
    diagnosis_raw = _read_regular(diagnosis)
    source_entries = {n:_entry(source/n,b) for n,b in data.items()}
    job_entries = {n:_entry(job_directory/n,b) for n,b in job.items()}
    other_entries = {'gate_bundle':_entry(gate_bundle,bundle_raw), 'diagnosis':_entry(diagnosis,diagnosis_raw)}
    code = verify_checker_integrity()
    manifest, validation, external, old_error, exit_code = _inspect_source(
        source, data, _json(bundle_raw), job, gate_bundle)
    journals = {}
    for gate in EXTERNAL:
        path = _canonical_path(external[gate]['evidence_path'])
        raw = _read_regular(path)
        if _digest(raw) != external[gate]['evidence_hash']:
            raise ValueError('original gate journal checksum changed')
        journals[gate] = _entry(path, raw)
    final = destination_root / manifest['run_id']
    # Reservation uses exclusive mkdir; no rename can replace another root.
    destination_root.mkdir(mode=0o700)
    stage = None
    published = False
    try:
        stage = Path(tempfile.mkdtemp(prefix='.artifact-certification-', dir=destination_root.parent))
        run = stage / manifest['run_id']
        run.mkdir()
        for name, raw in data.items():
            (run/name).write_bytes(raw)
        updated = dict(validation)
        updated.update({gate:external[gate] for gate in EXTERNAL})
        updated['save_reload'] = {'status':'passed', 'measured':checker._reload_measurement(),
            'tolerance':1e-10, 'evidence_path':str(final),
            'evidence_hash':manifest['artifacts']['matrices.npz'], 'reason':None}
        (run/'validation.json').write_bytes(_dump(updated))
        (run/'run.log').write_bytes(data['run.log'] +
            f'\n{started} separate certification of original {manifest["status"]} run; {REASON}\n'.encode())
        next_manifest = dict(manifest)
        next_manifest['status'] = 'building'
        next_manifest['artifacts'] = {name:_digest((run/name).read_bytes()) for name in FILES if name != 'manifest.json'}
        (run/'manifest.json').write_bytes(_dump(next_manifest))
        checker._read_payload(run, require_complete=False)
        next_manifest['status'] = 'complete'
        (run/'manifest.json').write_bytes(_dump(next_manifest))
        checker._read_payload(run, require_complete=True)
        receipt = {'certification_version':VERSION, 'run_id':manifest['run_id'],
            'destination':str(final), 'source_directory':str(source), 'original_status':manifest['status'],
            'source_files':source_entries, 'job_files':job_entries, 'inputs':other_entries,
            'journals':journals, 'producer_exit_code':exit_code, 'old_checker_rejection':old_error,
            'configuration_sha256':_digest(_dump(manifest['configuration'])),
            'producer_provenance_sha256':_digest(_dump(manifest['provenance'])),
            'scientific_manifest_sha256':_digest(_dump(_science(manifest))),
            'original_checker_sha256':ORIGINAL_SHA256, 'checker_sources':code,
            'destination_artifacts':dict(next_manifest['artifacts']),
            'unchanged_numerical_files':{n:manifest['artifacts'][n] for n in ('matrices.npz','summary.csv')},
            'reason':REASON, 'execution':{'start_utc':started, 'end_utc':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':time.monotonic()-start, 'result':'passed',
                'environment':_environment(), 'operation':'certify_artifact',
                'arguments':{'source':str(source), 'destination_root':str(destination_root),
                    'gate_bundle':str(gate_bundle), 'job_directory':str(job_directory), 'diagnosis':str(diagnosis)},
                'result_scope':'full corrected payload and exact external gates validated before receipt issuance'}}
        raw_receipt = _dump(receipt)
        (run/'certification.json').write_bytes(raw_receipt)
        receipt_hash = _digest(raw_receipt)
        next_manifest['artifacts']['certification.json'] = receipt_hash
        next_manifest['certification'] = {'version':VERSION, 'receipt':'certification.json', 'sha256':receipt_hash}
        (run/'manifest.json').write_bytes(_dump(next_manifest))
        _load(run, receipt_sha256=receipt_hash, destination=final)
        # Source and code must still match before publication.
        for entries, label in ((source_entries,'source'),(job_entries,'job'),(other_entries,'input'),(journals,'gate')):
            _check_entries(entries,label)
        verify_checker_integrity()
        run.rename(final)
        published = True
        load_revision_artifacts(final, receipt_sha256=receipt_hash)
        return Certification(final, receipt_hash)
    except BaseException:
        if published:
            shutil.rmtree(final)
        destination_root.rmdir()
        raise
    finally:
        if stage is not None:
            shutil.rmtree(stage)


def _load(path, *, receipt_sha256, destination):
    path, destination = _canonical_path(path), _canonical_path(destination)
    if not isinstance(receipt_sha256,str) or not re.fullmatch('[0-9a-f]{64}',receipt_sha256):
        raise ValueError('an externally pinned receipt SHA256 is required')
    manifest = _json(_read_regular(path/'manifest.json'))
    expected = {'version':VERSION,'receipt':'certification.json','sha256':receipt_sha256}
    if manifest.get('certification') != expected:
        raise ValueError('missing or mismatched certification receipt descriptor')
    if manifest.get('artifacts',{}).get('certification.json') != receipt_sha256:
        raise ValueError('certification receipt missing from checksums')
    raw = _read_regular(path/'certification.json')
    if _digest(raw) != receipt_sha256:
        raise ValueError('certification receipt checksum mismatch')
    receipt = _json(raw)
    if (receipt.get('certification_version') != VERSION or receipt.get('run_id') != path.name
            or receipt.get('destination') != str(destination)):
        raise ValueError('certification receipt identity/location mismatch')
    code = verify_checker_integrity()
    if receipt.get('checker_sources') != code or receipt.get('original_checker_sha256') != ORIGINAL_SHA256:
        raise ValueError('checker source substitution')
    for key in ('source_files','job_files','inputs','journals'):
        _check_entries(receipt[key],key)
    source = _canonical_path(receipt['source_directory'])
    data = _snapshot(source,FILES)
    if receipt['source_files'] != {n:_entry(source/n,b) for n,b in data.items()}:
        raise ValueError('source lineage mapping mismatch')
    arguments = receipt['execution']['arguments']
    job_dir = _canonical_path(arguments['job_directory'])
    job = _snapshot(job_dir,JOB_FILES)
    if receipt['job_files'] != {n:_entry(job_dir/n,b) for n,b in job.items()}:
        raise ValueError('job lineage mapping mismatch')
    bundle_path = _canonical_path(receipt['inputs']['gate_bundle']['path'])
    original_manifest, _, external, old_error, exit_code = _inspect_source(
        source,data,_json(_read_regular(bundle_path)),job,bundle_path)
    if receipt['original_status'] != original_manifest['status'] or receipt['old_checker_rejection'] != old_error or receipt['producer_exit_code'] != exit_code:
        raise ValueError('original failure/certification lifecycle mismatch')
    if _science(manifest) != _science(original_manifest):
        raise ValueError('source scientific metadata substitution')
    for key,value in (('configuration_sha256',original_manifest['configuration']),
                      ('producer_provenance_sha256',original_manifest['provenance']),
                      ('scientific_manifest_sha256',_science(original_manifest))):
        if receipt[key] != _digest(_dump(value)):
            raise ValueError('source identity digest mismatch')
    for gate in EXTERNAL:
        entry = receipt['journals'][gate]
        if entry['path'] != str(Path(external[gate]['evidence_path']).resolve()) or entry['sha256'] != external[gate]['evidence_hash']:
            raise ValueError('gate journal lineage mismatch')
    if set(receipt['destination_artifacts']) != set(FILES)-{'manifest.json'}:
        raise ValueError('destination receipt inventory mismatch')
    for name, digest in receipt['destination_artifacts'].items():
        if _digest(_read_regular(path/name)) != digest or manifest['artifacts'].get(name) != digest:
            raise ValueError('destination payload/validation/log receipt checksum mismatch')
    for name in ('matrices.npz','summary.csv'):
        if receipt['unchanged_numerical_files'][name] != _digest(data[name]) or _read_regular(path/name) != data[name]:
            raise ValueError('unchanged numerical byte identity mismatch')
    result = checker._read_payload(path,require_complete=True)
    expected_reload = {'status':'passed','measured':checker._reload_measurement(),'tolerance':1e-10,
        'evidence_path':str(destination),'evidence_hash':manifest['artifacts']['matrices.npz'],'reason':None}
    if result[3]['save_reload'] != expected_reload or any(result[3][g] != external[g] for g in EXTERNAL):
        raise ValueError('certified gate/save-reload mismatch')
    for key in ('source_files','job_files','inputs','journals'):
        _check_entries(receipt[key],key)
    return result


def load_revision_artifacts(path, *, receipt_sha256):
    """Supported final-use loader: every input must have a pinned valid receipt."""
    return _load(path,receipt_sha256=receipt_sha256,destination=path)
