import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / 'results/revision-qualification-d8dfcf4/production/I-k020-p1-all-ee4b092d0661'
BASE = ROOT / 'results/revision-qualification-d8dfcf4'


def command(*args):
    return subprocess.run([sys.executable, '-m', 'artifact_checker', *map(str,args)],
                          cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)


def test_cli_missing_pinned_receipt_rejects():
    result = command('check', REAL)
    assert result.returncode != 0
    assert 'receipt-sha256' in result.stderr


def test_real_i_private_process_recovery_and_consumer_reload():
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL.iterdir()}
    with tempfile.TemporaryDirectory(prefix='artifact-checker-pre-review-', dir='/private/tmp') as root:
        result = command('certify', '--source', REAL, '--destination-root', Path(root)/'certified',
            '--gate-bundle', BASE/'physical-baseline'/f'{REAL.name}.gates.json',
            '--job-directory', BASE/'produce-baseline-I',
            '--diagnosis', ROOT/'docs/revision/evidence/production-covariance-reload-diagnosis.md')
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output['status'] == 'certified'
        destination = Path(output['path'])
        checked = command('check', destination, '--receipt-sha256', output['receipt_sha256'])
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)['run_id'] == REAL.name
        receipt = json.loads((destination/'certification.json').read_text())
        assert receipt['original_status'] == 'failed'
        assert receipt['producer_exit_code'] == 1
        assert receipt['old_checker_rejection'] == 'covariance mismatch'
        assert receipt['execution']['environment']['executable'] == sys.executable
        assert receipt['execution']['environment']['argv'][1] == 'certify'
        for name in ('matrices.npz','summary.csv'):
            assert hashlib.sha256((destination/name).read_bytes()).hexdigest() == before[name]
        bad = command('check', destination, '--receipt-sha256', '0'*64)
        assert bad.returncode == 1
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL.iterdir()} == before
    assert json.loads((REAL/'manifest.json').read_text())['status'] == 'failed'
