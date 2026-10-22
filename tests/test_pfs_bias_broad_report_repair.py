"""A comparison summary belongs beside its four authenticated refinements."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.report_pfs_bias_broad_v2 import verify_comparison_inventory
from scripts.check_pfs_bias_broad_checkpoint import validate_stage_exits, validate_report_recovery


def comparison_directory(path):
    reports = {}
    for name in ('loop2048-to-loop4096.json', 'loop2048-to-angular64.json',
                 'angular64-to-shell304.json', 'shell304-to-final-loop.json'):
        content = b'{"status":"passed"}\n'
        (path/name).write_bytes(content)
        reports[name] = hashlib.sha256(content).hexdigest()
    summary = {'reports': reports}
    (path/'summary.json').write_text(json.dumps(summary))
    return summary


def test_summary_is_part_of_the_complete_comparison_directory(tmp_path):
    summary = comparison_directory(tmp_path)
    verify_comparison_inventory(tmp_path, summary)


@pytest.mark.parametrize('change', ['missing', 'changed', 'extra', 'stale_summary'])
def test_repaired_inventory_still_rejects_incomplete_or_changed_evidence(tmp_path, change):
    summary = comparison_directory(tmp_path)
    target = tmp_path/'loop2048-to-loop4096.json'
    if change == 'missing':
        target.unlink()
    elif change == 'changed':
        target.write_text('{"status":"failed"}')
    elif change == 'extra':
        (tmp_path/'unexpected.json').write_text('{}')
    else:
        (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(ValueError):
        verify_comparison_inventory(tmp_path, summary)


def failed_report_attempt():
    return {'status': 'stopped', 'after_stochastic_validation': True,
            'reason': 'numerical stage failed: report',
            'stages': [{'stage': name, 'actual_exit': code} for name, code in
                       [('inference', 0), ('common', 0), ('comparison', 0), ('report', 1)]]}


def test_separate_report_success_preserves_original_failed_exit():
    original = failed_report_attempt()
    saved = deepcopy(original)
    recovery = {'schema': 'pfs-bias-broad-report-recovery-v1',
                'status': 'passed', 'stage': 'report', 'actual_exit': 0}
    validate_stage_exits(original, recovery)
    assert original == saved
    for change in ('earlier_failure', 'missing_stage', 'recovery_failed', 'recovery_missing'):
        bad = deepcopy(original)
        repaired = deepcopy(recovery)
        if change == 'earlier_failure':
            bad['stages'][2]['actual_exit'] = 1
        elif change == 'missing_stage':
            bad['stages'].pop(0)
        elif change == 'recovery_failed':
            repaired['actual_exit'] = 1
        else:
            repaired = None
        with pytest.raises(ValueError):
            validate_stage_exits(bad, repaired)


@pytest.mark.parametrize('change', [None, 'original', 'source', 'command', 'input', 'log', 'output', 'extra_output'])
def test_separate_report_is_bound_to_the_failed_attempt_and_unchanged_evidence(tmp_path, change):
    root = Path(__file__).resolve().parents[1]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    original = failed_report_attempt()
    attempt = tmp_path/'continuation-after-validation.json'
    attempt.write_text(json.dumps(original))
    comparisons = tmp_path/'comparisons'
    comparisons.mkdir()
    summary = comparisons/'summary.json'
    summary.write_text('{"status":"passed"}')
    output = tmp_path/'report'
    output.mkdir()
    result = output/'report.json'
    result.write_text('{"rows":[]}')
    log = tmp_path/'report-recovery.log'
    log.write_text('report finished\n')
    receipt = {'schema': 'pfs-bias-broad-report-recovery-v1', 'stage': 'report',
               'status': 'passed', 'actual_exit': 0,
               'original_continuation_sha256': digest(attempt),
               'sources': {str(root/'scripts'/name): digest(root/'scripts'/name) for name in
                           ('report_pfs_bias_broad.py', 'report_pfs_bias_broad_v2.py',
                            'check_pfs_bias_broad_checkpoint.py')},
               'command': [str(root/'.venv/bin/python'), str(root/'scripts/report_pfs_bias_broad_v2.py'),
                           '--root', str(tmp_path), '--output', str(output)],
               'inputs': {str(summary): digest(summary)},
               'log': str(log), 'log_sha256': digest(log), 'outputs': {'report.json': digest(result)}}
    if change == 'original':
        attempt.write_text('{}')
    elif change == 'source':
        receipt['sources'][str(root/'scripts/report_pfs_bias_broad_v2.py')] = 'wrong'
    elif change == 'command':
        receipt['command'][-1] = str(tmp_path/'unrelated-report')
    elif change == 'input':
        summary.write_text('{}')
    elif change == 'log':
        log.write_text('different attempt\n')
    elif change == 'output':
        result.write_text('{"changed":true}')
    elif change == 'extra_output':
        (output/'unexpected.json').write_text('{}')
    (tmp_path/'report-recovery.json').write_text(json.dumps(receipt))
    if change is None:
        assert validate_report_recovery(tmp_path, original) == receipt
    else:
        with pytest.raises(ValueError):
            validate_report_recovery(tmp_path, original)
