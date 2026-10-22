#!/usr/bin/env python
"""Reconstruct and bind the complete broad-bias study, including failed origins."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from datetime import datetime,timezone
import os
import subprocess
import tempfile

from pfsfog.elf_survey_evidence import canonical,file_digest,verify_files,write_json
from pfsfog.extended_artifacts import read_json
from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.pfs_bias_broad_inference import verify_inference
from scripts.report_pfs_bias_broad_v2 import report
from scripts.run_pfs_bias_broad import GRIDS,check_campaign
from scripts.run_pfs_bias_stochastic_validation import (
    original_finished,validate_validation_execution,validation_root)
from scripts.run_elf_studies import check_bindings

CAMPAIGN=ROOT/'results/pfs-bias-broad-20261010-01'
HERE=ROOT/'docs/revision/evidence/pfs-bias-broad-2026-10-10'
CHECKPOINT=HERE/'BROAD_CHECKPOINT.json'


def summarize_controls(grids):
    expected={r['key'] for r in broad_schedule()}
    totals=dict.fromkeys(('physical_populations','profile_calculations','original_array_controls',
                         'original_global_controls','full_inverse_controls','cross_information_controls'),0)
    if set(grids)!=set(GRIDS):raise ValueError('control inventory lacks five exact grids')
    for grid,record in grids.items():
        points=record['points']
        if set(points)!=expected:raise ValueError('control inventory has missing populations')
        for point in points.values():
            totals['physical_populations']+=1
            if point.get('reference_physical_check',{}).get('status')=='passed':
                totals['original_array_controls']+=1
            if point.get('cross_information_check',{}).get('status')=='passed':
                totals['cross_information_controls']+=1
            for out in point['profiles'].values():
                totals['profile_calculations']+=1
                if out.get('original_bias_covariance_check')=='passed':totals['original_global_controls']+=1
                if out.get('full_inverse_control',{}).get('status')=='passed':totals['full_inverse_controls']+=1
    required={'physical_populations':175,'profile_calculations':525,'original_array_controls':15,
              'original_global_controls':45,'full_inverse_controls':42,'cross_information_controls':175}
    if totals!=required:raise ValueError('measured control counts differ from the specification: '+str(totals))
    return totals


def validate_stage_exits(record,recovery=None):
    rows=record['stages']
    common=(record.get('after_stochastic_validation') is True
            and [r['stage'] for r in rows]==['inference','common','comparison','report'])
    uninterrupted=(recovery is None and record['status']=='numerical_stages_finished_review_pending'
                   and all(r['actual_exit']==0 for r in rows))
    repaired=(recovery is not None and record['status']=='stopped'
              and record.get('reason')=='numerical stage failed: report'
              and [r['actual_exit'] for r in rows]==[0,0,0,1]
              and recovery.get('schema')=='pfs-bias-broad-report-recovery-v1'
              and recovery.get('status')=='passed' and recovery.get('stage')=='report'
              and recovery.get('actual_exit')==0)
    if not common or not (uninterrupted or repaired):
        raise ValueError('complete successful numerical continuation is required')


def validate_report_recovery(root,continuation):
    root=Path(root).resolve()
    recovery=read_json(root/'report-recovery.json')
    validate_stage_exits(continuation,recovery)
    if recovery['original_continuation_sha256']!=file_digest(root/'continuation-after-validation.json'):
        raise ValueError('report recovery refers to a different original attempt')
    expected_sources={str(ROOT/'scripts'/name):file_digest(ROOT/'scripts'/name) for name in
        ('report_pfs_bias_broad.py','report_pfs_bias_broad_v2.py','check_pfs_bias_broad_checkpoint.py')}
    if recovery['sources']!=expected_sources:
        raise ValueError('report recovery sources changed')
    command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/report_pfs_bias_broad_v2.py'),
             '--root',str(root),'--output',str(root/'report')]
    if recovery['command']!=command:
        raise ValueError('report recovery command differs from the stated calculation')
    expected_inputs={str(root/'comparisons/summary.json'):file_digest(root/'comparisons/summary.json')}
    if recovery['inputs']!=expected_inputs:
        raise ValueError('report recovery comparison summary changed')
    if (Path(recovery['log']).resolve()!=root/'report-recovery.log'
            or file_digest(recovery['log'])!=recovery['log_sha256']):
        raise ValueError('report recovery log changed')
    problems=verify_files(root/'report',recovery['outputs'])
    if problems:raise ValueError('; '.join(problems))
    return recovery


def validate_science(*,require_review=True):
    check_campaign(CAMPAIGN)
    original=read_json(CAMPAIGN/'physical-execution.json')
    if not original_finished(original):raise ValueError('physical measurement is incomplete')
    validation=read_json(validation_root(CAMPAIGN)/'execution.json')
    validate_validation_execution(validation)
    continuation=read_json(CAMPAIGN/'continuation-after-validation.json')
    recovery=validate_report_recovery(CAMPAIGN,continuation) if (CAMPAIGN/'report-recovery.json').exists() else None
    validate_stage_exits(continuation,recovery);check_bindings(continuation['sources'])
    for row in continuation['stages']:
        if file_digest(row['log'])!=row['log_sha256']:raise ValueError('numerical stage log changed')
    verified={g:verify_inference(CAMPAIGN,g) for g in GRIDS}
    checks=summarize_controls(verified)
    # Explicitly test path spelling: earlier report preparation exposed this failure mode.
    previous=Path.cwd()
    try:
        os.chdir(ROOT)
        with tempfile.TemporaryDirectory(prefix='pfs-bias-broad-check-') as temporary:
            temp=Path(temporary)
            absolute=report(CAMPAIGN,temp/'absolute')
            relative=report(CAMPAIGN.relative_to(ROOT),temp/'relative')
            if (canonical(absolute)!=canonical(relative)
                    or canonical(absolute)!=canonical(read_json(CAMPAIGN/'report/report.json'))):
                raise ValueError('saved scientific report is not independently reproducible')
            csv=[file_digest(p) for p in (temp/'absolute/response.csv',temp/'relative/response.csv',
                                         CAMPAIGN/'report/response.csv')]
            if len(set(csv))!=1:raise ValueError('saved response table differs from reconstruction')
    finally:os.chdir(previous)
    if (absolute['full_bias_grid_qualification'] is not True or len(absolute['rows'])!=140
            or len(absolute['rankings'])!=48):raise ValueError('incomplete qualified scientific report')
    for base in ('global-bias-response','cross-spectrum-bias-response'):
        for suffix in ('pdf','svg','png'):
            if not (CAMPAIGN/'report'/f'{base}.{suffix}').is_file():raise ValueError('missing scientific figure')
    original_rows=[row for g in GRIDS for row in read_json(CAMPAIGN/'physical'/g/'summary.json')['workers']]
    checks.update(original_failed_workers=sum(r['exit_code']!=0 for r in original_rows),
                  original_controller_exits={r['grid']:r['actual_exit'] for r in original['controllers']},
                  common_k_populations=absolute['common_k_populations'],report_rows=len(absolute['rows']),
                  maximum_relative_global_refinement_change=absolute['maximum_relative_global_refinement_change'])
    if require_review:
        exits=read_json(HERE/'numerical-exits.json')
        required=('stochastic-validation-batch','affected-tests',
                  'checkpoint-tests','full-suite','pilot-checkpoint','density-checkpoint')
        if any(exits.get(name)!=0 for name in required):raise ValueError('missing successful actual verification exit')
        if recovery is None:
            if exits.get('continuation-after-validation')!=0:raise ValueError('missing successful continuation exit')
        elif (exits.get('continuation-after-validation')!=1 or exits.get('report-recovery')!=0
              or exits.get('report-repair-tests')!=0 or exits.get('report-repair-full-suite')!=0):
            raise ValueError('original failure or successful report-repair verification exit missing')
        disposition=read_json(HERE/'REVIEW_DISPOSITION.json')
        if (disposition['status']!='passed' or disposition['review_type']!='independent'
                or disposition['unresolved_important'] or disposition['unresolved_critical']):
            raise ValueError('independent scientific review remains incomplete')
        check_bindings(disposition['reviewed_sources'])
        if file_digest(HERE/disposition['review_file'])!=disposition['review_sha256']:
            raise ValueError('independent review record changed')
        figures=read_json(HERE/'FIGURE_REVIEW.json')
        if figures['status']!='passed':raise ValueError('scientific figures have not been inspected')
        check_bindings(figures['files'])
        checks['actual_process_exits']=exits
    return checks


def inventory():
    files={p for p in CAMPAIGN.rglob('*') if p.is_file()}
    files.update(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name not in ('BROAD_CHECKPOINT.json','CHECKPOINT_CLOSURE.json'))
    patterns=('pfs_bias_broad_*.py','pfs_bias_stochastic_*.py','run_pfs_bias_stochastic_validation.py',
              'run_pfs_bias_broad.py','continue_pfs_bias_broad.py','report_pfs_bias_broad.py',
              'report_pfs_bias_broad_v2.py',
              'check_pfs_bias_broad_checkpoint.py')
    for pattern in patterns:files.update((ROOT/'scripts').glob(pattern))
    for pattern in ('test_pfs_bias_broad*.py','test_pfs_bias_stochastic*.py','test_continue_pfs_bias_broad.py'):
        files.update((ROOT/'tests').glob(pattern))
    files.update(ROOT/p for p in (
        'docs/revision/specs/2026-10-10-pfs-bias-broad.md',
        'docs/revision/specs/2026-10-10-pfs-bias-stochastic-zero-repair.md',
        'docs/revision/specs/2026-10-11-pfs-bias-report-inventory-repair.md',
        'docs/superpowers/plans/2026-10-10-pfs-bias-broad.md',
        'docs/revision/issues/pfs-bias-broad-response.md',
        'docs/revision/PFS_BIAS_BROAD_WORKING_STATE.json','docs/revision/PFS_BIAS_BROAD_HANDOFF.md',
        'docs/revision/PFS_BIAS_BROAD_NEXT_AGENT_PROMPT.md'))
    return {str(p.relative_to(ROOT)):{'sha256':file_digest(p),'bytes':p.stat().st_size} for p in sorted(files)}


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for name in ('write','verify','preflight'):group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.verify:
        saved=read_json(CHECKPOINT)
        if saved['status']!='complete' or saved['full_bias_grid_qualification'] is not True:
            raise ValueError('invalid broad checkpoint scope')
        for name,value in saved['files'].items():
            path=ROOT/name
            if path.stat().st_size!=value['bytes'] or file_digest(path)!=value['sha256']:
                raise ValueError('changed broad checkpoint file: '+name)
    checks=validate_science(require_review=not args.preflight)
    if args.preflight:
        print({'status':'numerical_preflight_passed_review_not_certified',**checks});return
    if args.write:
        if CHECKPOINT.exists():raise FileExistsError('checkpoint already exists')
        saved={'schema':'pfs-bias-broad-checkpoint-v1','status':'complete',
            'full_bias_grid_qualification':True,'created_utc':datetime.now(timezone.utc).isoformat(),
            'prepared_from_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'checks':checks,'files':inventory()}
        write_json(CHECKPOINT,saved)
    elif canonical(checks)!=canonical(saved['checks']):raise ValueError('scientific checkpoint checks changed')
    print({'status':'passed','checkpoint_files':len(saved['files']),**checks})


if __name__=='__main__':main()
