#!/usr/bin/env python
"""Advance authorized numerical stages after physical qualification, retaining exits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from datetime import datetime,timezone
import json
import os
import subprocess
import time
import traceback

from pfsfog.elf_survey_evidence import file_digest, write_json
from pfsfog.extended_artifacts import read_json
from scripts.run_pfs_bias_broad import GRIDS, check_campaign
from scripts.pfs_bias_broad_inference import inference_sources
from scripts.pfs_bias_broad_comparison import comparison_sources
from scripts.run_elf_studies import check_bindings


def physical_ready(record):
    if record['status']=='running':return False
    rows=record['controllers']
    if (record['status']!='passed' or len(rows)!=len(GRIDS)
            or {r['grid'] for r in rows}!=set(GRIDS) or any(r['actual_exit']!=0 for r in rows)):
        raise ValueError('physical grid controllers did not all finish successfully')
    return True


def stage_commands(root):
    root=str(Path(root).resolve());runner=str(ROOT/'scripts/run_pfs_bias_broad.py')
    return [('inference',[sys.executable,runner,'--phase','inference','--output',root]),
            ('common',[sys.executable,runner,'--phase','common','--output',root]),
            ('comparison',[sys.executable,str(ROOT/'scripts/pfs_bias_broad_comparison.py'),
                           '--root',root,'--output',root+'/comparisons']),
            ('report',[sys.executable,str(ROOT/'scripts/report_pfs_bias_broad.py'),
                       '--root',root,'--output',root+'/report'])]


def ensure_fresh_outputs(root):
    for name in ('inference','inference-execution.json','common-k','comparisons','report'):
        if (Path(root)/name).exists():raise FileExistsError('retained stage output already exists: '+name)


def continue_stages(root,*,timeout=10800,poll_seconds=15,after_stochastic_validation=False):
    root=Path(root).resolve();check_campaign(root);ensure_fresh_outputs(root)
    prefix='continuation-after-validation' if after_stochastic_validation else 'continuation'
    path=root/(prefix+'.json')
    if path.exists():raise FileExistsError('continuation receipt already exists')
    sources={**inference_sources(),**comparison_sources()}
    for file in (Path(__file__).resolve(),ROOT/'scripts/report_pfs_bias_broad.py'):
        sources[str(file)]=file_digest(file)
    gate=(root/'stochastic-validation/execution.json' if after_stochastic_validation
          else root/'physical-execution.json')
    record={'schema':'pfs-bias-broad-continuation-v1',
            'status':'waiting_for_stochastic_validation' if after_stochastic_validation else 'waiting_for_physical',
            'qualification_receipt':str(gate),'after_stochastic_validation':after_stochastic_validation,
            'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat(),
            'sources':sources,'request_sha256':file_digest(root/'request.json'),
            'stages':[],'review_complete':False}
    write_json(path,record);started=time.monotonic()
    try:
        while True:
            try:ready=gate.exists() and physical_ready(read_json(gate))
            except json.JSONDecodeError:
                # The producer may be replacing its progress receipt at this instant.
                ready=False
            if ready:break
            if time.monotonic()-started>timeout:raise TimeoutError('required qualification batch is still incomplete')
            time.sleep(poll_seconds)
        record['qualification_receipt_sha256']=file_digest(gate)
        for stage,command in stage_commands(root):
            check_bindings(sources);check_campaign(root)
            record['status']='running_'+stage;write_json(path,record)
            log=root/(prefix+'-'+stage+'.log');stage_start=time.monotonic()
            print('Starting '+stage,flush=True)
            with log.open('x') as stream:
                process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            row={'stage':stage,'command':command,'actual_exit':process.returncode,
                 'elapsed_seconds':time.monotonic()-stage_start,'log':str(log),'log_sha256':file_digest(log)}
            record['stages'].append(row);write_json(path,record)
            print('Finished '+stage+f': actual exit {process.returncode}',flush=True)
            if process.returncode!=0:raise ValueError('numerical stage failed: '+stage)
        check_bindings(sources);check_campaign(root)
        record['status']='numerical_stages_finished_review_pending'
    except Exception as error:
        record.update(status='stopped',reason=str(error),error_type=type(error).__name__)
        (root/(prefix+'-failure.log')).write_text(traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        record['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(path,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=10800)
    parser.add_argument('--after-stochastic-validation',action='store_true');args=parser.parse_args()
    print(continue_stages(args.root,timeout=args.timeout,
        after_stochastic_validation=args.after_stochastic_validation)['status'])
