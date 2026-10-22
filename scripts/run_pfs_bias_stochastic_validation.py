#!/usr/bin/env python
"""Record distinct validation exits for the unchanged broad-bias spectra."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess
import time
import traceback

from pfsfog.elf_survey_evidence import artifact_hashes,canonical,cell_directory,file_digest,verify_files,write_json
from pfsfog.extended_artifacts import read_json
from scripts.pfs_bias_stochastic_evidence import (
    correct_saved_cell,verify_corrected_cell,validation_sources)
from scripts.run_pfs_bias_broad import GRIDS,check_campaign
from scripts.run_elf_studies import check_bindings


def original_finished(record):
    if record['status']=='running':return False
    rows=record['controllers']
    if (record['status'] not in ('passed','failed') or len(rows)!=len(GRIDS)
            or {r['grid'] for r in rows}!=set(GRIDS)
            or any(r['actual_exit'] not in (0,1) for r in rows)):
        raise ValueError('original physical attempt is incomplete or interrupted')
    if (record['status']=='passed')!=all(r['actual_exit']==0 for r in rows):
        raise ValueError('original controller status contradicts exits')
    return True


def validate_original_grid(summary,expected):
    rows=summary['workers']
    if (summary['status'] not in ('passed','failed') or len(rows)!=35
            or [r['key'] for r in rows]!=[r['key'] for r in expected]
            or len({r['key'] for r in rows})!=35
            or any((r['status'],r['exit_code']) not in (('passed',0),('failed',1)) for r in rows)):
        raise ValueError('original physical grid lacks complete exact populations and exits')
    if (summary['status']=='passed')!=all(r['status']=='passed' for r in rows):
        raise ValueError('original grid summary contradicts workers')


def validate_validation_execution(record):
    if (record['status']!='passed' or len(record['controllers'])!=len(GRIDS)
            or {r['grid'] for r in record['controllers']}!=set(GRIDS)
            or any(r['actual_exit']!=0 for r in record['controllers'])):
        raise ValueError('five successful actual validation controller exits are required')


def validation_root(root):return Path(root)/'stochastic-validation'


def physical_directory(root,grid):return validation_root(root)/'physical'/grid


def campaign_sources():
    return {**validation_sources(),str(Path(__file__).resolve()):file_digest(Path(__file__))}


def prepare(root):
    root=Path(root).resolve();campaign=check_campaign(root)
    execution=read_json(root/'physical-execution.json')
    if not original_finished(execution):raise ValueError('original physical attempt is still running')
    bindings={str(root/name):file_digest(root/name) for name in ('request.json','state.json','physical-execution.json')}
    for grid in GRIDS:
        base=root/'physical'/grid;summary=read_json(base/'summary.json')
        validate_original_grid(summary,campaign['physical_requests'][grid]['expected'])
        for name in ('request.json','summary.json'):bindings[str(base/name)]=file_digest(base/name)
    directory=validation_root(root);directory.mkdir(parents=True,exist_ok=False)
    request={'schema':'pfs-bias-stochastic-campaign-v1','kind':'saved_array_validation',
        'original_campaign':str(root),'sources':campaign_sources(),'original_bindings':bindings,
        'physical_populations':175,'physical_arrays_changed':False,
        'original_controller_exits':{r['grid']:r['actual_exit'] for r in execution['controllers']}}
    write_json(directory/'request.json',request)
    return request


def check_request(root):
    root=Path(root).resolve();request=read_json(validation_root(root)/'request.json')
    if (request['schema']!='pfs-bias-stochastic-campaign-v1'
            or request['kind']!='saved_array_validation' or request['original_campaign']!=str(root)
            or request['sources']!=campaign_sources() or request['physical_populations']!=175
            or request['physical_arrays_changed'] is not False):
        raise ValueError('stochastic validation request changed')
    check_bindings(request['original_bindings']);check_campaign(root)
    return request


def original_records(root,grid):
    request=read_json(Path(root)/'physical'/grid/'request.json')
    summary=read_json(Path(root)/'physical'/grid/'summary.json')
    validate_original_grid(summary,request['expected'])
    return request,summary


def run_grid(root,grid):
    root=Path(root).resolve();bound=check_request(root);request,original=original_records(root,grid)
    directory=physical_directory(root,grid);directory.mkdir(parents=True,exist_ok=False)
    summary={'kind':'saved_array_validation','status':'running','grid':grid,'workers':[],
        'validation_request_sha256':file_digest(validation_root(root)/'request.json'),
        'sources':bound['sources'],'original_summary_sha256':file_digest(root/'physical'/grid/'summary.json')}
    write_json(directory/'summary.json',summary)
    for expected,raw in zip(request['expected'],original['workers']):
        key=expected['key'];command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),
                                    '--grid',grid,'--worker',key]
        started=time.monotonic();log=directory/(key+'.log')
        with log.open('x') as stream:process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row={'key':key,'kind':'saved_array_validation','status':'failed','exit_code':process.returncode,
             'original_actual_exit':raw['exit_code'],'command':command,'log':str(log),
             'elapsed_seconds':time.monotonic()-started}
        path=cell_directory(directory,key)
        try:
            row['verification']=verify_corrected_cell(path,expected,request,raw)
            if process.returncode==0:row['status']='passed'
        except Exception as error:row['error']=str(error)
        row['files']=artifact_hashes(path) if path.exists() else {}
        summary['workers'].append(row);write_json(directory/'summary.json',summary)
        print(grid+' '+key+': validation '+row['status']+f'; actual exit {process.returncode}',flush=True)
    check_request(root)
    summary['status']='passed' if all(r['status']=='passed' for r in summary['workers']) else 'failed'
    write_json(directory/'summary.json',summary)
    if summary['status']!='passed':raise ValueError('stochastic validation contains failed populations')
    return summary


def verify_validated_physical(root,grid):
    root=Path(root).resolve();bound=check_request(root)
    execution=read_json(validation_root(root)/'execution.json');validate_validation_execution(execution)
    request,original=original_records(root,grid);directory=physical_directory(root,grid)
    summary=read_json(directory/'summary.json')
    if (summary['kind']!='saved_array_validation' or summary['status']!='passed'
            or summary['grid']!=grid or summary['sources']!=bound['sources']
            or summary['validation_request_sha256']!=file_digest(validation_root(root)/'request.json')
            or summary['original_summary_sha256']!=file_digest(root/'physical'/grid/'summary.json')
            or len(summary['workers'])!=35
            or [r['key'] for r in summary['workers']]!=[r['key'] for r in request['expected']]):
        raise ValueError('incomplete or changed supplementary validation grid')
    checks=[]
    for expected,raw,row in zip(request['expected'],original['workers'],summary['workers']):
        if (row['status']!='passed' or row['exit_code']!=0 or row['kind']!='saved_array_validation'
                or row['original_actual_exit']!=raw['exit_code']):
            raise ValueError('supplementary validation lacks a successful actual worker exit')
        path=cell_directory(directory,row['key']);problems=verify_files(path,row['files'])
        if problems:raise ValueError('; '.join(problems))
        checks.append(verify_corrected_cell(path,expected,request,raw))
    return {'status':'passed','grid':grid,'kind':'saved_array_validation','populations':checks,
            'physical_directory':str(directory)}


def run_all(root):
    root=Path(root).resolve();prepare(root);directory=validation_root(root)
    summary={'kind':'saved_array_validation','status':'running','controllers':[],'max_concurrent':2}
    write_json(directory/'execution.json',summary)
    def one(grid):
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--grid',grid]
        log=directory/(grid+'.log');started=time.monotonic()
        with log.open('x') as stream:process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        return {'grid':grid,'actual_exit':process.returncode,'command':command,'log':str(log),
                'elapsed_seconds':time.monotonic()-started}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(one,g) for g in GRIDS]
        for future in as_completed(jobs):
            row=future.result();summary['controllers'].append(row);write_json(directory/'execution.json',summary)
            print('Validation '+row['grid']+f': actual exit {row["actual_exit"]}',flush=True)
    summary['status']='passed' if all(r['actual_exit']==0 for r in summary['controllers']) else 'failed'
    write_json(directory/'execution.json',summary);validate_validation_execution(summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--grid',choices=tuple(GRIDS));parser.add_argument('--worker')
    parser.add_argument('--wait',action='store_true');args=parser.parse_args();root=args.root.resolve()
    if args.wait:
        while not original_finished(read_json(root/'physical-execution.json')):time.sleep(15)
    if args.worker:
        if not args.grid:parser.error('--worker requires --grid')
        check_request(root);request,original=original_records(root,args.grid)
        expected=next(r for r in request['expected'] if r['key']==args.worker)
        row=next(r for r in original['workers'] if r['key']==args.worker)
        correct_saved_cell(cell_directory(root/'physical'/args.grid,args.worker),
            cell_directory(physical_directory(root,args.grid),args.worker),expected,request,row)
    elif args.grid:run_grid(root,args.grid)
    else:run_all(root)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:traceback.print_exc();sys.exit(1)
