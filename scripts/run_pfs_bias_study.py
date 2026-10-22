#!/usr/bin/env python
"""Fresh fixed-density PFS population pilot with retained process evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import subprocess
import time
import traceback

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_survey_evidence import (artifact_hashes, canonical, cell_directory,
    file_digest, request_digest, verify_files, write_json)
from pfsfog.extended_artifacts import read_json
from scripts.pfs_bias_evidence import (bias_source, make_bias_request,
    validate_bias_request, verify_bias_cell)
from scripts.pfs_bias_physical import measure_bias_population
from scripts.run_elf_studies import GRID_PROFILES, check_bindings

GRIDS={k:GRID_PROFILES[k] for k in ('shell304','final-loop')}
STUDY=ROOT/'results/elf-survey-studies-20261010-01'


def prepare(root):
    density=ROOT/'results/pfs-density-response-20261010-02/request.json'
    checkpoint=ROOT/'docs/revision/evidence/pfs-density-response-2026-10-10/DENSITY_CHECKPOINT.json'
    bindings=read_json(density)['bindings']
    bindings.update({str(p):file_digest(p) for p in (density,checkpoint)})
    check_bindings(bindings)
    cosmo=FiducialCosmology();requests={}
    old=read_json(STUDY/'request.json')
    for grid,settings in GRIDS.items():
        source=bias_source(settings)
        if canonical(source['details']['original'])!=canonical(old['sources'][grid]['details']['original']):
            raise ValueError('inherited physical model differs from saved reference: '+grid)
        requests[grid]=make_bias_request(settings,cosmo,source=source)
    root.mkdir(parents=True,exist_ok=False)
    request={'schema':'pfs-bias-pilot-v1','study':str(STUDY),'grids':GRIDS,
        'physical_requests':requests,'bindings':bindings,
        'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'fixed-density two-loop-resolution three-coordinate bias pilot'}
    write_json(root/'request.json',request)
    write_json(root/'state.json',{'status':'prepared','request_sha256':request_digest(request),
        'full_bias_grid_qualification':'pending','active_jobs':[]})
    return {'status':'prepared','root':str(root),'populations_per_grid':15,'grids':list(GRIDS)}


def check_campaign(root,grid=None,*,check_references=True):
    r=read_json(root/'request.json');state=read_json(root/'state.json')
    if (r['schema']!='pfs-bias-pilot-v1' or r['grids']!=GRIDS
            or r['study']!=str(STUDY) or request_digest(r)!=state['request_sha256']):
        raise ValueError('bias campaign request changed')
    if check_references:check_bindings(r['bindings'])
    cosmo=FiducialCosmology()
    for g in ([grid] if grid else GRIDS):
        validate_bias_request(r['physical_requests'][g],cosmo,bias_source(GRIDS[g]))
    return r


def run_physical(root,grid):
    campaign=check_campaign(root,grid);request=campaign['physical_requests'][grid]
    dest=root/'physical'/grid;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'request.json',request)
    summary={'status':'running','workers':[],'request_sha256':request_digest(request),
        'source_identity_hash':request['source']['identity_hash'],'complete_populations':False,
        'full_bias_grid_qualification':'pending'}
    write_json(dest/'summary.json',summary)
    for expected in request['expected']:
        check_campaign(root,grid,check_references=False)
        key=expected['key'];p=cell_directory(dest,key);p.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),
            '--phase','physical','--grid',grid,'--worker',key]
        print('Starting '+grid+' '+key,flush=True);started=time.monotonic()
        with (p/'worker.log').open('w') as f:
            process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        row={'key':key,'status':'failed','exit_code':process.returncode,
             'command':cmd,'elapsed_seconds':time.monotonic()-started,
             'request_sha256':request_digest(request),
             'source_identity_hash':request['source']['identity_hash']}
        try:
            row['verification']=verify_bias_cell(p,expected,request)
            if process.returncode==0:row['status']='passed'
        except Exception as error:row['error']=str(error)
        row['files']=artifact_hashes(p);summary['workers'].append(row)
        write_json(dest/'summary.json',summary)
        print('Finished '+grid+' '+key+': '+row['status']+f'; actual exit {process.returncode}',flush=True)
    check_campaign(root,grid)
    passed=len(summary['workers'])==15 and all(r['status']=='passed' for r in summary['workers'])
    summary.update(status='passed' if passed else 'failed',complete_populations=passed)
    write_json(dest/'summary.json',summary)
    if not passed:raise ValueError('bias physical pilot has failed populations: '+grid)
    return summary


def verify_physical(root,grid):
    campaign=check_campaign(root,grid);request=campaign['physical_requests'][grid]
    dest=root/'physical'/grid;summary=read_json(dest/'summary.json')
    keys=[r['key'] for r in request['expected']]
    if (read_json(dest/'request.json')!=request or summary['request_sha256']!=request_digest(request)
            or summary['source_identity_hash']!=request['source']['identity_hash']
            or [r['key'] for r in summary['workers']]!=keys
            or summary['status']!='passed' or summary['complete_populations'] is not True):
        raise ValueError('incomplete or unrelated bias physical cohort')
    checks=[]
    for expected,row in zip(request['expected'],summary['workers']):
        if (row['status']!='passed' or row['exit_code']!=0
                or row['request_sha256']!=request_digest(request)
                or row['source_identity_hash']!=request['source']['identity_hash']):
            raise ValueError('physical population failed or belongs to another request')
        path=cell_directory(dest,row['key']);problems=verify_files(path,row['files'])
        if problems:raise ValueError('; '.join(problems))
        checks.append(verify_bias_cell(path,expected,request))
    return {'status':'passed','grid':grid,'populations':checks}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','physical','verify','inference'),required=True)
    parser.add_argument('--grid',choices=tuple(GRIDS));parser.add_argument('--worker')
    args=parser.parse_args();root=args.output.resolve();started=time.monotonic()
    if args.phase!='prepare' and args.grid is None:parser.error('--grid is required')
    if args.worker:
        if args.phase!='physical':parser.error('--worker requires physical phase')
        request=read_json(root/'physical'/args.grid/'request.json')
        result=measure_bias_population(request,args.worker,cell_directory(root/'physical'/args.grid,args.worker))
        return 0 if result['status']=='passed' else 1
    try:
        if args.phase=='prepare':result=prepare(root)
        elif args.phase=='physical':result=run_physical(root,args.grid)
        elif args.phase=='verify':result=verify_physical(root,args.grid)
        else:
            from scripts.pfs_bias_inference import run_grid
            result=run_grid(root,args.grid)
        if args.phase=='prepare':print(result,flush=True)
        else:print(args.phase+' '+str(args.grid)+': '+result['status'],flush=True)
        return 0
    except Exception:
        traceback.print_exc();return 1
    finally:
        print(f'Elapsed {time.monotonic()-started:.1f} seconds',flush=True)


if __name__=='__main__':sys.exit(main())
