#!/usr/bin/env python
"""Run the source-bound wider PFS bias experiment, retaining every actual exit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
import traceback

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_survey_evidence import (artifact_hashes, canonical, cell_directory,
    file_digest, request_digest, verify_files, write_json)
from pfsfog.extended_artifacts import read_json
from scripts.pfs_bias_evidence import verify_bias_cell
from scripts.pfs_bias_broad_evidence import (
    broad_source, make_broad_request, validate_broad_request, validate_broad_summary)
from scripts.pfs_bias_broad_physical import measure_broad_population
from scripts.run_elf_studies import GRID_PROFILES, REFINEMENTS, check_bindings

GRIDS=GRID_PROFILES
STUDY=ROOT/'results/elf-survey-studies-20261010-01'
PILOT=ROOT/'results/pfs-bias-pilot-20261010-02'


def prepare(root):
    pilot_checkpoint=ROOT/'docs/revision/evidence/pfs-bias-response-2026-10-10/BIAS_CHECKPOINT.json'
    closure=pilot_checkpoint.with_name('CHECKPOINT_CLOSURE.json');closed=read_json(closure)
    if (closed['write_actual_exit']!=0 or closed['verify_actual_exit']!=0
            or closed['checkpoint_sha256']!=file_digest(pilot_checkpoint)):
        raise ValueError('pilot checkpoint is not closed')
    bindings=read_json(PILOT/'request.json')['bindings']
    bindings.update({str(p):file_digest(p) for p in (PILOT/'request.json',pilot_checkpoint,closure)})
    check_bindings(bindings)
    cosmo=FiducialCosmology();old=read_json(STUDY/'request.json');requests={}
    for grid,settings in GRIDS.items():
        source=broad_source(settings)
        if canonical(source['details']['original'])!=canonical(old['sources'][grid]['details']['original']):
            raise ValueError('inherited physical model differs: '+grid)
        requests[grid]=make_broad_request(settings,cosmo,source)
    root.mkdir(parents=True,exist_ok=False)
    request={'schema':'pfs-bias-broad-v1','study':str(STUDY),'grids':GRIDS,
        'refinements':REFINEMENTS,'physical_requests':requests,'bindings':bindings,
        'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'fixed-density 35-population five-grid three-coordinate bias study'}
    write_json(root/'request.json',request)
    write_json(root/'state.json',{'status':'prepared','request_sha256':request_digest(request),
        'full_bias_grid_qualification':False})
    return {'status':'prepared','root':str(root),'populations_per_grid':35,'grids':list(GRIDS)}


def check_campaign(root,grid=None,*,check_references=True):
    root=Path(root);r=read_json(root/'request.json');state=read_json(root/'state.json')
    if (r['schema']!='pfs-bias-broad-v1' or r['grids']!=GRIDS
            or canonical(r['refinements'])!=canonical(REFINEMENTS)
            or r['study']!=str(STUDY) or request_digest(r)!=state['request_sha256']):
        raise ValueError('broad campaign request changed')
    if check_references:check_bindings(r['bindings'])
    cosmo=FiducialCosmology()
    for g in ([grid] if grid else GRIDS):
        validate_broad_request(r['physical_requests'][g],cosmo,broad_source(GRIDS[g]))
    return r


def run_physical(root,grid):
    campaign=check_campaign(root,grid);request=campaign['physical_requests'][grid]
    dest=root/'physical'/grid;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'request.json',request)
    summary={'status':'running','workers':[],'request_sha256':request_digest(request),
        'source_identity_hash':request['source']['identity_hash'],'complete_populations':False,
        'full_bias_grid_qualification':False}
    write_json(dest/'summary.json',summary)
    for expected in request['expected']:
        check_campaign(root,grid,check_references=False)
        key=expected['key'];path=cell_directory(dest,key);path.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),
            '--phase','physical','--grid',grid,'--worker',key]
        print('Starting '+grid+' '+key,flush=True);started=time.monotonic()
        with (path/'worker.log').open('w') as stream:
            process=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row={'key':key,'status':'failed','exit_code':process.returncode,'command':cmd,
            'elapsed_seconds':time.monotonic()-started,'request_sha256':request_digest(request),
            'source_identity_hash':request['source']['identity_hash']}
        try:
            row['verification']=verify_bias_cell(path,expected,request)
            if process.returncode==0:row['status']='passed'
        except Exception as error:row['error']=str(error)
        row['files']=artifact_hashes(path);summary['workers'].append(row)
        write_json(dest/'summary.json',summary)
        print('Finished '+grid+' '+key+': '+row['status']+f'; actual exit {process.returncode}',flush=True)
    check_campaign(root,grid)
    passed=len(summary['workers'])==35 and all(r['status']=='passed' for r in summary['workers'])
    summary.update(status='passed' if passed else 'failed',complete_populations=passed)
    write_json(dest/'summary.json',summary)
    if not passed:raise ValueError('broad physical populations failed: '+grid)
    validate_broad_summary(request,summary)
    return summary


def verify_physical(root,grid):
    campaign=check_campaign(root,grid);request=campaign['physical_requests'][grid]
    dest=root/'physical'/grid;summary=read_json(dest/'summary.json')
    if canonical(read_json(dest/'request.json'))!=canonical(request):
        raise ValueError('physical request changed')
    validate_broad_summary(request,summary);checks=[]
    for expected,row in zip(request['expected'],summary['workers']):
        path=cell_directory(dest,row['key']);problems=verify_files(path,row['files'])
        if problems:raise ValueError('; '.join(problems))
        checks.append(verify_bias_cell(path,expected,request))
    return {'status':'passed','grid':grid,'populations':checks}


def run_all(root,phase):
    check_campaign(root);output=root/(phase+'-execution.json')
    if output.exists():raise FileExistsError('execution record already exists')
    summary={'phase':phase,'status':'running','controllers':[],'max_concurrent':2}
    write_json(output,summary)
    def one(grid):
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),
                 '--phase',phase,'--grid',grid]
        started=time.monotonic();log=root/(phase+'-'+grid+'.log')
        with log.open('x') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        return {'grid':grid,'actual_exit':process.returncode,'command':command,
                'log':str(log),'elapsed_seconds':time.monotonic()-started}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs={executor.submit(one,g):g for g in GRIDS}
        for future in as_completed(jobs):
            row=future.result();summary['controllers'].append(row)
            write_json(output,summary);print(phase+' '+row['grid']+f': actual exit {row["actual_exit"]}',flush=True)
    summary['status']='passed' if all(r['actual_exit']==0 for r in summary['controllers']) else 'failed'
    write_json(output,summary)
    if summary['status']!='passed':raise ValueError('one or more grid controllers failed')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','physical','verify','inference','common'),required=True)
    parser.add_argument('--grid',choices=tuple(GRIDS));parser.add_argument('--worker')
    args=parser.parse_args();root=args.output.resolve();started=time.monotonic()
    try:
        if args.worker:
            if args.phase=='physical' and args.grid:
                request=read_json(root/'physical'/args.grid/'request.json')
                result=measure_broad_population(request,args.worker,cell_directory(root/'physical'/args.grid,args.worker))
                return 0 if result['status']=='passed' else 1
            if args.phase=='common':
                from scripts.pfs_bias_broad_comparison import common_population
                common_population(root,args.worker);return 0
            parser.error('physical workers require --grid; only physical/common accept --worker')
        if args.phase=='prepare':result=prepare(root)
        elif args.phase=='common':
            from scripts.pfs_bias_broad_comparison import run_common
            result=run_common(root)
        elif args.grid is None:result=run_all(root,args.phase)
        elif args.phase=='physical':result=run_physical(root,args.grid)
        elif args.phase=='verify':result=verify_physical(root,args.grid)
        else:
            from scripts.pfs_bias_broad_inference import run_grid
            result=run_grid(root,args.grid)
        print(args.phase+' '+str(args.grid)+': '+result['status'],flush=True);return 0
    except Exception:
        traceback.print_exc();return 1
    finally:print(f'Elapsed {time.monotonic()-started:.1f} seconds',flush=True)


if __name__=='__main__':sys.exit(main())
