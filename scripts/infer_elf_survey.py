#!/usr/bin/env python
"""Global B6/C6 inference and native3 controls at one verified ELF resolution."""
from dataclasses import replace
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ps_1loop_jax'));sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.cosmology_projection import COSMOLOGY_TREATMENTS
from pfsfog.elf_inference import load_physical_likelihood, infer_treatments
from pfsfog.elf_inference_evidence import write_case, verify_case, verify_physical_binding
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_survey_evidence import (artifact_hashes, file_digest, request_digest,
    source_identity, verify_files, verify_run, write_json)
from pfsfog.extended_artifacts import (read_json, jsonable, canonical, PRECISION_FIELDS, GROWTH_FIELDS)
from pfsfog.extended_validation import measure_growth_jacobian, measure_sigma8_radius, validate_sigma8_radius
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_forecast import build_revision_likelihood, apply_revision_priors
from pfsfog.surveys import default_survey_group

SCOPE='single_resolution_global_inference'


def constraint_ratios(outcomes):
    if set(outcomes)!={'B','C'} or any(set(c)!=set(COSMOLOGY_TREATMENTS) for c in outcomes.values()):
        raise ValueError('ratio calculation requires every B/C treatment outcome')
    ratios={}
    for name in COSMOLOGY_TREATMENTS:
        failed=[c for c in ('B','C') if outcomes[c][name]['status']=='failed']
        if failed:
            ratios[name]={'status':'unavailable','failed_cases':failed}
            continue
        b,c=outcomes['B'][name],outcomes['C'][name]
        if b['status']!='passed' or c['status']!='passed' or b['free_cosmology']!=c['free_cosmology']:
            raise ValueError('ratio treatment status or cosmology basis differs')
        row={'status':'available','free_cosmology':b['free_cosmology']}
        for kind in ('marginalized','nuisance_fixed'):
            row[kind]={'B_over_C':{},'C_over_B':{}}
            for field in ('cosmology','growth'):
                left,right=b[kind][field],c[kind][field]
                if field=='cosmology':
                    if set(left)!=set(right) or set(left)!=set(b['free_cosmology']):
                        raise ValueError('ratio cosmology labels differ')
                    keys=list(b['free_cosmology']);left=[left[k] for k in keys];right=[right[k] for k in keys]
                elif len(left)!=8 or len(right)!=8:
                    raise ValueError('ratio growth count differs')
                left,right=np.asarray(left),np.asarray(right)
                if not np.isfinite([left,right]).all() or np.any(left<=0) or np.any(right<=0):
                    raise ValueError('ratio errors must be positive and finite')
                for direction,value in (('B_over_C',left/right),('C_over_B',right/left)):
                    row[kind][direction][field]=(dict(zip(keys,value.tolist())) if field=='cosmology' else value.tolist())
        ratios[name]=row
    return ratios


def execution_problems(request,summary):
    problems=[];digest=request_digest(request);workers=summary.get('workers',[])
    if ([w.get('case') for w in workers]!=['B','C'] or summary.get('status')!='passed'
            or summary.get('scope')!=SCOPE or summary.get('full_forecast_qualification')!='pending'
            or summary.get('request_sha256')!=digest or summary.get('source_changed') is not False):
        problems.append('incomplete, failed or overstated inference execution')
    for worker in workers:
        if (worker.get('exit_code')!=0 or worker.get('status')!='passed'
                or worker.get('request_sha256')!=digest
                or worker.get('source_before')!=request['source_identity_hash']
                or worker.get('source_after')!=request['source_identity_hash']):
            problems.append('failed or foreign inference worker: '+str(worker.get('case')))
    return problems


def _read_outcomes(path):
    record=read_json(path/'manifest.json');out={}
    with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
        names=record['likelihood']['names'][:6]
        for name in COSMOLOGY_TREATMENTS:
            if name in record['failures']:
                out[name]={'status':'failed','error':record['failures'][name]['error']};continue
            free=record['treatments'][name]['free_cosmology']
            row={'status':'passed','free_cosmology':free}
            for kind,suffix in (('marginalized','marg'),('nuisance_fixed','nuisance_fixed')):
                variances=np.diag(arrays[name+'/cov_cosmo_'+suffix])
                growth=np.diag(arrays[name+'/growth_cov_'+suffix])
                if any(variances[names.index(n)]<=0 for n in free) or np.any(growth<=0):
                    raise ValueError('nonpositive reported variance')
                row[kind]={'cosmology':{n:float(np.sqrt(variances[names.index(n)])) for n in free},
                           'growth':np.sqrt(growth).tolist()}
            out[name]=row
    return out


def _check_request(request):
    if request.get('scope')!=SCOPE or request.get('schema')!='elf-global-inference-v1':
        raise ValueError('unsupported inference request')
    physical=Path(request['physical_root'])
    checked=verify_run(physical)
    if checked['status']!='passed' or not checked['complete_survey']:
        raise ValueError('complete passed physical cohort required')
    original=read_json(physical/'request.json')
    if (request['physical_request_sha256']!=file_digest(physical/'request.json')
            or request['physical_summary_sha256']!=file_digest(physical/'summary.json')
            or request['source_identity_hash']!=original['source']['identity_hash']
            or request['settings']!=original['settings']
            or request['program_sha256']!=file_digest(Path(__file__))):
        raise ValueError('physical input, numerical source or inference program changed')
    return original


def verify_growth_checkpoint(root,case,request,worker):
    path=root/'workers'/f'{case}-growth.json'
    if file_digest(path)!=worker['growth_sha256']:
        raise ValueError('completed growth checkpoint changed')
    record=read_json(path)
    if (record.get('case')!=case
            or record.get('source_identity_hash')!=request['source_identity_hash']
            or record.get('request_sha256')!=request_digest(request)
            or record.get('full_forecast_qualification')!='pending'
            or canonical(record.get('measurement'))!=canonical(read_json(root/'cases'/case/'growth.json'))):
        raise ValueError('completed growth measurement or source binding differs')


def run_worker(root,case):
    request=read_json(root/'request.json');start=time.monotonic()
    receipt={'case':case,'status':'running','request_sha256':request_digest(request),
             'source_before':None,'source_after':None,'full_forecast_qualification':'pending'}
    dest=root/'workers'/f'{case}.json';write_json(dest,receipt)
    try:
        _check_request(request)
        identity=source_identity(request['settings'])['identity_hash']
        receipt['source_before']=identity
        like=load_physical_likelihood(request['physical_root'],case)
        engine=make_one_loop_engine(nfft=request['settings']['nfft'])
        space=elf_cosmology_spec(like.reference_cosmology)
        z=np.array([(lo+hi)/2 for lo,hi in like.config.nominal_bins])
        print(case+': measuring growth AD/FD and sigma8 radius',flush=True)
        growth=measure_growth_jacobian(space,like.reference_cosmology,z)
        growth_path=root/'workers'/f'{case}-growth.json'
        write_json(growth_path,{'case':case,'source_identity_hash':identity,
            'request_sha256':request_digest(request),'full_forecast_qualification':'pending',
            'measurement':jsonable(growth)})
        receipt['growth_sha256']=file_digest(growth_path)
        write_json(dest,receipt)
        radius=measure_sigma8_radius(space,like.reference_cosmology,z)
        write_json(root/'workers'/f'{case}-sigma8-radius.json',radius)
        receipt['sigma8_radius']=validate_sigma8_radius(radius,space,like.reference_cosmology,z)
        def checkpoint(name,result,failure):
            partial=root/'partial'/case/name;partial.mkdir(parents=True,exist_ok=False)
            if result is not None:
                np.savez_compressed(partial/'arrays.npz',**{f:getattr(result,f) for f in (*PRECISION_FIELDS,*GROWTH_FIELDS)})
            write_json(partial/'outcome.json',jsonable({'treatment':name,'status':'passed' if result is not None else 'failed',
                'failure':failure,'source_identity_hash':identity,'full_forecast_qualification':'pending'}))
            print(case+': '+name+' '+('solved' if result is not None else 'FAILED'),flush=True)
        results,failures=infer_treatments(like,on_outcome=checkpoint)
        print(case+': independently computing native3 spectra and derivatives',flush=True)
        native=None;native_result=None;native_start=time.monotonic()
        construction={'method':'build_revision_likelihood','cosmology_basis':'baseline3',
            'source_identity_hash':identity,'case':case,'status':'running',
            'builder_sha256':file_digest(ROOT/'pfsfog/revision_forecast.py')}
        try:
            native=build_revision_likelihood(replace(like.config,cosmology_basis='baseline3'),
                default_survey_group(),FiducialCosmology(),engine)
            native_result=apply_revision_priors(native,native.config.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
            construction['status']='passed'
        except ValueError as error:
            construction.update(status='failed',error=str(error))
        construction['elapsed_seconds']=time.monotonic()-native_start
        provenance={'case':case,'source_identity_hash':identity,'request_sha256':request_digest(request),
            'program_sha256':request['program_sha256'],'physical_root':request['physical_root'],
            'physical_request_sha256':request['physical_request_sha256'],
            'physical_summary_sha256':request['physical_summary_sha256'],'settings':request['settings'],
            'native_construction':construction}
        path=root/'cases'/case
        write_case(path,like,results,failures,native,native_result,growth,provenance)
        verify_physical_binding(path,like)
        checked=verify_case(path,provenance)
        receipt.update(status=checked['status'],verification=checked,provenance=provenance,
                       files=artifact_hashes(path),sigma8_radius_sha256=file_digest(root/'workers'/f'{case}-sigma8-radius.json'))
    except Exception as error:
        receipt.update(status='failed',reason=str(error),error_type=type(error).__name__)
        (root/'workers'/f'{case}-failure.log').write_text(traceback.format_exc())
    try:
        receipt['source_after']=source_identity(request['settings'])['identity_hash']
        if receipt['source_after']!=request['source_identity_hash']:
            receipt.update(status='failed',source_changed=True)
    except Exception as error:
        receipt.update(status='failed',source_changed=True,source_error=str(error))
    receipt['elapsed_seconds']=time.monotonic()-start;write_json(dest,jsonable(receipt))
    print(case+': '+receipt['status'],flush=True)
    return 0 if receipt['status']=='passed' else 1


def verify_inference(root):
    root=Path(root);result={'status':'failed','full_forecast_qualification':'pending','problems':[],'cases':{}}
    try:
        request=read_json(root/'request.json');summary=read_json(root/'summary.json')
        physical=_check_request(request)
        result['problems']=execution_problems(request,summary)
        outcomes={}
        for worker in summary['workers']:
            case=worker['case'];path=root/'cases'/case
            if case not in ('B','C') or case in result['cases']:
                raise ValueError('duplicate or foreign case')
            receipt=read_json(root/'workers'/f'{case}.json')
            if any(receipt.get(k)!=worker.get(k) for k in ('case','status','request_sha256','source_before','source_after')):
                raise ValueError('worker receipt differs from actual execution')
            problems=verify_files(path,worker['files'])
            if problems:raise ValueError(str(problems))
            provenance=worker['provenance']
            if (provenance['source_identity_hash']!=request['source_identity_hash']
                    or provenance['request_sha256']!=request_digest(request)
                    or provenance['program_sha256']!=request['program_sha256']
                    or provenance['settings']!=request['settings']
                    or provenance['physical_root']!=request['physical_root']
                    or provenance['physical_request_sha256']!=request['physical_request_sha256']
                    or provenance['physical_summary_sha256']!=request['physical_summary_sha256']
                    or provenance['native_construction']['builder_sha256']!=file_digest(ROOT/'pfsfog/revision_forecast.py')):
                raise ValueError('inference worker source/input binding differs')
            result['cases'][case]=verify_case(path,provenance)
            verify_growth_checkpoint(root,case,request,worker)
            result['cases'][case]['physical_binding']=verify_physical_binding(
                path,load_physical_likelihood(request['physical_root'],case))
            if result['cases'][case]['status']!='passed':result['problems'].append(case+' required inference control failed')
            radius_path=root/'workers'/f'{case}-sigma8-radius.json'
            if file_digest(radius_path)!=worker['sigma8_radius_sha256']:
                raise ValueError('sigma8 radius record changed')
            reference=physical['source']['details']['reference_cosmology']
            z=np.array([(a+b)/2 for a,b in physical['configurations'][case]['nominal_bins']])
            validate_sigma8_radius(read_json(radius_path),elf_cosmology_spec(reference),reference,z)
            outcomes[case]=_read_outcomes(path)
        if set(outcomes)!={'B','C'}:raise ValueError('missing inference case')
        ratios=constraint_ratios(outcomes)
        if canonical(summary['ratios'])!=canonical(ratios) or canonical(summary['outcomes'])!=canonical(outcomes):
            raise ValueError('saved uncertainties or ratios differ from covariance')
        result.update(ratios=ratios)
        if not result['problems']:result['status']='passed'
    except Exception as error:
        result['problems'].append(str(error))
    return result


def run_inference(args):
    output=args.output.resolve()
    if output.exists():raise FileExistsError('preserve existing inference output')
    physical=args.physical_root.resolve();checked=verify_run(physical)
    if checked['status']!='passed' or not checked['complete_survey']:
        raise ValueError('complete passed physical cohort required')
    original=read_json(physical/'request.json')
    request={'schema':'elf-global-inference-v1','scope':SCOPE,'physical_root':str(physical),
        'physical_request_sha256':file_digest(physical/'request.json'),
        'physical_summary_sha256':file_digest(physical/'summary.json'),
        'source_identity_hash':original['source']['identity_hash'],'settings':original['settings'],
        'program_sha256':file_digest(Path(__file__))}
    output.mkdir(parents=True);(output/'workers').mkdir();write_json(output/'request.json',request)
    start=time.monotonic();summary={'status':'running','scope':SCOPE,'full_forecast_qualification':'pending',
        'source_changed':False,'request_sha256':request_digest(request),'workers':[],
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write_json(output/'summary.json',summary)
    for case in ('B','C'):
        _check_request(request)
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker',case]
        print('Starting '+case+' inference',flush=True)
        with (output/'workers'/f'{case}.log').open('w') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        path=output/'workers'/f'{case}.json'
        receipt=read_json(path) if path.exists() else {'case':case,'status':'failed','reason':'missing worker receipt'}
        summary['workers'].append({**receipt,'exit_code':process.returncode,'command':command})
        summary.update(elapsed_seconds=time.monotonic()-start,source_changed=bool(receipt.get('source_changed',False)))
        write_json(output/'summary.json',summary)
        print(case+' inference '+receipt['status']+'; actual exit '+str(process.returncode),flush=True)
        if summary['source_changed']:break
    summary['status']='passed' if len(summary['workers'])==2 and all(w['status']=='passed' and w['exit_code']==0 for w in summary['workers']) and not summary['source_changed'] else 'failed'
    try:
        summary['outcomes']={c:_read_outcomes(output/'cases'/c) for c in ('B','C')}
        summary['ratios']=constraint_ratios(summary['outcomes'])
    except Exception as error:
        summary.update(status='failed',ratio_error=str(error))
    summary['elapsed_seconds']=time.monotonic()-start;write_json(output/'summary.json',summary)
    verified=verify_inference(output);write_json(output/'verification.json',verified)
    if verified['status']!='passed':summary['status']='failed';write_json(output/'summary.json',summary)
    print(summary['status']+'; full marginalized convergence remains pending',flush=True)
    return 0 if summary['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',type=Path);parser.add_argument('--worker',choices=('B','C'))
    args=parser.parse_args()
    if args.verify:
        import json
        result=verify_inference(args.verify);print(json.dumps(result,indent=2))
        return 0 if result['status']=='passed' else 1
    if not args.output:parser.error('--output is required')
    if args.worker:return run_worker(args.output.resolve(),args.worker)
    if not args.physical_root:parser.error('--physical-root is required')
    return run_inference(args)


if __name__=='__main__':
    raise SystemExit(main())
