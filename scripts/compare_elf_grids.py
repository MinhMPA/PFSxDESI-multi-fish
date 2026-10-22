#!/usr/bin/env python
"""Authenticate and compare two complete ELF B6/C6 numerical resolutions."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ps_1loop_jax'));sys.path.insert(0,str(ROOT))
import numpy as np
from pfsfog.elf_survey_evidence import artifact_hashes,file_digest
from pfsfog.extended_artifacts import read_json,canonical
from scripts.infer_elf_survey import verify_inference
from scripts.elf_grid_comparison import compare_local,compare_constraints,validate_pair_contract

SCHEMA='elf-grid-comparison-v1'
PROGRAMS=('scripts/elf_grid_comparison.py','scripts/compare_elf_grids.py','scripts/infer_elf_survey.py')


def _programs():
    return {name:file_digest(ROOT/name) for name in PROGRAMS}


def _inventory(roots):
    result={}
    for root in roots:
        for name,digest in artifact_hashes(root).items():result[str(root/name)]=digest
    return result


def _fixed_case(coarse,fine,cs,fs):
    """Priors, fiducials, identities and support are never quadrature choices."""
    cm,fm=[read_json(root/'manifest.json') for root in (coarse,fine)]
    fixed=[]
    for manifest,s in [(cm,cs),(fm,fs)]:
        record={}
        for prefix in ('likelihood','native'):
            original=manifest[prefix]
            row={k:deepcopy(original[k]) for k in ('configuration','names','parameters','reference_cosmology','model')}
            row['cells']=[{k:cell[k] for k in ('names','metadata')} for cell in original['cells']]
            if (row['model']['fft_n']!=s['nfft'] or row['model']['mean_n_mu']!=s['n_mu']
                    or row['model']['covariance_n_mu']!=s['n_mu']
                    or row['configuration']['dk']!=s['dk'] or row['configuration']['n_mu']!=s['n_mu']):
                raise ValueError('fixed case has inconsistent numerical settings')
            for key in ('dk','n_mu'):row['configuration'].pop(key)
            for key in ('fft_n','mean_n_mu','covariance_n_mu'):row['model'].pop(key)
            record[prefix]=row
        fixed.append(record)
    if canonical(fixed[0])!=canonical(fixed[1]):raise ValueError('fixed inference metadata changed')
    with np.load(coarse/'arrays.npz',allow_pickle=False) as a,np.load(fine/'arrays.npz',allow_pickle=False) as b:
        fixed_names={f'{prefix}/fiducial' for prefix in ('likelihood','native')}
        # All priors remain represented, including failed treatment precision inputs.
        is_fixed=lambda name:name.endswith(('/F_prior_cosmo','/F_prior_nuisance','/growth_z','/growth_fiducial'))
        fixed_names.update(k for k in a if is_fixed(k))
        if {k for k in a if is_fixed(k)}!={k for k in b if is_fixed(k)}:
            # Optional treatment status changes are reported globally. Common records
            # must match; completed growth arrays can be absent after a failed solve.
            prior=lambda z:{k for k in z if k.endswith(('/F_prior_cosmo','/F_prior_nuisance'))}
            if prior(a)!=prior(b):raise ValueError('fixed prior coverage changed')
            fixed_names={k for k in fixed_names if k in a and k in b}
        for key in fixed_names:
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):
                raise ValueError('fixed inference fiducial/prior/growth coordinate changed: '+key)


def compare_runs(coarse,fine,axis):
    coarse,fine=Path(coarse).resolve(),Path(fine).resolve()
    programs=_programs();physical=[];requests=[];summaries=[];checked={}
    # Locate the physical inputs from captured request bytes, then freeze the
    # entire evidence inventory BEFORE either expensive authentication begins.
    request_hashes={}
    try:
        for root in (coarse,fine):
            path=root/'request.json';raw=path.read_bytes()
            request_hashes[str(path)]=hashlib.sha256(raw).hexdigest()
            physical.append(Path(json.loads(raw)['physical_root']).resolve())
        roots=[coarse,fine,*physical];files=_inventory(roots)
        if any(files.get(path)!=digest for path,digest in request_hashes.items()):
            raise ValueError('request changed before authentication')
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise ValueError('input authentication prerequisites failed: '+str(error)) from error
    for label,root in [('coarse',coarse),('fine',fine)]:
        verification=verify_inference(root)
        if (verification.get('status')!='passed' or set(verification.get('cases',{}))!={'B','C'}
                or any(v.get('status')!='passed' for v in verification['cases'].values())):
            raise ValueError('input authentication failed: '+label+' '+str(verification))
        checked[label]={'status':'passed','cases':['B','C']}
    if files!=_inventory(roots) or programs!=_programs():
        raise ValueError('comparison inputs/programs changed during authentication')
    for root,p in zip((coarse,fine),physical):
        requests.append(read_json(p/'request.json'));summaries.append(read_json(root/'summary.json'))
    contract=validate_pair_contract(*requests,axis)
    expected=requests[0]['expected']
    if (requests[0]['scope']!='full_B6_C6' or len(expected)!=40
            or len({row['key'] for row in expected})!=40
            or any(sum(row['case']==c for row in expected)!=20 for c in ('B','C'))):
        raise ValueError('comparison requires the complete forty-cell survey')
    for case in ('B','C'):_fixed_case(coarse/'cases'/case,fine/'cases'/case,*[r['settings'] for r in requests])
    local={}
    shell='dk' in contract['changed_settings']
    for row in expected:
        loaded=[];fixed=[]
        for root in physical:
            path=root/'cells'/row['key'].replace(':','__');inputs=read_json(path/'inputs.json')
            fixed.append({k:inputs[k] for k in ('metadata','fiducials')})
            with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
                loaded.append({**{k:arrays[k] for k in ('k','dk','mean','D','C')},'names':row['support']['local_names']})
        if canonical(fixed[0])!=canonical(fixed[1]):raise ValueError('fixed physical cell content changed: '+row['key'])
        local[row['key']]=compare_local(*loaded,shell=shell)
    global_result=compare_constraints(*summaries)
    if files!=_inventory(roots) or programs!=_programs():raise ValueError('comparison inputs/programs changed during measurement')
    passed=global_result['status']=='passed' and all(v['status']=='passed' for v in local.values())
    return {'schema':SCHEMA,'scope':'paired_grid_comparison','coarse_root':str(coarse),'fine_root':str(fine),
        'axis':axis,'contract':contract,'authentication':checked,'input_files':files,
        'comparison_programs':programs,'local':local,'global':global_result,
        'status':'passed' if passed else 'failed','full_forecast_qualification':'pending'}


def verify_report(path):
    result={'status':'failed','full_forecast_qualification':'pending','problems':[]}
    try:
        saved=read_json(path)
        if saved.get('schema')!=SCHEMA:raise ValueError('unsupported comparison report')
        actual=compare_runs(saved['coarse_root'],saved['fine_root'],saved['axis'])
        if canonical(saved)!=canonical(actual):raise ValueError('saved report differs from fresh evidence/arithmetic')
        result.update(status='passed',comparison_status=actual['status'])
    except Exception as error:result['problems'].append(str(error))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse',type=Path);parser.add_argument('--fine',type=Path)
    parser.add_argument('--axis',choices=('loop','angular','shell','combined'))
    parser.add_argument('--output',type=Path);parser.add_argument('--verify',type=Path)
    args=parser.parse_args(argv)
    if args.verify:
        if any((args.coarse,args.fine,args.axis,args.output)):parser.error('--verify cannot be combined with measurement')
        result=verify_report(args.verify);print(json.dumps(result,indent=2))
        return 0 if result['status']=='passed' and result.get('comparison_status')=='passed' else 1
    if not all((args.coarse,args.fine,args.axis,args.output)):parser.error('measurement needs both cohorts, axis and output')
    # Reserve before reading inputs; no existing result can be overwritten.
    with args.output.open('x') as stream:
        try:result=compare_runs(args.coarse,args.fine,args.axis)
        except Exception as error:
            result={'schema':SCHEMA,'status':'failed','scope':'paired_grid_comparison',
                'coarse_root':str(args.coarse.resolve()),'fine_root':str(args.fine.resolve()),'axis':args.axis,
                'error_type':type(error).__name__,'reason':str(error),'full_forecast_qualification':'pending'}
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'output':str(args.output),'full_forecast_qualification':'pending'}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
