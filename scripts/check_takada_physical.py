#!/usr/bin/env python
"""Recheck every affected physical source at the corrected density and grid."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
import argparse
from copy import deepcopy
import hashlib
import json
import subprocess
import time
import numpy as np
from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_survey_evidence import file_digest,write_json,verify_files,artifact_hashes,cell_directory
from pfsfog.extended_artifacts import read_json,canonical,jsonable
from pfsfog.one_loop import make_one_loop_engine
from scripts.run_takada_revision import request_and_binding,corrected_source,GRID_PROFILES,REFINEMENTS
from scripts.run_takada_density_scan import reconstruct_reference_source
from scripts.run_takada_bias import BROAD,corrected_bias_window
from scripts.elf_study_evidence import load_source_cell,study_source
from scripts.elf_grid_comparison import compare_local
from scripts.pfs_density_response import rescale_density_source
from scripts.run_pfs_density_scan import FACTORS,InputBinding
from scripts.takada_physical import direct_query,change_query_density


def signature(record):
    original=read_json(Path(record['source_directory'])/'inputs.json')
    return hashlib.sha256(canonical({'metadata':record['source_target'],'fiducials':original['fiducials']})).hexdigest()


def records(baseline,grid,kind):
    request,binding=request_and_binding(baseline)
    path=baseline/'inference'/grid;s=read_json(path/'summary.json')
    if s['status']!='passed':raise ValueError('incomplete corrected physical reference')
    errors=verify_files(path/'projections',s['projection_files'])
    if errors:raise ValueError('; '.join(errors))
    output={}
    for name in ('C24','C22','C','D'):
        manifest=read_json(path/'profiles'/name/'manifest.json')
        for refid in manifest['projections']:
            p=path/'projections'/(refid+'.json');r=read_json(p)
            if r['source_target'] is None or 'PFS-ELG' not in r['source_target']['active_populations']:continue
            key=signature(r)
            if key not in output:output[key]={'kind':'main','record':r,'reference_file':str(p),'reference_sha256':file_digest(p)}
    if kind=='main':return output
    if kind!='bias':raise ValueError('unknown physical comparison')
    output={};checkpoint=ROOT/'docs/revision/evidence/pfs-bias-broad-2026-10-10/BROAD_CHECKPOINT.json'
    frozen=read_json(checkpoint)['files'];auth=InputBinding({str(ROOT/p):v['sha256'] if isinstance(v,dict) else v for p,v in frozen.items()})
    campaign=read_json(auth.check(BROAD/'request.json'));physical=BROAD/'stochastic-validation/physical'/grid
    cohort=read_json(auth.check(physical/'summary.json'))
    m=read_json(path/'profiles/C24/manifest.json')
    base_refs={c['metadata']['cell_id']:read_json(path/'projections'/(r+'.json'))
               for c,r in zip(m['likelihood']['cells'],m['projections'])}
    for item,worker in zip(campaign['physical_requests'][grid]['expected'],cohort['workers']):
        if item['key']!=worker['key']:raise ValueError('bias population order changed')
        window=corrected_bias_window(item['experiment']['window'],request['pfs_zmin'])
        directory=cell_directory(physical,item['key'])
        auth.check(directory/'inputs.json')
        meta=read_json(directory/'inputs.json')['metadata']
        for cell in m['likelihood']['cells']:
            row=cell['metadata']
            if row['region']!='overlap' or not window[0]<=row['radial_bin'][0]<row['radial_bin'][1]<=window[1]:continue
            target=deepcopy(meta)
            for field in ('cell_id','nominal_bin_index','nominal_bin','radial_bin','model_redshift','volume','region'):
                target[field]=deepcopy(row[field])
            for n,v in row['nbar'].items():target['nbar'][n]=v
            r={'source_directory':str(directory),'source_record':worker,'source_target':target,'expected':row,
               'unchanged_observation_anchor':base_refs[row['cell_id']].get('unchanged_observation_anchor')}
            key=item['key']+'--'+row['cell_id']
            output[key]={'kind':'bias','record':r,'reference_file':str(checkpoint),'reference_sha256':file_digest(checkpoint)}
    return output


def source(entry):
    p=Path(entry['reference_file'])
    if file_digest(p)!=entry['reference_sha256']:raise ValueError('physical reference changed')
    r=entry['record'];raw=load_source_cell(r['source_directory'],r['source_record'])
    return reconstruct_reference_source(raw,r)


def local(s):
    a=s['arrays']
    return {**{k:a[k] for k in ('k','dk','mean','D','C')},'names':s['inputs']['metadata']['local_names']}


def reconstruction(data,q):
    ix=np.searchsorted(q['k'],data['k'])
    if np.any(ix>=len(q['k'])) or not np.array_equal(q['k'][ix],data['k']):raise ValueError('common-k coverage differs')
    rebuilt={**data,'mean':q['mean'][ix],'D':q['D'][ix],
             'C':q['covariance_kernel'][ix]/data['dk'][:,None,None]}
    return compare_local(data,rebuilt)


def worker(baseline,kind,key):
    baseline=Path(baseline).resolve();request,binding=request_and_binding(baseline)
    dest=baseline/'physical-comparisons'/kind/key;dest.mkdir(parents=True,exist_ok=False)
    own={str(ROOT/p):file_digest(ROOT/p) for p in ('scripts/check_takada_physical.py','scripts/takada_physical.py')}
    entries={g:records(baseline,g,kind)[key] for g in GRID_PROFILES}
    sources={g:source(e) for g,e in entries.items()}
    reference=sources['shell304']['inputs']
    for s in sources.values():
        if (canonical(s['inputs']['metadata'])!=canonical(reference['metadata'])
                or s['inputs']['fiducials']!=reference['fiducials']):raise ValueError('physical population changes with resolution')
    engine=make_one_loop_engine(nfft=2048)
    if canonical(study_source(GRID_PROFILES['shell304'],engine))!=canonical(request['original_source']['shell304']):
        raise ValueError('common-k theory source changed')
    query=np.union1d(sources['angular64']['arrays']['k'],sources['shell304']['arrays']['k'])
    direct=direct_query(reference,query,engine,FiducialCosmology())
    np.savez_compressed(dest/'query.npz',**direct)
    rows=[];factors=FACTORS if kind=='main' else {'fiducial':1.}
    for label,factor in factors.items():
        changed={g:s if factor==1. else rescale_density_source(s,factor) for g,s in sources.items()}
        q=direct if factor==1. else change_query_density(direct,reference,factor)
        for coarse,fine,axis in REFINEMENTS:
            a=local(changed[coarse]);b=local(changed[fine]);r={'factor':factor,'factor_label':label,
                'coarse':coarse,'fine':fine,'axis':axis}
            if axis=='shell':
                checks=[reconstruction(d,q) for d in (a,b)]
                r.update(status='passed' if all(x['status']=='passed' for x in checks) else 'failed',
                         method='direct_common_k',measurements=checks)
            else:
                check=compare_local(a,b);r.update(status=check['status'],method='identical_k',measurements=[check])
            rows.append(r)
    bindings={}
    for s in sources.values():bindings.update(s['source_files'])
    bindings.update({e['reference_file']:e['reference_sha256'] for e in entries.values()});bindings.update(own)
    for p,h in bindings.items():
        if file_digest(p)!=h:raise ValueError('physical comparison input changed')
    if canonical(study_source(GRID_PROFILES['shell304'],engine))!=canonical(request['original_source']['shell304']):
        raise ValueError('common-k theory source changed after evaluation')
    request_and_binding(baseline)
    report={'status':'passed' if all(r['status']=='passed' for r in rows) else 'failed','kind':kind,'key':key,
            'request_sha256':file_digest(baseline/'request.json'),'input_files':bindings,'metadata':reference['metadata'],
            'fiducials':reference['fiducials'],'comparisons':rows,'query_sha256':file_digest(dest/'query.npz')}
    write_json(dest/'report.json',jsonable(report))
    return {'status':report['status'],'key':key,'comparisons':len(rows)}


def run(baseline,kind):
    baseline=Path(baseline).resolve();inventories={g:records(baseline,g,kind) for g in GRID_PROFILES}
    keys=list(inventories['shell304'])
    if any(set(v)!=set(keys) for v in inventories.values()):raise ValueError('different physical source inventories')
    dest=baseline/'physical-comparisons'/kind;dest.mkdir(parents=True,exist_ok=False)
    execution={'status':'running','kind':kind,'keys':keys,'workers':[]}
    write_json(dest/'execution.json',execution)
    for key in keys:
        started=time.monotonic();command=[sys.executable,str(Path(__file__).resolve()),'--baseline',str(baseline),
            '--kind',kind,'--worker',key]
        with (dest/(key+'.log')).open('w') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        execution['workers'].append({'key':key,'command':command,'actual_exit':result.returncode,
            'elapsed_seconds':time.monotonic()-started,'report_sha256':file_digest(dest/key/'report.json') if (dest/key/'report.json').exists() else None})
        write_json(dest/'execution.json',execution);print(kind+' physical '+key+': '+str(result.returncode),flush=True)
        if result.returncode:
            execution['status']='failed';write_json(dest/'execution.json',execution)
            raise ValueError('physical comparison failed: '+key)
    execution['status']='passed';write_json(dest/'execution.json',execution);return execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--kind',choices=('main','bias'),required=True);p.add_argument('--worker')
    a=p.parse_args();r=worker(a.baseline,a.kind,a.worker) if a.worker else run(a.baseline,a.kind)
    print(json.dumps({k:v for k,v in r.items() if k not in ('workers','keys')},indent=2))
    return 0 if r['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
