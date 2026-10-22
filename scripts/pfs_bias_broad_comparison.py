"""Direct physical and global convergence checks for the wider PFS bias range."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time

import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_survey_evidence import canonical, cell_directory, file_digest, verify_files, write_json
from pfsfog.extended_artifacts import read_json, jsonable
from pfsfog.one_loop import make_one_loop_engine
from scripts.elf_grid_comparison import compare_local
from scripts.elf_study_evidence import load_source_cell
from scripts.elf_study_physical import common_query_data
from scripts.pfs_bias_stochastic_evidence import verify_corrected_cell
from scripts.run_pfs_bias_stochastic_validation import (
    physical_directory, validation_root, original_records,
    campaign_sources as validation_campaign_sources)
from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.pfs_bias_broad_inference import require_grid_exit, verify_inference
from scripts.run_pfs_bias_broad import check_campaign, GRIDS, REFINEMENTS
from scripts.run_elf_studies import check_bindings


def comparison_sources():
    files=('scripts/pfs_bias_broad_comparison.py','scripts/elf_grid_comparison.py',
           'scripts/elf_study_physical.py','scripts/pfs_bias_broad_inference.py')
    return {**validation_campaign_sources(),**{str(ROOT/p):file_digest(ROOT/p) for p in files}}


def check_population_pair(coarse,fine):
    fields=('key','support','experiment','fiducials')
    if canonical({k:coarse[k] for k in fields})!=canonical({k:fine[k] for k in fields}):
        raise ValueError('refinement changes physical population or survey support')


def query_reconstruction(data,direct):
    query=np.asarray(direct['k']);k=np.asarray(data['k'])
    if query.ndim!=1 or not len(query) or np.any(np.diff(query)<=0):
        raise ValueError('invalid common-k coverage')
    indices=np.searchsorted(query,k)
    if np.any(indices>=len(query)) or not np.array_equal(query[indices],k):
        raise ValueError('common-k query does not cover saved wavenumbers')
    rebuilt={**data,'mean':np.asarray(direct['mean'])[indices],
        'D':np.asarray(direct['D'])[indices],
        'C':np.asarray(direct['covariance_kernel'])[indices]/np.asarray(data['dk'])[:,None,None]}
    return compare_local(data,rebuilt)


def project_local(data,mapping):
    rows=np.asarray(mapping['source_rows'],dtype=int);columns=np.asarray(mapping['source_columns'],dtype=int)
    if (mapping['independent_autos'] is not False
            or not np.array_equal(mapping['source_shells'],np.arange(len(data['k'])))
            or len(set(rows))!=len(rows) or len(set(columns))!=len(columns)
            or not len(rows) or not len(columns) or min(rows)<0 or min(columns)<0
            or max(rows)>=data['mean'].shape[1] or max(columns)>=len(data['names'])):
        raise ValueError('invalid broad profile projection')
    return {**data,'names':[data['names'][i] for i in columns],
            'mean':data['mean'][:,rows],'D':data['D'][:,rows][:,:,columns],
            'C':data['C'][:,rows][:,:,rows]}


def project_query(direct,mapping):
    rows=mapping['source_rows'];cols=mapping['source_columns']
    return {**direct,'mean':direct['mean'][:,rows],'D':direct['D'][:,rows][:,:,cols],
            'covariance_kernel':direct['covariance_kernel'][:,rows][:,:,rows]}


def _errors(out,kind):
    suffix='' if kind=='marginalized' else '_nuisance_fixed'
    result={}
    for field,key,labels in [('cosmology','cosmological_covariance',['ln10_10_As','Mnu','Omegam']),
                             ('growth','growth_covariance',out['growth_z'])]:
        matrix=np.asarray(out[key+suffix]);diagonal=np.diag(matrix)
        if matrix.shape!=(len(labels),len(labels)) or not np.isfinite(matrix).all() or np.any(diagonal<=0):
            raise ValueError('invalid constraint covariance')
        result.update({field+'/'+str(label):float(value) for label,value in zip(labels,np.sqrt(diagonal))})
    return result


def _constraint_values(point,reference=None):
    if point['status']!='passed':raise ValueError('failed population in convergence comparison')
    window=point['experiment']['window']
    left,right=('D','C') if window[1]<=1.6 else ('H21','C21')
    if set(point['profiles'])!={left,right,'C22'}:raise ValueError('incomplete comparison profiles')
    values={}
    for kind in ('marginalized','nuisance_fixed'):
        errors={n:_errors(out,kind) for n,out in point['profiles'].items()}
        for name,row in errors.items():values.update({name+'/'+kind+'/'+k:v for k,v in row.items()})
        for a,b in ((left,right),(right,left)):
            if set(errors[a])!=set(errors[b]):raise ValueError('unmatched growth coordinates')
            values.update({'ratio/'+a+'_over_'+b+'/'+kind+'/'+k:errors[a][k]/errors[b][k] for k in errors[a]})
        if reference is not None:
            original=_errors(reference['profiles']['C22'],kind)
            if set(original)!=set(errors['C22']):raise ValueError('unmatched original population')
            values.update({'ratio/C22_over_original/'+kind+'/'+k:errors['C22'][k]/original[k] for k in original})
    return values


def compare_profile_constraints(coarse,fine,reference_coarse=None,reference_fine=None):
    if (coarse['key']!=fine['key'] or canonical(coarse['experiment'])!=canonical(fine['experiment'])
            or (reference_coarse is None)!=(reference_fine is None)):
        raise ValueError('unmatched populations in cosmological comparison')
    a=_constraint_values(coarse,reference_coarse);b=_constraint_values(fine,reference_fine)
    if set(a)!=set(b):raise ValueError('cosmological comparison inventory differs')
    metrics=[]
    for name in sorted(a):
        change=abs(b[name]/a[name]-1);guard=8*np.finfo(float).eps*max(1.,abs(b[name]/a[name]))
        metrics.append({'label':name,'coarse':a[name],'fine':b[name],'relative_change':change,
                        'roundoff_guard':guard,'status':'passed' if change+guard<.01 else 'failed'})
    return {'status':'passed' if all(r['status']=='passed' for r in metrics) else 'failed',
            'tolerance':.01,'metrics':metrics,'maximum_relative_change':max(r['relative_change'] for r in metrics)}


def validate_common_execution(execution,keys):
    rows=execution['workers']
    if (execution['status']!='passed' or len(rows)!=len(keys)
            or {r['key'] for r in rows}!=set(keys) or any(r['actual_exit']!=0 for r in rows)):
        raise ValueError('incomplete or unsuccessful common-k execution')


def _source(root,grid,key,campaign):
    require_grid_exit(root,'validated_physical',grid)
    base=physical_directory(root,grid);request=campaign['physical_requests'][grid]
    summary=read_json(base/'summary.json')
    row=next(r for r in summary['workers'] if r['key']==key)
    expected=next(r for r in request['expected'] if r['key']==key)
    if row['status']!='passed' or row['exit_code']!=0:raise ValueError('failed physical source')
    path=cell_directory(base,key);problems=verify_files(path,row['files'])
    if problems:raise ValueError('; '.join(problems))
    _,original=original_records(root,grid)
    raw=next(r for r in original['workers'] if r['key']==key)
    verify_corrected_cell(path,expected,request,raw)
    return load_source_cell(path,row),expected


def _local(source):
    a=source['arrays']
    return {**{k:a[k] for k in ('k','dk','mean','D','C')},'names':source['inputs']['metadata']['local_names']}


def common_population(root,key):
    root=Path(root).resolve();campaign=check_campaign(root);before=comparison_sources()
    data=[];sources=[];expected=[];bindings={}
    for grid in ('angular64','shell304'):
        require_grid_exit(root,'validated_physical',grid)
        source,e=_source(root,grid,key,campaign)
        sources.append(source);data.append(_local(source));expected.append(e)
        bindings.update(source['source_files'])
        for p in (physical_directory(root,grid)/'summary.json',
                  validation_root(root)/'request.json',validation_root(root)/'execution.json'):
            bindings[str(p)]=file_digest(p)
    check_population_pair(*expected)
    if canonical(sources[0]['inputs']['metadata'])!=canonical(sources[1]['inputs']['metadata']):
        raise ValueError('common-k observation layouts differ')
    output=cell_directory(root/'common-k',key);output.mkdir(parents=True,exist_ok=False)
    engine=make_one_loop_engine(nfft=2048);query=np.union1d(data[0]['k'],data[1]['k'])
    direct=common_query_data(sources[1]['inputs'],query,engine,FiducialCosmology())
    checks=[query_reconstruction(d,direct) for d in data]
    np.savez_compressed(output/'query_arrays.npz',**direct)
    report={'status':'passed' if all(r['status']=='passed' for r in checks) else 'failed',
        'key':key,'experiment':expected[0]['experiment'],'reconstructions':checks,
        'original_interpolation_diagnostic':compare_local(*data,shell=True),
        'input_files':bindings,'sources':before,'query_arrays_sha256':file_digest(output/'query_arrays.npz'),
        'physical_source_identities':[campaign['physical_requests'][g]['source']['identity_hash'] for g in ('angular64','shell304')]}
    check_bindings(bindings);check_campaign(root)
    if before!=comparison_sources():raise ValueError('common-k sources changed')
    write_json(output/'report.json',jsonable(report))
    if report['status']!='passed':raise ValueError('direct common-k reconstruction failed')
    return report


def run_common(root):
    root=Path(root).resolve();campaign=check_campaign(root)
    for g in GRIDS:require_grid_exit(root,'validated_physical',g)
    destination=root/'common-k';destination.mkdir(parents=True,exist_ok=False)
    keys=[r['key'] for r in broad_schedule()]
    summary={'status':'running','workers':[],'max_concurrent':2}
    write_json(destination/'execution.json',summary)
    def one(key):
        command=[sys.executable,str(ROOT/'scripts/run_pfs_bias_broad.py'),'--output',str(root),
                 '--phase','common','--worker',key]
        started=time.monotonic();log=destination/(key+'.log')
        with log.open('x') as stream:process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        return {'key':key,'actual_exit':process.returncode,'command':command,'log':str(log),
                'elapsed_seconds':time.monotonic()-started}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(one,key) for key in keys]
        for future in as_completed(jobs):
            row=future.result();summary['workers'].append(row);write_json(destination/'execution.json',summary)
            print('common-k '+row['key']+f': actual exit {row["actual_exit"]}',flush=True)
    summary['status']='passed' if all(r['actual_exit']==0 for r in summary['workers']) else 'failed'
    write_json(destination/'execution.json',summary);validate_common_execution(summary,keys)
    return summary


def compare_campaign(root,output):
    root=Path(root).resolve();output=Path(output);campaign=check_campaign(root)
    before=comparison_sources();verified={g:verify_inference(root,g) for g in GRIDS}
    keys=[r['key'] for r in broad_schedule()]
    common_execution=read_json(root/'common-k/execution.json');validate_common_execution(common_execution,keys)
    inputs={str(root/'common-k/execution.json'):file_digest(root/'common-k/execution.json')}
    for value in verified.values():inputs.update(value['inputs'])
    output.mkdir(parents=True,exist_ok=False);reports=[]
    original={tuple(r['window']):r['key'] for r in broad_schedule() if r['ratio']==1.}
    for coarse,fine,axis in REFINEMENTS:
        result={'coarse':coarse,'fine':fine,'axis':axis,'populations':[]}
        for key in keys:
            a,ea=_source(root,coarse,key,campaign);b,eb=_source(root,fine,key,campaign)
            check_population_pair(ea,eb);da,db=_local(a),_local(b)
            inputs.update(a['source_files']);inputs.update(b['source_files'])
            local=[];direct=None
            if axis=='shell':
                directory=cell_directory(root/'common-k',key);record=read_json(directory/'report.json')
                if (record['status']!='passed' or record['key']!=key or record['sources']!=before
                        or canonical(record['experiment'])!=canonical(ea['experiment'])
                        or record['physical_source_identities']!=[campaign['physical_requests'][g]['source']['identity_hash'] for g in ('angular64','shell304')]
                        or record['query_arrays_sha256']!=file_digest(directory/'query_arrays.npz')):
                    raise ValueError('missing, changed or unrelated common-k evidence')
                check_bindings(record['input_files']);inputs.update(record['input_files'])
                for name in ('report.json','query_arrays.npz'):inputs[str(directory/name)]=file_digest(directory/name)
                with np.load(directory/'query_arrays.npz',allow_pickle=False) as saved:
                    direct={name:saved[name].copy() for name in saved.files}
                local.extend([{'profile':'full_six_coordinate','grid':g,'measurement':query_reconstruction(d,direct)}
                              for g,d in ((coarse,da),(fine,db))])
            else:local.append({'profile':'full_six_coordinate','measurement':compare_local(da,db)})
            pa=verified[coarse]['points'][key];pb=verified[fine]['points'][key]
            for profile in pa['profiles']:
                maps=[read_json(root/'inference'/g/key/(profile+'-physical-check.json'))['mapping'] for g in (coarse,fine)]
                projected=[project_local(data,mapping) for data,mapping in zip((da,db),maps)]
                if axis=='shell':
                    for g,data,mapping in zip((coarse,fine),projected,maps):
                        local.append({'profile':profile,'grid':g,
                                      'measurement':query_reconstruction(data,project_query(direct,mapping))})
                else:local.append({'profile':profile,'measurement':compare_local(*projected)})
            ref=original[tuple(ea['experiment']['window'])]
            global_check=compare_profile_constraints(pa,pb,verified[coarse]['points'][ref],verified[fine]['points'][ref])
            passed=global_check['status']=='passed' and all(r['measurement']['status']=='passed' for r in local)
            result['populations'].append({'key':key,'status':'passed' if passed else 'failed',
                                          'global':global_check,'local':local})
        result['status']='passed' if all(r['status']=='passed' for r in result['populations']) else 'failed'
        path=output/(coarse+'-to-'+fine+'.json');write_json(path,jsonable(result))
        reports.append(result);print(coarse+' -> '+fine+': '+result['status'],flush=True)
    check_bindings(inputs);check_campaign(root)
    if before!=comparison_sources():raise ValueError('comparison sources changed')
    passed=all(r['status']=='passed' for r in reports)
    summary={'status':'passed' if passed else 'failed','full_bias_grid_qualification':passed,
             'sources':before,'input_files':inputs,'refinements':REFINEMENTS,
             'reports':{p.name:file_digest(p) for p in output.glob('*-to-*.json')}}
    write_json(output/'summary.json',jsonable(summary))
    if not passed:raise ValueError('broad physical or cosmological convergence failed')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    compare_campaign(args.root,args.output)
