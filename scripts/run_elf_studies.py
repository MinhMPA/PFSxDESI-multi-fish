#!/usr/bin/env python
"""Complete the accepted13-paper/five-high-z ELF studies with retained evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
import argparse
from dataclasses import asdict,replace
import json
import subprocess
import time
import traceback
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from pfsfog.cosmo import FiducialCosmology,FIDUCIAL
from pfsfog.elf_survey_evidence import (artifact_hashes,cell_directory,file_digest,request_digest,
    source_identity,verify_files,verify_run,write_json)
from pfsfog.extended_artifacts import canonical,read_json,jsonable,require_close,PRECISION_FIELDS,GROWTH_FIELDS
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_forecast import make_shell_grid
from pfsfog.revision_validation import compare_nested_likelihoods
from scripts.elf_grid_comparison import compare_local
from scripts.elf_study_profiles import (study_cells,study_config,study_shell_grid,likelihood_key,paper_profiles,highz_profiles)
from scripts.elf_study_data import project_cell,select_cell,independent_autos
from scripts.elf_study_inference import assemble_study,infer_study
from scripts.elf_study_evidence import (study_source,make_study_request,validate_study_ledger,
    verify_highz,verify_highz_cell,load_source_cell)
from scripts.elf_study_physical import measure_highz_cell,common_query_data
from scripts.elf_study_report import (all_profiles,comparisons,constraint_row,compare_results,
    save_result,load_result,verify_result,nesting_snapshot,write_tables,check_projection_binding,
    verify_comparison_record)

GRID_PROFILES={
    'loop2048':{'nfft':2048,'n_mu':32,'dk':.00125},
    'loop4096':{'nfft':4096,'n_mu':32,'dk':.00125},
    'angular64':{'nfft':2048,'n_mu':64,'dk':.00125},
    'shell304':{'nfft':2048,'n_mu':64,'dk':.000625},
    'final-loop':{'nfft':4096,'n_mu':64,'dk':.000625}}
REFINEMENTS=[('loop2048','loop4096','loop'),('loop2048','angular64','angular'),
             ('angular64','shell304','shell'),('shell304','final-loop','loop')]
OLD=ROOT/'docs/revision/evidence/elf-grid-convergence-2026-10-09'


def fresh_directory(path):Path(path).mkdir(parents=True,exist_ok=False)


def old_physical(grid):return ROOT/f'results/elf-grid-20261009-{grid}-physical-01'


def old_inference(grid):return ROOT/f'results/elf-grid-20261009-{grid}-inference-01'


def source_for_cell(profile,cell_id):
    ibin=int(cell_id[3:5])
    if ibin==8 and cell_id.endswith('-nonoverlap'):return None
    if profile.pfs_zmax>1.6 and ibin>=6 and cell_id.endswith('-overlap'):return 'C22'
    return 'B' if profile.case in ('I','A','B') else 'C'


def check_bindings(bindings):
    if not bindings:raise ValueError('empty source/file binding inventory')
    for path,digest in bindings.items():
        if file_digest(path)!=digest:raise ValueError('changed input/evidence: '+path)


def prepare(root):
    fresh_directory(root)
    checkpoint=read_json(OLD/'GRID_CHECKPOINT.json')
    if checkpoint['status']!='passed_under_documented_criteria':raise ValueError('prior grid checkpoint is incomplete')
    check_bindings(checkpoint['checkpoint_hashes'])
    request={'schema':'elf-survey-studies-v1','profiles':{n:asdict(p) for n,p in all_profiles().items()},
        'grids':GRID_PROFILES,'refinements':REFINEMENTS,'sources':{},
        'prior_checkpoint':str(OLD/'GRID_CHECKPOINT.json'),
        'prior_checkpoint_sha256':file_digest(OLD/'GRID_CHECKPOINT.json')}
    for grid,s in GRID_PROFILES.items():
        source=study_source(s);old=read_json(old_physical(grid)/'request.json')
        if canonical(source['details']['original'])!=canonical(old['source']):
            raise ValueError('original physical source changed: '+grid)
        request['sources'][grid]=source
    write_json(root/'request.json',jsonable(request))
    write_json(root/'state.json',{'status':'prepared','request_sha256':request_digest(jsonable(request)),
        'full_forecast_qualification':'pending','active_jobs':[]})
    return {'status':'prepared','root':str(root)}


def check_campaign(root,grid=None):
    request=read_json(root/'request.json')
    if (request['schema']!='elf-survey-studies-v1' or request['grids']!=GRID_PROFILES
            or canonical(request['profiles'])!=canonical({n:asdict(p) for n,p in all_profiles().items()})
            or canonical(request['refinements'])!=canonical(REFINEMENTS)
            or file_digest(request['prior_checkpoint'])!=request['prior_checkpoint_sha256']):
        raise ValueError('campaign specification or prior evidence changed')
    for g in ([grid] if grid else GRID_PROFILES):
        if canonical(study_source(GRID_PROFILES[g]))!=canonical(request['sources'][g]):
            raise ValueError('campaign calculation sources changed: '+g)
    return request


def run_physical(root,grid):
    request=check_campaign(root,grid);dest=root/'physical'/grid;fresh_directory(dest)
    r=make_study_request(GRID_PROFILES[grid],source=request['sources'][grid]);write_json(dest/'request.json',r)
    summary={'status':'running','scope':r['scope'],'workers':[],'complete_new_cells':False,
        'source_changed':False,'source_identity_hash':r['source']['identity_hash'],
        'request_sha256':request_digest(r),'full_forecast_qualification':'pending'}
    write_json(dest/'summary.json',summary)
    for expected in r['expected']:
        check_campaign(root,grid);key=expected['key'];p=cell_directory(dest,key);fresh_directory(p)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--phase','physical','--grid',grid,'--worker',key]
        print('Starting physical '+grid+' '+key,flush=True);start=time.monotonic()
        with (p/'worker.log').open('w') as f:process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        receipt=read_json(p/'worker.json')
        row={'key':key,'exit_code':process.returncode,'status':'failed','command':cmd,
            'elapsed_seconds':time.monotonic()-start,'request_sha256':receipt['request_sha256'],
            'source_identity_hash':receipt['source_after']}
        try:
            row['verification']=verify_highz_cell(p,expected,r)
            if process.returncode==0 and receipt['status']=='passed':row['status']='passed'
        except Exception as error:row['error']=str(error)
        row['files']=artifact_hashes(p);summary['workers'].append(row)
        write_json(dest/'summary.json',summary)
        print('Finished '+key+': '+row['status']+f'; actual exit {process.returncode}',flush=True)
    check_campaign(root,grid)
    passed=all(x['status']=='passed' for x in summary['workers']) and len(summary['workers'])==3
    summary.update(status='passed' if passed else 'failed',complete_new_cells=passed)
    write_json(dest/'summary.json',summary)
    if not passed:raise ValueError('high-redshift physical cohort failed: '+grid)
    return verify_highz(dest)


def native_bridge(result,grid,case):
    p=old_inference(grid)/'cases'/case
    with np.load(p/'arrays.npz',allow_pickle=False) as a:
        require_close(result.F_like,a['native/F_like'],'native3 likelihood bridge',rtol=1e-8,atol=1e-12)
        for key in (*PRECISION_FIELDS,*GROWTH_FIELDS):
            require_close(getattr(result,key),a['native_result/'+key],'native3 '+key,rtol=1e-8,atol=1e-12)
    return {'status':'passed','case':case,'independent_control':str(p),
            'arrays_sha256':file_digest(p/'arrays.npz'),'rtol':1e-8,'atol':1e-12}


def run_inference(root,grid):
    campaign=check_campaign(root,grid);settings=GRID_PROFILES[grid];dest=root/'inference'/grid;fresh_directory(dest)
    sources={}
    for label,path in [('old',old_physical(grid)),('new',root/'physical'/grid)]:
        checked=verify_run(path) if label=='old' else verify_highz(path)
        write_json(dest/(label+'-physical-verification.json'),checked)
        if checked['status']!='passed':raise ValueError('physical input failed: '+str(checked))
        sources[label]=(path,read_json(path/'request.json'),read_json(path/'summary.json'))
    checkpoint=read_json(campaign['prior_checkpoint']);check_bindings(checkpoint['checkpoint_hashes'])
    bindings={str(p/f):file_digest(p/f) for p,_,_ in sources.values() for f in ('request.json','summary.json')}
    reference=campaign['sources'][grid]['details']['original']['details']['reference_cosmology']
    model=campaign['sources'][grid]['details']['original']['details']['engine_settings']
    cosmo=FiducialCosmology();cache={};likes={};results={};outcomes={};refs={}
    summary={'status':'running','grid':grid,'settings':settings,'source':campaign['sources'][grid],
        'outcomes':outcomes,'profiles':{},'native_bridges':{},'nesting':{},'input_files':bindings,
        'full_forecast_qualification':'pending'}
    write_json(dest/'summary.json',summary)
    for name,profile in all_profiles().items():
        print(grid+': '+name+' selecting physical observations',flush=True)
        identity=likelihood_key(profile)
        if identity not in likes:
            parts=[];references=[];config=study_config(profile,settings);g=study_shell_grid(config)
            for case in study_cells(profile,cosmo,settings=settings):
                expected=case['expected'];sourcecase=source_for_cell(profile,case['cell_id'])
                path=None;record=None
                if sourcecase is not None:
                    path,source_request,source_summary=sources['new' if sourcecase=='C22' else 'old']
                    key=sourcecase+':'+case['cell_id'];record=next(r for r in source_summary['workers'] if r['key']==key)
                    path=cell_directory(path,key)
                cachekey=request_digest({'source':str(path),'expected':expected,'k':g.centers.tolist(),
                    'widths':g.widths.tolist(),'independent_autos':profile.case=='I'})
                if cachekey not in cache:
                    source=load_source_cell(path,record) if path else None
                    part=project_cell(source,profile,expected,g)
                    ref={'id':cachekey,'source_directory':str(path) if path else None,
                        'source_record':record,'profile':asdict(profile),'expected':expected,'mapping':part['mapping']}
                    for field in ('derivative_check','neighborhood_check'):
                        if field in part:ref[field]=part.pop(field)
                    part.pop('arrays',None);part.pop('analytic_zero_support',None)
                    directory=dest/'projections';directory.mkdir(exist_ok=True)
                    write_json(directory/(cachekey+'.json'),jsonable(ref));cache[cachekey]=part
                    if source:bindings.update(source['source_files'])
                parts.append(cache[cachekey]);references.append(cachekey)
            like=assemble_study(profile,settings,parts,reference,model);likes[identity]=(like,references)
        like,references=likes[identity]
        # Same likelihood, different prior widths; always apply the prior once.
        like=replace(like,config=study_config(profile,settings))
        alias={'B16':'B','C16':'C'}.get(name)
        result=results[alias] if alias else infer_study(like)
        results[name]=result;outcomes[name]=constraint_row(result);refs[name]=references
        p=dest/'profiles'/name;save_result(p,profile,settings,result,references)
        summary['profiles'][name]={'files':artifact_hashes(p),'status':'passed'}
        if name in ('B','C'):summary['native_bridges'][name]=native_bridge(result,grid,name)
        write_json(dest/'summary.json',summary)
        print(grid+': '+name+' saved',flush=True)
    for a,b in comparisons():
        if a=='I':continue
        report=compare_nested_likelihoods(nesting_snapshot(results[a]),nesting_snapshot(results[b]))
        summary['nesting'][a+'_to_'+b]=report
    write_tables(outcomes,dest)
    check_bindings(bindings);check_campaign(root,grid)
    summary['projection_files']=artifact_hashes(dest/'projections')
    summary['status']='passed' if all(r['status']=='passed' for r in summary['nesting'].values()) else 'failed'
    write_json(dest/'summary.json',summary)
    if summary['status']!='passed':raise ValueError('nested information check failed: '+grid)
    return {'status':'passed','profiles':len(outcomes),'distinct_likelihoods':len(likes)}


def verify_inference(root,grid,*,recompute=True):
    check_campaign(root,grid);dest=root/'inference'/grid;s=read_json(dest/'summary.json')
    if (s['status']!='passed' or s['settings']!=GRID_PROFILES[grid] or set(s['profiles'])!=set(all_profiles())
            or s['source']!=read_json(root/'request.json')['sources'][grid]):raise ValueError('incomplete/foreign inference')
    check_bindings(s['input_files']);problems=verify_files(dest/'projections',s['projection_files'])
    if problems:raise ValueError('; '.join(problems))
    results={};sources={};projection_cache={}
    for name,profile in all_profiles().items():
        path=dest/'profiles'/name;problems=verify_files(path,s['profiles'][name]['files'])
        if problems:raise ValueError('; '.join(problems))
        result,meta=load_result(path);results[name]=result
        if recompute:verify_result(path,profile,GRID_PROFILES[grid])
        if canonical(constraint_row(result))!=canonical(s['outcomes'][name]):raise ValueError('constraint summary differs')
        expected=study_cells(profile,SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:1.),settings=GRID_PROFILES[grid])
        if len(expected)!=len(result.likelihood.cells) or len(meta['projections'])!=len(expected):raise ValueError('missing support')
        selected_parts=[]
        for case,cell,ref_id in zip(expected,result.likelihood.cells,meta['projections']):
            ref=read_json(dest/'projections'/(ref_id+'.json'))
            if canonical(ref['expected'])!=canonical(case['expected']):raise ValueError('projection support differs')
            if ref_id not in projection_cache:
                sourcepath=ref['source_directory']
                if sourcepath not in sources and sourcepath:
                    sources[sourcepath]=load_source_cell(sourcepath,ref['source_record'])
                source=sources.get(sourcepath)
                part=(select_cell(source,profile,case['expected'],result.grid) if source else
                      project_cell(None,profile,case['expected'],result.grid))
                projection_cache[ref_id]=part
            part=projection_cache[ref_id]
            selected_parts.append(part)
            if canonical(part['mapping'])!=canonical(ref['mapping']):raise ValueError('data selection mapping differs')
            for field in ('F_like','derivatives','covariance','fiducial_mean'):
                require_close(getattr(cell,field),getattr(part['cell'],field),'source-selected '+field)
        original=s['source']['details']['original']['details']
        # The physical reports already bind field positivity; rebuilding the
        # global information also checks source fiducials and unscaled priors.
        for part,cell in zip(selected_parts,result.likelihood.cells):
            part['cell'].validation=cell.validation
        check_projection_binding(result,profile,GRID_PROFILES[grid],selected_parts,
                                 original['reference_cosmology'],original['engine_settings'])
        if name in ('B','C'):
            bridge=native_bridge(result,grid,name)
            if bridge!=s['native_bridges'][name]:raise ValueError('independent native3 bridge differs')
    for a,b in comparisons():
        if a=='I':continue
        report=compare_nested_likelihoods(nesting_snapshot(results[a]),nesting_snapshot(results[b]))
        if report['status']!='passed' or canonical(report)!=canonical(s['nesting'][a+'_to_'+b]):
            raise ValueError('saved information nesting differs')
    check_bindings(s['input_files']);check_campaign(root,grid)
    return {'status':'passed','grid':grid,'profiles':len(results),'recomputed_precision_and_growth':recompute}


def common_highz(root,key):
    check_campaign(root);output=cell_directory(root/'common-k',key);fresh_directory(output)
    data=[];inputs=[];bindings={}
    for grid in ('angular64','shell304'):
        base=root/'physical'/grid;request=read_json(base/'request.json');s=read_json(base/'summary.json')
        errors=validate_study_ledger(request,s,study_source(GRID_PROFILES[grid]))
        if errors:raise ValueError(str(errors))
        rec=next(r for r in s['workers'] if r['key']==key);expected=next(r for r in request['expected'] if r['key']==key)
        path=cell_directory(base,key);verify_highz_cell(path,expected,request)
        source=load_source_cell(path,rec);bindings.update(source['source_files'])
        bindings.update({str(base/f):file_digest(base/f) for f in ('request.json','summary.json')})
        data.append(source['arrays']);inputs.append(source['inputs'])
    if canonical(inputs[0]['metadata'])!=canonical(inputs[1]['metadata']) or inputs[0]['fiducials']!=inputs[1]['fiducials']:
        raise ValueError('common-k physical definitions differ')
    engine=make_one_loop_engine(nfft=2048);query=np.union1d(data[0]['k'],data[1]['k'])
    direct=common_query_data(inputs[1],query,engine,FiducialCosmology());reports=[]
    for a in data:
        indices=np.searchsorted(query,a['k'])
        measured={k:a[k] for k in ('k','dk','mean','D','C')};measured['names']=inputs[1]['metadata']['local_names']
        rebuilt={'k':a['k'],'dk':a['dk'],'names':measured['names'],'mean':direct['mean'][indices],
            'D':direct['D'][indices],'C':direct['covariance_kernel'][indices]/a['dk'][:,None,None]}
        reports.append(compare_local(measured,rebuilt))
    np.savez_compressed(output/'query_arrays.npz',**direct)
    original=[{**{k:a[k] for k in ('k','dk','mean','D','C')},'names':inputs[1]['metadata']['local_names']} for a in data]
    report={'status':'passed' if all(r['status']=='passed' for r in reports) else 'failed','cell':key,
        'reconstructions':reports,'original_interpolation_diagnostic':compare_local(*original,shell=True),
        'input_files':bindings,'query_arrays_sha256':file_digest(output/'query_arrays.npz'),
        'sources':[read_json(root/'request.json')['sources'][g] for g in ('angular64','shell304')]}
    check_bindings(bindings);check_campaign(root);write_json(output/'report.json',report)
    if report['status']!='passed':raise ValueError('direct high-z common-k check failed')
    return report


def query_directory(root,key):
    if key.startswith('C22:'):return cell_directory(root/'common-k',key)
    identities=[read_json(old_physical(g)/'request.json')['source']['identity_hash'] for g in ('angular64','shell304')]
    for p in sorted(OLD.glob('shell-common-k-*/report.json')):
        r=read_json(p)
        if r.get('cell')==key and r.get('source_identities')==identities and r.get('status')=='passed':return p.parent
    raise ValueError('no bound common-k reconstruction for '+key)


def compare_campaign(root,*,verify=False):
    check_campaign(root);dest=root/'comparisons'
    if not verify:fresh_directory(dest)
    # The three independent high-z query evaluations retain their actual exits.
    common=read_json(dest/'common-execution.json') if verify else []
    if verify and (len(common)!=3 or any(r['exit_code']!=0 for r in common)):
        raise ValueError('common-k physical workers did not complete')
    for expected in ([] if verify else read_json(root/'physical/shell304/request.json')['expected']):
        key=expected['key'];cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),
            '--phase','compare','--worker',key];log=dest/(key.replace(':','__')+'-common.log')
        start=time.monotonic()
        with log.open('w') as f:process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        row={'key':key,'exit_code':process.returncode,'command':cmd,'elapsed_seconds':time.monotonic()-start}
        common.append(row);write_json(dest/'common-execution.json',common)
        if process.returncode:raise ValueError('common-k worker failed: '+key)
    reports=[];bindings={}
    for coarse,fine,axis in REFINEMENTS:
        a=root/'inference'/coarse;b=root/'inference'/fine
        sa=read_json(a/'summary.json');sb=read_json(b/'summary.json')
        for base,s in ((a,sa),(b,sb)):
            if s['status']!='passed':raise ValueError('unqualified inference in resolution comparison')
            bindings[str(base/'summary.json')]=file_digest(base/'summary.json')
        report={'coarse':coarse,'fine':fine,'axis':axis,'global':compare_results(sa['outcomes'],sb['outcomes']),'local':[]}
        for name,profile in all_profiles().items():
            ra,ma=load_result(a/'profiles'/name);rb,mb=load_result(b/'profiles'/name)
            for i,(ca,cb) in enumerate(zip(ra.likelihood.cells,rb.likelihood.cells)):
                if not ca.names:continue
                da={'k':ra.grid.centers,'dk':ra.grid.widths,'mean':ca.fiducial_mean,'D':ca.derivatives,'C':ca.covariance,'names':ca.names}
                db={'k':rb.grid.centers,'dk':rb.grid.widths,'mean':cb.fiducial_mean,'D':cb.derivatives,'C':cb.covariance,'names':cb.names}
                row={'profile':name,'cell':ca.metadata['cell_id']}
                if axis!='shell':row['measurement']=compare_local(da,db)
                else:
                    row['original_interpolation_diagnostic']=compare_local(da,db,shell=True)
                    sourcecase=source_for_cell(profile,ca.metadata['cell_id']);key=sourcecase+':'+ca.metadata['cell_id']
                    qpath=query_directory(root,key);qr=read_json(qpath/'report.json')
                    if file_digest(qpath/'query_arrays.npz')!=qr['query_arrays_sha256']:raise ValueError('common-k arrays changed')
                    for fname in ('report.json','query_arrays.npz'):bindings[str(qpath/fname)]=file_digest(qpath/fname)
                    checks=[]
                    with np.load(qpath/'query_arrays.npz',allow_pickle=False) as q:
                        for sourcebase,meta,data in ((a,ma,da),(b,mb,db)):
                            ref=read_json(sourcebase/'projections'/(meta['projections'][i]+'.json'))
                            rr=ref['mapping']['source_rows'];cc=ref['mapping']['source_columns'];ix=np.searchsorted(q['k'],data['k'])
                            if not np.array_equal(q['k'][ix],data['k']):raise ValueError('common k query does not cover study')
                            cov=q['covariance_kernel'][ix][:,rr][:,:,rr]/data['dk'][:,None,None]
                            if profile.case=='I':cov=independent_autos(cov)
                            direct={**data,'mean':q['mean'][ix][:,rr],'D':q['D'][ix][:,rr][:,:,cc],'C':cov}
                            checks.append(compare_local(data,direct))
                    row['measurement']={'status':'passed' if all(c['status']=='passed' for c in checks) else 'failed',
                        'criterion':'documented_direct_common_k','reconstructions':checks}
                report['local'].append(row)
        report['status']='passed' if report['global']['status']=='passed' and all(r['measurement']['status']=='passed' for r in report['local']) else 'failed'
        path=dest/(coarse+'-to-'+fine+'.json')
        if verify:verify_comparison_record(read_json(path),report)
        else:write_json(path,report)
        reports.append(report)
        print(coarse+' -> '+fine+': '+report['status'],flush=True)
    check_bindings(bindings);check_campaign(root)
    summary={'status':'passed' if all(r['status']=='passed' for r in reports) else 'failed',
        'pairs':[{k:r[k] for k in ('coarse','fine','axis','status')} for r in reports],
        'input_files':bindings,'common_workers':common,'full_forecast_qualification':'pending'}
    if verify:verify_comparison_record(read_json(dest/'summary.json'),summary)
    else:write_json(dest/'summary.json',summary)
    if summary['status']!='passed':raise ValueError('study refinement did not pass')
    return summary


def verify_campaign(root):
    check_campaign(root);reports={}
    for grid in GRID_PROFILES:
        physical=verify_highz(root/'physical'/grid)
        if physical['status']!='passed':raise ValueError('high-z physical verification failed '+grid)
        reports[grid]={'physical':physical,'inference':verify_inference(root,grid)}
    comp=compare_campaign(root,verify=True);check_bindings(comp['input_files'])
    if comp['status']!='passed' or len(comp['pairs'])!=4:raise ValueError('resolution comparisons incomplete')
    report={'status':'passed','grids':reports,'profiles':list(all_profiles()),'comparisons':comp['pairs'],
        'scope':'numerical_studies','independent_scientific_review':'pending'}
    write_json(root/'verification.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['prepare','physical','inference','compare','report','verify'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grid',choices=list(GRID_PROFILES))
    p.add_argument('--worker',help=argparse.SUPPRESS);args=p.parse_args(argv);root=args.output.resolve()
    if args.worker:
        if args.phase=='physical':
            r=read_json(root/'physical'/args.grid/'request.json')
            out=measure_highz_cell(r,args.worker,cell_directory(root/'physical'/args.grid,args.worker))
            return 0 if out['status']=='passed' else 1
        if args.phase=='compare':common_highz(root,args.worker);return 0
        raise ValueError('invalid worker phase')
    start=time.monotonic();record={'phase':args.phase,'grid':args.grid,'status':'running','command':sys.argv}
    try:
        if args.phase=='prepare':result=prepare(root)
        elif args.phase in ('physical','inference'):
            result={g:(run_physical if args.phase=='physical' else run_inference)(root,g)
                    for g in ([args.grid] if args.grid else GRID_PROFILES)}
        elif args.phase=='compare':result=compare_campaign(root)
        elif args.phase=='verify':result=verify_campaign(root)
        else:
            from scripts.elf_study_science import scientific_report
            result=scientific_report(root)
        record.update(status='passed',result=result,exit_code=0)
    except Exception as error:
        record.update(status='failed',error=str(error),error_type=type(error).__name__,exit_code=1)
        if root.is_dir():
            suffix=args.phase+('-'+args.grid if args.grid else '')
            (root/('failure-'+suffix+'.log')).write_text(traceback.format_exc())
    record['elapsed_seconds']=time.monotonic()-start
    if root.is_dir():
        path=root/('execution-'+args.phase+('-'+args.grid if args.grid else '')+'.json')
        if path.exists():path=root/(path.stem+'-'+str(time.time_ns())+'.json')
        write_json(path,record)
    print(json.dumps({k:v for k,v in record.items() if k!='result'},indent=2),flush=True)
    return record['exit_code']


if __name__=='__main__':raise SystemExit(main())
