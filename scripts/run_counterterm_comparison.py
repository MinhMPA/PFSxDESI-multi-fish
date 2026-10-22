#!/usr/bin/env python
"""Authenticated, resumable main-survey counterterm comparison."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import subprocess
import time
import traceback
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_adapter import ELFPowerSpectrum
from pfsfog.elf_survey_evidence import artifact_hashes,request_digest,verify_files
from pfsfog.extended_artifacts import read_json,require_close,canonical
from pfsfog.revision_forecast import CellLikelihood
from pfsfog.revision_validation import compare_nested_likelihoods
from scripts.run_pfs_density_scan import InputBinding,digest,write_json
from scripts.run_elf_studies import GRID_PROFILES,REFINEMENTS
from scripts.elf_study_evidence import load_source_cell,study_source
from scripts.elf_study_data import project_cell
from scripts.elf_study_inference import infer_study
from scripts.elf_study_report import save_result,nesting_snapshot
from scripts.elf_study_profiles import paper_profiles,study_shell_grid
from scripts.takada_study import load_corrected_result,assemble_from_template,corrected_cells,corrected_config
from scripts.takada_density import retarget_source,preserve_unchanged_observation
from scripts.counterterm_comparison import correct_source,validate_request,unique_cell_records,check_response_split
from scripts.counterterm_shared_observation import anchor_corrected_source
TRACK='CT-A'
PUBLIC={'A':'A','B':'B','D':'C','C':'D'}
OLD_REQUEST_SHA='6bf97e62d77f06897356fd65c387de2ae90e3a807117e6471d044f493b4cc652'
OWN=('scripts/counterterm_acceptance.py','tests/test_counterterm_acceptance.py','scripts/counterterm_shared_observation.py','tests/test_counterterm_shared_observation.py','tests/test_counterterm_comparison.py','scripts/counterterm_comparison.py','scripts/run_counterterm_comparison.py','scripts/takada_study.py',
     'scripts/takada_density.py','survey_specs/PFS_nz_takada2014_table2.txt')


def prepare(source_root,output):
    source_root=Path(source_root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('refusing to replace an immutable request')
    old=source_root/'results/takada-density-full-20261014-02';oldrequest=old/'request.json'
    if digest(oldrequest)!=OLD_REQUEST_SHA:raise ValueError('historical Takada request changed')
    previous=read_json(oldrequest);InputBinding(previous['bindings']).verify_all()
    # Evaluate the historical source in its own process and import context.
    code="from pathlib import Path;import sys,json;root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'ps_1loop_jax'));import jax;jax.config.update('jax_enable_x64',True);from scripts.elf_study_evidence import study_source;from scripts.run_elf_studies import GRID_PROFILES;print(json.dumps({g:study_source(s) for g,s in GRID_PROFILES.items()}))"
    live=json.loads(subprocess.check_output([str(source_root/'.venv/bin/python'),'-c',code],cwd=source_root,text=True))
    if canonical(live)!=canonical(previous['original_source']):raise ValueError('historical physical source changed')
    sources={g:study_source(s) for g,s in GRID_PROFILES.items()}
    bindings=dict(previous['bindings']);bindings[str(oldrequest)]=digest(oldrequest)
    # Bind profiles and projection records not necessarily part of the old request.
    for base in (old,source_root/'results/elf-survey-studies-20261010-01'):
        for grid in GRID_PROFILES:
            for name in PUBLIC:
                p=base/'inference'/grid/'profiles'/name
                for f in p.glob('*'):bindings[str(f)]=digest(f)
            for f in (base/'inference'/grid/'projections').glob('*.json'):bindings[str(f)]=digest(f)
    bindings.update({str(ROOT/n):digest(ROOT/n) for n in OWN})
    request={'schema_version':1,'track':TRACK,'source_root':str(source_root),'old_study':str(old),
             'historical_request_sha256':OLD_REQUEST_SHA,'historical_source':live,'sources':sources,
             'bindings':bindings,'grids':GRID_PROFILES,'public_labels':PUBLIC,
             'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'method':'pointwise complete new-minus-old counterterms; unchanged authenticated loop contribution',
             'numerical_contract':{'error_shift':.005,'gain_shift_percentage_points':.1,
                                   'fd_relative':.001,'fd_zero':1e-6,'two_successive_steps':True}}
    output.mkdir(parents=True);write_json(output/'request.json',request)
    write_json(output/'state.json',{'status':'prepared','request_sha256':digest(output/'request.json'),'grids':{},'active_jobs':[]})
    return {'status':'prepared','bound_files':len(bindings),'old_bound_files_verified':len(previous['bindings'])}


def load_reference(request,grid,name):
    old=Path(request['old_study']);result,manifest=load_corrected_result(old/'inference'/grid/'profiles'/name)
    base=old
    if name in ('A','B'):
        base=Path(request['source_root'])/'results/elf-survey-studies-20261010-01'
        manifest=read_json(base/'inference'/grid/'profiles'/name/'manifest.json')
    refs=[read_json(base/'inference'/grid/'projections'/(h+'.json')) for h in manifest['projections']]
    if len(refs)!=len(result.likelihood.cells):raise ValueError('incomplete original projections')
    return result,unique_cell_records([{**r,'cell_id':r['expected']['cell_id']} for r in refs])


def reconstructed_source(ref):
    source=load_source_cell(ref['source_directory'],ref['source_record'])
    if ref.get('source_target') is not None:source=retarget_source(source,ref['source_target'])
    ar=ref.get('unchanged_observation_anchor')
    if ar:
        anchor=load_source_cell(ar['source_directory'],ar['source_record'])
        source=preserve_unchanged_observation(source,anchor)
    return source


def verify_physical_files(directory,receipt):
    directory=Path(directory)
    actual={p.name for p in directory.iterdir() if p.is_file()}
    if actual!=set(receipt['files'])|{'receipt.json'}:raise ValueError('physical artifact inventory changed')
    for name,sha in receipt['files'].items():
        if digest(directory/name)!=sha:raise ValueError('changed physical artifact: '+name)
    for name,sha in receipt.get('projection_files',{}).items():
        if digest(name)!=sha:raise ValueError('changed physical projection: '+name)


def save_part(directory,part,receipt):
    directory.mkdir(parents=True,exist_ok=False);cell=part['cell']
    np.savez_compressed(directory/'arrays.npz',F_like=cell.F_like,D=cell.derivatives,C=cell.covariance,mean=cell.fiducial_mean)
    write_json(directory/'record.json',{'names':cell.names,'metadata':cell.metadata,'validation':cell.validation,
        'fiducials':part['fiducials'],'mapping':part['mapping'],'derivative_check':part.get('derivative_check'),
        'neighborhood_check':part.get('neighborhood_check'),'physical_receipt':str(receipt),
        'arrays_sha256':digest(directory/'arrays.npz')})


def load_part(directory,request_sha):
    r=read_json(directory/'record.json');physical=read_json(r['physical_receipt'])
    if physical['status']!='passed' or physical['request_sha256']!=request_sha:raise ValueError('unqualified physical cell')
    if physical.get('projection_files',{}).get(str(directory/'record.json'))!=digest(directory/'record.json'):raise ValueError('changed projected qualification record')
    if digest(directory/'arrays.npz')!=r['arrays_sha256']:raise ValueError('changed projected physical arrays')
    with np.load(directory/'arrays.npz',allow_pickle=False) as a:
        cell=CellLikelihood(a['F_like'],r['names'],r['metadata'],r['validation'],a['D'],a['C'],a['mean'])
    return {'cell':cell,'fiducials':r['fiducials']}


def qualify(output):
    """Fresh full-ELF test is an explicit request-bound prerequisite."""
    request=validate_request(output,TRACK);output=Path(output);start=time.monotonic()
    command=[sys.executable,'-m','pytest','tests/test_counterterm_comparison.py','-k','fresh_complete','-q']
    log=output/'fresh-elf.log'
    with log.open('w') as stream:r=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    receipt={'status':'passed' if r.returncode==0 else 'failed','exit_code':r.returncode,'command':command,
        'request_sha256':digest(output/'request.json'),'elapsed_seconds':time.monotonic()-start,'log_sha256':digest(log),
        'physical_comparison':'unequal DESI and DESI x PFS, nfft=2048, complete means and all six cosmological/nuisance derivatives'}
    write_json(output/'fresh-elf.json',receipt)
    if r.returncode:raise ValueError('fresh complete ELF comparisons failed')
    return receipt


def run_grid(output,grid,benchmark=False):
    output=Path(output);request=validate_request(output,TRACK);request_sha=digest(output/'request.json')
    fresh=read_json(output/'fresh-elf.json')
    if fresh['status']!='passed' or fresh['request_sha256']!=request_sha or digest(output/'fresh-elf.log')!=fresh['log_sha256']:
        raise ValueError('missing fresh complete ELF qualification')
    settings=GRID_PROFILES[grid];engine=ELFPowerSpectrum(nfft=settings['nfft']);cosmo=FiducialCosmology()
    if canonical(study_source(settings,engine))!=canonical(request['sources'][grid]):raise ValueError('new physical source changed')
    dest=output/'inference'/grid;dest.mkdir(parents=True,exist_ok=True)
    profiles=paper_profiles();templates={};refs={};expected={};groups={};jobs=[]
    for name in PUBLIC:
        templates[name],refs[name]=load_reference(request,grid,name)
        expected[name]=corrected_cells(profiles[name],cosmo,pfs_zmin=.6,settings=settings)
        for case in expected[name]:
            row=case['expected'];ref=refs[name][row['cell_id']]
            key=request_digest({k:ref.get(k) for k in ('source_directory','source_record','source_target','unchanged_observation_anchor')})
            if row['empty_observation']:key='empty-'+row['cell_id']
            groups.setdefault(key,{'ref':ref,'jobs':[]})['jobs'].append((name,row))
    # Benchmark a complete two-tracer physical cell with its A/B restrictions.
    if benchmark:
        groups=dict([next((k,g) for k,g in groups.items() if len(g['jobs'][0][1]['active_populations'])==2)])
    for key,group in groups.items():
        start=time.monotonic();physical=dest/'physical'/key;receiptfile=physical/'receipt.json'
        if receiptfile.exists() and read_json(receiptfile)['status']=='passed':
            saved=read_json(receiptfile)
            if saved['request_sha256']!=request_sha:raise ValueError('mixed physical request')
            verify_physical_files(physical,saved)
            continue
        if physical.exists():raise ValueError('previous failed/incomplete cell retained: '+str(physical))
        physical.mkdir(parents=True);receipt={'status':'running','request_sha256':request_sha,'physical_key':key,
            'source':group['ref'],'source_identity':request['sources'][grid]['identity_hash']}
        write_json(receiptfile,receipt)
        print(grid,'physical',key[:10],[n+':'+r['cell_id'] for n,r in group['jobs']],flush=True)
        try:
            source=None
            if not group['jobs'][0][1]['empty_observation']:
                old=reconstructed_source(group['ref']);source=correct_source(old,engine,cosmo)
                if any(name in ('D','C') for name,_ in group['jobs']):
                    cellid=group['jobs'][0][1]['cell_id']
                    anchorpart=read_json(dest/'parts/B'/cellid/'record.json')
                    anchorpath=Path(anchorpart['physical_receipt']).parent
                    source=anchor_corrected_source(source,anchorpath,request_sha)
                    receipt['corrected_observation_anchor']=source['corrected_observation_anchor']
                np.savez_compressed(physical/'arrays.npz',**source['arrays'])
                np.save(physical/'neighborhood_coordinates.npy',source['neighborhood_coordinates'])
                write_json(physical/'inputs.json',source['inputs'])
                receipt['diagnostic']=source['counterterm_diagnostic']
                receipt['source_files']=source['source_files']
                # Validate the entire original six-coordinate neighborhood before
                # restricting to the native-three survey cosmology.
                fullrow={**deepcopy(source['inputs']['metadata']),'empty_observation':False}
                fullpart=project_cell(source,profiles['C'],fullrow,study_shell_grid(corrected_config(profiles['C'],settings)))
                write_json(physical/'qualification.json',{'intrinsic_geometry':check_response_split(source),'derivative_check':fullpart['derivative_check'],
                    'neighborhood_check':fullpart['neighborhood_check'],'validation':fullpart['cell'].validation})
            for name,row in group['jobs']:
                shells=study_shell_grid(corrected_config(profiles[name],settings))
                part=project_cell(source,profiles[name],row,shells)
                save_part(dest/'parts'/name/row['cell_id'],part,receiptfile)
            receipt['projection_files']={str(dest/'parts'/name/row['cell_id']/f):digest(dest/'parts'/name/row['cell_id']/f) for name,row in group['jobs'] for f in ('record.json','arrays.npz')}
            receipt.update(status='passed',elapsed_seconds=time.monotonic()-start,files=artifact_hashes(physical))
            receipt['files'].pop('receipt.json',None);write_json(receiptfile,receipt)
            print(grid,'passed',key[:10],round(receipt['elapsed_seconds'],2),'s',flush=True)
        except Exception as error:
            receipt.update(status='failed',error=str(error),elapsed_seconds=time.monotonic()-start)
            if hasattr(error,'matrix'):np.save(physical/'failed_covariance.npy',error.matrix)
            (physical/'failure.log').write_text(traceback.format_exc());write_json(receiptfile,receipt);raise
        finally:
            source=None;old=None;gc.collect();jax.clear_caches()
    if benchmark:return {'status':'passed','grid':grid,'benchmark':read_json(receiptfile)}
    results={};rows=[];summary={'status':'running','request_sha256':request_sha,'grid':grid,'nesting':{}}
    for name in PUBLIC:
        template=templates[name];parts=[load_part(dest/'parts'/name/c['expected']['cell_id'],request_sha) for c in expected[name]]
        like=assemble_from_template(template.likelihood,parts,config=corrected_config(profiles[name],settings))
        like=replace(like,model=engine.numerical_settings());result=infer_study(like);results[name]=result
        path=dest/'profiles'/name
        if not path.exists():save_result(path,profiles[name],settings,result,[c['expected']['cell_id'] for c in expected[name]])
        saved,_=load_corrected_result(path)
        require_close(saved.covariance,result.covariance,'saved corrected full inverse',rtol=1e-10,atol=1e-12)
        oldagain=infer_study(template.likelihood)
        require_close(oldagain.covariance,template.covariance,'historical full inverse',rtol=1e-10,atol=1e-12)
        require_close(oldagain.growth_cov_marg,template.growth_cov_marg,'historical growth inverse',rtol=1e-10,atol=1e-12)
        rows.extend([result_row(grid,name,TRACK,path,result),result_row(grid,name,'old',Path(request['old_study'])/'inference'/grid/'profiles'/name,oldagain)])
    for a,b in [('A','B'),('B','D'),('D','C'),('B','C')]:
        check=compare_nested_likelihoods(nesting_snapshot(results[a]),nesting_snapshot(results[b]))
        summary['nesting'][a+'_to_'+b]=check
        if check['status']!='passed':
            summary['status']='failed';write_json(dest/'summary.json',summary);raise ValueError('nested observations failed: '+str(check['reasons']))
        for field in ('cov_cosmo_marg','cov_cosmo_nuisance_fixed'):
            ca=getattr(results[a],field);cb=getattr(results[b],field);scale=np.sqrt(np.diag(ca))
            eigen=np.linalg.eigvalsh((ca-cb)/scale[:,None]/scale[None,:])
            if eigen[0]<-1e-10:raise ValueError('nested common covariance ordering failed')
    summary.update(status='passed',rows=rows);write_json(dest/'summary.json',summary)
    state=read_json(output/'state.json');state['grids'][grid]={'status':'passed','summary_sha256':digest(dest/'summary.json')};write_json(output/'state.json',state)
    return {'status':'passed','grid':grid,'rows':len(rows)}


def result_row(grid,name,model,path,result):
    z=np.flatnonzero(np.isclose(result.growth_z,1.1,rtol=0,atol=1e-14))
    if len(z)!=1:raise ValueError('growth pivot must occur exactly once')
    row={'grid':grid,'public_case':PUBLIC[name],'internal_profile':name,'model':model,
         'extra_fog':'not_present','result_directory':str(path.resolve()),'status':'passed'}
    for kind,suffix in [('marginalized','marg'),('nuisance_fixed','nuisance_fixed')]:
        sigma=dict(zip(result.likelihood.cosmo_names,np.sqrt(np.diag(getattr(result,'cov_cosmo_'+suffix)))))
        sigma['fsigma8_z1p1']=float(np.sqrt(getattr(result,'growth_cov_'+suffix)[z[0],z[0]]))
        row['sigma_'+kind]=sigma
    return row


def convergence(rows):
    rows=[r for r in rows if r['model']==TRACK]
    keyed={(r['grid'],r['public_case']):r for r in rows}
    if len(keyed)!=20 or len(rows)!=20:raise ValueError('incomplete or duplicate grid/case comparison')
    checks=[]
    for coarse,fine,label in REFINEMENTS:
        for kind in ('marginalized','nuisance_fixed'):
            for case in 'ABCD':
                a=keyed[coarse,case]['sigma_'+kind];b=keyed[fine,case]['sigma_'+kind]
                for n in a:
                    shift=abs(b[n]/a[n]-1);checks.append({'refinement':label,'coarse':coarse,'fine':fine,
                        'kind':kind,'case':case,'quantity':n,'error_shift':shift,'passed':shift<.005})
            for a,b in [('A','B'),('B','C'),('C','D'),('B','D')]:
                for n in keyed[coarse,a]['sigma_'+kind]:
                    gains=[100*(1-keyed[g,b]['sigma_'+kind][n]/keyed[g,a]['sigma_'+kind][n]) for g in (coarse,fine)]
                    shift=abs(gains[1]-gains[0]);checks.append({'refinement':label,'coarse':coarse,'fine':fine,
                        'kind':kind,'gain':a+'_to_'+b,'quantity':n,'gain_shift_pp':shift,'passed':shift<.1})
    return {'status':'passed' if all(r['passed'] for r in checks) else 'failed','checks':checks}


def _reconstruct_verification(output,request):
    output=Path(output);sha=digest(output/'request.json');rows=[]
    for grid in GRID_PROFILES:
        dest=output/'inference'/grid;summary=read_json(dest/'summary.json')
        if summary['status']!='passed' or summary['request_sha256']!=sha:raise ValueError('unqualified grid')
        for name in PUBLIC:
            path=dest/'profiles'/name;saved,_=load_corrected_result(path);again=infer_study(saved.likelihood)
            require_close(saved.covariance,again.covariance,'reopened full inverse',rtol=1e-10,atol=1e-12)
            require_close(saved.cov_cosmo_marg,again.cov_cosmo_marg,'reopened Schur',rtol=1e-10,atol=1e-12)
            for cell in saved.likelihood.cells:
                part=load_part(dest/'parts'/name/cell.metadata['cell_id'],sha)['cell']
                for left,right in [(cell.F_like,part.F_like),(cell.derivatives,part.derivatives),(cell.covariance,part.covariance)]:
                    if not np.array_equal(left,right):raise ValueError('saved physical selection differs')
        for p in (dest/'physical').glob('*/receipt.json'):
            receipt=read_json(p)
            if receipt['status']!='passed' or receipt['request_sha256']!=sha:raise ValueError('unqualified physical receipt')
            verify_physical_files(p.parent,receipt)
        rows+=summary['rows']
    gridcheck=convergence(rows);status='qualified' if gridcheck['status']=='passed' else 'provisional'
    comparison={'schema_version':1,'track':TRACK,'request_sha256':sha,'status':status,'rows':rows,'convergence':gridcheck}
    return comparison


def _publish_verification(output,comparison,acceptance_receipt):
    output=Path(output);rows=comparison['rows'];status=comparison['status'];gridcheck=comparison['convergence']
    write_json(output/'comparison.json',comparison)
    state=read_json(output/'state.json');state.update(status=status,active_jobs=[],comparison_sha256=digest(output/'comparison.json'),acceptance_receipt=str(acceptance_receipt),acceptance_receipt_sha256=digest(acceptance_receipt));write_json(output/'state.json',state)
    findings=[f'# {TRACK}: {status}', '', 'Main PFS range 0.6–1.6, 1200 deg², Takada Table 2 density; all existing nuisance priors retained.', '']
    findings+=['| Model | Case | σ ln(10¹⁰As) | σ Mν [eV] | σ Ωm | σ fσ8(z=1.1) |','|---|---|---:|---:|---:|---:|']
    for row in rows:
        if row['grid']=='final-loop':findings.append('| '+row['model']+' | '+row['public_case']+' | '+' | '.join(f'{v:.9g}' for v in row['sigma_marginalized'].values())+' |')
    findings+=['',f'Grid comparison: {gridcheck["status"]}. Complete physical and inference receipts are retained per cell and per grid.']
    (output/'FINDINGS.md').write_text('\n'.join(findings)+'\n')


def verify(output):
    from scripts.counterterm_acceptance import qualify
    return qualify(output,lambda request:_reconstruct_verification(output,request),
                   lambda comparison,receipt:_publish_verification(output,comparison,receipt))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['prepare','qualify','benchmark','run','verify'])
    parser.add_argument('--source-root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--grid',choices=list(GRID_PROFILES),default='loop2048')
    args=parser.parse_args();output=Path(args.output);start=time.monotonic()
    try:
        if args.phase=='prepare':result=prepare(args.source_root,output)
        elif args.phase=='qualify':result=qualify(output)
        elif args.phase in ('benchmark','run'):result=run_grid(output,args.grid,benchmark=args.phase=='benchmark')
        else:result=verify(output)
        print(json.dumps(result,default=str),flush=True)
    except Exception:
        if output.exists():
            receipt={'status':'failed','phase':args.phase,'grid':args.grid,'elapsed_seconds':time.monotonic()-start,
                     'command':sys.argv,'traceback':traceback.format_exc()}
            write_json(output/f'failure-{args.phase}-{args.grid}-{time.time_ns()}.json',receipt)
        raise
if __name__=='__main__':main()
