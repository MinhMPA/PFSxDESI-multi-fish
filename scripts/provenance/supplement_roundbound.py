#!/usr/bin/env python
"""CT-A re-evaluation of the already adopted survey sensitivity studies.

Existing qualified outputs are read-only.  The unchanged ELF loop contribution
is reused only through the authenticated complete source and the previously
qualified pointwise counterterm difference. No old Fisher information is scaled.

Derived from supplement.py (sha256 cefe5b3a...b21542, output -01) with three
author-approved changes, 22 October 2026:
1. Newly computed profiles use roundbound_solver: the same 50-digit inverse,
   accepted by the bound implied by binary64 nearest rounding instead of the
   fixed 1e-10 cap that final-loop B_prior10 exceeded (1.079e-10).
2. Frozen source directories recorded relative to .worktrees/elf-engine
   (the 2.1-2.4 tail) are resolved against that root, independent of cwd.
3. compare() unpacks the three-element REFINEMENTS entries.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[5]
ROOT=REPO/'.worktrees/counterterms-leading'
OLDROOT=REPO/'.worktrees/elf-engine'
CTA=ROOT/'results/counterterm-leading-20261020-04'
OLD=OLDROOT/'results/takada-density-full-20261014-02'
OUT=ROOT/'results/ct-a-adoption-supplement-20261022-02'
DERIVED_FROM=Path(__file__).resolve().parent/'supplement.py'
SOLVER_MODULE=Path(__file__).resolve().parent/'roundbound_solver.py'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import time
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_adapter import ELFPowerSpectrum
from pfsfog.extended_artifacts import read_json,canonical,require_close,jsonable
from pfsfog.elf_survey_evidence import artifact_hashes,request_digest
from pfsfog.revision_validation import compare_nested_likelihoods
from scripts.run_pfs_density_scan import digest,write_json,InputBinding
from scripts.run_elf_studies import GRID_PROFILES,REFINEMENTS
from scripts.elf_study_evidence import study_source
from scripts.elf_study_profiles import study_shell_grid
from scripts.elf_study_data import project_cell
from scripts.elf_study_inference import validate_growth
from pfsfog.revision_forecast import apply_revision_priors
import roundbound_solver
roundbound_solver.install()
from scripts.elf_study_report import save_result,constraint_row,nesting_snapshot
from scripts.takada_study import corrected_profiles,corrected_cells,corrected_config,load_corrected_result,assemble_from_template
from scripts.counterterm_acceptance import authenticate_physical
from scripts.counterterm_comparison import Correction,CorrectedNeighborhood,correct_source,check_response_split
from scripts.counterterm_shared_observation import anchor_corrected_source
from scripts.run_counterterm_comparison import reconstructed_source,verify_physical_files
from scripts.overlap_checks import check_marginalization

NAMES=('A','B','D','C','B_k015','C_k015','B_prior3','C_prior3','B_prior10','C_prior10','B_no_elg1','C_no_elg1','H21','C21','C24')
PAIRS=(('A','B'),('B','D'),('D','C'),('B','C'),('B_k015','C_k015'),('B_prior3','C_prior3'),('B_prior10','C_prior10'),('B_no_elg1','C_no_elg1'),('C','H21'),('H21','C21'),('C21','C24'),('B','C24'))

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    request,exception=authenticate_physical(CTA)
    oldrequest=read_json(OLD/'request.json');InputBinding(oldrequest['bindings']).verify_all()
    bindings={str(Path(__file__).resolve()):digest(__file__),str(CTA/'request.json'):digest(CTA/'request.json'),str(OLD/'request.json'):digest(OLD/'request.json'),
        str(SOLVER_MODULE):digest(SOLVER_MODULE),str(DERIVED_FROM):digest(DERIVED_FROM)}
    for base in (CTA,OLD):
        for grid in GRID_PROFILES:
            for p in (base/'inference'/grid).glob('profiles/*/*'):
                if p.is_file():bindings[str(p)]=digest(p)
            for p in (base/'inference'/grid).glob('projections/*.json'):bindings[str(p)]=digest(p)
            for p in (base/'inference'/grid).glob('parts/*/*/record.json'):bindings[str(p)]=digest(p)
    sources={g:study_source(s) for g,s in GRID_PROFILES.items()}
    if canonical(sources)!=canonical(request['sources']):raise ValueError('qualified CT-A physical source changed')
    OUT.mkdir(parents=True)
    write_json(OUT/'request.json',{'schema':'ct-a-adoption-supplement-v1','model':'CT-A','profiles':NAMES,'pairs':PAIRS,
        'main_request_sha256':digest(CTA/'request.json'),'old_request_sha256':digest(OLD/'request.json'),
        'sources':sources,'bindings':bindings,'grids':GRID_PROFILES,'refinements':REFINEMENTS,
        'tolerances':{'relative_sigma':.005,'gain_percentage_points':.1,'FD_relative':.001,'FD_zero':1e-6},
        'fisher_solver':roundbound_solver.roundbound_solver_record(),
        'fisher_solver_scope':'newly computed profiles; A-D are copies of the qualified CT-A main results (decimal50-cholesky-v1)',
        'derived_from':{'script':str(DERIVED_FROM),'sha256':digest(DERIVED_FROM),'output':str(ROOT/'results/ct-a-adoption-supplement-20261022-01')},
        'historical_runner_exception':{k:v for k,v in exception.items() if k!='blob'} if exception else None})
    write_json(OUT/'state.json',{'status':'prepared','request_sha256':digest(OUT/'request.json'),'grids':{}})
    return {'status':'prepared','output':str(OUT),'bound_files':len(bindings)}

def bound_request():
    q=read_json(OUT/'request.json')
    if digest(OUT/'request.json')!=read_json(OUT/'state.json')['request_sha256']:raise ValueError('supplement request changed')
    InputBinding(q['bindings']).verify_all()
    authenticate_physical(CTA)
    return q

def load_corrected_physical(path,engine,cosmo):
    path=Path(path);receipt=read_json(path/'receipt.json')
    if receipt['status']!='passed':raise ValueError('unqualified CT physical source')
    verify_physical_files(path,receipt)
    old=reconstructed_source(receipt['source'])
    with np.load(path/'arrays.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    c=Correction(old,engine,cosmo)
    angular=jax.jit(lambda x:c.delta_pkmu(x,old['arrays']['signed_mu']))
    return {**old,'inputs':read_json(path/'inputs.json'),'arrays':arrays,
        'neighborhood_spectra':CorrectedNeighborhood(old,angular),
        'source_files':{**old['source_files'],**{str(path/n):digest(path/n) for n in ('receipt.json','arrays.npz','inputs.json')}},
        'corrected_physical_directory':str(path)}

def infer_roundbound(likelihood):
    result=apply_revision_priors(likelihood,likelihood.config.nuisance_factor,fisher_solver=roundbound_solver.ROUNDBOUND_FISHER_SOLVER)
    result.validation['growth_check']=validate_growth(result)
    return result

def resolve_frozen(ref):
    """Resolve source directories that the frozen Takada study stored relative to its worktree."""
    ref=deepcopy(ref)
    for r in (ref,ref.get('unchanged_observation_anchor') or {}):
        d=r.get('source_directory')
        if d is not None and not Path(d).is_absolute():
            r['frozen_source_directory']=d;r['source_directory']=str(OLDROOT/d)
    return ref

def main_physical(grid,name,cellid):
    r=read_json(CTA/'inference'/grid/'parts'/name/cellid/'record.json')
    return Path(r['physical_receipt']).parent

def new_highz(grid,row,engine,cosmo):
    cellid=row['cell_id'];dest=OUT/'inference'/grid/'physical'/cellid
    if (dest/'receipt.json').exists():return load_corrected_physical(dest,engine,cosmo)
    old,manifest=load_corrected_result(OLD/'inference'/grid/'profiles/C24')
    i=next(i for i,c in enumerate(old.likelihood.cells) if c.metadata['cell_id']==cellid)
    ref=resolve_frozen(read_json(OLD/'inference'/grid/'projections'/(manifest['projections'][i]+'.json')))
    source=correct_source(reconstructed_source(ref),engine,cosmo)
    if row['radial_bin'][0]<2.1:
        source=anchor_corrected_source(source,main_physical(grid,'B',cellid),digest(CTA/'request.json'))
    fullrow={**deepcopy(source['inputs']['metadata']),'empty_observation':False}
    config=corrected_config(corrected_profiles(2.4)['C24'],GRID_PROFILES[grid])
    part=project_cell(source,corrected_profiles(2.4)['C24'],fullrow,study_shell_grid(config))
    # The saved tail predates h-split diagnostics; its full six-coordinate AD/FD
    # and covariance-neighborhood qualification is still required here.
    split=check_response_split(source) if 'h_intrinsic_plus' in source['arrays'] else {'status':'not_present_in_historical_tail','full_six_coordinate_AD_FD':'passed'}
    dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'arrays.npz',**source['arrays'])
    np.save(dest/'neighborhood_coordinates.npy',source['neighborhood_coordinates'])
    write_json(dest/'inputs.json',source['inputs'])
    write_json(dest/'qualification.json',{'status':'passed','derivative_check':part['derivative_check'],'neighborhood_check':part['neighborhood_check'],'intrinsic_geometry':split})
    receipt={'status':'passed','request_sha256':digest(OUT/'request.json'),'source':ref,
        'source_files':source['source_files'],'diagnostic':source['counterterm_diagnostic'],
        'files':artifact_hashes(dest),'projection_files':{}}
    write_json(dest/'receipt.json',receipt)
    source['corrected_physical_directory']=str(dest)
    print(grid,'new high-z physical passed',cellid,flush=True)
    return source

def run_grid(grid):
    request=bound_request();settings=GRID_PROFILES[grid];engine=ELFPowerSpectrum(nfft=settings['nfft']);cosmo=FiducialCosmology()
    if canonical(study_source(settings,engine))!=canonical(request['sources'][grid]):raise ValueError('live CT-A theory differs')
    dest=OUT/'inference'/grid;dest.mkdir(parents=True,exist_ok=True)
    summary={'status':'running','grid':grid,'request_sha256':digest(OUT/'request.json'),'profiles':{},'outcomes':{},'nesting':{},'physical_sources':{}}
    if (dest/'summary.json').exists():summary=read_json(dest/'summary.json')
    profiles=corrected_profiles(2.4);cache={};parts_cache={};results={}
    for name in NAMES:
        path=dest/'profiles'/name
        if path.exists():
            results[name],_=load_corrected_result(path)
            continue
        started=time.monotonic();profile=profiles[name];refs=[]
        if name in ('A','B','D','C'):
            result,_=load_corrected_result(CTA/'inference'/grid/'profiles'/name)
            original_path=str(CTA/'inference'/grid/'profiles'/name)
        else:
            template,_=load_corrected_result(OLD/'inference'/grid/'profiles'/name)
            parts=[];config=corrected_config(profile,settings);shells=study_shell_grid(config)
            for case in corrected_cells(profile,cosmo,pfs_zmin=.6,settings=settings):
                row=case['expected'];cellid=row['cell_id'];source=None
                if row['empty_observation']:key='empty-'+cellid
                elif 'PFS-ELG' in row['active_populations'] and row['radial_bin'][0]>=1.6:
                    key='highz-'+cellid
                    if key not in cache:cache[key]=new_highz(grid,row,engine,cosmo)
                    source=cache[key]
                else:
                    base='B' if profile.case in ('A','B') else 'C'
                    physical=main_physical(grid,base,cellid);key=str(physical)
                    if key not in cache:cache[key]=load_corrected_physical(physical,engine,cosmo)
                    source=cache[key]
                partkey=request_digest({'physical':key,'row':row,'grid':shells.centers.tolist(),'case':profile.case})
                if partkey not in parts_cache:
                    part=project_cell(source,profile,row,shells)
                    record={'status':'passed','expected':row,'physical_directory':source['corrected_physical_directory'] if source else None,
                        'mapping':part['mapping'],'derivative_check':part.get('derivative_check'),'neighborhood_check':part.get('neighborhood_check')}
                    (dest/'projections').mkdir(exist_ok=True)
                    write_json(dest/'projections'/(partkey+'.json'),jsonable(record))
                    parts_cache[partkey]={'cell':part['cell'],'fiducials':part['fiducials']}
                    if source:summary['physical_sources'][source['corrected_physical_directory']]=digest(Path(source['corrected_physical_directory'])/'receipt.json')
                parts.append(parts_cache[partkey]);refs.append(partkey)
            like=assemble_from_template(template.likelihood,parts,config=config)
            result=infer_roundbound(replace(like,model=engine.numerical_settings()));original_path=None
        save_result(path,profile,settings,result,refs)
        saved,_=load_corrected_result(path)
        audit=check_marginalization(saved)
        write_json(path/'independent.json',jsonable({'status':'passed','checks':audit}))
        results[name]=saved;summary['outcomes'][name]=constraint_row(saved)
        summary['profiles'][name]={'status':'passed','files':artifact_hashes(path),'elapsed_seconds':time.monotonic()-started,'qualified_main_copy':original_path}
        write_json(dest/'summary.json',jsonable(summary))
        print(grid,name,'passed',round(time.monotonic()-started,2),'s',flush=True)
    for a,b in PAIRS:
        if b=='C24':
            from scripts.pfs_bias_inference import check_cross_information
            check=check_cross_information(results[a].cov_cosmo_marg,results[b].cov_cosmo_marg)
        else:check=compare_nested_likelihoods(nesting_snapshot(results[a]),nesting_snapshot(results[b]))
        if check['status']!='passed':raise ValueError('nested information failed '+a+' '+b)
        summary['nesting'][a+'_to_'+b]=check
    summary.update(status='passed',projection_files=artifact_hashes(dest/'projections'))
    write_json(dest/'summary.json',jsonable(summary))
    state=read_json(OUT/'state.json');state['grids'][grid]={'status':'passed','summary_sha256':digest(dest/'summary.json')};write_json(OUT/'state.json',state)
    bound_request()
    return {'status':'passed','grid':grid,'profiles':len(results)}

def compare():
    bound_request();metrics=[]
    for a,b,_ in REFINEMENTS:
        x=read_json(OUT/'inference'/a/'summary.json');y=read_json(OUT/'inference'/b/'summary.json')
        if x['status']!='passed' or y['status']!='passed':raise ValueError('unqualified grid')
        for name in NAMES:
            for mode in ('marginalized','nuisance_fixed'):
                for field in ('cosmology','growth'):
                    xx=np.array(x['outcomes'][name][mode][field]);yy=np.array(y['outcomes'][name][mode][field]);err=float(np.max(abs(yy/xx-1)))
                    if err>=.005:raise ValueError('uncertainty refinement failed')
                    metrics.append({'grids':[a,b],'profile':name,'mode':mode,'field':field,'relative_sigma':err})
        for before,after in PAIRS:
            for field in ('cosmology','growth'):
                xx=100*(1-np.array(x['outcomes'][after]['marginalized'][field])[:8]/np.array(x['outcomes'][before]['marginalized'][field])[:8]);yy=100*(1-np.array(y['outcomes'][after]['marginalized'][field])[:8]/np.array(y['outcomes'][before]['marginalized'][field])[:8]);err=float(np.max(abs(yy-xx)))
                if err>=.1:raise ValueError('gain refinement failed')
                metrics.append({'grids':[a,b],'pair':[before,after],'field':field,'gain_percentage_points':err})
    record={'status':'passed','request_sha256':digest(OUT/'request.json'),'checks':metrics,
        'max_relative_sigma':max(x.get('relative_sigma',0) for x in metrics),'max_gain_percentage_points':max(x.get('gain_percentage_points',0) for x in metrics),
        'inputs':{str(OUT/'inference'/g/'summary.json'):digest(OUT/'inference'/g/'summary.json') for g in GRID_PROFILES}}
    write_json(OUT/'comparison.json',record);return {k:v for k,v in record.items() if k not in ('checks','inputs')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','run','compare'],required=True);p.add_argument('--grid',choices=GRID_PROFILES,default='shell304');a=p.parse_args()
    r=prepare() if a.phase=='prepare' else run_grid(a.grid) if a.phase=='run' else compare()
    print(json.dumps(r,indent=2),flush=True)
