#!/usr/bin/env python
"""Recompute the manuscript comparisons with an explicit Takada Table 2 survey."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import subprocess
import time
import traceback

import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.extended_artifacts import canonical,jsonable,read_json,require_close
from pfsfog.elf_survey_evidence import artifact_hashes,write_json,file_digest,request_digest
from pfsfog.revision_validation import compare_nested_likelihoods
from scripts.run_pfs_density_scan import InputBinding
from scripts.run_elf_studies import GRID_PROFILES,REFINEMENTS
from scripts.elf_study_evidence import load_source_cell,study_source
from scripts.elf_study_data import project_cell
from scripts.elf_study_inference import infer_study
from scripts.elf_study_profiles import study_shell_grid
from scripts.elf_study_report import (load_result,save_result,constraint_row,nesting_snapshot,comparisons)
from scripts.pfs_density_response import gamma3_zero_support
from scripts.takada_density import retarget_source,preserve_unchanged_observation,TABLE
from scripts.takada_study import (corrected_profiles,corrected_cells,corrected_config,
    assemble_from_template,load_corrected_result)

STUDY=ROOT/'results/elf-survey-studies-20261010-01'
OWN_FILES=('scripts/takada_density.py','scripts/takada_study.py','scripts/run_takada_revision.py',
           'survey_specs/PFS_nz_takada2014_table2.txt')


def original_bindings():
    study=ROOT/'docs/revision/evidence/elf-survey-studies-2026-10-10/STUDY_CHECKPOINT.json'
    grid=ROOT/'docs/revision/evidence/elf-grid-convergence-2026-10-09/GRID_CHECKPOINT.json'
    frozen=read_json(study)
    files={str(ROOT/p):r['sha256'] for p,r in frozen['files'].items()}
    files.update(read_json(grid)['checkpoint_hashes'])
    files.update({str(p):file_digest(p) for p in (study,grid)})
    return files


def verify_original_source(original,current):
    """Authenticate every original source, allowing the explicitly added table.

    The historical source collector hashes all survey text files, including an
    unused new file. Removing this one declared addition must recover its exact
    previous identity. No old file, engine setting or dependency is exempted.
    """
    normalized=deepcopy(current)
    inner=normalized['details']['original']
    added=inner['details']['survey_input_hashes'].pop(str(TABLE),None)
    if added!=file_digest(TABLE):raise ValueError('new Takada table is missing or changed')
    inner['identity_hash']=request_digest(inner['details'])
    normalized['identity_hash']=request_digest(normalized['details'])
    if canonical(normalized)!=canonical(original):
        raise ValueError('historical theory source or dependency changed beyond the added table')
    return {'status':'passed','only_added_input':str(TABLE),'added_sha256':added,
            'original_identity':original['identity_hash'],'current_identity':current['identity_hash']}


def prepare(output,zmin,zmax):
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to replace a calculation')
    files=original_bindings()
    print('Authenticating original physical and inference evidence',flush=True)
    InputBinding(files).verify_all()
    source={g:study_source(s) for g,s in GRID_PROFILES.items()}
    old=read_json(STUDY/'request.json')['sources']
    controls={g:verify_original_source(old[g],source[g]) for g in GRID_PROFILES}
    files.update({str(ROOT/n):file_digest(ROOT/n) for n in OWN_FILES})
    output.mkdir(parents=True)
    request={'schema':'takada-revision-v1','pfs_zmin':zmin,'pfs_zmax':zmax,
        'scope':'Takada Table 2 densities; explicit coverage; all other physical assumptions unchanged',
        'profiles':{n:asdict(p) for n,p in corrected_profiles(zmax).items()},
        'grids':GRID_PROFILES,'refinements':REFINEMENTS,'bindings':files,'original_source':source,
        'original_source_authentication':controls,
        'original_study':str(STUDY),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write_json(output/'request.json',jsonable(request))
    write_json(output/'state.json',{'status':'prepared','request_sha256':file_digest(output/'request.json'),'grids':{}})
    return {'status':'prepared','bound_files':len(files)}


def request_and_binding(output,*,verify_all=False):
    output=Path(output);request=read_json(output/'request.json');state=read_json(output/'state.json')
    if file_digest(output/'request.json')!=state['request_sha256']:raise ValueError('corrected request changed')
    profiles=corrected_profiles(request['pfs_zmax'])
    if (request['schema']!='takada-revision-v1' or request['pfs_zmin'] not in (.6,.7)
            or canonical(request['profiles'])!=canonical({n:asdict(p) for n,p in profiles.items()})
            or request['grids']!=GRID_PROFILES):raise ValueError('corrected experiment changed')
    binding=InputBinding(request['bindings'])
    for name in OWN_FILES:binding.check(ROOT/name)
    if verify_all:binding.verify_all()
    return request,binding


def source_target(source,expected):
    target=deepcopy(source['inputs']['metadata'])
    for key in ('cell_id','nominal_bin_index','nominal_bin','radial_bin','model_redshift','volume','region'):
        target[key]=deepcopy(expected[key])
    for name,value in expected['nbar'].items():target['nbar'][name]=value
    return target


def corrected_source(source,expected,*,unchanged=None):
    changed=retarget_source(source,source_target(source,expected))
    if unchanged is not None:changed=preserve_unchanged_observation(changed,unchanged)
    support=changed['inputs']['analytic_zero_response']['support']
    existing={(n,tuple(p),ell) for n,p,ell in support}
    support.extend(row for row in gamma3_zero_support(changed['inputs']) if row not in existing)
    return changed


def measure_tail(output,grid):
    from pfsfog.one_loop import make_one_loop_engine
    from pfsfog.elf_covariance_audit import audit_neighborhood
    from scripts.validate_elf_survey import covariance_inputs
    from scripts.validate_elf_engine import measure_cell
    output=Path(output);request,binding=request_and_binding(output)
    directory=output/'physical'/grid/'C24-tail';directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipt={'status':'running','request_sha256':file_digest(output/'request.json')}
    write_json(directory/'worker.json',receipt)
    try:
        settings=GRID_PROFILES[grid];cosmo=FiducialCosmology();engine=make_one_loop_engine(nfft=settings['nfft'])
        if canonical(study_source(settings,engine))!=canonical(request['original_source'][grid]):
            raise ValueError('tail theory dependency changed')
        profile=corrected_profiles(2.4)['C24'];config=corrected_config(profile,settings,extended=True)
        case=next(c for c in corrected_cells(profile,cosmo,pfs_zmin=request['pfs_zmin'],settings=settings,extended=True)
                  if c['cell_id']=='bin08-radial00-overlap')
        case['scope']='takada_density_corrected_tail'
        covcell,values=covariance_inputs(case,cosmo)
        write_json(directory/'covariance_inputs.json',{'configuration':asdict(config),
            'metadata':covcell.metadata,'fiducials':values,'engine_settings':engine.numerical_settings()})
        audit=audit_neighborhood(covcell,config,cosmo.params,engine,values,directory)
        if audit['status']!='passed':raise ValueError('corrected tail covariance neighborhood failed')
        cell,measurement=measure_cell(case,config,cosmo,engine,directory)
        if measurement['status']!='passed':raise ValueError('corrected tail derivatives failed')
        receipt.update(status='passed',covariance_status=audit['status'],mean_status=measurement['status'])
        if canonical(study_source(settings,engine))!=canonical(request['original_source'][grid]):
            raise ValueError('tail theory dependency changed after measurement')
        request_and_binding(output)
    except Exception as error:
        receipt.update(status='failed',error=str(error));(directory/'failure.txt').write_text(traceback.format_exc())
    receipt['elapsed_seconds']=time.monotonic()-started;write_json(directory/'worker.json',receipt)
    return receipt


def run_grid(output,grid):
    output=Path(output);request,binding=request_and_binding(output)
    dest=output/'inference'/grid;dest.mkdir(parents=True,exist_ok=False)
    settings=GRID_PROFILES[grid];profiles=corrected_profiles(request['pfs_zmax']);cosmo=FiducialCosmology()
    summary={'status':'running','grid':grid,'request_sha256':file_digest(output/'request.json'),
             'profiles':{},'outcomes':{},'nesting':{},'original_inputs_checked':{},'reuse_controls':{}}
    write_json(dest/'summary.json',summary)
    tail_record=None
    if 'C24' in profiles:
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--phase','tail','--grid',grid]
        (output/'logs').mkdir(exist_ok=True)
        with (output/'logs'/(grid+'-tail.log')).open('w') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        directory=output/'physical'/grid/'C24-tail';receipt=read_json(directory/'worker.json')
        tail_record={'status':receipt['status'],'exit_code':process.returncode,'files':artifact_hashes(directory)}
        write_json(output/'physical'/grid/'tail-execution.json',{'command':command,**tail_record})
        if process.returncode!=0 or receipt['status']!='passed':raise ValueError('new tail physical calculation failed')
    source_cache={};part_cache={};results={};templates={};meta={};source_refs={}
    for name in profiles:
        original='C22' if name=='C24' else name
        if original in templates:continue
        path=STUDY/'inference'/grid/'profiles'/original
        for file in ('manifest.json','arrays.npz','constraints.json'):binding.check(path/file)
        templates[original],meta[original]=load_result(path)
    full_template=templates['C22'];full_manifest=meta['C22']
    for cell,ref in zip(full_template.likelihood.cells,full_manifest['projections']):
        path=STUDY/'inference'/grid/'projections'/(ref+'.json')
        source_refs[cell.metadata['cell_id']]=read_json(binding.check(path))
    for name,profile in profiles.items():
        started=time.monotonic();print(grid+': '+name+' rebuilding covariance and information',flush=True)
        template=templates['C22' if name=='C24' else name]
        references=[]
        if profile.case in ('I','A','B'):
            # Existing DESI likelihood remains exactly the same. Recompute its inverse.
            result=infer_study(template.likelihood)
            require_close(result.covariance,template.covariance,'unchanged DESI full covariance',rtol=1e-10,atol=1e-12)
            summary['reuse_controls'][name]={'status':'passed','likelihood_exact':True,'source':str(STUDY/'inference'/grid/'profiles'/name)}
        else:
            parts=[];config=corrected_config(profile,settings);shells=study_shell_grid(config)
            expected=corrected_cells(profile,cosmo,pfs_zmin=request['pfs_zmin'],settings=settings)
            for case in expected:
                row=case['expected'];cellid=row['cell_id'];ref=None;source=None;anchor_ref=None
                key=hashlib.sha256(canonical({'row':row,'k':shells.centers.tolist(),'widths':shells.widths.tolist(),
                                              'case':profile.case})).hexdigest()
                if key not in part_cache:
                    if not row['empty_observation']:
                        oldid=cellid
                        if cellid=='bin01-radial00-overlap' and 'PFS-ELG' in row['active_populations']:
                            oldid='bin01-radial01-overlap'
                        ref=source_refs[oldid]
                        directory=Path(ref['source_directory']);record=ref['source_record']
                        if name=='C24' and cellid=='bin08-radial00-overlap':
                            directory=output/'physical'/grid/'C24-tail';record=tail_record
                        if str(directory) not in source_cache:
                            if directory.is_relative_to(STUDY) or str(directory).startswith(str(ROOT/'results/elf-grid-')):
                                for file in record['files']:binding.check(directory/file)
                            source_cache[str(directory)]=load_source_cell(directory,record)
                        original=source_cache[str(directory)]
                        anchor=None
                        if oldid!=cellid:
                            anchor_ref=source_refs[cellid];anchor_path=Path(anchor_ref['source_directory'])
                            if str(anchor_path) not in source_cache:
                                for file in anchor_ref['source_record']['files']:binding.check(anchor_path/file)
                                source_cache[str(anchor_path)]=load_source_cell(anchor_path,anchor_ref['source_record'])
                            anchor=source_cache[str(anchor_path)]
                        source=corrected_source(original,row,unchanged=anchor)
                    part=project_cell(source,profile,row,shells)
                    record={'expected':row,'profile':asdict(profile),'source_directory':str(directory) if source else None,
                        'source_record':record if source else None,'source_target':source['inputs']['metadata'] if source else None,
                        'mapping':part['mapping'],'unchanged_observation_anchor':anchor_ref,
                        'unchanged_observation_check':source.get('unchanged_observation') if source else None}
                    for field in ('derivative_check','neighborhood_check'):
                        if field in part:record[field]=part.pop(field)
                    part.pop('arrays',None);part.pop('analytic_zero_support',None)
                    (dest/'projections').mkdir(exist_ok=True)
                    write_json(dest/'projections'/(key+'.json'),jsonable(record));part_cache[key]=part
                parts.append(part_cache[key]);references.append(key)
            like=assemble_from_template(template.likelihood,parts,config=config);result=infer_study(like)
        results[name]=result;path=dest/'profiles'/name
        save_result(path,profile,settings,result,references)
        # Reload and re-invert the saved result, independent of the in-memory object.
        saved,_=load_corrected_result(path);again=infer_study(saved.likelihood)
        require_close(again.covariance,saved.covariance,'corrected saved full covariance',rtol=1e-10,atol=1e-12)
        require_close(again.growth_cov_marg,saved.growth_cov_marg,'corrected saved growth covariance',rtol=1e-10,atol=1e-12)
        summary['outcomes'][name]=constraint_row(result)
        summary['profiles'][name]={'status':'passed','files':artifact_hashes(path),'elapsed_seconds':time.monotonic()-started}
        write_json(dest/'summary.json',summary)
        print(grid+': '+name+' saved and reconstructed',flush=True)
    pairs=comparisons()+([('C21','C24'),('B16','C24')] if 'C24' in profiles else [])
    for a,b in pairs:
        if a=='I':continue
        if b=='C24':
            from scripts.pfs_bias_inference import check_cross_information
            summary['nesting'][a+'_to_'+b]=check_cross_information(results[a].cov_cosmo_marg,results[b].cov_cosmo_marg)
        else:summary['nesting'][a+'_to_'+b]=compare_nested_likelihoods(nesting_snapshot(results[a]),nesting_snapshot(results[b]))
    if any(r['status']!='passed' for r in summary['nesting'].values()):
        summary['status']='failed';write_json(dest/'summary.json',summary)
        raise ValueError('corrected information nesting failed')
    for source in source_cache.values():
        for path,digest in source['source_files'].items():
            if file_digest(path)!=digest:raise ValueError('physical source changed during corrected inference')
            summary['original_inputs_checked'][path]=digest
    request_and_binding(output)
    summary.update(status='passed',projection_files=artifact_hashes(dest/'projections'))
    write_json(dest/'summary.json',summary)
    state=read_json(output/'state.json');state['grids'][grid]={'status':'passed','summary_sha256':file_digest(dest/'summary.json')}
    state['status']='calculations_in_progress';write_json(output/'state.json',state)
    return {'status':'passed','grid':grid,'profiles':len(profiles)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','run','tail'),required=True)
    parser.add_argument('--grid',choices=tuple(GRID_PROFILES),default='shell304')
    parser.add_argument('--pfs-zmin',type=float,choices=(.6,.7))
    parser.add_argument('--pfs-zmax',type=float,choices=(2.2,2.4))
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.pfs_zmin is None or args.pfs_zmax is None:parser.error('prepare requires explicit PFS coverage')
        result=prepare(args.output,args.pfs_zmin,args.pfs_zmax)
    elif args.phase=='tail':result=measure_tail(args.output,args.grid)
    else:result=run_grid(args.output,args.grid)
    print(json.dumps(jsonable(result),indent=2),flush=True)
    return 0 if result['status']=='passed' or result['status']=='prepared' else 1


if __name__=='__main__':raise SystemExit(main())
