#!/usr/bin/env python
"""Repeat the seven accepted cosmological-prior treatments at Takada density."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import json
import time
import numpy as np
from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_survey_evidence import file_digest,write_json,artifact_hashes
from pfsfog.extended_artifacts import (read_json,jsonable,require_close,unpack_likelihood,
    pack_likelihood,PRECISION_FIELDS,GROWTH_FIELDS)
from pfsfog.elf_inference import infer_treatments
from pfsfog.cosmology_projection import project_cosmology_precision
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from scripts.run_takada_revision import request_and_binding,corrected_source,STUDY,GRID_PROFILES
from scripts.takada_study import corrected_profiles,corrected_cells,corrected_config,assemble_from_template,load_corrected_result
from scripts.takada_extended_checks import project_extended_cell
from scripts.elf_study_evidence import load_source_cell


def load_extended_template(path):
    path=Path(path);record=read_json(path/'manifest.json')
    if record['schema']!='elf-global-inference-case-v1' or file_digest(path/'arrays.npz')!=record['arrays_sha256']:
        raise ValueError('original extended likelihood binding differs')
    with np.load(path/'arrays.npz',allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in a.files if k.startswith('likelihood/')}
    return unpack_likelihood(record['likelihood'],arrays),record


def run_grid(baseline,grid):
    baseline=Path(baseline).resolve();request,binding=request_and_binding(baseline)
    dest=baseline/'extended'/grid;dest.mkdir(parents=True,exist_ok=False)
    own_hash=file_digest(__file__);support=ROOT/'scripts/takada_extended_checks.py';support_hash=file_digest(support)
    settings=GRID_PROFILES[grid];profiles=corrected_profiles(request['pfs_zmax'])
    cosmo=FiducialCosmology();summary={'status':'running','grid':grid,
        'request_sha256':file_digest(baseline/'request.json'),'source_sha256':own_hash,
        'support_source':{str(support):support_hash},'outcomes':{},'cases':{}}
    write_json(dest/'summary.json',summary)
    for case in ('B','C'):
        started=time.monotonic();old=ROOT/f'results/elf-grid-20261009-{grid}-inference-01/cases'/case
        for name in ('manifest.json','arrays.npz'):binding.check(old/name)
        template,_=load_extended_template(old);refs=[];checked={}
        if case=='B':like=template
        else:
            original=STUDY/'inference'/grid/'profiles'/'C22'
            original_meta=read_json(binding.check(original/'manifest.json'))
            source_refs={c['metadata']['cell_id']:ref for c,ref in
                         zip(original_meta['likelihood']['cells'],original_meta['projections'])}
            parts=[];cache={};profile=profiles['C'];config=corrected_config(profile,settings,extended=True)
            for expected in corrected_cells(profile,cosmo,pfs_zmin=request['pfs_zmin'],settings=settings,extended=True):
                row=expected['expected'];oldid=row['cell_id']
                if oldid=='bin01-radial00-overlap' and 'PFS-ELG' in row['active_populations']:
                    oldid='bin01-radial01-overlap'
                refpath=STUDY/'inference'/grid/'projections'/(source_refs[oldid]+'.json')
                ref=read_json(binding.check(refpath));directory=Path(ref['source_directory'])
                if str(directory) not in cache:
                    for name in ref['source_record']['files']:binding.check(directory/name)
                    cache[str(directory)]=load_source_cell(directory,ref['source_record'])
                anchor=None
                if oldid!=row['cell_id']:
                    apath=STUDY/'inference'/grid/'projections'/(source_refs[row['cell_id']]+'.json')
                    aref=read_json(binding.check(apath));adirectory=Path(aref['source_directory'])
                    if str(adirectory) not in cache:
                        for name in aref['source_record']['files']:binding.check(adirectory/name)
                        cache[str(adirectory)]=load_source_cell(adirectory,aref['source_record'])
                    anchor=cache[str(adirectory)]
                source=corrected_source(cache[str(directory)],row,unchanged=anchor)
                part=project_extended_cell(source,profile,row,template.grid)
                audit={k:part[k] for k in ('derivative_check','neighborhood_check','mapping')}
                file=dest/(case+'-'+row['cell_id']+'-physical.json');write_json(file,jsonable(audit))
                checked[file.name]=file_digest(file);parts.append(part)
                refs.append({'source_directory':str(directory),'source_record':ref['source_record'],'expected':row})
            like=assemble_from_template(template,parts,config=config)
            for source in cache.values():
                for path,digest in source['source_files'].items():
                    if file_digest(path)!=digest:raise ValueError('extended physical input changed during calculation')
        print(grid+': extended '+case+' applying the seven prior treatments',flush=True)
        results,failures=infer_treatments(like)
        arrays={};manifest={'likelihood':pack_likelihood(like,arrays),'treatments':{},'failures':jsonable(failures),
            'physical_checks':checked,'sources':refs,'original_extended':str(old)}
        summary['outcomes'][case]={}
        for name,result in results.items():
            precision=project_cosmology_precision(like,name,fisher_solver=DECIMAL_FISHER_SOLVER)
            for field in PRECISION_FIELDS:
                require_close(getattr(result,field),precision[field],'reconstructed extended '+field,rtol=1e-10,atol=1e-12)
            for field in (*PRECISION_FIELDS,*GROWTH_FIELDS):arrays[name+'/'+field]=getattr(result,field)
            manifest['treatments'][name]={'free_cosmology':result.free_cosmology,'validation':result.validation,
                'active_indices':result.active_indices,'active_parameter_names':result.active_parameter_names}
            free=list(result.free_cosmology);cn=like.cosmo_names
            row={'status':'passed','free_cosmology':free,'growth_z':result.growth_z.tolist()}
            for kind,suffix in [('marginalized','marg'),('nuisance_fixed','nuisance_fixed')]:
                covariance=getattr(result,'cov_cosmo_'+suffix);growth=getattr(result,'growth_cov_'+suffix)
                row[kind]={'cosmology':{n:float(np.sqrt(covariance[cn.index(n),cn.index(n)])) for n in free},
                           'growth':np.sqrt(np.diag(growth)).tolist()}
            summary['outcomes'][case][name]=row
        for name,failed in failures.items():summary['outcomes'][case][name]=jsonable(failed)
        if not {'conditional3','marginal6'}<=set(results):raise ValueError('required corrected cosmology treatment failed')
        native,_=load_corrected_result(baseline/'inference'/grid/'profiles'/case)
        require_close(results['conditional3'].cov_cosmo_marg[:3,:3],native.cov_cosmo_marg,
                      'corrected six-to-three-coordinate control',rtol=1e-8,atol=1e-12)
        manifest['conditional3_control']='passed'
        path=dest/case;path.mkdir();np.savez_compressed(path/'arrays.npz',**arrays)
        manifest['arrays_sha256']=file_digest(path/'arrays.npz');write_json(path/'manifest.json',jsonable(manifest))
        summary['cases'][case]={'status':'passed','files':artifact_hashes(path),'elapsed_seconds':time.monotonic()-started}
        write_json(dest/'summary.json',jsonable(summary))
    from scripts.infer_elf_survey import constraint_ratios
    summary['ratios']=constraint_ratios(summary['outcomes'])
    if file_digest(__file__)!=own_hash:raise ValueError('extended calculation changed during run')
    if file_digest(support)!=support_hash:raise ValueError('extended physical checker changed during run')
    request_and_binding(baseline)
    summary['status']='passed';write_json(dest/'summary.json',jsonable(summary))
    return {'status':'passed','grid':grid,'cases':2,'treatments_per_case':7}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--grid',required=True)
    a=p.parse_args();out=run_grid(a.baseline,a.grid);print(json.dumps(out,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
