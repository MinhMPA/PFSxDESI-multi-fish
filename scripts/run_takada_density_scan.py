#!/usr/bin/env python
"""Density responses relative to a source-bound corrected Takada forecast."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import json
import time
import traceback
import numpy as np

from pfsfog.elf_survey_evidence import write_json,file_digest,verify_files,artifact_hashes
from pfsfog.extended_artifacts import jsonable,read_json,require_close
from scripts.run_takada_revision import request_and_binding,corrected_source
from scripts.takada_study import corrected_profiles,load_corrected_result
from scripts.elf_study_evidence import load_source_cell
from scripts.pfs_density_response import density_cell
from scripts.pfs_density_inference import (PRIOR_MODES,window_cells,window_count,
    verify_reference,density_bin_covariance,full_density_covariance)
from scripts.run_pfs_density_scan import FACTORS,constraint_row


def density_windows(zmin,zmax):
    if zmin not in (.6,.7) or zmax not in (2.2,2.4):
        raise ValueError('density scan requires a declared Table 2 coverage')
    return ((zmin,.8),(.8,1.),(1.,1.2),(1.2,1.4),(1.4,1.6),
            (1.6,2.),(2.,2.1),(2.1,zmax))


def reconstruct_reference_source(raw,record):
    ref=record.get('unchanged_observation_anchor')
    anchor=load_source_cell(ref['source_directory'],ref['source_record']) if ref else None
    changed=corrected_source(raw,record['expected'],unchanged=anchor)
    if changed['inputs']['metadata']!=record['source_target']:
        raise ValueError('corrected reference physical metadata changed')
    return changed


def run_grid(baseline,output,grid):
    baseline=Path(baseline).resolve();output=Path(output).resolve()
    request,binding=request_and_binding(baseline)
    original=baseline/'inference'/grid;summary=read_json(original/'summary.json')
    if summary['status']!='passed':raise ValueError('corrected reference is not complete')
    problems=verify_files(original/'projections',summary['projection_files'])
    if problems:raise ValueError('; '.join(problems))
    baseline_hash=file_digest(original/'summary.json');source_hash=file_digest(__file__)
    terminal='C24' if request['pfs_zmax']==2.4 else 'C22'
    windows=density_windows(request['pfs_zmin'],request['pfs_zmax'])
    profiles=corrected_profiles(request['pfs_zmax']);results={};manifests={};cache={}
    dest=output/grid;dest.mkdir(parents=True,exist_ok=False)
    state={'schema':'takada-density-variations-v1','status':'running','grid':grid,
        'baseline':str(baseline),'baseline_summary_sha256':baseline_hash,
        'source_file':str(Path(__file__).resolve()),'source_sha256':source_hash,
        'windows':windows,'factors':FACTORS,'prior_modes':PRIOR_MODES,'terminal_profile':terminal,
        'reference_checks':{},'points':{}}
    write_json(dest/'summary.json',jsonable(state))
    for name in ('B','D','C','H21','C21',terminal):
        path=original/'profiles'/name
        problems=verify_files(path,summary['profiles'][name]['files'])
        if problems:raise ValueError('; '.join(problems))
        results[name],manifests[name]=load_corrected_result(path)
        state['reference_checks'][name]=verify_reference(results[name])
    def sources_for(name,window):
        result=results[name];sources=[]
        for cell in window_cells(result,window):
            index=next(i for i,c in enumerate(result.likelihood.cells) if c.metadata['cell_id']==cell.metadata['cell_id'])
            key=manifests[name]['projections'][index]
            if key not in cache:
                record=read_json(original/'projections'/(key+'.json'))
                raw=load_source_cell(record['source_directory'],record['source_record'])
                cache[key]=reconstruct_reference_source(raw,record)
            sources.append((cache[key],cell))
        return sources
    for iw,window in enumerate(windows):
        names=('D','C') if iw<5 else ('H21','C21') if iw<7 else (terminal,)
        for label,factor in FACTORS.items():
            started=time.monotonic();key=f'w{iw:02d}-{label}';directory=dest/key;directory.mkdir()
            point={'status':'running','grid':grid,'window_index':iw,'window':window,
                'factor':factor,'factor_label':label,'profiles':{},'physical':{}}
            try:
                for name in names:
                    result=results[name];parts=[]
                    for source,cell in sources_for(name,window):
                        part=density_cell(source,profiles[name],cell.metadata,result.grid,factor);parts.append(part)
                        audit={k:part[k] for k in ('derivative_check','neighborhood_check','density_transform')}
                        filename=name+'-'+cell.metadata['cell_id']+'-physical.json'
                        write_json(directory/filename,jsonable(audit));point['physical'][filename]=file_digest(directory/filename)
                    point['profiles'][name]={'count_at_q1':window_count(result,window),'prior_modes':{}}
                    for mode in PRIOR_MODES:
                        out=density_bin_covariance(result,parts,factor,mode)
                        if factor==1.:
                            require_close(out['cosmological_covariance'],result.cov_cosmo_marg,'corrected q=1 covariance',rtol=1e-8,atol=1e-12)
                        if grid=='shell304' and iw in (0,5) and factor==2.:
                            full=full_density_covariance(result,parts,factor,mode)
                            require_close(out['cosmological_covariance'],full['cov_cosmo_marg'],'corrected density full inverse',rtol=1e-8,atol=1e-12)
                            out['full_inverse_control']={'status':'passed','validation':full['validation']}
                        out['sigma']=constraint_row(out);point['profiles'][name]['prior_modes'][mode]=out
                if terminal not in point['profiles']:
                    result=results[terminal]
                    point['profiles'][terminal]={'count_at_q1':window_count(result,window),'prior_modes':{}}
                    for mode in PRIOR_MODES:
                        out=density_bin_covariance(result,parts,factor,mode)
                        if factor==1.:
                            require_close(out['cosmological_covariance'],result.cov_cosmo_marg,'common Takada q=1 covariance',rtol=1e-8,atol=1e-12)
                        out['sigma']=constraint_row(out);point['profiles'][terminal]['prior_modes'][mode]=out
                point['status']='passed'
            except Exception as error:
                point.update(status='failed',error=str(error),error_type=type(error).__name__)
                (directory/'failure.txt').write_text(traceback.format_exc())
                if hasattr(error,'matrix'):np.save(directory/'failed_matrix.npy',error.matrix)
            point['elapsed_seconds']=time.monotonic()-started
            write_json(directory/'point.json',jsonable(point))
            state['points'][key]={'status':point['status'],'point_sha256':file_digest(directory/'point.json'),
                                  'elapsed_seconds':point['elapsed_seconds']}
            write_json(dest/'summary.json',jsonable(state))
            print(f'{grid}: {key} {point["status"]} {point["elapsed_seconds"]:.2f}s',flush=True)
    if file_digest(original/'summary.json')!=baseline_hash or file_digest(__file__)!=source_hash:
        raise ValueError('density baseline or source changed during calculation')
    for source in cache.values():
        for path,digest in source['source_files'].items():
            if file_digest(path)!=digest:raise ValueError('density physical input changed during calculation')
    request_and_binding(baseline)
    state['status']='passed' if len(state['points'])==64 and all(p['status']=='passed' for p in state['points'].values()) else 'completed_with_failures'
    write_json(dest/'summary.json',jsonable(state))
    return {'status':state['status'],'grid':grid,'points':len(state['points'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grid',required=True)
    a=p.parse_args();result=run_grid(a.baseline,a.output,a.grid);print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
