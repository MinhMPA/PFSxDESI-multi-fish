#!/usr/bin/env python
"""Re-evaluate the accepted 35 bias populations at the corrected PFS density."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from copy import deepcopy
import json
import time
import traceback
import numpy as np
from pfsfog.elf_survey_evidence import file_digest,write_json,verify_files,cell_directory
from pfsfog.extended_artifacts import read_json,jsonable,require_close
from scripts.run_takada_revision import request_and_binding,corrected_source
from scripts.takada_study import corrected_profiles,load_corrected_result
from scripts.elf_study_data import project_cell
from scripts.elf_study_evidence import load_source_cell
from scripts.pfs_density_inference import window_cells,verify_reference
from scripts.pfs_bias_inference import bias_bin_covariance,full_bias_covariance,check_cross_information
from scripts.run_pfs_density_scan import InputBinding,constraint_row

BROAD=ROOT/'results/pfs-bias-broad-20261010-01'


def corrected_bias_window(original,zmin):
    window=tuple(original)
    if zmin not in (.6,.7) or window not in ((.7,.8),(1.4,1.6),(1.6,2.)):
        raise ValueError('unrecognized original bias population window')
    return (zmin,.8) if window==(.7,.8) else window


def run_grid(baseline,grid):
    baseline=Path(baseline).resolve();request,binding=request_and_binding(baseline)
    original=baseline/'inference'/grid;s=read_json(original/'summary.json')
    if s['status']!='passed':raise ValueError('corrected bias reference is incomplete')
    original_hash=file_digest(original/'summary.json');own_hash=file_digest(__file__)
    checkpoint=ROOT/'docs/revision/evidence/pfs-bias-broad-2026-10-10/BROAD_CHECKPOINT.json'
    bound=read_json(checkpoint)['files']
    files={str(ROOT/p):v['sha256'] if isinstance(v,dict) else v for p,v in bound.items()}
    auth=InputBinding(files)
    campaign=read_json(auth.check(BROAD/'request.json'))
    physical=BROAD/'stochastic-validation/physical'/grid
    cohort=read_json(auth.check(physical/'summary.json'))
    expected=campaign['physical_requests'][grid]['expected']
    if (cohort['status']!='passed' or len(expected)!=35
            or [r['key'] for r in cohort['workers']]!=[r['key'] for r in expected]):
        raise ValueError('incomplete original bias population inventory')
    dest=baseline/'bias'/grid;dest.mkdir(parents=True,exist_ok=False)
    terminal='C24' if request['pfs_zmax']==2.4 else 'C22'
    profiles=corrected_profiles(request['pfs_zmax']);results={};anchors={}
    summary={'status':'running','grid':grid,'baseline':str(baseline),
        'baseline_summary_sha256':original_hash,'source_sha256':own_hash,
        'bias_checkpoint_sha256':file_digest(checkpoint),'reference_checks':{},'points':{}}
    write_json(dest/'summary.json',summary)
    for name in ('D','C','H21','C21',terminal):
        path=original/'profiles'/name;problems=verify_files(path,s['profiles'][name]['files'])
        if problems:raise ValueError('; '.join(problems))
        results[name],manifest=load_corrected_result(path);summary['reference_checks'][name]=verify_reference(results[name])
        for cell,refid in zip(results[name].likelihood.cells,manifest['projections']):
            if cell.metadata['cell_id']!='bin01-radial00-overlap':continue
            ref=read_json(original/'projections'/(refid+'.json')).get('unchanged_observation_anchor')
            if ref:anchors[cell.metadata['cell_id']]=load_source_cell(ref['source_directory'],ref['source_record'])
    for item,record in zip(expected,cohort['workers']):
        started=time.monotonic();key=item['key'];experiment=deepcopy(item['experiment'])
        window=corrected_bias_window(experiment['window'],request['pfs_zmin'])
        experiment['original_window']=experiment['window'];experiment['window']=window
        path=dest/key;path.mkdir()
        point={'status':'running','key':key,'experiment':experiment,'profiles':{},'physical':{}}
        try:
            sourcepath=cell_directory(physical,key)
            for name in record['files']:auth.check(sourcepath/name)
            raw=load_source_cell(sourcepath,record)
            names=('D','C',terminal) if window[1]<=1.6 else ('H21','C21',terminal)
            for name in names:
                result=results[name];parts=[]
                for cell in window_cells(result,window):
                    source=corrected_source(raw,cell.metadata,unchanged=anchors.get(cell.metadata['cell_id']))
                    part=project_cell(source,profiles[name],cell.metadata,result.grid)
                    audit={k:part[k] for k in ('derivative_check','neighborhood_check','mapping')}
                    audit.update(physical_source=str(sourcepath),source_files=record['files'],
                                 density_transform=source['density_transform'])
                    filename=name+'-'+cell.metadata['cell_id']+'-physical.json'
                    write_json(path/filename,jsonable(audit));point['physical'][filename]=file_digest(path/filename)
                    parts.append(part)
                out=bias_bin_covariance(result,parts);out['sigma']=constraint_row(out)
                if experiment['ratio']==1.:
                    require_close(out['cosmological_covariance'],result.cov_cosmo_marg,
                                  'corrected unit-bias reference',rtol=1e-8,atol=1e-12)
                    out['unit_bias_control']='passed'
                if grid=='shell304' and (experiment['ratio']==2. or experiment['equal_to'] is not None):
                    full=full_bias_covariance(result,parts)
                    require_close(out['cosmological_covariance'],full['cov_cosmo_marg'],
                                  'corrected bias full inverse',rtol=1e-8,atol=1e-12)
                    out['full_inverse_control']={'status':'passed','validation':full['validation']}
                point['profiles'][name]=out
            point['cross_information']=check_cross_information(point['profiles'][names[0]]['cosmological_covariance'],
                                                               point['profiles'][names[1]]['cosmological_covariance'])
            point['status']='passed'
        except Exception as error:
            point.update(status='failed',error=str(error),error_type=type(error).__name__)
            (path/'failure.txt').write_text(traceback.format_exc())
            if hasattr(error,'matrix'):np.save(path/'failed_matrix.npy',error.matrix)
        for name in record['files']:auth.check(cell_directory(physical,key)/name)
        point['elapsed_seconds']=time.monotonic()-started
        write_json(path/'point.json',jsonable(point))
        summary['points'][key]={'status':point['status'],'point_sha256':file_digest(path/'point.json'),
                                'elapsed_seconds':point['elapsed_seconds']}
        write_json(dest/'summary.json',jsonable(summary));print(grid+': '+key+' '+point['status'],flush=True)
    if file_digest(original/'summary.json')!=original_hash or file_digest(__file__)!=own_hash:
        raise ValueError('bias reference or calculation changed')
    request_and_binding(baseline)
    summary['status']='passed' if all(p['status']=='passed' for p in summary['points'].values()) else 'completed_with_failures'
    write_json(dest/'summary.json',jsonable(summary))
    return {'status':summary['status'],'grid':grid,'populations':len(summary['points'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--grid',required=True)
    a=p.parse_args();out=run_grid(a.baseline,a.grid);print(json.dumps(out,indent=2));return 0 if out['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
