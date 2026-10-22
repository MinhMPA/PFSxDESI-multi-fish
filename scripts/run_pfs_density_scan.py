#!/usr/bin/env python
"""Source-bound PFS density experiments using the completed ELF physical arrays."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import hashlib
import json
import subprocess
import time
import traceback

import numpy as np

from pfsfog.extended_artifacts import jsonable, require_close
from scripts.elf_study_evidence import load_source_cell
from scripts.elf_study_profiles import paper_profiles, highz_profiles
from scripts.elf_study_report import load_result
from scripts.run_elf_studies import GRID_PROFILES, REFINEMENTS
from scripts.pfs_density_response import density_cell
from scripts.pfs_density_inference import (WINDOWS, PRIOR_MODES, window_cells,
    window_count, verify_reference, density_bin_covariance, full_density_covariance)

STUDY=ROOT/'results/elf-survey-studies-20261010-01'
FACTORS={'fiducial':1.,'ln_m010':float(np.exp(-.1)),'ln_p010':float(np.exp(.1)),
         'ln_m005':float(np.exp(-.05)),'ln_p005':float(np.exp(.05)),
         'times2':2.,'times5':5.,'times10':10.}
SOURCE_NAMES=('pfs_density_response.py','pfs_density_inference.py','run_pfs_density_scan.py')


def digest(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def fresh_output(path):
    Path(path).mkdir(parents=True,exist_ok=False)


class InputBinding:
    def __init__(self, expected):
        self.expected={str(Path(p).resolve()):h for p,h in expected.items()}

    def check(self, path):
        path=Path(path).resolve()
        if str(path) not in self.expected: raise ValueError('unbound density input: '+str(path))
        if digest(path)!=self.expected[str(path)]: raise ValueError('changed density input: '+str(path))
        return path

    def verify_all(self):
        for path in self.expected: self.check(path)


def window_profiles(i):
    if i<5: return ('D','C')
    if i<7: return ('H21','C21')
    return ('C22',)


def prepare(output):
    output=Path(output)
    files={}
    study_checkpoint=ROOT/'docs/revision/evidence/elf-survey-studies-2026-10-10/STUDY_CHECKPOINT.json'
    grid_checkpoint=ROOT/'docs/revision/evidence/elf-grid-convergence-2026-10-09/GRID_CHECKPOINT.json'
    frozen=json.loads(study_checkpoint.read_text())
    files.update({str(ROOT/p):v['sha256'] for p,v in frozen['files'].items()})
    files.update(json.loads(grid_checkpoint.read_text())['checkpoint_hashes'])
    for path in (study_checkpoint,grid_checkpoint,*(ROOT/'scripts'/n for n in SOURCE_NAMES),
                 ROOT/'docs/revision/specs/2026-10-10-pfs-density-response.md'):
        files[str(path)]=digest(path)
    InputBinding(files).verify_all()
    fresh_output(output)
    request={'schema':'pfs-density-scan-v1','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'windows':WINDOWS,'factors':FACTORS,'prior_modes':PRIOR_MODES,'grids':GRID_PROFILES,
             'refinements':REFINEMENTS,'study':str(STUDY),'bindings':files,
             'scope':'Density only at fixed population clustering; adopted PFS density provenance unresolved.'}
    write_json(output/'request.json',request)
    write_json(output/'state.json',{'status':'prepared','request_sha256':digest(output/'request.json'),'grids':{}})
    return {'status':'prepared','output':str(output),'bound_inputs':len(files)}


def load_request(output):
    request=json.loads((Path(output)/'request.json').read_text())
    state=json.loads((Path(output)/'state.json').read_text())
    if digest(Path(output)/'request.json')!=state['request_sha256']:raise ValueError('density request changed')
    if (request['schema']!='pfs-density-scan-v1' or request['factors']!=FACTORS
            or request['windows']!=list(map(list,WINDOWS)) or request['grids']!=GRID_PROFILES
            or request['prior_modes']!=list(PRIOR_MODES)):
        raise ValueError('density request differs from declared experiment')
    binding=InputBinding(request['bindings']);binding.verify_all()
    return request,binding


def read_sources(grid, result, manifest, cells, binding, cache):
    sources=[]
    for c in cells:
        idx=next(i for i,old in enumerate(result.likelihood.cells) if old.metadata['cell_id']==c.metadata['cell_id'])
        refpath=STUDY/'inference'/grid/'projections'/(manifest['projections'][idx]+'.json')
        record=json.loads(binding.check(refpath).read_text());directory=Path(record['source_directory'])
        if directory not in cache:
            for name in record['source_record']['files']: binding.check(directory/name)
            cache[directory]=load_source_cell(directory,record['source_record'])
        sources.append((cache[directory],c))
    return sources


def constraint_row(calculation):
    z=np.asarray(calculation['growth_z']);indices=np.flatnonzero(np.isclose(z,1.1,rtol=0,atol=1e-12))
    if len(indices)!=1:raise ValueError('missing unique headline growth redshift')
    covariance=np.asarray(calculation['cosmological_covariance'])
    growth=np.asarray(calculation['growth_covariance'])
    return dict(zip(('ln10_10_As','Mnu','Omegam','fsigma8_z1.1'),
                    [*np.sqrt(np.diag(covariance)),np.sqrt(growth[indices[0],indices[0]])]))


def run_grid(output, grid):
    output=Path(output);request,binding=load_request(output)
    if grid not in GRID_PROFILES:raise ValueError('unknown density grid')
    dest=output/grid;fresh_output(dest)
    summary={'status':'running','grid':grid,'request_sha256':digest(output/'request.json'),
             'points':{},'reference_checks':{},'settings':GRID_PROFILES[grid]}
    write_json(dest/'summary.json',summary)
    profiles={**paper_profiles(),**highz_profiles()};results={};manifests={};cache={}
    for name in ('B','D','C','H21','C21','C22'):
        result,manifest=load_result(STUDY/'inference'/grid/'profiles'/name)
        results[name]=result;manifests[name]=manifest
        summary['reference_checks'][name]=verify_reference(result)
        print(grid+': reference '+name+' recovered',flush=True)
    for iw,window in enumerate(WINDOWS):
        for label,factor in FACTORS.items():
            start=time.monotonic();key=f'w{iw:02d}-{label}'
            point={'status':'running','grid':grid,'window':list(window),'window_index':iw,
                   'factor':factor,'factor_label':label,'profiles':{},'physical':{},
                   'request_sha256':summary['request_sha256']}
            directory=dest/key;fresh_output(directory)
            try:
                for name in window_profiles(iw):
                    result=results[name];cells=window_cells(result,window);parts=[]
                    for source,old in read_sources(grid,result,manifests[name],cells,binding,cache):
                        part=density_cell(source,profiles[name],old.metadata,result.grid,factor)
                        parts.append(part)
                        audit={'derivatives':part['derivative_check'],'neighborhood':part['neighborhood_check'],
                               'transformation':part['density_transform']}
                        filename=name+'-'+old.metadata['cell_id']+'-physical.json'
                        write_json(directory/filename,audit)
                        point['physical'][filename]=digest(directory/filename)
                    point['profiles'][name]={'count_at_q1':window_count(result,window),'prior_modes':{}}
                    for mode in PRIOR_MODES:
                        calculated=density_bin_covariance(result,parts,factor,mode)
                        if factor==1.:
                            require_close(calculated['cosmological_covariance'],result.cov_cosmo_marg,
                                          'unit-density reference',rtol=1e-8,atol=1e-12)
                        # Independent full-matrix controls cover low/high-z and both prior conventions.
                        if (grid=='shell304' and iw in (0,5) and factor==2.):
                            full=full_density_covariance(result,parts,factor,mode)
                            require_close(calculated['cosmological_covariance'],full['cov_cosmo_marg'],
                                          'density full-inverse control',rtol=1e-8,atol=1e-12)
                            calculated['full_inverse_control']={'status':'passed','validation':full['validation']}
                        calculated['sigma']=constraint_row(calculated)
                        point['profiles'][name]['prior_modes'][mode]=calculated
                # Same total survey coverage for a fair ranking of all eight windows.
                # The final profile above includes every measured cross in this bin;
                # the changed physical cells are identical in the common C22 survey.
                if 'C22' not in point['profiles']:
                    point['profiles']['C22']={'count_at_q1':window_count(results['C22'],window),'prior_modes':{}}
                    for mode in PRIOR_MODES:
                        calculated=density_bin_covariance(results['C22'],parts,factor,mode)
                        if factor==1.:
                            require_close(calculated['cosmological_covariance'],results['C22'].cov_cosmo_marg,
                                          'common-C22 density reference',rtol=1e-8,atol=1e-12)
                        calculated['sigma']=constraint_row(calculated)
                        point['profiles']['C22']['prior_modes'][mode]=calculated
                point['status']='passed'
            except Exception as error:
                point.update(status='failed',error=str(error),error_type=type(error).__name__)
                (directory/'failure.txt').write_text(traceback.format_exc())
                if hasattr(error,'matrix'):np.save(directory/'failed_matrix.npy',error.matrix)
            point['elapsed_seconds']=time.monotonic()-start
            write_json(directory/'point.json',point)
            summary['points'][key]={'status':point['status'],'point_sha256':digest(directory/'point.json'),
                                   'elapsed_seconds':point['elapsed_seconds']}
            write_json(dest/'summary.json',summary)
            print(f'{grid}: {key} {point["status"]} {point["elapsed_seconds"]:.1f}s',flush=True)
    binding.verify_all()
    summary['status']='passed' if all(p['status']=='passed' for p in summary['points'].values()) else 'failed'
    write_json(dest/'summary.json',summary)
    state=json.loads((output/'state.json').read_text())
    state['grids'][grid]={'status':summary['status'],'summary_sha256':digest(dest/'summary.json')}
    state['status']='running' if len(state['grids'])<len(GRID_PROFILES) else 'numerical_runs_finished'
    write_json(output/'state.json',state)
    return {'status':summary['status'],'grid':grid,'points':len(summary['points'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','run'),required=True)
    p.add_argument('--grid',choices=tuple(GRID_PROFILES),default='shell304')
    args=p.parse_args()
    result=prepare(args.output) if args.phase=='prepare' else run_grid(args.output,args.grid)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']!='failed' else 1


if __name__=='__main__':raise SystemExit(main())
