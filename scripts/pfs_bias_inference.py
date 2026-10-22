"""Global cosmological information from one changed PFS population.

At fixed density, the tested factor-one replacement algebra applies unchanged.
Population relations never enter nuisance priors or derivative constraints.
"""
from pathlib import Path
import time
import traceback

import numpy as np

from pfsfog.extended_artifacts import jsonable, read_json, require_close
from pfsfog.elf_survey_evidence import artifact_hashes, cell_directory, file_digest, write_json
from scripts.elf_study_data import project_cell
from scripts.elf_study_evidence import load_source_cell
from scripts.elf_study_report import all_profiles, load_result
from scripts.pfs_density_inference import (density_bin_covariance, full_density_covariance,
    window_cells, verify_reference)
from scripts.run_pfs_density_scan import constraint_row, InputBinding, read_sources

ROOT=Path(__file__).resolve().parents[1]


def bias_bin_covariance(result,replacements):
    output=density_bin_covariance(result,replacements,1.,'dimensionless')
    output.pop('factor');output.pop('prior_mode')
    output['density_and_prior_widths_fixed']=True
    return output


def full_bias_covariance(result,replacements):
    return full_density_covariance(result,replacements,1.,'dimensionless')


def check_cross_information(less,more):
    less=np.asarray(less);more=np.asarray(more)
    if less.shape!=(3,3) or more.shape!=(3,3) or not np.isfinite([less,more]).all():
        raise ValueError('invalid cross spectra covariance comparison')
    if np.any(np.diag(less)<=0):raise ValueError('invalid reference variances')
    scale=np.sqrt(np.outer(np.diag(less),np.diag(less)))
    delta=(less-more)/scale
    minimum=float(np.linalg.eigvalsh((delta+delta.T)/2).min())
    if minimum < -1e-9:
        raise ValueError('added cross spectra lost a cosmological combination')
    return {'status':'passed','minimum_scaled_eigenvalue':minimum,'tolerance':1e-9}


def inference_sources():
    files=('scripts/pfs_bias_inference.py','scripts/pfs_density_inference.py',
           'scripts/run_pfs_density_scan.py','scripts/pfs_density_response.py')
    return {str(ROOT/p):file_digest(ROOT/p) for p in files}


def reference_physical_check(source,original):
    if set(source['arrays'])!=set(original['arrays']):
        raise ValueError('reference physical array inventory differs')
    checks={}
    for name,old in original['arrays'].items():
        new=source['arrays'][name]
        require_close(new,old,'unchanged population '+name,rtol=1e-10,atol=1e-12)
        norm=float(np.linalg.norm(old))
        checks[name]={'exact':bool(np.array_equal(new,old)),
                     'relative_norm_difference':float(np.linalg.norm(new-old)/norm) if norm else 0.}
    return {'status':'passed','arrays':checks,'rtol':1e-10,'atol':1e-12}


def run_grid(root,grid):
    from scripts.run_pfs_bias_study import verify_physical,check_campaign,STUDY
    root=Path(root);campaign=check_campaign(root,grid)
    physical=verify_physical(root,grid)
    request=campaign['physical_requests'][grid]
    physical_root=root/'physical'/grid;raw=read_json(physical_root/'summary.json')
    before=inference_sources();binding=InputBinding(campaign['bindings'])
    destination=root/'inference'/grid;destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'physical-verification.json',physical)
    summary={'status':'running','grid':grid,'request_sha256':file_digest(root/'request.json'),
        'sources':before,'physical_summary_sha256':file_digest(physical_root/'summary.json'),
        'points':{},'reference_checks':{},'full_bias_grid_qualification':'pending'}
    write_json(destination/'summary.json',summary)
    profiles=all_profiles();results={};manifests={};cache={}
    for name in ('D','C','H21','C21','C22'):
        path=STUDY/'inference'/grid/'profiles'/name
        for filename in ('manifest.json','arrays.npz'):binding.check(path/filename)
        results[name],manifests[name]=load_result(path)
        summary['reference_checks'][name]=verify_reference(results[name])
    for expected,row in zip(request['expected'],raw['workers']):
        started=time.monotonic();key=row['key'];experiment=expected['experiment']
        path=destination/key;path.mkdir(exist_ok=False)
        point={'status':'running','key':key,'experiment':experiment,'grid':grid,'profiles':{},
            'physical_files':row['files'],'physical_directory':str(cell_directory(physical_root,key))}
        try:
            source=load_source_cell(cell_directory(physical_root,key),row)
            window=experiment['window']
            if experiment['ratio']==1.:
                cells=window_cells(results['C22'],window)
                old=read_sources(grid,results['C22'],manifests['C22'],cells,binding,cache)
                if len(old)!=1:raise ValueError('pilot requires one complete physical radial cell')
                point['reference_physical_check']=reference_physical_check(source,old[0][0])
            names=('D','C','C22') if window[1]<=1.6 else ('H21','C21','C22')
            for name in names:
                result=results[name];cells=window_cells(result,window)
                if len(cells)!=1:raise ValueError('pilot must replace complete single-cell population')
                part=project_cell(source,profiles[name],cells[0].metadata,result.grid)
                audit={k:part[k] for k in ('derivative_check','neighborhood_check','mapping')}
                write_json(path/(name+'-physical-check.json'),jsonable(audit))
                out=bias_bin_covariance(result,[part]);out['sigma']=constraint_row(out)
                # Cover both prescriptions, low/high-z, all observation choices.
                if grid=='shell304' and experiment['ratio']==1.2:
                    full=full_bias_covariance(result,[part])
                    require_close(out['cosmological_covariance'],full['cov_cosmo_marg'],
                                  'bias full-inverse control',rtol=1e-8,atol=1e-12)
                    require_close(full['F_prior_cosmo'],result.F_prior_cosmo,'unchanged cosmological priors')
                    require_close(full['F_prior_nuisance'],result.F_prior_nuisance,'unchanged nuisance priors')
                    out['full_inverse_control']={'status':'passed','validation':full['validation']}
                if experiment['ratio']==1.:
                    require_close(out['cosmological_covariance'],result.cov_cosmo_marg,
                                  'original bias covariance',rtol=1e-8,atol=1e-12)
                    out['original_bias_covariance_check']='passed'
                point['profiles'][name]=out
            point['cross_information_check']=check_cross_information(
                point['profiles'][names[0]]['cosmological_covariance'],
                point['profiles'][names[1]]['cosmological_covariance'])
            point['status']='passed'
        except Exception as error:
            point.update(status='failed',error=str(error),error_type=type(error).__name__)
            (path/'failure.txt').write_text(traceback.format_exc())
            if hasattr(error,'matrix'):np.save(path/'failed_matrix.npy',error.matrix)
        point['elapsed_seconds']=time.monotonic()-started
        write_json(path/'point.json',jsonable(point))
        summary['points'][key]={'status':point['status'],'files':artifact_hashes(path),
                               'elapsed_seconds':point['elapsed_seconds']}
        write_json(destination/'summary.json',jsonable(summary))
        print(f'{grid}: {key} {point["status"]} {point["elapsed_seconds"]:.1f}s',flush=True)
    binding.verify_all();check_campaign(root,grid)
    if before!=inference_sources():raise ValueError('inference sources changed during calculation')
    summary['status']='passed' if len(summary['points'])==15 and all(
        p['status']=='passed' for p in summary['points'].values()) else 'failed'
    write_json(destination/'summary.json',jsonable(summary))
    if summary['status']!='passed':raise ValueError('bias inference has failed populations')
    return summary
