"""Matched broad-population inference; the accepted shared-bin algebra is unchanged."""
from pathlib import Path
import time
import traceback

import numpy as np

from pfsfog.extended_artifacts import jsonable, read_json, require_close
from pfsfog.elf_survey_evidence import artifact_hashes, canonical, cell_directory, file_digest, verify_files, write_json
from scripts.elf_study_data import project_cell
from scripts.elf_study_evidence import load_source_cell
from scripts.elf_study_report import all_profiles, load_result
from scripts.pfs_density_inference import (density_bin_covariance, full_density_covariance,
    window_cells, verify_reference)
from scripts.run_pfs_density_scan import constraint_row, InputBinding, read_sources

ROOT=Path(__file__).resolve().parents[1]


from scripts.pfs_bias_inference import (
    bias_bin_covariance, full_bias_covariance, check_cross_information,
    reference_physical_check, inference_sources as pilot_inference_sources)
from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.run_pfs_bias_stochastic_validation import (
    physical_directory, validation_root, verify_validated_physical,
    validate_validation_execution, campaign_sources as validation_campaign_sources)


def full_inverse_required(grid,experiment):
    return grid=='shell304' and (experiment['ratio']==2. or experiment['equal_to'] is not None)


def require_grid_exit(root,phase,grid):
    if phase=='validated_physical':
        record=read_json(validation_root(root)/'execution.json')
        validate_validation_execution(record)
    else:record=read_json(Path(root)/(phase+'-execution.json'))
    rows=[r for r in record['controllers'] if r['grid']==grid]
    if len(rows)!=1 or rows[0]['actual_exit']!=0:
        raise ValueError('grid has no unique successful actual controller exit: '+phase+' '+grid)
    return rows[0]


def inference_sources():
    return {**pilot_inference_sources(),**validation_campaign_sources(),
            str(Path(__file__).resolve()):file_digest(Path(__file__))}


def verify_point(point,expected,grid):
    if (point['status']!='passed' or point['key']!=expected['key'] or point['grid']!=grid
            or canonical(point['experiment'])!=canonical(expected['experiment'])):
        raise ValueError('unrelated or mislabeled broad inference population')
    experiment=expected['experiment'];window=experiment['window']
    names=('D','C','C22') if window[1]<=1.6 else ('H21','C21','C22')
    if set(point['profiles'])!=set(names):raise ValueError('incomplete profile inventory')
    profiles=all_profiles()
    for name,out in point['profiles'].items():
        if out['density_and_prior_widths_fixed'] is not True or out['changed_prior_widths']!={}:
            raise ValueError('density or prior widths changed')
        cov=np.asarray(out['cosmological_covariance']);info=np.asarray(out['global_information'])
        if cov.shape!=(3,3) or info.shape!=(3,3) or not np.isfinite([cov,info]).all():
            raise ValueError('invalid cosmological matrices')
        if not np.allclose(cov,cov.T,rtol=1e-12,atol=0):raise ValueError('asymmetric covariance')
        if np.linalg.eigvalsh(cov).min()<=0:raise ValueError('nonpositive cosmological covariance')
        scale=np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
        require_close((info*scale)@(cov/scale),np.eye(3),'broad saved inverse',rtol=1e-10,atol=1e-10)
        z=np.asarray(out['growth_z']);j=np.asarray(out['growth_jacobian'])
        want_z=np.mean(profiles[name].nominal_bins,axis=1)
        if z.shape!=want_z.shape or j.shape!=(len(z),3) or not np.isfinite(j).all():
            raise ValueError('invalid growth labels or Jacobian')
        require_close(z,want_z,'broad growth labels',rtol=0,atol=1e-12)
        for suffix in ('','_nuisance_fixed'):
            c=np.asarray(out['cosmological_covariance'+suffix]);growth=np.asarray(out['growth_covariance'+suffix])
            if c.shape!=(3,3) or growth.shape!=(len(z),len(z)) or not np.isfinite(c).all():
                raise ValueError('invalid growth covariance shape')
            if np.linalg.eigvalsh(c).min()<=0:raise ValueError('nonpositive cosmological covariance')
            require_close(growth,j@c@j.T,'broad saved growth propagation',rtol=1e-10,atol=1e-12)
        actual=constraint_row(out)
        if set(actual)!=set(out['sigma']):raise ValueError('incorrect constraint labels')
        for parameter,value in actual.items():
            require_close(out['sigma'][parameter],value,'broad saved '+parameter,rtol=1e-12,atol=1e-14)
        if experiment['ratio']==1. and out.get('original_bias_covariance_check')!='passed':
            raise ValueError('missing original population global control')
        if full_inverse_required(grid,experiment) and out.get('full_inverse_control',{}).get('status')!='passed':
            raise ValueError('missing independent full-inverse control')
    if experiment['ratio']==1. and point.get('reference_physical_check',{}).get('status')!='passed':
        raise ValueError('missing original physical array control')
    cross=check_cross_information(point['profiles'][names[0]]['cosmological_covariance'],
                                  point['profiles'][names[1]]['cosmological_covariance'])
    return {'status':'passed','cross_information_check':cross}

def run_grid(root,grid):
    from scripts.run_pfs_bias_broad import check_campaign,STUDY
    root=Path(root);campaign=check_campaign(root,grid)
    require_grid_exit(root,'validated_physical',grid)
    physical=verify_validated_physical(root,grid)
    request=campaign['physical_requests'][grid]
    physical_root=physical_directory(root,grid);raw=read_json(physical_root/'summary.json')
    before=inference_sources();binding=InputBinding(campaign['bindings'])
    destination=root/'inference'/grid;destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'physical-verification.json',physical)
    summary={'status':'running','grid':grid,'request_sha256':file_digest(root/'request.json'),
        'sources':before,'physical_summary_sha256':file_digest(physical_root/'summary.json'),
        'points':{},'reference_checks':{},'full_bias_grid_qualification':False}
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
                if len(old)!=1:raise ValueError('broad study requires one complete physical radial cell')
                point['reference_physical_check']=reference_physical_check(source,old[0][0])
            names=('D','C','C22') if window[1]<=1.6 else ('H21','C21','C22')
            for name in names:
                result=results[name];cells=window_cells(result,window)
                if len(cells)!=1:raise ValueError('broad study must replace complete single-cell population')
                part=project_cell(source,profiles[name],cells[0].metadata,result.grid)
                audit={k:part[k] for k in ('derivative_check','neighborhood_check','mapping')}
                write_json(path/(name+'-physical-check.json'),jsonable(audit))
                out=bias_bin_covariance(result,[part]);out['sigma']=constraint_row(out)
                # Cover both prescriptions, low/high-z, all observation choices.
                if full_inverse_required(grid,experiment):
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
            point['independent_check']=verify_point(point,expected,grid)
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
    summary['status']='passed' if len(summary['points'])==35 and all(
        p['status']=='passed' for p in summary['points'].values()) else 'failed'
    write_json(destination/'summary.json',jsonable(summary))
    if summary['status']!='passed':raise ValueError('bias inference has failed populations')
    return summary


def verify_inference(root,grid):
    from scripts.run_pfs_bias_broad import check_campaign, STUDY
    root=Path(root);campaign=check_campaign(root,grid);require_grid_exit(root,'inference',grid)
    verify_validated_physical(root,grid)
    directory=root/'inference'/grid;summary=read_json(directory/'summary.json')
    request=campaign['physical_requests'][grid]
    physical_root=physical_directory(root,grid);raw=read_json(physical_root/'summary.json')
    if (summary['status']!='passed' or summary['grid']!=grid or summary['sources']!=inference_sources()
            or summary['request_sha256']!=file_digest(root/'request.json')
            or summary['physical_summary_sha256']!=file_digest(physical_root/'summary.json')
            or set(summary['points'])!={r['key'] for r in request['expected']}):
        raise ValueError('failed, changed or incomplete broad inference')
    references={name:load_result(STUDY/'inference'/grid/'profiles'/name)[0]
                for name in ('D','C','H21','C21','C22')}
    paths=[directory/'summary.json',physical_root/'summary.json',
           validation_root(root)/'request.json',validation_root(root)/'execution.json']
    inputs={str(p):file_digest(p) for p in paths};points={}
    for expected,physical in zip(request['expected'],raw['workers']):
        key=expected['key'];receipt=summary['points'][key];path=directory/key
        if receipt['status']!='passed':raise ValueError('failed broad inference population')
        problems=verify_files(path,receipt['files'])
        if problems:raise ValueError('; '.join(problems))
        point=read_json(path/'point.json');verify_point(point,expected,grid)
        if (point['physical_files']!=physical['files']
                or Path(point['physical_directory']).resolve()!=cell_directory(physical_root,key).resolve()):
            raise ValueError('inference uses a different physical population')
        for name,out in point['profiles'].items():
            require_close(out['growth_jacobian'],references[name].growth_jacobian,
                          'inherited global growth Jacobian',rtol=1e-12,atol=1e-14)
            if expected['experiment']['ratio']==1.:
                require_close(out['cosmological_covariance'],references[name].cov_cosmo_marg,
                              'original broad covariance',rtol=1e-8,atol=1e-12)
        inputs.update({str(path/name):digest for name,digest in receipt['files'].items()})
        points[key]=point
    return {'status':'passed','grid':grid,'points':points,'inputs':inputs}
