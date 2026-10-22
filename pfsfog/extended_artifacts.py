"""Numerically checked, immutable extended-cosmology payloads.

These payloads establish local numerical consistency. A complete study additionally
requires the externally pinned qualification and supervisor selection produced by
run_extended_cosmology. The frozen checker and schema are deliberately untouched.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from .cosmology_projection import COSMOLOGY_TREATMENTS, CosmologyTreatmentResult, project_cosmology_precision
from .cosmology_space import make_cosmology_spec
from .ap_numerics import profile_fields, validate_profile_fields
from .shell_numerics import shell_profile_from_record, shell_profile_fields, shell_fields, validate_shell_fields
from .extended_validation import validate_extended_response
from .fisher_core import fisher_from_shells
from .revision_forecast import RevisionConfig, RevisionLikelihood, CellLikelihood, ShellGrid, make_shell_grid
from .fisher_precision import fisher_solver_record, solver_from_record, common_fisher_solver

SCHEMA = 'pfs-desi-cosmology-v1'
IDENTITY_KEYS = ('project_head','source_content_hash','backend_snapshot','network_hash',
                 'survey_input_hash','runtime_hash')
PRECISION_FIELDS = ('F_like','F_prior_cosmo','F_prior_nuisance','F_total','covariance',
                    'cov_cosmo_marg','cov_cosmo_nuisance_fixed')
GROWTH_FIELDS = ('growth_z','growth_fiducial','growth_jacobian','growth_cov_marg',
                 'growth_cov_nuisance_fixed')


def jsonable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [jsonable(v) for v in value]
    return value


def canonical(value):
    return json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def hash_value(value): return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value): Path(path).write_bytes(canonical(value)+b'\n')


def read_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError(f'duplicate JSON key {key}')
            result[key]=value
        return result
    def invalid(value): raise ValueError(f'nonfinite JSON {value}')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def require_close(actual,expected,label,*,rtol=1e-10,atol=1e-12):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if (actual.shape!=expected.shape or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(expected))
            or not np.allclose(actual,expected,rtol=rtol,atol=atol)):
        raise ValueError(f'{label}: shape, finite value or numerical identity mismatch')


def identity_provenance(provenance):
    if any(not isinstance(provenance.get(k),str) or not provenance[k] for k in IDENTITY_KEYS):
        raise ValueError('missing source/input/runtime identity')
    for key in IDENTITY_KEYS[1:]:
        value=provenance[key]
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'invalid provenance digest {key}')
    return {**{k:provenance[k] for k in IDENTITY_KEYS},
            **shell_profile_fields(shell_profile_from_record(provenance))}


def physical_likelihood_id(likelihood,provenance):
    config=asdict(likelihood.config); config.pop('nuisance_factor')
    return hash_value({'configuration':config,'reference':likelihood.reference_cosmology,
                       'model':likelihood.model,'provenance':identity_provenance(provenance)})


def pack_likelihood(like,arrays,prefix='likelihood'):
    for name in ('F_like','fiducial'):
        arrays[f'{prefix}/{name}']=np.asarray(getattr(like,name))
    arrays[f'{prefix}/bin_F_like']=np.asarray(like.bin_F_like)
    for name in ('edges','centers','widths'): arrays[f'{prefix}/grid_{name}']=getattr(like.grid,name)
    cells=[]
    for i,cell in enumerate(like.cells):
        for field in ('F_like','derivatives','covariance','fiducial_mean'):
            arrays[f'{prefix}/cell{i}/{field}']=np.asarray(getattr(cell,field))
        cells.append({'names':cell.names,'metadata':cell.metadata,'validation':cell.validation})
    return jsonable({**profile_fields(like.config),'configuration':asdict(like.config),'names':like.names,
        'parameters':like.parameters,'cells':cells,'reference_cosmology':like.reference_cosmology,
        'model':like.model,'validation':like.validation})


def unpack_likelihood(meta,arrays,prefix='likelihood'):
    config=dict(meta['configuration']); config['nominal_bins']=tuple(map(tuple,config['nominal_bins']))
    configured = RevisionConfig(**config)
    validate_profile_fields(meta, configured)
    cells=[CellLikelihood(*(arrays[f'{prefix}/cell{i}/F_like'],c['names'],c['metadata'],
            c['validation'],arrays[f'{prefix}/cell{i}/derivatives'],
            arrays[f'{prefix}/cell{i}/covariance'],arrays[f'{prefix}/cell{i}/fiducial_mean']))
           for i,c in enumerate(meta['cells'])]
    for cell in cells:
        cell.metadata=dict(cell.metadata)
        if 'retained_observations' in cell.metadata:
            cell.metadata['retained_observations']=[(tuple(pair),ell) for pair,ell in cell.metadata['retained_observations']]
    grid=ShellGrid(*(arrays[f'{prefix}/grid_{f}'] for f in ('edges','centers','widths')))
    return RevisionLikelihood(configured,arrays[f'{prefix}/F_like'],meta['names'],
        arrays[f'{prefix}/fiducial'],meta['parameters'],cells,list(arrays[f'{prefix}/bin_F_like']),
        grid,meta['reference_cosmology'],meta['model'],meta['validation'])


def validate_likelihood(like):
    """Rebuild local, bin and global information from every original D/C."""
    names=like.names; n=len(names)
    if len(set(names))!=n or [p['name'] for p in like.parameters]!=names:
        raise ValueError('parameter ordering mismatch')
    require_close(like.fiducial,[p['fiducial'] for p in like.parameters],'fiducial metadata')
    if any('unit' not in p for p in like.parameters): raise ValueError('parameter unit metadata missing')
    expected_grid=make_shell_grid(like.config.kmin,like.config.kmax,like.config.dk)
    for field in ('edges','centers','widths'):
        if not np.array_equal(getattr(like.grid,field),getattr(expected_grid,field)):
            raise ValueError('observed grid differs from configuration')
    bins=[np.zeros((n,n)) for _ in like.config.nominal_bins]
    if not like.cells: raise ValueError('missing retained cell response arrays')
    ids=[c.metadata['cell_id'] for c in like.cells]
    if len(set(ids))!=len(ids): raise ValueError('duplicate cell identity')
    for cell in like.cells:
        if cell.metadata['retained_pairs']:
            if any(cell.names.count(c)!=1 for c in like.cosmo_names):
                raise ValueError('omitted cosmology derivative column')
            expected=(len(like.grid.centers),3*len(cell.metadata['retained_pairs']),len(cell.names))
            if cell.derivatives.shape!=expected: raise ValueError('derivative column or shell shape mismatch')
            require_close(cell.fiducial_mean,np.asarray(cell.fiducial_mean).reshape(expected[:2]),'mean')
            fisher=fisher_from_shells(cell.derivatives,cell.covariance,like.grid.centers)
            require_close(cell.F_like,fisher,'cell Fisher')
        elif cell.names or np.size(cell.F_like) or np.size(cell.derivatives) or np.size(cell.covariance):
            raise ValueError('empty cell has consumable arrays')
        local=[names.index(name) for name in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(local,local)]+=cell.F_like
    require_close(like.bin_F_like,bins,'bin Fisher')
    require_close(like.F_like,np.sum(bins,axis=0),'global Fisher')


def units(like):
    return {'h_ref':like.reference_cosmology['h'],'unit_convention':'fixed_h_ref',
            'sigma8_radius':'8/h_trial Mpc','sigma8_radius_ref':'8*h_ref/h_trial',
            'k':'h_ref/Mpc','power':'(Mpc/h_ref)^3','growth':'dimensionless f*sigma8'}


def treatment_metadata(like,name,precision,physical_id):
    free=list(precision['free_cosmology'])
    record={'treatment':name,'free_cosmology':free,
        'fixed_cosmology':[n for n in like.cosmo_names if n not in free],
        'active_parameter_names':list(precision['active_parameter_names']),
        'active_indices':list(precision['active_indices']),'prior_center':'fiducial',
        'prior_identity':hash_value({f:precision[f] for f in ('F_prior_cosmo','F_prior_nuisance')}),
        'physical_likelihood_id':physical_id,'cosmology_columns':list(like.cosmo_names),
        'fisher_solver':fisher_solver_record(precision['fisher_solver'])}
    record['treatment_id']=hash_value(record)
    return record


def _validate_payload(manifest,arrays,evidence,*,intrinsic_root=None):
    if manifest.get('schema_version')!=SCHEMA or manifest.get('status')!='numerically_validated':
        raise ValueError('unsupported extended schema or status')
    solver=solver_from_record(manifest)
    if solver_from_record(manifest['provenance'])!=solver:
        raise ValueError('artifact/provenance Fisher solver differs')
    expected_arrays={}
    like=unpack_likelihood(manifest['likelihood'],arrays)
    if like.config.ap_interpolation_nk>16384:
        raise ValueError('AP refinement controls use the output-only saved-control path')
    if like.config.ap_interpolation_nk==16384:
        if manifest.get('intrinsic_sidecars') != evidence.get('response',{}).get('intrinsic_geometry'):
            raise ValueError('intrinsic-table manifest differs from raw response evidence')
    elif 'intrinsic_sidecars' in manifest or 'intrinsic_geometry' in evidence.get('response',{}):
        raise ValueError('legacy profile cannot claim intrinsic-table evidence')
    pack_likelihood(like,expected_arrays)
    expected_keys=set(expected_arrays)|{name+'/'+field for name in manifest['treatments'] for field in (*PRECISION_FIELDS,*GROWTH_FIELDS)}
    if set(arrays)!=expected_keys: raise ValueError('array inventory differs from declared likelihood/treatments')
    if any(v.dtype!=np.dtype('float64') for v in arrays.values()): raise ValueError('scientific arrays require float64')
    space=make_cosmology_spec(like.reference_cosmology,extended=True,
                             ap_nk=like.config.ap_interpolation_nk)
    for record in (manifest,manifest['provenance'],evidence):
        validate_profile_fields(record,like.config)
    if (like.config.cosmology_basis!='extended6' or manifest.get('n_cosmo')!=6
            or manifest.get('cosmology_columns')!=list(space.names)
            or manifest.get('full_parameter_names')!=like.names or manifest.get('units')!=units(like)):
        raise ValueError('full names, units or six-coordinate basis mismatch')
    validate_shell_fields(manifest,manifest['provenance'],like.config)
    validate_shell_fields(manifest['likelihood'],manifest['provenance'],like.config)
    pid=physical_likelihood_id(like,manifest['provenance'])
    if manifest['physical_likelihood_id']!=pid: raise ValueError('physical source/configuration identity mismatch')
    if evidence.get('response_identity')!=identity_provenance(manifest['provenance']):
        raise ValueError('independent response source identity mismatch')
    validate_likelihood(like)
    response=validate_extended_response(evidence['response'],like,intrinsic_root=intrinsic_root)
    if manifest['raw_gate_hashes']!={k:hash_value(v) for k,v in evidence.items()}:
        raise ValueError('raw gate evidence hash mismatch')
    failed=evidence.get('failed_treatments',{})
    if manifest['failed_treatments']!=failed: raise ValueError('failed treatment records differ')
    successful=manifest['treatments']
    if (set(successful)&set(failed) or set(successful)|set(failed)!=set(COSMOLOGY_TREATMENTS)
            or not {'conditional3','marginal6'}<=set(successful)):
        raise ValueError('seven treatments including required controls must be accounted for')
    for name,record in failed.items():
        if record.get('status')!='failed' or not record.get('error') or set(record)-{'status','error','raw'}:
            raise ValueError('failed record must not carry consumable covariance or complete status')
        if any(key.startswith(name+'/') for key in arrays): raise ValueError('failed covariance is consumable')
    growth=evidence['response']['growth']; rows=[]
    for name in COSMOLOGY_TREATMENTS:
        if name not in successful: continue
        record=successful[name]
        if solver_from_record(record)!=solver: raise ValueError('treatment Fisher solver differs')
        precision=project_cosmology_precision(like,name,fisher_solver=solver)
        if record!=treatment_metadata(like,name,precision,pid): raise ValueError('treatment names/prior identity mismatch')
        for field in PRECISION_FIELDS: require_close(arrays[f'{name}/{field}'],precision[field],name+'/'+field)
        for field,reference in (('growth_z','redshifts'),('growth_fiducial','fiducial'),('growth_jacobian','ad')):
            require_close(arrays[f'{name}/{field}'],growth[reference],name+'/'+field,rtol=1e-10,atol=1e-12)
        if not np.array_equal(arrays[f'{name}/growth_z'],growth['redshifts']):
            raise ValueError('growth redshifts must match exactly')
        jac=arrays[f'{name}/growth_jacobian']
        for suffix in ('marg','nuisance_fixed'):
            require_close(arrays[f'{name}/growth_cov_{suffix}'],jac@precision[f'cov_cosmo_{suffix}']@jac.T,
                          name+'/full growth propagation')
        free=[space.names.index(n) for n in record['free_cosmology']]
        native=np.sqrt(np.diag(precision['cov_cosmo_marg']))
        fixed=np.sqrt(np.diag(precision['cov_cosmo_nuisance_fixed']))
        g=np.sqrt(np.diag(arrays[f'{name}/growth_cov_marg']))
        gf=np.sqrt(np.diag(arrays[f'{name}/growth_cov_nuisance_fixed']))
        covariance=precision['cov_cosmo_marg'][np.ix_(free,free)]
        scale=native[free]
        rows.append({'treatment':name,'free_cosmology':record['free_cosmology'],
            'native_sigma':dict(zip(record['free_cosmology'],native[free].tolist())),
            'growth_sigma':g.tolist(),'growth_z':arrays[f'{name}/growth_z'].tolist(),
            'headline_z1p1':float(g[list(arrays[f'{name}/growth_z']).index(1.1)]),
            'native_nuisance_penalty':(native[free]/fixed[free]).tolist(),
            'growth_nuisance_penalty':(g/gf).tolist(),
            'native_correlation':(covariance/scale[:,None]/scale[None,:]).tolist(),
            'growth_correlation':(arrays[f'{name}/growth_cov_marg']/g[:,None]/g[None,:]).tolist()})
    return jsonable(rows),{'response':response,'numerical_reload':'passed'}


def write_extended_artifacts(likelihood:RevisionLikelihood,treatments:dict[str,CosmologyTreatmentResult],
                             root:Path,provenance:dict,evidence:dict,*,intrinsic_root=None)->Path:
    root=Path(root)
    if (root/'manifest.json').exists(): raise FileExistsError('refusing existing run directory')
    arrays={}; meta=pack_likelihood(likelihood,arrays)
    solver=common_fisher_solver(*treatments.values())
    provenance={**provenance,'fisher_solver':provenance.get('fisher_solver',fisher_solver_record(solver))}
    if solver_from_record(provenance)!=solver: raise ValueError('result/provenance Fisher solver differs')
    meta.update(shell_fields(provenance,likelihood.config))
    pid=physical_likelihood_id(likelihood,provenance)
    manifest={**profile_fields(likelihood.config),**shell_fields(provenance,likelihood.config),
        'fisher_solver':fisher_solver_record(solver),
        'schema_version':SCHEMA,'status':'numerically_validated','n_cosmo':6,
        'cosmology_columns':list(likelihood.cosmo_names),'full_parameter_names':likelihood.names,
        'units':units(likelihood),'physical_likelihood_id':pid,'likelihood':meta,
        'provenance':jsonable(provenance),'treatments':{},
        'failed_treatments':jsonable(evidence.get('failed_treatments',{})),
        'raw_gate_hashes':{k:hash_value(v) for k,v in evidence.items()}}
    if likelihood.config.ap_interpolation_nk>=16384:
        manifest['intrinsic_sidecars']=jsonable(evidence.get('response',{}).get('intrinsic_geometry'))
    for name,result in treatments.items():
        if result.likelihood is not likelihood or result.treatment!=name: raise ValueError('treatment attached to wrong likelihood')
        precision={field:getattr(result,field) for field in (*PRECISION_FIELDS,'free_cosmology',
                                                        'active_parameter_names','active_indices','fisher_solver')}
        manifest['treatments'][name]=treatment_metadata(likelihood,name,precision,pid)
        for field in (*PRECISION_FIELDS,*GROWTH_FIELDS): arrays[f'{name}/{field}']=getattr(result,field)
    rows,validation=_validate_payload(manifest,arrays,jsonable(evidence),intrinsic_root=intrinsic_root)
    path=root/f'{likelihood.config.case}6-{pid[:12]}-{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True,exist_ok=False)
    np.savez(path/'matrices.npz',**arrays)
    write_json(path/'evidence.json',evidence); write_json(path/'summary.json',rows)
    write_json(path/'validation.json',validation)
    manifest['files']={name:file_hash(path/name) for name in ('matrices.npz','evidence.json','summary.json','validation.json')}
    for entry in manifest.get('intrinsic_sidecars',[]):
        from .revision_validation_ap_intrinsic import _sidecar_path
        target=_sidecar_path(path,entry['path']); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(_sidecar_path(intrinsic_root,entry['path']),target)
        if file_hash(target)!=entry['sha256']:
            raise ValueError('intrinsic sidecar changed during artifact copy')
        manifest['files'][entry['path']]=entry['sha256']
    write_json(path/'manifest.json',manifest)
    load_extended_artifacts(path,manifest_sha256=file_hash(path/'manifest.json'))
    return path


def load_extended_artifacts(path:Path,*,manifest_sha256:str)->tuple[dict,dict,list,dict]:
    path=Path(path)
    if file_hash(path/'manifest.json')!=manifest_sha256: raise ValueError('external manifest hash mismatch')
    manifest=read_json(path/'manifest.json')
    expected_files={'matrices.npz','evidence.json','summary.json','validation.json'}
    if 'intrinsic_sidecars' in manifest:
        from .revision_validation_ap_intrinsic import _sidecar_path
        entries=manifest['intrinsic_sidecars']
        if not isinstance(entries,list): raise ValueError('intrinsic sidecar inventory must be an ordered list')
        sidecar_paths=[]
        for entry in entries:
            if not isinstance(entry,dict) or not isinstance(entry.get('path'),str):
                raise ValueError('incomplete intrinsic sidecar declaration')
            _sidecar_path(path,entry['path'])
            sidecar_paths.append(entry['path'])
        if len(set(sidecar_paths))!=len(sidecar_paths): raise ValueError('duplicate intrinsic sidecar inventory')
        expected_files.update(sidecar_paths)
        actual_files={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() or p.is_symlink()}
        if actual_files!=expected_files|{'manifest.json'}: raise ValueError('actual payload file inventory mismatch')
        if any((path/name).is_symlink() for name in expected_files):
            raise ValueError('payload symlinks are not accepted')
    if set(manifest['files'])!=expected_files:
        raise ValueError('payload file inventory mismatch')
    for name,digest in manifest['files'].items():
        if file_hash(path/name)!=digest: raise ValueError('payload bytes changed: '+name)
    with np.load(path/'matrices.npz',allow_pickle=False) as saved:
        arrays={k:saved[k] for k in saved.files}
    rows,validation=_validate_payload(manifest,arrays,read_json(path/'evidence.json'),intrinsic_root=path)
    if canonical(rows)!=canonical(read_json(path/'summary.json')): raise ValueError('summary differs from arrays')
    if canonical(validation)!=canonical(read_json(path/'validation.json')): raise ValueError('validation differs from raw evidence')
    return manifest,arrays,rows,validation
