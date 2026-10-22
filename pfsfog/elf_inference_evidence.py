"""Saved ELF inference arithmetic, with explicit independent native3 evidence."""
from pathlib import Path

import numpy as np

from .cosmology_projection import (COSMOLOGY_TREATMENTS, CosmologyTreatmentResult,
    prepare_cosmology_precision, project_cosmology_precision)
from .elf_survey import elf_cosmology_spec
from .elf_survey_evidence import file_digest, write_json
from .extended_artifacts import (PRECISION_FIELDS, GROWTH_FIELDS, canonical, jsonable,
    pack_likelihood, unpack_likelihood, validate_likelihood, read_json, require_close)
from .extended_validation import _validate_growth, validate_conditional_bridge, validate_treatment_cohort
from .fisher_precision import DECIMAL_FISHER_SOLVER, fisher_solver_record
from .revision_forecast import PriorResult, project_baseline_precision

SCHEMA = 'elf-global-inference-case-v1'
REQUIRED = {'conditional3','marginal6'}


def _result_record(result, arrays, prefix):
    for field in (*PRECISION_FIELDS,*GROWTH_FIELDS):
        arrays[prefix+'/'+field] = np.asarray(getattr(result,field))
    record = {'validation':result.validation,'fisher_solver':fisher_solver_record(result.fisher_solver)}
    if hasattr(result,'treatment'):
        record.update(treatment=result.treatment,free_cosmology=result.free_cosmology,
            active_parameter_names=result.active_parameter_names,active_indices=result.active_indices)
    return jsonable(record)


def write_case(path, likelihood, results, failures, native, native_result, growth, provenance):
    """Save all outcomes before verification can reject a required control."""
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    arrays = {}
    record = {'schema':SCHEMA,'full_forecast_qualification':'pending',
        'provenance':provenance,'likelihood':pack_likelihood(likelihood,arrays),
        'native':None if native is None else pack_likelihood(native,arrays,'native'),
        'treatments':{n:_result_record(r,arrays,n) for n,r in results.items()},
        'failures':jsonable(failures),
        'native_result':None if native_result is None else _result_record(native_result,arrays,'native_result'),
        'fisher_solver':fisher_solver_record(DECIMAL_FISHER_SOLVER)}
    np.savez_compressed(path/'arrays.npz',**arrays)
    write_json(path/'growth.json',jsonable(growth))
    record.update(arrays_sha256=file_digest(path/'arrays.npz'),growth_sha256=file_digest(path/'growth.json'))
    write_json(path/'manifest.json',jsonable(record))
    return record


def _check_native_identity(provenance, case):
    native = provenance.get('native_construction',{})
    if (provenance.get('case') != case or native.get('method') != 'build_revision_likelihood'
            or native.get('cosmology_basis') != 'baseline3' or native.get('case') != case
            or not provenance.get('source_identity_hash')
            or native.get('source_identity_hash') != provenance['source_identity_hash']):
        raise ValueError('independent native3 construction/source identity differs')


def _growth_fields(arrays, prefix, precision, growth, *, native=False):
    expected={'growth_z':np.asarray(growth['redshifts']),
        'growth_fiducial':np.asarray(growth['fiducial']),
        'growth_jacobian':np.asarray(growth['ad'])[:,:3] if native else np.asarray(growth['ad'])}
    jac = expected['growth_jacobian']
    expected['growth_cov_marg'] = jac @ precision['cov_cosmo_marg'] @ jac.T
    expected['growth_cov_nuisance_fixed'] = jac @ precision['cov_cosmo_nuisance_fixed'] @ jac.T
    for field,value in expected.items():
        require_close(arrays[prefix+'/'+field],value,prefix+'/'+field)
    return expected


def verify_physical_binding(path, likelihood):
    """Bind every saved physical array to the separately authenticated survey."""
    path = Path(path)
    record = read_json(path/'manifest.json')
    expected = {}
    metadata = pack_likelihood(likelihood, expected)
    if canonical(record['likelihood']) != canonical(metadata):
        raise ValueError('saved physical likelihood metadata differs from measured survey')
    with np.load(path/'arrays.npz', allow_pickle=False) as arrays:
        for name,value in expected.items():
            if name not in arrays or not np.array_equal(arrays[name], value):
                raise ValueError('saved physical array differs from measured survey: '+name)
    return {'status':'passed', 'array_count':len(expected)}


def verify_case(path, provenance):
    """Reconstruct every saved precision, propagated covariance and native3 comparison."""
    path=Path(path);record=read_json(path/'manifest.json')
    if (record.get('schema') != SCHEMA or record.get('full_forecast_qualification') != 'pending'
            or canonical(record.get('provenance')) != canonical(provenance)
            or record.get('fisher_solver') != fisher_solver_record(DECIMAL_FISHER_SOLVER)):
        raise ValueError('inference schema, source or scope differs')
    for name in ('arrays','growth'):
        suffix='npz' if name=='arrays' else 'json'
        if record.get(name+'_sha256') != file_digest(path/(name+'.'+suffix)):
            raise ValueError('changed saved '+name)
    with np.load(path/'arrays.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    if any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('nonfinite or non-float64 inference array')
    like=unpack_likelihood(record['likelihood'],arrays)
    validate_likelihood(like)
    _check_native_identity(provenance,like.config.case)
    success,failed=record['treatments'],record['failures']
    if set(success)&set(failed) or set(success)|set(failed)!=set(COSMOLOGY_TREATMENTS):
        raise ValueError('all seven inference outcomes must be accounted for')
    growth=read_json(path/'growth.json')
    z=np.array([(a+b)/2 for a,b in like.config.nominal_bins])
    growth_check=_validate_growth(growth,elf_cosmology_spec(like.reference_cosmology),like.reference_cosmology,z)
    expected_arrays={};pack_likelihood(like,expected_arrays)
    results={}
    for name in COSMOLOGY_TREATMENTS:
        if name in failed:
            failure=failed[name]
            if set(failure)!={'status','error','precision_inputs'} or failure['status']!='failed' or not failure['error']:
                raise ValueError('invalid failed treatment record')
            raw=prepare_cosmology_precision(like,name)
            if canonical(raw)!=canonical(failure['precision_inputs']):
                raise ValueError('failed treatment precision inputs differ')
            try:
                project_cosmology_precision(like,name,fisher_solver=DECIMAL_FISHER_SOLVER)
            except ValueError:
                continue
            raise ValueError('saved failed precision now solves: '+name)
        precision=project_cosmology_precision(like,name,fisher_solver=DECIMAL_FISHER_SOLVER)
        meta=success[name]
        if (meta.get('treatment')!=name or meta.get('fisher_solver')!=fisher_solver_record(DECIMAL_FISHER_SOLVER)
                or any(canonical(meta.get(f))!=canonical(precision[f]) for f in
                       ('free_cosmology','active_parameter_names','active_indices'))):
            raise ValueError('treatment basis or precision identity differs')
        for field in PRECISION_FIELDS:
            require_close(arrays[name+'/'+field],precision[field],name+'/'+field)
            expected_arrays[name+'/'+field]=arrays[name+'/'+field]
        fields=_growth_fields(arrays,name,precision,growth)
        expected_arrays.update({name+'/'+f:arrays[name+'/'+f] for f in GROWTH_FIELDS})
        results[name]=CosmologyTreatmentResult(likelihood=like,treatment=name,**precision,**fields)
    native=None;bridge={'status':'failed','reason':'required conditional3 or native3 is unavailable'}
    if record['native'] is not None:
        native=unpack_likelihood(record['native'],arrays,'native')
        validate_likelihood(native);pack_likelihood(native,expected_arrays,'native')
        if canonical(native.model) != canonical(like.model):
            raise ValueError('native and extended numerical model differ')
    if record['native_result'] is not None:
        if native is None or record['native_result'].get('fisher_solver')!=fisher_solver_record(DECIMAL_FISHER_SOLVER):
            raise ValueError('native result lacks matching physical calculation/solver')
        precision=project_baseline_precision(native,native.config.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
        for field in PRECISION_FIELDS:
            require_close(arrays['native_result/'+field],precision[field],'native_result/'+field)
            expected_arrays['native_result/'+field]=arrays['native_result/'+field]
        fields=_growth_fields(arrays,'native_result',precision,growth,native=True)
        expected_arrays.update({'native_result/'+f:arrays['native_result/'+f] for f in GROWTH_FIELDS})
        baseline=PriorResult(likelihood=native,**{k:v for k,v in precision.items() if k!='F_like'},**fields)
        if 'conditional3' in results:
            bridge=validate_conditional_bridge(like,results['conditional3'],native,baseline)
    if set(arrays)!=set(expected_arrays):
        raise ValueError('inference array inventory differs from explicit outcomes')
    monotonic=validate_treatment_cohort(results) if REQUIRED<=set(results) else {'status':'not_checked_required_control_failed'}
    passed=REQUIRED<=set(results) and bridge['status']=='passed'
    return {'status':'passed' if passed else 'failed','case':like.config.case,
        'successful_treatments':[n for n in COSMOLOGY_TREATMENTS if n in results],
        'failed_treatments':[n for n in COSMOLOGY_TREATMENTS if n in failed],
        'native_bridge':bridge,'growth':growth_check,'prior_monotonicity':monotonic,
        'full_forecast_qualification':'pending'}
