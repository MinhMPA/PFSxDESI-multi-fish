#!/usr/bin/env python3
"""Prepare, qualify, produce and report the two-likelihood cosmology supplement."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace
import uuid

import jax
jax.config.update('jax_enable_x64',True)

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from pfsfog.extended_artifacts import (canonical,file_hash,hash_value,write_json,read_json,
    jsonable,identity_provenance,physical_likelihood_id,pack_likelihood,unpack_likelihood,
    validate_likelihood,load_extended_artifacts,write_extended_artifacts,require_close,
    PRECISION_FIELDS,GROWTH_FIELDS)
from pfsfog.cosmology_projection import COSMOLOGY_TREATMENTS,project_cosmology_precision
from pfsfog.extended_validation import validate_study_inventory,validate_conditional_bridge
from pfsfog.ap_numerics import (LEGACY_PROFILE,INTRINSIC_PROFILE,PRODUCTION_PROFILES,
    profile_record,profile_fields,validate_profile_fields)
from pfsfog.fisher_precision import (LEGACY_FISHER_SOLVER,DECIMAL_FISHER_SOLVER,FISHER_SOLVERS,
    fisher_solver_record,solver_from_record,common_fisher_solver)

from pfsfog.shell_numerics import (LEGACY_SHELL_PROFILE, FINE_SHELL_PROFILE, SHELL_PROFILES,
    shell_profile_record, shell_profile_from_record, shell_profile_fields, validate_shell_selection,
    shell_fields, validate_shell_fields)


def inventory(*,profile=LEGACY_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER,shell_profile=LEGACY_SHELL_PROFILE):
    from pfsfog.revision_forecast import RevisionConfig
    if profile not in PRODUCTION_PROFILES: raise ValueError('unsupported production numerical profile')
    validate_shell_selection(shell_profile,profile,fisher_solver)
    shell=shell_profile_record(shell_profile)
    numerical=profile_record(profile); nk=numerical['ap_interpolation_nk']
    return {**({} if profile==LEGACY_PROFILE else {'numerical_profile':numerical}),
        **shell_profile_fields(shell_profile),
        'fisher_solver':fisher_solver_record(fisher_solver),
        'physical_likelihood_count':2,'matrix_treatment_count':14,'growth_output_count':8,
        'growth_outputs_independent':False,'baseline3_role':'independent bridge control',
        'physical_likelihoods':[asdict(RevisionConfig(c,dk=shell['adopted_dk'],ap_interpolation_nk=nk,
            ap_refinement_nk=(2*nk,4*nk),cosmology_basis='extended6')) for c in ('B','C')],
        'treatments':list(COSMOLOGY_TREATMENTS),'requested_dk':(.005 if shell_profile==LEGACY_SHELL_PROFILE else shell['adopted_dk']),'pfs_ceiling':1.6,
        'approximation':'total-matter emulator; scale-independent Heath growth; Gaussian covariance'}


def _inventory_profile(record,requested=None,*,fisher_solver=None,shell_profile=None):
    """Bind both declared settings and requested selection before numerical use."""
    selected_shell=shell_profile_from_record(record,requested=shell_profile)
    declaration=record.get('numerical_profile',{})
    if not isinstance(declaration,dict): raise ValueError('invalid numerical profile declaration')
    name=declaration.get('name',LEGACY_PROFILE)
    if (requested is not None and name!=requested) or name not in PRODUCTION_PROFILES:
        raise ValueError('numerical profile differs from requested study')
    solver=solver_from_record(record)
    if fisher_solver is not None and solver!=fisher_solver:
        raise ValueError('Fisher solver differs from requested study')
    if canonical(record)!=canonical(inventory(profile=name,fisher_solver=solver,shell_profile=selected_shell)):
        raise ValueError('numerical profile or study inventory differs')
    return name


def prepare(root,*,profile=LEGACY_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER,shell_profile=LEGACY_SHELL_PROFILE):
    """Declare inventory without constructing backend or evaluating theory."""
    record=inventory(profile=profile,fisher_solver=fisher_solver,shell_profile=shell_profile); validate_study_inventory(record)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/f'prepare-{uuid.uuid4().hex}.json'; write_json(path,record)
    return record


def observe_process(command,path,*,cwd=None,completion_path=None):
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    start=datetime.now(timezone.utc).isoformat()
    with (path/'process.log').open('wb') as log:
        completed=subprocess.run(command,cwd=cwd or ROOT,stdout=log,stderr=subprocess.STDOUT)
    record={'command':command,'cwd':str(cwd or ROOT),'started_at':start,
        'finished_at':datetime.now(timezone.utc).isoformat(),'returncode':completed.returncode,
        'status':'passed' if completed.returncode==0 else 'failed',
        'log':str((path/'process.log').resolve()),'log_sha256':file_hash(path/'process.log')}
    # Preserve the observation even if completion attachment subsequently fails.
    write_json(path/'execution.json',record)
    if completed.returncode==0 and completion_path is not None:
        cp=Path(completion_path).resolve()
        record.update(completion=read_json(cp),completion_path=str(cp),completion_sha256=file_hash(cp))
    write_json(path/'execution.json',record)
    return record


def require_passed(record,label):
    if record.get('status')!='passed': raise ValueError(label+' failed: '+str(record.get('reasons',record)))
    return record


def validate_exit(record):
    if (record.get('returncode')!=0 or record.get('status')!='passed' or not record.get('command')
            or file_hash(record['log'])!=record['log_sha256']):
        raise ValueError('missing or failed actual process exit/log evidence')


def _pin(path,digest,label):
    if not digest or file_hash(path)!=digest: raise ValueError('external '+label+' hash mismatch')
    return read_json(path)


def load_oracle(root,pin):
    """Consume frozen export bytes; never import the frozen checker here."""
    from scripts.export_cosmology_oracle import HEAD,SELECTION_PIN,RECEIPTS,verify_inputs
    root=Path(root); selected=_pin(root/'export-index.json',pin,'oracle export')
    if (selected.get('status')!='complete' or selected.get('selection_sha256')!=SELECTION_PIN
            or selected.get('receipt_pins')!=RECEIPTS): raise ValueError('oracle selection/receipt pins differ')
    for field in ('execution','verification'):
        if selected[field]['returncode']!=0: raise ValueError('oracle failed actual exit')
        log=root/('export.log' if field=='execution' else 'verify.log')
        if file_hash(log)!=selected[field]['log_sha256']: raise ValueError('oracle log changed')
    index=_pin(root/'worker-index.json',selected['worker_index_sha256'],'oracle worker')
    if index['source_head']!=HEAD or index['selection_sha256']!=SELECTION_PIN or set(index['cases'])!={'B','C'}:
        raise ValueError('oracle source or case identity mismatch')
    from scripts.export_cosmology_oracle import ORACLE,verify_runtime,verify_dependencies,verify_import_records,authenticate_historical_export
    from pfsfog.emulator_paths import approved_relocation_record
    relocation=approved_relocation_record()
    source_receipt=authenticate_historical_export(root,selected,index)
    verify_active_imports()
    if Path(index['cwd']).resolve()!=ORACLE: raise ValueError('oracle frozen cwd changed')
    verify_import_records(index['imports'],relocation_record=relocation)
    active_runtime=verify_runtime(index['runtime'],recorded_root=ORACLE,recorded_cwd=index['cwd'],
                                  active_root=ROOT,relocation_record=relocation)
    import numpy as np
    controls={};input_receipts={}
    for case,r in index['cases'].items():
        if r['receipt_sha256']!=RECEIPTS[case] or r['original_selected']['receipt_sha256']!=RECEIPTS[case]:
            raise ValueError('oracle receipt pin changed')
        input_receipts[case]={key:verify_inputs(r[key],relocation_record=relocation)
                              for key in ('original_provenance','recovered_provenance')}
        verify_dependencies(r['original_provenance'])
        verify_dependencies(r['recovered_provenance'],index['runtime'])
        for key in ('source_content_hash','backend_snapshot','network_hash','survey_input_hash'):
            if r['original_provenance'][key]!=r['recovered_provenance'][key]: raise ValueError('oracle recovered source/input identity differs')
        meta=_pin(r['control'],r['control_sha256'],'oracle control')
        if file_hash(r['arrays'])!=r['arrays_sha256']: raise ValueError('oracle arrays changed')
        with np.load(r['arrays'],allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
        if set(arrays)!=set(r['array_inventory']): raise ValueError('oracle array inventory differs')
        import hashlib
        for k,v in arrays.items():
            if r['array_inventory'][k]!={'shape':list(v.shape),'dtype':v.dtype.str,
                    'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()}:
                raise ValueError('oracle array layout changed')
        like=unpack_likelihood(meta,arrays); validate_likelihood(like)
        result=SimpleNamespace(**{f:arrays['result/'+f] for f in (*PRECISION_FIELDS,*GROWTH_FIELDS)},
            fisher_solver=LEGACY_FISHER_SOLVER,
            historical_parent_identity={'oracle_export':{'path':str(root.resolve()),'sha256':pin,'case':case},
                'worker_index_sha256':selected['worker_index_sha256'],
                'arrays_sha256':r['arrays_sha256'],'control_sha256':r['control_sha256']})
        controls[case]=(like,result)
    _pin(root/'export-index.json',pin,'oracle export')
    if source_receipt!=authenticate_historical_export(root,selected,index):
        raise ValueError('historical oracle source verification changed during reload')
    for case,r in index['cases'].items():
        for key,receipt in input_receipts[case].items():
            if receipt!=verify_inputs(r[key],relocation_record=relocation):
                raise ValueError('historical inputs changed during reload')
    verify_active_imports()
    final_runtime=verify_runtime(index['runtime'],recorded_root=ORACLE,recorded_cwd=index['cwd'],
                                  active_root=ROOT,relocation_record=relocation)
    if final_runtime!=active_runtime: raise ValueError('runtime changed during historical reload')
    receipt={'schema_version':'pfs-desi-historical-oracle-consumption-v1',
             'export_index_sha256':pin,'worker_index_sha256':selected['worker_index_sha256'],
             'source_verification':source_receipt,'input_verification':input_receipts,
             'historical_runtime':index['runtime'],'active_runtime':active_runtime,
             'relocation_record_sha256':hash_value(relocation),'historical_exact_path_replay':False}
    return controls,receipt


def collect_provenance(backend,*,profile=LEGACY_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER,shell_profile=LEGACY_SHELL_PROFILE):
    from pfsfog.one_loop import reject_legacy_ap_qualification
    reject_legacy_ap_qualification(backend)
    from scripts.run_revision import collect_provenance as original
    from scripts.export_cosmology_oracle import runtime_record
    validate_shell_selection(shell_profile,profile,fisher_solver)
    numerical=profile_record(profile)
    requested_dk=.005 if shell_profile==LEGACY_SHELL_PROFILE else shell_profile_record(shell_profile)['adopted_dk']
    p=original(backend,requested_dk=requested_dk,ap_nk=numerical['ap_interpolation_nk'])
    p.update(shell_profile_fields(shell_profile))
    if shell_profile!=LEGACY_SHELL_PROFILE: p['original_grid_dk']=.0025
    if profile!=LEGACY_PROFILE: p['numerical_profile']=numerical
    p['fisher_solver']=fisher_solver_record(fisher_solver)
    p['source_file_hashes'].update({str(ROOT/'scripts'/f):file_hash(ROOT/'scripts'/f)
        for f in ('run_extended_cosmology.py','export_cosmology_oracle.py')})
    p['source_content_hash']=hash_value(p['source_file_hashes'])
    p['runtime']=runtime_record(); p['runtime_hash']=hash_value(p['runtime'])
    p['runtime_cwd']=str(Path.cwd())
    p['driver_source']={'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)}
    return p


def verify_provenance(p):
    identity_provenance(p)
    solver=solver_from_record(p)
    shell=shell_profile_from_record(p)
    validate_shell_selection(shell,p.get('numerical_profile',{}).get('name',LEGACY_PROFILE),solver)
    if shell!=LEGACY_SHELL_PROFILE and (p.get('requested_dk')!=.00125 or p.get('original_grid_dk')!=.0025):
        raise ValueError('shell provenance spacing differs')
    from scripts.export_cosmology_oracle import verify_scientific_inventory
    expected=verify_scientific_inventory(p,ROOT,extended=True)
    if hash_value(p['runtime'])!=p['runtime_hash']: raise ValueError('runtime identity changed')
    from scripts.export_cosmology_oracle import verify_runtime,verify_dependencies
    verify_active_imports(expected)
    verify_runtime(p['runtime'],recorded_root=ROOT,recorded_cwd=p['runtime_cwd'],active_root=ROOT)
    verify_dependencies(p,p['runtime'])


def verify_active_imports(inventory=None):
    from scripts.export_cosmology_oracle import scientific_inventory,verify_project_origins
    expected=scientific_inventory(ROOT,extended=True) if inventory is None else inventory
    verify_project_origins(ROOT,expected)


def _requested_solver(provenance):
    """Legacy Python callers keep their default; every newly saved record labels it."""
    return solver_from_record(provenance) if 'fisher_solver' in provenance else LEGACY_FISHER_SOLVER


def treatments(like,*,failure_root,provenance):
    from pfsfog.cosmology_projection import apply_cosmology_treatment
    failure_root=Path(failure_root); failure_root.mkdir(parents=True,exist_ok=True)
    raw=save_control(failure_root/'raw-likelihood',like,{},provenance=provenance)
    results,failed={},{}
    solver=_requested_solver(provenance)
    for name in COSMOLOGY_TREATMENTS:
        try: results[name]=apply_cosmology_treatment(like,name,
            **({} if solver==LEGACY_FISHER_SOLVER else {'fisher_solver':solver}))
        except (ValueError,ArithmeticError) as exc:
            failed[name]={'status':'failed','error':str(exc),'raw':{'exception_type':type(exc).__name__}}
    write_json(failure_root/'failed-treatments.json',failed)
    if not {'conditional3','marginal6'}<=set(results):
        write_json(failure_root/'failure.json',{'schema_version':'pfs-desi-cosmology-failure-v1',
            'status':'failed','consumable':False,'failed_treatments':failed,
            'raw_likelihood':raw,'provenance':provenance})
        raise ValueError('required treatment failed: '+str(failed))
    return results,failed


def treatment_outcomes(results,failures):
    """Retain all seven prior choices, including explicit failed solutions."""
    if (not isinstance(results,dict) or not isinstance(failures,dict)
            or set(results)&set(failures)
            or set(results)|set(failures)!=set(COSMOLOGY_TREATMENTS)
            or not {'conditional3','marginal6'}<=set(results)):
        raise ValueError('treatment coverage requires seven disjoint outcomes and both required successes')
    for value in failures.values():
        if (not isinstance(value,dict) or set(value)!={'status','error','raw'}
                or value['status']!='failed' or not isinstance(value['error'],str) or not value['error']
                or not isinstance(value['raw'],dict)
                or not isinstance(value['raw'].get('exception_type'),str) or not value['raw']['exception_type']):
            raise ValueError('treatment failure evidence is incomplete or malformed')
    return {**results,**failures}


def save_control(path,like,results,*,provenance):
    import numpy as np
    solver=_requested_solver(provenance)
    if results and common_fisher_solver(*results.values())!=solver:
        raise ValueError('control result/provenance Fisher solver differs')
    arrays={}; meta=pack_likelihood(like,arrays)
    meta['fisher_solver']=fisher_solver_record(solver)
    meta.update(shell_fields(provenance,like.config))
    for name,result in results.items():
        for f in (*PRECISION_FIELDS,*GROWTH_FIELDS): arrays[name+'/'+f]=getattr(result,f)
    path.mkdir(parents=True,exist_ok=False)
    np.savez(path/'arrays.npz',**arrays); write_json(path/'likelihood.json',meta)
    return {**profile_fields(like.config),**shell_fields(provenance,like.config),'fisher_solver':fisher_solver_record(solver),
            'producer_identity':identity_provenance(provenance),'path':str(path.resolve()),'metadata_sha256':file_hash(path/'likelihood.json'),
            'arrays_sha256':file_hash(path/'arrays.npz'),'treatments':list(results)}


def load_control(record,*,provenance):
    if record.get('producer_identity')!=identity_provenance(provenance): raise ValueError('control source/input/runtime identity mismatch')
    import numpy as np
    solver=solver_from_record(record)
    if solver!=_requested_solver(provenance): raise ValueError('control/provenance Fisher solver differs')
    path=Path(record['path']); meta=_pin(path/'likelihood.json',record['metadata_sha256'],'control metadata')
    if solver_from_record(meta)!=solver: raise ValueError('control metadata Fisher solver differs')
    if file_hash(path/'arrays.npz')!=record['arrays_sha256']: raise ValueError('control arrays changed')
    with np.load(path/'arrays.npz',allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
    try: like=unpack_likelihood(meta,arrays)
    except (KeyError,TypeError) as exc: raise ValueError('control likelihood arrays are incomplete') from exc
    validate_shell_fields(record,provenance,like.config)
    validate_shell_fields(meta,provenance,like.config)
    names=record['treatments']
    if not isinstance(names,list) or len(names)!=len(set(names)):
        raise ValueError('control treatment inventory is malformed')
    expected={};pack_likelihood(like,expected)
    inventory=set(expected)|{name+'/'+field for name in names for field in (*PRECISION_FIELDS,*GROWTH_FIELDS)}
    if set(arrays)!=inventory or any(a.dtype!=np.dtype('float64') for a in arrays.values()):
        raise ValueError('control scientific array inventory or float64 precision differs')
    validate_likelihood(like)
    validate_profile_fields(record,like.config)
    results={}
    for name in record['treatments']:
        result=SimpleNamespace(**{f:arrays[name+'/'+f] for f in (*PRECISION_FIELDS,*GROWTH_FIELDS)})
        if like.config.cosmology_basis=='extended6':
            precision=project_cosmology_precision(like,name,fisher_solver=solver)
            result.free_cosmology=precision['free_cosmology']; result.active_parameter_names=precision['active_parameter_names']
        else:
            from pfsfog.revision_forecast import project_baseline_precision
            if name!='baseline3': raise ValueError('native control treatment differs')
            precision=project_baseline_precision(like,like.config.nuisance_factor,fisher_solver=solver)
        for f in PRECISION_FIELDS: require_close(getattr(result,f),precision[f],'refinement '+f)
        result.fisher_solver=solver
        result.treatment=name;result.likelihood=like
        z=np.array([(lo+hi)/2 for lo,hi in like.config.nominal_bins]);nc=len(like.cosmo_names)
        require_close(result.growth_z,z,'control growth redshifts',rtol=0.,atol=0.)
        if (result.growth_fiducial.shape!=z.shape or result.growth_jacobian.shape!=(len(z),nc)
                or not np.all(np.isfinite(result.growth_fiducial))
                or not np.all(np.isfinite(result.growth_jacobian))):
            raise ValueError('control growth prediction/Jacobian differs from its coordinates')
        for suffix in ('marg','nuisance_fixed'):
            covariance=getattr(result,'cov_cosmo_'+suffix)
            if covariance.shape!=(nc,nc): raise ValueError('control cosmology covariance shape differs')
            require_close(getattr(result,'growth_cov_'+suffix),
                result.growth_jacobian@covariance@result.growth_jacobian.T,'control full growth propagation')
        results[name]=result
    return like,results


def diagnostic_value(value):
    """Preserve nonfinite failure diagnostics explicitly, never as usable numerics."""
    import math
    value=jsonable(value)
    if hasattr(value,'tolist'): value=value.tolist()
    if isinstance(value,float) and not math.isfinite(value): return {'nonfinite':repr(value)}
    if isinstance(value,dict): return {k:diagnostic_value(v) for k,v in value.items()}
    if isinstance(value,list): return [diagnostic_value(v) for v in value]
    return value


def diagnostic_identity(like,provenance):
    return {'consumable':False,'physical_likelihood_id':physical_likelihood_id(like,provenance),
            'configuration':asdict(like.config),'producer_identity':identity_provenance(provenance)}


def save_gate_measurements(raw,like,provenance,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);target=path/'gate-measurements.json'
    record=diagnostic_value({'schema_version':'pfs-desi-cosmology-gate-diagnostics-v1',
        'status':'measured',**diagnostic_identity(like,provenance),'raw':raw})
    def contains_nonfinite(value):
        if isinstance(value,dict): return 'nonfinite' in value or any(contains_nonfinite(v) for v in value.values())
        if isinstance(value,list): return any(contains_nonfinite(v) for v in value)
        return False
    if contains_nonfinite(record): record['status']='failed'
    write_json(target,record)
    return {'path':str(target.resolve()),'sha256':file_hash(target)}


def radius_failure_diagnostics(report):
    import numpy as np
    rows=[]
    for i,row in enumerate(report.get('rows',[])):
        record={'point_index':i,**{k:row.get(k) for k in ('coordinate','sign','step','redshift','radius_ref','network_sigma8')}}
        try:
            k=np.asarray(row['k_ref']);power=np.asarray(row['power_ref']);x=k*row['radius_ref']
            with np.errstate(all='ignore'):
                window=3*(np.sin(x)-x*np.cos(x))/x**3
                integrated=float(np.sqrt(np.trapezoid(k**3*power*window**2,x=np.log(k))/(2*np.pi**2)))
                error=abs(integrated/row['network_sigma8']-1)
            record.update(integrated_sigma8=integrated,relative_error=error,
                          status='failed' if not np.isfinite(error) or error>=.002 else 'within_tolerance')
        except (ValueError,ArithmeticError,KeyError,TypeError) as exc:
            record.update(status='unavailable',error=str(exc))
        rows.append(record)
    return rows


def save_failure(path,like,provenance,stage,exc,inputs,*,raw=None):
    record={'schema_version':'pfs-desi-cosmology-failure-v1','status':'failed',
        **diagnostic_identity(like,provenance),'stage':stage,'inputs':inputs,
        'exception':{'type':type(exc).__name__,'message':str(exc),
                     'diagnostics':getattr(exc,'diagnostics',None)}}
    if raw is not None and 'sigma8_radius' in raw:
        record['sigma8_radius_diagnostics']=radius_failure_diagnostics(raw['sigma8_radius'])
    write_json(Path(path)/'failure.json',diagnostic_value(record))


def check_measured_gates(raw,like,provenance,path):
    saved=save_gate_measurements(raw,like,provenance,path)
    try: validate_physical(raw,like,physical_likelihood_id(like,provenance))
    except (ValueError,ArithmeticError) as exc:
        save_failure(path,like,provenance,'physical_validation',exc,{'gate_measurements':saved},raw=raw)
        raise


def _historical_reference(oracle,oracle_identity,solver):
    from pfsfog.historical_precision import historical_precision_reference
    parent=getattr(oracle[1],'historical_parent_identity',None)
    if not isinstance(parent,dict) or parent.get('oracle_export')!=oracle_identity:
        raise ValueError('derived reference has no authenticated historical parent')
    return historical_precision_reference(*oracle,fisher_solver=solver,parent_identity=parent)


def _reload_historical_reference(saved,oracle,oracle_identity,provenance):
    import numpy as np
    expected,receipt=_historical_reference(oracle,oracle_identity,_requested_solver(provenance))
    if saved.get('receipt')!=receipt: raise ValueError('historical derived reference receipt differs')
    like,results=load_control(saved['control'],provenance=provenance)
    if set(results)!={'baseline3'}: raise ValueError('historical derived result inventory differs')
    original_arrays={};derived_arrays={}
    original_meta=pack_likelihood(oracle[0],original_arrays)
    derived_meta=pack_likelihood(like,derived_arrays)
    if (canonical(original_meta)!=canonical(derived_meta)
            or set(original_arrays)!=set(derived_arrays)
            or any(not np.array_equal(v,derived_arrays[k]) for k,v in original_arrays.items())):
        raise ValueError('historical derived likelihood is not its exact authenticated parent')
    for field in (*PRECISION_FIELDS,*GROWTH_FIELDS):
        require_close(getattr(results['baseline3'],field),getattr(expected,field),'derived historical '+field)
    return like,results['baseline3'],receipt


def check_baseline_control(like,conditional,baseline,oracle,path,provenance,*,oracle_identity=None):
    from pfsfog.revision_forecast import apply_revision_priors
    from pfsfog.extended_validation import validate_native_baseline_identity
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    inputs={'raw_baseline':save_control(path/'baseline3-raw',baseline,{},provenance=provenance),
            'oracle_export':oracle_identity}
    stage='baseline_prior'
    solver=_requested_solver(provenance)
    derived_reference={}
    try:
        result=apply_revision_priors(baseline,baseline.config.nuisance_factor,
            **({} if solver==LEGACY_FISHER_SOLVER else {'fisher_solver':solver}))
        inputs['baseline3']=save_control(path/'baseline3',baseline,{'baseline3':result},provenance=provenance)
        stage='baseline_baseline3'
        checks={'conditional3_native3_same_resolution':validate_conditional_bridge(like,conditional,baseline,result)}
        if oracle is not None:
            stage='baseline_oracle'
            if solver==LEGACY_FISHER_SOLVER:
                checks['conditional3_historical_oracle']=validate_conditional_bridge(like,conditional,*oracle)
                checks['native3_historical_oracle']=validate_native_baseline_identity(baseline,result,*oracle)
            else:
                derived,receipt=_historical_reference(oracle,oracle_identity,solver)
                reference={'control':save_control(path/'historical-derived',oracle[0],{'baseline3':derived},provenance=provenance),
                           'receipt':receipt}
                inputs['historical_derived_reference']=reference
                checks['historical_legacy_replay']=receipt['legacy_replay']
                checks['conditional3_historical_derived']=validate_conditional_bridge(like,conditional,oracle[0],derived)
                checks['native3_historical_derived']=validate_native_baseline_identity(baseline,result,oracle[0],derived)
                derived_reference={'historical_derived_reference':reference}
    except (ValueError,ArithmeticError) as exc:
        save_failure(path,like,provenance,stage,exc,inputs)
        raise
    return {**inputs['baseline3'],'reference_checks':checks,'oracle_export':oracle_identity,**derived_reference}


def validate_baseline_control(like,conditional,record,oracle,provenance,*,oracle_identity=None):
    """Recompute every same-grid comparison and authenticate its recorded meaning."""
    from pfsfog.extended_validation import validate_native_baseline_identity
    baseline,results=load_control(record,provenance=provenance)
    if set(results)!={'baseline3'}: raise ValueError('native3 reference result inventory differs')
    if record.get('oracle_export')!=oracle_identity:
        raise ValueError('matched-resolution oracle export identity differs')
    result=results['baseline3']
    checks={'conditional3_native3_same_resolution':validate_conditional_bridge(like,conditional,baseline,result)}
    if oracle is None:
        if 'historical_derived_reference' in record:
            raise ValueError('same-grid control without oracle cannot claim a derived historical reference')
    elif _requested_solver(provenance)==LEGACY_FISHER_SOLVER:
        if 'historical_derived_reference' in record:
            raise ValueError('legacy comparison cannot substitute a derived reference')
        checks.update(conditional3_historical_oracle=validate_conditional_bridge(like,conditional,*oracle),
            native3_historical_oracle=validate_native_baseline_identity(baseline,result,*oracle))
    else:
        if not isinstance(record.get('historical_derived_reference'),dict):
            raise ValueError('missing historical derived reference')
        dl,dr,receipt=_reload_historical_reference(record['historical_derived_reference'],oracle,oracle_identity,provenance)
        checks.update(historical_legacy_replay=receipt['legacy_replay'],
            conditional3_historical_derived=validate_conditional_bridge(like,conditional,dl,dr),
            native3_historical_derived=validate_native_baseline_identity(baseline,result,dl,dr))
    if record.get('reference_checks')!=checks:
        raise ValueError('matched-resolution reference comparison receipt differs')
    return baseline,result


AP_OUTPUT_COMPARISONS=((8192,16384),(16384,32768),(32768,65536),(16384,65536))


def _output_reference(record,nk,role,*,failed=None):
    return {**record,'reference_role':role,'evaluated_ap_grids':[nk],
            'derivative_qualification_performed':False,
            **({} if failed is None else {'failed_treatments':failed})}


def measure_ap_references(base,base_results,oracle,path,provenance,*,oracle_identity,build_likelihood):
    """Construct native3 independently and measure four output resolutions.

    build_likelihood receives a complete RevisionConfig and must use its native
    basis. No derivative qualification or table above65536 is launched here.
    Nonconverged reports are saved for diagnosis; the cohort enforces release.
    """
    from pfsfog.cosmology_resolution import compare_ap_resolution
    if base.config.ap_interpolation_nk!=16384 or base.config.cosmology_basis!='extended6':
        raise ValueError('independent AP references require the candidate16384 extended likelihood')
    shell=shell_profile_from_record(provenance)
    candidate=shell!=LEGACY_SHELL_PROFILE
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    native=build_likelihood(replace(base.config,cosmology_basis='baseline3'))
    current=check_baseline_control(base,base_results['conditional3'],native,None,
        path/'native3_16384',provenance)
    controls={'native3_16384':_output_reference(current,16384,'native3_same_resolution')}
    measured={16384:(base,base_results)}
    for nk in (8192,32768,65536):
        config=replace(base.config,ap_interpolation_nk=nk,ap_refinement_nk=(2*nk,4*nk))
        like=build_likelihood(config)
        results,failed=treatments(like,failure_root=path/(f'extended6_{nk}-projection'),provenance=provenance)
        controls[f'extended6_{nk}']=_output_reference(
            save_control(path/f'extended6_{nk}',like,results,provenance=provenance),
            nk,'ap_likelihood_output',failed=failed)
        measured[nk]=(like,{**results,**failed})
        if nk==8192:
            old_native=build_likelihood(replace(config,cosmology_basis='baseline3'))
            historical=check_baseline_control(like,results['conditional3'],old_native,None if candidate else oracle,
                path/'native3_8192',provenance,oracle_identity=None if candidate else oracle_identity)
            controls['native3_8192']=_output_reference(historical,8192,'native3_same_resolution')
    comparisons={f'{a}_to_{b}':compare_ap_resolution(measured[a][0],measured[b][0],
                  measured[a][1],measured[b][1]) for a,b in AP_OUTPUT_COMPARISONS}
    extra={}
    assessment='historical_change_requires_assessment'
    if candidate:
        from pfsfog.shell_references import measure_original_grid_references
        extra={**shell_profile_fields(shell),'original_grid_replay':measure_original_grid_references(
            base,oracle,path/'original-grid',provenance,oracle_identity=oracle_identity,
            build_likelihood=build_likelihood,control_api=sys.modules[__name__])}
        assessment='candidate_shell_ap8192_requires_assessment'
    record={'definition':'ap-independent-references-v2' if candidate else 'ap-independent-references-v1','case':base.config.case,
        'fisher_solver':fisher_solver_record(_requested_solver(provenance)),
        'base_physical_likelihood_id':physical_likelihood_id(base,provenance),
        'oracle_export':oracle_identity,'controls':controls,'comparisons':comparisons,
        assessment:comparisons['8192_to_16384']['status']!='passed',**extra}
    # Snapshot the supplied external identity; editing a returned report must
    # not mutate the independently held expected oracle pin through an alias.
    record=jsonable(record)
    write_json(path/'reference-comparisons.json',record)
    validate_ap_references(base,base_results,record,oracle,provenance,oracle_identity=oracle_identity)
    return record


def validate_ap_references(base,base_results,record,oracle,provenance,*,oracle_identity):
    """Reload each exact reference and recompute all same-grid/resolution tests."""
    from pfsfog.cosmology_resolution import compare_ap_resolution
    shell=shell_profile_from_record(provenance)
    candidate=shell!=LEGACY_SHELL_PROFILE
    shell_profile_from_record(record,requested=shell)
    solver=solver_from_record(record)
    if (solver!=_requested_solver(provenance)
            or solver!=common_fisher_solver(*(r for r in base_results.values() if not isinstance(r,dict)))):
        raise ValueError('AP reference Fisher solver differs')
    expected={'native3_16384':(16384,'baseline3','native3_same_resolution'),
        'native3_8192':(8192,'baseline3','native3_same_resolution'),
        **{f'extended6_{nk}':(nk,'extended6','ap_likelihood_output') for nk in (8192,32768,65536)}}
    if (base.config.ap_interpolation_nk!=16384 or base.config.cosmology_basis!='extended6'
            or record.get('definition')!=('ap-independent-references-v2' if candidate else 'ap-independent-references-v1')
            or record.get('case')!=base.config.case
            or record.get('base_physical_likelihood_id')!=physical_likelihood_id(base,provenance)
            or record.get('oracle_export')!=oracle_identity
            or not isinstance(record.get('controls'),dict) or set(record['controls'])!=set(expected)):
        raise ValueError('AP reference identity or inventory differs')
    loaded={16384:(base,base_results)}
    for key,(nk,basis,role) in expected.items():
        saved=record['controls'][key]
        if (saved.get('reference_role')!=role or saved.get('evaluated_ap_grids')!=[nk]
                or saved.get('derivative_qualification_performed') is not False):
            raise ValueError('AP output-reference role or evaluated grid differs')
        like,results=load_control(saved,provenance=provenance)
        config=replace(base.config,cosmology_basis=basis,ap_interpolation_nk=nk,ap_refinement_nk=(2*nk,4*nk))
        if like.config!=config: raise ValueError('AP reference configuration differs')
        if basis=='baseline3':
            if set(results)!={'baseline3'}: raise ValueError('native3 reference result inventory differs')
        else:
            failures=saved.get('failed_treatments')
            if not isinstance(failures,dict) or set(failures)&set(results):
                raise ValueError('AP reference treatment failures overlap or are absent')
            loaded[nk]=(like,{**results,**failures})
    validate_baseline_control(base,base_results['conditional3'],record['controls']['native3_16384'],None,provenance)
    old,old_results=loaded[8192]
    validate_baseline_control(old,old_results['conditional3'],record['controls']['native3_8192'],
                              None if candidate else oracle,provenance,
                              oracle_identity=None if candidate else oracle_identity)
    assessment='historical_change_requires_assessment'
    if candidate:
        from pfsfog.shell_references import validate_original_grid_references
        loaded['original_grid']=validate_original_grid_references(base,record.get('original_grid_replay'),
            oracle,provenance,oracle_identity=oracle_identity,control_api=sys.modules[__name__])
        assessment='candidate_shell_ap8192_requires_assessment'
        if 'historical_change_requires_assessment' in record:
            raise ValueError('candidate-shell AP change cannot claim original-grid assessment')
    elif 'original_grid_replay' in record or 'candidate_shell_ap8192_requires_assessment' in record:
        raise ValueError('legacy AP references cannot claim candidate-shell roles')
    comparisons={f'{a}_to_{b}':compare_ap_resolution(loaded[a][0],loaded[b][0],loaded[a][1],loaded[b][1])
                 for a,b in AP_OUTPUT_COMPARISONS}
    if (canonical(record.get('comparisons'))!=canonical(comparisons)
            or record.get(assessment)!=(comparisons['8192_to_16384']['status']!='passed')):
        raise ValueError('saved AP output-resolution comparison differs from its references')
    return loaded,comparisons


def measure_case(like,backend,provenance,survey,path):
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.extended_validation import measure_growth_jacobian,validate_extended_response,measure_sigma8_radius
    from pfsfog.revision_validation_ap import measure_ap_cell,ap_cell_record,measure_pure_ap_controls
    from pfsfog.revision_validation_physical import (expected_support_schedule,declare_fd_manifest,
        assembled_fd_certificate,audit_cell_covariance_neighborhood)
    raw={}
    try:
        space=make_cosmology_spec(like.reference_cosmology,extended=True,
                                 ap_nk=like.config.ap_interpolation_nk)
        pid=physical_likelihood_id(like,provenance)
        support=expected_support_schedule(like.config,survey,space=space)
        ids=[c.metadata['cell_id'] for c in like.cells]
        expected=declare_fd_manifest(like,ids,pid,expected_support=support,space=space)
        raw={**profile_fields(like.config),'response_identity':identity_provenance(provenance),'expected_support':support}
        raw['physical_fd']=assembled_fd_certificate(like,backend,ids,expected,expected_support=support,
                                                    provenance=provenance,space=space)
        write_json(path/'physical-fd.json',raw['physical_fd'])
        raw['covariance_neighborhood']=[]
        for c in like.cells:
            raw['covariance_neighborhood'].append(audit_cell_covariance_neighborhood(like,c,backend,space=space))
            write_json(path/'covariance-neighborhood.json',raw['covariance_neighborhood'])
        response={'cosmology_columns':list(space.names),'points':[],'cells':[],
            'growth':measure_growth_jacobian(space,like.reference_cosmology,[(lo+hi)/2 for lo,hi in like.config.nominal_bins])}
        intrinsic_profile=like.config.ap_interpolation_nk==16384
        if intrinsic_profile: response['intrinsic_geometry']=[]
        raw['response']=response
        pure=measure_pure_ap_controls(like.config.ap_interpolation_nk,like.config.ap_interpolation_nmu)
        raw['pure_ap_controls']=pure
        save_gate_measurements(raw,like,provenance,path)
        for icell,c in enumerate(like.cells):
            if not c.metadata['retained_pairs']: continue
            kwargs={}
            if intrinsic_profile:
                from pfsfog.revision_validation_ap_intrinsic import write_intrinsic_sidecar
                def retain_intrinsic(record,arrays):
                    pairs=[list(pair) for pair in c.metadata['retained_pairs']]
                    ipair=pairs.index(record['pair'])
                    entry=write_intrinsic_sidecar(path,f'intrinsic/cell{icell:03d}-pair{ipair:03d}.npz',record,arrays)
                    response['intrinsic_geometry'].append(entry)
                    write_json(path/'response-partial.json',response)
                kwargs['intrinsic_sink']=retain_intrinsic
            rows=measure_ap_cell(like,c,backend,pure_controls=pure,space=space,**kwargs)
            response['points'].extend(rows); response['cells'].append(ap_cell_record(like,c,rows,space=space))
            write_json(path/'response-partial.json',response)
        raw['response']=response; raw['pure_ap_controls']=pure
        raw['sigma8_radius']=measure_sigma8_radius(space,like.reference_cosmology,response['growth']['redshifts'])
        save_gate_measurements(raw,like,provenance,path)
        validate_extended_response(response,like,intrinsic_root=path)
        return raw
    except (ValueError,ArithmeticError) as exc:
        saved=save_gate_measurements(raw,like,provenance,path)
        save_failure(path,like,provenance,'physical_measurement',exc,{'gate_measurements':saved},raw=raw)
        raise


def validate_physical(raw,like,pid):
    """Re-evaluate quantitative acceptance and exact full nuisance/FD coverage."""
    import numpy as np
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.revision_validation import require_complete_coverage
    from pfsfog.revision_validation_physical import declare_fd_manifest,_steps
    from pfsfog.revision_validation_ap import validate_pure_ap_controls
    validate_profile_fields(raw,like.config)
    space=make_cosmology_spec(like.reference_cosmology,extended=True,
                             ap_nk=like.config.ap_interpolation_nk)
    support=raw['expected_support']; ids=[c.metadata['cell_id'] for c in like.cells]
    declared=declare_fd_manifest(like,ids,pid,expected_support=support,space=space)
    fd=raw['physical_fd']
    require_passed(fd,'physical FD')
    if fd.get('failures'): raise ValueError('physical FD contains explicit failed measurements')
    require_complete_coverage(declared,fd['measured_coverage'])
    require_complete_coverage(declared,fd['expected_coverage'])
    if fd['scope']!='all_cells' or fd['selected_cell_ids']!=ids or fd['all_cell_ids']!=ids:
        raise ValueError('physical FD requires complete cells')
    expected={(pid,c.metadata['cell_id'],n):list(_steps(n,like.fiducial[like.names.index(n)],space=space))
              for c in like.cells if c.names for n in c.names}
    columns={}
    for row in fd['column_results']:
        key=(row['configuration'],row['cell_id'],row['column'])
        if key in columns or row['steps']!=expected.get(key): raise ValueError('FD column schedule mismatch')
        errors=np.asarray(row['step_errors']); limits=np.asarray(row['step_tolerances'])
        if errors.shape!=(3,) or limits.shape!=(3,) or not np.all(np.isfinite(errors)) or np.any(errors<0) or not np.all(np.isin(limits,[1e-3,1e-6])):
            raise ValueError('FD quantitative columns missing')
        good=errors<limits
        if not ((good[0] and good[1]) or (good[1] and good[2])): raise ValueError('physical FD failed')
        columns[key]=row
    if set(columns)!=set(expected): raise ValueError('omitted FD columns')
    groups={}
    for row in fd['errors']:
        key=(row['configuration'],row['cell_id'],tuple(row['pair']),row['ell'],row['column'])
        groups.setdefault(key,[]).append(row)
    expected_groups={(r[0],r[1],tuple(r[2]),r[3],r[4]) for r in declared}
    if set(groups)!=expected_groups: raise ValueError('physical FD pair/multipole coverage differs')
    for key,rows in groups.items():
        rows.sort(key=lambda r:r['step'],reverse=True)
        if [r['step'] for r in rows]!=expected[(key[0],key[1],key[4])]: raise ValueError('FD pair steps differ')
        good=[]
        for r in rows:
            limit=1e-3 if r['reason']=='relative' else 1e-6
            if r['limit']!=limit or not np.isfinite(r['error']) or r['error']<0: raise ValueError('FD error/limit malformed')
            good.append(r['error']<limit)
        accepted=all(good) if all(r['reason']=='zero block' for r in rows) else ((good[0] and good[1]) or (good[1] and good[2]))
        if not accepted: raise ValueError('physical pair/multipole FD failed')
    validate_covariance_neighborhood(raw['covariance_neighborhood'],like)
    validate_pure_ap_controls(raw['pure_ap_controls'],like.config.ap_interpolation_nk,
                             like.config.ap_interpolation_nmu)
    from pfsfog.extended_validation import validate_sigma8_radius
    validate_sigma8_radius(raw['sigma8_radius'],space,like.reference_cosmology,[(a+b)/2 for a,b in like.config.nominal_bins])


def validate_covariance_neighborhood(checks,like):
    import numpy as np
    ids=[c.metadata['cell_id'] for c in like.cells]
    if [r['cell_id'] for r in checks]!=ids: raise ValueError('covariance neighborhood cells differ')
    for c,r in zip(like.cells,checks):
        require_passed(r,'physical covariance neighborhood')
        if any(r.get(k) for k in ('failures','failure','error','failure_diagnostics','diagnostics')):
            raise ValueError('physical covariance neighborhood contains explicit failure diagnostics')
        n=len(like.grid.centers) if c.names else 0
        if r['expected_shells']!=n or r['processed_shells']!=n or r['points']!=(1+6*len(c.names) if c.names else 0):
            raise ValueError('covariance neighborhood coverage differs')
        if c.names:
            for k in ('field','retained_covariance'):
                if (not np.isfinite(r['min_scaled_eigenvalue'][k]) or r['min_scaled_eigenvalue'][k]<=0
                        or not np.isfinite(r['max_scaled_residual'][k]) or not 0<=r['max_scaled_residual'][k]<=1e-10):
                    raise ValueError('physical covariance neighborhood failed')


def qualify_worker(args):
    profile=getattr(args,'numerical_profile',LEGACY_PROFILE)
    shell_profile=getattr(args,'shell_profile',LEGACY_SHELL_PROFILE)
    solver=getattr(args,'fisher_solver',LEGACY_FISHER_SOLVER)
    fisher_solver_record(solver)
    if profile not in PRODUCTION_PROFILES: raise ValueError('unsupported production numerical profile')
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.surveys import default_survey_group
    from pfsfog.revision_forecast import RevisionConfig,build_revision_likelihood,apply_revision_priors
    from scripts.run_revision import _load_backend
    oracle,oracle_consumption=load_oracle(args.oracle_export,args.oracle_export_sha256)
    backend=_load_backend(); survey=default_survey_group(); cosmo=FiducialCosmology(backend='cosmopower')
    provenance=collect_provenance(backend,**({} if profile==LEGACY_PROFILE else {'profile':profile}),
        **({} if solver==LEGACY_FISHER_SOLVER else {'fisher_solver':solver}),
        **({} if shell_profile==LEGACY_SHELL_PROFILE else {'shell_profile':shell_profile}))
    if solver_from_record(provenance)!=solver: raise ValueError('requested/provenance Fisher solver differs')
    records={}; resumed=[]
    if args.resume:
        records,resumed=load_checkpoint(args.resume,args.resume_sha256,provenance=provenance,
                                       oracle_pin=args.oracle_export_sha256,profile=profile,fisher_solver=solver,shell_profile=shell_profile)
    for config_record in inventory(profile=profile,fisher_solver=solver,shell_profile=shell_profile)['physical_likelihoods']:
        config=RevisionConfig(**config_record); case=config.case
        if case in records: continue
        case_root=args.output_root/case; case_root.mkdir()
        like=build_revision_likelihood(config,survey,cosmo,backend)
        result,failed=treatments(like,failure_root=case_root,provenance=provenance)
        raw=measure_case(like,backend,provenance,survey,case_root)
        raw['failed_treatments']=failed
        check_measured_gates(raw,like,provenance,case_root)
        oracle_identity={'path':str(args.oracle_export.resolve()),'sha256':args.oracle_export_sha256,'case':case}
        references={}
        if profile==INTRINSIC_PROFILE:
            ap_references=measure_ap_references(like,{**result,**failed},oracle[case],case_root/'ap-references',
                provenance,oracle_identity=oracle_identity,
                build_likelihood=lambda control:build_revision_likelihood(control,survey,cosmo,backend))
            bridge=ap_references['controls']['native3_16384']
            references={'ap_references':ap_references}
        else:
            baseline=build_revision_likelihood(replace(config,cosmology_basis='baseline3'),survey,cosmo,backend)
            bridge=check_baseline_control(like,result['conditional3'],baseline,oracle[case],case_root,provenance,
                oracle_identity=oracle_identity)
        artifact=write_extended_artifacts(like,result,case_root/'artifacts',provenance,raw,
                                          intrinsic_root=case_root)
        refinements={}
        control_dk=shell_profile_record(shell_profile)['control_dk']
        for label,refined_config in (('shell',replace(config,dk=control_dk)),('angular',replace(config,n_mu=64))):
            refined=build_revision_likelihood(refined_config,survey,cosmo,backend)
            projected,failures=treatments(refined,failure_root=case_root/(label+'-projection'),provenance=provenance)
            refinements[label]=save_control(case_root/label,refined,projected,provenance=provenance)
            refinements[label]['failed_treatments']=failures
        records[case]={'artifact':str(artifact.resolve()),'manifest_sha256':file_hash(artifact/'manifest.json'),
                       'baseline3':bridge,'refinements':refinements,**references}
        checkpoint={'schema_version':'pfs-desi-cosmology-case-checkpoint-v1','status':'local_cases_validated',
            'inventory':jsonable(inventory(profile=profile,fisher_solver=solver,shell_profile=shell_profile)),'provenance':provenance,'cases':records,
            'oracle_export':str(args.oracle_export.resolve()),'oracle_export_sha256':args.oracle_export_sha256,
            'oracle_consumption':oracle_consumption,
            'origin_execution':str((args.output_root/'worker'/'execution.json').resolve()),'resumed_from':resumed}
        _validate_cases(checkpoint)
        write_json(args.output_root/('checkpoint-'+''.join(sorted(records))+'.json'),checkpoint)
    verify_provenance(provenance)
    record={'schema_version':'pfs-desi-cosmology-qualification-v1','inventory':inventory(profile=profile,fisher_solver=solver,shell_profile=shell_profile),'resumed_from':resumed,
        'provenance':provenance,'cases':records,'oracle_export':str(args.oracle_export.resolve()),
        'oracle_export_sha256':args.oracle_export_sha256,'oracle_consumption':oracle_consumption}
    if profile==INTRINSIC_PROFILE:
        loaded,references=_validate_cases(record)
        record['ap_resolution_ratios']=_ap_resolution_ratios(references)
        assessment='historical_ap_assessment' if shell_profile==LEGACY_SHELL_PROFILE else 'candidate_shell_ap8192_assessment'
        record[assessment]=historical_ap_assessment(records,record['ap_resolution_ratios'],shell_profile=shell_profile)
        if shell_profile!=LEGACY_SHELL_PROFILE:
            record.update(_grid_resolution_reports(loaded,references,provenance))
    write_json(args.output_root/'qualification.json',record)
    qpath=args.output_root/'qualification.json'; qpin=file_hash(qpath)
    validate_qualification(qpath,qpin)
    return qpath.resolve(),qpin


def validate_refinement_binding(base,refined):
    if (base.names!=refined.names or canonical(base.parameters)!=canonical(refined.parameters)
            or canonical(base.reference_cosmology)!=canonical(refined.reference_cosmology)
            or len(base.cells)!=len(refined.cells)):
        raise ValueError('refinement parameter/prior/reference layout differs')
    require_close(base.fiducial,refined.fiducial,'refinement fiducial',rtol=0,atol=0)
    for original,updated in zip(base.cells,refined.cells):
        if original.names!=updated.names or canonical(original.metadata)!=canonical(updated.metadata):
            raise ValueError('refinement physical support/observation metadata differs')


def _load_resolution_control(base,record,provenance,*,label):
    """Authenticate the saved producer before using its quadrature comparison."""
    shell=shell_profile_from_record(provenance,requested=FINE_SHELL_PROFILE)
    if label not in ('shell','angular'): raise ValueError('unknown resolution control')
    other,results=load_control(record,provenance=provenance)
    outcomes=treatment_outcomes(results,record.get('failed_treatments'))
    changes={'dk':shell_profile_record(shell)['control_dk']} if label=='shell' else {'n_mu':64}
    if other.config!=replace(base.config,**changes): raise ValueError('resolution control configuration differs')
    validate_refinement_binding(base,other)
    return (other,outcomes),{'producer_identity':identity_provenance(provenance),
        'physical_likelihood_id':physical_likelihood_id(other,provenance),'configuration':asdict(other.config)}


def _validate_cases(q):
    import numpy as np
    from pfsfog.revision_validation_physical import expected_support_schedule
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.surveys import default_survey_group
    validate_study_inventory(q['inventory'])
    profile=_inventory_profile(q['inventory'])
    shell_profile=shell_profile_from_record(q['inventory'])
    shell_profile_from_record(q['provenance'],requested=shell_profile)
    solver=solver_from_record(q['inventory'])
    if solver_from_record(q['provenance'])!=solver: raise ValueError('qualification/provenance Fisher solver differs')
    if not set(q['cases'])<=set(('B','C')) or not q['cases']:
        raise ValueError('checkpoint case inventory differs')
    verify_provenance(q['provenance']); oracle,consumption=load_oracle(q['oracle_export'],q['oracle_export_sha256'])
    if q.get('oracle_consumption')!=consumption: raise ValueError('historical oracle consumption identity differs')
    loaded={}; refined={}; survey=default_survey_group()
    for case,r in q['cases'].items():
        manifest,arrays,rows,_=load_extended_artifacts(Path(r['artifact']),manifest_sha256=r['manifest_sha256'])
        like=unpack_likelihood(manifest['likelihood'],arrays)
        if asdict(like.config)!=next(c for c in inventory(profile=profile,fisher_solver=solver,shell_profile=shell_profile)['physical_likelihoods'] if c['case']==case):
            raise ValueError('physical configuration differs from approved study')
        validate_profile_fields(q['provenance'],like.config)
        if identity_provenance(manifest['provenance'])!=identity_provenance(q['provenance']): raise ValueError('qualification source mismatch')
        if solver_from_record(manifest)!=solver: raise ValueError('qualification/artifact Fisher solver differs')
        raw=read_json(Path(r['artifact'])/'evidence.json')
        space=make_cosmology_spec(like.reference_cosmology,extended=True,
                                 ap_nk=like.config.ap_interpolation_nk)
        if canonical(raw['expected_support'])!=canonical(expected_support_schedule(like.config,survey,space=space)):
            raise ValueError('survey support differs')
        validate_physical(raw,like,manifest['physical_likelihood_id'])
        results={}
        for t,meta in manifest['treatments'].items():
            results[t]=SimpleNamespace(**{f:arrays[t+'/'+f] for f in (*PRECISION_FIELDS,*GROWTH_FIELDS)},
                likelihood=like,treatment=t,free_cosmology=meta['free_cosmology'],active_parameter_names=meta['active_parameter_names'],
                fisher_solver=solver_from_record(meta))
        treatment_outcomes(results,manifest['failed_treatments'])
        refined[case]={}
        if profile==INTRINSIC_PROFILE:
            if r.get('baseline3')!=r.get('ap_references',{}).get('controls',{}).get('native3_16384'):
                raise ValueError('selected native3 reference differs from the new-profile cohort')
            controls,comparisons=validate_ap_references(like,{**results,**manifest['failed_treatments']},
                r['ap_references'],oracle[case],q['provenance'],
                oracle_identity={'path':q['oracle_export'],'sha256':q['oracle_export_sha256'],'case':case})
            for a,b in AP_OUTPUT_COMPARISONS[1:]:
                require_passed(comparisons[f'{a}_to_{b}'],'AP standard-error convergence')
            refined[case]['ap']=controls
        else:
            validate_baseline_control(like,results['conditional3'],r['baseline3'],oracle[case],q['provenance'],
                oracle_identity={'path':q['oracle_export'],'sha256':q['oracle_export_sha256'],'case':case})
        from pfsfog.extended_validation import validate_treatment_cohort
        validate_treatment_cohort(results)
        loaded[case]=(like,results,rows,manifest)
        if shell_profile!=LEGACY_SHELL_PROFILE:
            from pfsfog.shell_resolution import compare_shell_resolution
            if set(r['refinements'])!={'shell','angular'}: raise ValueError('resolution control inventory differs')
            outcomes=treatment_outcomes(results,manifest['failed_treatments'])
            refined[case]['comparisons']={}
            for label in ('shell','angular'):
                pair,_=_load_resolution_control(like,r['refinements'][label],q['provenance'],label=label)
                refined[case][label]=pair
                comparison=compare_shell_resolution(like,pair[0],outcomes,pair[1],role='convergence')
                require_passed(comparison,label+' absolute-error convergence')
                refined[case]['comparisons'][label]=comparison
            old,old_outcomes=refined[case]['ap']['original_grid'][16384]
            assessment=compare_shell_resolution(old,like,old_outcomes,outcomes,role='historical_assessment')
            if assessment['status']!='assessed': raise ValueError('original-grid shell assessment coverage differs')
            refined[case]['shell_grid_assessment']=assessment
            continue
        for label in ('shell','angular'):
            other,rr=load_control(r['refinements'][label],provenance=q['provenance'])
            treatment_outcomes(rr,r['refinements'][label].get('failed_treatments'))
            expected_config=replace(like.config,**({'dk':.00125} if label=='shell' else {'n_mu':64}))
            if other.config!=expected_config or set(rr)!=set(results): raise ValueError('resolution configuration/treatment mismatch')
            validate_refinement_binding(like,other)
            for t,res in rr.items():
                for field in ('growth_z','growth_fiducial','growth_jacobian'):
                    require_close(getattr(res,field),getattr(results[t],field),'refined growth binding')
                jac=res.growth_jacobian
                for suffix in ('marg','nuisance_fixed'):
                    require_close(getattr(res,'growth_cov_'+suffix),jac@getattr(res,'cov_cosmo_'+suffix)@jac.T,'refined full propagation')
                for field in ('cov_cosmo_marg','growth_cov_marg','cov_cosmo_nuisance_fixed','growth_cov_nuisance_fixed'):
                    original=np.sqrt(np.diag(getattr(results[t],field))); updated=np.sqrt(np.diag(getattr(res,field)))
                    active=original>0
                    if np.any(np.abs(updated[active]/original[active]-1)>=.01): raise ValueError('resolution absolute errors change by at least 1%')
            refined[case][label]=rr
    return loaded,refined


def _grid_resolution_reports(loaded,refined,provenance):
    """Only called after every constituent has passed saved-source authentication."""
    from pfsfog.shell_resolution import compare_shell_ratios
    if set(loaded)!=set(refined) or set(loaded)!={'B','C'}:
        raise ValueError('shell error ratios require complete B/C controls')
    identity=identity_provenance(provenance)
    pairs={}
    for case,(like,results,_,manifest) in loaded.items():
        if identity_provenance(manifest['provenance'])!=identity:
            raise ValueError('shell ratio source/input/runtime identities differ')
        pairs[case]=(like,treatment_outcomes(results,manifest['failed_treatments']))
    reports={}
    for label in ('shell','angular'):
        ratios=compare_shell_ratios(pairs['B'],pairs['C'],refined['B'][label],refined['C'][label],role='convergence')
        require_passed(ratios,label+' B/C and C/B error-ratio convergence')
        reports[label]={'producer_identity':identity,'cases':{case:refined[case]['comparisons'][label] for case in ('B','C')},
            'ratios':ratios,'physical_likelihood_ids':{case:{'base':physical_likelihood_id(pairs[case][0],provenance),
                'refined':physical_likelihood_id(refined[case][label][0],provenance)} for case in ('B','C')}}
    original={case:refined[case]['ap']['original_grid'][16384] for case in ('B','C')}
    ratios=compare_shell_ratios(original['B'],original['C'],pairs['B'],pairs['C'],role='historical_assessment')
    if ratios['status']!='assessed': raise ValueError('historical shell ratio coverage differs')
    historical={'producer_identity':identity,'scope':'original76 to candidate152 shells at fixed AP16384; assessment, not candidate convergence',
        'cases':{case:refined[case]['shell_grid_assessment'] for case in ('B','C')},'ratios':ratios,
        'physical_likelihood_ids':{case:{'original':physical_likelihood_id(original[case][0],provenance),
            'candidate':physical_likelihood_id(pairs[case][0],provenance)} for case in ('B','C')}}
    return {'resolution_comparisons':reports,'shell_grid_assessment':historical}


def _ap_resolution_ratios(refined):
    from pfsfog.cosmology_resolution import compare_resolution_ratios
    return {f'{a}_to_{b}':compare_resolution_ratios(refined['B']['ap'][a][1],refined['C']['ap'][a][1],
        refined['B']['ap'][b][1],refined['C']['ap'][b][1]) for a,b in AP_OUTPUT_COMPARISONS}


def historical_ap_assessment(cases,ratios,*,shell_profile=LEGACY_SHELL_PROFILE):
    """Flag changes in absolute errors or survey gains relative to AP8192."""
    absolute={case:cases[case]['ap_references']['comparisons']['8192_to_16384']
              for case in ('B','C')}
    ratio=ratios['8192_to_16384']
    reasons=[case+': '+reason for case,value in absolute.items() for reason in value['reasons']]
    reasons+=['C/B or B/C: '+reason for reason in ratio['reasons']]
    scope={'scope':'historical AP8192 versus candidate AP16384; separate from candidate convergence'}
    if shell_profile!=LEGACY_SHELL_PROFILE:
        scope={'scope':'AP8192 versus AP16384 on fixed candidate shells; separate from original-grid replay and shell convergence',
               'shell_dk':shell_profile_record(shell_profile)['adopted_dk']}
    return {'required':any(r['status']!='passed' for r in (*absolute.values(),ratio)),
        'case_errors':{case:value['status'] for case,value in absolute.items()},
        'error_ratios':ratio['status'],'reasons':reasons,
        **scope}


def validate_qualification(path,pin):
    import numpy as np
    from pfsfog.revision_validation import compare_nested_likelihoods,compare_marginal_errors
    from pfsfog.revision_validation_physical import likelihood_snapshot
    from pfsfog.cosmology_space import make_cosmology_spec
    q=_pin(path,pin,'qualification')
    if q.get('schema_version')!='pfs-desi-cosmology-qualification-v1' or set(q['cases'])!={'B','C'}:
        raise ValueError('complete B/C qualification required; a local checkpoint is not a cohort')
    loaded,refined=_validate_cases(q)
    shell_profile=shell_profile_from_record(q['inventory'])
    if _inventory_profile(q['inventory'])==INTRINSIC_PROFILE:
        ratios=_ap_resolution_ratios(refined)
        if canonical(q.get('ap_resolution_ratios'))!=canonical(ratios):
            raise ValueError('saved AP-resolution error ratios differ from their references')
        assessment='historical_ap_assessment' if shell_profile==LEGACY_SHELL_PROFILE else 'candidate_shell_ap8192_assessment'
        if q.get(assessment)!=historical_ap_assessment(q['cases'],ratios,shell_profile=shell_profile):
            raise ValueError('saved historical AP assessment differs from absolute errors or gain ratios')
        for a,b in AP_OUTPUT_COMPARISONS[1:]:
            require_passed(ratios[f'{a}_to_{b}'],'AP C/B and B/C error-ratio convergence')
    if shell_profile!=LEGACY_SHELL_PROFILE:
        if 'historical_ap_assessment' in q: raise ValueError('candidate AP assessment mislabeled as original-grid history')
        recomputed=_grid_resolution_reports(loaded,refined,q['provenance'])
        if any(canonical(q.get(key))!=canonical(value) for key,value in recomputed.items()):
            raise ValueError('saved shell/angular ratios or historical assessment differ from authenticated controls')
    for previous in q.get('resumed_from',[]):
        _pin(previous['checkpoint'],previous['checkpoint_sha256'],'resume checkpoint')
        if file_hash(previous['execution_path'])!=previous['execution_sha256']:
            raise ValueError('original resume exit evidence changed')
        validate_observed_exit(previous['execution'])
    b,c=loaded['B'],loaded['C']
    require_passed(compare_nested_likelihoods(likelihood_snapshot(b[0]),likelihood_snapshot(c[0])),'likelihood nesting')
    common=set(b[1])&set(c[1]); space=make_cosmology_spec(b[0].reference_cosmology,extended=True,
                                                      ap_nk=b[0].config.ap_interpolation_nk)
    for t in common:
        records=[{'cosmology_columns':list(space.names),'treatment':t,'free_cosmology':list(r.free_cosmology),
                  'growth_z':r.growth_z,'growth_jacobian':r.growth_jacobian,'cov_cosmo_marg':r.cov_cosmo_marg}
                 for r in (b[1][t],c[1][t])]
        require_passed(compare_marginal_errors(*records,space=space),'common marginal errors')
        if shell_profile!=LEGACY_SHELL_PROFILE: continue
        for label in ('shell','angular'):
            for field in ('cov_cosmo_marg','growth_cov_marg'):
                base_b=np.sqrt(np.diag(getattr(b[1][t],field))); base_c=np.sqrt(np.diag(getattr(c[1][t],field)))
                rb=np.sqrt(np.diag(getattr(refined['B'][label][t],field)))
                rc=np.sqrt(np.diag(getattr(refined['C'][label][t],field)))
                active=base_b>0
                if np.any(np.abs((rb[active]/rc[active])/(base_b[active]/base_c[active])-1)>=.01):
                    raise ValueError('resolution B/C gain changes by at least 1%')
    return q,loaded


def validate_observed_exit(record):
    if (type(record.get('returncode')) is not int or not record.get('command')
            or record.get('status')!=('passed' if record['returncode']==0 else 'failed')
            or file_hash(record['log'])!=record['log_sha256']):
        raise ValueError('original observed process exit/log evidence is missing or changed')


def load_checkpoint(path,pin,*,provenance,oracle_pin,profile=LEGACY_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER,shell_profile=LEGACY_SHELL_PROFILE):
    checkpoint=_pin(path,pin,'resume checkpoint')
    if (checkpoint.get('schema_version')!='pfs-desi-cosmology-case-checkpoint-v1'
            or checkpoint.get('status')!='local_cases_validated'):
        raise ValueError('resume requires a completed-case checkpoint')
    _inventory_profile(checkpoint['inventory'],requested=profile,fisher_solver=fisher_solver,shell_profile=shell_profile)
    if solver_from_record(checkpoint['provenance'])!=solver_from_record(provenance):
        raise ValueError('resume Fisher solver identity mismatch')
    if (identity_provenance(checkpoint['provenance'])!=identity_provenance(provenance)
            or checkpoint['oracle_export_sha256']!=oracle_pin):
        raise ValueError('resume source/runtime/input/oracle identity mismatch')
    execution_path=Path(checkpoint['origin_execution'])
    execution=read_json(execution_path); validate_observed_exit(execution)
    _validate_cases(checkpoint)
    previous={'checkpoint':str(Path(path).resolve()),'checkpoint_sha256':pin,
              'execution_path':str(execution_path),'execution_sha256':file_hash(execution_path),
              'execution':execution}
    return checkpoint['cases'],checkpoint.get('resumed_from',[])+[previous]


def _execution_command(source,interpreter,selected,q,role):
    run=selected['run_root']
    phase='produce' if role=='reload' else selected['phase']
    command=[interpreter,'-B',source,'--phase',phase,'--output-root',run]
    if q.get('inventory',{}).get('numerical_profile') is not None:
        profile=_inventory_profile(q['inventory'])
        command+=['--numerical-profile',profile]
    shell=shell_profile_from_record(q['inventory'])
    if shell!=LEGACY_SHELL_PROFILE: command+=['--shell-profile',shell]
    solver=solver_from_record(q['inventory'])
    if solver!=LEGACY_FISHER_SOLVER: command+=['--fisher-solver',solver]
    if role=='reload':
        command+=['--selection',selected['qualification'],'--selection-sha256',selected['qualification_sha256'],'--reload']
    elif phase=='qualify':
        command+=['--oracle-export',q['oracle_export'],'--oracle-export-sha256',q['oracle_export_sha256']]
        if selected.get('resume'):
            command+=['--resume',selected['resume'],'--resume-sha256',selected['resume_sha256']]
        command+=['--worker']
    else:
        command+=['--selection',selected['upstream_selection'],'--selection-sha256',selected['upstream_selection_sha256'],'--worker']
    return command


def validate_execution_binding(selected,q,role):
    record=selected[role]; validate_exit(record)
    completion=_pin(record['completion_path'],record['completion_sha256'],'execution completion')
    source=q['provenance']['driver_source']
    if (completion!=record.get('completion') or completion.get('source')!=source
            or q['provenance']['source_file_hashes'].get(source['path'])!=source['sha256']
            or file_hash(source['path'])!=source['sha256']):
        raise ValueError('execution source/completion identity mismatch')
    expected=_execution_command(source['path'],q['provenance']['runtime']['interpreter'],selected,q,role)
    if (record['command']!=expected or completion.get('command')!=expected
            or completion.get('phase')!=('produce' if role=='reload' else selected['phase'])
            or completion.get('role')!=role
            or completion.get('schema_version')!='pfs-desi-cosmology-completion-v1'
            or completion.get('qualification')!=selected['qualification']
            or completion.get('qualification_sha256')!=selected['qualification_sha256']
            or Path(record['completion_path']).resolve()!=Path(selected['run_root'])/(role+'-completion.json')
            or Path(record['log']).resolve()!=Path(selected['run_root'])/role/'process.log'
            or record.get('cwd')!=str(ROOT)):
        raise ValueError('execution command/completion is not bound to selected qualification')


def emit_completion(args,qpath,qpin,role):
    # This is the already validated identity, never a newly computed replacement pin.
    _pin(qpath,qpin,'qualification')
    source={'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)}
    record={'schema_version':'pfs-desi-cosmology-completion-v1','phase':args.phase,'role':role,
        'qualification':str(Path(qpath).resolve()),'qualification_sha256':qpin,
        'command':list(sys.orig_argv),'source':source}
    write_json(args.output_root/(role+'-completion.json'),record)


def read_selection(path,pin,*,kind):
    selected=_pin(path,pin,'selection')
    if (selected.get('schema_version')!='pfs-desi-cosmology-selection-v1'
            or selected.get('status')!='complete' or selected.get('phase')!=kind):
        raise ValueError('selection phase/status mismatch')
    if kind=='produce':
        if not selected.get('upstream_selection') or not selected.get('upstream_selection_sha256'):
            raise ValueError('missing authenticated upstream qualification selection')
        upstream,q,loaded=read_selection(selected['upstream_selection'],selected['upstream_selection_sha256'],kind='qualify')
        if any(selected[key]!=upstream[key] for key in ('qualification','qualification_sha256')):
            raise ValueError('upstream qualification identity mismatch')
        if selected.get('resume') or selected.get('resume_sha256'):
            raise ValueError('produce selection cannot introduce resume inputs')
    else:
        if selected.get('upstream_selection') or selected.get('upstream_selection_sha256'):
            raise ValueError('qualification selection cannot have upstream production selection')
        if Path(selected['qualification'])!=Path(selected['run_root'])/'qualification.json':
            raise ValueError('qualification output path differs from its worker run')
        q,loaded=validate_qualification(selected['qualification'],selected['qualification_sha256'])
        resumed=q.get('resumed_from',[])
        expected=(resumed[-1]['checkpoint'],resumed[-1]['checkpoint_sha256']) if resumed else (None,None)
        if (selected.get('resume'),selected.get('resume_sha256'))!=expected:
            raise ValueError('qualification resume command identity mismatch')
    for role in ('worker','reload'): validate_execution_binding(selected,q,role)
    _pin(selected['qualification'],selected['qualification_sha256'],'qualification')
    _pin(path,pin,'selection')
    return selected,q,loaded


def report(selection,pin,output,*,profile=LEGACY_PROFILE,fisher_solver=LEGACY_FISHER_SOLVER,shell_profile=LEGACY_SHELL_PROFILE):
    selected,q,loaded=read_selection(selection,pin,kind='produce')
    _inventory_profile(q['inventory'],requested=profile,fisher_solver=fisher_solver,shell_profile=shell_profile)
    b={r['treatment']:r for r in loaded['B'][2]}; c={r['treatment']:r for r in loaded['C'][2]}
    rows=[]
    for t in COSMOLOGY_TREATMENTS:
        if t not in b or t not in c:
            rows.append({'treatment':t,'status':'failed','B':b.get(t),'C':c.get(t),'failures':{case:loaded[case][3]['failed_treatments'].get(t)
                                                                  for case in ('B','C')}}); continue
        rows.append({'treatment':t,'status':'complete','B':b[t],'C':c[t],
            'native_B_to_C_gain':{n:b[t]['native_sigma'][n]/c[t]['native_sigma'][n] for n in b[t]['native_sigma']},
            'growth_B_to_C_gain':[x/y for x,y in zip(b[t]['growth_sigma'],c[t]['growth_sigma'])]})
    result={**inventory(profile=_inventory_profile(q['inventory']),fisher_solver=fisher_solver,shell_profile=shell_profile),'treatment_comparisons':rows,'selection_sha256':pin,
            'qualification_sha256':selected['qualification_sha256']}
    if profile==INTRINSIC_PROFILE:
        assessment='historical_ap_assessment' if shell_profile==LEGACY_SHELL_PROFILE else 'candidate_shell_ap8192_assessment'
        result[assessment]=historical_ap_assessment(q['cases'],q['ap_resolution_ratios'],shell_profile=shell_profile)
        result.update(ap_resolution_comparisons={case:q['cases'][case]['ap_references']['comparisons'] for case in ('B','C')},
            ap_resolution_ratios=q['ap_resolution_ratios'])
    if shell_profile!=LEGACY_SHELL_PROFILE:
        result.update(original_grid_replay={case:q['cases'][case]['ap_references']['original_grid_replay'] for case in ('B','C')},
                      shell_grid_assessment=q['shell_grid_assessment'],resolution_comparisons=q['resolution_comparisons'])
    if fisher_solver==DECIMAL_FISHER_SOLVER:
        controls={case:(q['cases'][case]['ap_references']['original_grid_replay']['controls']['original_grid_native3_8192']
                       if shell_profile!=LEGACY_SHELL_PROFILE else
                       q['cases'][case]['ap_references']['controls']['native3_8192']
                       if profile==INTRINSIC_PROFILE else q['cases'][case]['baseline3']) for case in ('B','C')}
        result['fisher_precision_assessment']={
            'scope':'accurate inversion of unchanged historical Fisher matrices; separate from AP grid convergence',
            'historical_covariances_preserved':True,'fisher_solver':fisher_solver_record(fisher_solver),
            'cases':{case:control['historical_derived_reference']['receipt'] for case,control in controls.items()}}
    write_json(output,result); return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['B','C']); p.add_argument('--phase',choices=('prepare','qualify','produce','report'),required=True)
    p.add_argument('--output-root',type=Path,required=True); p.add_argument('--oracle-export',type=Path)
    p.add_argument('--oracle-export-sha256'); p.add_argument('--selection',type=Path); p.add_argument('--selection-sha256')
    p.add_argument('--resume',type=Path); p.add_argument('--resume-sha256')
    p.add_argument('--numerical-profile',choices=PRODUCTION_PROFILES,default=LEGACY_PROFILE)
    p.add_argument('--fisher-solver',choices=FISHER_SOLVERS,default=LEGACY_FISHER_SOLVER)
    p.add_argument('--shell-profile',choices=SHELL_PROFILES,default=LEGACY_SHELL_PROFILE)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--reload',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    validate_shell_selection(args.shell_profile,args.numerical_profile,args.fisher_solver)
    if bool(args.resume)!=bool(args.resume_sha256) or (args.resume and args.phase!='qualify'):
        raise ValueError('resume requires qualify phase and external checkpoint pin')
    if args.cases!=['B','C']: raise ValueError('exactly ordered cases B C required')
    if args.phase=='prepare': prepare(args.output_root,profile=args.numerical_profile,fisher_solver=args.fisher_solver,shell_profile=args.shell_profile); return 0
    if args.phase=='qualify' and (not args.oracle_export or not args.oracle_export_sha256):
        raise ValueError('qualification needs externally pinned oracle export')
    if args.phase in ('produce','report') and (not args.selection or not args.selection_sha256):
        raise ValueError('externally pinned selection required')
    if args.phase=='report':
        args.output_root.mkdir(parents=True,exist_ok=True)
        report(args.selection,args.selection_sha256,args.output_root/f'report-{uuid.uuid4().hex}.json',
               profile=args.numerical_profile,fisher_solver=args.fisher_solver,shell_profile=args.shell_profile); return 0
    if args.reload:
        q,_=validate_qualification(args.selection,args.selection_sha256)
        _inventory_profile(q['inventory'],requested=args.numerical_profile,fisher_solver=args.fisher_solver,shell_profile=args.shell_profile)
        emit_completion(args,args.selection,args.selection_sha256,'reload'); return 0
    if args.worker:
        if args.phase=='qualify': qpath,qpin=qualify_worker(args)
        else:
            upstream,q,_=read_selection(args.selection,args.selection_sha256,kind='qualify')
            _inventory_profile(q['inventory'],requested=args.numerical_profile,fisher_solver=args.fisher_solver,shell_profile=args.shell_profile)
            qpath,qpin=upstream['qualification'],upstream['qualification_sha256']
        emit_completion(args,qpath,qpin,'worker'); return 0
    upstream=None
    if args.phase=='produce':
        upstream=_pin(args.selection,args.selection_sha256,'upstream selection')
        qpath=Path(upstream['qualification']).resolve(); qpin=upstream['qualification_sha256']
        q=_pin(qpath,qpin,'qualification')
        _inventory_profile(q['inventory'],requested=args.numerical_profile,fisher_solver=args.fisher_solver,shell_profile=args.shell_profile)
    run=(args.output_root/f'{args.phase}-{uuid.uuid4().hex}').resolve(); run.mkdir(parents=True)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--phase',args.phase,'--output-root',str(run)]
    if args.numerical_profile!=LEGACY_PROFILE:
        command+=['--numerical-profile',args.numerical_profile]
    if args.shell_profile!=LEGACY_SHELL_PROFILE: command+=['--shell-profile',args.shell_profile]
    if args.fisher_solver!=LEGACY_FISHER_SOLVER: command+=['--fisher-solver',args.fisher_solver]
    if args.phase=='qualify': command+=['--oracle-export',str(args.oracle_export.resolve()),'--oracle-export-sha256',args.oracle_export_sha256]
    else: command+=['--selection',str(args.selection.resolve()),'--selection-sha256',args.selection_sha256]
    if args.resume: command+=['--resume',str(args.resume.resolve()),'--resume-sha256',args.resume_sha256]
    worker=observe_process(command+['--worker'],run/'worker',completion_path=run/'worker-completion.json')
    if worker['returncode']: return worker['returncode']
    if args.phase=='qualify':
        qpath=run/'qualification.json'; qpin=worker['completion']['qualification_sha256']
    else: _pin(args.selection,args.selection_sha256,'upstream selection')
    q=_pin(qpath,qpin,'qualification')
    selected={'schema_version':'pfs-desi-cosmology-selection-v1','phase':args.phase,'run_root':str(run),
        'status':'complete','worker':worker,'qualification':str(qpath.resolve()),'qualification_sha256':qpin,
        'upstream_selection':str(args.selection.resolve()) if args.selection else None,
        'upstream_selection_sha256':args.selection_sha256,
        'resume':str(args.resume.resolve()) if args.resume else None,'resume_sha256':args.resume_sha256}
    validate_execution_binding(selected,q,'worker')
    reload_command=[sys.executable,'-B',str(Path(__file__).resolve()),'--phase','produce','--output-root',str(run)]
    if args.numerical_profile!=LEGACY_PROFILE:
        reload_command+=['--numerical-profile',args.numerical_profile]
    if args.shell_profile!=LEGACY_SHELL_PROFILE: reload_command+=['--shell-profile',args.shell_profile]
    if args.fisher_solver!=LEGACY_FISHER_SOLVER: reload_command+=['--fisher-solver',args.fisher_solver]
    reload_command+=['--selection',str(qpath.resolve()),'--selection-sha256',qpin,'--reload']
    reload=observe_process(reload_command,run/'reload',completion_path=run/'reload-completion.json')
    if reload['returncode']: return reload['returncode']
    selected['reload']=reload
    validate_execution_binding(selected,q,'reload')
    _pin(qpath,qpin,'qualification')
    if args.selection: _pin(args.selection,args.selection_sha256,'upstream selection')
    write_json(run/'selection.json',selected)
    print(str(run/'selection.json'),file_hash(run/'selection.json'),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
