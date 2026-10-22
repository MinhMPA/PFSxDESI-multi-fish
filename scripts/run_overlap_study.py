"""Resumable, authenticated overlap-area and residual cross-stochastic forecasts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np

from pfsfog.extended_artifacts import (canonical, jsonable, pack_likelihood,
                                      PRECISION_FIELDS, GROWTH_FIELDS)
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from scripts.overlap_inputs import (PROJECT_ROOT, LABELS, Condition, conditions,
                                   canonical_condition, load_inputs)
from scripts.overlap_study import (select_observations, scale_area,
                                  fix_cross_coefficients, infer_saved_growth)
from scripts.overlap_checks import check_marginalization, _scaled_error
from scripts.run_elf_studies import GRID_PROFILES

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'overlap-information-v1'
SPEC = PROJECT_ROOT/'docs/revision/specs/2026-10-19-cross-stochasticity-super-pfs.md'
OWN_FILES = ('scripts/overlap_inputs.py', 'scripts/overlap_study.py',
             'scripts/overlap_checks.py', 'scripts/run_overlap_study.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    """Atomically publish each receipt after its numerical files are flushed."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('missing or unreadable campaign evidence: '+str(path)) from error


def condition_id(c):
    return f'{c.case}-area{c.area_deg2:05d}-{c.stochastic}'


def _collect_sources(source_root):
    records = {}
    for grid in GRID_PROFILES:
        print(f'{utc()} authenticating physical references: {grid}', flush=True)
        inputs = load_inputs(source_root, grid)
        if inputs.grid != grid or inputs.source_root != Path(source_root).absolute():
            raise ValueError('mixed source grid or root')
        records[grid] = inputs.hashes
    return records


def _bound_code():
    try:
        return {str(p): digest(p) for p in [*(ROOT/n for n in OWN_FILES), SPEC, LABELS]}
    except OSError as error:
        raise ValueError('missing bound numerical module or scientific specification') from error


def prepare(source_root: Path, output: Path) -> dict:
    output = Path(output).absolute(); source_root = Path(source_root).absolute()
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing calculation')
    sources = _collect_sources(source_root)
    request = {'schema': SCHEMA, 'source_root':str(source_root), 'grids': GRID_PROFILES,
               'conditions':[asdict(c) for c in conditions()], 'code_hashes':_bound_code(),
               'source_hashes':sources, 'growth_definition':'derived f sigma8(z); headline z=1.1',
               'solver':DECIMAL_FISHER_SOLVER, 'prior':'once after regional likelihood sum; nuisance factor 1',
               'created_utc':utc(), 'source_head':subprocess.check_output(
                   ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    output.mkdir(parents=True)
    write(output/'request.json',request)
    state = {'schema':SCHEMA, 'request_sha256':digest(output/'request.json'), 'status':'prepared',
             'created_utc':utc(), 'updated_utc':utc(), 'pid':None, 'active_grid':None,
             'active_condition':None, 'completed_records':0, 'exit_status':None, 'grids':{}}
    write(output/'state.json',state)
    return {'status':'prepared','expected_records':160,'expected_distinct':75}


def _verify_bindings(bindings):
    for path, expected in bindings.items():
        try: actual = digest(path)
        except OSError as error: raise ValueError('missing bound physical input: '+path) from error
        if actual != expected:
            raise ValueError('changed bound physical input: '+path)


def _request(output, *, verify_sources=True):
    output = Path(output)
    request, state = read(output/'request.json'), read(output/'state.json')
    if digest(output/'request.json') != state.get('request_sha256'):
        raise ValueError('changed campaign request')
    if (request.get('schema') != SCHEMA or state.get('schema') != SCHEMA
            or request.get('grids') != GRID_PROFILES
            or canonical(request.get('conditions')) != canonical([asdict(c) for c in conditions()])
            or request.get('solver') != DECIMAL_FISHER_SOLVER
            or request.get('code_hashes') != _bound_code()
            or set(request.get('source_hashes',{})) != set(GRID_PROFILES)):
        raise ValueError('incompatible physical campaign or numerical modules')
    if verify_sources:
        bindings = {}
        for hashes in request['source_hashes'].values():
            if not hashes: raise ValueError('empty physical input inventory')
            for p,h in hashes.items():
                if p in bindings and bindings[p] != h: raise ValueError('mixed source hashes')
                bindings[p] = h
        _verify_bindings(bindings)
    return request, state


def _validate_grid(output, grid, row, request, request_sha):
    records = row.get('records',[]); ids = [r['condition_id'] for r in records]
    expected = {condition_id(c):c for c in conditions()}
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected):
        raise ValueError('duplicate or unknown condition IDs')
    canonical_ids = {condition_id(canonical_condition(expected[n])) for n in ids}
    for record in records:
        name = record['condition_id']; c = expected[name]
        path = output/'inference'/grid/name
        meta = read(path/'manifest.json'); check = read(path/'checks.json')
        target = condition_id(canonical_condition(c))
        if (digest(path/'manifest.json') != record['manifest_sha256']
                or meta.get('schema') != SCHEMA or meta.get('grid') != grid
                or meta.get('condition_id') != name or meta.get('condition') != asdict(c)
                or meta.get('canonical_condition') != asdict(canonical_condition(c))
                or meta.get('request_sha256') != request_sha
                or meta.get('source_hashes') != request['source_hashes'][grid]
                or meta.get('status') != 'passed' or meta.get('consumable') is not True
                or check.get('status') != 'passed'
                or digest(path/'checks.json') != meta.get('checks_sha256')):
            raise ValueError('failed, mixed or changed condition evidence: '+name)
        if name != target:
            if (meta.get('alias_of') != target or target not in ids
                    or check.get('equivalence',{}).get('status') != 'passed'):
                raise ValueError('incomplete or unjustified alias: '+name)
            parent = read(output/'inference'/grid/target/'manifest.json')
            if (parent.get('alias_of') or parent.get('status') != 'passed'
                    or meta.get('alias_manifest_sha256') != digest(output/'inference'/grid/target/'manifest.json')):
                raise ValueError('alias does not reference a distinct authenticated result')
        elif meta.get('alias_of') or digest(path/'arrays.npz') != meta.get('arrays_sha256'):
            raise ValueError('changed canonical result arrays: '+name)
    if row.get('status') == 'passed':
        checks = read(output/'inference'/grid/'checks.json')
        if (set(ids) != set(expected) or len(canonical_ids) != 15
                or checks.get('status') != 'passed'
                or checks.get('area_identities',{}).get('status') != 'passed'
                or digest(output/'inference'/grid/'checks.json') != row.get('checks_sha256')):
            raise ValueError('incomplete named/distinct inventory or physical checks')
    return len(ids), len({n for n in ids if n == condition_id(canonical_condition(expected[n]))})


def status(output: Path) -> dict:
    output = Path(output).absolute(); request, state = _request(output)
    totals = [0,0]
    for grid,row in state['grids'].items():
        if grid not in GRID_PROFILES: raise ValueError('mixed source grid')
        counts = _validate_grid(output,grid,row,request,state['request_sha256'])
        totals = [a+b for a,b in zip(totals,counts)]
    complete = (set(state['grids']) == set(GRID_PROFILES)
                and all(r['status']=='passed' for r in state['grids'].values())
                and totals == [160,75])
    if state['status']=='passed' and not complete:
        raise ValueError('false campaign completion')
    alive = False
    if state.get('pid') and state['status']=='running':
        try: os.kill(state['pid'],0); alive=True
        except ProcessLookupError: pass
    return {**state, 'status':'passed' if complete else state['status'],
            'process_alive':alive, 'completed_records':totals[0], 'completed_distinct':totals[1],
            'expected_records':160, 'expected_distinct':75,
            'qualification':'per-grid checks only; refinement comparison pending'}


def _equivalence(like, other):
    if like.names != other.names or canonical(like.parameters) != canonical(other.parameters):
        raise ValueError('alias free parameters or priors differ')
    error = _scaled_error(like.F_like,other.F_like)
    if not np.isfinite(error) or error >= 1e-10:
        raise ValueError('alias likelihood information differs')
    return {'status':'passed','fisher_scaled_error':error,
            'reason':'identical named likelihood information and priors; absent cross family or DESI area partition'}


def _area_checks(selected, areas):
    errors = {}; low = {}
    for case in ('A','B','C','D'):
        source=selected[case]
        if _scaled_error(areas[case,1200].F_like,source.F_like)>=1e-10:
            raise ValueError('1200 deg2 did not recover the source information')
        for area in (1200,14000):
            like=areas[case,area]
            if like.names != source.names: raise ValueError('area changed shared nuisance identities')
            low[f'{case}-{area}'] = _scaled_error(like.bin_F_like[0], source.bin_F_like[0])
            if low[f'{case}-{area}'] >= 1e-10: raise ValueError('DESI below z=0.6 changed')
            for original,cell in zip(source.cells,like.cells):
                w=cell.metadata['area_scaling']['weight']
                if cell.metadata['volume'] != w*original.metadata['volume']:
                    raise ValueError('area-volume identity failed')
                if w:
                    if (not np.array_equal(original.derivatives,cell.derivatives)
                            or not np.array_equal(original.fiducial_mean,cell.fiducial_mean)
                            or not np.array_equal(original.covariance/w,cell.covariance)):
                        raise ValueError('area changed mean, response or covariance weight')
                elif cell.names or cell.derivatives.shape[1] or cell.covariance.shape[1]:
                    raise ValueError('zero-volume region retained measured information')
        if case in ('A','B'):
            errors[case]=_scaled_error(areas[case,14000].F_like,areas[case,1200].F_like)
            if errors[case]>=1e-10: raise ValueError('DESI information depends on PFS area')
    return {'status':'passed','desi_area_scaled_errors':errors,'low_redshift_scaled_errors':low,
            'volume_response_covariance_identities':'passed for every cell and both areas'}


def _save_result(path, result, diagnostics):
    arrays={}; meta=pack_likelihood(result.likelihood,arrays)
    for key in (*PRECISION_FIELDS,*GROWTH_FIELDS): arrays['result/'+key]=getattr(result,key)
    metrics={}
    for key,value in diagnostics.items():
        if isinstance(value,np.ndarray): arrays['diagnostics/'+key]=value
        else: metrics[key]=value
    np.savez_compressed(path/'arrays.npz',**arrays)
    return {'likelihood':meta, 'parameters':result.parameters,
            'nuisance_factor':result.nuisance_factor, 'validation':result.validation,
            'fisher_solver':result.fisher_solver, 'arrays_sha256':digest(path/'arrays.npz')}, metrics


def run_grid(output: Path, grid: str) -> dict:
    output=Path(output).absolute(); request,state=_request(output)
    if grid not in GRID_PROFILES: raise ValueError('unknown numerical grid')
    for oldgrid,row in state['grids'].items():
        _validate_grid(output,oldgrid,row,request,state['request_sha256'])
    old=state['grids'].get(grid,{})
    if old.get('status')=='passed': return old
    if state['status']=='running' and state.get('pid'):
        try: os.kill(state['pid'],0)
        except ProcessLookupError: pass
        else: raise ValueError('another numerical calculation is still running')
    if old.get('status')=='running' and old.get('attempt'):
        write(Path(old['attempt'])/'interruption.json', {**old,
            'active_condition':old.get('active_condition'), 'observed_interrupted_utc':utc(),
            'request_sha256':state['request_sha256'], 'reason':'recorded process no longer alive'})
    dest=output/'inference'/grid; dest.mkdir(parents=True,exist_ok=True)
    known={r['condition_id'] for r in old.get('records',[])}
    for path in dest.iterdir():
        if path.is_dir() and path.name!='attempts' and path.name not in known:
            if (path/'manifest.json').exists():
                raise ValueError('unrecorded partial output requires diagnosis: '+str(path))
    attempts=dest/'attempts'; attempts.mkdir(exist_ok=True)
    attempt=attempts/f'{len(list(attempts.iterdir()))+1:03d}'; attempt.mkdir()
    row={'status':'running','records':old.get('records',[]),'pid':os.getpid(),
         'start_utc':utc(),'attempt':str(attempt),'elapsed_seconds':0,'active_condition':None}
    write(attempt/'start.json', {**row, 'grid':grid,
        'request_sha256':state['request_sha256'], 'resumed_from_status':old.get('status')})
    state['grids'][grid]=row
    started=time.monotonic()
    log=output/'logs'/f'{grid}.log'; log.parent.mkdir(exist_ok=True)
    def progress(message):
        state.update(status='running',pid=os.getpid(),active_grid=grid,
                     active_condition=row['active_condition'],updated_utc=utc(),exit_status=None)
        row['updated_utc']=state['updated_utc']; row['elapsed_seconds']=time.monotonic()-started
        state['completed_records']=sum(len(r['records']) for r in state['grids'].values())
        write(output/'state.json',state)
        line=f'{utc()} PID={os.getpid()} {grid} {state.get("active_condition")} completed={len(row["records"])}/32 {message}'
        print(line,flush=True)
        with log.open('a') as stream: stream.write(line+'\n'); stream.flush()
    progress('authenticating physical source')
    active=None
    try:
        inputs=load_inputs(Path(request['source_root']),grid)
        if inputs.grid!=grid or inputs.hashes!=request['source_hashes'][grid]:
            raise ValueError('mixed or changed authenticated source grid')
        reference=inputs.results['C21']
        selected={c:select_observations(reference.likelihood,c) for c in ('A','B','C','D')}
        areas={(c,a):scale_area(selected[c],a) for c in selected for a in (1200,14000)}
        area_checks=_area_checks(selected,areas)
        cached={}; completed={r['condition_id'] for r in row['records']}
        for c in conditions():
            name=condition_id(c); cc=canonical_condition(c); target=condition_id(cc)
            if name in completed: continue
            active=name; row['active_condition']=name; progress('selecting free cross-stochastic coefficients')
            like,fixed=fix_cross_coefficients(areas[c.case,c.area_deg2],c.stochastic)
            if target not in cached:
                cached[target]=fix_cross_coefficients(areas[cc.case,cc.area_deg2],cc.stochastic)[0]
            equivalent=_equivalence(like,cached[target])
            path=dest/name
            if path.exists():
                # Preserve an interrupted publication; a fresh receipt cannot mix with it.
                path.rename(attempt/(name+'-incomplete'))
            path.mkdir()
            checks={'status':'passed','equivalence':equivalent,'area_identities':area_checks}
            meta={'schema':SCHEMA,'status':'passed','consumable':True,'grid':grid,
                  'condition_id':name,'condition':asdict(c),'canonical_condition':asdict(cc),
                  'request_sha256':state['request_sha256'],'source_hashes':inputs.hashes,
                  'fixed_names':list(fixed),'fixed_values':[0.]*len(fixed),
                  'free_parameters':like.names,'prior_identity':request['prior'],
                  'solver_identity':request['solver'],'growth_definition':request['growth_definition'],
                  'selection_maps':{x.metadata['cell_id']:x.metadata['area_scaling']['source_metadata']['selection'] for x in like.cells},
                  'area_weights':{x.metadata['cell_id']:x.metadata['area_scaling']['weight'] for x in like.cells}}
            if name!=target:
                parent=read(dest/target/'manifest.json')
                if parent.get('status')!='passed': raise ValueError('canonical solve not consumable')
                meta.update(alias_of=target,alias_manifest_sha256=digest(dest/target/'manifest.json'))
            else:
                progress('Decimal inverse and independent Schur/QR checks')
                result=infer_saved_growth(like,reference)
                diagnostics=check_marginalization(result)
                packed,metrics=_save_result(path,result,diagnostics)
                meta.update(packed); checks['marginalization']=metrics
            write(path/'checks.json',checks); meta['checks_sha256']=digest(path/'checks.json')
            write(path/'manifest.json',meta)
            row['records'].append({'condition_id':name,'manifest_sha256':digest(path/'manifest.json')})
            progress('saved and checked')
        _request(output)
        gridchecks={'status':'passed','area_identities':area_checks,
                    'named_records':32,'distinct_results':15}
        write(dest/'checks.json',gridchecks)
        row.update(status='passed',finish_utc=utc(),exit_status=0,
                   elapsed_seconds=time.monotonic()-started,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   checks_sha256=digest(dest/'checks.json'))
        _validate_grid(output,grid,row,request,state['request_sha256'])
        state.update(status='passed' if len(state['grids'])==5 and all(r['status']=='passed' for r in state['grids'].values()) else 'partial',
                     active_grid=None,active_condition=None,updated_utc=utc(),finish_utc=utc(),exit_status=0)
        write(output/'state.json',state); write(attempt/'outcome.json',row)
        return row
    except BaseException as error:
        failure={'status':'failed','consumable':False,'condition_id':active,'error':str(error),
                 'traceback':traceback.format_exc(),'utc':utc(),'pid':os.getpid()}
        write(attempt/'error.json',failure)
        if active and (dest/active).is_dir() and not (dest/active/'manifest.json').exists():
            write(dest/active/'error.json',failure)
        row.update(status='failed',error=str(error),exit_status=1,finish_utc=utc(),
                   elapsed_seconds=time.monotonic()-started)
        state.update(status='failed',exit_status=1,updated_utc=utc())
        write(output/'state.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run-grid');p.add_argument('--output',type=Path,required=True);p.add_argument('--grid',choices=list(GRID_PROFILES),required=True)
    p=sub.add_parser('status');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare': result=prepare(args.source,args.output)
    elif args.action=='run-grid': result=run_grid(args.output,args.grid)
    else: result=status(args.output)
    print(json.dumps(jsonable(result),sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
