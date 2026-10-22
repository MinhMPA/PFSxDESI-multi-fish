#!/usr/bin/env python3
"""Recover receipt-selected B/C response arrays using only frozen numerical code.

Invoke this standalone file with frozen cwd/PYTHONPATH. Its supervisor records the
worker's observed exit and performs a second, independently pinned verification.
No numerical imports occur until selection, receipts and environment authenticate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ORACLE=Path('/Users/nguyenmn/PFSxDESI-multi-fish/.worktrees/jcap-revision-20260930')
RECOVERY=ORACLE.parent/'runtime-recovery-20261001/venv'
HEAD='d8dfcf4ab3e390b0365a98af365a511f6d7c1e9e'
SELECTION_PIN='f71de20a8f6f26cc6d2d6e03b1ea2029be9a77cec698c5c4e0098c334ce329a2'
RECEIPTS={'B':'36d059b7e37d3be5e7bc02170ed61d04e6e900857594d0c049e93243562e9176',
          'C':'d1da90a13b45b9cd02747ceffbb1c393860f5e0a67e2e5ffd44790a27d2801ae'}
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write_json(path,value):
    payload=canonical(value)+b'\n'; Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def git(*args):
    return subprocess.run(['git',*args],cwd=ORACLE,check=True,capture_output=True,text=True).stdout


def check_environment(cwd=None):
    require((Path.cwd() if cwd is None else Path(cwd))==ORACLE,'run from frozen oracle cwd')
    require(Path(sys.prefix)==RECOVERY,'use reviewed recovery interpreter')
    require(sys.dont_write_bytecode,'disable bytecode writes')
    require(os.environ.get('PYTHONPATH')==f'{ORACLE}:{ORACLE}/ps_1loop_jax','frozen PYTHONPATH required')
    require('XLA_FLAGS' not in os.environ,'unset XLA_FLAGS')
    require(all(os.environ.get(k)=='1' for k in THREADS),'single-thread environment required')
    require(git('rev-parse','HEAD').strip()==HEAD,'frozen HEAD changed')
    require(not git('status','--porcelain=v1','--untracked-files=all','--','pfsfog','scripts','tests'),'frozen numerical source changed')


EMULATOR_ROOT=Path('/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024')
EMULATOR_MODEL='jense_2023_camb_mnu'
EMULATOR_FILES={
    'jense_2023_pk_camb_mnu.yaml':'771bd7184c430540f167bcb39e8e1bdeb4725fa158aad97d5cbfb3d5ed915e3e',
    'jense_2023_camb_mnu/networks/jense_2023_camb_mnu_Pk_lin.npz':'a8b5b12d11131fce8c64c9d1e0f65b6dd0a18f95541943a0f2f2e04738fad3e2',
    'jense_2023_camb_mnu/networks/jense_2023_camb_mnu_sigma8.npz':'6b2d8039497a11ecd6e36237aaedcc54cddb02e554c2321b6c1d1feefdd35b95'}
INVENTORY_KEYS=(('source_file_hashes','source_content_hash'),('backend_file_hashes','backend_snapshot'),
                ('survey_input_hashes','survey_input_hash'),('emulator_input_hashes','network_hash'))


def emulator_locator():
    """Load only the stdlib locator, including from frozen-oracle verification."""
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'pfsfog/emulator_paths.py'
    spec=importlib.util.spec_from_file_location('_pfs_emulator_locator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def scientific_inventory(root,*,extended,input_role='active',relocation_record=None):
    """Derive complete approved input sets independently of a supplied record."""
    root=Path(root).resolve()
    source=[*sorted((root/'pfsfog').glob('*.py')),root/'scripts/run_revision.py',root/'scripts/validate_revision.py']
    if extended: source += [root/'scripts/run_extended_cosmology.py',root/'scripts/export_cosmology_oracle.py']
    locator=emulator_locator();extra={}
    if input_role=='active':
        require(relocation_record is None,'active inputs must not be relabelled as historical')
        expected_networks=locator.active_emulator_inputs()
        extra['active_emulator_provenance']=locator.active_emulator_provenance()
    elif input_role=='historical':
        expected_networks=locator.historical_emulator_inputs()
        manifest=json.loads((root/'docs/revision/evidence/emulator-provenance.json').read_text())
        declared={str(Path(row['local_path']).resolve()):row['sha256'] for row in manifest['files']}
        require(len(manifest['files'])==len(expected_networks) and declared==expected_networks,
                'emulator inventory manifest differs from approved consumed inputs')
        if relocation_record is not None:
            extra['historical_relocation']=locator.verify_historical_relocation(expected_networks,relocation_record)
        else:
            for path,sha in expected_networks.items(): require(digest(path)==sha,'historical emulator input bytes changed')
    else: raise ValueError('unknown scientific input role')
    groups={'source_file_hashes':source,
        'backend_file_hashes':[p for p in (root/'ps_1loop_jax').rglob('*') if p.is_file() and p.suffix in ('.py','.txt')],
        'survey_input_hashes':[*sorted((root/'survey_specs').rglob('*.txt')),root/'configs/default.yaml']}
    result={key:{str(path.resolve()):digest(path) for path in sorted(files)} for key,files in groups.items()}
    result.update(extra);result['emulator_input_hashes']=expected_networks
    for group,key in INVENTORY_KEYS:
        require(bool(result[group]),'empty scientific inventory: '+group)
        result[key]=hashlib.sha256(canonical(result[group])).hexdigest()
    return result


def scientific_modules(root,inventory,*,checker=False):
    root=Path(root).resolve();modules={}
    packages={'pfsfog':root/'pfsfog','ps_1loop_jax':root/'ps_1loop_jax/ps_1loop_jax'}
    files={**inventory['source_file_hashes'],**inventory['backend_file_hashes']}
    if checker:
        files.update({str(p.resolve()):digest(p) for p in (root/'artifact_checker').glob('*.py')})
        packages['artifact_checker']=root/'artifact_checker'
    for path,sha in files.items():
        path=Path(path)
        if path.suffix!='.py': continue
        package=next(((name,base) for name,base in packages.items() if path.is_relative_to(base)),None)
        if package:
            name,base=package;parts=list(path.relative_to(base).with_suffix('').parts)
            if parts[-1]=='__init__': parts.pop()
            modules['.'.join([name,*parts])]={'path':str(path),'sha256':sha}
        elif path.parent==root/'scripts':
            modules['scripts.'+path.stem]={'path':str(path),'sha256':sha}
    return modules


def verify_recorded_modules(records,modules):
    for name,record in records.items():
        require(name in modules and record==modules[name], 'recorded scientific module origin/hash differs: '+name)


def verify_scientific_inventory(provenance,root,*,extended,input_role='active',relocation_record=None):
    expected=scientific_inventory(root,extended=extended,input_role=input_role,relocation_record=relocation_record)
    for group,key in INVENTORY_KEYS:
        require(provenance.get(group)==expected[group] and provenance.get(key)==expected[key],
                'complete scientific inventory/digest differs: '+group)
    verify_recorded_modules(provenance.get('executed_module_hashes',{}),scientific_modules(root,expected))
    if input_role=='active':
        require(provenance.get('active_emulator_provenance')==expected['active_emulator_provenance'],
                'active emulator provenance inventory differs')
    return expected


def verify_project_origins(root,inventory,*,checker=False):
    """Resolve without imports, then independently bind currently loaded subsets."""
    from importlib.machinery import PathFinder
    root=Path(root).resolve();modules=scientific_modules(root,inventory,checker=checker)
    specs={}
    def resolve(name):
        if name in specs: return specs[name]
        parent=name.rpartition('.')[0]
        if parent:
            parent_spec=resolve(parent)
            loaded=sys.modules.get(parent)
            search=getattr(loaded,'__path__',parent_spec.submodule_search_locations)
        else: search=None
        spec=PathFinder.find_spec(name,search)
        require(spec is not None,'scientific module resolution missing: '+name)
        if name in modules:
            require(spec.origin is not None and str(Path(spec.origin).resolve())==modules[name]['path'],
                    'scientific module resolution origin differs: '+name)
        else:
            require(name=='scripts' and root/'scripts' in {Path(p).resolve() for p in spec.submodule_search_locations},
                    'scientific namespace resolution differs: '+name)
        specs[name]=spec;return spec
    for name in modules: resolve(name)
    prefixes=('pfsfog','ps_1loop_jax','artifact_checker') if checker else ('pfsfog','ps_1loop_jax')
    loaded_records={}
    for name,module in tuple(sys.modules.items()):
        if name not in modules and not any(name==prefix or name.startswith(prefix+'.') for prefix in prefixes): continue
        require(name in modules,'unexpected scientific module source: '+name)
        record=modules[name];filename=getattr(module,'__file__',None);spec=getattr(module,'__spec__',None)
        require(filename is not None and str(Path(filename).resolve())==record['path'] and
                spec is not None and spec.origin is not None and str(Path(spec.origin).resolve())==record['path'],
                'active scientific module file/spec origin differs: '+name)
        require(digest(filename)==record['sha256'],'active scientific module bytes differ: '+name)
        loaded_records[name]=record
    # A standalone exporter is outside frozen scripts; bind its actual main file separately.
    main=sys.modules.get('__main__');main_file=getattr(main,'__file__',None)
    if main_file and Path(main_file).name in ('run_extended_cosmology.py','export_cosmology_oracle.py'):
        expected=Path(__file__).resolve() if Path(main_file).name=='export_cosmology_oracle.py' else root/'scripts/run_extended_cosmology.py'
        require(Path(main_file).resolve()==expected,'active driver/exporter main origin differs')
        spec=getattr(main,'__spec__',None)
        require(spec is None or (spec.origin is not None and Path(spec.origin).resolve()==expected),
                'active driver/exporter main spec origin differs')
    cosmo=sys.modules.get('pfsfog.cosmo')
    if cosmo is not None:
        model_root=emulator_locator().MODEL_ROOT if 'active_emulator_provenance' in inventory else EMULATOR_ROOT
        require(Path(cosmo._JENSE_DIR).resolve()==model_root and cosmo._JENSE_MODEL==EMULATOR_MODEL,
                'active consumed emulator input paths differ')
    if 'active_emulator_provenance' in inventory:
        require(inventory['active_emulator_provenance']==emulator_locator().active_emulator_provenance(),
                'active emulator package source/origin differs')
    return loaded_records


def import_inventory():
    expected=scientific_inventory(ORACLE,extended=False,input_role='historical')
    return verify_project_origins(ORACLE,expected,checker=True)


def verify_inputs(provenance,*,relocation_record=None):
    require(provenance['project_head']==HEAD,'original source pin changed')
    expected=verify_scientific_inventory(provenance,ORACLE,extended=False,input_role='historical',
                                         relocation_record=relocation_record)
    return expected.get('historical_relocation',{'historical_exact_path_replay':True})


def pack_control(like,result=None):
    """Standalone serialization: never import the modified supplement serializer."""
    import numpy as np
    arrays={'likelihood/F_like':like.F_like,'likelihood/fiducial':like.fiducial,
            'likelihood/bin_F_like':np.asarray(like.bin_F_like)}
    for f in ('edges','centers','widths'): arrays['likelihood/grid_'+f]=getattr(like.grid,f)
    cells=[]
    for i,c in enumerate(like.cells):
        for f in ('F_like','derivatives','covariance','fiducial_mean'): arrays[f'likelihood/cell{i}/{f}']=getattr(c,f)
        cells.append({'names':c.names,'metadata':c.metadata,'validation':c.validation})
    config=asdict(like.config); config['cosmology_basis']='baseline3'
    meta={'configuration':config,'names':like.names,'parameters':like.parameters,'cells':cells,
          'reference_cosmology':like.reference_cosmology,'model':like.model,'validation':like.validation}
    if result is not None:
        for f in ('F_like','F_prior_cosmo','F_prior_nuisance','F_total','covariance',
                  'cov_cosmo_marg','cov_cosmo_nuisance_fixed','growth_z','growth_fiducial',
                  'growth_jacobian','growth_cov_marg','growth_cov_nuisance_fixed'):
            arrays['result/'+f]=getattr(result,f)
    return meta,arrays


def compare_saved_controls(live_arrays,saved_arrays):
    import numpy as np
    require(set(live_arrays)==set(saved_arrays),'saved control array inventory differs')
    for name,expected in saved_arrays.items():
        actual=live_arrays[name]
        require(actual.shape==expected.shape,'saved control shape differs: '+name)
        if expected.dtype.kind in 'SU': require(np.array_equal(actual,expected),'saved labels differ: '+name)
        else:
            require(np.all(np.isfinite(actual)) and np.allclose(actual,expected,rtol=1e-8,atol=1e-12),
                    'saved control recovery differs: '+name)


def runtime_record():
    import numpy as np
    import scipy
    from contextlib import redirect_stdout
    from io import StringIO
    stream=StringIO()
    with redirect_stdout(stream): scipy.show_config()
    binaries={str(p):digest(p) for root in (Path(np.__file__).parent,Path(scipy.__file__).parent)
              for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.so','.dylib')}
    return {'interpreter':sys.executable,'prefix':sys.prefix,'binary_sha256':digest(sys.executable),
        'scipy_path':scipy.__file__,'scipy_build':stream.getvalue(),'binary_files':binaries,
        'package_versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','jax','jaxlib','cosmopower-jax','PyYAML')},
        'environment':{k:os.environ.get(k) for k in (*THREADS,'PYTHONPATH','XLA_FLAGS')},
        'runtime_limitation':'Recovered SciPy build; historical bitwise replay remains failed.'}


def verify_import_records(records,*,relocation_record=None):
    require('pfsfog' in records,'missing frozen import inventory')
    expected=scientific_inventory(ORACLE,extended=False,input_role='historical',relocation_record=relocation_record)
    verify_recorded_modules(records,scientific_modules(ORACLE,expected,checker=True))


def verify_dependencies(provenance,runtime=None):
    versions=provenance.get('package_versions',{})
    require(set(versions)=={'numpy','scipy','jax','jaxlib','cosmopower-jax','PyYAML'},'dependency version inventory differs')
    require(hashlib.sha256(canonical(versions)).hexdigest()==provenance.get('dependency_hash'),'dependency digest differs')
    if runtime is not None:
        require(versions==runtime['package_versions'],'dependency versions differ from recovered runtime')


def verify_runtime(recorded,*,recorded_root,recorded_cwd,active_root,relocation_record=None):
    """Only the authenticated frozen/consumer source-root distinction is allowed."""
    active=runtime_record()
    require(Path(active['prefix'])==RECOVERY and Path(active['interpreter']).parent==RECOVERY/'bin',
            'active runtime is not the reviewed recovery interpreter')
    def paths(record,root,cwd):
        value=record.get('environment',{}).get('PYTHONPATH')
        require(isinstance(value,str),'runtime PYTHONPATH missing')
        entries=[(Path(cwd)/p).resolve() for p in value.split(os.pathsep)]
        required=[Path(root).resolve(),(Path(root)/'ps_1loop_jax').resolve()]
        locator=emulator_locator()
        require(entries in (required,required+[locator.PACKAGE_ROOT]),'runtime PYTHONPATH has unexpected source roots')
        if len(entries)==3: locator.active_emulator_provenance()
        return len(entries)
    old_count=paths(recorded,recorded_root,recorded_cwd);active_count=paths(active,active_root,Path.cwd())
    if old_count!=active_count:
        require(old_count==2 and active_count==3 and relocation_record is not None,
                'runtime emulator relocation requires explicit verification')
        emulator_locator().verify_historical_relocation({},relocation_record)
    if Path(recorded_root).resolve()==Path(active_root).resolve() and old_count==active_count:
        require(recorded==active,'active runtime differs from recorded runtime')
    else:
        # Validate each role above before normalizing this one necessary difference.
        normalized={**recorded,'environment':{**recorded['environment'],
                    'PYTHONPATH':active['environment']['PYTHONPATH']}}
        require(normalized==active,'active runtime differs from recovered oracle runtime')
    return active


def oracle_command(output,selection,role,pin=None,*,source=None):
    command=[sys.executable,'-B',source or str(Path(__file__).resolve()),'--selection',str(Path(selection).resolve()),
             '--selection-sha256',SELECTION_PIN,'--output',str(Path(output).resolve())]
    return command+(['--worker'] if role=='worker' else ['--verify-only','--index-sha256',pin])


def authenticate_worker_index(output,index,*,source_identity=None):
    source=source_identity or {'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}
    require(index.get('schema_version')=='pfs-desi-cosmology-oracle-v1' and
            index.get('status')=='worker_completed','oracle worker schema/status differs')
    require(index.get('source_head')==HEAD and index.get('selection_sha256')==SELECTION_PIN,
            'oracle worker selection/source differs')
    require(index.get('exporter_path')==source['path'] and
            index.get('exporter_sha256')==source['sha256'],'oracle exporter source differs')
    require(index.get('output')==str(Path(output).resolve()) and index.get('cwd')==str(ORACLE),
            'oracle worker output/cwd differs')
    require(index['runtime']['interpreter']==sys.executable,'oracle worker interpreter differs')
    require(digest(index['selection'])==SELECTION_PIN,'oracle selected bytes changed')
    require(index.get('command')==oracle_command(output,index['selection'],'worker',source=source['path']),
            'oracle worker command differs')


def authenticate_observation(output,index,pin,record,role,*,source_identity=None):
    source=source_identity or {'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}
    output=Path(output).resolve(); log=output/('export.log' if role=='worker' else 'verify.log')
    require(type(record.get('returncode')) is int and record['returncode']==0 and
            record.get('status')=='passed','oracle observed process failed')
    require(record.get('role')==role and record.get('cwd')==str(ORACLE) and
            record.get('command')==oracle_command(output,index['selection'],role,pin,source=source['path']),
            'oracle observed command/role/cwd differs')
    require(record.get('log')==str(log) and digest(log)==record['log_sha256'],'oracle observed log differs')
    start=datetime.fromisoformat(record['started_at']);end=datetime.fromisoformat(record['finished_at'])
    require(start.utcoffset()==timezone.utc.utcoffset(start) and end.utcoffset()==timezone.utc.utcoffset(end)
            and end>=start,'oracle observed UTC interval differs')
    path=output/(role+'-completion.json')
    require(record.get('completion_path')==str(path) and digest(path)==record['completion_sha256'],
            'oracle completion pin differs')
    completion=json.loads(path.read_text())
    require(completion==record.get('completion') and
            completion.get('schema_version')=='pfs-desi-cosmology-oracle-completion-v1' and
            completion.get('worker_index_sha256')==pin and completion.get('role')==role and
            completion.get('output')==str(output) and completion.get('cwd')==str(ORACLE) and
            completion.get('command')==record['command'] and
            completion.get('source')==source,
            'oracle completion is not bound to selected worker index/source')


def authenticate_export(output,selected,index,*,source_identity=None):
    """Stdlib-only association checks, shared with the separate-study consumer."""
    require(selected.get('schema_version')=='pfs-desi-cosmology-oracle-selection-v1' and
            selected.get('status')=='complete','oracle selection schema/status differs')
    require(selected.get('selection_sha256')==SELECTION_PIN and selected.get('receipt_pins')==RECEIPTS,
            'oracle selection/receipt identity differs')
    pin=selected['worker_index_sha256']
    require(digest(Path(output)/'worker-index.json')==pin,'oracle worker index pin changed')
    require(json.loads((Path(output)/'worker-index.json').read_text())==index,'oracle supplied index differs from pinned bytes')
    authenticate_worker_index(output,index,source_identity=source_identity)
    for key,role in (('execution','worker'),('verification','verification')):
        authenticate_observation(output,index,pin,selected[key],role,source_identity=source_identity)


def authenticate_historical_export(output,selected,index):
    """Bind old observations to their archived producer, not the changed verifier."""
    active={'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}
    historical={'path':index.get('exporter_path'),'sha256':index.get('exporter_sha256')}
    proof={'kind':'active_source','sha256':active['sha256']}
    if historical!=active:
        base='69e33b3e63995070b0a62e93996c4cc09bfbe710'
        expected='9430355447701ad69ea53bb0f6f1ac587151fa415583f386dc2448e4d17d6377'
        original_path=str(ORACLE.parent/'extended-cache-repair/scripts/export_cosmology_oracle.py')
        require(historical['sha256']==expected and historical['path'] in (active['path'],original_path),
                'unapproved historical exporter source')
        original=subprocess.check_output(['git','show',base+':scripts/export_cosmology_oracle.py'],
                                         cwd=Path(__file__).resolve().parents[1])
        require(hashlib.sha256(original).hexdigest()==expected,'historical exporter source bytes differ')
        proof={'kind':'git_blob','commit':base,'file':'scripts/export_cosmology_oracle.py','sha256':expected}
    authenticate_export(output,selected,index,source_identity=historical)
    return {'historical_exact_path_replay':False,'historical_exporter':historical,
            'historical_source_verification':proof,'active_verifier':active}


def emit_oracle_completion(args,pin,role):
    require(digest(args.output/'worker-index.json')==pin,'oracle completed index changed')
    write_json(args.output/(role+'-completion.json'),{
        'schema_version':'pfs-desi-cosmology-oracle-completion-v1','role':role,
        'worker_index_sha256':pin,'output':str(args.output.resolve()),'cwd':str(Path.cwd()),
        'command':list(sys.orig_argv),'source':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}})


def observe_oracle(command,output,role):
    output=Path(output).resolve();log=output/('export.log' if role=='worker' else 'verify.log')
    started=datetime.now(timezone.utc).isoformat()
    with log.open('wb') as stream:
        result=subprocess.run(command,cwd=ORACLE,stdout=stream,stderr=subprocess.STDOUT)
    record={'command':command,'cwd':str(ORACLE),'role':role,'started_at':started,
            'finished_at':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,
            'status':'passed' if result.returncode==0 else 'failed','log':str(log),'log_sha256':digest(log)}
    receipt=output/('execution.json' if role=='worker' else 'verify-execution.json')
    write_json(receipt,record)  # Preserve actual exit even if completion attachment rejects.
    if result.returncode==0:
        completion=output/(role+'-completion.json')
        record.update(completion=json.loads(completion.read_text()),completion_path=str(completion),
                      completion_sha256=digest(completion))
        write_json(receipt,record)
    return record


def worker(args):
    check_environment()
    require(args.selection_sha256==SELECTION_PIN and digest(args.selection)==SELECTION_PIN,'external selection pin mismatch')
    selection=json.loads(args.selection.read_text())
    selected=[e for e in selection['runs'] if e['configuration']['case'] in RECEIPTS]
    require(len(selected)==2 and {e['configuration']['case'] for e in selected}==set(RECEIPTS),'B/C selection differs')
    # Reject incomplete source/input identities before receipt-loader reconstruction.
    # This precheck never replaces the mandatory receipt authentication below.
    for item in selected:
        raw=json.loads((Path(item['path'])/'manifest.json').read_text())
        verify_inputs(raw['provenance'])
    import_inventory()
    # Public frozen receipt loader authenticates before backend/theory initialization.
    from artifact_checker import load_revision_artifacts
    authenticated={}
    for item in selected:
        case=item['configuration']['case']
        require(item['receipt_sha256']==RECEIPTS[case],'altered receipt pin')
        loaded=load_revision_artifacts(item['path'],receipt_sha256=RECEIPTS[case])
        require(loaded[0]['run_id']==item['run_id'] and loaded[0]['configuration']==item['configuration'],'selected identity mismatch')
        verify_inputs(loaded[0]['provenance']); verify_dependencies(loaded[0]['provenance']); authenticated[case]=(item,loaded)
    import_inventory()
    import jax
    jax.config.update('jax_enable_x64',True)
    import numpy as np
    from scripts import run_revision as producer
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.surveys import default_survey_group
    from pfsfog.revision_forecast import RevisionConfig,build_revision_likelihood,apply_revision_priors
    from pfsfog.revision_artifacts import _arrays
    import_inventory()
    backend=producer._load_backend(); survey=default_survey_group()
    runtime=runtime_record(); verify_runtime(runtime,recorded_root=ORACLE,recorded_cwd=ORACLE,active_root=ORACLE); cases={}
    for case,(item,loaded) in authenticated.items():
        manifest,saved,_,_=loaded
        out=args.output/case; out.mkdir()
        original=out/'original'; original.mkdir()
        for filename in ('manifest.json','matrices.npz','summary.csv','validation.json','certification.json','run.log'):
            shutil.copyfile(Path(item['path'])/filename,original/filename)
        fields={k:v for k,v in manifest['configuration'].items() if k in RevisionConfig.__dataclass_fields__}
        fields['nominal_bins']=tuple(map(tuple,fields['nominal_bins']))
        config=RevisionConfig(**fields)
        cosmo=FiducialCosmology(params=manifest['cosmology']['reference_cosmology'],backend='cosmopower')
        live=build_revision_likelihood(config,survey,cosmo,backend)
        # Persist D/C before any solve, so failed recovery remains inspectable.
        meta,arrays=pack_control(live)
        np.savez(out/'response-before-solve.npz',**arrays); write_json(out/'control.json',meta)
        for actual,expected in zip(live.cells,manifest['cells']):
            require(all(canonical(v)==canonical(expected[k]) for k,v in actual.metadata.items()),'saved physical cell metadata differs')
        result=apply_revision_priors(live,config.nuisance_factor)
        compare_saved_controls(_arrays(result),saved)
        meta,arrays=pack_control(live,result)
        np.savez(out/'controls.npz',**arrays)
        provenance=producer.collect_provenance(backend,requested_dk=.005,ap_nk=8192)
        verify_inputs(manifest['provenance']); verify_dependencies(provenance,runtime); import_inventory()
        cases[case]={'original_selected':item,'original_provenance':manifest['provenance'],
            'recovered_provenance':provenance,'receipt_sha256':RECEIPTS[case],
            'control':str((out/'control.json').resolve()),'arrays':str((out/'controls.npz').resolve()),
            'control_sha256':digest(out/'control.json'),'arrays_sha256':digest(out/'controls.npz'),
            'array_inventory':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()} for k,v in arrays.items()},
            'original_files':{p.name:digest(p) for p in original.iterdir()},
            'recovery_tolerance':{'rtol':1e-8,'atol':1e-12}}
    check_environment()
    index={'schema_version':'pfs-desi-cosmology-oracle-v1','status':'worker_completed',
        'source_head':HEAD,'selection_sha256':SELECTION_PIN,'selection':str(args.selection.resolve()),
        'exporter_path':str(Path(__file__).resolve()),'exporter_sha256':digest(__file__),
        'output':str(args.output.resolve()),'cases':cases,'runtime':runtime,'imports':import_inventory(),
        'command':list(sys.orig_argv),'cwd':str(Path.cwd()),'completed_at':datetime.now(timezone.utc).isoformat()}
    return write_json(args.output/'worker-index.json',index)


def verify_export(output,index_sha256):
    """Fresh-process byte/array and numerical checks, with an external index pin."""
    check_environment(); path=output/'worker-index.json'
    require(digest(path)==index_sha256,'external export index pin mismatch')
    index=json.loads(path.read_text())
    authenticate_worker_index(output,index)
    authenticate_observation(output,index,index_sha256,json.loads((output/'execution.json').read_text()),'worker')
    require(digest(index['selection'])==SELECTION_PIN,'selected bytes changed')
    require(index['exporter_sha256']==digest(__file__),'exporter source changed')
    require(Path(index['cwd']).resolve()==ORACLE,'frozen cwd changed')
    verify_import_records(index['imports'])
    verify_runtime(index['runtime'],recorded_root=ORACLE,recorded_cwd=index['cwd'],active_root=ORACLE)
    for record in index['cases'].values():
        verify_inputs(record['original_provenance']); verify_inputs(record['recovered_provenance'])
        verify_dependencies(record['original_provenance'])
        verify_dependencies(record['recovered_provenance'],index['runtime'])
    import_inventory()
    import numpy as np
    from pfsfog.fisher_core import fisher_from_shells
    from pfsfog.revision_forecast import _scaled_inverse
    for case,record in index['cases'].items():
        verify_dependencies(record['original_provenance'])
        verify_dependencies(record['recovered_provenance'],index['runtime'])
        require(record['receipt_sha256']==RECEIPTS[case],'receipt changed')
        require(digest(record['control'])==record['control_sha256'] and digest(record['arrays'])==record['arrays_sha256'],'export bytes changed')
        meta=json.loads(Path(record['control']).read_text())
        with np.load(record['arrays'],allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
        require(set(arrays)==set(record['array_inventory']),'array inventory changed')
        for key,v in arrays.items():
            require(record['array_inventory'][key]=={'shape':list(v.shape),'dtype':v.dtype.str,'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()},'array layout changed')
        n=len(meta['names']); bins=np.zeros_like(arrays['likelihood/bin_F_like'])
        for i,cell in enumerate(meta['cells']):
            key=f'likelihood/cell{i}/'; f=arrays[key+'F_like']
            if cell['names']:
                actual=fisher_from_shells(arrays[key+'derivatives'],arrays[key+'covariance'],arrays['likelihood/grid_centers'])
                require(np.allclose(f,actual,rtol=1e-10,atol=1e-12),'recovered D/C Fisher mismatch')
                indices=[meta['names'].index(n) for n in cell['names']]
                bins[cell['metadata']['nominal_bin_index']][np.ix_(indices,indices)]+=actual
        require(np.allclose(bins,arrays['likelihood/bin_F_like'],rtol=1e-10,atol=1e-12),'bin Fisher mismatch')
        require(np.allclose(bins.sum(axis=0),arrays['likelihood/F_like'],rtol=1e-10,atol=1e-12),'global Fisher mismatch')
        prior=np.zeros((n,n)); prior[1,1]=.04; prior[2,2]=1.
        nuisance=np.zeros((n,n))
        for i,p in enumerate(meta['parameters']):
            if p['role']!='cosmo' and p['prior_sigma'] is not None:
                nuisance[i,i]=1/(p['prior_sigma']*meta['configuration']['nuisance_factor'])**2
        for name,value in (('F_prior_cosmo',prior),('F_prior_nuisance',nuisance),('F_total',arrays['likelihood/F_like']+prior+nuisance)):
            require(np.allclose(arrays['result/'+name],value,rtol=1e-10,atol=1e-12),'prior identity mismatch')
        inverse,_=_scaled_inverse(arrays['result/F_total'],meta['names'],'oracle reload')
        require(np.allclose(inverse,arrays['result/covariance'],rtol=1e-10,atol=1e-12),'scaled covariance reload mismatch')
        jac=arrays['result/growth_jacobian']
        require(np.allclose(jac@inverse[:3,:3]@jac.T,arrays['result/growth_cov_marg'],rtol=1e-10,atol=1e-12),'growth propagation mismatch')
        verify_inputs(record['original_provenance']); verify_inputs(record['recovered_provenance'])
    import_inventory()
    require(digest(path)==index_sha256,'oracle verified index changed')
    return index


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True); p.add_argument('--selection-sha256',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--verify-only',action='store_true'); p.add_argument('--index-sha256')
    args=p.parse_args(argv); check_environment()
    require(args.selection_sha256==SELECTION_PIN and digest(args.selection)==SELECTION_PIN,
            'external oracle selection pin mismatch')
    if args.verify_only:
        require(args.index_sha256 is not None,'external index pin required')
        verify_export(args.output,args.index_sha256)
        emit_oracle_completion(args,args.index_sha256,'verification'); return 0
    if args.worker:
        pin=worker(args); emit_oracle_completion(args,pin,'worker'); return 0
    require(not args.output.exists(),'refusing existing oracle export'); args.output.mkdir(parents=True)
    receipt=observe_oracle(oracle_command(args.output,args.selection,'worker'),args.output,'worker')
    if receipt['returncode']: return receipt['returncode']
    pin=receipt['completion']['worker_index_sha256']
    require(digest(args.output/'worker-index.json')==pin,'oracle worker index changed after completion')
    index=json.loads((args.output/'worker-index.json').read_text())
    authenticate_worker_index(args.output,index)
    authenticate_observation(args.output,index,pin,receipt,'worker')
    verify=observe_oracle(oracle_command(args.output,args.selection,'verification',pin),args.output,'verification')
    if verify['returncode']: return verify['returncode']
    selected={'schema_version':'pfs-desi-cosmology-oracle-selection-v1',
        'status':'complete','worker_index_sha256':pin,'execution':receipt,'verification':verify,
        'selection_sha256':SELECTION_PIN,'receipt_pins':RECEIPTS}
    authenticate_export(args.output,selected,index)
    write_json(args.output/'export-index.json',selected)
    print(digest(args.output/'export-index.json'),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
