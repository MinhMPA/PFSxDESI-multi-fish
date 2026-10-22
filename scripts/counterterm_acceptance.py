"""Require independent inference evidence before CT-A qualification.

Historical acceptance is append-only. Only the exact request-recorded runner
may be authenticated from Git; every physical source remains a live binding.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
from scripts.run_pfs_density_scan import digest,write_json
from pfsfog.extended_artifacts import read_json,canonical

ROOT=Path(__file__).resolve().parents[1]
GRIDS=('loop2048','loop4096','angular64','shell304','final-loop')
PUBLIC={'A':'A','B':'B','D':'C','C':'D'}
RUNNER='scripts/run_counterterm_comparison.py'
AUDIT_SOURCES=('scripts/audit_counterterm_inference.py','scripts/overlap_checks.py','tests/test_counterterm_inference_audit.py')
CHECKER_SOURCES=(RUNNER,'scripts/counterterm_acceptance.py','tests/test_counterterm_acceptance.py')
LIMITS={'full_schur_scaled_error':1e-8,'qr_covariance_error':1e-8,'assembly_scaled_error':1e-10,
        'qr_inverse_residual':1e-10,'growth_scaled_error':1e-8,'fixed_growth_scaled_error':1e-8}


def require(ok,message):
    if not ok:raise ValueError(message)


def verify_bindings(bindings):
    for name,sha in bindings.items():
        require(digest(name)==sha,'changed binding: '+name)


def git_blob(root,commit,path):
    try:return subprocess.check_output(['git','show',commit+':'+path],cwd=root,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:raise ValueError('historical runner commit unavailable') from error


def physical_bindings(request):
    bindings=dict(request['bindings'])
    def visit(value):
        if not isinstance(value,dict):return
        for key,item in value.items():
            if str(key).startswith('/'):
                require(isinstance(item,str) and len(item)==64 and all(c in '0123456789abcdef' for c in item),'invalid physical source digest')
                require(key not in bindings or bindings[key]==item,'conflicting physical source binding: '+key)
                bindings[key]=item
            else:visit(item)
    visit(request.get('sources',{}));visit(request.get('historical_source',{}))
    return bindings


def authenticate_physical(output,root=ROOT):
    output=Path(output).resolve();root=Path(root).resolve();request=read_json(output/'request.json')
    require(read_json(output/'state.json')['request_sha256']==digest(output/'request.json'),'immutable physical request changed')
    require(request['track']=='CT-A','mixed counterterm track')
    runner=str(root/RUNNER);require(runner in request['bindings'],'missing exact runner binding')
    exception=None
    for name,sha in physical_bindings(request).items():
        if digest(name)==sha:continue
        require(name==runner,'changed physical/source binding: '+name)
        blob=git_blob(root,request['source_head'],RUNNER)
        require(hashlib.sha256(blob).hexdigest()==sha,'historical runner blob differs from request binding')
        exception={'path':runner,'git_path':RUNNER,'commit':request['source_head'],'sha256':sha,'blob':blob}
    return request,exception


def validate_independent(output,request,root=ROOT):
    output=Path(output).resolve();root=Path(root).resolve();audit=output/'independent-audit'
    parent=digest(output/'request.json');aq=read_json(audit/'request.json');aqsha=digest(audit/'request.json')
    verification=read_json(audit/'verification.json')
    require(verification['status']=='passed','independent verification did not pass')
    require(verification['physical_request_sha256']==parent==aq['physical_request_sha256'],'wrong independent physical parent')
    require(aq['physical_request']==str(output/'request.json'),'wrong independent physical request path')
    require(verification['request_sha256']==aqsha,'wrong independent request digest')
    require(set(aq['source_files'])=={str(root/n) for n in AUDIT_SOURCES},'incomplete independent checker bindings')
    verify_bindings(aq['source_files'])
    expected={str(audit/n) for n in ('request.json','auto-attribution.json',*(g+'.json' for g in GRIDS))}
    require(set(verification['receipts'])==expected,'incomplete independent receipt coverage')
    verify_bindings(verification['receipts'])
    files={str(audit/'verification.json'):digest(audit/'verification.json'),**verification['receipts'],**aq['source_files']}
    seen=[]
    for grid in GRIDS:
        record=read_json(audit/(grid+'.json'))
        require(record['status']=='passed' and record['grid']==grid and record['request_sha256']==aqsha,'failed or mismatched grid audit')
        local=[]
        for check in record['checks']:
            key=(check['model'],check['grid'],check['public_case']);local.append(key);seen.append(key)
            require(check['status']=='passed' and check['grid']==grid,'failed or wrong-grid result audit')
            require(check['model'] in ('old','CT-A') and check['public_case'] in PUBLIC.values(),'invalid audited result label')
            internal=next(k for k,v in PUBLIC.items() if v==check['public_case'])
            base=Path(request['old_study']) if check['model']=='old' else output
            result=(base/'inference'/grid/'profiles'/internal).resolve()
            require(check['result_directory']==str(result),'wrong audited result path')
            require(set(check['input_files'])=={str(result/n) for n in ('manifest.json','arrays.npz','constraints.json')},'incomplete result binding')
            verify_bindings(check['input_files']);files.update(check['input_files'])
            require(set(check['checks'])==set(LIMITS),'incomplete Schur/QR/growth checks')
            for name,limit in LIMITS.items():
                value=check['checks'][name]
                require(math.isfinite(value) and 0<=value<=limit,'failed independent numerical check: '+name)
        expected_local={(m,grid,p) for m in ('old','CT-A') for p in PUBLIC.values()}
        require(len(local)==8 and set(local)==expected_local,'duplicate or incomplete independent result coverage')
    require(len(seen)==40 and len(set(seen))==40,'duplicate independent result coverage')
    attribution=read_json(audit/'auto-attribution.json')
    require(attribution['status']=='passed' and attribution['request_sha256']==aqsha,'failed or mismatched attribution')
    require(set(attribution['controls'])=={'old','AP_response_only','covariance_only','CT-A'},'incomplete attribution controls')
    expected_physical={str(p) for p in (output/'inference/final-loop/physical').glob('*/receipt.json')}
    listed=[x['physical_receipt'] for x in attribution['angular_cells']]
    require(bool(expected_physical) and len(listed)==len(set(listed)) and set(listed)==expected_physical,'duplicate or incomplete angular coverage')
    for cell in attribution['angular_cells']:
        p=Path(cell['physical_receipt']);r=read_json(p)
        require(r['status']=='passed' and r['request_sha256']==parent,'unqualified angular physical parent')
        require(digest(p)==cell['receipt_sha256'],'changed attribution physical receipt')
        residual=cell['auto_ell6_reconstruction_max_abs']
        require(math.isfinite(residual) and 0<=residual<=1e-8,'failed angular completion check')
        files[str(p)]=digest(p)
    return {'status':'passed','result_count':len(seen),'physical_request_sha256':parent,
            'audit_request_sha256':aqsha,'verification_sha256':digest(audit/'verification.json'),
            'attribution_sha256':digest(audit/'auto-attribution.json'),'files':files}


def supporting_files(output,parent):
    state=read_json(output/'state.json');files={}
    for grid in GRIDS:
        summary=output/'inference'/grid/'summary.json';r=read_json(summary)
        require(state['grids'][grid]['status']=='passed' and r['status']=='passed' and r['request_sha256']==parent,'unqualified grid summary')
        require(state['grids'][grid]['summary_sha256']==digest(summary),'changed grid summary binding')
        files[str(summary)]=digest(summary)
        for p in (summary.parent/'physical').glob('*/receipt.json'):
            r=read_json(p);require(r['status']=='passed' and r['request_sha256']==parent,'unqualified physical receipt')
            files[str(p)]=digest(p)
    return files


def immutable_write(path,value):
    if path.exists():require(read_json(path)==value,'refusing to replace acceptance evidence')
    else:write_json(path,value)


def qualify(output,reconstruct,publish,root=ROOT):
    output=Path(output).resolve();root=Path(root).resolve()
    request,exception=authenticate_physical(output,root)
    independent=validate_independent(output,request,root)
    files={str(output/'request.json'):digest(output/'request.json'),**independent['files'],**supporting_files(output,independent['physical_request_sha256'])}
    existing=output/'comparison.json'
    if exception:
        files[str(output/'state.json')]=digest(output/'state.json')
        if existing.exists():files[str(existing)]=digest(existing)
    checker={str(root/n):digest(root/n) for n in CHECKER_SOURCES}
    comparison=reconstruct(request)
    if exception and existing.exists():
        require(canonical(read_json(existing))==canonical(comparison),'historical numerical comparison changed')
    # Check again after reconstruction, before writing any accepted evidence.
    authenticate_physical(output,root);verify_bindings(files);verify_bindings(checker)
    historical=None if exception is None else {**{k:v for k,v in exception.items() if k!='blob'},'snapshot':'historical-runner.py'}
    acceptance={'schema_version':1,'track':'CT-A','physical_request_sha256':independent['physical_request_sha256'],
                'checker_files':checker,'input_files':files,'historical_runner':historical,
                'independent':{k:v for k,v in independent.items() if k!='files'}}
    key=hashlib.sha256(json.dumps(acceptance,sort_keys=True).encode()).hexdigest()
    dest=output/'acceptance'/key;dest.mkdir(parents=True,exist_ok=True)
    if exception:
        snapshot=dest/'historical-runner.py'
        if snapshot.exists():require(digest(snapshot)==exception['sha256'],'changed historical runner snapshot')
        else:snapshot.write_bytes(exception['blob'])
    immutable_write(dest/'request.json',acceptance);immutable_write(dest/'comparison.json',comparison)
    receipt={'status':comparison['status'],'physical_request_sha256':independent['physical_request_sha256'],
             'request_sha256':digest(dest/'request.json'),'comparison_sha256':digest(dest/'comparison.json'),
             'independent':acceptance['independent'],'historical_runner':historical,'checker_files':checker}
    immutable_write(dest/'receipt.json',receipt)
    # The historical calculation and its old acceptance records remain untouched.
    if exception is None:publish(comparison,dest/'receipt.json')
    return {'status':comparison['status'],'rows':len(comparison['rows']),
            'failed_refinement_checks':sum(not c['passed'] for c in comparison['convergence']['checks']),
            'acceptance_receipt':str(dest/'receipt.json'),'historical_runner_exception':exception is not None}
