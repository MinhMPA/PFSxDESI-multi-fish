"""Qualification requires complete independent physical-inference evidence."""
import copy
import json
from pathlib import Path
import pytest
from scripts.run_pfs_density_scan import digest

GRIDS=('loop2048','loop4096','angular64','shell304','final-loop')
PUBLIC={'A':'A','B':'B','D':'C','C':'D'}

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return digest(p)

@pytest.fixture
def evidence(tmp_path):
    root=tmp_path/'root';out=root/'results/new';old=root/'results/old';audit=out/'independent-audit'
    runner=root/'scripts/run_counterterm_comparison.py';put(runner,{'old':'runner'})
    physical=root/'scripts/counterterm_comparison.py';put(physical,{'physical':'unchanged'})
    request={'track':'CT-A','source_head':'abc123','old_study':str(old),'bindings':{str(runner):digest(runner),str(physical):digest(physical)}}
    parent=put(out/'request.json',request);put(out/'state.json',{'request_sha256':parent,'grids':{},'active_jobs':[]})
    sources={}
    for n in ('scripts/audit_counterterm_inference.py','scripts/overlap_checks.py','tests/test_counterterm_inference_audit.py'):
        sources[str(root/n)]=put(root/n,{'source':n})
    aq={'physical_request':str(out/'request.json'),'physical_request_sha256':parent,'source_files':sources}
    aqsha=put(audit/'request.json',aq);receipts={str(audit/'request.json'):aqsha}
    for g in GRIDS:
        checks=[]
        for model,base in [('old',old),('CT-A',out)]:
            for internal,public in PUBLIC.items():
                p=base/'inference'/g/'profiles'/internal
                files={str(p/n):put(p/n,{'value':n}) for n in ('manifest.json','arrays.npz','constraints.json')}
                checks.append({'model':model,'grid':g,'public_case':public,'status':'passed','result_directory':str(p),'input_files':files,'checks':{n:0. for n in ('full_schur_scaled_error','qr_covariance_error','assembly_scaled_error','qr_inverse_residual','growth_scaled_error','fixed_growth_scaled_error')}})
        receipts[str(audit/(g+'.json'))]=put(audit/(g+'.json'),{'status':'passed','grid':g,'request_sha256':aqsha,'checks':checks})
    receipt=out/'inference/final-loop/physical/cell/receipt.json';psha=put(receipt,{'status':'passed','request_sha256':parent})
    ar={'status':'passed','request_sha256':aqsha,'controls':{n:{} for n in ('old','AP_response_only','covariance_only','CT-A')},'angular_cells':[{'physical_receipt':str(receipt),'receipt_sha256':psha,'auto_ell6_reconstruction_max_abs':0.}]}
    receipts[str(audit/'auto-attribution.json')]=put(audit/'auto-attribution.json',ar)
    put(audit/'verification.json',{'status':'passed','physical_request_sha256':parent,'request_sha256':aqsha,'receipts':receipts})
    return root,out,request

def reseal(out):
    audit=out/'independent-audit';p=audit/'verification.json';r=json.loads(p.read_text())
    r['receipts']={name:digest(name) for name in r['receipts']};put(p,r)

def test_complete_independent_evidence_passes(evidence):
    from scripts.counterterm_acceptance import validate_independent
    root,out,request=evidence
    assert validate_independent(out,request,root=root)['result_count']==40

@pytest.mark.parametrize('damage',['absent','failed','wrong_parent','duplicate','incomplete','wrong_grid','corrupt_result','corrupt_attribution','missing_source','missing_result_binding','missing_angular','duplicate_angular','failed_grid','wrong_audit_request','missing_grid_receipt'])
def test_invalid_independent_evidence_cannot_qualify(evidence,damage):
    from scripts.counterterm_acceptance import validate_independent
    root,out,request=evidence;a=out/'independent-audit';p=a/'verification.json';v=json.loads(p.read_text())
    if damage=='absent':p.unlink()
    elif damage in ('failed','wrong_parent','missing_grid_receipt'):
        if damage=='failed':v['status']='failed'
        elif damage=='wrong_parent':v['physical_request_sha256']='wrong'
        else:v['receipts'].pop(str(a/'loop2048.json'))
        put(p,v)
    elif damage=='corrupt_result':(out/'inference/loop2048/profiles/A/arrays.npz').write_text('corrupt')
    elif damage=='corrupt_attribution':(a/'auto-attribution.json').write_text('{}')
    elif damage=='missing_source':
        q=json.loads((a/'request.json').read_text());q['source_files']={};put(a/'request.json',q);reseal(out)
    elif damage in ('missing_angular','duplicate_angular'):
        q=json.loads((a/'auto-attribution.json').read_text());q['angular_cells']=[] if damage=='missing_angular' else q['angular_cells']*2;put(a/'auto-attribution.json',q);reseal(out)
    else:
        q=json.loads((a/'loop2048.json').read_text())
        if damage=='duplicate':q['checks'][-1]=copy.deepcopy(q['checks'][0])
        elif damage=='incomplete':q['checks'].pop()
        elif damage=='wrong_grid':q['checks'][0]['grid']='final-loop'
        elif damage=='failed_grid':q['status']='failed'
        elif damage=='wrong_audit_request':q['request_sha256']='wrong'
        else:q['checks'][0]['input_files'].pop(next(iter(q['checks'][0]['input_files'])))
        put(a/'loop2048.json',q);reseal(out)
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_independent(out,request,root=root)

def test_historical_exception_is_exact_and_does_not_cover_physical_change(evidence,monkeypatch):
    from scripts import counterterm_acceptance as ca
    root,out,request=evidence;runner=root/'scripts/run_counterterm_comparison.py';blob=runner.read_bytes()
    monkeypatch.setattr(ca,'git_blob',lambda r,c,p:blob)
    runner.write_text('new acceptance only')
    _,exception=ca.authenticate_physical(out,root=root)
    assert exception['sha256']==request['bindings'][str(runner)]
    (root/'scripts/counterterm_comparison.py').write_text('changed physics')
    with pytest.raises(ValueError,match='binding'):
        ca.authenticate_physical(out,root=root)

@pytest.mark.parametrize('damage',['wrong_blob','wrong_path','wrong_commit'])
def test_historical_exception_rejects_unproved_runner(evidence,monkeypatch,damage):
    from scripts import counterterm_acceptance as ca
    root,out,request=evidence;runner=root/'scripts/run_counterterm_comparison.py';blob=runner.read_bytes();runner.write_text('new')
    if damage=='wrong_path':
        request['bindings'][str(root/'scripts/other.py')]=request['bindings'].pop(str(runner));put(out/'request.json',request)
        state=json.loads((out/'state.json').read_text());state['request_sha256']=digest(out/'request.json');put(out/'state.json',state)
    def get_blob(r,c,p):
        if damage=='wrong_commit':raise ValueError('unavailable commit')
        return b'wrong' if damage=='wrong_blob' else blob
    monkeypatch.setattr(ca,'git_blob',get_blob)
    with pytest.raises((ValueError,FileNotFoundError)):
        ca.authenticate_physical(out,root=root)

def test_fresh_request_uses_live_runner_without_exception(evidence):
    from scripts.counterterm_acceptance import authenticate_physical
    root,out,_=evidence
    assert authenticate_physical(out,root=root)[1] is None

def prepared_acceptance(evidence):
    root,out,request=evidence;state=json.loads((out/'state.json').read_text())
    for g in GRIDS:
        summary=out/'inference'/g/'summary.json'
        state['grids'][g]={'status':'passed','summary_sha256':put(summary,{'status':'passed','request_sha256':digest(out/'request.json')})}
    put(out/'state.json',state)
    for n in ('scripts/counterterm_acceptance.py','tests/test_counterterm_acceptance.py'):put(root/n,{'checker':n})
    rows=[{'grid':g,'model':m,'public_case':p,'sigma_marginalized':{'x':1.}} for g in GRIDS for m in ('old','CT-A') for p in PUBLIC.values()]
    return root,out,request,{'schema_version':1,'track':'CT-A','request_sha256':digest(out/'request.json'),'status':'qualified','rows':rows,'convergence':{'status':'passed','checks':[{'passed':True}]}}

def test_public_verify_rejects_missing_audit_before_reconstruction(evidence,monkeypatch):
    from scripts import counterterm_acceptance as ca,run_counterterm_comparison as runner
    root,out,_,comparison=prepared_acceptance(evidence)
    (out/'independent-audit/verification.json').unlink();calls=[]
    monkeypatch.setattr(runner,'_reconstruct_verification',lambda *args:calls.append(args))
    original=ca.qualify
    monkeypatch.setattr(ca,'qualify',lambda output,reconstruct,publish:original(output,reconstruct,publish,root=root))
    with pytest.raises(FileNotFoundError):runner.verify(out)
    assert calls==[] and not (out/'comparison.json').exists() and not (out/'acceptance').exists()

def test_fresh_public_verification_does_not_require_old_comparison(evidence,monkeypatch):
    from scripts import counterterm_acceptance as ca,run_counterterm_comparison as runner
    root,out,_,comparison=prepared_acceptance(evidence)
    monkeypatch.setattr(runner,'_reconstruct_verification',lambda *args:comparison)
    original=ca.qualify
    monkeypatch.setattr(ca,'qualify',lambda output,reconstruct,publish:original(output,reconstruct,publish,root=root))
    assert not (out/'comparison.json').exists()
    result=runner.verify(out)
    assert result['status']=='qualified' and not result['historical_runner_exception']
    assert json.loads((out/'comparison.json').read_text())==comparison
    assert json.loads(Path(result['acceptance_receipt']).read_text())['independent']['result_count']==40
    accepted=json.loads((Path(result['acceptance_receipt']).parent/'request.json').read_text())
    ca.verify_bindings(accepted['input_files'])

def test_historical_acceptance_is_append_only(evidence,monkeypatch):
    from scripts import counterterm_acceptance as ca
    root,out,_,comparison=prepared_acceptance(evidence);put(out/'comparison.json',comparison)
    runner=root/'scripts/run_counterterm_comparison.py';blob=runner.read_bytes();runner.write_text('new acceptance')
    monkeypatch.setattr(ca,'git_blob',lambda *args:blob)
    before={p:digest(p) for p in out.rglob('*') if p.is_file()};published=[]
    result=ca.qualify(out,lambda request:comparison,lambda *args:published.append(args),root=root)
    assert not published and result['historical_runner_exception']
    assert all(digest(p)==sha for p,sha in before.items())
    assert (Path(result['acceptance_receipt']).parent/'historical-runner.py').read_bytes()==blob

def test_nested_physical_source_inventory_stays_strict_with_runner_exception(evidence,monkeypatch):
    from scripts import counterterm_acceptance as ca
    root,out,request=evidence;model=root/'pfsfog/counterterm_fields.py';modelsha=put(model,{'physics':'original'})
    request['sources']={'loop2048':{'details':{'original':{'details':{'source_file_hashes':{str(model):modelsha}}}}}}
    put(out/'request.json',request);state=json.loads((out/'state.json').read_text());state['request_sha256']=digest(out/'request.json');put(out/'state.json',state)
    runner=root/'scripts/run_counterterm_comparison.py';blob=runner.read_bytes();runner.write_text('new acceptance')
    monkeypatch.setattr(ca,'git_blob',lambda *args:blob)
    model.write_text('changed physical model')
    with pytest.raises(ValueError,match='binding'):ca.authenticate_physical(out,root=root)
