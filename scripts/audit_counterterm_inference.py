#!/usr/bin/env python
"""Independent Schur/QR and angular-auto attribution supplement.

This adds checks to an immutable physical request without modifying its bound
source, arrays or receipts. Supplemental source and result hashes are explicit.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
import argparse
from dataclasses import replace
import time
import numpy as np
from pfsfog.extended_artifacts import read_json,require_close
from pfsfog.fisher_core import fisher_from_shells
from scripts.run_pfs_density_scan import digest,write_json
from scripts.takada_study import load_corrected_result,assemble_from_template
from scripts.overlap_checks import check_marginalization
from scripts.elf_study_inference import infer_study
from scripts.run_counterterm_comparison import reconstructed_source,result_row,PUBLIC,GRID_PROFILES

SOURCES=('scripts/audit_counterterm_inference.py','scripts/overlap_checks.py','tests/test_counterterm_inference_audit.py')


def audit_result(path):
    path=Path(path);saved,_=load_corrected_result(path);check=check_marginalization(saved)
    return {'status':'passed','result_directory':str(path.resolve()),
        'input_files':{str(path/n):digest(path/n) for n in ('manifest.json','arrays.npz','constraints.json')},
        'checks':{k:float(v) for k,v in check.items() if np.ndim(v)==0},
        'method':'independent Schur complement and QR of whitened physical response rows with one global prior insertion'}


def binding(output):
    output=Path(output);dest=output/'independent-audit';dest.mkdir(exist_ok=True)
    request={'physical_request':str(output/'request.json'),'physical_request_sha256':digest(output/'request.json'),
             'source_files':{str(ROOT/n):digest(ROOT/n) for n in SOURCES},
             'requirements':['Schur/full','QR/full','one prior insertion','growth projection','angular ell=6 reconstruction','auto-only AP/covariance attribution']}
    p=dest/'request.json'
    if p.exists():
        if read_json(p)!=request:raise ValueError('supplemental audit binding changed')
    else:write_json(p,request)
    return dest,request


def audit_grid(output,grid):
    output=Path(output);dest,request=binding(output);physical=read_json(output/'request.json');checks=[];start=time.monotonic()
    for model,base in [('old',Path(physical['old_study'])),('CT-A',output)]:
        for name in PUBLIC:
            p=base/'inference'/grid/'profiles'/name
            checks.append({'model':model,'grid':grid,'public_case':PUBLIC[name],**audit_result(p)})
            print(grid,model,PUBLIC[name],'Schur/QR passed',flush=True)
    record={'status':'passed','grid':grid,'request_sha256':digest(dest/'request.json'),'checks':checks,'elapsed_seconds':time.monotonic()-start}
    write_json(dest/(grid+'.json'),record);return record


def angular_auto_completion(delta,pairs,nmu):
    autos=[i for i,p in enumerate(pairs) if p[0]==p[1]]
    mu,w=np.polynomial.legendre.leggauss(nmu);l6=np.polynomial.legendre.Legendre.basis(6)(mu)
    p6=13/2*np.einsum('pkm,m,m->pk',delta[autos],w,l6)
    residual=delta[autos]-p6[:,:,None]*l6
    if np.max(abs(residual))>1e-8:raise ValueError('changed angular autos are not explained by ell=6')
    return {'auto_delta_P6_max_abs':float(np.max(abs(p6))),
            'auto_ell6_reconstruction_max_abs':float(np.max(abs(residual))),
            'auto_delta_angular_max_abs':float(np.max(abs(delta[autos])))}


def auto_attribution(output):
    output=Path(output);dest,request=binding(output);physical=read_json(output/'request.json');grid='final-loop'
    old,_=load_corrected_result(Path(physical['old_study'])/'inference'/grid/'profiles/A')
    new,_=load_corrected_result(output/'inference'/grid/'profiles/A');controls={}
    for label,newd,newc in [('old',False,False),('AP_response_only',True,False),('covariance_only',False,True),('CT-A',True,True)]:
        parts=[]
        for oc,nc in zip(old.likelihood.cells,new.likelihood.cells):
            if oc.names!=nc.names or oc.metadata!=nc.metadata:raise ValueError('auto-only cell inventories differ')
            d=nc.derivatives if newd else oc.derivatives;c=nc.covariance if newc else oc.covariance
            f=fisher_from_shells(d,c,new.likelihood.grid.centers) if nc.names else nc.F_like
            cell=replace(nc,derivatives=d,covariance=c,F_like=f)
            parts.append({'cell':cell,'fiducials':{n:float(new.likelihood.fiducial[new.names.index(n)]) for n in nc.names}})
        like=assemble_from_template(new.likelihood,parts);result=infer_study(like)
        if label in ('old','CT-A'):
            reference=old if label=='old' else new
            require_close(result.covariance,reference.covariance,'auto attribution endpoint',rtol=1e-10,atol=1e-12)
        controls[label]=result_row(grid,'A',label,output/'inference'/grid/'profiles/A',result)
    records=[]
    for receiptpath in (output/'inference'/grid/'physical').glob('*/receipt.json'):
        receipt=read_json(receiptpath)
        if 'diagnostic' not in receipt:continue
        original=reconstructed_source(receipt['source'])
        with np.load(receiptpath.parent/'arrays.npz',allow_pickle=False) as data:
            delta=data['pkmu']-original['arrays']['pkmu']
        pairs=original['inputs']['metadata']['theoretical_pairs']
        records.append({'physical_receipt':str(receiptpath),'receipt_sha256':digest(receiptpath),
            'diagnostic':receipt['diagnostic'],**angular_auto_completion(delta,pairs,original['inputs']['configuration']['n_mu'])})
    result={'status':'passed','request_sha256':digest(dest/'request.json'),
        'interpretation':'The two hybrid A controls are attribution diagnostics, not independent physical forecast prescriptions. Intrinsic fitted auto responses are preserved, so the changed A mean derivative is its AP completion; the changed covariance includes both angular auto completion and corrected theoretical cross-spectra in mutual auto covariance. The covariance-only control does not isolate these two contributions.',
        'controls':controls,'angular_cells':records}
    write_json(dest/'auto-attribution.json',result);return result


def verify(output):
    output=Path(output);dest,request=binding(output)
    records=[read_json(dest/(g+'.json')) for g in GRID_PROFILES]+[read_json(dest/'auto-attribution.json')]
    for r in records:
        if r['status']!='passed' or r['request_sha256']!=digest(dest/'request.json'):raise ValueError('incomplete independent audit')
        for c in r.get('checks',[]):
            for name,sha in c['input_files'].items():
                if digest(name)!=sha:raise ValueError('audited result changed')
    result={'status':'passed','physical_request_sha256':request['physical_request_sha256'],
        'request_sha256':digest(dest/'request.json'),'receipts':{str(p):digest(p) for p in dest.glob('*.json') if p.name!='verification.json'}}
    write_json(dest/'verification.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--grid',choices=list(GRID_PROFILES));p.add_argument('--phase',choices=['grid','attribution','verify'],required=True);a=p.parse_args()
    if a.phase=='grid':r=audit_grid(a.output,a.grid)
    elif a.phase=='attribution':r=auto_attribution(a.output)
    else:r=verify(a.output)
    print(r['status'],flush=True)
if __name__=='__main__':main()
