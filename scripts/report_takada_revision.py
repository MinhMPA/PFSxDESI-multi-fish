#!/usr/bin/env python
"""Read-only numerical qualification and scientific tables for Takada n(z)."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))
import argparse
import json
import numpy as np
from pfsfog.elf_survey_evidence import file_digest,write_json,verify_files
from pfsfog.extended_artifacts import read_json,jsonable,canonical,require_close
from pfsfog.revision_validation import compare_nested_likelihoods
from scripts.run_takada_revision import request_and_binding,GRID_PROFILES,REFINEMENTS
from scripts.run_takada_density_scan import density_windows
from scripts.takada_study import corrected_profiles,load_corrected_result
from scripts.elf_study_report import constraint_row,nesting_snapshot,comparisons
from scripts.elf_grid_comparison import compare_constraints
from scripts.report_pfs_density_scan import verify_calculation,density_rankings,QUANTITIES
from scripts.pfs_bias_broad_comparison import _errors
from scripts.pfs_density_inference import PRIOR_MODES
from scripts.run_pfs_density_scan import FACTORS
from scripts.takada_report import compare_values,density_responses,main_values


def bind_file(path,bindings):
    bindings[str(path)]=file_digest(path);return read_json(path)


def require_files(path,record,bindings):
    errors=verify_files(path,record)
    if errors:raise ValueError('; '.join(errors))
    bindings.update({str(path/n):h for n,h in record.items()})


def verify_extended_audits(root,case,manifest,bindings):
    expected=set() if case=='B' else {
        case+'-'+c['metadata']['cell_id']+'-physical.json' for c in manifest['likelihood']['cells']}
    if set(manifest['physical_checks'])!=expected:
        raise ValueError('extended physical audit inventory differs')
    cosmology={'ln10_10_As','Mnu','Omegam','h','omega_b','n_s'}
    for name,digest in manifest['physical_checks'].items():
        path=root/name
        if file_digest(path)!=digest:raise ValueError('extended physical audit changed')
        audit=bind_file(path,bindings)
        checked={r['column'] for r in audit['derivative_check']['column_results']}
        if (any(audit[k]['status']!='passed' for k in ('derivative_check','neighborhood_check'))
                or audit['mapping']['fixed_cosmology'] or not cosmology<=checked):
            raise ValueError('incomplete six-coordinate physical audit')


def main_grid(base,grid,bindings):
    path=base/'inference'/grid;s=bind_file(path/'summary.json',bindings)
    receipt=bind_file(base/'logs'/(grid+'-execution.json'),bindings)
    if s['status']!='passed' or receipt['actual_exit']!=0 or set(s['outcomes'])!=set(corrected_profiles(2.4)):
        raise ValueError('unqualified corrected main comparison: '+grid)
    require_files(path/'projections',s['projection_files'],bindings)
    for name,p in s['profiles'].items():
        require_files(path/'profiles'/name,p['files'],bindings)
        result,_=load_corrected_result(path/'profiles'/name)
        if canonical(constraint_row(result))!=canonical(s['outcomes'][name]):raise ValueError('saved constraint row differs')
    if any(r['status']!='passed' for r in s['nesting'].values()):raise ValueError('main observation nesting failed')
    # The new tail also satisfies the stronger full observation/parameter test.
    tail,_=load_corrected_result(path/'profiles/C24');checks={}
    for name in ('B16','C21'):
        before,_=load_corrected_result(path/'profiles'/name)
        checks[name+'_to_C24']=compare_nested_likelihoods(nesting_snapshot(before),nesting_snapshot(tail))
    if any(r['status']!='passed' for r in checks.values()):raise ValueError('new tail strict nesting failed')
    return s,checks


def read_points(base,grid,kind,bindings):
    root=base/kind/grid;s=bind_file(root/'summary.json',bindings)
    receipt=bind_file(base/'logs'/(kind+'-'+grid+'-execution.json'),bindings)
    if s['status'] not in ('passed','completed_with_failures') or receipt['actual_exit'] not in (0,1):
        raise ValueError('incomplete '+kind+' calculation')
    if s['baseline_summary_sha256']!=file_digest(base/'inference'/grid/'summary.json'):
        raise ValueError('followup baseline changed')
    program=ROOT/('scripts/run_takada_'+('density_scan' if kind=='density' else kind)+'.py')
    if s['source_sha256']!=file_digest(program):raise ValueError('followup source changed')
    bindings[str(program)]=file_digest(program)
    expected=64 if kind=='density' else 35
    if len(s['points'])!=expected:raise ValueError('incomplete followup inventory')
    points={}
    for key,record in s['points'].items():
        path=root/key/'point.json'
        if file_digest(path)!=record['point_sha256']:raise ValueError('point changed')
        p=bind_file(path,bindings)
        if p['status']!=record['status']:raise ValueError('point status changed')
        for n,h in p.get('physical',{}).items():
            if file_digest(path.parent/n)!=h:raise ValueError('physical audit changed')
            audit=bind_file(path.parent/n,bindings)
            if any(audit[k]['status']!='passed' for k in ('derivative_check','neighborhood_check')):
                raise ValueError('failed physical audit presented as accepted')
        if p['status']=='passed':
            for profile in p['profiles'].values():
                for c in (profile['prior_modes'].values() if kind=='density' else [profile]):verify_calculation(c)
            if kind=='bias' and p['cross_information']['status']!='passed':raise ValueError('bias cross information failed')
        elif not p.get('error'):raise ValueError('failed point lacks reason')
        points[key]=p
    return points


def four(row):
    i=row['growth_z'].index(1.1)
    return dict(zip(QUANTITIES,[*row['marginalized']['cosmology'],row['marginalized']['growth'][i]]))


def point_values(points,kind,baseline):
    values={};statuses={}
    for key,p in points.items():
        statuses[key]=p['status']
        if p['status']!='passed':continue
        profiles=p['profiles'];errors={}
        if kind=='density':
            for name,r in profiles.items():
                for mode in PRIOR_MODES:errors[name+'/'+mode]=r['prior_modes'][mode]['sigma']
            for mode in PRIOR_MODES:
                before=four(baseline['B'])
                for name in profiles:
                    after=errors[name+'/'+mode]
                    for q in QUANTITIES:
                        values[key+'/ratio/B_over_'+name+'/'+mode+'/'+q]=before[q]/after[q]
                        values[key+'/ratio/'+name+'_over_B/'+mode+'/'+q]=after[q]/before[q]
        else:
            for name,r in profiles.items():
                for mode in ('marginalized','nuisance_fixed'):errors[name+'/'+mode]=_errors(r,mode)
        for name,row in errors.items():
            for q,value in row.items():values[key+'/'+name+'/'+q]=value
        for a,b in (('D','C'),('H21','C21')):
            if a not in profiles:continue
            modes=PRIOR_MODES if kind=='density' else ('marginalized','nuisance_fixed')
            for mode in modes:
                for q,x in errors[a+'/'+mode].items():
                    y=errors[b+'/'+mode][q]
                    values[key+'/ratio/'+a+'_over_'+b+'/'+mode+'/'+q]=x/y
                    values[key+'/ratio/'+b+'_over_'+a+'/'+mode+'/'+q]=y/x
    return values,statuses


def density_physics(points):
    reference=points['w00-fiducial']['profiles']['C24']['prior_modes']['dimensionless']['sigma'];checks=[]
    for key,p in points.items():
        if p['status']!='passed':continue
        modes=p['profiles']['C24']['prior_modes'];q=p['factor']
        for name in QUANTITIES:
            dim=modes['dimensionless']['sigma'][name];fixed=modes['fixed_physical']['sigma'][name]
            if q==1.:
                good=np.isclose(dim,reference[name],rtol=1e-10,atol=1e-14) and np.isclose(fixed,reference[name],rtol=1e-10,atol=1e-14)
            elif q>=1.:good=dim<=fixed*(1+1e-10) and fixed<=reference[name]*(1+1e-10)
            else:continue
            checks.append({'point':key,'parameter':name,'passed':bool(good)})
    return {'status':'passed' if all(r['passed'] for r in checks) else 'failed','checks':checks}


def physical_report(base,kind,bindings):
    root=base/'physical-comparisons'/kind;s=bind_file(root/'execution.json',bindings)
    if s['status']!='passed' or {r['key'] for r in s['workers']}!=set(s['keys']):raise ValueError('physical comparisons incomplete')
    for row in s['workers']:
        p=root/row['key']/'report.json'
        if row['actual_exit']!=0 or file_digest(p)!=row['report_sha256']:raise ValueError('physical comparison exit or binding failed')
        r=bind_file(p,bindings)
        if r['status']!='passed' or any(v['status']!='passed' for v in r['comparisons']):raise ValueError('physical refinement failed')
        for f,h in r['input_files'].items():
            if file_digest(f)!=h:raise ValueError('physical source changed after comparison')
            bindings[f]=h
        if file_digest(p.parent/'query.npz')!=r['query_sha256']:raise ValueError('direct query changed')
        bindings[str(p.parent/'query.npz')]=r['query_sha256']
    return {'status':'passed','sources':len(s['keys'])}


def report(base,output,*,main_only=False):
    base=Path(base).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    request,binding=request_and_binding(base,verify_all=True);bindings={str(base/'request.json'):file_digest(base/'request.json')}
    for source in (Path(__file__).resolve(),ROOT/'scripts/takada_report.py'):
        bindings[str(source)]=file_digest(source)
    mains={};nesting={};density={};bias={};extended={};responses={}
    for g in GRID_PROFILES:
        mains[g],nesting[g]=main_grid(base,g,bindings)
        if main_only:continue
        density[g]=read_points(base,g,'density',bindings);bias[g]=read_points(base,g,'bias',bindings)
        responses[g]=density_responses(density[g],density_windows(request['pfs_zmin'],request['pfs_zmax']),'C24')
        extended[g]=bind_file(base/'extended'/g/'summary.json',bindings)
        if extended[g]['status']!='passed':raise ValueError('extended result incomplete')
        receipts=[p for p in (base/'logs').glob('extended-'+g+'*-execution.json')
                  if read_json(p).get('actual_exit')==0]
        if len(receipts)!=1:raise ValueError('missing or ambiguous successful extended execution')
        bind_file(receipts[0],bindings)
        p=ROOT/'scripts/run_takada_extended.py'
        if file_digest(p)!=extended[g]['source_sha256']:raise ValueError('extended source changed')
        bindings[str(p)]=file_digest(p)
        support=extended[g]['support_source']
        if set(support)!={str(ROOT/'scripts/takada_extended_checks.py')}:raise ValueError('extended checker inventory differs')
        for source,digest in support.items():
            if file_digest(source)!=digest:raise ValueError('extended checker changed')
            bindings[source]=digest
        for name,row in extended[g]['cases'].items():
            require_files(base/'extended'/g/name,row['files'],bindings)
            manifest=read_json(base/'extended'/g/name/'manifest.json')
            verify_extended_audits(base/'extended'/g,name,manifest,bindings)
    reports=[]
    for a,b,axis in REFINEMENTS:
        row={'coarse':a,'fine':b,'axis':axis,'main':compare_values(main_values(mains[a]['outcomes']),main_values(mains[b]['outcomes']))}
        if not main_only:
            for kind,points in (('density',density),('bias',bias)):
                x,xs=point_values(points[a],kind,mains[a]['outcomes']);y,ys=point_values(points[b],kind,mains[b]['outcomes'])
                if xs!=ys:raise ValueError('followup physical domain changes with resolution')
                row[kind]=compare_values(x,y);row[kind]['unqualified']=[k for k,v in xs.items() if v!='passed']
            elastic=[]
            for x,y in zip(responses[a],responses[b]):
                if any(x[k]!=y[k] for k in ('window_index','prior_mode','parameter','count_at_q1')):raise ValueError('elasticity identities differ')
                change=abs(y['elasticity']-x['elasticity']);limit=max(1e-5,.01*abs(y['elasticity']))
                elastic.append({'window':x['window_index'],'mode':x['prior_mode'],'parameter':x['parameter'],
                    'change':change,'limit':limit,'status':'passed' if change<limit else 'failed'})
            row['elasticity']={'status':'passed' if all(r['status']=='passed' for r in elastic) else 'failed','metrics':elastic}
            row['extended']=compare_constraints(extended[a],extended[b])
        row['status']='passed' if all(v['status']=='passed' for v in row.values() if isinstance(v,dict)) else 'failed'
        write_json(output/(a+'-to-'+b+'.json'),jsonable(row));reports.append(row)
    comparisons={}
    selected=mains['shell304']['outcomes']
    for a,b in comparisons_pairs():
        x=four(selected[a]);y=four(selected[b]);comparisons[a+'_to_'+b]={p:100*(1-y[p]/x[p]) for p in QUANTITIES}
    result={'status':'passed' if all(r['status']=='passed' for r in reports) else 'failed','main_only':main_only,
        'selected_grid':'shell304','constraints':selected,'error_reductions_percent':comparisons,
        'main_nesting':nesting,'refinements':[{k:r[k] for k in ('coarse','fine','axis','status')} for r in reports]}
    if not main_only:
        result['density_physics']={g:density_physics(p) for g,p in density.items()}
        result['density_response']=responses;result['density_rankings']=density_rankings(responses)
        result['bias_points']=bias['shell304'];result['extended']=extended['shell304']
        result['physical']={kind:physical_report(base,kind,bindings) for kind in ('main','bias')}
        if (any(r['step_check']!='passed' for rr in responses.values() for r in rr)
                or any(r['status']!='passed' for r in result['density_physics'].values())):result['status']='failed'
    for p,h in bindings.items():
        if file_digest(p)!=h:raise ValueError('report input changed during verification')
    request_and_binding(base);result['inputs']=bindings
    write_json(output/'summary.json',jsonable(result))
    return {'status':result['status'],'main_only':main_only,'input_files':len(bindings),'output':str(output)}


def comparisons_pairs():
    return comparisons()+[('C21','C24'),('B','C24'),('C','C24')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--main-only',action='store_true')
    a=p.parse_args();r=report(a.baseline,a.output,main_only=a.main_only);print(json.dumps(r,indent=2));return 0 if r['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
