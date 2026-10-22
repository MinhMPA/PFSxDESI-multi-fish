#!/usr/bin/env python
"""Validate density sensitivities and report the common-survey comparison."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import csv
import json

import numpy as np

from scripts.run_pfs_density_scan import (FACTORS, GRID_PROFILES, REFINEMENTS,
    digest, load_request, write_json, fresh_output, window_profiles)
from scripts.pfs_density_inference import WINDOWS, PRIOR_MODES

QUANTITIES=('ln10_10_As','Mnu','Omegam','fsigma8_z1.1')


def response_rows(points):
    expected={f'w{i:02d}-{label}' for i in range(8) for label in FACTORS}
    if set(points)!=expected:raise ValueError('incomplete density-point inventory')
    rows=[]
    for i,window in enumerate(WINDOWS):
        count=points[f'w{i:02d}-fiducial']['profiles']['C22']['count_at_q1']
        if not np.isfinite(count) or count<=0:raise ValueError('invalid PFS reference count')
        for label,q in FACTORS.items():
            point=points[f'w{i:02d}-{label}']
            if (point['status']!='passed' or point['factor']!=q or point['window_index']!=i
                    or point['profiles']['C22']['count_at_q1']!=count):
                raise ValueError('failed or inconsistent density point')
        for mode in PRIOR_MODES:
            values={label:points[f'w{i:02d}-{label}']['profiles']['C22']['prior_modes'][mode]['sigma']
                    for label in FACTORS}
            if any(set(v)!=set(QUANTITIES) or not np.isfinite(list(v.values())).all()
                   or min(v.values())<=0 for v in values.values()):
                raise ValueError('invalid marginalized uncertainty')
            for parameter in QUANTITIES:
                fine=-(np.log(values['ln_p005'][parameter])-np.log(values['ln_m005'][parameter]))/.1
                coarse=-(np.log(values['ln_p010'][parameter])-np.log(values['ln_m010'][parameter]))/.2
                limit=max(1e-6,.02*abs(fine));error=abs(fine-coarse)
                rows.append({'window_index':i,'z_min':window[0],'z_max':window[1],
                    'comparison_profile':'C22','prior_mode':mode,'parameter':parameter,
                    'count_at_q1':count,'elasticity':float(fine),'coarse_elasticity':float(coarse),
                    'per_added_galaxy':float(fine/count),'step_difference':float(error),
                    'step_tolerance':float(limit),'step_check':'passed' if error<limit else 'failed',
                    'finite_error_reduction_percent':{label:float(100*(1-values[label][parameter]/values['fiducial'][parameter]))
                                                     for label in ('times2','times5','times10')}})
    return rows


def _constraint_values(points, references=None):
    values={}
    for key,point in points.items():
        for name,profile in point['profiles'].items():
            for mode in PRIOR_MODES:
                for parameter,value in profile['prior_modes'][mode]['sigma'].items():
                    values[f'{key}/{name}/{mode}/{parameter}']=value
        for a,b in [('D','C'),('H21','C21')]:
            if a not in point['profiles']:continue
            for mode in PRIOR_MODES:
                x=point['profiles'][a]['prior_modes'][mode]['sigma']
                y=point['profiles'][b]['prior_modes'][mode]['sigma']
                for parameter in QUANTITIES:
                    values[f'{key}/{a}_over_{b}/{mode}/{parameter}']=x[parameter]/y[parameter]
                    values[f'{key}/{b}_over_{a}/{mode}/{parameter}']=y[parameter]/x[parameter]
    if references is not None:
        for name,sigma in references.items():
            for parameter,value in sigma.items():values[f'reference/{name}/{parameter}']=value
        for row in baseline_gain_rows(points,references):
            key=f'w{row["window_index"]:02d}/q{row["factor"]:.17g}/{row["comparison"]}/{row["prior_mode"]}'
            for direction in ('ratio_before_over_after','ratio_after_over_before'):
                for parameter,value in row[direction].items():values[f'{key}/{direction}/{parameter}']=value
    return values


def compare_grids(coarse, fine, coarse_rows, fine_rows, coarse_reference=None, fine_reference=None):
    if (coarse_reference is None)!=(fine_reference is None):raise ValueError('both grid references are required')
    a=_constraint_values(coarse,coarse_reference);b=_constraint_values(fine,fine_reference)
    if a.keys()!=b.keys() or len(coarse_rows)!=len(fine_rows):raise ValueError('grid comparison inventory differs')
    metrics=[]
    for key,x in a.items():
        change=abs(b[key]/x-1.)
        metrics.append({'kind':'uncertainty_or_ordered_ratio','label':key,'change':float(change),
                        'limit':.01,'status':'passed' if change+8*np.finfo(float).eps<.01 else 'failed'})
    for x,y in zip(coarse_rows,fine_rows):
        if any(x[k]!=y[k] for k in ('window_index','prior_mode','parameter','count_at_q1')):
            raise ValueError('density response identities differ between grids')
        change=abs(y['elasticity']-x['elasticity']);limit=max(1e-5,.01*abs(y['elasticity']))
        metrics.append({'kind':'density_elasticity','label':f'w{y["window_index"]:02d}/{y["prior_mode"]}/{y["parameter"]}',
                        'change':change,'limit':limit,'status':'passed' if change<limit else 'failed'})
    return {'status':'passed' if all(m['status']=='passed' for m in metrics) else 'failed','metrics':metrics}


def verify_calculation(c):
    cov=np.asarray(c['cosmological_covariance']);f=np.asarray(c['global_information'])
    if cov.shape!=(3,3) or f.shape!=(3,3) or not np.isfinite(cov).all() or not np.isfinite(f).all():
        raise ValueError('invalid saved density covariance/information')
    for a in (cov,f):
        if np.any(np.diag(a)<=0):raise ValueError('nonpositive saved covariance/information diagonal')
        normalized=a/np.sqrt(np.outer(np.diag(a),np.diag(a)))
        if np.max(abs(normalized-normalized.T))>1e-10:raise ValueError('saved density matrices are not symmetric')
        np.linalg.cholesky(normalized)
    scale=np.sqrt(np.diag(f));res=(f@cov-np.eye(3))*scale[None,:]/scale[:,None]
    if np.max(abs(res))>1e-10:raise ValueError('saved density inverse does not reconstruct')
    j=np.asarray(c['growth_jacobian']);g=np.asarray(c['growth_covariance']);z=np.asarray(c['growth_z'])
    if not np.allclose(g,j@cov@j.T,rtol=1e-10,atol=1e-12):raise ValueError('saved growth propagation differs')
    i=np.flatnonzero(np.isclose(z,1.1,rtol=0,atol=1e-12))
    if len(i)!=1:raise ValueError('invalid growth redshift identity')
    sigma=np.r_[np.sqrt(np.diag(cov)),np.sqrt(g[i[0],i[0]])]
    if not np.allclose(sigma,[c['sigma'][q] for q in QUANTITIES],rtol=1e-12,atol=1e-14):
        raise ValueError('saved uncertainty table differs from covariance')


def read_grid(output, grid, request_hash):
    root=Path(output)/grid
    summary=json.loads((root/'summary.json').read_text())
    if (summary['status']!='passed' or summary['grid']!=grid or summary['request_sha256']!=request_hash
            or summary['settings']!=GRID_PROFILES[grid]):raise ValueError('failed or unrelated density grid')
    points={};bindings={str(root/'summary.json'):digest(root/'summary.json')}
    for key,record in summary['points'].items():
        path=root/key/'point.json'
        if digest(path)!=record['point_sha256']:raise ValueError('changed density point')
        p=json.loads(path.read_text());bindings[str(path)]=record['point_sha256']
        if p['status']!='passed' or p['request_sha256']!=request_hash:raise ValueError('failed density point')
        i=p['window_index']
        if (p['grid']!=grid or p['window']!=list(WINDOWS[i])
                or set(p['profiles'])!=set(window_profiles(i))|{'C22'}):
            raise ValueError('density point support differs')
        for name,h in p['physical'].items():
            if digest(path.parent/name)!=h:raise ValueError('changed physical density qualification')
            bindings[str(path.parent/name)]=h
        for name,profile in p['profiles'].items():
            for c in profile['prior_modes'].values():verify_calculation(c)
        points[key]=p
    response_rows(points)
    return points,bindings


def scientific_checks(points):
    checks=[];reference=None
    for i in range(8):
        fid=points[f'w{i:02d}-fiducial']['profiles']['C22']['prior_modes']['dimensionless']['sigma']
        if reference is None:reference=fid
        for p in QUANTITIES:
            checks.append({'test':'common_survey_reference','window':i,'parameter':p,
                           'passed':bool(np.isclose(fid[p],reference[p],rtol=1e-10,atol=1e-14))})
        for label,q in FACTORS.items():
            pair=points[f'w{i:02d}-{label}']['profiles']['C22']['prior_modes']
            if q<1.:continue
            for p in QUANTITIES:
                dim=pair['dimensionless']['sigma'][p];fixed=pair['fixed_physical']['sigma'][p]
                checks.append({'test':'more_sampling_and_stronger_prior_order','window':i,
                               'factor':q,'parameter':p,'passed':bool(dim<=fixed*(1+1e-10) and fixed<=fid[p]*(1+1e-10))})
    return {'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks}


def cross_spectrum_rows(points):
    rows=[]
    for key,p in points.items():
        i=p['window_index']
        if i==7:continue
        a,b=('D','C') if i<5 else ('H21','C21')
        for mode in PRIOR_MODES:
            x=p['profiles'][a]['prior_modes'][mode]['sigma'];y=p['profiles'][b]['prior_modes'][mode]['sigma']
            rows.append({'window_index':i,'factor':p['factor'],'prior_mode':mode,'comparison':a+'_to_'+b,
                         'error_reduction_percent':{q:100*(1-y[q]/x[q]) for q in QUANTITIES}})
    return rows


def total_pfs_rows(points, baseline):
    rows=[]
    for point in points.values():
        for mode in PRIOR_MODES:
            sigma=point['profiles']['C22']['prior_modes'][mode]['sigma']
            rows.append({'window_index':point['window_index'],'factor':point['factor'],
                         'prior_mode':mode,'comparison':'B_to_C22',
                         'error_reduction_percent':{q:100*(1-sigma[q]/baseline[q]) for q in QUANTITIES}})
    return rows


def read_references(request,binding,grid):
    path=Path(request['study'])/'inference'/grid/'summary.json'
    binding.check(path);h=digest(path);saved=json.loads(path.read_text())
    if saved['status']!='passed' or saved['grid']!=grid or saved['settings']!=GRID_PROFILES[grid]:
        raise ValueError('unqualified or unrelated density comparison reference')
    references={}
    for name in ('B','C21'):
        row=saved['outcomes'][name]
        z=np.asarray(row['growth_z']);i=np.flatnonzero(np.isclose(z,1.1,rtol=0,atol=1e-12))
        if row['status']!='passed' or tuple(row['cosmology_names'])!=QUANTITIES[:3] or len(i)!=1:
            raise ValueError('invalid reference cosmology/growth identity')
        sigma=[*row['marginalized']['cosmology'],row['marginalized']['growth'][i[0]]]
        if len(sigma)!=4 or not np.isfinite(sigma).all() or min(sigma)<=0:
            raise ValueError('invalid reference uncertainty')
        references[name]=dict(zip(QUANTITIES,sigma))
    binding.check(path)
    return references,{str(path):h}


def baseline_gain_rows(points,references):
    rows=[]
    for point in points.values():
        i=point['window_index']
        comparisons=[('B','C22')]+([('B','D'),('B','C')] if i<5 else
                                  [('B','H21'),('B','C21')] if i<7 else [('C21','C22')])
        for a,b in comparisons:
            for mode in PRIOR_MODES:
                x=references[a];y=point['profiles'][b]['prior_modes'][mode]['sigma']
                rows.append({'window_index':i,'factor':point['factor'],'prior_mode':mode,
                    'comparison':a+'_to_'+b,'reference_density_factor':1.,
                    'error_reduction_percent':{q:100*(1-y[q]/x[q]) for q in QUANTITIES},
                    'ratio_before_over_after':{q:x[q]/y[q] for q in QUANTITIES},
                    'ratio_after_over_before':{q:y[q]/x[q] for q in QUANTITIES}})
    return rows


def density_rankings(rows):
    """Conservative numerical-resolution groups, not statistical confidence limits."""
    selected=rows['shell304'];rankings={}
    for mode in PRIOR_MODES:
        rankings[mode]={}
        for q in QUANTITIES:
            own=[r for r in selected if r['prior_mode']==mode and r['parameter']==q]
            for r in own:
                corresponding=[next(t for t in rr if t['window_index']==r['window_index']
                    and t['prior_mode']==mode and t['parameter']==q) for rr in rows.values()]
                r['grid_spread']=max(abs(t['elasticity']-r['elasticity']) for t in corresponding)
                r['step_spread']=max(t['step_difference'] for t in corresponding)
                # Do not claim ordering more precise than the declared acceptance scales.
                grid_resolution=max(r['grid_spread'],1e-5,.01*abs(r['elasticity']))
                step_resolution=max(r['step_spread'],1e-6,.02*abs(r['elasticity']))
                r['elasticity_numerical_resolution']=grid_resolution+step_resolution
                r['per_added_galaxy_numerical_resolution']=(grid_resolution+step_resolution)/r['count_at_q1']
                r['per_added_galaxy_grid_spread']=r['grid_spread']/r['count_at_q1']
            rankings[mode][q]={}
            for field in ('elasticity','per_added_galaxy'):
                groups=[];lower=None
                # Upper-edge ordering visits wide bridging intervals before
                # any lower group they connect; central-value ordering cannot.
                for r in sorted(own,key=lambda x:x[field]+x[field+'_numerical_resolution'],reverse=True):
                    delta=r[field+'_numerical_resolution'];lo=r[field]-delta;hi=r[field]+delta
                    if lower is None or hi<lower:
                        groups.append([r['window_index']]);lower=lo
                    else:
                        groups[-1].append(r['window_index']);lower=min(lower,lo)
                rankings[mode][q][field]=[sorted(group) for group in groups]
    return rankings


def make_figure(rows,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex=True,sharey=True)
    fields=('elasticity','per_added_galaxy')
    for ir,field in enumerate(fields):
        matrices=[]
        for mode in PRIOR_MODES:
            matrices.append(np.array([[next(r[field] for r in rows if r['window_index']==i
                                and r['prior_mode']==mode and r['parameter']==q) for q in QUANTITIES] for i in range(8)]))
        factor=1. if ir==0 else 1e8
        vmax=max(np.max(m) for m in matrices)*factor
        for j,m in enumerate(matrices):
            im=axes[ir,j].imshow(m*factor,aspect='auto',vmin=0,vmax=vmax,cmap='viridis')
            axes[ir,j].set_yticks(range(8),[f'{a:g}–{b:g}'+(' (ext.)' if i>=5 else '')
                                         for i,(a,b) in enumerate(WINDOWS)])
            axes[ir,j].axhline(4.5,color='white',linewidth=1.2)
            axes[ir,j].set_xticks(range(4),[r'$\ln(10^{10}A_s)$',r'$M_\nu$',r'$\Omega_m$',r'$f\sigma_8(1.1)$'])
        fig.colorbar(im,ax=axes[ir,:],label=r'$E=-\partial\ln\sigma/\partial\ln n_{\rm PFS}$' if ir==0 else r'$10^8 E/N_{\rm PFS}$')
    axes[0,0].set_title('Fixed dimensionless noise priors');axes[0,1].set_title('Fixed physical noise priors')
    axes[0,0].set_ylabel('PFS redshift interval');axes[1,0].set_ylabel('PFS redshift interval')
    for suffix in ('png','pdf','svg'):fig.savefig(dest/('density-response.'+suffix),dpi=180)
    plt.close(fig)


def _report_to(output,dest,request,binding,request_hash):
    points={};rows={};bindings={};references={};program_hash=digest(__file__)
    for grid in GRID_PROFILES:
        points[grid],files=read_grid(output,grid,request_hash);bindings.update(files)
        references[grid],files=read_references(request,binding,grid);bindings.update(files)
        rows[grid]=response_rows(points[grid])
    comparisons={a+'_to_'+b:compare_grids(points[a],points[b],rows[a],rows[b],references[a],references[b])
                 for a,b,_ in REFINEMENTS}
    science={g:scientific_checks(p) for g,p in points.items()}
    passed=(all(x['status']=='passed' for x in comparisons.values()) and
            all(x['status']=='passed' for x in science.values()) and
            all(r['step_check']=='passed' for rr in rows.values() for r in rr))
    selected=rows['shell304'];rankings=density_rankings(rows)
    result={'status':'incomplete','scope':'Conditional fixed-clustering density experiment; common C22 coverage through z=2.2.',
            'response_rows':selected,'rankings':rankings,'grid_comparisons':comparisons,'scientific_checks':science,
            'ranking_convention':'Groups ordered by benefit; bins connected by overlapping numerical-resolution intervals remain unresolved. Resolution is max(grid spread,1e-5,1% |E|) + max(step difference,1e-6,2% |E|), divided by N for E/N. This is not a statistical confidence interval.',
            'all_grid_response_rows':rows,'cross_spectrum_gains':cross_spectrum_rows(points['shell304']),
            'baseline_PFS_gains':baseline_gain_rows(points['shell304'],references['shell304']),
            'total_PFS_gains':total_pfs_rows(points['shell304'],references['shell304']['B']),
            'reference_uncertainties':references,
            'input_files':bindings,'report_program_sha256':program_hash,'request_sha256':request_hash}
    write_json(dest/'summary.json',result)
    with (dest/'density-response.csv').open('w',newline='') as handle:
        fields=[k for k in selected[0] if k!='finite_error_reduction_percent']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in selected)
    lines=['# PFS density response in a common cosmological fit','',
           'Status: '+result['status']+'. All windows use C22 (PFS through z=2.2) as the common reference.',
           'Clustering statistics and observing geometry are fixed; the adopted density table remains conditional.',
           'Growth is derived from the same three global cosmological coordinates.','',
           '| PFS interval | Noise-prior convention | E for ln As | E for Mnu | E for Omega_m | E for growth at 1.1 |','|---|---|---:|---:|---:|---:|']
    for i,(a,b) in enumerate(WINDOWS):
        for mode in PRIOR_MODES:
            vals=[next(r['elasticity'] for r in selected if r['window_index']==i and r['prior_mode']==mode and r['parameter']==q) for q in QUANTITIES]
            lines.append(f'| {a:g}–{b:g} | {mode} | '+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines+=['','E is a local fractional error response to a fractional density increase. E/N compares equal added galaxies.',
            'The two prior conventions separate improved sampling from density-dependent tightening of physical stochastic-power priors.',
            'Finite-factor, DESI-baseline, tail and cross-spectrum gains are in summary.json. D/C and H21/C21 use their original coverages;',
            'their percentages have different denominators and must not be added. Rankings retain unresolved groups under the declared numerical resolution.',
            'Neither ranking is an observing-time optimization, a bias optimization, or a six-coordinate cosmological forecast.']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if passed:make_figure(selected,dest)
    (dest/'FIGURE_CAPTION.md').write_text('Density sensitivity in a common DESI+PFS forecast through z=2.2. Top: E=-d ln sigma/d ln n_PFS. Bottom: 10^8 E/N_PFS, the local response per added galaxy. Columns compare fixed dimensionless stochastic priors with fixed physical stochastic-power priors. Each row changes one PFS interval only, at fixed clustering and survey geometry. The white line separates the original 0.7–1.6 PFS baseline from the high-z extension (ext.). Density steps and numerical refinements are checked separately. The high-z density provenance remains unresolved; these are conditional responses, not an observing-time or target-selection optimum.\n')
    load_request(output)
    if digest(output/'request.json')!=request_hash:raise ValueError('density request changed during reporting')
    if digest(__file__)!=program_hash:raise ValueError('density reporting program changed during reporting')
    if any(digest(p)!=h for p,h in bindings.items()):raise ValueError('density results changed during reporting')
    result['status']='passed' if passed else 'failed'
    write_json(dest/'summary.json',result)
    prose=dest/'RESULTS.md';prose.write_text(prose.read_text().replace('Status: incomplete.','Status: '+result['status']+'.',1))
    write_json(dest/'files.json',{p.name:digest(p) for p in sorted(dest.iterdir()) if p.is_file()})
    return {'status':result['status'],'response_rows':len(selected)}


def report(output):
    output=Path(output);request,binding=load_request(output);request_hash=digest(output/'request.json')
    final=output/'report';dest=output/'report.incomplete'
    if final.exists():raise FileExistsError(str(final))
    fresh_output(dest)
    write_json(dest/'summary.json',{'status':'incomplete'})
    try:
        result=_report_to(output,dest,request,binding,request_hash)
        dest.rename(final)
    except Exception as error:
        import traceback
        saved=json.loads((dest/'summary.json').read_text())
        saved.update(status='failed',failure=str(error),failure_type=type(error).__name__)
        write_json(dest/'summary.json',saved)
        prose=dest/'RESULTS.md'
        body=prose.read_text() if prose.exists() else ''
        prose.write_text('Status: failed. Report was not published.\n\n'+body.replace('Status: passed.','Status: failed.').replace('Status: incomplete.','Status: failed.'))
        (dest/'failure.txt').write_text(traceback.format_exc())
        if (dest/'files.json').exists():(dest/'files.json').rename(dest/'incomplete-files.json')
        raise
    return {**result,'report':str(final)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    result=report(parser.parse_args().output);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)
