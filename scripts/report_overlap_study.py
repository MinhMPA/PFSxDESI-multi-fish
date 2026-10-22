"""Authenticated errors, gain convergence and conditional cross-spectrum responses.

This reads saved multipole likelihoods only. No spectrum engine is initialized.
All-nuisance-fixed errors retain the other two cosmological coordinates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from pfsfog.extended_artifacts import canonical, unpack_likelihood
from scripts import run_overlap_study as campaign
from scripts.overlap_inputs import TREATMENTS, Condition, conditions, canonical_condition
from scripts.overlap_checks import conditional_information, _whiten
from scripts.overlap_study import scale_area
from scripts.run_elf_studies import GRID_PROFILES

PAIRS = (('A','B'), ('B','C'), ('C','D'), ('B','D'), ('A','C'), ('A','D'))
REFINEMENTS = (('loop2048','loop4096'), ('loop2048','angular64'),
               ('angular64','shell304'), ('shell304','final-loop'))
EPS = np.finfo(float).eps
MODES = ('marginalized', 'nuisance_fixed')


def gain_percent(before: float, after: float) -> float:
    """Percentage reduction in an absolute error, with the preceding error below."""
    if not np.isfinite([before,after]).all() or before <= 0 or after <= 0:
        raise ValueError('uncertainties must be positive and finite')
    return float(100*(1-after/before))


def _comparison(before, after):
    if set(before) != set(after):
        raise ValueError('comparison uncertainty inventory differs')
    return {q: {'before':before[q], 'after':after[q], 'ratio':after[q]/before[q],
                'gain_percent':gain_percent(before[q],after[q])} for q in before}


def case_comparisons(errors: dict) -> dict:
    if set(errors) != set('ABCD'):
        raise ValueError('missing A/B/C/D error inventory')
    result = {f'{a}-{b}':_comparison(errors[a],errors[b]) for a,b in PAIRS}
    for q in errors['A']:
        product=np.prod([result[p][q]['ratio'] for p in ('A-B','B-C','C-D')])
        if abs(product/result['A-D'][q]['ratio']-1) >= 1e-12:
            raise ValueError('successive error ratios do not close multiplicatively')
    return result


def area_comparisons(errors: dict) -> dict:
    expected={campaign.condition_id(c) for c in conditions()}
    if set(errors) != expected:
        raise ValueError('missing area/case/stochasticity condition')
    return {f'{case}/{s}':_comparison(errors[f'{case}-area01200-{s}'],
                                     errors[f'{case}-area14000-{s}'])
            for case in 'ABCD' for s in TREATMENTS}


def compare_grids(coarse: dict, fine: dict) -> dict:
    """Strict <1% relative errors/ratios; <0.05 percentage points for gains."""
    if canonical(coarse['identity']) != canonical(fine['identity']):
        raise ValueError('physical identity differs between grids')
    if not coarse['metrics'] or set(coarse['metrics']) != set(fine['metrics']):
        raise ValueError('named refinement metric inventory differs')
    checks={}
    for name,a in coarse['metrics'].items():
        b=fine['metrics'][name]
        kind=a['kind']; unit=a['unit']
        if kind not in ('uncertainty','ratio','gain') or (kind,unit)!=(b['kind'],b['unit']):
            raise ValueError('refinement metric kind or units differ')
        if ((kind=='gain' and unit!='percentage points')
                or (kind=='ratio' and unit!='dimensionless')):
            raise ValueError('incorrect refinement units')
        x,y=float(a['value']),float(b['value'])
        if not np.isfinite([x,y]).all(): raise ValueError('nonfinite refinement value')
        if kind=='gain':
            change=abs(y-x); guard=8*EPS*max(1.,abs(x),abs(y)); limit=.05
            change_unit='percentage points'
        else:
            if min(x,y)<=0: raise ValueError('nonpositive uncertainty or ratio')
            ratio=y/x; change=abs(ratio-1); guard=8*EPS*max(1.,abs(ratio)); limit=.01
            change_unit='fractional'
        checks[name]={'coarse':x,'fine':y,'unit':unit,'kind':kind,
                      'change':change,'change_unit':change_unit,'guard':guard,
                      'limit':limit,'status':'passed' if change+guard<limit else 'failed'}
    maxima={kind: max(((v['change'],k) for k,v in checks.items() if v['kind']==kind),
                     default=(0.,None)) for kind in ('uncertainty','ratio','gain')}
    return {'status':'passed' if all(v['status']=='passed' for v in checks.values()) else 'failed',
            'checks':checks,'maxima':{k:{'change':v[0],'metric':v[1]} for k,v in maxima.items()}}


def variance_ordering(before: dict, after: dict) -> dict:
    """Common named parameter and derived-growth variances may not increase."""
    old=dict(zip(before['names'],before['variance'],strict=True))
    new=dict(zip(after['names'],after['variance'],strict=True))
    if len(old)!=len(before['names']) or len(new)!=len(after['names']):
        raise ValueError('duplicate parameter in variance ordering')
    values={n:float(new[n]/old[n]-1) for n in old.keys() & new.keys()}
    a,b=np.asarray(before['growth_variance']),np.asarray(after['growth_variance'])
    if a.shape!=b.shape or not values: raise ValueError('variance ordering inventory differs')
    if (not np.isfinite([*old.values(),*new.values(),*a,*b]).all()
            or min(*old.values(),*new.values(),*a,*b)<=0):
        raise ValueError('nonpositive or nonfinite variance')
    values.update({f'growth/{i}':float(y/x-1) for i,(x,y) in enumerate(zip(a,b))})
    name=max(values,key=values.get)
    return {'status':'passed' if values[name]<=1e-8 else 'failed',
            'maximum_fractional_increase':values[name],'sensitive_coordinate':name,
            'allowance':1e-8,'common_coordinate_count':len(values),'changes':values}


def compensated_direction(cosmo_cov, k, n, gradient):
    """After-case unit-sigma cosmological displacement and best nuisance response."""
    c=np.asarray(cosmo_cov); g=np.asarray(gradient)
    variance=float(g@c@g)
    if not np.isfinite(variance) or variance<=0: raise ValueError('invalid directional variance')
    v=c@g/np.sqrt(variance)
    if len(n):
        scale=np.sqrt(np.diag(n))
        if np.any(scale<=0): raise ValueError('nonpositive nuisance precision')
        l=np.linalg.cholesky(n/scale[:,None]/scale[None,:])
        w=-np.linalg.solve(l.T,np.linalg.solve(l,k.T@v/scale))/scale
    else: w=np.zeros(0)
    return v,w


def response_norm(conditional, global_names, direction):
    """D_cond u squared in the inverse conditional covariance metric."""
    if len(set(global_names))!=len(global_names): raise ValueError('duplicate global names')
    index={n:i for i,n in enumerate(global_names)}
    try: local=np.asarray(direction)[[index[n] for n in conditional['names']]]
    except KeyError as exc: raise ValueError('conditional coordinate missing globally') from exc
    projected=np.einsum('koi,i->ko',conditional['D'],local)[...,None]
    white=_whiten(projected,conditional['C'])
    return float(np.sum(white**2))


def _save_array(archive, name, value):
    with archive.open(name+'.npy','w',force_zip64=True) as stream:
        np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False)


def _load_results(output, grid):
    cache={}; records={}
    for condition in conditions():
        name=campaign.condition_id(condition)
        target=campaign.condition_id(canonical_condition(condition))
        if target not in cache:
            path=output/'inference'/grid/target
            meta=campaign.read(path/'manifest.json')
            with np.load(path/'arrays.npz',allow_pickle=False) as data:
                arrays={k:data[k] for k in data.files}
            cache[target]={'meta':meta,'arrays':arrays,
                           'like':unpack_likelihood(meta['likelihood'],arrays),
                           'checks':campaign.read(path/'checks.json')['marginalization']}
        records[name]=cache[target]
    return records


def _physical_identity(records):
    result={}
    for name,r in records.items():
        like=r['like']; a=r['arrays']
        parameters=[{k:v for k,v in p.items() if k!='prior_only'} for p in like.parameters]
        result[name]={'redshifts':a['result/growth_z'].tolist(),
            'reference_cosmology':like.reference_cosmology,
            'physical_model':{k:v for k,v in like.model.items() if k not in
                ('fft_n','multipole_default_nmu','ap_interpolation_nk',
                 'ap_interpolation_nmu','ap_refinement_nk')},
            'spectrum_configuration':{k:v for k,v in vars(like.config).items()
                if k not in ('dk','n_mu','ap_interpolation_nk','ap_interpolation_nmu',
                             'ap_refinement_nk')},
            'nominal_bins':like.config.nominal_bins,'parameters':parameters,
            'density_and_cells':[{k:c.metadata[k] for k in
                ('cell_id','radial_bin','nominal_bin','model_redshift','nbar','active_populations')}
                                for c in like.cells],
            'priors':r['meta']['prior_identity'], 'solver':r['meta']['solver_identity'],
            'k_range':[like.config.kmin,like.config.kmax]}
    return result


def _errors(record):
    arrays=record['arrays']; like=record['like']
    z=arrays['result/growth_z']; pivot=np.flatnonzero(np.isclose(z,1.1,rtol=0,atol=1e-12))
    if len(z)!=8 or len(pivot)!=1: raise ValueError('missing eight growth predictions or headline pivot')
    quantities=[*like.cosmo_names,*[f'fsigma8_z{value:.12g}' for value in z]]
    units={**{p['name']:p['unit'] or 'dimensionless' for p in like.parameters[:3]},
           **{q:'dimensionless' for q in quantities[3:]}}
    result={}
    for mode,cosmo,growth in (
            ('marginalized','cov_cosmo_marg','growth_cov_marg'),
            ('nuisance_fixed','cov_cosmo_nuisance_fixed','growth_cov_nuisance_fixed')):
        variances=np.r_[np.diag(arrays['result/'+cosmo]),np.diag(arrays['result/'+growth])]
        if not np.isfinite(variances).all() or np.any(variances<=0): raise ValueError('invalid error variance')
        result[mode]=dict(zip(quantities,map(float,np.sqrt(variances)),strict=True))
    return result,units,['fsigma8_z1.1','Mnu','Omegam']


def _variance_record(record, fixed=False):
    a=record['arrays']
    return {'names':record['like'].cosmo_names if fixed else record['like'].names,
            'variance':np.diag(a['result/cov_cosmo_nuisance_fixed' if fixed else 'result/covariance']),
            'growth_variance':np.diag(a['result/growth_cov_nuisance_fixed' if fixed else 'result/growth_cov_marg'])}


def _ordering(records):
    checks={}
    def add(label,before,after):
        for fixed in (False,True):
            checks[label+('/nuisance_fixed' if fixed else '/marginalized')]=variance_ordering(
                _variance_record(records[before],fixed),_variance_record(records[after],fixed))
    for area in (1200,14000):
        for s in TREATMENTS:
            for a,b in PAIRS:
                add(f'observations/{area}/{s}/{a}-{b}',f'{a}-area{area:05d}-{s}',f'{b}-area{area:05d}-{s}')
        for case in 'ABCD':
            for a,b in (('marginalized','desi_fixed'),('marginalized','pfs_fixed'),
                        ('marginalized','all_cross_fixed'),('desi_fixed','all_cross_fixed'),
                        ('pfs_fixed','all_cross_fixed')):
                add(f'fixing/{area}/{case}/{a}-{b}',f'{case}-area{area:05d}-{a}',f'{case}-area{area:05d}-{b}')
    for case in 'ABCD':
        for s in TREATMENTS:
            add(f'area/{case}/{s}',f'{case}-area01200-{s}',f'{case}-area14000-{s}')
    # Each nuisance-fixed cosmological covariance still marginalizes the other cosmological coordinates.
    for name,r in records.items():
        a=_variance_record(r,True); b=_variance_record(r)
        b={'names':list(r['like'].cosmo_names),'variance':b['variance'][:3],
           'growth_variance':b['growth_variance']}
        checks['all_nuisance_fixing/'+name]=variance_ordering(b,a)
    return {'status':'passed' if all(c['status']=='passed' for c in checks.values()) else 'failed','checks':checks}


def _conditional_diagnostics(records,grid,archive):
    summaries={}
    for area in (1200,14000):
        for s in TREATMENTS:
            for a,b in (('A','B'),('C','D')):
                before=records[f'{a}-area{area:05d}-{s}']; after=records[f'{b}-area{area:05d}-{s}']
                old,new=before['like'],after['like']
                # A/B aliases are saved at 1200 deg². Recover their matching angular
                # partition for a cellwise conditional comparison at full area.
                if old.validation['area_scaling']['area_deg2']!=area: old=scale_area(old,area)
                if new.validation['area_scaling']['area_deg2']!=area: new=scale_area(new,area)
                if [c.metadata['cell_id'] for c in old.cells]!=[c.metadata['cell_id'] for c in new.cells]:
                    raise ValueError('conditional cell inventory differs')
                arrays=after['arrays']; prefix=f'{grid}/conditional/{area}/{s}/{a}-{b}'
                piv=np.flatnonzero(np.isclose(arrays['result/growth_z'],1.1,rtol=0,atol=1e-12))[0]
                gradients={'fsigma8_z1.1':arrays['result/growth_jacobian'][piv],
                           'Mnu':np.eye(3)[1],'Omegam':np.eye(3)[2]}
                directions={}; norms={q:{'cosmology_only':0.,'nuisance_compensated':0.} for q in gradients}
                for q,g in gradients.items():
                    v,w=compensated_direction(arrays['result/cov_cosmo_marg'],
                        arrays['diagnostics/K'],arrays['diagnostics/N'],g)
                    directions[q]=(np.r_[v,np.zeros(len(w))],np.r_[v,w])
                    _save_array(archive,f'{prefix}/{q}/gradient',g)
                    _save_array(archive,f'{prefix}/{q}/v',v); _save_array(archive,f'{prefix}/{q}/w',w)
                cells=[]
                for old_cell,new_cell in zip(old.cells,new.cells):
                    conditional=conditional_information(old_cell,new_cell)
                    cell_id=new_cell.metadata['cell_id']; cell_prefix=f'{prefix}/{cell_id}'
                    for field in ('D','C','F'): _save_array(archive,cell_prefix+'/'+field,conditional[field])
                    local_norms={}
                    for q,(bare,compensated) in directions.items():
                        local_norms[q]={'cosmology_only':response_norm(conditional,new.names,bare),
                                       'nuisance_compensated':response_norm(conditional,new.names,compensated)}
                        for key,value in local_norms[q].items(): norms[q][key]+=value
                    cells.append({'cell_id':cell_id,'array_prefix':cell_prefix,'names':conditional['names'],
                        'added_observations':conditional['observations'],'norms':local_norms,
                        'reconstruction_scaled_error':conditional['reconstruction_scaled_error'],
                        'minimum_normalized_eigenvalue':conditional['minimum_normalized_eigenvalue']})
                summaries[prefix]={'after_case':b,'global_names':new.names,
                    'parameter_definitions_and_priors':after['meta']['parameters'],
                    'norms':norms,'cells':cells,'direction_definition':'v=C_cos g/sqrt(g^T C_cos g); w=-N^-1 K^T v',
                    'norm_definition':'(D_cond u)^T C_cond^-1 (D_cond u)',
                    'norm_unit':'dimensionless',
                    'interpretation':'Local diagnostics along the after-case marginalized direction; neither an additive gain attribution nor a guaranteed suppression by nuisance compensation.'}
    return summaries


def _grid_summary(records,grid,archive):
    metrics={}; constraints=[]; gains=[]; area_gains=[]; errors={}; units=None
    diagnostic_records={}
    for name,r in records.items():
        errors[name],units,headlines=_errors(r)
        for key,value in r['arrays'].items():
            if key.startswith('diagnostics/') or key in ('result/growth_jacobian','result/growth_z','result/growth_fiducial'):
                _save_array(archive,f'{grid}/{name}/{key}',value)
        diagnostic_records[name]={'array_prefix':f'{grid}/{name}/diagnostics',
            'names':r['like'].names,'parameter_definitions_and_priors':r['meta']['parameters'],
            'independent_covariance_agreements':r['checks']}
        c=next(c for c in conditions() if campaign.condition_id(c)==name)
        for q,unit in units.items():
            marg=errors[name]['marginalized'][q]; fixed=errors[name]['nuisance_fixed'][q]
            penalty=marg/fixed
            constraints.append({'grid':grid,'condition':name,'case':c.case,'area_deg2':c.area_deg2,
                'stochastic':c.stochastic,'quantity':q,'unit':unit,'sigma_marginalized':marg,
                'sigma_nuisance_fixed':fixed,'nuisance_penalty':penalty,'headline':q in headlines,
                'growth_pivot_z':float(q.split('_z')[1]) if q.startswith('fsigma8_z') else ''})
            metrics[f'penalty/{name}/{q}']={'kind':'ratio','unit':'dimensionless','value':penalty}
            for mode in MODES:
                metrics[f'sigma/{name}/{mode}/{q}']={'kind':'uncertainty','unit':unit,'value':errors[name][mode][q]}
    def save_comparisons(comparisons,mode,collection,base):
        for pair,qs in comparisons.items():
            for q,values in qs.items():
                row={**base,'comparison':pair,'mode':mode,'quantity':q,'unit':units[q],**values}
                if 'before_area_deg2' in base:
                    row.update(case=pair.split('/')[0],stochastic=pair.split('/')[1])
                collection.append(row)
                key='/'.join(map(str,[base['grid'],base.get('area_deg2','area'),base.get('stochastic','matched'),mode,pair,q]))
                for field,kind,unit in (('ratio','ratio','dimensionless'),('gain_percent','gain','percentage points')):
                    metrics[f'{field}/{key}']={'kind':kind,'unit':unit,'value':values[field]}
    for area in (1200,14000):
        for s in TREATMENTS:
            for mode in MODES:
                matched={c:errors[f'{c}-area{area:05d}-{s}'][mode] for c in 'ABCD'}
                save_comparisons(case_comparisons(matched),mode,gains,
                                 {'grid':grid,'area_deg2':area,'stochastic':s})
    for mode in MODES:
        matched={name:value[mode] for name,value in errors.items()}
        save_comparisons(area_comparisons(matched),mode,area_gains,
                         {'grid':grid,'before_area_deg2':1200,'after_area_deg2':14000})
    # Grid labels are bookkeeping, not part of a matched metric's identity.
    metrics={k.replace('/'+grid+'/', '/'):v for k,v in metrics.items()}
    return {'identity':_physical_identity(records),'metrics':metrics,'constraints':constraints,
            'gains':gains,'area_gains':area_gains,'diagnostic_records':diagnostic_records,
            'ordering':_ordering(records),'conditional':_conditional_diagnostics(records,grid,archive)}


def _csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _physical_report(summary):
    selected=summary['grids']['shell304']
    lines=['# DESI and PFS overlap information', '',
        f"Numerical qualification: **{summary['status']}**. Presentation uses shell304; all five grids and four refinements are retained.", '',
        'The forecast varies ln(10¹⁰ A_s), Mν and Ω_m, fixing the remaining cosmological coordinates. PFS covers 0.6 < z < 2.1, while DESI retains 0.4 < z < 2.1. The adopted PFS density, clustering model and priors are unchanged between 1,200 and 14,000 deg². The headline growth quantity is fσ₈(z=1.1).', '',
        'A measures correlated DESI autos; B adds internal-DESI crosses; C adds PFS autos; D adds DESI × PFS crosses. Each gain is 100(1−σ_after/σ_before), with the preceding error as its denominator. Area comparisons hold case and residual cross-stochastic treatment fixed. Successive gains close through multiplication of error ratios.', '',
        'The four stochastic treatments fix only the specified residual cross-stochastic coefficients. Other nuisance parameters remain marginalized. The separate all-nuisance-fixed errors still marginalize the other two cosmological coordinates; they use H⁻¹ rather than the reciprocal diagonal of H.', '',
        '| PFS area [deg²] | Cross-stochastic treatment | Case | σ[fσ₈(z=1.1)] | σ(Mν) [eV] | σ(Ω_m) |',
        '| ---: | --- | --- | ---: | ---: | ---: |']
    for area in (1200,14000):
        for s in TREATMENTS:
            for case in 'ABCD':
                rows={r['quantity']:r for r in selected['constraints'] if r['case']==case and r['area_deg2']==area and r['stochastic']==s}
                values=[rows[q]['sigma_marginalized'] for q in ('fsigma8_z1.1','Mnu','Omegam')]
                lines.append(f'| {area} | {s} | {case} | '+ ' | '.join(f'{v:.4g}' for v in values)+' |')
    lines+=['', 'The following reductions use the marginalized errors above. For A→B, B→C, C→D and B→D, the denominators are respectively σ_A, σ_B, σ_C and σ_B; the overall A-reference reductions use σ_A.', '',
        '| PFS area [deg²] | Cross-stochastic treatment | Comparison | fσ₈(z=1.1) gain [%] | Mν gain [%] | Ω_m gain [%] |',
        '| ---: | --- | --- | ---: | ---: | ---: |']
    for area in (1200,14000):
        for s in TREATMENTS:
            for a,b in PAIRS:
                rows={r['quantity']:r for r in selected['gains']
                    if r['area_deg2']==area and r['stochastic']==s
                    and r['comparison']==f'{a}-{b}' and r['mode']=='marginalized'}
                values=[rows[q]['gain_percent'] for q in ('fsigma8_z1.1','Mnu','Omegam')]
                lines.append(f'| {area} | {s} | {a}→{b} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['', 'The footprint reductions compare 14,000 deg² with 1,200 deg² at the same case and cross-stochastic treatment, using the 1,200 deg² error as denominator.', '',
        '| Case | Cross-stochastic treatment | fσ₈(z=1.1) area gain [%] | Mν area gain [%] | Ω_m area gain [%] |',
        '| --- | --- | ---: | ---: | ---: |']
    for case in 'ABCD':
        for s in TREATMENTS:
            rows={r['quantity']:r for r in selected['area_gains'] if r['case']==case
                and r['stochastic']==s and r['mode']=='marginalized'}
            values=[rows[q]['gain_percent'] for q in ('fsigma8_z1.1','Mnu','Omegam')]
            lines.append(f'| {case} | {s} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','| Refinement | Maximum error change | Maximum ratio change | Maximum gain change [pp] | Status |',
            '| --- | ---: | ---: | ---: | --- |']
    for pair,r in summary['refinements'].items():
        m=r['maxima']; lines.append(f"| {pair} | {100*m['uncertainty']['change']:.6g}% | {100*m['ratio']['change']:.6g}% | {m['gain']['change']:.6g} | {r['status']} |")
    lines+=['', 'The relative changes in every error and positive ratio must remain strictly below 1%, and every reported gain must change by strictly less than 0.05 percentage points. Both tests include their floating-point boundary guard. No tolerance is relaxed when a comparison fails.', '',
        'H, K and N are the cosmological, mixed and nuisance precision blocks. The marginalized cosmological precision is S=H−KN⁻¹Kᵀ. The saved uncertainty penalties compare the same quantity using S⁻¹ and H⁻¹. Independent full, Schur and QR covariance agreements are retained for every result.', '',
        'For A→B and C→D, the conditional response subtracts the cross-spectrum response predicted by the existing observations. Its squared norm uses the inverse conditional covariance and the after-case direction v=C_cos g/√(gᵀC_cos g), together with either zero nuisance displacement or w=−N⁻¹Kᵀv. Named cell coordinates are matched to the global parameters before projection. These local diagnostics do not give an additive allocation of a gain to nuisance families or redshift bins. Nuisance compensation need not lower the norm of an individual conditional response.', '',
        'The source derivative qualification is inherited only for the authenticated unchanged physical inputs. This study separately checks the selected observations, area weighting, fixed coefficients, independent marginalization, common-parameter variance ordering and refinement convergence. The saved forecast is conditional on the stated cosmology and Gaussian covariance model; numerical qualification is not a new test of that nonlinear model.', '',
        'Machine-readable absolute errors, gains with both denominators, area comparisons, named conditional responses and all refinement values accompany this report. The identities file authenticates each output and its numerical inputs.']
    failed=[(pair,key,value) for pair,r in summary['refinements'].items() for key,value in r['checks'].items() if value['status']=='failed']
    if failed:
        lines+=['','Failed refinement comparisons:']
        lines += [f"- {p}: {k}, {v['coarse']:.12g} → {v['fine']:.12g}; change {v['change']:.12g} {v['change_unit']}, strict limit {v['limit']}." for p,k,v in failed]
    return '\n'.join(lines)+'\n'


def report(output: Path, destination: Path) -> dict:
    output=Path(output).absolute(); destination=Path(destination).absolute()
    if destination.exists(): raise FileExistsError('refusing to overwrite an existing report')
    status=campaign.status(output)
    if status['status']!='passed': raise ValueError('complete authenticated campaign required')
    # status authenticates all physical-source bytes, every named alias and every saved array.
    destination.mkdir(parents=True)
    campaign.write(destination/'attempt.json',{'status':'running','started_utc':campaign.utc(),
                    'input':str(output),'request_sha256':campaign.digest(output/'request.json')})
    try:
        grids={}
        with zipfile.ZipFile(destination/'diagnostics.npz','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for grid in GRID_PROFILES:
                print(f'{campaign.utc()} conditional responses, absolute errors and ordering: {grid}',flush=True)
                records=_load_results(output,grid)
                grids[grid]=_grid_summary(records,grid,archive)
                del records
        refinements={f'{a}-to-{b}':compare_grids(grids[a],grids[b]) for a,b in REFINEMENTS}
        passed=all(r['status']=='passed' for r in refinements.values()) and all(g['ordering']['status']=='passed' for g in grids.values())
        summary={'schema':'overlap-report-v1','status':'passed' if passed else 'failed',
            'presentation_grid':'shell304','source_qualification':'authenticated existing physical derivatives; unchanged source only',
            'study_qualification':'selected observations, area weights, conditioning, marginalization, ordering and resolution',
            'headlines':{grid:[r for r in g['constraints'] if r['headline']] for grid,g in grids.items()},
            'grids':grids,'refinements':refinements}
        for field in ('constraints','gains','area_gains'):
            _csv(destination/(field+'.csv'),[r for g in grids.values() for r in g[field]])
        campaign.write(destination/'refinements.json',refinements)
        campaign.write(destination/'summary.json',summary)
        (destination/'REPORT.md').write_text(_physical_report(summary))
        campaign.write(destination/'attempt.json',{'status':summary['status'],'finished_utc':campaign.utc(),'input':str(output)})
        files={str(p.relative_to(destination)):campaign.digest(p) for p in sorted(destination.iterdir()) if p.is_file()}
        request=campaign.read(output/'request.json')
        inputs={str(output/'request.json'):campaign.digest(output/'request.json'),
                str(output/'state.json'):campaign.digest(output/'state.json')}
        for p in sorted((output/'inference').rglob('*')):
            if p.is_file() and p.name in ('manifest.json','checks.json','arrays.npz'): inputs[str(p)]=campaign.digest(p)
        campaign.write(destination/'identities.json',{'schema':'overlap-report-identities-v1',
            'report_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=campaign.ROOT,text=True).strip(),
            'report_code':{str(Path(__file__).resolve()):campaign.digest(__file__)},
            'request_sha256':campaign.digest(output/'request.json'),'numerical_code':request['code_hashes'],
            'physical_sources':request['source_hashes'],'result_files':inputs,'exports':files})
        return {'status':summary['status'],'output':str(destination),
                'refinements':{k:{'status':v['status'],'maxima':v['maxima']} for k,v in refinements.items()}}
    except BaseException as exc:
        campaign.write(destination/'failure.json',{'status':'failed','type':type(exc).__name__,
                                                  'error':str(exc),'utc':campaign.utc()})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=report(args.input,args.output)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
