"""Exact restrictions of source-bound physical Gaussian data and their checks."""
from copy import deepcopy

import numpy as np

from pfsfog.elf_covariance_audit import audit_spd_batch, covariance_from_spectra
from pfsfog.extended_artifacts import canonical, require_close
from pfsfog.fisher_core import fisher_from_shells
from pfsfog.revision_forecast import CellLikelihood
from pfsfog.revision_validation import _active_for_pair

FIXED_COSMOLOGY = {'h','omega_b','n_s'}


def independent_autos(covariance):
    result=np.array(covariance,copy=True)
    for a in range(result.shape[1]//3):
        for b in range(result.shape[1]//3):
            if a!=b: result[:,3*a:3*a+3,3*b:3*b+3]=0.
    return result


def select_cell(source, profile, expected, grid):
    """Select data before inversion; this alone is not physical qualification."""
    a=source['arrays'];original=source['inputs']['metadata'];names=list(expected['local_names'])
    for key in ('nominal_bin','radial_bin','region','model_redshift','volume','nominal_bin_index'):
        if canonical(original[key])!=canonical(expected[key]):
            raise ValueError('source support differs: '+key)
    for n,value in expected['nbar'].items():
        if value!=original['nbar'].get(n): raise ValueError('source number density differs')
    old_names=list(original['local_names']);old_pairs=list(map(tuple,original['retained_pairs']))
    pairs=list(map(tuple,expected['retained_pairs']))
    if (len(set(old_names))!=len(old_names) or len(set(names))!=len(names)
            or not set(names)<=set(old_names) or not set(pairs)<=set(old_pairs)):
        raise ValueError('source data or parameter inventory differs')
    shells=np.searchsorted(a['k'],grid.centers)
    if (np.any(shells>=len(a['k'])) or not np.array_equal(a['k'][shells],grid.centers)
            or not np.array_equal(a['dk'][shells],grid.widths)):
        raise ValueError('exact shell selection unavailable')
    rows=[3*old_pairs.index(p)+ell for p in pairs for ell in range(3)]
    cols=[old_names.index(n) for n in names]
    selectedD=a['D'][shells][:,rows]
    omitted=[i for i,n in enumerate(old_names) if n not in names and n not in FIXED_COSMOLOGY]
    if omitted and np.any(selectedD[:,:,omitted]!=0):
        raise ValueError('omitted nuisance affects retained means')
    if omitted and np.any((a['fd_plus'][omitted]-a['fd_minus'][omitted])[:,:,shells][:,:,:,rows]!=0):
        raise ValueError('omitted nuisance affects retained signed means')
    selected={key:np.array(a[key][shells][:,rows],copy=True) for key in ('mean',)}
    selected['D']=selectedD[:,:,cols].copy()
    selected['C']=a['C'][shells][:,rows][:,:,rows].copy()
    if profile.case=='I': selected['C']=independent_autos(selected['C'])
    for key in ('fd_plus','fd_minus','fd_derivatives'):
        selected[key]=a[key][cols][:,:,shells][:,:,:,rows].copy()
    selected['fd_steps']=a['fd_steps'][cols].copy()
    for key in ('D_intrinsic','D_geometry'):
        if key in a: selected[key]=a[key][shells][:,rows][:,:,cols].copy()
    selected['k']=grid.centers.copy();selected['dk']=grid.widths.copy()
    values={n:source['inputs']['fiducials'][n] for n in names}
    if not np.array_equal(a['fiducial'][cols],[values[n] for n in names]):
        raise ValueError('source fiducial arrays differ')
    selected['fiducial']=np.array([values[n] for n in names])
    for key,value in selected.items():
        if not np.isfinite(value).all(): raise ValueError('nonfinite selected array: '+key)
    active=expected['active_populations'];meta=deepcopy(expected)
    meta['theoretical_pairs']=[(n,n) for n in active]+[(x,y) for i,x in enumerate(active) for y in active[i+1:]]
    meta['retained_observations']=[(p,ell) for p in pairs for ell in (0,2,4)]
    covcheck=audit_spd_batch(selected['C'],'selected multipole covariance')
    validation={'expected_shells':len(grid.centers),'processed_shells':len(grid.centers),
        'covariance_min_scaled_eigenvalue':covcheck['min_scaled_eigenvalue'],
        'covariance_max_scaled_condition':covcheck['max_scaled_condition'],
        'max_scaled_residual':covcheck['max_scaled_residual']}
    f=fisher_from_shells(selected['D'],selected['C'],grid.centers)
    cell=CellLikelihood(f,names,meta,validation,selected['D'],selected['C'],selected['mean'])
    zeros=[z for z in source['inputs']['analytic_zero_response']['support']
           if z[0] in names and tuple(z[1]) in pairs]
    return {'cell':cell,'fiducials':values,'arrays':selected,'analytic_zero_support':zeros,
        'mapping':{'source_shells':shells.tolist(),'source_rows':rows,'source_columns':cols,
                   'fixed_cosmology':sorted(FIXED_COSMOLOGY),'independent_autos':profile.case=='I'}}


def check_saved_derivatives(projected):
    """Same full-vector and pair/ell FD metrics, evaluated in selected C."""
    cell=projected['cell'];a=projected['arrays'];D=cell.derivatives;C=cell.covariance
    steps=a['fd_steps'];plus=a['fd_plus'];minus=a['fd_minus']
    npar=len(cell.names);nk,nobs=D.shape[:2]
    if (steps.shape!=(npar,3) or np.any(steps<=0) or not np.isfinite(steps).all()
            or plus.shape!=(npar,3,nk,nobs) or minus.shape!=plus.shape):
        raise ValueError('finite difference array layout differs')
    fd=(plus-minus)/(2*steps[:,:,None,None])
    if not np.array_equal(fd,a['fd_derivatives']): raise ValueError('saved finite difference identity differs')
    chol=np.linalg.cholesky(C)
    W=np.linalg.solve(chol,D)
    fd_rhs=fd.transpose(2,3,0,1).reshape(nk,nobs,npar*3)
    Wfd=np.linalg.solve(chol,fd_rhs).reshape(nk,nobs,npar,3)
    scale=float(np.linalg.norm(np.linalg.solve(chol,cell.fiducial_mean[...,None])))
    if not np.isfinite(scale) or scale<=0: raise ValueError('finite difference physical scale is invalid')
    denom=np.linalg.norm(Wfd,axis=(0,1));diff=np.linalg.norm(W[:,:,:,None]-Wfd,axis=(0,1))
    tiny=denom<=1e-12*scale
    error=diff/np.where(tiny,scale,denom);limits=np.where(tiny,1e-6,1e-3)
    good=error<limits;success=(good[:,0]&good[:,1])|(good[:,1]&good[:,2])
    # Norm of a single observation embedded in the full covariance metric.
    invchol=np.linalg.solve(chol,np.broadcast_to(np.eye(nobs),C.shape))
    weight=np.sum(invchol**2,axis=1)
    adrow=np.sqrt(np.einsum('kop,ko->po',D**2,weight))
    fdrow=np.sqrt(np.einsum('psko,ko->pso',fd**2,weight))
    rowdiff=np.sqrt(np.einsum('psko,ko->pso',(D.transpose(2,0,1)[:,None]-fd)**2,weight))
    pairs=list(map(tuple,cell.metadata['retained_pairs']))
    zeros={(n,tuple(p),ell) for n,p,ell in projected['analytic_zero_support']}
    failures=[];rows=[]
    for i,n in enumerate(cell.names):
        for r in range(nobs):
            pair=pairs[r//3];ell=(0,2,4)[r%3]
            declared=(n not in FIXED_COSMOLOGY and not _active_for_pair(n,pair)) or (n,pair,ell) in zeros
            iszero=(fdrow[i,:,r]<=1e-12*scale)|declared
            errs=rowdiff[i,:,r]/np.where(iszero,scale,fdrow[i,:,r])
            lim=np.where(iszero,1e-6,1e-3);passed=errs<lim
            ok=bool(np.all(passed) if declared else ((passed[0] and passed[1]) or (passed[1] and passed[2])))
            if declared: ok=ok and adrow[i,r]<=1e-6*scale and np.all(fdrow[i,:,r]<=1e-6*scale)
            record={'column':n,'pair':list(pair),'ell':ell,'step_errors':errs.tolist(),
                    'step_tolerances':lim.tolist(),'declared_zero':bool(declared),'passed':bool(ok)}
            rows.append(record)
            if not ok: failures.append(record)
    columns=[{'column':n,'steps':steps[i].tolist(),'step_errors':error[i].tolist(),
        'step_tolerances':limits[i].tolist(),'two_successive_pass':bool(success[i])} for i,n in enumerate(cell.names)]
    failures += [r for r in columns if not r['two_successive_pass']]
    result={'status':'failed' if failures else 'passed','physical_scale':scale,
        'column_results':columns,'row_results':rows,'failures':failures,
        'relative_tolerance':1e-3,'zero_tolerance':1e-6,
        'coverage_count':npar*nobs*3,'analytic_zero_support':projected['analytic_zero_support']}
    if failures: raise ValueError('selected finite difference failed: '+str(failures[:2]))
    return result


def check_neighborhood(source, projected, *, n_mu):
    """Rebuild selected field and data covariances at every retained displacement."""
    original=source['inputs']['metadata'];old_names=original['local_names']
    spectra=source['neighborhood_spectra'];coordinates=source['neighborhood_coordinates']
    cell=projected['cell'];a=projected['arrays'];mapping=projected['mapping']
    fid=np.array([source['inputs']['fiducials'][n] for n in old_names])
    old_pairs=list(map(tuple,original['theoretical_pairs']))
    paircols=[old_pairs.index(tuple(p)) for p in cell.metadata['theoretical_pairs']]
    shells=mapping['source_shells'];sourcecols=mapping['source_columns']
    if (spectra.shape[0]!=1+6*len(old_names) or coordinates.shape!=(1+6*len(old_names),len(old_names))
            or not np.array_equal(coordinates[0],fid)):
        raise ValueError('source neighborhood layout or fiducial differs')
    points=[(0,fid)]
    for i,col in enumerate(sourcecols):
        for j,step in enumerate(a['fd_steps'][i]):
            for sign,offset in ((1,0),(-1,1)):
                point=fid.copy();point[col]+=sign*step
                points.append((1+6*col+2*j+offset,point))
    records=[]
    for idx,point in points:
        if not np.array_equal(coordinates[idx],point): raise ValueError('source neighborhood displacement differs')
        signals=np.asarray(spectra[idx])[paircols][:,shells]
        field,cov=covariance_from_spectra(signals,cell.metadata,a['k'],a['dk'],n_mu=n_mu)
        if mapping['independent_autos']: cov=independent_autos(cov)
        if idx==0: require_close(cov,cell.covariance,'selected covariance from saved physical spectra')
        records.append({'source_point':idx,'field':audit_spd_batch(field,'selected field'),
                        'covariance':audit_spd_batch(cov,'selected data covariance')})
    fieldmin=min(r['field']['min_scaled_eigenvalue'] for r in records)
    cell.validation.update(field_min_scaled_eigenvalue=fieldmin,
        field_max_scaled_condition=max(r['field']['max_scaled_condition'] for r in records))
    return {'status':'passed','expected_points':len(points),'processed_points':len(records),
            'points':records,'min_field_scaled_eigenvalue':fieldmin}


def project_cell(source,profile,expected,grid):
    if expected['empty_observation']:
        if expected['local_names'] or expected['retained_pairs'] or expected['active_populations']:
            raise ValueError('empty region contains observations or coordinates')
        meta=deepcopy(expected);meta.update(theoretical_pairs=[],retained_observations=[])
        cell=CellLikelihood(np.zeros((0,0)),[],meta,{'expected_shells':0,'processed_shells':0},
            np.zeros((len(grid.centers),0,0)),np.zeros((len(grid.centers),0,0)),np.zeros((len(grid.centers),0)))
        return {'cell':cell,'fiducials':{},'mapping':{'empty_observation':True}}
    result=select_cell(source,profile,expected,grid)
    result['derivative_check']=check_saved_derivatives(result)
    result['neighborhood_check']=check_neighborhood(source,result,n_mu=source['inputs']['configuration']['n_mu'])
    return result
