"""Six-coordinate form of the frozen saved-response audit.

The algebra and tolerances are identical to elf_study_data.check_saved_derivatives;
only the declared cosmology space is extended. Historical source stays untouched.
"""
import numpy as np
from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.revision_validation import _active_for_pair
from scripts.elf_study_data import select_cell,check_neighborhood,project_cell

def check_extended_derivatives(projected):
    """Same full-vector and pair/ell FD metrics, evaluated in selected C."""
    space=make_cosmology_spec({'h':.6736},extended=True)
    if tuple(projected['cell'].names[:6])!=space.names:
        raise ValueError('six-coordinate cosmology is required')
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
            declared=(not _active_for_pair(n,pair,space=space)) or (n,pair,ell) in zeros
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



def project_extended_cell(source,profile,expected,grid):
    if expected['empty_observation']:
        return project_cell(source,profile,expected,grid)
    result=select_cell(source,profile,expected,grid)
    result['mapping']['fixed_cosmology']=[]
    result['derivative_check']=check_extended_derivatives(result)
    result['neighborhood_check']=check_neighborhood(source,result,n_mu=source['inputs']['configuration']['n_mu'])
    return result
