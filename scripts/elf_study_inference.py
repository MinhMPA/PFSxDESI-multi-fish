"""Three-coordinate cosmology with shared population priors and derived growth."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.eft_params import broad_priors,CROSS_STOCH_PRIOR_SIGMA
from pfsfog.extended_artifacts import canonical,require_close,validate_likelihood
from pfsfog.fisher_core import fisher_from_shells
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from pfsfog.elf_covariance_audit import audit_spd_batch
from pfsfog.revision_forecast import RevisionLikelihood,make_shell_grid,apply_revision_priors
from pfsfog.revision_theory import NUISANCE_NAMES
from scripts.elf_study_profiles import study_config,study_cells,study_shell_grid


def validate_study_likelihood(like):
    expected=study_shell_grid(like.config)
    for field in ('edges','centers','widths'):
        if not np.array_equal(getattr(like.grid,field),getattr(expected,field)):
            raise ValueError('study grid is not the exact observed parent restriction')
    # Validate arithmetic against the actual stored upper edge. The nominal cut
    # differs by at most one representation step and remains in the manifest.
    validate_likelihood(replace(like,config=replace(like.config,kmax=float(expected.edges[-1]))))


def assemble_study(profile,settings,parts,reference,model):
    config=study_config(profile,settings)
    space=make_cosmology_spec(reference,extended=False)
    expected=[c['expected'] for c in study_cells(profile,SimpleNamespace(params=reference,sigma8=lambda z:1.),settings=settings)]
    cells=[p['cell'] for p in parts]
    if (len(cells)!=len(expected) or [c.metadata['cell_id'] for c in cells]!=[r['cell_id'] for r in expected]):
        raise ValueError('missing, duplicate or reordered study cell')
    grid=study_shell_grid(config)
    values={};theta={n:float(reference[{'Mnu':'mnu','Omegam':'Omega_m'}.get(n,n)]) for n in space.names}
    for part,row in zip(parts,expected):
        cell=part['cell'];point=part['fiducials']
        if cell.names!=row['local_names'] or set(point)!=set(cell.names):
            raise ValueError('local names or fiducials differ from study support')
        for key,value in row.items():
            if canonical(cell.metadata.get(key))!=canonical(value): raise ValueError('study support differs: '+key)
        for n,v in point.items():
            if (not np.isfinite(v) or (n in values and v!=values[n]) or (n in theta and v!=theta[n])):
                raise ValueError('inconsistent shared or cosmological fiducial: '+n)
            values[n]=float(v)
        shape=(len(grid.centers),3*len(row['retained_pairs']),len(cell.names))
        if (cell.derivatives.shape!=shape or cell.fiducial_mean.shape!=shape[:2]
                or cell.covariance.shape!=(*shape[:2],shape[1]) or not np.isfinite(cell.derivatives).all()
                or not np.isfinite(cell.fiducial_mean).all()): raise ValueError('cell array layout differs')
        expected_shells=len(grid.centers) if cell.names else 0
        if any(cell.validation.get(k)!=expected_shells for k in ('expected_shells','processed_shells')):
            raise ValueError('physical shell coverage differs from study support')
        if cell.names:
            audit_spd_batch(cell.covariance,'study multipole covariance')
            require_close(cell.F_like,fisher_from_shells(cell.derivatives,cell.covariance,grid.centers),'cell Fisher')
        elif np.size(cell.F_like): raise ValueError('empty cell has information')
    names=list(space.names)
    parameters=[{'name':n,'role':'cosmo','nominal_bin':None,'tracers':[],
        'coordinate':n,'unit':'eV' if n=='Mnu' else '','fiducial':theta[n],
        'prior_sigma':{'Mnu':5.,'Omegam':1.}.get(n)} for n in names]
    widths=broad_priors().sigma_dict()
    for ibin in range(len(config.nominal_bins)):
        own=[c for c in cells if c.metadata['nominal_bin_index']==ibin]
        populations=sorted({n for c in own for n in c.metadata['active_populations']})
        crosses=sorted({tuple(p) for c in own for p in c.metadata['retained_pairs'] if p[0]!=p[1]})
        for population,coords,role,tracers in (
                [(n,NUISANCE_NAMES,'tracer',[n]) for n in populations]
                +[('|'.join(p),tuple(CROSS_STOCH_PRIOR_SIGMA),'cross',list(p)) for p in crosses]):
            for q in coords:
                n=f'bin{ibin:02d}:{population}:{q}';names.append(n)
                parameters.append({'name':n,'role':role,'nominal_bin':ibin,'tracers':tracers,
                    'coordinate':q,'fiducial':values[n],
                    'prior_sigma':(widths if role=='tracer' else CROSS_STOCH_PRIOR_SIGMA)[q],
                    'unit':'(Mpc/h)^2' if q in ('c0','c2','c4') else '(Mpc/h)^4' if q=='c_tilde' else ''})
    if set(names)!=set(values) or len(names)!=len(set(names)): raise ValueError('global parameter inventory differs')
    index={n:i for i,n in enumerate(names)};bins=[np.zeros((len(names),len(names))) for _ in config.nominal_bins]
    for cell in cells:
        local=[index[n] for n in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(local,local)]+=cell.F_like
    fisher=np.sum(bins,axis=0)
    for i,p in enumerate(parameters): p['prior_only']=bool(fisher[i,i]==0 and p['prior_sigma'] is not None)
    nonempty=[c for c in cells if c.names]
    validation={'cell_count':len(cells),'empty_cells':[c.metadata['cell_id'] for c in cells if not c.names],
        'expected_shells':sum(c.validation['expected_shells'] for c in cells),
        'processed_shells':sum(c.validation['processed_shells'] for c in cells),
        'min_field_scaled_eigenvalue':min(c.validation['field_min_scaled_eigenvalue'] for c in nonempty),
        'min_covariance_scaled_eigenvalue':min(c.validation['covariance_min_scaled_eigenvalue'] for c in nonempty)}
    like=RevisionLikelihood(config,fisher,names,np.array([values[n] for n in names]),parameters,cells,
        bins,grid,dict(reference),dict(model),validation)
    validate_study_likelihood(like)
    return like


def validate_growth(result):
    z=np.asarray(result.growth_z);j=np.asarray(result.growth_jacobian)
    if (j.shape!=(len(z),3) or np.shape(result.growth_fiducial)!=(len(z),)
            or not np.isfinite(j).all() or not np.isfinite(result.growth_fiducial).all()):
        raise ValueError('invalid derived growth labels or Jacobian')
    for suffix,cosmo in (('marg',result.cov_cosmo_marg),('nuisance_fixed',result.cov_cosmo_nuisance_fixed)):
        cov=np.asarray(getattr(result,'growth_cov_'+suffix))
        require_close(cov,j@cosmo@j.T,'growth covariance '+suffix)
        if np.any(np.diag(cov)<=0): raise ValueError('nonpositive derived growth variance')
    penalty=np.sqrt(np.diag(result.growth_cov_marg)/np.diag(result.growth_cov_nuisance_fixed))
    if np.min(penalty)<1-1e-10: raise ValueError('growth nuisance penalty below unity')
    return {'rank_upper_bound':3,'nuisance_penalty':penalty.tolist()}


def infer_study(likelihood):
    result=apply_revision_priors(likelihood,likelihood.config.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
    result.validation['growth_check']=validate_growth(result)
    return result
