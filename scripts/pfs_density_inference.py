"""One-bin density changes in a common cosmology with independent bin nuisances."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from pfsfog.extended_artifacts import canonical, require_close
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER, decimal_inverse
from pfsfog.revision_forecast import project_baseline_precision
from scripts.pfs_density_response import PFS, _factor, density_column_scales

WINDOWS = ((.7,.8),(.8,1.),(1.,1.2),(1.2,1.4),(1.4,1.6),
           (1.6,2.),(2.,2.1),(2.1,2.2))
PRIOR_MODES = ('dimensionless', 'fixed_physical')


def window_cells(result, window):
    lo, hi = window
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError('invalid density window')
    cells=[]
    for c in result.likelihood.cells:
        a,b=c.metadata['radial_bin']
        if PFS not in c.metadata['active_populations'] or b<=lo or a>=hi:
            continue
        if a<lo or b>hi:
            raise ValueError('density window would cut a saved radial cell')
        cells.append(c)
    if not cells:
        raise ValueError('density window contains no PFS observations')
    bounds=sorted(tuple(c.metadata['radial_bin']) for c in cells)
    if bounds[0][0]!=lo or bounds[-1][1]!=hi or any(a[1]!=b[0] for a,b in zip(bounds,bounds[1:])):
        raise ValueError('density window has incomplete or duplicate radial coverage')
    if len({c.metadata['nominal_bin_index'] for c in cells})!=1:
        raise ValueError('density window must belong to one nominal bin')
    return cells


def window_count(result, window):
    return float(sum(c.metadata['nbar'][PFS]*c.metadata['volume'] for c in window_cells(result,window)))


def _inverse(a, labels, context):
    return decimal_inverse(a, labels, context)[0]


def _no_cross_bin_nuisance(like):
    if like.names[:3]!=['ln10_10_As','Mnu','Omegam']:
        raise ValueError('density recombination requires the three global coordinates')
    for i,p in enumerate(like.parameters):
        if p['role']=='cosmo': continue
        other=[j for j,q in enumerate(like.parameters)
               if q['role']!='cosmo' and q['nominal_bin']!=p['nominal_bin']]
        if np.any(like.F_like[i,other]!=0) or np.any(like.F_like[other,i]!=0):
            raise ValueError('cross-bin nuisance information invalidates recombination')


def _bin_precision(result, ibin, fisher=None, parameters=None):
    like=result.likelihood
    retained=[i for i,p in enumerate(like.parameters) if p['role']=='cosmo' or p['nominal_bin']==ibin]
    f=like.bin_F_like[ibin] if fisher is None else fisher
    records=like.parameters if parameters is None else parameters
    local=SimpleNamespace(config=like.config,names=[like.names[i] for i in retained],
        F_like=f[np.ix_(retained,retained)],parameters=[records[i] for i in retained])
    precision=project_baseline_precision(local,result.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
    information=_inverse(precision['cov_cosmo_marg'],like.names[:3],'marginalized bin information')
    return information-precision['F_prior_cosmo'][:3,:3], precision['validation']


def verify_reference(result):
    like=result.likelihood;_no_cross_bin_nuisance(like)
    require_close(np.sum(like.bin_F_like,axis=0),like.F_like,'sum of reference bins')
    information=result.F_prior_cosmo[:3,:3].copy();checks=[]
    for ibin in range(len(like.bin_F_like)):
        value,check=_bin_precision(result,ibin)
        information+=value;checks.append(check)
    covariance=_inverse(information,like.names[:3],'recombined reference')
    require_close(covariance,result.cov_cosmo_marg,'density reference recombination',rtol=1e-8,atol=1e-12)
    return {'status':'passed','cross_bin_nuisance_blocks_exactly_zero':True,
            'maximum_scaled_covariance_difference':float(np.max(np.abs(covariance-result.cov_cosmo_marg)
                /np.sqrt(np.outer(np.diag(result.cov_cosmo_marg),np.diag(result.cov_cosmo_marg))))),
            'bin_solve_checks':checks}


def _replacement_state(result, replacements, factor, prior_mode):
    factor=_factor(factor)
    if prior_mode not in PRIOR_MODES: raise ValueError('unknown stochastic prior convention')
    like=result.likelihood;_no_cross_bin_nuisance(like)
    changed=[p['cell'] for p in replacements]
    if not changed: raise ValueError('density replacement must contain complete PFS support')
    ids=[c.metadata['cell_id'] for c in changed]
    if len(set(ids))!=len(ids): raise ValueError('duplicate changed density cell')
    bins={c.metadata['nominal_bin_index'] for c in changed}
    if len(bins)!=1: raise ValueError('density replacement spans multiple nominal bins')
    ibin=bins.pop()
    original={c.metadata['cell_id']:c for c in like.cells
              if c.metadata['nominal_bin_index']==ibin and PFS in c.metadata['active_populations']}
    if set(ids)!=set(original): raise ValueError('density replacement requires complete PFS support within its bin')
    index={n:i for i,n in enumerate(like.names)};delta=np.zeros_like(like.F_like)
    for c in changed:
        old=original[c.metadata['cell_id']]
        expected=deepcopy(old.metadata);expected['nbar'][PFS]*=factor
        if canonical(c.metadata)!=canonical(expected) or c.names!=old.names:
            raise ValueError('density replacement changed support or unexpected number densities')
        ix=[index[n] for n in c.names]
        delta[np.ix_(ix,ix)]+=c.F_like-old.F_like
    records=deepcopy(like.parameters)
    for p,scale in zip(records,density_column_scales(like.names,factor)):
        if (prior_mode=='fixed_physical' and p['nominal_bin']==ibin
                and p['prior_sigma'] is not None):
            p['prior_sigma']/=scale
    return ibin,delta,records


def density_bin_covariance(result, replacements, factor, prior_mode):
    ibin,delta,records=_replacement_state(result,replacements,factor,prior_mode)
    like=result.likelihood
    old,_=_bin_precision(result,ibin)
    new,check=_bin_precision(result,ibin,like.bin_F_like[ibin]+delta,records)
    information=_inverse(result.cov_cosmo_marg,like.names[:3],'reference global information')+new-old
    covariance,solve=decimal_inverse(information,like.names[:3],'changed density global information')
    fixed=_inverse(result.F_total[:3,:3]+delta[:3,:3],like.names[:3],'density nuisance-fixed information')
    j=result.growth_jacobian
    return {'cosmological_covariance':covariance,'cosmological_covariance_nuisance_fixed':fixed,
            'old_bin_information':old,'new_bin_information':new,'global_information':information,
            'growth_z':result.growth_z,'growth_covariance':j@covariance@j.T,
            'growth_covariance_nuisance_fixed':j@fixed@j.T,
            'growth_jacobian':j,'growth_fiducial':result.growth_fiducial,
            'nominal_bin_index':ibin,'factor':float(factor),'prior_mode':prior_mode,
            'changed_cell_ids':[p['cell'].metadata['cell_id'] for p in replacements],
            'changed_prior_widths':{p['name']:p['prior_sigma'] for p,oldp in zip(records,like.parameters)
                                    if p['prior_sigma']!=oldp['prior_sigma']},
            'bin_solve':check,'global_solve':solve}


def full_density_covariance(result, replacements, factor, prior_mode):
    """Independent full-matrix control for the one-bin recombination."""
    _,delta,records=_replacement_state(result,replacements,factor,prior_mode)
    like=result.likelihood
    full=SimpleNamespace(config=like.config,names=like.names,F_like=like.F_like+delta,parameters=records)
    return project_baseline_precision(full,result.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
