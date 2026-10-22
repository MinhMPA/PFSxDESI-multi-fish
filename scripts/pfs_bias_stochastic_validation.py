"""Explicit angular-zero validation of immutable broad-bias measurements."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pfsfog.cosmo import FIDUCIAL
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.revision_validation import compare_cell_derivatives
from scripts.validate_elf_engine import _white


def cross_shot_zero_support(names,pairs,*,alphas):
    """A constant cross stochastic term has no quadrupole or hexadecapole.

    This narrow witness is enabled only at identity AP; it does not assert
    anything about cosmological derivatives or other stochastic coefficients.
    """
    alphas=np.asarray(alphas,dtype=float)
    if (alphas.shape!=(2,) or not np.isfinite(alphas).all()
            or np.any(np.abs(alphas-1.)>32*np.finfo(float).eps)):
        return ()
    zeros=[]
    for name in names:
        fields=name.split(':')
        if len(fields)!=3 or fields[2]!='Pshot_cross':continue
        population=tuple(fields[1].split('|'))
        for pair in pairs:
            if len(population)==2 and population==tuple(pair) and pair[0]!=pair[1]:
                zeros.extend((name,tuple(pair),ell) for ell in (2,4))
    return tuple(zeros)


def replay_saved_derivatives(inputs,arrays,*,configuration,add_cross_shot_zeros=True):
    """Re-evaluate the original checks using every saved signed mean, unchanged."""
    meta=inputs['metadata'];names=meta['local_names'];pairs=meta['retained_pairs']
    fiducials=inputs['fiducials'];steps=inputs['steps']
    fid=np.array([fiducials[n] for n in names]);evaluations={}
    if not np.array_equal(arrays['fd_steps'],[steps[n] for n in names]):
        raise ValueError('saved finite-difference schedule differs')
    reconstructed=(arrays['fd_plus']-arrays['fd_minus'])/(2*arrays['fd_steps'][:,:,None,None])
    if not np.array_equal(reconstructed,arrays['fd_derivatives']):
        raise ValueError('saved finite differences differ from signed means')
    for i,name in enumerate(names):
        for j,step in enumerate(steps[name]):
            for sign,field in ((1,'fd_plus'),(-1,'fd_minus')):
                theta=fid.copy();theta[i]+=sign*step;key=tuple(theta)
                if key in evaluations:raise ValueError('duplicate signed coordinate')
                evaluations[key]=arrays[field][i,j]
    means={}
    for index,pair in enumerate(pairs):
        def mean(coordinates,index=index):
            value=evaluations[tuple(coordinates[n] for n in names)]
            return value[:,3*index:3*index+3]
        means[tuple(pair)]=mean
    zeros=[(name,tuple(pair),ell) for name,pair,ell in inputs['analytic_zero_response']['support']]
    if add_cross_shot_zeros:
        zeros+=list(cross_shot_zero_support(names,pairs,alphas=inputs['analytic_zero_response']['alphas']))
    cell=SimpleNamespace(metadata=meta,names=names,derivatives=arrays['D'],covariance=arrays['C'])
    scale=float(np.linalg.norm(_white(arrays['C'],arrays['mean'][...,None])))
    return compare_cell_derivatives(cell,means,fiducials,steps,configuration=configuration,
        physical_scale=scale,space=elf_cosmology_spec(FIDUCIAL),analytic_zero_support=zeros)


def allow_original_failure(worker,measurement,actual_exit,zeros):
    """Admit only a completed measurement with the diagnosed exact-zero failure."""
    if (worker.get('covariance_status')!='passed' or worker.get('reason')
            or worker.get('source_changed') or worker.get('error_type')
            or measurement['chain_tolerance']!=1e-10 or measurement['chain_relative_error']>=1e-10
            or any(v['status']!='passed' for v in measurement['h_split_fd'].values())
            or not all(r['two_successive_pass'] for r in measurement['ad_fd']['column_results'])):
        raise ValueError('original measurement has an unrelated physical failure')
    if worker['status']==measurement['status']==worker['mean_status']=='passed' and actual_exit==0:
        if measurement['ad_fd']['failures']:raise ValueError('passed measurement contains failures')
        return
    if not (worker['status']==measurement['status']==worker['mean_status']=='failed' and actual_exit==1):
        raise ValueError('original measurement did not finish with the diagnosed validation failure')
    failures=measurement['ad_fd']['failures'];support=set(zeros)
    if not failures or any(r.get('reason')!='pair/ell whitened FD' or
        (r.get('column'),tuple(r.get('pair',())),r.get('ell')) not in support for r in failures):
        raise ValueError('original failure is outside constant cross-shot angular zeros')
