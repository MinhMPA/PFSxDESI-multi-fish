"""Cosmology/linear-bias correlations with explicit bin-only versus global data."""
from types import SimpleNamespace
import numpy as np
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from pfsfog.revision_forecast import project_baseline_precision


def correlation(cov):
    std=np.sqrt(np.diag(cov))
    if np.any(std<=0) or not np.isfinite(cov).all():raise ValueError('invalid diagnostic covariance')
    return cov/std[:,None]/std[None,:]


def bin_diagnostics(result):
    like=result.likelihood;rows=[]
    for ibin,bounds in enumerate(like.config.nominal_bins):
        retained=[i for i,p in enumerate(like.parameters) if p['role']=='cosmo' or p['nominal_bin']==ibin]
        names=[like.names[i] for i in retained]
        labels=[n for n in names if ':' not in n or n.endswith(':b1_sigma8')]
        global_ix=[like.names.index(n) for n in labels];local_ix=[names.index(n) for n in labels]
        global_cov=result.covariance[np.ix_(global_ix,global_ix)]
        local_f=like.bin_F_like[ibin][np.ix_(retained,retained)]
        row={'nominal_bin_index':ibin,'nominal_bin':list(bounds),'model_redshift':float(result.growth_z[ibin]),
            'labels':labels,'global_marginal_covariance':global_cov.tolist(),
            'global_marginal_correlation':correlation(global_cov).tolist(),
            'bin_only':{'status':'pending','names':names,'F_like':local_f.tolist(),
                        'conditioning':'only this nominal bin; priors inserted once; other cosmology marginalized'}}
        local=SimpleNamespace(config=like.config,names=names,F_like=local_f,
            parameters=[like.parameters[i] for i in retained],fiducial=like.fiducial[retained])
        try:
            q=project_baseline_precision(local,result.nuisance_factor,fisher_solver=DECIMAL_FISHER_SOLVER)
            selected=q['covariance'][np.ix_(local_ix,local_ix)];j=result.growth_jacobian[ibin]
            gm=float(j@q['cov_cosmo_marg']@j);gf=float(j@q['cov_cosmo_nuisance_fixed']@j)
            if not (gm>0 and gf>0 and gm>=gf*(1-1e-10)):raise ValueError('invalid bin-only growth uncertainty')
            row['bin_only'].update(status='passed',marginal_covariance=selected.tolist(),
                marginal_correlation=correlation(selected).tolist(),
                nuisance_penalty_cosmology=np.sqrt(np.diag(q['cov_cosmo_marg'])/np.diag(q['cov_cosmo_nuisance_fixed'])).tolist(),
                nuisance_penalty_growth=float(np.sqrt(gm/gf)),sigma_growth=float(np.sqrt(gm)),
                sigma_growth_nuisance_fixed=float(np.sqrt(gf)),validation=q['validation'])
        except ValueError as error:row['bin_only'].update(status='failed',error=str(error))
        rows.append(row)
    return rows
