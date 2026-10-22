"""Global information from physically verified ELF survey regions."""
from pathlib import Path

import numpy as np

from .cosmology_space import make_cosmology_spec
from .eft_params import broad_priors, CROSS_STOCH_PRIOR_SIGMA
from .elf_survey_evidence import canonical, config_from_request, verify_run
from .extended_artifacts import read_json, require_close, validate_likelihood
from .fisher_core import fisher_from_shells
from .revision_forecast import (CellLikelihood, RevisionLikelihood, NOMINAL_BINS,
    _scaled_spd, make_shell_grid)
from .revision_theory import NUISANCE_NAMES
from .revision_validation_physical import expected_support_schedule
from .surveys import default_survey_group


def assemble_likelihood(config, reference, model, cells, fiducials):
    """Embed each regional D^T C^-1 D with one shared nominal-bin nuisance basis."""
    if (config.case not in ('B','C') or config.cosmology_basis != 'extended6'
            or tuple(map(tuple,config.nominal_bins)) != NOMINAL_BINS):
        raise ValueError('global ELF inference requires baseline B6/C6 support')
    space = make_cosmology_spec(reference, extended=True)
    expected = expected_support_schedule(config, default_survey_group(), space=space)
    if len(cells) != len(expected) or len(fiducials) != len(cells):
        raise ValueError('missing physical cell or fiducial map')
    ids = [c.metadata['cell_id'] for c in cells]
    if ids != [e['cell_id'] for e in expected] or len(set(ids)) != len(ids):
        raise ValueError('duplicate, missing or reordered physical cell')
    grid = make_shell_grid(config.kmin,config.kmax,config.dk)
    values = {}
    theta = {n:float(reference[{'Mnu':'mnu','Omegam':'Omega_m'}.get(n,n)]) for n in space.names}
    for cell, point, row in zip(cells,fiducials,expected):
        if (len(set(cell.names)) != len(cell.names) or set(cell.names) != set(row['local_names'])
                or cell.metadata['local_names'] != cell.names or set(point) != set(cell.names)):
            raise ValueError('local parameter names or fiducials differ from physical support')
        for key,value in row.items():
            if key not in ('local_names','empty_observation','absent_observations'):
                if canonical(cell.metadata.get(key)) != canonical(value):
                    raise ValueError('physical support mismatch: '+key)
        for name,value in point.items():
            if not np.isfinite(value) or (name in values and value != values[name]):
                raise ValueError('inconsistent shared fiducial: '+name)
            if name in theta and value != theta[name]:
                raise ValueError('cosmological fiducial differs from reference: '+name)
            values[name] = float(value)
        shape = (len(grid.centers),3*len(row['retained_pairs']),len(cell.names))
        if (cell.derivatives.shape != shape or cell.fiducial_mean.shape != shape[:2]
                or cell.covariance.shape != (*shape[:2],shape[1])
                or not np.isfinite(cell.derivatives).all() or not np.isfinite(cell.fiducial_mean).all()):
            raise ValueError('physical derivative, mean or covariance shell shape differs')
        for covariance in cell.covariance:
            _scaled_spd(covariance,'saved ELF multipole covariance')
        require_close(cell.F_like,fisher_from_shells(cell.derivatives,cell.covariance,grid.centers),
                      'saved cell Fisher')
    names = list(space.names)
    parameters = [{'name':n,'role':'cosmo','nominal_bin':None,'tracers':[],
        'coordinate':n,'unit':'eV' if n=='Mnu' else '', 'fiducial':theta[n],
        'prior_sigma':{'Mnu':5.,'Omegam':1.}.get(n)} for n in names]
    widths = broad_priors().sigma_dict()
    for ibin in range(len(config.nominal_bins)):
        own = [c for c in cells if c.metadata['nominal_bin_index']==ibin]
        populations = sorted({n for c in own for n in c.metadata['active_populations']})
        crosses = sorted({tuple(p) for c in own for p in c.metadata['retained_pairs'] if p[0]!=p[1]})
        for population,coordinates,role,tracers in (
            [(n,NUISANCE_NAMES,'tracer',[n]) for n in populations]
            +[('|'.join(p),tuple(CROSS_STOCH_PRIOR_SIGMA),'cross',list(p)) for p in crosses]):
            for coordinate in coordinates:
                name = f'bin{ibin:02d}:{population}:{coordinate}'
                names.append(name)
                parameters.append({'name':name,'role':role,'nominal_bin':ibin,
                    'tracers':tracers,'coordinate':coordinate,'fiducial':values[name],
                    'prior_sigma':(widths if role=='tracer' else CROSS_STOCH_PRIOR_SIGMA)[coordinate],
                    'unit':('(Mpc/h)^2' if coordinate in ('c0','c2','c4') else
                            '(Mpc/h)^4' if coordinate=='c_tilde' else '')})
    if set(names) != set(values) or len(names) != len(set(names)):
        raise ValueError('global physical parameter inventory differs')
    index = {n:i for i,n in enumerate(names)}
    bins = [np.zeros((len(names),len(names))) for _ in config.nominal_bins]
    for cell in cells:
        local = [index[n] for n in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(local,local)] += cell.F_like
    fisher = np.sum(bins,axis=0)
    for i,record in enumerate(parameters):
        record['prior_only'] = bool(fisher[i,i]==0 and record['prior_sigma'] is not None)
    validation = {'cell_count':len(cells),
        'expected_shells':sum(c.validation['expected_shells'] for c in cells),
        'processed_shells':sum(c.validation['processed_shells'] for c in cells),
        'min_field_scaled_eigenvalue':min(c.validation['field_min_scaled_eigenvalue'] for c in cells),
        'min_covariance_scaled_eigenvalue':min(c.validation['covariance_min_scaled_eigenvalue'] for c in cells)}
    result = RevisionLikelihood(config,fisher,names,np.array([values[n] for n in names]),
        parameters,list(cells),bins,grid,dict(reference),dict(model),validation)
    validate_likelihood(result)
    return result


def load_physical_likelihood(root, case):
    """Accept only a complete, current-source physical cohort, never a selected probe."""
    root = Path(root)
    verified = verify_run(root)
    if verified['status'] != 'passed' or not verified['complete_survey']:
        raise ValueError('physical cohort is failed, incomplete or foreign-source: '+str(verified['problems']))
    if case not in ('B','C'):
        raise ValueError('physical case must be B or C')
    request = read_json(root/'request.json')
    config = config_from_request(request,case)
    cells,fiducials = [],[]
    grid = make_shell_grid(config.kmin,config.kmax,config.dk)
    for row in request['expected']:
        if row['case'] != case:
            continue
        path = root/'cells'/row['key'].replace(':','__')
        inputs = read_json(path/'inputs.json'); measurement = read_json(path/'measurement.json')
        with np.load(path/'arrays.npz',allow_pickle=False) as a:
            if not np.array_equal(a['k'],grid.centers) or not np.array_equal(a['dk'],grid.widths):
                raise ValueError('physical shell arrays differ')
            metadata = inputs['metadata']
            cells.append(CellLikelihood(a['F_like'].copy(),list(metadata['local_names']),metadata,
                measurement['covariance'],a['D'].copy(),a['C'].copy(),a['mean'].copy()))
        fiducials.append(inputs['fiducials'])
    flags = request['source']['details']['model_flags']
    model = {'backend':flags['backend_class'],'subtract_k0_limit':flags['subtract_k0_limit'],
        'mean_n_mu':config.n_mu,'covariance_n_mu':config.n_mu,'fixed_observed_geometry':True,
        'fixed_covariance':True,'covariance_derivative_trace':False,'serial':True,
        **request['source']['details']['engine_settings']}
    return assemble_likelihood(config,request['source']['details']['reference_cosmology'],
                               model,cells,fiducials)


def infer_treatments(likelihood, *, on_outcome=None):
    """Apply all accepted priors, retaining exact precision inputs on failure."""
    from .cosmology_projection import (COSMOLOGY_TREATMENTS, apply_cosmology_treatment,
                                       prepare_cosmology_precision)
    from .fisher_precision import DECIMAL_FISHER_SOLVER
    results, failures = {}, {}
    for name in COSMOLOGY_TREATMENTS:
        inputs = prepare_cosmology_precision(likelihood, name)
        try:
            results[name] = apply_cosmology_treatment(likelihood, name,
                                                     fisher_solver=DECIMAL_FISHER_SOLVER)
        except ValueError as error:
            failures[name] = {'status':'failed', 'error':str(error),
                              'precision_inputs':inputs}
        if on_outcome is not None:
            on_outcome(name, results.get(name), failures.get(name))
    return results, failures
