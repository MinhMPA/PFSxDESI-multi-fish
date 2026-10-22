"""Shell/angular output convergence at fixed physical support and cosmology.

Each grid reconstructs its own full D/C/F and all correlated cosmological and
growth covariances. No spectra at different k points are subtracted. The
qualification caller authenticates a common source/runtime/input identity
before invoking these mathematical comparisons and records that identity.
"""
from dataclasses import asdict
import numpy as np

from .ap_numerics import configuration_profile
from .cosmology_projection import COSMOLOGY_TREATMENTS
from .cosmology_resolution import _cohort, _failure, _coverage, _bind_growth, _FIELDS, _matrix_change
from .extended_artifacts import canonical, jsonable, require_close, validate_likelihood
from .fisher_precision import common_fisher_solver, fisher_solver_record

_TOLERANCE = .01


def _model(like):
    value = dict(like.model)
    for field in ('mean_n_mu', 'covariance_n_mu'):
        if field in value and value.pop(field) != like.config.n_mu:
            raise ValueError('quadrature model declaration differs from configuration')
    return value


def _bind(base, refined, role):
    if role not in ('convergence', 'historical_assessment'):
        raise ValueError('unknown shell comparison role')
    for like in (base, refined):
        configuration_profile(like.config)
        if (like.config.cosmology_basis != 'extended6' or like.config.kmin != .01
                or like.config.kmax != .20 or like.config.ap_interpolation_nk != 16384):
            raise ValueError('shell comparison requires declared extended6 support and fixed AP16384')
        validate_likelihood(like)
    a, b = asdict(base.config), asdict(refined.config)
    changed = {key for key in a if canonical(a[key]) != canonical(b[key])}
    if changed == {'dk'}:
        expected = (.0025, .00125) if role == 'historical_assessment' else (.00125, .000625)
        if (base.config.dk, refined.config.dk) != expected or base.config.n_mu != 32:
            raise ValueError('shell spacing does not match comparison role')
        require_close(base.grid.edges, refined.grid.edges[::2], 'nested shell edges', rtol=0., atol=1e-15)
        comparison = 'shell output resolution'
    elif changed == {'n_mu'} and role == 'convergence':
        if (base.config.n_mu, refined.config.n_mu) != (32, 64) or base.config.dk != .00125:
            raise ValueError('angular comparison grid differs')
        for field in ('edges', 'centers', 'widths'):
            require_close(getattr(base.grid, field), getattr(refined.grid, field), 'angular observed shells', rtol=0., atol=0.)
        comparison = 'angular output resolution'
    else:
        raise ValueError('resolution comparison must change only the declared quadrature')
    if (base.names != refined.names or canonical(base.parameters) != canonical(refined.parameters)
            or canonical(base.reference_cosmology) != canonical(refined.reference_cosmology)
            or canonical(_model(base)) != canonical(_model(refined)) or len(base.cells) != len(refined.cells)):
        raise ValueError('shell physical parameter/prior/reference/model identity differs')
    require_close(base.fiducial, refined.fiducial, 'shell fiducial', rtol=0., atol=0.)
    for first, second in zip(base.cells, refined.cells):
        if first.names != second.names or canonical(first.metadata) != canonical(second.metadata):
            raise ValueError('shell physical cell support or parameter names differ')
    return comparison


def _grid(like):
    return {key:getattr(like.config, key) for key in ('kmin', 'kmax', 'dk', 'n_mu', 'ap_interpolation_nk')} | {
        'shells':len(like.grid.centers), 'edges':like.grid.edges.tolist(),
        'centers':like.grid.centers.tolist(), 'widths':like.grid.widths.tolist()}


def _change(base, refined, labels):
    base, refined = np.asarray(base), np.asarray(refined)
    if (base.shape != refined.shape or base.shape != (len(labels),)
            or not np.all(np.isfinite(base)) or not np.all(np.isfinite(refined))
            or np.any(base <= 0.) or np.any(refined <= 0.)):
        raise ValueError('quadrature comparison requires matched positive finite outputs')
    delta, reverse = np.abs(refined/base-1.), np.abs(base/refined-1.)
    return {'labels':labels, 'base':base.tolist(), 'refined':refined.tolist(),
            'fractional_changes':delta.tolist(), 'reverse_fractional_changes':reverse.tolist(),
            'status':'passed' if np.all(delta < _TOLERANCE) else 'failed'}


def _summary(role, coverage, changes):
    reasons = coverage+changes
    status = 'failed' if reasons else 'passed'
    if role == 'historical_assessment' and not coverage:
        status = 'assessed'
    return {'status':status, 'reasons':reasons, 'within_tolerance':not reasons, 'role':role,
            'tolerance':_TOLERANCE, 'definition':'abs(refined/base-1)<0.01',
            'reverse_definition':'abs(base/refined-1); reported, not the acceptance denominator'}


def _covariances(base, refined):
    output = {}
    for field in _FIELDS:
        kind, suffix = field.split('_', 1)
        attribute = ('cov_cosmo_' if kind == 'cosmology' else 'growth_cov_')+suffix
        a, b = np.asarray(getattr(base, attribute)), np.asarray(getattr(refined, attribute))
        output[field] = {'base':a.tolist(), 'refined':b.tolist(),
                         'absolute_frobenius':float(np.linalg.norm(b-a))}
    return output


def compare_shell_resolution(base, refined, base_results, refined_results, *, role):
    comparison = _bind(base, refined, role)
    solver = common_fisher_solver(*(r for rows in (base_results, refined_results)
                                    for r in rows.values() if not _failure(r)))
    a, fa, ga = _cohort(base, base_results)
    b, fb, gb = _cohort(refined, refined_results)
    _bind_growth(ga, gb)
    coverage = [s.replace('AP resolutions', 'quadratures') for s in _coverage(a, b)]
    treatments, changes = {}, []
    for name in COSMOLOGY_TREATMENTS:
        if name not in a or name not in b:
            treatments[name] = {'status':'unavailable', 'failures':{
                k:f[name] for k,f in (('base', fa), ('refined', fb)) if name in f}}
            continue
        rows = {}
        for field in _FIELDS:
            x, y = a[name]['outputs'][field], b[name]['outputs'][field]
            if x['labels'] != y['labels']:
                raise ValueError('shell output labels differ')
            rows[field] = _change(x['sigma'], y['sigma'], x['labels'])
        if any(row['status'] != 'passed' for row in rows.values()):
            changes.append(name+': a standard error changes by at least 1%')
        treatments[name] = {'outputs':rows, 'covariances':_covariances(base_results[name], refined_results[name]),
                            'solves':{'base':a[name]['solves'], 'refined':b[name]['solves']}}
    fisher = {'global':_matrix_change(base.F_like, refined.F_like, base.names),
        'bins':[_matrix_change(x, y, base.names) for x,y in zip(base.bin_F_like, refined.bin_F_like)],
        'cells':[_matrix_change(x.F_like, y.F_like, x.names) for x,y in zip(base.cells, refined.cells)]}
    return jsonable({**_summary(role, coverage, changes), 'comparison':comparison, 'case':base.config.case,
        'base_grid':_grid(base), 'refined_grid':_grid(refined), 'treatments':treatments, 'fisher':fisher,
        'fisher_solver':fisher_solver_record(solver)})


def compare_shell_ratios(base_b, base_c, refined_b, refined_c, *, role):
    pairs = (base_b, base_c, refined_b, refined_c)
    likes, results = zip(*pairs)
    if [like.config.case for like in likes] != ['B', 'C', 'B', 'C']:
        raise ValueError('ratio roles must be B,C on both quadratures')
    comparison = _bind(likes[0], likes[2], role)
    if _bind(likes[1], likes[3], role) != comparison:
        raise ValueError('B/C resolution roles differ')
    for i,j in ((0,1), (2,3)):
        x,y = asdict(likes[i].config), asdict(likes[j].config)
        x.pop('case'); y.pop('case')
        if (canonical(x) != canonical(y)
                or canonical(likes[i].reference_cosmology) != canonical(likes[j].reference_cosmology)
                or canonical(_model(likes[i])) != canonical(_model(likes[j]))):
            raise ValueError('B/C cosmology/model/quadrature configuration differs')
        p,q = ({r['name']:r for r in like.parameters} for like in (likes[i],likes[j]))
        if any(p[n]['prior_sigma'] != q[n]['prior_sigma'] for n in p.keys() & q.keys()):
            raise ValueError('B/C common nuisance priors differ')
    solver = common_fisher_solver(*(r for rows in results for r in rows.values() if not _failure(r)))
    cohorts = [_cohort(like, rows) for like,rows in pairs]
    coverage = []
    for item in cohorts[1:]:
        _bind_growth(cohorts[0][2], item[2])
        coverage += [s.replace('AP resolutions', 'survey/quadrature comparisons') for s in _coverage(cohorts[0][0], item[0])]
    treatments, changes, count = {}, [], 0
    for name in COSMOLOGY_TREATMENTS:
        failures = {k:row[1][name] for k,row in zip(('base_B','base_C','refined_B','refined_C'),cohorts) if name in row[1]}
        if failures:
            treatments[name] = {'status':'unavailable', 'failures':failures}
            continue
        rows = {}
        for field in _FIELDS:
            values = [item[0][name]['outputs'][field] for item in cohorts]
            labels = values[0]['labels']
            if any(v['labels'] != labels for v in values):
                raise ValueError('B/C free-coordinate or growth-output labels differ')
            b,c,br,cr = [v['sigma'] for v in values]
            rows[field] = {'B_over_C':_change(b/c, br/cr, labels), 'C_over_B':_change(c/b, cr/br, labels)}
            if any(r['status'] != 'passed' for r in rows[field].values()):
                changes.append(name+': '+field+' ratio changes by at least 1%')
        treatments[name] = rows
        count += 1
    return jsonable({**_summary(role, coverage, changes), 'comparison':comparison,
        'base_grid':_grid(likes[0]), 'refined_grid':_grid(likes[2]), 'treatments':treatments,
        'successful_ratio_treatments':count, 'fisher_solver':fisher_solver_record(solver)})
