"""Compare AP discretizations without confusing convergence with identity.

Inputs are independently constructed likelihoods and all seven treatment
outcomes. The caller authenticates their source/runtime/input identities and
retains the raw matrices. This module reconstructs Fisher information and
propagated errors from each resolution's own saved arrays; it initializes no
emulator and never certifies the physical accuracy of a reference calculation.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from .ap_numerics import configuration_profile
from .cosmology_projection import COSMOLOGY_TREATMENTS, project_cosmology_precision
from .extended_artifacts import (PRECISION_FIELDS, canonical, jsonable, require_close,
                                 validate_likelihood)
from .fisher_precision import common_fisher_solver, fisher_solver_record

_REQUIRED = {'conditional3', 'marginal6'}
_FIELDS = ('cosmology_marg', 'cosmology_nuisance_fixed', 'growth_marg', 'growth_nuisance_fixed')
_TOLERANCE = .01


def _physical_configuration(like):
    configuration_profile(like.config)
    if like.config.cosmology_basis != 'extended6':
        raise ValueError('AP resolution requires native extended6 likelihoods')
    value = asdict(like.config)
    for key in ('ap_interpolation_nk', 'ap_refinement_nk'):
        value.pop(key)
    return value


def _physical_model(like):
    value = dict(like.model)
    for key in ('ap_interpolation_nk', 'ap_refinement_nk'):
        if key in value:
            if canonical(value.pop(key)) != canonical(getattr(like.config, key)):
                raise ValueError('AP model declaration differs from its actual configuration')
    return value


def _bind_resolutions(base, refined):
    a, b = base.config.ap_interpolation_nk, refined.config.ap_interpolation_nk
    if a not in (8192, 16384, 32768) or b not in (16384, 32768, 65536) or a >= b:
        raise ValueError('resolution comparison requires increasing declared AP grids; same-grid identity is separate')
    if canonical(_physical_configuration(base)) != canonical(_physical_configuration(refined)):
        raise ValueError('AP resolution physical configuration differs')
    if (base.names != refined.names or canonical(base.parameters) != canonical(refined.parameters)
            or canonical(base.reference_cosmology) != canonical(refined.reference_cosmology)
            or canonical(_physical_model(base)) != canonical(_physical_model(refined))
            or len(base.cells) != len(refined.cells)):
        raise ValueError('AP resolution parameter/prior/reference/model identity differs')
    require_close(base.fiducial, refined.fiducial, 'resolution fiducial', rtol=0., atol=0.)
    for field in ('edges', 'centers', 'widths'):
        require_close(getattr(base.grid, field), getattr(refined.grid, field),
                      'resolution observed shells', rtol=0., atol=0.)
    for c, d in zip(base.cells, refined.cells):
        if c.names != d.names or canonical(c.metadata) != canonical(d.metadata):
            raise ValueError('AP resolution physical support or local names differ')
    # A covariance substituted without its own consistent D/C/F reconstruction
    # fails here. Independent physical construction is the caller's obligation.
    validate_likelihood(base)
    validate_likelihood(refined)


def _failure(value):
    if not isinstance(value, dict):
        return False
    if (set(value) != {'status', 'error', 'raw'} or value['status'] != 'failed'
            or not isinstance(value['error'], str) or not value['error']
            or not isinstance(value['raw'], dict)):
        raise ValueError('invalid explicit treatment failure')
    return True


def _cohort(like, results):
    if not isinstance(results, dict) or set(results) != set(COSMOLOGY_TREATMENTS):
        raise ValueError('all seven treatment outcomes must be present')
    names = list(like.cosmo_names)
    z = np.array([(lo+hi)/2 for lo, hi in like.config.nominal_bins])
    successful, failures, common_growth = {}, {}, None
    solver=common_fisher_solver(*(r for r in results.values() if not _failure(r)))
    for name in COSMOLOGY_TREATMENTS:
        result = results[name]
        if _failure(result):
            failures[name] = result
            continue
        if getattr(result, 'likelihood', None) is not like or result.treatment != name:
            raise ValueError('treatment belongs to another likelihood or disposition')
        precision = project_cosmology_precision(like, name, fisher_solver=solver)
        if (tuple(result.free_cosmology) != precision['free_cosmology']
                or tuple(result.active_parameter_names) != precision['active_parameter_names']):
            raise ValueError('treatment free-coordinate or active-parameter mask differs')
        for field in PRECISION_FIELDS:
            require_close(getattr(result, field), precision[field], 'resolution '+field)
        active = [names.index(n) for n in precision['free_cosmology']]
        fixed = [i for i in range(6) if i not in active]
        require_close(result.growth_z, z, 'resolution growth redshifts', rtol=0., atol=0.)
        predicted, jac = np.asarray(result.growth_fiducial), np.asarray(result.growth_jacobian)
        if (predicted.shape != z.shape or jac.shape != (len(z), 6)
                or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(jac))):
            raise ValueError('resolution growth prediction/Jacobian is malformed')
        if common_growth is None:
            common_growth = (predicted, jac)
        else:
            require_close(predicted, common_growth[0], 'within-cohort growth fiducial')
            require_close(jac, common_growth[1], 'within-cohort growth Jacobian')
        outputs = {}
        for suffix in ('marg', 'nuisance_fixed'):
            cov = np.asarray(getattr(result, 'cov_cosmo_'+suffix))
            if np.any(cov[fixed, :] != 0.) or np.any(cov[:, fixed] != 0.):
                raise ValueError('fixed cosmological coordinates must have exactly zero covariance')
            propagated = jac @ cov @ jac.T
            require_close(getattr(result, 'growth_cov_'+suffix), propagated, 'resolution full growth propagation')
            for kind, labels, variance in (
                    ('cosmology', list(precision['free_cosmology']), np.diag(cov)[active]),
                    ('growth', z.tolist(), np.diag(propagated))):
                if not np.all(np.isfinite(variance)) or np.any(variance <= 0.):
                    raise ValueError('active reported variance must be finite and positive')
                outputs[kind+'_'+suffix] = {'labels':labels, 'sigma':np.sqrt(variance)}
        successful[name] = {'outputs':outputs, 'solves':precision['validation']}
    return successful, failures, common_growth


def _bind_growth(a, b):
    if a is not None and b is not None:
        require_close(a[0], b[0], 'AP resolution growth fiducial')
        require_close(a[1], b[1], 'AP resolution growth Jacobian')


def _coverage(a, b):
    reasons = []
    if set(a) != set(b):
        reasons.append('successful treatment sets differ across AP resolutions')
    if not _REQUIRED <= set(a) or not _REQUIRED <= set(b):
        reasons.append('a required conditional3 or marginal6 treatment failed')
    return reasons


def _change(coarse, fine, labels):
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if (coarse.shape != fine.shape or coarse.shape != (len(labels),)
            or not np.all(np.isfinite(coarse)) or not np.all(np.isfinite(fine))
            or np.any(coarse <= 0.) or np.any(fine <= 0.)):
        raise ValueError('resolution comparison requires matched positive finite outputs')
    delta = np.abs(coarse/fine-1.)
    return {'labels':labels, 'coarse':coarse.tolist(), 'fine':fine.tolist(),
            'fractional_changes':delta.tolist(), 'status':'passed' if np.all(delta < _TOLERANCE) else 'failed'}


def _whiten(cov, values):
    scale = np.sqrt(np.diag(cov))
    chol = np.linalg.cholesky((cov+cov.T)/2/scale[:, None]/scale[None, :])
    return np.linalg.solve(chol, values/scale[:, None]), scale, chol


def _spectral_changes(base, refined):
    output = []
    for c, d in zip(base.cells, refined.cells):
        if not d.metadata['retained_pairs']:
            continue
        row = {'cell_id':d.metadata['cell_id'], 'parameter_names':d.names, 'k':refined.grid.centers.tolist()}
        for field, label in (('fiducial_mean', 'mean'), ('derivatives', 'derivatives')):
            coarse, fine = np.asarray(getattr(c, field)), np.asarray(getattr(d, field))
            delta = coarse-fine
            if fine.ndim == 2:
                fine, delta = fine[..., None], delta[..., None]
            white_delta, white_fine = [], []
            for i, cov in enumerate(d.covariance):
                white_delta.append(_whiten(cov, delta[i])[0])
                white_fine.append(_whiten(cov, fine[i])[0])
            absnorm = np.sqrt(np.sum(np.asarray(white_delta)**2, axis=(0, 1)))
            refnorm = np.sqrt(np.sum(np.asarray(white_fine)**2, axis=(0, 1)))
            row[label] = {'raw_l2':np.sqrt(np.sum(delta**2, axis=(0, 1))).tolist(),
                'raw_max_abs':np.max(np.abs(delta), axis=(0, 1)).tolist(),
                'whitened_norm':absnorm.tolist(), 'fine_whitened_norm':refnorm.tolist(),
                'relative_to_fine':[float(a/b) if b > 0. else None for a, b in zip(absnorm, refnorm)]}
        changes = []
        for old, new in zip(c.covariance, d.covariance):
            _, scale, chol = _whiten(new, np.zeros((len(new), 1)))
            delta = (old-new)/scale[:, None]/scale[None, :]
            changes.append(np.linalg.solve(chol, np.linalg.solve(chol, delta).T).T)
        row['covariance'] = {'raw_frobenius':float(np.linalg.norm(c.covariance-d.covariance)),
                             'whitened_frobenius':float(np.linalg.norm(changes))}
        output.append(row)
    return output


def _matrix_change(coarse, fine, names):
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    delta = coarse-fine
    diagonal = np.diag(fine)
    indices = np.flatnonzero(diagonal > 0.)
    # Zero-information coordinates stay explicitly unscaled; no eigenvalue is
    # clipped and no elementwise near-zero Fisher acceptance gate is introduced.
    scale = np.ones(len(names)); scale[indices] = np.sqrt(diagonal[indices])
    scaled = fine/scale[:, None]/scale[None, :]
    eig, vec = np.linalg.eigh((scaled+scaled.T)/2)
    return {'names':list(names), 'absolute_frobenius':float(np.linalg.norm(delta)),
            'diagonal_changes':np.diag(delta).tolist(),
            'scaled_delta_frobenius':float(np.linalg.norm(delta/scale[:, None]/scale[None, :])),
            'scaled_coordinate_indices':indices.tolist(), 'fine_scaled_eigenvalues':eig.tolist(),
            'fine_weakest_direction':vec[:, 0].tolist() if len(names) else []}


def compare_ap_resolution(base, refined, base_results: dict, refined_results: dict) -> dict:
    """Report errors and raw information changes against the finer AP grid."""
    _bind_resolutions(base, refined)
    solver=common_fisher_solver(*(r for results in (base_results,refined_results)
                                  for r in results.values() if not _failure(r)))
    a, fa, ga = _cohort(base, base_results)
    b, fb, gb = _cohort(refined, refined_results)
    _bind_growth(ga, gb)
    reasons = _coverage(a, b)
    treatments = {}
    for name in COSMOLOGY_TREATMENTS:
        if name not in a or name not in b:
            treatments[name] = {'status':'unavailable', 'failures':{
                key:failures[name] for key, failures in (('base', fa), ('refined', fb)) if name in failures}}
            continue
        rows = {field:_change(a[name]['outputs'][field]['sigma'], b[name]['outputs'][field]['sigma'],
                              a[name]['outputs'][field]['labels']) for field in _FIELDS}
        if any(row['status'] != 'passed' for row in rows.values()):
            reasons.append(name+': a standard error changes by at least 1%')
        treatments[name] = {**rows, 'solves':{'base':a[name]['solves'], 'refined':b[name]['solves']}}
    fisher = {'global':_matrix_change(base.F_like, refined.F_like, base.names),
        'bins':[_matrix_change(a, b, base.names) for a, b in zip(base.bin_F_like, refined.bin_F_like)],
        'cells':[_matrix_change(a.F_like, b.F_like, a.names) for a, b in zip(base.cells, refined.cells)]}
    return jsonable({'status':'failed' if reasons else 'passed', 'reasons':reasons,
        'case':base.config.case, 'base_ap_nk':base.config.ap_interpolation_nk,
        'refined_ap_nk':refined.config.ap_interpolation_nk, 'tolerance':_TOLERANCE,
        'definition':'abs(coarse/fine-1)<0.01', 'comparison':'AP output resolution',
        'treatments':treatments, 'spectra':_spectral_changes(base, refined), 'fisher':fisher,
        'fisher_solver':fisher_solver_record(solver)})


def _attached_likelihood(results):
    if not isinstance(results, dict) or set(results) != set(COSMOLOGY_TREATMENTS):
        raise ValueError('all seven treatment outcomes must be present')
    likes = [result.likelihood for result in results.values() if not _failure(result)]
    if not likes or any(like is not likes[0] for like in likes):
        raise ValueError('ratio outcomes must identify one physical likelihood')
    return likes[0]


def compare_resolution_ratios(base_B: dict, base_C: dict, refined_B: dict, refined_C: dict) -> dict:
    """Compare C/B and B/C explicitly for both nuisance dispositions."""
    inputs = (base_B, base_C, refined_B, refined_C)
    solver=common_fisher_solver(*(r for results in inputs for r in results.values() if not _failure(r)))
    likes = [_attached_likelihood(r) for r in inputs]
    if [like.config.case for like in likes] != ['B', 'C', 'B', 'C']:
        raise ValueError('ratio roles must be B,C at coarse and finer AP grids')
    _bind_resolutions(likes[0], likes[2]); _bind_resolutions(likes[1], likes[3])
    for i, j in ((0, 1), (2, 3)):
        x, y = asdict(likes[i].config), asdict(likes[j].config)
        x.pop('case'); y.pop('case')
        if canonical(x) != canonical(y) or canonical(likes[i].reference_cosmology) != canonical(likes[j].reference_cosmology):
            raise ValueError('B/C ratio cosmology or numerical configuration differs')
        if canonical(_physical_model(likes[i])) != canonical(_physical_model(likes[j])):
            raise ValueError('B/C ratio physical model differs')
    cohorts = [_cohort(like, result) for like, result in zip(likes, inputs)]
    for item in cohorts[1:]:
        _bind_growth(cohorts[0][2], item[2])
    reasons = _coverage(cohorts[0][0], cohorts[2][0])+_coverage(cohorts[1][0], cohorts[3][0])
    treatments, count = {}, 0
    for name in COSMOLOGY_TREATMENTS:
        failures = {label:item[1][name] for label, item in zip(('base_B', 'base_C', 'refined_B', 'refined_C'), cohorts) if name in item[1]}
        if failures:
            treatments[name] = {'status':'unavailable', 'failures':failures}
            continue
        rows = {}
        for field in _FIELDS:
            values = [item[0][name]['outputs'][field] for item in cohorts]
            labels = values[0]['labels']
            if any(item['labels'] != labels for item in values):
                raise ValueError('ratio free-coordinate or growth-output labels differ')
            a, c, b, d = [item['sigma'] for item in values]
            rows[field] = {'C_over_B':_change(c/a, d/b, labels), 'B_over_C':_change(a/c, b/d, labels)}
            if any(row['status'] != 'passed' for row in rows[field].values()):
                reasons.append(name+': '+field+' ratio changes by at least 1%')
        treatments[name] = rows
        count += 1
    return jsonable({'status':'failed' if reasons else 'passed', 'reasons':reasons,
        'base_ap_nk':likes[0].config.ap_interpolation_nk, 'refined_ap_nk':likes[2].config.ap_interpolation_nk,
        'tolerance':_TOLERANCE, 'definition':'abs(coarse/fine-1)<0.01', 'treatments':treatments,
        'successful_ratio_treatments':count, 'fisher_solver':fisher_solver_record(solver)})
