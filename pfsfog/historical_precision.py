"""Preserved historical replay and separately identified accurate references."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from .extended_artifacts import (GROWTH_FIELDS, PRECISION_FIELDS, hash_value,
                                 jsonable, require_close)
from .fisher_precision import (DECIMAL_FISHER_SOLVER, LEGACY_FISHER_SOLVER,
                               fisher_solver_record)
from .revision_forecast import project_baseline_precision


def _array_pin(value):
    a = np.asarray(value)
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def _changes(new, old, rtol):
    allowance = 1e-12 + rtol*np.abs(old)
    difference = np.abs(new-old)
    bad = np.argwhere(difference > allowance)
    return {'rtol': rtol, 'atol': 1e-12, 'failed_entries': len(bad),
            'max_allowance_multiple': float(np.max(difference/allowance)),
            'failed_values': [{'indices': [int(i), int(j)], 'original': float(old[i, j]),
                               'derived': float(new[i, j]), 'absolute_difference': float(difference[i, j]),
                               'allowance': float(allowance[i, j])} for i, j in bad]}


def historical_precision_reference(likelihood, result, *, fisher_solver: str,
                                    parent_identity: dict) -> tuple[SimpleNamespace, dict]:
    """Caller must authenticate the original export; never overwrite its arrays."""
    if fisher_solver != DECIMAL_FISHER_SOLVER or getattr(result, 'fisher_solver', None) != LEGACY_FISHER_SOLVER:
        raise ValueError('historical derived reference requires declared original and new solver identities')
    required = {'oracle_export', 'worker_index_sha256', 'arrays_sha256', 'control_sha256'}
    if not isinstance(parent_identity, dict) or set(parent_identity) != required:
        raise ValueError('historical parent identity is incomplete')
    export = parent_identity['oracle_export']
    if (not isinstance(export, dict) or set(export) != {'path','sha256','case'}
            or not isinstance(export['path'], str) or not export['path']
            or export['case'] != likelihood.config.case):
        raise ValueError('historical parent case or export identity differs')
    for value in [export['sha256'], *(parent_identity[k] for k in required-{'oracle_export'})]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('historical parent digest is malformed')
    original = project_baseline_precision(likelihood, likelihood.config.nuisance_factor,
                                          fisher_solver=LEGACY_FISHER_SOLVER)
    replay = {}
    for field in PRECISION_FIELDS:
        saved = getattr(result, field)
        require_close(original[field], saved, 'historical legacy replay '+field)
        if field.startswith('F_') and not np.array_equal(original[field], saved):
            raise ValueError('historical Fisher/prior matrix is not the exact replayed input: '+field)
        replay[field] = bool(np.array_equal(original[field], saved))
    z = np.array([(lo+hi)/2 for lo,hi in likelihood.config.nominal_bins])
    require_close(result.growth_z, z, 'historical growth redshifts', rtol=0., atol=0.)
    if (np.shape(result.growth_fiducial) != z.shape
            or np.shape(result.growth_jacobian) != (len(z), 3)
            or not np.all(np.isfinite(result.growth_fiducial))
            or not np.all(np.isfinite(result.growth_jacobian))):
        raise ValueError('invalid historical growth predictions/Jacobian')
    for suffix in ('marg', 'nuisance_fixed'):
        growth = result.growth_jacobian @ original['cov_cosmo_'+suffix] @ result.growth_jacobian.T
        require_close(growth, getattr(result, 'growth_cov_'+suffix), 'historical legacy growth '+suffix)
        replay['growth_cov_'+suffix] = bool(np.array_equal(growth, getattr(result, 'growth_cov_'+suffix)))
    precise = project_baseline_precision(likelihood, likelihood.config.nuisance_factor,
                                         fisher_solver=fisher_solver)
    values = {**precise, **{field:np.array(getattr(result, field), copy=True) for field in GROWTH_FIELDS}}
    for suffix in ('marg', 'nuisance_fixed'):
        values['growth_cov_'+suffix] = result.growth_jacobian @ precise['cov_cosmo_'+suffix] @ result.growth_jacobian.T
        for field in ('cov_cosmo_'+suffix, 'growth_cov_'+suffix):
            require_close(values[field], getattr(result, field), 'historical scientific identity '+field)
    receipt = {'definition': 'fisher-historical-derived-v1', 'parent_identity': parent_identity,
        'parameter_names': list(likelihood.names), 'parameters_sha256': hash_value(likelihood.parameters),
        'parent_matrix_hashes': {field:_array_pin(getattr(result, field)) for field in (*PRECISION_FIELDS,*GROWTH_FIELDS)},
        'original_fisher_solver': fisher_solver_record(LEGACY_FISHER_SOLVER),
        'fisher_solver': fisher_solver_record(fisher_solver),
        'legacy_replay': {'status':'passed','rtol':1e-10,'atol':1e-12,'bitwise_equal':replay},
        'scientific_identity': {'status':'passed','rtol':1e-10,'atol':1e-12,
            'fields':['cov_cosmo_marg','cov_cosmo_nuisance_fixed','growth_cov_marg','growth_cov_nuisance_fixed']},
        'full_covariance_changes': {
            'same_resolution_allowance': _changes(precise['covariance'], result.covariance, 1e-8),
            'reconstruction_allowance': _changes(precise['covariance'], result.covariance, 1e-10)}}
    return SimpleNamespace(likelihood=likelihood, treatment='baseline3', **values), jsonable(receipt)
