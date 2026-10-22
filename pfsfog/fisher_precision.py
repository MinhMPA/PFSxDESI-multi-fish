"""Declared arithmetic for recovering a covariance from fixed information.

The accurate calculation inverts the exact binary64 Fisher entries; it does
not first round a normalized matrix or alter weak directions with a prior.
"""
from __future__ import annotations

from decimal import (Context, Decimal, DecimalException, DivisionByZero,
                     InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext)

import numpy as np

LEGACY_FISHER_SOLVER = 'scaled-cholesky-float64-v1'
DECIMAL_FISHER_SOLVER = 'decimal50-cholesky-v1'
FISHER_SOLVERS = (LEGACY_FISHER_SOLVER, DECIMAL_FISHER_SOLVER)


def fisher_solver_record(name: str) -> dict:
    if name not in FISHER_SOLVERS:
        raise ValueError(f'unknown Fisher solver {name!r}')
    if name == LEGACY_FISHER_SOLVER:
        return {'name': name, 'input': 'binary64', 'decimal_digits': None,
                'target': 'binary64_diagonal_normalization_of_symmetrized_F',
                'method': 'numpy_cholesky_two_dense_solves', 'output': 'binary64',
                'residual': 'binary64_normalized_matrix_times_inverse',
                'maximum_scaled_residual': 1e-10, 'failure': 'raise_no_regularization'}
    return {'name': name, 'input': 'exact_binary64', 'decimal_digits': 50,
            'target': 'decimal_symmetrized_F', 'method': 'decimal_cholesky_inverse_triangular_product',
            'decimal_rounding': 'ROUND_HALF_EVEN', 'output': 'binary64_nearest',
            'residual': 'decimal_diagonal_scaled_F_times_covariance_before_and_after_rounding',
            'maximum_scaled_residual': 1e-10, 'failure': 'raise_no_regularization'}


def solver_from_record(record: dict) -> str:
    declared = record.get('fisher_solver')
    if not isinstance(declared, dict):
        raise ValueError('missing Fisher solver identity')
    name = declared.get('name')
    if declared != fisher_solver_record(name):
        raise ValueError('Fisher solver declaration differs')
    return name


def common_fisher_solver(*results) -> str:
    names = [getattr(result, 'fisher_solver', None) for result in results]
    if not names:
        raise ValueError('missing Fisher solver result identity')
    for name in names:
        fisher_solver_record(name)
    if any(name != names[0] for name in names):
        raise ValueError('mixed Fisher solver identities')
    return names[0]


def _residual(matrix, inverse, scales):
    n = len(matrix)
    zero, one = Decimal(0), Decimal(1)
    largest = zero
    for i in range(n):
        for j in range(n):
            value = sum((matrix[i][k]*inverse[k][j] for k in range(n)), zero)
            value = (value - (one if i == j else zero))*scales[j]/scales[i]
            largest = max(largest, abs(value))
    return largest


def decimal_inverse(matrix: np.ndarray, labels, context: str) -> tuple[np.ndarray, dict]:
    from .revision_forecast import _scaled_spd

    f = np.asarray(matrix, dtype=float)
    if (f.ndim != 2 or not len(f) or f.shape != (len(labels), len(labels))
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError(f'invalid matrix or parameter labels in {context}')
    metrics = _scaled_spd(f, context)
    arithmetic = Context(prec=50, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                         traps=[InvalidOperation, DivisionByZero, Overflow])
    try:
        with localcontext(arithmetic):
            n = len(f)
            zero, one, two = Decimal(0), Decimal(1), Decimal(2)
            a = [[(Decimal.from_float(float(f[i, j]))
                   + Decimal.from_float(float(f[j, i])))/two
                  for j in range(n)] for i in range(n)]
            lower = [[zero]*n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    v = a[i][j] - sum((lower[i][k]*lower[j][k] for k in range(j)), zero)
                    if i == j:
                        if v <= zero:
                            raise ValueError(f'nonpositive Fisher pivot for {labels[i]} in {context}')
                        lower[i][j] = v.sqrt()
                    else:
                        lower[i][j] = v/lower[j][j]
            linv = [[zero]*n for _ in range(n)]
            for j in range(n):
                linv[j][j] = one/lower[j][j]
                for i in range(j + 1, n):
                    linv[i][j] = -sum((lower[i][k]*linv[k][j]
                                       for k in range(j, i)), zero)/lower[i][i]
            inverse = [[zero]*n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    v = sum((linv[k][i]*linv[k][j] for k in range(i, n)), zero)
                    inverse[i][j] = inverse[j][i] = v
            result = np.asarray(inverse, dtype=float)
            if not np.all(np.isfinite(result)):
                raise ValueError(f'nonfinite rounded Fisher covariance in {context}')
            rounded = [[Decimal.from_float(float(v)) for v in row] for row in result]
            scales = [a[i][i].sqrt() for i in range(n)]
            before = _residual(a, inverse, scales)
            after = _residual(a, rounded, scales)
            if not after.is_finite() or after > Decimal('1e-10'):
                raise ValueError(f'scaled inverse residual {after} in {context}')
            return result, {**metrics, 'scaled_inverse_residual': float(after),
                            'decimal_inverse_residual_before_rounding': str(before),
                            'decimal_inverse_residual_after_rounding': str(after)}
    except DecimalException as exc:
        raise ValueError(f'Decimal Fisher inverse failed in {context}: {exc}') from exc
