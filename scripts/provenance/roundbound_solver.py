"""Decimal Fisher inverse whose binary64 rounding residual is bounded, not capped.

The arithmetic is copied verbatim from pfsfog.fisher_precision.decimal_inverse
(qualified source, unchanged).  Only the acceptance rule differs.  The original
requires the diagonal-scaled residual |F C - I| of the binary64-rounded
covariance C to be at most 1e-10.  At nuisance priors x10 the scaled condition
number is about 1.24e7, so nearest rounding of an exact inverse can exceed that
cap (final-loop B_prior10: 1.079e-10), although each covariance element is
correct to one binary64 rounding.

Here the 50-digit inverse must have scaled residual <= 1e-30.  The rounded
covariance must then satisfy, elementwise, |C~ - C| <= u|C| with u = 2^-53,
and its scaled residual must obey the bound implied by nearest rounding,
    r_after <= r_before + u max_ij sum_k |F_ik| |C_kj| s_j/s_i ,
evaluated in the same 50-digit arithmetic.  Approved by the author on
22 October 2026.
"""
from __future__ import annotations

from decimal import (Context, Decimal, DecimalException, DivisionByZero,
                     InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext)

import numpy as np

import pfsfog.revision_forecast as revision_forecast
from pfsfog.fisher_precision import _residual

ROUNDBOUND_FISHER_SOLVER = 'decimal50-cholesky-roundbound-v1'
UNIT_ROUNDOFF = Decimal.from_float(2.0**-53)  # exact; Decimal(2)**-53 rounds at 28 digits
EXACT_RESIDUAL_MAX = Decimal('1e-30')


def roundbound_solver_record() -> dict:
    return {'name': ROUNDBOUND_FISHER_SOLVER, 'input': 'exact_binary64', 'decimal_digits': 50,
            'target': 'decimal_symmetrized_F', 'method': 'decimal_cholesky_inverse_triangular_product',
            'decimal_rounding': 'ROUND_HALF_EVEN', 'output': 'binary64_nearest',
            'residual': 'decimal_diagonal_scaled_F_times_covariance_before_and_after_rounding',
            'maximum_exact_scaled_residual': float(EXACT_RESIDUAL_MAX),
            'rounded_acceptance': 'elementwise |C~-C|<=u|C| and r_after<=r_before+u*max_ij(sum_k|F_ik||C_kj|s_j/s_i)',
            'unit_roundoff': float(UNIT_ROUNDOFF), 'failure': 'raise_no_regularization',
            'arithmetic_source': 'pfsfog.fisher_precision.decimal_inverse (verbatim)'}


def _rounding_bound(matrix, inverse, scales):
    n = len(matrix)
    zero = Decimal(0)
    largest = zero
    for i in range(n):
        for j in range(n):
            value = sum((abs(matrix[i][k])*abs(inverse[k][j]) for k in range(n)), zero)
            largest = max(largest, value*scales[j]/scales[i])
    return UNIT_ROUNDOFF*largest


def roundbound_inverse(matrix: np.ndarray, labels, context: str) -> tuple[np.ndarray, dict]:
    f = np.asarray(matrix, dtype=float)
    if (f.ndim != 2 or not len(f) or f.shape != (len(labels), len(labels))
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError(f'invalid matrix or parameter labels in {context}')
    metrics = revision_forecast._scaled_spd(f, context)
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
            # Acceptance differs from decimal_inverse from here on.
            for i in range(n):
                for j in range(n):
                    if abs(rounded[i][j] - inverse[i][j]) > UNIT_ROUNDOFF*abs(inverse[i][j]):
                        raise ValueError(f'covariance element not nearest-rounded in {context}')
            scales = [a[i][i].sqrt() for i in range(n)]
            before = _residual(a, inverse, scales)
            after = _residual(a, rounded, scales)
            bound = _rounding_bound(a, inverse, scales)
            if not before.is_finite() or before > EXACT_RESIDUAL_MAX:
                raise ValueError(f'exact scaled inverse residual {before} in {context}')
            if not after.is_finite() or after > before + bound:
                raise ValueError(f'rounded residual {after} exceeds rounding bound {before + bound} in {context}')
            return result, {**metrics, 'scaled_inverse_residual': float(after),
                            'decimal_inverse_residual_before_rounding': str(before),
                            'decimal_inverse_residual_after_rounding': str(after),
                            'rounding_residual_bound': str(bound),
                            'legacy_1e-10_cap_satisfied': bool(after <= Decimal('1e-10'))}
    except DecimalException as exc:
        raise ValueError(f'Decimal Fisher inverse failed in {context}: {exc}') from exc


_ORIGINAL_SCALED_INVERSE = revision_forecast._scaled_inverse


def _routed_scaled_inverse(matrix, labels, context, *, fisher_solver='scaled-cholesky-float64-v1'):
    if fisher_solver == ROUNDBOUND_FISHER_SOLVER:
        return roundbound_inverse(matrix, labels, context)
    return _ORIGINAL_SCALED_INVERSE(matrix, labels, context, fisher_solver=fisher_solver)


def install():
    """Route the new solver name at run time; all other names are unchanged."""
    revision_forecast._scaled_inverse = _routed_scaled_inverse
