"""Forward accuracy of Fisher covariances, without any change to information."""
import copy
from decimal import getcontext, ROUND_DOWN
from fractions import Fraction
import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('pfsfog.fisher_precision')


def rational_inverse(a):
    """Independent exact Gauss--Jordan arithmetic for a small binary matrix."""
    n = len(a)
    rows = [[Fraction(float(v)) for v in row]
            + [Fraction(i == j) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot = rows[i][i]
        rows[i] = [v/pivot for v in rows[i]]
        for j in range(n):
            if j != i:
                factor = rows[j][i]
                rows[j] = [v-factor*w for v, w in zip(rows[j], rows[i])]
    return np.array([[float(v) for v in row[n:]] for row in rows])


def test_exact_analytic_inverse_and_residuals():
    a = api()
    result, metrics = a.decimal_inverse(np.array([[4., 1.], [1., 3.]]), ['x', 'y'], 'analytic')
    np.testing.assert_array_equal(result, np.array([[3., -1.], [-1., 4.]])/11)
    assert float(metrics['decimal_inverse_residual_before_rounding']) < 1e-40
    assert metrics['scaled_inverse_residual'] <= 1e-10
    assert result.dtype == np.dtype('float64')


def test_whole_weak_direction_inverse_matches_exact_rational_arithmetic():
    a = api()
    design = np.array([[10., 10., 3., 1.], [5., 5., -2., 7.],
                       [1., 1., 8., -4.], [-6., -6., 1., 2.]])
    f = design.T @ design + np.diag([1/1024, 1/2048, 1/64, 1/32])
    scale = np.array([2.**-10, 2.**5, 2.**-2, 2.**7])
    f = f*scale[:, None]*scale[None, :]
    expected = rational_inverse(f)
    result, metrics = a.decimal_inverse(f, ['x', 'y', 'u', 'v'], 'weak')
    np.testing.assert_array_equal(result, expected)
    assert metrics['scaled_condition'] > 1e5
    assert metrics['scaled_inverse_residual'] <= 1e-10


@pytest.mark.parametrize('matrix', [
    np.ones((2, 3)), np.empty((0, 0)), np.array([[1., np.nan], [np.nan, 1.]]),
    np.array([[np.inf, 0.], [0., 1.]]), np.array([[0., 0.], [0., 1.]]),
    np.array([[1., .1], [.2, 1.]]), np.ones((2, 2)), np.array([[1., 2.], [2., 1.]]),
])
def test_invalid_fisher_does_not_get_regularized(matrix):
    with pytest.raises(ValueError):
        api().decimal_inverse(matrix, ['x', 'y'], 'invalid')


@pytest.mark.parametrize('labels', [['x'], ['x', 'x'], [1, 2]])
def test_labels_bind_the_matrix(labels):
    with pytest.raises(ValueError, match='label'):
        api().decimal_inverse(np.eye(2), labels, 'labels')


def test_decimal_context_is_local_and_declared_rounding_is_used():
    a = api()
    ctx = getcontext()
    original = copy.copy(ctx)
    try:
        ctx.prec = 11
        ctx.rounding = ROUND_DOWN
        before = repr(ctx)
        result, _ = a.decimal_inverse(np.array([[4., 1.], [1., 3.]]), ['x', 'y'], 'local')
        np.testing.assert_array_equal(result, np.array([[3., -1.], [-1., 4.]])/11)
        assert repr(ctx) == before
    finally:
        from decimal import setcontext
        setcontext(original)


def test_solver_declarations_are_explicit_and_reject_mutation():
    a = api()
    record = a.fisher_solver_record(a.DECIMAL_FISHER_SOLVER)
    assert record['decimal_digits'] == 50
    assert record['input'] == 'exact_binary64'
    assert a.solver_from_record({'fisher_solver': record}) == a.DECIMAL_FISHER_SOLVER
    for item in ({}, {'fisher_solver': {**record, 'decimal_digits': 49}}):
        with pytest.raises(ValueError, match='solver'):
            a.solver_from_record(item)
    with pytest.raises(ValueError, match='solver'):
        a.fisher_solver_record('unknown')


def test_legacy_arithmetic_remains_bitwise_equal():
    from pfsfog.revision_forecast import _scaled_inverse
    a = api()
    f = np.array([[10., 9.], [9., 10.]])
    scales = np.sqrt(np.diag(f))
    corr = (f+f.T)/(2*scales[:, None]*scales[None, :])
    chol = np.linalg.cholesky(corr)
    expected = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(2)))
    expected = expected/scales[:, None]/scales[None, :]
    default, _ = _scaled_inverse(f, ['a', 'b'], 'legacy')
    explicit, _ = _scaled_inverse(f, ['a', 'b'], 'legacy', fisher_solver=a.LEGACY_FISHER_SOLVER)
    np.testing.assert_array_equal(default, expected)
    np.testing.assert_array_equal(explicit, expected)


def test_extended_and_native_projection_use_declared_arithmetic_in_both_blocks():
    a = api()
    from pfsfog.cosmology_projection import project_cosmology_precision
    from pfsfog.revision_forecast import project_baseline_precision
    from test_cosmology_resolution import sample, native3
    like, results = sample()
    p = project_cosmology_precision(like, 'conditional3', fisher_solver=a.DECIMAL_FISHER_SOLVER)
    native, _ = native3(like, results['conditional3'])
    q = project_baseline_precision(native, 1., fisher_solver=a.DECIMAL_FISHER_SOLVER)
    np.testing.assert_array_equal(p['covariance'], q['covariance'])
    for value in (p, q):
        assert value['fisher_solver'] == a.DECIMAL_FISHER_SOLVER
        for block in ('global_solve', 'cosmology_block_solve'):
            assert 'decimal_inverse_residual_before_rounding' in value['validation'][block]
    np.testing.assert_array_equal(p['cov_cosmo_nuisance_fixed'][:3, :3], q['cov_cosmo_nuisance_fixed'])
