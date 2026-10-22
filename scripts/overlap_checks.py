"""Independent Gaussian marginalization and added-spectrum information checks.

The full inverse uses the declared Decimal calculation in overlap_study.
Here Schur elimination and QR of whitened response rows supply independent
checks, including the once-only prior rows and derived growth covariance.
"""
from __future__ import annotations

import numpy as np

from pfsfog.revision_forecast import CellLikelihood, PriorResult


def _product(a, b):
    return np.einsum('...ij,...jk->...ik', a, b, optimize=False)


def _scaled_error(actual, expected):
    """Matrix discrepancy in units of the reference diagonal standard scales."""
    diagonal = np.diag(expected)
    if not np.isfinite(expected).all() or np.any(diagonal < 0):
        raise ValueError('nonfinite matrix or negative reference diagonal')
    scale = np.sqrt(diagonal)
    # A truly absent response has an exactly zero row and column. This only
    # defines units for the comparison; no Fisher or covariance is modified.
    scale[scale == 0] = 1.
    return float(np.max(abs(actual-expected)/scale[:, None]/scale[None, :], initial=0.))


def _require_error(value, limit, context):
    if not np.isfinite(value) or value >= limit:
        raise ValueError(f'{context}: scaled discrepancy {value} >= {limit}')
    return value


def _inverse(matrix):
    """Scaled positive-definite inverse, without regularizing weak directions."""
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all() or np.any(np.diag(matrix) <= 0):
        raise ValueError('nonpositive or nonfinite precision')
    scale = np.sqrt(np.diag(matrix))
    a = matrix/scale[:, None]/scale[None, :]
    _require_error(float(np.max(abs(a-a.T))), 1e-10, 'precision symmetry')
    try:
        lower = np.linalg.cholesky((a+a.T)/2)
        invlower = np.linalg.solve(lower, np.eye(len(a)))
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular or nonpositive precision') from exc
    inverse = _product(invlower.T, invlower)
    _require_error(float(np.max(abs(_product(a, inverse)-np.eye(len(a))))),
                   1e-10, 'scaled precision inverse')
    return inverse/scale[:, None]/scale[None, :]


def _whiten(d, c):
    if not np.isfinite(d).all() or not np.isfinite(c).all():
        raise ValueError('nonfinite conditional response or covariance')
    if d.shape[-2] == 0:
        return d.copy()
    diagonal = np.diagonal(c, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0):
        raise ValueError('nonpositive covariance diagonal')
    scale = np.sqrt(diagonal)
    corr = c/scale[..., :, None]/scale[..., None, :]
    _require_error(float(np.max(abs(corr-np.swapaxes(corr, -1, -2)))), 1e-10, 'covariance symmetry')
    try:
        return np.linalg.solve(np.linalg.cholesky(corr), d/scale[..., :, None])
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular or nonpositive covariance') from exc


def check_marginalization(result: PriorResult) -> dict:
    """Compare full, Schur and QR covariance; retain H,K,N,S and nuisance loss.

    H is the complete cosmological block: H⁻¹ still marginalizes the other
    cosmological coordinates while fixing all nuisance coefficients.
    """
    like = result.likelihood
    names = like.names
    if names[:3] != list(like.cosmo_names) or len(set(names)) != len(names):
        raise ValueError('cosmological order or global names differ')
    index = {n: i for i, n in enumerate(names)}
    rows = []
    for cell in like.cells:
        if len(set(cell.names)) != len(cell.names) or any(n not in index for n in cell.names):
            raise ValueError('local parameter inventory differs')
        if not cell.names:
            if cell.derivatives.shape[1] or cell.covariance.shape[1]:
                raise ValueError('unlabelled measured rows')
            continue
        white = _whiten(cell.derivatives, cell.covariance).reshape(-1, len(cell.names))
        global_rows = np.zeros((len(white), len(names)))
        global_rows[:, [index[n] for n in cell.names]] = white
        rows.append(global_rows)
    if not rows:
        raise ValueError('no measured response rows')
    response = np.concatenate(rows)
    assembled_like = _product(response.T, response)
    assembly_error = _require_error(_scaled_error(assembled_like, like.F_like), 1e-10,
                                    'response-row likelihood assembly')
    if [p['name'] for p in result.parameters] != names:
        raise ValueError('prior parameter inventory differs')
    prior = np.zeros(len(names))
    for i, p in enumerate(result.parameters):
        sigma = p['prior_sigma']
        if sigma is not None:
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('invalid finite prior width')
            prior[i] = 1/sigma**2
    if prior[:3].tolist() != [0., 1/25., 1.]:
        raise ValueError('cosmological prior differs')
    expected_prior = np.diag(prior)
    _require_error(_scaled_error(result.F_prior_cosmo+result.F_prior_nuisance, expected_prior),
                   1e-10, 'prior assembly')
    _require_error(_scaled_error(result.F_total, like.F_like+expected_prior), 1e-10,
                   'total precision assembly')
    prior_rows = np.diag(np.sqrt(prior))[prior > 0]
    augmented = np.concatenate((response, prior_rows))
    scale = np.sqrt(np.sum(augmented**2, axis=0))
    if np.any(scale == 0):
        raise ValueError('unconstrained parameter in whitened responses')
    r = np.linalg.qr(augmented/scale, mode='r')
    if r.shape != (len(names), len(names)):
        raise ValueError('too few independent response rows')
    singular = np.linalg.svd(r, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps*len(names)*singular[0]:
        raise ValueError('singular whitened response direction')
    ri = np.linalg.solve(r, np.eye(len(names)))
    qr_residual = _require_error(float(np.max(abs(_product(r, ri)-np.eye(len(names))))),
                                 1e-10, 'QR triangular inverse')
    qcov = _product(ri, ri.T)/scale[:, None]/scale[None, :]
    qr_error = _require_error(_scaled_error(qcov, result.covariance), 1e-8, 'QR/full covariance')
    f = result.F_total
    h, k, n = f[:3, :3], f[:3, 3:], f[3:, 3:]
    fixed = _inverse(h)
    if len(n):
        ns = np.sqrt(np.diag(n))
        white_k = np.linalg.solve(np.linalg.cholesky(n/ns[:, None]/ns[None, :]), k.T/ns[:, None])
        loss = _product(white_k.T, white_k)
    else:
        loss = np.zeros_like(h)
    s = h-loss
    marginal = _inverse(s)
    schur_error = _require_error(_scaled_error(marginal, result.covariance[:3, :3]),
                                 1e-8, 'Schur/full covariance')
    _require_error(_scaled_error(result.cov_cosmo_marg, result.covariance[:3, :3]),
                   1e-8, 'stored cosmological covariance')
    _require_error(_scaled_error(fixed, result.cov_cosmo_nuisance_fixed), 1e-8,
                   'nuisance-fixed cosmological covariance')
    j = result.growth_jacobian
    growth = _product(_product(j, marginal), j.T)
    growth_fixed = _product(_product(j, fixed), j.T)
    growth_error = _require_error(_scaled_error(growth, result.growth_cov_marg), 1e-8, 'growth covariance')
    fixed_growth_error = _require_error(_scaled_error(growth_fixed, result.growth_cov_nuisance_fixed),
                                        1e-8, 'nuisance-fixed growth covariance')
    eig, vectors = np.linalg.eigh(h)
    hinvroot = _product(vectors/np.sqrt(eig), vectors.T)
    penalty = _product(_product(hinvroot, loss), hinvroot)
    penalty_eig, penalty_vec = np.linalg.eigh((penalty+penalty.T)/2)
    return {'H': h, 'K': k, 'N': n, 'S': s, 'loss': loss, 'C_fixed': fixed,
            'C_marg': marginal, 'C_full': result.covariance, 'C_qr': qcov,
            'penalty_eigenvalues': penalty_eig,
            'penalty_directions': _product(hinvroot, penalty_vec),
            'full_schur_scaled_error': schur_error, 'qr_covariance_error': qr_error,
            'assembly_scaled_error': assembly_error, 'qr_inverse_residual': qr_residual,
            'growth_scaled_error': growth_error, 'fixed_growth_scaled_error': fixed_growth_error}


def conditional_information(old: CellLikelihood, new: CellLikelihood) -> dict:
    """Information in the additional rows, conditioned on already measured rows.

    The returned D, C and F use the new cell's complete named parameter basis.
    C includes all cross-covariance terms, and D removes the response predicted
    by the old rows. A new spectrum can therefore supply exactly zero information.
    """
    def labels(cell):
        return [(tuple(pair) if not isinstance(pair, str) else pair, ell)
                for pair, ell in cell.metadata['retained_observations']]

    previous, current = labels(old), labels(new)
    if (old.metadata['cell_id'] != new.metadata['cell_id']
            or len(set(previous)) != len(previous) or len(set(current)) != len(current)
            or not set(previous) <= set(current)
            or len(set(old.names)) != len(old.names) or len(set(new.names)) != len(new.names)
            or not set(old.names) <= set(new.names)):
        raise ValueError('conditional comparison is not nested in rows and parameters')
    nk, nobs, npar = new.derivatives.shape
    if (nobs != len(current) or npar != len(new.names)
            or new.covariance.shape != (nk, nobs, nobs)
            or old.derivatives.shape != (nk, len(previous), len(old.names))
            or old.covariance.shape != (nk, len(previous), len(previous))):
        raise ValueError('conditional cell dimensions differ')
    kept = [current.index(o) for o in previous]
    extra = [i for i in range(nobs) if i not in kept]
    ids = [new.names.index(n) for n in old.names]
    embedded_d = np.zeros((nk, len(kept), npar))
    embedded_d[:, :, ids] = old.derivatives
    if (not np.array_equal(new.derivatives[:, kept], embedded_d)
            or not np.array_equal(new.covariance[:, kept][:, :, kept], old.covariance)
            or not np.array_equal(new.fiducial_mean[:, kept], old.fiducial_mean)):
        raise ValueError('old spectra, response or covariance differ within new cell')
    d, c = new.derivatives, new.covariance
    dy, yy = d[:, extra], c[:, extra][:, :, extra]
    if kept and extra:
        dx, xx, yx = d[:, kept], c[:, kept][:, :, kept], c[:, extra][:, :, kept]
        # Cholesky whitening validates positive definiteness before the solves.
        _whiten(dx, xx)
        residual = dy-_product(yx, np.linalg.solve(xx, dx))
        conditional = yy-_product(yx, np.linalg.solve(xx, np.swapaxes(yx, -1, -2)))
    else:
        residual, conditional = dy.copy(), yy.copy()
    white = _whiten(residual, conditional)
    increment = np.einsum('koi,koj->ij', white, white, optimize=False)
    old_fisher = np.zeros((npar, npar))
    old_fisher[np.ix_(ids, ids)] = old.F_like
    error = _require_error(_scaled_error(old_fisher+increment, new.F_like), 1e-10,
                           'conditional information reconstruction')
    scale = np.sqrt(np.diag(new.F_like)); scale[scale == 0] = 1.
    eigen = np.linalg.eigvalsh(increment/scale[:, None]/scale[None, :]) if npar else np.zeros(0)
    minimum = float(np.min(eigen, initial=0.))
    if minimum < -1e-10:
        raise ValueError('negative conditional information direction')
    return {'names': list(new.names), 'observations': [current[i] for i in extra],
            'D': residual, 'C': conditional, 'F': increment,
            'reconstruction_scaled_error': error, 'minimum_normalized_eigenvalue': minimum}
