"""AP geometry at an independently verified native intrinsic spectrum table.

Validation only: the native spectra, intrinsic response and total response are
calculated without observation. The observer may be entered only in a serial
validation process, after those native outputs have been evaluated.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import re
import textwrap
from threading import Lock

import interpax
import jax
import jax.numpy as jnp
import numpy as np


def remap_frozen_table(k, mu, x, y, table, alpha_perp, alpha_para,
                       *, cross_noise=None):
    """Remap a fixed intrinsic table, with native interpolation and AP volume.

    Auto stochasticity is already in ``table``. ``cross_noise`` contains the
    separately frozen (q0, q2, k_nl, nbar_a, nbar_b) for a cross spectrum.
    Operation order matches the independently checked diagnostic.
    """
    fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
    kt = k[:, None] * fac[None, :] / alpha_perp
    mt = mu * (alpha_perp / alpha_para) / fac
    mt = jnp.broadcast_to(mt[None, :], kt.shape)
    jac = alpha_perp**2 * alpha_para
    power = interpax.interp2d(kt.ravel(), mt.ravel(), x, y, table,
                             extrap=True).reshape(kt.shape) / jac
    inside = ((kt >= x[0]) & (kt <= x[-1])
              & (mu[None, :] >= 0) & (mu[None, :] <= 1)
              & (alpha_perp > 0) & (alpha_para > 0) & jnp.isfinite(mt))
    power = jnp.where(inside, power, jnp.nan)
    if cross_noise is not None:
        q0, q2, k_nl, nbar_a, nbar_b = cross_noise
        power = power + (q0 + q2 * (kt * mt / k_nl)**2) / (
            jnp.sqrt(nbar_a * nbar_b) * jac)
    return power


# Exact dedented, decorated method source from the reviewed backend snapshot.
NATIVE_AP_SOURCE_SHA256 = 'c4121bcb8779e1464748febfd20138da44878bf23749efdd6c1016a136a4c17e'
OBSERVED_AP_SOURCE_SHA256 = '9fbc4d1989c770926ce734e38713c52b4e7f8709c01598a4bb83b6c39025ef72'
_OBSERVER_LOCK = Lock()


@contextmanager
def observe_intrinsic_table():
    """Observe tables without changing arithmetic, and always restore the method.

    Only the caller's intrinsic calculation belongs inside this context. Means,
    native totals and likelihood production must run outside it. A nonblocking
    lock rejects overlapping observers; physical jobs must also remain serial.
    """
    from ps_1loop_jax import PowerSpectrum1Loop
    if not _OBSERVER_LOCK.acquire(blocking=False):
        raise RuntimeError('intrinsic-table observation is already active')
    original = PowerSpectrum1Loop._get_pkmu_pair_ref_interp
    try:
        try:
            source = textwrap.dedent(inspect.getsource(original.__wrapped__))
        except (AttributeError, OSError, TypeError) as exc:
            raise ValueError('native AP observer source cannot be authenticated') from exc
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        if source_hash != NATIVE_AP_SOURCE_SHA256:
            raise ValueError('native AP observer source differs from reviewed method')
        line = '    pkmu_grid = self.get_pkmu_pair(self._kn, self._mu, pk_data, params, add_stochasticity=add_stochasticity)'
        addition = '    _observe_table(pkmu_grid)'
        if source.count(line) != 1:
            raise ValueError('native AP observer source insertion is ambiguous')
        observed_source = source.replace(line, line+'\n'+addition)
        if ast.dump(ast.parse(observed_source.replace('\n'+addition, ''))) != ast.dump(ast.parse(source)):
            raise ValueError('native AP observer changed calculation source')
        observations = []
        def capture(value):
            observations.append(np.asarray(value).copy())
        namespace = dict(original.__wrapped__.__globals__)
        namespace['_observe_table'] = lambda table: jax.debug.callback(capture, table, ordered=True)
        exec(compile(observed_source, '<observe-native-ap-table>', 'exec'), namespace)
        receipt = {'observations':observations, 'source_sha256':source_hash,
                   'observed_source_sha256':hashlib.sha256(observed_source.encode()).hexdigest()}
        PowerSpectrum1Loop._get_pkmu_pair_ref_interp = namespace['_get_pkmu_pair_ref_interp']
        yield receipt
    finally:
        try:
            # Drain asynchronous observations even when a calculation raises.
            jax.effects_barrier()
        finally:
            PowerSpectrum1Loop._get_pkmu_pair_ref_interp = original
            _OBSERVER_LOCK.release()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite64(value, shape, label):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.dtype('float64') or not np.isfinite(value).all():
        raise ValueError(f'{label} requires finite float64 values with shape {shape}')
    return value


def _table_geometry(ap_alphas, packed, k, mu, weights, x, y, table, noise, n_cosmo):
    # This function accepts a table, never a desired total or closure residual.
    from .revision_validation_ap import _project
    def function(v):
        ap, al = ap_alphas(v[:n_cosmo])
        power = remap_frozen_table(k, mu, x, y, table, ap, al, cross_noise=noise)
        return _project(power, mu, weights)
    return np.asarray(jax.jacfwd(function)(packed))


def measure_intrinsic_geometry(response, packed, k, *, n_cosmo, n_mu,
                               native_mean, native_total, native_intrinsic):
    """Recover and observe the full native intrinsic JVP table independently.

    All complete native inputs are preserved. The observer surrounds only a
    repeat of the intrinsic calculation; it never surrounds the mean or total.
    """
    from .one_loop import reject_legacy_ap_qualification
    reject_legacy_ap_qualification(response.ps)
    from .revision_theory import nuisance_to_backend, ap_coordinates, NUISANCE_NAMES
    from .eft_params import CROSS_STOCH_NAMES
    from .revision_validation_ap import _project
    cross = response.nbar_b is not None
    width = 30 if cross else 17
    if n_cosmo != 6 or response.space is None or len(response.space.names) != 6:
        raise ValueError('intrinsic-table validation requires the six-coordinate basis')
    p = jnp.asarray(_finite64(packed, (width,), 'packed fiducial'))
    k = jnp.asarray(_finite64(k, (len(k),), 'observed shells'))
    arrays = {'packed':np.asarray(p), 'k':np.asarray(k),
              'mean':_finite64(native_mean, (len(k), 3), 'native mean').copy(),
              'total':_finite64(native_total, (len(k), 3, width), 'native total').copy(),
              'intrinsic':_finite64(native_intrinsic, (len(k), 3, width), 'native intrinsic').copy()}
    nodes, w = np.polynomial.legendre.leggauss(n_mu)
    mu, weights = jnp.array((nodes+1)/2), jnp.array(w/2)
    x, y = response.ap_backend._kn, response.ap_backend._mu

    def split(v):
        return v[6:17], None if not cross else v[17:28], None if not cross else v[28:30]

    def inputs(v, frozen=None):
        cosmo = response._cosmology(v[:6]) if frozen is None else frozen
        s8 = response.sigma8(response.z, cosmo)
        f = response.growth_rate(response.z, cosmo)
        qa, qb, qc = split(v)
        pa = nuisance_to_backend(qa, s8, response.sigma8_ref, f,
                                 cosmo['h'], response.nbar_a, response.k_nl)
        if qb is not None:
            pb = nuisance_to_backend(qb, s8, response.sigma8_ref, f,
                                     cosmo['h'], response.nbar_b, response.k_nl)
            pa = {**pa, 'bias2':pb['bias'], 'ctr2':pb['ctr']}
        return response.plin(response.z, cosmo), pa

    def table(v):
        plin, params = inputs(v)
        return response.ps.get_pkmu_pair(x, y, plin, params, add_stochasticity=not cross)

    def intrinsic(v):
        return response.multipoles(k, v[:6], *split(v), ap=False, n_mu=n_mu)

    frozen_plin, frozen_params = inputs(p, response.reference)
    qc = split(p)[2]
    noise = None if qc is None else (qc[0], qc[1], response.k_nl, response.nbar_a, response.nbar_b)
    def original_geometry(v):
        ap, al = response.ap_alphas(v[:6])
        power = response.ap_backend._get_pkmu_pair_ref_interp(
            k, mu, ap, al, frozen_plin, frozen_params, add_stochasticity=not cross)
        if qc is not None:
            kt, mt, jac = ap_coordinates(k, mu, ap, al)
            power += (qc[0]+qc[1]*(kt*mt/response.k_nl)**2)/(
                jnp.sqrt(response.nbar_a*response.nbar_b)*jac)
        return _project(power, mu, weights)

    arrays['geometry_original'] = np.asarray(jax.jacfwd(original_geometry)(p))
    primals, tangents = jax.vmap(lambda d: jax.jvp(table, (p,), (d,)))(jnp.eye(len(p), dtype=p.dtype))
    matched = np.asarray(primals[0]).copy()
    expected_shape = (width, len(x), len(y))
    if primals.shape != expected_shape or not all(np.array_equal(np.asarray(item), matched) for item in primals):
        raise ValueError('full-width repeated intrinsic primal tables differ')
    tangent_values = _finite64(tangents, expected_shape, 'full-width table tangents')
    tangent_hash = array_digest(tangent_values)
    del primals, tangents, tangent_values
    forward = np.asarray(table(p))
    arrays['geometry_matched'] = _table_geometry(response.ap_alphas, p, k, mu, weights, x, y,
                                                jnp.asarray(matched), noise, n_cosmo)
    arrays['geometry_forward'] = _table_geometry(response.ap_alphas, p, k, mu, weights, x, y,
                                                jnp.asarray(forward), noise, n_cosmo)
    with observe_intrinsic_table() as observed:
        actual = np.asarray(jax.jacfwd(intrinsic)(p))
        jax.effects_barrier()
        observations = observed['observations']
        if not np.array_equal(actual, arrays['intrinsic']):
            raise ValueError('intrinsic observation changed the native derivative')
        if not observations or not all(np.array_equal(value, matched) for value in observations):
            raise ValueError('observed intrinsic table differs from full-width recovered table')
        arrays['observed_intrinsic'] = actual
        arrays['observed_intrinsic_table'] = observations[0].copy()
    arrays.update(table_matched=matched, table_forward=forward,
                  table_k=np.asarray(x), table_mu=np.asarray(y))
    if not all(v.dtype == np.dtype('float64') and np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('intrinsic evidence contains nonfinite or non-float64 arrays')
    if np.any(arrays['geometry_matched'][..., 6:] != 0):
        raise ValueError('frozen geometry has a nuisance derivative')
    if np.any(arrays['geometry_matched'][..., 0] != 0):
        raise ValueError('frozen geometry has an amplitude derivative')
    labels = list(response.space.names)+['a:'+name for name in NUISANCE_NAMES]
    if cross:
        labels += ['b:'+name for name in NUISANCE_NAMES]+['cross:'+name for name in CROSS_STOCH_NAMES]
    record = {'definition':'intrinsic-table-v1', 'packed_width':width, 'n_cosmo':6,
              'packed_roles':labels, 'n_mu':n_mu, 'ap_nk':len(x), 'ap_nmu':len(y),
              'redshift':response.z, 'nbar_a':response.nbar_a, 'nbar_b':response.nbar_b,
              'k_nl':response.k_nl, 'reference_cosmology':dict(response.reference),
              'tangent_shape':list(expected_shape), 'tangent_sha256':tangent_hash,
              'repeated_primal_count':width, 'repeated_primals_equal':True, 'tangents_finite':True,
              'table_sha256':array_digest(matched),
              'observation':{'outputs_bitwise_equal':True, 'table_bitwise_equal':True,
                  'table_observations':len(observations), 'source_sha256':observed['source_sha256'],
                  'observed_source_sha256':observed['observed_source_sha256'],
                  'scope':'intrinsic-only; native mean and total unobserved'}}
    return record, arrays


INTRINSIC_ARRAYS = frozenset((
    'packed', 'k', 'mean', 'total', 'intrinsic', 'geometry_original',
    'geometry_matched', 'geometry_forward', 'observed_intrinsic',
    'observed_intrinsic_table', 'table_matched', 'table_forward', 'table_k',
    'table_mu', 'pair_covariance'))


def packed_labels(likelihood, cell, pair):
    """Native pair-local ordering, independently mapped to assembled columns."""
    from .cosmology_space import make_cosmology_spec
    from .revision_theory import NUISANCE_NAMES
    from .eft_params import CROSS_STOCH_NAMES
    a, b = pair
    ibin = cell.metadata['nominal_bin_index']
    labels = list(make_cosmology_spec(likelihood.reference_cosmology, extended=True).names)
    labels += [f'bin{ibin:02d}:{a}:{n}' for n in NUISANCE_NAMES]
    if a != b:
        labels += [f'bin{ibin:02d}:{b}:{n}' for n in NUISANCE_NAMES]
        labels += [f'bin{ibin:02d}:{a}|{b}:{n}' for n in CROSS_STOCH_NAMES]
    if (len(set(cell.names)) != len(cell.names) or len(set(likelihood.names)) != len(likelihood.names)
            or any(name not in cell.names or name not in likelihood.names for name in labels)):
        raise ValueError('native packed labels absent from assembled likelihood')
    return labels


def _same(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise ValueError(f'{label} differs from independently assembled evidence')


def validate_intrinsic_geometry(record, arrays, *, likelihood, cell, pair):
    """Bind every full native vector/table to its actual likelihood context."""
    from . import revision_theory as theory
    from .cosmology_space import make_cosmology_spec
    if not isinstance(arrays, dict) or set(arrays) != INTRINSIC_ARRAYS:
        raise ValueError('intrinsic array inventory is incomplete or contains extra arrays')
    a, b = pair
    width = 17 if a == b else 30
    n = len(likelihood.grid.centers)
    nk, nm = likelihood.config.ap_interpolation_nk, likelihood.config.ap_interpolation_nmu
    shapes = {'packed':(width,), 'k':(n,), 'mean':(n, 3), 'table_k':(nk,),
              'table_mu':(nm,), 'pair_covariance':(n, 3, 3)}
    shapes.update({name:(nk, nm) for name in ('table_matched', 'table_forward', 'observed_intrinsic_table')})
    shapes.update({name:(n, 3, width) for name in (
        'total', 'intrinsic', 'geometry_original', 'geometry_matched', 'geometry_forward', 'observed_intrinsic')})
    for name, shape in shapes.items():
        _finite64(arrays[name], shape, name)
    labels = packed_labels(likelihood, cell, pair)
    if record.get('packed_labels') != labels:
        raise ValueError('native packed labels or ordering differ')
    roles = list(make_cosmology_spec(likelihood.reference_cosmology, extended=True).names)
    roles += ['a:'+name for name in theory.NUISANCE_NAMES]
    if a != b:
        from .eft_params import CROSS_STOCH_NAMES
        roles += ['b:'+name for name in theory.NUISANCE_NAMES]+['cross:'+name for name in CROSS_STOCH_NAMES]
    expected = {'definition':'intrinsic-table-v1', 'cell_id':cell.metadata['cell_id'],
        'pair':list(pair), 'packed_width':width, 'n_cosmo':6, 'packed_roles':roles,
        'n_mu':likelihood.config.n_mu, 'ap_nk':nk, 'ap_nmu':nm,
        'redshift':cell.metadata['model_redshift'], 'nbar_a':cell.metadata['nbar'][a],
        'nbar_b':None if a == b else cell.metadata['nbar'][b], 'k_nl':likelihood.config.k_nl,
        'reference_cosmology':dict(likelihood.reference_cosmology), 'tangent_shape':[width, nk, nm],
        'repeated_primal_count':width, 'repeated_primals_equal':True, 'tangents_finite':True}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('intrinsic context, full tangent coverage or physical identity differs')
    if not re.fullmatch('[0-9a-f]{64}', str(record.get('tangent_sha256', ''))):
        raise ValueError('full-width table tangent digest is missing')
    _same(arrays['k'], likelihood.grid.centers, 'observed shell order')
    _same(arrays['packed'], np.asarray(likelihood.fiducial)[[likelihood.names.index(n) for n in labels]],
          'native fiducial')
    pairs = [tuple(p) for p in cell.metadata['retained_pairs']]
    if pairs.count(tuple(pair)) != 1:
        raise ValueError('intrinsic context pair is omitted or duplicated')
    i = pairs.index(tuple(pair))
    sl = slice(3*i, 3*i+3)
    _same(arrays['mean'], np.asarray(cell.fiducial_mean)[:, sl], 'native mean')
    _same(arrays['total'], np.asarray(cell.derivatives)[:, sl, :][:, :, [cell.names.index(n) for n in labels]],
          'native total')
    _same(arrays['pair_covariance'], np.asarray(cell.covariance)[:, sl, sl], 'pair covariance identity')
    _same(arrays['table_k'], np.asarray(jnp.geomspace(.002, .5, nk)), 'intrinsic table k grid')
    _same(arrays['table_mu'], np.asarray(jnp.linspace(0., 1., nm)), 'intrinsic table mu grid')
    obs = record.get('observation', {})
    if (obs.get('outputs_bitwise_equal') is not True or obs.get('table_bitwise_equal') is not True
            or type(obs.get('table_observations')) is not int or obs['table_observations'] < 1
            or obs.get('source_sha256') != NATIVE_AP_SOURCE_SHA256
            or obs.get('observed_source_sha256') != OBSERVED_AP_SOURCE_SHA256
            or obs.get('scope') != 'intrinsic-only; native mean and total unobserved'):
        raise ValueError('intrinsic observation receipt is incomplete or unverified')
    if record.get('table_sha256') != array_digest(arrays['table_matched']):
        raise ValueError('differentiated intrinsic table digest differs')
    _same(arrays['observed_intrinsic_table'], arrays['table_matched'], 'observed intrinsic table')
    _same(arrays['observed_intrinsic'], arrays['intrinsic'], 'intrinsic observation output')
    if np.any(arrays['geometry_matched'][..., 6:] != 0):
        raise ValueError('frozen geometry has a nuisance derivative')
    if np.any(arrays['geometry_matched'][..., 0] != 0):
        raise ValueError('frozen geometry has an amplitude derivative')
    # Re-evaluate only background geometry, with no emulator or loop initialization.
    response = object.__new__(theory.PairResponse)
    response.space = make_cosmology_spec(likelihood.reference_cosmology, extended=True)
    response.reference = dict(likelihood.reference_cosmology)
    response.z = float(cell.metadata['model_redshift'])
    ref = response.reference
    response.da_ref = theory.bg.angular_diameter_distance(ref['omega_b'], ref['omega_cdm'], ref['h'], response.z, ref['mnu'])
    response.hubble_ref = theory.bg.Hz(ref['omega_b'], ref['omega_cdm'], ref['h'], response.z, ref['mnu'])
    nodes, weights = np.polynomial.legendre.leggauss(likelihood.config.n_mu)
    p, k = jnp.asarray(arrays['packed']), jnp.asarray(arrays['k'])
    noise = None if a == b else (p[28], p[29], record['k_nl'], record['nbar_a'], record['nbar_b'])
    for kind in ('matched', 'forward'):
        calculated = _table_geometry(response.ap_alphas, p, k, jnp.asarray((nodes+1)/2),
            jnp.asarray(weights/2), jnp.asarray(arrays['table_k']), jnp.asarray(arrays['table_mu']),
            jnp.asarray(arrays['table_'+kind]), noise, 6)
        if not np.allclose(arrays['geometry_'+kind], calculated, rtol=1e-10, atol=1e-12):
            raise ValueError(f'{kind} geometry differs from independent saved-table remapping')
    from .revision_validation_ap import _project
    mu, weights = jnp.asarray((nodes+1)/2), jnp.asarray(weights/2)
    power = remap_frozen_table(k, mu, jnp.asarray(arrays['table_k']), jnp.asarray(arrays['table_mu']),
        jnp.asarray(arrays['table_forward']), *response.ap_alphas(p[:6]), cross_noise=noise)
    if not np.allclose(np.asarray(_project(power, mu, weights)), arrays['mean'], rtol=1e-8, atol=1e-12):
        raise ValueError('forward table mean differs from native fiducial spectrum')


def _sidecar_path(root, relative):
    if not isinstance(relative, str) or not re.fullmatch(r'intrinsic/cell[0-9]{3,}-pair[0-9]{3,}\.npz', relative):
        raise ValueError('unsafe or noncanonical intrinsic sidecar path')
    root = Path(root)
    path = root/relative
    if any(part.is_symlink() for part in (path, path.parent)) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe intrinsic sidecar symlink or path')
    return path


def _array_inventory(arrays):
    if set(arrays) != INTRINSIC_ARRAYS:
        raise ValueError('intrinsic sidecar array inventory differs')
    result = []
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype('float64') or not np.isfinite(value).all():
            raise ValueError('intrinsic sidecar arrays require finite float64 values')
        result.append({'name':name, 'shape':list(value.shape), 'dtype':str(value.dtype), 'sha256':array_digest(value)})
    return result


def write_intrinsic_sidecar(root, relative_path, record, arrays):
    """Write one immutable pair's complete raw evidence; return JSON metadata."""
    inventory = _array_inventory(arrays)
    path = _sidecar_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez(stream, **{name:arrays[name] for name in sorted(arrays)})
    return {'cell_id':record['cell_id'], 'pair':record['pair'], 'path':relative_path,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'record':record, 'arrays':inventory}


def load_intrinsic_sidecar(root, entry):
    if not isinstance(entry, dict) or set(entry) != {'cell_id','pair','path','sha256','record','arrays'}:
        raise ValueError('intrinsic sidecar declaration is incomplete')
    path = _sidecar_path(root, entry['path'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('intrinsic sidecar is missing or its bytes changed')
    with np.load(path, allow_pickle=False) as saved:
        if saved.files != sorted(INTRINSIC_ARRAYS):
            raise ValueError('intrinsic sidecar array inventory or order differs')
        arrays = {name:saved[name] for name in saved.files}
    if _array_inventory(arrays) != entry['arrays']:
        raise ValueError('intrinsic sidecar content inventory differs')
    if entry['cell_id'] != entry['record'].get('cell_id') or entry['pair'] != entry['record'].get('pair'):
        raise ValueError('intrinsic sidecar context differs from its record')
    return arrays


def validate_intrinsic_evidence(report, likelihood, *, root):
    """Require one ordered sidecar per retained pair, with original AP rows."""
    expected = [(i, cell, j, tuple(pair)) for i, cell in enumerate(likelihood.cells)
                for j, pair in enumerate(cell.metadata['retained_pairs'])]
    entries = report.get('intrinsic_geometry')
    if (root is None or not isinstance(entries, list) or len(entries) != len(expected)
            or not isinstance(report.get('points'), list)):
        raise ValueError('intrinsic-table evidence coverage or sidecar root is missing')
    paths = [f'intrinsic/cell{i:03d}-pair{j:03d}.npz' for i, cell, j, pair in expected]
    for entry, path, (_, cell, _, pair) in zip(entries, paths, expected):
        if (not isinstance(entry, dict) or entry.get('path') != path
                or entry.get('cell_id') != cell.metadata['cell_id'] or entry.get('pair') != list(pair)):
            raise ValueError('intrinsic-table pair evidence is omitted, duplicated, substituted or reordered')
    directory = Path(root)/'intrinsic'
    actual = {str(path.relative_to(root)) for path in directory.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != set(paths):
        raise ValueError('intrinsic sidecar file inventory is incomplete or has extra files')
    points = report['points']
    expected_keys = [(cell.metadata['cell_id'], pair, float(k))
                     for _, cell, _, pair in expected for k in likelihood.grid.centers]
    actual_keys = [(r.get('cell_id'), tuple(r.get('pair', ())), r.get('k')) for r in points]
    if actual_keys != expected_keys:
        raise ValueError('intrinsic point evidence is incomplete or reordered')
    for entry, (_, cell, _, pair) in zip(entries, expected):
        arrays = load_intrinsic_sidecar(root, entry)
        validate_intrinsic_geometry(entry['record'], arrays, likelihood=likelihood, cell=cell, pair=pair)
        rows = [r for r in points if r['cell_id'] == cell.metadata['cell_id'] and tuple(r['pair']) == pair]
        for i, row in enumerate(rows):
            if row.get('decomposition') != 'intrinsic-table-v1':
                raise ValueError('AP row decomposition definition differs')
            for row_key, array_key in (
                    ('fiducial_multipoles','mean'), ('pair_covariance','pair_covariance'),
                    ('total_ad','total'), ('intrinsic_ad','intrinsic'),
                    ('geometry_ad_original','geometry_original'), ('geometry_ad','geometry_matched')):
                value = arrays[array_key][i]
                if array_key not in ('mean','pair_covariance'):
                    value = value[:, :6]
                _same(np.asarray(row.get(row_key)), value, 'AP row '+row_key)
            chol = np.linalg.cholesky(arrays['pair_covariance'][i])
            def norm(v): return float(np.linalg.norm(np.linalg.solve(chol, v)))
            scale = norm(arrays['mean'][i])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('intrinsic original closure has invalid physical scale')
            original_errors = []
            for j in range(6):
                total = arrays['total'][i, :, j]
                delta = total-arrays['intrinsic'][i, :, j]-arrays['geometry_original'][i, :, j]
                denominator = norm(total)
                original_errors.append(norm(delta)/(scale if denominator <= 1e-12*scale else denominator))
            saved = _finite64(np.asarray(row.get('chain_errors_original')), (6,), 'original closure')
            # This checks faithful storage of the original diagnostic, not its
            # scientific acceptance. An absolute floor could erase that result.
            if not np.allclose(saved, original_errors, rtol=1e-12, atol=0.):
                raise ValueError('original closure residual differs from raw original geometry')
    return {'definition':'intrinsic-table-v1', 'pairs':len(expected), 'points':len(expected_keys),
            'native_widths':[17 if pair[0] == pair[1] else 30 for _, _, _, pair in expected],
            'status':'passed'}
