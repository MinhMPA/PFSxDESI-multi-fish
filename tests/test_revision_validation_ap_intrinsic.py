"""Independent AP geometry and transparent intrinsic-table observation controls."""
import importlib
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def api():
    name = 'pfsfog.revision_validation_ap_intrinsic'
    assert importlib.util.find_spec(name), 'intrinsic-table validation module is missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('cross', [False, True], ids=['auto', 'cross'])
def test_finite_ap_and_identity_partials_from_cartesian_dilation(cross):
    # Removing volume, exchanging distances, or counting cross noise twice fails.
    remap = api().remap_frozen_table
    x, y = jnp.array([.002, .04, .08, .15, .3, .5]), jnp.array([0., .1, .4, .6, 1.])
    k, mu = jnp.array([.071, .113]), jnp.array([.19, .61, .73])
    amplitude, beta = (2., .6) if cross else (1., .4)
    table = amplitude*x[:, None]*(1+beta*y[None, :])
    noise = (.07, 3., .7, 2., 8.) if cross else None

    def actual(alphas):
        return remap(k, mu, x, y, table, *alphas, cross_noise=noise)

    ap, al = 1.03, .97
    kk, mm = np.asarray(k)[:, None], np.asarray(mu)[None, :]
    transverse, radial = kk*np.sqrt(1-mm**2)/ap, kk*mm/al
    expected = amplitude*(np.hypot(transverse, radial)+beta*radial)
    if cross:
        expected += (.07+3*(radial/.7)**2)/4
    expected /= ap**2*al
    np.testing.assert_allclose(actual(jnp.array([ap, al])), expected, rtol=2e-13, atol=2e-15)
    power = amplitude*kk*(1+beta*mm)
    perp = -amplitude*kk*(1-mm**2)-2*power
    para = -amplitude*kk*(mm**2+beta*mm)-power
    if cross:
        n2 = 3*(kk*mm/.7)**2
        perp += -2*(.07+n2)/4
        para += -(.07+3*n2)/4
    np.testing.assert_allclose(jax.jacfwd(actual)(jnp.ones(2)),
                               np.stack((perp, para), axis=-1), rtol=2e-13, atol=2e-15)


def test_remap_rejects_outside_support_and_nonpositive_distance():
    remap = api().remap_frozen_table
    x, y = jnp.linspace(.002, .5, 6), jnp.linspace(0., 1., 5)
    table = jnp.ones((6, 5))
    assert np.isnan(remap(jnp.array([.001, .6]), jnp.array([.5]), x, y, table, 1., 1.)).all()
    assert np.isnan(remap(jnp.array([.1]), jnp.array([-.1, 1.1]), x, y, table, 1., 1.)).all()
    assert np.isnan(remap(jnp.array([.1]), jnp.array([.5]), x, y, table, -1., 1.)).all()


def test_intrinsic_observer_restores_after_success_and_exception():
    observer = api().observe_intrinsic_table
    from ps_1loop_jax import PowerSpectrum1Loop
    original = PowerSpectrum1Loop._get_pkmu_pair_ref_interp
    with observer() as receipt:
        assert receipt['observations'] == []
        assert PowerSpectrum1Loop._get_pkmu_pair_ref_interp is not original
    assert PowerSpectrum1Loop._get_pkmu_pair_ref_interp is original
    with pytest.raises(RuntimeError, match='deliberate'):
        with observer():
            raise RuntimeError('deliberate calculation failure')
    assert PowerSpectrum1Loop._get_pkmu_pair_ref_interp is original


def test_intrinsic_observer_rejects_nested_and_concurrent_use():
    observer = api().observe_intrinsic_table
    def nested():
        with pytest.raises(RuntimeError, match='already active'):
            with observer():
                pass
    with observer():
        nested()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(nested).result(timeout=10)


def test_intrinsic_observer_rejects_changed_backend_method(monkeypatch):
    observer = api().observe_intrinsic_table
    from ps_1loop_jax import PowerSpectrum1Loop
    def changed(*args, **kwargs):
        return None
    monkeypatch.setattr(PowerSpectrum1Loop, '_get_pkmu_pair_ref_interp', changed)
    with pytest.raises(ValueError, match='source'):
        with observer():
            pass
    assert PowerSpectrum1Loop._get_pkmu_pair_ref_interp is changed


def test_observer_captures_actual_intrinsic_table_without_changing_derivative():
    observer = api().observe_intrinsic_table
    from pfsfog.revision_theory import APInterpolationBackend
    class Spectrum:
        def get_pkmu_pair(self, k, mu, pk, params, add_stochasticity=False):
            return pk*k[:, None]*(1+.6*mu[None, :])
    backend = APInterpolationBackend(Spectrum(), 8, 5)
    def intrinsic(p):
        return backend._get_pkmu_pair_ref_interp(jnp.array([.071, .113]),
            jnp.array([.19, .61]), 1., 1., p[0]+2*p[1], {}, add_stochasticity=True)
    p = jnp.array([1., 3.])
    unobserved = np.asarray(jax.jacfwd(intrinsic)(p))
    with observer() as receipt:
        observed = np.asarray(jax.jacfwd(intrinsic)(p))
    np.testing.assert_array_equal(observed, unobserved)
    assert len(receipt['observations']) >= 1
    expected = 7*np.asarray(backend._kn)[:, None]*(1+.6*np.asarray(backend._mu)[None, :])
    for table in receipt['observations']:
        np.testing.assert_array_equal(table, expected)


def analytic_pair(monkeypatch, cross):
    """Keep real background/AP/nuisance coordinates; control only expensive leaves."""
    from pfsfog import revision_theory as theory
    from pfsfog.cosmo import FIDUCIAL
    from pfsfog.cosmology_space import make_cosmology_spec
    monkeypatch.setattr(theory, 'make_reference_plin_func', lambda h_ref: lambda z, p:
        jnp.exp(.1*p['ln10_10_As']+.2*p['mnu']+.3*p['Omega_m']+
                .4*p['h']+5*p['omega_b']+.6*p['n_s']))
    monkeypatch.setattr(theory, 'make_sigma8_func', lambda: lambda z, p:
        .8*jnp.exp(.02*(p['ln10_10_As']-FIDUCIAL['ln10_10_As'])+.01*(p['h']-FIDUCIAL['h'])))
    monkeypatch.setattr(theory, 'make_growth_rate_func', lambda: lambda z, p: .6+.1*p['Omega_m'])
    class Spectrum:
        def get_pkmu_pair(self, k, mu, pk, params, add_stochasticity=False):
            coefficient = sum(params['bias'].values())+sum(params['ctr'].values())
            if 'bias2' in params:
                coefficient += 2*sum(params['bias2'].values())+3*sum(params['ctr2'].values())
            if add_stochasticity:
                coefficient += sum(params['stoch'].values())
            return pk*coefficient*k[:, None]*(1+.6*jnp.abs(mu)[None, :])
    space = make_cosmology_spec(FIDUCIAL, extended=True, ap_nk=16384)
    response = theory.make_pair_response(Spectrum(), .9, FIDUCIAL, 1., 2. if cross else None,
                                         .7, ap_nk=8, ap_nmu=5, space=space)
    theta = [FIDUCIAL['ln10_10_As'], FIDUCIAL['mnu'], FIDUCIAL['Omega_m'],
             FIDUCIAL['h'], FIDUCIAL['omega_b'], FIDUCIAL['n_s']]
    qa = np.arange(1, 12)/20
    packed = jnp.asarray(np.r_[theta, qa, qa*1.2, [.07, .3]] if cross else np.r_[theta, qa])
    k = jnp.array([.071, .113])
    def mean(p, ap=True):
        return response.multipoles(k, p[:6], p[6:17], p[17:28] if cross else None,
                                   p[28:30] if cross else None, ap=ap, n_mu=8)
    native = {'native_mean':np.asarray(mean(packed)),
              'native_total':np.asarray(jax.jacfwd(mean)(packed)),
              'native_intrinsic':np.asarray(jax.jacfwd(lambda p: mean(p, ap=False))(packed))}
    return response, packed, k, native


@pytest.mark.parametrize('cross', [False, True], ids=['auto17', 'cross30'])
def test_full_width_recovery_observes_native_table_and_preserves_native_outputs(monkeypatch, cross):
    module = api()
    assert hasattr(module, 'measure_intrinsic_geometry'), 'full-width table recovery is missing'
    response, packed, k, native = analytic_pair(monkeypatch, cross)
    record, arrays = module.measure_intrinsic_geometry(response, packed, k, n_cosmo=6, n_mu=8, **native)
    width = 30 if cross else 17
    assert record['packed_width'] == width
    assert record['tangent_shape'] == [width, 8, 5]
    assert record['repeated_primal_count'] == width
    assert record['observation']['table_observations'] >= 1
    for key, old in [('mean', 'native_mean'), ('total', 'native_total'), ('intrinsic', 'native_intrinsic')]:
        np.testing.assert_array_equal(arrays[key], native[old])
    np.testing.assert_array_equal(arrays['observed_intrinsic'], native['native_intrinsic'])
    np.testing.assert_array_equal(arrays['table_matched'], arrays['observed_intrinsic_table'])
    assert arrays['geometry_matched'].shape == (2, 3, width)
    assert not np.any(arrays['geometry_matched'][..., 6:])
    assert not np.any(arrays['geometry_matched'][..., 0])
    np.testing.assert_allclose(arrays['total'], arrays['intrinsic']+arrays['geometry_matched'],
                               rtol=1e-10, atol=1e-12)
    assert arrays['geometry_original'].shape == (2, 3, width)
    assert arrays['geometry_forward'].shape == (2, 3, width)


def test_recovery_rejects_shortened_native_tangent_input(monkeypatch):
    module = api()
    assert hasattr(module, 'measure_intrinsic_geometry'), 'full-width table recovery is missing'
    response, packed, k, native = analytic_pair(monkeypatch, False)
    native['native_total'] = native['native_total'][..., :6]
    with pytest.raises(ValueError, match='total.*shape'):
        module.measure_intrinsic_geometry(response, packed, k, n_cosmo=6, n_mu=8, **native)


@pytest.fixture
def bound_cross(monkeypatch):
    module = api()
    response, packed, k, native = analytic_pair(monkeypatch, True)
    record, arrays = module.measure_intrinsic_geometry(response, packed, k, n_cosmo=6, n_mu=8, **native)
    from pfsfog.revision_theory import NUISANCE_NAMES
    from pfsfog.eft_params import CROSS_STOCH_NAMES
    arrays = {name:value.copy() for name, value in arrays.items()}
    pair = ('DESI-LRG', 'PFS-ELG')
    labels = list(response.space.names)+[f'bin02:{tracer}:{name}' for tracer in pair for name in NUISANCE_NAMES]
    labels += [f'bin02:{pair[0]}|{pair[1]}:{name}' for name in CROSS_STOCH_NAMES]
    cell = SimpleNamespace(names=labels, metadata={'cell_id':'overlap-z09',
        'retained_pairs':[list(pair)], 'model_redshift':.9, 'nominal_bin_index':2,
        'nbar':{pair[0]:1., pair[1]:2.}}, fiducial_mean=native['native_mean'].copy(),
        derivatives=native['native_total'].copy(), covariance=np.repeat(np.eye(3)[None], 2, axis=0))
    # A small table is sufficient for this independent validation test. Real
    # production profile selection is separately constrained to the declared grids.
    config = SimpleNamespace(ap_interpolation_nk=8, ap_interpolation_nmu=5, n_mu=8, k_nl=.7)
    like = SimpleNamespace(config=config, reference_cosmology=dict(response.reference),
        names=labels, fiducial=np.asarray(packed).copy(), cells=[cell],
        grid=SimpleNamespace(centers=np.asarray(k)))
    record.update(cell_id=cell.metadata['cell_id'], pair=list(pair), packed_labels=labels)
    arrays['pair_covariance'] = cell.covariance.copy()
    return record, arrays, like, cell, pair


def test_full_native_and_observed_table_binding_accepts_measured_cross(bound_cross):
    record, arrays, like, cell, pair = bound_cross
    assert hasattr(api(), 'validate_intrinsic_geometry'), 'intrinsic evidence binding is missing'
    api().validate_intrinsic_geometry(record, arrays, likelihood=like, cell=cell, pair=pair)


@pytest.mark.parametrize('mutation,match', [
    ('native_total','native total'), ('short_width','shape'), ('shells','shell'),
    ('fiducial','fiducial'), ('observation','observation'), ('table','table'),
    ('absent_table','inventory'), ('nuisance','nuisance'), ('amplitude','amplitude'),
    ('positive_covariance','covariance'), ('incomplete_covariance','shape'),
    ('context','context'), ('labels','labels'), ('forged_geometry','geometry'),
    ('forged_table','geometry'), ('forged_mean','table mean'),
])
def test_rehashed_but_inconsistent_intrinsic_evidence_is_rejected(bound_cross, mutation, match):
    assert hasattr(api(), 'validate_intrinsic_geometry'), 'intrinsic evidence binding is missing'
    record, arrays, like, cell, pair = bound_cross
    if mutation == 'native_total':
        arrays['total'][0,0,3] = np.nextafter(arrays['total'][0,0,3], np.inf)
    elif mutation == 'short_width': arrays['total'] = arrays['total'][...,:6]
    elif mutation == 'shells': arrays['k'] = arrays['k'][::-1].copy()
    elif mutation == 'fiducial': arrays['packed'] = arrays['packed'].copy(); arrays['packed'][3] += .001
    elif mutation == 'observation': record['observation']['outputs_bitwise_equal'] = False
    elif mutation == 'table': arrays['table_matched'][0,0] += 1.
    elif mutation == 'absent_table': arrays.pop('observed_intrinsic_table')
    elif mutation == 'nuisance': arrays['geometry_matched'][0,0,28] = .01
    elif mutation == 'amplitude': arrays['geometry_matched'][0,0,0] = .01
    elif mutation == 'positive_covariance':
        arrays['pair_covariance'][0,0,0] = np.nextafter(arrays['pair_covariance'][0,0,0], np.inf)
        np.linalg.cholesky(arrays['pair_covariance'])
    elif mutation == 'incomplete_covariance': arrays['pair_covariance'] = arrays['pair_covariance'][:1]
    elif mutation == 'context': record['nbar_b'] = 3.
    elif mutation == 'labels': record['packed_labels'] = record['packed_labels'][::-1]
    elif mutation == 'forged_geometry': arrays['geometry_matched'][0,0,3] += 1.
    elif mutation == 'forged_mean':
        arrays['mean'][0,0] += 1.; cell.fiducial_mean[0,0] += 1.
    elif mutation == 'forged_table':
        arrays['table_matched'] = arrays['table_matched']*1.01
        arrays['observed_intrinsic_table'] = arrays['table_matched'].copy()
        record['table_sha256'] = hashlib.sha256(arrays['table_matched'].tobytes()).hexdigest()
    with pytest.raises(ValueError, match=match):
        api().validate_intrinsic_geometry(record, arrays, likelihood=like, cell=cell, pair=pair)


def test_sidecar_roundtrip_is_exact_and_cannot_be_overwritten(bound_cross, tmp_path):
    module = api()
    assert hasattr(module, 'write_intrinsic_sidecar'), 'intrinsic sidecars are missing'
    record, arrays, *_ = bound_cross
    entry = module.write_intrinsic_sidecar(tmp_path, 'intrinsic/cell000-pair000.npz', record, arrays)
    restored = module.load_intrinsic_sidecar(tmp_path, entry)
    assert set(restored) == set(arrays)
    for name in arrays:
        np.testing.assert_array_equal(restored[name], arrays[name])
    with pytest.raises(FileExistsError):
        module.write_intrinsic_sidecar(tmp_path, entry['path'], record, arrays)


@pytest.mark.parametrize('mutation', ['unsafe','absolute','symlink','hash','missing','extra','duplicate','reordered','dtype'])
def test_sidecar_loader_rejects_changed_or_unsafe_evidence(bound_cross, tmp_path, mutation):
    module = api()
    assert hasattr(module, 'write_intrinsic_sidecar'), 'intrinsic sidecars are missing'
    record, arrays, *_ = bound_cross
    entry = module.write_intrinsic_sidecar(tmp_path, 'intrinsic/cell000-pair000.npz', record, arrays)
    path = tmp_path/entry['path']
    if mutation == 'unsafe': entry['path'] = '../outside.npz'
    elif mutation == 'absolute': entry['path'] = str(path)
    elif mutation == 'symlink':
        old = tmp_path/'original.npz'; path.rename(old); path.symlink_to(old)
    elif mutation == 'hash': entry['sha256'] = '0'*64
    elif mutation == 'missing': entry['arrays'] = entry['arrays'][:-1]
    elif mutation == 'extra': entry['arrays'].append(copy.deepcopy(entry['arrays'][0]) | {'name':'extra'})
    elif mutation == 'duplicate': entry['arrays'].append(copy.deepcopy(entry['arrays'][0]))
    elif mutation == 'reordered': entry['arrays'] = entry['arrays'][::-1]
    elif mutation == 'dtype': entry['arrays'][0]['dtype'] = 'float32'
    with pytest.raises(ValueError): module.load_intrinsic_sidecar(tmp_path, entry)


def sidecar_report(bound, root):
    record, arrays, like, cell, pair = bound
    entry = api().write_intrinsic_sidecar(root, 'intrinsic/cell000-pair000.npz', record, arrays)
    rows = []
    for i, k in enumerate(arrays['k']):
        total, intrinsic, original = (arrays[name][i,:,:6] for name in ('total','intrinsic','geometry_original'))
        chol = np.linalg.cholesky(arrays['pair_covariance'][i])
        def norm(v): return np.linalg.norm(np.linalg.solve(chol, v))
        scale = norm(arrays['mean'][i])
        errors = [norm(total[:,j]-intrinsic[:,j]-original[:,j]) /
                  (norm(total[:,j]) if norm(total[:,j]) > 1e-12*scale else scale) for j in range(6)]
        rows.append({'cell_id':cell.metadata['cell_id'], 'pair':list(pair), 'k':float(k),
            'decomposition':'intrinsic-table-v1', 'fiducial_multipoles':arrays['mean'][i].tolist(),
            'total_ad':total.tolist(), 'intrinsic_ad':intrinsic.tolist(),
            'geometry_ad_original':original.tolist(), 'geometry_ad':arrays['geometry_matched'][i,:,:6].tolist(),
            'pair_covariance':arrays['pair_covariance'][i].tolist(), 'chain_errors_original':errors})
    return {'intrinsic_geometry':[entry], 'points':rows}, like


def test_sidecar_cohort_binds_all_rows_and_original_residuals(bound_cross, tmp_path):
    assert hasattr(api(), 'validate_intrinsic_evidence'), 'complete intrinsic evidence binding is missing'
    report, like = sidecar_report(bound_cross, tmp_path)
    assert api().validate_intrinsic_evidence(report, like, root=tmp_path)['pairs'] == 1


@pytest.mark.parametrize('residual,replacement', [(3e-11, 0.), (1.008e-10, 1e-12)])
def test_original_small_closure_discrepancy_cannot_be_erased(bound_cross, tmp_path, residual, replacement):
    # Preserve a discrepancy of the size measured in the physical auto/cross
    # controls. A serialization check must not hide it behind the closure gate.
    _, arrays, *_ = bound_cross
    total = arrays['total'][..., 2]
    arrays['geometry_original'][..., 2] = total-arrays['intrinsic'][..., 2]-residual*total
    report, like = sidecar_report(bound_cross, tmp_path)
    actual = report['points'][0]['chain_errors_original'][2]
    assert np.isclose(actual, residual, rtol=1e-4, atol=0.)
    api().validate_intrinsic_evidence(report, like, root=tmp_path)
    report['points'][0]['chain_errors_original'][2] = replacement
    with pytest.raises(ValueError, match='original closure residual'):
        api().validate_intrinsic_evidence(report, like, root=tmp_path)


@pytest.mark.parametrize('mutation', ['missing','duplicate','other_pair','order','extra_file',
    'row_total','row_intrinsic','row_original','original_residual','row_geometry','row_covariance'])
def test_cohort_rejects_missing_substituted_or_reordered_evidence(bound_cross, tmp_path, mutation):
    assert hasattr(api(), 'validate_intrinsic_evidence'), 'complete intrinsic evidence binding is missing'
    report, like = sidecar_report(bound_cross, tmp_path)
    if mutation == 'missing': report['intrinsic_geometry'] = []
    elif mutation == 'duplicate': report['intrinsic_geometry'] *= 2
    elif mutation == 'other_pair': report['intrinsic_geometry'][0]['pair'] = ['PFS-ELG','PFS-ELG']
    elif mutation == 'order': report['points'].reverse()
    elif mutation == 'extra_file': (tmp_path/'intrinsic/extra.npz').write_bytes(b'extra')
    else:
        field = {'row_total':'total_ad','row_intrinsic':'intrinsic_ad','row_original':'geometry_ad_original',
                 'original_residual':'chain_errors_original','row_geometry':'geometry_ad',
                 'row_covariance':'pair_covariance'}[mutation]
        report['points'][0][field] = (np.asarray(report['points'][0][field])+1.).tolist()
    with pytest.raises(ValueError): api().validate_intrinsic_evidence(report, like, root=tmp_path)
