"""Complete new-profile artifact transport and hostile saved-evidence changes.

The supplied matrices below are a synthetic, well-conditioned local likelihood.
They test serialization and numerical binding, not physical forecast accuracy;
the real full17/30 observer experiments are separately authenticated evidence.
"""
import copy
from dataclasses import replace
import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from test_extended_artifacts import fixture, digest
from test_extended_validation import _row, _growth


def intrinsic_fixture(monkeypatch, root):
    from pfsfog import revision_theory as theory
    from pfsfog import revision_validation_ap_intrinsic as intrinsic
    from pfsfog.ap_numerics import profile_fields
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.cosmology_projection import apply_cosmology_treatment, COSMOLOGY_TREATMENTS
    from pfsfog.revision_validation_ap import _project, ap_cell_record
    import test_extended_validation as helpers
    like, _, provenance, evidence = fixture(monkeypatch)
    like.config = replace(like.config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    space = make_cosmology_spec(like.reference_cosmology,extended=True,ap_nk=16384)
    monkeypatch.setattr(helpers,'STEPS',tuple(space.fd_steps.values()))
    response = object.__new__(theory.PairResponse)
    response.reference = dict(like.reference_cosmology); response.space = space; response.z = 1.1
    ref = response.reference
    response.da_ref = theory.bg.angular_diameter_distance(ref['omega_b'],ref['omega_cdm'],ref['h'],1.1,ref['mnu'])
    response.hubble_ref = theory.bg.Hz(ref['omega_b'],ref['omega_cdm'],ref['h'],1.1,ref['mnu'])
    x,y = jnp.geomspace(.002,.5,16384),jnp.linspace(0.,1.,101)
    table = jnp.ones((16384,101)); k = jnp.asarray(like.grid.centers)
    nodes,w = np.polynomial.legendre.leggauss(32); mu,weights = jnp.asarray((nodes+1)/2),jnp.asarray(w/2)
    packed = np.r_[like.fiducial,np.zeros(11)]
    geometry = intrinsic._table_geometry(response.ap_alphas,jnp.asarray(packed),k,mu,weights,x,y,table,None,6)
    mean = np.asarray(_project(intrinsic.remap_frozen_table(k,mu,x,y,table,*response.ap_alphas(packed[:6])),mu,weights))
    old_cosmology = like.parameters.copy()
    like.names = list(space.names)+[f'bin{i:02d}:DESI-ELG:{n}' for i in range(3) for n in theory.NUISANCE_NAMES]
    like.fiducial = np.r_[like.fiducial,np.zeros(33)]
    like.parameters = old_cosmology+[{'name':n,'role':'nuisance','prior_sigma':1.,'fiducial':0.,'unit':''}
                                   for n in like.names[6:]]
    bins = [np.zeros((39,39)) for _ in like.config.nominal_bins]
    rows, entries = [], []
    for i,cell in enumerate(like.cells):
        cosmology_ad = cell.derivatives.copy()
        cell.names = list(space.names)+[f'bin{i:02d}:DESI-ELG:{n}' for n in theory.NUISANCE_NAMES]
        cell.derivatives = np.concatenate((cosmology_ad,np.zeros((1,3,11))),axis=-1)
        cell.fiducial_mean = mean.copy(); cell.F_like = cell.derivatives[0].T@cell.derivatives[0]
        indices = [like.names.index(n) for n in cell.names]
        bins[i][np.ix_(indices,indices)] = cell.F_like
        cell.metadata['nbar'] = {'DESI-ELG':1.}
        row = _row(space,cosmology_ad[0]); row['cell_id'] = cell.metadata['cell_id']
        row.update(decomposition='intrinsic-table-v1',fiducial_multipoles=mean[0].tolist(),
            direct_multipoles=mean[0].tolist(),whitened_physical_scale=float(np.linalg.norm(mean)),
            intrinsic_ad=(cosmology_ad[0]-geometry[0,:,:6]).tolist(),
            geometry_ad=geometry[0,:,:6].tolist(),geometry_ad_original=geometry[0,:,:6].tolist())
        delta = cosmology_ad[0]-np.asarray(row['intrinsic_ad'])-geometry[0,:,:6]
        row['chain_errors'] = row['chain_errors_original'] = [float(np.linalg.norm(delta[:,j])/np.linalg.norm(cosmology_ad[0,:,j])) for j in range(6)]
        for r,nk in zip(row['interpolation'],(32768,65536)): r.update(nk=nk,mean=mean[0].tolist())
        row['angular']['mean'] = mean[0].tolist()
        arrays = {'packed':packed.copy(),'k':np.asarray(k),'mean':mean.copy(),'total':cell.derivatives.copy(),
            'intrinsic':cell.derivatives-geometry,'geometry_original':geometry.copy(),
            'geometry_matched':geometry.copy(),'geometry_forward':geometry.copy(),
            'observed_intrinsic':cell.derivatives-geometry,'observed_intrinsic_table':np.asarray(table),
            'table_matched':np.asarray(table),'table_forward':np.asarray(table),
            'table_k':np.asarray(x),'table_mu':np.asarray(y),'pair_covariance':cell.covariance.copy()}
        record = {'definition':'intrinsic-table-v1','cell_id':cell.metadata['cell_id'],
            'pair':['DESI-ELG','DESI-ELG'],'packed_width':17,'n_cosmo':6,
            'packed_labels':cell.names,'packed_roles':list(space.names)+['a:'+n for n in theory.NUISANCE_NAMES],
            'n_mu':32,'ap_nk':16384,'ap_nmu':101,'redshift':1.1,'nbar_a':1.,'nbar_b':None,'k_nl':.7,
            'reference_cosmology':ref,'tangent_shape':[17,16384,101],'tangent_sha256':'1'*64,
            'repeated_primal_count':17,'repeated_primals_equal':True,'tangents_finite':True,
            'table_sha256':intrinsic.array_digest(table),'observation':{'outputs_bitwise_equal':True,
                'table_bitwise_equal':True,'table_observations':1,'source_sha256':intrinsic.NATIVE_AP_SOURCE_SHA256,
                'observed_source_sha256':intrinsic.OBSERVED_AP_SOURCE_SHA256,
                'scope':'intrinsic-only; native mean and total unobserved'}}
        entries.append(intrinsic.write_intrinsic_sidecar(root,f'intrinsic/cell{i:03d}-pair000.npz',record,arrays))
        rows.append(row)
    like.bin_F_like = bins; like.F_like = sum(bins)
    evidence['response'] = {'cosmology_columns':list(space.names),'points':rows,
        'cells':[ap_cell_record(like,c,[r],space=space) for c,r in zip(like.cells,rows)],
        'intrinsic_geometry':entries,'growth':_growth(monkeypatch,space)}
    for record in (provenance,evidence): record.update(profile_fields(like.config))
    results = {t:apply_cosmology_treatment(like,t) for t in COSMOLOGY_TREATMENTS}
    return like,results,provenance,evidence


def test_new_profile_roundtrip_keeps_sidecars_out_of_json(monkeypatch,tmp_path):
    from pfsfog import extended_artifacts as a
    stage = tmp_path/'stage'
    like,results,provenance,evidence = intrinsic_fixture(monkeypatch,stage)
    path = a.write_extended_artifacts(like,results,tmp_path/'artifacts',provenance,evidence,intrinsic_root=stage)
    manifest,_,rows,validation = a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))
    assert len(rows) == 7
    assert validation['response']['intrinsic_geometry']['pairs'] == 3
    expected = {f'intrinsic/cell{i:03d}-pair000.npz' for i in range(3)}
    assert set(manifest['files']) == expected|{'matrices.npz','evidence.json','summary.json','validation.json'}
    assert manifest['intrinsic_sidecars'] == evidence['response']['intrinsic_geometry']
    assert (path/'evidence.json').stat().st_size < 500000
    for name in expected: assert digest(path/name) == digest(stage/name)


@pytest.mark.parametrize('mutation', ['missing','extra','reorder','wrong_pair','table','native','unsafe'])
def test_new_profile_loader_rejects_rehashed_sidecar_corruption(monkeypatch,tmp_path,mutation):
    from pfsfog import extended_artifacts as a
    stage = tmp_path/'stage'
    like,results,provenance,evidence = intrinsic_fixture(monkeypatch,stage)
    path = a.write_extended_artifacts(like,results,tmp_path/'artifacts',provenance,evidence,intrinsic_root=stage)
    manifest = a.read_json(path/'manifest.json'); raw = a.read_json(path/'evidence.json')
    if mutation == 'missing': manifest['files'].pop(raw['response']['intrinsic_geometry'][0]['path'])
    elif mutation == 'extra': (path/'intrinsic/extra.npz').write_bytes(b'extra')
    else:
        entries = raw['response']['intrinsic_geometry']
        if mutation == 'reorder': entries.reverse()
        elif mutation == 'wrong_pair': entries[0]['pair'] = ['DESI-LRG','DESI-LRG']
        elif mutation == 'unsafe': entries[0]['path'] = '../outside.npz'
        else:
            entry = entries[0]; target = path/entry['path']
            with np.load(target,allow_pickle=False) as f: arrays = {name:f[name] for name in f.files}
            field = 'observed_intrinsic_table' if mutation == 'table' else 'total'
            arrays[field].flat[0] += .01
            np.savez(target,**arrays)
            entry['sha256'] = manifest['files'][entry['path']] = digest(target)
            for record in entry['arrays']:
                if record['name'] == field: record['sha256'] = hashlib.sha256(arrays[field].tobytes()).hexdigest()
        manifest['intrinsic_sidecars'] = copy.deepcopy(entries)
        a.write_json(path/'evidence.json',raw)
        manifest['files']['evidence.json'] = digest(path/'evidence.json')
        manifest['raw_gate_hashes'] = {k:a.hash_value(v) for k,v in raw.items()}
    a.write_json(path/'manifest.json',manifest)
    with pytest.raises(ValueError): a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))


def test_case_measurement_persists_each_intrinsic_pair_before_acceptance(monkeypatch,tmp_path):
    from scripts import run_extended_cosmology as driver
    from pfsfog import revision_validation_physical as physical
    from pfsfog import revision_validation_ap as ap
    from pfsfog import extended_validation as validation
    from pfsfog.revision_validation_ap_intrinsic import load_intrinsic_sidecar
    source = tmp_path/'inputs'; output = tmp_path/'case'; output.mkdir()
    like,_,provenance,evidence = intrinsic_fixture(monkeypatch,source)
    # Other physical gates are separately tested; this boundary control checks
    # the case runner's storage of actual shaped AP evidence and its real reload.
    monkeypatch.setattr(physical,'expected_support_schedule',lambda *a,**kw:[])
    monkeypatch.setattr(physical,'declare_fd_manifest',lambda *a,**kw:{})
    monkeypatch.setattr(physical,'assembled_fd_certificate',lambda *a,**kw:{'status':'passed'})
    monkeypatch.setattr(physical,'audit_cell_covariance_neighborhood',lambda *a,**kw:{'status':'passed'})
    monkeypatch.setattr(validation,'measure_sigma8_radius',lambda *a,**kw:{'status':'separate control'})
    def measured(likelihood,cell,backend,**kwargs):
        assert callable(kwargs.get('intrinsic_sink')), 'case measurement must retain raw intrinsic evidence'
        i = next(i for i,c in enumerate(like.cells) if c is cell)
        entry = evidence['response']['intrinsic_geometry'][i]
        kwargs['intrinsic_sink'](entry['record'],load_intrinsic_sidecar(source,entry))
        return [evidence['response']['points'][i]]
    monkeypatch.setattr(ap,'measure_ap_cell',measured)
    raw = driver.measure_case(like,object(),provenance,object(),output)
    entries = raw['response']['intrinsic_geometry']
    assert [entry['path'] for entry in entries] == [f'intrinsic/cell{i:03d}-pair000.npz' for i in range(3)]
    assert validation.validate_extended_response(raw['response'],like,intrinsic_root=output)['status'] == 'passed'
    for original,saved in zip(evidence['response']['intrinsic_geometry'],entries):
        assert saved['sha256'] == original['sha256']
        assert digest(output/saved['path']) == saved['sha256']
