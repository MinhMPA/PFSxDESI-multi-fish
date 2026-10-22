"""Versioned, externally pinned six-coordinate payloads and hostile mutations."""
import copy
import hashlib
import importlib
import json
from dataclasses import replace

import numpy as np
import pytest

from pfsfog.cosmology_projection import apply_cosmology_treatment, COSMOLOGY_TREATMENTS
from pfsfog.revision_forecast import RevisionLikelihood, CellLikelihood, make_shell_grid
from test_extended_validation import _report, _row, NAMES


def api():
    assert importlib.util.find_spec('pfsfog.extended_artifacts'), 'extended artifact API missing'
    return importlib.import_module('pfsfog.extended_artifacts')


def fixture(monkeypatch):
    ev, space, old, response = _report(monkeypatch)
    rng = np.random.default_rng(21)
    cells, rows, records = [], [], []
    from pfsfog.revision_validation_ap import ap_cell_record
    from pfsfog import cosmology_projection as projection
    monkeypatch.setattr(projection,'make_growth_rate_func',ev.make_growth_rate_func)
    monkeypatch.setattr(projection,'make_sigma8_func',ev.make_sigma8_func)
    for i in range(3):
        d = rng.normal(size=(3,6))
        row = _row(space,d); row['cell_id'] = f'cell{i}'
        cell = CellLikelihood(d.T@d,list(NAMES),
            {'cell_id':f'cell{i}','retained_pairs':[['DESI-ELG','DESI-ELG']],
             'model_redshift':1.1,'nominal_bin_index':i}, {},d[None],np.eye(3)[None],
             np.array([[10.,20.,30.]]))
        cells.append(cell); rows.append(row)
    theta = old.fiducial[[old.names.index(n) for n in NAMES]]
    params = [{'name':n,'role':'cosmo','prior_sigma':{'Mnu':5.,'Omegam':1.}.get(n),
               'fiducial':float(v),'unit':'eV' if n=='Mnu' else ''} for n,v in zip(NAMES,theta)]
    bins = [c.F_like for c in cells]+[np.zeros((6,6)) for _ in range(5)]
    like = RevisionLikelihood(replace(old.config,kmax=.02,dk=.01),sum(bins),list(NAMES),
        theta,params,cells,bins,make_shell_grid(.01,.02,.01),old.reference_cosmology,{}, {})
    response['points'] = rows
    response['cells'] = [ap_cell_record(like,c,[r],space=space) for c,r in zip(cells,rows)]
    results = {t:apply_cosmology_treatment(like,t) for t in COSMOLOGY_TREATMENTS}
    provenance = {key:hashlib.sha256(key.encode()).hexdigest() for key in
        ('source_content_hash','backend_snapshot','network_hash','survey_input_hash','runtime_hash')}
    provenance['project_head'] = 'a'*40
    evidence = {'response':response,'response_identity':provenance.copy(),
                'failed_treatments':{}}
    from pfsfog.fisher_precision import fisher_solver_record, LEGACY_FISHER_SOLVER
    provenance['fisher_solver'] = fisher_solver_record(LEGACY_FISHER_SOLVER)
    return like,results,provenance,evidence


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_trip_preserves_full_active_and_growth_arrays(monkeypatch,tmp_path):
    a=api(); like,results,prov,evidence=fixture(monkeypatch)
    path=a.write_extended_artifacts(like,results,tmp_path,prov,evidence)
    manifest,arrays,rows,validation=a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))
    assert manifest['schema_version']=='pfs-desi-cosmology-v1'
    assert manifest['n_cosmo']==6 and manifest['cosmology_columns']==list(NAMES)
    assert manifest['units']['unit_convention']=='fixed_h_ref'
    assert manifest['units']['sigma8_radius']=='8/h_trial Mpc'
    assert len(rows)==7
    for name,r in results.items():
        for field,shape in [('cov_cosmo_marg',(6,6)),('growth_jacobian',(8,6)),('growth_cov_marg',(8,8))]:
            np.testing.assert_allclose(arrays[f'{name}/{field}'],getattr(r,field),rtol=1e-10,atol=1e-12)
            assert arrays[f'{name}/{field}'].shape==shape
    assert manifest['status']=='numerically_validated'
    with pytest.raises(FileExistsError): a.write_extended_artifacts(like,results,path,prov,evidence)


@pytest.mark.parametrize('mutation',['labels','units','prior','growth','cross','baseline','source','missing_column'])
def test_writer_rejects_malformed_numeric_or_evidence_identity(monkeypatch,tmp_path,mutation):
    a=api(); like,r,p,e=fixture(monkeypatch)
    if mutation=='labels': r['marginal6'].active_parameter_names=tuple(reversed(NAMES))
    elif mutation=='units': e['response']['growth']['units'].pop('sigma8_radius')
    elif mutation=='prior': r['marginal6'].F_prior_cosmo[0,0]=1.
    elif mutation=='growth':
        r['release_h'].growth_jacobian=r['release_h'].growth_jacobian.copy()
        r['release_h'].growth_jacobian[0,3]+=1.
    elif mutation=='cross': r['marginal6'].growth_cov_marg=np.diag(np.diag(r['marginal6'].growth_cov_marg))
    elif mutation=='baseline': e['response']['cosmology_columns']=list(NAMES[:3])
    elif mutation=='source': e['response_identity']['source_content_hash']='f'*64
    elif mutation=='missing_column': like.cells[0].derivatives=like.cells[0].derivatives[:,:,:5]
    with pytest.raises(ValueError): a.write_extended_artifacts(like,r,tmp_path,p,e)


def test_consumer_recomputes_even_with_rehashed_manifest(monkeypatch,tmp_path):
    a=api(); like,r,p,e=fixture(monkeypatch)
    path=a.write_extended_artifacts(like,r,tmp_path,p,e)
    m=json.loads((path/'manifest.json').read_text()); m['units'].pop('h_ref')
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))


def test_external_pin_and_prior_identity_are_required(monkeypatch,tmp_path):
    a=api(); like,r,p,e=fixture(monkeypatch)
    path=a.write_extended_artifacts(like,r,tmp_path,p,e)
    with pytest.raises(ValueError,match='manifest'): a.load_extended_artifacts(path,manifest_sha256='0'*64)
    changed=replace(like,config=replace(like.config,nuisance_factor=2.))
    assert a.physical_likelihood_id(changed,p)==a.physical_likelihood_id(like,p)
    assert a.physical_likelihood_id(like,{**p,'network_hash':'e'*64})!=a.physical_likelihood_id(like,p)


def test_failed_diagnostic_has_no_consumable_covariance(monkeypatch,tmp_path):
    a=api(); like,r,p,e=fixture(monkeypatch)
    failed=r.pop('marginal6_no_extra')
    e['failed_treatments']={'marginal6_no_extra':{'status':'failed','error':'unconstrained directions','raw':{'eigenvalue':0.}}}
    path=a.write_extended_artifacts(like,r,tmp_path,p,e)
    m,arrays,rows,_=a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))
    assert m['failed_treatments']['marginal6_no_extra']['status']=='failed'
    assert not any(k.startswith('marginal6_no_extra/') for k in arrays)
    e['failed_treatments']['marginal6_no_extra']['covariance']=failed.covariance.tolist()
    with pytest.raises(ValueError): a.write_extended_artifacts(like,r,tmp_path/'bad',p,e)


def test_loader_rejects_extra_arrays_and_grid_configuration_mismatch(monkeypatch,tmp_path):
    a=api(); like,r,p,e=fixture(monkeypatch)
    path=a.write_extended_artifacts(like,r,tmp_path,p,e)
    with np.load(path/'matrices.npz') as saved: arrays={k:saved[k] for k in saved.files}
    arrays['baseline3/covariance']=np.eye(3)
    np.savez(path/'matrices.npz',**arrays)
    m=a.read_json(path/'manifest.json');m['files']['matrices.npz']=digest(path/'matrices.npz')
    a.write_json(path/'manifest.json',m)
    with pytest.raises(ValueError,match='inventory'):
        a.load_extended_artifacts(path,manifest_sha256=digest(path/'manifest.json'))
    like.grid=make_shell_grid(.01,.03,.02)
    with pytest.raises(ValueError,match='grid'):
        a.write_extended_artifacts(like,r,tmp_path/'wrong-grid',p,e)
