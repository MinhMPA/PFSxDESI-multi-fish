"""Inference preserves shared information, prior algebra and explicit failures."""
from dataclasses import replace
import copy
import importlib
import importlib.util

import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.cosmology_projection import COSMOLOGY_TREATMENTS
from pfsfog.revision_forecast import (RevisionConfig,RevisionLikelihood,CellLikelihood,
    make_shell_grid,apply_revision_priors)
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER


def api():
    spec=importlib.util.find_spec('pfsfog.elf_inference_evidence')
    assert spec is not None,'ELF inference evidence is missing'
    return importlib.import_module('pfsfog.elf_inference_evidence')


def fixture(monkeypatch):
    # The physical algebra is analytic here; the CLI later measures real growth.
    import pfsfog.cosmology_projection as cp
    import pfsfog.revision_forecast as rf
    import pfsfog.extended_validation as ev
    rate=lambda:lambda z,c:(c['ln10_10_As']+c['mnu']*z+c['Omega_m']*z*z
                            +c['h']+c['omega_b']*z**3+c['n_s']*z**4)
    for mod in (cp,rf,ev):
        monkeypatch.setattr(mod,'make_growth_rate_func',rate)
        monkeypatch.setattr(mod,'make_sigma8_func',lambda:lambda z,c:1.)
    config=RevisionConfig('B',cosmology_basis='extended6',kmax=.025,dk=.005)
    names=list(make_cosmology_spec(FIDUCIAL,extended=True).names)+['bin00:DESI-LRG:c2']
    values=np.array([FIDUCIAL[k] for k in ('ln10_10_As','mnu','Omega_m','h','omega_b','n_s')]+[1.])
    rng=np.random.default_rng(739)
    d=rng.normal(size=(3,3,7));c=np.tile(np.eye(3),(3,1,1));f=np.einsum('koi,koj->ij',d,d)
    meta={'cell_id':'bin00-radial00-overlap','nominal_bin_index':0,
        'retained_pairs':[('DESI-LRG','DESI-LRG')], 'local_names':names,
        'retained_observations':[(('DESI-LRG','DESI-LRG'),ell) for ell in (0,2,4)]}
    cell=CellLikelihood(f,names.copy(),meta,{},d,c,np.ones((3,3)))
    params=[{'name':n,'role':'cosmo' if i<6 else 'tracer','unit':'',
             'fiducial':values[i],'prior_sigma':2. if i==6 else None} for i,n in enumerate(names)]
    bins=[f]+[np.zeros_like(f) for _ in range(7)]
    like=RevisionLikelihood(config,f,names,values,params,[cell],bins,
        make_shell_grid(.01,.025,.005),FIDUCIAL,{'engine':'elf.EPT'}, {})
    # Independent hand-constructed analytic four-coordinate reference for tests.
    ids=[0,1,2,6];nd=d[:,:,ids].copy();nf=np.einsum('koi,koj->ij',nd,nd)
    nn=[names[i] for i in ids];nm={**meta,'local_names':nn}
    nc=CellLikelihood(nf,nn.copy(),nm,{},nd,c.copy(),np.ones((3,3)))
    native=RevisionLikelihood(replace(config,cosmology_basis='baseline3'),nf,nn,values[ids],
        [copy.deepcopy(params[i]) for i in ids],[nc],[nf]+[np.zeros_like(nf) for _ in range(7)],
        like.grid,FIDUCIAL,like.model.copy(),{})
    nr=apply_revision_priors(native,1.,fisher_solver=DECIMAL_FISHER_SOLVER)
    from pfsfog.elf_survey import elf_cosmology_spec
    growth=ev.measure_growth_jacobian(elf_cosmology_spec(FIDUCIAL),FIDUCIAL,
                                    np.array([(a+b)/2 for a,b in config.nominal_bins]))
    provenance={'source_identity_hash':'synthetic-fixture','case':'B',
        'native_construction':{'method':'build_revision_likelihood','cosmology_basis':'baseline3',
          'source_identity_hash':'synthetic-fixture','case':'B'}}
    return like,native,nr,growth,provenance


def test_preparation_retains_exact_prior_inputs_when_solve_is_singular(monkeypatch):
    import pfsfog.cosmology_projection as cp
    assert hasattr(cp,'prepare_cosmology_precision'),'pre-inversion precision inputs are missing'
    like,*_=fixture(monkeypatch)
    like.F_like[:]=0.
    raw=cp.prepare_cosmology_precision(like,'marginal6')
    assert raw['F_prior_cosmo'][1,1]==pytest.approx(.04)
    assert raw['F_prior_cosmo'][4,4]==pytest.approx(.00055**-2)
    assert raw['F_prior_cosmo'][5,5]==pytest.approx(.042**-2)
    assert raw['F_prior_nuisance'][6,6]==.25
    np.testing.assert_array_equal(raw['F_total'],raw['F_prior_cosmo']+raw['F_prior_nuisance'])
    with pytest.raises(ValueError):
        cp.project_cosmology_precision(like,'marginal6',fisher_solver=DECIMAL_FISHER_SOLVER)


def test_all_seven_outcomes_and_correlated_growth_survive_roundtrip(monkeypatch,tmp_path):
    module=api();like,native,nr,growth,p=fixture(monkeypatch)
    from pfsfog.elf_inference import infer_treatments
    results,failed=infer_treatments(like)
    assert set(results)==set(COSMOLOGY_TREATMENTS) and not failed
    out=tmp_path/'case'
    module.write_case(out,like,results,failed,native,nr,growth,p)
    checked=module.verify_case(out,p)
    assert checked['status']=='passed' and checked['full_forecast_qualification']=='pending'
    assert checked['successful_treatments']==list(COSMOLOGY_TREATMENTS)
    covariance=results['marginal6'].growth_cov_marg
    assert covariance.shape==(8,8) and abs(covariance[0,-1])>0.
    assert checked['native_bridge']['status']=='passed'


def test_failed_solves_retain_input_matrices_and_no_consumable_covariance(monkeypatch):
    module=api();like,*_=fixture(monkeypatch)
    from pfsfog.elf_inference import infer_treatments
    like.F_like[:]=0.
    results,failed=infer_treatments(like)
    assert not results and set(failed)==set(COSMOLOGY_TREATMENTS)
    for record in failed.values():
        assert record['status']=='failed' and record['error']
        assert 'covariance' not in record
        raw=record['precision_inputs']
        np.testing.assert_array_equal(raw['F_total'],raw['F_like']+raw['F_prior_cosmo']+raw['F_prior_nuisance'])


@pytest.mark.parametrize('damage',['prior','D','C','growth','missing','native_source','native_slice','overclaim'])
def test_saved_scientific_corruption_is_rejected(monkeypatch,tmp_path,damage):
    module=api();like,native,nr,growth,p=fixture(monkeypatch)
    from pfsfog.elf_inference import infer_treatments
    from pfsfog.elf_survey_evidence import file_digest,write_json
    from pfsfog.extended_artifacts import read_json
    out=tmp_path/'case';results,failed=infer_treatments(like)
    module.write_case(out,like,results,failed,native,nr,growth,p)
    manifest=read_json(out/'manifest.json')
    with np.load(out/'arrays.npz',allow_pickle=False) as a: arrays={k:a[k] for k in a.files}
    if damage=='prior':arrays['marginal6/F_prior_cosmo'][4,4]*=2
    elif damage=='D':arrays['likelihood/cell0/derivatives'][0,0,0]+=1.
    elif damage=='C':arrays['likelihood/cell0/covariance'][0,0,0]*=2
    elif damage=='growth':arrays['marginal6/growth_cov_marg'][0,1]=0.
    elif damage=='missing':manifest['treatments'].pop('release_h')
    elif damage=='native_source':manifest['provenance']['native_construction']['source_identity_hash']='foreign'
    elif damage=='native_slice':manifest['provenance']['native_construction']['method']='slice_extended'
    else:manifest['full_forecast_qualification']='passed'
    np.savez_compressed(out/'arrays.npz',**arrays)
    manifest['arrays_sha256']=file_digest(out/'arrays.npz')
    write_json(out/'manifest.json',manifest)
    with pytest.raises(ValueError):module.verify_case(out,p)


def test_failed_required_controls_are_saved_and_reported_failed(monkeypatch,tmp_path):
    module=api();like,_,_,growth,p=fixture(monkeypatch)
    from pfsfog.elf_inference import infer_treatments
    like.F_like[:]=0.;like.cells[0].derivatives[:]=0.
    observed=[]
    results,failures=infer_treatments(like,on_outcome=lambda n,r,f:observed.append((n,r,f['status'])))
    assert [n for n,_,_ in observed]==list(COSMOLOGY_TREATMENTS)
    assert all(r is None and status=='failed' for _,r,status in observed)
    out=tmp_path/'failed'
    module.write_case(out,like,results,failures,None,None,growth,p)
    checked=module.verify_case(out,p)
    assert checked['status']=='failed'
    assert checked['failed_treatments']==list(COSMOLOGY_TREATMENTS)
    assert (out/'arrays.npz').is_file() and (out/'manifest.json').is_file()


def test_consistent_but_changed_extra_response_is_not_the_measured_likelihood(monkeypatch,tmp_path):
    module=api();like,native,nr,growth,p=fixture(monkeypatch)
    assert hasattr(module,'verify_physical_binding'),'saved constraints need binding to measured D/C'
    from pfsfog.elf_inference import infer_treatments
    from pfsfog.extended_artifacts import read_json,unpack_likelihood,validate_likelihood
    results,failures=infer_treatments(like);out=tmp_path/'case'
    module.write_case(out,like,results,failures,native,nr,growth,p)
    module.verify_physical_binding(out,like)
    record=read_json(out/'manifest.json')
    with np.load(out/'arrays.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    arrays['likelihood/cell0/derivatives'][:,:,3]*=1.1
    d=arrays['likelihood/cell0/derivatives'];f=np.einsum('koi,koj->ij',d,d)
    arrays['likelihood/cell0/F_like']=f;arrays['likelihood/F_like']=f
    arrays['likelihood/bin_F_like'][0]=f
    # It is internally consistent, and conditional3 cannot see the changed h column.
    validate_likelihood(unpack_likelihood(record['likelihood'],arrays))
    np.savez_compressed(out/'arrays.npz',**arrays)
    with pytest.raises(ValueError,match='physical'):
        module.verify_physical_binding(out,like)


def test_native_reference_must_use_the_same_numerical_model(monkeypatch,tmp_path):
    module=api();like,native,nr,growth,p=fixture(monkeypatch)
    from pfsfog.elf_inference import infer_treatments
    results,failures=infer_treatments(like)
    native.model={**native.model,'fft_n':512}
    out=tmp_path/'case';module.write_case(out,like,results,failures,native,nr,growth,p)
    with pytest.raises(ValueError,match='model'):
        module.verify_case(out,p)
