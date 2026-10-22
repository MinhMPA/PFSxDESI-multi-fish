"""Independent reference construction roles and authenticated saved comparisons."""
import copy
from dataclasses import replace

import pytest

from test_cosmology_resolution import sample,projected,native3,control_provenance


def context(monkeypatch,amplitudes=None):
    from scripts import run_extended_cosmology as driver
    from pfsfog import revision_forecast as forecast
    amplitudes=amplitudes or {8192:1.008,16384:1.005,32768:1.003,65536:1.}
    calls=[]
    def build(config):
        calls.append((config.cosmology_basis,config.ap_interpolation_nk))
        like,results=sample(config.ap_interpolation_nk,amplitudes[config.ap_interpolation_nk],config.case)
        assert config==replace(like.config,cosmology_basis=config.cosmology_basis)
        return native3(like,results['conditional3'])[0] if config.cosmology_basis=='baseline3' else like
    def priors(like,*args):
        ext,results=sample(like.config.ap_interpolation_nk,amplitudes[like.config.ap_interpolation_nk],like.config.case)
        return native3(ext,results['conditional3'])[1]
    monkeypatch.setattr(forecast,'apply_revision_priors',priors)
    monkeypatch.setattr(driver,'treatments',lambda like,**kwargs:(projected(like),{}))
    base,results=sample(16384,amplitudes[16384]);old,old_results=sample(8192,amplitudes[8192])
    oracle=native3(old,old_results['conditional3'])
    identity={'path':'authenticated/historical/oracle','sha256':'b'*64,'case':'B'}
    return driver,base,results,oracle,identity,build,calls


def measure(monkeypatch,path,amplitudes=None):
    driver,base,results,oracle,identity,build,calls=context(monkeypatch,amplitudes)
    assert hasattr(driver,'measure_ap_references'),'AP reference sequence is missing'
    record=driver.measure_ap_references(base,results,oracle,path,control_provenance(),
        oracle_identity=identity,build_likelihood=build)
    return driver,base,results,oracle,identity,calls,record


def test_reference_sequence_builds_native_bases_and_only_authorized_output_grids(monkeypatch,tmp_path):
    driver,base,results,oracle,identity,calls,record=measure(monkeypatch,tmp_path)
    assert calls==[('baseline3',16384),('extended6',8192),('baseline3',8192),
                   ('extended6',32768),('extended6',65536)]
    assert set(record['controls'])=={'native3_16384','native3_8192','extended6_8192','extended6_32768','extended6_65536'}
    assert set(record['comparisons'])=={'8192_to_16384','16384_to_32768','32768_to_65536','16384_to_65536'}
    for key,control in record['controls'].items():
        nk=int(key.split('_')[1])
        assert control['evaluated_ap_grids']==[nk]
        assert control['derivative_qualification_performed'] is False
        assert control['reference_role']==('native3_same_resolution' if key.startswith('native3') else 'ap_likelihood_output')
    loaded,recomputed=driver.validate_ap_references(base,results,record,oracle,control_provenance(),oracle_identity=identity)
    assert set(loaded)=={8192,16384,32768,65536}
    assert recomputed==record['comparisons']
    assert record['controls']['native3_8192']['reference_checks']['native3_historical_oracle']['status']=='passed'


@pytest.mark.parametrize('mutation',['wrong_role','grid_declaration','reference_profile','reference_hash',
                                    'missing_treatment','missing_control','oracle_identity','comparison','version'])
def test_reference_reload_rejects_substituted_or_incomplete_controls(monkeypatch,tmp_path,mutation):
    driver,base,results,oracle,identity,_,record=measure(monkeypatch,tmp_path)
    control=record['controls']['extended6_32768']
    if mutation=='wrong_role': control['reference_role']='production_qualified'
    elif mutation=='grid_declaration': control['evaluated_ap_grids']=[32768,65536,131072]
    elif mutation=='reference_profile': control['numerical_profile']['name']='legacy8192-v1'
    elif mutation=='reference_hash': control['arrays_sha256']='0'*64
    elif mutation=='missing_treatment': control['treatments'].remove('release_h')
    elif mutation=='missing_control': record['controls'].pop('native3_8192')
    elif mutation=='oracle_identity': record['oracle_export']['sha256']='0'*64
    elif mutation=='comparison': record['comparisons']['16384_to_65536']['treatments']['conditional3']['cosmology_marg']['coarse'][0]*=1.001
    elif mutation=='version': record['definition']='ap-independent-references-v2'
    with pytest.raises(ValueError):
        driver.validate_ap_references(base,results,record,oracle,control_provenance(),oracle_identity=identity)


def test_historical_change_is_recorded_separately_from_new_grid_convergence(monkeypatch,tmp_path):
    driver,base,results,oracle,identity,_,record=measure(monkeypatch,tmp_path,
        {8192:1.03,16384:1.,32768:1.,65536:1.})
    assert record['comparisons']['8192_to_16384']['status']=='failed'
    assert record['historical_change_requires_assessment'] is True
    assert all(record['comparisons'][key]['status']=='passed' for key in
        ('16384_to_32768','32768_to_65536','16384_to_65536'))
    driver.validate_ap_references(base,results,record,oracle,control_provenance(),oracle_identity=identity)


def test_failed_direct_finest_comparison_survives_saved_reload(monkeypatch,tmp_path):
    driver,base,results,oracle,identity,_,record=measure(monkeypatch,tmp_path,
        {8192:1.018081,16384:1.018081,32768:1.009,65536:1.})
    assert record['comparisons']['16384_to_32768']['status']=='passed'
    assert record['comparisons']['32768_to_65536']['status']=='passed'
    assert record['comparisons']['16384_to_65536']['status']=='failed'
    _,recomputed=driver.validate_ap_references(base,results,record,oracle,control_provenance(),oracle_identity=identity)
    assert recomputed['16384_to_65536']['status']=='failed'
    record['comparisons']['16384_to_65536']['status']='passed'
    with pytest.raises(ValueError):
        driver.validate_ap_references(base,results,record,oracle,control_provenance(),oracle_identity=identity)
