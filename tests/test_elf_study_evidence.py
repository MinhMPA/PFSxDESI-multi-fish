"""A new high-z cohort cannot borrow a baseline-only numerical certificate."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from pfsfog.elf_survey_evidence import request_digest
from scripts.elf_study_evidence import (make_study_request,validate_study_ledger,study_source,
    study_config_from_request,verify_highz,load_source_cell)

S={'nfft':2048,'n_mu':64,'dk':.000625}

def ledger():
    req=make_study_request(S,source={'identity_hash':'synthetic'})
    out={'status':'passed','scope':req['scope'],'source_changed':False,
        'source_identity_hash':'synthetic','request_sha256':request_digest(req),
        'complete_new_cells':True,'full_forecast_qualification':'pending',
        'workers':[{'key':r['key'],'status':'passed','exit_code':0,
            'source_identity_hash':'synthetic','request_sha256':request_digest(req)} for r in req['expected']]}
    return req,out


def test_request_is_exactly_three_new_overlap_fields_including_tail():
    r,_=ledger();assert len(r['expected'])==3
    assert [x['support']['model_redshift'] for x in r['expected']]==pytest.approx([1.8,2.05,2.15])
    assert [len(x['support']['local_names']) for x in r['expected']]==[30,30,17]
    assert r['expected'][-1]['support']['active_populations']==['PFS-ELG']
    config=study_config_from_request(r)
    assert config.nominal_bins[-1]==(2.1,2.2) and config.cosmology_basis=='extended6'
    assert all(x['support']['region']=='overlap' for x in r['expected'])


@pytest.mark.parametrize('damage',['missing','duplicate','failed','exit','source','grid','request','overstated','cross','empty'])
def test_changed_or_incomplete_physical_evidence_is_rejected(damage):
    r,s=ledger();live=deepcopy(r['source'])
    assert not validate_study_ledger(r,s,live)
    if damage=='missing':s['workers'].pop()
    if damage=='duplicate':s['workers'].append(deepcopy(s['workers'][0]))
    if damage=='failed':s['workers'][0]['status']='failed'
    if damage=='exit':s['workers'][0]['exit_code']=9
    if damage=='source':live['identity_hash']='changed'
    if damage=='grid':r['configuration']['dk']=.00125
    if damage=='request':s['workers'][0]['request_sha256']='wrong'
    if damage=='overstated':s['full_forecast_qualification']='passed'
    if damage=='cross':r['expected'][0]['support']['retained_pairs'].pop()
    if damage=='empty':r['expected'][0]['support']['empty_observation']=True
    assert validate_study_ledger(r,s,live)


def test_source_binds_every_study_program_and_original_physics():
    record=study_source(S)
    assert 'original' in record['details']
    actual={p.name for p in Path('scripts').glob('elf_study*.py')}
    bound={Path(p).name for p in record['details']['programs']}
    assert actual<=bound
    assert record['details']['original']['details']['settings']==S


def test_missing_run_and_unbound_source_cell_fail_closed(tmp_path):
    assert verify_highz(tmp_path)['status']=='failed'
    with pytest.raises((ValueError,FileNotFoundError)):
        load_source_cell(tmp_path,{'files':{}})
