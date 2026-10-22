"""A wider-bias certificate must identify the actual population and complete cohort."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.elf_survey_evidence import request_digest
from scripts.pfs_bias_broad_evidence import (
    make_broad_request, validate_broad_request, validate_broad_summary,
)
from scripts.pfs_bias_broad_physical import measure_broad_population

SETTINGS={'nfft':2048,'n_mu':64,'dk':.000625}
COSMO=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.6)
SOURCE={'identity_hash':'test','details':{'original':{}}}


@pytest.mark.parametrize('mutation',['missing','duplicate','ratio','equality','fiducial','source','grid'])
def test_changed_physical_request_is_rejected(mutation):
    r=make_broad_request(SETTINGS,COSMO,source=SOURCE)
    validate_broad_request(r,COSMO,SOURCE)
    if mutation=='missing':r['expected'].pop()
    elif mutation=='duplicate':r['expected'][1]=r['expected'][0]
    elif mutation=='ratio':r['expected'][1]['experiment']['ratio']+=.01
    elif mutation=='equality':r['expected'][1]['experiment']['equal_to']='DESI-QSO'
    elif mutation=='fiducial':r['expected'][1]['fiducials']['bin01:PFS-ELG:b1_sigma8']*=1.01
    elif mutation=='source':r['source']['identity_hash']='changed'
    else:r['configuration']['dk']*=2
    with pytest.raises(ValueError,match='request'):validate_broad_request(r,COSMO,SOURCE)


@pytest.mark.parametrize('mutation',['missing','duplicate','failed','exit','identity'])
def test_incomplete_or_failed_cohort_cannot_supply_inference(mutation):
    request=make_broad_request(SETTINGS,COSMO,source=SOURCE)
    digest=request_digest(request)
    summary={'status':'passed','complete_populations':True,'request_sha256':digest,
        'source_identity_hash':'test','workers':[{'key':r['key'],'status':'passed',
            'exit_code':0,'request_sha256':digest,'source_identity_hash':'test'}
            for r in request['expected']]}
    validate_broad_summary(request,summary)
    if mutation=='missing':summary['workers'].pop()
    elif mutation=='duplicate':summary['workers'][1]=summary['workers'][0]
    elif mutation=='failed':summary['workers'][0]['status']='failed'
    elif mutation=='exit':summary['workers'][0]['exit_code']=1
    else:summary['workers'][0]['source_identity_hash']='other'
    with pytest.raises(ValueError):validate_broad_summary(request,summary)


def test_physical_worker_refuses_to_overwrite_evidence(tmp_path):
    sentinel=tmp_path/'kept.json';sentinel.write_text('preserve')
    with pytest.raises(FileExistsError):measure_broad_population({},'unused',tmp_path)
    assert sentinel.read_text()=='preserve'


def test_live_broad_worker_authenticates_saved_population_and_signed_responses(tmp_path):
    from pfsfog.cosmo import FiducialCosmology
    from scripts.pfs_bias_evidence import verify_bias_cell
    settings={'nfft':128,'n_mu':8,'dk':.05}
    request=make_broad_request(settings,FiducialCosmology())
    expected=request['expected'][0]
    receipt=measure_broad_population(request,expected['key'],tmp_path)
    assert receipt['status']=='passed',receipt
    assert receipt['source_before']==receipt['source_after']==request['source']['identity_hash']
    verified=verify_bias_cell(tmp_path,expected,request)
    assert verified['status']=='passed'
