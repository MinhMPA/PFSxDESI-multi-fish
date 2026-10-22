"""Physical evidence must belong to the requested population and source."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import numpy as np

from pfsfog.cosmo import FIDUCIAL
from scripts.pfs_bias_evidence import (
    make_bias_request, validate_bias_request, verify_bias_fiducials, bias_zero_support,
)

SETTINGS={'nfft':2048,'n_mu':64,'dk':.000625}
COSMO=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.6)
SOURCE={'identity_hash':'test','details':{'original':{}}}


def test_request_binds_the_measured_bias_not_only_the_survey_support():
    request=make_bias_request(SETTINGS,COSMO,source=SOURCE)
    validate_bias_request(request,COSMO,SOURCE)
    expected=request['expected'][1]
    assert expected['fiducials']['bin01:PFS-ELG:b1_sigma8']==pytest.approx(.5664)
    inputs={'fiducials':deepcopy(expected['fiducials'])}
    verify_bias_fiducials(inputs,expected)
    inputs['fiducials']['bin01:PFS-ELG:b1_sigma8']*=1.01
    with pytest.raises(ValueError,match='fiducial'):verify_bias_fiducials(inputs,expected)


@pytest.mark.parametrize('mutation',['missing','duplicate','factor','source','configuration'])
def test_changed_requests_cannot_reuse_a_passed_population(mutation):
    r=make_bias_request(SETTINGS,COSMO,source=SOURCE)
    if mutation=='missing':r['expected'].pop()
    elif mutation=='duplicate':r['expected'][1]=r['expected'][0]
    elif mutation=='factor':r['expected'][1]['experiment']['ratio']=1.3
    elif mutation=='source':r['source']['identity_hash']='changed'
    else:r['configuration']['n_mu']=32
    with pytest.raises(ValueError):validate_bias_request(r,COSMO,SOURCE)


@pytest.mark.parametrize('alphas,ir,want',[
    ([1.,1.],False,True),([1.,1.001],False,False),([1.,1.],True,False)])
def test_gamma3_hexadecapole_zero_requires_identity_ap_and_ir_off(alphas,ir,want):
    names=['ln10_10_As','bin01:PFS-ELG:bGamma3']
    pairs=[['PFS-ELG','PFS-ELG'],['DESI-LRG','PFS-ELG'],['DESI-LRG','DESI-LRG']]
    zeros=bias_zero_support(names,pairs,alphas=alphas,settings={'engine':'elf.EPT','do_irres':ir})
    expected=[(names[1],tuple(p),4) for p in pairs[:2]] if want else []
    assert zeros==expected


def test_live_cross_measurement_saves_the_physical_shot_noise_response(tmp_path):
    # Small live calculation: catches missing imports/routing in the measured
    # cross-spectrum path; this deliberately does not certify grid convergence.
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.one_loop import make_one_loop_engine
    from scripts.pfs_bias_cases import bias_cases
    from scripts.pfs_bias_physical import measure_bias_cell
    from scripts.elf_study_profiles import highz_profiles, study_config
    settings={'nfft':128,'n_mu':8,'dk':.05}
    cosmo=FiducialCosmology();case=bias_cases(cosmo,settings)[0]
    cell,record=measure_bias_cell(case,study_config(highz_profiles()['C22'],settings,extended=True),
                                 cosmo,make_one_loop_engine(nfft=128),tmp_path)
    pairs=list(map(tuple,cell.metadata['retained_pairs']))
    pair=('DESI-LRG','PFS-ELG');row=3*pairs.index(pair)
    column=cell.names.index('bin01:DESI-LRG|PFS-ELG:Pshot_cross')
    density=cell.metadata['nbar']
    np.testing.assert_allclose(cell.derivatives[:,row,column],
        1/np.sqrt(density[pair[0]]*density[pair[1]]),rtol=1e-12)
    with np.load(tmp_path/'arrays.npz') as arrays:
        assert arrays['fd_plus'].shape[0]==len(cell.names)
        assert np.isfinite(arrays['D_geometry']).all()
