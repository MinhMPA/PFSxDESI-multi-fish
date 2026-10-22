"""Published PFS densities and their effect on physical Gaussian observables."""
from copy import deepcopy

import numpy as np
import pytest

from pfsfog.survey_geometry import cell_number_density, ShellGeometry
from pfsfog.surveys import default_survey_group
from tests.test_pfs_density_response import physical_toy


@pytest.mark.parametrize('lo,hi,want', [
    (.6,.8,1.9e-4), (.8,1.,6.0e-4), (1.,1.2,5.8e-4),
    (1.2,1.4,7.8e-4), (1.4,1.6,5.5e-4), (1.6,2.,3.1e-4),
    (2.,2.4,2.7e-4), (1.,1.1,5.8e-4), (1.1,1.2,5.8e-4),
    (2.,2.1,2.7e-4), (2.1,2.4,2.7e-4)])
def test_published_density_reaches_actual_analysis_cells(lo,hi,want):
    # Catches wrong input selection, count/density confusion and bin averaging.
    from scripts.takada_density import takada_survey
    survey=takada_survey()
    assert cell_number_density(survey,lo,hi)==pytest.approx(want,rel=2e-15)
    assert survey.area_deg2==1200.
    assert survey.b1_of_z(1.8)==pytest.approx(1.62)


def test_complete_table_support_preserves_existing_observed_volumes():
    from scripts.takada_density import takada_survey
    survey=takada_survey()
    assert survey.z_min_all[0]==.6
    assert survey.z_max_all[-1]==2.4
    np.testing.assert_array_equal(survey.z_max_all[:-1],survey.z_min_all[1:])
    original=ShellGeometry.from_group(default_survey_group())
    for lo,hi in [(.6,.7),(.7,.8),(.8,1.),(1.,1.1),(1.1,1.2),
                  (1.2,1.4),(1.4,1.6),(1.6,2.),(2.,2.1)]:
        assert survey.volume(lo,hi)==pytest.approx(original.volume(lo,hi,1200.),rel=1e-6)
    # Existing full PFS-native geometry, not Table 2's rounded 1464 deg2 volume.
    assert survey.volume(2.,2.4)==pytest.approx(1.898263920857254267e9,rel=2e-15)
    assert survey.nbar_eff(2.1,2.4)*survey.volume(2.1,2.4)>0.


def test_density_and_volume_transform_preserves_signal_and_rebuilds_noise():
    # Catches treating all cross columns as PFS-only or scaling parameter errors.
    from scripts.takada_density import retarget_source
    source,row,grid=physical_toy()
    original=deepcopy(source['inputs'])
    target=deepcopy(source['inputs']['metadata'])
    target['nbar']={'DESI-QSO':4e-4,'PFS-ELG':1e-4}
    target['volume']=4e8
    changed=retarget_source(source,target)
    np.testing.assert_array_equal(changed['arrays']['mean'],source['arrays']['mean'])
    np.testing.assert_array_equal(changed['arrays']['D'][:,:,:3],source['arrays']['D'][:,:,:3])
    scale=np.array([1.,1.,1.,.25,2.,2.,2.,1/np.sqrt(2.),1/np.sqrt(2.)])
    np.testing.assert_allclose(changed['arrays']['D'],source['arrays']['D']*scale,rtol=1e-15,atol=0)
    s=source['arrays']['pkmu'];w=np.polynomial.legendre.leggauss(8)[1]
    modes=grid.centers**2*grid.widths*4e8/(2*np.pi**2)
    np.testing.assert_allclose(changed['arrays']['C'][:,0,0],np.sum(w*(s[0]+2500.)**2,axis=1)/modes)
    np.testing.assert_allclose(changed['arrays']['C'][:,3,3],np.sum(w*(s[1]+10000.)**2,axis=1)/modes)
    np.testing.assert_allclose(changed['arrays']['C'][:,0,3],np.sum(w*s[2]**2,axis=1)/modes)
    for col in (0,3,4,7):
        i=1+6*col
        np.testing.assert_allclose(changed['neighborhood_spectra'][i],
            s+scale[col]*(source['neighborhood_spectra'][i]-s))
    assert source['inputs']==original
    assert changed['inputs']['metadata']==target


@pytest.mark.parametrize('mutation', ['redshift','fiducial','population','density','volume'])
def test_reuse_rejects_changed_physics_or_invalid_observed_cell(mutation):
    from scripts.takada_density import retarget_source
    source,_,_=physical_toy();target=deepcopy(source['inputs']['metadata'])
    if mutation=='redshift':target['model_redshift']=2.1
    if mutation=='fiducial':source['inputs']['fiducials']['bin00:PFS-ELG:Pshot']=.1
    if mutation=='population':target['active_populations']=['PFS-ELG']
    if mutation=='density':target['nbar']['PFS-ELG']=0.
    if mutation=='volume':target['volume']=-1.
    with pytest.raises(ValueError):retarget_source(source,target)


def test_joint_auto_cross_density_scaling_matches_elf_with_nonidentity_ap():
    import jax.numpy as jnp
    from pfsfog.cosmo import FiducialCosmology,FIDUCIAL
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.elf_survey import elf_cosmology_spec
    from pfsfog.eft_params import tracer_fiducials
    from pfsfog.revision_theory import make_pair_response,fiducial_nuisance,NUISANCE_NAMES
    c=FiducialCosmology();z=.9;sigma=float(c.sigma8(z))
    qa=fiducial_nuisance(tracer_fiducials('DESI-QSO',2.,sigma,b1_ref=1.4))
    qb=fiducial_nuisance(tracer_fiducials('PFS-ELG',1.26,sigma,b1_ref=1.4))
    qnoise=qb.at[NUISANCE_NAMES.index('Pshot')].set(.3).at[NUISANCE_NAMES.index('a2')].set(-.2)
    space=elf_cosmology_spec(FIDUCIAL)
    theta=jnp.array([FIDUCIAL[{'Mnu':'mnu','Omegam':'Omega_m'}.get(n,n)] for n in space.names])
    theta=theta.at[space.names.index('h')].add(.01)
    engine=ELFPowerSpectrum(nfft=512);k=jnp.array([.03,.09,.17]);mu=jnp.array([-.8,.2,.9])
    old=make_pair_response(engine,z,FIDUCIAL,1e-4,2e-4,space=space)
    new=make_pair_response(engine,z,FIDUCIAL,4e-4,1e-4,space=space)
    assert max(abs(np.asarray(old.ap_alphas(theta))-1.))>1e-3
    zero=jnp.zeros(2);cross=jnp.array([.4,-.2])
    signal=old.pkmu(k,mu,theta,qa,qb,zero)
    np.testing.assert_allclose(new.pkmu(k,mu,theta,qa,qb,zero),signal,rtol=1e-12,atol=1e-9)
    a=old.pkmu(k,mu,theta,qa,qb,cross)-signal
    b=new.pkmu(k,mu,theta,qa,qb,cross)-signal
    np.testing.assert_allclose(b,a/np.sqrt(2.),rtol=1e-10,atol=1e-9)
    oldauto=make_pair_response(engine,z,FIDUCIAL,2e-4,space=space)
    newauto=make_pair_response(engine,z,FIDUCIAL,1e-4,space=space)
    signal=oldauto.pkmu(k,mu,theta,qb)
    np.testing.assert_allclose(newauto.pkmu(k,mu,theta,qnoise)-signal,
                              2*(oldauto.pkmu(k,mu,theta,qnoise)-signal),rtol=1e-10,atol=1e-9)
