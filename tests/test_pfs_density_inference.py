"""Density changes must preserve global cosmology and once-only shared priors."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from pfsfog.fisher_core import fisher_from_shells
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from pfsfog.revision_forecast import CellLikelihood, project_baseline_precision, make_shell_grid


def toy_result():
    names=['ln10_10_As','Mnu','Omegam','bin00:DESI-QSO:b1_sigma8',
           'bin01:DESI-QSO:b1_sigma8','bin01:PFS-ELG:Pshot',
           'bin01:DESI-QSO|PFS-ELG:Pshot_cross']
    parameters=[]
    for i,name in enumerate(names):
        parameters.append(dict(name=name, coordinate=name.split(':')[-1],
            role='cosmo' if i<3 else 'cross' if '|' in name else 'tracer',
            nominal_bin=None if i<3 else 0 if i==3 else 1,
            prior_sigma=[None,5.,1.,None,None,1.,1.][i],fiducial=0.))
    g=make_shell_grid(.05,.07,.01);cells=[];bins=[np.zeros((7,7)) for _ in range(2)]
    rng=np.random.default_rng(2481)
    for j,(ib,ix,radial) in enumerate([(0,[0,1,2,3],[.8,1.]),(1,[0,1,2,4,5,6],[1.,1.1]),(1,[0,1,2,4,5,6],[1.1,1.2])]):
        D=rng.normal(size=(2,9,len(ix)));C=np.tile(np.eye(9),(2,1,1))
        f=fisher_from_shells(D,C,g.centers)
        m=dict(cell_id=f'bin{ib:02d}-radial{j:02d}-overlap',nominal_bin_index=ib,
               nominal_bin=[[.8,1.],[1.,1.2]][ib],radial_bin=radial,region='overlap',
               active_populations=['DESI-QSO','PFS-ELG'] if ib else ['DESI-QSO'],
               nbar={'DESI-QSO':1e-4,'PFS-ELG':2e-4} if ib else {'DESI-QSO':1e-4},
               volume=1e8)
        c=CellLikelihood(f,[names[i] for i in ix],m,{},D,C,np.ones((2,9)))
        cells.append(c);bins[ib][np.ix_(ix,ix)]+=f
    config=SimpleNamespace(cosmology_basis='baseline3',nominal_bins=[(.8,1.),(1.,1.2)])
    like=SimpleNamespace(config=config,names=names,parameters=parameters,F_like=sum(bins),
                         bin_F_like=bins,cells=cells,fiducial=np.zeros(7),grid=g)
    precision=project_baseline_precision(like,1.,fisher_solver=DECIMAL_FISHER_SOLVER)
    j=np.array([[.1,.2,.3],[.15,.21,.4]])
    return SimpleNamespace(likelihood=like,parameters=parameters,**{k:v for k,v in precision.items() if k!='parameters'},
        growth_z=np.array([.9,1.1]),growth_jacobian=j,growth_fiducial=np.array([.4,.3]))


def changed_parts(result,factor):
    from scripts.pfs_density_response import density_column_scales
    parts=[]
    for old in result.likelihood.cells[1:]:
        c=deepcopy(old);c.metadata['nbar']['PFS-ELG']*=factor
        c.derivatives*=density_column_scales(c.names,factor)[None,None,:]
        c.covariance/=np.sqrt(factor)
        c.F_like=fisher_from_shells(c.derivatives,c.covariance,result.likelihood.grid.centers)
        parts.append({'cell':c})
    return parts


@pytest.mark.parametrize('prior_mode',['dimensionless','fixed_physical'])
def test_bin_recombination_matches_full_inverse_and_shared_prior(prior_mode):
    from scripts.pfs_density_inference import density_bin_covariance, full_density_covariance
    result=toy_result();parts=changed_parts(result,4.)
    out=density_bin_covariance(result,parts,4.,prior_mode)
    full=full_density_covariance(result,parts,4.,prior_mode)
    np.testing.assert_allclose(out['cosmological_covariance'],full['cov_cosmo_marg'],rtol=1e-11,atol=1e-12)
    scales=[1,1,1,1,1,1/4,1/2]
    expected=1/(4.**2) if prior_mode=='fixed_physical' else 1.
    assert full['F_prior_nuisance'][5,5]==expected
    assert full['F_prior_cosmo'][1,1]==1/25
    np.testing.assert_array_equal(out['growth_covariance'],result.growth_jacobian@out['cosmological_covariance']@result.growth_jacobian.T)


def test_fixed_physical_noise_prior_is_coordinate_invariant():
    from scripts.pfs_density_inference import density_bin_covariance, full_density_covariance
    result=toy_result();parts=changed_parts(result,4.)
    fixed=density_bin_covariance(result,parts,4.,'fixed_physical')
    # Express all nuisance amplitudes in their original physical-power units.
    for part in parts:
        c=part['cell'];c.derivatives[:,:,-2:]*=[4.,2.]
        c.F_like=fisher_from_shells(c.derivatives,c.covariance,result.likelihood.grid.centers)
    original_coordinates=full_density_covariance(result,parts,4.,'dimensionless')
    np.testing.assert_allclose(fixed['cosmological_covariance'],original_coordinates['cov_cosmo_marg'],rtol=1e-11)


def test_unit_density_recovers_reference_and_rejects_incomplete_or_coupled_bin():
    from scripts.pfs_density_inference import density_bin_covariance, verify_reference
    result=toy_result();parts=changed_parts(result,1.)
    assert verify_reference(result)['status']=='passed'
    out=density_bin_covariance(result,parts,1.,'dimensionless')
    np.testing.assert_allclose(out['cosmological_covariance'],result.cov_cosmo_marg,rtol=1e-11)
    with pytest.raises(ValueError,match='complete PFS'): density_bin_covariance(result,parts[:1],1.,'dimensionless')
    with pytest.raises(ValueError,match='duplicate'): density_bin_covariance(result,parts+parts[:1],1.,'dimensionless')
    with pytest.raises(ValueError,match='prior'): density_bin_covariance(result,parts,1.,'unrecognized')
    result.likelihood.F_like[3,4]=result.likelihood.F_like[4,3]=.1
    with pytest.raises(ValueError,match='cross-bin'): verify_reference(result)


def test_windows_require_complete_radial_support_and_density_count():
    from scripts.pfs_density_inference import window_cells, window_count
    result=toy_result()
    assert len(window_cells(result,(1.,1.2)))==2
    assert window_count(result,(1.,1.2))==40000.
    with pytest.raises(ValueError,match='cut'): window_cells(result,(1.05,1.2))
    with pytest.raises(ValueError,match='PFS'): window_cells(result,(.8,1.))


def test_saved_list_and_live_tuple_metadata_have_identical_physical_support():
    from scripts.pfs_density_inference import density_bin_covariance
    result=toy_result()
    for c in result.likelihood.cells:
        c.metadata['theoretical_pairs']=[['DESI-QSO','PFS-ELG']]
    parts=changed_parts(result,2.)
    for part in parts: part['cell'].metadata['theoretical_pairs']=[('DESI-QSO','PFS-ELG')]
    assert np.isfinite(density_bin_covariance(result,parts,2.,'dimensionless')['cosmological_covariance']).all()


def test_input_binding_rejects_tampering_and_existing_output(tmp_path):
    from scripts.run_pfs_density_scan import InputBinding, fresh_output
    p=tmp_path/'source.json';p.write_text('{}')
    binding=InputBinding({str(p):hashlib.sha256(p.read_bytes()).hexdigest()})
    assert binding.check(p)==p
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'): binding.check(p)
    output=tmp_path/'output';fresh_output(output)
    with pytest.raises(FileExistsError): fresh_output(output)
