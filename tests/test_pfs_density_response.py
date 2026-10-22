"""Physical density identities, with independent linear spectra and ELF checks."""
from copy import deepcopy

import numpy as np
import pytest

from pfsfog.elf_covariance_audit import covariance_from_spectra
from pfsfog.revision_forecast import make_shell_grid
from scripts.elf_study_profiles import paper_profiles


def physical_toy():
    grid = make_shell_grid(.05, .07, .01)
    mu, weight = np.polynomial.legendre.leggauss(8)
    leg = np.array([np.ones(8), (3*mu**2-1)/2, (35*mu**4-30*mu**2+3)/8])
    def project(pk):
        return np.einsum('pkm,lm,m,l->kpl', pk, leg, weight/2, [1, 5, 9]).reshape(2, 9)
    names = ['ln10_10_As', 'Mnu', 'Omegam', 'bin00:DESI-QSO:Pshot',
             'bin00:PFS-ELG:Pshot', 'bin00:PFS-ELG:a0', 'bin00:PFS-ELG:a2',
             'bin00:DESI-QSO|PFS-ELG:Pshot_cross', 'bin00:DESI-QSO|PFS-ELG:a2_cross']
    pairs = [['DESI-QSO', 'DESI-QSO'], ['PFS-ELG', 'PFS-ELG'], ['DESI-QSO', 'PFS-ELG']]
    meta = dict(cell_id='bin00-radial00-overlap', nominal_bin_index=0,
                nominal_bin=[1.6, 2.0], radial_bin=[1.6, 2.0], region='overlap',
                model_redshift=1.8, volume=2.e8,
                nbar={'DESI-QSO':1.e-4, 'PFS-ELG':2.e-4},
                active_populations=['DESI-QSO', 'PFS-ELG'],
                candidate_populations=['DESI-QSO', 'PFS-ELG'],
                theoretical_pairs=pairs, retained_pairs=pairs, local_names=names)
    k = grid.centers[:, None]
    pm = 500*(k/.06)**-.8
    d, p = 2.2+.8*mu**2, 1.3+.8*mu**2
    signal = np.array([pm*d*d, pm*p*p, pm*d*p])
    derivatives = np.zeros((len(names), *signal.shape))
    derivatives[0] = signal
    derivatives[1] = signal*k
    derivatives[2] = signal*mu**2
    derivatives[3, 0] = 1.e4
    derivatives[4, 1] = 5.e3
    derivatives[5, 1] = 5.e3*(k/.7)**2
    derivatives[6, 1] = 5.e3*(k/.7)**2*mu**2
    derivatives[7, 2] = 1/np.sqrt(2.e-8)
    derivatives[8, 2] = (k/.7)**2*mu**2/np.sqrt(2.e-8)
    D = np.stack([project(a) for a in derivatives], axis=-1)
    mean = project(signal)
    steps = np.tile([.001, .0005, .00025], (len(names), 1))
    plus = mean[None, None]+steps[:, :, None, None]*D.transpose(2, 0, 1)[:, None]
    minus = mean[None, None]-steps[:, :, None, None]*D.transpose(2, 0, 1)[:, None]
    neighbors, coordinates = [signal], [np.zeros(len(names))]
    for i in range(len(names)):
        for step in steps[i]:
            for sign in (1, -1):
                neighbors.append(signal+sign*step*derivatives[i])
                point=np.zeros(len(names)); point[i]=sign*step; coordinates.append(point)
    arrays = dict(k=grid.centers, dk=grid.widths, signed_mu=mu, pkmu=signal,
                  mean=mean, D=D, D_intrinsic=D.copy(), D_geometry=np.zeros_like(D),
                  C=covariance_from_spectra(signal, meta, grid.centers, grid.widths, n_mu=8)[1],
                  fiducial=np.zeros(len(names)), fd_steps=steps, fd_plus=plus, fd_minus=minus,
                  fd_derivatives=(plus-minus)/(2*steps[:, :, None, None]))
    source=dict(inputs={'metadata':meta, 'fiducials':dict.fromkeys(names, 0.),
                        'configuration':{'n_mu':8}, 'analytic_zero_response':{'support':[]}},
                arrays=arrays, neighborhood_spectra=np.stack(neighbors),
                neighborhood_coordinates=np.array(coordinates))
    expected=deepcopy(meta);expected.update(empty_observation=False, absent_observations=[])
    return source, expected, grid


def test_density_scaling_distinguishes_auto_cross_and_unrelated_columns():
    from scripts.pfs_density_response import density_column_scales
    source, _, _=physical_toy()
    np.testing.assert_allclose(density_column_scales(source['inputs']['metadata']['local_names'], 4.),
                               [1, 1, 1, 1, .25, .25, .25, .5, .5])


def test_unit_factor_is_identity_and_input_is_not_mutated():
    from scripts.pfs_density_response import rescale_density_source
    source, _, _=physical_toy(); saved=deepcopy(source)
    same=rescale_density_source(source, 1.)
    for name, arr in source['arrays'].items(): np.testing.assert_array_equal(same['arrays'][name], arr)
    changed=rescale_density_source(source, 4.)
    assert changed['inputs']['metadata']['nbar']['PFS-ELG']==8.e-4
    assert source['inputs']==saved['inputs']
    for name, arr in source['arrays'].items(): np.testing.assert_array_equal(arr, saved['arrays'][name])
    np.testing.assert_array_equal(changed['arrays']['D'][:, :, :4], source['arrays']['D'][:, :, :4])


def test_density_cell_retains_theoretical_cross_in_autos_covariance():
    from scripts.pfs_density_response import density_cell
    source, row, grid=physical_toy()
    row['retained_pairs']=row['retained_pairs'][:2]
    row['local_names']=row['local_names'][:-2]
    out=density_cell(source, paper_profiles()['D'], row, grid, 4.)
    c=out['cell'];s=source['arrays']['pkmu'];n=c.metadata['nbar']
    modes=grid.centers**2*grid.widths*row['volume']/(2*np.pi**2)
    w=np.polynomial.legendre.leggauss(8)[1]
    np.testing.assert_allclose(c.covariance[:, 0, 3], np.sum(w*s[2]**2, axis=1)/modes)
    np.testing.assert_allclose(c.covariance[:, 3, 3], np.sum(w*(s[1]+1/n['PFS-ELG'])**2, axis=1)/modes)
    assert out['derivative_check']['status']=='passed'
    assert out['neighborhood_check']['processed_points']==1+6*len(c.names)


def test_signed_stochastic_neighborhoods_obey_the_same_density_scaling():
    from scripts.pfs_density_response import rescale_density_source
    source, _, _=physical_toy();changed=rescale_density_source(source, 4.)
    for index, scale in ((1, 1.), (1+6*4, .25), (1+6*7, .5)):
        expected=source['neighborhood_spectra'][0]+scale*(source['neighborhood_spectra'][index]-source['neighborhood_spectra'][0])
        np.testing.assert_allclose(changed['neighborhood_spectra'][index], expected, rtol=1.e-14)


@pytest.mark.parametrize('factor', [0., -1., np.inf, np.nan, True])
def test_invalid_density_factor_rejected(factor):
    from scripts.pfs_density_response import rescale_density_source
    with pytest.raises(ValueError, match='factor'): rescale_density_source(physical_toy()[0], factor)


def test_nonzero_stochastic_fiducial_and_nonphysical_neighborhood_rejected():
    from scripts.pfs_density_response import rescale_density_source, density_cell
    source, row, grid=physical_toy()
    source['inputs']['fiducials']['bin00:PFS-ELG:Pshot']=.1
    with pytest.raises(ValueError, match='zero stochastic'): rescale_density_source(source, 2.)
    source, row, grid=physical_toy();source['neighborhood_spectra'][1, 2]*=100.
    with pytest.raises(ValueError, match='nonpositive'): density_cell(source, paper_profiles()['C'], row, grid, 2.)


def test_density_law_matches_direct_elf_with_nontrivial_ap():
    from scripts.pfs_density_response import density_column_scales
    import jax.numpy as jnp
    from pfsfog.elf_adapter import ELFPowerSpectrum
    engine=ELFPowerSpectrum(nfft=512)
    kgrid=jnp.geomspace(1.e-5, 100., 512)
    pd={'k':kgrid, 'pk':2.e4*(kgrid/.05)**.96/(1+(kgrid/.05)**2)**2}
    k=jnp.array([.06, .12]); mu=jnp.array([.1, .6, .9])
    p=dict(h=.6736, f=.8, bias=dict(b1=1.3,b2=.2,bG2=-.1,bGamma3=.05),
           ctr=dict(c0=0.,c2=0.,c4=0.,cfog=100.),
           stoch=dict(P_shot=.2,a0=-.3,a2=.4),ndens=2.e-4,k_nl=.7)
    signal=engine.get_pkmu_pair_ref(k,mu,1.02,.97,pd,p)
    before=engine.get_pkmu_pair_ref(k,mu,1.02,.97,pd,p,add_stochasticity=True)
    after=engine.get_pkmu_pair_ref(k,mu,1.02,.97,pd,{**p,'ndens':8.e-4},add_stochasticity=True)
    s=density_column_scales(['bin00:PFS-ELG:Pshot'],4.)[0]
    np.testing.assert_allclose(after-signal,s*(before-signal),rtol=1e-10,atol=1e-9)


def test_gamma3_hexadecapole_zero_requires_identity_ap_and_ir_off():
    from scripts.pfs_density_response import gamma3_zero_support
    inputs={'metadata':{'local_names':['bin00:PFS-ELG:bGamma3','bin00:PFS-ELG:b2_sigma8sq'],
                        'retained_pairs':[['PFS-ELG','PFS-ELG'],['DESI-QSO','PFS-ELG']]},
            'analytic_zero_response':{'alphas':[1.,1.]},
            'engine_settings':{'engine':'elf.EPT','do_irres':False}}
    assert gamma3_zero_support(inputs)==[
        ('bin00:PFS-ELG:bGamma3',('PFS-ELG','PFS-ELG'),4),
        ('bin00:PFS-ELG:bGamma3',('DESI-QSO','PFS-ELG'),4)]
    for key,value in [('alphas',[1.02,.97]),('ir',True),('engine','other')]:
        changed=deepcopy(inputs)
        if key=='alphas':changed['analytic_zero_response']['alphas']=value
        else:changed['engine_settings']['do_irres' if key=='ir' else key]=value
        assert gamma3_zero_support(changed)==[]


def test_gamma3_zero_is_checked_absolutely_when_subtraction_noise_exceeds_tiny_cut():
    from scripts.pfs_density_response import density_cell
    source,row,grid=physical_toy();old='bin00:PFS-ELG:a2';new='bin00:PFS-ELG:bGamma3'
    source['inputs']['metadata']['local_names'][6]=new;row['local_names'][6]=new
    source['inputs']['fiducials'][new]=source['inputs']['fiducials'].pop(old)
    source['inputs']['engine_settings']={'engine':'elf.EPT','do_irres':False}
    source['inputs']['analytic_zero_response']['alphas']=[1.,1.]
    a=source['arrays'];step=a['fd_steps'][6,:,None]
    a['fd_plus'][6,:,:,5]+=step*1e-5;a['fd_minus'][6,:,:,5]-=step*1e-5
    a['fd_derivatives']=(a['fd_plus']-a['fd_minus'])/(2*a['fd_steps'][:,:,None,None])
    result=density_cell(source,paper_profiles()['C'],row,grid,10.)
    check=next(r for r in result['derivative_check']['row_results']
               if r['column']==new and r['pair']==['PFS-ELG','PFS-ELG'] and r['ell']==4)
    assert check['declared_zero'] and check['passed']
    assert check['step_tolerances']==[1e-6]*3


@pytest.mark.parametrize('cross',[False,True])
def test_direct_elf_gamma3_response_has_only_mu0_and_mu2(cross):
    from scripts.pfs_density_response import gamma3_zero_support
    import jax
    import jax.numpy as jnp
    from pfsfog.elf_adapter import ELFPowerSpectrum
    engine=ELFPowerSpectrum(nfft=512);kg=jnp.geomspace(1.e-5,100.,512)
    pd={'k':kg,'pk':2.e4*(kg/.05)**.96/(1+(kg/.05)**2)**2}
    k=jnp.array([.02,.07,.15,.2]);mu,weight=np.polynomial.legendre.leggauss(32)
    p=dict(h=.6736,f=.8,bias=dict(b1=1.3,b2=.2,bG2=-.1,bGamma3=.05),
           ctr=dict(c0=0.,c2=0.,c4=0.,cfog=100.),
           stoch=dict(P_shot=0.,a0=0.,a2=0.),ndens=2.e-4,k_nl=.7)
    if cross:p.update(bias2=dict(b1=2.4,b2=.7,bG2=-.2,bGamma3=.3),ctr2=deepcopy(p['ctr']))
    def spectrum(g):
        varied={**p,'bias':{**p['bias'],'bGamma3':g}}
        return engine.get_pkmu_pair_ref(k,jnp.asarray(mu),1.,1.,pd,varied)
    response=np.asarray(jax.jacfwd(spectrum)(.05))
    polynomial=np.stack([np.ones_like(mu),mu**2],axis=1)
    fitted=np.linalg.lstsq(polynomial,response.T,rcond=None)[0]
    np.testing.assert_allclose(response,(polynomial@fitted).T,rtol=1e-10,atol=1e-8)
    ell4=9*np.sum(response*weight*(35*mu**4-30*mu**2+3)/16,axis=1)
    assert np.max(abs(ell4)) < 1e-10*np.max(abs(response))
