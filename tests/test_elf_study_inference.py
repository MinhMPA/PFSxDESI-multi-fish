"""Shared regional parameters, proper priors and derived growth information."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from pfsfog.cosmo import FIDUCIAL
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from pfsfog.revision_forecast import CellLikelihood,make_shell_grid,project_baseline_precision
from scripts.elf_study_profiles import paper_profiles,highz_profiles,study_config,study_cells
from scripts.elf_study_inference import assemble_study,infer_study,validate_growth

SETTINGS={'nfft':2048,'n_mu':4,'dk':.01}

def analytic(profile):
    cosmo=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.7)
    g=make_shell_grid(.01,profile.kmax,SETTINGS['dk']);parts=[]
    for case in study_cells(profile,cosmo,settings=SETTINGS):
        row=case['expected'];names=row['local_names'];nobs=3*len(row['retained_pairs'])
        d=np.zeros((len(g.centers),nobs,len(names)));c=np.tile(np.eye(nobs),(len(g.centers),1,1))
        if names:
            d[:,:,0]=1.
            for i,n in enumerate(names):
                if n.endswith(':b1_sigma8'):
                    population=n.split(':')[1]
                    row_index=3*row['retained_pairs'].index([population,population])
                    d[:,row_index,i]=2.+.05*np.arange(len(g.centers))
        f=np.einsum('koi,koj->ij',d,d)
        values={n: FIDUCIAL[{'Mnu':'mnu','Omegam':'Omega_m'}.get(n,n)] if n in names[:3]
                else 0. if '|' in n else 1. for n in names}
        v=dict(expected_shells=len(g.centers) if names else 0,processed_shells=len(g.centers) if names else 0,
            field_min_scaled_eigenvalue=1.,covariance_min_scaled_eigenvalue=1.)
        parts.append({'cell':CellLikelihood(f,names,deepcopy(row),v,d,c,np.ones((len(g.centers),nobs))),
                      'fiducials':values})
    return parts


def test_regional_fisher_sums_and_shared_priors_enter_once():
    p=paper_profiles()['B'];parts=analytic(p)
    like=assemble_study(p,SETTINGS,parts,FIDUCIAL,{'engine':'elf.EPT'})
    name='bin00:DESI-LRG:b1_sigma8';i=like.names.index(name)
    assert like.names.count(name)==1 and like.F_like[1,1]==0
    wanted=sum(c['cell'].F_like[c['cell'].names.index(name),c['cell'].names.index(name)]
               for c in parts if name in c['cell'].names)
    assert like.F_like[i,i]==wanted
    base=project_baseline_precision(like,1.,fisher_solver=DECIMAL_FISHER_SOLVER)
    for factor in (3,10):
        other=project_baseline_precision(like,factor,fisher_solver=DECIMAL_FISHER_SOLVER)
        np.testing.assert_allclose(other['F_prior_nuisance'],base['F_prior_nuisance']/factor**2)
        np.testing.assert_array_equal(other['F_prior_cosmo'],base['F_prior_cosmo'])
    assert base['cov_cosmo_marg'][0,0] > base['cov_cosmo_nuisance_fixed'][0,0]


@pytest.mark.parametrize('damage',['missing','duplicate','fiducial','cosmology','F','covariance','empty','coverage'])
def test_assembly_rejects_missing_inconsistent_or_invented_information(damage):
    p=highz_profiles()['C22'];parts=analytic(p)
    if damage=='missing': parts.pop()
    if damage=='duplicate': parts[-1]=deepcopy(parts[0])
    if damage=='fiducial': parts[1]['fiducials']['bin00:DESI-LRG:b1_sigma8']+=.1
    if damage=='cosmology': parts[0]['fiducials']['Mnu']+=.1
    if damage=='F': parts[0]['cell'].F_like[0,0]+=1
    if damage=='covariance': parts[0]['cell'].covariance[0,0,0]=-1
    if damage=='empty': parts[-1]['cell'].names=['invented']
    if damage=='coverage': parts[0]['cell'].validation['processed_shells']-=1
    with pytest.raises(ValueError): assemble_study(p,SETTINGS,parts,FIDUCIAL,{'engine':'elf.EPT'})


def test_terminal_empty_cell_has_no_nuisance_or_information():
    p=highz_profiles()['C22'];like=assemble_study(p,SETTINGS,analytic(p),FIDUCIAL,{})
    assert len(like.cells)==22 and like.cells[-1].names==[]
    assert len([n for n in like.names if n.startswith('bin08:')])==11
    assert not any('DESI' in n for n in like.names if n.startswith('bin08:'))


def test_growth_full_correlations_and_arithmetic_rejection():
    j=np.array([[1.,2.,3.],[2.,3.,4.]])
    result=SimpleNamespace(growth_z=np.array([1.,2.]),growth_fiducial=np.array([.4,.3]),
        growth_jacobian=j,cov_cosmo_marg=np.eye(3),cov_cosmo_nuisance_fixed=np.eye(3)/2,
        growth_cov_marg=j@j.T,growth_cov_nuisance_fixed=j@j.T/2)
    assert validate_growth(result)['rank_upper_bound']==3
    result.growth_cov_marg[0,1]+=1
    with pytest.raises(ValueError,match='growth'): validate_growth(result)


def test_scale_cut_assembles_from_exactly_truncated_parent_arrays():
    from scripts.elf_study_profiles import study_shell_grid
    from scripts.elf_study_inference import validate_study_likelihood
    p=paper_profiles()['B_k015'];parts=analytic(paper_profiles()['B'])
    grid=study_shell_grid(study_config(p,SETTINGS));nk=len(grid.centers)
    for part in parts:
        cell=part['cell'];cell.derivatives=cell.derivatives[:nk]
        cell.covariance=cell.covariance[:nk];cell.fiducial_mean=cell.fiducial_mean[:nk]
        cell.F_like=np.einsum('koi,koj->ij',cell.derivatives,cell.derivatives)
        cell.validation.update(expected_shells=nk,processed_shells=nk)
    like=assemble_study(p,SETTINGS,parts,FIDUCIAL,{})
    validate_study_likelihood(like)
    assert like.config.kmax==.15
    np.testing.assert_array_equal(like.grid.centers,grid.centers)
