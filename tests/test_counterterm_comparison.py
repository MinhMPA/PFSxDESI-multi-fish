"""Exact physical correction, source authentication and resumable inventories."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest

OLD=Path('/Users/nguyenmn/PFSxDESI-multi-fish/.worktrees/elf-engine')

def test_unequal_real_space_correction_is_not_old_pair_average():
    from scripts.counterterm_comparison import counterterm_difference
    from pfsfog.elf_adapter import ELFPowerSpectrum
    import jax.numpy as jnp
    model=ELFPowerSpectrum(nfft=64).model
    a={'bias':{'b1':1.},'ctr':{'c0':2.,'c2':0.,'c4':0.,'cfog':0.}}
    b={'bias':{'b1':3.},'ctr':{'c0':8.,'c2':0.,'c4':0.,'cfog':0.}}
    k=jnp.array([[.1]]);mu=jnp.array([[.4]]);pk=jnp.array([[100.]])
    # new=-k²P*(1*8/3+3*2), old=-k²P*(2+8)
    got=counterterm_difference(model,k,mu,0.,a,b,pk)
    np.testing.assert_allclose(got,[[4/3]],rtol=1e-13)

@pytest.fixture
def source():
    from scripts.elf_study_evidence import load_source_cell
    p=OLD/'results/elf-survey-studies-20261010-01/inference/loop2048'
    m=json.loads((p/'profiles/B/manifest.json').read_text())
    refs=[json.loads((p/'projections'/f'{h}.json').read_text()) for h in m['projections']]
    ref=next(r for r in refs if r['source_directory'] and len(r['expected']['active_populations'])==2)
    return load_source_cell(ref['source_directory'],ref['source_record'])

def test_correction_preserves_intrinsic_auto_multipoles_and_derivatives(source):
    from scripts.counterterm_comparison import Correction
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.cosmo import FiducialCosmology
    import jax
    c=Correction(source,ELFPowerSpectrum(nfft=2048),FiducialCosmology())
    fid=source['arrays']['fiducial'];pairs=source['inputs']['metadata']['retained_pairs']
    delta=np.asarray(c.delta_mean(fid,ap=False));jac=np.asarray(jax.jacfwd(lambda x:c.delta_mean(x,ap=False))(fid))
    autos=[3*i+j for i,p in enumerate(pairs) if p[0]==p[1] for j in range(3)]
    assert np.max(abs(delta[:,autos]))<1e-9
    assert np.max(abs(jac[:,autos]))<1e-7
    assert np.max(abs(np.asarray(c.delta_pkmu(fid,source['arrays']['signed_mu']))))>1e-5

def test_request_rejects_changed_input_and_mixed_track(tmp_path):
    from scripts.counterterm_comparison import validate_request
    from scripts.run_pfs_density_scan import digest,write_json
    f=tmp_path/'source';f.write_text('original')
    request={'track':'CT-A','bindings':{str(f):digest(f)}}
    write_json(tmp_path/'request.json',request)
    write_json(tmp_path/'state.json',{'request_sha256':digest(tmp_path/'request.json')})
    validate_request(tmp_path,'CT-A')
    with pytest.raises(ValueError,match='track'):validate_request(tmp_path,'CT-B')
    f.write_text('changed')
    with pytest.raises(ValueError,match='changed'):validate_request(tmp_path,'CT-A')

def test_duplicate_cells_rejected():
    from scripts.counterterm_comparison import unique_cell_records
    with pytest.raises(ValueError,match='duplicate'):
        unique_cell_records([{'cell_id':'x'},{'cell_id':'x'}])

@pytest.mark.parametrize('profile',['B','C'])
def test_exact_additive_matches_fresh_complete_elf_mean_and_all_derivatives(profile):
    from scripts.counterterm_comparison import Correction
    from scripts.elf_study_evidence import load_source_cell
    from scripts.elf_study_physical import common_query_data
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.cosmo import FiducialCosmology
    import jax
    p=OLD/'results/elf-survey-studies-20261010-01/inference/loop2048'
    m=json.loads((p/f'profiles/{profile}/manifest.json').read_text())
    refs=[json.loads((p/'projections'/f'{h}.json').read_text()) for h in m['projections']]
    r=next(r for r in refs if r['source_directory'] and len(r['expected']['active_populations'])==2
           and (profile=='B' or 'PFS-ELG' in r['expected']['active_populations']))
    s=load_source_cell(r['source_directory'],r['source_record']);a=s['arrays']
    engine=ELFPowerSpectrum(nfft=2048);cosmo=FiducialCosmology();c=Correction(s,engine,cosmo)
    ix=np.array([0,len(a['k'])//2,len(a['k'])-1]);fresh=common_query_data(s['inputs'],a['k'][ix],engine,cosmo)
    mean=a['mean']+np.asarray(jax.jit(c.delta_mean)(a['fiducial']))
    D=a['D']+np.asarray(jax.jit(jax.jacfwd(c.delta_mean))(a['fiducial']))
    np.testing.assert_allclose(mean[ix],fresh['mean'],rtol=2e-11,atol=1e-8)
    np.testing.assert_allclose(D[ix],fresh['D'],rtol=2e-8,atol=2e-6)

def test_refinement_requires_errors_and_gains_and_complete_rows():
    from scripts.run_counterterm_comparison import convergence
    base=[]
    for grid in ('loop2048','loop4096','angular64','shell304','final-loop'):
        for i,case in enumerate('ABCD'):
            base.append({'grid':grid,'public_case':case,'model':'CT-A',
                         'sigma_marginalized':{'x':1-i*.1},'sigma_nuisance_fixed':{'x':1-i*.05}})
    assert convergence(base)['status']=='passed'
    wrong=deepcopy(base);wrong[-1]['sigma_marginalized']['x']*=1.002
    assert convergence(wrong)['status']=='failed' # <.5% errors, but >.1 pp gain
    with pytest.raises(ValueError,match='incomplete'):convergence(base[:-1])

def test_physical_receipt_checks_own_files_without_self_hash(tmp_path):
    from scripts.run_counterterm_comparison import verify_physical_files
    from scripts.run_pfs_density_scan import digest,write_json
    a=tmp_path/'arrays.npz';a.write_bytes(b'data')
    r={'files':{'arrays.npz':digest(a)}};write_json(tmp_path/'receipt.json',r)
    verify_physical_files(tmp_path,r)
    a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_physical_files(tmp_path,r)

def test_complete_six_coordinate_corrected_cell_passes_physical_audit(source):
    from scripts.counterterm_comparison import correct_source
    from scripts.elf_study_data import project_cell
    from scripts.elf_study_profiles import paper_profiles,study_shell_grid
    from scripts.takada_study import corrected_config
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.cosmo import FiducialCosmology
    new=correct_source(source,ELFPowerSpectrum(nfft=2048),FiducialCosmology())
    row={**new['inputs']['metadata'],'empty_observation':False};p=paper_profiles()['B']
    part=project_cell(new,p,row,study_shell_grid(corrected_config(p,{'nfft':2048,'n_mu':32,'dk':.00125})))
    assert part['derivative_check']['status']=='passed'
    assert part['neighborhood_check']['processed_points']==1+6*len(row['local_names'])

def test_unknown_global_coordinate_remains_rejected():
    from scripts.elf_study_data import check_saved_derivatives
    from pfsfog.revision_forecast import CellLikelihood
    D=np.ones((1,3,1));C=np.eye(3)[None];mean=np.ones((1,3));steps=np.ones((1,3))
    plus=np.ones((1,3,1,3));minus=-plus
    cell=CellLikelihood(np.ones((1,1)),['unrecognized_global'],{'retained_pairs':[['X','X']]},{},D,C,mean)
    with pytest.raises(ValueError,match='unrecognized'):
        check_saved_derivatives({'cell':cell,'analytic_zero_support':[],
            'arrays':{'fd_steps':steps,'fd_plus':plus,'fd_minus':minus,'fd_derivatives':plus}})


def test_geometry_response_is_direct_AP_derivative_and_nuisance_null(source):
    from scripts.counterterm_comparison import correct_source,Correction
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.cosmo import FiducialCosmology
    import jax
    import jax.numpy as jnp
    engine=ELFPowerSpectrum(nfft=2048);cosmo=FiducialCosmology()
    c=Correction(source,engine,cosmo);fid=source['arrays']['fiducial']
    changed=correct_source(source,engine,cosmo)
    nuisance=[i for i,n in enumerate(c.names) if ':' in n]
    np.testing.assert_array_equal(changed['arrays']['D_geometry'][...,nuisance],source['arrays']['D_geometry'][...,nuisance])
    expected=np.asarray(jax.jit(jax.jacfwd(lambda x:c.delta_mean(fid,ap_theta=x[jnp.array(c.cidx)])))(fid))
    np.testing.assert_array_equal(changed['arrays']['D_geometry'],source['arrays']['D_geometry']+expected)


def test_intrinsic_geometry_chain_and_nonzero_geometry_FD(source):
    import scripts.counterterm_comparison as module
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.cosmo import FiducialCosmology
    changed=module.correct_source(source,ELFPowerSpectrum(nfft=2048),FiducialCosmology())
    assert hasattr(module,'check_response_split'), 'corrected intrinsic/AP split audit absent'
    report=module.check_response_split(changed)
    assert report['status']=='passed' and report['chain_relative_error']<1e-10
    assert report['h_geometry']['status']=='passed' and report['h_intrinsic']['status']=='passed'
    i=changed['inputs']['metadata']['local_names'].index('h')
    assert np.max(abs(changed['arrays']['D_geometry'][...,i]))>1.
    bad={**changed,'arrays':dict(changed['arrays'])}
    bad['arrays']['D_geometry']=changed['arrays']['D_geometry'].copy()
    bad['arrays']['D_geometry'][...,i]+=1.
    with pytest.raises(ValueError,match='chain'):module.check_response_split(bad)
