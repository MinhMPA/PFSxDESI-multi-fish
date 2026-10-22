"""Exact changes of Gaussian data vector, with explicit physical checks."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from pfsfog.revision_forecast import make_shell_grid
from pfsfog.revision_validation import compare_cell_derivatives
from scripts.elf_study_profiles import paper_profiles
from scripts.elf_study_data import select_cell, project_cell, check_saved_derivatives, check_neighborhood


def toy():
    grid=make_shell_grid(.01,.03,.01)
    names=['ln10_10_As','Mnu','Omegam','h','omega_b','n_s','bin00:A:b1_sigma8','bin00:A|B:N_ab']
    pairs=[['A','A'],['B','B'],['A','B']]
    meta=dict(cell_id='bin00-radial00-overlap',nominal_bin_index=0,nominal_bin=[.4,.6],
        radial_bin=[.4,.6],region='overlap',model_redshift=.5,volume=1.,nbar={'A':1.,'B':1.},
        active_populations=['A','B'],candidate_populations=['A','B'],retained_pairs=pairs,
        theoretical_pairs=pairs,local_names=names)
    d=np.zeros((2,9,8));d[:,:,0]=1.;d[:,:3,6]=2.;d[:,6:,7]=3.;d[:,:,3:6]=.2
    mean=np.arange(18.).reshape(2,9)+10.
    c=np.broadcast_to(np.eye(9)+.2,(2,9,9)).copy()
    fid=np.zeros(8);steps=np.broadcast_to([.001,.0005,.00025],(8,3)).copy()
    fd=np.broadcast_to(d.transpose(2,0,1)[:,None],(8,3,2,9)).copy()
    plus=mean[None,None]+steps[:,:,None,None]*fd;minus=mean[None,None]-steps[:,:,None,None]*fd
    arrays=dict(k=grid.centers,dk=grid.widths,mean=mean,D=d,C=c,fiducial=fid,fd_steps=steps,
        fd_plus=plus,fd_minus=minus,fd_derivatives=(plus-minus)/(2*steps[:,:,None,None]))
    source=dict(arrays=arrays,inputs={'metadata':meta,'fiducials':dict(zip(names,fid)),
        'analytic_zero_response':{'support':[]}},measurement={})
    expected=deepcopy(meta);expected.update(local_names=names[:3]+names[6:7],
        retained_pairs=pairs[:2],empty_observation=False,absent_observations=[[pairs[2],ell] for ell in (0,2,4)])
    return source,expected,grid


def test_row_first_fisher_is_not_a_full_fisher_parameter_block():
    src,row,g=toy();out=select_cell(src,paper_profiles()['A'],row,g)
    D=src['arrays']['D'][:,:6][:,:,[0,1,2,6]];C=src['arrays']['C'][:,:6,:6]
    direct=np.einsum('koi,koj->ij',D,np.linalg.solve(C,D))
    np.testing.assert_allclose(out['cell'].F_like,direct,rtol=1e-14)
    full=np.einsum('koi,koj->ij',src['arrays']['D'],np.linalg.solve(src['arrays']['C'],src['arrays']['D']))
    assert not np.allclose(direct,full[np.ix_([0,1,2,6],[0,1,2,6])])
    np.testing.assert_array_equal(out['cell'].covariance,C)


def test_independent_auto_blocks_keep_internal_multipole_covariance():
    src,row,g=toy();out=select_cell(src,paper_profiles()['I'],row,g)
    C=out['cell'].covariance
    np.testing.assert_array_equal(C[:,:3,:3],src['arrays']['C'][:,:3,:3])
    assert np.count_nonzero(C[:,:3,3:])==0


def test_exact_shell_selection_and_nonzero_omitted_nuisance_rejection():
    src,row,g=toy();sub=make_shell_grid(.01,.02,.01)
    assert select_cell(src,paper_profiles()['A'],row,sub)['cell'].derivatives.shape[0]==1
    shifted=replace(sub,centers=sub.centers+1e-12)
    with pytest.raises(ValueError,match='shell'): select_cell(src,paper_profiles()['A'],row,shifted)
    src['arrays']['D'][0,0,7]=1e-8
    with pytest.raises(ValueError,match='omitted nuisance'): select_cell(src,paper_profiles()['A'],row,g)


def test_fd_rechecked_in_selected_covariance_and_requires_adjacent_steps():
    src,row,g=toy();out=select_cell(src,paper_profiles()['A'],row,g)
    assert check_saved_derivatives(out)['status']=='passed'
    # Good first and third steps do not qualify a failed middle step.
    out['arrays']['fd_plus'][0,1]+=0.01
    a=out['arrays'];a['fd_derivatives']=(a['fd_plus']-a['fd_minus'])/(2*a['fd_steps'][:,:,None,None])
    with pytest.raises(ValueError,match='finite difference'): check_saved_derivatives(out)


def test_angular_zero_requires_both_ad_and_fd_to_vanish():
    src,row,g=toy();out=select_cell(src,paper_profiles()['A'],row,g)
    out['analytic_zero_support']=[['ln10_10_As',['A','A'],0]]
    with pytest.raises(ValueError,match='finite difference'): check_saved_derivatives(out)


def test_omitted_nuisance_requires_zero_signed_response_too():
    src,row,g=toy();src['arrays']['fd_plus'][7,0,0,0]+=1e-8
    with pytest.raises(ValueError,match='omitted nuisance'): select_cell(src,paper_profiles()['A'],row,g)


def test_empty_region_has_no_consumable_information_or_prior_coordinates():
    src,row,g=toy();row.update(retained_pairs=[],local_names=[],active_populations=[],nbar={},empty_observation=True)
    part=project_cell(None,paper_profiles()['C'],row,g)
    assert part['cell'].names==[] and part['cell'].F_like.size==0 and not part['fiducials']


def test_vectorized_fd_agrees_with_reference_full_and_pair_metrics():
    src,row,g=toy();out=select_cell(src,paper_profiles()['A'],row,g)
    report=check_saved_derivatives(out);cell=out['cell'];a=out['arrays'];fids=out['fiducials']
    def mean(pair):
        pos=row['retained_pairs'].index(list(pair))*3
        def evaluate(point):
            changed=[i for i,n in enumerate(cell.names) if point[n]!=fids[n]]
            i=changed[0];delta=point[cell.names[i]]-fids[cell.names[i]]
            j=np.argmin(abs(a['fd_steps'][i]-abs(delta)))
            return a['fd_plus' if delta>0 else 'fd_minus'][i,j,:,pos:pos+3]
        return evaluate
    ref=compare_cell_derivatives(cell,{tuple(p):mean(p) for p in row['retained_pairs']},fids,
        dict(zip(cell.names,a['fd_steps'])),configuration='test',physical_scale=report['physical_scale'])
    assert ref['status']=='passed'
    np.testing.assert_allclose([r['step_errors'] for r in report['column_results']],
        [r['step_errors'] for r in ref['column_results']],rtol=1e-9,atol=1e-15)


def test_covariance_uses_unmeasured_theoretical_cross_and_rejects_negative_field():
    from pfsfog.elf_covariance_audit import covariance_from_spectra
    src,row,g=toy();meta=deepcopy(src['inputs']['metadata']);meta['retained_pairs']=row['retained_pairs']
    signals=np.ones((3,2,8));signals[:2]*=2
    _,C=covariance_from_spectra(signals,meta,g.centers,g.widths,n_mu=8)
    assert np.any(C[:,:3,3:]!=0)
    src['arrays']['C']=covariance_from_spectra(signals,src['inputs']['metadata'],g.centers,g.widths,n_mu=8)[1]
    coordinates=[np.zeros(8)]
    for i in range(8):
        for step in src['arrays']['fd_steps'][i]:
            for sign in (1,-1):
                point=np.zeros(8);point[i]=sign*step;coordinates.append(point)
    source={'inputs':src['inputs'],'neighborhood_spectra':np.stack([signals]*49),
        'neighborhood_coordinates':np.array(coordinates)}
    out=select_cell(src,paper_profiles()['A'],row,g)
    assert check_neighborhood(source,out,n_mu=8)['processed_points']==25
    source['neighborhood_spectra'][1,2]=4  # field cross exceeds shot-noise-inclusive autos (3)
    with pytest.raises(ValueError,match='nonpositive'):
        check_neighborhood(source,out,n_mu=8)
