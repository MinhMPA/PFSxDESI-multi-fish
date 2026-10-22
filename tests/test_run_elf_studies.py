"""Whole-study inventory, strict resolution checks and immutable evidence."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from pfsfog.cosmo import FIDUCIAL
from pfsfog.revision_forecast import project_baseline_precision,PriorResult
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from scripts.elf_study_profiles import paper_profiles,highz_profiles
from scripts.elf_study_inference import assemble_study
from scripts.elf_study_report import (all_profiles,comparisons,constraint_row,compare_results,
    save_result,load_result,verify_result,nesting_snapshot)
from scripts.run_elf_studies import fresh_directory,source_for_cell,GRID_PROFILES
from test_elf_study_inference import analytic,SETTINGS


def fake_result():
    p=paper_profiles()['B'];like=assemble_study(p,SETTINGS,analytic(p),FIDUCIAL,{})
    q=project_baseline_precision(like,1.,fisher_solver=DECIMAL_FISHER_SOLVER)
    z=np.array([sum(b)/2 for b in p.nominal_bins]);j=np.stack([z,z**2,np.ones(8)],axis=1)
    return PriorResult(likelihood=like,**{k:v for k,v in q.items() if k!='F_like'},growth_z=z,
        growth_fiducial=np.ones(8),growth_jacobian=j,growth_cov_marg=j@q['cov_cosmo_marg']@j.T,
        growth_cov_nuisance_fixed=j@q['cov_cosmo_nuisance_fixed']@j.T)


def outcomes():
    return {name:{'status':'passed','cosmology_names':['ln10_10_As','Mnu','Omegam'],
        'growth_z':[sum(b)/2 for b in p.nominal_bins],
        'marginalized':{'cosmology':[1.,2.,3.],'growth':[1.]*len(p.nominal_bins)},
        'nuisance_fixed':{'cosmology':[.5,1.,1.5],'growth':[.5]*len(p.nominal_bins)}}
        for name,p in all_profiles().items()}


def test_exact_inventory_required_ratios_and_five_refinement_grids():
    profiles=all_profiles();assert len(profiles)==18
    assert len(set(profiles)&set(paper_profiles()))==13
    assert len(GRID_PROFILES)==5
    pairs=comparisons();assert ('I','A') in pairs and ('H21','C21') in pairs and ('C21','C22') in pairs
    r=compare_results(outcomes(),outcomes())
    assert r['status']=='passed'
    assert any('C21_over_C22' in x['label'] for x in r['metrics'])
    assert any('C22_over_C21' in x['label'] for x in r['metrics'])
    assert any('/growth/8' in x['label'] for x in r['metrics'])


@pytest.mark.parametrize('damage',['missing','failed','growth','cosmology','nonfinite','boundary','reciprocal'])
def test_failures_or_one_percent_boundary_cannot_be_hidden(damage):
    a=outcomes();b=outcomes()
    if damage=='missing':b.pop('I')
    if damage=='failed':b['H21']['status']='failed'
    if damage=='growth':b['C22']['growth_z'][-1]=2.3
    if damage=='cosmology':b['B']['cosmology_names'].reverse()
    if damage=='nonfinite':b['C21']['marginalized']['growth'][0]=np.nan
    if damage=='boundary':b['H21']['marginalized']['growth'][0]=1.01
    if damage=='reciprocal':b['I']['marginalized']['growth'][0]=.99005
    try:result=compare_results(a,b)
    except ValueError:return
    assert result['status']=='failed'


def test_output_roots_are_immutable_and_source_selection_never_invents_tail_desi(tmp_path):
    p=tmp_path/'run';fresh_directory(p)
    with pytest.raises(FileExistsError):fresh_directory(p)
    assert source_for_cell(paper_profiles()['A'],'bin02-radial00-overlap')=='B'
    assert source_for_cell(highz_profiles()['H21'],'bin06-radial00-overlap')=='C22'
    assert source_for_cell(highz_profiles()['C22'],'bin08-radial00-nonoverlap') is None
    assert source_for_cell(highz_profiles()['C21'],'bin06-radial00-nonoverlap')=='C'


def test_saved_likelihood_prior_and_growth_arithmetic_are_rebuilt(tmp_path,monkeypatch):
    r=fake_result();p=paper_profiles()['B'];save_result(tmp_path,p,SETTINGS,r,[])
    other,meta=load_result(tmp_path)
    np.testing.assert_array_equal(other.F_like,r.F_like)
    # This test isolates saved arithmetic; physical growth response has separate checks.
    monkeypatch.setattr('scripts.elf_study_report.infer_study',lambda like:r)
    assert verify_result(tmp_path,p,SETTINGS)['status']=='passed'
    a=dict(np.load(tmp_path/'arrays.npz'));a['result/F_prior_nuisance']*=2
    np.savez_compressed(tmp_path/'arrays.npz',**a)
    with pytest.raises(ValueError):verify_result(tmp_path,p,SETTINGS)


def test_constraint_labels_and_nesting_snapshot_preserve_full_covariance():
    r=fake_result();row=constraint_row(r)
    assert len(row['growth_z'])==8 and row['parameter_count']==len(r.names)
    assert len(row['nuisance_penalty']['growth'])==8
    s=nesting_snapshot(r)
    assert len(s['cells'])==20
    np.testing.assert_array_equal(s['cells'][0]['covariance'],r.likelihood.cells[0].covariance)


def test_physical_projection_fiducials_and_prior_records_cannot_be_changed():
    from scripts.elf_study_report import check_projection_binding
    r=fake_result();p=paper_profiles()['B'];parts=analytic(p)
    assert check_projection_binding(r,p,SETTINGS,parts,FIDUCIAL,{})['status']=='passed'
    r.likelihood.parameters[3]['prior_sigma']=.2
    with pytest.raises(ValueError):check_projection_binding(r,p,SETTINGS,parts,FIDUCIAL,{})


def test_saved_resolution_certificate_must_match_recomputed_metrics():
    from scripts.elf_study_report import verify_comparison_record
    a=compare_results(outcomes(),outcomes());b=compare_results(outcomes(),outcomes())
    assert verify_comparison_record(a,b)['status']=='passed'
    a['metrics'][0]['relative_change']=.5
    with pytest.raises(ValueError,match='comparison'):verify_comparison_record(a,b)
