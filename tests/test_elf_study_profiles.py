"""Scientific coverage: omissions, false high-z crosses and volume duplication."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.survey_geometry import ShellGeometry
from pfsfog.surveys import default_survey_group
from scripts.elf_study_profiles import (
    paper_profiles, highz_profiles, study_config, study_cells, selected_pairs,
    study_geometry, likelihood_key,
)


def test_exact_paper_inventory_shares_only_prior_variants():
    profiles = paper_profiles()
    assert set(profiles) == {'I','A','B','D','C','B_k015','C_k015',
                            'B_prior3','C_prior3','B_prior10','C_prior10',
                            'B_no_elg1','C_no_elg1'}
    assert len({likelihood_key(p) for p in profiles.values()}) == 9
    assert likelihood_key(profiles['B']) == likelihood_key(profiles['B_prior10'])
    assert likelihood_key(profiles['B']) != likelihood_key(profiles['B_k015'])
    assert likelihood_key(profiles['C']) != likelihood_key(profiles['C_no_elg1'])


@pytest.mark.parametrize('case,count,crosses', [('I',2,0),('A',2,0),('B',3,1),('D',4,1),('C',6,3)])
def test_baseline_data_rows_preserve_desired_autos_and_crosses(case,count,crosses):
    pairs = selected_pairs(paper_profiles()[case],(.8,1.),('DESI-ELG','DESI-QSO','PFS-ELG'))
    assert len(pairs) == count
    assert sum(a!=b for a,b in pairs) == crosses
    if case == 'D': assert ('DESI-ELG','PFS-ELG') not in pairs


def test_highz_autos_and_crosses_are_separate_observations():
    p = highz_profiles()
    assert set(p) == {'B16','C16','H21','C21','C22'}
    names=('DESI-QSO','PFS-ELG')
    assert selected_pairs(p['C16'],(1.6,2.),names)==[('DESI-QSO','DESI-QSO')]
    assert selected_pairs(p['H21'],(1.6,2.),names)==[('DESI-QSO','DESI-QSO'),('PFS-ELG','PFS-ELG')]
    assert selected_pairs(p['C21'],(1.6,2.),names)[-1]==('DESI-QSO','PFS-ELG')
    assert selected_pairs(p['H21'],(1.4,1.6),names)[-1]==('DESI-QSO','PFS-ELG')
    assert len(p['C21'].nominal_bins)==8 and len(p['C22'].nominal_bins)==9


def test_cross_ceiling_cannot_be_hidden_inside_unsplit_cell():
    with pytest.raises(ValueError,match='boundary'):
        selected_pairs(highz_profiles()['H21'],(1.5,1.7),('DESI-QSO','PFS-ELG'))


def test_terminal_config_preserves_physical_bounds():
    settings={'nfft':2048,'n_mu':64,'dk':.000625}
    c=study_config(highz_profiles()['C22'],settings)
    assert c.cosmology_basis=='baseline3' and c.nominal_bins[-1]==(2.1,2.2)
    assert c.kmax==.2 and c.n_mu==64 and c.dk==.000625
    assert study_config(highz_profiles()['C22'],settings,extended=True).cosmology_basis=='extended6'
    with pytest.raises(ValueError): replace(c,nominal_bins=((2.1,2.4),))


def test_terminal_geometry_is_native_pfs_only_and_common_geometry_unchanged():
    group=default_survey_group(); base=ShellGeometry.from_group(group)
    g=study_geometry(highz_profiles()['C22'],group)
    np.testing.assert_array_equal(g.edges[:len(base.edges)],base.edges)
    np.testing.assert_array_equal(g.volume_per_deg2[:len(base.volume_per_deg2)],base.volume_per_deg2)
    pfs=group.pfs; rows=(pfs.z_min_all>=2.1)&(pfs.z_max_all<=2.2)
    assert g.edges[-1]==2.2
    assert g.volume(2.1,2.2,1200.)==pytest.approx(pfs.Vz_all[rows].sum(),rel=1e-14)


def test_shared_native_volume_mismatch_rejected_before_extension():
    group=default_survey_group(); group=replace(group,pfs=replace(group.pfs,Vz_all=group.pfs.Vz_all*1.01))
    with pytest.raises(ValueError,match='volume mismatch'):
        study_geometry(highz_profiles()['C22'],group)


def test_terminal_nonoverlap_stays_empty_and_highz_population_sharing_is_exact():
    cosmo=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.7)
    cells=study_cells(highz_profiles()['C22'],cosmo)
    assert len(cells)==22
    last=cells[-2:]
    assert last[0]['expected']['active_populations']==['PFS-ELG']
    assert last[1]['expected']['empty_observation'] is True
    assert last[1]['expected']['local_names']==[]
    assert len(last[0]['expected']['local_names'])==14
    h=study_cells(highz_profiles()['H21'],cosmo)
    overlap=next(c for c in h if c['cell_id']=='bin06-radial00-overlap')
    assert overlap['expected']['active_populations']==['DESI-QSO','PFS-ELG']
    assert len(overlap['expected']['local_names'])==25
    assert not any('|' in n for n in overlap['expected']['local_names'])
    assert overlap['expected']['absent_observations']==[
        [['DESI-QSO','PFS-ELG'],ell] for ell in (0,2,4)]


def test_elg1_removal_keeps_reference_fiducials_and_exact_partial_support():
    cosmo=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.7)
    baseline={c['cell_id']:c for c in study_cells(paper_profiles()['C'],cosmo)}
    removed=study_cells(paper_profiles()['C_no_elg1'],cosmo)
    for c in removed:
        if c['cell_id'] in baseline: assert c['fiducials']==baseline[c['cell_id']]['fiducials']
        lo,hi=c['expected']['radial_bin']
        if lo>=.8 and hi<=1.1:
            assert 'DESI-ELG' not in c['expected']['active_populations']
        if lo==1.1:
            assert 'DESI-ELG' in c['expected']['active_populations']


def test_scale_cut_is_an_exact_restriction_of_the_observed_parent_shells():
    from scripts.elf_study_profiles import study_shell_grid
    from pfsfog.revision_forecast import make_shell_grid
    for dk in (.00125,.000625):
        p=paper_profiles()['B_k015'];s={'nfft':2048,'n_mu':64,'dk':dk}
        parent=make_shell_grid(.01,.2,dk);cut=study_shell_grid(study_config(p,s))
        assert len(cut.centers)==round(.14/dk)
        np.testing.assert_array_equal(cut.centers,parent.centers[:len(cut.centers)])
        np.testing.assert_array_equal(cut.widths,parent.widths[:len(cut.widths)])
        assert abs(cut.edges[-1]-.15)<=np.spacing(.15)
