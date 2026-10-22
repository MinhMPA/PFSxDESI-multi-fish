"""Corrected selection retains the DESI comparison and shared-bin nuisance model."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from scripts.elf_study_profiles import paper_profiles,study_cells,SELECTED_SETTINGS


@pytest.mark.parametrize('start',[.6,.7])
def test_coverage_changes_pfs_without_averaging_away_desi_selection(start):
    from scripts.takada_study import corrected_profiles,corrected_cells
    cosmo=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.8)
    profile=corrected_profiles(2.4)['C']
    old=study_cells(paper_profiles()['C'],cosmo)
    new=corrected_cells(profile,cosmo,pfs_zmin=start)
    assert len(new)==len(old)
    for a,b in zip(old,new):
        x,y=a['expected'],b['expected']
        assert x['radial_bin']==y['radial_bin']
        assert x['volume']==y['volume']
        assert x['model_redshift']==y['model_redshift']
        for n in x['nbar']:
            if n.startswith('DESI-'):assert x['nbar'][n]==y['nbar'][n]
    row=next(c['expected'] for c in new if c['cell_id']=='bin01-radial00-overlap')
    assert ('PFS-ELG' in row['active_populations'])==(start==.6)
    if start==.6:
        assert row['nbar']['PFS-ELG']==pytest.approx(1.9e-4)
        assert 'bin01:DESI-LRG|PFS-ELG:Pshot_cross' in row['local_names']


def test_full_table_tail_has_new_physical_redshift_and_no_desi():
    from scripts.takada_study import corrected_profiles,corrected_cells
    cosmo=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.8)
    profile=corrected_profiles(2.4)['C24']
    cells=corrected_cells(profile,cosmo,pfs_zmin=.6)
    overlap=next(c for c in cells if c['cell_id']=='bin08-radial00-overlap')
    row=overlap['expected']
    assert row['nominal_bin']==[2.1,2.4]
    assert row['model_redshift']==2.25
    assert row['nbar']=={'PFS-ELG':pytest.approx(2.7e-4)}
    assert row['retained_pairs']==[['PFS-ELG','PFS-ELG']]
    empty=cells[-1]['expected']
    assert empty['empty_observation'] and empty['local_names']==[]
    assert empty['volume']==0.


def test_assembly_reproduces_authenticated_reference_without_reapplying_priors():
    from scripts.takada_study import assemble_from_template
    from scripts.elf_study_report import load_result
    path=Path(__file__).resolve().parents[1]/'results/elf-survey-studies-20261010-01/inference/shell304/profiles/C'
    old,_=load_result(path)
    values=dict(zip(old.names,old.likelihood.fiducial))
    parts=[{'cell':c,'fiducials':{n:values[n] for n in c.names}} for c in old.likelihood.cells]
    new=assemble_from_template(old.likelihood,parts)
    np.testing.assert_array_equal(new.F_like,old.F_like)
    assert new.parameters==old.likelihood.parameters
    with pytest.raises(ValueError,match='cell'):
        assemble_from_template(old.likelihood,parts[:-1])
    with pytest.raises(ValueError,match='cell'):
        assemble_from_template(old.likelihood,parts+[parts[-1]])
