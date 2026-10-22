"""The corrected campaign must bind and preserve its physical source."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_pfs_density_response import physical_toy


def test_target_metadata_changes_only_observed_cell_and_density():
    from scripts.run_takada_revision import source_target
    source,row,_=physical_toy();row['nbar']['PFS-ELG']=3.1e-4
    row['volume']=3e8;row['radial_bin']=[1.7,2.]
    target=source_target(source,row)
    assert target['nbar']['PFS-ELG']==3.1e-4
    assert target['nbar']['DESI-QSO']==1e-4
    assert target['volume']==3e8 and target['radial_bin']==[1.7,2.]
    assert target['local_names']==source['inputs']['metadata']['local_names']
    assert source['inputs']['metadata']['nbar']['PFS-ELG']==2e-4


def test_changed_redshift_cannot_be_disguised_as_density_rescaling():
    from scripts.run_takada_revision import source_target
    from scripts.takada_density import retarget_source
    source,row,_=physical_toy();row['model_redshift']=2.25
    with pytest.raises(ValueError,match='model_redshift'):
        retarget_source(source,source_target(source,row))


def test_source_authentication_allows_only_the_new_table_and_rejects_old_input_changes():
    from scripts.run_takada_revision import verify_original_source
    from scripts.takada_density import TABLE
    from pfsfog.elf_survey_evidence import request_digest,file_digest
    def identity(surveys):
        core={'survey_input_hashes':surveys,'engine':'unchanged ELF'}
        outer={'original':{'details':core,'identity_hash':request_digest(core)},'programs':{'model':'same'}}
        return {'details':outer,'identity_hash':request_digest(outer)}
    old=identity({'original.txt':'abc'})
    new=identity({'original.txt':'abc',str(TABLE):file_digest(TABLE)})
    assert verify_original_source(old,new)['status']=='passed'
    for files in ({'original.txt':'changed',str(TABLE):file_digest(TABLE)},
                  {'original.txt':'abc',str(TABLE):'wrong'},
                  {'original.txt':'abc',str(TABLE):file_digest(TABLE),'unrelated.txt':'x'}):
        with pytest.raises(ValueError):verify_original_source(old,identity(files))


def test_new_overlap_keeps_existing_desi_responses_exact_and_rejects_physical_change():
    """Density transfer must not round the already measured DESI observation."""
    from pathlib import Path
    from scripts.elf_study_evidence import load_source_cell
    from scripts.run_takada_revision import corrected_source,STUDY
    from scripts.takada_study import corrected_cells,corrected_profiles
    from pfsfog.cosmo import FiducialCosmology
    import json
    base=STUDY/'inference/loop2048'
    meta=json.loads((base/'profiles/C22/manifest.json').read_text())
    refs={c['metadata']['cell_id']:r for c,r in zip(meta['likelihood']['cells'],meta['projections'])}
    def load(name):
        ref=json.loads((base/'projections'/(refs[name]+'.json')).read_text())
        return load_source_cell(ref['source_directory'],ref['source_record'])
    raw=load('bin01-radial01-overlap');anchor=load('bin01-radial00-overlap')
    target=next(c['expected'] for c in corrected_cells(corrected_profiles(2.4)['C'],FiducialCosmology(),pfs_zmin=.6)
                if c['cell_id']=='bin01-radial00-overlap')
    source=corrected_source(raw,target,unchanged=anchor)
    names=source['inputs']['metadata']['local_names'];cols=[names.index(n) for n in anchor['inputs']['metadata']['local_names']]
    np.testing.assert_array_equal(source['arrays']['D'][:,:3,cols],anchor['arrays']['D'])
    np.testing.assert_array_equal(source['arrays']['C'][:,:3,:3],anchor['arrays']['C'])
    damaged={**anchor,'arrays':dict(anchor['arrays'])}
    damaged['arrays']['D']=anchor['arrays']['D'].copy();damaged['arrays']['D'][0,0,0]*=1.01
    with pytest.raises(ValueError,match='unchanged observation'):
        corrected_source(raw,target,unchanged=damaged)
