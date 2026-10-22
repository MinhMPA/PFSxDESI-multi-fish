"""Direct common-k spectra must retain the covariance's angular information."""
import json
import numpy as np


def test_direct_query_reconstructs_saved_spectra_and_density_changed_covariance():
    from scripts.takada_physical import direct_query,change_query_density
    from scripts.run_takada_revision import STUDY
    from scripts.elf_study_evidence import load_source_cell
    from scripts.elf_study_physical import common_query_data
    from pfsfog.one_loop import make_one_loop_engine
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.elf_covariance_audit import covariance_from_spectra
    base=STUDY/'inference/loop2048'
    m=json.loads((base/'profiles/C22/manifest.json').read_text())
    i=next(i for i,c in enumerate(m['likelihood']['cells']) if c['metadata']['cell_id']=='bin06-radial00-overlap')
    ref=json.loads((base/'projections'/(m['projections'][i]+'.json')).read_text())
    source=load_source_cell(ref['source_directory'],ref['source_record'])
    inputs=source['inputs'];idx=[0,10,40];k=source['arrays']['k'][idx]
    engine=make_one_loop_engine(nfft=2048);cosmo=FiducialCosmology()
    q=direct_query(inputs,k,engine,cosmo)
    np.testing.assert_allclose(q['pkmu'],source['arrays']['pkmu'][:,idx],rtol=1e-12,atol=1e-10)
    doubled=change_query_density(q,inputs,2.)
    changed={**inputs,'metadata':{**inputs['metadata'],'nbar':{**inputs['metadata']['nbar'],
        'PFS-ELG':2*inputs['metadata']['nbar']['PFS-ELG']}}}
    independent=common_query_data(changed,k,engine,cosmo)
    np.testing.assert_allclose(doubled['D'],independent['D'],rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(doubled['covariance_kernel'],independent['covariance_kernel'],rtol=1e-12,atol=1e-10)
    np.testing.assert_array_equal(doubled['mean'],q['mean'])
