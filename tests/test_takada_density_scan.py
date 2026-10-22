"""The follow-up must vary density around the corrected, common reference."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_pfs_density_response import physical_toy


def test_density_windows_cover_the_selected_table_without_old_tail_truncation():
    from scripts.run_takada_density_scan import density_windows
    windows=density_windows(.6,2.4)
    assert windows==((.6,.8),(.8,1.),(1.,1.2),(1.2,1.4),(1.4,1.6),
                     (1.6,2.),(2.,2.1),(2.1,2.4))
    assert density_windows(.7,2.2)[0]==(.7,.8)
    assert density_windows(.7,2.2)[-1]==(2.1,2.2)
    with pytest.raises(ValueError):density_windows(.8,2.4)


def test_scan_reconstructs_the_new_reference_and_rejects_stale_density():
    from scripts.run_takada_density_scan import reconstruct_reference_source
    from scripts.run_takada_revision import source_target
    source,row,grid=physical_toy();row['nbar']['PFS-ELG']=3.1e-4
    record={'expected':row,'source_target':source_target(source,row)}
    changed=reconstruct_reference_source(source,record)
    assert changed['inputs']['metadata']['nbar']['PFS-ELG']==3.1e-4
    assert source['inputs']['metadata']['nbar']['PFS-ELG']==2e-4
    np.testing.assert_allclose(changed['arrays']['D'][:,:,4],source['arrays']['D'][:,:,4]*(2/3.1))
    wrong=deepcopy(record);wrong['source_target']['nbar']['PFS-ELG']=2e-4
    with pytest.raises(ValueError,match='reference'):
        reconstruct_reference_source(source,wrong)
