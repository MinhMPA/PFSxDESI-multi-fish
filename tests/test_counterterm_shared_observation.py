"""Canonical shared responses require an unchanged physical observation."""
from copy import deepcopy
import json
import numpy as np
import pytest


def fixture(tmp_path):
    from scripts.run_pfs_density_scan import digest,write_json
    meta={'cell_id':'x','nominal_bin':[.4,.6],'radial_bin':[.4,.6], 'model_redshift':.5,
          'volume':100.,'region':'overlap','nbar':{'X':.001},'local_names':['ln10_10_As','bin00:X:c0'],
          'retained_pairs':[['X','X']]}
    source={'inputs':{'metadata':meta,'fiducials':{'ln10_10_As':3.,'bin00:X:c0':0.}},
            'arrays':{'k':np.array([.1,.2]),'dk':np.array([.01,.01]),'mean':np.ones((2,3)),
                      'C':np.tile(np.eye(3),(2,1,1)),'D':np.ones((2,3,2)),
                      'D_intrinsic':np.ones((2,3,2)),'D_geometry':np.zeros((2,3,2))},'source_files':{}}
    np.savez(tmp_path/'arrays.npz',**source['arrays']);write_json(tmp_path/'inputs.json',source['inputs'])
    write_json(tmp_path/'receipt.json',{'status':'passed','request_sha256':'same','source_files':{},
        'files':{n:digest(tmp_path/n) for n in ('arrays.npz','inputs.json')}})
    return source


def test_roundoff_only_anchor_restores_exact_shared_derivatives(tmp_path):
    from scripts.counterterm_shared_observation import anchor_corrected_source
    source=fixture(tmp_path);source['arrays']['D'][0,0,0]+=1e-15
    anchored=anchor_corrected_source(source,tmp_path,'same')
    np.testing.assert_array_equal(anchored['arrays']['D'],np.ones((2,3,2)))
    assert anchored['unchanged_observation']['responses']['D']['roundoff_limit_in_eps']==64

@pytest.mark.parametrize('kind',['physical_D','mean','C','request'])
def test_anchor_never_accepts_changed_physics_or_request(tmp_path,kind):
    from scripts.counterterm_shared_observation import anchor_corrected_source
    source=fixture(tmp_path)
    if kind=='physical_D':source['arrays']['D'][0,0,0]+=.01
    elif kind=='mean':source['arrays']['mean'][0,0]+=.01
    elif kind=='C':source['arrays']['C'][0,0,0]+=.01
    with pytest.raises(ValueError):anchor_corrected_source(source,tmp_path,'other' if kind=='request' else 'same')

def test_saved_two_tracer_cell_keeps_signed_differences_and_passes_qualification():
    from pathlib import Path
    from types import SimpleNamespace
    from pfsfog.extended_artifacts import read_json
    from scripts.counterterm_shared_observation import anchor_corrected_source
    from scripts.elf_study_data import select_cell,check_saved_derivatives
    root=Path(__file__).resolve().parents[1]/'results/counterterm-leading-20261020-03'
    part=root/'inference/loop2048/parts'
    b=Path(read_json(part/'B/bin01-radial00-overlap/record.json')['physical_receipt']).parent
    c=Path(read_json(part/'D/bin01-radial00-overlap/record.json')['physical_receipt']).parent
    receipt=read_json(c/'receipt.json')
    with np.load(c/'arrays.npz',allow_pickle=False) as data:a={n:data[n].copy() for n in data.files}
    source={'arrays':a,'inputs':read_json(c/'inputs.json'),'source_files':receipt['source_files']}
    anchored=anchor_corrected_source(source,b,receipt['request_sha256'])
    for name in ('fd_plus','fd_minus','fd_derivatives'):
        np.testing.assert_array_equal(anchored['arrays'][name],a[name])
    row={**anchored['inputs']['metadata'],'empty_observation':False}
    projected=select_cell(anchored,SimpleNamespace(case='C'),row,SimpleNamespace(centers=a['k'],widths=a['dk']))
    assert check_saved_derivatives(projected)['status']=='passed'
    assert anchored['unchanged_observation']['responses']['D']['max_absolute_difference']>0
