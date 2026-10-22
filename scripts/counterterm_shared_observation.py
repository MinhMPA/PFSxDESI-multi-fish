"""Retain the same measured DESI response across nested tracer inventories."""
from pathlib import Path
import numpy as np
from pfsfog.extended_artifacts import read_json
from scripts.run_pfs_density_scan import digest
from scripts.takada_density import preserve_unchanged_observation


def anchor_corrected_source(source,physical_directory,request_sha):
    """Apply the existing exact mean/C and 64-epsilon response witness.

    All signed finite-difference means and covariance neighborhoods remain
    independent. Their subsequent qualification uses the canonical AD columns.
    """
    path=Path(physical_directory);receipt=read_json(path/'receipt.json')
    if receipt['status']!='passed' or receipt['request_sha256']!=request_sha:
        raise ValueError('corrected DESI anchor is not qualified under this request')
    for name,sha in receipt['files'].items():
        if digest(path/name)!=sha:raise ValueError('corrected DESI anchor changed')
    with np.load(path/'arrays.npz',allow_pickle=False) as a:arrays={n:a[n].copy() for n in a.files}
    inputs=read_json(path/'inputs.json')
    anchor={'arrays':arrays,'inputs':inputs,'source_files':{**receipt['source_files'],
        **{str(path/n):digest(path/n) for n in ('arrays.npz','inputs.json','receipt.json')}}}
    result=preserve_unchanged_observation(source,anchor)
    result['corrected_observation_anchor']={'directory':str(path),'receipt_sha256':digest(path/'receipt.json'),
                                           'witness':result['unchanged_observation']}
    return result
