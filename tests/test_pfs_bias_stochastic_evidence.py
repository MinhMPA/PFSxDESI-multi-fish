"""Supplementary validation must preserve and authenticate the failed measurement."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pfsfog.elf_survey_evidence import file_digest
from scripts.pfs_bias_stochastic_evidence import correct_saved_cell, verify_corrected_cell

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/pfs-bias-broad-20261010-01/physical/angular64'
KEY='w0-adopted_relations-equal-DESI-LRG'


@pytest.mark.skipif(not (RAW/'summary.json').exists(),reason='requires retained broader measurement')
def test_corrected_evidence_keeps_failed_origin_and_all_arrays_unchanged(tmp_path):
    request=json.loads((RAW/'request.json').read_text())
    row=next(r for r in json.loads((RAW/'summary.json').read_text())['workers'] if r['key']==KEY)
    expected=next(r for r in request['expected'] if r['key']==KEY)
    raw=RAW/'cells'/KEY;output=tmp_path/'validated'
    before={name:file_digest(raw/name) for name in row['files']}
    correct_saved_cell(raw,output,expected,request,row)
    assert verify_corrected_cell(output,expected,request,row)['status']=='passed'
    receipt=json.loads((output/'validation.json').read_text())
    assert receipt['original_actual_exit']==1
    assert receipt['kind']=='saved_array_validation'
    assert {name:file_digest(raw/name) for name in row['files']}==before
    assert file_digest(output/'arrays.npz')==before['arrays.npz']
    assert file_digest(output/'neighborhood_spectra.npy')==before['neighborhood_spectra.npy']
    with pytest.raises(FileExistsError):correct_saved_cell(raw,output,expected,request,row)
    bad=deepcopy(expected);bad['experiment']['ratio']=.8
    with pytest.raises(ValueError):verify_corrected_cell(output,bad,request,row)
    record=json.loads((output/'inputs.json').read_text());record['analytic_zero_response']['support']=[]
    (output/'inputs.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):verify_corrected_cell(output,expected,request,row)
