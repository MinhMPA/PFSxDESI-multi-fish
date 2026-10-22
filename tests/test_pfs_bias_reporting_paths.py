"""A saved scientific report must be reproducible from either path spelling."""
from pathlib import Path

import pytest

from scripts.report_pfs_bias_study import ROOT, report

CAMPAIGN=ROOT/'results/pfs-bias-pilot-20261010-02'


@pytest.mark.skipif(not (CAMPAIGN/'inference/final-loop/summary.json').exists(),
                    reason='requires the retained completed pilot; no spectra are rerun')
def test_relative_and_absolute_campaign_paths_give_identical_bound_reports(tmp_path,monkeypatch):
    monkeypatch.chdir(ROOT)
    relative=report(CAMPAIGN.relative_to(ROOT),tmp_path/'relative')
    absolute=report(CAMPAIGN,tmp_path/'absolute')
    assert relative==absolute
    assert (tmp_path/'relative/response.csv').read_bytes()==(tmp_path/'absolute/response.csv').read_bytes()
