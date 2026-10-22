"""Long numerical stages advance only after completed physical qualification."""
import pytest
import json

from scripts.run_pfs_bias_broad import GRIDS
from scripts.continue_pfs_bias_broad import physical_ready, stage_commands, ensure_fresh_outputs


def test_no_numerical_stage_starts_before_all_physical_controllers_pass():
    assert physical_ready({'status':'running','controllers':[]}) is False
    rows=[{'grid':g,'actual_exit':0} for g in GRIDS]
    assert physical_ready({'status':'passed','controllers':rows}) is True
    for status,records in [('failed',rows),('passed',rows[:-1]),('passed',[rows[0]]*5),
                           ('passed',[{**r,'actual_exit':1} for r in rows])]:
        with pytest.raises(ValueError):physical_ready({'status':status,'controllers':records})


def test_sequential_stages_stop_at_report_and_do_not_publish(tmp_path):
    commands=stage_commands(tmp_path)
    assert [stage for stage,_ in commands]==['inference','common','comparison','report']
    assert all(str(tmp_path.resolve()) in ' '.join(command) for _,command in commands)
    assert all('git' not in command and 'submit' not in command for _,command in commands)


def test_existing_future_evidence_is_never_overwritten(tmp_path):
    ensure_fresh_outputs(tmp_path)
    (tmp_path/'comparisons').mkdir()
    with pytest.raises(FileExistsError):ensure_fresh_outputs(tmp_path)


def test_repaired_continuation_waits_on_supplementary_validation_and_preserves_original_receipt(tmp_path,monkeypatch):
    from scripts import continue_pfs_bias_broad as module
    # Isolate scheduling from the independently tested, expensive physics authentication.
    monkeypatch.setattr(module,'check_campaign',lambda root:None)
    (tmp_path/'request.json').write_text('{}')
    (tmp_path/'continuation.json').write_text('{"retained_original":true}')
    (tmp_path/'physical-execution.json').write_text('{"status":"failed","controllers":[]}')
    folder=tmp_path/'stochastic-validation';folder.mkdir()
    (folder/'execution.json').write_text('{"status":"running","controllers":[]}')
    with pytest.raises(TimeoutError):
        module.continue_stages(tmp_path,timeout=0,poll_seconds=0,after_stochastic_validation=True)
    assert json.loads((tmp_path/'continuation.json').read_text())=={'retained_original':True}
    receipt=json.loads((tmp_path/'continuation-after-validation.json').read_text())
    assert receipt['stages']==[] and receipt['review_complete'] is False
    assert receipt['error_type']=='TimeoutError'
