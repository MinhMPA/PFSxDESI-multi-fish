"""Actual observed phase orchestration on controlled physical leaves, not a forecast."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest
from test_shell_numerics import AP,SOLVER,NEW,grid_sample,candidate_provenance


@pytest.mark.parametrize('change',['source','shell','failure'])
def test_refinement_reload_authenticates_source_before_comparing_errors(tmp_path,change):
    from scripts import run_extended_cosmology as d
    assert hasattr(d,'_load_resolution_control'), 'authenticated shell comparison boundary missing'
    base,_=grid_sample();other,outcomes=grid_sample(.000625);p=candidate_provenance()
    saved=d.save_control(tmp_path/'refined',other,outcomes,provenance=p)
    saved['failed_treatments']={}
    loaded,report=d._load_resolution_control(base,saved,p,label='shell')
    assert len(loaded[0].grid.centers)==304
    if change=='source':saved['producer_identity']['source_content_hash']='e'*64
    elif change=='shell':saved['shell_grid_role']='adopted'
    elif change=='failure':saved['failed_treatments']={'marginal6_no_extra':{'status':'failed','error':'changed','raw':{}}}
    with pytest.raises(ValueError):d._load_resolution_control(base,saved,p,label='shell')


def test_complete_candidate_phase_retains_both_ratios_and_all_reference_roles(tmp_path,monkeypatch):
    from scripts import run_extended_cosmology as d
    from test_run_extended_cosmology import _route_driver_processes
    tests=Path(__file__).resolve().parent
    bootstrap=tmp_path/'bootstrap.py'
    bootstrap.write_text(f'import sys\nsys.path.insert(0,{str(tests)!r})\nsys.path.insert(0,{str(tests.parent)!r})\n'
        'from test_run_extended_cosmology import _install_synthetic_leaves,_synthetic_oracle,_route_driver_processes\n'
        f'from pathlib import Path\nroot=Path({str(tmp_path)!r})\n'
        'd,build,survey=_install_synthetic_leaves(root,intrinsic=True)\n'
        'if not (root/"oracle").exists():\n _synthetic_oracle(root,build,survey)\n'
        '_route_driver_processes(d,__file__)\n'
        'if len(sys.argv)>1: raise SystemExit(d.main())\n')
    initial=subprocess.run([sys.executable,'-B',str(bootstrap)],capture_output=True,text=True)
    assert initial.returncode==0,initial.stderr
    _route_driver_processes(d,bootstrap,monkeypatch)
    oracle=tmp_path/'oracle';out=tmp_path/'study'
    profile=['--numerical-profile',AP,'--shell-profile',NEW,'--fisher-solver',SOLVER]
    rc=d.main(['--phase','qualify','--output-root',str(out),*profile,
        '--oracle-export',str(oracle),'--oracle-export-sha256',d.file_hash(oracle/'export-index.json')])
    assert rc==0,'\n'.join(p.read_text()[-10000:] for p in out.glob('qualify-*/worker/process.log'))
    selection=next(out.glob('qualify-*/selection.json'));q=d.read_json(d.read_json(selection)['qualification'])
    assert q['inventory']['shell_profile']['name']==NEW
    assert 'historical_ap_assessment' not in q
    assert q['candidate_shell_ap8192_assessment']['shell_dk']==.00125
    assert q['shell_grid_assessment']['ratios']['status']=='assessed'
    for label in ('shell','angular'):
        report=q['resolution_comparisons'][label]
        assert report['ratios']['status']=='passed' and report['producer_identity']['shell_profile']['name']==NEW
        for case in ('B','C'):
            assert report['cases'][case]['status']=='passed'
            assert report['cases'][case]['refined_grid']['shells']==(304 if label=='shell' else 152)
        for treatment in report['ratios']['treatments'].values():
            assert set(treatment)=={'cosmology_marg','cosmology_nuisance_fixed','growth_marg','growth_nuisance_fixed'}
            assert all(set(row)=={'B_over_C','C_over_B'} for row in treatment.values())
    assert d.main(['--phase','produce','--output-root',str(out),*profile,
                   '--selection',str(selection),'--selection-sha256',d.file_hash(selection)])==0
    produced=next(out.glob('produce-*/selection.json'))
    run=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','report','--output-root',str(out),*profile,
        '--selection',str(produced),'--selection-sha256',d.file_hash(produced)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    for change in ('missing_ratios','wrong_source','wrong_assessment','missing_original'):
        bad=copy.deepcopy(q)
        if change=='missing_ratios':bad['resolution_comparisons']['shell']['ratios']['treatments']['marginal6']['cosmology_nuisance_fixed'].pop('C_over_B')
        elif change=='wrong_source':bad['cases']['C']['refinements']['shell']['producer_identity']['source_content_hash']='e'*64
        elif change=='wrong_assessment':bad['shell_grid_assessment']['ratios']['status']='passed'
        else:bad['cases']['B']['ap_references'].pop('original_grid_replay')
        path=tmp_path/(change+'.json');d.write_json(path,bad)
        rejected=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','produce','--output-root',str(out),*profile,
            '--selection',str(path),'--selection-sha256',d.file_hash(path),'--reload'],capture_output=True,text=True)
        assert rejected.returncode!=0,change
