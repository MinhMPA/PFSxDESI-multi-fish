"""Grid reports authenticate their data before any convergence claim."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_elf_grid_comparison import local,request,summary,TREATMENTS


def api():
    assert importlib.util.find_spec('scripts.compare_elf_grids') is not None, 'authenticated grid comparison missing'
    return importlib.import_module('scripts.compare_elf_grids')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def cohort(root,nfft):
    physical=root/'physical';r=request(nfft=nfft);r['expected']=[]
    for case in ('B','C'):
        cells=[]
        for i in range(20):
            key=f'{case}:cell{i}';a=local()
            support={'cell_id':f'cell{i}','local_names':a['names'],'volume':100.}
            r['expected'].append({'key':key,'case':case,'support':support})
            cell=physical/'cells'/key.replace(':','__');cell.mkdir(parents=True)
            np.savez(cell/'arrays.npz',**{k:v for k,v in a.items() if k!='names'})
            write(cell/'inputs.json',{'metadata':support,'fiducials':{'amplitude':1.,'tilt':1.}})
            cells.append({'names':a['names'],'metadata':support,'validation':{'status':'passed'}})
        meta={'names':['amplitude','tilt'],'parameters':[{'name':'amplitude','prior_sigma':None}],
              'reference_cosmology':{'h':.67},'configuration':copy.deepcopy(r['configurations'][case]),
              'model':{'fft_n':nfft,'mean_n_mu':32,'covariance_n_mu':32,'fixed_covariance':True},
              'cells':cells,'validation':{'status':'passed'}}
        path=root/'cases'/case
        write(path/'manifest.json',{'likelihood':meta,'native':copy.deepcopy(meta)})
        write(path/'growth.json',{'redshifts':list(range(8))})
        arrays={f'{prefix}/fiducial':np.ones(2) for prefix in ('likelihood','native')}
        for t in (*TREATMENTS,'native_result'):
            for field in ('F_prior_cosmo','F_prior_nuisance'):arrays[t+'/'+field]=np.eye(2)
            arrays[t+'/growth_z']=np.arange(8);arrays[t+'/growth_fiducial']=np.ones(8)
        np.savez(path/'arrays.npz',**arrays)
    write(physical/'request.json',r);write(physical/'summary.json',{'status':'passed'})
    write(root/'request.json',{'physical_root':str(physical.resolve()),'settings':r['settings']})
    write(root/'summary.json',summary())
    return root


@pytest.fixture
def pair(tmp_path,monkeypatch):
    module=api()
    # Expensive theory/native3 validation is the external boundary only.
    # All local/global arithmetic, file loading, fixed-content and report checks are real.
    monkeypatch.setattr(module,'verify_inference',lambda root:{'status':'passed',
        'cases':{case:{'status':'passed'} for case in ('B','C')},'full_forecast_qualification':'pending'})
    return module,cohort(tmp_path/'coarse',1024),cohort(tmp_path/'fine',2048)


def test_complete_saved_comparison_and_reconstruction(pair,tmp_path):
    m,c,f=pair;r=m.compare_runs(c,f,'loop')
    assert r['status']=='passed' and r['full_forecast_qualification']=='pending'
    assert len(r['local'])==40 and len(r['global']['metrics'])==712
    assert any(p.endswith('/cases/B/arrays.npz') for p in r['input_files'])
    assert set(r['comparison_programs'])=={'scripts/elf_grid_comparison.py','scripts/compare_elf_grids.py','scripts/infer_elf_survey.py'}
    path=tmp_path/'report.json';write(path,r)
    assert m.verify_report(path)['status']=='passed'


@pytest.mark.parametrize('checked',[{'status':'failed','problems':['foreign source']},
    {'status':'passed','cases':{'B':{'status':'passed'}}},
    {'status':'passed','cases':{'B':{'status':'failed'},'C':{'status':'passed'}}}])
def test_incomplete_or_foreign_input_fails_before_reading(monkeypatch,tmp_path,checked):
    m=api();monkeypatch.setattr(m,'verify_inference',lambda root:checked)
    with pytest.raises(ValueError,match='authenticat'):m.compare_runs(tmp_path/'absent',tmp_path/'also_absent','loop')


@pytest.mark.parametrize('field',['D','C','mean'])
def test_local_failure_cannot_be_hidden_by_fresh_hashes(pair,field):
    m,c,f=pair;p=f/'physical/cells/C__cell19/arrays.npz'
    with np.load(p) as z:a={k:z[k] for k in z}
    a[field]*=1.002;np.savez(p,**a)
    r=m.compare_runs(c,f,'loop')
    assert r['status']=='failed' and r['local']['C:cell19']['status']=='failed'


def test_global_failure_cannot_be_hidden_by_fresh_hashes(pair):
    m,c,f=pair;p=f/'summary.json';s=json.loads(p.read_text());o=s['outcomes']
    o['C']['marginal6']['nuisance_fixed']['growth'][7]*=1.02;write(p,summary(o))
    r=m.compare_runs(c,f,'loop')
    assert r['status']=='failed' and r['global']['status']=='failed'


@pytest.mark.parametrize('damage',['fiducial','prior','metadata','model'])
def test_inference_fixed_physics_must_match(pair,damage):
    m,c,f=pair;p=f/'cases/B/manifest.json';v=json.loads(p.read_text())
    if damage in ('metadata','model'):
        if damage=='metadata':v['likelihood']['cells'][0]['metadata']['volume']=101.
        else:v['likelihood']['model']['fixed_covariance']=False
        write(p,v)
    else:
        p=f/'cases/B/arrays.npz'
        with np.load(p) as z:a={k:z[k] for k in z}
        a['likelihood/fiducial' if damage=='fiducial' else 'marginal6/F_prior_cosmo']*=2
        np.savez(p,**a)
    with pytest.raises(ValueError,match='fixed'):m.compare_runs(c,f,'loop')


@pytest.mark.parametrize('damage',['report_metric','input','missing_hash','program_hash'])
def test_verification_rejects_changed_or_missing_report_evidence(pair,tmp_path,damage):
    m,c,f=pair;r=m.compare_runs(c,f,'loop');p=tmp_path/'report.json'
    if damage=='report_metric':r['global']['metrics'][0]['fine']=123.
    elif damage=='input':write(f/'cases/C/growth.json',{'foreign':True})
    elif damage=='missing_hash':r['input_files'].pop(next(iter(r['input_files'])))
    else:r['comparison_programs']['scripts/elf_grid_comparison.py']='foreign'
    write(p,r);assert m.verify_report(p)['status']=='failed'


def test_existing_report_is_never_overwritten(tmp_path):
    m=api();p=tmp_path/'report.json';p.write_text('preserved failure')
    with pytest.raises(FileExistsError):m.main(['--coarse','absent','--fine','absent','--axis','loop','--output',str(p)])
    assert p.read_text()=='preserved failure'


def test_failure_exit_and_record_are_explicit(pair,tmp_path):
    m,c,f=pair;p=f/'summary.json';s=json.loads(p.read_text());s['outcomes']['C']['marginal6']['marginalized']['growth'][0]=110.;write(p,summary(s['outcomes']))
    dest=tmp_path/'failed.json'
    assert m.main(['--coarse',str(c),'--fine',str(f),'--axis','loop','--output',str(dest)])==1
    assert json.loads(dest.read_text())['status']=='failed'


def test_comparison_sources_do_not_enter_producer_inventory():
    api()
    from scripts.run_revision import collect_provenance
    import inspect
    source=inspect.getsource(collect_provenance)
    # A new producer file would invalidate the already measured baseline.
    assert not any(Path('pfsfog').glob('*grid_compar*'))
    assert 'compare_elf_grids.py' not in source and 'elf_grid_comparison.py' not in source


def test_help_exposes_saved_verification():
    api();r=subprocess.run([sys.executable,'scripts/compare_elf_grids.py','--help'],capture_output=True,text=True)
    assert r.returncode==0 and all(x in r.stdout for x in ('--coarse','--fine','--axis','--verify'))


def test_mutation_during_authentication_cannot_pass(pair,monkeypatch):
    m,c,f=pair;p=c/'physical/cells/B__cell0/arrays.npz';original=p.read_bytes()
    def authenticate(root):
        if Path(root)==c and p.read_bytes()!=original:return {'status':'failed','problems':['changed array']}
        if Path(root)==f:
            with np.load(p) as data:a={k:data[k] for k in data}
            a['D']*=1.0001;np.savez(p,**a)
        return {'status':'passed','cases':{case:{'status':'passed'} for case in ('B','C')}}
    monkeypatch.setattr(m,'verify_inference',authenticate)
    with pytest.raises(ValueError,match='changed.*authenticat|authenticat.*changed'):
        m.compare_runs(c,f,'loop')
