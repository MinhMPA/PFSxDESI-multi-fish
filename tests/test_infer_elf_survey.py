"""A complete constraint comparison requires actual successful source-bound work."""
import copy
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pfsfog.cosmology_projection import COSMOLOGY_TREATMENTS


def api():
    spec=importlib.util.find_spec('scripts.infer_elf_survey')
    assert spec is not None,'ELF inference command is missing'
    return importlib.import_module('scripts.infer_elf_survey')


def outcomes():
    out={}
    for case,scale in [('B',4.),('C',2.)]:
        out[case]={name:{'status':'passed','free_cosmology':['ln10_10_As','Mnu','Omegam'],
            **{kind:{'cosmology':dict.fromkeys(['ln10_10_As','Mnu','Omegam'],scale),
                     'growth':[scale]*8} for kind in ('marginalized','nuisance_fixed')}}
                  for name in COSMOLOGY_TREATMENTS}
    return out


def test_ratios_cover_both_directions_and_nuisance_dispositions():
    ratios=api().constraint_ratios(outcomes())
    assert set(ratios)==set(COSMOLOGY_TREATMENTS)
    for result in ratios.values():
        assert result['status']=='available'
        for kind in ('marginalized','nuisance_fixed'):
            assert result[kind]['B_over_C']['growth']==[2.]*8
            assert result[kind]['C_over_B']['growth']==[.5]*8
            assert result[kind]['B_over_C']['cosmology']['Mnu']==2.
            assert result[kind]['C_over_B']['cosmology']['Mnu']==.5


def test_failed_treatment_has_no_fabricated_gain():
    o=outcomes();o['B']['marginal6_no_extra']={'status':'failed','error':'unconstrained h'}
    result=api().constraint_ratios(o)['marginal6_no_extra']
    assert result=={'status':'unavailable','failed_cases':['B']}


@pytest.mark.parametrize('damage',['missing','zero','negative','wrong_basis','growth_count'])
def test_incomplete_or_incompatible_errors_cannot_form_ratios(damage):
    o=outcomes()
    if damage=='missing':o['B'].pop('release_h')
    elif damage=='zero':o['B']['release_h']['marginalized']['growth'][0]=0.
    elif damage=='negative':o['C']['marginal6']['marginalized']['cosmology']['Mnu']=-1.
    elif damage=='wrong_basis':o['C']['marginal6']['free_cosmology']=['h']
    else:o['B']['marginal6']['nuisance_fixed']['growth'].pop()
    with pytest.raises(ValueError):api().constraint_ratios(o)


def execution():
    request={'source_identity_hash':'example','scope':'single_resolution_global_inference'}
    from pfsfog.elf_survey_evidence import request_digest
    summary={'status':'passed','scope':request['scope'],'full_forecast_qualification':'pending',
        'source_changed':False,'request_sha256':request_digest(request),
        'workers':[{'case':c,'exit_code':0,'status':'passed','source_before':'example',
                    'source_after':'example','request_sha256':request_digest(request)} for c in ('B','C')]}
    return request,summary


@pytest.mark.parametrize('damage',['missing','duplicate','nonzero','foreign','failed','overclaim'])
def test_bad_execution_cannot_qualify_inference(damage):
    module=api();request,summary=execution()
    assert module.execution_problems(request,summary)==[]
    if damage=='missing':summary['workers'].pop()
    elif damage=='duplicate':summary['workers'].append(copy.deepcopy(summary['workers'][0]))
    elif damage=='nonzero':summary['workers'][0]['exit_code']=2
    elif damage=='foreign':summary['workers'][0]['source_after']='other'
    elif damage=='failed':summary['workers'][0]['status']='failed'
    else:summary['full_forecast_qualification']='passed'
    assert module.execution_problems(request,summary)


def test_selected_physical_probe_cannot_start_inference(monkeypatch,tmp_path):
    module=api()
    monkeypatch.setattr(module,'verify_run',lambda root:{'status':'passed','complete_survey':False})
    args=SimpleNamespace(physical_root=tmp_path/'selected',output=tmp_path/'new')
    with pytest.raises(ValueError,match='physical'):module.run_inference(args)
    assert not args.output.exists()


def test_output_is_never_overwritten(tmp_path):
    module=api();(tmp_path/'precious.txt').write_text('retained failure')
    with pytest.raises(FileExistsError):
        module.run_inference(SimpleNamespace(physical_root=tmp_path/'missing',output=tmp_path))
    assert (tmp_path/'precious.txt').read_text()=='retained failure'


def test_cli_help_exposes_physical_input_and_saved_verification():
    api();root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/infer_elf_survey.py','--help'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--physical-root' in result.stdout and '--verify' in result.stdout


def test_completed_growth_is_retained_when_sigma8_check_fails(monkeypatch,tmp_path):
    module=api();(tmp_path/'workers').mkdir()
    request={'settings':{'nfft':1024},'source_identity_hash':'measured-source',
             'physical_root':'physical-cohort'}
    module.write_json(tmp_path/'request.json',request)
    growth={'fiducial':[.4], 'jacobian':[[.2]], 'finite_difference':[[.20001]]}
    like=SimpleNamespace(reference_cosmology={},config=SimpleNamespace(nominal_bins=[(.8,1.)]))
    # Replace costly external measurements, keeping worker persistence and failure handling real.
    monkeypatch.setattr(module,'_check_request',lambda request:None)
    monkeypatch.setattr(module,'source_identity',lambda settings:{'identity_hash':'measured-source'})
    monkeypatch.setattr(module,'load_physical_likelihood',lambda root,case:like)
    monkeypatch.setattr(module,'make_one_loop_engine',lambda **kwargs:None)
    monkeypatch.setattr(module,'elf_cosmology_spec',lambda reference:None)
    monkeypatch.setattr(module,'measure_growth_jacobian',lambda *args:growth)
    monkeypatch.setattr(module,'measure_sigma8_radius',lambda *args:{'disagreement':.01})
    def failed_radius(*args):
        raise ValueError('sigma8 integral disagrees')
    monkeypatch.setattr(module,'validate_sigma8_radius',failed_radius)
    assert module.run_worker(tmp_path,'B')==1
    receipt=module.read_json(tmp_path/'workers/B.json')
    assert receipt['status']=='failed' and receipt['reason']=='sigma8 integral disagrees'
    saved=tmp_path/'workers/B-growth.json'
    assert saved.exists(),'Completed growth measurement was lost after a later failure'
    report=module.read_json(saved)
    assert report['measurement']==growth
    assert report['case']=='B' and report['source_identity_hash']=='measured-source'
    assert report['request_sha256']==module.request_digest(request)
    assert receipt['growth_sha256']==module.file_digest(saved)


@pytest.mark.parametrize('damage',['source','request','case','measurement'])
def test_growth_checkpoint_rejects_foreign_or_changed_measurement(tmp_path,damage):
    module=api();(tmp_path/'workers').mkdir();(tmp_path/'cases/B').mkdir(parents=True)
    request={'source_identity_hash':'measured-source'}
    report={'case':'B','source_identity_hash':'measured-source',
            'request_sha256':module.request_digest(request),
            'full_forecast_qualification':'pending','measurement':{'fiducial':[.4]}}
    saved=tmp_path/'workers/B-growth.json'
    module.write_json(saved,report);module.write_json(tmp_path/'cases/B/growth.json',report['measurement'])
    worker={'growth_sha256':module.file_digest(saved)}
    module.verify_growth_checkpoint(tmp_path,'B',request,worker)
    if damage=='source':report['source_identity_hash']='foreign'
    elif damage=='request':report['request_sha256']='foreign'
    elif damage=='case':report['case']='C'
    else:report['measurement']['fiducial']=[.5]
    module.write_json(saved,report);worker['growth_sha256']=module.file_digest(saved)
    with pytest.raises(ValueError,match='growth'):
        module.verify_growth_checkpoint(tmp_path,'B',request,worker)
