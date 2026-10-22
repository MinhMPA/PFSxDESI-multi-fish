"""Fresh spectra for broader populations using the frozen physical primitives."""
from pathlib import Path
import time
import traceback

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_covariance_audit import audit_neighborhood
from pfsfog.elf_survey_evidence import canonical, request_digest, write_json
from pfsfog.one_loop import make_one_loop_engine
from scripts.elf_study_evidence import study_config_from_request
from scripts.pfs_bias_physical import measure_bias_cell
from scripts.pfs_bias_broad_cases import broad_cases
from scripts.pfs_bias_broad_evidence import broad_source, validate_broad_request
from scripts.validate_elf_survey import covariance_inputs


def measure_broad_population(request,key,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(p.name!='worker.log' for p in output.iterdir()):
        raise FileExistsError('refusing to overwrite broad physical evidence')
    started=time.monotonic();engine=None
    receipt={'key':key,'status':'running','request_sha256':request_digest(request),
        'source_before':None,'source_after':None,'full_forecast_qualification':'pending'}
    write_json(output/'worker.json',receipt)
    try:
        engine=make_one_loop_engine(nfft=request['settings']['nfft'])
        before=broad_source(request['settings'],engine);receipt['source_before']=before['identity_hash']
        cosmo=FiducialCosmology();validate_broad_request(request,cosmo,before)
        expected=next(r for r in request['expected'] if r['key']==key)
        case=next(c for c in broad_cases(cosmo,request['settings']) if c['label']==key)
        if canonical(case['expected'])!=canonical(expected['support']):
            raise ValueError('independent broad support differs')
        config=study_config_from_request(request)
        covcell,values=covariance_inputs(case,cosmo)
        if canonical(values)!=canonical(expected['fiducials']):
            raise ValueError('independent broad fiducials differ')
        write_json(output/'covariance_inputs.json',{'configuration':request['configuration'],
            'metadata':covcell.metadata,'fiducials':values,'engine_settings':engine.numerical_settings()})
        neighborhood=audit_neighborhood(covcell,config,cosmo.params,engine,values,output)
        receipt['covariance_status']=neighborhood['status']
        if neighborhood['status']!='passed':raise ValueError('nonphysical covariance neighborhood')
        cell,measurement=measure_bias_cell(case,config,cosmo,engine,output)
        for field,value in case['expected'].items():
            if field not in ('empty_observation','absent_observations') and canonical(cell.metadata.get(field))!=canonical(value):
                raise ValueError('measured support differs: '+field)
        receipt.update(status=measurement['status'],mean_status=measurement['status'])
    except Exception as error:
        receipt.update(status='failed',reason=str(error),error_type=type(error).__name__)
        (output/'failure.log').write_text(traceback.format_exc())
    try:
        after=broad_source(request['settings'],engine);receipt['source_after']=after['identity_hash']
        if canonical(after)!=canonical(request['source']):receipt.update(status='failed',source_changed=True)
    except Exception as error:receipt.update(status='failed',source_changed=True,source_error=str(error))
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json(output/'worker.json',receipt)
    return receipt

