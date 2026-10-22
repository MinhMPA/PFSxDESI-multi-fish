"""New PFS overlap spectra, physical neighborhoods and direct common-k checks."""
from pathlib import Path
import time
import traceback

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_covariance_audit import audit_neighborhood,covariance_from_spectra
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_survey_evidence import canonical,request_digest,write_json
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_theory import NUISANCE_NAMES,make_pair_response
from pfsfog.eft_params import CROSS_STOCH_NAMES
from scripts.validate_elf_engine import measure_cell
from scripts.validate_elf_survey import covariance_inputs
from scripts.elf_study_profiles import highz_profiles,study_cells
from scripts.elf_study_evidence import study_source,make_study_request,study_config_from_request


def measure_highz_cell(request,key,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(p.name!='worker.log' for p in output.iterdir()):
        raise FileExistsError('refusing to overwrite physical evidence')
    started=time.monotonic();engine=None
    receipt={'key':key,'status':'running','request_sha256':request_digest(request),
        'source_before':None,'source_after':None,'full_forecast_qualification':'pending'}
    write_json(output/'worker.json',receipt)
    try:
        engine=make_one_loop_engine(nfft=request['settings']['nfft'])
        before=study_source(request['settings'],engine);receipt['source_before']=before['identity_hash']
        if canonical(make_study_request(request['settings'],source=before))!=canonical(request):
            raise ValueError('worker source or requested physical support differs')
        expected=next(r for r in request['expected'] if r['key']==key)
        config=study_config_from_request(request);cosmo=FiducialCosmology()
        case=next(c for c in study_cells(highz_profiles()['C22'],cosmo,settings=request['settings'],extended=True)
                  if c['label']==key)
        if canonical(case['expected'])!=canonical(expected['support']):raise ValueError('independent support differs')
        covcell,values=covariance_inputs(case,cosmo)
        write_json(output/'covariance_inputs.json',{'configuration':request['configuration'],
            'metadata':covcell.metadata,'fiducials':values,'engine_settings':engine.numerical_settings()})
        neighborhood=audit_neighborhood(covcell,config,cosmo.params,engine,values,output)
        receipt['covariance_status']=neighborhood['status']
        if neighborhood['status']!='passed':raise ValueError('nonphysical covariance neighborhood')
        cell,measurement=measure_cell(case,config,cosmo,engine,output)
        for field,value in case['expected'].items():
            if field not in ('empty_observation','absent_observations') and canonical(cell.metadata.get(field))!=canonical(value):
                raise ValueError('measured support differs: '+field)
        receipt.update(status=measurement['status'],mean_status=measurement['status'])
    except Exception as error:
        receipt.update(status='failed',reason=str(error),error_type=type(error).__name__)
        (output/'failure.log').write_text(traceback.format_exc())
    try:
        after=study_source(request['settings'],engine);receipt['source_after']=after['identity_hash']
        if canonical(after)!=canonical(request['source']):receipt.update(status='failed',source_changed=True)
    except Exception as error:receipt.update(status='failed',source_changed=True,source_error=str(error))
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json(output/'worker.json',receipt)
    return receipt


def common_query_data(inputs,query,engine,cosmo):
    """Direct ELF evaluation on the union of shell queries; no Fisher summation."""
    query=np.asarray(query);meta=inputs['metadata'];values=inputs['fiducials'];names=meta['local_names']
    if query.ndim!=1 or np.any(query<=0) or not np.isfinite(query).all() or np.any(np.diff(query)<=0):
        raise ValueError('invalid common k query')
    config=inputs['configuration'];space=elf_cosmology_spec(cosmo.params)
    ibin=meta['nominal_bin_index'];nmu=config['n_mu'];mu=np.polynomial.legendre.leggauss(nmu)[0]
    pairs=list(map(tuple,meta['retained_pairs']));theoretical=list(map(tuple,meta['theoretical_pairs']))
    mean=np.zeros((len(query),3*len(pairs)));derivative=np.zeros((*mean.shape,len(names)));spectra=[]
    for a,b in theoretical:
        cross=a!=b;labels=list(space.names)+[f'bin{ibin:02d}:{a}:{n}' for n in NUISANCE_NAMES]
        if cross:labels += [f'bin{ibin:02d}:{b}:{n}' for n in NUISANCE_NAMES]+[f'bin{ibin:02d}:{a}|{b}:{n}' for n in CROSS_STOCH_NAMES]
        packed=jnp.array([values[n] for n in labels]);nc=len(space.names);nq=len(NUISANCE_NAMES)
        response=make_pair_response(engine,meta['model_redshift'],cosmo.params,meta['nbar'][a],
            meta['nbar'][b] if cross else None,config['k_nl'],space=space)
        def split(v):return v[:nc],v[nc:nc+nq],v[nc+nq:nc+2*nq] if cross else None,v[-2:] if cross else None
        spectra.append(np.asarray(response.pkmu(query,mu,*split(packed))))
        if (a,b) in pairs:
            def multipoles(v):return response.multipoles(query,*split(v),n_mu=nmu)
            rows=slice(3*pairs.index((a,b)),3*pairs.index((a,b))+3)
            mean[:,rows]=np.asarray(multipoles(packed));jac=np.asarray(jax.jacfwd(multipoles)(packed))
            for local,label in enumerate(labels):derivative[:,rows,names.index(label)]=jac[:,:,local]
    _,kernel=covariance_from_spectra(np.asarray(spectra),meta,query,np.ones(len(query)),n_mu=nmu)
    return {'k':query,'mean':mean,'D':derivative,'covariance_kernel':kernel}
