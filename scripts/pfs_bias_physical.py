"""Fresh means, independent nuisance responses and AP checks for each population.

Measurement follows the frozen validate_elf_engine.measure_cell algebra with
an explicit guarded Gamma3 zero witness; no inherited function is patched.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import time
import traceback

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import CROSS_STOCH_NAMES
from pfsfog.cosmology_space import validate_coordinate_neighborhood
from pfsfog.elf_covariance_audit import audit_neighborhood
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_survey_evidence import canonical,request_digest,write_json
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_forecast import build_cell_likelihood,make_shell_grid
from pfsfog.revision_theory import NUISANCE_NAMES,fiducial_nuisance,make_pair_response
from pfsfog.revision_validation import compare_cell_derivatives
from pfsfog.revision_validation_physical import _steps
from scripts.validate_elf_engine import pair_functions,_mean_evaluators,_fd_arrays,_white,_write_json
from scripts.validate_elf_survey import covariance_inputs
from scripts.elf_study_evidence import study_config_from_request
from scripts.pfs_bias_cases import bias_cases
from scripts.pfs_bias_evidence import bias_source,validate_bias_request,bias_zero_support


def measure_bias_cell(case, config, cosmo, engine, output):
    """Save the live assembled D,C and all independent signed FD means."""
    support, fids = case["cell"], case["fiducials"]
    region = case.get("region", "overlap")
    cell = build_cell_likelihood(config, support, region, fids, cosmo, engine)
    ibin = cell.metadata["nominal_bin_index"]
    cell.metadata["cell_id"] = case.get("cell_id", f"elf-{case['label']}-radial0-{region}")
    space = elf_cosmology_spec(cosmo.params)
    theta = np.array([cosmo.params[{"Mnu": "mnu", "Omegam": "Omega_m"}.get(n, n)]
                      for n in space.names])
    domain = validate_coordinate_neighborhood(theta, {**cosmo.params, "z": support.model_redshift}, space)
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    q = {n: np.asarray(fiducial_nuisance(fids[n])) for n in cell.metadata["active_populations"]}
    values = dict(zip(space.names, theta))
    for tracer in q:
        values.update({f"bin{ibin:02d}:{tracer}:{name}": value for name, value in zip(NUISANCE_NAMES, q[tracer])})
    pair_data = {}
    intrinsic = np.zeros_like(cell.derivatives)
    geometric = np.zeros_like(cell.derivatives)
    spectra = []
    signed_mu = np.polynomial.legendre.leggauss(config.n_mu)[0]
    for ipair, pair in enumerate(cell.metadata["retained_pairs"]):
        a, b = pair
        cross = a != b
        response = make_pair_response(engine, support.model_redshift, cosmo.params,
            support.nbar[a], support.nbar[b] if cross else None, config.k_nl, space=space)
        local = list(space.names)+[f"bin{ibin:02d}:{a}:{n}" for n in NUISANCE_NAMES]
        if cross:
            local += [f"bin{ibin:02d}:{b}:{n}" for n in NUISANCE_NAMES]
            labels = [f"bin{ibin:02d}:{a}|{b}:{n}" for n in CROSS_STOCH_NAMES]
            local += labels
            values.update(dict.fromkeys(labels, 0.))
        packed = jnp.array([values[n] for n in local])
        functions = pair_functions(response, jnp.array(grid.centers), packed, n_mu=config.n_mu)
        indices = [cell.names.index(n) for n in local]
        pair_data[tuple(pair)] = {"functions": functions, "indices": indices, "packed": np.asarray(packed)}
        rows = slice(3*ipair, 3*ipair+3)
        for component, dest in [("intrinsic", intrinsic), ("geometry", geometric)]:
            derivative = np.asarray(jax.jacfwd(functions[component])(packed))
            dest[:, rows, indices] = derivative
        spectra.append(np.asarray(response.pkmu(grid.centers, signed_mu, theta, q[a],
                         q[b] if cross else None, jnp.zeros(2) if cross else None)))
    means = _mean_evaluators(pair_data, cell.names)
    alphas = [float(value) for value in response.ap_alphas(theta)]
    zero_support = bias_zero_support(cell.names, cell.metadata["retained_pairs"],
        alphas=alphas, settings=engine.numerical_settings())
    steps = {n: _steps(n, values[n], space=space) for n in cell.names}
    scale = float(np.linalg.norm(_white(cell.covariance, cell.fiducial_mean[..., None])))
    arrays = {"k": grid.centers, "dk": grid.widths, "signed_mu": signed_mu,
              "pkmu": np.stack(spectra), "mean": cell.fiducial_mean,
              "D": cell.derivatives, "C": cell.covariance, "F_like": cell.F_like,
              "D_intrinsic": intrinsic, "D_geometry": geometric,
              "fiducial": np.array([values[n] for n in cell.names])}
    linear = response.plin(response.z, response._cosmology(jnp.array(theta)))
    arrays.update(linear_k=np.asarray(linear["k"]), linear_pk=np.asarray(linear["pk"]))
    _write_json(output / "inputs.json", {"scope": case["scope"], "configuration": asdict(config),
        "metadata": cell.metadata, "fiducials": values, "steps": steps, "domain": domain,
        "engine_settings": engine.numerical_settings(), "units": space.unit_convention,
        "h_ref": space.h_ref, "inactive_ps_controls": ["ap_interpolation_nk", "ap_refinement_nk", "ap_interpolation_nmu"],
        "analytic_zero_response": {"alphas": alphas,
            "condition": "identity AP; counterterm Legendre support; IR-off Gamma3 A(k)+B(k)mu^2",
            "support": [[name, list(pair), ell] for name, pair, ell in zero_support]}})
    np.savez_compressed(output / "arrays.npz", **arrays)
    print(f"{case['label']}: assembled {len(cell.names)} columns, {len(spectra)} pairs, {len(grid.centers)} shells", flush=True)
    fd = _fd_arrays(cell, means, values, steps)
    arrays.update({f"fd_{key}": value for key, value in fd.items()})
    total_check = compare_cell_derivatives(cell, means, values, steps,
        configuration=case["label"], physical_scale=scale, space=space,
        analytic_zero_support=zero_support)
    splits = {}
    for component, derivative in [("intrinsic", intrinsic), ("geometry", geometric)]:
        hcell = replace(cell, names=["h"], derivatives=derivative[..., 3:4])
        hmeans = _mean_evaluators(pair_data, cell.names, component=component, h_only=True)
        hvalues, hsteps = {"h": values["h"]}, {"h": steps["h"]}
        hfd = _fd_arrays(hcell, hmeans, hvalues, hsteps)
        arrays.update({f"h_{component}_{key}": value for key, value in hfd.items()})
        splits[component] = compare_cell_derivatives(hcell, hmeans, hvalues, hsteps,
            configuration=f"{case['label']}-{component}-h", physical_scale=scale, space=space)
    chain = _white(cell.covariance, cell.derivatives-intrinsic-geometric)
    reference = _white(cell.covariance, cell.derivatives)
    chain_error = float(np.linalg.norm(chain)/np.linalg.norm(reference))
    status = "passed" if total_check["status"] == "passed" and all(
        x["status"] == "passed" for x in splits.values()) and chain_error < 1.e-10 else "failed"
    result = {"status": status, "scope": case["scope"], "metadata": cell.metadata,
              "covariance": cell.validation, "ad_fd": total_check, "h_split_fd": splits,
              "chain_relative_error": chain_error, "chain_tolerance": 1.e-10}
    np.savez_compressed(output / "arrays.npz", **arrays)
    result["arrays_sha256"] = hashlib.sha256((output / "arrays.npz").read_bytes()).hexdigest()
    _write_json(output / "measurement.json", result)
    return cell, result



def measure_bias_population(request,key,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(p.name!='worker.log' for p in output.iterdir()):
        raise FileExistsError('refusing to overwrite physical evidence')
    started=time.monotonic();engine=None
    receipt={'key':key,'status':'running','request_sha256':request_digest(request),
        'source_before':None,'source_after':None,'full_forecast_qualification':'pending'}
    write_json(output/'worker.json',receipt)
    try:
        engine=make_one_loop_engine(nfft=request['settings']['nfft'])
        before=bias_source(request['settings'],engine);receipt['source_before']=before['identity_hash']
        cosmo=FiducialCosmology()
        validate_bias_request(request,cosmo,before)
        expected=next(r for r in request['expected'] if r['key']==key)
        config=study_config_from_request(request);cosmo=FiducialCosmology()
        case=next(c for c in bias_cases(cosmo,request['settings'])
                  if c['label']==key)
        if canonical(case['expected'])!=canonical(expected['support']):raise ValueError('independent support differs')
        covcell,values=covariance_inputs(case,cosmo)
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
        after=bias_source(request['settings'],engine);receipt['source_after']=after['identity_hash']
        if canonical(after)!=canonical(request['source']):receipt.update(status='failed',source_changed=True)
    except Exception as error:receipt.update(status='failed',source_changed=True,source_error=str(error))
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json(output/'worker.json',receipt)
    return receipt

