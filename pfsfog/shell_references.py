"""Original-shell replay, kept separate from candidate-shell basis equality.

The caller supplies the authenticated control operations. Native3 is built
independently at its own physical configuration, never sliced from extended6.
These controls measure output resolution; they do not qualify derivatives.
"""
from dataclasses import replace
from pathlib import Path

from .extended_artifacts import jsonable, physical_likelihood_id, identity_provenance
from .fisher_precision import fisher_solver_record, solver_from_record
from .shell_numerics import (FINE_SHELL_PROFILE, shell_profile_from_record,
                             shell_profile_record, validate_shell_configuration)

_CONTROLS = {
    'original_grid_extended6_8192': (8192, 'extended6', 'original_grid_replay'),
    'original_grid_native3_8192': (8192, 'baseline3', 'original_grid_replay'),
    'original_grid_extended6_16384': (16384, 'extended6', 'original_grid_shell_assessment'),
}


def _configuration(base, nk, basis):
    return replace(base.config, dk=.0025, ap_interpolation_nk=nk,
                   ap_refinement_nk=(2*nk, 4*nk), cosmology_basis=basis)


def measure_original_grid_references(base, oracle, path, provenance, *,
                                     oracle_identity, build_likelihood, control_api):
    profile = shell_profile_from_record(provenance, requested=FINE_SHELL_PROFILE)
    validate_shell_configuration(base.config, profile, 'adopted')
    path = Path(path)
    controls = {}
    old = old_results = None
    for key, (nk, basis, role) in _CONTROLS.items():
        like = build_likelihood(_configuration(base, nk, basis))
        if basis == 'baseline3':
            saved = control_api.check_baseline_control(old, old_results['conditional3'],
                like, oracle, path/key, provenance, oracle_identity=oracle_identity)
            failures = None
        else:
            results, failures = control_api.treatments(like,
                failure_root=path/(key+'-projection'), provenance=provenance)
            control_api.treatment_outcomes(results, failures)
            saved = control_api.save_control(path/key, like, results, provenance=provenance)
            if nk == 8192:
                old, old_results = like, results
        controls[key] = control_api._output_reference(saved, nk, role, failed=failures)
    record = jsonable({'definition':'original-shell-references-v1',
        'shell_profile':shell_profile_record(profile), 'case':base.config.case,
        'base_physical_likelihood_id':physical_likelihood_id(base, provenance),
        'producer_identity':identity_provenance(provenance),
        'fisher_solver':fisher_solver_record(control_api._requested_solver(provenance)),
        'oracle_export':oracle_identity, 'controls':controls})
    control_api.write_json(path/'original-grid-references.json', record)
    validate_original_grid_references(base, record, oracle, provenance,
                                      oracle_identity=oracle_identity, control_api=control_api)
    return record


def validate_original_grid_references(base, record, oracle, provenance, *,
                                      oracle_identity, control_api):
    profile = shell_profile_from_record(provenance, requested=FINE_SHELL_PROFILE)
    validate_shell_configuration(base.config, profile, 'adopted')
    if (not isinstance(record, dict) or record.get('definition') != 'original-shell-references-v1'
            or record.get('shell_profile') != shell_profile_record(profile)
            or record.get('case') != base.config.case
            or record.get('base_physical_likelihood_id') != physical_likelihood_id(base, provenance)
            or record.get('producer_identity') != identity_provenance(provenance)
            or record.get('oracle_export') != oracle_identity
            or solver_from_record(record) != control_api._requested_solver(provenance)
            or not isinstance(record.get('controls'), dict) or set(record['controls']) != set(_CONTROLS)):
        raise ValueError('original-grid reference identity or inventory differs')
    loaded = {}
    for key, (nk, basis, role) in _CONTROLS.items():
        saved = record['controls'][key]
        if (saved.get('reference_role') != role or saved.get('evaluated_ap_grids') != [nk]
                or saved.get('derivative_qualification_performed') is not False):
            raise ValueError('original-grid reference role differs')
        like, results = control_api.load_control(saved, provenance=provenance)
        if like.config != _configuration(base, nk, basis):
            raise ValueError('original-grid reference configuration differs')
        if basis == 'baseline3':
            if set(results) != {'baseline3'}:
                raise ValueError('original-grid native3 result inventory differs')
        else:
            loaded[nk] = (like, control_api.treatment_outcomes(results, saved.get('failed_treatments')))
    like, results = loaded[8192]
    control_api.validate_baseline_control(like, results['conditional3'],
        record['controls']['original_grid_native3_8192'], oracle, provenance,
        oracle_identity=oracle_identity)
    return loaded
