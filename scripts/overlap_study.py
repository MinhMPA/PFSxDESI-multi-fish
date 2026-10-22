"""Matched DESI+PFS comparisons selected from measured spectrum likelihoods."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from pfsfog.elf_covariance_audit import audit_spd_batch
from pfsfog.fisher_core import fisher_from_shells
from pfsfog.revision_forecast import (CellLikelihood, RevisionLikelihood, ShellGrid, PriorResult,
                                     project_baseline_precision)
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from scripts.overlap_inputs import PublicCase, Treatment, TREATMENTS, _cross_families


# Only RevisionConfig retains the historical case convention.
_LEGACY_CONFIG_CASE = {'A': 'A', 'B': 'B', 'C': 'D', 'D': 'C'}


def _pairs(populations, case):
    active = sorted(n for n in populations if case in ('C', 'D') or n.startswith('DESI-'))
    pairs = [(n, n) for n in active]
    if case != 'A':
        pairs += [(a, b) for i, a in enumerate(active) for b in active[i+1:]
                  if case == 'D' or 'PFS-ELG' not in (a, b)]
    return active, pairs


def _retained_parameter(record, active, pairs, nominal_bin):
    role = record['role']
    if role == 'cosmo':
        return True
    if record['nominal_bin'] != nominal_bin:
        raise ValueError('local nuisance belongs to another nominal bin')
    tracers = tuple(record['tracers'])
    if role == 'tracer' and len(tracers) == 1:
        return tracers[0] in active
    if role == 'cross' and len(tracers) == 2:
        return tracers in pairs
    raise ValueError('unknown nuisance population or pair record')


def select_observations(source: RevisionLikelihood, case: PublicCase) -> RevisionLikelihood:
    """Restrict means, responses and both covariance axes before summing DᵀC⁻¹D.

    No priors, area rescaling or nuisance conditioning enter this operation.
    An absent population or measured pair loses its nuisance coordinates only
    when their responses vanish exactly on every retained mean.
    """
    if case not in _LEGACY_CONFIG_CASE:
        raise ValueError('unknown public case')
    cosmology = list(source.cosmo_names)
    if any(source.names.count(n) != 1 for n in cosmology):
        raise ValueError('missing or duplicate global cosmological column')
    if (len(set(source.names)) != len(source.names)
            or [p['name'] for p in source.parameters] != source.names
            or np.shape(source.fiducial) != (len(source.names),)):
        raise ValueError('global parameter records, names or fiducials differ')
    records = {p['name']: p for p in source.parameters}
    if {p['name'] for p in source.parameters if p['role'] == 'cosmo'} != set(cosmology):
        raise ValueError('cosmological parameter records differ')
    if not np.array_equal(source.fiducial, [p['fiducial'] for p in source.parameters]):
        raise ValueError('source fiducials differ from parameter records')
    source_index = {n: i for i, n in enumerate(source.names)}
    ids = [c.metadata['cell_id'] for c in source.cells]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('missing or duplicate source cell')
    nk = len(source.grid.centers)
    cells = []
    for original in source.cells:
        meta = deepcopy(original.metadata)
        ibin = meta['nominal_bin_index']
        if not 0 <= ibin < len(source.config.nominal_bins):
            raise ValueError('source cell nominal bin differs')
        old_names = original.names
        if (len(set(old_names)) != len(old_names) or meta['local_names'] != old_names
                or any(n not in records for n in old_names)):
            raise ValueError('local parameter inventory differs')
        old_obs = [(tuple(pair), ell) for pair, ell in meta['retained_observations']]
        old_pairs = list(map(tuple, meta['retained_pairs']))
        if (len(set(old_obs)) != len(old_obs)
                or set(old_obs) != {(p, ell) for p in old_pairs for ell in (0, 2, 4)}):
            raise ValueError('source measured pair/multipole inventory differs')
        if old_obs and any(old_names.count(n) != 1 for n in cosmology):
            raise ValueError('missing local cosmological derivative column')
        active, pairs = _pairs(meta['active_populations'], case)
        observations = [(p, ell) for p in pairs for ell in (0, 2, 4)]
        if not set(observations) <= set(old_obs):
            raise ValueError('source lacks a required measured pair/multipole')
        if (original.derivatives.shape != (nk, len(old_obs), len(old_names))
                or original.covariance.shape != (nk, len(old_obs), len(old_obs))
                or original.fiducial_mean.shape != (nk, len(old_obs))):
            raise ValueError('source mean, response or covariance layout differs')
        rows = [old_obs.index(o) for o in observations]
        names = [n for n in old_names
                 if _retained_parameter(records[n], active, pairs, ibin)] if pairs else []
        cols = [old_names.index(n) for n in names]
        omitted = [i for i, n in enumerate(old_names) if n not in names]
        retained_response = original.derivatives[:, rows]
        if omitted and np.any(retained_response[:, :, omitted] != 0.):
            raise ValueError('omitted nuisance has nonzero retained response')
        derivatives = retained_response[:, :, cols].copy()
        covariance = original.covariance[:, rows][:, :, rows].copy()
        mean = original.fiducial_mean[:, rows].copy()
        if not np.isfinite(mean).all():
            raise ValueError('nonfinite retained mean')
        theoretical = [(n, n) for n in active] + [(a, b) for i, a in enumerate(active) for b in active[i+1:]]
        meta.update(active_populations=active, retained_pairs=pairs,
                    retained_observations=observations, theoretical_pairs=theoretical,
                    absent_observations=[(p, ell) for p in theoretical if p not in pairs for ell in (0, 2, 4)],
                    local_names=names, nbar={n: meta['nbar'][n] for n in active},
                    empty_observation=not bool(pairs),
                    selection={'case': case, 'source_cell_id': meta['cell_id'],
                               'source_rows': rows, 'source_columns': cols})
        validation = deepcopy(original.validation)
        validation.update(expected_shells=nk if pairs else 0, processed_shells=nk if pairs else 0)
        if pairs:
            audit = audit_spd_batch(covariance, 'selected multipole covariance')
            validation.update(covariance_min_scaled_eigenvalue=audit['min_scaled_eigenvalue'],
                              covariance_max_scaled_condition=audit['max_scaled_condition'],
                              max_scaled_residual=audit['max_scaled_residual'])
        fisher = fisher_from_shells(derivatives, covariance, source.grid.centers)
        cells.append(CellLikelihood(fisher, names, meta, validation, derivatives, covariance, mean))

    retained = {n for c in cells for n in c.names}
    names = cosmology + [n for n in source.names if n in retained and n not in cosmology]
    if not set(cosmology) <= retained:
        raise ValueError('selected observations have no cosmological response columns')
    columns = [source_index[n] for n in names]
    index = {n: i for i, n in enumerate(names)}
    bins = [np.zeros((len(names), len(names))) for _ in source.config.nominal_bins]
    for cell in cells:
        local = [index[n] for n in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(local, local)] += cell.F_like
    fisher = np.sum(bins, axis=0)
    parameters = [deepcopy(records[n]) for n in names]
    for i, p in enumerate(parameters):
        p['prior_only'] = bool(fisher[i, i] == 0. and p['prior_sigma'] is not None)
    validation = deepcopy(source.validation)
    validation.update(cell_count=len(cells), empty_cells=[c.metadata['cell_id'] for c in cells if not c.names],
                      expected_shells=sum(c.validation['expected_shells'] for c in cells),
                      processed_shells=sum(c.validation['processed_shells'] for c in cells),
                      min_covariance_scaled_eigenvalue=min(c.validation['covariance_min_scaled_eigenvalue']
                                                          for c in cells if c.names),
                      selection={'case': case, 'source_config_case': source.config.case,
                                 'source_columns': columns,
                                 'new_forecast_qualification': 'pending'})
    grid = ShellGrid(*(np.array(getattr(source.grid, n), copy=True) for n in ('edges', 'centers', 'widths')))
    return RevisionLikelihood(replace(source.config, case=_LEGACY_CONFIG_CASE[case]), fisher, names,
                              source.fiducial[columns].copy(), parameters, cells, bins, grid,
                              deepcopy(source.reference_cosmology), deepcopy(source.model), validation)


def volume_weight(radial_bin: tuple[float, float], region: str, area_deg2: int) -> float:
    """Common-volume weight at fixed 14,000 deg² DESI coverage.

    The two stored angular pieces below z=0.6 (or above z=2.1) retain
    their original volumes. A radial cell must be split at either PFS
    selection boundary before an angular weight can be assigned.
    """
    if type(area_deg2) is not int or area_deg2 not in (1200, 14000):
        raise ValueError('unsupported PFS area')
    if region not in ('overlap', 'nonoverlap'):
        raise ValueError('unknown angular region')
    bounds = np.asarray(radial_bin, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('invalid radial bin')
    lo, hi = bounds
    if any(lo < boundary < hi for boundary in (.6, 2.1)):
        raise ValueError('radial bin crosses an unsplit PFS selection boundary')
    if hi <= .6 or lo >= 2.1:
        return 1.
    return area_deg2 / 1200 if region == 'overlap' else (14000 - area_deg2) / 12800


def scale_area(like: RevisionLikelihood, area_deg2: int) -> RevisionLikelihood:
    """Rescale independent angular volumes, keeping means and responses fixed.

    At positive weight w the Gaussian covariance is C/w and the cell
    information is rebuilt from the measured multipole responses. A vanished
    region has no consumable rows or local parameters. Named nuisance
    coordinates in the surviving cells remain shared in the survey sum;
    no prior is added here.
    """
    # Validate even when the supplied cell inventory happens to be empty.
    volume_weight((.4, .6), 'overlap', area_deg2)
    cells = []
    nk = len(like.grid.centers)
    for original in like.cells:
        source_meta = original.metadata
        w = volume_weight(source_meta['radial_bin'], source_meta['region'], area_deg2)
        meta = deepcopy(source_meta)
        meta.update(volume=w * source_meta['volume'],
                    area_scaling={'area_deg2': area_deg2, 'weight': w,
                                  'source_metadata': deepcopy(source_meta)})
        validation = deepcopy(original.validation)
        if w == 0:
            names = []
            derivatives, covariance, mean = np.zeros((nk, 0, 0)), np.zeros((nk, 0, 0)), np.zeros((nk, 0))
            fisher = np.zeros((0, 0))
            meta.update(local_names=[], retained_pairs=[], retained_observations=[],
                        absent_observations=[(tuple(p), ell) for p in meta['theoretical_pairs'] for ell in (0, 2, 4)],
                        empty_observation=True)
            validation = {'expected_shells': 0, 'processed_shells': 0,
                          'empty_reason': 'zero angular volume'}
        else:
            names = list(original.names)
            derivatives = original.derivatives.copy()
            covariance = original.covariance / w
            mean = original.fiducial_mean.copy()
            fisher = fisher_from_shells(derivatives, covariance, like.grid.centers)
            expected = w * original.F_like
            scale = np.sqrt(np.diag(expected)).copy()
            scale[scale == 0] = 1.
            residual = np.max(np.abs(fisher - expected) / scale[:, None] / scale[None, :], initial=0.)
            if not np.isfinite(residual) or residual > 1e-10:
                raise ValueError('rescaled cell Fisher differs from volume-weighted source')
            validation['area_fisher_scaled_residual'] = float(residual)
            if names:
                audit = audit_spd_batch(covariance, 'area-scaled multipole covariance')
                validation.update(covariance_min_scaled_eigenvalue=audit['min_scaled_eigenvalue'],
                                  covariance_max_scaled_condition=audit['max_scaled_condition'],
                                  max_scaled_residual=audit['max_scaled_residual'])
        cells.append(CellLikelihood(fisher, names, meta, validation, derivatives, covariance, mean))

    surviving = {n for cell in cells for n in cell.names}
    names = [n for n in like.names if n in surviving]
    if not set(like.cosmo_names) <= surviving:
        raise ValueError('area-scaled observations have no cosmological response columns')
    columns = [like.names.index(n) for n in names]
    index = {n: i for i, n in enumerate(names)}
    bins = [np.zeros((len(names), len(names))) for _ in like.config.nominal_bins]
    for cell in cells:
        local = [index[n] for n in cell.names]
        bins[cell.metadata['nominal_bin_index']][np.ix_(local, local)] += cell.F_like
    fisher = np.sum(bins, axis=0)
    parameters = [deepcopy(like.parameters[i]) for i in columns]
    for i, p in enumerate(parameters):
        p['prior_only'] = bool(fisher[i, i] == 0. and p['prior_sigma'] is not None)
    validation = deepcopy(like.validation)
    validation.update(cell_count=len(cells), empty_cells=[c.metadata['cell_id'] for c in cells if not c.names],
                      expected_shells=sum(c.validation['expected_shells'] for c in cells),
                      processed_shells=sum(c.validation['processed_shells'] for c in cells),
                      min_covariance_scaled_eigenvalue=min(c.validation['covariance_min_scaled_eigenvalue']
                                                          for c in cells if c.names),
                      area_scaling={'area_deg2': area_deg2, 'desi_area_deg2': 14000,
                                    'source_columns': columns, 'new_forecast_qualification': 'pending'})
    grid = ShellGrid(*(np.array(getattr(like.grid, n), copy=True) for n in ('edges', 'centers', 'widths')))
    return RevisionLikelihood(like.config, fisher, names, like.fiducial[columns].copy(), parameters,
                              cells, bins, grid, deepcopy(like.reference_cosmology), deepcopy(like.model), validation)


def fix_cross_coefficients(like: RevisionLikelihood, treatment: Treatment) -> tuple[RevisionLikelihood, tuple[str, ...]]:
    """Condition chosen residual cross-stochastic coefficients at exactly zero.

    Conditioning restricts the response columns. It does not restrict the
    observed spectra or their Gaussian covariance, nor add a limiting prior.
    The returned fixed-coordinate record is separate from spectrum selection.
    """
    if treatment not in TREATMENTS:
        raise ValueError('unknown residual cross-stochastic treatment')
    if (len(set(like.names)) != len(like.names)
            or [p['name'] for p in like.parameters] != like.names
            or not np.array_equal(like.fiducial, [p['fiducial'] for p in like.parameters])):
        raise ValueError('conditioning parameter names or fiducials differ')
    families = _cross_families(like.parameters, like.cells)
    chosen = set()
    if treatment in ('desi_fixed', 'all_cross_fixed'):
        chosen.update(families['desi'])
    if treatment in ('pfs_fixed', 'all_cross_fixed'):
        chosen.update(families['pfs'])
    fixed = tuple(n for n in like.names if n in chosen)
    names = [n for n in like.names if n not in chosen]
    columns = [like.names.index(n) for n in names]
    index = {n: i for i, n in enumerate(names)}
    conditioning = {'treatment': treatment, 'fixed_names': list(fixed),
                    'fixed_values': [0.] * len(fixed)}
    cells = []
    bins = [np.zeros((len(names), len(names))) for _ in like.config.nominal_bins]
    for original in like.cells:
        local_names = [n for n in original.names if n not in chosen]
        keep = [original.names.index(n) for n in local_names]
        d = original.derivatives[:, :, keep].copy()
        c, mean = original.covariance.copy(), original.fiducial_mean.copy()
        fisher = fisher_from_shells(d, c, like.grid.centers)
        meta, validation = deepcopy(original.metadata), deepcopy(original.validation)
        meta.update(local_names=local_names, conditioning=deepcopy(conditioning))
        cells.append(CellLikelihood(fisher, local_names, meta, validation, d, c, mean))
        ids = [index[n] for n in local_names]
        bins[meta['nominal_bin_index']][np.ix_(ids, ids)] += fisher
    fisher = np.sum(bins, axis=0)
    parameters = [deepcopy(like.parameters[i]) for i in columns]
    for i, p in enumerate(parameters):
        p['prior_only'] = bool(fisher[i, i] == 0. and p['prior_sigma'] is not None)
    validation = deepcopy(like.validation)
    validation['conditioning'] = conditioning
    return replace(like, names=names, F_like=fisher, fiducial=like.fiducial[columns].copy(),
                   parameters=parameters, cells=cells, bin_F_like=bins, validation=validation), fixed


def infer_saved_growth(like: RevisionLikelihood, reference: PriorResult) -> PriorResult:
    """Add the agreed priors once and propagate the authenticated fσ8 response.

    Spectrum/area selection and nuisance conditioning can change the Fisher
    matrix, but cannot change the background, parameter units or growth model.
    Authentication of the reference arrays is the responsibility of load_inputs.
    """
    source = reference.likelihood
    if (like.cosmo_names != source.cosmo_names
            or like.names[:3] != list(source.cosmo_names)
            or source.names[:3] != list(source.cosmo_names)
            or like.reference_cosmology != source.reference_cosmology
            or like.model != source.model
            or like.config.nominal_bins != source.config.nominal_bins):
        raise ValueError('saved growth cosmology, model or nominal redshifts differ')
    records = {p['name']: p for p in source.parameters}
    if ([p['name'] for p in like.parameters] != like.names
            or not np.array_equal(like.fiducial, [p['fiducial'] for p in like.parameters])
            or not np.array_equal(source.fiducial, [p['fiducial'] for p in source.parameters])):
        raise ValueError('saved growth parameter names or fiducials differ')
    for p in like.parameters:
        if p['name'] not in records or any(p[k] != records[p['name']][k]
                for k in ('coordinate', 'role', 'fiducial', 'unit', 'nominal_bin', 'tracers', 'prior_sigma')):
            raise ValueError('saved growth parameter definition differs')
    source_cells = {c.metadata['cell_id']: c for c in source.cells}
    for cell in like.cells:
        original = source_cells.get(cell.metadata['cell_id'])
        if original is None or any(cell.metadata[k] != original.metadata[k]
                for k in ('nominal_bin_index', 'model_redshift')):
            raise ValueError('saved growth cell model redshift differs')
    z = np.asarray([(a+b)/2 for a, b in like.config.nominal_bins])
    if (reference.growth_z.shape != z.shape
            or not np.allclose(reference.growth_z, z, rtol=0, atol=1e-12)
            or reference.growth_fiducial.shape != z.shape
            or reference.growth_jacobian.shape != (len(z), 3)
            or not all(np.isfinite(a).all() for a in
                       (reference.growth_z, reference.growth_fiducial, reference.growth_jacobian))):
        raise ValueError('saved growth arrays or nominal model redshifts differ')
    precision = project_baseline_precision(like, 1., fisher_solver=DECIMAL_FISHER_SOLVER)
    jac = reference.growth_jacobian.copy()
    return PriorResult(likelihood=like, **{k: v for k, v in precision.items() if k != 'F_like'},
                       growth_z=reference.growth_z.copy(),
                       growth_fiducial=reference.growth_fiducial.copy(), growth_jacobian=jac,
                       growth_cov_marg=jac @ precision['cov_cosmo_marg'] @ jac.T,
                       growth_cov_nuisance_fixed=jac @ precision['cov_cosmo_nuisance_fixed'] @ jac.T)
