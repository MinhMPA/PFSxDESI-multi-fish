"""Named extended spectrum/AP coverage and independent growth AD/FD gates.

Reports are measurements, not production certificates. The response contract is
``cosmology_columns, points, cells, growth``. Points are raw AP rows, cells retain
full covariance diagnostics, and growth is a separate measurement from
``measure_growth_jacobian``; no spectrum status substitutes for that record.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .cosmology_space import (CosmologySpec, make_cosmology_spec,
                              cosmology_from_coordinates, validate_coordinate_neighborhood)
from .revision_artifacts import _canonical
from .revision_forecast import RevisionLikelihood
from .revision_theory import make_growth_rate_func, make_sigma8_func
from .revision_validation_ap import (_finite, _validate_cell_record,
                                     validate_ap_point_rows, angular_grid_declaration)
from .revision_validation_physical import validate_physical_point

RELATIVE_TOLERANCE = 1e-3
ZERO_TOLERANCE = 1e-6
NEAR_ZERO_SWITCH = 1e-12


def _reference_theta(space, reference):
    source = reference.params if hasattr(reference, 'params') else reference
    return np.array([source['mnu' if name == 'Mnu' else
                            'Omega_m' if name == 'Omegam' else name] for name in space.names])


def _likelihood_space(likelihood):
    space = make_cosmology_spec(likelihood.reference_cosmology, extended=True,
                                ap_nk=likelihood.config.ap_interpolation_nk)
    if (likelihood.config.cosmology_basis != 'extended6'
            or tuple(likelihood.cosmo_names) != space.names
            or len(set(likelihood.names)) != len(likelihood.names)
            or any(likelihood.names.count(name) != 1 for name in space.names)):
        raise ValueError('extended response requires exactly six named cosmology columns')
    return space


def extended_expected_schedule(likelihood: RevisionLikelihood) -> list[dict]:
    """Declare every retained shell/pair/multipole/coordinate/step before evidence."""
    space = _likelihood_space(likelihood)
    rows = []
    ids = [cell.metadata['cell_id'] for cell in likelihood.cells]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('extended cell inventory is empty or duplicated')
    for cell in likelihood.cells:
        pairs = [tuple(pair) for pair in cell.metadata['retained_pairs']]
        if len(set(pairs)) != len(pairs):
            raise ValueError('extended retained pairs are duplicated')
        if pairs and any(cell.names.count(name) != 1 for name in space.names):
            raise ValueError('cell omits a named cosmology column')
        for pair in pairs:
            for k in likelihood.grid.centers:
                for ell in (0, 2, 4):
                    for name in space.names:
                        for step in space.fd_steps[name]:
                            rows.append({'cell_id':cell.metadata['cell_id'], 'pair':list(pair),
                                         'k':float(k), 'ell':ell, 'column':name, 'step':float(step)})
    if not rows:
        raise ValueError('extended spectrum schedule contains no observations')
    return rows


def _growth_units(space):
    return {'output':'dimensionless f*sigma8', 'coordinates':{
        name:('eV' if name == 'Mnu' else 'dimensionless') for name in space.names},
        'h_ref':space.h_ref, 'power_unit_convention':space.unit_convention,
        'sigma8_radius':'8/h_trial Mpc'}


def _growth_diagnostics(ad, fd, fiducial, redshifts, space):
    """Use FD reference scales and retain every failed step and physical raw norm."""
    rows, coverage = [], []
    for iz, z in enumerate(redshifts):
        physical_scale = abs(float(fiducial[iz]))
        if physical_scale <= 0:
            raise ValueError('growth physical scale must be positive')
        for ic, name in enumerate(space.names):
            raw_steps = []
            for js, step in enumerate(space.fd_steps[name]):
                difference = abs(float(ad[iz,ic]-fd[iz,ic,js]))
                reference_norm = abs(float(fd[iz,ic,js]))
                zero = reference_norm <= NEAR_ZERO_SWITCH*physical_scale
                error = difference/(physical_scale if zero else reference_norm)
                limit = ZERO_TOLERANCE if zero else RELATIVE_TOLERANCE
                raw_steps.append({'step':float(step), 'ad_norm':abs(float(ad[iz,ic])),
                    'fd_norm':reference_norm, 'difference_norm':difference,
                    'physical_scale':physical_scale, 'error':error, 'limit':limit,
                    'branch':'normalized absolute' if zero else 'relative', 'passed':bool(error < limit)})
                coverage.append({'redshift':float(z),'column':name,'step':float(step)})
            passed = any(raw_steps[j]['passed'] and raw_steps[j+1]['passed'] for j in (0,1))
            rows.append({'redshift':float(z),'column':name,'steps':raw_steps,
                         'two_successive_pass':bool(passed)})
    failures = [row for row in rows if not row['two_successive_pass']]
    return {'status':'failed' if failures else 'passed', 'rows':rows, 'coverage':coverage,
            'failures':failures, 'raw_column_norms':{
                name:{'ad':float(np.linalg.norm(ad[:,ic])),
                      'fd':[float(np.linalg.norm(fd[:,ic,js])) for js in range(3)]}
                for ic,name in enumerate(space.names)},
            'physical_norm':float(np.linalg.norm(fiducial)),
            'relative_tolerance':RELATIVE_TOLERANCE,'zero_tolerance':ZERO_TOLERANCE,
            'near_zero_switch':NEAR_ZERO_SWITCH}


def measure_growth_jacobian(space: CosmologySpec, reference: dict,
                            redshifts: np.ndarray) -> dict:
    """Measure rate*sigma8 AD and three independent central FDs at all eight z.

    ``ad`` is (8,ncosmo), ``fd`` is (8,ncosmo,3), ``signed_values`` is
    (8,ncosmo,3,2), ordered negative then positive. Domain theta points are
    ordered center then coordinate/step/sign. ``rows`` and ``coverage`` enumerate
    all 8*ncosmo*3 checks and retain raw scales; two successive steps must pass
    independently for each redshift/coordinate. The frozen reference is copied.
    """
    source = reference.params if hasattr(reference,'params') else reference
    source = dict(source)
    z = _finite(redshifts,(8,),'growth redshifts')
    if np.any(np.diff(z) <= 0) or np.any(z < 0) or np.any(z > 5):
        raise ValueError('growth redshifts must be ordered within [0, 5]')
    theta = _reference_theta(space,source)
    if source['h'] != space.h_ref:
        raise ValueError('growth h_ref differs from frozen reference')
    domain = validate_coordinate_neighborhood(theta,source,space)
    # Factories are captured only after the whole declared mapped neighborhood passes.
    rate, sigma8 = make_growth_rate_func(), make_sigma8_func()
    def growth(t):
        mapped = cosmology_from_coordinates(t,source,space)
        return jnp.stack([rate(zi,mapped)*sigma8(zi,mapped) for zi in z])
    point = jnp.asarray(theta,dtype=jnp.float64)
    fid = _finite(growth(point),(8,),'growth fiducial')
    ad = _finite(jax.jacfwd(growth)(point),(8,len(space.names)),'growth AD')
    values = np.empty((8,len(space.names),3,2))
    signed_theta = []
    for ic,name in enumerate(space.names):
        for js,step in enumerate(space.fd_steps[name]):
            for isig,sign in enumerate((-1,1)):
                varied = theta.copy()
                varied[ic] += sign*step
                validate_physical_point(varied,source,space,float(z[0]))
                values[:,ic,js,isig] = _finite(growth(jnp.asarray(varied)),(8,),'signed growth')
                signed_theta.append({'column':name,'step':float(step),'sign':sign,
                                     'theta':varied.tolist()})
    steps = np.array([space.fd_steps[name] for name in space.names])
    fd = (values[:,:,:,1]-values[:,:,:,0])/(2*steps[None])
    return {'cosmology_columns':list(space.names),'reference_cosmology':source,
        'theta':theta.tolist(),'redshifts':z.tolist(),'units':_growth_units(space),
        'fiducial':fid.tolist(),'ad':ad.tolist(),'fd':fd.tolist(),
        'signed_values':values.tolist(),'signed_theta':signed_theta,'domain':domain,
        **_growth_diagnostics(ad,fd,fid,z,space)}


def _validate_growth(report,space,reference,redshifts):
    if not isinstance(report,dict):
        raise ValueError('independent growth FD report is required')
    theta = _reference_theta(space,reference)
    if (report.get('cosmology_columns') != list(space.names)
            or report.get('redshifts') != redshifts.tolist()
            or _canonical(report.get('reference_cosmology')) != _canonical(reference)
            or report.get('theta') != theta.tolist()
            or report.get('units') != _growth_units(space)):
        raise ValueError('growth reference/column/redshift/unit coverage mismatch')
    want_domain = validate_coordinate_neighborhood(theta,reference,space)
    if _canonical(report.get('domain')) != _canonical(want_domain):
        raise ValueError('growth mapped domain evidence mismatch')
    want_signed = []
    for ic,name in enumerate(space.names):
        for step in space.fd_steps[name]:
            for sign in (-1,1):
                varied = theta.copy(); varied[ic] += sign*step
                want_signed.append({'column':name,'step':float(step),'sign':sign,'theta':varied.tolist()})
    if report.get('signed_theta') != want_signed:
        raise ValueError('growth signed theta coverage mismatch')
    n = len(space.names)
    fid = _finite(report.get('fiducial'),(8,),'growth fiducial')
    ad = _finite(report.get('ad'),(8,n),'growth AD')
    values = _finite(report.get('signed_values'),(8,n,3,2),'growth signed values')
    fd = _finite(report.get('fd'),(8,n,3),'growth FD')
    steps = np.array([space.fd_steps[name] for name in space.names])
    calculated_fd = (values[:,:,:,1]-values[:,:,:,0])/(2*steps[None])
    if not np.array_equal(fd,calculated_fd):
        raise ValueError('growth FD differs from raw signed predictions')
    calculated = _growth_diagnostics(ad,fd,fid,redshifts,space)
    if any(_canonical(report.get(key)) != _canonical(value) for key,value in calculated.items()):
        raise ValueError('growth raw diagnostics or coverage mismatch')
    if calculated['status'] != 'passed':
        raise ValueError('independent growth AD/FD failed')
    return calculated


def validate_extended_response(report: dict, likelihood: RevisionLikelihood, *, intrinsic_root=None) -> dict:
    """Recompute raw six-column AP and separate growth gates against live cells.

    Independent expectations come from live retained pairs/shells, canonical
    spec names/steps, and eight nominal-bin midpoints. Label omissions, swapped
    axes, baseline-only reports and missing growth FD fail closed. Results expose
    exact spectrum coverage and raw growth diagnostics for the later exporter.
    """
    space = _likelihood_space(likelihood)
    intrinsic = {}
    if likelihood.config.ap_interpolation_nk >= 16384:
        from .revision_validation_ap_intrinsic import validate_intrinsic_evidence
        intrinsic = {'intrinsic_geometry':validate_intrinsic_evidence(report,likelihood,root=intrinsic_root)}
    expected = extended_expected_schedule(likelihood)
    if report.get('cosmology_columns') != list(space.names):
        raise ValueError('extended cosmology column mismatch')
    points, saved = report.get('points'), report.get('cells')
    cells = [c for c in likelihood.cells if c.metadata['retained_pairs']]
    if (not isinstance(points,list) or not isinstance(saved,list)
            or [r.get('cell_id') for r in saved] != [c.metadata['cell_id'] for c in cells]):
        raise ValueError('extended retained cell coverage mismatch')
    expected_keys = {(row['cell_id'],tuple(row['pair']),row['k']) for row in expected}
    actual_keys = [(r.get('cell_id'),tuple(r.get('pair',())),r.get('k')) for r in points]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        raise ValueError('extended spectrum point coverage mismatch')
    theta = np.asarray(likelihood.fiducial)[[likelihood.names.index(n) for n in space.names]]
    if not np.array_equal(theta,_reference_theta(space,likelihood.reference_cosmology)):
        raise ValueError('extended fiducial differs from frozen growth reference')
    diagnostics = []
    for saved_cell,cell in zip(saved,cells):
        rows = [r for r in points if r['cell_id'] == cell.metadata['cell_id']]
        z = float(cell.metadata['model_redshift'])
        validate_coordinate_neighborhood(theta,{**likelihood.reference_cosmology,'z':z},space)
        keys = [(cell.metadata['cell_id'],tuple(pair),float(k))
                for pair in cell.metadata['retained_pairs'] for k in likelihood.grid.centers]
        validate_ap_point_rows(rows,keys,expected_mu=np.asarray(
            angular_grid_declaration(likelihood.config.n_mu)['mean_projection']['mu']),ap_nk=likelihood.config.ap_interpolation_nk,
            refinement_nk=likelihood.config.ap_refinement_nk,require_grid_coverage=True,
            space=space,reference=likelihood.reference_cosmology,theta=theta,redshift=z)
        diagnostics.append(_validate_cell_record(saved_cell,likelihood,cell,rows,space=space))
    redshifts = np.array([(lo+hi)/2 for lo,hi in likelihood.config.nominal_bins])
    growth = _validate_growth(report.get('growth'),space,likelihood.reference_cosmology,redshifts)
    measured = []
    for row in points:
        coordinates = row['domain']['parameter_coordinates']
        for ell in (0,2,4):
            for ic,name in enumerate(row['cosmology_columns']):
                for js in range(3):
                    recorded = coordinates[1+ic*6+js*2]
                    measured.append({'cell_id':row['cell_id'],'pair':row['pair'],
                        'k':row['k'],'ell':ell,'column':name,'step':recorded['step']})
    return {**intrinsic,'status':'passed','cosmology_columns':list(space.names),
            'expected_coverage':expected,'measured_coverage':measured,
            'cells':diagnostics,'growth':growth}


def validate_study_inventory(record):
    """Two physical likelihoods; fourteen projections; eight correlated outputs."""
    expected={'physical_likelihood_count':2,'matrix_treatment_count':14,
              'growth_output_count':8,'growth_outputs_independent':False}
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('study must contain two physical likelihoods, fourteen matrix treatments and eight correlated growth outputs')
    return expected


def validate_conditional_bridge(extended,conditional,baseline,baseline_result):
    """Compare to a separately built native three-basis likelihood or oracle control.

    Caller authenticates provenance and establishes the independent build. This
    numerical comparison never derives its baseline from the six-dimensional F.
    """
    from dataclasses import asdict
    from .extended_artifacts import require_close, canonical
    from .fisher_precision import common_fisher_solver
    common_fisher_solver(conditional,baseline_result)
    if baseline.config.cosmology_basis!='baseline3' or extended.config.cosmology_basis!='extended6':
        raise ValueError('bridge requires independent baseline3 and extended6 builds')
    a,b=asdict(extended.config),asdict(baseline.config)
    a.pop('cosmology_basis'); b.pop('cosmology_basis')
    if a!=b or conditional.treatment!='conditional3': raise ValueError('bridge configuration mismatch')
    if (list(conditional.active_parameter_names)!=baseline.names
            or canonical(extended.reference_cosmology)!=canonical(baseline.reference_cosmology)):
        raise ValueError('bridge parameter labels/reference mismatch')
    common=[extended.names.index(n) for n in baseline.names]
    require_close(extended.fiducial[common],baseline.fiducial,'bridge fiducial',rtol=1e-8)
    # All physical/nuisance metadata remain exact; only added cosmology columns disappear.
    if canonical([extended.parameters[i] for i in common])!=canonical(baseline.parameters):
        raise ValueError('bridge prior/parameter metadata mismatch')
    for field in ('edges','centers','widths'):
        if not np.array_equal(getattr(extended.grid,field),getattr(baseline.grid,field)):
            raise ValueError('bridge observed shell grid differs')
    if len(extended.cells)!=len(baseline.cells): raise ValueError('bridge cell inventory differs')
    for new,old in zip(extended.cells,baseline.cells):
        meta=dict(new.metadata)
        if 'local_names' in meta: meta['local_names']=[n for n in meta['local_names'] if n in baseline.names]
        if canonical(meta)!=canonical(old.metadata): raise ValueError('bridge physical cell metadata differs')
        local=[new.names.index(n) for n in old.names]
        if [n for n in new.names if n in baseline.names]!=old.names:
            raise ValueError('bridge local labels differ')
        for field in ('fiducial_mean','covariance'):
            require_close(getattr(new,field),getattr(old,field),'bridge '+field,rtol=1e-8)
        require_close(new.derivatives[:,:,local],old.derivatives,'bridge derivative',rtol=1e-8)
        require_close(new.F_like[np.ix_(local,local)],old.F_like,'bridge local Fisher',rtol=1e-8)
    require_close(np.asarray(extended.bin_F_like)[:,common][:,:,common],baseline.bin_F_like,
                  'bridge bin Fisher',rtol=1e-8)
    for field in ('F_like','F_prior_cosmo','F_prior_nuisance','F_total','covariance',
                  'growth_z','growth_fiducial','growth_cov_marg','growth_cov_nuisance_fixed'):
        require_close(getattr(conditional,field),getattr(baseline_result,field),'bridge '+field,rtol=1e-8)
    for field in ('cov_cosmo_marg','cov_cosmo_nuisance_fixed'):
        require_close(getattr(conditional,field)[:3,:3],getattr(baseline_result,field),'bridge '+field,rtol=1e-8)
    require_close(conditional.growth_jacobian[:,:3],baseline_result.growth_jacobian,'bridge growth Jacobian',rtol=1e-8)
    return {'status':'passed','rtol':1e-8,'atol':1e-12,'case':extended.config.case,
            'comparison':'conditional3 versus independently built baseline3'}


def validate_native_baseline_identity(current,current_result,historical,historical_result):
    """Strict native3 identity at one AP grid; source authentication is external."""
    from dataclasses import asdict
    from .extended_artifacts import (PRECISION_FIELDS,GROWTH_FIELDS,canonical,
                                     require_close,validate_likelihood)
    from .fisher_precision import common_fisher_solver
    common_fisher_solver(current_result,historical_result)
    if (current.config.cosmology_basis!='baseline3' or historical.config.cosmology_basis!='baseline3'
            or canonical(asdict(current.config))!=canonical(asdict(historical.config))):
        raise ValueError('native3 identity configuration differs')
    for field in ('names','parameters','reference_cosmology','model'):
        if canonical(getattr(current,field))!=canonical(getattr(historical,field)):
            raise ValueError('native3 identity physical metadata differs: '+field)
    if len(current.cells)!=len(historical.cells): raise ValueError('native3 identity cell inventory differs')
    validate_likelihood(current);validate_likelihood(historical)
    for a,b in zip(current.cells,historical.cells):
        if a.names!=b.names or canonical(a.metadata)!=canonical(b.metadata):
            raise ValueError('native3 identity support/local labels differ')
        for field in ('fiducial_mean','covariance','derivatives','F_like'):
            require_close(getattr(a,field),getattr(b,field),'native3 identity '+field,rtol=1e-8)
    for field in ('F_like','bin_F_like','fiducial'):
        require_close(getattr(current,field),getattr(historical,field),'native3 identity '+field,rtol=1e-8)
    for field in ('edges','centers','widths'):
        require_close(getattr(current.grid,field),getattr(historical.grid,field),
                      'native3 identity shells',rtol=0.,atol=0.)
    for field in (*PRECISION_FIELDS,*GROWTH_FIELDS):
        require_close(getattr(current_result,field),getattr(historical_result,field),
                      'native3 identity '+field,rtol=1e-8)
    return {'status':'passed','comparison':'native3 versus historical native3 at the same AP resolution',
            'case':current.config.case,'ap_interpolation_nk':current.config.ap_interpolation_nk,
            'rtol':1e-8,'atol':1e-12}


def _radius_points(space,reference):
    theta=_reference_theta(space,reference)
    points=[('fiducial',0,0.,theta)]
    for name in ('h','omega_b','n_s'):
        for sign in (-1,1):
            value=theta.copy(); step=space.fd_steps[name][0]
            value[space.names.index(name)]+=sign*step
            points.append((name,sign,step,value))
    return points


def measure_sigma8_radius(space,reference,redshifts):
    """Save network and independent top-hat inputs at seven declared points."""
    from .reference_units import make_reference_plin_func
    power=make_reference_plin_func(space.h_ref); sigma8=make_sigma8_func()
    rows=[]
    for name,sign,step,theta in _radius_points(space,reference):
        mapped=validate_physical_point(theta,reference,space,float(redshifts[0]))
        for z in redshifts:
            data=power(float(z),mapped)
            rows.append({'coordinate':name,'sign':sign,'step':step,'theta':theta.tolist(),
                'redshift':float(z),'radius_ref':8*space.h_ref/mapped['h'],
                'k_ref':np.asarray(data['k']).tolist(),'power_ref':np.asarray(data['pk']).tolist(),
                'network_sigma8':float(sigma8(float(z),mapped))})
    return {'rows':rows,'h_ref':space.h_ref,'unit_convention':'fixed_h_ref',
            'sigma8_radius':'8/h_trial Mpc','tolerance':.002}


def validate_sigma8_radius(report,space,reference,redshifts):
    """Recompute the integral; fixed eight is invalid when trial h changes."""
    if (report.get('h_ref')!=space.h_ref or report.get('unit_convention')!='fixed_h_ref'
            or report.get('sigma8_radius')!='8/h_trial Mpc' or report.get('tolerance')!=.002):
        raise ValueError('sigma8 radius unit/tolerance mismatch')
    expected=[(name,sign,step,theta,float(z)) for name,sign,step,theta in _radius_points(space,reference)
              for z in redshifts]
    if len(report['rows'])!=len(expected): raise ValueError('sigma8 radius point coverage differs')
    errors=[]
    for row,(name,sign,step,theta,z) in zip(report['rows'],expected):
        mapped=validate_physical_point(theta,reference,space,z); radius=8*space.h_ref/mapped['h']
        if (row['coordinate']!=name or row['sign']!=sign or row['step']!=step or row['theta']!=theta.tolist()
                or row['redshift']!=z or row['radius_ref']!=radius): raise ValueError('sigma8 radius/signed input mismatch')
        k=np.asarray(row['k_ref']);power=np.asarray(row['power_ref']); network=row['network_sigma8']
        if (k.ndim!=1 or len(k)<3 or power.shape!=k.shape or not np.all(np.isfinite(k))
                or not np.all(np.isfinite(power)) or np.any(k<=0) or np.any(np.diff(k)<=0)
                or np.any(power<=0) or not np.isfinite(network) or network<=0):
            raise ValueError('sigma8 raw power/grid malformed')
        x=k*radius; window=3*(np.sin(x)-x*np.cos(x))/x**3
        integrated=np.sqrt(np.trapezoid(k**3*power*window**2,x=np.log(k))/(2*np.pi**2))
        error=abs(integrated/network-1)
        if not np.isfinite(error) or error>=.002: raise ValueError('sigma8 radius network/integral disagreement')
        errors.append(float(error))
    return {'status':'passed','relative_errors':errors,'tolerance':.002}


def validate_treatment_cohort(results):
    """Compatible prior/conditioning releases cannot improve marginal errors."""
    from .fisher_precision import common_fisher_solver
    common_fisher_solver(*results.values())
    edges=[('conditional3',name) for name in ('release_h','release_omega_b','release_n_s','marginal6')]
    edges += [(name,'marginal6') for name in ('release_h','release_omega_b','release_n_s')]
    edges += [('marginal6','marginal6_weak10'),('marginal6_weak10','marginal6_no_extra')]
    comparisons=[]
    for fixed,released in edges:
        if fixed not in results or released not in results: continue
        a,b=results[fixed],results[released]
        for field in ('cov_cosmo_marg','cov_cosmo_nuisance_fixed','growth_cov_marg','growth_cov_nuisance_fixed'):
            low=np.sqrt(np.diag(getattr(a,field))); high=np.sqrt(np.diag(getattr(b,field)))
            active=low>0
            if (not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
                    or np.any(high[active]<low[active]*(1-1e-10))):
                raise ValueError(f'release monotonicity failed: {fixed} to {released}, {field}')
        comparisons.append([fixed,released])
    return {'status':'passed','comparisons':comparisons,'relative_tolerance':1e-10}
