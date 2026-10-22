"""Six spectrum columns and independent eight-redshift growth evidence."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec, cosmology_from_coordinates
from pfsfog.revision_forecast import RevisionConfig
from pfsfog.revision_theory import NUISANCE_NAMES

NAMES = ('ln10_10_As', 'Mnu', 'Omegam', 'h', 'omega_b', 'n_s')
STEPS = ((.001, .0005, .00025), (.005, .0025, .00125),
         (.001, .0005, .00025), (.001, .0005, .00025),
         (.0001, .00005, .000025), (.001, .0005, .00025))
Z = np.array([(.4+.6)/2, (.6+.8)/2, (.8+1.)/2, (1.+1.2)/2,
              (1.2+1.4)/2, (1.4+1.6)/2, (1.6+2.)/2, (2.+2.1)/2])


def _theta():
    return np.array([FIDUCIAL['ln10_10_As'], FIDUCIAL['mnu'], FIDUCIAL['Omega_m'],
                     FIDUCIAL['h'], FIDUCIAL['omega_b'], FIDUCIAL['n_s']])


def _fixture():
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    config = RevisionConfig('B', cosmology_basis='extended6')
    pair = ('DESI-ELG', 'DESI-ELG')
    # Hand-declared six distinct slopes; h must not disappear as a zero response.
    ad = np.array([[1., 2., 3., 4., 5., 6.], [2., 4., 6., 8., 10., 12.],
                   [3., 6., 9., 12., 15., 18.]])
    cell = SimpleNamespace(names=list(NAMES),
        metadata={'cell_id':'cell', 'retained_pairs':[pair], 'model_redshift':1.1},
        fiducial_mean=np.array([[10., 20., 30.]]), derivatives=ad[None],
        covariance=np.eye(3)[None])
    # Full likelihood order intentionally differs from canonical cosmology order.
    order = (3, 0, 5, 1, 4, 2)
    likelihood = SimpleNamespace(config=config, cells=[cell],
        names=[NAMES[i] for i in order], fiducial=_theta()[list(order)],
        cosmo_names=NAMES, reference_cosmology=dict(FIDUCIAL),
        grid=SimpleNamespace(centers=np.array([.015])))
    return space, likelihood, cell, ad


def _domain(mu, space):
    points = [('fiducial', 0, 0.)] + [(name, sign, step) for name, steps in zip(NAMES, STEPS)
              for step in steps for sign in (-1, 1)]
    coordinates = []
    for name, sign, step in points:
        theta = _theta().copy()
        if name != 'fiducial':
            theta[NAMES.index(name)] += sign*step
        mapped = {key:float(value) for key, value in
                  cosmology_from_coordinates(theta, FIDUCIAL, space).items()}
        coordinates.append({'parameter':name, 'sign':sign, 'step':step,
            'theta':theta.tolist(), 'mapped_cosmology':mapped,
            'alpha_perp':1., 'alpha_para':1., 'response_shape':[1,len(mu)],
            'response_min':1., 'response_max':1., 'response_values':[1.]*len(mu)})
    return {'parameter_coordinates':coordinates, 'k_true_min':.015, 'k_true_max':.015,
            'mu_true_min':float(np.min(np.abs(mu))), 'mu_true_max':float(np.max(np.abs(mu))),
            'k_margin_lower':.013, 'k_margin_upper':.485}


def _row(space, ad):
    from pfsfog.revision_validation_ap import angular_grid_declaration
    grids = angular_grid_declaration(32)
    mu = grids['mean_projection']['mu']
    refinements = [{'nk':size, 'mean_error':0., 'derivative_errors':[0.]*6,
                    'mean':[10.,20.,30.], 'ad':ad.tolist()} for size in (4096,8192)]
    return {'cell_id':'cell', 'pair':['DESI-ELG','DESI-ELG'], 'k':.015,
        'cosmology_columns':list(NAMES), 'mu':mu, 'shape':[1,len(mu)],
        'domain':_domain(mu,space), 'observed_domains':{
            role:_domain(grids[role]['mu'],space) for role in ('covariance','refined_projection')},
        'fiducial_multipoles':[10.,20.,30.], 'direct_multipoles':[10.,20.,30.],
        'pair_covariance':np.eye(3).tolist(), 'whitened_physical_scale':float(np.sqrt(1400)),
        'total_ad':ad.tolist(), 'intrinsic_ad':ad.tolist(), 'geometry_ad':np.zeros((3,6)).tolist(),
        'fd_derivatives':np.repeat(ad.T[:,None,:],3,axis=1).tolist(),
        'alpha_derivatives':np.zeros((2,6)).tolist(), 'continuity_deltas':np.zeros((2,3)).tolist(),
        'identity_error':0., 'continuity_error':0., 'pure_ap_isotropic_error':0.,
        'pure_ap_anisotropic_error':0., 'ad_fd_errors':np.zeros((6,3)).tolist(),
        'chain_errors':[0.]*6, 'alpha_fd_errors':np.zeros((6,3)).tolist(),
        'interpolation':refinements, 'angular':{'n_mu':64,'mean_error':0.,
            'derivative_errors':[0.]*6,'mean':[10.,20.,30.],'ad':ad.tolist()}}


def _growth(monkeypatch, space):
    from pfsfog import extended_validation as ev
    # Only external growth/network leaves are controlled; map, AD, FD and gate are real.
    monkeypatch.setattr(ev, 'make_growth_rate_func', lambda: lambda z,p: 1/(1+z))
    monkeypatch.setattr(ev, 'make_sigma8_func', lambda: lambda z,p:
        jnp.exp(.1*p['ln10_10_As']+.2*p['mnu']+.3*p['omega_cdm']+
                .4*p['h']+.5*p['omega_b']+.6*p['n_s']))
    return ev.measure_growth_jacobian(space, FIDUCIAL, Z)


def _report(monkeypatch):
    from pfsfog import extended_validation as ev
    from pfsfog.revision_validation_ap import ap_cell_record
    space, likelihood, cell, ad = _fixture()
    row = _row(space,ad)
    report = {'cosmology_columns':list(NAMES), 'points':[row],
        'cells':[ap_cell_record(likelihood,cell,[row],space=space)],
        'growth':_growth(monkeypatch,space)}
    return ev, space, likelihood, report


def test_extended_schedule_covers_six_columns_three_multipoles_three_steps():
    from pfsfog.extended_validation import extended_expected_schedule
    _, likelihood, _, _ = _fixture()
    got = extended_expected_schedule(likelihood)
    expected = [{'cell_id':'cell','pair':['DESI-ELG','DESI-ELG'],'k':.015,
                 'ell':ell,'column':name,'step':step}
                for ell in (0,2,4) for name, steps in zip(NAMES,STEPS) for step in steps]
    assert got == expected


@pytest.mark.parametrize('name', ['h','omega_b','n_s'])
def test_missing_new_column_is_rejected(monkeypatch,name):
    ev, _, likelihood, report = _report(monkeypatch)
    assert ev.validate_extended_response(report,likelihood)['status'] == 'passed'
    report['points'][0]['cosmology_columns'].remove(name)
    with pytest.raises(ValueError,match='column'):
        ev.validate_extended_response(report,likelihood)


def test_swapped_column_and_baseline_report_cannot_attach(monkeypatch):
    ev, space, likelihood, report = _report(monkeypatch)
    report['cells'][0]['cosmology_columns'][3:5] = ['omega_b','h']
    with pytest.raises(ValueError,match='ordering'):
        ev.validate_extended_response(report,likelihood)
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    base = _row(space,np.zeros((3,6)))
    base['cosmology_columns'] = list(NAMES[:3])
    with pytest.raises(ValueError,match='column'):
        validate_ap_point_rows([base],[('cell',('DESI-ELG','DESI-ELG'),.015)],space=space)


def test_zeroed_h_derivative_fails_independent_fd(monkeypatch):
    ev, _, likelihood, report = _report(monkeypatch)
    row = report['points'][0]
    row['total_ad'] = np.asarray(row['total_ad']).copy()
    row['total_ad'][:,3] = 0.
    with pytest.raises(ValueError):
        ev.validate_extended_response(report,likelihood)


def test_independent_growth_has_eight_rows_and_all_three_steps(monkeypatch):
    from pfsfog import extended_validation as ev
    space, _, _, _ = _fixture()
    result = _growth(monkeypatch,space)
    assert result['status'] == 'passed'
    assert result['cosmology_columns'] == list(NAMES)
    assert np.shape(result['ad']) == (8,6)
    assert np.shape(result['fd']) == (8,6,3)
    assert len(result['coverage']) == 8*6*3
    np.testing.assert_array_equal(result['redshifts'],Z)
    prediction = np.asarray(result['fiducial'])
    # Exact derivatives of the independent analytic physical leaf above.
    slopes = [.1, .2-.3/93.14, .3*FIDUCIAL['h']**2,
              .4+.6*FIDUCIAL['Omega_m']*FIDUCIAL['h'], .2, .6]
    np.testing.assert_allclose(result['ad'],prediction[:,None]*np.array(slopes),rtol=1e-12)
    assert len(result['rows']) == 48
    assert all(len(row['steps']) == 3 and row['two_successive_pass'] for row in result['rows'])


@pytest.mark.parametrize('mutation', ['missing','row','fd','labels','step','raw','ad'])
def test_growth_fd_cannot_be_inferred_from_spectrum_success(monkeypatch,mutation):
    ev, _, likelihood, report = _report(monkeypatch)
    growth = report['growth']
    if mutation == 'missing': del report['growth']
    elif mutation == 'row': growth['redshifts'] = growth['redshifts'][:-1]
    elif mutation == 'fd': del growth['fd']
    elif mutation == 'labels': growth['cosmology_columns'][3:5] = ['omega_b','h']
    elif mutation == 'step': growth['coverage'].pop()
    elif mutation == 'raw': growth['signed_values'][0][3][0][0] += 1.
    elif mutation == 'ad': growth['ad'][0][3] = 0.
    with pytest.raises(ValueError): ev.validate_extended_response(report,likelihood)


@pytest.mark.parametrize('role', ['mean','covariance','refined_projection'])
def test_all_domain_grids_bind_signed_theta_and_mapped_density(monkeypatch,role):
    ev, _, likelihood, report = _report(monkeypatch)
    row = report['points'][0]
    domain = row['domain'] if role == 'mean' else row['observed_domains'][role]
    domain['parameter_coordinates'][20]['mapped_cosmology']['omega_cdm'] = .16
    with pytest.raises(ValueError,match='mapped'):
        ev.validate_extended_response(report,likelihood)


def test_invalid_growth_neighborhood_rejected_before_external_prediction(monkeypatch):
    from pfsfog import extended_validation as ev
    space, _, _, _ = _fixture()
    def forbidden(): raise AssertionError('prediction factory reached before domain check')
    monkeypatch.setattr(ev,'make_sigma8_func',forbidden)
    with pytest.raises(ValueError,match='omega_cdm'):
        ev.measure_growth_jacobian(space,{**FIDUCIAL,'Omega_m':.45},Z)


def test_baseline_schedule_keeps_nineteen_points_and_exact_steps():
    from pfsfog.revision_validation_ap import ap_parameter_schedule
    assert ap_parameter_schedule() == [('fiducial',0,0.)] + [
        (name,sign,step) for name,steps in zip(NAMES[:3],STEPS[:3])
        for step in steps for sign in (-1,1)]


def test_explicit_physical_space_routes_six_theta_and_small_baryon_steps(monkeypatch):
    from pfsfog import revision_validation_physical as physical
    space, likelihood, cell, _ = _fixture()
    nuisance = [f'bin00:DESI-ELG:{name}' for name in NUISANCE_NAMES]
    cell.names += nuisance
    cell.metadata.update({'nominal_bin_index':0,'model_redshift':1.,
        'active_populations':('DESI-ELG',),'local_names':cell.names,
        'nbar':{'DESI-ELG':1.}})
    likelihood.names += nuisance
    likelihood.fiducial = np.r_[likelihood.fiducial,np.ones(len(nuisance))]
    seen = []
    class Response:
        def multipoles(self,k,theta,qa,qb,qc,*,n_mu):
            seen.append(np.asarray(theta))
            return np.full((len(k),3),np.dot(theta,[1,2,3,4,5,6]))
    def factory(*args,**kwargs):
        assert kwargs['space'] is space
        return Response()
    monkeypatch.setattr(physical,'make_pair_response',factory)
    monkeypatch.setattr(physical,'_checked_pairs',lambda *args:[('DESI-ELG','DESI-ELG')])
    monkeypatch.setattr(physical,'_check_shells',lambda *args:None)
    means = physical.physical_cell_means(likelihood,cell,object(),space=space)
    coords = dict(zip(likelihood.names,likelihood.fiducial))
    means[('DESI-ELG','DESI-ELG')](coords)
    np.testing.assert_array_equal(seen[-1],_theta())
    assert physical._steps('omega_b',FIDUCIAL['omega_b'],space=space) == (.0001,.00005,.000025)
    coords['h'] = .8
    with pytest.raises(ValueError,match='omega_cdm'):
        means[('DESI-ELG','DESI-ELG')](coords)
    assert len(seen) == 1


def test_full_growth_covariance_and_matched_treatment_errors():
    from pfsfog.revision_validation import compare_marginal_errors
    space, _, _, _ = _fixture()
    jac = np.array([[1.,2.,3.,4.,5.,6.]]*8)
    cov = np.eye(6)
    cov[0,3] = cov[3,0] = .2
    record = {'cosmology_columns':list(NAMES),'treatment':'marginal6',
        'free_cosmology':list(NAMES),'growth_z':Z,'growth_jacobian':jac,
        'cov_cosmo_marg':cov}
    new = {**record,'cov_cosmo_marg':cov/4}
    result = compare_marginal_errors(record,new,space=space)
    np.testing.assert_allclose(result['native_ratio'],[.5]*6)
    np.testing.assert_allclose(result['growth_ratio'],[.5]*8)
    assert result['old_growth_sigma'][0] == pytest.approx(np.sqrt(92.6))
    with pytest.raises(ValueError,match='treatment'):
        compare_marginal_errors(record,{**new,'treatment':'conditional3'},space=space)
    conditional = cov.copy(); conditional[3:,:] = 0.; conditional[:,3:] = 0.
    record = {**record,'treatment':'conditional3','free_cosmology':list(NAMES[:3]),
              'cov_cosmo_marg':conditional}
    assert len(compare_marginal_errors(record,record,space=space)['native_ratio']) == 3


@pytest.mark.parametrize('cross,new_profile',[(False,False),(True,False),(False,True)],
                         ids=['auto','cross','intrinsic-auto'])
def test_live_extended_ap_measurement_uses_six_packed_tangents_and_signed_domains(monkeypatch,cross,new_profile):
    import jax
    from pfsfog import revision_theory as theory
    from pfsfog import revision_validation_ap as ap
    space = make_cosmology_spec(FIDUCIAL,extended=True,ap_nk=16384 if new_profile else None)
    # The released spectrum/network leaves are replaced by analytic physical
    # leaves. The real PairResponse, physical backgrounds and AP interpolation run.
    monkeypatch.setattr(theory,'make_reference_plin_func',lambda h_ref: lambda z,p:
        jnp.exp(.1*p['ln10_10_As']+.2*p['mnu']+.3*p['Omega_m']+
                .4*p['h']+5*p['omega_b']+.6*p['n_s']))
    monkeypatch.setattr(theory,'make_sigma8_func',lambda: lambda z,p:
        .8*jnp.exp(.02*(p['ln10_10_As']-FIDUCIAL['ln10_10_As'])+
                    .01*(p['h']-FIDUCIAL['h'])+.2*(p['omega_b']-FIDUCIAL['omega_b'])))
    monkeypatch.setattr(theory,'make_growth_rate_func',lambda: lambda z,p:
        .6+.1*p['Omega_m']+.01*p['mnu'])
    class Backend:
        def get_pkmu_pair(self,k,mu,pk,params,add_stochasticity=False):
            coefficient = sum(params['bias'].values())+sum(params['ctr'].values())
            if 'bias2' in params:
                coefficient += 2*sum(params['bias2'].values())+3*sum(params['ctr2'].values())
            if add_stochasticity: coefficient += sum(params['stoch'].values())
            return pk*coefficient*jnp.asarray(k)[:,None]*(1+.6*jnp.abs(jnp.asarray(mu))[None,:])
    backend = Backend()
    pair = ('DESI-ELG','DESI-LRG' if cross else 'DESI-ELG')
    active = pair if cross else pair[:1]
    names = list(NAMES)+[f'bin00:{a}:{name}' for a in active for name in NUISANCE_NAMES]
    if cross: names += [f'bin00:{pair[0]}|{pair[1]}:{name}' for name in ('Pshot_cross','a2_cross')]
    qa = np.arange(1,12)/20
    packed = jnp.asarray(np.r_[_theta(),qa,qa*1.2,[.07,.3]] if cross else np.r_[_theta(),qa])
    config = RevisionConfig('B',cosmology_basis='extended6')
    if new_profile:
        config = replace(config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    reference = dict(FIDUCIAL)
    response = theory.make_pair_response(backend,1.,reference,1.,2. if cross else None,
                                        config.k_nl,ap_nk=config.ap_interpolation_nk,space=space)
    k = jnp.asarray([.015])
    def assembled_mean(p,ap=True,n_mu=32):
        return response.multipoles(k,p[:6],p[6:17],p[17:28] if cross else None,
                                   p[-2:] if cross else None,ap=ap,n_mu=n_mu)
    mean = np.asarray(assembled_mean(packed))
    derivative = np.asarray(jax.jacfwd(assembled_mean)(packed))
    cell = SimpleNamespace(names=names,metadata={'cell_id':'live','retained_pairs':[pair],
        'nominal_bin_index':0,'model_redshift':1.,'nbar':{pair[0]:1.,**({pair[1]:2.} if cross else {})}},
        fiducial_mean=mean,derivatives=derivative,covariance=np.eye(3)[None])
    likelihood = SimpleNamespace(config=config,names=names,fiducial=np.asarray(packed),
        cells=[cell],grid=SimpleNamespace(centers=np.asarray(k)),reference_cosmology=reference)
    monkeypatch.setattr(ap,'_checked_pairs',lambda *args:[pair])
    monkeypatch.setattr(ap,'_check_shells',lambda *args:None)
    bundles = []
    kwargs = {'intrinsic_sink':lambda record,arrays:bundles.append((record,arrays))} if new_profile else {}
    rows = ap.measure_ap_cell(likelihood,cell,backend,space=space,**kwargs)
    np.testing.assert_array_equal(rows[0]['total_ad'],derivative[0,:,:6])
    assert np.shape(rows[0]['fd_derivatives']) == (6,3,3)
    assert np.shape(rows[0]['alpha_derivatives']) == (2,6)
    assert np.linalg.norm(np.asarray(rows[0]['total_ad'])[:,3]) > 0
    for i,steps in enumerate(space.fd_steps.values()):
        for j,step in enumerate(steps):
            expected = (np.asarray(assembled_mean(packed.at[i].add(step)))-
                        np.asarray(assembled_mean(packed.at[i].add(-step))))/(2*step)
            np.testing.assert_array_equal(rows[0]['fd_derivatives'][i][j],expected[0])
    assert ap.validate_ap_point_rows(rows,[('live',pair,.015)],space=space,reference=reference,
        ap_nk=config.ap_interpolation_nk,refinement_nk=config.ap_refinement_nk,
        theta=_theta(),redshift=1.,require_grid_coverage=True)['status'] == 'passed'
    record = ap.ap_cell_record(likelihood,cell,rows,space=space)
    assert record['diagnostics']['status'] == 'passed'
    for domain in [rows[0]['domain'],*rows[0]['observed_domains'].values()]:
        assert len(domain['parameter_coordinates']) == 37
    # Physical AP amplitude has exact zero AD/FD and passes absolute checks.
    np.testing.assert_array_equal(np.asarray(rows[0]['alpha_derivatives'])[:,0],[0.,0.])
    if new_profile:
        from pfsfog.revision_validation_ap_intrinsic import validate_intrinsic_geometry
        assert len(bundles) == 1
        saved, arrays = bundles[0]
        validate_intrinsic_geometry(saved,arrays,likelihood=likelihood,cell=cell,pair=pair)
        np.testing.assert_array_equal(arrays['total'],derivative)
        np.testing.assert_array_equal(rows[0]['geometry_ad_original'],arrays['geometry_original'][0,:,:6])
        np.testing.assert_array_equal(rows[0]['geometry_ad'],arrays['geometry_matched'][0,:,:6])
        assert len(rows[0]['chain_errors_original']) == 6


def test_new_profile_requires_intrinsic_evidence_before_response_acceptance(monkeypatch,tmp_path):
    ev, _, like, report = _report(monkeypatch)
    like.config = replace(like.config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    with pytest.raises(ValueError,match='intrinsic-table'):
        ev.validate_extended_response(report,like,intrinsic_root=tmp_path)


def test_extended_ap_zero_fd_uses_normalized_absolute_rule():
    from pfsfog import revision_validation_ap as ap
    space, likelihood, cell, ad = _fixture()
    # A truly zero n_s FD with tiny AD roundoff is accepted on physical scale.
    ad[:,5] = np.sqrt(1400)*5e-8
    cell.derivatives = ad[None].copy()
    row = _row(space,ad)
    row['fd_derivatives'][5] = [[0.,0.,0.]]*3
    row['ad_fd_errors'][5] = [np.sqrt(3)*5e-8]*3
    assert ap.validate_ap_point_rows([row],[('cell',('DESI-ELG','DESI-ELG'),.015)],
        space=space,reference=FIDUCIAL,theta=_theta(),redshift=1.1,
        require_grid_coverage=True)['status'] == 'passed'
    result = ap.ap_cell_record(likelihood,cell,[row],space=space)
    assert result['diagnostics']['status'] == 'passed'
    assert result['diagnostics']['blocks'][0]['ad_fd'][5][0]['branch'] == 'normalized absolute'


def test_growth_zero_columns_and_failed_steps_are_retained(monkeypatch):
    from pfsfog import extended_validation as ev
    space, _, _, _ = _fixture()
    monkeypatch.setattr(ev,'make_growth_rate_func',lambda: lambda z,p:1/(1+z))
    monkeypatch.setattr(ev,'make_sigma8_func',lambda: lambda z,p:jnp.exp(p['ln10_10_As']))
    zero = ev.measure_growth_jacobian(space,FIDUCIAL,Z)
    assert zero['status'] == 'passed'
    assert all(step['branch'] == 'normalized absolute' and step['limit'] == 1e-6
        for row in zero['rows'] if row['column'] == 'h' for step in row['steps'])
    original = ev.jax.jacfwd
    def broken_ad(function):
        evaluate = original(function)
        def derivative(theta):
            return evaluate(theta).at[:,0].set(0.)
        return derivative
    monkeypatch.setattr(ev.jax,'jacfwd',broken_ad)
    failed = ev.measure_growth_jacobian(space,FIDUCIAL,Z)
    assert failed['status'] == 'failed'
    assert len(failed['coverage']) == 144
    assert len(failed['failures']) == 8
    assert all(len(row['steps']) == 3 for row in failed['failures'])
    assert all(step['fd_norm'] > 0 and step['ad_norm'] == 0 and not step['passed']
        for row in failed['failures'] for step in row['steps'])


def _physical_fixture():
    space, likelihood, cell, _ = _fixture()
    likelihood.config = replace(likelihood.config,kmin=.01,kmax=.02,dk=.01)
    nuisance = [f'bin00:DESI-ELG:{name}' for name in NUISANCE_NAMES]
    cell.names += nuisance
    cell.metadata.update({'nominal_bin_index':0,'nominal_bin':(.4,.6),'radial_bin':(.4,.6),
        'region':'overlap','volume':1.,'model_redshift':.5,
        'candidate_populations':('DESI-ELG',),'active_populations':('DESI-ELG',),
        'theoretical_pairs':[('DESI-ELG','DESI-ELG')],
        'retained_observations':[(('DESI-ELG','DESI-ELG'),ell) for ell in (0,2,4)],
        'local_names':cell.names,'nbar':{'DESI-ELG':1.}})
    cell.validation = {'expected_shells':1,'processed_shells':1}
    likelihood.names += nuisance
    likelihood.fiducial = np.r_[likelihood.fiducial,np.ones(len(nuisance))]
    likelihood.grid.widths = np.array([.01])
    cell.derivatives = np.zeros((1,3,len(cell.names)))
    cell.derivatives[0,:,:6] = np.array([[1,2,3,4,5,6]]*3)
    support = {key:cell.metadata[key] for key in ('cell_id','nominal_bin_index','nominal_bin',
        'radial_bin','region','model_redshift','volume','nbar','candidate_populations',
        'active_populations','retained_pairs','local_names')}
    support.update({'absent_observations':[],'empty_observation':False})
    return space,likelihood,cell,[support]


def test_extended_physical_certificate_measures_named_columns_and_covariance_neighborhood(monkeypatch):
    from pfsfog import revision_validation_physical as physical
    space,likelihood,cell,support = _physical_fixture()
    class Response:
        def multipoles(self,k,theta,qa,qb,qc,*,n_mu):
            return np.full((len(k),3),100+np.dot(theta,[1,2,3,4,5,6]))
        def pkmu(self,k,mu,theta,qa,qb,qc):
            return np.full((len(k),len(mu)),100+np.dot(theta,[1,2,3,4,5,6]))
    def factory(*args,**kwargs):
        assert kwargs['space'] is space
        return Response()
    monkeypatch.setattr(physical,'make_pair_response',factory)
    manifest = physical.declare_fd_manifest(likelihood,['cell'],'run',expected_support=support,space=space)
    assert {row[-1] for row in manifest if row[-2] == 'omega_b'} == {.0001,.00005,.000025}
    measured = physical.assembled_fd_certificate(likelihood,object(),['cell'],manifest,
                                                 expected_support=support,space=space)
    assert measured['status'] == 'passed'
    assert len(measured['column_results']) == 6+len(NUISANCE_NAMES)
    cell.derivatives[0,:,3] = 0.
    failed = physical.assembled_fd_certificate(likelihood,object(),['cell'],manifest,
                                               expected_support=support,space=space)
    assert failed['status'] == 'failed'
    assert {r['column'] for r in failed['failures']} == {'h'}
    neighborhood = physical.audit_cell_covariance_neighborhood(likelihood,cell,object(),space=space)
    assert neighborhood['points'] == 1+(6+len(NUISANCE_NAMES))*6
    assert neighborhood['min_scaled_eigenvalue']['field'] > 0
    assert neighborhood['min_scaled_eigenvalue']['retained_covariance'] > 0


def test_real_growth_factories_have_independent_six_column_fd_agreement():
    from pfsfog.extended_validation import measure_growth_jacobian
    space = make_cosmology_spec(FIDUCIAL,extended=True)
    measured = measure_growth_jacobian(space,FIDUCIAL,Z)
    assert measured['status'] == 'passed'
    assert len(measured['rows']) == 48
    assert measured['raw_column_norms']['h']['ad'] > 0


def test_extended_ap_attachment_uses_explicit_space_and_no_historical_relaxation(monkeypatch):
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_validation_suite import physical_dependencies
    space,likelihood,cell,support = _physical_fixture()
    row = _row(space,cell.derivatives[0,:,:6])
    cell.fiducial_mean = np.array([[10.,20.,30.]])
    provenance = {'project_head':'head','source_content_hash':'source','backend_snapshot':'backend',
                  'network_hash':'network','survey_input_hash':'survey'}
    monkeypatch.setattr(ap,'expected_support_schedule',lambda *args,**kwargs:support)
    report = {'status':'passed','scope':'all_cells','run_id':'run',
        'configuration':ap.ap_configuration(likelihood.config),
        'physical_dependencies':physical_dependencies(provenance),
        'reference_cosmology':dict(FIDUCIAL),'metric_contract':ap.AP_METRIC_CONTRACT,
        'angular_grids':ap.angular_grid_declaration(32),
        'pure_ap_controls':ap.measure_pure_ap_controls(),
        'selected_cell_ids':['cell'],'completed_cells':['cell'],'empty_cells':[],
        'points':[row],'cells':[ap.ap_cell_record(likelihood,cell,[row],space=space)]}
    assert ap.validate_ap_evidence(report,likelihood.config,provenance,object(),'run',
                                  likelihood=likelihood,space=space)['status'] == 'passed'
    with pytest.raises(ValueError,match='historical'):
        ap.validate_ap_point_rows([row],[('cell',('DESI-ELG','DESI-ELG'),.015)],
                                 space=space,historical_only=True)
