"""AP/domain evidence must bind actual points and quantitative measurements."""

import pytest
import numpy as np
import copy
from types import SimpleNamespace


def _row(cell="bin00-radial00-overlap", k=.01125):
    from pfsfog.revision_validation_ap import ap_parameter_schedule
    nodes, _ = np.polynomial.legendre.leggauss(32)
    mu = ((nodes+1)/2).tolist()
    coordinates = [{"parameter":name,"sign":sign,"step":step,
                    "alpha_perp":1., "alpha_para":1.,
                    "response_shape":[1,len(mu)], "response_min":1.,
                    "response_max":2., "response_values":[1.]*16+[2.]*16}
                   for name,sign,step in ap_parameter_schedule()]
    return {"cell_id":cell, "pair":["DESI-ELG", "DESI-ELG"], "k":k,
            "mu":mu, "shape":[1, len(mu)],
            "domain":{"k_true_min":k, "k_true_max":k,
                      "mu_true_min":mu[0], "mu_true_max":mu[-1],
                      "k_margin_lower":k-.002, "k_margin_upper":.5-k,
                      "parameter_coordinates":coordinates},
            "pair_covariance":np.eye(3).tolist(),
            "whitened_physical_scale":float(np.sqrt(3)),
            "fiducial_multipoles":[1.,1.,1.], "direct_multipoles":[1.,1.,1.],
            "total_ad":np.zeros((3,3)).tolist(),
            "intrinsic_ad":np.zeros((3,3)).tolist(),
            "geometry_ad":np.zeros((3,3)).tolist(),
            "fd_derivatives":np.zeros((3,3,3)).tolist(),
            "alpha_derivatives":np.zeros((2,3)).tolist(),
            "continuity_deltas":np.zeros((2,3)).tolist(),
            "identity_error":0., "continuity_error":0.,
            "pure_ap_isotropic_error":0.,
            "pure_ap_anisotropic_error":0.,
            "ad_fd_errors":np.zeros((3,3)).tolist(),
            "chain_errors":[0.]*3,
            "alpha_fd_errors":np.zeros((3,3)).tolist(),
            "interpolation":[{"nk":4096,"mean_error":0.,
                              "derivative_errors":[0.]*3,
                              "mean":[1.,1.,1.], "ad":np.zeros((3,3)).tolist()},
                             {"nk":8192,"mean_error":0.,
                              "derivative_errors":[0.]*3,
                              "mean":[1.,1.,1.], "ad":np.zeros((3,3)).tolist()}],
            "angular":{"n_mu":64,"mean_error":0.,
                       "derivative_errors":[0.]*3,
                       "mean":[1.,1.,1.], "ad":np.zeros((3,3)).tolist()}}


def test_ap_quantitative_rows_require_exact_keys_and_finite_metrics():
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    expected = [(_row()["cell_id"], tuple(_row()["pair"]), .01125),
                (_row()["cell_id"], tuple(_row()["pair"]), .01375)]
    rows = [_row(k=.01125), _row(k=.01375)]
    assert validate_ap_point_rows(rows, expected)["status"] == "passed"
    for bad in ([rows[0]], [rows[0], rows[0]],
                [rows[0], {**rows[1], "identity_error":float("nan")}],
                [rows[0], {**rows[1], "interpolation":rows[1]["interpolation"][:1]}],
                [rows[0], {**rows[1], "angular":{**rows[1]["angular"],
                                                "derivative_errors":[.002]*3}}]):
        with pytest.raises(ValueError):
            validate_ap_point_rows(bad, expected)


def test_ap_quantitative_rows_reject_false_domain_and_failed_chain():
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    expected = [(_row()["cell_id"], tuple(_row()["pair"]), .01125)]
    for change in ({"domain":{**_row()["domain"], "k_margin_lower":-1e-3}},
                   {"domain":{**_row()["domain"],
                              "parameter_coordinates":_row()["domain"]["parameter_coordinates"][:-1]}},
                   {"chain_errors":[1e-9]*3},
                   {"ad_fd_errors":[[.002]*3]*3},
                   {"shape":[2, 2]}):
        bad = {**_row(), **change}
        with pytest.raises(ValueError):
            validate_ap_point_rows([bad], expected)


def test_ap_validator_recomputes_errors_from_raw_arrays():
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    base = _row()
    expected = [(base["cell_id"], tuple(base["pair"]), base["k"])]
    assert validate_ap_point_rows([base], expected)["status"] == "passed"
    for field, value in (("direct_multipoles",[2.,1.,1.]),
                         ("fd_derivatives",[[[1.,0.,0.]]*3]*3),
                         ("pair_covariance",np.diag([-1.,1.,1.]).tolist())):
        bad = copy.deepcopy(base)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_ap_point_rows([bad], expected)
    bad = copy.deepcopy(base)
    bad["interpolation"][0]["ad"][0][2] = 1.
    with pytest.raises(ValueError):
        validate_ap_point_rows([bad], expected)
    bad = copy.deepcopy(base)
    bad["domain"]["parameter_coordinates"][1]["response_values"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_ap_point_rows([bad], expected)


def test_ap_attachment_binds_run_configuration_dependencies_and_complete_scope(monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    provenance = {"project_head":"head", "source_content_hash":"source",
                  "backend_snapshot":"backend", "network_hash":"network",
                  "survey_input_hash":"survey"}
    report = _bound_report(ap,config,rows,cell,likelihood,provenance)
    assert ap.validate_ap_evidence(report, config, provenance, object(), "run",
                                   likelihood=likelihood)["status"] == "passed"
    for change in ({"run_id":"other"}, {"scope":"representative"},
                   {"physical_dependencies":{**report["physical_dependencies"],
                                             "backend_snapshot":"stale"}},
                   {"configuration":{**report["configuration"], "n_mu":64}},
                   {"points":[]}):
        with pytest.raises(ValueError):
            ap.validate_ap_evidence({**report, **change}, config, provenance, object(), "run",likelihood=likelihood)


def test_independent_pure_ap_toys_check_jacobian_and_anisotropic_mapping():
    from pfsfog.revision_validation_ap import pure_ap_toy_errors
    isotropic, anisotropic = pure_ap_toy_errors()
    assert isotropic < 1e-12
    assert anisotropic < 1e-12


def test_ap_probe_persists_failed_cell_and_preserves_evidence(tmp_path, monkeypatch):
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig("B", dk=.0025)
    cell = SimpleNamespace(metadata={"cell_id":"bin00-radial00-overlap",
                                     "retained_pairs":[("DESI-ELG", "DESI-ELG")]})
    likelihood = SimpleNamespace(cells=[cell], reference_cosmology={}, grid=SimpleNamespace(centers=np.array([.01125])))
    monkeypatch.setattr(ap, "build_revision_likelihood", lambda *args:likelihood)
    monkeypatch.setattr(ap, "expected_support_schedule", lambda *args:[
        {"cell_id":cell.metadata["cell_id"],
         "retained_pairs":cell.metadata["retained_pairs"]}])
    monkeypatch.setattr(ap, "validate_support_schedule", lambda *args:None)
    def fail(*args,**kwargs):
        raise RuntimeError("measured response out of domain")
    monkeypatch.setattr(ap, "measure_ap_cell", fail)
    with pytest.raises(RuntimeError, match="out of domain"):
        ap.run_ap_probe(config, {"source_content_hash":"source"}, object(),
                        object(), object(), tmp_path, "run", max_cells=1)
    report = __import__("json").loads((tmp_path / "run.ap-evidence.json").read_text())
    assert report["status"] == "failed"
    assert report["active_cell"] == cell.metadata["cell_id"]
    assert report["failure"]["exception"] == "measured response out of domain"


def test_cli_exposes_ap_execution_and_attachment_modes():
    from scripts.validate_revision import _arguments
    assert _arguments(["--ap-probe"]).ap_probe
    assert str(_arguments(["--ap-evidence", "evidence.json"]).ap_evidence) == "evidence.json"
    options = _arguments(["--resolution-complete", "base.json",
                          "--changed-shell-ap-B", "b.json",
                          "--changed-shell-ap-C", "c.json"])
    assert options.resolution_complete.name == "base.json"
    assert options.changed_shell_ap_b.name == "b.json"
    assert options.changed_shell_ap_c.name == "c.json"


def test_ap_table_configuration_requires_two_successive_doublings():
    from pfsfog.revision_forecast import RevisionConfig
    default = RevisionConfig("B")
    assert (default.ap_interpolation_nk, default.ap_refinement_nk,
            default.ap_interpolation_nmu) == (2048,(4096,8192),101)
    candidate = RevisionConfig("B", ap_interpolation_nk=8192,
                               ap_refinement_nk=(16384,32768))
    assert candidate.ap_refinement_nk == (16384,32768)
    for wrong in ((8192,16384),(16384,16384),(16384,),()):
        with pytest.raises(ValueError, match="AP.*refinement"):
            RevisionConfig("B",ap_interpolation_nk=8192,ap_refinement_nk=wrong)


def test_ap_probe_rejects_failed_quantitative_partial_cell(tmp_path, monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    empty = SimpleNamespace(metadata={"cell_id":"empty","retained_pairs":[]})
    likelihood.cells.append(empty)
    monkeypatch.setattr(ap, "build_revision_likelihood", lambda *args:likelihood)
    monkeypatch.setattr(ap, "expected_support_schedule", lambda *args:[
        {"cell_id":c.metadata["cell_id"],"retained_pairs":c.metadata["retained_pairs"]}
        for c in likelihood.cells])
    rows[0]['chain_errors'] = [1e-4]*3
    monkeypatch.setattr(ap, "measure_ap_cell", lambda *args,**kwargs:rows)
    with pytest.raises(ValueError, match="chain"):
        ap.run_ap_probe(config, {}, object(), object(), object(), tmp_path, "run", max_cells=1)
    report = __import__("json").loads((tmp_path / "run.ap-evidence.json").read_text())
    assert report["status"] == "failed"


def test_resolution_companion_requires_six_exact_runs_and_both_changed_ap_reports(monkeypatch):
    from dataclasses import replace
    from pfsfog import revision_validation_ap as ap
    from pfsfog import revision_validation_physical as physical
    from pfsfog.revision_validation_suite import validate_resolution_evidence, physical_dependencies
    from pfsfog.revision_artifacts import _configuration, run_identity
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig("B", dk=.0025)
    provenance = {"project_head":"head", "source_content_hash":"source",
                  "model_flags":{"fft_n":256}, "requested_dk":.005}
    def identity(variant):
        return run_identity(_configuration(variant, variant.nuisance_factor,
                            requested_dk=.005, model_flags=provenance["model_flags"]),
                            provenance)
    monkeypatch.setattr(physical, "expected_support_schedule", lambda *args:[
        {"retained_pairs":[("DESI-ELG", "DESI-ELG")]}])
    runs = []
    for case in ("B", "C"):
        for tag, dk, nmu in (("base",.0025,32),("angular",.0025,64),("shell",.00125,32)):
            variant = replace(config, case=case, dk=dk, n_mu=nmu)
            from pfsfog.revision_forecast import make_shell_grid
            shells = len(make_shell_grid(variant.kmin,variant.kmax,variant.dk).centers)
            runs.append({"key":f"{case}_{tag}","run_id":identity(variant),
                         "configuration":_configuration(variant,variant.nuisance_factor,
                             requested_dk=.005,model_flags=provenance["model_flags"]),
                         "dk":dk,"n_mu":nmu,"expected_shells":shells,"processed_shells":shells})
    monkeypatch.setattr(ap, "expected_ap_point_keys", lambda *args:[
        (_row(k=.010625)["cell_id"], tuple(_row()["pair"]), .010625)])
    changed = {}
    live = {}
    for case in ("B","C"):
        variant = replace(config, case=case, dk=.00125)
        live[case] = SimpleNamespace(config=variant)
        changed[case] = {"status":"passed","scope":"all_cells",
                         "run_id":identity(variant),
                         "configuration":ap.ap_configuration(variant),
                         "physical_dependencies":physical_dependencies(provenance)}
    def attached(report, variant, provenance, survey, run_id, *, likelihood):
        assert likelihood is live[variant.case]
        if report['run_id'] != run_id:
            raise ValueError('stale AP identity')
        return {'status':'passed'}
    monkeypatch.setattr(ap,'validate_ap_evidence',attached)
    report = {"status":"passed","scope":"representative_BC_full_redshift",
              "physical_dependencies":physical_dependencies(provenance),
              "configuration":{"kmin":.01,"kmax":.20,"dk":.0025,"n_mu":32,
                               "nuisance_factor":1.,"exclude_elg1":False,
                               "nominal_bins":[list(x) for x in config.nominal_bins]},
              "values":{r["key"]:[1.]*11 for r in runs},"runs":runs,
              "changed_shell_domain_ap":{"status":"passed","reports":changed}}
    assert validate_resolution_evidence(report,config,provenance,object(),changed_shell_likelihoods=live)["status"] == "passed"
    bad = {**report,"runs":runs[:5]}
    with pytest.raises(ValueError):
        validate_resolution_evidence(bad,config,provenance,object(),changed_shell_likelihoods=live)
    bad = {**report,"changed_shell_domain_ap":{"status":"passed","reports":{"B":changed["B"]}}}
    with pytest.raises(ValueError):
        validate_resolution_evidence(bad,config,provenance,object(),changed_shell_likelihoods=live)
    stale = {**changed["C"],"run_id":"wrong"}
    bad = {**report,"changed_shell_domain_ap":{"status":"passed","reports":{**changed,"C":stale}}}
    with pytest.raises(ValueError):
        validate_resolution_evidence(bad,config,provenance,object(),changed_shell_likelihoods=live)


@pytest.mark.parametrize("settings", [
    {"kmax":.20}, {"kmax":.15}, {"kmax":.20, "nuisance_factor":3.},
    {"kmax":.20, "nuisance_factor":10.}, {"kmax":.20, "exclude_elg1":True},
])
def test_resolution_producer_records_six_task2_identities_even_while_pending(tmp_path, monkeypatch, settings):
    from scripts import validate_revision as cli
    from pfsfog.revision_artifacts import _configuration, run_identity
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig("B", dk=.0025, **settings)
    provenance = {"project_head":"head", "source_content_hash":"source",
                  "model_flags":{"fft_n":256}, "requested_dk":.005}
    likelihood = SimpleNamespace(validation={"expected_shells":1,"processed_shells":1})
    result = SimpleNamespace(cov_cosmo_marg=np.eye(3),growth_cov_marg=np.eye(8),
                             names=["x"],growth_z=np.arange(8.),validation={})
    monkeypatch.setattr(cli, "build_revision_likelihood", lambda *args:likelihood)
    monkeypatch.setattr(cli, "apply_revision_priors", lambda *args:result)
    path = cli._resolution_probe(config,provenance,object(),object(),object(),tmp_path)
    assert path == tmp_path / "bc-resolution-evidence.json"
    import json
    report = json.loads((tmp_path / "bc-resolution-evidence.json").read_text())
    assert report["status"] == "pending"
    assert len(report["runs"]) == 6
    assert report["changed_shell_domain_ap"]["reason"] == (
        "awaiting exact changed-shell B/C AP reports and live attachment")
    for row in report["runs"]:
        variant = RevisionConfig(row["key"][0], dk=row["dk"],n_mu=row["n_mu"],
                                 **settings)
        expected_config = _configuration(variant,variant.nuisance_factor,
            requested_dk=.005,model_flags=provenance["model_flags"])
        assert row["configuration"] == expected_config
        assert row["run_id"] == run_identity(expected_config,provenance)


@pytest.mark.parametrize("settings", [
    {"kmax":.18}, {"kmax":.15, "nuisance_factor":3.},
    {"kmax":.15, "exclude_elg1":True},
])
def test_resolution_producer_rejects_unsupported_cohort_before_build(tmp_path, monkeypatch, settings):
    from scripts import validate_revision as cli
    from pfsfog.revision_forecast import RevisionConfig
    monkeypatch.setattr(cli, "build_revision_likelihood",
                        lambda *args:pytest.fail("build reached"))
    config = RevisionConfig("B", dk=.0025, **settings)
    with pytest.raises(ValueError, match="unsupported suite setting"):
        cli._resolution_probe(config, {}, object(), object(), object(), tmp_path)


def test_resolution_producer_rejects_smoke_bin_before_build(tmp_path, monkeypatch):
    from scripts import validate_revision as cli
    from pfsfog.revision_forecast import RevisionConfig
    monkeypatch.setattr(cli, "build_revision_likelihood",
                        lambda *args:pytest.fail("build reached"))
    config = RevisionConfig("B", kmax=.15, nominal_bins=((1., 1.2),))
    with pytest.raises(ValueError, match="eight-bin"):
        cli._resolution_probe(config, {}, object(), object(), object(), tmp_path)


def test_resolution_completion_attaches_quantitative_reports_and_preserves_failure(tmp_path, monkeypatch):
    import json
    from scripts import validate_revision as cli
    from pfsfog.revision_forecast import RevisionConfig
    base = tmp_path / "base.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    base.write_text(json.dumps({"status":"pending","scope":"representative_BC_full_redshift"}))
    b.write_text(json.dumps({"status":"passed","run_id":"B"}))
    c.write_text(json.dumps({"status":"passed","run_id":"C"}))
    def reject(report,*args,**kwargs):
        assert set(report["changed_shell_domain_ap"]["reports"]) == {"B","C"}
        raise ValueError("missing measured k point")
    monkeypatch.setattr(cli,"validate_resolution_evidence",reject)
    monkeypatch.setattr(cli,"build_revision_likelihood",lambda *args:object())
    with pytest.raises(ValueError, match="missing measured k point"):
        cli._resolution_complete(base,b,c,RevisionConfig("B",dk=.0025),{},object(),tmp_path,cosmo=object(),backend=object())
    output = json.loads((tmp_path / "bc-resolution-evidence.complete.json").read_text())
    assert output["status"] == "failed"
    assert output["failure"]["exception"] == "missing measured k point"


def test_amplitude_geometry_must_be_invariant_even_when_ad_fd_agree():
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    row = _row()
    for point in row['domain']['parameter_coordinates']:
        if point['parameter'] == 'ln10_10_As':
            point['alpha_perp'] = 1. + .02*point['sign']*point['step']
    row['alpha_derivatives'][0][0] = .02
    km, mm = [], []
    mu = np.asarray(row['mu'])
    for point in row['domain']['parameter_coordinates']:
        ap, al = point['alpha_perp'], point['alpha_para']
        fac = np.sqrt(1 + mu**2*((ap/al)**2-1))
        km.extend(row['k']*fac/ap)
        mm.extend(mu*(ap/al)/fac)
    row['domain'].update(k_true_min=min(km), k_true_max=max(km),
                         mu_true_min=min(mm), mu_true_max=max(mm),
                         k_margin_lower=min(km)-.002, k_margin_upper=.5-max(km))
    # A coherent amplitude dependence is forbidden independently of AD/FD.
    with pytest.raises(ValueError, match='amplitude'):
        validate_ap_point_rows([row], [(row['cell_id'], tuple(row['pair']), row['k'])])


def test_ap_rows_accept_only_configured_candidate_sequence():
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    row = _row()
    row['interpolation'][0]['nk'] = 16384
    row['interpolation'][1]['nk'] = 32768
    key = (row['cell_id'], tuple(row['pair']), row['k'])
    assert validate_ap_point_rows([row], [key], ap_nk=8192,
                                 refinement_nk=(16384,32768))['status'] == 'passed'
    with pytest.raises(ValueError, match='sequence'):
        validate_ap_point_rows([row], [key])


@pytest.mark.parametrize('mutation', ['jacobian', 'swap', 'sign'])
def test_pure_ap_controls_reach_production_remap_and_reject_mutations(monkeypatch, mutation):
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_theory import APInterpolationBackend
    original = APInterpolationBackend._get_pkmu_pair_ref_interp
    def wrong(self, k, mu, transverse, radial, *args, **kwargs):
        if mutation == 'swap':
            transverse, radial = radial, transverse
        elif mutation == 'sign':
            transverse, radial = 2-transverse, 2-radial
        result = original(self, k, mu, transverse, radial, *args, **kwargs)
        return result * transverse**2*radial if mutation == 'jacobian' else result
    monkeypatch.setattr(APInterpolationBackend, '_get_pkmu_pair_ref_interp', wrong)
    with pytest.raises(ValueError, match='pure AP'):
        ap.validate_pure_ap_controls(ap.measure_pure_ap_controls())


def test_pure_ap_raw_controls_cover_auto_cross_and_recompute_outputs():
    from pfsfog import revision_validation_ap as ap
    controls = ap.measure_pure_ap_controls()
    assert ap.validate_pure_ap_controls(controls)['status'] == 'passed'
    assert {(r['pair_kind'],r['control']) for r in controls} == {
        (kind,control) for kind in ('auto','cross')
        for control in ('isotropic','transverse','radial','anisotropic')}
    controls[0]['observed'][0][0] *= 1.01
    with pytest.raises(ValueError, match='pure AP'):
        ap.validate_pure_ap_controls(controls)


@pytest.mark.parametrize('mutation', ['jacobian', 'radial_template'])
def test_pure_ap_cross_stochastic_branch_rejects_geometry_mutation(monkeypatch, mutation):
    from pfsfog import revision_theory, revision_validation_ap as ap
    original = revision_theory.ap_coordinates

    def wrong_cross_geometry(*args):
        true_k, true_mu, jacobian = original(*args)
        if mutation == 'jacobian':
            return true_k, true_mu, 1.01*jacobian
        return true_k, .8*true_mu, jacobian

    # The interpolation backend uses only the mapped coordinates from this
    # helper. Changing its Jacobian or radial template therefore isolates
    # PairResponse's separately remapped cross-stochastic contribution.
    monkeypatch.setattr(revision_theory, 'ap_coordinates', wrong_cross_geometry)
    with pytest.raises(ValueError, match='pure AP cross'):
        ap.validate_pure_ap_controls(ap.measure_pure_ap_controls())


@pytest.mark.parametrize('role', ['covariance', 'refined_projection'])
def test_ap_domain_requires_all_observed_grids_and_all_nineteen_points(role):
    from pfsfog.revision_validation_ap import validate_ap_point_rows
    row = _row()
    key = (row['cell_id'],tuple(row['pair']),row['k'])
    with pytest.raises(ValueError, match='angular domain'):
        validate_ap_point_rows([row], [key], require_grid_coverage=True)


def _complete_domains(row):
    from pfsfog.revision_validation_ap import angular_grid_declaration
    row['observed_domains'] = {}
    for role in ('covariance','refined_projection'):
        mu = np.asarray(angular_grid_declaration(32)[role]['mu'])
        domain = copy.deepcopy(row['domain'])
        domain.update(mu_true_min=float(np.abs(mu).min()),mu_true_max=float(np.abs(mu).max()))
        for point in domain['parameter_coordinates']:
            point.update(response_shape=[1,len(mu)],response_values=[1.]*len(mu),
                         response_min=1.,response_max=1.)
        row['observed_domains'][role] = domain
    return row


def _bound_fixture(monkeypatch):
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig('B', kmax=.0125, dk=.0025)
    rows = [_complete_domains(_row()), _complete_domains(_row())]
    rows[1]['pair'] = ['DESI-ELG','DESI-LRG']
    total = np.arange(1.,19.).reshape(6,3)
    for ipair,row in enumerate(rows):
        ad = total[3*ipair:3*ipair+3]
        row['total_ad'] = row['intrinsic_ad'] = ad.tolist()
        row['fd_derivatives'] = np.repeat(ad.T[:,None,:],3,axis=1).tolist()
        for refined in [*row['interpolation'], row['angular']]:
            refined['ad'] = ad.tolist()
    covariance = np.eye(6)
    covariance[0,3] = covariance[3,0] = .4
    cell = SimpleNamespace(metadata={'cell_id':rows[0]['cell_id'],
                                     'retained_pairs':[tuple(r['pair']) for r in rows]},
                           names=['Omegam','nuisance','ln10_10_As','Mnu'],
                           derivatives=np.stack([total[:,2],np.ones(6),total[:,0],total[:,1]],axis=-1)[None],
                           covariance=covariance[None],fiducial_mean=np.ones((1,6)))
    likelihood = SimpleNamespace(config=config,cells=[cell],
                                 grid=SimpleNamespace(centers=np.array([.01125])),
                                 reference_cosmology={'h':.67})
    keys = [(r['cell_id'],tuple(r['pair']),r['k']) for r in rows]
    monkeypatch.setattr(ap, 'expected_ap_point_keys', lambda *args:keys)
    monkeypatch.setattr(ap, 'validate_support_schedule', lambda *args:None)
    monkeypatch.setattr(ap, 'expected_support_schedule', lambda *args:[{'cell_id':cell.metadata['cell_id'],'retained_pairs':cell.metadata['retained_pairs']}])
    return ap,config,rows,cell,likelihood


def test_attachment_binds_live_mean_columns_pair_order_and_full_covariance(monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    report = {'status':'passed','scope':'all_cells','run_id':'run',
              'configuration':ap.ap_configuration(config),'physical_dependencies':ap.physical_dependencies({}),
              'angular_grids':ap.angular_grid_declaration(32),
              'pure_ap_controls':ap.measure_pure_ap_controls(),
              'reference_cosmology':likelihood.reference_cosmology,
              'metric_contract':ap.AP_METRIC_CONTRACT,
              'selected_cell_ids':[cell.metadata['cell_id']],
              'completed_cells':[cell.metadata['cell_id']], 'empty_cells':[],
              'points':rows,'cells':[ap.ap_cell_record(likelihood,cell,rows)]}
    assert ap.validate_ap_evidence(report,config,{},object(),'run',likelihood=likelihood)['status'] == 'passed'
    # Each tampering remains internally self-consistent but contradicts live assembly.
    for field in ('mean','column','off_pair_covariance','pair_order'):
        bad = copy.deepcopy(report)
        if field == 'mean':
            bad['points'][0]['fiducial_multipoles'][0] = 2.
        elif field == 'column':
            bad['points'][0]['total_ad'][0][0] = 0.
        elif field == 'off_pair_covariance':
            bad['cells'][0]['covariance'][0][0][3] = .1
            bad['cells'][0]['covariance'][0][3][0] = .1
        else:
            bad['cells'][0]['retained_pairs'].reverse()
        with pytest.raises(ValueError, match='assembled|retained'):
            ap.validate_ap_evidence(bad,config,{},object(),'run',likelihood=likelihood)
    surrogate = copy.deepcopy(report)
    for row in surrogate['points']:
        row['total_ad'] = row['intrinsic_ad'] = np.zeros((3,3)).tolist()
        row['fd_derivatives'] = np.zeros((3,3,3)).tolist()
        for check in [*row['interpolation'],row['angular']]:
            check['ad'] = np.zeros((3,3)).tolist()
    with pytest.raises(ValueError, match='assembled'):
        ap.validate_ap_evidence(surrogate,config,{},object(),'run',likelihood=likelihood)


def test_full_covariance_metric_retains_pair_ell_gate_and_zero_branch(monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    record = ap.ap_cell_record(likelihood,cell,rows)
    assert record['diagnostics']['status'] == 'passed'
    # A large unrelated component must not hide a small pair/ell error.
    rows[0]['fd_derivatives'][0] = [[1.01,4.,7.]]*3
    diagnostics = ap.ap_cell_record(likelihood,cell,rows)['diagnostics']
    assert diagnostics['status'] == 'failed'
    residual = diagnostics['blocks'][1]['ad_fd'][0][0]
    assert residual['error'] == pytest.approx(.01/1.01)
    assert residual['reference_norm'] == pytest.approx(1.01/np.sqrt(1-.4**2))


def _bound_report(ap,config,rows,cell,likelihood,provenance=None):
    return {'status':'passed','scope':'all_cells','run_id':'run',
            'configuration':ap.ap_configuration(config),'physical_dependencies':ap.physical_dependencies(provenance or {}),
            'angular_grids':ap.angular_grid_declaration(config.n_mu),
            'pure_ap_controls':ap.measure_pure_ap_controls(config.ap_interpolation_nk,config.ap_interpolation_nmu),
            'reference_cosmology':likelihood.reference_cosmology,
            'metric_contract':ap.AP_METRIC_CONTRACT,
            'selected_cell_ids':[cell.metadata['cell_id']],
            'completed_cells':[cell.metadata['cell_id']], 'empty_cells':[],
            'points':rows,'cells':[ap.ap_cell_record(likelihood,cell,rows)]}


def test_legacy_all_cell_validator_cannot_certify_intrinsic_profile_without_tables(monkeypatch):
    from dataclasses import replace
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    config = replace(config,ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    likelihood.config = config
    for row in rows:
        for item,nk in zip(row['interpolation'],config.ap_refinement_nk): item['nk'] = nk
    report = _bound_report(ap,config,rows,cell,likelihood)
    with pytest.raises(ValueError,match='intrinsic-table'):
        ap.validate_ap_evidence(report,config,{},object(),'run',likelihood=likelihood)


def test_legacy_probe_rejects_intrinsic_profile_before_theory_initialization(monkeypatch,tmp_path):
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig('B',ap_interpolation_nk=16384,ap_refinement_nk=(32768,65536))
    def forbidden(*args,**kwargs): raise AssertionError('legacy probe must not initialize theory for this profile')
    monkeypatch.setattr(ap,'build_revision_likelihood',forbidden)
    with pytest.raises(ValueError,match='intrinsic-table'):
        ap.run_ap_probe(config,{},object(),object(),object(),tmp_path,'run')


@pytest.mark.parametrize('role', ['covariance','refined_projection'])
@pytest.mark.parametrize('mutation', ['missing_point','nonfinite','false_extrema','missing_grid','different_alphas'])
def test_all_angular_neighborhood_raw_evidence_is_revalidated(role,mutation):
    from pfsfog import revision_validation_ap as ap
    row = _complete_domains(_row())
    key = (row['cell_id'],tuple(row['pair']),row['k'])
    assert ap.validate_ap_point_rows([row],[key],require_grid_coverage=True)['status'] == 'passed'
    domain = row['observed_domains'][role]
    if mutation == 'missing_point':
        domain['parameter_coordinates'].pop()
    elif mutation == 'nonfinite':
        domain['parameter_coordinates'][-1]['response_values'][0] = float('nan')
    elif mutation == 'false_extrema':
        domain['mu_true_min'] += .001
    elif mutation == 'missing_grid':
        del row['observed_domains'][role]
    else:
        domain['parameter_coordinates'][-1]['alpha_perp'] = 1.001
        mu = np.abs(np.asarray(ap.angular_grid_declaration(32)[role]['mu']))
        ks,ms = [],[]
        for point in domain['parameter_coordinates']:
            a,b = point['alpha_perp'],point['alpha_para']
            f = np.sqrt(1+mu**2*((a/b)**2-1))
            ks.extend(row['k']*f/a)
            ms.extend(mu*(a/b)/f)
        domain.update(k_true_min=min(ks),k_true_max=max(ks),mu_true_min=min(ms),mu_true_max=max(ms),
                      k_margin_lower=min(ks)-.002,k_margin_upper=.5-max(ks))
    with pytest.raises(ValueError):
        ap.validate_ap_point_rows([row],[key],require_grid_coverage=True)


def test_metric_zero_branch_uses_physical_scale_and_own_ap_thresholds(monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    # A vanishing pair/ell response is allowed; it must still have an absolute check.
    cell.derivatives[0,0,2] = 0.
    rows[0]['total_ad'][0][0] = 0.
    rows[0]['intrinsic_ad'][0][0] = 0.
    for step in rows[0]['fd_derivatives'][0]:
        step[0] = 0.
    for check in [*rows[0]['interpolation'],rows[0]['angular']]:
        check['ad'][0][0] = 0.
    diagnostics = ap.ap_cell_record(likelihood,cell,rows)['diagnostics']
    block = diagnostics['blocks'][1]
    assert block['ad_fd'][0][0]['branch'] == 'normalized absolute'
    assert block['ad_fd'][0][0]['limit'] == 1e-6
    assert block['chain'][0]['limit'] == 1e-10
    assert block['interpolation'][0]['ad'][0]['limit'] == 5e-4
    # Wrong zero-block FD exceeds the normalized absolute threshold despite tiny AD.
    for step in rows[0]['fd_derivatives'][0]:
        step[0] = .01
    assert ap.ap_cell_record(likelihood,cell,rows)['diagnostics']['status'] == 'failed'


def test_candidate_probe_producer_matches_live_binder_with_all_domain_grids(tmp_path,monkeypatch):
    """Cheap analytic response isolates orchestration; remap itself is tested above."""
    import json
    import jax
    import jax.numpy as jnp
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_forecast import RevisionConfig
    config = RevisionConfig('B',kmax=.0125,dk=.0025,ap_interpolation_nk=8192,
                            ap_refinement_nk=(16384,32768))
    pair = ('DESI-ELG','DESI-ELG')
    reference = {'h':.67,'ln10_10_As':3.,'mnu':.06,'Omega_m':.3}
    values = dict(zip(ap.COSMO_NAMES,[3.,.06,.3]))
    values.update({f'bin00:{pair[0]}:{name}':0. for name in ap.NUISANCE_NAMES})
    class Backend:
        def get_pkmu_pair(self,k,mu,pk,params,add_stochasticity=False):
            return pk*jnp.asarray(k)[:,None]*(1+.6*jnp.abs(jnp.asarray(mu))[None,:])
        def _get_pkmu_pair_ref_interp(self,k,mu,transverse,radial,pk,params,add_stochasticity=False):
            return self.get_pkmu_pair(k,mu,pk,params,add_stochasticity)
    backend = Backend()
    class Response:
        reference = {'h':.67,'ln10_10_As':3.,'mnu':.06,'Omega_m':.3}
        sigma8_ref = 1.
        ap_backend = backend
        sigma8 = staticmethod(lambda *args:1.)
        growth_rate = staticmethod(lambda *args:0.)
        plin = staticmethod(lambda z,p:p['ln10_10_As']+p['mnu']+p['Omega_m'])
        ap_alphas = staticmethod(lambda t:jnp.ones(2))
        def pkmu(self,k,mu,t,*args,**kwargs):
            return backend.get_pkmu_pair(k,mu,jnp.sum(t),{})
        def multipoles(self,k,t,*args,n_mu=32,**kwargs):
            nodes,weights = np.polynomial.legendre.leggauss(n_mu)
            mu = jnp.asarray((nodes+1)/2)
            return ap._project(self.pkmu(k,mu,t),mu,jnp.asarray(weights/2))
    response = Response()
    theta = jnp.asarray([3.,.06,.3])
    k = jnp.asarray([.01125])
    # The live assembler differentiates all pair-local coordinates, including
    # nuisance directions which this orchestration toy holds inactive.
    packed = jnp.asarray(list(values.values()))
    def assembled_mean(p):
        return response.multipoles(k,p[:3],p[3:])
    cell = SimpleNamespace(metadata={'cell_id':'cell','retained_pairs':[pair],
                                     'nominal_bin_index':0,'model_redshift':1.,'nbar':{pair[0]:1.}},
                           names=list(values),
                           derivatives=np.asarray(jax.jacfwd(assembled_mean)(packed)),
                           covariance=np.eye(3)[None],
                           fiducial_mean=np.asarray(assembled_mean(packed)))
    likelihood = SimpleNamespace(config=config,cells=[cell],grid=SimpleNamespace(centers=np.asarray(k)),
                                 reference_cosmology=reference)
    requests = []
    def factory(*args,ap_nk,ap_nmu):
        requests.append((ap_nk,ap_nmu))
        return Response()
    monkeypatch.setattr(ap,'make_pair_response',factory)
    monkeypatch.setattr(ap,'_cell_fiducials',lambda *args:values)
    monkeypatch.setattr(ap,'_checked_pairs',lambda *args:[pair])
    monkeypatch.setattr(ap,'_check_shells',lambda *args:None)
    monkeypatch.setattr(ap,'build_revision_likelihood',lambda *args:likelihood)
    monkeypatch.setattr(ap,'validate_support_schedule',lambda *args:None)
    monkeypatch.setattr(ap,'expected_support_schedule',lambda *args:[{'cell_id':'cell','retained_pairs':[pair]}])
    path = ap.run_ap_probe(config,{},object(),object(),backend,tmp_path,'run')
    report = json.loads(path.read_text())
    assert report['status'] == 'passed'
    assert requests == [(8192,101),(16384,101),(32768,101)]
    assert len(report['cells']) == 1
    assert len(report['points'][0]['observed_domains']['covariance']['parameter_coordinates']) == 19
    assert ap.validate_ap_evidence(report,config,{},object(),'run',likelihood=likelihood)['status'] == 'passed'
    report['configuration']['ap_interpolation_nk'] = 2048
    with pytest.raises(ValueError,match='configuration'):
        ap.validate_ap_evidence(report,config,{},object(),'run',likelihood=likelihood)


def test_binding_rejects_missing_completed_and_empty_cell_inventory(monkeypatch):
    ap,config,rows,cell,likelihood = _bound_fixture(monkeypatch)
    empty = SimpleNamespace(metadata={'cell_id':'empty','retained_pairs':[]})
    likelihood.cells.append(empty)
    support = [{'cell_id':c.metadata['cell_id'],'retained_pairs':c.metadata['retained_pairs']}
               for c in likelihood.cells]
    monkeypatch.setattr(ap,'expected_support_schedule',lambda *args:support)
    report = _bound_report(ap,config,rows,cell,likelihood)
    report.update(selected_cell_ids=[c['cell_id'] for c in support],
                  completed_cells=[c['cell_id'] for c in support],
                  empty_cells=[{'cell_id':'empty','reason':'no retained observations'}])
    assert ap.validate_ap_evidence(report,config,{},object(),'run',likelihood=likelihood)['status'] == 'passed'
    for field in ('completed_cells','empty_cells','selected_cell_ids'):
        bad = copy.deepcopy(report)
        bad[field] = []
        with pytest.raises(ValueError,match='inventory'):
            ap.validate_ap_evidence(bad,config,{},object(),'run',likelihood=likelihood)


@pytest.mark.parametrize('cross', [False, True], ids=['auto', 'cross'])
def test_ap_probe_uses_assembled_packed_ad_layout_and_preserves_fiducials(monkeypatch,cross):
    """AD tangent width must match assembly even for a cosmology-only report.

    Exercise the real PairResponse remap with cheap intrinsic physics. Distinct
    nonzero nuisance values, including cross noise, expose dropped or reordered
    coordinates independently of the three cosmological output columns.
    """
    import jax
    import jax.numpy as jnp
    from pfsfog import revision_validation_ap as ap
    from pfsfog.revision_forecast import RevisionConfig
    from pfsfog.revision_theory import PairResponse, APInterpolationBackend

    config = RevisionConfig('B',kmax=.015,dk=.0025)
    pair = ('DESI-ELG','DESI-LRG' if cross else 'DESI-ELG')
    reference = {'h':.67,'omega_b':.022,'n_s':.96,'omega_cdm':.11,
                 'ln10_10_As':3.,'mnu':.06,'Omega_m':.3}
    theta = jnp.asarray([3.,.06,.3])
    qa = jnp.arange(1.,len(ap.NUISANCE_NAMES)+1)/10
    qb = qa+.4 if cross else None
    qc = jnp.asarray([.07,3.]) if cross else None
    packed = jnp.concatenate((theta,qa,*((qb,qc) if cross else ())))
    names = list(ap.COSMO_NAMES)
    names += [f'bin00:{pair[0]}:{name}' for name in ap.NUISANCE_NAMES]
    if cross:
        names += [f'bin00:{pair[1]}:{name}' for name in ap.NUISANCE_NAMES]
        names += [f'bin00:{pair[0]}|{pair[1]}:{name}' for name in ap.CROSS_STOCH_NAMES]

    class Backend:
        def get_pkmu_pair(self,k,mu,pk,params,add_stochasticity=False):
            coefficient = sum(params['bias'].values())+sum(params['ctr'].values())
            if 'bias2' in params:
                coefficient += 2*sum(params['bias2'].values())+3*sum(params['ctr2'].values())
            if add_stochasticity:
                coefficient += sum(params['stoch'].values())
            return pk*coefficient*jnp.asarray(k)[:,None]*(1+.6*jnp.abs(jnp.asarray(mu))[None,:])

    backend = Backend()

    def factory(*args,ap_nk,ap_nmu):
        response = object.__new__(PairResponse)
        response.space = None
        response.reference, response.z = reference, 1.
        response.nbar_a, response.nbar_b = 1., 2. if cross else None
        response.k_nl = config.k_nl
        response.sigma8 = lambda z,p:jnp.exp(.1*p['ln10_10_As']+.2*p['mnu']+.3*p['Omega_m'])
        response.growth_rate = lambda z,p:.5+.4*p['Omega_m']+.1*p['mnu']
        response.plin = lambda z,p:jnp.exp(p['ln10_10_As'])+p['mnu']+2*p['Omega_m']
        response.sigma8_ref = response.sigma8(1.,reference)
        response.ap_alphas = lambda t:jnp.exp(jnp.asarray([.1*(t[1]-.06)+.2*(t[2]-.3),
                                                         .3*(t[1]-.06)-.1*(t[2]-.3)]))
        response.ap_backend = APInterpolationBackend(backend,ap_nk,ap_nmu)
        return response

    response = factory(ap_nk=config.ap_interpolation_nk,ap_nmu=config.ap_interpolation_nmu)
    k = jnp.asarray([.01125,.01375])

    # Independent oracle follows build_cell_likelihood's pair-local packing.
    def assembled_mean(p,apply_ap=True,n_mu=32):
        first = p[3:3+len(ap.NUISANCE_NAMES)]
        second = p[3+len(ap.NUISANCE_NAMES):3+2*len(ap.NUISANCE_NAMES)] if cross else None
        stochastic = p[-2:] if cross else None
        return response.multipoles(k,p[:3],first,second,stochastic,ap=apply_ap,n_mu=n_mu)

    expected_mean = np.asarray(assembled_mean(packed))
    expected_ad = np.asarray(jax.jacfwd(assembled_mean)(packed))
    expected_intrinsic = np.asarray(jax.jacfwd(lambda p:assembled_mean(p,apply_ap=False))(packed))[:,:,:3]
    expected_angular = np.asarray(jax.jacfwd(lambda p:assembled_mean(p,n_mu=64))(packed))[:,:,:3]
    cell = SimpleNamespace(metadata={'cell_id':'cell','retained_pairs':[pair],
                                     'nominal_bin_index':0,'model_redshift':1.,
                                     'nbar':{pair[0]:1.,**({pair[1]:2.} if cross else {})}},
                           names=names,derivatives=expected_ad,
                           covariance=np.repeat(np.eye(3)[None],len(k),axis=0),
                           fiducial_mean=expected_mean)
    likelihood = SimpleNamespace(config=config,cells=[cell],names=names,fiducial=np.asarray(packed),
                                 grid=SimpleNamespace(centers=np.asarray(k)),
                                 reference_cosmology=reference)
    original_jacfwd = jax.jacfwd
    differentiated_points = []

    def record_jacfwd(function,*args,**kwargs):
        derivative = original_jacfwd(function,*args,**kwargs)
        def evaluate(point):
            differentiated_points.append(np.asarray(point).copy())
            return derivative(point)
        return evaluate

    monkeypatch.setattr(ap,'make_pair_response',factory)
    monkeypatch.setattr(ap,'_checked_pairs',lambda *args:[pair])
    monkeypatch.setattr(ap,'_check_shells',lambda *args:None)
    monkeypatch.setattr(ap.jax,'jacfwd',record_jacfwd)
    rows = ap.measure_ap_cell(likelihood,cell,backend)
    # Total, intrinsic, geometry, distance ratios, two tables and angular AD.
    assert len(differentiated_points) == 7
    for point in differentiated_points:
        np.testing.assert_array_equal(point,packed)
    np.testing.assert_array_equal([row['fiducial_multipoles'] for row in rows],expected_mean)
    np.testing.assert_array_equal([row['total_ad'] for row in rows],expected_ad[:,:,:3])
    np.testing.assert_array_equal([row['intrinsic_ad'] for row in rows],expected_intrinsic)
    np.testing.assert_array_equal([row['angular']['ad'] for row in rows],expected_angular)
    for idir,name in enumerate(ap.COSMO_NAMES):
        for istep,step in enumerate(ap.COSMO_STEPS[name]):
            expected_fd = (np.asarray(assembled_mean(packed.at[idir].add(step)))-
                           np.asarray(assembled_mean(packed.at[idir].add(-step))))/(2*step)
            np.testing.assert_array_equal(
                [row['fd_derivatives'][idir][istep] for row in rows],expected_fd)
    # Retain the strict live-array binder and the nuisance fiducials themselves.
    ap.ap_cell_record(likelihood,cell,rows)
    np.testing.assert_array_equal(likelihood.fiducial,packed)
