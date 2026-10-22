"""Revision likelihood assembly contracts with hand-computed small examples."""

import numpy as np
import pytest
import jax.numpy as jnp

from pfsfog.revision_forecast import (
    RevisionConfig, make_shell_grid, retained_pairs, build_cell_likelihood,
    build_revision_likelihood, apply_revision_priors, RevisionLikelihood,
    validate_reference_domain, NOMINAL_BINS,
)
from pfsfog.covariance_mt_general import multi_tracer_cov_general
from pfsfog.survey_geometry import SupportCell
from pfsfog.cosmo import FIDUCIAL
from pfsfog.eft_params import tracer_fiducials
from pfsfog.revision_theory import fiducial_nuisance
from pfsfog.surveys import default_survey_group


class LinearPairResponse:
    """Analytic signal with independent derivatives on both cross sides."""

    def __init__(self, a, b, z):
        self.a, self.b, self.z = a, b, z

    def pkmu(self, k, mu, theta, q_a, q_b=None, q_cross=None):
        signal = 5. if self.a == self.b else 1.
        return jnp.full((len(k), len(mu)), signal)

    def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu=32):
        # All three cosmological coordinates affect all three multipoles.
        cosmo = jnp.dot(theta, jnp.array([1., 2., 3.]))
        qa = q_a[0] + 2 * q_a[1]
        qb = 3 * q_b[0] if q_b is not None else 0.
        qc = 4 * q_cross[0] + 5 * q_cross[1] if q_cross is not None else 0.
        return jnp.ones((len(k), 3)) * (cosmo + qa + qb + qc) * jnp.array([1., 2., 3.])


def _fake_cell(monkeypatch, names=("DESI-ELG", "DESI-LRG", "PFS-ELG")):
    import pfsfog.revision_forecast as rf

    def factory(ps, z, reference, nbar_a, nbar_b=None, k_nl=.7, **kwargs):
        # Distinct densities identify the tracer pair without changing model z.
        reverse = {2e-4: "DESI-ELG", 3e-4: "DESI-LRG", 4e-4: "PFS-ELG"}
        return LinearPairResponse(reverse[nbar_a], reverse[nbar_b] if nbar_b else reverse[nbar_a], z)

    monkeypatch.setattr(rf, "make_pair_response", factory)
    densities = {"DESI-ELG": 2e-4, "DESI-LRG": 3e-4, "PFS-ELG": 4e-4}
    cell = SupportCell((1., 1.2), (1., 1.1), 1.1, names,
                       tuple(n for n in names if n.startswith("DESI-")),
                       {n: densities[n] for n in names}, 100., 200.)
    fid = {n: tracer_fiducials(n, 1.5, .6) for n in names}
    return cell, fid


def test_shell_grid_preserves_physical_endpoint_with_short_last_shell():
    grid = make_shell_grid(0.01, 0.022, 0.005)
    np.testing.assert_allclose(grid.edges, [0.01, 0.015, 0.02, 0.022])
    np.testing.assert_allclose(grid.centers, [0.0125, 0.0175, 0.021])
    np.testing.assert_allclose(grid.widths, [0.005, 0.005, 0.002])
    assert grid.edges[-1] == 0.022


def test_candidate_ap_table_reaches_every_production_pair_response(monkeypatch):
    import pfsfog.revision_forecast as rf
    cell, fid = _fake_cell(monkeypatch, ("DESI-ELG", "DESI-LRG"))
    received = []
    def factory(ps, z, reference, nbar_a, nbar_b=None, k_nl=.7, *,
                ap_nk=None, ap_nmu=None):
        received.append((ap_nk, ap_nmu))
        return LinearPairResponse("A", "B" if nbar_b else "A", z)
    monkeypatch.setattr(rf, "make_pair_response", factory)
    config = RevisionConfig("B", kmax=.015, ap_interpolation_nk=8192,
                            ap_refinement_nk=(16384,32768))
    rf.build_cell_likelihood(config,cell,"overlap",fid,FIDUCIAL,object())
    assert received == [(8192,101)]*3


def test_case_pair_selection_is_sorted_and_observation_specific():
    names = ["PFS-ELG", "DESI-QSO", "DESI-ELG"]
    auto_desi = [("DESI-ELG", "DESI-ELG"), ("DESI-QSO", "DESI-QSO")]
    desi_cross = [("DESI-ELG", "DESI-QSO")]
    pfs_auto = [("PFS-ELG", "PFS-ELG")]
    pfs_cross = [("DESI-ELG", "PFS-ELG"), ("DESI-QSO", "PFS-ELG")]
    assert retained_pairs("I", names) == auto_desi
    assert retained_pairs("A", names) == auto_desi
    assert retained_pairs("B", names) == auto_desi + desi_cross
    assert retained_pairs("D", names) == auto_desi + pfs_auto + desi_cross
    assert retained_pairs("C", names) == auto_desi + pfs_auto + desi_cross + pfs_cross


def test_invalid_configurations_are_rejected_before_survey_work():
    for kwargs in ({"case": "X"}, {"dk": 0.0}, {"kmax": 0.61}, {"n_mu": 2},
                   {"nuisance_factor": 0.0}, {"kmin": 0.2, "kmax": 0.1}):
        with pytest.raises(ValueError):
            RevisionConfig(**({"case": "B"} | kwargs))


def test_covariance_uses_each_physical_shell_width():
    k = np.array([.1, .2])
    widths = np.array([.01, .002])
    cov = multi_tracer_cov_general(
        ["X"], {("X", "X"): lambda k, mu: np.full((len(k), len(mu)), 3.)},
        {"X": 1.}, k, volume=100., dk=widths, n_mu=32,
    )
    modes = k**2 * widths * 100. / (2 * np.pi**2)
    np.testing.assert_allclose(cov[:, 0, 0], 32. / modes, rtol=1e-12)


def test_cell_selection_keeps_theoretical_cross_covariance_and_both_derivative_sides(monkeypatch):
    cell, fid = _fake_cell(monkeypatch)
    outputs = {case: build_cell_likelihood(
        RevisionConfig(case, kmax=.015), cell, "overlap", fid, FIDUCIAL, object())
        for case in ("I", "A", "B", "D", "C")}
    i, a, b, d, c = (outputs[x] for x in ("I", "A", "B", "D", "C"))
    assert [len(x.metadata["retained_pairs"]) for x in (i, a, b, d, c)] == [2, 2, 3, 4, 6]
    assert i.covariance.shape == (1, 6, 6)
    np.testing.assert_allclose(i.covariance[0, :3, 3:], 0.)
    assert np.any(a.covariance[0, :3, 3:] != 0.)
    assert np.linalg.norm(a.covariance[0, :3, :3] - i.covariance[0, :3, :3]) == 0
    assert b.metadata["theoretical_pairs"] == [
        ("DESI-ELG", "DESI-ELG"), ("DESI-LRG", "DESI-LRG"),
        ("DESI-ELG", "DESI-LRG")]
    cross = b.derivatives[0, 6:9]
    assert b.fiducial_mean.shape == (1, 9)
    np.testing.assert_allclose(b.fiducial_mean[0, :3],
                               outputs["A"].fiducial_mean[0, :3])
    assert np.all(np.isfinite(b.fiducial_mean))
    theta = np.array([FIDUCIAL[name] for name in ("ln10_10_As", "mnu", "Omega_m")])
    qa = np.asarray(fiducial_nuisance(fid["DESI-ELG"]))
    qb = np.asarray(fiducial_nuisance(fid["DESI-LRG"]))
    cosmo = np.dot(theta, [1., 2., 3.])
    np.testing.assert_allclose(b.fiducial_mean[0, 0], cosmo + qa[0] + 2*qa[1])
    np.testing.assert_allclose(b.fiducial_mean[0, 6],
                               cosmo + qa[0] + 2*qa[1] + 3*qb[0])
    assert np.all(cross[:, :3] != 0)
    qa = b.names.index("bin03:DESI-ELG:b1_sigma8")
    qb = b.names.index("bin03:DESI-LRG:b1_sigma8")
    qc = b.names.index("bin03:DESI-ELG|DESI-LRG:Pshot_cross")
    np.testing.assert_allclose(cross[0, [qa, qb, qc]], [1., 3., 4.])
    assert c.metadata["model_redshift"] == 1.1


def test_cell_fisher_has_absolute_mode_count_and_volume_scaling(monkeypatch):
    from dataclasses import replace

    cell, fid = _fake_cell(monkeypatch, ("DESI-ELG",))
    config = RevisionConfig("B", kmin=.01, kmax=.015)
    first = build_cell_likelihood(config, cell, "overlap", fid, FIDUCIAL, object())
    doubled = build_cell_likelihood(config, replace(cell, overlap_volume=200.),
                                    "overlap", fid, FIDUCIAL, object())
    np.testing.assert_allclose(doubled.F_like, 2*first.F_like, rtol=1e-12)
    modes = .0125**2 * .005 * 100. / (2*np.pi**2)
    # Ptot=5+1/n=5005; monopole covariance is 2*Ptot²/Nmodes.
    assert first.covariance[0, 0, 0] == pytest.approx(2*5005.**2/modes)


def test_unimplemented_cosmological_response_fails_instead_of_embedding_zeros(monkeypatch):
    import pfsfog.revision_forecast as rf

    class MissingCosmology(LinearPairResponse):
        def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu=32):
            return jnp.ones((len(k), 3)) * (q_a[0] + (q_b[0] if q_b is not None else 0.))

    cell, fid = _fake_cell(monkeypatch, ("DESI-ELG",))
    # _fake_cell sets a valid factory; replace it after building the fixture.
    monkeypatch.setattr(rf, "make_pair_response", lambda ps, z, reference, nbar_a,
                        nbar_b=None, k_nl=.7, **kwargs: MissingCosmology("A", "A", z))
    with pytest.raises(ValueError, match="cosmological response"):
        build_cell_likelihood(RevisionConfig("B", kmax=.015), cell,
                              "overlap", fid, FIDUCIAL, object())


def test_global_embedding_shares_nominal_nuisances_across_support_cells(monkeypatch):
    import pfsfog.revision_forecast as rf

    monkeypatch.setattr(rf, "make_pair_response", lambda ps, z, reference, nbar_a,
                        nbar_b=None, k_nl=.7, **kwargs: LinearPairResponse(
                            "A", "A" if nbar_b is None else "B", z))

    class Cosmo:
        params = FIDUCIAL

        def sigma8(self, z):
            return .6

    config = RevisionConfig("C", kmax=.015, nominal_bins=((1., 1.2),))
    result = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert len(result.cells) == 4  # two radial supports, two footprints
    assert len(result.bin_F_like) == 1
    np.testing.assert_allclose(result.F_like, result.bin_F_like[0])
    assert all(cell.metadata["model_redshift"] == 1.1 for cell in result.cells)
    names = result.names
    assert names[:3] == ["ln10_10_As", "Mnu", "Omegam"]
    assert names.count("bin00:DESI-ELG:b1_sigma8") == 1
    assert names.count("bin00:PFS-ELG:b1_sigma8") == 1
    assert not any("c1" in name for name in names)
    reconstructed = np.zeros_like(result.F_like)
    for cell in result.cells:
        index = [names.index(name) for name in cell.names]
        reconstructed[np.ix_(index, index)] += cell.F_like
        assert cell.validation["expected_shells"] == cell.validation["processed_shells"] == 1
    np.testing.assert_allclose(result.F_like, reconstructed)
    assert sum(r["role"] == "tracer" for r in result.parameters) == 44
    assert sum(r["role"] == "cross" for r in result.parameters) == 12
    assert len(names) == 59
    by_name = {r["name"]: r for r in result.parameters}
    assert by_name["bin00:DESI-ELG:c2"]["unit"] == "(Mpc/h)^2"
    assert by_name["bin00:DESI-ELG:c_tilde"]["unit"] == "(Mpc/h)^4"
    assert by_name["bin00:DESI-ELG:bGamma3"]["unit"] == ""


def test_empty_support_regions_keep_zero_records_without_changing_active_basis(monkeypatch):
    from dataclasses import replace
    import pfsfog.revision_forecast as rf

    measured, _ = _fake_cell(monkeypatch, ("DESI-ELG",))
    zero_volume = replace(measured, overlap_tracers=("DESI-LRG", "PFS-ELG"),
                          nonoverlap_tracers=("DESI-LRG",),
                          nbar={"DESI-LRG": 3e-4, "PFS-ELG": 4e-4},
                          overlap_volume=0., nonoverlap_volume=0.)
    pfs_only = replace(measured, overlap_tracers=("PFS-ELG",),
                       nonoverlap_tracers=(), nbar={"PFS-ELG": 4e-4})
    supports = [measured, zero_volume, pfs_only]
    monkeypatch.setattr(rf, "support_cells", lambda *args, **kwargs: supports)

    class Cosmo:
        params = FIDUCIAL

        def sigma8(self, z):
            return .6

    config = RevisionConfig("B", kmax=.015, nominal_bins=((1., 1.2),))
    result = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert len(result.cells) == 6
    assert [c.metadata["cell_id"] for c in result.cells] == [
        f"bin00-radial{i:02d}-{region}" for i in range(3)
        for region in ("overlap", "nonoverlap")]
    assert [c.validation.get("reason") for c in result.cells] == [
        None, None, "zero_volume", "zero_volume",
        "no_retained_population", "no_retained_population"]
    assert result.cells[2].metadata["candidate_populations"] == ("DESI-LRG", "PFS-ELG")
    for cell in result.cells[2:]:
        assert cell.names == []
        assert cell.F_like.shape == (0, 0)
        assert cell.derivatives.size == cell.covariance.size == 0
        assert cell.metadata["active_populations"] == ()
        assert cell.metadata["retained_observations"] == []
        assert cell.validation["expected_shells"] == cell.validation["processed_shells"] == 0
        assert cell.validation["field_min_scaled_eigenvalue"] is None
        assert cell.validation["covariance_min_scaled_eigenvalue"] is None
    assert result.names == result.cells[0].names
    np.testing.assert_allclose(result.F_like,
                               result.cells[0].F_like + result.cells[1].F_like)
    assert not any("DESI-LRG" in name or "PFS-ELG" in name for name in result.names)
    assert result.validation["cell_count"] == 6
    assert result.validation["expected_shells"] == result.validation["processed_shells"] == 2
    assert result.validation["min_field_scaled_eigenvalue"] > 0
    assert result.validation["min_covariance_scaled_eigenvalue"] > 0

    monkeypatch.setattr(rf, "support_cells", lambda *args, **kwargs: [measured])
    measured_only = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert result.names == measured_only.names
    assert result.parameters == measured_only.parameters
    np.testing.assert_allclose(result.F_like, measured_only.F_like)

    monkeypatch.setattr(rf, "support_cells", lambda *args, **kwargs: [zero_volume, pfs_only])
    all_empty = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert all_empty.names == ["ln10_10_As", "Mnu", "Omegam"]
    np.testing.assert_array_equal(all_empty.F_like, np.zeros((3, 3)))
    assert all_empty.validation["min_field_scaled_eigenvalue"] is None
    assert all_empty.validation["min_covariance_scaled_eigenvalue"] is None


def test_reference_domain_checks_actual_emulator_inputs_and_consistent_matter():
    reference = dict(FIDUCIAL)
    validate_reference_domain(reference, NOMINAL_BINS)
    for field, bad in (("omega_b", .031), ("omega_cdm", .151),
                       ("ln10_10_As", 3.51), ("n_s", 1.06),
                       ("h", .39), ("mnu", .501)):
        changed = dict(reference, **{field: bad})
        changed["Omega_m"] = (changed["omega_b"] + changed["omega_cdm"]
                              + changed["mnu"]/93.14)/changed["h"]**2
        with pytest.raises(ValueError, match="emulator domain"):
            validate_reference_domain(changed, NOMINAL_BINS)
    with pytest.raises(ValueError, match="inconsistent Omega_m"):
        validate_reference_domain(dict(reference, Omega_m=.4), NOMINAL_BINS)


def test_once_only_priors_q13_full_cosmo_inverse_and_growth_rank(monkeypatch):
    import pfsfog.revision_forecast as rf

    monkeypatch.setattr(rf, "make_growth_rate_func", lambda: (
        lambda z, cosmo: cosmo["ln10_10_As"] + z*cosmo["mnu"] + z*z*cosmo["Omega_m"]))
    monkeypatch.setattr(rf, "make_sigma8_func", lambda: lambda z, cosmo: 1.)
    config = RevisionConfig("B")
    names = ["ln10_10_As", "Mnu", "Omegam", "bin00:DESI-ELG:b1_sigma8"]
    f = np.array([[4., 0., 0., 1.], [0., 9., 0., 1.],
                  [0., 0., 16., 1.], [1., 1., 1., 2.]])
    parameters = [{"name": name, "role": "cosmo" if i < 3 else "tracer",
                   "coordinate": name.split(":")[-1], "fiducial": 0.,
                   "prior_sigma": None, "nominal_bin": None if i < 3 else 0,
                   "tracers": [] if i < 3 else ["DESI-ELG"], "unit": ""}
                  for i, name in enumerate(names)]
    grid = make_shell_grid(.01, .02, .005)
    like = RevisionLikelihood(config, f, names, np.array([3., .06, .31, .9]),
                              parameters, [], [f], grid, FIDUCIAL, {}, {})
    result = apply_revision_priors(like, 1.)
    assert result.F_prior_cosmo[0, 0] == 0
    np.testing.assert_allclose(np.diag(result.F_prior_cosmo), [0., 1/25., 1., 0.])
    np.testing.assert_allclose(result.F_prior_nuisance, 0.)
    np.testing.assert_allclose(result.cov_cosmo_nuisance_fixed,
                               np.diag([1/4., 1/9.04, 1/17.]), rtol=1e-12)
    np.testing.assert_allclose(result.cov_cosmo_marg, np.linalg.inv(result.F_total)[:3, :3])
    assert result.cov_cosmo_marg[0, 0] > result.cov_cosmo_nuisance_fixed[0, 0]
    z = np.array([(a+b)/2 for a, b in config.nominal_bins])
    np.testing.assert_allclose(result.growth_jacobian, np.stack([np.ones(8), z, z*z], axis=1))
    assert np.linalg.matrix_rank(result.growth_cov_marg, tol=1e-10) == 3


def test_nuisance_widths_scale_once_without_c1_or_unmeasured_cross(monkeypatch):
    import pfsfog.revision_forecast as rf

    monkeypatch.setattr(rf, "make_growth_rate_func", lambda: lambda z, cosmo: cosmo["ln10_10_As"])
    monkeypatch.setattr(rf, "make_sigma8_func", lambda: lambda z, cosmo: 1.)
    config = RevisionConfig("B", nominal_bins=((1., 1.2),))
    names = ["ln10_10_As", "Mnu", "Omegam", "bin00:DESI-ELG:c2",
             "bin00:DESI-ELG|DESI-LRG:Pshot_cross"]
    records = [{"name": name, "role": role, "prior_sigma": width}
               for name, role, width in zip(names,
                   ["cosmo"]*3 + ["tracer", "cross"], [None, 5., 1., 30., 1.])]
    f = np.eye(5)
    like = RevisionLikelihood(config, f, names, np.array([3., .06, .31, 30., 0.]),
                              records, [], [f], make_shell_grid(.01, .02, .005), FIDUCIAL, {}, {})
    result = apply_revision_priors(like, 3.)
    assert result.nuisance_factor == 3.
    np.testing.assert_allclose(np.diag(result.F_prior_nuisance), [0., 0., 0., 1/90.**2, 1/9.])
    assert result.parameters[-2]["prior_sigma"] == 90.
    assert result.parameters[-1]["prior_sigma"] == 3.
    assert records[-2]["prior_sigma"] == 30.  # no mutation of baseline


def test_assembled_model_records_executed_backend_and_fixed_likelihood_flags(monkeypatch):
    import pfsfog.revision_forecast as rf

    cell, _ = _fake_cell(monkeypatch)
    monkeypatch.setattr(rf, "support_cells", lambda *args, **kwargs: [cell])

    class Cosmo:
        params = FIDUCIAL

        def sigma8(self, z):
            return .6

    class Backend:
        do_irres = False
        subtract_k0_limit = True
        _kmin = 1e-5
        _kmax = 1e3
        _nfft = 256

    result = build_revision_likelihood(
        RevisionConfig("B", kmax=.015, nominal_bins=((1., 1.2),)),
        default_survey_group(), Cosmo(), Backend())
    assert result.model["subtract_k0_limit"] is True
    assert result.model["fft_kmin"] == 1e-5
    assert result.model["fft_kmax"] == 1e3
    assert result.model["fft_n"] == 256
    assert result.model["fixed_observed_geometry"] is True
    assert result.model["fixed_covariance"] is True
    assert result.model["covariance_derivative_trace"] is False
