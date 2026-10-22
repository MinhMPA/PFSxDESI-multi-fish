"""Exercise production cell dispatch/embedding with an analytic toy likelihood."""
from dataclasses import replace

import numpy as np
import pytest

import pfsfog.fisher_joint as joint
from pfsfog.config import ForecastConfig
from pfsfog.eft_params import COSMO_NAMES
from pfsfog.surveys import default_survey_group


@pytest.fixture
def toy_builder(monkeypatch):
    calls = []

    def build(zbin, active, volume, cosmo, ps, cfg, cross_shot=None,
              marginalize_cross_stoch=None, *, density_bin=None, fiducial_tracers=None):
        assert density_bin is not None
        assert fiducial_tracers is not None
        calls.append((zbin, tuple(sorted(active)), volume, density_bin, fiducial_tracers))
        names = list(COSMO_NAMES) + [f"nuis_{n}_{zbin}" for n in sorted(active)]
        F = np.zeros((len(names), len(names)))
        for i, name in enumerate(sorted(active), len(COSMO_NAMES)):
            F[i, i] = volume
        F[0, 0] = volume * len(active)
        return F, names

    monkeypatch.setattr(joint, "build_zbin_fisher", build)
    return calls


def test_partition_uses_supported_volume_and_shared_nominal_nuisance(toy_builder):
    sg, cfg, zb = default_survey_group(), ForecastConfig(), (1., 1.2)
    F, names = joint.volume_partitioned_zbin_fisher(zb, sg, None, None, cfg)
    assert len(toy_builder) == 4
    assert {c[3] for c in toy_builder} == {(1., 1.1), (1.1, 1.2)}
    assert all(c[0] == zb for c in toy_builder)
    for tracer in sg.desi_tracers:
        idx = names.index(f"nuis_{tracer}_{zb}")
        expected = sg.desi_tracers[tracer].volume(*zb)
        np.testing.assert_allclose(F[idx, idx], expected, rtol=1e-12)
    # Exactly one coordinate per population/nominal bin, despite four cells.
    assert len(names) == len(COSMO_NAMES) + 4


def test_elg_selection_dispatch_preserves_pfs_reference(toy_builder):
    sg, cfg = default_survey_group(), ForecastConfig()
    cfg.exclude_elg1 = True
    joint.volume_partitioned_zbin_fisher((.8, 1.), sg, None, None, cfg)
    for _, active, _, _, reference in toy_builder:
        assert "DESI-ELG" not in active
        assert reference["DESI-ELG"] is sg.desi_tracers["DESI-ELG"]


def test_baseline_reference_does_not_invent_an_unsupported_elg_population(toy_builder):
    sg, cfg = default_survey_group(), ForecastConfig()
    joint.volume_partitioned_zbin_fisher((.6, .8), sg, None, None, cfg)
    # PFS's inherited b1_ref=1.3 fallback applies where DESI-ELG is absent
    # even before selection. Freezing fiducials must not extrapolate ELG here.
    assert all("DESI-ELG" not in reference for *_, reference in toy_builder)


def test_broad_comparator_uses_the_same_support(toy_builder):
    sg, cfg, zb = default_survey_group(), ForecastConfig(), (1., 1.2)
    # Per-bin helper does not invert this deliberately rank-deficient toy or add priors.
    F, names = joint.independent_autos_zbin_fisher(zb, sg, None, None, cfg)
    assert all(len(c[1]) == 1 for c in toy_builder)
    for tracer, survey in sg.desi_tracers.items():
        idx = names.index(f"nuis_{tracer}_{zb}")
        np.testing.assert_allclose(F[idx, idx], survey.volume(*zb), rtol=1e-12)
    assert len(names) == len(COSMO_NAMES) + 3


def test_baseline_fiducials_are_independent_of_selection():
    sg, cfg = default_survey_group(), ForecastConfig()
    # Simple biases avoid loading an emulator in a pure selection test.
    reference = {name: replace(s, b1_of_z=lambda z: 2.) for name, s in sg.all_surveys.items()}
    selected = {"PFS-ELG": reference["PFS-ELG"]}
    fids_all, pars_all, _ = joint._prepare_tracer_parameters(
        reference, (1., 1.2), (1., 1.1), reference, .5, .8, .67, cfg,
    )
    fids_cut, pars_cut, nbars = joint._prepare_tracer_parameters(
        selected, (1., 1.2), (1., 1.1), reference, .5, .8, .67, cfg,
    )
    assert fids_cut["PFS-ELG"] == fids_all["PFS-ELG"]
    assert pars_cut["PFS-ELG"] == pars_all["PFS-ELG"]
    assert nbars["PFS-ELG"] == reference["PFS-ELG"].nbar_eff(1, 1.1)
    cfg.k_nl = .8
    _, pars, _ = joint._prepare_tracer_parameters(selected, (1., 1.2), (1., 1.1), reference, .5, .8, .67, cfg)
    assert pars["PFS-ELG"]["k_nl"] == .8
