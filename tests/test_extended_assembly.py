"""Six distinct analytic responses exercise real observed cell assembly."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.eft_params import tracer_fiducials
from pfsfog.revision_forecast import RevisionConfig, build_cell_likelihood, build_revision_likelihood
from pfsfog.revision_theory import NUISANCE_NAMES
from pfsfog.survey_geometry import SupportCell
from pfsfog.surveys import default_survey_group

NAMES = make_cosmology_spec(FIDUCIAL, extended=True).names
POPULATIONS = ("DESI-ELG", "DESI-LRG", "PFS-ELG")


class SixResponse:
    def __init__(self, missing=None):
        self.missing = missing

    def pkmu(self, k, mu, theta, q_a, q_b=None, q_cross=None):
        return jnp.full((len(k), len(mu)), 5. if q_b is None else 1.)

    def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, n_mu=32):
        directions = jnp.stack([jnp.asarray(k)**(i//3)
                                * jnp.array([1., 2., 3.])[i%3] for i in range(6)], axis=-1)
        # Independent ell and shell directions; every individual response is nonzero.
        ell = jnp.array([[1., 1., 1., 1., 1., 1.],
                         [1., 2., 3., 1., 2., 3.],
                         [1., 4., 9., 1., 4., 9.]])
        if self.missing is not None:
            theta = theta.at[self.missing].set(0.)
        signal = jnp.einsum("ki,li,i->kl", directions, ell, theta)
        nuisance = q_a[0] + 2*q_a[1]
        if q_b is not None:
            nuisance += 3*q_b[0] + 4*q_cross[0] + 5*q_cross[1]
        return signal + nuisance*jnp.array([1., 2., 3.])


def setup(monkeypatch, missing=None):
    import pfsfog.revision_forecast as rf
    calls = []
    def factory(ps, z, reference, nbar_a, nbar_b=None, k_nl=.7, **kwargs):
        calls.append((z, nbar_a, nbar_b, kwargs))
        assert kwargs["space"].names == NAMES
        return SixResponse(missing)
    monkeypatch.setattr(rf, "make_pair_response", factory)
    densities = dict(zip(POPULATIONS, (2e-4, 3e-4, 4e-4)))
    cell = SupportCell((1., 1.2), (1., 1.1), 1.1, POPULATIONS, POPULATIONS[:2],
                       densities, 100., 200.)
    fid = {n: tracer_fiducials(n, 1.5, .6) for n in POPULATIONS}
    config = RevisionConfig("C", cosmology_basis="extended6", kmax=.02)
    return config, cell, fid, calls


def test_basis_is_explicit_and_only_known_layouts_are_accepted():
    assert RevisionConfig("B").cosmology_basis == "baseline3"
    with pytest.raises(ValueError, match="cosmology"):
        RevisionConfig("B", cosmology_basis="anything")


def test_six_columns_autos_crosses_offsets_and_tracer_reordering(monkeypatch):
    config, cell, fid, calls = setup(monkeypatch)
    result = build_cell_likelihood(config, cell, "overlap", fid, FIDUCIAL, object())
    assert result.names[:6] == list(NAMES)
    assert result.derivatives.shape == (2, 18, 45)
    k = result.metadata["retained_pairs"]
    for ipair, pair in enumerate(k):
        block = result.derivatives[:, 3*ipair:3*ipair+3]
        assert np.linalg.matrix_rank(block[:, :, :6].reshape(6, 6)) == 6
        for i in range(6):
            expected = np.array([.0125, .0175])[:, None]**(i//3)
            expected = expected * np.array([1., 2., 3.])[i%3]
            expected = expected * np.array([1., i%3+1., (i%3+1.)**2])[None, :]
            np.testing.assert_allclose(block[:, :, i], expected, rtol=1e-12)
        a = result.names.index(f"bin03:{pair[0]}:b1_sigma8")
        np.testing.assert_allclose(block[0, :, a], [1., 2., 3.])
        if pair[0] != pair[1]:
            b = result.names.index(f"bin03:{pair[1]}:b1_sigma8")
            cross = result.names.index(f"bin03:{pair[0]}|{pair[1]}:Pshot_cross")
            np.testing.assert_allclose(block[0, :, [a, b, cross]].T,
                                       np.array([1., 2., 3.])[:, None]*[1., 3., 4.])
    reordered = replace(cell, overlap_tracers=tuple(reversed(POPULATIONS)))
    second = build_cell_likelihood(config, reordered, "overlap", fid, FIDUCIAL, object())
    assert result.names == second.names
    np.testing.assert_array_equal(result.F_like, second.F_like)
    assert len(calls) == 12


@pytest.mark.parametrize("column", [3, 4, 5])
def test_omitted_extra_response_is_rejected(monkeypatch, column):
    config, cell, fid, _ = setup(monkeypatch, column)
    with pytest.raises(ValueError, match="cosmological response"):
        build_cell_likelihood(config, cell, "overlap", fid, FIDUCIAL, object())


def test_unmeasured_cross_owns_no_active_nuisance_and_observed_quantities_are_fixed(monkeypatch):
    config, cell, fid, _ = setup(monkeypatch)
    config = replace(config, case="D")
    result = build_cell_likelihood(config, cell, "overlap", fid, FIDUCIAL, object())
    assert len(result.metadata["theoretical_pairs"]) == 6
    assert len(result.metadata["retained_pairs"]) == 4
    assert not any("PFS-ELG|" in n or ("|PFS-ELG" in n) for n in result.names)
    varied = dict(FIDUCIAL, h=FIDUCIAL["h"]+.001, omega_b=FIDUCIAL["omega_b"]+.0001,
                  n_s=FIDUCIAL["n_s"]+.001)
    second = build_cell_likelihood(config, cell, "overlap", fid, varied, object())
    for key in ("volume", "nbar", "model_redshift", "radial_bin", "retained_observations"):
        assert result.metadata[key] == second.metadata[key]
    np.testing.assert_array_equal(result.covariance, second.covariance)
    assert not np.array_equal(result.fiducial_mean, second.fiducial_mean)


def test_global_cells_sum_named_fisher_without_regional_priors(monkeypatch):
    import pfsfog.revision_forecast as rf
    config, cell, _, _ = setup(monkeypatch)
    config = replace(config, nominal_bins=((1., 1.2),))
    monkeypatch.setattr(rf, "support_cells", lambda *args, **kwargs: [cell])
    class Cosmo:
        params = FIDUCIAL
        def sigma8(self, z):
            return .6
    result = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert result.cosmo_names == NAMES
    assert len(result.names) == 6 + 3*len(NUISANCE_NAMES) + 3*2
    expected = np.zeros_like(result.F_like)
    for c in result.cells:
        index = [result.names.index(n) for n in c.names]
        expected[np.ix_(index, index)] += c.F_like
    np.testing.assert_allclose(result.F_like, expected, rtol=1e-12, atol=0.)
    for i in (1, 2):
        assert result.F_like[i, i] == expected[i, i]
    original = rf.build_cell_likelihood
    def reordered(*args, **kwargs):
        c = original(*args, **kwargs)
        order = list(reversed(range(len(c.names))))
        return replace(c, names=[c.names[i] for i in order],
                       F_like=c.F_like[np.ix_(order, order)], derivatives=c.derivatives[:, :, order])
    monkeypatch.setattr(rf, "build_cell_likelihood", reordered)
    other = build_revision_likelihood(config, default_survey_group(), Cosmo(), object())
    assert other.names == result.names
    np.testing.assert_allclose(other.F_like, result.F_like, rtol=1e-12, atol=0.)
