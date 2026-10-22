"""Independent Fourier mode-count normalization, shared by every assembler."""

import numpy as np
import pytest

from pfsfog.covariance_mt_general import multi_tracer_cov_general
from pfsfog.eft_params import COSMO_NAMES, NUISANCE_NAMES
from pfsfog.fisher import fisher_matrix
from pfsfog.fisher_full_area import full_area_fisher_per_zbin
from pfsfog.fisher_joint import _assemble_fisher_with_cosmo
from pfsfog.fisher_mt import multi_tracer_fisher, multi_tracer_fisher_asymmetric
from pfsfog.fisher_mt_general import multi_tracer_fisher_general


POWER, NBAR, VOLUME = 1000.0, 0.001, 1e9
NC, NN = len(COSMO_NAMES), len(NUISANCE_NAMES)


def shell_fixture(dk=0.005, volume=VOLUME):
    edges = np.linspace(0.01, 0.20, round(0.19 / dk) + 1)
    k = (edges[1:] + edges[:-1]) / 2

    def signal(k, mu):
        return np.full((len(k), len(mu)), POWER)

    cov = multi_tracer_cov_general(
        ["X"], {("X", "X"): signal}, {"X": NBAR}, k, volume, dk, ells=(0,),
    )
    nmodes = k**2 * dk * volume / (2 * np.pi**2)
    expected = np.sum(nmodes / 2 * (POWER / (POWER + 1 / NBAR))**2)
    return k, cov, expected


def joint_response(k, cov, tracer_names=("X",), response=(POWER,), pair_kmax=None, dk=0.005):
    autos = {}
    for name, amplitude in zip(tracer_names, response):
        cosmo = np.zeros((NC, 1, len(k)))
        cosmo[0, 0] = amplitude
        autos[name] = (np.zeros((NN, 1, len(k))), cosmo)
    pairs = [(name, name) for name in tracer_names]
    pairs += [(a, b) for i, a in enumerate(tracer_names) for b in tracer_names[i+1:]]
    return _assemble_fisher_with_cosmo(
        tracer_names, pairs, autos, {}, cov, k, dk, (0.8, 1.0), (0,),
        pair_kmax=pair_kmax,
    )[0, 0]


def assemble(kind, dk=0.005, volume=VOLUME):
    k, cov, expected = shell_fixture(dk, volume)
    derivative = {NUISANCE_NAMES[0]: {0: np.full(len(k), POWER)}}
    if kind == "single":
        F = fisher_matrix(derivative, 1 / cov, k, volume, dk, [NUISANCE_NAMES[0]], (0,))
        actual = F[0, 0]
    elif kind == "joint":
        actual = joint_response(k, cov, dk=dk)
    elif kind == "full_area":
        actual = full_area_fisher_per_zbin(
            derivative, cov, k, dk, np.zeros(NN), (0.8, 1.0), 0.2, (0,),
        ).F[NC, NC]
    elif kind == "general":
        actual = multi_tracer_fisher_general(
            ["X"], {"X": derivative}, {}, cov, k, dk, (0.8, 1.0), (0,),
        ).F[NC, NC]
    else:
        expanded = np.broadcast_to(np.eye(3), (len(k), 3, 3)).copy()
        expanded[:, 0, 0] = cov[:, 0, 0]
        if kind == "symmetric":
            result = multi_tracer_fisher(derivative, {}, {}, {}, expanded, k, dk, (0.8, 1.0), (0,))
        else:
            split = len(k) // 2
            high_cov = expanded[split:][:, [0, 2]][:, :, [0, 2]]
            result = multi_tracer_fisher_asymmetric(
                derivative, {}, {}, {}, expanded[:split], high_cov,
                k[:split], k[split:], dk, (0.8, 1.0), (0,),
            )
        actual = result.F[NC, NC]
    return actual, expected


@pytest.mark.parametrize("kind", ["single", "joint", "general", "symmetric", "asymmetric", "full_area"])
def test_independent_mode_count(kind):
    actual, expected = assemble(kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("kind", ["single", "joint", "general", "symmetric", "asymmetric", "full_area"])
def test_fixed_edges_refinement_and_volume(kind):
    coarse = assemble(kind)[0]
    fine = assemble(kind, dk=0.0025)[0]
    doubled = assemble(kind, volume=2 * VOLUME)[0]
    np.testing.assert_allclose(fine / coarse, 1.0, rtol=2e-4)
    np.testing.assert_allclose(doubled, 2 * coarse, rtol=1e-12)


def test_invalid_retained_covariance_raises_with_shell_context():
    k, cov, _ = shell_fixture()
    cov[0, 0, 0] = -1
    with pytest.raises(ValueError, match="shell 0"):
        joint_response(k, cov)


def test_joint_selects_covariance_before_inversion():
    cov = np.array([[[2, 1, 0.5], [1, 3, 0.4], [0.5, 0.4, 2]]], dtype=float)
    response = np.array([1, 2])
    actual = joint_response(np.array([0.1]), cov, ("X", "Y"), response, [0.2, 0.2, 0])
    expected = response @ np.linalg.solve(cov[0, :2, :2], response)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("cov", [
    [[1, 2], [2, 1]],  # positive diagonals are insufficient
    [[1, 0.2], [0.3, 1]],
    [[1, 0], [0, np.nan]],
])
def test_invalid_covariance_is_not_silently_skipped(cov):
    from pfsfog.fisher_core import fisher_from_shells
    with pytest.raises(ValueError, match="shell 0 at k=0.1"):
        fisher_from_shells(np.ones((1, 2, 1)), np.array([cov]), np.array([0.1]))


def test_masked_invalid_values_and_empty_shell_do_not_affect_information():
    from pfsfog.fisher_core import fisher_from_shells
    D = np.array([[[2.0], [np.nan]], [[np.nan], [np.nan]]])
    C = np.full((2, 2, 2), np.nan)
    C[0, 0, 0] = 4.0
    F = fisher_from_shells(D, C, [0.1, 0.2], active=np.array([[True, False], [False, False]]))
    np.testing.assert_allclose(F, [[1.0]], rtol=1e-15)


def test_information_invariant_to_observable_units_and_order():
    from pfsfog.fisher_core import fisher_from_shells
    D = np.array([[[1.0, 2.0], [3.0, -1.0]]])
    C = np.array([[[2.0, 1.0], [1.0, 3.0]]])
    expected = D[0].T @ np.linalg.solve(C[0], D[0])
    scales = np.array([1e-8, 1e8])
    scaled_C = C * scales[None, :, None] * scales[None, None, :]
    scaled_D = D * scales[None, :, None]
    actual = fisher_from_shells(scaled_D[:, ::-1], scaled_C[:, ::-1, ::-1], [0.1])
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
def test_single_tracer_convenience_rejects_negative_variance():
    from pfsfog.fisher import single_tracer_fisher

    with pytest.raises(ValueError, match="[Cc]ovariance|positive"):
        single_tracer_fisher(
            {"amplitude": {0: np.ones(1)}}, -np.ones((1, 1, 1)),
            np.array([0.1]), 0.01, ["amplitude"], (0.8, 1.0),
            "toy", 0.2, ells=(0,),
        )


def test_finite_dense_shell_gram_under_strict_floating_point_checks():
    """Finite whitened data must work with genuine overflow detection enabled.

    This shape triggers false Accelerate status flags on the affected Mac;
    the expected information is also computed without BLAS independently.
    """
    from pfsfog.fisher_core import fisher_from_shells

    d = np.random.default_rng(19).normal(size=(18, 45))
    expected = np.array([
        [sum(d[k, i] * d[k, j] for k in range(18)) for j in range(45)]
        for i in range(45)
    ])
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        actual = fisher_from_shells(d[None], np.eye(18)[None], np.array([0.1]))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
