"""Independent matrix, Gaussian-normalization and displaced-point checks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_batched_spd_agrees_with_original_scalar_diagnostics():
    from pfsfog.elf_covariance_audit import audit_spd_batch
    from pfsfog.revision_validation_physical import _audited_spd
    rng = np.random.default_rng(917)
    a = rng.normal(size=(2, 3, 4, 4))
    matrices = np.einsum("...ik,...jk->...ij", a, a)+np.eye(4)
    actual = audit_spd_batch(matrices, "test")
    reference = [_audited_spd(m, "test") for m in matrices.reshape(-1, 4, 4)]
    np.testing.assert_allclose(actual["min_scaled_eigenvalue"],
                               min(x["min_scaled_eigenvalue"] for x in reference), rtol=1.e-13)
    np.testing.assert_allclose(actual["max_scaled_condition"],
                               max(x["scaled_condition"] for x in reference), rtol=1.e-13)
    assert actual["matrix_count"] == 6
    assert actual["max_scaled_residual"] <= 1.e-10


@pytest.mark.parametrize("kind", ["negative", "asymmetric", "nonfinite"])
def test_rejected_matrix_is_retained_at_its_actual_index(kind):
    from pfsfog.elf_covariance_audit import audit_spd_batch, BatchCovarianceFailure
    matrices = np.broadcast_to(np.eye(2), (3, 4, 2, 2)).copy()
    if kind == "negative":
        matrices[2, 1] = [[1., 2.], [2., 1.]]
    elif kind == "asymmetric":
        matrices[2, 1, 0, 1] = .1
    else:
        matrices[2, 1, 1, 1] = np.nan
    before = matrices.copy()
    with pytest.raises(BatchCovarianceFailure) as caught:
        audit_spd_batch(matrices, "field cell=test")
    assert caught.value.index == (2, 1)
    np.testing.assert_array_equal(caught.value.matrix, before[2, 1])
    np.testing.assert_array_equal(matrices, before)


def test_roundoff_singular_cholesky_failure_retains_original_matrix():
    from pfsfog.elf_covariance_audit import audit_spd_batch, BatchCovarianceFailure
    rng = np.random.default_rng(901)
    for _ in range(10):
        a = rng.normal(size=(4, 3))
        singular = a @ a.T
    matrices = np.broadcast_to(np.eye(4), (3, 4, 4, 4)).copy()
    matrices[2, 1] = singular
    with pytest.raises(BatchCovarianceFailure) as caught:
        audit_spd_batch(matrices, "roundoff-singular field")
    assert caught.value.index == (2, 1)
    np.testing.assert_array_equal(caught.value.matrix, singular)


def test_gaussian_cross_covariance_counts_noise_and_shell_weight_once():
    from pfsfog.elf_covariance_audit import covariance_from_spectra
    k, dk, volume = np.array([.05, .1]), np.array([.01, .02]), 1000.
    pairs = [("A", "A"), ("B", "B"), ("A", "B")]
    metadata = {"active_populations": ("A", "B"), "theoretical_pairs": pairs,
                "retained_pairs": pairs, "nbar": {"A": .1, "B": .2}, "volume": volume}
    signals = np.broadcast_to(np.array([3., 4., 1.])[:, None, None], (3, 2, 32)).copy()
    field, covariance = covariance_from_spectra(signals, metadata, k, dk, n_mu=32)
    np.testing.assert_array_equal(field[0, 0], [[13., 1.], [1., 9.]])
    gaussian = np.array([[2*13**2, 2., 2*13], [2., 2*9**2, 2*9],
                         [2*13, 2*9, 13*9+1.]])
    modes = k**2*dk*volume/(2*np.pi**2)
    expected = np.zeros_like(covariance)
    for ell_index, ell in enumerate((0, 2, 4)):
        positions = np.arange(3)*3+ell_index
        for ik in range(2):
            expected[ik][np.ix_(positions, positions)] = (2*ell+1)*gaussian/modes[ik]
    np.testing.assert_allclose(covariance, expected, rtol=2.e-13, atol=2.e-8)
    _, twice_width = covariance_from_spectra(signals, metadata, k, 2*dk, n_mu=32)
    np.testing.assert_array_equal(twice_width, covariance/2)


def fixture_cell(cross=False):
    from pfsfog.cosmo import FIDUCIAL
    from pfsfog.eft_params import CROSS_STOCH_NAMES
    from pfsfog.revision_theory import NUISANCE_NAMES
    from scripts.validate_elf_engine import elf_cosmology_spec
    from pfsfog.revision_forecast import RevisionConfig
    space = elf_cosmology_spec(FIDUCIAL)
    values = {n: FIDUCIAL[{"Mnu": "mnu", "Omegam": "Omega_m"}.get(n, n)] for n in space.names}
    tracers = ("DESI-LRG", "DESI-QSO") if cross else ("DESI-LRG",)
    for tracer in tracers:
        values.update({f"bin00:{tracer}:{n}": (1. if i == 0 else 0.)
                       for i, n in enumerate(NUISANCE_NAMES)})
    pairs = [(t, t) for t in tracers]
    if cross:
        pairs += [tracers]
        values.update({f"bin00:{tracers[0]}|{tracers[1]}:{n}": 0. for n in CROSS_STOCH_NAMES})
    metadata = {"cell_id": "toy", "nominal_bin_index": 0, "model_redshift": .5,
                "active_populations": tracers, "theoretical_pairs": pairs,
                "retained_pairs": pairs, "nbar": {t: .1 for t in tracers}, "volume": 1000.}
    return SimpleNamespace(names=list(values), metadata=metadata), values, RevisionConfig(
        "C" if cross else "B", cosmology_basis="extended6", kmax=.03, dk=.01, n_mu=8)


def test_all_signed_neighborhood_points_and_cross_coordinates_are_saved(tmp_path, monkeypatch):
    from pfsfog import elf_covariance_audit as audit
    from pfsfog.cosmo import FIDUCIAL
    def response(*args, **kwargs):
        return SimpleNamespace(pkmu=lambda k, mu, theta, qa, qb=None, qc=None:
            np.full((len(k), len(mu)), 2. if qc is None else 1.+qc[0]))
    monkeypatch.setattr(audit, "make_pair_response", response)
    cell, values, config = fixture_cell(cross=True)
    report = audit.audit_neighborhood(cell, config, FIDUCIAL, None, values, tmp_path)
    expected = 1+6*len(values)
    assert report["status"] == "passed"
    assert report["processed_points"] == report["expected_points"] == expected
    assert len(report["points"]) == expected
    data = np.load(tmp_path/"neighborhood_spectra.npy", mmap_mode="r")
    assert np.isfinite(data).all()
    index = next(i for i, p in enumerate(report["points"])
                 if str(p["column"]).endswith(":Pshot_cross") and p["sign"] == 1)
    np.testing.assert_allclose(data[index, -1]-data[0, -1], report["points"][index]["step"],
                               rtol=1.e-12, atol=1.e-15)


def test_failure_away_from_fiducial_preserves_matrix_and_point(tmp_path, monkeypatch):
    from pfsfog import elf_covariance_audit as audit
    from pfsfog.cosmo import FIDUCIAL
    def response(*args, **kwargs):
        return SimpleNamespace(pkmu=lambda k, mu, theta, qa, qb=None, qc=None:
            np.full((len(k), len(mu)), -12. if qa[0] > 1. else 2.))
    monkeypatch.setattr(audit, "make_pair_response", response)
    cell, values, config = fixture_cell()
    report = audit.audit_neighborhood(cell, config, FIDUCIAL, None, values, tmp_path)
    assert report["status"] == "failed"
    assert 1 < report["processed_points"] < report["expected_points"]
    failed = report["points"][-1]
    assert failed["sign"] == 1 and failed["column"] == cell.names[6]
    np.testing.assert_array_equal(np.load(tmp_path/"failed_matrix.npy"), [[-2.]])
    assert json.loads((tmp_path/"neighborhood.json").read_text())["status"] == "failed"
