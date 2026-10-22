"""ELF selection, the actual cosmological response, and source authentication."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


def test_revision_selects_elf_and_records_actual_ap_method():
    from scripts import run_revision as run
    from pfsfog.elf_adapter import ELFPowerSpectrum
    engine = run._load_backend()
    assert isinstance(engine, ELFPowerSpectrum)
    provenance = run.collect_provenance(engine)
    assert provenance["backend_path"] == str(Path("/Users/nguyenmn/elf/src/elf/__init__.py"))
    assert any("pt_coeff/" in path for path in provenance["backend_file_hashes"])
    assert provenance["background_file_hashes"]
    flags = provenance["model_flags"]
    assert flags["ap_method"] == "elf-full-mean-cubic-logk-v1"
    assert flags["ap_interpolation_nk"] is None
    assert flags["fft_n"] == engine._nfft
    from pfsfog.revision_artifacts import _configuration
    config = _configuration(run.inventory()[0], 1., model_flags=flags)
    assert config.get("ap_interpolation_nk") is None
    assert config.get("ap_refinement_nk") is None


def test_changed_elf_coefficients_are_rejected(monkeypatch):
    from pfsfog.one_loop import elf_source_identity
    original = Path.read_bytes
    def changed(path, *args, **kwargs):
        if "/elf/src/elf/pt_coeff/" in str(path):
            return b"changed physical coefficients"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="ELF source"):
        elf_source_identity()


def test_foreign_elf_import_is_rejected(monkeypatch, tmp_path):
    import elf
    from pfsfog.one_loop import elf_source_identity
    monkeypatch.setattr(elf, "__file__", str(tmp_path/"elf/__init__.py"))
    with pytest.raises(ValueError, match="ELF.*origin"):
        elf_source_identity()


def test_pair_response_uses_complete_elf_mean_and_explicit_cross_noise():
    from pfsfog.one_loop import make_one_loop_engine
    from pfsfog.cosmo import FIDUCIAL, FiducialCosmology
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.eft_params import desi_elg_fiducials
    from pfsfog.revision_theory import (make_pair_response, fiducial_nuisance,
                                        nuisance_to_backend)
    engine = make_one_loop_engine()
    cosmo = FiducialCosmology()
    z = 1.1
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    theta = jnp.array([FIDUCIAL[n] for n in ("ln10_10_As", "mnu", "Omega_m", "h", "omega_b", "n_s")])
    qa = fiducial_nuisance(desi_elg_fiducials(b1=1.8, sigma8_z=cosmo.sigma8(z)))
    qb = fiducial_nuisance(desi_elg_fiducials(b1=1.3, sigma8_z=cosmo.sigma8(z)))
    n_a, n_b = 3.e-4, 7.e-4
    response = make_pair_response(engine, z, FIDUCIAL, n_a, n_b, space=space)
    assert response.ap_backend is engine
    k, mu = jnp.array([.03, .1, .19]), jnp.array([-.8, 0., .8])
    p = response._cosmology(theta)
    pa = nuisance_to_backend(qa, response.sigma8(z, p), response.sigma8_ref,
                             response.growth_rate(z, p), p["h"], n_a, .7)
    pb = nuisance_to_backend(qb, response.sigma8(z, p), response.sigma8_ref,
                             response.growth_rate(z, p), p["h"], n_b, .7)
    direct = engine.get_pkmu_pair(k, mu, response.plin(z, p),
                                  {**pa, "bias2": pb["bias"], "ctr2": pb["ctr"]})
    actual = response.pkmu(k, mu, theta, qa, qb, jnp.array([.4, -.3]))
    expected = direct + (.4-.3*(k[:, None]*mu[None, :]/.7)**2)/np.sqrt(n_a*n_b)
    np.testing.assert_allclose(actual, expected, rtol=2.e-11, atol=2.e-8)
