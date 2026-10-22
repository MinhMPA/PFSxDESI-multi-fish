"""ELF evidence must measure the new physics path, never an inactive AP table."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("entry", ["extended", "ap_cell", "intrinsic", "baseline_cli"])
def test_old_table_qualification_rejects_elf_before_calculation(entry):
    from pfsfog.one_loop import make_one_loop_engine
    engine = make_one_loop_engine()
    with pytest.raises(ValueError, match="ELF.*validate_elf_engine.py"):
        if entry == "extended":
            from scripts.run_extended_cosmology import collect_provenance
            collect_provenance(engine)
        elif entry == "ap_cell":
            from pfsfog.revision_validation_ap import measure_ap_cell
            measure_ap_cell(None, None, engine)
        elif entry == "intrinsic":
            from pfsfog.revision_validation_ap_intrinsic import measure_intrinsic_geometry
            measure_intrinsic_geometry(SimpleNamespace(ps=engine), None, None,
                n_cosmo=6, n_mu=32, native_mean=None, native_total=None, native_intrinsic=None)
        else:
            from scripts.validate_revision import main
            main(["--case", "C", "--ap-probe"])


def test_physical_cases_keep_highz_pfs_separate_from_baseline_selection():
    from scripts.validate_elf_engine import diagnostic_cells
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.revision_forecast import RevisionConfig
    from pfsfog.surveys import default_survey_group
    cases = diagnostic_cells(FiducialCosmology(), RevisionConfig("C", cosmology_basis="extended6"))
    assert [item["cell"].model_redshift for item in cases] == [1.1, 1.8]
    assert "PFS-ELG" in cases[0]["cell"].overlap_tracers
    assert set(cases[1]["cell"].overlap_tracers) == {"DESI-QSO", "PFS-ELG"}
    assert cases[1]["scope"] == "highz_sensitivity_only"
    assert default_survey_group().pfs_zmax == 1.6
    assert all(f.c_tilde > 0 for item in cases for f in item["fiducials"].values())


def test_intrinsic_and_geometry_derivatives_reconstruct_physical_cross_response():
    from scripts.validate_elf_engine import pair_functions, elf_cosmology_spec
    from pfsfog.one_loop import make_one_loop_engine
    from pfsfog.cosmo import FIDUCIAL, FiducialCosmology
    from pfsfog.eft_params import tracer_fiducials
    from pfsfog.revision_theory import make_pair_response, fiducial_nuisance
    space = elf_cosmology_spec(FIDUCIAL)
    assert space.fd_steps["h"] == (.000125, .0000625, .00003125)
    theta = jnp.array([FIDUCIAL[n] for n in ("ln10_10_As", "mnu", "Omega_m", "h", "omega_b", "n_s")])
    cosmo = FiducialCosmology()
    qa = fiducial_nuisance(tracer_fiducials("DESI-QSO", 2.1, cosmo.sigma8(1.8)))
    qb = fiducial_nuisance(tracer_fiducials("PFS-ELG", 1.62, cosmo.sigma8(1.8)))
    response = make_pair_response(make_one_loop_engine(), 1.8, FIDUCIAL, 1.e-5,
                                  3.e-4, space=space)
    # Nonzero noise also has to contribute to the geometric derivative.
    packed = jnp.concatenate((theta, qa, qb, jnp.array([.4, -.3])))
    functions = pair_functions(response, jnp.array([.035, .105, .195]), packed, n_mu=32)
    total, intrinsic, geometry = [np.asarray(jax.jacfwd(functions[key])(packed))
                                  for key in ("total", "intrinsic", "geometry")]
    np.testing.assert_allclose(total, intrinsic+geometry, rtol=2.e-10, atol=2.e-7)
    assert np.linalg.norm(geometry[..., 3]) > 1.
    step = space.fd_steps["h"][-1]
    delta = jnp.zeros_like(packed).at[3].set(step)
    fd = (functions["geometry"](packed+delta)-functions["geometry"](packed-delta))/(2*step)
    np.testing.assert_allclose(geometry[..., 3], fd, rtol=1.e-3, atol=1.e-4)


def test_missing_cell_or_requested_refinement_cannot_be_reported_as_passed():
    from scripts.validate_elf_engine import finish_status
    report = {"cases": {"z1.1": {"status": "passed"}}, "refinements": {}}
    assert finish_status(report, ["z1.1", "z1.8"], 0) == 1
    report["cases"]["z1.8"] = {"status": "passed"}
    assert finish_status(report, ["z1.1", "z1.8"], 8) == 1
    assert report["refinement_diagnostics"] == "incomplete"
    assert finish_status(report, ["z1.1", "z1.8"], 0) == 0
    assert report["refinement_diagnostics"] == "not_requested"
    report["source_changed"] = True
    assert finish_status(report, ["z1.1", "z1.8"], 0) == 1
