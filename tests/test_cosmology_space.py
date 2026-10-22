"""Declared coordinate basis and mapped emulator validity domain."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.cosmology_space import (
    cosmology_from_coordinates, make_cosmology_spec,
    validate_coordinate_neighborhood,
)


NAMES = ("ln10_10_As", "Mnu", "Omegam", "h", "omega_b", "n_s")
STEPS = ((.001, .0005, .00025), (.005, .0025, .00125),
         (.001, .0005, .00025), (.001, .0005, .00025),
         (.0001, .00005, .000025), (.001, .0005, .00025))


def theta6(reference=FIDUCIAL):
    return jnp.array([reference["ln10_10_As"], reference["mnu"],
                      reference["Omega_m"], reference["h"],
                      reference["omega_b"], reference["n_s"]])


def test_declared_basis_steps_and_immutable_reference_scale():
    reference = dict(FIDUCIAL)
    space = make_cosmology_spec(reference, extended=True)
    assert space.names == NAMES
    assert dict(space.fd_steps) == dict(zip(NAMES, STEPS))
    assert space.unit_convention == "fixed_h_ref"
    reference["h"] = .7
    assert space.h_ref == FIDUCIAL["h"]
    with pytest.raises(FrozenInstanceError):
        space.h_ref = .7
    with pytest.raises(TypeError):
        space.fd_steps["h"] = (.1, .01, .001)
    baseline = make_cosmology_spec(FIDUCIAL, extended=False)
    assert baseline.names == NAMES[:3]
    assert tuple(baseline.fd_steps) == NAMES[:3]


def test_mapping_physical_density_and_jax_jacobian():
    reference = dict(FIDUCIAL)
    space = make_cosmology_spec(reference, extended=True)
    theta = theta6().at[1].set(.12).at[2].set(.32).at[3].set(.7).at[4].set(.023)
    mapped = jax.jit(lambda x: cosmology_from_coordinates(x, reference, space))(theta)
    assert float(mapped["omega_cdm"]) == pytest.approx(.32 * .7**2 - .023 - .12 / 93.14)
    assert float(mapped["Omega_m"]) == pytest.approx(.32)
    assert reference == FIDUCIAL
    jac = jax.jacfwd(lambda x: cosmology_from_coordinates(x, reference, space)["omega_cdm"])(theta)
    np.testing.assert_allclose(jac, [0, -1/93.14, .7**2, 2*.32*.7, -1, 0], rtol=1e-12, atol=1e-12)
    baseline = make_cosmology_spec(reference, extended=False)
    from pfsfog.revision_theory import cosmology_from_theta
    old = cosmology_from_theta(theta6()[:3], reference)
    new = cosmology_from_coordinates(theta6()[:3], reference, baseline)
    assert set(new) == set(old)
    for key in old:
        np.testing.assert_allclose(new[key], old[key], rtol=1e-12)
    with pytest.raises(ValueError, match="shape"):
        cosmology_from_coordinates(theta6()[:3], reference, space)


def test_neighborhood_checks_all_signed_steps_and_exact_mapped_bounds():
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    result = validate_coordinate_neighborhood(np.array(theta6()), FIDUCIAL, space)
    assert result["valid"]
    assert len(result["points"]) == 1 + 6 * 3 * 2
    assert result["bounds"]["omega_cdm"] == (.09, .15)
    assert result["bounds"]["redshift"] == (0., 5.)
    assert result["validated_redshift"] is None
    # h and Omega_m are individually plausible, but their mapped density is not.
    bad = theta6().at[3].set(.8)
    with pytest.raises(ValueError, match="omega_cdm"):
        validate_coordinate_neighborhood(np.array(bad), FIDUCIAL, space)
    # Center valid, largest signed h step crosses the density boundary.
    edge = theta6().at[2].set((.15 - .00048 + FIDUCIAL["omega_b"] + .06/93.14) / FIDUCIAL["h"]**2)
    with pytest.raises(ValueError, match="h.*omega_cdm"):
        validate_coordinate_neighborhood(np.array(edge), FIDUCIAL, space)


@pytest.mark.parametrize("key,value", [("omega_b", .031), ("n_s", 1.06),
                                      ("ln10_10_As", 3.51), ("Mnu", -.01), ("h", 1.01)])
def test_coordinate_training_bounds(key, value):
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    theta = theta6().at[NAMES.index(key)].set(value)
    if key == "h":
        # Isolate the h limit from the separately tested mapped-density limit.
        theta = theta.at[2].set((.12 + FIDUCIAL["omega_b"] + .06/93.14) / value**2)
    with pytest.raises(ValueError, match=key if key != "Mnu" else "mnu"):
        validate_coordinate_neighborhood(np.array(theta), FIDUCIAL, space)


def test_optional_redshift_is_validated_without_claiming_missing_validation():
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    valid = validate_coordinate_neighborhood(np.array(theta6()), {**FIDUCIAL, "z": 1.1}, space)
    assert valid["validated_redshift"] == 1.1
    for extra in ({"z": 5.1}, {"redshift": -.1}, {"z": 1., "redshift": 2.}):
        with pytest.raises(ValueError, match="redshift"):
            validate_coordinate_neighborhood(np.array(theta6()), {**FIDUCIAL, **extra}, space)
