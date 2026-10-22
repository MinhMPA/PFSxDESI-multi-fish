"""Declared AP resolutions and cosmological verification steps.

These are numerical prescriptions, not evidence that a forecast has passed.
The two finest likelihoods are output references; their formal 2N/4N metadata
must not start recursive derivative qualifications beyond the measured range.
"""

LEGACY_PROFILE = "legacy8192-v1"
INTRINSIC_PROFILE = "ap16384-intrinsic-v1"
PRODUCTION_PROFILES = (LEGACY_PROFILE, INTRINSIC_PROFILE)
AP_GRIDS = (2048, 8192, 16384, 32768, 65536)
_NAMES = ("ln10_10_As", "Mnu", "Omegam", "h", "omega_b", "n_s")
_STEPS = ((.001, .0005, .00025), (.005, .0025, .00125),
          (.001, .0005, .00025), (.001, .0005, .00025),
          (.0001, .00005, .000025), (.001, .0005, .00025))
_GRID_NAMES = dict(zip(AP_GRIDS, ("legacy2048-v1", LEGACY_PROFILE,
    INTRINSIC_PROFILE, "ap32768-reference-v1", "ap65536-reference-v1")))


def cosmology_steps(ap_nk=None):
    """Return a fresh schedule; omitting the grid preserves legacy behavior."""
    if ap_nk is not None and ap_nk not in AP_GRIDS:
        raise ValueError("undeclared AP numerical profile grid")
    steps = dict(zip(_NAMES, _STEPS))
    if ap_nk is not None and ap_nk >= 16384:
        steps["h"] = (.000125, .0000625, .00003125)
    return steps


def decomposition_contract(ap_nk):
    cosmology_steps(ap_nk)
    return "intrinsic-table-v1" if ap_nk >= 16384 else "forward-table-v1"


def profile_record(name):
    """Fresh JSON data: callers cannot mutate a shared numerical declaration."""
    if name not in _GRID_NAMES.values():
        raise ValueError("unknown AP numerical profile")
    nk = next(n for n, label in _GRID_NAMES.items() if label == name)
    return {
        "schema_version": "pfs-desi-ap-numerics-v1", "name": name,
        "ap_interpolation_nk": nk, "ap_refinement_nk": [2*nk, 4*nk],
        "ap_interpolation_nmu": 101, "fft_n": 256,
        "fft_k_bounds": [1e-5, 1e3], "projection_nodes": [32, 64],
        "fd_steps": {key: list(steps) for key, steps in cosmology_steps(nk).items()},
        "decomposition": decomposition_contract(nk),
        "purpose": "output_resolution" if nk > 16384 else "main_likelihood",
        "derivative_grids_to_evaluate": [] if nk > 16384 else [2*nk, 4*nk],
        "quantitative_contract": "ap-original-tolerances-v1",
    }


def configuration_profile(config):
    """Derive a profile from actual configuration, never from supplied metadata."""
    nk = config.ap_interpolation_nk
    if (nk not in AP_GRIDS or tuple(config.ap_refinement_nk) != (2*nk, 4*nk)
            or config.ap_interpolation_nmu != 101):
        raise ValueError("configuration differs from AP numerical profile")
    return profile_record(_GRID_NAMES[nk])


def profile_fields(config):
    """New profiles require explicit evidence; old serialized layouts survive."""
    return ({"numerical_profile": configuration_profile(config)}
            if config.ap_interpolation_nk >= 16384 else {})


def validate_profile_fields(record, config):
    expected = profile_fields(config)
    if expected:
        if record.get("numerical_profile") != expected["numerical_profile"]:
            raise ValueError("numerical profile missing or differs from configuration")
    elif "numerical_profile" in record:
        if record["numerical_profile"] != configuration_profile(config):
            raise ValueError("legacy numerical profile differs from configuration")
