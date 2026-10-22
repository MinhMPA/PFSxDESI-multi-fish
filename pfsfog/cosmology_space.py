"""Explicit physical cosmology coordinates and released-network domain checks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from .ap_numerics import cosmology_steps


_NAMES = ("ln10_10_As", "Mnu", "Omegam", "h", "omega_b", "n_s")
_STEPS = ((.001, .0005, .00025), (.005, .0025, .00125),
          (.001, .0005, .00025), (.001, .0005, .00025),
          (.0001, .00005, .000025), (.001, .0005, .00025))
TRAINING_BOUNDS = MappingProxyType({
    "omega_b": (.015, .030), "omega_cdm": (.09, .15),
    "ln10_10_As": (2.5, 3.5), "n_s": (.85, 1.05),
    "h": (.4, 1.), "mnu": (0., .5), "redshift": (0., 5.),
})


@dataclass(frozen=True)
class CosmologySpec:
    """Immutable declared basis, reference scale and finite-difference schedule."""

    names: tuple[str, ...]
    h_ref: float
    fd_steps: Mapping[str, tuple[float, float, float]]
    unit_convention: str

    def __post_init__(self):
        names = tuple(self.names)
        if names not in (_NAMES[:3], _NAMES):
            raise ValueError("cosmology names must be ordered baseline3 or extended6")
        if not np.isfinite(self.h_ref) or self.h_ref <= 0:
            raise ValueError("h_ref must be finite and positive")
        if self.unit_convention != "fixed_h_ref":
            raise ValueError("unit_convention must be fixed_h_ref")
        if set(self.fd_steps) != set(names):
            raise ValueError("fd_steps must contain exactly the declared names")
        steps = {name: tuple(self.fd_steps[name]) for name in names}
        if any(len(values) != 3 or not all(np.isfinite(v) and v > 0 for v in values)
               for values in steps.values()):
            raise ValueError("fd_steps require three finite positive steps per coordinate")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "h_ref", float(self.h_ref))
        object.__setattr__(self, "fd_steps", MappingProxyType(steps))


def make_cosmology_spec(reference: dict, *, extended: bool,
                        ap_nk: int | None = None) -> CosmologySpec:
    """Capture the baseline h without retaining a mutable reference dictionary."""
    source = reference.params if hasattr(reference, "params") else reference
    names = _NAMES if extended else _NAMES[:3]
    steps = cosmology_steps(ap_nk)
    return CosmologySpec(names, source["h"], {n: steps[n] for n in names}, "fixed_h_ref")


def cosmology_from_coordinates(theta: jax.Array, reference: dict,
                               space: CosmologySpec) -> dict:
    """Pure JAX map to total-matter physical inputs; no implicit extra columns."""
    source = reference.params if hasattr(reference, "params") else reference
    theta = jnp.asarray(theta)
    if theta.shape != (len(space.names),):
        raise ValueError(f"theta shape must be {(len(space.names),)}, got {theta.shape}")
    values = dict(zip(space.names, theta))
    h = values.get("h", source["h"])
    omega_b = values.get("omega_b", source["omega_b"])
    mnu, omega_m = values["Mnu"], values["Omegam"]
    return {
        "h": h, "omega_b": omega_b,
        "omega_cdm": omega_m * h**2 - omega_b - mnu / 93.14,
        "n_s": values.get("n_s", source["n_s"]),
        "ln10_10_As": values["ln10_10_As"], "mnu": mnu, "Omega_m": omega_m,
    }


def _check_mapped_bounds(mapped, label):
    for key, (lower, upper) in TRAINING_BOUNDS.items():
        if key == "redshift":
            continue
        value = float(mapped[key])
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{label}: mapped {key}={value} outside [{lower}, {upper}]")


def validate_coordinate_neighborhood(theta: np.ndarray, reference: dict,
                                     space: CosmologySpec) -> dict:
    """Validate the center and every signed scheduled point outside JAX tracing.

    Optional reference z/redshift is checked here; absent redshift is recorded
    explicitly. A response separately checks its actual evaluation redshift.
    Training bounds are validity limits, never additional forecast priors.
    """
    source = reference.params if hasattr(reference, "params") else reference
    redshifts = [float(source[key]) for key in ("z", "redshift") if key in source]
    lower, upper = TRAINING_BOUNDS["redshift"]
    if any(not np.isfinite(z) or not lower <= z <= upper for z in redshifts):
        raise ValueError(f"redshift outside [{lower}, {upper}]")
    if len(redshifts) == 2 and redshifts[0] != redshifts[1]:
        raise ValueError("reference z and redshift disagree")
    theta = np.asarray(theta, dtype=float)
    # Check center shape before constructing perturbations.
    center = cosmology_from_coordinates(theta, source, space)
    _check_mapped_bounds(center, "center")
    points = [{"coordinate": None, "step": 0., "sign": 0,
               "mapped": {key: float(value) for key, value in center.items()}}]
    for index, name in enumerate(space.names):
        for step in space.fd_steps[name]:
            for sign in (-1, 1):
                shifted = theta.copy()
                shifted[index] += sign * step
                mapped = cosmology_from_coordinates(shifted, source, space)
                _check_mapped_bounds(mapped, f"{name} {sign:+d} step {step}")
                points.append({"coordinate": name, "step": step, "sign": sign,
                               "mapped": {key: float(value) for key, value in mapped.items()}})
    return {"valid": True, "points": points, "bounds": dict(TRAINING_BOUNDS),
            "validated_redshift": redshifts[0] if redshifts else None}
