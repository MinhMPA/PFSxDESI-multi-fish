"""Linear power and sigma8 radius in immutable baseline h_ref units."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .cosmo import make_plin_func


def native_power_to_reference(k_native, p_native, h_trial, h_ref
                              ) -> tuple[jax.Array, jax.Array]:
    """Convert k in 1/Mpc and released P in (Mpc/h_trial)^3.

    The released target already includes its native h convention: this is a
    unit conversion, not amplitude renormalization or an extra h**3 factor.
    """
    return (jnp.asarray(k_native) / h_ref,
            jnp.asarray(p_native) * (h_ref / h_trial)**3)


def sigma8_radius_reference(h_trial, h_ref) -> jax.Array:
    """Standard 8/h_trial Mpc sphere in Mpc/h_ref coordinate units."""
    return jnp.asarray(8.) * h_ref / h_trial


def make_reference_plin_func(h_ref: float) -> Callable:
    """Wrap the same released network, preserving its prediction unchanged.

    The old wrapper supplies native trial-h k and power. Undo only its k-axis
    conversion to reconstruct k_native, then convert both axes to h_ref.
    The immutable scalar h_ref is captured outside the differentiated closure.
    """
    native = make_plin_func()
    h_ref = float(h_ref)

    def plin(z, cosmo_dict):
        data = native(z, cosmo_dict)
        h_trial = cosmo_dict["h"]
        k, pk = native_power_to_reference(data["k"] * h_trial, data["pk"],
                                          h_trial, h_ref)
        return {"k": k, "pk": pk}

    return plin
