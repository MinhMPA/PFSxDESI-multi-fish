"""Exact named precision projections of a saved six-coordinate likelihood.

Fixed cosmology is removed before solving. Nuisances remain active; the
nuisance-fixed diagnostic separately solves the whole free cosmology block.
Failed solves raise ValueError for the later driver to persist as failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .cosmology_space import make_cosmology_spec, cosmology_from_coordinates
from .revision_forecast import RevisionLikelihood, _scaled_inverse
from .revision_theory import make_growth_rate_func, make_sigma8_func
from .fisher_precision import LEGACY_FISHER_SOLVER


COSMOLOGY_TREATMENTS = ("conditional3", "release_h", "release_omega_b", "release_n_s",
                        "marginal6", "marginal6_weak10", "marginal6_no_extra")


@dataclass
class CosmologyTreatmentResult:
    likelihood: RevisionLikelihood
    treatment: str
    free_cosmology: tuple[str, ...]
    active_parameter_names: tuple[str, ...]
    active_indices: tuple[int, ...]
    F_like: np.ndarray
    F_prior_cosmo: np.ndarray
    F_prior_nuisance: np.ndarray
    F_total: np.ndarray
    covariance: np.ndarray
    cov_cosmo_marg: np.ndarray
    cov_cosmo_nuisance_fixed: np.ndarray
    growth_z: np.ndarray
    growth_fiducial: np.ndarray
    growth_jacobian: np.ndarray
    growth_cov_marg: np.ndarray
    growth_cov_nuisance_fixed: np.ndarray
    validation: dict
    fisher_solver: str = LEGACY_FISHER_SOLVER


def prepare_cosmology_precision(likelihood: RevisionLikelihood, treatment: str) -> dict:
    """Retain exact named prior/conditioning inputs before an inverse can fail."""
    if treatment not in COSMOLOGY_TREATMENTS:
        raise ValueError(f"unknown cosmology treatment {treatment!r}")
    if likelihood.config.cosmology_basis != "extended6":
        raise ValueError("cosmology treatments require an extended6 likelihood")
    space = make_cosmology_spec(likelihood.reference_cosmology, extended=True,
                                ap_nk=likelihood.config.ap_interpolation_nk)
    names = likelihood.names
    size = len(names)
    if (len(set(names)) != size or len(likelihood.parameters) != size
            or np.shape(likelihood.F_like) != (size, size)
            or np.shape(likelihood.fiducial) != (size,)):
        raise ValueError("invalid named six-coordinate likelihood layout")
    records = {record["name"]: record for record in likelihood.parameters}
    if (set(records) != set(names)
            or {name for name, record in records.items() if record["role"] == "cosmo"}
            != set(space.names)):
        raise ValueError("likelihood must contain exactly the six named cosmology coordinates")
    free = space.names[:3]
    if treatment.startswith("release_"):
        free += (treatment.removeprefix("release_"),)
    elif treatment.startswith("marginal6"):
        free = space.names
    # Order active cosmology by its explicit basis; nuisances keep full named order.
    active_names = free + tuple(name for name in names if name not in space.names)
    indices = tuple(names.index(name) for name in active_names)
    active_like = np.asarray(likelihood.F_like, dtype=float)[np.ix_(indices, indices)].copy()
    cosmo_prior = np.zeros_like(active_like)
    nuisance_prior = np.zeros_like(active_like)
    widths = {}
    if treatment in ("release_omega_b", "marginal6", "marginal6_weak10"):
        widths["omega_b"] = .00055
    if treatment in ("release_n_s", "marginal6", "marginal6_weak10"):
        widths["n_s"] = .042
    if treatment == "marginal6_weak10":
        widths = {name: width*10 for name, width in widths.items()}
    precision = {"Mnu": .04, "Omegam": 1., **{n: w**-2 for n, w in widths.items()}}
    factor = likelihood.config.nuisance_factor
    for i, name in enumerate(active_names):
        if name in free:
            cosmo_prior[i, i] = precision.get(name, 0.)
        else:
            width = records[name]["prior_sigma"]
            if width is not None:
                if not np.isfinite(width) or width <= 0:
                    raise ValueError(f"invalid nuisance prior width for {name}")
                nuisance_prior[i, i] = 1/(width*factor)**2
    total = active_like + cosmo_prior + nuisance_prior
    return dict(free_cosmology=free, active_parameter_names=active_names,
                active_indices=indices, F_like=active_like, F_prior_cosmo=cosmo_prior,
                F_prior_nuisance=nuisance_prior, F_total=total,
                added_prior_widths=widths, nuisance_factor=float(factor))


def project_cosmology_precision(likelihood: RevisionLikelihood, treatment: str, *,
                                fisher_solver=LEGACY_FISHER_SOLVER) -> dict:
    """Pure named prior/conditioning algebra; does not initialize theory."""
    inputs = prepare_cosmology_precision(likelihood, treatment)
    free, active_names = inputs['free_cosmology'], inputs['active_parameter_names']
    total = inputs['F_total']
    space = make_cosmology_spec(likelihood.reference_cosmology, extended=True,
                                ap_nk=likelihood.config.ap_interpolation_nk)
    covariance, solve = _scaled_inverse(total, active_names, f"{treatment} active Fisher",
                                        fisher_solver=fisher_solver)
    count = len(free)
    fixed, fixed_solve = _scaled_inverse(total[:count, :count], free,
                                        f"{treatment} cosmology block", fisher_solver=fisher_solver)
    free_indices = [space.names.index(name) for name in free]
    marginal6 = np.zeros((6, 6))
    fixed6 = np.zeros((6, 6))
    marginal6[np.ix_(free_indices, free_indices)] = covariance[:count, :count]
    fixed6[np.ix_(free_indices, free_indices)] = fixed
    validation = {"global_solve": solve, "cosmology_block_solve": fixed_solve,
                  "nuisance_factor": inputs['nuisance_factor'], "growth_rank_upper_bound": min(count, len(likelihood.config.nominal_bins)),
                  "added_prior_widths": inputs['added_prior_widths'], "prior_center": "fiducial",
                  "fixed_cosmology": tuple(name for name in space.names if name not in free)}
    return dict(free_cosmology=free, active_parameter_names=active_names,
                active_indices=inputs['active_indices'], F_like=inputs['F_like'],
                F_prior_cosmo=inputs['F_prior_cosmo'], F_prior_nuisance=inputs['F_prior_nuisance'],
                F_total=total, covariance=covariance,
                cov_cosmo_marg=marginal6, cov_cosmo_nuisance_fixed=fixed6,
                validation=validation, fisher_solver=fisher_solver)


def apply_cosmology_treatment(likelihood: RevisionLikelihood,
                              treatment: str, *, fisher_solver=LEGACY_FISHER_SOLVER) -> CosmologyTreatmentResult:
    """Apply one of seven fiducial-centered local Gaussian prior treatments."""
    precision = project_cosmology_precision(likelihood, treatment, fisher_solver=fisher_solver)
    space = make_cosmology_spec(likelihood.reference_cosmology, extended=True,
                                ap_nk=likelihood.config.ap_interpolation_nk)
    names = likelihood.names
    z = np.asarray([(lo + hi)/2 for lo, hi in likelihood.config.nominal_bins])
    rate, sigma8 = make_growth_rate_func(), make_sigma8_func()
    reference = likelihood.reference_cosmology

    def growth(theta):
        mapped = cosmology_from_coordinates(theta, reference, space)
        return jnp.stack([rate(zi, mapped)*sigma8(zi, mapped) for zi in z])

    theta = jnp.asarray(likelihood.fiducial[[names.index(name) for name in space.names]])
    predicted = np.asarray(growth(theta), dtype=float)
    jac = np.asarray(jax.jacfwd(growth)(theta), dtype=float)
    if (predicted.shape != (len(z),) or jac.shape != (len(z), 6)
            or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(jac))):
        raise ValueError("invalid extended growth predictions or Jacobian")
    return CosmologyTreatmentResult(
        likelihood=likelihood, treatment=treatment, **precision,
        growth_z=z, growth_fiducial=predicted, growth_jacobian=jac,
        growth_cov_marg=jac @ precision['cov_cosmo_marg'] @ jac.T,
        growth_cov_nuisance_fixed=jac @ precision['cov_cosmo_nuisance_fixed'] @ jac.T)
