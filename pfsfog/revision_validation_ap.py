"""Quantitative, point-complete AP and response-domain evidence checks.

The interpolation and angular budget is half the accepted 1e-3 derivative
comparison tolerance. It is declared here before any physical measurements.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import numpy as np
import jax
import jax.numpy as jnp

from .revision_artifacts import _canonical
from .revision_forecast import build_revision_likelihood, make_shell_grid
from .revision_validation_physical import (
    _cell_fiducials, _checked_pairs, _check_shells, expected_support_schedule,
    validate_support_schedule)
from .revision_validation_suite import physical_dependencies
from .revision_validation import COSMO_STEPS
from .revision_theory import (COSMO_NAMES, NUISANCE_NAMES, ap_coordinates,
                              make_pair_response, nuisance_to_backend,
                              PairResponse, APInterpolationBackend)
from .cosmology_space import make_cosmology_spec, validate_coordinate_neighborhood
from .revision_validation_physical import validate_physical_point
from .eft_params import CROSS_STOCH_NAMES


IDENTITY_TOLERANCE = 1e-3
DERIVATIVE_TOLERANCE = 1e-3
CHAIN_TOLERANCE = 1e-10
REFINEMENT_TOLERANCE = 5e-4
AP_K_BOUNDS = (.002, .5)


def ap_parameter_schedule(*, space=None):
    """Fiducial plus the accepted ± finite-difference cosmology points."""
    return [("fiducial", 0, 0.)] + [
        (name, sign, float(step))
        for name in (COSMO_NAMES if space is None else space.names)
        for step in (COSMO_STEPS if space is None else space.fd_steps)[name] for sign in (-1, 1)]


def _toy_spec():
    # Bilinear intrinsic power is reproduced exactly by the production cubic
    # interpolation; no interpolation error budget is hidden in this control.
    return [(kind, name, ap, al, beta) for kind in ("auto", "cross")
            for name, ap, al, beta in (("isotropic",1.04,1.04,0.),
                                      ("transverse",1.03,1.,.6),
                                      ("radial",1.,.97,.6),
                                      ("anisotropic",1.03,.97,.6))]


def measure_pure_ap_controls(ap_nk=2048, ap_nmu=101):
    """Hold intrinsic physics fixed, exercising unmodified PairResponse.pkmu.

    Only the intrinsic spectrum and background functions are controlled. Both
    auto/cross dispatch and the APInterpolationBackend remap are production
    methods. P=k(1+beta*abs(mu)) and the separate cross-stochastic term have
    independent Cartesian expectations.
    """
    class IntrinsicSpectrum:
        def get_pkmu_pair(self, k, mu, pk_data, params, add_stochasticity=False):
            cross = "bias2" in params
            if cross == add_stochasticity:
                raise ValueError("pure AP auto/cross dispatch mismatch")
            amplitude = 2. if cross else 1.
            return amplitude*jnp.asarray(k)[:, None]*(1+pk_data*jnp.asarray(mu)[None, :])

    controls = []
    for kind, name, transverse, radial, beta in _toy_spec():
        response = object.__new__(PairResponse)
        response.space = None
        response.reference = {"h":.67,"omega_b":.022,"n_s":.96}
        response.z, response.nbar_a = 1., 1.
        response.nbar_b = 1. if kind == "cross" else None
        response.k_nl, response.sigma8_ref = .7, 1.
        response.sigma8 = lambda *args: jnp.asarray(1.)
        response.growth_rate = lambda *args: jnp.asarray(0.)
        response.plin = lambda *args: beta
        response.ap_alphas = lambda theta: (transverse, radial)
        response.ap_backend = APInterpolationBackend(IntrinsicSpectrum(), ap_nk, ap_nmu)
        k, mu = [.071,.113], [-.73,.19,.61]
        q = jnp.zeros(len(NUISANCE_NAMES))
        cross_stochastic = [.07, 3.] if kind == "cross" else None
        q_cross = None if cross_stochastic is None else jnp.asarray(cross_stochastic)
        observed = response.pkmu(k, mu, jnp.asarray([3.,.06,.3]), q,
                                 q if kind == "cross" else None, q_cross)
        controls.append({"pair_kind":kind,"control":name,"k":k,"mu":mu,
                         "alpha_perp":transverse,"alpha_para":radial,
                         "slope":1.,"beta":beta,"amplitude":2. if kind == "cross" else 1.,
                         "cross_stochastic":cross_stochastic,"k_nl":.7,
                         "nbar_a":1.,"nbar_b":1. if kind == "cross" else None,
                         "ap_nk":ap_nk,"ap_nmu":ap_nmu,
                         "observed":np.asarray(observed).tolist()})
    return controls


def validate_pure_ap_controls(controls, ap_nk=2048, ap_nmu=101):
    if not isinstance(controls, list) or len(controls) != len(_toy_spec()):
        raise ValueError("pure AP control coverage incomplete")
    errors = {}
    for row, (kind, name, ap, al, beta) in zip(controls, _toy_spec()):
        expected_metadata = {"pair_kind":kind,"control":name,"k":[.071,.113],
                             "mu":[-.73,.19,.61],"alpha_perp":ap,"alpha_para":al,
                             "slope":1.,"beta":beta,"amplitude":2. if kind == "cross" else 1.,
                             "cross_stochastic":[.07,3.] if kind == "cross" else None,
                             "k_nl":.7,"nbar_a":1.,"nbar_b":1. if kind == "cross" else None,
                             "ap_nk":ap_nk,"ap_nmu":ap_nmu}
        if any(row.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("pure AP control coordinates mismatch")
        k, mu = np.asarray(row["k"])[:,None], np.abs(np.asarray(row["mu"]))[None,:]
        transverse, radial = k*np.sqrt(1-mu**2)/ap, k*mu/al
        true_k = np.hypot(transverse, radial)
        expected = row["amplitude"]*(true_k+beta*radial)/(ap**2*al)
        if kind == "cross":
            q0, q2 = .07, 3.
            expected += (q0+q2*(radial/.7)**2)/(ap**2*al)
        observed = _finite(row.get("observed"), expected.shape, "pure AP output")
        error = float(np.max(np.abs(observed/expected-1)))
        if error > CHAIN_TOLERANCE:
            raise ValueError(f"pure AP {kind} {name} failed: {error}")
        errors[f"{kind}:{name}"] = error
    return {"status":"passed", "errors":errors,"tolerance":CHAIN_TOLERANCE}


def pure_ap_toy_errors():
    """Compatibility diagnostic, now measured through the production remap."""
    result = validate_pure_ap_controls(measure_pure_ap_controls())
    return (max(v for k,v in result["errors"].items() if k.endswith(":isotropic")),
            max(v for k,v in result["errors"].items() if not k.endswith(":isotropic")))


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def _successive(errors, tolerance):
    return bool(any(errors[i] <= tolerance and errors[i + 1] <= tolerance
                    for i in range(len(errors) - 1)))


def _response_error(norm, delta, reference, scale, *, space=None, fraction=1e-6):
    """Legacy diagnostic floor, or the declared normalized-zero physical rule."""
    denominator = norm(reference)
    if space is None:
        return norm(delta)/max(denominator,fraction*scale)
    if fraction == 1. or denominator <= 1e-12*scale:
        return norm(delta)/scale
    return norm(delta)/denominator


def _check_raw_diagnostics(row, context, *, space=None):
    """Recompute the saved scalar evidence in the actual pair covariance metric."""
    names = COSMO_NAMES if space is None else space.names
    steps = COSMO_STEPS if space is None else space.fd_steps
    nc = len(names)
    covariance = _finite(row.get("pair_covariance"), (3, 3), "pair covariance")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"nonpositive AP pair covariance at {context}") from exc

    def norm(vector):
        return float(np.linalg.norm(np.linalg.solve(chol, vector)))

    fid = _finite(row.get("fiducial_multipoles"), (3,), "fiducial multipoles")
    direct = _finite(row.get("direct_multipoles"), (3,), "direct multipoles")
    total = _finite(row.get("total_ad"), (3, nc), "total AD")
    intrinsic = _finite(row.get("intrinsic_ad"), (3, nc), "intrinsic AD")
    geometry = _finite(row.get("geometry_ad"), (3, nc), "geometry AD")
    fd = _finite(row.get("fd_derivatives"), (nc, 3, 3), "physical FD vectors")
    alpha_ad = _finite(row.get("alpha_derivatives"), (2, nc), "distance-ratio AD")
    if np.any(np.abs(alpha_ad[:, names.index("ln10_10_As")]) > CHAIN_TOLERANCE):
        raise ValueError(f"AP amplitude geometry derivative must vanish at {context}")
    continuity = _finite(row.get("continuity_deltas"), (2, 3), "continuity vectors")
    scale = norm(fid)
    if scale <= 0 or not np.isclose(row.get("whitened_physical_scale"), scale,
                                    rtol=1e-8, atol=1e-12):
        raise ValueError(f"AP physical scale mismatch at {context}")

    def error(delta, reference, fraction=1e-6):
        return _response_error(norm,delta,reference,scale,space=space,fraction=fraction)

    def same(name, observed, calculated):
        observed = np.asarray(observed, dtype=float)
        calculated = np.asarray(calculated, dtype=float)
        if (observed.shape != calculated.shape or not np.all(np.isfinite(observed))
                or not np.allclose(observed, calculated, rtol=1e-7, atol=1e-10)):
            raise ValueError(f"AP {name} differs from raw arrays at {context}")

    same("identity error", row["identity_error"], error(fid-direct, direct))
    same("continuity error", row["continuity_error"],
         max(error(delta, fid, fraction=1.) for delta in continuity))
    same("AD/FD errors", row["ad_fd_errors"],
         [[error(total[:, i]-fd[i, j], total[:, i] if space is None else fd[i,j]) for j in range(3)]
          for i in range(nc)])
    same("chain errors", row["chain_errors"],
         [error(total[:, i]-intrinsic[:, i]-geometry[:, i], total[:, i])
          for i in range(nc)])
    coordinates = row["domain"]["parameter_coordinates"]
    if not np.allclose([coordinates[0]["alpha_perp"],coordinates[0]["alpha_para"]],
                       [1.,1.], rtol=0, atol=1e-10):
        raise ValueError(f"AP frozen-reference identity failed at {context}")
    alpha_errors = []
    for i in range(nc):
        errors = []
        for j, step in enumerate(steps[names[i]]):
            negative = coordinates[1+i*6+j*2]
            positive = coordinates[2+i*6+j*2]
            central = (np.asarray([positive["alpha_perp"],positive["alpha_para"]]) -
                       np.asarray([negative["alpha_perp"],negative["alpha_para"]]))/(2*step)
            errors.append(float(_response_error(np.linalg.norm,alpha_ad[:,i]-central,
                alpha_ad[:,i] if space is None else central,1.,space=space)))
        alpha_errors.append(errors)
    same("distance-ratio errors", row["alpha_fd_errors"], alpha_errors)
    previous_mean, previous_ad = fid, total
    for check in row["interpolation"]:
        mean = _finite(check.get("mean"), (3,), "refined mean")
        ad = _finite(check.get("ad"), (3, nc), "refined AD")
        same("interpolation mean error", check["mean_error"],
             error(mean-previous_mean, mean))
        same("interpolation derivative errors", check["derivative_errors"],
             [error(ad[:, i]-previous_ad[:, i], ad[:, i]) for i in range(nc)])
        previous_mean, previous_ad = mean, ad
    check = row["angular"]
    mean = _finite(check.get("mean"), (3,), "angular mean")
    ad = _finite(check.get("ad"), (3, nc), "angular AD")
    same("angular mean error", check["mean_error"], error(mean-fid, mean))
    same("angular derivative errors", check["derivative_errors"],
         [error(ad[:, i]-total[:, i], ad[:, i]) for i in range(nc)])


def angular_grid_declaration(n_mu):
    nodes, _ = np.polynomial.legendre.leggauss(n_mu)
    refined, _ = np.polynomial.legendre.leggauss(64)
    return {"mean_projection":{"role":"observed", "mu":((nodes+1)/2).tolist()},
            "covariance":{"role":"observed_signed_folded_by_PairResponse", "mu":nodes.tolist()},
            "refined_projection":{"role":"observed", "mu":((refined+1)/2).tolist()},
            "intrinsic_table":{"role":"intrinsic_support_not_remapped", "mu":np.linspace(0.,1.,101).tolist(),
                               "k_bounds":list(AP_K_BOUNDS)}}


def _validate_domain_grid(domain, k, mu, context, *, space=None, reference=None, theta=None, redshift=None):
    if not isinstance(domain, dict):
        raise ValueError(f"AP domain points incomplete at {context}")
    coordinates = domain.get("parameter_coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != len(ap_parameter_schedule(space=space)):
        raise ValueError(f"AP parameter points incomplete at {context}")
    mapped_k, mapped_mu = [], []
    for recorded, (name, sign, step) in zip(coordinates, ap_parameter_schedule(space=space)):
        if ([recorded.get("parameter"), recorded.get("sign"), recorded.get("step")]
                != [name, sign, step]
                or recorded.get("response_shape") != [1, len(mu)]):
            raise ValueError(f"AP parameter coordinate or response shape mismatch at {context}")
        if space is not None:
            if reference is None or theta is None or redshift is None:
                raise ValueError("explicit AP domain requires frozen theta/reference/redshift")
            varied = np.asarray(theta, dtype=float).copy()
            if name != "fiducial":
                varied[space.names.index(name)] += sign*step
            if not np.array_equal(recorded.get("theta"), varied):
                raise ValueError(f"AP signed theta mismatch at {context}")
            mapped = validate_physical_point(varied, reference, space, redshift)
            if _canonical(recorded.get("mapped_cosmology")) != _canonical(mapped):
                raise ValueError(f"AP mapped cosmology mismatch at {context}")
        alphas = _finite([recorded.get("alpha_perp"), recorded.get("alpha_para")],
                         (2,), "AP factors")
        extremes = _finite([recorded.get("response_min"), recorded.get("response_max")],
                           (2,), "AP physical response extremes")
        response_values = _finite(recorded.get("response_values"), (len(mu),),
                                  "AP physical response values")
        if (np.any(alphas <= 0) or extremes[0] > extremes[1]
                or not np.allclose(extremes,
                                   [response_values.min(), response_values.max()],
                                   rtol=1e-12, atol=1e-14)):
            raise ValueError(f"invalid AP factors or response at {context}")
        if name == "ln10_10_As" and not np.allclose(alphas, 1., rtol=0, atol=CHAIN_TOLERANCE):
            raise ValueError(f"AP amplitude geometry must remain unity at {context}")
        ap, al = alphas
        fac = np.sqrt(1 + np.abs(mu)**2 * ((ap/al)**2 - 1))
        mapped_k.extend((float(k) * fac / ap).tolist())
        mapped_mu.extend((np.abs(mu) * (ap/al) / fac).tolist())
    mapped_k = np.asarray(mapped_k)
    mapped_mu = np.asarray(mapped_mu)
    bounds = _finite([domain.get(name) for name in (
        "k_true_min", "k_true_max", "mu_true_min", "mu_true_max",
        "k_margin_lower", "k_margin_upper")], (6,), "domain margins")
    if (bounds[0] < AP_K_BOUNDS[0] or bounds[1] > AP_K_BOUNDS[1]
            or bounds[2] < -1e-12 or bounds[3] > 1 + 1e-12
            or bounds[4] < 0 or bounds[5] < 0
            or not np.isclose(bounds[4], bounds[0] - AP_K_BOUNDS[0])
            or not np.isclose(bounds[5], AP_K_BOUNDS[1] - bounds[1])
            or not np.allclose(bounds[:4], [mapped_k.min(), mapped_k.max(),
                                            mapped_mu.min(), mapped_mu.max()],
                               rtol=1e-12, atol=1e-14)):
        raise ValueError(f"AP mapped domain failed at {context}")


def _domain_record(k, mu, spectra, ik):
    coordinates, mapped_k, mapped_mu = [], [], []
    for entry in spectra:
        name, sign, step, alphas, spectrum = entry[:5]
        ap, al = alphas
        folded = np.abs(np.asarray(mu))
        fac = np.sqrt(1+folded**2*((ap/al)**2-1))
        mapped_k.extend((k*fac/ap).tolist())
        mapped_mu.extend((folded*(ap/al)/fac).tolist())
        coordinates.append({"parameter":name,"sign":sign,"step":step,
                            "alpha_perp":float(ap),"alpha_para":float(al),
                            "response_shape":[1,len(mu)],
                            "response_min":float(spectrum[ik].min()),
                            "response_max":float(spectrum[ik].max()),
                            "response_values":spectrum[ik].tolist(),
                            **(entry[5] if len(entry) == 6 else {})})
    return {"parameter_coordinates":coordinates,"k_true_min":min(mapped_k),
            "k_true_max":max(mapped_k),"mu_true_min":min(mapped_mu),
            "mu_true_max":max(mapped_mu),"k_margin_lower":min(mapped_k)-AP_K_BOUNDS[0],
            "k_margin_upper":AP_K_BOUNDS[1]-max(mapped_k)}


def validate_ap_point_rows(rows, expected_keys, *, expected_mu=None,
                           ap_nk=2048, refinement_nk=(4096, 8192),
                           require_grid_coverage=False, historical_only=False, space=None,
                           reference=None, theta=None, redshift=None):
    """Reject omission, duplication, failed criteria and flag-only evidence.

    Each row corresponds to one physical observed k and retained pair. Full
    cell/pair/k coverage is supplied by the independent support schedule.
    """
    names = COSMO_NAMES if space is None else space.names
    nc = len(names)
    if space is not None and historical_only:
        raise ValueError("explicit physical gates cannot use historical-only relaxation")
    if tuple(refinement_nk) != (2*ap_nk, 4*ap_nk):
        raise ValueError("AP interpolation sequence requires two successive doublings")
    expected = list(expected_keys)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("independent AP expected key schedule is empty or duplicated")
    keys = []
    for row in rows:
        try:
            key = (row["cell_id"], tuple(row["pair"]), float(row["k"]))
            if len(key[1]) != 2 or not np.isfinite(key[2]):
                raise ValueError("invalid AP pair/k key")
            keys.append(key)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid AP point key") from exc
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("AP point key coverage omitted, duplicated or unexpected")
    for row in rows:
        if space is not None and row.get("cosmology_columns") != list(names):
            raise ValueError("AP cosmology column ordering mismatch")
        context = (row["cell_id"], row["pair"], row["k"])
        mu = np.asarray(row.get("mu"), dtype=float)
        if (mu.ndim != 1 or len(mu) < 2 or not np.all(np.isfinite(mu))
                or np.any(mu < 0) or np.any(mu > 1)
                or np.any(np.diff(mu) <= 0)):
            raise ValueError(f"invalid AP angular coordinates at {context}")
        if expected_mu is not None and not np.array_equal(mu, expected_mu):
            raise ValueError(f"AP angular grid mismatch at {context}")
        if row.get("shape") != [1, len(mu)]:
            raise ValueError(f"AP response shape mismatch at {context}")
        _validate_domain_grid(row.get("domain"), row["k"], mu, context, space=space,
                              reference=reference,theta=theta,redshift=redshift)
        if require_grid_coverage:
            grids = row.get("observed_domains")
            if not isinstance(grids, dict) or set(grids) != {"covariance", "refined_projection"}:
                raise ValueError(f"AP angular domain coverage incomplete at {context}")
            for role in ("covariance", "refined_projection"):
                grid_mu = np.asarray(angular_grid_declaration(len(mu))[role]["mu"])
                _validate_domain_grid(grids[role], row["k"], grid_mu, (context, role), space=space,
                                      reference=reference,theta=theta,redshift=redshift)
                for base, point in zip(row['domain']['parameter_coordinates'],
                                       grids[role]['parameter_coordinates']):
                    if any(base[key] != point[key] for key in ('alpha_perp','alpha_para')):
                        raise ValueError(f"AP angular domain geometry differs between grids at {context}")
        for name, tol in (("identity_error", IDENTITY_TOLERANCE),
                          ("continuity_error", IDENTITY_TOLERANCE),
                          ("pure_ap_isotropic_error", CHAIN_TOLERANCE),
                          ("pure_ap_anisotropic_error", CHAIN_TOLERANCE)):
            value = _finite(row.get(name), (), name).item()
            if value < 0 or (value > tol and (not historical_only or name.startswith("pure_ap"))):
                raise ValueError(f"{name} failed at {context}: {value}")
        for name, tol in (("ad_fd_errors", DERIVATIVE_TOLERANCE),
                          ("alpha_fd_errors", DERIVATIVE_TOLERANCE)):
            errors = _finite(row.get(name), (nc, 3), name)
            limits = np.full((nc,3),tol)
            if space is not None:
                if name == "ad_fd_errors":
                    vectors = _finite(row.get("fd_derivatives"),(nc,3,3),"physical FD vectors")
                    chol = np.linalg.cholesky(_finite(row.get("pair_covariance"),(3,3),"pair covariance"))
                    norms = np.array([[np.linalg.norm(np.linalg.solve(chol,v)) for v in group] for group in vectors])
                    scale = float(row["whitened_physical_scale"])
                else:
                    coordinates = row["domain"]["parameter_coordinates"]
                    norms = np.empty((nc,3))
                    for i,coordinate in enumerate(names):
                        for j,step in enumerate(space.fd_steps[coordinate]):
                            minus,plus = coordinates[1+i*6+j*2:3+i*6+j*2]
                            norms[i,j] = np.linalg.norm((np.array([plus["alpha_perp"],plus["alpha_para"]])-
                                np.array([minus["alpha_perp"],minus["alpha_para"]]))/(2*step))
                    scale = 1.
                limits[norms <= 1e-12*scale] = 1e-6
            if np.any(errors < 0) or (not all(_successive(e,tol) if space is None else
                        any(e[j] < limit[j] and e[j+1] < limit[j+1] for j in (0,1))
                        for e,limit in zip(errors,limits))
                                     and (not historical_only or name == "alpha_fd_errors")):
                raise ValueError(f"{name} failed at {context}")
        chain = _finite(row.get("chain_errors"), (nc,), "chain_errors")
        if np.any(chain < 0) or (not historical_only and np.any(chain > CHAIN_TOLERANCE)):
            raise ValueError(f"AP chain closure failed at {context}")
        refinement = row.get("interpolation")
        if not isinstance(refinement, list) or [r.get("nk") for r in refinement] != list(refinement_nk):
            raise ValueError(f"AP interpolation sequence incomplete at {context}")
        for check in [*refinement, row.get("angular")]:
            if not isinstance(check, dict):
                raise ValueError(f"missing AP refinement at {context}")
            mean = _finite(check.get("mean_error"), (), "refinement mean").item()
            derivative = _finite(check.get("derivative_errors"), (nc,),
                                 "refinement derivatives")
            if mean < 0 or np.any(derivative < 0) or (not historical_only and
                    (mean > REFINEMENT_TOLERANCE or np.any(derivative > REFINEMENT_TOLERANCE))):
                raise ValueError(f"AP refinement failed at {context}")
        if row["angular"].get("n_mu") != 64:
            raise ValueError(f"AP angular refinement incomplete at {context}")
        _check_raw_diagnostics(row, context, space=space)
    return {"status":"passed", "expected_points":len(expected),
            "measured_points":len(keys), "identity_tolerance":IDENTITY_TOLERANCE,
            "derivative_tolerance":DERIVATIVE_TOLERANCE,
            "chain_tolerance":CHAIN_TOLERANCE,
            "refinement_tolerance":REFINEMENT_TOLERANCE}


AP_METRIC_CONTRACT = {
    "norm":"sqrt(sum_k ||solve(cholesky(C_retained[k]), vector[k])||^2); no dk weight",
    "blocks":"joint and each pair/ell zero-embedded in full retained space; all must pass",
    "physical_scale":"full fiducial mean norm across all retained pairs, multipoles and k",
    "near_zero_switch":1e-12, "ad_fd_relative_tolerance":DERIVATIVE_TOLERANCE,
    "ad_fd_absolute_tolerance":1e-6, "ad_fd_reference":"FD",
    "refinement_tolerance":REFINEMENT_TOLERANCE,
    "chain_tolerance":CHAIN_TOLERANCE, "identity_tolerance":IDENTITY_TOLERANCE,
    "other_zero_branch":"difference/physical_scale with each check's own threshold",
    "historical_pair_metric":"3x3 pair covariance, 1e-6 physical-scale denominator floor; diagnostic only",
    "amplitude_geometry_absolute_tolerance":CHAIN_TOLERANCE,
}


def _same_assembled(observed, actual, label):
    actual = np.asarray(actual, dtype=float)
    value = _finite(observed, actual.shape, label)
    if not np.all(np.isfinite(actual)) or not np.array_equal(value, actual):
        raise ValueError(f"AP evidence differs from assembled {label}")


def _cell_arrays(likelihood, cell, rows, *, space=None):
    names = COSMO_NAMES if space is None else space.names
    nc = len(names)
    pairs = [tuple(p) for p in cell.metadata['retained_pairs']]
    centers = np.asarray(likelihood.grid.centers)
    lookup = {(tuple(r['pair']),float(r['k'])):r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {(p,float(k)) for p in pairs for k in centers}:
        raise ValueError("AP retained cell point coverage mismatch")
    ordered = [[lookup[(p,float(k))] for p in pairs] for k in centers]
    def packed(field):
        return np.asarray([[r[field] for r in shell] for shell in ordered])
    nk, nr = len(centers),3*len(pairs)
    mean = packed('fiducial_multipoles').reshape(nk,nr)
    ad = packed('total_ad').reshape(nk,nr,nc)
    _same_assembled(mean, cell.fiducial_mean, 'fiducial mean')
    indices = [cell.names.index(name) for name in names]
    _same_assembled(ad, np.asarray(cell.derivatives)[:,:,indices], 'cosmology AD columns')
    for ik,shell in enumerate(ordered):
        for ipair,row in enumerate(shell):
            _same_assembled(row['pair_covariance'],cell.covariance[ik,3*ipair:3*ipair+3,3*ipair:3*ipair+3],
                            'pair covariance')
    arrays = {'mean':mean,'ad':ad,
              'direct':packed('direct_multipoles').reshape(nk,nr),
              'intrinsic':packed('intrinsic_ad').reshape(nk,nr,nc),
              'geometry':packed('geometry_ad').reshape(nk,nr,nc),
              'fd':packed('fd_derivatives').transpose(0,1,4,2,3).reshape(nk,nr,nc,3),
              'continuity':packed('continuity_deltas').transpose(0,1,3,2).reshape(nk,nr,2)}
    for tag, accessor in [('angular',lambda r:r['angular'])] + [
            (f'refinement_{i}',lambda r,i=i:r['interpolation'][i]) for i in range(2)]:
        arrays[tag+'_mean'] = np.asarray([[accessor(r)['mean'] for r in shell] for shell in ordered]).reshape(nk,nr)
        arrays[tag+'_ad'] = np.asarray([[accessor(r)['ad'] for r in shell] for shell in ordered]).reshape(nk,nr,nc)
    if not all(np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError('AP retained arrays must be finite')
    return arrays


def ap_cell_record(likelihood, cell, rows, *, space=None):
    """Bind measured mean/AD and compute the A1 full-C acceptance diagnostics."""
    names = COSMO_NAMES if space is None else space.names
    nc = len(names)
    pairs = [list(p) for p in cell.metadata['retained_pairs']]
    arrays = _cell_arrays(likelihood,cell,rows,space=space)
    covariance = np.asarray(cell.covariance)
    if (not np.all(np.isfinite(covariance)) or
            not np.allclose(covariance, covariance.transpose(0,2,1),rtol=0,atol=1e-12)):
        raise ValueError('AP assembled covariance is nonfinite or asymmetric')
    chol = np.linalg.cholesky(covariance)
    def norm(vector):
        return float(np.linalg.norm(np.linalg.solve(chol, np.asarray(vector)[...,None])[...,0]))
    physical_scale = norm(arrays['mean'])
    if physical_scale <= 0:
        raise ValueError('AP assembled physical scale must be positive')
    blocks = []
    for index in [None,*range(3*len(pairs))]:
        mask = np.ones(arrays['mean'].shape[1]) if index is None else np.eye(arrays['mean'].shape[1])[index]
        def compare(delta, reference, tolerance, zero_tolerance=None):
            difference, denominator = norm(delta*mask),norm(reference*mask)
            zero = denominator <= 1e-12*physical_scale
            limit = (tolerance if zero_tolerance is None else zero_tolerance) if zero else tolerance
            error = difference/physical_scale if zero else difference/denominator
            return {'difference_norm':difference,'reference_norm':denominator,
                    'error':error,'limit':limit,'branch':'normalized absolute' if zero else 'relative',
                    'passed':bool(error < limit)}
        fid,ad = arrays['mean'],arrays['ad']
        block = {'pair':None if index is None else pairs[index//3],
                 'ell':None if index is None else (0,2,4)[index%3],
                 'identity':compare(fid-arrays['direct'],arrays['direct'],IDENTITY_TOLERANCE),
                 'continuity':[compare(arrays['continuity'][:,:,i],fid,IDENTITY_TOLERANCE) for i in range(2)],
                 'ad_fd':[[compare(ad[:,:,i]-arrays['fd'][:,:,i,j], arrays['fd'][:,:,i,j],
                                   DERIVATIVE_TOLERANCE,1e-6) for j in range(3)] for i in range(nc)],
                 'chain':[compare(ad[:,:,i]-arrays['intrinsic'][:,:,i]-arrays['geometry'][:,:,i],
                                  ad[:,:,i],CHAIN_TOLERANCE) for i in range(nc)]}
        refinements = []
        previous_mean,previous_ad = fid,ad
        for i,nk in enumerate(likelihood.config.ap_refinement_nk):
            refined_mean,refined_ad = arrays[f'refinement_{i}_mean'],arrays[f'refinement_{i}_ad']
            refinements.append({'nk':nk,'mean':compare(refined_mean-previous_mean,refined_mean,REFINEMENT_TOLERANCE),
                                'ad':[compare(refined_ad[:,:,c]-previous_ad[:,:,c],refined_ad[:,:,c],
                                              REFINEMENT_TOLERANCE) for c in range(nc)]})
            previous_mean,previous_ad = refined_mean,refined_ad
        block['interpolation'] = refinements
        block['angular'] = {'n_mu':64,
                            'mean':compare(arrays['angular_mean']-fid,arrays['angular_mean'],REFINEMENT_TOLERANCE),
                            'ad':[compare(arrays['angular_ad'][:,:,i]-ad[:,:,i],arrays['angular_ad'][:,:,i],
                                          REFINEMENT_TOLERANCE) for i in range(nc)]}
        checks = [block['identity'],*block['continuity'],*block['chain'],block['angular']['mean'],*block['angular']['ad']]
        for refinement in refinements:
            checks.extend([refinement['mean'],*refinement['ad']])
        block['passed'] = all(c['passed'] for c in checks) and all(
            any(group[i]['passed'] and group[i+1]['passed'] for i in range(2)) for group in block['ad_fd'])
        blocks.append(block)
    return {'cell_id':cell.metadata['cell_id'],'retained_pairs':pairs,
            'k':np.asarray(likelihood.grid.centers).tolist(),'multipoles':[0,2,4],
            'cosmology_columns':list(names),'covariance':covariance.tolist(),
            'diagnostics':{'status':'passed' if all(b['passed'] for b in blocks) else 'failed',
                           'physical_scale':physical_scale,'blocks':blocks}}


def _validate_cell_record(saved, likelihood, cell, rows, *, space=None):
    names = COSMO_NAMES if space is None else space.names
    if (saved.get('retained_pairs') != [list(p) for p in cell.metadata['retained_pairs']]
            or saved.get('k') != np.asarray(likelihood.grid.centers).tolist()
            or saved.get('multipoles') != [0,2,4]
            or saved.get('cosmology_columns') != list(names)):
        raise ValueError('AP retained row/column ordering mismatch')
    _same_assembled(saved.get('covariance'),cell.covariance,'full retained covariance')
    calculated = ap_cell_record(likelihood,cell,rows,space=space)
    if _canonical(saved.get('diagnostics')) != _canonical(calculated['diagnostics']):
        raise ValueError('AP retained diagnostics differ from raw arrays')
    if calculated['diagnostics']['status'] != 'passed':
        raise ValueError('AP full retained covariance / pair-ell quantitative check failed')
    return calculated['diagnostics']


def ap_configuration(config):
    """Capture every setting that selects the physical AP point schedule."""
    return asdict(config)


def expected_ap_point_keys(config, survey, *, space=None):
    """Declare support, retained pairs and shells before inspecting evidence."""
    support = expected_support_schedule(config, survey, **({} if space is None else {"space":space}))
    centers = make_shell_grid(config.kmin, config.kmax, config.dk).centers
    return [(row["cell_id"], tuple(pair), float(k)) for row in support
            for pair in row["retained_pairs"] for k in centers]


def validate_ap_evidence(report, config, provenance, survey, run_id, *, likelihood, space=None):
    """Attach only an all-cell certificate bound to the live assembled likelihood."""
    if config.ap_interpolation_nk >= 16384:
        raise ValueError('intrinsic-table profiles require the extended response validation path')
    if space is not None:
        declared = make_cosmology_spec(likelihood.reference_cosmology,
                                      extended=config.cosmology_basis == "extended6",
                                      ap_nk=config.ap_interpolation_nk)
        if space != declared:
            raise ValueError("AP declared cosmology space mismatch")
    if (report.get("status") != "passed" or report.get("scope") != "all_cells"
            or report.get("run_id") != run_id):
        raise ValueError("AP evidence is incomplete or bound to another run")
    if (_canonical(report.get("configuration")) != _canonical(ap_configuration(config))
            or _canonical(ap_configuration(likelihood.config)) != _canonical(ap_configuration(config))):
        raise ValueError("AP evidence configuration mismatch")
    if _canonical(report.get("physical_dependencies")) != _canonical(physical_dependencies(provenance)):
        raise ValueError("AP evidence physical dependencies mismatch")
    if _canonical(report.get("reference_cosmology")) != _canonical(likelihood.reference_cosmology):
        raise ValueError("AP frozen reference differs from assembled reference")
    if report.get("metric_contract") != AP_METRIC_CONTRACT:
        raise ValueError("AP metric contract mismatch")
    grids = angular_grid_declaration(config.n_mu)
    if report.get("angular_grids") != grids:
        raise ValueError("AP angular domain grid declaration mismatch")
    validate_pure_ap_controls(report.get("pure_ap_controls"),config.ap_interpolation_nk,
                              config.ap_interpolation_nmu)
    support = expected_support_schedule(config,survey, **({} if space is None else {"space":space}))
    validate_support_schedule(likelihood,support,**({} if space is None else {"space":space}))
    ids = [entry['cell_id'] for entry in support]
    empty = [{'cell_id':entry['cell_id'],'reason':'no retained observations'}
             for entry in support if not entry['retained_pairs']]
    if (report.get('selected_cell_ids') != ids or report.get('completed_cells') != ids
            or report.get('empty_cells') != empty):
        raise ValueError('AP complete/empty support inventory mismatch')
    expected = expected_ap_point_keys(config,survey, **({} if space is None else {"space":space}))
    rows = report.get("points",[])
    cells = [cell for cell in likelihood.cells if cell.metadata['retained_pairs']]
    saved_cells = report.get('cells',[])
    if (len(saved_cells) != len(cells) or [c.get('cell_id') for c in saved_cells] !=
            [c.metadata['cell_id'] for c in cells]):
        raise ValueError('AP retained cell ledger omitted, duplicated or reordered')
    diagnostics = []
    for saved,cell in zip(saved_cells,cells):
        selected = [r for r in rows if r.get('cell_id') == cell.metadata['cell_id']]
        diagnostics.append(_validate_cell_record(saved,likelihood,cell,selected,space=space))
    if space is None:
        result = validate_ap_point_rows(rows,expected,expected_mu=np.asarray(grids['mean_projection']['mu']),
                                        ap_nk=config.ap_interpolation_nk,refinement_nk=config.ap_refinement_nk,
                                        require_grid_coverage=True,historical_only=True)
    else:
        theta = np.asarray(likelihood.fiducial)[[likelihood.names.index(n) for n in space.names]]
        for cell in cells:
            cell_id = cell.metadata['cell_id']
            selected = [r for r in rows if r['cell_id'] == cell_id]
            result = validate_ap_point_rows(selected,[key for key in expected if key[0] == cell_id],
                expected_mu=np.asarray(grids['mean_projection']['mu']),ap_nk=config.ap_interpolation_nk,
                refinement_nk=config.ap_refinement_nk,require_grid_coverage=True,space=space,
                reference=likelihood.reference_cosmology,theta=theta,redshift=cell.metadata['model_redshift'])
        # Also reject points from unretained or unknown cells outside the local checks.
        keys = [(r['cell_id'],tuple(r['pair']),float(r['k'])) for r in rows]
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError("AP complete point coverage mismatch")
        result = result | {"expected_points":len(expected),"measured_points":len(rows)}
    return result | {"metric_contract":AP_METRIC_CONTRACT,"cells":diagnostics}


def _project(pkmu, mu, weights):
    leg = jnp.stack((jnp.ones_like(mu), (3*mu**2-1)/2,
                     (35*mu**4-30*mu**2+3)/8))
    return jnp.einsum("km,lm,m,l->kl", pkmu, leg, weights,
                      jnp.asarray([1., 5., 9.]))


def measure_ap_cell(likelihood, cell, backend, *, pure_controls=None, space=None,
                    intrinsic_sink=None):
    """Measure every retained pair and observed shell on the live mean path."""
    from .one_loop import reject_legacy_ap_qualification
    reject_legacy_ap_qualification(backend)
    names = COSMO_NAMES if space is None else space.names
    steps = COSMO_STEPS if space is None else space.fd_steps
    nc = len(names)
    pairs = _checked_pairs(likelihood, cell)
    _check_shells(likelihood, cell)
    if not pairs:
        return []
    values = _cell_fiducials(likelihood, cell)
    ibin = cell.metadata["nominal_bin_index"]
    z = cell.metadata["model_redshift"]
    nbar = cell.metadata["nbar"]
    k = jnp.asarray(likelihood.grid.centers, dtype=jnp.float64)
    nodes, quadrature = np.polynomial.legendre.leggauss(likelihood.config.n_mu)
    mu = jnp.asarray((nodes+1)/2, dtype=jnp.float64)
    weights = jnp.asarray(quadrature/2, dtype=jnp.float64)
    theta = jnp.asarray([values[name] for name in names])
    if space is not None:
        validate_coordinate_neighborhood(np.asarray(theta),
            {**likelihood.reference_cosmology,"z":z},space)
    adopted = likelihood.config.ap_interpolation_nk
    comparisons = likelihood.config.ap_refinement_nk
    if adopted >= 16384 and (adopted != 16384 or nc != 6 or not callable(intrinsic_sink)):
        raise ValueError('intrinsic-table measurement requires AP16384, six coordinates and a raw-evidence sink')
    toy_result = validate_pure_ap_controls(
        measure_pure_ap_controls(adopted, likelihood.config.ap_interpolation_nmu)
        if pure_controls is None else pure_controls,
        adopted, likelihood.config.ap_interpolation_nmu)
    toy_iso = max(v for key,v in toy_result['errors'].items() if key.endswith(':isotropic'))
    toy_aniso = max(v for key,v in toy_result['errors'].items() if not key.endswith(':isotropic'))
    output = []

    for ipair, pair in enumerate(pairs):
        a, b = pair
        qa = jnp.asarray([values[f"bin{ibin:02d}:{a}:{name}"] for name in NUISANCE_NAMES])
        qb = (None if a == b else jnp.asarray([
            values[f"bin{ibin:02d}:{b}:{name}"] for name in NUISANCE_NAMES]))
        qc = (None if a == b else jnp.asarray([
            values[f"bin{ibin:02d}:{a}|{b}:{name}"] for name in CROSS_STOCH_NAMES]))
        # Match build_cell_likelihood's complete pair-local tangent layout.
        # Differentiating only theta can change floating-point operation order
        # and break exact binding to its assembled cosmological columns.
        packed = jnp.concatenate((theta, qa) if a == b else (theta, qa, qb, qc))
        responses = {size:make_pair_response(
            backend, z, likelihood.reference_cosmology, nbar[a],
            None if a == b else nbar[b], likelihood.config.k_nl,
            ap_nk=size, ap_nmu=likelihood.config.ap_interpolation_nmu,
            **({} if space is None else {"space":space}))
            for size in (adopted, *comparisons)}
        response = responses[adopted]

        def mean(p, size=adopted, angular=None, ap=True):
            first = p[nc:nc+len(NUISANCE_NAMES)]
            second = None if a == b else p[nc+len(NUISANCE_NAMES):nc+2*len(NUISANCE_NAMES)]
            stochastic = None if a == b else p[-len(CROSS_STOCH_NAMES):]
            return responses[size].multipoles(
                k, p[:nc], first, second, stochastic, ap=ap,
                n_mu=likelihood.config.n_mu if angular is None else angular)

        def cosmology_ad(function):
            return np.asarray(jax.jacfwd(function)(packed))[..., :len(names)]

        fid = np.asarray(mean(packed))
        if fid.shape != (len(k), 3) or not np.all(np.isfinite(fid)):
            raise ValueError(f"invalid fiducial AP multipoles in {cell.metadata['cell_id']} {pair}")
        total_full = np.asarray(jax.jacfwd(mean)(packed))
        intrinsic_full = np.asarray(jax.jacfwd(lambda p:mean(p, ap=False))(packed))
        total_ad = total_full[..., :nc]
        intrinsic_ad = intrinsic_full[..., :nc]
        # Freeze P_lin, growth, sigma8, nuisance backend coordinates and noise.
        frozen = response.reference
        s8 = response.sigma8(z, frozen)
        f = response.growth_rate(z, frozen)
        pa = nuisance_to_backend(qa, s8, response.sigma8_ref, f,
                                 frozen["h"], nbar[a], likelihood.config.k_nl)
        if a != b:
            pb = nuisance_to_backend(qb, s8, response.sigma8_ref, f,
                                     frozen["h"], nbar[b], likelihood.config.k_nl)
            pa = {**pa, "bias2":pb["bias"], "ctr2":pb["ctr"]}
        pk_data = response.plin(z, frozen)

        def geometry_alpha(ap, al):
            power = response.ap_backend._get_pkmu_pair_ref_interp(
                k, mu, ap, al, pk_data, pa, add_stochasticity=a == b)
            if qc is not None:
                kt, mt, jac = ap_coordinates(k, mu, ap, al)
                power = power + (qc[0] + qc[1]*(kt*mt/likelihood.config.k_nl)**2) / (
                    jnp.sqrt(nbar[a]*nbar[b])*jac)
            return _project(power, mu, weights)

        def geometry(p):
            return geometry_alpha(*response.ap_alphas(p[:nc]))

        geometry_full = np.asarray(jax.jacfwd(geometry)(packed))
        geometry_original = geometry_full[..., :nc]
        geometry_ad = geometry_original
        if adopted == 16384:
            from .revision_validation_ap_intrinsic import (
                measure_intrinsic_geometry, validate_intrinsic_geometry, packed_labels)
            record, arrays = measure_intrinsic_geometry(response, packed, k,
                n_cosmo=nc, n_mu=likelihood.config.n_mu, native_mean=fid,
                native_total=total_full, native_intrinsic=intrinsic_full)
            if not np.array_equal(arrays['geometry_original'], geometry_full):
                raise ValueError('original AP geometry changed during intrinsic-table extraction')
            record.update(cell_id=cell.metadata['cell_id'], pair=list(pair),
                          packed_labels=packed_labels(likelihood,cell,pair))
            arrays['pair_covariance'] = np.asarray(cell.covariance)[:,3*ipair:3*ipair+3,3*ipair:3*ipair+3].copy()
            validate_intrinsic_geometry(record,arrays,likelihood=likelihood,cell=cell,pair=pair)
            geometry_ad = arrays['geometry_matched'][..., :nc]
            intrinsic_sink(record,arrays)
        direct_power = backend.get_pkmu_pair(k, mu, pk_data, pa,
                                             add_stochasticity=a == b)
        if qc is not None:
            direct_power = direct_power + (qc[0] + qc[1]*(k[:, None]*mu[None, :]/
                likelihood.config.k_nl)**2) / jnp.sqrt(nbar[a]*nbar[b])
        direct = np.asarray(_project(direct_power, mu, weights))
        alpha_ad = cosmology_ad(lambda p:response.ap_alphas(p[:nc]))
        alpha_fd = np.empty((nc, 3), dtype=float)
        fd = np.empty((nc, 3, len(k), 3), dtype=float)
        for i, name in enumerate(names):
            for j, step in enumerate(steps[name]):
                plus = packed.at[i].add(step)
                minus = packed.at[i].add(-step)
                if space is not None:
                    validate_physical_point(np.asarray(plus[:nc]),likelihood.reference_cosmology,space,z)
                    validate_physical_point(np.asarray(minus[:nc]),likelihood.reference_cosmology,space,z)
                fd[i, j] = (np.asarray(mean(plus)) - np.asarray(mean(minus))) / (2*step)
                alpha_central = (np.asarray(response.ap_alphas(plus[:nc])) -
                                 np.asarray(response.ap_alphas(minus[:nc]))) / (2*step)
                alpha_fd[i, j] = float(_response_error(np.linalg.norm,alpha_ad[:,i]-alpha_central,
                    alpha_ad[:,i] if space is None else alpha_central,1.,space=space))
        refined = {}
        for size in comparisons:
            refined[size] = (np.asarray(mean(packed, size=size)),
                             cosmology_ad(lambda p:mean(p, size=size)))
        angular = (np.asarray(mean(packed, angular=64)),
                   cosmology_ad(lambda p:mean(p, angular=64)))
        eps = 1e-5
        continuity = []
        for ap, al in ((1+eps, 1.), (1., 1+eps)):
            opposite = (2-ap, 2-al)
            continuity.append(np.asarray(geometry_alpha(ap, al)) +
                              np.asarray(geometry_alpha(*opposite)) - 2*fid)
        grid_declaration = angular_grid_declaration(likelihood.config.n_mu)
        domain_spectra = []
        other_spectra = {role:[] for role in ("covariance", "refined_projection")}
        for name, sign, step in ap_parameter_schedule(space=space):
            varied = theta if name == "fiducial" else theta.at[names.index(name)].add(sign*step)
            point_metadata = {}
            if space is not None:
                point_metadata = {"theta":np.asarray(varied).tolist(),"mapped_cosmology":
                    validate_physical_point(np.asarray(varied),likelihood.reference_cosmology,space,z)}
            alphas = np.asarray(response.ap_alphas(varied), dtype=float)
            spectrum = np.asarray(response.pkmu(k, mu, varied, qa, qb, qc), dtype=float)
            if spectrum.shape != (len(k), len(mu)) or not np.all(np.isfinite(spectrum)):
                raise ValueError(f"AP response shape/nonfinite: {cell.metadata['cell_id']} {pair} "
                                 f"{name} sign={sign} step={step}")
            domain_spectra.append((name, sign, step, alphas, spectrum, point_metadata))
            for role, records in other_spectra.items():
                grid_mu = jnp.asarray(grid_declaration[role]["mu"])
                power = np.asarray(response.pkmu(k, grid_mu, varied, qa, qb, qc))
                if power.shape != (len(k), len(grid_mu)) or not np.all(np.isfinite(power)):
                    raise ValueError(f"AP angular domain nonfinite/shape: {cell.metadata['cell_id']} {pair} {role} {name} {sign} {step}")
                records.append((name, sign, step, alphas, power, point_metadata))

        for ik, kval in enumerate(np.asarray(k)):
            covariance = cell.covariance[ik, 3*ipair:3*ipair+3,
                                         3*ipair:3*ipair+3]
            chol = np.linalg.cholesky(covariance)

            def norm(vector):
                return float(np.linalg.norm(np.linalg.solve(chol, np.asarray(vector))))

            physical_scale = max(norm(fid[ik]), np.finfo(float).tiny)

            def error(delta, reference, *, absolute_fraction=1e-6):
                return _response_error(norm,delta,reference,physical_scale,
                                       space=space,fraction=absolute_fraction)

            previous_value, previous_ad = fid, total_ad
            interpolation = []
            for size in comparisons:
                value, derivative = refined[size]
                interpolation.append({"nk":size,
                    "mean_error":error(value[ik]-previous_value[ik], value[ik]),
                    "derivative_errors":[error(derivative[ik, :, i]-previous_ad[ik, :, i],
                                                derivative[ik, :, i]) for i in range(nc)],
                    "mean":value[ik].tolist(), "ad":derivative[ik].tolist()})
                previous_value, previous_ad = value, derivative
            angular_value, angular_ad = angular
            row = {**({} if space is None else {"cosmology_columns":list(names)}),
                   "cell_id":cell.metadata["cell_id"], "pair":list(pair),
                   "k":float(kval), "mu":np.asarray(mu).tolist(),
                   "shape":[1, len(mu)],
                   "domain":_domain_record(kval, mu, domain_spectra, ik),
                   "observed_domains":{role:_domain_record(kval, grid_declaration[role]["mu"], records, ik)
                                       for role,records in other_spectra.items()},
                   "fiducial_multipoles":fid[ik].tolist(),
                   "direct_multipoles":direct[ik].tolist(),
                   "pair_covariance":np.asarray(covariance).tolist(),
                   "total_ad":total_ad[ik].tolist(),
                   "intrinsic_ad":intrinsic_ad[ik].tolist(),
                   "geometry_ad":geometry_ad[ik].tolist(),
                   "fd_derivatives":fd[:, :, ik, :].tolist(),
                   "alpha_derivatives":alpha_ad.tolist(),
                   "continuity_deltas":[value[ik].tolist() for value in continuity],
                   "whitened_physical_scale":physical_scale,
                   "identity_error":error(fid[ik]-direct[ik], direct[ik]),
                   "continuity_error":max(error(value[ik], fid[ik], absolute_fraction=1.)
                                          for value in continuity),
                   "pure_ap_isotropic_error":toy_iso,
                   "pure_ap_anisotropic_error":toy_aniso,
                   "ad_fd_errors":[[error(total_ad[ik, :, i]-fd[i, j, ik],
                                           total_ad[ik, :, i] if space is None else fd[i,j,ik]) for j in range(3)]
                                   for i in range(nc)],
                   "chain_errors":[error(total_ad[ik, :, i]-intrinsic_ad[ik, :, i]-
                                          geometry_ad[ik, :, i], total_ad[ik, :, i])
                                   for i in range(nc)],
                   "alpha_fd_errors":alpha_fd.tolist(),
                   "interpolation":interpolation,
                   "angular":{"n_mu":64,
                              "mean_error":error(angular_value[ik]-fid[ik], angular_value[ik]),
                              "derivative_errors":[error(angular_ad[ik, :, i]-total_ad[ik, :, i],
                                                          angular_ad[ik, :, i]) for i in range(nc)],
                              "mean":angular_value[ik].tolist(),
                              "ad":angular_ad[ik].tolist()}}
            if adopted == 16384:
                row.update(decomposition='intrinsic-table-v1',
                    geometry_ad_original=geometry_original[ik].tolist(),
                    chain_errors_original=[error(total_ad[ik,:,i]-intrinsic_ad[ik,:,i]-
                        geometry_original[ik,:,i],total_ad[ik,:,i]) for i in range(nc)])
            output.append(row)
    return output


def run_ap_probe(config, provenance, survey, cosmo, backend, root, run_id, *, max_cells=0, space=None):
    """Persist per-cell AP measurements atomically, including failed work."""
    if config.ap_interpolation_nk >= 16384:
        raise ValueError('intrinsic-table profiles require the extended response validation path')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.ap-evidence.json"
    if path.exists():
        raise FileExistsError(f"preserving existing AP evidence: {path}")
    support = expected_support_schedule(config, survey, **({} if space is None else {"space":space}))
    selected = [row["cell_id"] for row in support]
    if max_cells:
        selected = selected[:max_cells]
    report = {"status":"running", "scope":"all_cells" if len(selected) == len(support)
              else "representative", "run_id":run_id,
              "configuration":ap_configuration(config),
              "physical_dependencies":physical_dependencies(provenance),
              "selected_cell_ids":selected,
              "empty_cells":[{"cell_id":row["cell_id"],
                              "reason":"no retained observations"
                              if not row["retained_pairs"] else None}
                             for row in support if not row["retained_pairs"]],
              "points":[], "completed_cells":[], "cells":[],
              "metric_contract":AP_METRIC_CONTRACT,
              "angular_grids":angular_grid_declaration(config.n_mu),
              "pure_ap_controls":[]}

    def save():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)

    save()
    started = time.monotonic()
    try:
        report['pure_ap_controls'] = measure_pure_ap_controls(
            config.ap_interpolation_nk,config.ap_interpolation_nmu)
        validate_pure_ap_controls(report['pure_ap_controls'],config.ap_interpolation_nk,
                                  config.ap_interpolation_nmu)
        save()
        likelihood = build_revision_likelihood(config, survey, cosmo, backend)
        report['reference_cosmology'] = likelihood.reference_cosmology
        validate_support_schedule(likelihood,support,**({} if space is None else {"space":space}))
        by_id = {cell.metadata["cell_id"]:cell for cell in likelihood.cells}
        support_by_id = {row["cell_id"]:row for row in support}
        centers = make_shell_grid(config.kmin, config.kmax, config.dk).centers
        angular_nodes, _ = np.polynomial.legendre.leggauss(config.n_mu)
        expected_mu = (angular_nodes+1)/2
        for cell_id in selected:
            report["active_cell"] = cell_id
            save()
            measured = measure_ap_cell(likelihood, by_id[cell_id], backend,
                                       pure_controls=report["pure_ap_controls"],
                                       **({} if space is None else {"space":space}))
            report["points"].extend(measured)
            expected = [(cell_id, tuple(pair), float(k))
                        for pair in support_by_id[cell_id]["retained_pairs"]
                        for k in centers]
            if expected:
                record = ap_cell_record(likelihood,by_id[cell_id],measured,space=space)
                report['cells'].append(record)
                save()
                validate_ap_point_rows(measured, expected, expected_mu=expected_mu,
                                       ap_nk=config.ap_interpolation_nk,
                                       refinement_nk=config.ap_refinement_nk,
                                       require_grid_coverage=True,historical_only=space is None,
                                       space=space,reference=likelihood.reference_cosmology,
                                       theta=None if space is None else np.asarray(likelihood.fiducial)[
                                           [likelihood.names.index(n) for n in space.names]],
                                       redshift=by_id[cell_id].metadata.get('model_redshift'))
                _validate_cell_record(record,likelihood,by_id[cell_id],measured,space=space)
            elif measured:
                raise ValueError(f"unexpected AP points in empty cell {cell_id}")
            report["completed_cells"].append(cell_id)
            save()
        if report["scope"] == "all_cells":
            result = validate_ap_evidence(report | {"status":"passed"}, config,
                                          provenance, survey, run_id,likelihood=likelihood,space=space)
            report["validation"] = result
            report["status"] = "passed"
        else:
            report["status"] = "pending"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"exception_type":type(exc).__name__,
                             "exception":str(exc), "traceback":traceback.format_exc()}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        save()
    return path
