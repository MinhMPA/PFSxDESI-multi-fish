#!/usr/bin/env python
"""Bounded physical ELF checks; this is not a full-survey Fisher certificate.

Measure real spectra and complete cell derivatives at z=1.1 and 1.8, using
the accepted covariance metric and all three finite-difference steps. The
high-z PFS cell is explicitly a sensitivity case, not a baseline extension.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ps_1loop_jax"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog.cosmo import FIDUCIAL, FiducialCosmology
from pfsfog.cosmology_space import CosmologySpec, make_cosmology_spec, validate_coordinate_neighborhood
from pfsfog.eft_params import CROSS_STOCH_NAMES, tracer_fiducials
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_validation_zeros import counterterm_zero_support
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_forecast import RevisionConfig, build_cell_likelihood, make_shell_grid
from pfsfog.revision_theory import (NUISANCE_NAMES, ap_coordinates, fiducial_nuisance,
                                   make_pair_response, nuisance_to_backend)
from pfsfog.revision_validation import compare_cell_derivatives
from pfsfog.revision_validation_physical import _steps
from pfsfog.survey_geometry import support_cells
from pfsfog.surveys import default_survey_group


def diagnostic_cells(cosmo, config):
    group = default_survey_group()
    output = []
    for nominal, scope in [((1., 1.2), "baseline_overlap_representative"),
                            ((1.6, 2.), "highz_sensitivity_only")]:
        selected = group if scope.startswith("baseline") else replace(group, pfs_zmax=2.1)
        cells = support_cells(selected, nominal)
        # The first subcell retains the LRG population at z=1.1.
        cell = cells[0]
        names = sorted({name for c in cells for name in c.overlap_tracers})
        z = cell.model_redshift
        b_ref = selected.desi_tracers["DESI-ELG"].b1_of_z(z) if "DESI-ELG" in names else 1.3
        fids = {name: tracer_fiducials(name, selected.all_surveys[name].b1_of_z(z),
                    float(cosmo.sigma8(z)), b1_ref=b_ref, r_sigma_v=config.r_sigma_v)
                for name in names}
        output.append({"label": f"z{z:.1f}", "scope": scope, "cell": cell, "fiducials": fids})
    return output


def pair_functions(response, k, packed, *, n_mu):
    """Total, intrinsic-only and frozen-intrinsic AP means in the same basis."""
    nc, nq = len(response.space.names), len(NUISANCE_NAMES)
    cross = response.nbar_b is not None

    def split(v):
        return (v[:nc], v[nc:nc+nq], v[nc+nq:nc+2*nq] if cross else None,
                v[-2:] if cross else None)

    def total(v):
        return response.multipoles(k, *split(v), n_mu=n_mu)

    def intrinsic(v):
        return response.multipoles(k, *split(v), ap=False, n_mu=n_mu)

    theta, qa, qb, qc = split(packed)
    cosmology = response._cosmology(theta)
    sigma8 = response.sigma8(response.z, cosmology)
    growth = response.growth_rate(response.z, cosmology)
    params = nuisance_to_backend(qa, sigma8, response.sigma8_ref, growth,
                                 cosmology["h"], response.nbar_a, response.k_nl)
    if cross:
        second = nuisance_to_backend(qb, sigma8, response.sigma8_ref, growth,
                                     cosmology["h"], response.nbar_b, response.k_nl)
        params = {**params, "bias2": second["bias"], "ctr2": second["ctr"]}
    linear = response.plin(response.z, cosmology)
    nodes, weights = np.polynomial.legendre.leggauss(n_mu)
    mu, weights = jnp.array((nodes+1)/2), jnp.array(weights/2)
    leg = jnp.stack((jnp.ones_like(mu), (3*mu**2-1)/2, (35*mu**4-30*mu**2+3)/8))

    def geometry(v):
        ap, al = response.ap_alphas(v[:nc])
        power = response.ps.get_pkmu_pair_ref(k, mu, ap, al, linear, params,
                                              add_stochasticity=not cross)
        if cross:
            kt, mt, jacobian = ap_coordinates(k, mu, ap, al)
            power = power + (qc[0]+qc[1]*(kt*mt/response.k_nl)**2) / (
                np.sqrt(response.nbar_a*response.nbar_b)*jacobian)
        return jnp.einsum("km,lm,m,l->kl", power, leg, weights, jnp.array([1., 5., 9.]))

    return {"total": jax.jit(total), "intrinsic": jax.jit(intrinsic),
            "geometry": jax.jit(geometry)}


def _white(covariance, vectors):
    return np.linalg.solve(np.linalg.cholesky(covariance), vectors)


def _write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+"\n")


def _mean_evaluators(pair_data, names, *, component="total", h_only=False):
    evaluators = {}
    for pair, item in pair_data.items():
        cache = {}
        def mean(coordinates, item=item, cache=cache):
            v = np.array(item["packed"], copy=True)
            if h_only:
                v[3] = coordinates["h"]
            else:
                v = np.asarray([coordinates[names[i]] for i in item["indices"]])
            key = tuple(v)
            if key not in cache:
                cache[key] = np.asarray(item["functions"][component](jnp.asarray(v)))
            return cache[key]
        evaluators[pair] = mean
    return evaluators


def _fd_arrays(cell, means, fiducials, steps):
    pairs = [tuple(p) for p in cell.metadata["retained_pairs"]]
    plus = np.empty((len(cell.names), 3, *cell.fiducial_mean.shape))
    minus = np.empty_like(plus)
    for i, name in enumerate(cell.names):
        for j, step in enumerate(steps[name]):
            for sign, destination in [(1, plus), (-1, minus)]:
                point = {**fiducials, name: fiducials[name]+sign*step}
                destination[i, j] = np.concatenate([means[pair](point) for pair in pairs], axis=1)
    return {"plus": plus, "minus": minus,
            "steps": np.array([steps[n] for n in cell.names]),
            "derivatives": (plus-minus)/(2*np.array([steps[n] for n in cell.names])[:, :, None, None])}


def measure_cell(case, config, cosmo, engine, output):
    """Save the live assembled D,C and all independent signed FD means."""
    support, fids = case["cell"], case["fiducials"]
    region = case.get("region", "overlap")
    cell = build_cell_likelihood(config, support, region, fids, cosmo, engine)
    ibin = cell.metadata["nominal_bin_index"]
    cell.metadata["cell_id"] = case.get("cell_id", f"elf-{case['label']}-radial0-{region}")
    space = elf_cosmology_spec(cosmo.params)
    theta = np.array([cosmo.params[{"Mnu": "mnu", "Omegam": "Omega_m"}.get(n, n)]
                      for n in space.names])
    domain = validate_coordinate_neighborhood(theta, {**cosmo.params, "z": support.model_redshift}, space)
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    q = {n: np.asarray(fiducial_nuisance(fids[n])) for n in cell.metadata["active_populations"]}
    values = dict(zip(space.names, theta))
    for tracer in q:
        values.update({f"bin{ibin:02d}:{tracer}:{name}": value for name, value in zip(NUISANCE_NAMES, q[tracer])})
    pair_data = {}
    intrinsic = np.zeros_like(cell.derivatives)
    geometric = np.zeros_like(cell.derivatives)
    spectra = []
    signed_mu = np.polynomial.legendre.leggauss(config.n_mu)[0]
    for ipair, pair in enumerate(cell.metadata["retained_pairs"]):
        a, b = pair
        cross = a != b
        response = make_pair_response(engine, support.model_redshift, cosmo.params,
            support.nbar[a], support.nbar[b] if cross else None, config.k_nl, space=space)
        local = list(space.names)+[f"bin{ibin:02d}:{a}:{n}" for n in NUISANCE_NAMES]
        if cross:
            local += [f"bin{ibin:02d}:{b}:{n}" for n in NUISANCE_NAMES]
            labels = [f"bin{ibin:02d}:{a}|{b}:{n}" for n in CROSS_STOCH_NAMES]
            local += labels
            values.update(dict.fromkeys(labels, 0.))
        packed = jnp.array([values[n] for n in local])
        functions = pair_functions(response, jnp.array(grid.centers), packed, n_mu=config.n_mu)
        indices = [cell.names.index(n) for n in local]
        pair_data[tuple(pair)] = {"functions": functions, "indices": indices, "packed": np.asarray(packed)}
        rows = slice(3*ipair, 3*ipair+3)
        for component, dest in [("intrinsic", intrinsic), ("geometry", geometric)]:
            derivative = np.asarray(jax.jacfwd(functions[component])(packed))
            dest[:, rows, indices] = derivative
        spectra.append(np.asarray(response.pkmu(grid.centers, signed_mu, theta, q[a],
                         q[b] if cross else None, jnp.zeros(2) if cross else None)))
    means = _mean_evaluators(pair_data, cell.names)
    alphas = [float(value) for value in response.ap_alphas(theta)]
    zero_support = counterterm_zero_support(cell.names, cell.metadata["retained_pairs"],
        alphas=alphas, settings=engine.numerical_settings())
    steps = {n: _steps(n, values[n], space=space) for n in cell.names}
    scale = float(np.linalg.norm(_white(cell.covariance, cell.fiducial_mean[..., None])))
    arrays = {"k": grid.centers, "dk": grid.widths, "signed_mu": signed_mu,
              "pkmu": np.stack(spectra), "mean": cell.fiducial_mean,
              "D": cell.derivatives, "C": cell.covariance, "F_like": cell.F_like,
              "D_intrinsic": intrinsic, "D_geometry": geometric,
              "fiducial": np.array([values[n] for n in cell.names])}
    linear = response.plin(response.z, response._cosmology(jnp.array(theta)))
    arrays.update(linear_k=np.asarray(linear["k"]), linear_pk=np.asarray(linear["pk"]))
    _write_json(output / "inputs.json", {"scope": case["scope"], "configuration": asdict(config),
        "metadata": cell.metadata, "fiducials": values, "steps": steps, "domain": domain,
        "engine_settings": engine.numerical_settings(), "units": space.unit_convention,
        "h_ref": space.h_ref, "inactive_ps_controls": ["ap_interpolation_nk", "ap_refinement_nk", "ap_interpolation_nmu"],
        "analytic_zero_response": {"alphas": alphas,
            "condition": "identity AP, linear isotropic counterterms, Q0/Q2/Q4 Legendre support",
            "support": [[name, list(pair), ell] for name, pair, ell in zero_support]}})
    np.savez_compressed(output / "arrays.npz", **arrays)
    print(f"{case['label']}: assembled {len(cell.names)} columns, {len(spectra)} pairs, {len(grid.centers)} shells", flush=True)
    fd = _fd_arrays(cell, means, values, steps)
    arrays.update({f"fd_{key}": value for key, value in fd.items()})
    total_check = compare_cell_derivatives(cell, means, values, steps,
        configuration=case["label"], physical_scale=scale, space=space,
        analytic_zero_support=zero_support)
    splits = {}
    for component, derivative in [("intrinsic", intrinsic), ("geometry", geometric)]:
        hcell = replace(cell, names=["h"], derivatives=derivative[..., 3:4])
        hmeans = _mean_evaluators(pair_data, cell.names, component=component, h_only=True)
        hvalues, hsteps = {"h": values["h"]}, {"h": steps["h"]}
        hfd = _fd_arrays(hcell, hmeans, hvalues, hsteps)
        arrays.update({f"h_{component}_{key}": value for key, value in hfd.items()})
        splits[component] = compare_cell_derivatives(hcell, hmeans, hvalues, hsteps,
            configuration=f"{case['label']}-{component}-h", physical_scale=scale, space=space)
    chain = _white(cell.covariance, cell.derivatives-intrinsic-geometric)
    reference = _white(cell.covariance, cell.derivatives)
    chain_error = float(np.linalg.norm(chain)/np.linalg.norm(reference))
    status = "passed" if total_check["status"] == "passed" and all(
        x["status"] == "passed" for x in splits.values()) and chain_error < 1.e-10 else "failed"
    result = {"status": status, "scope": case["scope"], "metadata": cell.metadata,
              "covariance": cell.validation, "ad_fd": total_check, "h_split_fd": splits,
              "chain_relative_error": chain_error, "chain_tolerance": 1.e-10}
    np.savez_compressed(output / "arrays.npz", **arrays)
    result["arrays_sha256"] = hashlib.sha256((output / "arrays.npz").read_bytes()).hexdigest()
    _write_json(output / "measurement.json", result)
    return cell, result


def refinement_difference(base, changed):
    """Covariance-weighted local diagnostics, never marginalized forecast errors."""
    if base.names != changed.names or base.derivatives.shape != changed.derivatives.shape:
        raise ValueError("refinement changed physical observations or coordinates")
    ref = _white(base.covariance, base.derivatives)
    delta = _white(base.covariance, changed.derivatives-base.derivatives)
    den = np.linalg.norm(ref, axis=(0, 1))
    absolute = np.linalg.norm(delta, axis=(0, 1))
    by_column = {name: (float(num/norm) if norm > 0 else None)
                 for name, num, norm in zip(base.names, absolute, den)}
    chol = np.linalg.cholesky(base.covariance)
    cov_left = np.linalg.solve(chol, changed.covariance-base.covariance)
    whitened_cov = np.linalg.solve(chol, cov_left.swapaxes(-1, -2)).swapaxes(-1, -2)
    mean_scale = np.linalg.norm(_white(base.covariance, base.fiducial_mean[..., None]))
    return {"status": "measured_not_full_fisher_qualified", "derivative_relative_by_column": by_column,
            "maximum_derivative_relative": max(x for x in by_column.values() if x is not None),
            "mean_relative": float(np.linalg.norm(_white(base.covariance,
                (changed.fiducial_mean-base.fiducial_mean)[..., None]))/mean_scale),
            "covariance_maximum_spectral_change": float(max(np.linalg.norm(x, ord=2) for x in whitened_cov)),
            "zero_column_absolute_change": {name: float(num) for name, num, norm in zip(base.names, absolute, den) if norm == 0}}


def finish_status(report, case_labels, requested_refinements):
    """A missing measurement or changed source cannot become a success."""
    coverage = set(report["cases"]) == set(case_labels)
    coverage &= all(x["status"] == "passed" for x in report["cases"].values())
    refinements = len(report["refinements"]) == requested_refinements
    report["refinement_diagnostics"] = ("not_requested" if requested_refinements == 0
        else "measured" if refinements else "incomplete")
    report["status"] = "passed" if coverage and refinements and not report.get("source_changed", False) else "failed"
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--n-mu", type=int, default=32)
    parser.add_argument("--dk", type=float, default=.005)
    parser.add_argument("--no-refinements", action="store_true", help="Record refinement diagnostics as pending")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    from scripts.run_revision import collect_provenance
    engine = make_one_loop_engine(nfft=args.nfft)
    provenance = collect_provenance(engine, requested_dk=args.dk)
    provenance["measurement_source"] = {"path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_json(args.output / "provenance.json", provenance)
    config = RevisionConfig("C", cosmology_basis="extended6", dk=args.dk, n_mu=args.n_mu)
    cosmo = FiducialCosmology()
    report = {"status": "running", "scope": "two_representative_overlap_cells",
              "full_forecast_qualification": "pending", "cases": {}, "refinements": {},
              "not_measured": ["all-cell shell convergence", "all prior choices and nuisance dispositions",
                               "marginalized cosmology and correlated growth convergence", "13-configuration production"]}
    cases = diagnostic_cells(cosmo, config)
    for case in cases:
        dest = args.output / case["label"]
        dest.mkdir()
        try:
            base, measurement = measure_cell(case, config, cosmo, engine, dest)
            report["cases"][case["label"]] = {"status": measurement["status"], "path": str(dest),
                "chain_relative_error": measurement["chain_relative_error"],
                "ad_fd_failures": len(measurement["ad_fd"]["failures"])}
            if not args.no_refinements:
                for loop_factor, angular_factor in [(2, 1), (4, 1), (1, 2), (1, 4)]:
                    changed_engine = make_one_loop_engine(nfft=args.nfft*loop_factor)
                    changed_config = replace(config, n_mu=args.n_mu*angular_factor)
                    changed = build_cell_likelihood(changed_config, case["cell"], "overlap",
                                                   case["fiducials"], cosmo, changed_engine)
                    label = f"fft{args.nfft*loop_factor}-mu{args.n_mu*angular_factor}"
                    np.savez_compressed(dest / f"{label}.npz", D=changed.derivatives, C=changed.covariance,
                                        mean=changed.fiducial_mean, F_like=changed.F_like)
                    report["refinements"][case["label"]+":"+label] = {
                        **refinement_difference(base, changed), "settings": changed_engine.numerical_settings(),
                        "n_mu": changed_config.n_mu, "covariance": changed.validation,
                        "arrays_sha256": hashlib.sha256((dest / f"{label}.npz").read_bytes()).hexdigest()}
                    print(f"{case['label']}: {label} measured", flush=True)
        except Exception as exc:
            (dest / "failure.log").write_text(traceback.format_exc())
            report["cases"][case["label"]] = {"status": "failed", "reason": str(exc), "path": str(dest)}
            print(f"{case['label']}: FAILED: {exc}", flush=True)
        _write_json(args.output / "summary.json", report)
    final_source = collect_provenance(engine, requested_dk=args.dk)
    report["source_changed"] = any(provenance[key] != final_source[key] for key in (
        "source_file_hashes", "backend_file_hashes", "background_file_hashes",
        "emulator_input_hashes", "dependency_hash", "survey_input_hash"))
    report["source_changed"] |= provenance["measurement_source"]["sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["elapsed_seconds"] = time.monotonic()-start
    exit_code = finish_status(report, [c["label"] for c in cases], 0 if args.no_refinements else 4*len(cases))
    _write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
