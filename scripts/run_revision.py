#!/usr/bin/env python
"""Inventory or explicitly execute the frozen DESI/PFS revision configurations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "ps_1loop_jax"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

from pfsfog.cosmo import FIDUCIAL, FiducialCosmology
from pfsfog.revision_artifacts import (MANDATORY_GATES, _check_external_gates,
                                       _configuration, _strict_json,
                                       finalize_revision_artifacts, likelihood_identity,
                                       load_cached_revision_likelihood, run_identity,
                                       write_revision_artifacts)
from pfsfog.revision_forecast import (CASES, RevisionConfig,
                                      apply_revision_priors, build_revision_likelihood)
from pfsfog.surveys import default_survey_group


def inventory(*, dk=.005, n_mu=32, requested_dk=.005, ap_nk=2048):
    """Five baseline cases and four B/C robustness pairs: 13 runs, 9 likelihoods."""
    ap_settings = {"ap_interpolation_nk": ap_nk,
                   "ap_refinement_nk": (2*ap_nk, 4*ap_nk)}
    configs = [RevisionConfig(case, kmax=.20, dk=dk, n_mu=n_mu, **ap_settings)
               for case in CASES]
    configs += [RevisionConfig(case, kmax=.15, dk=dk, n_mu=n_mu, **ap_settings)
                for case in ("B", "C")]
    configs += [RevisionConfig(case, kmax=.20, dk=dk, n_mu=n_mu,
                               nuisance_factor=factor, **ap_settings)
                for factor in (3., 10.) for case in ("B", "C")]
    configs += [RevisionConfig(case, kmax=.20, dk=dk, n_mu=n_mu,
                               exclude_elg1=True, **ap_settings) for case in ("B", "C")]
    assert len(configs) == 13
    return configs


def backend_module_path():
    from pfsfog.one_loop import elf_source_identity
    return Path(elf_source_identity()["module"])


def _load_backend():
    from pfsfog.one_loop import make_one_loop_engine
    return make_one_loop_engine()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_hashes(paths):
    return {str(path.resolve()):_hash(path) for path in sorted(set(paths))}


def _digest_mapping(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def collect_provenance(backend, *, requested_dk=.005, command=None, ap_nk=2048):
    """Hash local sources and released inputs afresh for content identity."""
    source_files = list((PROJECT_ROOT / "pfsfog").glob("*.py"))
    source_files.append(Path(__file__))
    source_files.append(PROJECT_ROOT / "scripts/validate_revision.py")
    source_files.append(PROJECT_ROOT / "pfsfog/elf_source.json")
    source_hashes = _file_hashes(source_files)
    backend_files = [path for path in BACKEND_ROOT.rglob("*")
                     if path.is_file() and path.suffix in (".py", ".txt")]
    background_hashes = _file_hashes(backend_files)
    import ps_1loop_jax
    background_module = Path(ps_1loop_jax.__file__).resolve()
    if not background_module.is_relative_to(BACKEND_ROOT.resolve()):
        raise RuntimeError("background source differs from the retained local calculation")
    is_elf = getattr(backend, "native_full_mean_ap", False)
    engine_source = None
    if is_elf:
        from pfsfog.one_loop import elf_source_identity
        engine_source = elf_source_identity()
    backend_hashes = engine_source["file_hashes"] if is_elf else background_hashes
    from pfsfog.emulator_paths import active_emulator_inputs,active_emulator_provenance
    network_hashes = active_emulator_inputs()
    emulator_provenance = active_emulator_provenance()
    survey_paths = list((PROJECT_ROOT / "survey_specs").rglob("*.txt"))
    survey_paths.append(PROJECT_ROOT / "configs/default.yaml")
    survey_hashes = _file_hashes(survey_paths)
    versions = {}
    for package in ("numpy", "scipy", "jax", "jaxlib", "cosmopower-jax", "PyYAML",
                    "sympy", "interpax", "quadax", "equinox"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    executed = {}
    for name, module in sorted(sys.modules.items()):
        # Gate validation is not on the likelihood path. Its source is already
        # covered by source_file_hashes; importing it while preparing evidence
        # must not change the producer's exact run identity.
        if name in ("pfsfog.revision_validation",
                    "pfsfog.revision_validation_physical",
                    "pfsfog.revision_validation_suite",
                    "pfsfog.revision_validation_ap"):
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if path.suffix == ".py" and (path.is_relative_to(PROJECT_ROOT / "pfsfog")
                                     or path.is_relative_to(BACKEND_ROOT)
                                     or (is_elf and path.is_relative_to(engine_source["root"]))):
            executed[name] = {"path":str(path), "sha256":_hash(path)}
    return {"project_head":head,
            "source_content_hash":_digest_mapping(source_hashes),
            "source_file_hashes":source_hashes,
            "backend_snapshot":_digest_mapping(backend_hashes),
            "backend_file_hashes":backend_hashes,
            "engine_source":engine_source,
            "background_file_hashes":background_hashes,
            "executed_module_hashes":executed,
            "network_hash":_digest_mapping(network_hashes),
            "emulator_input_hashes":network_hashes,
            "active_emulator_provenance":emulator_provenance,
            "survey_input_hash":_digest_mapping(survey_hashes),
            "survey_input_hashes":survey_hashes,
            "dependency_hash":_digest_mapping(versions),
            "package_versions":versions,
            "interpreter":{"path":sys.executable, "version":sys.version.split()[0]},
            "precision":"float64",
            "reference_cosmology":dict(FIDUCIAL),
            "requested_dk":float(requested_dk),
            "thread_settings":{key:os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "XLA_FLAGS")},
            "process_settings":{"mode":"serial", "workers":1},
            "backend_path":engine_source["module"] if is_elf else str(background_module),
            "model_flags":{"backend_class":type(backend).__name__,
                           "do_irres":backend.do_irres,
                           "subtract_k0_limit":backend.subtract_k0_limit,
                           "fft_kmin":float(backend._kmin),
                           "fft_kmax":float(backend._kmax),
                           "fft_n":int(backend._nfft),
                           "ap_interpolation_nk":ap_nk,
                           "ap_refinement_nk":[2*ap_nk, 4*ap_nk],
                           "ap_interpolation_nmu":101,
                           "fixed_observed_geometry":True,
                           "fixed_covariance":True,
                           "covariance_derivative_trace":False,
                           **(backend.numerical_settings() if is_elf else {})},
            "command":command or " ".join(sys.argv)}


def load_gates(path):
    """Read exact-run evidence and bind passed records to their journals."""
    bundle = _strict_json(Path(path).read_text())
    if not isinstance(bundle, dict):
        raise ValueError("gate file must be a JSON object")
    runs = bundle.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("gate file must contain run-bound evidence in a runs mapping")
    for run_id, gates in runs.items():
        if not isinstance(gates, dict):
            raise ValueError(f"run {run_id} has no gate records")
        _check_external_gates(gates, run_id=run_id, require_passed=False)
    return bundle


def gates_for_run(config, provenance, bundle):
    if bundle is None:
        return {}
    requested = provenance.get("requested_dk", config.dk)
    configuration = _configuration(config, config.nuisance_factor,
                                   requested_dk=requested,
                                   model_flags=provenance.get("model_flags"))
    run_id = run_identity(configuration, provenance)
    gates = bundle.get("runs", {}).get(run_id)
    if gates is None:
        raise ValueError(f"gate file has no exact run identity for {run_id}")
    return gates


def _failure_context(config, exc):
    message = str(exc)
    match = re.search(r"cell=(.*?) k=([0-9.eE+-]+)", message)
    return {"case":config.case, "kmin":config.kmin, "kmax":config.kmax,
            "nominal_bins":list(config.nominal_bins),
            "cell_context":match.group(1) if match else None,
            "k":float(match.group(2)) if match else None,
            "exception_type":type(exc).__name__, "exception":message,
            "traceback":traceback.format_exc()}


def execute(configs, output_root, gate_bundle, *, requested_dk=.005, command=None):
    """Run validation or gated production, caching matching likelihoods."""
    backend = _load_backend()
    if not configs or len({config.ap_interpolation_nk for config in configs}) != 1:
        raise ValueError("producer requires one declared AP table size")
    provenance = collect_provenance(backend, requested_dk=requested_dk,
                                    command=command, ap_nk=configs[0].ap_interpolation_nk)
    plans = []
    for config in configs:
        gates = gates_for_run(config, provenance, gate_bundle)
        identity = run_identity(_configuration(config, config.nuisance_factor,
                                               requested_dk=provenance.get("requested_dk", config.dk),
                                               model_flags=provenance.get("model_flags")),
                                provenance)
        target = Path(output_root) / identity
        if target.exists():
            raise FileExistsError(f"preserving existing run directory: {target}")
        plans.append((config, gates))
    survey = default_survey_group()
    cosmo = FiducialCosmology()
    cache = {}
    completed = []
    for config, gates in plans:
        identity = likelihood_identity(
            _configuration(config, config.nuisance_factor,
                           requested_dk=provenance.get("requested_dk", config.dk),
                           model_flags=provenance.get("model_flags")), provenance)
        try:
            likelihood = cache.get(identity)
            if likelihood is None and config.nuisance_factor in (3., 10.):
                baseline = replace(config, nuisance_factor=1.)
                base_configuration = _configuration(
                    baseline, 1., requested_dk=provenance.get("requested_dk", config.dk),
                    model_flags=provenance.get("model_flags"))
                base_path = Path(output_root) / run_identity(base_configuration, provenance)
                if base_path.exists():
                    likelihood = load_cached_revision_likelihood(
                        base_path, likelihood_identity(base_configuration, provenance), provenance)
            if likelihood is None:
                likelihood = build_revision_likelihood(config, survey, cosmo, backend)
            cache[identity] = likelihood
            result = apply_revision_priors(likelihood, config.nuisance_factor)
        except Exception as exc:
            failure = {"status":"failed", "failure":_failure_context(config, exc),
                       **gates}
            write_revision_artifacts(None, output_root,
                                     {**provenance, "configuration":config}, failure)
            raise
        path = write_revision_artifacts(result, output_root, provenance, gates)
        if gate_bundle is not None and all(gates.get(name, {}).get("status") == "passed"
                                           for name in MANDATORY_GATES if name != "save_reload"):
            path = finalize_revision_artifacts(path, gate_bundle)
        completed.append(path)
    return completed


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="B")
    parser.add_argument("--kmax", type=float, default=.20)
    parser.add_argument("--kmin", type=float, default=.01)
    parser.add_argument("--dk", type=float, default=.005,
                        help="Adopted physical shell width for this run")
    parser.add_argument("--requested-dk", type=float, default=.005,
                        help="Original requested shell width recorded alongside adopted --dk")
    parser.add_argument("--n-mu", type=int, default=32)
    parser.add_argument("--ap-nk", type=int, choices=(2048, 8192), default=2048)
    parser.add_argument("--nuisance-factor", type=float, default=1.)
    parser.add_argument("--exclude-elg1", action="store_true")
    parser.add_argument("--grid", action="store_true", help="Inventory the exact 13 configurations (9 likelihoods)")
    parser.add_argument("--identities", action="store_true",
                        help="Hash current inputs and list exact run IDs for a gate bundle")
    parser.add_argument("--execute", action="store_true",
                        help="Compute selected configuration; without passed gates it remains building")
    parser.add_argument("--finalize", type=Path,
                        help="Attach passed evidence to a matching building run without recomputation")
    parser.add_argument("--gates", type=Path, help="JSON evidence bundle keyed by exact run ID")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results/revision-2026-09-30")
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(argv)
    if args.finalize is not None:
        if args.gates is None:
            raise ValueError("--finalize requires a gate file bound to the existing run identity")
        bundle = load_gates(args.gates)
        path = finalize_revision_artifacts(args.finalize, bundle)
        print(json.dumps({"scope":"complete", "path":str(path)}, indent=2))
        return 0
    if args.grid:
        configs = inventory(dk=args.dk, n_mu=args.n_mu,
                            requested_dk=args.requested_dk, ap_nk=args.ap_nk)
    else:
        configs = [RevisionConfig(args.case, kmin=args.kmin, kmax=args.kmax,
                                  dk=args.dk, n_mu=args.n_mu,
                                  nuisance_factor=args.nuisance_factor,
                                  exclude_elg1=args.exclude_elg1,
                                  ap_interpolation_nk=args.ap_nk,
                                  ap_refinement_nk=(2*args.ap_nk, 4*args.ap_nk))]
    if not args.execute:
        plan = {"scope":"validation_inventory", "requested_dk":args.requested_dk,
                "configurations":[_configuration(config, config.nuisance_factor,
                                                  requested_dk=args.requested_dk)
                                  for config in configs]}
        if args.identities:
            provenance = collect_provenance(_load_backend(), requested_dk=args.requested_dk,
                                            ap_nk=args.ap_nk)
            plan["configurations"] = [_configuration(config, config.nuisance_factor,
                                                   requested_dk=args.requested_dk,
                                                   model_flags=provenance.get("model_flags"))
                                      for config in configs]
            plan["run_ids"] = [run_identity(config, provenance)
                               for config in plan["configurations"]]
            plan["likelihood_identities"] = [likelihood_identity(config, provenance)
                                             for config in plan["configurations"]]
        print(json.dumps(plan, indent=2))
        return 0
    if args.grid and args.gates is None:
        raise ValueError("--grid --execute requires a gate file, even for building validation runs")
    gates = load_gates(args.gates) if args.gates is not None else None
    paths = execute(configs, args.output_root, gates, requested_dk=args.requested_dk,
                    command=" ".join(sys.argv if argv is None else [sys.argv[0], *argv]))
    print(json.dumps({"scope":"executed", "paths":[str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
