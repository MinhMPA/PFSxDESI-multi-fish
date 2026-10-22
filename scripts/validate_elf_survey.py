#!/usr/bin/env python
"""Measure ELF physical derivatives/covariances across the baseline survey.

Each cell is a separate serial process. A passed run certifies this numerical
resolution's physical checks only, never marginalized Fisher convergence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"ps_1loop_jax"))
sys.path.insert(0, str(ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from pfsfog.cosmo import FiducialCosmology
from pfsfog.elf_covariance_audit import audit_neighborhood
from pfsfog.elf_survey import elf_cosmology_spec, survey_cases
from pfsfog.elf_survey_evidence import (
    artifact_hashes, canonical, cell_directory, config_from_request, make_request, request_digest,
    source_identity, verify_cell, verify_run, write_json)
from pfsfog.one_loop import make_one_loop_engine
from pfsfog.revision_forecast import RevisionConfig, make_shell_grid
from pfsfog.revision_theory import NUISANCE_NAMES, fiducial_nuisance
from pfsfog.revision_validation_physical import validate_support_schedule
from scripts.validate_elf_engine import measure_cell


def covariance_inputs(case, cosmo):
    row = case["expected"]
    metadata = {k: v for k, v in row.items() if k not in ("empty_observation", "absent_observations")}
    active, pairs = row["active_populations"], row["retained_pairs"]
    metadata["theoretical_pairs"] = [(a, a) for a in active]+[
        (a, b) for i, a in enumerate(active) for b in active[i+1:]]
    metadata["retained_observations"] = [(pair, ell) for pair in pairs for ell in (0, 2, 4)]
    space = elf_cosmology_spec(cosmo.params)
    values = {n: float(cosmo.params[{"Mnu": "mnu", "Omegam": "Omega_m"}.get(n, n)]) for n in space.names}
    ibin = row["nominal_bin_index"]
    for tracer in active:
        values.update({f"bin{ibin:02d}:{tracer}:{n}": float(v) for n, v in
                       zip(NUISANCE_NAMES, np.asarray(fiducial_nuisance(case["fiducials"][tracer])))})
    values.update({n: 0. for n in row["local_names"] if "|" in n})
    return SimpleNamespace(metadata=metadata, names=list(row["local_names"])), values


def run_worker(root, case_name, cell_id):
    request = json.loads((root/"request.json").read_text())
    key = case_name+":"+cell_id
    matches = [row for row in request["expected"] if row["key"] == key]
    if len(matches) != 1:
        raise ValueError("worker is absent or duplicated in the requested survey")
    output = cell_directory(root, key)
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name != "worker.log" for path in output.iterdir()):
        raise FileExistsError("refusing to reuse a measured cell directory")
    start = time.monotonic()
    receipt = {"key": key, "status": "running", "request_sha256": request_digest(request),
               "source_before": None, "source_after": None,
               "full_forecast_qualification": "pending"}
    write_json(output/"worker.json", receipt)
    engine = None
    try:
        engine = make_one_loop_engine(nfft=request["settings"]["nfft"])
        before = source_identity(request["settings"], engine)
        receipt["source_before"] = before["identity_hash"]
        rebuilt = make_request(request["settings"], cases=request["cases"],
                               cell_ids=request["cell_ids"], source=before)
        if canonical(rebuilt) != canonical(request):
            raise ValueError("worker sources/configuration differ from requested calculation")
        config = config_from_request(request, case_name)
        cosmo = FiducialCosmology()
        case = next(item for item in survey_cases(config, cosmo) if item["cell_id"] == cell_id)
        if canonical(case["expected"]) != canonical(matches[0]["support"]):
            raise ValueError("worker support differs from independent requested schedule")
        # Save the field before likelihood assembly can reject its fiducial.
        covariance_cell, values = covariance_inputs(case, cosmo)
        write_json(output/"covariance_inputs.json", {"configuration": request["configurations"][case_name],
            "metadata": covariance_cell.metadata, "fiducials": values,
            "engine_settings": engine.numerical_settings()})
        neighborhood = audit_neighborhood(covariance_cell, config, cosmo.params, engine, values, output)
        receipt["covariance_status"] = neighborhood["status"]
        if neighborhood["status"] != "passed":
            receipt.update(status="failed", mean_status="not_measured_nonphysical_covariance",
                           reason=neighborhood["points"][-1].get("reason", "covariance neighborhood failed"))
        else:
            cell, measurement = measure_cell(case, config, cosmo, engine, output)
            inputs = json.loads((output/"inputs.json").read_text())
            if canonical(inputs["fiducials"]) != canonical(values):
                raise ValueError("mean and covariance use different fiducial coordinates")
            assembled = SimpleNamespace(config=config, cells=[cell], names=cell.names,
                fiducial=[values[n] for n in cell.names],
                grid=make_shell_grid(config.kmin, config.kmax, config.dk))
            receipt["support_check"] = validate_support_schedule(assembled, [case["expected"]],
                                                                 space=elf_cosmology_spec(cosmo.params))
            receipt.update(mean_status=measurement["status"], status=measurement["status"])
    except Exception as error:
        receipt.update(status="failed", reason=str(error), error_type=type(error).__name__)
        (output/"failure.log").write_text(traceback.format_exc())
    try:
        after = source_identity(request["settings"], engine)
        receipt["source_after"] = after["identity_hash"]
        if canonical(after) != canonical(request["source"]):
            receipt.update(status="failed", source_changed=True)
    except Exception as error:
        receipt.update(status="failed", source_changed=True, final_source_error=str(error))
    receipt["elapsed_seconds"] = time.monotonic()-start
    write_json(output/"worker.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("key", "status", "elapsed_seconds")}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


def run_survey(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    settings = {"nfft": args.nfft, "dk": args.dk, "n_mu": args.n_mu}
    engine = make_one_loop_engine(nfft=args.nfft)
    request = make_request(settings, cases=args.cases, cell_ids=args.cell_id,
                           source=source_identity(settings, engine))
    write_json(output/"request.json", request)
    identity, digest = request["source"]["identity_hash"], request_digest(request)
    summary = {"status": "running", "scope": request["scope"], "source_identity_hash": identity,
               "request_sha256": digest, "workers": [], "complete_survey": False,
               "source_changed": False, "full_forecast_qualification": "pending",
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "command": sys.argv}
    write_json(output/"summary.json", summary)
    for expected in request["expected"]:
        try:
            if canonical(source_identity(settings, engine)) != canonical(request["source"]):
                raise ValueError("sources changed before the next survey cell")
        except Exception as error:
            summary.update(source_changed=True, source_error=str(error))
            break
        key = expected["key"]
        dest = cell_directory(output, key)
        dest.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                   "--worker", expected["case"], expected["support"]["cell_id"]]
        cell_start = time.monotonic()
        print(f"Starting {key} ({len(summary['workers'])+1}/{len(request['expected'])})", flush=True)
        with (dest/"worker.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {"key": key, "exit_code": process.returncode, "command": command,
                  "elapsed_seconds": time.monotonic()-cell_start, "status": "failed",
                  "request_sha256": digest, "source_identity_hash": identity}
        try:
            receipt = json.loads((dest/"worker.json").read_text())
            record["worker_status"] = receipt["status"]
            record["source_identity_hash"] = receipt["source_after"]
            record["request_sha256"] = receipt["request_sha256"]
            record["verification"] = verify_cell(dest, expected, request)
            if process.returncode == 0 and receipt["status"] == "passed":
                record["status"] = "passed"
        except Exception as error:
            record["reason"] = str(error)
        record["files"] = artifact_hashes(dest)
        summary["workers"].append(record)
        try:
            if canonical(source_identity(settings, engine)) != canonical(request["source"]):
                raise ValueError("sources changed during a survey cell")
        except Exception as error:
            summary.update(source_changed=True, source_error=str(error))
        summary["elapsed_seconds"] = time.monotonic()-start
        write_json(output/"summary.json", summary)
        print(f"Finished {key}: {record['status']}; actual exit {process.returncode}; "
              f"{record['elapsed_seconds']:.1f} s", flush=True)
        if summary["source_changed"]:
            break
    passed = (not summary["source_changed"] and len(summary["workers"]) == len(request["expected"]))
    passed &= all(row["status"] == "passed" for row in summary["workers"])
    summary.update(status="passed" if passed else "failed", elapsed_seconds=time.monotonic()-start,
                   complete_survey=bool(passed and request["scope"] == "full_B6_C6"))
    write_json(output/"summary.json", summary)
    print(json.dumps({"status": summary["status"], "scope": summary["scope"],
                      "completed_cells": len(summary["workers"]),
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--output", type=Path)
    selection.add_argument("--verify", type=Path)
    parser.add_argument("--cases", nargs="+", choices=("B", "C"), default=["B", "C"])
    parser.add_argument("--cell-id", action="append")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--dk", type=float, default=.00125)
    parser.add_argument("--n-mu", type=int, default=32)
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "CELL_ID"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_run(args.verify)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "passed" else 1
    if args.worker:
        return run_worker(args.output.resolve(), *args.worker)
    return run_survey(args)


if __name__ == "__main__":
    raise SystemExit(main())
