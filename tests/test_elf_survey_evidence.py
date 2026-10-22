"""Evidence integrity checks; synthetic ledgers are not physical certificates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def synthetic_ledger(*, selected=False):
    from pfsfog.elf_survey_evidence import make_request, request_digest
    request = make_request({"nfft": 1024, "dk": .00125, "n_mu": 32},
                           cases=("B", "C"),
                           cell_ids=("bin00-radial00-overlap",) if selected else None,
                           source={"identity_hash": "synthetic-source"})
    summary = {"status": "passed", "scope": request["scope"],
               "source_identity_hash": "synthetic-source",
               "request_sha256": request_digest(request), "source_changed": False,
               "complete_survey": not selected, "full_forecast_qualification": "pending",
               "workers": [{"key": item["key"], "exit_code": 0, "status": "passed",
                            "source_identity_hash": "synthetic-source",
                            "request_sha256": request_digest(request)}
                           for item in request["expected"]]}
    return request, summary


def test_complete_ledger_covers_forty_cells_but_never_full_forecast():
    from pfsfog.elf_survey_evidence import validate_ledger
    request, summary = synthetic_ledger()
    assert len(request["expected"]) == 40
    assert sum(1+6*len(r["support"]["local_names"]) for r in request["expected"]
               if r["case"] == "B") == 3200
    assert sum(1+6*len(r["support"]["local_names"]) for r in request["expected"]
               if r["case"] == "C") == 3752
    assert validate_ledger(request, summary, request["source"]) == []
    summary["full_forecast_qualification"] = "passed"
    assert validate_ledger(request, summary, request["source"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign", "nonzero_exit",
    "failed_worker", "changed_source", "different_worker_source", "different_request",
    "changed_config", "source_changed"])
def test_no_damaged_ledger_can_pass(damage):
    from pfsfog.elf_survey_evidence import validate_ledger
    request, summary = synthetic_ledger()
    live = copy.deepcopy(request["source"])
    if damage == "missing":
        summary["workers"].pop()
    elif damage == "duplicate":
        summary["workers"].append(copy.deepcopy(summary["workers"][0]))
    elif damage == "foreign":
        summary["workers"][0]["key"] = "C:unknown"
    elif damage == "nonzero_exit":
        summary["workers"][0]["exit_code"] = 7
    elif damage == "failed_worker":
        summary["workers"][0]["status"] = "failed"
    elif damage == "changed_source":
        live["identity_hash"] = "new-source"
    elif damage == "different_worker_source":
        summary["workers"][0]["source_identity_hash"] = "new-source"
    elif damage == "different_request":
        summary["workers"][0]["request_sha256"] = "unrelated"
    elif damage == "changed_config":
        request["configurations"]["B"]["dk"] = .005
    elif damage == "source_changed":
        summary["source_changed"] = True
    assert validate_ledger(request, summary, live)


def test_selected_cells_cannot_claim_complete_survey():
    from pfsfog.elf_survey_evidence import validate_ledger
    request, summary = synthetic_ledger(selected=True)
    assert len(request["expected"]) == 2
    assert request["scope"] == "selected_cells"
    assert validate_ledger(request, summary, request["source"]) == []
    summary["complete_survey"] = True
    assert validate_ledger(request, summary, request["source"])


@pytest.mark.parametrize("damage", ["missing", "corrupt", "config", "unlisted"])
def test_saved_file_changes_are_detected(tmp_path, damage):
    from pfsfog.elf_survey_evidence import artifact_hashes, verify_files
    np.savez_compressed(tmp_path/"arrays.npz", D=np.eye(2), C=np.eye(2))
    (tmp_path/"inputs.json").write_text(json.dumps({"dk": .00125}))
    hashes = artifact_hashes(tmp_path)
    assert verify_files(tmp_path, hashes) == []
    if damage == "missing":
        (tmp_path/"arrays.npz").unlink()
    elif damage == "corrupt":
        (tmp_path/"arrays.npz").write_bytes(b"not the measured derivative")
    elif damage == "config":
        (tmp_path/"inputs.json").write_text(json.dumps({"dk": .005}))
    else:
        (tmp_path/"unexpected.npy").write_bytes(b"unlisted")
    assert verify_files(tmp_path, hashes)


def test_actual_source_identity_distinguishes_loop_resolution():
    from pfsfog.elf_survey_evidence import source_identity
    a = source_identity({"nfft": 512, "dk": .00125, "n_mu": 32})
    b = source_identity({"nfft": 1024, "dk": .00125, "n_mu": 32})
    assert a["identity_hash"] != b["identity_hash"]
    assert a["details"]["model_flags"]["engine"] == "elf.EPT"
    assert a["details"]["model_flags"]["ap_interpolation_nk"] is None


def test_cli_can_parse_its_measurement_and_verification_options():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/validate_elf_survey.py", "--help"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--verify" in result.stdout and "--cell-id" in result.stdout


def test_missing_run_cannot_be_verified(tmp_path):
    from pfsfog.elf_survey_evidence import verify_run
    result = verify_run(tmp_path)
    assert result["status"] == "failed" and result["problems"]
    assert result["complete_survey"] is False
    assert result["full_forecast_qualification"] == "pending"


def test_saved_redshift_bounds_recover_the_same_nominal_bin():
    from pfsfog.elf_survey_evidence import config_from_request
    from pfsfog.revision_forecast import _bin_number
    request, _ = synthetic_ledger()
    config = config_from_request(json.loads(json.dumps(request)), "C")
    assert _bin_number(config, (1., 1.2)) == 3
    assert _bin_number(config, (2., 2.1)) == 7


def test_worker_retains_spectra_and_matrix_when_fiducial_field_is_negative(tmp_path, monkeypatch):
    from pfsfog import elf_covariance_audit, revision_forecast
    from pfsfog.elf_survey_evidence import make_request, write_json, cell_directory
    from scripts.validate_elf_survey import run_worker
    def negative_response(*args, **kwargs):
        return SimpleNamespace(pkmu=lambda k, mu, *coordinates:
                               np.full((len(k), len(mu)), -1.e12))
    # Replace only the physical response: geometry, source records, worker flow
    # and covariance rejection all remain real. This is a deliberate bad field.
    monkeypatch.setattr(revision_forecast, "make_pair_response", negative_response)
    monkeypatch.setattr(elf_covariance_audit, "make_pair_response", negative_response)
    request = make_request({"nfft": 512, "dk": .095, "n_mu": 8}, cases=("B",),
                           cell_ids=("bin00-radial00-overlap",))
    write_json(tmp_path/"request.json", request)
    assert run_worker(tmp_path, "B", "bin00-radial00-overlap") == 1
    directory = cell_directory(tmp_path, "B:bin00-radial00-overlap")
    assert (directory/"covariance_inputs.json").is_file()
    report = json.loads((directory/"neighborhood.json").read_text())
    assert report["status"] == "failed" and report["processed_points"] == 1
    assert report["points"][0]["column"] is None
    signal = np.load(directory/"neighborhood_spectra.npy", mmap_mode="r")[0, 0, 0, 0]
    nbar = request["expected"][0]["support"]["nbar"]["DESI-LRG"]
    np.testing.assert_array_equal(np.load(directory/"failed_matrix.npy"), [[signal+1./nbar]])
