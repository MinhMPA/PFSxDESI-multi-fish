"""Cheap failure-path checks for the representative resolution diagnostics."""
import hashlib
import json
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "docs/revision/evidence"


def _script(tmp_path, name):
    here = tmp_path / "docs/revision/evidence"
    here.mkdir(parents=True)
    path = here / name
    shutil.copyfile(SCRIPTS / name, path)
    namespace = runpy.run_path(str(path))
    main = namespace["main"]
    main.__globals__.update(
        FiducialCosmology=lambda: object(),
        default_survey_group=lambda: object(),
        PowerSpectrum1Loop=lambda **kw: object(),
    )
    return path, main


def _failed(path):
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    return report


def test_coarse_setup_failure_persists(tmp_path):
    path, main = _script(tmp_path, "validate-assembled-resolution.py")
    def fail_setup():
        raise RuntimeError("setup failure")
    main.__globals__["FiducialCosmology"] = fail_setup
    with pytest.raises(RuntimeError, match="setup failure"):
        main()
    report = _failed(path.with_name("assembled-resolution-probe.json"))
    assert "setup failure" in report["error"]


@pytest.mark.parametrize("bad", [float("nan"), 0.0])
def test_coarse_invalid_fixed_errors_persist_origin(bad, tmp_path):
    path, main = _script(tmp_path, "validate-assembled-resolution.py")
    main.__globals__["build_revision_likelihood"] = lambda *args: SimpleNamespace(validation={})
    main.__globals__["apply_revision_priors"] = lambda *args: SimpleNamespace(
        cov_cosmo_marg=np.eye(3), growth_cov_marg=np.eye(1),
        cov_cosmo_nuisance_fixed=np.diag([bad, 1., 1.]),
        growth_cov_nuisance_fixed=np.eye(1),
    )
    with pytest.raises(ValueError, match="Invalid forecast errors for B_base"):
        main()
    report = _failed(path.with_name("assembled-resolution-probe.json"))
    assert "Invalid forecast errors for B_base" in report["error"]
    assert report["runs"] == []


def test_coarse_failed_tolerance_exits_nonzero_with_evidence(tmp_path):
    path, main = _script(tmp_path, "validate-assembled-resolution.py")
    main.__globals__["build_revision_likelihood"] = lambda *args: SimpleNamespace(
        validation={}, grid=SimpleNamespace(edges=np.array([.01, .2])))
    def result(like, factor):
        config = main.__globals__["_current_config"]
        scale = 1.1 if config.dk == .0025 else 1.
        return SimpleNamespace(
            cov_cosmo_marg=np.eye(3) * scale**2,
            growth_cov_marg=np.eye(1) * scale**2,
            cov_cosmo_nuisance_fixed=np.eye(3), growth_cov_nuisance_fixed=np.eye(1),
            F_total=np.eye(4), covariance=np.eye(4), names=["a", "b", "c", "d"],
            grid=SimpleNamespace(edges=np.array([.01, .2])), validation={})
    original_config = main.__globals__["RevisionConfig"]
    def config(**kwargs):
        value = original_config(**kwargs)
        main.__globals__["_current_config"] = value
        return value
    main.__globals__["RevisionConfig"] = config
    main.__globals__["apply_revision_priors"] = result
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0
    report = _failed(path.with_name("assembled-resolution-probe.json"))
    assert any(not row["passed"] for row in report["comparisons"])


def _baseline(path):
    root = path.parents[3]
    source_file = root / "dummy-source.py"
    source_file.write_text("executed source\n")
    archive = path.with_name("assembled-resolution-probe.npz")
    archive.write_bytes(b"archived numeric output")
    record = lambda case: {"key": f"{case}_shell", "dk": .0025, "n_mu": 32,
                           "sigma_marg": [1., 2., 3., 4.]}
    data = {"scope": "limited_representative_nominal_bin",
            "nominal_bins": [[1., 1.2]], "model_z": 1.1,
            "kmin": .01, "kmax": .20, "do_irres": False,
            "source_sha256": {"dummy-source.py": hashlib.sha256(source_file.read_bytes()).hexdigest()},
            "arrays_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "runs": [record("B"), record("C")]}
    baseline_path = path.with_name("assembled-resolution-probe.json")
    baseline_path.write_text(json.dumps(data))
    return baseline_path, data


def test_finer_missing_baseline_persists_failure(tmp_path):
    path, main = _script(tmp_path, "refine-assembled-shells.py")
    with pytest.raises(FileNotFoundError):
        main()
    report = _failed(path.with_name("assembled-shell-refinement.json"))
    assert "FileNotFoundError" in report["error"]


def test_finer_malformed_baseline_persists_failure(tmp_path):
    path, main = _script(tmp_path, "refine-assembled-shells.py")
    path.with_name("assembled-resolution-probe.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        main()
    report = _failed(path.with_name("assembled-shell-refinement.json"))
    assert "JSONDecodeError" in report["error"]


def test_finer_missing_script_persists_failure(tmp_path):
    path, main = _script(tmp_path, "refine-assembled-shells.py")
    _baseline(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        main()
    report = _failed(path.with_name("assembled-shell-refinement.json"))
    assert "FileNotFoundError" in report["error"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(scope="wrong"),
    lambda d: d.update(nominal_bins=[[1., 1.3]]),
    lambda d: d.update(model_z=1.2),
    lambda d: d.update(kmin=.02),
    lambda d: d.update(kmax=.21),
    lambda d: d.update(do_irres=True),
    lambda d: d["runs"][0].update(dk=.005),
    lambda d: d["runs"][1].update(n_mu=64),
    lambda d: d["runs"][0].update(sigma_marg=[1., 2., 3.]),
    lambda d: d["runs"][0].update(sigma_marg=[1., 2., 3., float("nan")]),
    lambda d: d["runs"][0].update(sigma_marg=[1., 2., 3., 0.]),
    lambda d: d["runs"].pop(),
    lambda d: d["runs"].append(dict(d["runs"][0])),
])
def test_finer_rejects_mismatched_baseline_before_physics(tmp_path, mutate):
    path, main = _script(tmp_path, "refine-assembled-shells.py")
    baseline_path, data = _baseline(path)
    mutate(data)
    baseline_path.write_text(json.dumps(data))
    def unexpected_physics():
        raise AssertionError("physics started")
    main.__globals__["FiducialCosmology"] = unexpected_physics
    with pytest.raises((ValueError, KeyError, TypeError)):
        main()
    report = _failed(path.with_name("assembled-shell-refinement.json"))
    assert "physics started" not in report["error"]


@pytest.mark.parametrize("tamper", ["archive", "source"])
def test_finer_rejects_changed_hashes_before_physics(tmp_path, tamper):
    path, main = _script(tmp_path, "refine-assembled-shells.py")
    _baseline(path)
    target = path.with_name("assembled-resolution-probe.npz") if tamper == "archive" else tmp_path / "dummy-source.py"
    target.write_bytes(b"changed")
    main.__globals__["FiducialCosmology"] = lambda: pytest.fail("physics started")
    with pytest.raises(ValueError):
        main()
    _failed(path.with_name("assembled-shell-refinement.json"))
