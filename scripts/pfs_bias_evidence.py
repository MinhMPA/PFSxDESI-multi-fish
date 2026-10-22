"""Source and population binding for fresh fixed-density bias spectra.

The verifier below preserves the frozen high-z verifier's checks and adds the
established Gamma3 zero witness and exact changed-population identity. The
frozen verifier itself remains unchanged so earlier certificates stay valid.
"""
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from pfsfog.cosmo import FIDUCIAL
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_survey_evidence import (canonical, request_digest, file_digest,
    _require, _check_derivative_report)
from pfsfog.elf_validation_zeros import counterterm_zero_support
from pfsfog.extended_artifacts import read_json
from pfsfog.revision_forecast import make_shell_grid
from pfsfog.revision_validation_physical import _steps
from scripts.elf_study_evidence import study_source, study_config_from_request
from scripts.elf_study_profiles import highz_profiles, study_config
from scripts.pfs_bias_cases import bias_cases
from scripts.pfs_density_response import gamma3_zero_support
from scripts.validate_elf_survey import covariance_inputs

ROOT=Path(__file__).resolve().parents[1]
PHYSICAL_SOURCES=('scripts/pfs_bias_cases.py','scripts/pfs_bias_evidence.py',
    'scripts/pfs_bias_physical.py','scripts/run_pfs_bias_study.py',
    'scripts/pfs_density_response.py',
    'docs/revision/specs/2026-10-10-pfs-bias-response.md')


def bias_source(settings,engine=None):
    inherited=study_source(settings,engine)
    details={**inherited['details'],'bias_programs':{
        str(ROOT/p):file_digest(ROOT/p) for p in PHYSICAL_SOURCES}}
    return {'identity_hash':request_digest(details),'details':details}


def make_bias_request(settings,cosmo,*,source=None):
    if (set(settings)!={'nfft','n_mu','dk'} or
            any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in settings.values())
            or any(int(settings[k])!=settings[k] for k in ('nfft','n_mu'))):
        raise ValueError('invalid physical resolution')
    expected=[]
    for c in bias_cases(cosmo,settings):
        _,values=covariance_inputs(c,cosmo)
        expected.append({'key':c['label'],'support':c['expected'],
                         'experiment':c['experiment'],'fiducials':values})
    return json.loads(canonical({'schema':'pfs-bias-physical-v1',
        'scope':'fixed_density_bias_pilot','settings':settings,
        'configuration':asdict(study_config(highz_profiles()['C22'],settings,extended=True)),
        'expected':expected,'source':bias_source(settings) if source is None else source}))


def validate_bias_request(request,cosmo,source):
    rebuilt=make_bias_request(request['settings'],cosmo,source=source)
    if canonical(rebuilt)!=canonical(request):
        raise ValueError('bias request source, population or support changed')


def verify_bias_fiducials(inputs,expected):
    if canonical(inputs['fiducials'])!=canonical(expected['fiducials']):
        raise ValueError('measured fiducials differ from requested bias population')


def bias_zero_support(names,pairs,*,alphas,settings):
    zeros=list(counterterm_zero_support(names,pairs,alphas=alphas,settings=settings))
    zeros+=gamma3_zero_support({'metadata':{'local_names':names,'retained_pairs':pairs},
        'engine_settings':settings,'analytic_zero_response':{'alphas':alphas}})
    return zeros


def verify_bias_cell(directory, expected, request):
    """Check saved scope/coverage and reconstruct elementary array identities."""
    directory = Path(directory)
    read = lambda name: read_json(directory/name)
    inputs, measurement, neighborhood, worker = map(read,
        ("inputs.json", "measurement.json", "neighborhood.json", "worker.json"))
    covariance_inputs = read("covariance_inputs.json")
    verify_bias_fiducials(inputs, expected)
    row, key = expected["support"], expected["key"]
    names, pairs = row["local_names"], row["retained_pairs"]
    config = study_config_from_request(request)
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    metadata = inputs["metadata"]
    _require(covariance_inputs["configuration"] == inputs["configuration"]
             and covariance_inputs["fiducials"] == inputs["fiducials"]
             and covariance_inputs["engine_settings"] == inputs["engine_settings"]
             and all(metadata.get(k) == v for k, v in covariance_inputs["metadata"].items()),
             "mean and covariance input records differ")
    _require(inputs["configuration"] == request["configuration"],
             "cell configuration differs from requested calculation")
    for name in ("cell_id", "nominal_bin_index", "nominal_bin", "radial_bin", "region",
                 "model_redshift", "volume", "nbar", "candidate_populations",
                 "active_populations", "retained_pairs", "local_names"):
        _require(metadata[name] == row[name], "cell support differs: "+name)
    active = row["active_populations"]
    theoretical = [[a, a] for a in active]+[[a, b] for i, a in enumerate(active) for b in active[i+1:]]
    _require(metadata["theoretical_pairs"] == theoretical and metadata["retained_observations"] ==
             [[p, ell] for p in pairs for ell in (0, 2, 4)], "incorrect observation layout")
    _require(measurement["metadata"] == metadata and measurement["status"] == "passed"
             and inputs["scope"] == measurement["scope"] == "elf_survey_study_physical",
             "failed or unrelated mean measurement")
    _require(worker["status"] == "passed" and worker["key"] == key
             and worker["request_sha256"] == request_digest(request)
             and worker["source_before"] == worker["source_after"] == request["source"]["identity_hash"],
             "failed or unrelated worker receipt")
    _require(inputs["engine_settings"] == request["source"]["details"]["original"]["details"]["engine_settings"],
             "cell engine resolution or physical model differs")
    _require(set(inputs["fiducials"]) == set(names), "missing local fiducial")
    space = elf_cosmology_spec(FIDUCIAL)
    steps = {n: list(_steps(n, inputs["fiducials"][n], space=space)) for n in names}
    _require(inputs["steps"] == steps, "finite-difference steps changed")
    witness = inputs["analytic_zero_response"]
    zeros = bias_zero_support(names, pairs, alphas=witness["alphas"], settings=inputs["engine_settings"])
    _require(witness["support"] == [[n, list(p), ell] for n, p, ell in zeros], "analytic zero witness differs")
    _check_derivative_report(measurement["ad_fd"], names, pairs, steps, key, row["cell_id"],
                             analytic_zero_support=zeros)
    for component in ("intrinsic", "geometry"):
        _check_derivative_report(measurement["h_split_fd"][component], ["h"], pairs,
                                 {"h": steps["h"]}, key+"-"+component+"-h", row["cell_id"])
    _require(measurement["chain_tolerance"] == 1.e-10 and
             measurement["chain_relative_error"] < 1.e-10, "AP chain rule failed")
    nk, nobs, npar = len(grid.centers), 3*len(pairs), len(names)
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        shapes = {"D": (nk, nobs, npar), "C": (nk, nobs, nobs), "mean": (nk, nobs),
                  "F_like": (npar, npar), "D_intrinsic": (nk, nobs, npar),
                  "D_geometry": (nk, nobs, npar), "fiducial": (npar,),
                  "fd_plus": (npar, 3, nk, nobs), "fd_minus": (npar, 3, nk, nobs),
                  "fd_steps": (npar, 3), "fd_derivatives": (npar, 3, nk, nobs)}
        for component in ("intrinsic", "geometry"):
            shapes.update({f"h_{component}_{n}": shape for n, shape in
                (("plus", (1, 3, nk, nobs)), ("minus", (1, 3, nk, nobs)),
                 ("steps", (1, 3)), ("derivatives", (1, 3, nk, nobs)))})
        for name, shape in shapes.items():
            _require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(),
                     "invalid saved array: "+name)
        _require(np.array_equal(arrays["k"], grid.centers) and np.array_equal(arrays["dk"], grid.widths)
                 and np.array_equal(arrays["signed_mu"], np.polynomial.legendre.leggauss(config.n_mu)[0]),
                 "saved Fourier/quadrature grid differs")
        fiducial = np.array([inputs["fiducials"][n] for n in names])
        _require(np.array_equal(arrays["fiducial"], fiducial), "saved fiducial differs")
        _require(np.array_equal(arrays["fd_steps"], [steps[n] for n in names]), "saved steps differ")
        for prefix in ("fd", "h_intrinsic", "h_geometry"):
            direct = (arrays[prefix+"_plus"]-arrays[prefix+"_minus"])/(2*arrays[prefix+"_steps"][:, :, None, None])
            _require(np.array_equal(direct, arrays[prefix+"_derivatives"]), "saved finite difference differs")
        white = np.linalg.solve(np.linalg.cholesky(arrays["C"]), arrays["D"])
        inverse = np.linalg.solve(arrays["C"], np.broadcast_to(np.eye(nobs), arrays["C"].shape))
        scale = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(arrays["C"]), arrays["mean"][..., None]))
        for name, pair, ell in zeros:
            column = names.index(name)
            position = 3*pairs.index(list(pair))+(0, 2, 4).index(ell)
            weight = inverse[:, position, position]
            ad_norm = np.sqrt(np.sum(arrays["D"][:, position, column]**2*weight))
            fd_norm = np.sqrt(np.sum(arrays["fd_derivatives"][column, :, :, position]**2*weight[None, :], axis=1))
            _require(ad_norm <= 1.e-6*scale and np.all(fd_norm <= 1.e-6*scale), "analytic angular zero has physical leakage")
        fisher = np.einsum("kop,koq->pq", white, white, optimize=False)
        fisher_error = float(np.linalg.norm(fisher-arrays["F_like"])/np.linalg.norm(fisher))
        _require(fisher_error < 1.e-10, "saved Fisher sum differs")
    _require(measurement["arrays_sha256"] == file_digest(directory/"arrays.npz"), "mean arrays changed")
    points = [{"column": None, "sign": 0, "step": 0.}]
    coordinates = [fiducial]
    for i, name in enumerate(names):
        for step in steps[name]:
            for sign in (1, -1):
                point = fiducial.copy(); point[i] += sign*step
                coordinates.append(point); points.append({"column": name, "sign": sign, "step": step})
    _require(neighborhood["status"] == "passed" and neighborhood["cell_id"] == row["cell_id"]
             and neighborhood["expected_points"] == neighborhood["processed_points"] == len(points)
             and neighborhood["parameter_names"] == names and neighborhood["theoretical_pairs"] == theoretical
             and neighborhood["shell_count"] == nk and neighborhood["angular_order"] == config.n_mu
             and neighborhood["full_forecast_qualification"] == "pending"
             and len(neighborhood["points"]) == len(points), "incomplete covariance neighborhood")
    _require(np.array_equal(np.load(directory/"neighborhood_coordinates.npy", allow_pickle=False), coordinates),
             "saved covariance displacements differ")
    spectra = np.load(directory/"neighborhood_spectra.npy", mmap_mode="r", allow_pickle=False)
    _require(spectra.shape == (len(points), len(theoretical), nk, config.n_mu), "neighborhood spectra missing")
    for index, (record, description) in enumerate(zip(neighborhood["points"], points)):
        _require(all(record[k] == v for k, v in description.items()) and record["status"] == "passed"
                 and np.isfinite(spectra[index]).all(), "missing/nonfinite covariance point")
        for kind, count in (("field", nk*config.n_mu), ("covariance", nk)):
            audit = record[kind]
            _require(audit["matrix_count"] == count and audit["min_scaled_eigenvalue"] > 0
                     and audit["max_scaled_residual"] <= 1.e-10 and audit["max_scaled_asymmetry"] <= 1.e-10,
                     "failed field/covariance point")
    for name in ("spectra", "coordinates"):
        _require(neighborhood[name+"_sha256"] == file_digest(directory/f"neighborhood_{name}.npy"),
                 "neighborhood arrays changed")
    return {"key": key, "status": "passed", "points": len(points), "fisher_relative_error": fisher_error}

