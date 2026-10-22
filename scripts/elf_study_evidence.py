"""Source-bound high-z physical evidence, including the native PFS-only tail."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from pfsfog.cosmo import FIDUCIAL
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.elf_survey_evidence import (canonical,request_digest,source_identity,file_digest,
    artifact_hashes,verify_files,cell_directory,_require,_check_derivative_report)
from pfsfog.elf_validation_zeros import counterterm_zero_support
from pfsfog.extended_artifacts import read_json
from pfsfog.revision_forecast import make_shell_grid
from pfsfog.revision_validation_physical import _steps
from scripts.elf_study_profiles import highz_profiles,study_cells,study_config,StudyConfig

ROOT=Path(__file__).resolve().parents[1]
SCOPE='new_highz_overlap_only'


def study_source(settings,engine=None):
    files=set((ROOT/'scripts').glob('elf_study*.py'))
    files.add(ROOT/'scripts/elf_grid_comparison.py')
    runner=ROOT/'scripts/run_elf_studies.py'
    if runner.is_file(): files.add(runner)
    details={'original':source_identity(settings,engine),
             'programs':{str(p):file_digest(p) for p in sorted(files)}}
    return {'identity_hash':request_digest(details),'details':details}


def make_study_request(settings,*,source=None):
    if (set(settings)!={'nfft','n_mu','dk'} or any(not np.isfinite(v) or v<=0 for v in settings.values())
            or settings['n_mu']<8 or settings['nfft']<16
            or any(isinstance(settings[k],bool) or int(settings[k])!=settings[k] for k in ('nfft','n_mu'))):
        raise ValueError('invalid physical grid settings')
    p=highz_profiles()['C22'];config=study_config(p,settings,extended=True)
    cases=study_cells(p,SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:1.),settings=settings,extended=True)
    expected=[{'key':c['label'],'support':c['expected']} for c in cases
              if c['region']=='overlap' and c['expected']['radial_bin'][0]>=1.6]
    return json.loads(canonical({'schema':'elf-study-highz-v1','scope':SCOPE,'settings':settings,
        'configuration':asdict(config),'expected':expected,
        'source':study_source(settings) if source is None else source}))


def study_config_from_request(request):
    c=dict(request['configuration']);c['nominal_bins']=tuple(map(tuple,c['nominal_bins']))
    return StudyConfig(**c)


def validate_study_ledger(request,summary,live):
    problems=[]
    try:
        if canonical(make_study_request(request['settings'],source=live))!=canonical(request):
            problems.append('source, configuration or support changed')
        digest=request_digest(request);identity=live['identity_hash']
        keys=[r['key'] for r in request['expected']]
        if [r['key'] for r in summary['workers']]!=keys or len(set(keys))!=3:
            problems.append('incomplete or duplicated three-cell schedule')
        if (summary['status']!='passed' or summary['scope']!=SCOPE or summary['source_changed']
                or summary['source_identity_hash']!=identity or summary['request_sha256']!=digest
                or summary['complete_new_cells'] is not True or summary['full_forecast_qualification']!='pending'):
            problems.append('failed, unrelated or overstated physical summary')
        for r in summary['workers']:
            if (r['status']!='passed' or r['exit_code']!=0 or r['source_identity_hash']!=identity
                    or r['request_sha256']!=digest): problems.append('failed or unrelated worker: '+r['key'])
    except (KeyError,TypeError,ValueError) as error: problems.append(str(error))
    return problems


def load_source_cell(directory,record):
    directory=Path(directory)
    if record.get('status')!='passed' or record.get('exit_code')!=0:
        raise ValueError('source cell is not a passed physical worker')
    problems=verify_files(directory,record['files'])
    if problems: raise ValueError('; '.join(problems))
    with np.load(directory/'arrays.npz',allow_pickle=False) as a: arrays={k:a[k].copy() for k in a.files}
    return {'inputs':read_json(directory/'inputs.json'),'measurement':read_json(directory/'measurement.json'),
        'arrays':arrays,'neighborhood_spectra':np.load(directory/'neighborhood_spectra.npy',mmap_mode='r',allow_pickle=False),
        'neighborhood_coordinates':np.load(directory/'neighborhood_coordinates.npy',allow_pickle=False),
        'source_files':{str(directory/p):h for p,h in record['files'].items()}}


# Retain the original physical array/FD/AP checks, with explicit study scope and tail config.
def verify_highz_cell(directory, expected, request):
    """Check saved scope/coverage and reconstruct elementary array identities."""
    directory = Path(directory)
    read = lambda name: read_json(directory/name)
    inputs, measurement, neighborhood, worker = map(read,
        ("inputs.json", "measurement.json", "neighborhood.json", "worker.json"))
    covariance_inputs = read("covariance_inputs.json")
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
    zeros = counterterm_zero_support(names, pairs, alphas=witness["alphas"], settings=inputs["engine_settings"])
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


def verify_highz(root):
    root=Path(root);result={'status':'failed','problems':[],'cells':[],
                           'complete_new_cells':False,'full_forecast_qualification':'pending'}
    try:
        request=read_json(root/'request.json');summary=read_json(root/'summary.json')
        result['problems']=validate_study_ledger(request,summary,study_source(request['settings']))
        bykey={r['key']:r for r in summary['workers']}
        for expected in request['expected']:
            key=expected['key']
            if key not in bykey: continue
            try:
                path=cell_directory(root,key);problems=verify_files(path,bykey[key]['files'])
                if problems: raise ValueError('; '.join(problems))
                result['cells'].append(verify_highz_cell(path,expected,request))
            except Exception as error: result['problems'].append(key+': '+str(error))
        if not result['problems']:result.update(status='passed',complete_new_cells=True)
    except Exception as error:result['problems'].append(str(error))
    return result
