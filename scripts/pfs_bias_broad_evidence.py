"""Bind the wider population range to its exact physical model and fiducials."""
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from pfsfog.elf_survey_evidence import canonical, file_digest, request_digest
from scripts.elf_study_profiles import highz_profiles, study_config
from scripts.pfs_bias_evidence import bias_source
from scripts.pfs_bias_broad_cases import broad_cases, CANDIDATES
from scripts.validate_elf_survey import covariance_inputs

ROOT=Path(__file__).resolve().parents[1]
PHYSICAL_SOURCES=('scripts/pfs_bias_broad_cases.py','scripts/pfs_bias_broad_evidence.py',
    'scripts/pfs_bias_broad_physical.py','scripts/run_pfs_bias_broad.py',
    'docs/revision/specs/2026-10-10-pfs-bias-broad.md')


def broad_source(settings,engine=None):
    inherited=bias_source(settings,engine)
    files=[ROOT/p for p in PHYSICAL_SOURCES]+[CANDIDATES]
    details={**inherited['details'],'broad_bias_programs':{
        str(p):file_digest(p) for p in files}}
    return {'identity_hash':request_digest(details),'details':details}


def make_broad_request(settings,cosmo,source=None):
    if (set(settings)!={'nfft','n_mu','dk'} or
            any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in settings.values())
            or any(int(settings[k])!=settings[k] for k in ('nfft','n_mu'))):
        raise ValueError('invalid broad physical resolution')
    expected=[]
    for case in broad_cases(cosmo,settings):
        _,fiducials=covariance_inputs(case,cosmo)
        expected.append({'key':case['label'],'support':case['expected'],
                         'experiment':case['experiment'],'fiducials':fiducials})
    return json.loads(canonical({'schema':'pfs-bias-broad-physical-v1',
        'scope':'fixed_density_broad_bias','settings':settings,
        'configuration':asdict(study_config(highz_profiles()['C22'],settings,extended=True)),
        'expected':expected,'source':broad_source(settings) if source is None else source}))


def validate_broad_request(request,cosmo,source):
    if canonical(make_broad_request(request['settings'],cosmo,source))!=canonical(request):
        raise ValueError('broad request source, population or support changed')


def validate_broad_summary(request,summary):
    digest=request_digest(request);identity=request['source']['identity_hash']
    keys=[r['key'] for r in request['expected']]
    if (len(keys)!=35 or len(set(keys))!=35 or summary['status']!='passed'
            or summary['complete_populations'] is not True
            or summary['request_sha256']!=digest or summary['source_identity_hash']!=identity
            or [r['key'] for r in summary['workers']]!=keys):
        raise ValueError('incomplete or unrelated broad physical cohort')
    for row in summary['workers']:
        if (row['status']!='passed' or row['exit_code']!=0
                or row['request_sha256']!=digest or row['source_identity_hash']!=identity):
            raise ValueError('failed or unrelated broad physical population')

