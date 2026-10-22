"""Prepared wider bias range, retaining the measured pilot's population model."""
from pathlib import Path

import numpy as np

from pfsfog.extended_artifacts import read_json
from scripts.pfs_bias_cases import bias_cases, vary_case, PRESCRIPTIONS, PFS

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=ROOT/'docs/revision/evidence/pfs-bias-response-2026-10-10/BROADER_CANDIDATES.json'


def broad_schedule():
    inventory=read_json(CANDIDATES);rows=[]
    for i,window in enumerate(inventory['windows']):
        ratios=window['bias_ratios'];equal=window['DESI_equal_bias_ratios']
        if (len(ratios)!=len(set(ratios)) or 1. not in ratios
                or any(not np.isfinite(r) or r<=0 for r in ratios)
                or any(r not in ratios for r in equal.values())):
            raise ValueError('invalid prepared bias range')
        trials=[('b1_only',1.)]+[(p,r) for r in sorted(ratios,reverse=True)
                               if r!=1. for p in PRESCRIPTIONS]
        for prescription,ratio in trials:
            targets=[name for name,value in equal.items() if value==ratio]
            if len(targets)>1:raise ValueError('ambiguous equal-bias target')
            target=targets[0] if targets else None
            trial='equal-'+target if target else 'r'+format(ratio,'.12g').replace('.','p')
            rows.append({'key':f'w{i}-{prescription}-{trial}','window':window['window'],
                         'ratio':ratio,'prescription':prescription,'trial':trial,'equal_to':target})
    if len(rows)!=35 or len({r['key'] for r in rows})!=35:
        raise ValueError('incomplete or duplicate broad bias schedule')
    return rows


def broad_cases(cosmo,settings):
    templates={tuple(c['experiment']['window']):c for c in bias_cases(cosmo,settings)
               if c['experiment']['ratio']==1.}
    cases=[]
    for row in broad_schedule():
        original=templates[tuple(row['window'])];e=original['experiment']
        case=vary_case(original,row['ratio'],row['prescription'],e['sigma8_z'],e['b1_reference'])
        case['label']=row['key']
        case['experiment'].update({k:row[k] for k in ('window','trial','equal_to')})
        if row['equal_to']:
            actual=case['fiducials'][PFS].b1_sigma8
            expected=case['fiducials'][row['equal_to']].b1_sigma8
            if not np.isclose(actual,expected,rtol=1e-13,atol=0):
                raise ValueError('prepared equal bias differs from physical DESI fiducial')
        cases.append(case)
    return cases
