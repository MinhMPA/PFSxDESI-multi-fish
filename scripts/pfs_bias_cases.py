"""Controlled PFS population changes at fixed density and independent nuisances."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from pfsfog.eft_params import pfs_elg_fiducials
from pfsfog.surveys import default_survey_group
from scripts.elf_study_profiles import highz_profiles, study_cells

WINDOWS = ((.7,.8), (1.4,1.6), (1.6,2.))
PRESCRIPTIONS = ('b1_only', 'adopted_relations')
PFS = 'PFS-ELG'


def bias_schedule():
    rows=[]
    for i,window in enumerate(WINDOWS):
        for prescription,ratio in [('b1_only',1.)]+[
                (p,r) for p in PRESCRIPTIONS for r in (.8,1.2)]:
            rows.append({'key':f'w{i}-{prescription}-r{ratio:.1f}',
                         'window':list(window),'ratio':ratio,'prescription':prescription})
    return rows


def vary_case(case, ratio, prescription, sigma8_z, b1_ref):
    if (isinstance(ratio,(bool,np.bool_)) or not np.isscalar(ratio)
            or not np.isfinite(ratio) or ratio<=0):
        raise ValueError('bias ratio must be positive and finite')
    if prescription not in PRESCRIPTIONS:
        raise ValueError('unknown population prescription')
    if not np.isfinite([sigma8_z,b1_ref]).all() or min(sigma8_z,b1_ref)<=0:
        raise ValueError('invalid population amplitude or reference bias')
    if PFS not in case['expected']['active_populations']:
        raise ValueError('bias change requires observed PFS support')
    changed=deepcopy(case);old=case['fiducials'][PFS]
    if ratio!=1.:
        changed['fiducials'][PFS]=(replace(old,b1_sigma8=old.b1_sigma8*ratio)
            if prescription=='b1_only' else pfs_elg_fiducials(
                old.b1_sigma8/sigma8_z*ratio,b1_ref,sigma8_z,r_sigma_v=.75))
    changed['experiment']={'ratio':float(ratio),'prescription':prescription,
        'sigma8_z':float(sigma8_z),'b1_reference':float(b1_ref),
        'b1_original':float(old.b1_sigma8/sigma8_z),
        'b1_changed':float(changed['fiducials'][PFS].b1_sigma8/sigma8_z),
        'density_changed':False,'nuisance_relations_imposed_in_fit':False}
    return changed


def bias_cases(cosmo, settings):
    original=study_cells(highz_profiles()['C22'],cosmo,settings=settings,extended=True)
    group=default_survey_group();output=[]
    for row in bias_schedule():
        matches=[c for c in original if c['region']=='overlap'
                 and c['expected']['radial_bin']==row['window']]
        if len(matches)!=1:raise ValueError('pilot window requires one complete radial cell')
        case=matches[0];z=case['expected']['model_redshift']
        nominal=case['expected']['nominal_bin_index']
        support=[c for c in original if c['expected']['nominal_bin_index']==nominal
                 and PFS in c['expected']['active_populations']]
        if len(support)!=1:raise ValueError('pilot must replace complete nominal-bin PFS support')
        population=case['fiducials']
        b_ref=(group.desi_tracers['DESI-ELG'].b1_of_z(z) if 'DESI-ELG' in population else 1.3)
        changed=vary_case(case,row['ratio'],row['prescription'],float(cosmo.sigma8(z)),b_ref)
        changed['label']=row['key'];changed['experiment']['window']=row['window']
        output.append(changed)
    return output
