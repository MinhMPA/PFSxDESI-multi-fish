"""Uncertainty and density-response comparisons for the corrected PFS survey."""
import numpy as np
from scripts.run_pfs_density_scan import FACTORS
from scripts.pfs_density_inference import PRIOR_MODES
from scripts.report_pfs_density_scan import QUANTITIES
from scripts.elf_study_report import all_profiles,constraint_values


def compare_values(coarse,fine):
    if coarse.keys()!=fine.keys() or not coarse:raise ValueError('comparison inventory differs')
    rows=[]
    for name,a in sorted(coarse.items()):
        b=fine[name]
        if not np.isfinite([a,b]).all() or min(a,b)<=0:raise ValueError('nonpositive uncertainty or ratio')
        difference=abs(b/a-1);guard=8*np.finfo(float).eps*max(1.,abs(b/a))
        rows.append({'label':name,'coarse':a,'fine':b,'relative_change':difference,
                     'roundoff_guard':guard,'status':'passed' if difference+guard<.01 else 'failed'})
    worst=max(rows,key=lambda r:r['relative_change'])
    return {'status':'passed' if all(r['status']=='passed' for r in rows) else 'failed',
            'tolerance':.01,'metrics':rows,'maximum_relative_change':worst['relative_change'],
            'worst_label':worst['label']}


def density_responses(points,windows,terminal):
    expected={f'w{i:02d}-{label}' for i in range(8) for label in FACTORS}
    if set(points)!=expected or len(windows)!=8:raise ValueError('incomplete density-point inventory')
    rows=[]
    for i,window in enumerate(windows):
        required=('fiducial','ln_m010','ln_p010','ln_m005','ln_p005')
        if any(points[f'w{i:02d}-{label}']['status']!='passed' for label in required):
            raise ValueError('failed local density response in window '+str(i))
        fid=points[f'w{i:02d}-fiducial']['profiles'][terminal]
        count=fid['count_at_q1']
        if not np.isfinite(count) or count<=0:raise ValueError('nonpositive PFS count')
        for label,q in FACTORS.items():
            p=points[f'w{i:02d}-{label}']
            if p['status']=='passed' and (p['factor']!=q or p['window_index']!=i
                    or p['profiles'][terminal]['count_at_q1']!=count):
                raise ValueError('density point has another reference or factor')
        for mode in PRIOR_MODES:
            values={label:points[f'w{i:02d}-{label}']['profiles'][terminal]['prior_modes'][mode]['sigma']
                    for label in FACTORS if points[f'w{i:02d}-{label}']['status']=='passed'}
            if any(set(v)!=set(QUANTITIES) or not np.isfinite(list(v.values())).all()
                    or min(v.values())<=0 for v in values.values()):raise ValueError('invalid density uncertainty')
            for parameter in QUANTITIES:
                fine=-np.log(values['ln_p005'][parameter]/values['ln_m005'][parameter])/.1
                coarse=-np.log(values['ln_p010'][parameter]/values['ln_m010'][parameter])/.2
                limit=max(1e-6,.02*abs(fine));difference=abs(fine-coarse)
                finite={label:float(100*(1-values[label][parameter]/values['fiducial'][parameter]))
                        if label in values else None for label in ('times2','times5','times10')}
                failures={label:points[f'w{i:02d}-{label}']['error']
                          for label in ('times2','times5','times10') if label not in values}
                rows.append({'window_index':i,'z_min':window[0],'z_max':window[1],
                    'comparison_profile':terminal,'prior_mode':mode,'parameter':parameter,
                    'count_at_q1':count,'elasticity':float(fine),'coarse_elasticity':float(coarse),
                    'per_added_galaxy':float(fine/count),'step_difference':float(difference),
                    'step_tolerance':float(limit),'step_check':'passed' if difference<limit else 'failed',
                    'finite_error_reduction_percent':finite,'finite_failures':failures})
    return rows


def main_values(outcomes):
    values=constraint_values({n:outcomes[n] for n in all_profiles()})
    if set(outcomes)!=set(all_profiles())|{'C24'}:raise ValueError('incomplete corrected forecast inventory')
    r=outcomes['C24']
    if r['status']!='passed' or r['cosmology_names']!=['ln10_10_As','Mnu','Omegam']:
        raise ValueError('invalid new tail cosmology')
    for kind in ('marginalized','nuisance_fixed'):
        for field in ('cosmology','growth'):
            for i,x in enumerate(r[kind][field]):
                if not np.isfinite(x) or x<=0:raise ValueError('invalid tail uncertainty')
                values[f'C24/{kind}/{field}/{i}']=x
        for a in ('B','C','C21'):
            for field in ('cosmology','growth'):
                left=outcomes[a][kind][field];right=r[kind][field]
                for i,x in enumerate(left):
                    if field=='growth' and outcomes[a]['growth_z'][i]!=r['growth_z'][i]:
                        raise ValueError('unmatched common growth coordinates')
                    values[f'ratios/{a}_over_C24/{kind}/{field}/{i}']=x/right[i]
                    values[f'ratios/C24_over_{a}/{kind}/{field}/{i}']=right[i]/x
    return values
