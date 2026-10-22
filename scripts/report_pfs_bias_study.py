#!/usr/bin/env python
"""Compare the two-resolution pilot without claiming a survey-design optimum."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
import csv
import numpy as np

from pfsfog.extended_artifacts import jsonable,read_json
from pfsfog.elf_survey_evidence import canonical,file_digest,verify_files,write_json
from scripts.pfs_bias_cases import bias_schedule
from scripts.pfs_bias_inference import inference_sources
from scripts.run_pfs_bias_study import check_campaign

PARAMETERS=('ln10_10_As','Mnu','Omegam','fsigma8_z1.1')


def percent_improvement(reference,changed):
    if not np.isfinite([reference,changed]).all() or min(reference,changed)<=0:
        raise ValueError('uncertainties must be finite and positive')
    return float(100*(1-changed/reference))


def loop_response(coarse,fine):
    if not np.isfinite([coarse,fine]).all():raise ValueError('nonfinite loop comparison')
    half=abs(fine-coarse)
    return {'fine_percentage_points':float(fine),'coarse_percentage_points':float(coarse),
        'half_width_percentage_points':float(half),
        'sign_resolved_by_loop_pair':bool(abs(fine)>half),
        'full_grid_qualification':False}


def check_point_identity(point,expected,grid):
    if (point['status']!='passed' or point['key']!=expected['key'] or point['grid']!=grid
            or canonical(point['experiment'])!=canonical(expected['experiment'])):
        raise ValueError('unrelated or mislabeled bias population')


def report(root,output):
    root=Path(root).resolve();output=Path(output);campaign=check_campaign(root)
    inputs={};points={};summaries={}
    for grid in campaign['grids']:
        directory=root/'inference'/grid;s=read_json(directory/'summary.json');summaries[grid]=s
        if (s['status']!='passed' or s['grid']!=grid or s['sources']!=inference_sources()
                or s['request_sha256']!=file_digest(root/'request.json')
                or s['physical_summary_sha256']!=file_digest(root/'physical'/grid/'summary.json')
                or set(s['points'])!={r['key'] for r in bias_schedule()}):
            raise ValueError('failed, changed or incomplete pilot inference')
        inputs[str(directory/'summary.json')]=file_digest(directory/'summary.json')
        points[grid]={}
        for row in campaign['physical_requests'][grid]['expected']:
            key=row['key'];receipt=s['points'][key];path=directory/key
            if receipt['status']!='passed':raise ValueError('failed bias population')
            problems=verify_files(path,receipt['files'])
            if problems:raise ValueError('; '.join(problems))
            for filename,digest in receipt['files'].items():inputs[str(path/filename)]=digest
            p=read_json(path/'point.json')
            check_point_identity(p,row,grid)
            points[grid][key]=p
    rows=[];sigma_checks=[]
    for row in bias_schedule():
        key=row['key'];coarse=points['shell304'][key];fine=points['final-loop'][key]
        window=row['window'];iw=key.split('-')[0];basekey=iw+'-b1_only-r1.0'
        a,b=('D','C') if window[1]<=1.6 else ('H21','C21')
        for parameter in PARAMETERS:
            metric={'key':key,'window':window,'prescription':row['prescription'],'ratio':row['ratio'],
                    'b1':fine['experiment']['b1_changed'],'parameter':parameter,
                    'cross_comparison':a+' to '+b}
            changes={};crosses={}
            for grid,point in [('shell304',coarse),('final-loop',fine)]:
                baseline=points[grid][basekey]['profiles']['C22']['sigma'][parameter]
                sigma=point['profiles']['C22']['sigma'][parameter]
                changes[grid]=percent_improvement(baseline,sigma)
                crosses[grid]=percent_improvement(point['profiles'][a]['sigma'][parameter],
                                                  point['profiles'][b]['sigma'][parameter])
            metric['C22_improvement_over_original_bias']=loop_response(changes['shell304'],changes['final-loop'])
            metric['PFS_cross_improvement']=loop_response(crosses['shell304'],crosses['final-loop'])
            rows.append(metric)
        for name,profile in fine['profiles'].items():
            for parameter,sigma in profile['sigma'].items():
                residual=abs(sigma/coarse['profiles'][name]['sigma'][parameter]-1)
                sigma_checks.append({'key':key,'profile':name,'parameter':parameter,
                    'relative_change':residual,'below_one_percent':bool(residual<.01)})
    output.mkdir(parents=True,exist_ok=False)
    result={'status':'passed_two_resolution_pilot','scope':campaign['scope'],
        'grids':list(points),'populations_per_grid':15,'cosmology_coordinates':list(PARAMETERS[:3]),
        'inputs':inputs,'report_source':file_digest(Path(__file__)),
        'rows':rows,'loop_constraint_checks':sigma_checks,
        'maximum_loop_constraint_change':max(r['relative_change'] for r in sigma_checks),
        'all_loop_constraints_below_one_percent':all(r['below_one_percent'] for r in sigma_checks),
        'full_bias_grid_qualification':False,
        'interpretation':['One redshift window changes at a time; number density and prior widths are fixed.',
            'D to C removes/adds all baseline PFS cross spectra, with the stated one-window bias change.',
            'H21 to C21 removes/adds high-z PFS cross spectra; lower-z cross spectra remain.',
            'Loop differences are numerical sensitivity indicators, not statistical confidence intervals.',
            'Relations choose fiducials only; all nuisance coordinates remain independently marginalized.',
            'No feasible selection or optimal bias claim; the PFS density provenance remains unresolved.']}
    write_json(output/'report.json',jsonable(result))
    with (output/'response.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['window','prescription','bias_ratio','physical_b1','parameter',
            'C22_gain_percent','C22_loop_difference_pp','cross_comparison','cross_gain_percent','cross_loop_difference_pp'])
        for r in rows:
            gain=r['C22_improvement_over_original_bias'];cross=r['PFS_cross_improvement']
            writer.writerow([str(r['window']),r['prescription'],r['ratio'],r['b1'],r['parameter'],
                gain['fine_percentage_points'],gain['half_width_percentage_points'],r['cross_comparison'],
                cross['fine_percentage_points'],cross['half_width_percentage_points']])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=report(a.root,a.output)
    print(result['status'],'maximum relative loop change',result['maximum_loop_constraint_change'])
