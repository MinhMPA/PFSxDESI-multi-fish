#!/usr/bin/env python
"""Report-only inventory repair; the original report attempt remains frozen."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ps_1loop_jax'))

import argparse
from copy import deepcopy
import csv

import numpy as np

from pfsfog.elf_survey_evidence import canonical, file_digest, verify_files, write_json
from pfsfog.extended_artifacts import read_json, jsonable
from scripts.pfs_bias_broad_cases import broad_schedule
from scripts.pfs_bias_broad_comparison import comparison_sources
from scripts.pfs_bias_broad_inference import verify_inference
from scripts.report_pfs_bias_study import percent_improvement
from scripts.run_pfs_bias_broad import check_campaign, GRIDS, REFINEMENTS
from scripts.run_elf_studies import check_bindings

PARAMETERS=('ln10_10_As','Mnu','Omegam','fsigma8_z1.1')
METRICS=('C22_gain','cross_gain')


from scripts.report_pfs_bias_broad import (numerical_interval, unresolved_groups,
    selected_curve, qualified_comparisons, render_figures)


def verify_comparison_inventory(directory, summary):
    directory = Path(directory)
    if canonical(read_json(directory/'summary.json')) != canonical(summary):
        raise ValueError('comparison summary changed during verification')
    recorded = {**summary['reports'], 'summary.json': file_digest(directory/'summary.json')}
    problems = verify_files(directory, recorded)
    if problems:
        raise ValueError('; '.join(problems))


def report(root,output):
    root=Path(root).resolve();output=Path(output);campaign=check_campaign(root)
    directory=root/'comparisons';summary=read_json(directory/'summary.json')
    qualified_comparisons(summary)
    if summary['sources']!=comparison_sources():raise ValueError('comparison sources changed')
    verify_comparison_inventory(directory,summary)
    check_bindings(summary['input_files'])
    inputs={str(directory/'summary.json'):file_digest(directory/'summary.json'),**summary['input_files']}
    schedule=broad_schedule();keys={r['key'] for r in schedule};maximum_global=0.
    for coarse,fine,axis in REFINEMENTS:
        path=directory/(coarse+'-to-'+fine+'.json');r=read_json(path)
        if (r['status']!='passed' or (r['coarse'],r['fine'],r['axis'])!=(coarse,fine,axis)
                or len(r['populations'])!=35 or {p['key'] for p in r['populations']}!=keys
                or any(p['status']!='passed' or p['global']['status']!='passed'
                       or not p['local'] or any(x['measurement']['status']!='passed' for x in p['local'])
                       for p in r['populations'])):
            raise ValueError('incomplete or failed broad population refinement')
        maximum_global=max(maximum_global,max(p['global']['maximum_relative_change'] for p in r['populations']))
        inputs[str(path)]=file_digest(path)
    verified={g:verify_inference(root,g) for g in GRIDS}
    for value in verified.values():inputs.update(value['inputs'])
    original={tuple(r['window']):r['key'] for r in schedule if r['ratio']==1.}
    rows=[]
    for row in schedule:
        key=row['key'];ref=original[tuple(row['window'])]
        a,b=('D','C') if row['window'][1]<=1.6 else ('H21','C21')
        fine=verified['final-loop']['points'][key]
        for parameter in PARAMETERS:
            values={metric:{} for metric in METRICS}
            for grid,record in verified.items():
                point=record['points'][key];baseline=record['points'][ref]
                values['C22_gain'][grid]=percent_improvement(
                    baseline['profiles']['C22']['sigma'][parameter],point['profiles']['C22']['sigma'][parameter])
                values['cross_gain'][grid]=percent_improvement(
                    point['profiles'][a]['sigma'][parameter],point['profiles'][b]['sigma'][parameter])
            rows.append({**row,'parameter':parameter,'b1':fine['experiment']['b1_changed'],
                         'cross_comparison':a+' to '+b,
                         **{metric:numerical_interval(value) for metric,value in values.items()}})
    rankings=[]
    for window in ([.7,.8],[1.4,1.6],[1.6,2.]):
        for prescription in ('b1_only','adopted_relations'):
            for parameter in PARAMETERS:
                curve=selected_curve(rows,window,prescription,parameter)
                for metric in METRICS:
                    groups=unresolved_groups([{'key':r['key'],**r[metric]} for r in curve])
                    rankings.append({'window':window,'prescription':prescription,'parameter':parameter,
                        'metric':metric,'groups':groups,'boundary_keys':[r['key'] for r in curve if r['on_tested_boundary']],
                        'interpretation':'conditional ordering of sampled populations; no extrapolated or feasible optimum'})
    output.mkdir(parents=True,exist_ok=False)
    result={'schema':'pfs-bias-broad-report-v1','status':'passed_five_grid_study',
        'full_bias_grid_qualification':True,'scope':campaign['scope'],
        'physical_populations':175,'common_k_populations':35,'profile_calculations':525,
        'grids':list(GRIDS),'refinements':REFINEMENTS,'cosmology_coordinates':list(PARAMETERS[:3]),
        'inputs':inputs,'report_source':file_digest(Path(__file__)),
        'maximum_relative_global_refinement_change':maximum_global,
        'rows':rows,'rankings':rankings,
        'numerical_resolution_convention':'max(four-refinement gain changes, final-grid spread, 1e-5 percentage points, 1% of absolute gain); connected overlapping intervals remain unresolved',
        'interpretation':['Density, geometry, DESI fiducials and independent nuisance-prior widths are fixed.',
            'Population relations set fiducials only; all nuisance coordinates remain independent.',
            'C22 gains change one PFS population window at a time in the same global survey.',
            'D to C adds all baseline PFS cross spectra; H21 to C21 adds all PFS crosses over 1.6–2.1.',
            'Numerical intervals are sensitivity indicators, not statistical confidence intervals.',
            'A boundary preference is not an optimum beyond the measured range.',
            'Feasible selection requires a joint density/bias/stochasticity/success/cost model and identified density provenance.']}
    write_json(output/'report.json',jsonable(result))
    with (output/'response.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['window','prescription','bias_ratio','physical_b1','equal_to','parameter',
                         'C22_gain_percent','C22_resolution_pp','cross_comparison','cross_gain_percent','cross_resolution_pp'])
        for row in rows:
            writer.writerow([str(row['window']),row['prescription'],row['ratio'],row['b1'],row['equal_to'],
                row['parameter'],row['C22_gain']['value'],row['C22_gain']['half_width'],row['cross_comparison'],
                row['cross_gain']['value'],row['cross_gain']['half_width']])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=report(args.root,args.output);render_figures(result,args.output)
    print(result['status'],len(result['rows']),'parameter-response rows')
