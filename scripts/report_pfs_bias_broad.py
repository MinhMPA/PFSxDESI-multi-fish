#!/usr/bin/env python
"""Conditional PFS bias responses with explicit numerical resolution and scope."""
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


def numerical_interval(values,refinements=REFINEMENTS):
    if set(values)!=set(GRIDS) or not np.isfinite(list(values.values())).all():
        raise ValueError('numerical resolution requires all five finite grid values')
    fine=float(values['final-loop'])
    changes={a+'-to-'+b:abs(values[b]-values[a]) for a,b,_ in refinements}
    spread=max(abs(v-fine) for v in values.values())
    width=max(spread,max(changes.values()),1e-5,.01*abs(fine))
    return {'value':fine,'half_width':float(width),'lower':fine-width,'upper':fine+width,
            'grid_values':values,'refinement_changes':changes,'final_grid_spread':spread,
            'sign_resolved':bool(abs(fine)>width),'statistical_confidence_interval':False}


def unresolved_groups(rows):
    if len({r['key'] for r in rows})!=len(rows):raise ValueError('duplicate ranking population')
    if any(not np.isfinite([r['value'],r['half_width']]).all() or r['half_width']<0 for r in rows):
        raise ValueError('invalid numerical interval')
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(rows):
        for j,b in enumerate(rows[:i]):
            guard=8*np.finfo(float).eps*max(1.,abs(a['value']),abs(b['value']))
            if abs(a['value']-b['value'])<=a['half_width']+b['half_width']+guard:
                parent[root(i)]=root(j)
    groups={}
    for i,row in enumerate(rows):groups.setdefault(root(i),[]).append(row)
    ordered=sorted(groups.values(),key=lambda g:max(r['value'] for r in g),reverse=True)
    return [{'members':[r['key'] for r in sorted(g,key=lambda r:r['value'],reverse=True)],
             'unresolved':len(g)>1,'value_range':[min(r['value'] for r in g),max(r['value'] for r in g)]}
            for g in ordered]


def selected_curve(rows,window,prescription,parameter):
    selected=[deepcopy(r) for r in rows if r['window']==window and r['parameter']==parameter
              and (r['prescription']==prescription or r['ratio']==1.)]
    if len({r['key'] for r in selected})!=len(selected):raise ValueError('duplicated curve population')
    if not selected:raise ValueError('missing curve population')
    lo=min(r['ratio'] for r in selected);hi=max(r['ratio'] for r in selected)
    for row in selected:
        row['reused_original_population']=row['ratio']==1. and row['prescription']!=prescription
        row['on_tested_boundary']=row['ratio'] in (lo,hi)
    return sorted(selected,key=lambda r:r['ratio'])


def qualified_comparisons(summary):
    filenames={a+'-to-'+b+'.json' for a,b,_ in REFINEMENTS}
    if (summary['status']!='passed' or summary['full_bias_grid_qualification'] is not True
            or canonical(summary['refinements'])!=canonical(REFINEMENTS)
            or set(summary['reports'])!=filenames):
        raise ValueError('broad report requires all four qualified refinements')


def report(root,output):
    root=Path(root).resolve();output=Path(output);campaign=check_campaign(root)
    directory=root/'comparisons';summary=read_json(directory/'summary.json')
    qualified_comparisons(summary)
    if summary['sources']!=comparison_sources():raise ValueError('comparison sources changed')
    problems=verify_files(directory,summary['reports'])
    if problems:raise ValueError('; '.join(problems))
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


def render_figures(result,output):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    titles=(r'$\ln(10^{10}A_s)$',r'$M_\nu$',r'$\Omega_m$',r'$f\sigma_8(z=1.1)$')
    windows=([.7,.8],[1.4,1.6],[1.6,2.])
    for metric,filename in [('C22_gain','global-bias-response'),('cross_gain','cross-spectrum-bias-response')]:
        fig,axes=plt.subplots(3,4,figsize=(11.2,7.4),layout='constrained')
        for i,window in enumerate(windows):
            for j,parameter in enumerate(PARAMETERS):
                ax=axes[i,j]
                for prescription,color,style,label in [
                    ('b1_only','#176b91','-','Change $b_1$ only'),
                    ('adopted_relations','#bc652b','--','Adopted population relations')]:
                    curve=selected_curve(result['rows'],window,prescription,parameter)
                    x=np.array([r['ratio'] for r in curve]);y=np.array([r[metric]['value'] for r in curve])
                    width=np.array([r[metric]['half_width'] for r in curve])
                    ax.plot(x,y,style,color=color,marker='o',ms=3,lw=1.4,label=label)
                    ax.fill_between(x,y-width,y+width,color=color,alpha=.12,linewidth=0)
                equal={r['equal_to']:r['ratio'] for r in result['rows'] if r['window']==window and r['equal_to']}
                for target,ratio in equal.items():
                    ax.axvline(ratio,color='.55',ls=':',lw=.8)
                    if i==0 or j==0:
                        ax.text(ratio,.97,target.removeprefix('DESI-'),transform=ax.get_xaxis_transform(),
                                rotation=90,va='top',ha='right',fontsize=7,color='.35')
                ax.axhline(0,color='.65',lw=.6);ax.tick_params(labelsize=8)
                ax.spines[['top','right']].set_visible(False)
                if i==0:ax.set_title(titles[j],fontsize=11)
                if i==2:ax.set_xlabel(r'$b_{1,\rm PFS}/b_{1,\rm fid}$',fontsize=10)
                if j==0:ax.set_ylabel(f'{window[0]:g} < z < {window[1]:g}\nError reduction [%]',fontsize=10)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=10)
        for suffix in ('pdf','svg','png'):
            fig.savefig(output/(filename+'.'+suffix),dpi=180)
        plt.close(fig)
    (output/'FIGURE_CAPTIONS.md').write_text(
        'Global bias response: percent reductions of marginalized C22 errors relative to the original population. '
        'Each row changes one PFS radial window at fixed density and survey geometry. The three cosmological '
        'coordinates determine the correlated derived growth constraint at z=1.1.\n\n'
        'Cross-spectrum bias response: matched D to C error reductions for the first two rows and H21 to C21 '
        'for the third. These add all PFS cross spectra below z=1.6 or over 1.6–2.1, respectively, at the '
        'same changed population; they do not isolate crosses in the one varied window.\n\n'
        'Solid/dashed curves distinguish changing b1 alone from changing adopted population fiducials. '
        'Both retain independent nuisance marginalization and identical prior widths. Shading shows the '
        'declared numerical resolution across four refinements, not statistical uncertainty. Dotted lines '
        'mark equality with the indicated DESI linear bias. Curves join sampled conditional forecasts; '
        'neither a boundary preference nor a matched bias establishes a feasible selection optimum.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=report(args.root,args.output);render_figures(result,args.output)
    print(result['status'],len(result['rows']),'parameter-response rows')
