"""Export the scientific results with explicit numerical and model boundaries."""
from pathlib import Path
import csv

import numpy as np
from pfsfog.extended_artifacts import read_json,write_json,file_hash
from scripts.elf_study_report import write_tables,all_profiles,comparisons,load_result
from scripts.elf_study_diagnostics import bin_diagnostics


def extended_review(path):
    summary=read_json(path/'summary.json')
    if summary['status']!='passed':raise ValueError('extended-cosmology study did not pass')
    treatments=summary['outcomes']['B'];rows={}
    for name in treatments:
        rows[name]={'cases':{}}
        for case in ('B','C'):
            source=summary['outcomes'][case][name]
            if source['status']!='passed':raise ValueError('extended treatment failed: '+case+' '+name)
            with np.load(path/'cases'/case/'arrays.npz',allow_pickle=False) as a:
                cov=a[name+'/growth_cov_marg'];fixed=a[name+'/growth_cov_nuisance_fixed']
                sigma=np.sqrt(np.diag(cov));row={**source,'growth_covariance':cov.tolist(),
                    'growth_correlation':(cov/sigma[:,None]/sigma[None,:]).tolist(),
                    'growth_nuisance_penalty':np.sqrt(np.diag(cov)/np.diag(fixed)).tolist()}
                rows[name]['cases'][case]=row
        x=rows[name]['cases']['B']['marginalized']['growth'][3]
        y=rows[name]['cases']['C']['marginalized']['growth'][3]
        rows[name]['headline_z1.1_error_reduction_percent']=100*(1-y/x)
    return {'status':'reviewed_from_bound_full_covariances','treatments':rows,
            'source_summary':str(path/'summary.json'),'source_sha256':file_hash(path/'summary.json')}


def scientific_report(root):
    # Import here to avoid a cycle with the command entry point.
    from scripts.run_elf_studies import check_campaign,check_bindings,old_inference,fresh_directory
    check_campaign(root);verification=read_json(root/'verification.json')
    if verification['status']!='passed':raise ValueError('fresh whole-study verification is required')
    comparison=read_json(root/'comparisons/summary.json');check_bindings(comparison['input_files'])
    if comparison['status']!='passed':raise ValueError('unqualified resolution comparison')
    output=root/'report';fresh_directory(output)
    outcomes=read_json(root/'inference/shell304/summary.json')['outcomes']
    gains=write_tables(outcomes,output);extended=extended_review(old_inference('shell304'))
    write_json(output/'extended_cosmology_review.json',extended)
    diagnostic_profiles=('I','A','B','D','C','H21','C21','C22')
    bins={n:bin_diagnostics(load_result(root/'inference/shell304/profiles'/n)[0]) for n in diagnostic_profiles}
    write_json(output/'bin_only_and_global_correlations.json',bins)
    failed_bins=[n+':bin'+str(r['nominal_bin_index']) for n,rows in bins.items() for r in rows if r['bin_only']['status']!='passed']
    lines=['# DESI–PFS information-content study with ELF','',
        'The accepted13main configurations and five high-redshift profiles are numerically verified. '
        'Independent final scientific/source review is recorded separately. No manuscript numbers are '
        'promoted by this report alone.','',
        'The main question is the incremental information from the smaller PFS survey after exploiting '
        'DESI internal multi-tracer information. All headline forecasts vary ln(10^10 A_s), Mnu and Omega_m, '
        'with h, omega_b and n_s fixed. Growth is derived from this one global cosmology; the redshift '
        'outputs have a full correlated covariance and are not independent growth parameters.','',
        '| Configuration | sigma[ln(10^10 A_s)] | sigma(Mnu) [eV] | sigma(Omega_m) | sigma[f sigma8](z=1.1) |',
        '|---|---:|---:|---:|---:|']
    for name in all_profiles():
        r=outcomes[name];c=r['marginalized']['cosmology'];growth=r['marginalized']['growth'][3]
        lines.append(f'| {name} | {c[0]:.7g} | {c[1]:.7g} | {c[2]:.7g} | {growth:.7g} |')
    lines += ['', '## Information attribution','',
        '| Change of observations | Reduction in sigma[f sigma8](z=1.1) | Reduction in sigma(Mnu) |',
        '|---|---:|---:|']
    for a,b in comparisons():
        r=gains[a+'_to_'+b]['marginalized'];lines.append(f'| {a} → {b} | {r["growth"][3]:.5f}% | {r["cosmology"][1]:.5f}% |')
    lines += ['',
        'I→A changes the independent-auto approximation to the full mutual covariance; it is not an '
        'information-nesting statement. A→B adds measured DESI crosses, B→D adds PFS autos, and D→C '
        'adds measured PFS–DESI crosses. B→C is the total incremental PFS gain. The successive '
        'percentages use different denominators and must not be added.','',
        'C16 ends PFS at z=1.6. H21 adds PFS autos to z=2.1 while retaining the theoretical PFS–QSO '
        'cross signal in the covariance. C21 also measures those crosses. C22 adds only the native '
        'PFS volume at 2.1<z<2.2; its terminal non-overlap region is empty. The common eight growth '
        'outputs are compared consistently; the additional z=2.15 output is supplementary.','',
        '## Prior dependence and nuisance degeneracies','',
        'The main nuisance factors3and10 multiply finite nuisance widths; flat linear-bias priors '
        'remain flat. The broad Mnu/Omega_m widths remain5eV/1, and no external amplitude prior enters. '
        'The nuisance penalty is the ratio of marginalized to nuisance-fixed uncertainty. Exact '
        'prior-only parameter lists, full growth covariance/correlation, and all penalties are retained '
        'in the JSON tables and profile matrices.','',
        'The accompanying bin_only_and_global_correlations.json records cosmology/linear-bias '
        'marginal correlations for every nominal bin. Global slices use all survey bins. Bin-only '
        'covariances use only that nominal bin and insert its priors once; they are separately inverted. '
        'No global covariance submatrix is described as a bin-only forecast. Explicit failed bin-only '
        'inversions: '+(', '.join(failed_bins) if failed_bins else 'none')+'.','',
        '| Baseline case | Nuisance penalty: f sigma8(z=1.1) | Mnu | Omega_m |',
        '|---|---:|---:|---:|']
    for name in ('I','A','B','D','C'):
        q=outcomes[name]['nuisance_penalty']
        lines.append(f'| {name} | {q["growth"][3]:.5g} | {q["cosmology"][1]:.5g} | {q["cosmology"][2]:.5g} |')
    lines += ['',
        'Each penalty marginalizes the other cosmological coordinates in both calculations. '
        'It measures the effect of freeing the declared nuisance coordinates under their stated priors; '
        'it is not an additive information fraction or a unique attribution to linear bias.','',
        '| Extended-cosmology treatment | PFS growth-error reduction at z=1.1 |','|---|---:|']
    for name,r in extended['treatments'].items():
        lines.append(f'| {name} | {r["headline_z1.1_error_reduction_percent"]:.5f}% |')
    lines += ['',
        'The extended check retains the separately accepted omega_b and n_s prior prescriptions '
        '(sigma=0.00055 and0.042 where specified). Its weak10 variant changes those cosmological '
        'priors; it must not be confused with the main-paper nuisance-width variants.','',
        '## Numerical evidence and scientific limits','',
        'Every profile uses the selected2048loop-node,64angle,304shell grid, with '
        'k=0.01–0.20 h_ref/Mpc (0.15 for the declared scale-cut variants). All four required '
        'loop/angular/shell refinement directions are checked on each absolute uncertainty and both '
        'ratio directions. The threshold is strictly1%; local mean/response/covariance checks use5e-4. '
        'The shell test uses the documented direct-common-k criterion and retains the original '
        'interpolation diagnostics. The direct reconstruction does not independently bound a shared '
        'physical-model error; the full-Fisher refinement bounds the stated integration change.','',
        'All means include the complete AP mapping and its derivatives. Signed finite differences, '
        'field and data covariance positivity, native-three-coordinate controls, shared priors, '
        'information nesting and saved-data reconstruction are retained. No clipping, jitter or '
        'pseudoinverse is applied.','',
        'These remain Fisher forecasts for the accepted population-uniformity assumptions, total-matter '
        'one-loop approximation, IR-off ELF model and declared nuisance/FoG prescription. Numerical '
        'convergence is not a mock-data or nonlinear-model validity test. ELF does not introduce '
        'neutrino-specific loop dynamics. Revised manuscript and referee-response integration remains '
        'a separate final step before the October14deadline.','']
    (output/'SCIENTIFIC_REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for tag,names,base in [('main',['I','A','B','D','C'],'B'),('highz',['B16','C16','H21','C21','C22'],'B16')]:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
        for ax,field,index,label in [(axes[0],'growth',3,r'$\sigma[f\sigma_8](z=1.1)$'),
                                     (axes[1],'cosmology',1,r'$\sigma(M_\nu)$ [eV]')]:
            values=[outcomes[n]['marginalized'][field][index] for n in names]
            ax.bar(names,values,color=['#557a95' if n==base else '#86b3ad' for n in names]);ax.set_ylabel(label)
        for ext in ('pdf','svg','png'):fig.savefig(output/(tag+'_constraints.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained');names=['I','A','B','D','C']
    for ax,field,index,label in [(axes[0],'growth',3,r'$f\sigma_8(z=1.1)$'),
                                 (axes[1],'cosmology',1,r'$M_\nu$'),(axes[2],'cosmology',2,r'$\Omega_m$')]:
        ax.bar(names,[outcomes[n]['nuisance_penalty'][field][index] for n in names],color='#557a95')
        ax.axhline(1,color='0.4',lw=.8);ax.set_title(label);ax.set_ylabel(r'$\sigma_{\rm marg}/\sigma_{\rm fixed}$')
    for ext in ('pdf','svg','png'):fig.savefig(output/('nuisance_penalty.'+ext),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4.6),layout='constrained')
    r=outcomes['C22'];im=ax.imshow(r['growth_correlation'],vmin=-1,vmax=1,cmap='RdBu_r')
    labels=[f'{z:g}' for z in r['growth_z']];ax.set_xticks(range(9),labels);ax.set_yticks(range(9),labels)
    ax.set_xlabel('Redshift');ax.set_ylabel('Redshift');fig.colorbar(im,ax=ax,label=r'Correlation of $f\sigma_8$')
    for ext in ('pdf','svg','png'):fig.savefig(output/('growth_correlation.'+ext),dpi=180)
    plt.close(fig)
    # A representative overlapping bin plus complete per-bin numerical evidence.
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    for column,name in enumerate(('B','D','C')):
        row=bins[name][3];labels=[n if ':' not in n else n.split(':')[1]+' b1 sigma8' for n in row['labels']]
        for iy,(kind,tag) in enumerate((('global_marginal_correlation','All bins'),('marginal_correlation','Bin only'))):
            ax=axes[iy,column]
            matrix=row[kind] if iy==0 else row['bin_only'].get(kind)
            if matrix is None:ax.text(.5,.5,'Unqualified bin-only inversion',ha='center');ax.set_axis_off();continue
            im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set_title(name+' — '+tag+', z=1.1');ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right')
            ax.set_yticks(range(len(labels)),labels)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Correlation')
    for ext in ('pdf','svg','png'):fig.savefig(output/('cosmology_bias_correlations.'+ext),dpi=180)
    plt.close(fig)
    return {'status':'written','path':str(output),'profiles':18,'independent_final_review':'pending'}
