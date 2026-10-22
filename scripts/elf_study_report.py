"""Saved uncertainties, full growth covariances and survey information comparisons."""
from dataclasses import asdict
from pathlib import Path
import csv

import numpy as np
from pfsfog.extended_artifacts import (PRECISION_FIELDS,GROWTH_FIELDS,pack_likelihood,
    validate_likelihood,read_json,write_json,canonical,jsonable,require_close,file_hash)
from pfsfog.revision_forecast import CellLikelihood,RevisionLikelihood,ShellGrid,PriorResult
from pfsfog.fisher_precision import DECIMAL_FISHER_SOLVER
from scripts.elf_study_profiles import paper_profiles,highz_profiles,StudyConfig,study_config,study_cells
from scripts.elf_study_inference import assemble_study,infer_study,validate_growth,validate_study_likelihood


def all_profiles(): return {**paper_profiles(),**highz_profiles()}


def comparisons():
    return [('I','A'),('A','B'),('B','D'),('D','C'),('B','C')]+[
        ('B'+suffix,'C'+suffix) for suffix in ('_k015','_prior3','_prior10','_no_elg1')]+[
        ('B16','C16'),('B16','C21'),('B16','C22'),('C16','H21'),('H21','C21'),('C21','C22')]


def constraint_row(result):
    row={'status':'passed','cosmology_names':list(result.likelihood.cosmo_names),
        'growth_z':result.growth_z.tolist(),'growth_fiducial':result.growth_fiducial.tolist(),
        'parameter_count':len(result.names),'prior_only':result.validation['prior_only'],
        'cell_count':len(result.likelihood.cells),'empty_cells':result.likelihood.validation['empty_cells']}
    for kind,suffix in [('marginalized','marg'),('nuisance_fixed','nuisance_fixed')]:
        row[kind]={'cosmology':np.sqrt(np.diag(getattr(result,'cov_cosmo_'+suffix))).tolist(),
                   'growth':np.sqrt(np.diag(getattr(result,'growth_cov_'+suffix))).tolist()}
    row['nuisance_penalty']={k:(np.array(row['marginalized'][k])/row['nuisance_fixed'][k]).tolist()
                             for k in ('cosmology','growth')}
    v=np.sqrt(np.diag(result.growth_cov_marg));row['growth_correlation']=(result.growth_cov_marg/v[:,None]/v[None,:]).tolist()
    return row


def constraint_values(outcomes):
    profiles=all_profiles()
    if set(outcomes)!=set(profiles):raise ValueError('incomplete18-profile inventory')
    values={}
    for name,p in profiles.items():
        r=outcomes[name];z=[sum(b)/2 for b in p.nominal_bins]
        if r['status']!='passed' or r['cosmology_names']!=['ln10_10_As','Mnu','Omegam'] or r['growth_z']!=z:
            raise ValueError('failed outcome or incorrect growth/cosmology labels: '+name)
        for kind in ('marginalized','nuisance_fixed'):
            for field,count in [('cosmology',3),('growth',len(z))]:
                a=np.asarray(r[kind][field])
                if a.shape!=(count,) or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('invalid uncertainty')
                values.update({f'{name}/{kind}/{field}/{i}':float(x) for i,x in enumerate(a)})
    for a,b in comparisons():
        count=min(len(profiles[a].nominal_bins),len(profiles[b].nominal_bins))
        for kind in ('marginalized','nuisance_fixed'):
            for field,n in [('cosmology',3),('growth',count)]:
                for i in range(n):
                    x=values[f'{a}/{kind}/{field}/{i}'];y=values[f'{b}/{kind}/{field}/{i}']
                    values[f'ratios/{a}_over_{b}/{kind}/{field}/{i}']=x/y
                    values[f'ratios/{b}_over_{a}/{kind}/{field}/{i}']=y/x
    return values


def compare_results(coarse,fine):
    a=constraint_values(coarse);b=constraint_values(fine);metrics=[]
    for name in sorted(a):
        delta=abs(b[name]/a[name]-1);guard=8*np.finfo(float).eps*max(1.,abs(b[name]/a[name]))
        metrics.append({'label':name,'coarse':a[name],'fine':b[name],'relative_change':delta,
                        'roundoff_guard':guard,'status':'passed' if delta+guard<.01 else 'failed'})
    worst=max(metrics,key=lambda r:r['relative_change'])
    return {'status':'passed' if all(m['status']=='passed' for m in metrics) else 'failed',
        'tolerance':.01,'metrics':metrics,'maximum_relative_change':worst['relative_change'],'worst_label':worst['label']}


def verify_comparison_record(saved,recomputed):
    if recomputed['status']!='passed' or canonical(saved)!=canonical(recomputed):
        raise ValueError('saved resolution comparison differs from recomputed evidence')
    return {'status':'passed'}


def save_result(path,profile,settings,result,projection_refs):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if list(path.iterdir()):raise FileExistsError('study result already exists')
    arrays={};meta=pack_likelihood(result.likelihood,arrays)
    for key in (*PRECISION_FIELDS,*GROWTH_FIELDS):arrays['result/'+key]=getattr(result,key)
    np.savez_compressed(path/'arrays.npz',**arrays)
    write_json(path/'manifest.json',{'schema':'elf-study-result-v1','profile':asdict(profile),
        'settings':settings,'likelihood':meta,'nuisance_factor':result.nuisance_factor,
        'parameters':result.parameters,'validation':result.validation,'fisher_solver':result.fisher_solver,
        'projections':projection_refs,'arrays_sha256':file_hash(path/'arrays.npz')})
    write_json(path/'constraints.json',constraint_row(result))


def load_result(path):
    path=Path(path);record=read_json(path/'manifest.json');meta=record['likelihood']
    if record['schema']!='elf-study-result-v1' or file_hash(path/'arrays.npz')!=record['arrays_sha256']:
        raise ValueError('saved study array binding differs')
    c=dict(meta['configuration']);c['nominal_bins']=tuple(map(tuple,c['nominal_bins']));config=StudyConfig(**c)
    with np.load(path/'arrays.npz',allow_pickle=False) as data:a={k:data[k].copy() for k in data.files}
    cells=[CellLikelihood(a[f'likelihood/cell{i}/F_like'],r['names'],r['metadata'],r['validation'],
        a[f'likelihood/cell{i}/derivatives'],a[f'likelihood/cell{i}/covariance'],a[f'likelihood/cell{i}/fiducial_mean'])
        for i,r in enumerate(meta['cells'])]
    like=RevisionLikelihood(config,a['likelihood/F_like'],meta['names'],a['likelihood/fiducial'],meta['parameters'],cells,
        list(a['likelihood/bin_F_like']),ShellGrid(*(a['likelihood/grid_'+n] for n in ('edges','centers','widths'))),
        meta['reference_cosmology'],meta['model'],meta['validation'])
    result=PriorResult(likelihood=like,nuisance_factor=record['nuisance_factor'],parameters=record['parameters'],
        validation=record['validation'],fisher_solver=record['fisher_solver'],
        **{k:a['result/'+k] for k in (*PRECISION_FIELDS,*GROWTH_FIELDS) if k!='F_like'})
    require_close(result.F_like,a['result/F_like'],'saved likelihood Fisher')
    return result,record


def verify_result(path,profile,settings):
    result,meta=load_result(path)
    if (canonical(meta['profile'])!=canonical(asdict(profile)) or meta['settings']!=settings
            or canonical(asdict(result.likelihood.config))!=canonical(asdict(study_config(profile,settings)))
            or result.fisher_solver!=DECIMAL_FISHER_SOLVER):raise ValueError('study result profile or solver differs')
    validate_study_likelihood(result.likelihood);validate_growth(result)
    recomputed=infer_study(result.likelihood)
    for key in (*PRECISION_FIELDS,*GROWTH_FIELDS):
        require_close(getattr(result,key),getattr(recomputed,key),'saved '+key,rtol=1e-8,atol=1e-12)
    if canonical(result.parameters)!=canonical(recomputed.parameters):raise ValueError('saved prior parameter records differ')
    if canonical(constraint_row(result))!=canonical(read_json(Path(path)/'constraints.json')):
        raise ValueError('saved constraint table differs')
    return {'status':'passed','profile':profile.name,'cells':len(result.likelihood.cells),'parameters':len(result.names)}


def nesting_snapshot(result):
    like=result.likelihood;cells=[]
    for c in like.cells:
        m=c.metadata
        if not m['retained_pairs']:continue
        cells.append({'cell_id':m['cell_id'],'geometry':{k:m[k] for k in
            ('nominal_bin','radial_bin','region','model_redshift','volume','nbar')},
            'pairs':m['retained_pairs'],'names':c.names,'derivatives':c.derivatives,'covariance':c.covariance})
    return {'names':like.names,'fiducial':like.fiducial,'prior_sigma':[p['prior_sigma'] for p in result.parameters],
        'F_like':like.F_like,'k':like.grid.centers,'dk':like.grid.widths,'cells':cells}


def check_projection_binding(result,profile,settings,parts,reference,model):
    rebuilt=assemble_study(profile,settings,parts,reference,model)
    like=result.likelihood
    for field in ('names','parameters','reference_cosmology','model'):
        if canonical(getattr(like,field))!=canonical(getattr(rebuilt,field)):
            raise ValueError('source-selected global '+field+' differs')
    for field in ('fiducial','F_like','bin_F_like'):
        require_close(getattr(like,field),getattr(rebuilt,field),'source-selected global '+field)
    return {'status':'passed','source_selected_cells':len(parts)}


def write_tables(outcomes,path):
    path=Path(path);constraint_values(outcomes);write_json(path/'constraints.json',outcomes)
    with (path/'constraints.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['profile','kind','quantity','sigma'])
        for name,row in outcomes.items():
            for kind in ('marginalized','nuisance_fixed'):
                for label,x in zip(row['cosmology_names'],row[kind]['cosmology']):w.writerow([name,kind,label,x])
                for z,x in zip(row['growth_z'],row[kind]['growth']):w.writerow([name,kind,'fsigma8_z'+str(z),x])
    changes={}
    for a,b in comparisons():
        changes[a+'_to_'+b]={}
        for kind in ('marginalized','nuisance_fixed'):
            changes[a+'_to_'+b][kind]={}
            for field in ('cosmology','growth'):
                x=np.array(outcomes[a][kind][field]);y=np.array(outcomes[b][kind][field])[:len(x)]
                changes[a+'_to_'+b][kind][field]=(100*(1-y/x)).tolist()
    write_json(path/'error_reductions_percent.json',changes)
    return changes
