"""Phase inventory, explicit external pins, actual exits and report semantics."""
import importlib
import subprocess
import sys

import pytest


def _synthetic_runtime(monkeypatch=None):
    """Authorize only this controlled test process as its synthetic runtime.

    Actual frozen-export authentication is exercised, in its original Python,
    by test_emulator_paths; production recovery authority is not changed.
    """
    import os
    from pathlib import Path
    from scripts import export_cosmology_oracle as e, run_extended_cosmology as d
    pythonpath = f'{d.ROOT}:{d.ROOT}/ps_1loop_jax:/Users/nguyenmn/cosmopower-jax'
    if monkeypatch is None:  # Analytic child process only.
        e.RECOVERY = Path(sys.prefix)
        os.environ['PYTHONPATH'] = pythonpath
    else:
        monkeypatch.setattr(e, 'RECOVERY', Path(sys.prefix))
        monkeypatch.setenv('PYTHONPATH', pythonpath)


@pytest.fixture(autouse=True)
def synthetic_runtime_authority(monkeypatch):
    _synthetic_runtime(monkeypatch)


def api():
    assert importlib.util.find_spec('scripts.run_extended_cosmology'), 'extended phase driver missing'
    return importlib.import_module('scripts.run_extended_cosmology')


def test_prepare_inventory_is_two_likelihoods_fourteen_treatments(tmp_path):
    a=api(); record=a.prepare(tmp_path)
    assert record['physical_likelihood_count']==2
    assert record['matrix_treatment_count']==14
    assert record['growth_output_count']==8
    assert record['growth_outputs_independent'] is False
    assert {r['case'] for r in record['physical_likelihoods']}=={'B','C'}
    assert record['baseline3_role']=='independent bridge control'


def test_report_requires_external_selection_pin(tmp_path):
    a=api()
    with pytest.raises(ValueError,match='selection'): a.main(['--phase','report','--output-root',str(tmp_path)])


def test_qualify_requires_external_oracle_pin(tmp_path):
    a=api()
    with pytest.raises(ValueError,match='oracle'): a.main(['--phase','qualify','--output-root',str(tmp_path),'--oracle-export',str(tmp_path/'oracle')])


def test_supervisor_records_actual_nonzero_exit_and_preserves_log(tmp_path):
    a=api()
    result=a.observe_process([sys.executable,'-c','print("raw failure"); raise SystemExit(7)'],tmp_path/'child')
    assert result['returncode']==7 and result['status']=='failed'
    assert (tmp_path/'child'/'process.log').read_text().strip()=='raw failure'
    assert result['command'][-1].endswith('SystemExit(7)')


def test_oracle_exporter_rejects_nonfrozen_cwd_before_scientific_import(tmp_path):
    assert importlib.util.find_spec('scripts.export_cosmology_oracle'), 'formal exporter missing'
    a=importlib.import_module('scripts.export_cosmology_oracle')
    with pytest.raises(ValueError,match='frozen'): a.check_environment(tmp_path)


def test_bridge_rejects_same_six_basis_and_changed_derivative(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from test_extended_artifacts import fixture
    from pfsfog import extended_validation as ev
    assert hasattr(ev,'validate_conditional_bridge'), 'independent baseline bridge missing'
    like,results,_,_=fixture(monkeypatch)
    cond=results['conditional3']
    with pytest.raises(ValueError,match='baseline3'):
        ev.validate_conditional_bridge(like,cond,like,cond)
    indices=[like.names.index(n) for n in cond.active_parameter_names]
    base=replace(like,config=replace(like.config,cosmology_basis='baseline3'),
        names=list(cond.active_parameter_names),parameters=[like.parameters[i] for i in indices],
        fiducial=like.fiducial[indices],F_like=cond.F_like,
        bin_F_like=[f[np.ix_(indices,indices)] for f in like.bin_F_like],
        cells=[replace(c,names=list(cond.active_parameter_names),F_like=c.F_like[:3,:3],
                       derivatives=c.derivatives[:,:,:3]) for c in like.cells])
    base_result=SimpleNamespace(**{f:getattr(cond,f) for f in
        ('F_like','F_prior_cosmo','F_prior_nuisance','F_total','covariance','growth_z',
         'growth_fiducial','growth_cov_marg','growth_cov_nuisance_fixed')},
        cov_cosmo_marg=cond.cov_cosmo_marg[:3,:3],
        cov_cosmo_nuisance_fixed=cond.cov_cosmo_nuisance_fixed[:3,:3],
        growth_jacobian=cond.growth_jacobian[:,:3],fisher_solver=cond.fisher_solver)
    assert ev.validate_conditional_bridge(like,cond,base,base_result)['status']=='passed'
    base.cells[0].derivatives=base.cells[0].derivatives.copy()
    base.cells[0].derivatives[0,0,0]+=1e-4
    with pytest.raises(ValueError,match='derivative'):
        ev.validate_conditional_bridge(like,cond,base,base_result)


def test_cohort_rejects_misleading_counts_and_independent_growth_claim():
    from pfsfog import extended_validation as ev
    assert hasattr(ev,'validate_study_inventory'), 'cohort inventory validation missing'
    good={'physical_likelihood_count':2,'matrix_treatment_count':14,
          'growth_output_count':8,'growth_outputs_independent':False}
    ev.validate_study_inventory(good)
    with pytest.raises(ValueError): ev.validate_study_inventory({**good,'physical_likelihood_count':14})
    with pytest.raises(ValueError): ev.validate_study_inventory({**good,'growth_outputs_independent':True})


def test_rejects_failed_quantitative_comparison_and_changed_actual_exit(tmp_path):
    a=api()
    assert hasattr(a,'require_passed'), 'quantitative failed statuses are not enforced'
    with pytest.raises(ValueError,match='nesting'):
        a.require_passed({'status':'failed','reasons':['not PSD']},'nesting')
    r=a.observe_process([sys.executable,'-c','print("done")'],tmp_path/'ok')
    a.validate_exit(r)
    r['returncode']=9
    with pytest.raises(ValueError,match='exit'): a.validate_exit(r)


def test_exporter_control_comparison_rejects_labels_and_values():
    import numpy as np
    from scripts.export_cosmology_oracle import compare_saved_controls
    original={'parameter_names':np.array(['a','b']),'F':np.eye(2)}
    compare_saved_controls(original,original)
    with pytest.raises(ValueError,match='labels'):
        compare_saved_controls({**original,'parameter_names':np.array(['b','a'])},original)
    with pytest.raises(ValueError,match='recovery'):
        compare_saved_controls({**original,'F':np.eye(2)*1.00001},original)


def _install_synthetic_leaves(root,*,intrinsic=False):
    """Child-only physical leaves; real gates, geometry, algebra and supervisors."""
    _synthetic_runtime()
    from pathlib import Path
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    import scripts.run_extended_cosmology as d
    import scripts.run_revision as producer
    import pfsfog.revision_forecast as rf
    import pfsfog.surveys as surveys
    import pfsfog.extended_validation as ev
    import pfsfog.cosmology_projection as projection
    from pfsfog.cosmo import FIDUCIAL
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.revision_validation_physical import expected_support_schedule,declare_fd_manifest,validate_support_schedule
    from pfsfog.revision_validation_ap import ap_cell_record,measure_pure_ap_controls
    from test_extended_validation import _row
    root=Path(root)
    def survey():
        # Independently declared survey geometry: only bin0 contains one tracer.
        bins=np.array(rf.NOMINAL_BINS); density=np.r_[.001,np.zeros(7)]
        def one(name,area,nz):
            return surveys.Survey(name,area,bins[:,0],bins[:,1],nz,np.ones(8)*area,lambda z:1.)
        return surveys.SurveyGroup(one('PFS-ELG',1200,np.zeros(8)),
                                   {'DESI-ELG':one('DESI-ELG',14000,density)})
    surveys.default_survey_group=survey
    def rate(): return lambda z,p: 1/(1+z)
    def sigma(): return lambda z,p:jnp.exp(.1*p['ln10_10_As']+.2*p['mnu']+.3*p['omega_cdm']+.4*p['h']+.5*p['omega_b']+.6*p['n_s'])
    for module in (ev,projection,rf):
        module.make_growth_rate_func=rate; module.make_sigma8_func=sigma
    def build(config,group,cosmo,backend):
        if config.cosmology_basis=='extended6' and config.dk==.0025 and config.n_mu==32:
            with (root/'build-counts.txt').open('a') as log: log.write(config.case+'\n')
            if config.case=='C' and (root/'interrupt-C').exists(): raise RuntimeError('synthetic interruption after completed B')
        space=make_cosmology_spec(FIDUCIAL,extended=config.cosmology_basis=='extended6',ap_nk=config.ap_interpolation_nk)
        support=expected_support_schedule(config,group,space=space)
        names=list(space.names)+[n for r in support for n in r['local_names'] if n not in space.names]
        names=list(dict.fromkeys(names)); grid=rf.make_shell_grid(config.kmin,config.kmax,config.dk)
        values={'Mnu':FIDUCIAL['mnu'],'Omegam':FIDUCIAL['Omega_m'],**FIDUCIAL}
        theta=np.array([values.get(n,1.) for n in names])
        params=[{'name':n,'role':'cosmo' if n in space.names else 'tracer','fiducial':float(v),
                 'unit':'eV' if n=='Mnu' else '',
                 'prior_sigma':{'Mnu':5.,'Omegam':1.}.get(n) if n in space.names else 2.}
                for n,v in zip(names,theta)]
        cells=[]; bins=[np.zeros((len(names),len(names))) for _ in config.nominal_bins]
        for r in support:
            meta={k:v for k,v in r.items() if k not in ('absent_observations','empty_observation')}
            pairs=r['retained_pairs']; meta['theoretical_pairs']=pairs
            meta['retained_observations']=[(p,ell) for p in pairs for ell in (0,2,4)]
            local=r['local_names']; nk=len(grid.centers)
            if local:
                full=list(make_cosmology_spec(FIDUCIAL,extended=True).names)+[n for n in local if n not in space.names]
                frequency=np.array([full.index(n)+1 for n in local])
                phase=(grid.centers-config.kmin)/(config.kmax-config.kmin)
                deriv=np.repeat(np.sin(2*np.pi*phase[:,None]*frequency)[...,None,:],3,axis=1)
                covariance=np.repeat((np.eye(3)*nk)[None],nk,axis=0)
                mean=np.repeat([[10.,20.,30.]],nk,axis=0)
                if intrinsic:
                    from pfsfog.revision_validation_ap import _project
                    nodes,weights=np.polynomial.legendre.leggauss(config.n_mu)
                    mean=np.asarray(_project(jnp.ones((nk,config.n_mu)),jnp.asarray((nodes+1)/2),jnp.asarray(weights/2)))
                from pfsfog.fisher_core import fisher_from_shells
                fisher=fisher_from_shells(deriv,covariance,grid.centers)
            else:
                fisher=np.zeros((0,0)); deriv=covariance=np.zeros((0,0,0)); mean=np.zeros((0,0))
            c=rf.CellLikelihood(fisher,list(local),meta,{'expected_shells':nk if local else 0,'processed_shells':nk if local else 0},deriv,covariance,mean)
            cells.append(c); indices=[names.index(n) for n in local]
            bins[r['nominal_bin_index']][np.ix_(indices,indices)]+=fisher
        return rf.RevisionLikelihood(config,np.sum(bins,axis=0),names,theta,params,cells,bins,grid,dict(FIDUCIAL),{'do_irres':False},{})
    rf.build_revision_likelihood=build
    producer._load_backend=lambda:SimpleNamespace()
    def provenance(backend,*,profile=d.LEGACY_PROFILE,fisher_solver=d.LEGACY_FISHER_SOLVER,shell_profile=d.LEGACY_SHELL_PROFILE):
        from scripts.export_cosmology_oracle import runtime_record
        p={**_real_scientific_inventory(d.ROOT,True),'project_head':'a'*40,
            'runtime':runtime_record(),'runtime_cwd':str(Path.cwd())}
        p['package_versions']=p['runtime']['package_versions'];p['dependency_hash']=d.hash_value(p['package_versions'])
        p['runtime_hash']=d.hash_value(p['runtime'])
        p['driver_source']={'path':str(Path(d.__file__).resolve()),'sha256':d.file_hash(d.__file__)}
        if profile!=d.LEGACY_PROFILE: p['numerical_profile']=d.profile_record(profile)
        p['fisher_solver']=d.fisher_solver_record(fisher_solver)
        p.update(d.shell_profile_fields(shell_profile))
        if shell_profile!=d.LEGACY_SHELL_PROFILE:
            p.update(requested_dk=.00125,original_grid_dk=.0025)
        return p
    d.collect_provenance=provenance
    def measure(like,backend,p,group,path):
        space=make_cosmology_spec(FIDUCIAL,extended=True,ap_nk=like.config.ap_interpolation_nk); nk=len(like.grid.centers)
        support=expected_support_schedule(like.config,group,space=space)
        ids=[c.metadata['cell_id'] for c in like.cells]; pid=d.physical_likelihood_id(like,p)
        declared=declare_fd_manifest(like,ids,pid,expected_support=support,space=space)
        response={'cosmology_columns':list(space.names),'points':[],'cells':[],
            'growth':ev.measure_growth_jacobian(space,FIDUCIAL,[(a+b)/2 for a,b in like.config.nominal_bins])}
        for c in like.cells:
            if not c.names: continue
            rows=[]
            for i,k in enumerate(like.grid.centers):
                import test_extended_validation as row_helpers
                from unittest.mock import patch
                with patch.object(row_helpers,'STEPS',tuple(space.fd_steps.values())):
                    r=_row(space,c.derivatives[i,:,:6])
                r['cell_id']=c.metadata['cell_id'];r['k']=float(k)
                r['pair_covariance']=(np.eye(3)*nk).tolist();r['whitened_physical_scale']=float(np.sqrt(1400/nk))
                for size,item in zip(like.config.ap_refinement_nk,r['interpolation']): item['nk']=size
                for domain in [r['domain'],*r['observed_domains'].values()]:
                    domain.update(k_true_min=float(k),k_true_max=float(k),k_margin_lower=float(k-.002),k_margin_upper=float(.5-k))
                rows.append(r)
            if intrinsic:
                from test_intrinsic_phase import attach_intrinsic_rows
                response.setdefault('intrinsic_geometry',[]).append(attach_intrinsic_rows(like,c,rows,path))
            response['points'].extend(rows);response['cells'].append(ap_cell_record(like,c,rows,space=space))
        groups={}
        for run,c,pair,ell,name,step in declared: groups.setdefault((run,c,name),[]).append(step)
        fd={'status':'passed','failures':[],'scope':'all_cells','selected_cell_ids':ids,'all_cell_ids':ids,'expected_coverage':declared,'measured_coverage':declared,
            'column_results':[{'configuration':run,'cell_id':c,'column':n,'steps':sorted(set(steps),reverse=True),
                'step_errors':[0.,0.,0.],'step_tolerances':[1e-3]*3} for (run,c,n),steps in groups.items()],
            'errors':[{'configuration':run,'cell_id':c,'pair':pair,'ell':ell,'column':n,'step':step,
                       'error':0.,'limit':1e-3,'reason':'relative'} for run,c,pair,ell,n,step in declared]}
        checks=[{'status':'passed','cell_id':c.metadata['cell_id'],'expected_shells':nk if c.names else 0,
            'processed_shells':nk if c.names else 0,'points':1+6*len(c.names) if c.names else 0,
            'min_scaled_eigenvalue':{'field':1.,'retained_covariance':1.},
            'max_scaled_residual':{'field':0.,'retained_covariance':0.}} for c in like.cells]
        radius={'h_ref':space.h_ref,'unit_convention':'fixed_h_ref','sigma8_radius':'8/h_trial Mpc','tolerance':.002,'rows':[]}
        from pfsfog.cosmology_space import cosmology_from_coordinates
        for name,sign,step,theta in ev._radius_points(space,FIDUCIAL):
            mapped=cosmology_from_coordinates(theta,FIDUCIAL,space)
            for z in response['growth']['redshifts']:
                k=np.geomspace(.0001,10,128); rad=float(8*space.h_ref/mapped['h']); x=k*rad
                window=3*(np.sin(x)-x*np.cos(x))/x**3
                network=float(sigma()(z,mapped)); amplitude=network**2/(np.trapezoid(k**3*window**2,x=np.log(k))/(2*np.pi**2))
                radius['rows'].append({'coordinate':name,'sign':sign,'step':step,'theta':theta.tolist(),
                    'redshift':z,'radius_ref':rad,'k_ref':k.tolist(),'power_ref':np.full_like(k,amplitude).tolist(),'network_sigma8':network})
        return {**d.profile_fields(like.config),'response':response,'response_identity':d.identity_provenance(p),'expected_support':support,
                'physical_fd':fd,'covariance_neighborhood':checks,'pure_ap_controls':measure_pure_ap_controls(like.config.ap_interpolation_nk,like.config.ap_interpolation_nmu),'sigma8_radius':radius}
    d.measure_case=measure
    return d,build,survey


def _route_driver_processes(d,bootstrap,monkeypatch=None):
    """Test-only command adapter; keep true modules/inventories and actual exit records.

    Only command/source association is normalized for analytic bootstrap routing.
    The unmodified binding units separately exercise that production boundary.
    """
    _synthetic_runtime(monkeypatch)
    import copy
    from pathlib import Path
    bootstrap=str(Path(bootstrap).resolve());driver=str(Path(d.__file__).resolve())
    observe=d.observe_process;binding=d.validate_execution_binding
    def routed(command,path,**kwargs):
        actual=list(command)
        if actual[2]==driver: actual[2]=bootstrap
        return observe(actual,path,**kwargs)
    def normalize(record):
        out=copy.deepcopy(record)
        if out.get('command',[None]*3)[2]==bootstrap: out['command'][2]=driver
        return out
    def bound(selected,q,role):
        adapted=copy.deepcopy(selected);adapted[role]=normalize(adapted[role])
        adapted[role]['completion']=normalize(adapted[role]['completion'])
        pin=d._pin
        def checked(path,digest,label):
            result=pin(path,digest,label)
            return normalize(result) if label=='execution completion' else result
        d._pin=checked
        try: return binding(adapted,q,role)
        finally: d._pin=pin
    if monkeypatch is None:
        d.observe_process=routed;d.validate_execution_binding=bound
    else:
        monkeypatch.setattr(d,'observe_process',routed);monkeypatch.setattr(d,'validate_execution_binding',bound)


def _complete_synthetic_oracle(out,index):
    """Protocol-complete fixture observations; actual process behavior has separate tests."""
    from pathlib import Path
    from datetime import datetime,timezone
    from scripts import export_cosmology_oracle as e
    d=api();selected=e.ORACLE/'results/revision-qualification-d8dfcf4/certified-baseline-selection.json'
    index.update(schema_version='pfs-desi-cosmology-oracle-v1',status='worker_completed',
        exporter_path=str(Path(e.__file__).resolve()),exporter_sha256=e.digest(e.__file__),
        output=str(out.resolve()),selection=str(selected),selection_sha256=e.SELECTION_PIN,
        command=e.oracle_command(out,selected,'worker'),cwd=str(e.ORACLE))
    d.write_json(out/'worker-index.json',index);pin=d.file_hash(out/'worker-index.json');observed={}
    for role,key,logname in (('worker','execution','export.log'),('verification','verification','verify.log')):
        command=e.oracle_command(out,selected,role,pin);now=datetime.now(timezone.utc).isoformat()
        completion={'schema_version':'pfs-desi-cosmology-oracle-completion-v1','role':role,
            'worker_index_sha256':pin,'output':str(out.resolve()),'cwd':str(e.ORACLE),'command':command,
            'source':{'path':str(Path(e.__file__).resolve()),'sha256':e.digest(e.__file__)}}
        cp=out/(role+'-completion.json');d.write_json(cp,completion)
        log=out/logname;log.write_text('synthetic fixture observation; no scientific qualification\n')
        observed[key]={'command':command,'cwd':str(e.ORACLE),'role':role,'status':'passed','returncode':0,
            'started_at':now,'finished_at':now,'log':str(log.resolve()),'log_sha256':d.file_hash(log),
            'completion':completion,'completion_path':str(cp.resolve()),'completion_sha256':d.file_hash(cp)}
    d.write_json(out/'export-index.json',{'schema_version':'pfs-desi-cosmology-oracle-selection-v1',
        'status':'complete','selection_sha256':e.SELECTION_PIN,'receipt_pins':e.RECEIPTS,
        'worker_index_sha256':pin,**observed})
    return out,d.file_hash(out/'export-index.json')


def _synthetic_oracle(root,build,survey):
    import hashlib
    import numpy as np
    from pathlib import Path
    from pfsfog.revision_forecast import RevisionConfig,apply_revision_priors
    from scripts.export_cosmology_oracle import pack_control,HEAD,SELECTION_PIN,RECEIPTS,ORACLE,runtime_record
    from scripts.run_extended_cosmology import inventory,write_json,file_hash
    out=root/'oracle';out.mkdir();cases={}
    from scripts.run_extended_cosmology import hash_value
    runtime=runtime_record(); runtime['environment']['PYTHONPATH']=f'{ORACLE}:{ORACLE}/ps_1loop_jax'
    provenance={'project_head':HEAD,**_real_scientific_inventory(ORACLE,False),
                'package_versions':runtime['package_versions'],'dependency_hash':hash_value(runtime['package_versions'])}
    for c in inventory()['physical_likelihoods']:
        c={**c,'cosmology_basis':'baseline3'}; like=build(RevisionConfig(**c),survey(),None,None)
        result=apply_revision_priors(like,1.);meta,arrays=pack_control(like,result)
        case=c['case'];sub=out/case;sub.mkdir();write_json(sub/'control.json',meta);np.savez(sub/'controls.npz',**arrays)
        cases[case]={'original_provenance':provenance,'recovered_provenance':provenance,'control':str(sub/'control.json'),'control_sha256':file_hash(sub/'control.json'),
            'arrays':str(sub/'controls.npz'),'arrays_sha256':file_hash(sub/'controls.npz'),
            'receipt_sha256':RECEIPTS[case],'original_selected':{'receipt_sha256':RECEIPTS[case]},
            'array_inventory':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':hashlib.sha256(v.tobytes(order='C')).hexdigest()} for k,v in arrays.items()}}
    return _complete_synthetic_oracle(out,{'source_head':HEAD,'cases':cases,'runtime':runtime,
        'imports':{'pfsfog':{'path':str(ORACLE/'pfsfog/__init__.py'),'sha256':file_hash(ORACLE/'pfsfog/__init__.py')}}})


def test_successful_synthetic_phase_lifecycle_and_corrupt_support(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts import run_extended_cosmology as d
    # Isolate bootstrap mutations in child processes; parent only chooses executable entry.
    tests=Path(__file__).resolve().parent
    bootstrap=tmp_path/'bootstrap.py'
    bootstrap.write_text(f'import sys\nsys.path.insert(0,{str(tests)!r})\nsys.path.insert(0,{str(tests.parent)!r})\n'
        f'from test_run_extended_cosmology import _install_synthetic_leaves, _synthetic_oracle\n'
        f'from pathlib import Path\nroot=Path({str(tmp_path)!r})\n'
        'd,build,survey=_install_synthetic_leaves(root)\nfrom test_run_extended_cosmology import _route_driver_processes\n_route_driver_processes(d,__file__)\n'
        'if sys.argv[1:]==["setup"]:\n'
        '    oracle,pin=_synthetic_oracle(root,build,survey)\n    print(pin)\n'
        'else:\n    raise SystemExit(d.main())\n')
    (tmp_path/'input.txt').write_text('synthetic physical inputs\n')
    setup=subprocess.run([sys.executable,'-B',str(bootstrap),'setup'],cwd=d.ROOT,text=True,capture_output=True)
    assert setup.returncode==0,setup.stderr
    pin=setup.stdout.strip().splitlines()[-1]
    _route_driver_processes(d,bootstrap,monkeypatch)
    out=tmp_path/'study'
    args=['--phase','qualify','--output-root',str(out),'--oracle-export',str(tmp_path/'oracle'),'--oracle-export-sha256',pin]
    (tmp_path/'interrupt-C').write_text('controlled failing physical leaf')
    assert d.main(args)==1
    checkpoint=next(out.glob('qualify-*/checkpoint-B.json'))
    old_execution=d.read_json(checkpoint.parent/'worker'/'execution.json')
    assert old_execution['returncode']==1 and not (checkpoint.parent/'selection.json').exists()
    bad=d.read_json(checkpoint);bad['provenance']['source_content_hash']='e'*64
    bad_path=tmp_path/'wrong-identity.json';d.write_json(bad_path,bad)
    assert d.main(args+['--resume',str(bad_path),'--resume-sha256',d.file_hash(bad_path)])==1
    (tmp_path/'interrupt-C').unlink()
    assert d.main(args+['--resume',str(checkpoint),'--resume-sha256',d.file_hash(checkpoint)])==0
    assert (tmp_path/'build-counts.txt').read_text().splitlines().count('B')==1
    selection=next(out.glob('qualify-*/selection.json'))
    resumed_q=d.read_json(d.read_json(selection)['qualification'])
    assert resumed_q['resumed_from'][0]['execution']['returncode']==1
    assert d.read_json(checkpoint.parent/'worker'/'execution.json')==old_execution
    assert d.main(['--phase','produce','--output-root',str(out),'--selection',str(selection),'--selection-sha256',d.file_hash(selection)])==0
    produced=next(out.glob('produce-*/selection.json'))
    run=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','report','--output-root',str(out),
        '--selection',str(produced),'--selection-sha256',d.file_hash(produced)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    result=d.read_json(next(out.glob('report-*.json')))
    assert result['physical_likelihood_count']==2 and len(result['treatment_comparisons'])==7
    assert len(result['treatment_comparisons'][0]['growth_B_to_C_gain'])==8
    # Corrupt one saved support row and repin bytes: consumer still rejects structure.
    q=d.read_json(d.read_json(produced)['qualification']); artifact=Path(q['cases']['B']['artifact'])
    evidence=d.read_json(artifact/'evidence.json'); evidence['expected_support'].pop()
    d.write_json(artifact/'evidence.json',evidence)
    manifest=d.read_json(artifact/'manifest.json');manifest['files']['evidence.json']=d.file_hash(artifact/'evidence.json')
    manifest['raw_gate_hashes']['expected_support']=d.hash_value(evidence['expected_support'])
    d.write_json(artifact/'manifest.json',manifest)
    q['cases']['B']['manifest_sha256']=d.file_hash(artifact/'manifest.json')
    qpath=Path(d.read_json(produced)['qualification']);d.write_json(qpath,q)
    selected=d.read_json(produced);selected['qualification_sha256']=d.file_hash(qpath);d.write_json(produced,selected)
    run=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','report','--output-root',str(out),
        '--selection',str(produced),'--selection-sha256',d.file_hash(produced)],capture_output=True,text=True)
    assert run.returncode!=0 and any(word in run.stderr for word in ('support','hash','identity'))


def test_sigma8_radius_gate_recomputes_raw_power_and_rejects_fixed_eight():
    from pfsfog import extended_validation as ev
    assert hasattr(ev,'measure_sigma8_radius'), 'raw sigma8 radius qualification missing'
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.cosmo import FIDUCIAL
    from test_extended_validation import Z
    space=make_cosmology_spec(FIDUCIAL,extended=True)
    measured=ev.measure_sigma8_radius(space,FIDUCIAL,Z)
    assert len(measured['rows'])==56
    assert ev.validate_sigma8_radius(measured,space,FIDUCIAL,Z)['status']=='passed'
    measured['rows'][8]['radius_ref']=8.
    with pytest.raises(ValueError,match='radius'):
        ev.validate_sigma8_radius(measured,space,FIDUCIAL,Z)


def test_control_source_binding_and_refined_growth_propagation(monkeypatch,tmp_path):
    from test_extended_artifacts import fixture
    d=api(); like,results,p,_=fixture(monkeypatch)
    import inspect
    assert 'provenance' in inspect.signature(d.save_control).parameters, 'control producer identity missing'
    record=d.save_control(tmp_path/'control',like,results,provenance=p)
    d.load_control(record,provenance=p)
    with pytest.raises(ValueError,match='identity'):
        d.load_control(record,provenance={**p,'source_content_hash':'e'*64})


def test_release_cohort_rejects_improved_errors_when_releasing(monkeypatch):
    import numpy as np
    from test_extended_artifacts import fixture
    from pfsfog import extended_validation as ev
    assert hasattr(ev,'validate_treatment_cohort'), 'release monotonicity gate missing'
    _,results,_,_=fixture(monkeypatch)
    ev.validate_treatment_cohort(results)
    results['release_h'].cov_cosmo_marg=results['release_h'].cov_cosmo_marg*.1
    with pytest.raises(ValueError,match='release'):
        ev.validate_treatment_cohort(results)


def test_resume_requires_external_checkpoint_pin(tmp_path):
    d=api()
    assert hasattr(d,'load_checkpoint'), 'bounded completed-case resume missing'
    with pytest.raises(ValueError,match='resume'):
        d.main(['--phase','qualify','--output-root',str(tmp_path/'new'),'--oracle-export',str(tmp_path/'oracle'),
                '--oracle-export-sha256','a'*64,'--resume',str(tmp_path/'checkpoint.json')])


def test_oracle_input_inventory_digest_is_checked(tmp_path):
    from scripts import export_cosmology_oracle as e
    from scripts.run_extended_cosmology import hash_value
    p={'project_head':e.HEAD,**_real_scientific_inventory(e.ORACLE,False)}
    relocation=e.emulator_locator().approved_relocation_record()
    e.verify_inputs(p,relocation_record=relocation)
    p['source_content_hash']='0'*64
    with pytest.raises(ValueError,match='digest'): e.verify_inputs(p,relocation_record=relocation)


def test_exporter_import_origin_rejects_modified_package(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from scripts import export_cosmology_oracle as e
    fake=tmp_path/'not-frozen.py';fake.write_text('pass')
    monkeypatch.setitem(sys.modules,'pfsfog.fake_oracle_module',SimpleNamespace(__file__=str(fake)))
    with pytest.raises(ValueError,match='source|origin'):
        e.verify_project_origins(api().ROOT,e.scientific_inventory(api().ROOT,extended=True))


def test_oracle_consumer_checks_embedded_source_and_receipt_pins(tmp_path):
    d=api()
    from scripts.export_cosmology_oracle import HEAD,SELECTION_PIN,RECEIPTS
    oracle=tmp_path/'oracle';oracle.mkdir()
    for f in ('export.log','verify.log'): (oracle/f).write_text('completed')
    wrong={**RECEIPTS,'B':'0'*64}
    selected={'status':'complete','selection_sha256':SELECTION_PIN,'receipt_pins':wrong}
    d.write_json(oracle/'export-index.json',selected)
    with pytest.raises(ValueError,match='receipt'):
        d.load_oracle(oracle,d.file_hash(oracle/'export-index.json'))
    selected['receipt_pins']=RECEIPTS
    selected.update(execution={'returncode':0,'log_sha256':d.file_hash(oracle/'export.log')},
                    verification={'returncode':0,'log_sha256':d.file_hash(oracle/'verify.log')})
    from scripts.export_cosmology_oracle import ORACLE,runtime_record
    runtime=runtime_record();runtime['environment']['PYTHONPATH']=f'{ORACLE}:{ORACLE}/ps_1loop_jax'
    worker={'source_head':HEAD,'selection_sha256':SELECTION_PIN,'runtime':runtime,'cwd':str(ORACLE),
        'imports':{'pfsfog':{'path':str(ORACLE/'pfsfog/__init__.py'),'sha256':d.file_hash(ORACLE/'pfsfog/__init__.py')}},
        'cases':{c:{'receipt_sha256':RECEIPTS[c],'original_selected':{'receipt_sha256':RECEIPTS[c]},
                    'original_provenance':{'project_head':'changed'}} for c in ('B','C')}}
    _complete_synthetic_oracle(oracle,worker)
    with pytest.raises(ValueError,match='source'):
        d.load_oracle(oracle,d.file_hash(oracle/'export-index.json'))


def test_report_preserves_successful_other_case_when_diagnostic_fails(monkeypatch,tmp_path):
    d=api()
    successful={'treatment':'marginal6_no_extra','native_sigma':{'h':2.},'growth_sigma':[3.]*8}
    failure={'status':'failed','error':'unconstrained directions [h]'}
    loaded={'B':(None,None,[],{'failed_treatments':{'marginal6_no_extra':failure}}),
            'C':(None,None,[successful],{'failed_treatments':{}})}
    monkeypatch.setattr(d,'read_selection',lambda *a,**k:({'qualification_sha256':'a'*64},
                                                        {'inventory':d.inventory()},loaded))
    report=d.report(tmp_path/'selection','b'*64,tmp_path/'report.json')
    row=next(r for r in report['treatment_comparisons'] if r['treatment']=='marginal6_no_extra')
    assert row['status']=='failed' and row['C']==successful and row['failures']['B']==failure


def test_physical_gate_preserves_explicit_failed_raw_records(monkeypatch):
    import numpy as np
    from test_extended_validation import _physical_fixture,Z
    from pfsfog import revision_validation_physical as p
    from pfsfog.revision_validation_ap import measure_pure_ap_controls
    from pfsfog.extended_validation import measure_sigma8_radius
    d=api();space,like,cell,support=_physical_fixture()
    class Response:
        def multipoles(self,k,theta,qa,qb,qc,*,n_mu):
            return np.full((len(k),3),100+np.dot(theta,[1,2,3,4,5,6]))
        def pkmu(self,k,mu,theta,qa,qb,qc):
            return np.full((len(k),len(mu)),100+np.dot(theta,[1,2,3,4,5,6]))
    monkeypatch.setattr(p,'make_pair_response',lambda *a,**k:Response())
    declared=p.declare_fd_manifest(like,['cell'],'run',expected_support=support,space=space)
    fd=p.assembled_fd_certificate(like,object(),['cell'],declared,expected_support=support,space=space)
    raw={'expected_support':support,'physical_fd':fd,
         'covariance_neighborhood':[p.audit_cell_covariance_neighborhood(like,cell,object(),space=space)],
         'pure_ap_controls':measure_pure_ap_controls(like.config.ap_interpolation_nk,like.config.ap_interpolation_nmu),
         'sigma8_radius':measure_sigma8_radius(space,like.reference_cosmology,Z)}
    d.validate_physical(raw,like,'run')
    fd['status']='failed';fd['failures']=[{'reason':'nonzero inactive block'}]
    with pytest.raises(ValueError,match='physical'):
        d.validate_physical(raw,like,'run')


def test_refinement_preserves_physical_metadata_and_prior_layout(monkeypatch):
    import copy
    from test_extended_artifacts import fixture
    d=api();like,_,_,_=fixture(monkeypatch)
    assert hasattr(d,'validate_refinement_binding'), 'refinement physical binding missing'
    other=copy.deepcopy(like)
    d.validate_refinement_binding(like,other)
    other.cells[0].metadata['volume']=42.
    with pytest.raises(ValueError,match='refinement'):
        d.validate_refinement_binding(like,other)


def _bound_selection_fixture(tmp_path,monkeypatch):
    """Real completed processes; scientific qualification stubbed only in these binding units."""
    import json
    from pathlib import Path
    d=api();script=tmp_path/'worker.py'
    script.write_text('''import argparse, hashlib, json, pathlib, sys
p=argparse.ArgumentParser()
for key in ('phase','output-root','selection','selection-sha256','oracle-export','oracle-export-sha256','resume','resume-sha256'): p.add_argument('--'+key)
p.add_argument('--worker',action='store_true');p.add_argument('--reload',action='store_true')
a=p.parse_args(); root=pathlib.Path(a.output_root)
if a.phase=='qualify': target=root/'qualification.json'; pin=hashlib.sha256(target.read_bytes()).hexdigest()
elif a.reload: target=pathlib.Path(a.selection);pin=a.selection_sha256
else:
 selected=json.loads(pathlib.Path(a.selection).read_text());target=pathlib.Path(selected['qualification']);pin=selected['qualification_sha256']
role='reload' if a.reload else 'worker'
record={'schema_version':'pfs-desi-cosmology-completion-v1','phase':a.phase,'role':role,
 'qualification':str(target.resolve()),'qualification_sha256':pin,
 'command':sys.orig_argv,'source':{'path':str(pathlib.Path(__file__).resolve()),'sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}}
(root/(role+'-completion.json')).write_text(json.dumps(record))
''')
    source={str(script):d.file_hash(script)}
    q={'inventory':d.inventory(),'provenance':{'source_file_hashes':source,'driver_source':{'path':str(script),'sha256':d.file_hash(script)},'runtime':{'interpreter':sys.executable}},
       'oracle_export':str(tmp_path/'oracle'),'oracle_export_sha256':'a'*64}
    monkeypatch.setattr(d,'validate_qualification',lambda path,pin:(d._pin(path,pin,'qualification'),{}))
    def one(name):
        run=tmp_path/name;run.mkdir();qpath=run/'qualification.json';d.write_json(qpath,q)
        command=[sys.executable,'-B',str(script),'--phase','qualify','--output-root',str(run),
                 '--oracle-export',q['oracle_export'],'--oracle-export-sha256',q['oracle_export_sha256'],'--worker']
        worker=d.observe_process(command,run/'worker')
        # Completion attachment is explicit here so this fixture works during RED.
        for record,role in ((worker,'worker'),):
            cp=run/(role+'-completion.json');record.update(completion=d.read_json(cp),completion_path=str(cp),completion_sha256=d.file_hash(cp))
        pin=d.file_hash(qpath)
        command=[sys.executable,'-B',str(script),'--phase','produce','--output-root',str(run),
                 '--selection',str(qpath),'--selection-sha256',pin,'--reload']
        reload=d.observe_process(command,run/'reload');cp=run/'reload-completion.json'
        reload.update(completion=d.read_json(cp),completion_path=str(cp),completion_sha256=d.file_hash(cp))
        selected={'schema_version':'pfs-desi-cosmology-selection-v1','phase':'qualify','status':'complete',
            'run_root':str(run),'worker':worker,'reload':reload,'qualification':str(qpath),'qualification_sha256':pin,
            'upstream_selection':None,'upstream_selection_sha256':None,'resume':None,'resume_sha256':None}
        path=run/'selection.json';d.write_json(path,selected)
        return path,selected
    return d,script,one


@pytest.mark.parametrize('role',['worker','reload'])
def test_selection_rejects_successful_execution_from_other_qualification(tmp_path,monkeypatch,role):
    d,_,one=_bound_selection_fixture(tmp_path,monkeypatch)
    path,a=one('a');_,b=one('b')
    d.read_selection(path,d.file_hash(path),kind='qualify')
    a[role]=b[role];d.write_json(path,a)
    with pytest.raises(ValueError,match='execution|completion|command'):
        d.read_selection(path,d.file_hash(path),kind='qualify')


@pytest.mark.parametrize('changed',['qualification','upstream'])
def test_supervisor_does_not_repin_bytes_changed_after_worker(tmp_path,monkeypatch,changed):
    d,script,one=_bound_selection_fixture(tmp_path,monkeypatch)
    upstream,selected=one('qualified');pin=d.file_hash(upstream)
    monkeypatch.setattr(d,'__file__',str(script))
    observe=d.observe_process
    def observed(command,path,**kwargs):
        result=observe(command,path,**kwargs)
        if '--worker' in command:
            target=upstream if changed=='upstream' else __import__('pathlib').Path(selected['qualification'])
            target.write_bytes(target.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(d,'observe_process',observed)
    out=tmp_path/'produce'
    with pytest.raises(ValueError,match='hash'):
        d.main(['--phase','produce','--output-root',str(out),'--selection',str(upstream),'--selection-sha256',pin])
    assert not list(out.glob('*/selection.json'))

@pytest.mark.parametrize('required',['conditional3','marginal6'])
def test_required_treatment_failure_preserves_raw_likelihood(monkeypatch,tmp_path,required):
    import numpy as np
    from test_extended_artifacts import fixture
    from pfsfog import cosmology_projection as p
    d=api(); like,_,prov,_=fixture(monkeypatch)
    apply=p.apply_cosmology_treatment
    def fail(like,name):
        if name==required: raise ValueError('controlled required inversion failure')
        return apply(like,name)
    monkeypatch.setattr(p,'apply_cosmology_treatment',fail)
    with pytest.raises(ValueError,match='required treatment failed'):
        d.treatments(like,failure_root=tmp_path,provenance=prov)
    record=d.read_json(tmp_path/'failure.json')
    assert record['status']=='failed' and record['consumable'] is False
    assert record['failed_treatments'][required]['error']=='controlled required inversion failure'
    assert d.read_json(tmp_path/'failed-treatments.json')==record['failed_treatments']
    saved,results=d.load_control(record['raw_likelihood'],provenance=prov)
    assert not results
    for a,b in zip(saved.cells,like.cells):
        for field in ('derivatives','covariance','fiducial_mean','F_like'):
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    np.testing.assert_array_equal(saved.F_like,like.F_like)
    assert not list(tmp_path.rglob('selection.json')) and not list(tmp_path.rglob('manifest.json'))
    with pytest.raises((FileNotFoundError,ValueError)):
        d.load_extended_artifacts(tmp_path,manifest_sha256=d.file_hash(tmp_path/'failure.json'))

@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('mutation',['status','failures'])
def test_covariance_gate_rejects_explicit_failure_even_with_good_extrema(empty,mutation):
    from types import SimpleNamespace
    import copy
    d=api(); cell=SimpleNamespace(names=[] if empty else ['x'],metadata={'cell_id':'cell'})
    like=SimpleNamespace(cells=[cell],grid=SimpleNamespace(centers=[.01]))
    row={'status':'passed','cell_id':'cell','expected_shells':0 if empty else 1,
         'processed_shells':0 if empty else 1,'points':0 if empty else 7,
         'min_scaled_eigenvalue':{'field':1.,'retained_covariance':1.},
         'max_scaled_residual':{'field':0.,'retained_covariance':0.}}
    d.validate_covariance_neighborhood([row],like)
    bad=copy.deepcopy(row); bad[mutation]='failed' if mutation=='status' else [{'error':'controlled failure'}]
    with pytest.raises(ValueError,match='covariance'):
        d.validate_covariance_neighborhood([bad],like)
    if not empty:
        row['min_scaled_eigenvalue']['field']=-1.
        with pytest.raises(ValueError,match='covariance'):
            d.validate_covariance_neighborhood([row],like)


def _runtime_fixture(monkeypatch,tmp_path):
    from scripts import export_cosmology_oracle as e
    d=api(); runtime={'interpreter':sys.executable,'prefix':str(e.RECOVERY),'binary_sha256':'a'*64,
        'package_versions':{k:'1' for k in ('numpy','scipy','jax','jaxlib','cosmopower-jax','PyYAML')},
        'environment':{'PYTHONPATH':'.:ps_1loop_jax','OMP_NUM_THREADS':'1'}}
    monkeypatch.setattr(e,'runtime_record',lambda:runtime)
    p={'project_head':'a'*40,'runtime':runtime,'runtime_hash':d.hash_value(runtime),
       'fisher_solver':d.fisher_solver_record(d.LEGACY_FISHER_SOLVER),
       'runtime_cwd':str(d.ROOT),'package_versions':runtime['package_versions'],
       'dependency_hash':d.hash_value(runtime['package_versions']),**_real_scientific_inventory(d.ROOT,True)}
    return d,e,runtime,p


@pytest.mark.parametrize('changed',['interpreter','package_versions','dependency_hash','consistent_wrong_versions'])
def test_consumer_rejects_runtime_mismatch_before_reconstruction(monkeypatch,tmp_path,changed):
    import copy
    d,e,runtime,p=_runtime_fixture(monkeypatch,tmp_path)
    d.verify_provenance(p)
    q={'inventory':d.jsonable(d.inventory()),'cases':{'B':{}},'provenance':copy.deepcopy(p)}
    if changed=='dependency_hash': q['provenance']['dependency_hash']='0'*64
    elif changed=='consistent_wrong_versions':
        q['provenance']['package_versions']['scipy']='different'
        q['provenance']['dependency_hash']=d.hash_value(q['provenance']['package_versions'])
        q['provenance']['runtime']=copy.deepcopy(runtime)
    else:
        active=copy.deepcopy(runtime)
        if changed=='interpreter': active[changed]='/different/python'
        else: active[changed]['scipy']='different'
        monkeypatch.setattr(e,'runtime_record',lambda:active)
    monkeypatch.setattr(d,'load_oracle',lambda *a:pytest.fail('reconstruction boundary reached'))
    with pytest.raises(ValueError,match='runtime|dependency'):
        d._validate_cases(q)


def test_oracle_runtime_allows_only_validated_source_root_difference(monkeypatch,tmp_path):
    import copy
    d,e,active,_=_runtime_fixture(monkeypatch,tmp_path)
    frozen=copy.deepcopy(active);frozen['environment']['PYTHONPATH']=f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax'
    e.verify_runtime(frozen,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,active_root=d.ROOT)
    for bad in (f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax:/unrelated', '/unrelated'):
        changed=copy.deepcopy(frozen);changed['environment']['PYTHONPATH']=bad
        with pytest.raises(ValueError,match='PYTHONPATH'):
            e.verify_runtime(changed,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,active_root=d.ROOT)
    changed=copy.deepcopy(frozen);changed['environment']['OMP_NUM_THREADS']='2'
    with pytest.raises(ValueError,match='runtime'):
        e.verify_runtime(changed,recorded_root=e.ORACLE,recorded_cwd=e.ORACLE,active_root=d.ROOT)

@pytest.mark.parametrize('required',['conditional3','marginal6'])
def test_required_failure_supervisor_retains_failed_exit_without_selection(monkeypatch,tmp_path,required):
    from pathlib import Path
    d=api(); tests=Path(__file__).resolve().parent
    bootstrap=tmp_path/'failure-worker.py'
    bootstrap.write_text(f'import sys\nsys.path[:0]={[str(tests),str(tests.parent)]!r}\n'
        'from pathlib import Path\nimport pytest\nfrom test_extended_artifacts import fixture\n'
        'from scripts import run_extended_cosmology as d\nfrom pfsfog import cosmology_projection as p\n'
        'mp=pytest.MonkeyPatch()\nlike,_,prov,_=fixture(mp)\napply=p.apply_cosmology_treatment\n'
        f'def fail(like,name):\n    if name=={required!r}: raise ValueError("controlled required failure")\n    return apply(like,name)\n'
        'p.apply_cosmology_treatment=fail\n'
        'd.qualify_worker=lambda args:d.treatments(like,failure_root=args.output_root/"B",provenance=prov)\n'
        'raise SystemExit(d.main())\n')
    monkeypatch.setattr(d,'__file__',str(bootstrap))
    assert d.main(['--phase','qualify','--output-root',str(tmp_path/'study'),
                  '--oracle-export',str(tmp_path/'unused-oracle'),'--oracle-export-sha256','a'*64])==1
    run=next((tmp_path/'study').glob('qualify-*'))
    assert d.read_json(run/'worker'/'execution.json')['returncode']==1
    assert d.read_json(run/'B'/'failure.json')['failed_treatments'][required]['status']=='failed'
    assert (run/'B'/'raw-likelihood'/'arrays.npz').exists()
    assert not (run/'selection.json').exists()


def test_missing_completion_preserves_successful_actual_exit(tmp_path):
    d=api(); out=tmp_path/'worker'
    with pytest.raises(FileNotFoundError):
        d.observe_process([sys.executable,'-c','print("actual child success")'],out,
                          completion_path=tmp_path/'missing-completion.json')
    record=d.read_json(out/'execution.json')
    assert record['returncode']==0 and record['status']=='passed'
    assert 'actual child success' in (out/'process.log').read_text()


def test_oracle_recorded_imports_reject_nonfrozen_source(tmp_path):
    from scripts import export_cosmology_oracle as e
    source=tmp_path/'__init__.py';source.write_text('unchanged bytes')
    with pytest.raises(ValueError,match='scientific module origin'):
        e.verify_import_records({'pfsfog':{'path':str(source),'sha256':e.digest(source)}},
                                relocation_record=e.emulator_locator().approved_relocation_record())


def test_production_consumer_rechecks_pinned_upstream_selection(tmp_path,monkeypatch):
    from pathlib import Path
    d,_,one=_bound_selection_fixture(tmp_path,monkeypatch)
    selected,_=one('upstream'); q=d.read_json(d.read_json(selected)['qualification'])
    monkeypatch.setattr(d,'__file__',q['provenance']['driver_source']['path'])
    out=tmp_path/'produced'
    assert d.main(['--phase','produce','--output-root',str(out),'--selection',str(selected),
                   '--selection-sha256',d.file_hash(selected)])==0
    produced=next(out.glob('produce-*/selection.json'));pin=d.file_hash(produced)
    d.read_selection(produced,pin,kind='produce')
    selected.write_bytes(selected.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='selection hash'):
        d.read_selection(produced,pin,kind='produce')


@pytest.mark.parametrize('slot,value',[(2,'/unrelated/driver.py'),(4,'prepare')])
def test_selection_rejects_unrelated_source_or_phase_command(tmp_path,monkeypatch,slot,value):
    d,_,one=_bound_selection_fixture(tmp_path,monkeypatch)
    selected,_=one('run'); record=d.read_json(selected); record['worker']['command'][slot]=value
    d.write_json(selected,record)
    with pytest.raises(ValueError,match='command/completion'):
        d.read_selection(selected,d.file_hash(selected),kind='qualify')


def test_qualification_supervisor_holds_worker_completion_pin(tmp_path,monkeypatch):
    from pathlib import Path
    d,_,one=_bound_selection_fixture(tmp_path,monkeypatch)
    selected,_=one('template'); q=d.read_json(d.read_json(selected)['qualification'])
    monkeypatch.setattr(d,'__file__',q['provenance']['driver_source']['path'])
    observe=d.observe_process
    def mutate(command,path,**kwargs):
        run=Path(command[command.index('--output-root')+1]); target=run/'qualification.json'
        if '--worker' in command: d.write_json(target,q)
        record=observe(command,path,**kwargs)
        if '--worker' in command: target.write_bytes(target.read_bytes()+b'\n')
        return record
    monkeypatch.setattr(d,'observe_process',mutate)
    out=tmp_path/'qualified'
    with pytest.raises(ValueError,match='qualification hash'):
        d.main(['--phase','qualify','--output-root',str(out),'--oracle-export',q['oracle_export'],
                '--oracle-export-sha256',q['oracle_export_sha256']])
    assert not list(out.glob('qualify-*/selection.json'))


def test_historical_versions_remain_distinct_from_recovered_runtime(monkeypatch,tmp_path):
    import copy
    d,e,runtime,p=_runtime_fixture(monkeypatch,tmp_path)
    old=copy.deepcopy(p);old['package_versions']['scipy']='historical'
    old['dependency_hash']=d.hash_value(old['package_versions'])
    e.verify_dependencies(old)  # Historical identity is checked without claiming replay.
    with pytest.raises(ValueError,match='recovered runtime'):
        e.verify_dependencies(old,runtime)


def _observed_oracle_fixture(tmp_path,monkeypatch):
    """Actual isolated subprocesses; replace only expensive export/reconstruction leaves."""
    from pathlib import Path
    from scripts import export_cosmology_oracle as e
    d=api(); selection=tmp_path/'frozen-selection.json'; selection.write_text('{}')
    monkeypatch.setattr(e,'SELECTION_PIN',e.digest(selection))
    monkeypatch.setattr(e,'check_environment',lambda:None)
    script=tmp_path/'export-leaf.py'
    script.write_text('''import argparse, hashlib, json, pathlib, sys
p=argparse.ArgumentParser()
for k in ('selection','selection-sha256','output','index-sha256'): p.add_argument('--'+k)
p.add_argument('--worker',action='store_true');p.add_argument('--verify-only',action='store_true')
a=p.parse_args();out=pathlib.Path(a.output)
hashfile=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source={'path':str(pathlib.Path(__file__).resolve()),'sha256':hashfile(__file__)}
role='worker' if a.worker else 'verification'
if a.worker:
 index={'schema_version':'pfs-desi-cosmology-oracle-v1','status':'worker_completed',
 'source_head':HEAD,'selection':a.selection,'selection_sha256':a.selection_sha256,
 'exporter_path':source['path'],'exporter_sha256':source['sha256'],'output':str(out),
 'runtime':{'interpreter':sys.executable},'cases':{'B':{},'C':{}},
 'command':sys.orig_argv,'cwd':str(pathlib.Path.cwd())}
 (out/'worker-index.json').write_text(json.dumps(index))
 pin=hashfile(out/'worker-index.json')
else:
 pin=a.index_sha256
 assert pin==hashfile(out/'worker-index.json')
record={'schema_version':'pfs-desi-cosmology-oracle-completion-v1','role':role,
 'command':sys.orig_argv,'cwd':str(pathlib.Path.cwd()),'source':source,
 'output':str(out),'worker_index_sha256':pin}
(out/(role+'-completion.json')).write_text(json.dumps(record))
'''.replace('HEAD',repr(e.HEAD)))
    monkeypatch.setattr(e,'__file__',str(script))
    def one(name):
        out=tmp_path/name
        assert e.main(['--selection',str(selection),'--selection-sha256',e.SELECTION_PIN,'--output',str(out)])==0
        return out
    return d,e,one


@pytest.mark.parametrize('mutation',['execution','verification','verifier_pin','exporter','failed_worker','selection_schema','worker_schema'])
def test_oracle_rejects_unbound_successful_observations(tmp_path,monkeypatch,mutation):
    d,e,one=_observed_oracle_fixture(tmp_path,monkeypatch)
    out=one('a');other=one('b')
    class Authenticated(Exception): pass
    monkeypatch.setattr(d,'verify_active_imports',lambda:(_ for _ in ()).throw(Authenticated()))
    with pytest.raises(Authenticated): d.load_oracle(out,d.file_hash(out/'export-index.json'))
    selected=d.read_json(out/'export-index.json');index=d.read_json(out/'worker-index.json')
    if mutation in ('execution','verification'): selected[mutation]=d.read_json(other/'export-index.json')[mutation]
    elif mutation=='verifier_pin': selected['verification']['command'][-1]='0'*64
    elif mutation=='exporter': index['exporter_sha256']='0'*64
    elif mutation=='failed_worker': index['status']='failed'
    elif mutation=='selection_schema': selected.pop('schema_version')
    else: index.pop('schema_version')
    if mutation in ('exporter','failed_worker','worker_schema'):
        d.write_json(out/'worker-index.json',index);selected['worker_index_sha256']=d.file_hash(out/'worker-index.json')
    d.write_json(out/'export-index.json',selected)
    with pytest.raises(ValueError,match='oracle|export|worker|command|schema|completion'):
        d.load_oracle(out,d.file_hash(out/'export-index.json'))


def test_oracle_supervisor_rejects_post_worker_index_changes(tmp_path,monkeypatch):
    d,e,one=_observed_oracle_fixture(tmp_path,monkeypatch)
    run=e.subprocess.run
    def change(command,**kwargs):
        result=run(command,**kwargs)
        if '--worker' in command:
            from pathlib import Path
            p=Path(command[command.index('--output')+1])/'worker-index.json'
            p.write_bytes(p.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(e.subprocess,'run',change)
    with pytest.raises(ValueError,match='index|completion'):
        one('changed')
    assert not (tmp_path/'changed'/'export-index.json').exists()


@pytest.mark.parametrize('role',['worker','verification'])
def test_oracle_observations_record_utc_intervals_including_failure(tmp_path,monkeypatch,role):
    from datetime import datetime,timezone
    d,e,one=_observed_oracle_fixture(tmp_path,monkeypatch)
    out=one('good');selected=d.read_json(out/'export-index.json')
    for field in ('execution','verification'):
        record=selected[field]
        start=datetime.fromisoformat(record['started_at']);end=datetime.fromisoformat(record['finished_at'])
        assert start.tzinfo==timezone.utc and end>=start
    run=e.subprocess.run
    def fail(command,**kwargs):
        if ('--worker' in command)==(role=='worker'):
            return run([sys.executable,'-c','raise SystemExit(7)'],**kwargs)
        return run(command,**kwargs)
    monkeypatch.setattr(e.subprocess,'run',fail)
    target=tmp_path/'failed'
    assert e.main(['--selection',str(tmp_path/'frozen-selection.json'),'--selection-sha256',e.SELECTION_PIN,'--output',str(target)])==7
    record=d.read_json(target/('execution.json' if role=='worker' else 'verify-execution.json'))
    assert record['returncode']==7 and record['status']=='failed'
    assert datetime.fromisoformat(record['finished_at'])>=datetime.fromisoformat(record['started_at'])
    assert not (target/'export-index.json').exists()


def _baseline_persistence_fixture(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from test_extended_artifacts import fixture
    like,results,prov,_=fixture(monkeypatch);cond=results['conditional3']
    ids=[like.names.index(n) for n in cond.active_parameter_names]
    baseline=replace(like,config=replace(like.config,cosmology_basis='baseline3'),
        names=list(cond.active_parameter_names),parameters=[like.parameters[i] for i in ids],
        fiducial=like.fiducial[ids],F_like=cond.F_like,
        bin_F_like=[f[np.ix_(ids,ids)] for f in like.bin_F_like],
        cells=[replace(c,names=list(cond.active_parameter_names),F_like=c.F_like[:3,:3],
                       derivatives=c.derivatives[:,:,:3]) for c in like.cells])
    fields=(*api().PRECISION_FIELDS,*api().GROWTH_FIELDS)
    projected=SimpleNamespace(**{f:getattr(cond,f) for f in fields})
    for key in ('cov_cosmo_marg','cov_cosmo_nuisance_fixed'): setattr(projected,key,getattr(cond,key)[:3,:3])
    projected.growth_jacobian=cond.growth_jacobian[:,:3]
    projected.fisher_solver=cond.fisher_solver
    return like,cond,baseline,projected,prov


@pytest.mark.parametrize('boundary',['prior','baseline3','oracle'])
def test_baseline_failure_preserves_independent_raw_and_completed_result(monkeypatch,tmp_path,boundary):
    import copy
    import numpy as np
    from pfsfog import revision_forecast as f
    d=api();like,cond,base,projected,prov=_baseline_persistence_fixture(monkeypatch)
    oracle=copy.deepcopy(base)
    if boundary=='baseline3': base.cells[0].fiducial_mean=base.cells[0].fiducial_mean+1.
    if boundary=='oracle': oracle.cells[0].fiducial_mean=oracle.cells[0].fiducial_mean+1.
    def solve(*args):
        if boundary=='prior': raise ValueError('controlled baseline prior failure')
        return projected
    monkeypatch.setattr(f,'apply_revision_priors',solve)
    with pytest.raises(ValueError):
        d.check_baseline_control(like,cond,base,(oracle,projected),tmp_path,prov)
    failure=d.read_json(tmp_path/'failure.json')
    assert failure['status']=='failed' and failure['consumable'] is False
    assert failure['stage']=='baseline_'+boundary
    raw,_=d.load_control(failure['inputs']['raw_baseline'],provenance=prov)
    for saved,original in zip(raw.cells,base.cells):
        for field in ('derivatives','covariance','fiducial_mean','F_like'):
            np.testing.assert_array_equal(getattr(saved,field),getattr(original,field))
    np.testing.assert_array_equal(raw.F_like,base.F_like)
    if boundary!='prior':
        saved,results=d.load_control(failure['inputs']['baseline3'],provenance=prov)
        for field in (*d.PRECISION_FIELDS,*d.GROWTH_FIELDS):
            np.testing.assert_array_equal(getattr(results['baseline3'],field),getattr(projected,field))
    assert not list(tmp_path.rglob('selection.json')) and not list(tmp_path.rglob('manifest.json'))


def test_radius_rejection_preserves_exact_inputs_and_point_diagnostics(monkeypatch,tmp_path):
    import numpy as np
    from test_extended_validation import _physical_fixture,Z
    from pfsfog import revision_validation_physical as p,extended_validation as ev
    from pfsfog.cosmology_space import cosmology_from_coordinates
    from pfsfog.revision_validation_ap import measure_pure_ap_controls
    d=api();space,like,cell,support=_physical_fixture()
    class Response:
        def multipoles(self,k,theta,qa,qb,qc,*,n_mu):
            return np.full((len(k),3),100+np.dot(theta,[1,2,3,4,5,6]))
        def pkmu(self,k,mu,theta,qa,qb,qc):
            return np.full((len(k),len(mu)),100+np.dot(theta,[1,2,3,4,5,6]))
    monkeypatch.setattr(p,'make_pair_response',lambda *a,**k:Response())
    prov={k:'a'*64 for k in ('source_content_hash','backend_snapshot','network_hash','survey_input_hash','runtime_hash')};prov['project_head']='a'*40
    like.model={}
    pid=d.physical_likelihood_id(like,prov)
    declared=p.declare_fd_manifest(like,['cell'],pid,expected_support=support,space=space)
    radius={'h_ref':space.h_ref,'unit_convention':'fixed_h_ref','sigma8_radius':'8/h_trial Mpc','tolerance':.002,'rows':[]}
    for name,sign,step,theta in ev._radius_points(space,like.reference_cosmology):
        mapped=cosmology_from_coordinates(theta,like.reference_cosmology,space);rad=8*space.h_ref/mapped['h']
        for z in Z:
            k=np.geomspace(.001,4,64);power=np.ones_like(k);x=k*rad;w=3*(np.sin(x)-x*np.cos(x))/x**3
            network=float(np.sqrt(np.trapezoid(k**3*power*w*w,x=np.log(k))/(2*np.pi**2)))
            radius['rows'].append({'coordinate':name,'sign':sign,'step':step,'theta':theta.tolist(),'redshift':float(z),
                                  'radius_ref':rad,'k_ref':k.tolist(),'power_ref':power.tolist(),'network_sigma8':network})
    radius['rows'][9]['network_sigma8']*=1.1
    raw={'expected_support':support,'physical_fd':p.assembled_fd_certificate(like,object(),['cell'],declared,expected_support=support,space=space),
         'covariance_neighborhood':[p.audit_cell_covariance_neighborhood(like,cell,object(),space=space)],
         'pure_ap_controls':measure_pure_ap_controls(like.config.ap_interpolation_nk,like.config.ap_interpolation_nmu),'sigma8_radius':radius}
    with pytest.raises(ValueError,match='network/integral'):
        d.check_measured_gates(raw,like,prov,tmp_path)
    measured=d.read_json(tmp_path/'gate-measurements.json')
    assert measured['raw']==d.jsonable(raw)
    assert measured['physical_likelihood_id']==pid and measured['consumable'] is False
    failed=d.read_json(tmp_path/'failure.json')
    assert failed['status']=='failed' and failed['consumable'] is False
    assert failed['sigma8_radius_diagnostics'][9]['status']=='failed'
    assert failed['sigma8_radius_diagnostics'][9]['relative_error']==pytest.approx(1/11)
    assert not list(tmp_path.rglob('selection.json'))


def test_covariance_audit_exception_keeps_structured_diagnostics(monkeypatch,tmp_path):
    from test_extended_artifacts import fixture
    from pfsfog import revision_validation_physical as p
    d=api();like,_,prov,_=fixture(monkeypatch)
    monkeypatch.setattr(p,'expected_support_schedule',lambda *a,**k:[])
    monkeypatch.setattr(p,'declare_fd_manifest',lambda *a,**k:[])
    monkeypatch.setattr(p,'assembled_fd_certificate',lambda *a,**k:{'status':'passed','measured':[]})
    diagnostics={'context':'cell=cell0 column=h sign=-1 step=.005 k=.015 mu=0.5',
                 'min_scaled_eigenvalue':-1.,'scaled_condition':float('inf'),'scaled_residual':.2}
    def fail(*a,**k): raise p.CovarianceAuditFailure(diagnostics,'controlled covariance failure')
    monkeypatch.setattr(p,'audit_cell_covariance_neighborhood',fail)
    with pytest.raises(p.CovarianceAuditFailure): d.measure_case(like,object(),prov,object(),tmp_path)
    failed=d.read_json(tmp_path/'failure.json')
    assert failed['status']=='failed' and failed['consumable'] is False
    assert failed['exception']['type']=='CovarianceAuditFailure'
    assert failed['exception']['diagnostics']['context']==diagnostics['context']
    assert failed['exception']['diagnostics']['min_scaled_eigenvalue']==-1.
    assert failed['exception']['diagnostics']['scaled_condition']=={'nonfinite':'inf'}
    assert d.read_json(tmp_path/'gate-measurements.json')['raw']['physical_fd']['status']=='passed'


def test_nonfinite_tags_are_explicit_failed_diagnostics_only(monkeypatch,tmp_path):
    from test_extended_artifacts import fixture
    d=api();like,_,prov,_=fixture(monkeypatch)
    d.save_gate_measurements({'failed_value':float('inf')},like,prov,tmp_path)
    saved=d.read_json(tmp_path/'gate-measurements.json')
    assert saved['status']=='failed' and saved['consumable'] is False
    assert saved['raw']['failed_value']=={'nonfinite':'inf'}
    with pytest.raises((ValueError,FileNotFoundError)):
        d.load_extended_artifacts(tmp_path,manifest_sha256=d.file_hash(tmp_path/'gate-measurements.json'))


@pytest.mark.parametrize('failure',['radius','baseline3','oracle'])
def test_rejected_qualification_preserves_gate_controls_and_actual_exit(tmp_path,monkeypatch,failure):
    from pathlib import Path
    d=api();tests=Path(__file__).resolve().parent;bootstrap=tmp_path/'reject-worker.py'
    bootstrap.write_text(f'import sys\nsys.path[:0]={[str(tests),str(tests.parent)]!r}\n'
        'from pathlib import Path\nfrom test_run_extended_cosmology import _install_synthetic_leaves,_synthetic_oracle\n'
        f'root=Path({str(tmp_path)!r})\nmode={failure!r}\n'
        'd,build,survey=_install_synthetic_leaves(root)\nfrom test_run_extended_cosmology import _route_driver_processes\n_route_driver_processes(d,__file__)\n'
        'from pfsfog import revision_forecast as f\n'
        'def changed_build(config,*args,**kwargs):\n'
        '    like=build(config,*args,**kwargs)\n'
        '    if config.cosmology_basis=="baseline3" and config.case=="B" and ((mode=="oracle")== (sys.argv[1:]==["setup"])) and mode!="radius":\n'
        '        like.cells[0].fiducial_mean=like.cells[0].fiducial_mean+1.\n'
        '    return like\n'
        'f.build_revision_likelihood=changed_build\n'
        'measure=d.measure_case\n'
        'def measured(*args,**kwargs):\n'
        '    raw=measure(*args,**kwargs)\n'
        '    if mode=="radius": raw["sigma8_radius"]["rows"][9]["network_sigma8"]*=1.1\n'
        '    return raw\n'
        'd.measure_case=measured\n'
        'if sys.argv[1:]==["setup"]:\n'
        '    _,pin=_synthetic_oracle(root,changed_build,survey)\n    print(pin)\n'
        'else:\n    raise SystemExit(d.main())\n')
    (tmp_path/'input.txt').write_text('synthetic physical inputs\n')
    setup=subprocess.run([sys.executable,'-B',str(bootstrap),'setup'],cwd=d.ROOT,capture_output=True,text=True)
    assert setup.returncode==0,setup.stderr
    _route_driver_processes(d,bootstrap,monkeypatch);out=tmp_path/'study'
    assert d.main(['--phase','qualify','--output-root',str(out),'--oracle-export',str(tmp_path/'oracle'),
                   '--oracle-export-sha256',setup.stdout.strip().splitlines()[-1]])==1
    run=next(out.glob('qualify-*'));record=d.read_json(run/'B'/'failure.json')
    assert record['status']=='failed' and record['consumable'] is False
    assert record['stage']==('physical_validation' if failure=='radius' else 'baseline_'+failure)
    assert d.read_json(run/'worker'/'execution.json')['returncode']==1
    assert not (run/'selection.json').exists() and not (run/'B'/'artifacts').exists()
    if failure=='radius': assert record['sigma8_radius_diagnostics'][9]['status']=='failed'
    else:
        assert (run/'B'/'baseline3-raw'/'arrays.npz').exists()
        assert (run/'B'/'baseline3'/'arrays.npz').exists()


def _real_scientific_inventory(root,extended):
    """Independent read-only fixture declaration matching the native collector contract."""
    from pathlib import Path
    import json
    d=api();root=Path(root)
    sources=list((root/'pfsfog').glob('*.py'))+[root/'scripts/run_revision.py',root/'scripts/validate_revision.py']
    if extended: sources += [root/'scripts/run_extended_cosmology.py',root/'scripts/export_cosmology_oracle.py']
    sets={'source_file_hashes':sources,
        'backend_file_hashes':[p for p in (root/'ps_1loop_jax').rglob('*') if p.is_file() and p.suffix in ('.py','.txt')],
        'survey_input_hashes':[*list((root/'survey_specs').rglob('*.txt')),root/'configs/default.yaml']}
    result={k:{str(p.resolve()):d.file_hash(p) for p in files} for k,files in sets.items()}
    relocation=json.loads((d.ROOT/'docs/revision/evidence/extended-cosmology-study/ap-global-continuation-2026-10-05/RELOCATION_AUTHENTICATION.json').read_text())
    result['emulator_input_hashes']={r['active_path' if extended else 'historical_path']:d.file_hash(r['active_path'])
        for r in relocation['relocated_paths'] if '/cosmology/' in r['historical_path']}
    if extended:
        from pfsfog.emulator_paths import active_emulator_provenance
        result['active_emulator_provenance']=active_emulator_provenance()
    for group,key in (('source_file_hashes','source_content_hash'),('backend_file_hashes','backend_snapshot'),
                      ('survey_input_hashes','survey_input_hash'),('emulator_input_hashes','network_hash')):
        result[key]=d.hash_value(result[group])
    return result


@pytest.mark.parametrize('mutation',['source_file_hashes','backend_file_hashes','survey_input_hashes','emulator_input_hashes','false_digest','old_checkout'])
def test_complete_inventory_required_before_reconstruction(monkeypatch,tmp_path,mutation):
    from scripts import export_cosmology_oracle as e
    d,_,runtime,p=_runtime_fixture(monkeypatch,tmp_path)
    p.update(_real_scientific_inventory(d.ROOT,True));d.verify_provenance(p)
    if mutation=='old_checkout': p.update(_real_scientific_inventory(e.ORACLE,False))
    elif mutation=='false_digest': p['source_content_hash']='0'*64
    else:
        group=p[mutation];group.pop(next(iter(group)))
        digest={'source_file_hashes':'source_content_hash','backend_file_hashes':'backend_snapshot',
                'survey_input_hashes':'survey_input_hash','emulator_input_hashes':'network_hash'}[mutation]
        p[digest]=d.hash_value(group)
    monkeypatch.setattr(d,'load_oracle',lambda *a:pytest.fail('numerical reconstruction reached'))
    with pytest.raises(ValueError,match='inventory|source|digest'):
        d._validate_cases({'inventory':d.jsonable(d.inventory()),'cases':{'B':{}},'provenance':p})


@pytest.mark.parametrize('module_name',['pfsfog.cosmo','scripts.run_extended_cosmology','scripts.export_cosmology_oracle','ps_1loop_jax.background'])
@pytest.mark.parametrize('field',['file','spec','resolution'])
def test_named_active_origins_reject_other_existing_file_within_root(monkeypatch,tmp_path,field,module_name):
    import importlib.machinery
    from pfsfog import cosmo
    d,_,_,p=_runtime_fixture(monkeypatch,tmp_path);p.update(_real_scientific_inventory(d.ROOT,True))
    from types import SimpleNamespace
    if module_name=='ps_1loop_jax.background':
        original=str(d.ROOT/'ps_1loop_jax/ps_1loop_jax/background.py')
        monkeypatch.setitem(sys.modules,module_name,SimpleNamespace(__file__=original,__spec__=SimpleNamespace(origin=original)))
    module=sys.modules[module_name]
    wrong=str(d.ROOT/('ps_1loop_jax/ps_1loop_jax/ps_1loop.py' if module_name.startswith('ps_1loop') else 'scripts/run_revision.py' if module_name.startswith('scripts') else 'pfsfog/cosmology_space.py'))
    if field=='file': monkeypatch.setattr(module,'__file__',wrong)
    elif field=='spec': monkeypatch.setattr(module.__spec__,'origin',wrong)
    else:
        find=importlib.machinery.PathFinder.find_spec
        def altered(name,path=None,target=None):
            spec=find(name,path,target)
            if name==module_name: spec.origin=wrong
            return spec
        monkeypatch.setattr(importlib.machinery.PathFinder,'find_spec',altered)
    with pytest.raises(ValueError,match='origin|resolution|source'):
        d.verify_provenance(p)


@pytest.mark.parametrize('which',['historical','recovered'])
def test_oracle_requires_complete_native_inventory(monkeypatch,tmp_path,which):
    import copy
    from scripts import export_cosmology_oracle as e
    d=api();out=tmp_path/'oracle';out.mkdir()
    p={'project_head':e.HEAD,**_real_scientific_inventory(e.ORACLE,False)}
    runtime=e.runtime_record();runtime['environment']['PYTHONPATH']=f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax'
    index={'source_head':e.HEAD,'runtime':runtime,
        'imports':{'pfsfog':{'path':str(e.ORACLE/'pfsfog/__init__.py'),'sha256':e.digest(e.ORACLE/'pfsfog/__init__.py')}},
        'cases':{case:{'receipt_sha256':e.RECEIPTS[case],'original_selected':{'receipt_sha256':e.RECEIPTS[case]},
            'original_provenance':copy.deepcopy(p),'recovered_provenance':copy.deepcopy(p)} for case in ('B','C')}}
    class Authenticated(Exception): pass
    monkeypatch.setattr(e,'verify_dependencies',lambda *a:(_ for _ in ()).throw(Authenticated()))
    _,pin=_complete_synthetic_oracle(out,index)
    with pytest.raises(Authenticated): d.load_oracle(out,pin)
    target=index['cases']['B']['original_provenance' if which=='historical' else 'recovered_provenance']
    target['backend_file_hashes'].pop(next(iter(target['backend_file_hashes'])))
    target['backend_snapshot']=d.hash_value(target['backend_file_hashes'])
    _,pin=_complete_synthetic_oracle(out,index)
    with pytest.raises(ValueError,match='inventory'): d.load_oracle(out,pin)


def test_bootstrap_is_not_accepted_as_production_driver(monkeypatch,tmp_path):
    d,_,_,p=_runtime_fixture(monkeypatch,tmp_path)
    bootstrap=tmp_path/'bootstrap.py';bootstrap.write_text('pass')
    monkeypatch.setattr(d,'__file__',str(bootstrap))
    with pytest.raises(ValueError,match='origin'):
        d.verify_provenance(p)


def test_active_emulator_paths_and_recorded_module_entries_are_bound(monkeypatch,tmp_path):
    import copy
    from pfsfog import cosmo
    d,_,_,p=_runtime_fixture(monkeypatch,tmp_path)
    expected={'path':str(d.ROOT/'pfsfog/cosmo.py'),'sha256':d.file_hash(d.ROOT/'pfsfog/cosmo.py')}
    p['executed_module_hashes']={'pfsfog.cosmo':expected};d.verify_provenance(p)
    bad=copy.deepcopy(p);bad['executed_module_hashes']['pfsfog.cosmo']['path']=str(d.ROOT/'pfsfog/cosmology_space.py')
    with pytest.raises(ValueError,match='recorded scientific module'):
        d.verify_provenance(bad)
    monkeypatch.setattr(cosmo,'_JENSE_DIR',tmp_path)
    with pytest.raises(ValueError,match='consumed emulator'):
        d.verify_provenance(p)


def test_loaded_scientific_subset_may_differ(monkeypatch,tmp_path):
    d,_,_,p=_runtime_fixture(monkeypatch,tmp_path)
    p['executed_module_hashes']={};d.verify_provenance(p)
    import pfsfog.reference_units
    d.verify_provenance(p)


def test_fresh_verifier_authenticates_inventory_before_array_reconstruction(monkeypatch,tmp_path):
    import copy
    import numpy as np
    from scripts import export_cosmology_oracle as e
    d=api();out=tmp_path/'oracle';out.mkdir()
    runtime=e.runtime_record();runtime['environment']['PYTHONPATH']=f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax'
    p={'project_head':e.HEAD,**_real_scientific_inventory(e.ORACLE,False),
       'package_versions':runtime['package_versions'],'dependency_hash':d.hash_value(runtime['package_versions'])}
    bad=copy.deepcopy(p);bad['backend_file_hashes'].pop(next(iter(bad['backend_file_hashes'])))
    bad['backend_snapshot']=d.hash_value(bad['backend_file_hashes'])
    meta=out/'control.json';d.write_json(meta,{})
    arrays=out/'controls.npz';np.savez(arrays)
    case={'original_provenance':p,'recovered_provenance':bad,'receipt_sha256':e.RECEIPTS['B'],
          'control':str(meta),'control_sha256':d.file_hash(meta),'arrays':str(arrays),'arrays_sha256':d.file_hash(arrays)}
    index={'source_head':e.HEAD,'runtime':runtime,'cases':{'B':case},'imports':{}}
    _complete_synthetic_oracle(out,index);selected=d.read_json(out/'export-index.json')
    d.write_json(out/'execution.json',selected['execution'])
    monkeypatch.setattr(e,'check_environment',lambda:None)
    monkeypatch.setattr(e,'verify_runtime',lambda *a,**k:None)
    monkeypatch.setattr(e,'verify_import_records',lambda *a:None)
    verify=e.verify_inputs
    monkeypatch.setattr(e,'verify_inputs',lambda p:verify(p,relocation_record=e.emulator_locator().approved_relocation_record()))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('numeric array reconstruction reached'))
    with pytest.raises(ValueError,match='inventory'):
        e.verify_export(out,selected['worker_index_sha256'])


def test_worker_authenticates_raw_manifest_before_receipt_reconstruction(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from scripts import export_cosmology_oracle as e
    d=api();p={'project_head':e.HEAD,**_real_scientific_inventory(e.ORACLE,False)}
    p['survey_input_hashes'].pop(next(iter(p['survey_input_hashes'])))
    p['survey_input_hash']=d.hash_value(p['survey_input_hashes'])
    selected=[]
    for case in ('B','C'):
        root=tmp_path/case;root.mkdir();d.write_json(root/'manifest.json',{'provenance':p})
        selected.append({'configuration':{'case':case},'path':str(root),'receipt_sha256':e.RECEIPTS[case]})
    path=tmp_path/'selection.json';d.write_json(path,{'runs':selected})
    monkeypatch.setattr(e,'SELECTION_PIN',d.file_hash(path));monkeypatch.setattr(e,'check_environment',lambda:None)
    verify=e.verify_inputs
    monkeypatch.setattr(e,'verify_inputs',lambda p:verify(p,relocation_record=e.emulator_locator().approved_relocation_record()))
    monkeypatch.setitem(sys.modules,'artifact_checker',SimpleNamespace(load_revision_artifacts=lambda *a,**k:pytest.fail('receipt numerical reconstruction reached')))
    with pytest.raises(ValueError,match='inventory'):
        e.worker(SimpleNamespace(selection=path,selection_sha256=d.file_hash(path),output=tmp_path/'out'))


def test_frozen_resolution_requires_no_scientific_imports():
    import os
    from scripts import export_cosmology_oracle as e
    code=f'''import importlib.util,sys
spec=importlib.util.spec_from_file_location('inventory_preflight_helper',{e.__file__!r})
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
expected=helper.scientific_inventory(helper.ORACLE,extended=False,input_role='historical',
    relocation_record=helper.emulator_locator().approved_relocation_record())
loaded=helper.verify_project_origins(helper.ORACLE,expected,checker=True)
assert loaded=={{}}
assert not any(n.split('.')[0] in ('pfsfog','artifact_checker','ps_1loop_jax','numpy','jax') for n in sys.modules)
print('Frozen scientific resolution verified without numerical imports')
'''
    historical_python=e.ORACLE.parent/'runtime-recovery-20261001/venv/bin/python'
    result=subprocess.run([str(historical_python),'-B','-c',code],cwd=e.ORACLE,
        env={**os.environ,'PYTHONPATH':f'{e.ORACLE}:{e.ORACLE}/ps_1loop_jax:/Users/nguyenmn/cosmopower-jax'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'without numerical imports' in result.stdout
