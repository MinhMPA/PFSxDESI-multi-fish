"""Synthetic complete phase exercise; no physical forecast is produced here.

The controlled signal is a constant table and an independently supplied linear
response. Real full-width model/observer parity is separate authenticated
physical evidence. These fixtures exercise storage, references and phase gates.
"""
from pathlib import Path
import copy
import subprocess
import sys

import numpy as np
import pytest
from pfsfog.fisher_precision import LEGACY_FISHER_SOLVER, DECIMAL_FISHER_SOLVER


def attach_intrinsic_rows(like,cell,rows,path):
    import jax.numpy as jnp
    from pfsfog import revision_theory as theory, revision_validation_ap_intrinsic as intrinsic
    from pfsfog.cosmology_space import make_cosmology_spec
    response=object.__new__(theory.PairResponse)
    response.reference=dict(like.reference_cosmology)
    response.space=make_cosmology_spec(response.reference,extended=True,ap_nk=16384)
    response.z=float(cell.metadata['model_redshift']);ref=response.reference
    response.da_ref=theory.bg.angular_diameter_distance(ref['omega_b'],ref['omega_cdm'],ref['h'],response.z,ref['mnu'])
    response.hubble_ref=theory.bg.Hz(ref['omega_b'],ref['omega_cdm'],ref['h'],response.z,ref['mnu'])
    x,y=jnp.geomspace(.002,.5,16384),jnp.linspace(0.,1.,101)
    table=jnp.ones((16384,101));k=jnp.asarray(like.grid.centers)
    packed=np.asarray(like.fiducial)[[like.names.index(n) for n in cell.names]]
    nodes,weights=np.polynomial.legendre.leggauss(like.config.n_mu)
    geometry=intrinsic._table_geometry(response.ap_alphas,jnp.asarray(packed),k,
        jnp.asarray((nodes+1)/2),jnp.asarray(weights/2),x,y,table,None,6)
    partial=cell.derivatives-geometry
    for i,row in enumerate(rows):
        mean=cell.fiducial_mean[i];total=cell.derivatives[i,:,:6];g=geometry[i,:,:6]
        row.update(decomposition='intrinsic-table-v1',fiducial_multipoles=mean.tolist(),
            direct_multipoles=mean.tolist(),intrinsic_ad=partial[i,:,:6].tolist(),
            geometry_ad=g.tolist(),geometry_ad_original=g.tolist())
        chol=np.linalg.cholesky(cell.covariance[i])
        def norm(v): return float(np.linalg.norm(np.linalg.solve(chol,v)))
        scale=norm(mean);row['whitened_physical_scale']=scale
        errors=[]
        for j in range(6):
            delta=total[:,j]-partial[i,:,j]-g[:,j];denominator=norm(total[:,j])
            errors.append(norm(delta)/(scale if denominator<=1e-12*scale else denominator))
        row['chain_errors']=row['chain_errors_original']=errors
        for other in [*row['interpolation'],row['angular']]: other['mean']=mean.tolist()
    arrays={'packed':packed,'k':np.asarray(k),'mean':cell.fiducial_mean.copy(),'total':cell.derivatives.copy(),
        'intrinsic':partial,'geometry_original':geometry,'geometry_matched':geometry,'geometry_forward':geometry,
        'observed_intrinsic':partial.copy(),'observed_intrinsic_table':np.asarray(table),
        'table_matched':np.asarray(table),'table_forward':np.asarray(table),'table_k':np.asarray(x),
        'table_mu':np.asarray(y),'pair_covariance':cell.covariance.copy()}
    pair=cell.metadata['retained_pairs'][0]
    record={'definition':'intrinsic-table-v1','cell_id':cell.metadata['cell_id'],'pair':list(pair),
        'packed_width':17,'n_cosmo':6,'packed_labels':cell.names,
        'packed_roles':list(response.space.names)+['a:'+n for n in theory.NUISANCE_NAMES],
        'n_mu':like.config.n_mu,'ap_nk':16384,'ap_nmu':101,'redshift':response.z,
        'nbar_a':cell.metadata['nbar'][pair[0]],'nbar_b':None,'k_nl':like.config.k_nl,
        'reference_cosmology':ref,'tangent_shape':[17,16384,101],'tangent_sha256':'1'*64,
        'repeated_primal_count':17,'repeated_primals_equal':True,'tangents_finite':True,
        'table_sha256':intrinsic.array_digest(table),'observation':{'outputs_bitwise_equal':True,
            'table_bitwise_equal':True,'table_observations':1,'source_sha256':intrinsic.NATIVE_AP_SOURCE_SHA256,
            'observed_source_sha256':intrinsic.OBSERVED_AP_SOURCE_SHA256,
            'scope':'intrinsic-only; native mean and total unobserved'}}
    i=next(i for i,c in enumerate(like.cells) if c is cell)
    return intrinsic.write_intrinsic_sidecar(path,f'intrinsic/cell{i:03d}-pair000.npz',record,arrays)


@pytest.mark.parametrize('fisher_solver',[LEGACY_FISHER_SOLVER,DECIMAL_FISHER_SOLVER])
def test_new_profile_complete_synthetic_phase_and_reference_rejection(tmp_path,monkeypatch,fisher_solver):
    from scripts import run_extended_cosmology as driver
    from test_run_extended_cosmology import _route_driver_processes
    tests=Path(__file__).resolve().parent
    bootstrap=tmp_path/'bootstrap.py'
    bootstrap.write_text(f'import sys\nsys.path.insert(0,{str(tests)!r})\nsys.path.insert(0,{str(tests.parent)!r})\n'
        'from test_run_extended_cosmology import _install_synthetic_leaves,_synthetic_oracle,_route_driver_processes\n'
        f'from pathlib import Path\nroot=Path({str(tmp_path)!r})\n'
        'd,build,survey=_install_synthetic_leaves(root,intrinsic=True)\n'
        'import pfsfog.cosmology_projection as projection\n'
        'original_projection=projection.apply_cosmology_treatment\n'
        'def optional_failure(like,name,**kwargs):\n'
        ' if name=="marginal6_no_extra": raise ValueError("controlled optional weak-direction failure")\n'
        ' return original_projection(like,name,**kwargs)\n'
        'projection.apply_cosmology_treatment=optional_failure\n'
        'if not (root/"oracle").exists():\n _synthetic_oracle(root,build,survey)\n'
        '_route_driver_processes(d,__file__)\n'
        'if len(sys.argv)>1: raise SystemExit(d.main())\n')
    initial=subprocess.run([sys.executable,'-B',str(bootstrap)],capture_output=True,text=True)
    assert initial.returncode==0,initial.stderr
    _route_driver_processes(driver,bootstrap,monkeypatch)
    export=tmp_path/'oracle';out=tmp_path/'study'
    precision_args=[] if fisher_solver==LEGACY_FISHER_SOLVER else ['--fisher-solver',fisher_solver]
    args=['--phase','qualify','--output-root',str(out),'--numerical-profile',driver.INTRINSIC_PROFILE,
        *precision_args,'--oracle-export',str(export),'--oracle-export-sha256',driver.file_hash(export/'export-index.json')]
    result=driver.main(args)
    assert result==0,'\n'.join(p.read_text() for p in out.glob('qualify-*/worker/process.log'))
    selection=next(out.glob('qualify-*/selection.json'));pin=driver.file_hash(selection)
    q=driver.read_json(driver.read_json(selection)['qualification'])
    assert q['inventory']['fisher_solver']['name']==fisher_solver
    assert set(q['ap_resolution_ratios'])=={'8192_to_16384','16384_to_32768','32768_to_65536','16384_to_65536'}
    assert all(v['status']=='passed' for v in q['ap_resolution_ratios'].values())
    for case in ('B','C'):
        assert set(q['cases'][case]['ap_references']['controls'])=={
            'native3_16384','native3_8192','extended6_8192','extended6_32768','extended6_65536'}
        assert all(row['fisher_solver']['name']==fisher_solver
                   for row in q['cases'][case]['ap_references']['controls'].values())
        if fisher_solver==DECIMAL_FISHER_SOLVER:
            historical=q['cases'][case]['ap_references']['controls']['native3_8192']
            assert historical['reference_checks']['historical_legacy_replay']['status']=='passed'
            assert historical['historical_derived_reference']['receipt']['parent_identity']['oracle_export']['case']==case
    assert driver.main(['--phase','produce','--output-root',str(out),'--numerical-profile',driver.INTRINSIC_PROFILE,
        *precision_args,'--selection',str(selection),'--selection-sha256',pin])==0
    produced=next(out.glob('produce-*/selection.json'))
    done=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','report','--output-root',str(out),
        '--numerical-profile',driver.INTRINSIC_PROFILE,*precision_args,'--selection',str(produced),
        '--selection-sha256',driver.file_hash(produced)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    # An optional failed solution must survive shell/angular saved reload too.
    missing=copy.deepcopy(q)
    missing['cases']['B']['refinements']['shell']['failed_treatments'].pop('marginal6_no_extra')
    bad=tmp_path/'missing-refinement-failure.json';driver.write_json(bad,missing)
    rejected=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','produce','--output-root',str(out),
        '--numerical-profile',driver.INTRINSIC_PROFILE,*precision_args,'--selection',str(bad),
        '--selection-sha256',driver.file_hash(bad),'--reload'],capture_output=True,text=True)
    assert rejected.returncode!=0,'reload accepted an omitted refinement failure'
    assert q['historical_ap_assessment']['required'] is False
    changed=copy.deepcopy(q);changed['historical_ap_assessment']['required']=True
    bad=tmp_path/'changed-historical-assessment.json';driver.write_json(bad,changed)
    rejected=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','produce','--output-root',str(out),
        '--numerical-profile',driver.INTRINSIC_PROFILE,*precision_args,'--selection',str(bad),
        '--selection-sha256',driver.file_hash(bad),'--reload'],capture_output=True,text=True)
    assert rejected.returncode!=0,'reload accepted an inconsistent historical assessment'
    if fisher_solver==DECIMAL_FISHER_SOLVER:
        changed=copy.deepcopy(q)
        changed['cases']['B']['ap_references']['controls']['native3_8192']['historical_derived_reference']['receipt']['parent_identity']['arrays_sha256']='0'*64
        bad=tmp_path/'changed-derived-parent.json';driver.write_json(bad,changed)
        rejected=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','produce','--output-root',str(out),
            '--numerical-profile',driver.INTRINSIC_PROFILE,*precision_args,'--selection',str(bad),
            '--selection-sha256',driver.file_hash(bad),'--reload'],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'derived reference receipt' in rejected.stderr
    # A repinned high-resolution reference may not lose its actual output role.
    q['cases']['B']['ap_references']['controls']['extended6_65536']['evaluated_ap_grids']=[65536,131072]
    corrupt=tmp_path/'corrupt-qualification.json';driver.write_json(corrupt,q)
    check=subprocess.run([sys.executable,'-B',str(bootstrap),'--phase','produce','--output-root',str(out),
        '--numerical-profile',driver.INTRINSIC_PROFILE,*precision_args,'--selection',str(corrupt),
        '--selection-sha256',driver.file_hash(corrupt),'--reload'],capture_output=True,text=True)
    assert check.returncode!=0
