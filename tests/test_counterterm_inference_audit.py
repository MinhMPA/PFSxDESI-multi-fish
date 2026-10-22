"""The supplement must use an independent Schur and response-row QR solve."""
from pathlib import Path

def test_old_control_independently_reconstructs_schur_qr_and_priors():
    from scripts.audit_counterterm_inference import audit_result
    p=Path('/Users/nguyenmn/PFSxDESI-multi-fish/.worktrees/elf-engine/results/takada-density-full-20261014-02/inference/loop2048/profiles/A')
    r=audit_result(p)
    assert r['status']=='passed'
    assert r['checks']['full_schur_scaled_error']<1e-8
    assert r['checks']['qr_covariance_error']<1e-8
    assert r['checks']['assembly_scaled_error']<1e-10
    assert len(r['input_files'])==3

def test_angular_auto_completion_is_ell6_and_rejects_fitted_multipole_leakage():
    import numpy as np
    import pytest
    from scripts.audit_counterterm_inference import angular_auto_completion
    mu,_=np.polynomial.legendre.leggauss(32)
    delta=2*np.polynomial.legendre.Legendre.basis(6)(mu)[None,None,:]
    r=angular_auto_completion(delta,[['X','X']],32)
    assert abs(r['auto_delta_P6_max_abs']-2)<1e-12
    assert r['auto_ell6_reconstruction_max_abs']<1e-12
    with pytest.raises(ValueError,match='ell=6'):
        angular_auto_completion(delta+1e-3,[['X','X']],32)


def test_exact_angular_correction_matches_fresh_elf_at_ap_and_nuisance_displacements():
    import json
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pfsfog.cosmo import FiducialCosmology
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.elf_survey import elf_cosmology_spec
    from pfsfog.revision_theory import make_pair_response,NUISANCE_NAMES
    from scripts.counterterm_comparison import Correction
    from scripts.elf_study_evidence import load_source_cell
    old=Path('/Users/nguyenmn/PFSxDESI-multi-fish/.worktrees/elf-engine/results/elf-survey-studies-20261010-01/inference/loop2048')
    for profile in ('B','C'):
        manifest=json.loads((old/f'profiles/{profile}/manifest.json').read_text())
        refs=[json.loads((old/'projections'/f'{h}.json').read_text()) for h in manifest['projections']]
        ref=next(r for r in refs if r['source_directory'] and len(r['expected']['active_populations'])==2
                 and (profile=='B' or 'PFS-ELG' in r['expected']['active_populations']))
        source=load_source_cell(ref['source_directory'],ref['source_record']);meta=source['inputs']['metadata']
        names=meta['local_names'];ibin=meta['nominal_bin_index'];a=source['arrays'];ix=[0,len(a['k'])//2,len(a['k'])-1]
        engine=ELFPowerSpectrum(nfft=2048);cosmo=FiducialCosmology();space=elf_cosmology_spec(cosmo.params)
        correction=Correction(source,engine,cosmo);mu=a['signed_mu'];responses=[]
        for left,right in meta['theoretical_pairs']:
            cross=left!=right
            response=make_pair_response(engine,meta['model_redshift'],cosmo.params,meta['nbar'][left],
                                        meta['nbar'][right] if cross else None,space=space)
            ci=[names.index(n) for n in space.names]
            qa=[names.index(f'bin{ibin:02d}:{left}:{n}') for n in NUISANCE_NAMES]
            qb=[names.index(f'bin{ibin:02d}:{right}:{n}') for n in NUISANCE_NAMES] if cross else None
            qc=[names.index(f'bin{ibin:02d}:{left}|{right}:{n}') for n in ('Pshot_cross','a2_cross')] if cross else None
            responses.append((response,ci,qa,qb,qc))
        def fresh(point):
            return jnp.stack([r.pkmu(a['k'][ix],mu,point[jnp.array(ci)],point[jnp.array(qa)],
                                    point[jnp.array(qb)] if qb else None,point[jnp.array(qc)] if qc else None)
                              for r,ci,qa,qb,qc in responses])
        call=jax.jit(fresh);delta=jax.jit(lambda point:correction.delta_pkmu(point,mu))
        qname=next(n for n in names if n.endswith(':c2'))
        for p in (0,1+6*names.index('h'),1+6*names.index(qname)):
            point=source['neighborhood_coordinates'][p]
            reconstructed=np.asarray(source['neighborhood_spectra'][p])+np.asarray(delta(point))
            np.testing.assert_allclose(reconstructed[:,ix],np.asarray(call(jnp.asarray(point))),rtol=2e-11,atol=1e-8)
        jax.clear_caches()
