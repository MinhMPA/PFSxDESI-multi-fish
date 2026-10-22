"""New measurements use the accepted high-z physical support and common k."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from pfsfog.cosmo import FIDUCIAL
from scripts.elf_study_physical import common_query_data,measure_highz_cell
from scripts.elf_study_evidence import make_study_request


def test_common_k_evaluation_reconstructs_each_query_without_interpolation(monkeypatch):
    import scripts.elf_study_physical as m
    names=['ln10_10_As','Mnu','Omegam','h','omega_b','n_s']
    from pfsfog.revision_theory import NUISANCE_NAMES
    names += ['bin08:PFS-ELG:'+n for n in NUISANCE_NAMES]
    values={n:1. for n in names}
    meta={'model_redshift':2.15,'nominal_bin_index':8,'nbar':{'PFS-ELG':.001},'volume':1e8,
        'local_names':names,'active_populations':['PFS-ELG'],
        'theoretical_pairs':[['PFS-ELG','PFS-ELG']],'retained_pairs':[['PFS-ELG','PFS-ELG']]}
    class Response:
        def pkmu(self,k,mu,theta,q,*rest): return jnp.broadcast_to(10+jnp.asarray(k)[:,None]**2,(len(k),len(mu)))
        def multipoles(self,k,theta,q,*rest,n_mu):
            return (jnp.asarray(k)[:,None]**2*theta[0]+q[0])*jnp.array([1.,2.,3.])
    monkeypatch.setattr(m,'make_pair_response',lambda *a,**k:Response())
    inputs={'metadata':meta,'fiducials':values,'configuration':{'n_mu':8,'k_nl':.7}}
    coarse=np.array([.015,.025]);fine=np.array([.0125,.0175,.0225,.0275]);query=np.union1d(coarse,fine)
    data=common_query_data(inputs,query,object(),SimpleNamespace(params=FIDUCIAL))
    for k in (coarse,fine):
        take=np.searchsorted(query,k)
        np.testing.assert_array_equal(data['mean'][take],(k[:,None]**2+1)*np.array([1.,2.,3.]))
        np.testing.assert_array_equal(data['D'][take,:,0],k[:,None]**2*np.array([1.,2.,3.]))
    assert data['covariance_kernel'].shape==(6,3,3)


def test_worker_rejects_changed_request_before_measurement(tmp_path):
    r=make_study_request({'nfft':2048,'n_mu':64,'dk':.000625},source={'identity_hash':'foreign'})
    result=measure_highz_cell(r,r['expected'][0]['key'],tmp_path)
    assert result['status']=='failed' and (tmp_path/'failure.log').is_file()
    assert not (tmp_path/'arrays.npz').exists()
