"""Pointwise counterterm replacement in authenticated complete ELF spectra.

The loop contribution is unchanged. Both prescriptions are evaluated in the
same physical Q,T coordinates, linear-power interpolation and AP coordinates.
"""
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pfsfog.counterterm_fields import pair_counterterms
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.revision_theory import make_pair_response, nuisance_to_backend, NUISANCE_NAMES
from pfsfog.elf_covariance_audit import covariance_from_spectra
from pfsfog.elf_validation_zeros import counterterm_zero_support
from scripts.pfs_density_response import gamma3_zero_support
from scripts.run_pfs_density_scan import InputBinding,digest
from pfsfog.extended_artifacts import read_json


def old_counterterms(a,b,f):
    """Historical seven coefficients, including the unchanged tied FoG term."""
    ca,cb=a['ctr'],b['ctr'];ba,bb=a['bias']['b1'],b['bias']['b1']
    fog=(ca['cfog']+cb['cfog'])/4
    return jnp.array([(ca[n]+cb[n])/2 for n in ('c0','c2','c4')]+[0.,fog*ba*bb,fog*(ba+bb),fog])


def counterterm_difference(model,k,mu,f,a,b,pk):
    old=old_counterterms(a,b,f)
    new=pair_counterterms(a['ctr'],b['ctr'],a['bias']['b1'],b['bias']['b1'],f)
    delta=new-old
    return model._counterterms.leading(k,mu,f,delta,pk)+model._counterterms.nlo(k,mu,f,delta,pk)


class Correction:
    """Analytic new-minus-old response in a source's explicit coordinate order."""
    def __init__(self,source,engine,cosmo):
        self.source=source;self.engine=engine
        self.inputs=source['inputs'];self.meta=self.inputs['metadata']
        self.names=list(self.meta['local_names']);self.space=elf_cosmology_spec(cosmo.params)
        self.nc=len(self.space.names);self.cidx=[self.names.index(n) for n in self.space.names]
        self.k=jnp.asarray(source['arrays']['k']);self.nmu=self.inputs['configuration']['n_mu']
        self.pairs=list(map(tuple,self.meta['theoretical_pairs']))
        self.retained=[self.pairs.index(tuple(p)) for p in self.meta['retained_pairs']]
        self.response=make_pair_response(engine,self.meta['model_redshift'],cosmo.params,1.,space=self.space)
        ibin=self.meta['nominal_bin_index']
        self.qidx={p:[self.names.index(f'bin{ibin:02d}:{p}:{n}') for n in NUISANCE_NAMES]
                   for p in self.meta['active_populations']}
        nodes,weights=np.polynomial.legendre.leggauss(self.nmu)
        self.mu=jnp.asarray((nodes+1)/2)
        self.weights=jnp.asarray(np.stack([weights/2*(2*l+1)*np.polynomial.legendre.Legendre.basis(l)((nodes+1)/2)
                                          for l in (0,2,4)]))

    def delta_pkmu(self,point,mu,*,ap=True,ap_theta=None):
        point=jnp.asarray(point);theta=point[jnp.array(self.cidx)];r=self.response
        cosmo=r._cosmology(theta);s8=r.sigma8(r.z,cosmo);f=r.growth_rate(r.z,cosmo)
        pars={p:nuisance_to_backend(point[jnp.array(idx)],s8,r.sigma8_ref,f,cosmo['h'],self.meta['nbar'][p],
                                   self.inputs['configuration']['k_nl']) for p,idx in self.qidx.items()}
        alphas=r.ap_alphas(theta if ap_theta is None else ap_theta) if ap else (1.,1.)
        kt,mt,_=self.engine.model._true_coordinates(self.k,jnp.abs(jnp.asarray(mu)),*alphas)
        power=r.plin(r.z,cosmo);pk=self.engine.model._pk_at(kt,jnp.stack((power['k'],power['pk'])))
        return jnp.stack([counterterm_difference(self.engine.model,kt,mt,f,pars[a],pars[b],pk)
                          for a,b in self.pairs])/(alphas[0]**2*alphas[1])

    def delta_mean(self,point,*,ap=True,ap_theta=None):
        angular=self.delta_pkmu(point,self.mu,ap=ap,ap_theta=ap_theta)
        fitted=jnp.einsum('pkm,lm->kpl',angular[jnp.array(self.retained)],self.weights)
        return fitted.reshape((len(self.k),-1))


class CorrectedNeighborhood:
    def __init__(self,source,call):
        self.original=source['neighborhood_spectra'];self.coordinates=source['neighborhood_coordinates']
        self.shape=self.original.shape;self.call=call
    def __getitem__(self,index):
        if not isinstance(index,(int,np.integer)) or not 0<=index<self.shape[0]:raise IndexError(index)
        return np.asarray(self.original[index])+np.asarray(self.call(self.coordinates[index]))


def correct_source(source,engine,cosmo):
    """Update all signed means, AD columns and complete covariance neighborhoods."""
    c=Correction(source,engine,cosmo);old=source['arrays'];a=dict(old);fid=old['fiducial']
    mean=jax.jit(c.delta_mean);intrinsic=jax.jit(lambda x:c.delta_mean(x,ap=False))
    angular=jax.jit(lambda x:c.delta_pkmu(x,old['signed_mu']))
    d=np.asarray(jax.jit(jax.jacfwd(c.delta_mean))(fid))
    di=np.asarray(jax.jit(jax.jacfwd(lambda x:c.delta_mean(x,ap=False)))(fid))
    geometry=jax.jit(lambda x:c.delta_mean(fid,ap_theta=x[jnp.array(c.cidx)]))
    dg=np.asarray(jax.jit(jax.jacfwd(geometry))(fid))
    a['mean']=old['mean']+np.asarray(mean(fid));a['D']=old['D']+d
    for name,delta in [('D_intrinsic',di),('D_geometry',dg)]:
        if name in old:a[name]=old[name]+delta
    for label,offset in [('fd_plus',0),('fd_minus',1)]:
        a[label]=old[label].copy()
        for i in range(len(c.names)):
            for j in range(3):a[label][i,j]+=np.asarray(mean(source['neighborhood_coordinates'][1+6*i+2*j+offset]))
    a['fd_derivatives']=(a['fd_plus']-a['fd_minus'])/(2*a['fd_steps'][:,:,None,None])
    # Keep the six-coordinate intrinsic/AP split witness, including its signed h means.
    if 'h_intrinsic_plus' in old:
        hi=c.names.index('h');fidtheta=jnp.asarray(fid)[jnp.array(c.cidx)]
        geometry=jax.jit(lambda x:c.delta_mean(fid,ap_theta=x[jnp.array(c.cidx)]))
        for component,call in [('intrinsic',intrinsic),('geometry',geometry)]:
            for label,sign in [('plus',1),('minus',-1)]:
                key=f'h_{component}_{label}';a[key]=old[key].copy()
                for j,step in enumerate(old[f'h_{component}_steps'][0]):
                    point=fid.copy();point[hi]+=sign*step;a[key][0,j]+=np.asarray(call(point))
            a[f'h_{component}_derivatives']=(a[f'h_{component}_plus']-a[f'h_{component}_minus'])/(2*a[f'h_{component}_steps'][:,:,None,None])
    a['pkmu']=old['pkmu']+np.asarray(angular(fid))
    _,a['C']=covariance_from_spectra(a['pkmu'],c.meta,a['k'],a['dk'],n_mu=c.nmu)
    a.pop('F_like',None) # Every retained Fisher is reconstructed after data selection.
    inputs=deepcopy(source['inputs']);inputs['engine_settings']=engine.numerical_settings()
    zeros=counterterm_zero_support(c.names,c.meta['retained_pairs'],alphas=inputs['analytic_zero_response']['alphas'],settings=inputs['engine_settings'])
    inputs['analytic_zero_response']['support']=[[n,list(p),ell] for n,p,ell in (*zeros,*gamma3_zero_support(inputs))]
    autos=[3*i+j for i,p in enumerate(c.meta['retained_pairs']) if p[0]==p[1] for j in range(3)]
    diagnostic={'intrinsic_auto_mean_max_abs':float(np.max(abs(np.asarray(intrinsic(fid))[:,autos]))),
        'intrinsic_auto_derivative_max_abs':float(np.max(abs(di[:,autos]))),
        'angular_max_abs':float(np.max(abs(a['pkmu']-old['pkmu']))),
        'covariance_relative_norm':float(np.linalg.norm(a['C']-old['C'])/np.linalg.norm(old['C'])),
        'ap_auto_derivative_max_abs':float(np.max(abs(dg[:,autos])))}
    return {**source,'inputs':inputs,'arrays':a,'neighborhood_spectra':CorrectedNeighborhood(source,angular),
            'counterterm_diagnostic':diagnostic}


def check_response_split(source):
    """Audit the physical intrinsic/AP chain and independent signed h means."""
    from scripts.elf_study_data import check_saved_derivatives
    from pfsfog.revision_forecast import CellLikelihood
    a=source['arrays'];meta=source['inputs']['metadata'];i=meta['local_names'].index('h')
    lower=np.linalg.cholesky(a['C'])
    chain=np.linalg.solve(lower,a['D']-a['D_intrinsic']-a['D_geometry'])
    reference=np.linalg.solve(lower,a['D'])
    error=float(np.linalg.norm(chain)/np.linalg.norm(reference))
    if not np.isfinite(error) or error>=1e-10:raise ValueError('intrinsic/AP chain relative error '+str(error))
    result={'status':'passed','chain_relative_error':error,'chain_tolerance':1e-10}
    for component in ('intrinsic','geometry'):
        cell=CellLikelihood(np.zeros((1,1)),['h'],meta,{},a['D_'+component][...,i:i+1],a['C'],a['mean'])
        arrays={f'fd_{key}':a[f'h_{component}_{key}'] for key in ('plus','minus','steps','derivatives')}
        result['h_'+component]=check_saved_derivatives({'cell':cell,'arrays':arrays,'analytic_zero_support':[]})
    return result


def validate_request(output,track):
    output=Path(output);r=read_json(output/'request.json');s=read_json(output/'state.json')
    if digest(output/'request.json')!=s['request_sha256']:raise ValueError('immutable request changed')
    if r['track']!=track:raise ValueError('mixed counterterm track')
    InputBinding(r['bindings']).verify_all()
    return r


def unique_cell_records(records):
    ids=[r['cell_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate physical cell')
    return {r['cell_id']:r for r in records}
