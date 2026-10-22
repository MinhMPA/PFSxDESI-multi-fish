"""Direct common-k checks with explicit angular spectra at corrected densities."""
from copy import deepcopy
import jax.numpy as jnp
import numpy as np
from pfsfog.elf_covariance_audit import covariance_from_spectra
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.revision_theory import NUISANCE_NAMES,make_pair_response
from pfsfog.eft_params import CROSS_STOCH_NAMES
from scripts.elf_study_physical import common_query_data
from scripts.pfs_density_response import density_column_scales,AUTO_STOCH,CROSS_STOCH


def direct_query(inputs,query,engine,cosmo):
    direct=common_query_data(inputs,query,engine,cosmo)
    meta=inputs['metadata'];values=inputs['fiducials'];config=inputs['configuration']
    space=elf_cosmology_spec(cosmo.params);ibin=meta['nominal_bin_index']
    mu=np.polynomial.legendre.leggauss(config['n_mu'])[0];spectra=[]
    for a,b in meta['theoretical_pairs']:
        cross=a!=b;labels=list(space.names)+[f'bin{ibin:02d}:{a}:{n}' for n in NUISANCE_NAMES]
        if cross:labels += [f'bin{ibin:02d}:{b}:{n}' for n in NUISANCE_NAMES]+[f'bin{ibin:02d}:{a}|{b}:{n}' for n in CROSS_STOCH_NAMES]
        v=jnp.array([values[n] for n in labels]);nc=len(space.names);nq=len(NUISANCE_NAMES)
        response=make_pair_response(engine,meta['model_redshift'],cosmo.params,meta['nbar'][a],
            meta['nbar'][b] if cross else None,config['k_nl'],space=space)
        spectra.append(np.asarray(response.pkmu(query,mu,v[:nc],v[nc:nc+nq],
            v[nc+nq:nc+2*nq] if cross else None,v[-2:] if cross else None)))
    direct['pkmu']=np.asarray(spectra)
    _,kernel=covariance_from_spectra(direct['pkmu'],meta,query,np.ones(len(query)),n_mu=config['n_mu'])
    if not np.array_equal(kernel,direct['covariance_kernel']):raise ValueError('direct angular covariance differs')
    return direct


def change_query_density(query,inputs,factor):
    if any(v!=0 for n,v in inputs['fiducials'].items() if n.split(':')[-1] in AUTO_STOCH|CROSS_STOCH):
        raise ValueError('common-k density change requires zero stochastic fiducials')
    meta=deepcopy(inputs['metadata']);scales=density_column_scales(meta['local_names'],factor)
    meta['nbar']['PFS-ELG']*=factor
    _,kernel=covariance_from_spectra(query['pkmu'],meta,query['k'],np.ones(len(query['k'])),
                                    n_mu=inputs['configuration']['n_mu'])
    return {**query,'D':query['D']*scales[None,None,:],'covariance_kernel':kernel}
