"""Takada Table 2 density prescription and exact zero-stochastic transformations.

Historical survey inputs are preserved. The corrected calculation explicitly
uses ``takada_survey``; the old density never supplies its number density.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from pfsfog.elf_covariance_audit import covariance_from_spectra
from pfsfog.fisher_core import fisher_from_shells
from pfsfog.survey_geometry import ShellGeometry
from pfsfog.surveys import default_survey_group
from scripts.pfs_density_response import AUTO_STOCH, CROSS_STOCH, _DensityNeighborhood

ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/'survey_specs/PFS_nz_takada2014_table2.txt'


def takada_survey():
    """Complete 0.6–2.4 Table 2 density with the adopted 1200 deg2 geometry.

    Retain existing native PFS volumes over 0.7–2.2, common DESI geometry below
    0.7, and the existing coarse PFS volume for the previously unused 2.2–2.4
    remainder. Only the volume column of that coarse historical input is used.
    This neither copies Table 2's rounded 1464 deg2 volumes nor introduces a
    new background cosmology when extending the observed radial support.
    """
    bins=np.loadtxt(TABLE)
    if (bins.shape!=(7,3) or not np.isfinite(bins).all() or np.any(bins[:,2]<=0)
            or not np.array_equal(bins[:-1,1],bins[1:,0])):
        raise ValueError('invalid published Takada density table')
    group=default_survey_group();old=group.pfs
    geometry=ShellGeometry.from_group(group)
    edges=geometry.edges[(geometry.edges>=.6)&(geometry.edges<=.7)]
    lo=np.concatenate((edges[:-1],old.z_min_all,[2.2]))
    hi=np.concatenate((edges[1:],old.z_max_all,[2.4]))
    coarse=np.loadtxt(ROOT/'survey_specs/PFS_nz_pfs_fdr.txt')
    terminal=coarse[(coarse[:,0]==2.)&(coarse[:,1]==2.4),4]
    if terminal.shape!=(1,):raise ValueError('missing original full PFS terminal volume')
    volume=np.concatenate((
        [geometry.volume(a,b,1200.) for a,b in zip(edges[:-1],edges[1:])],
        old.Vz_all,[terminal[0]-old.volume(2.,2.2)]))
    density=[]
    for a,b in zip(lo,hi):
        row=bins[(bins[:,0]<=a)&(bins[:,1]>=b)]
        if len(row)!=1:raise ValueError('native row crosses a Table 2 density boundary')
        density.append(row[0,2])
    if np.any(volume<=0):raise ValueError('nonpositive Takada survey volume')
    return replace(old,z_min_all=lo,z_max_all=hi,nz_all=np.asarray(density),Vz_all=volume)


def retarget_source(source, target):
    """Change density/observed volume at identical clustering and model redshift.

    A source may supply another radial cell only within the same nominal model
    bin and with exactly the same populations, spectrum and parameter inventory.
    The mean, k grid and physical cosmology are unchanged. Every signed stochastic
    displacement and the complete theoretical Gaussian covariance are rebuilt.
    """
    original=source['inputs']['metadata']
    for key in ('model_redshift','nominal_bin','nominal_bin_index','region',
                'active_populations','local_names','retained_pairs','theoretical_pairs'):
        if target.get(key)!=original.get(key):
            raise ValueError('density reuse changes physical support: '+key)
    if (not np.isfinite(target['volume']) or target['volume']<=0
            or set(target['nbar'])!=set(original['nbar'])
            or any(not np.isfinite(n) or n<=0 for n in target['nbar'].values())):
        raise ValueError('invalid target number density or volume')
    if any(v!=0. for n,v in source['inputs']['fiducials'].items()
           if n.split(':')[-1] in AUTO_STOCH|CROSS_STOCH):
        raise ValueError('density reuse requires zero stochastic fiducials')
    inputs=deepcopy(source['inputs']);inputs['metadata']=deepcopy(target)
    names=original['local_names'];ratios={n:original['nbar'][n]/target['nbar'][n]
                                        for n in original['nbar']}
    scales=np.ones(len(names))
    for i,name in enumerate(names):
        fields=name.split(':')
        if len(fields)!=3:continue
        population,coordinate=fields[1:]
        if coordinate in AUTO_STOCH:scales[i]=ratios[population]
        elif coordinate in CROSS_STOCH:
            a,b=population.split('|');scales[i]=np.sqrt(ratios[a]*ratios[b])
    old=source['arrays']
    if not np.array_equal(old['fiducial'],[inputs['fiducials'][n] for n in names]):
        raise ValueError('source physical fiducial layout differs')
    arrays=dict(old)
    for key in ('D','D_intrinsic','D_geometry'):
        if key in old:arrays[key]=old[key]*scales[None,None,:]
    for key in ('fd_plus','fd_minus'):
        arrays[key]=old[key].copy()
        for i in np.flatnonzero(scales!=1.):
            arrays[key][i]=old['mean']+scales[i]*(old[key][i]-old['mean'])
    arrays['fd_derivatives']=(arrays['fd_plus']-arrays['fd_minus'])/(2*old['fd_steps'][:,:,None,None])
    _,arrays['C']=covariance_from_spectra(old['pkmu'],target,old['k'],old['dk'],
                                         n_mu=inputs['configuration']['n_mu'])
    if 'F_like' in arrays:arrays['F_like']=fisher_from_shells(arrays['D'],arrays['C'],old['k'])
    return {**source,'inputs':inputs,'arrays':arrays,
        'neighborhood_spectra':_DensityNeighborhood(source['neighborhood_spectra'],scales),
        'density_transform':{'source_metadata':deepcopy(original),'target_metadata':deepcopy(target),
            'column_scales':scales.tolist(),'zero_stochastic_fiducials':True}}


def preserve_unchanged_observation(source, anchor):
    """Retain an existing observation exactly when adding a tracer to its cell.

    Density transfer from the adjacent radial subcell changes the last bits of
    stochastic multipole projection. Verify the same physical observation to
    roundoff, then use its original responses. Covariance and mean must already
    be bitwise identical. The signed finite differences remain independent.
    """
    a=anchor['inputs']['metadata'];b=source['inputs']['metadata']
    for key in ('cell_id','nominal_bin','radial_bin','model_redshift','volume','region'):
        if a[key]!=b[key]:raise ValueError('unchanged observation geometry differs: '+key)
    if any(b['nbar'].get(n)!=v for n,v in a['nbar'].items()):
        raise ValueError('unchanged observation density differs')
    an=a['local_names'];bn=b['local_names']
    if not set(an)<=set(bn) or any(anchor['inputs']['fiducials'][n]!=source['inputs']['fiducials'][n] for n in an):
        raise ValueError('unchanged observation coordinates differ')
    cols=[bn.index(n) for n in an]
    pairs=list(map(tuple,b['retained_pairs']))
    rows=[3*pairs.index(tuple(p))+ell for p in a['retained_pairs'] for ell in range(3)]
    aa=anchor['arrays'];bb=source['arrays']
    for key in ('k','dk'):
        if not np.array_equal(aa[key],bb[key]):raise ValueError('unchanged observation shell grid differs')
    if (not np.array_equal(aa['mean'],bb['mean'][:,rows])
            or not np.array_equal(aa['C'],bb['C'][:,rows][:,:,rows])):
        raise ValueError('unchanged observation mean or covariance differs')
    changed=dict(bb);reports={}
    for key in ('D','D_intrinsic','D_geometry'):
        if key not in aa or key not in bb:continue
        shared=bb[key][:,rows][:,:,cols];reference=aa[key]
        scale=np.maximum(1.,np.max(np.abs(reference),axis=(0,1)))
        error=np.max(np.abs(shared-reference),axis=(0,1))
        limit=64*np.finfo(float).eps*scale
        if not np.isfinite(shared).all() or np.any(error>limit):
            raise ValueError('unchanged observation response differs physically: '+key)
        changed[key]=bb[key].copy()
        for i,row in enumerate(rows):changed[key][:,row,cols]=reference[:,i,:]
        reports[key]={'max_absolute_difference':float(error.max()),
                      'max_column_relative_difference':float(np.max(error/scale)),
                      'roundoff_limit_in_eps':64}
    if 'F_like' in changed:
        changed['F_like']=fisher_from_shells(changed['D'],changed['C'],changed['k'])
    return {**source,'arrays':changed,
            'source_files':{**source['source_files'],**anchor['source_files']},
            'unchanged_observation':{'status':'passed','cell_id':a['cell_id'],'responses':reports,
                                     'source_files':anchor['source_files']}}
