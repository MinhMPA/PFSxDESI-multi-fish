"""Matched Takada observations and assembly with the existing nuisance priors."""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import json

import numpy as np

from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.eft_params import CROSS_STOCH_NAMES,tracer_fiducials
from pfsfog.fisher_core import fisher_from_shells
from pfsfog.extended_artifacts import require_close
from pfsfog.revision_forecast import NOMINAL_BINS,RevisionConfig
from pfsfog.revision_theory import NUISANCE_NAMES
from pfsfog.revision_validation_physical import _support_row
from pfsfog.survey_geometry import cell_number_density
from scripts.elf_study_profiles import (StudyProfile,TAIL_BINS,SELECTED_SETTINGS,
    paper_profiles,highz_profiles,study_cells,selected_pairs,study_shell_grid)
from scripts.elf_study_inference import validate_study_likelihood
from scripts.takada_density import takada_survey


@dataclass(frozen=True)
class TakadaConfig(RevisionConfig):
    def __post_init__(self):
        if tuple(map(tuple,self.nominal_bins)) not in (
                NOMINAL_BINS,TAIL_BINS,NOMINAL_BINS+((2.1,2.4),)):
            raise ValueError('unsupported corrected nominal bins')
        base=asdict(self);base['nominal_bins']=NOMINAL_BINS
        RevisionConfig(**base)


def corrected_profiles(zmax):
    if zmax not in (2.2,2.4):raise ValueError('coverage must be explicitly selected')
    profiles={**paper_profiles(),**highz_profiles()}
    if zmax==2.4:
        profiles['C24']=replace(profiles['C22'],name='C24',pfs_zmax=2.4,
                                nominal_bins=NOMINAL_BINS+((2.1,2.4),))
    return profiles


def corrected_config(profile,settings=None,*,extended=False):
    settings=SELECTED_SETTINGS if settings is None else settings
    return TakadaConfig(case=profile.case,kmin=.01,kmax=profile.kmax,dk=settings['dk'],
        n_mu=settings['n_mu'],nuisance_factor=profile.nuisance_factor,
        exclude_elg1=profile.exclude_elg1,nominal_bins=profile.nominal_bins,
        cosmology_basis='extended6' if extended else 'baseline3')


def corrected_cells(profile,cosmo,*,pfs_zmin,settings=None,extended=False):
    """Keep the existing DESI radial quadrature and nominal nuisance identities.

    The former 0.7 boundary is retained as an integration subdivision to avoid
    changing the DESI density averaging. It adds no independent nuisance prior.
    """
    if pfs_zmin not in (.6,.7):raise ValueError('PFS lower coverage must be explicit')
    original=profile
    if profile.name=='C24':
        original=replace(profile,name='C22',pfs_zmax=2.2,nominal_bins=TAIL_BINS)
    cases=study_cells(original,cosmo,settings=settings,extended=extended)
    if profile.case in ('I','A','B'):return cases
    survey=takada_survey();config=corrected_config(profile,settings,extended=extended)
    space=elf_cosmology_spec(cosmo.params) if extended else make_cosmology_spec(cosmo.params,extended=False)
    out=[]
    for case in cases:
        cell=case['cell'];row=case['expected'];ibin=row['nominal_bin_index']
        icell=int(row['cell_id'].split('-')[1][6:]);region=case['region']
        nominal=profile.nominal_bins[ibin];radial=cell.radial_bin
        fids=deepcopy(case['fiducials'])
        if profile.name=='C24' and ibin==8:
            radial=nominal
            cell=replace(cell,nominal_bin=nominal,radial_bin=radial,
                model_redshift=sum(nominal)/2,overlap_volume=survey.volume(*nominal),nonoverlap_volume=0.)
            z=cell.model_redshift
            fids={'PFS-ELG':tracer_fiducials('PFS-ELG',survey.b1_of_z(z),float(cosmo.sigma8(z)),
                                          b1_ref=1.3,r_sigma_v=config.r_sigma_v)}
        nbar=dict(cell.nbar);populations=set(cell.overlap_tracers)
        if radial[0]>=pfs_zmin and radial[1]<=profile.pfs_zmax:
            populations.add('PFS-ELG');nbar['PFS-ELG']=cell_number_density(survey,*radial)
        else:
            populations.discard('PFS-ELG');nbar.pop('PFS-ELG',None)
        cell=replace(cell,nbar=nbar,overlap_tracers=tuple(sorted(populations)))
        row=_support_row(config,cell,ibin,icell,region,space=space)
        pairs=selected_pairs(profile,radial,row['candidate_populations']) if row['volume']>0 else []
        active=sorted({n for p in pairs for n in p})
        names=list(space.names) if active else []
        names += [f'bin{ibin:02d}:{n}:{q}' for n in active for q in NUISANCE_NAMES]
        names += [f'bin{ibin:02d}:{a}|{b}:{q}' for a,b in pairs if a!=b for q in CROSS_STOCH_NAMES]
        theoretical=[(n,n) for n in active]+[(a,b) for i,a in enumerate(active) for b in active[i+1:]]
        row.update(active_populations=active,retained_pairs=pairs,local_names=names,
                   empty_observation=not bool(pairs),nbar={n:nbar[n] for n in active},
                   absent_observations=[[list(p),ell] for p in theoretical if p not in pairs for ell in (0,2,4)])
        row=json.loads(json.dumps(row))
        out.append({**case,'label':profile.name+':'+row['cell_id'],'cell':cell,
                    'fiducials':fids,'expected':row})
    return out


def assemble_from_template(template,parts,*,config=None):
    """Sum corrected cells, preserving the declared global coordinates and priors.

    A low-z PFS support extension shares nuisances already present in that bin.
    A longer terminal interval retains the one terminal population. Thus the
    parameter inventory of the corresponding original case is unchanged.
    """
    cells=[p['cell'] for p in parts]
    oldids=[c.metadata['cell_id'] for c in template.cells]
    if [c.metadata['cell_id'] for c in cells]!=oldids or len(set(oldids))!=len(oldids):
        raise ValueError('missing, duplicate or reordered corrected cell')
    config=template.config if config is None else config
    names=template.names;index={n:i for i,n in enumerate(names)}
    bins=[np.zeros_like(template.F_like) for _ in config.nominal_bins];values={}
    for part in parts:
        cell=part['cell'];fid=part['fiducials'];ibin=cell.metadata['nominal_bin_index']
        if not set(cell.names)<=set(names) or set(fid)!=set(cell.names):
            raise ValueError('corrected cell changes parameter inventory')
        if cell.names:
            require_close(cell.F_like,fisher_from_shells(cell.derivatives,cell.covariance,template.grid.centers),
                          'corrected cell Fisher')
        for name,value in fid.items():
            if not np.isfinite(value) or (name in values and values[name]!=value):
                raise ValueError('inconsistent corrected shared fiducial')
            values[name]=value
        ix=[index[n] for n in cell.names];bins[ibin][np.ix_(ix,ix)]+=cell.F_like
    if set(values)!=set(names):raise ValueError('corrected cells omit a global coordinate')
    fisher=np.sum(bins,axis=0);parameters=deepcopy(template.parameters)
    for i,p in enumerate(parameters):
        p['fiducial']=float(values[names[i]])
        p['prior_only']=bool(fisher[i,i]==0 and p['prior_sigma'] is not None)
    nonempty=[c for c in cells if c.names]
    validation={'cell_count':len(cells),'empty_cells':[c.metadata['cell_id'] for c in cells if not c.names],
        'expected_shells':sum(c.validation['expected_shells'] for c in cells),
        'processed_shells':sum(c.validation['processed_shells'] for c in cells),
        'min_field_scaled_eigenvalue':min(c.validation['field_min_scaled_eigenvalue'] for c in nonempty),
        'min_covariance_scaled_eigenvalue':min(c.validation['covariance_min_scaled_eigenvalue'] for c in nonempty)}
    like=replace(template,config=config,F_like=fisher,cells=cells,bin_F_like=bins,
        parameters=parameters,fiducial=np.array([values[n] for n in names]),validation=validation)
    validate_study_likelihood(like)
    return like


def load_corrected_result(path):
    """Read the existing result format, admitting the explicitly declared tail."""
    from pathlib import Path
    from pfsfog.extended_artifacts import read_json,file_hash,PRECISION_FIELDS,GROWTH_FIELDS
    from pfsfog.revision_forecast import CellLikelihood,RevisionLikelihood,ShellGrid,PriorResult
    path=Path(path);record=read_json(path/'manifest.json');meta=record['likelihood']
    if record['schema']!='elf-study-result-v1' or file_hash(path/'arrays.npz')!=record['arrays_sha256']:
        raise ValueError('corrected saved array binding differs')
    config=dict(meta['configuration']);config['nominal_bins']=tuple(map(tuple,config['nominal_bins']))
    config=TakadaConfig(**config)
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
    require_close(result.F_like,a['result/F_like'],'saved corrected likelihood')
    require_close(result.growth_z,np.mean(config.nominal_bins,axis=1),'saved corrected model redshifts')
    return result,record
