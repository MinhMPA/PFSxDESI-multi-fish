"""Accepted DESI/PFS observations and native high-redshift survey geometry."""
from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from pfsfog.cosmology_space import make_cosmology_spec
from pfsfog.eft_params import CROSS_STOCH_NAMES, tracer_fiducials
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.revision_forecast import NOMINAL_BINS, RevisionConfig, ShellGrid, make_shell_grid
from pfsfog.revision_theory import NUISANCE_NAMES
from pfsfog.revision_validation_physical import _support_row
from pfsfog.survey_geometry import ShellGeometry, support_cells
from pfsfog.surveys import default_survey_group

SELECTED_SETTINGS = {'nfft': 2048, 'n_mu': 64, 'dk': .000625}
TAIL_BINS = NOMINAL_BINS + ((2.1, 2.2),)


@dataclass(frozen=True)
class StudyProfile:
    name: str
    case: str
    kmax: float = .20
    nuisance_factor: float = 1.
    exclude_elg1: bool = False
    pfs_zmax: float = 1.6
    pfs_cross_zmax: float = 1.6
    nominal_bins: tuple = NOMINAL_BINS


def paper_profiles():
    profiles = {c: StudyProfile(c, c) for c in ('I','A','B','D','C')}
    for c in ('B','C'):
        profiles[c+'_k015'] = replace(profiles[c], name=c+'_k015', kmax=.15)
        for factor in (3,10):
            name = c+'_prior'+str(factor)
            profiles[name] = replace(profiles[c], name=name, nuisance_factor=float(factor))
        name = c+'_no_elg1'
        profiles[name] = replace(profiles[c], name=name, exclude_elg1=True)
    return profiles


def highz_profiles():
    return {name: StudyProfile(name, case, pfs_zmax=auto,
                              pfs_cross_zmax=cross, nominal_bins=bins)
            for name,case,auto,cross,bins in (
                ('B16','B',1.6,1.6,NOMINAL_BINS),
                ('C16','C',1.6,1.6,NOMINAL_BINS),
                ('H21','C',2.1,1.6,NOMINAL_BINS),
                ('C21','C',2.1,2.1,NOMINAL_BINS),
                ('C22','C',2.2,2.1,TAIL_BINS))}


def likelihood_key(profile):
    return (profile.case, profile.kmax, profile.exclude_elg1,
            profile.pfs_zmax, profile.pfs_cross_zmax, profile.nominal_bins)


@dataclass(frozen=True)
class StudyConfig(RevisionConfig):
    """Same physical configuration, also admitting the approved terminal bin."""
    def __post_init__(self):
        if tuple(map(tuple,self.nominal_bins)) not in (NOMINAL_BINS, TAIL_BINS):
            raise ValueError('study requires the exact eight bins or approved PFS tail')
        base = asdict(self)
        base['nominal_bins'] = NOMINAL_BINS
        RevisionConfig(**base)


def study_config(profile, settings=None, *, extended=False):
    settings = SELECTED_SETTINGS if settings is None else settings
    if set(settings) != {'nfft','n_mu','dk'} or settings['nfft'] < 16:
        raise ValueError('explicit complete ELF settings are required')
    return StudyConfig(case=profile.case, kmin=.01, kmax=profile.kmax,
        dk=settings['dk'], n_mu=settings['n_mu'], nuisance_factor=profile.nuisance_factor,
        exclude_elg1=profile.exclude_elg1, nominal_bins=profile.nominal_bins,
        cosmology_basis='extended6' if extended else 'baseline3')


def study_shell_grid(config):
    """Observed scale cuts restrict the fixed .20 parent grid exactly."""
    parent=make_shell_grid(config.kmin,.20,config.dk)
    if config.kmax==.20:return parent
    if config.kmax!=.15:raise ValueError('unsupported study scale cut')
    end=int(np.argmin(abs(parent.edges-config.kmax)))
    if end<1 or abs(parent.edges[end]-config.kmax)>np.spacing(config.kmax):
        raise ValueError('scale cut is not a boundary of the observed parent grid')
    return ShellGrid(parent.edges[:end+1].copy(),parent.centers[:end].copy(),parent.widths[:end].copy())


def selected_pairs(profile, radial, tracers):
    lo,hi = radial
    if (not np.isfinite([lo,hi]).all() or hi <= lo
            or any(lo < boundary < hi for boundary in (profile.pfs_zmax,profile.pfs_cross_zmax))):
        raise ValueError('invalid interval or unsplit PFS support boundary')
    names = sorted(set(tracers))
    if profile.case in ('I','A','B') or lo >= profile.pfs_zmax:
        names = [n for n in names if n.startswith('DESI-')]
    if lo >= 2.1:
        names = [n for n in names if n == 'PFS-ELG']
    pairs = [(n,n) for n in names]
    if profile.case not in ('I','A'):
        pairs += [(a,b) for i,a in enumerate(names) for b in names[i+1:]
                  if ('PFS-ELG' not in (a,b)
                      or (profile.case != 'D' and hi <= profile.pfs_cross_zmax))]
    return pairs


def study_geometry(profile, group):
    ordinary = ShellGeometry.from_group(group, rtol=1e-6)
    if profile.nominal_bins != TAIL_BINS:
        return ordinary
    if ordinary.edges[-1] != 2.1:
        raise ValueError('terminal study requires common DESI geometry ending at2.1')
    pfs = group.pfs
    native_edges = np.unique(np.concatenate((pfs.z_min_all,pfs.z_max_all)))
    common = np.intersect1d(ordinary.edges,native_edges)
    checked = 0
    for lo,hi in zip(common[:-1],common[1:]):
        rows = (pfs.z_min_all >= lo)&(pfs.z_max_all <= hi)
        a,b = pfs.z_min_all[rows],pfs.z_max_all[rows]
        if not len(a) or a[0]!=lo or b[-1]!=hi or not np.array_equal(a[1:],b[:-1]):
            continue
        native = pfs.Vz_all[rows].sum()/pfs.area_deg2
        if not np.isclose(native,ordinary.volume(lo,hi,1.),rtol=1e-6,atol=0):
            raise ValueError('shared PFS/DESI volume mismatch')
        checked += 1
    rows = (pfs.z_max_all > 2.1)&(pfs.z_min_all < 2.2)
    lo,hi = pfs.z_min_all[rows],pfs.z_max_all[rows]
    if (not checked or not len(lo) or lo[0]!=2.1 or hi[-1]!=2.2
            or not np.array_equal(lo[1:],hi[:-1])):
        raise ValueError('terminal PFS native volume coverage is incomplete')
    return ShellGeometry(np.concatenate((ordinary.edges,hi)),
        np.concatenate((ordinary.volume_per_deg2,pfs.Vz_all[rows]/pfs.area_deg2)))


def study_cells(profile, cosmo, *, settings=None, extended=False):
    config = study_config(profile,settings,extended=extended)
    group = replace(default_survey_group(),pfs_zmax=profile.pfs_zmax)
    geometry = study_geometry(profile,group)
    space = (elf_cosmology_spec(cosmo.params) if extended
             else make_cosmology_spec(cosmo.params,extended=False))
    result = []
    for ibin,nominal in enumerate(profile.nominal_bins):
        baseline = support_cells(group,nominal,geometry=geometry)
        population = sorted({n for c in baseline for n in c.overlap_tracers})
        z = sum(nominal)/2
        b_ref = group.desi_tracers['DESI-ELG'].b1_of_z(z) if 'DESI-ELG' in population else 1.3
        fids = {n:tracer_fiducials(n,group.all_surveys[n].b1_of_z(z),float(cosmo.sigma8(z)),
                                 b1_ref=b_ref,r_sigma_v=config.r_sigma_v) for n in population}
        cells = support_cells(group,nominal,geometry=geometry,exclude_elg1=profile.exclude_elg1)
        for icell,cell in enumerate(cells):
            for region in ('overlap','nonoverlap'):
                row = _support_row(config,cell,ibin,icell,region,space=space)
                pairs = (selected_pairs(profile,cell.radial_bin,row['candidate_populations'])
                         if row['volume'] > 0 else [])
                active = sorted({n for pair in pairs for n in pair})
                names = list(space.names) if active else []
                names += [f'bin{ibin:02d}:{n}:{q}' for n in active for q in NUISANCE_NAMES]
                names += [f'bin{ibin:02d}:{a}|{b}:{q}' for a,b in pairs if a!=b for q in CROSS_STOCH_NAMES]
                row.update(active_populations=active,retained_pairs=pairs,local_names=names,
                           empty_observation=not bool(pairs))
                row['nbar'] = {n:cell.nbar[n] for n in active}
                theoretical = [(a,a) for a in active] + [
                    (a,b) for i,a in enumerate(active) for b in active[i+1:]]
                row['absent_observations'] = [[list(pair),ell]
                    for pair in theoretical if pair not in pairs for ell in (0,2,4)]
                row = json.loads(json.dumps(row))
                result.append({'label':profile.name+':'+row['cell_id'],'cell_id':row['cell_id'],
                    'region':region,'scope':'elf_survey_study_physical','cell':cell,
                    'fiducials':fids,'expected':row})
    return result
