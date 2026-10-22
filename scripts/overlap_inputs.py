"""Authenticated DESI+PFS references and residual cross-stochastic conditions.

Loading a reference authenticates the archived calculation. It does not qualify
any new choice of measured spectra, area weights or fixed nuisance parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from pfsfog.cosmo import FIDUCIAL
from pfsfog.extended_artifacts import canonical
from pfsfog.revision_forecast import PriorResult
from scripts.elf_study_profiles import study_shell_grid
from scripts.elf_study_report import comparisons, constraint_row
from scripts.run_elf_studies import GRID_PROFILES
from scripts.takada_study import corrected_cells, corrected_config, corrected_profiles, load_corrected_result

PublicCase = Literal['A', 'B', 'C', 'D']
Treatment = Literal['marginalized', 'desi_fixed', 'pfs_fixed', 'all_cross_fixed']
TREATMENTS = ('marginalized', 'desi_fixed', 'pfs_fixed', 'all_cross_fixed')
ARCHIVED_PROFILES = ('A', 'B', 'D', 'C', 'C21', 'H21')
CURRENT_CASES = {'A': 'A', 'B': 'B', 'D': 'C', 'C': 'D', 'C21': 'D', 'H21': None}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS = PROJECT_ROOT / 'notebooks/data/author_revision/case_labels.json'
G4 = PROJECT_ROOT / '.worktrees/forecast-discrepancy/results/forecast-discrepancy-20261015-01/G4/measurement-02/measurement.json'
G4_SHA256 = '7ab1a2d0b16721c822569fd49f4b2371c58780e2cb2292aefdda9bdcf58e1c45'


class EvidenceError(ValueError):
    """Required physical evidence is absent, altered or unqualified."""


class ArrayHashError(EvidenceError):
    """The saved numerical arrays differ from their recorded identity."""


class MissingGridError(EvidenceError):
    """The requested numerical refinement is absent."""


class CellInventoryError(EvidenceError):
    """A physical integration cell is missing, repeated or reordered."""


class PhysicalReferenceError(EvidenceError):
    """Density, bins, cosmology, units or model differ from the reference."""


class CosmologyOrderError(PhysicalReferenceError):
    """The conditional cosmological coordinates are not the agreed three."""


class GrowthPivotError(PhysicalReferenceError):
    """The headline growth quantity has no unique z=1.1 coordinate."""


class StochasticParameterError(PhysicalReferenceError):
    """Residual cross-stochastic parameters differ from the physical pairs."""


@dataclass(frozen=True)
class Condition:
    case: PublicCase
    area_deg2: int
    stochastic: Treatment

    def __post_init__(self):
        if (self.case not in ('A', 'B', 'C', 'D') or type(self.area_deg2) is not int
                or self.area_deg2 not in (1200, 14000) or self.stochastic not in TREATMENTS):
            raise ValueError('undeclared overlap condition')


def conditions() -> tuple[Condition, ...]:
    return tuple(Condition(c, a, s) for c in ('A', 'B', 'C', 'D')
                 for a in (1200, 14000) for s in TREATMENTS)


def canonical_condition(c: Condition) -> Condition:
    if c.case == 'A':
        return Condition('A', 1200, 'marginalized')
    if c.case in ('B', 'C'):
        desi = c.stochastic in ('desi_fixed', 'all_cross_fixed')
        return Condition(c.case, 1200 if c.case == 'B' else c.area_deg2,
                         'desi_fixed' if desi else 'marginalized')
    return c


@dataclass
class StudyInputs:
    grid: str
    source_root: Path
    results: dict[str, PriorResult]
    metadata: dict[str, dict]
    hashes: dict[str, str]


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError(f'missing or unreadable required evidence: {path}') from error


def _digest(path):
    try:
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except OSError as error:
        raise EvidenceError(f'missing required evidence: {path}') from error


class _Bindings:
    """One per-load inventory; every declared byte is hashed before returning."""
    def __init__(self, source_root, archived_root):
        self.source_root, self.archived_root = source_root, archived_root
        self.expected = {}
        self.hashes = {}

    def path(self, path):
        path = Path(path)
        if path.is_relative_to(self.archived_root):
            return self.source_root / path.relative_to(self.archived_root)
        return path

    def add(self, records):
        if not records:
            raise EvidenceError('missing required binding inventory')
        for path, digest in records.items():
            path = str(self.path(path))
            if path in self.expected and self.expected[path] != digest:
                raise EvidenceError('conflicting required evidence hashes: ' + path)
            self.expected[path] = digest

    def check(self, path, digest=None):
        path = str(self.path(path))
        expected = self.expected.get(path) if digest is None else digest
        if expected is None:
            raise EvidenceError('unbound required evidence: ' + path)
        actual = _digest(path)
        if actual != expected:
            raise EvidenceError('altered required evidence: ' + path)
        self.hashes[path] = actual

    def verify_all(self):
        for path in self.expected:
            if path not in self.hashes:
                self.check(path)


def _source_bindings(source, binding):
    if 'identity_hash' in source:
        digest = hashlib.sha256(canonical(source['details'])).hexdigest()
        if digest != source['identity_hash']:
            raise EvidenceError('altered physical source identity')
    for value in source.values():
        if isinstance(value, dict):
            if value and all(Path(p).is_absolute() and isinstance(h, str) and len(h) == 64
                             for p, h in value.items()):
                binding.add(value)
            else:
                _source_bindings(value, binding)


def _cross_families(parameters, cells):
    expected = {(c.metadata['nominal_bin_index'], tuple(pair), coordinate)
                for c in cells for pair in c.metadata['retained_pairs'] if pair[0] != pair[1]
                for coordinate in ('Pshot_cross', 'a2_cross')}
    found = set()
    families = {'desi': [], 'pfs': []}
    for p in parameters:
        cross_coordinate = p['coordinate'] in ('Pshot_cross', 'a2_cross')
        if (p['role'] == 'cross') != cross_coordinate:
            raise StochasticParameterError('cross-stochastic role or coordinate differs')
        if not cross_coordinate:
            continue
        pair = tuple(p['tracers'])
        key = (p['nominal_bin'], pair, p['coordinate'])
        if (key not in expected or key in found or len(pair) != 2 or pair[0] == pair[1]
                or p['fiducial'] != 0. or p['prior_sigma'] != 1. or p['unit'] != ''):
            raise StochasticParameterError('cross-stochastic pair, fiducial, prior or unit differs')
        found.add(key)
        if all(n.startswith('DESI-') for n in pair):
            family = 'desi'
        elif pair.count('PFS-ELG') == 1 and sum(n.startswith('DESI-') for n in pair) == 1:
            family = 'pfs'
        else:
            raise StochasticParameterError('unknown cross-stochastic family')
        families[family].append(p['name'])
    if found != expected:
        raise StochasticParameterError('missing cross-stochastic parameters for retained pairs')
    return families


def _validate_reference(result, manifest, profile, settings, source):
    like = result.likelihood
    if like.names[:3] != ['ln10_10_As', 'Mnu', 'Omegam']:
        raise CosmologyOrderError('conditional cosmology order differs')
    if np.count_nonzero(np.isclose(result.growth_z, 1.1, rtol=0, atol=1e-12)) != 1:
        raise GrowthPivotError('missing unique headline growth pivot')
    config = corrected_config(profile, settings)
    if (canonical(manifest['profile']) != canonical(asdict(profile)) or manifest['settings'] != settings
            or canonical(asdict(like.config)) != canonical(asdict(config))):
        raise PhysicalReferenceError('physical profile, bins or scale cuts differ')
    if (like.reference_cosmology != FIDUCIAL or like.reference_cosmology != source['reference_cosmology']
            or like.model != source['engine_settings'] or like.model['engine'] != 'elf.EPT'):
        raise PhysicalReferenceError('physical cosmology or ELF model differs')
    expected = [c['expected'] for c in corrected_cells(profile,
                SimpleNamespace(params=FIDUCIAL, sigma8=lambda z: 1.), pfs_zmin=.6, settings=settings)]
    ids = [c.metadata['cell_id'] for c in like.cells]
    if ids != [c['cell_id'] for c in expected] or len(set(ids)) != 20:
        raise CellInventoryError('missing, duplicate or reordered physical cell')
    for cell, row in zip(like.cells, expected):
        for field, value in row.items():
            if canonical(cell.metadata.get(field)) != canonical(value):
                raise PhysicalReferenceError('physical density/bin/support differs: ' + field)
        observations = [[p, ell] for p in row['retained_pairs'] for ell in (0, 2, 4)]
        if canonical(cell.metadata['retained_observations']) != canonical(observations):
            raise PhysicalReferenceError('physical multipoles differ')
    grid = study_shell_grid(config)
    for field in ('edges', 'centers', 'widths'):
        if not np.array_equal(getattr(like.grid, field), getattr(grid, field)):
            raise PhysicalReferenceError('physical observed Fourier shells differ')
    for records in (result.parameters, like.parameters):
        if [p['name'] for p in records] != like.names:
            raise CosmologyOrderError('cosmology or nuisance parameter order differs')
        for p, name, sigma, unit in zip(records[:3], like.names[:3], (None, 5., 1.), ('', 'eV', '')):
            if (p['role'] != 'cosmo' or p['coordinate'] != name
                    or p['prior_sigma'] != sigma or p['unit'] != unit):
                raise PhysicalReferenceError('physical cosmological prior or unit differs')
        _cross_families(records, like.cells)
    if canonical(result.parameters) != canonical(like.parameters):
        raise PhysicalReferenceError('physical nuisance fiducials or prior records differ')
    if not np.array_equal(result.growth_z, np.mean(config.nominal_bins, axis=1)):
        raise PhysicalReferenceError('physical growth redshifts differ')
    return _cross_families(result.parameters, like.cells)


def load_inputs(source_root: Path, grid: str) -> StudyInputs:
    try:
        return _load_inputs(source_root, grid)
    except (KeyError, TypeError) as error:
        raise EvidenceError('missing required evidence record: ' + str(error)) from error


def _load_inputs(source_root: Path, grid: str) -> StudyInputs:
    source_root = Path(source_root).absolute()
    if grid not in GRID_PROFILES or not (source_root / 'inference' / grid / 'summary.json').is_file():
        raise MissingGridError('missing required grid: ' + grid)
    request = _read(source_root / 'request.json')
    report = _read(source_root / 'report-01/summary.json')
    state = _read(source_root / 'state.json')
    roots = [Path(p).parent for p, h in report['inputs'].items()
             if Path(p).name == 'request.json' and h == state['request_sha256']]
    if len(roots) != 1:
        raise EvidenceError('missing unique archived request binding')
    binding = _Bindings(source_root, roots[0])
    binding.add(request['bindings'])
    binding.add(report['inputs'])
    qualification_paths = {str(binding.path(p)) for p in report['inputs']}
    binding.check(source_root / 'request.json', state['request_sha256'])
    if (request['schema'] != 'takada-revision-v1' or request['pfs_zmin'] != .6
            or request['pfs_zmax'] != 2.4 or request['grids'] != GRID_PROFILES
            or report['status'] != 'passed' or report['main_only'] is not False):
        raise EvidenceError('unqualified or different physical reference calculation')
    summary_path = source_root / 'inference' / grid / 'summary.json'
    if str(summary_path) not in qualification_paths:
        raise EvidenceError('missing grid summary qualification binding: ' + grid)
    binding.check(summary_path)
    summary = _read(summary_path)
    expected_nesting = {a + '_to_' + b for a, b in
                        comparisons() + [('C21', 'C24'), ('B16', 'C24')] if a != 'I'}
    if set(summary['nesting']) != expected_nesting:
        raise EvidenceError('physical nesting comparison inventory differs: ' + grid)
    if (summary['status'] != 'passed' or summary['grid'] != grid
            or summary['request_sha256'] != state['request_sha256']
            or any(r['status'] != 'passed' for r in summary['nesting'].values())):
        raise EvidenceError('unqualified physical grid reference')
    source = request['original_source'][grid]
    _source_bindings(source, binding)
    if request['original_source_authentication'][grid]['status'] != 'passed':
        raise EvidenceError('unqualified original physical source')
    physical = source['details']['original']['details']
    profiles = corrected_profiles(2.4)
    results, metadata = {}, {}
    for name in ARCHIVED_PROFILES:
        path = source_root / 'inference' / grid / 'profiles' / name
        manifest = _read(path / 'manifest.json')
        if _digest(path / 'arrays.npz') != manifest['arrays_sha256']:
            raise ArrayHashError('altered archived array hash: ' + name)
        with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
            if np.count_nonzero(np.isclose(arrays['result/growth_z'], 1.1, rtol=0, atol=1e-12)) != 1:
                raise GrowthPivotError('missing unique headline growth pivot: ' + name)
        try:
            result, manifest = load_corrected_result(path)
        except ValueError as error:
            raise PhysicalReferenceError('physical saved reference differs: ' + str(error)) from error
        families = _validate_reference(result, manifest, profiles[name], GRID_PROFILES[grid], physical)
        binding.add({str(path / file): digest for file, digest in summary['profiles'][name]['files'].items()})
        if any(str(path / f) not in qualification_paths for f in ('manifest.json', 'arrays.npz', 'constraints.json')):
            raise EvidenceError('missing profile qualification binding: ' + name)
        if canonical(constraint_row(result)) != canonical(_read(path / 'constraints.json')):
            raise PhysicalReferenceError('physical saved constraints differ')
        if name in ('A', 'B'):
            if manifest['projections'] or summary['reuse_controls'][name]['status'] != 'passed':
                raise EvidenceError('missing authenticated unchanged DESI reference')
        else:
            if len(manifest['projections']) != len(result.likelihood.cells):
                raise CellInventoryError('missing physical cell projection')
            for cell, ref in zip(result.likelihood.cells, manifest['projections']):
                file = ref + '.json'
                projection_path = source_root / 'inference' / grid / 'projections' / file
                if str(projection_path) not in qualification_paths:
                    raise EvidenceError('missing projection qualification binding: ' + file)
                binding.add({str(projection_path): summary['projection_files'][file]})
                projection = _read(projection_path)
                if any(canonical(cell.metadata.get(k)) != canonical(v) for k, v in projection['expected'].items()):
                    raise PhysicalReferenceError('physical projection support differs')
                record = projection['source_record']
                if record is not None:
                    if (record['status'] != 'passed' or record['exit_code'] != 0
                            or any(projection[k]['status'] != 'passed' for k in ('derivative_check', 'neighborhood_check'))):
                        raise EvidenceError('unqualified physical projection')
                    binding.add({str(Path(projection['source_directory']) / f): h for f, h in record['files'].items()})
        info = {'archived_identifier': name, 'current_case': CURRENT_CASES[name],
                'selection': 'extended_to_2.1' if name == 'C21' else 'intermediate_high_z' if name == 'H21' else 'main_selection',
                'source_cell_count': len(result.likelihood.cells), 'cross_families': families,
                'pfs_nominal_bins': sorted({c.metadata['nominal_bin_index'] for c in result.likelihood.cells
                                           if 'PFS-ELG' in c.metadata['active_populations']}),
                'desi_nominal_bins': sorted({c.metadata['nominal_bin_index'] for c in result.likelihood.cells
                                            if any(n.startswith('DESI-') for n in c.metadata['active_populations'])}),
                'units': {'k': 'h_ref/Mpc', 'volume': '(Mpc/h_ref)^3', 'nbar': '(h_ref/Mpc)^3',
                          'h_ref': .6736, 'Mnu': 'eV'},
                'new_forecast_qualification': 'pending'}
        metadata[name] = {**manifest, 'overlap_reference': info}
        results[name] = result
    labels = _read(LABELS)
    if labels['legacy_to_current'] != {'I': None, 'A': 'A', 'B': 'B', 'D': 'C', 'C': 'D'}:
        raise PhysicalReferenceError('physical case label mapping differs')
    binding.hashes[str(LABELS)] = _digest(LABELS)
    g4 = _read(G4)
    binding.check(G4, G4_SHA256)
    for previous, archived in (('B', 'A'), ('C', 'B'), ('D', 'D'), ('E', 'C')):
        binding.add(g4[previous]['source'])
        if g4[previous]['growth_pivot'] != 1.1:
            raise GrowthPivotError('G4 headline growth pivot differs')
        metadata[archived]['overlap_reference']['g4_reference'] = {
            'path': str(G4), 'previous_label': previous, 'scope': 'main_selection_only',
            'quantity_order': ['fsigma8_z1.1', 'Mnu', 'Omegam'],
            'all_fixed_means': 'all_nuisance_coordinates_fixed'}
        if grid == 'shell304':
            result = results[archived]
            pivot = np.flatnonzero(np.isclose(result.growth_z, 1.1, rtol=0, atol=1e-12))[0]
            errors = np.sqrt([result.growth_cov_marg[pivot, pivot],
                              result.cov_cosmo_marg[1, 1], result.cov_cosmo_marg[2, 2]])
            if not np.allclose(errors, g4[previous]['treatments']['all_marginalized']['errors'], rtol=1e-8, atol=1e-12):
                raise PhysicalReferenceError('main-selection G4 reference constraints differ')
    binding.add(summary['original_inputs_checked'])
    binding.verify_all()
    binding.hashes[str(source_root / 'report-01/summary.json')] = _digest(source_root / 'report-01/summary.json')
    return StudyInputs(grid, source_root, results, metadata, binding.hashes)
