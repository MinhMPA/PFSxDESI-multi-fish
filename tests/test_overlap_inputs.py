"""Authenticate the physical reference before changing the observed spectra."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from overlap_fixtures import overlap_inputs_for_grid, overlap_inputs, overlap_selected

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/takada-density-full-20261014-02'
PROFILES = ('A', 'B', 'D', 'C', 'C21', 'H21')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def copied_records(tmp_path):
    """Copy only editable records; the large archived arrays stay read-only."""
    dest = tmp_path / SOURCE.name
    (dest / 'report-01').mkdir(parents=True)
    for name in ('request.json', 'state.json', 'report-01/summary.json'):
        shutil.copy2(SOURCE / name, dest / name)
    grid = dest / 'inference/shell304'
    grid.mkdir(parents=True)
    shutil.copy2(SOURCE / 'inference/shell304/summary.json', grid / 'summary.json')
    (grid / 'projections').symlink_to(SOURCE / 'inference/shell304/projections', target_is_directory=True)
    for name in PROFILES:
        path = grid / 'profiles' / name
        path.mkdir(parents=True)
        original = SOURCE / 'inference/shell304/profiles' / name
        for file in ('manifest.json', 'constraints.json'):
            shutil.copy2(original / file, path / file)
        (path / 'arrays.npz').symlink_to(original / 'arrays.npz')
    # Complete the remaining archive through read-only links, so an omitted
    # qualification record cannot hide behind an unrelated missing input.
    for original, target in ((SOURCE, dest), (SOURCE / 'inference', dest / 'inference'),
                             (SOURCE / 'report-01', dest / 'report-01'),
                             (SOURCE / 'inference/shell304', grid),
                             (SOURCE / 'inference/shell304/profiles', grid / 'profiles')):
        for child in original.iterdir():
            link = target / child.name
            if not link.exists():
                link.symlink_to(child, target_is_directory=child.is_dir())
    return dest


def alter_manifest(root, change):
    path = root / 'inference/shell304/profiles/C21/manifest.json'
    record = read(path)
    change(record)
    write(path, record)


def rebind_grid_summary(root):
    from pfsfog.elf_survey_evidence import file_digest
    path = root / 'report-01/summary.json'
    report = read(path)
    report['inputs'][str(SOURCE / 'inference/shell304/summary.json')] = file_digest(
        root / 'inference/shell304/summary.json')
    write(path, report)


def test_conditions_and_physical_equivalences():
    from scripts.overlap_inputs import Condition, canonical_condition, conditions
    rows = conditions()
    assert isinstance(rows, tuple) and len(rows) == 32
    assert len(set(rows)) == 32
    assert len({canonical_condition(c) for c in rows}) == 15
    assert rows[:4] == tuple(Condition('A', 1200, s) for s in
                           ('marginalized', 'desi_fixed', 'pfs_fixed', 'all_cross_fixed'))
    for c in rows:
        expected = c
        if c.case == 'A':
            expected = Condition('A', 1200, 'marginalized')
        elif c.case in ('B', 'C'):
            treatment = 'desi_fixed' if c.stochastic in ('desi_fixed', 'all_cross_fixed') else 'marginalized'
            expected = Condition(c.case, 1200 if c.case == 'B' else c.area_deg2, treatment)
        assert canonical_condition(c) == expected
    with pytest.raises(FrozenInstanceError):
        rows[0].area_deg2 = 14000


@pytest.mark.parametrize('case,area,treatment', [('I',1200,'marginalized'),
    ('D',1464,'marginalized'), ('D',1200,'all_fixed')])
def test_undeclared_condition_is_rejected(case, area, treatment):
    from scripts.overlap_inputs import Condition
    with pytest.raises(ValueError, match='condition'):
        Condition(case, area, treatment)


@pytest.mark.parametrize('grid', ['loop2048','loop4096','angular64','shell304','final-loop'])
def test_complete_archived_reference_is_authenticated(grid, overlap_inputs_for_grid):
    inputs = overlap_inputs_for_grid(grid)
    assert inputs.grid == grid and inputs.source_root == SOURCE
    assert tuple(inputs.results) == PROFILES
    result = inputs.results['C21']
    assert result.likelihood.config.kmin == .01
    assert result.likelihood.config.kmax == .20
    assert result.names[:3] == ['ln10_10_As', 'Mnu', 'Omegam']
    assert len(result.names) == 276
    assert len(result.likelihood.cells) == 20
    info = inputs.metadata['C21']['overlap_reference']
    assert info['current_case'] == 'D'
    assert info['pfs_nominal_bins'] == list(range(1, 8))
    assert info['desi_nominal_bins'] == list(range(8))
    assert inputs.metadata['H21']['overlap_reference']['current_case'] is None
    assert inputs.metadata['D']['overlap_reference']['current_case'] == 'C'
    assert inputs.metadata['C']['overlap_reference']['g4_reference']['scope'] == 'main_selection_only'
    assert inputs.hashes[str(SOURCE / 'request.json')]
    assert inputs.hashes[str(SOURCE / 'report-01/summary.json')]
    assert inputs.hashes[str(SOURCE / 'inference' / grid / 'summary.json')]
    cross = [p for p in result.parameters if p['role'] == 'cross']
    assert cross and {p['coordinate'] for p in cross} == {'Pshot_cross', 'a2_cross'}
    assert all(p['fiducial'] == 0. and p['prior_sigma'] == 1. for p in cross)


def test_missing_grid_has_named_error():
    from scripts.overlap_inputs import load_inputs, MissingGridError
    with pytest.raises(MissingGridError, match='grid'):
        load_inputs(SOURCE, 'not-a-grid')


def test_changed_array_hash_is_rejected(copied_records):
    from scripts.overlap_inputs import load_inputs, ArrayHashError
    alter_manifest(copied_records, lambda m: m.update(arrays_sha256='0' * 64))
    with pytest.raises(ArrayHashError, match='array'):
        load_inputs(copied_records, 'shell304')


def test_copied_archive_can_have_another_directory_name(copied_records):
    from scripts.overlap_inputs import load_inputs, ArrayHashError
    moved = copied_records.rename(copied_records.parent / 'copied-physical-reference')
    alter_manifest(moved, lambda m: m.update(arrays_sha256='0' * 64))
    with pytest.raises(ArrayHashError, match='array'):
        load_inputs(moved, 'shell304')


def test_duplicate_cell_is_rejected(copied_records):
    from scripts.overlap_inputs import load_inputs, CellInventoryError
    def change(m):
        m['likelihood']['cells'][-1]['metadata']['cell_id'] = m['likelihood']['cells'][0]['metadata']['cell_id']
    alter_manifest(copied_records, change)
    with pytest.raises(CellInventoryError, match='cell'):
        load_inputs(copied_records, 'shell304')


@pytest.mark.parametrize('field', ['nbar', 'nominal_bin'])
def test_changed_density_or_bin_is_rejected(copied_records, field):
    from scripts.overlap_inputs import load_inputs, PhysicalReferenceError
    def change(m):
        row = m['likelihood']['cells'][0]['metadata']
        if field == 'nbar':
            row[field]['DESI-LRG'] *= 1.1
        else:
            row[field][0] = .41
    alter_manifest(copied_records, change)
    with pytest.raises(PhysicalReferenceError, match='density|bin|physical'):
        load_inputs(copied_records, 'shell304')


def test_wrong_cosmology_order_is_rejected(copied_records):
    from scripts.overlap_inputs import load_inputs, CosmologyOrderError
    def change(m):
        m['likelihood']['names'][:2] = ['Mnu', 'ln10_10_As']
    alter_manifest(copied_records, change)
    with pytest.raises(CosmologyOrderError, match='cosmolog'):
        load_inputs(copied_records, 'shell304')


@pytest.mark.parametrize('duplicate', [False, True])
def test_growth_pivot_must_be_unique(copied_records, duplicate):
    from scripts.overlap_inputs import load_inputs, GrowthPivotError
    path = copied_records / 'inference/shell304/profiles/C21/arrays.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    z = arrays['result/growth_z']
    z[2 if duplicate else 3] = 1.1 if duplicate else 1.11
    path.unlink()  # Remove the copied link, never the archived array.
    np.savez_compressed(path, **arrays)
    from pfsfog.elf_survey_evidence import file_digest
    alter_manifest(copied_records, lambda m: m.update(arrays_sha256=file_digest(path)))
    with pytest.raises(GrowthPivotError, match='growth'):
        load_inputs(copied_records, 'shell304')


def test_cross_family_is_read_from_parameter_records(copied_records):
    from scripts.overlap_inputs import load_inputs, StochasticParameterError
    def change(m):
        for records in (m['parameters'], m['likelihood']['parameters']):
            p = next(p for p in records if p['role'] == 'cross')
            p['tracers'] = ['PFS-ELG', 'PFS-ELG']
    alter_manifest(copied_records, change)
    with pytest.raises(StochasticParameterError, match='cross|stochastic'):
        load_inputs(copied_records, 'shell304')


def test_required_evidence_never_silently_skips(copied_records):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    (copied_records / 'report-01/summary.json').unlink()
    with pytest.raises(EvidenceError, match='missing'):
        load_inputs(copied_records, 'shell304')


def test_named_projection_requires_a_qualification_binding(copied_records):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    path = copied_records / 'report-01/summary.json'
    report = read(path)
    manifest = read(copied_records / 'inference/shell304/profiles/C21/manifest.json')
    original = SOURCE / 'inference/shell304/projections' / (manifest['projections'][0] + '.json')
    del report['inputs'][str(original)]
    write(path, report)
    with pytest.raises(EvidenceError, match='qualification binding'):
        load_inputs(copied_records, 'shell304')


def test_missing_named_profile_has_evidence_error(copied_records):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    path = copied_records / 'inference/shell304/summary.json'
    summary = read(path)
    del summary['profiles']['C21']
    write(path, summary)
    rebind_grid_summary(copied_records)
    with pytest.raises(EvidenceError, match='missing required evidence record'):
        load_inputs(copied_records, 'shell304')


def test_grid_summary_requires_a_qualification_binding(copied_records):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    path = copied_records / 'report-01/summary.json'
    report = read(path)
    del report['inputs'][str(SOURCE / 'inference/shell304/summary.json')]
    write(path, report)
    summary_path = copied_records / 'inference/shell304/summary.json'
    summary = read(summary_path)
    summary['nesting'] = {}
    write(summary_path, summary)
    with pytest.raises(EvidenceError, match='grid summary qualification binding'):
        load_inputs(copied_records, 'shell304')


@pytest.mark.parametrize('empty', [False, True])
def test_grid_summary_requires_complete_nesting_comparisons(copied_records, empty):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    path = copied_records / 'inference/shell304/summary.json'
    summary = read(path)
    if empty:
        summary['nesting'] = {}
    else:
        del summary['nesting']['H21_to_C21']
    write(path, summary)
    rebind_grid_summary(copied_records)
    with pytest.raises(EvidenceError, match='nesting comparison inventory'):
        load_inputs(copied_records, 'shell304')


def test_grid_summary_recorded_hash_is_verified(copied_records):
    from scripts.overlap_inputs import load_inputs, EvidenceError
    path = copied_records / 'inference/shell304/summary.json'
    summary = read(path)
    summary['qualification_note'] = 'changed copied record'
    write(path, summary)
    with pytest.raises(EvidenceError, match='altered required evidence: .*summary.json'):
        load_inputs(copied_records, 'shell304')
