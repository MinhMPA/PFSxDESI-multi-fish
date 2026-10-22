"""Shell refinement changes quadrature points, not physical support or priors."""
import copy
import importlib
from dataclasses import replace

import numpy as np
import pytest

OLD = 'shell0025-v1'
NEW = 'shell00125-v1'
AP = 'ap16384-intrinsic-v1'
SOLVER = 'decimal50-cholesky-v1'


def api():
    assert importlib.util.find_spec('pfsfog.shell_numerics'), 'explicit shell profile missing'
    return importlib.import_module('pfsfog.shell_numerics')


def grid_sample(dk=.00125, case='B', nk=16384, **kwargs):
    """Constant correlated Fisher density with real saved-likelihood algebra."""
    from test_cosmology_resolution import sample
    from test_fisher_solver_identity import precise
    from pfsfog.revision_forecast import make_shell_grid
    from pfsfog.fisher_core import fisher_from_shells
    like, _ = sample(nk, case=case, **kwargs)
    like.config = replace(like.config, dk=dk, kmax=.20)
    like.grid = make_shell_grid(.01, .20, dk)
    cell = like.cells[0]
    shape = np.eye(9)
    shape[0, 1] = shape[1, 0] = .2
    density = np.einsum('ij,jk->ik', np.linalg.cholesky(shape),
                        cell.derivatives[0], optimize=False) / np.sqrt(.19)
    cell.derivatives = np.broadcast_to(density, (len(like.grid.centers), 9, 8)).copy()
    cell.covariance = shape[None] / like.grid.widths[:, None, None]
    cell.fiducial_mean = np.ones((len(like.grid.centers), 9))
    cell.F_like = fisher_from_shells(cell.derivatives, cell.covariance, like.grid.centers)
    like.F_like = cell.F_like.copy()
    like.bin_F_like[2] = cell.F_like.copy()
    return like, precise(like)


def candidate_provenance():
    from test_fisher_solver_identity import provenance
    from pfsfog.ap_numerics import profile_record
    return {**provenance(), 'numerical_profile': profile_record(AP),
            'shell_profile': api().shell_profile_record(NEW)}


@pytest.mark.parametrize('profile,dk,count,refined', [(OLD,.0025,76,.00125),(NEW,.00125,152,.000625)])
def test_profile_produces_nested_midpoints_on_unchanged_support(profile,dk,count,refined):
    from pfsfog.revision_forecast import RevisionConfig, make_shell_grid
    a = api(); record = a.shell_profile_record(profile)
    config = RevisionConfig('B', dk=record['adopted_dk'])
    a.validate_shell_configuration(config, profile, 'adopted')
    coarse = make_shell_grid(config.kmin,config.kmax,config.dk)
    fine = make_shell_grid(config.kmin,config.kmax,record['control_dk'])
    assert config.dk == dk and record['control_dk'] == refined
    assert len(coarse.centers) == count and len(fine.centers) == 2*count
    np.testing.assert_allclose(coarse.edges, fine.edges[::2], rtol=0, atol=1e-15)
    assert coarse.edges[0] == fine.edges[0] == .01 and coarse.edges[-1] == fine.edges[-1] == .20
    assert np.isclose(coarse.widths.sum(), .19) and np.isclose(fine.widths.sum(), .19)


@pytest.mark.parametrize('change', ['dk','kmax','role','record','missing'])
def test_changed_or_omitted_candidate_grid_declaration_is_rejected(change):
    from pfsfog.revision_forecast import RevisionConfig
    a = api(); config = RevisionConfig('B', dk=.00125)
    if change in ('record','missing'):
        record = {'shell_profile': a.shell_profile_record(NEW)}
        if change == 'record': record['shell_profile']['control_dk'] = .00125
        else: record.pop('shell_profile')
        with pytest.raises(ValueError, match='shell'):
            a.shell_profile_from_record(record, requested=NEW)
    else:
        if change == 'dk': config = replace(config, dk=.0025)
        if change == 'kmax': config = replace(config, kmax=.15)
        with pytest.raises(ValueError, match='shell'):
            a.validate_shell_configuration(config, NEW, 'invented' if change=='role' else 'adopted')


@pytest.mark.parametrize('mutation', ['record','metadata','producer','configuration'])
def test_saved_control_cannot_lose_candidate_shell_identity(tmp_path,mutation):
    from scripts import run_extended_cosmology as d
    api(); like, results = grid_sample(); provenance = candidate_provenance()
    saved = d.save_control(tmp_path/'control',like,results,provenance=provenance)
    restored, _ = d.load_control(saved,provenance=provenance)
    assert len(restored.grid.centers) == 152
    if mutation == 'record': saved.pop('shell_profile')
    elif mutation == 'producer': saved['producer_identity'].pop('shell_profile')
    else:
        path=tmp_path/'control/likelihood.json'; meta=d.read_json(path)
        if mutation == 'metadata': meta.pop('shell_profile')
        else: meta['configuration']['dk'] = .0025
        d.write_json(path,meta);saved['metadata_sha256']=d.file_hash(path)
    with pytest.raises(ValueError): d.load_control(saved,provenance=provenance)
