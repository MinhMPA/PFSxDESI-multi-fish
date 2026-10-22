"""Shared physical information must survive regional assembly unchanged."""
from dataclasses import replace
import copy
import importlib
import importlib.util

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from pfsfog.elf_survey import elf_cosmology_spec
from pfsfog.revision_forecast import CellLikelihood, RevisionConfig, make_shell_grid
from pfsfog.revision_validation_physical import expected_support_schedule
from pfsfog.surveys import default_survey_group


def api():
    spec = importlib.util.find_spec('pfsfog.elf_inference')
    assert spec is not None, 'ELF global assembly is missing'
    return importlib.import_module('pfsfog.elf_inference')


def analytic_cells(case='B'):
    config = RevisionConfig(case, cosmology_basis='extended6', kmax=.02, dk=.005)
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    schedule = expected_support_schedule(config, default_survey_group(),
                                         space=elf_cosmology_spec(FIDUCIAL))
    cells, fids = [], []
    cosmology = dict(zip(elf_cosmology_spec(FIDUCIAL).names,
        [FIDUCIAL[n] for n in ('ln10_10_As','mnu','Omega_m','h','omega_b','n_s')]))
    for row in schedule:
        meta = {k:copy.deepcopy(v) for k,v in row.items() if k not in ('empty_observation','absent_observations')}
        active = row['active_populations']
        meta['theoretical_pairs'] = [(a,a) for a in active]+[(a,b) for i,a in enumerate(active) for b in active[i+1:]]
        meta['retained_observations'] = [(pair,ell) for pair in row['retained_pairs'] for ell in (0,2,4)]
        names = list(row['local_names']); nobs = 3*len(row['retained_pairs'])
        d = np.zeros((len(grid.centers),nobs,len(names)))
        # Hand-checkable amplitude/nuisance degeneracy. Regional volumes enter C.
        d[:,:,0] = 1.
        d[:,:,6] = 2.
        c = np.tile(np.eye(nobs),(len(grid.centers),1,1)) / row['volume']
        f = np.einsum('koi,koj->ij',d,np.linalg.solve(c,d))
        validation = dict(expected_shells=len(grid.centers),processed_shells=len(grid.centers),
            field_min_scaled_eigenvalue=1.,field_max_scaled_condition=1.,
            covariance_min_scaled_eigenvalue=1.,covariance_max_scaled_condition=1.,max_scaled_residual=0.)
        cells.append(CellLikelihood(f,names,meta,validation,d,c,np.ones(d.shape[:2])))
        fids.append({name:cosmology.get(name,0. if '|' in name else 1.) for name in names})
    return config, cells, fids


def test_regions_share_one_nuisance_and_information_has_no_prior():
    module=api(); config,cells,fids=analytic_cells()
    result=module.assemble_likelihood(config,FIDUCIAL,{'engine':'elf.EPT'},cells,fids)
    target='bin00:DESI-LRG:b1_sigma8'
    assert result.names.count(target)==1
    i=result.names.index(target)
    own=[c for c in cells if target in c.names]
    weight=sum(2*3*len(c.metadata['retained_pairs'])*c.metadata['volume'] for c in own)
    assert result.F_like[i,i]==pytest.approx(4*weight)
    assert result.F_like[0,i]==pytest.approx(2*weight)
    assert result.F_like[1,1]==0.  # Mnu prior has not been added regionally or globally.
    assert result.F_like[2,2]==0.  # Omega_m prior has not been added.
    np.testing.assert_array_equal(sum(result.bin_F_like),result.F_like)
    assert result.parameters[i]['fiducial']==1.
    assert len(result.cells)==20


def test_local_column_reordering_preserves_global_information():
    module=api(); config,cells,fids=analytic_cells('C')
    original=module.assemble_likelihood(config,FIDUCIAL,{'engine':'elf.EPT'},cells,fids)
    for cell in cells:
        order=list(reversed(range(len(cell.names))))
        cell.names=[cell.names[i] for i in order]
        cell.metadata['local_names']=cell.names.copy()
        cell.derivatives=cell.derivatives[:,:,order]
        cell.F_like=cell.F_like[np.ix_(order,order)]
    other=module.assemble_likelihood(config,FIDUCIAL,{'engine':'elf.EPT'},cells,fids)
    assert original.names==other.names
    np.testing.assert_array_equal(original.F_like,other.F_like)


@pytest.mark.parametrize('damage',['missing','duplicate','fiducial','cosmology','local_names','D','C','F','shells'])
def test_inconsistent_physical_inputs_are_rejected(damage):
    module=api(); config,cells,fids=analytic_cells()
    if damage=='missing': cells.pop();fids.pop()
    elif damage=='duplicate': cells[-1]=copy.deepcopy(cells[0]);fids[-1]=dict(fids[0])
    elif damage=='fiducial': fids[1]['bin00:DESI-LRG:b1_sigma8']+=.1
    elif damage=='cosmology': fids[0]['h']+=.001
    elif damage=='local_names': cells[0].names[6]='bin99:DESI-LRG:b1_sigma8'
    elif damage=='D': cells[0].derivatives[0,0,0]+=1.
    elif damage=='C': cells[0].covariance[0,0,0]=-1.
    elif damage=='F': cells[0].F_like[0,0]+=1.e10
    elif damage=='shells': cells[0].derivatives=cells[0].derivatives[:-1]
    with pytest.raises(ValueError):
        module.assemble_likelihood(config,FIDUCIAL,{'engine':'elf.EPT'},cells,fids)


def test_missing_physical_cohort_cannot_supply_global_information(tmp_path):
    with pytest.raises(ValueError,match='physical'):
        api().load_physical_likelihood(tmp_path,'B')


def test_saved_loader_preserves_measured_k0_subtraction_and_engine_flags(monkeypatch,tmp_path):
    from dataclasses import asdict
    import json
    module=api();config,cells,fids=analytic_cells()
    settings={'engine':'elf.EPT','counterterm_base':'linear','subtract_k0_const':True,'do_irres':False,
              'fft_n':1024,'fft_kmin':1.e-5,'fft_kmax':100.}
    request={'configurations':{'B':asdict(config)},'expected':[],
        'source':{'details':{'reference_cosmology':FIDUCIAL,'engine_settings':settings,
                             'model_flags':{'backend_class':'ELFPowerSpectrum','subtract_k0_limit':True}}}}
    grid=make_shell_grid(config.kmin,config.kmax,config.dk)
    for cell,point in zip(cells,fids):
        key='B:'+cell.metadata['cell_id'];request['expected'].append({'case':'B','key':key})
        p=tmp_path/'cells'/key.replace(':','__');p.mkdir(parents=True)
        (p/'inputs.json').write_text(json.dumps({'metadata':cell.metadata,'fiducials':point}))
        (p/'measurement.json').write_text(json.dumps({'covariance':cell.validation}))
        np.savez_compressed(p/'arrays.npz',F_like=cell.F_like,D=cell.derivatives,C=cell.covariance,
                            mean=cell.fiducial_mean,k=grid.centers,dk=grid.widths)
    (tmp_path/'request.json').write_text(json.dumps(request))
    # Upstream authenticity is tested by the physical verifier; this exercises loading.
    monkeypatch.setattr(module,'verify_run',lambda root:{'status':'passed','complete_survey':True})
    loaded=module.load_physical_likelihood(tmp_path,'B')
    assert loaded.model['subtract_k0_limit'] is True
    assert all(loaded.model[k]==v for k,v in settings.items())
