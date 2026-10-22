"""A bias experiment must change a population, not its density or fit freedoms."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from scripts.elf_study_profiles import highz_profiles, study_cells
from scripts.pfs_bias_cases import bias_schedule, bias_cases, vary_case

SETTINGS = {'nfft':2048, 'n_mu':64, 'dk':.000625}
COSMO = SimpleNamespace(params=FIDUCIAL, sigma8=lambda z:.6)


def original():
    return next(c for c in study_cells(highz_profiles()['C22'], COSMO,
        settings=SETTINGS, extended=True) if c['cell_id']=='bin01-radial01-overlap')


def test_change_physical_bias_preserves_density_support_and_other_fiducials():
    old = original()
    new = vary_case(old, 1.2, 'b1_only', .6, 1.3)
    assert new['expected'] == old['expected']
    assert new['fiducials']['PFS-ELG'].b1_sigma8 == pytest.approx(1.18*1.2*.6)
    for tracer, fid in old['fiducials'].items():
        want=asdict(fid)
        if tracer=='PFS-ELG': want['b1_sigma8']*=1.2
        assert asdict(new['fiducials'][tracer])==want
    assert old['fiducials']['PFS-ELG'].b1_sigma8 == pytest.approx(1.18*.6)


def test_relation_fiducials_do_not_remove_independent_nuisance_coordinates():
    old=original();new=vary_case(old, 1.2, 'adopted_relations', .6, 1.3)
    f=new['fiducials']['PFS-ELG'];b=1.416
    assert f.b2_sigma8sq==pytest.approx((.412-2.143*b+.929*b*b+.008*b**3)*.36)
    assert f.bG2_sigma8sq==pytest.approx(-2*(b-1)/7*.36)
    assert f.bGamma3==pytest.approx(23*(b-1)/42)
    assert f.c2==pytest.approx(30*b/1.3)
    assert f.c_tilde==pytest.approx(126.5625)
    assert new['expected']['local_names']==old['expected']['local_names']


@pytest.mark.parametrize('prescription',['b1_only','adopted_relations'])
def test_unvaried_population_is_exact_even_with_floating_amplitude(prescription):
    old=original();new=vary_case(old,1.,prescription,.6000000000001,1.3)
    assert new['fiducials']==old['fiducials']


@pytest.mark.parametrize('ratio',[0,-1,np.nan,np.inf,True,[1.]])
def test_reject_nonphysical_bias_ratios(ratio):
    with pytest.raises(ValueError):vary_case(original(),ratio,'b1_only',.6,1.3)


def test_reject_unknown_population_prescription():
    with pytest.raises(ValueError):vary_case(original(),1.,'unrecognized',.6,1.3)


def test_schedule_measures_unvaried_population_once_and_uses_nominal_redshift():
    cases=bias_cases(COSMO,SETTINGS)
    assert len(cases)==len(bias_schedule())==15
    assert len({c['label'] for c in cases})==15
    assert sum(c['experiment']['ratio']==1. for c in cases)==3
    assert {tuple(c['expected']['radial_bin']) for c in cases}=={(.7,.8),(1.4,1.6),(1.6,2.)}
    for c in cases:
        lo=c['expected']['radial_bin'][0]
        assert c['expected']['model_redshift']==pytest.approx({.7:.7,1.4:1.5,1.6:1.8}[lo])
        if lo in (.7,1.6): assert c['experiment']['b1_reference']==1.3
        assert c['expected']['region']=='overlap'
