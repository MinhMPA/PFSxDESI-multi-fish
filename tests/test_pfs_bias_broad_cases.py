"""The broader scan changes real population fiducials at fixed survey support."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from pfsfog.cosmo import FIDUCIAL
from scripts.pfs_bias_cases import bias_cases
from scripts.pfs_bias_broad_cases import broad_cases, broad_schedule

SETTINGS={'nfft':2048,'n_mu':64,'dk':.000625}
COSMO=SimpleNamespace(params=FIDUCIAL,sigma8=lambda z:.6)


def test_exact_distinct_schedule_contains_the_full_prepared_population_range():
    rows=broad_schedule()
    assert len(rows)==len({r['key'] for r in rows})==35
    for i,(window,count,targets) in enumerate([
        ((.7,.8),11,{'DESI-LRG'}),
        ((1.4,1.6),13,{'DESI-ELG','DESI-QSO'}),
        ((1.6,2.),11,{'DESI-QSO'})]):
        selected=[r for r in rows if tuple(r['window'])==window]
        assert len(selected)==count
        assert sum(r['ratio']==1. for r in selected)==1
        assert selected[0]['ratio']==1.
        assert {r['equal_to'] for r in selected if r['equal_to']}==targets
        for prescription in ('b1_only','adopted_relations'):
            varied=[r for r in selected if r['prescription']==prescription]
            assert {.8,1.2,1.5,2.} <= {r['ratio'] for r in varied}
            assert {r['equal_to'] for r in varied if r['equal_to']}==targets
            assert all(r['key'].startswith(f'w{i}-{prescription}-') for r in varied)


def test_populations_keep_nominal_redshifts_support_and_independent_nuisances():
    bases={tuple(c['experiment']['window']):c for c in bias_cases(COSMO,SETTINGS)
           if c['experiment']['ratio']==1.}
    cases=broad_cases(COSMO,SETTINGS)
    assert [c['label'] for c in cases]==[r['key'] for r in broad_schedule()]
    for c in cases:
        experiment=c['experiment'];old=bases[tuple(experiment['window'])]
        assert c['expected']==old['expected']
        assert c['expected']['model_redshift']==old['expected']['model_redshift']
        assert c['expected']['local_names']==old['expected']['local_names']
        assert experiment['density_changed'] is False
        assert experiment['nuisance_relations_imposed_in_fit'] is False
        for tracer,fid in old['fiducials'].items():
            if tracer!='PFS-ELG':assert c['fiducials'][tracer]==fid
        target=old['fiducials']['PFS-ELG'].b1_sigma8*experiment['ratio']
        assert c['fiducials']['PFS-ELG'].b1_sigma8==pytest.approx(target,rel=1e-14)
        if experiment['ratio']==1.:assert c['fiducials']==old['fiducials']
        elif experiment['prescription']=='b1_only':
            want=asdict(old['fiducials']['PFS-ELG']);want['b1_sigma8']=target
            assert asdict(c['fiducials']['PFS-ELG'])==want


def test_each_equal_bias_case_matches_the_actual_desi_fiducial():
    cases=[c for c in broad_cases(COSMO,SETTINGS) if c['experiment']['equal_to']]
    assert len(cases)==8
    for c in cases:
        target=c['experiment']['equal_to']
        np.testing.assert_allclose(c['fiducials']['PFS-ELG'].b1_sigma8,
                                   c['fiducials'][target].b1_sigma8,rtol=1e-13,atol=0)


def test_adopted_relations_apply_at_large_bias_without_changing_fit_freedoms():
    cases=broad_cases(COSMO,SETTINGS)
    for c in cases:
        e=c['experiment']
        if e['prescription']!='adopted_relations':continue
        b=e['b1_changed'];f=c['fiducials']['PFS-ELG']
        assert f.b2_sigma8sq==pytest.approx((.412-2.143*b+.929*b*b+.008*b**3)*.36)
        assert f.bG2_sigma8sq==pytest.approx(-2*(b-1)/7*.36)
        assert f.bGamma3==pytest.approx(23*(b-1)/42)
        assert f.c2==pytest.approx(30*b/e['b1_reference'])
