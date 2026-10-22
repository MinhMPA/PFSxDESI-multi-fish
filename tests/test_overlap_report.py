"""Reporting checks preserve absolute-error denominators and strict resolution."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from scripts import report_overlap_study as report


def record(ratio=0.999, gain=None):
    return {'identity': {'redshifts': [.5, 1.1], 'density': 'Takada', 'priors': [None, 5., 1.]},
            'metrics': {'sigma/A': {'value': 10., 'kind': 'uncertainty', 'unit': 'eV'},
                        'ratio/A-B': {'value': ratio, 'kind': 'ratio', 'unit': 'dimensionless'},
                        'gain/A-B': {'value': 100*(1-ratio) if gain is None else gain,
                                     'kind': 'gain', 'unit': 'percentage points'}}}


def test_gain_denominator_and_invalid_errors():
    assert report.gain_percent(10., 8.) == pytest.approx(20.)
    for a,b in [(0,1), (1,0), (-1,2), (np.nan,1), (1,np.inf)]:
        with pytest.raises(ValueError): report.gain_percent(a,b)


def test_strict_gain_boundary_and_small_gain_resolution():
    assert report.compare_grids(record(.999), record(.99849))['status'] == 'failed'
    assert report.compare_grids(record(gain=0), record(gain=.05))['status'] == 'failed'
    assert report.compare_grids(record(gain=0), record(gain=.049999))['status'] == 'passed'


def test_strict_relative_boundary():
    fine=record(); fine['metrics']['sigma/A']['value']=10.1
    assert report.compare_grids(record(), fine)['status'] == 'failed'


@pytest.mark.parametrize('field', ['redshifts','density','priors'])
def test_mixed_physical_identities_rejected(field):
    fine=record(); fine['identity'][field]='different'
    with pytest.raises(ValueError, match='identity'): report.compare_grids(record(),fine)


def test_missing_metric_and_wrong_units_rejected():
    fine=record(); fine['metrics'].pop('ratio/A-B')
    with pytest.raises(ValueError): report.compare_grids(record(),fine)
    fine=record(); fine['metrics']['gain/A-B']['unit']='percent ratio'
    with pytest.raises(ValueError): report.compare_grids(record(),fine)


def test_successive_gain_closure_is_multiplicative():
    rows={c:{'q':v} for c,v in zip('ABCD',[10.,8.,6.,3.])}
    gains=report.case_comparisons(rows)
    assert gains['A-B']['q']['before']==10.
    assert gains['B-C']['q']['before']==8.
    assert gains['C-D']['q']['before']==6.
    assert gains['B-D']['q']['before']==8.
    assert gains['A-D']['q']['before']==10.
    product=np.prod([gains[p]['q']['ratio'] for p in ('A-B','B-C','C-D')])
    assert product==pytest.approx(gains['A-D']['q']['ratio'])
    assert sum(gains[p]['q']['gain_percent'] for p in ('A-B','B-C','C-D')) != gains['A-D']['q']['gain_percent']


def test_area_gain_holds_case_and_stochasticity_fixed():
    records={f'{c}-area{a:05d}-{s}':{'q':float(20-i-j)}
             for i,c in enumerate('ABCD') for j,a in enumerate((1200,14000))
             for s in report.TREATMENTS}
    rows=report.area_comparisons(records)
    assert len(rows)==16
    for key,row in rows.items():
        c,s=key.split('/',1)
        assert row['q']['before']==records[f'{c}-area01200-{s}']['q']
        assert row['q']['after']==records[f'{c}-area14000-{s}']['q']


def test_information_ordering_checks_common_named_variances():
    old={'names':['x','y'], 'variance':[2.,3.], 'growth_variance':[4.]}
    new={'names':['y','x','z'], 'variance':[2.,1.,9.], 'growth_variance':[3.]}
    assert report.variance_ordering(old,new)['status']=='passed'
    new['variance'][1]=2.0001
    assert report.variance_ordering(old,new)['status']=='failed'


def test_conditional_norm_uses_covariance_and_named_coordinates():
    conditional={'names':['n','c'], 'D':np.array([[[2.,3.]]]), 'C':np.array([[[4.]]])}
    got=report.response_norm(conditional,['c','n'],np.array([2.,1.]))
    assert got==pytest.approx(16.) # (2*1+3*2)^2 / 4


def test_after_direction_preserves_other_cosmological_marginalization():
    c=np.array([[4.,1.],[1.,2.]])
    v,w=report.compensated_direction(c,np.array([[1.],[2.]]),np.array([[5.]]),np.array([1.,0.]))
    np.testing.assert_allclose(v,[2.,.5])
    np.testing.assert_allclose(w,[-.6])


def test_unqualified_campaign_cannot_report(tmp_path,monkeypatch):
    monkeypatch.setattr(report.campaign,'status',lambda _: {'status':'partial'})
    with pytest.raises(ValueError,match='complete'): report.report(tmp_path,tmp_path/'report')
    assert not (tmp_path/'report').exists()


@pytest.fixture
def copied_campaign(tmp_path,monkeypatch):
    """Isolate mutated output receipts; original numerical arrays stay read-only."""
    import shutil
    source=Path(__file__).resolve().parents[1]/'results/overlap-information-20261020-02'
    for filename in ('request.json','state.json'): shutil.copyfile(source/filename,tmp_path/filename)
    (tmp_path/'inference').mkdir()
    for grid in report.GRID_PROFILES:
        original=source/'inference'/grid; dest=tmp_path/'inference'/grid
        if grid!='shell304': dest.symlink_to(original,target_is_directory=True); continue
        dest.mkdir()
        for p in original.iterdir():
            if p.is_file(): shutil.copyfile(p,dest/p.name)
            elif (p/'manifest.json').exists():
                target=dest/p.name; target.mkdir()
                for q in p.iterdir():
                    if q.name=='arrays.npz': (target/q.name).symlink_to(q)
                    elif q.is_file(): shutil.copyfile(q,target/q.name)
    # Only the already immutable 38.8 GB source is omitted in these corruption
    # tests. Output manifests, arrays, aliases and completion stay authenticated.
    monkeypatch.setattr(report.campaign,'_verify_bindings',lambda _:None)
    return tmp_path


def _rehash_manifest(output,name):
    state=report.campaign.read(output/'state.json')
    for r in state['grids']['shell304']['records']:
        if r['condition_id']==name:
            r['manifest_sha256']=report.campaign.digest(output/'inference'/'shell304'/name/'manifest.json')
    report.campaign.write(output/'state.json',state)


@pytest.mark.parametrize('corruption',['missing_grid','missing_condition','wrong_alias','changed_arrays','unqualified'])
def test_corrupted_saved_campaign_cannot_pass(copied_campaign,corruption):
    output=copied_campaign
    if corruption in ('missing_grid','missing_condition'):
        state=report.campaign.read(output/'state.json')
        if corruption=='missing_grid': state['grids'].pop('shell304')
        else: state['grids']['shell304']['records'].pop()
        report.campaign.write(output/'state.json',state)
    elif corruption=='changed_arrays':
        p=output/'inference/shell304/A-area01200-marginalized/arrays.npz'
        p.unlink(); p.write_bytes(b'altered output arrays')
    else:
        name='A-area01200-desi_fixed'
        p=output/'inference'/'shell304'/name/'manifest.json'; meta=report.campaign.read(p)
        if corruption=='wrong_alias': meta['alias_of']='B-area01200-marginalized'
        else: meta['consumable']=False
        report.campaign.write(p,meta); _rehash_manifest(output,name)
    with pytest.raises(ValueError): report.report(output,output/'report')
    assert not (output/'report').exists()


@pytest.fixture(scope='module')
def shell304_summary(tmp_path_factory):
    import zipfile
    output=Path(__file__).resolve().parents[1]/'results/overlap-information-20261020-02'
    records=report._load_results(output,'shell304')
    destination=tmp_path_factory.mktemp('overlap-report')/'diagnostics.npz'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        summary=report._grid_summary(records,'shell304',archive)
    return summary,destination


def test_real_saved_likelihood_ordering_and_metric_inventory(shell304_summary):
    summary,_=shell304_summary
    assert summary['ordering']['status']=='passed'
    assert len(summary['constraints'])==32*11
    assert len(summary['gains'])==8*2*6*11
    assert len(summary['area_gains'])==16*2*11
    assert len(summary['metrics'])==3872
    assert report.compare_grids(summary,summary)['status']=='passed'
    assert all(r['nuisance_penalty']>=1-1e-8 for r in summary['constraints'])


def test_real_conditional_norm_and_schur_fixed_covariance(shell304_summary):
    summary,path=shell304_summary
    with np.load(path,allow_pickle=False) as arrays:
        for name,record in summary['diagnostic_records'].items():
            prefix=record['array_prefix']
            h=arrays[prefix+'/H']; c=arrays[prefix+'/C_fixed']
            np.testing.assert_allclose(h@c,np.eye(3),rtol=0,atol=1e-9)
            # Correlated cosmological coordinates remain marginalized when nuisances are fixed.
            assert np.any(np.diag(c)>1/np.diag(h)*(1+1e-3))
        for prefix,record in summary['conditional'].items():
            assert record['after_case'] in ('B','D')
            for q,total in record['norms'].items():
                v=arrays[prefix+'/'+q+'/v']; w=arrays[prefix+'/'+q+'/w']
                direction=np.r_[v,w]; global_names=record['global_names']
                independent=0.
                for cell in record['cells']:
                    local=direction[[global_names.index(n) for n in cell['names']]]
                    f=arrays[cell['array_prefix']+'/F']
                    independent+=float(np.einsum('i,ij,j->',local,f,local,optimize=False))
                assert total['nuisance_compensated']==pytest.approx(independent,rel=1e-8,abs=1e-10)


def test_report_headlines_state_growth_redshift_explicitly(shell304_summary):
    summary,_=shell304_summary
    text=report._physical_report({'status':'passed','grids':{'shell304':summary},'refinements':{}})
    assert 'fσ₈(1.1)' not in text
    assert 'σ[fσ₈(z=1.1)]' in text
    assert 'fσ₈(z=1.1) gain [%]' in text
    assert 'fσ₈(z=1.1) area gain [%]' in text
