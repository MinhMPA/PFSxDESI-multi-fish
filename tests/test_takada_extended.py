"""The density correction must retain the actual six-parameter likelihood."""
from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace


def test_all_six_cosmology_responses_are_checked_as_physical_columns():
    from scripts.takada_extended_checks import check_extended_derivatives
    names=['ln10_10_As','Mnu','Omegam','h','omega_b','n_s']
    D=np.arange(1,37,dtype=float).reshape(2,3,6)/20
    steps=np.full((6,3),.001)*np.array([1.,.5,.25])
    mean=np.ones((2,3))*10
    displacement=D.transpose(2,0,1)[:,None]*steps[:,:,None,None]
    plus=mean[None,None]+displacement;minus=mean[None,None]-displacement
    projected={'cell':SimpleNamespace(names=names,derivatives=D.copy(),
        covariance=np.broadcast_to(np.eye(3),(2,3,3)).copy(),fiducial_mean=mean,
        metadata={'retained_pairs':[('PFS-ELG','PFS-ELG')]}),
        'arrays':{'fd_steps':steps,'fd_plus':plus,'fd_minus':minus,
                  'fd_derivatives':(plus-minus)/(2*steps[:,:,None,None])},
        'analytic_zero_support':[]}
    good=check_extended_derivatives(projected)
    assert good['status']=='passed'
    assert {r['column'] for r in good['column_results']}==set(names)
    assert good['coverage_count']==6*3*3
    projected['cell'].derivatives[:,:,3]*=1.01
    with pytest.raises(ValueError,match='finite difference failed'):
        check_extended_derivatives(projected)


def test_extended_template_is_the_saved_likelihood_with_all_six_responses():
    from scripts.run_takada_extended import load_extended_template
    path=Path(__file__).resolve().parents[1]/'results/elf-grid-20261009-shell304-inference-01/cases/C'
    like,record=load_extended_template(path)
    assert like.names[:6]==['ln10_10_As','Mnu','Omegam','h','omega_b','n_s']
    assert like.config.cosmology_basis=='extended6'
    assert like.fiducial[:6].tolist()==[3.0445224377,.06,0.315191849347291,.6736,.02237,.9649]
    np.testing.assert_allclose(sum(like.bin_F_like),like.F_like,rtol=1e-14,atol=1e-10)
    assert set(record['treatments'])=={'conditional3','release_h','release_omega_b','release_n_s',
                                     'marginal6','marginal6_weak10','marginal6_no_extra'}
