"""The complete cosmological and nuisance response of the CT-A spectrum."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('cross', [False, True])
def test_complete_response_finite_differences_at_three_steps(cross):
    from pfsfog.cosmo import FIDUCIAL
    from pfsfog.cosmology_space import make_cosmology_spec
    from pfsfog.elf_adapter import ELFPowerSpectrum
    from pfsfog.eft_params import tracer_fiducials
    from pfsfog.revision_theory import make_pair_response, fiducial_nuisance
    space = make_cosmology_spec(FIDUCIAL, extended=True)
    response = make_pair_response(ELFPowerSpectrum(), .9, FIDUCIAL, 3.e-4,
                                  1.e-4 if cross else None, space=space)
    theta = jnp.array([FIDUCIAL[n] for n in ('ln10_10_As', 'mnu', 'Omega_m', 'h', 'omega_b', 'n_s')])
    # Displace all cosmological coordinates: the AP map is then nonidentity.
    theta += jnp.array([.01, .002, .001, .001, .00002, .001])
    qa = fiducial_nuisance(tracer_fiducials('DESI-ELG', 1.3, response.sigma8_ref))
    qb = fiducial_nuisance(tracer_fiducials('DESI-LRG', 2., response.sigma8_ref))
    qa = qa.at[4:8].set(jnp.array([4., -7., 9., 110.]))
    qb = qb.at[4:8].set(jnp.array([-2., 3., 5., 250.]))
    point = jnp.concatenate((theta, qa, qb)) if cross else jnp.concatenate((theta, qa))
    def mean(x):
        return response.multipoles(jnp.array([.04, .11]), x[:6], x[6:17],
                                    x[17:] if cross else None, n_mu=16)
    call = jax.jit(mean)
    derivative = np.asarray(jax.jit(jax.jacfwd(mean))(point))
    assert np.isfinite(derivative).all()
    # All six cosmological coordinates and each population's eleven nuisance
    # coordinates move independently. Fixed Q,T do not freeze d,b,f or A.
    steps = np.r_[np.array([2.e-4, 2.e-4, 2.e-5, 2.e-5, 2.e-6, 2.e-5]),
                  2.e-4*np.maximum(abs(np.asarray(point[6:])), 1.)]
    for i, step in enumerate(steps):
        errors = []
        for factor in (1., .5, .25):
            dx = jnp.zeros_like(point).at[i].set(step*factor)
            fd = np.asarray((call(point+dx)-call(point-dx))/(2*step*factor))
            scale = max(np.linalg.norm(derivative[..., i]), 1.)
            errors.append(np.linalg.norm(fd-derivative[..., i])/scale)
        assert errors[1] < 1.e-3 and errors[2] < 1.e-3, (cross, i, errors)
