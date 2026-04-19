"""
Background cosmology module for flat nuLCDM.

Provides H(z) and D_A(z) as pure JAX functions, fully compatible with
jax.jit and jax.jacfwd.  Uses quadax.cumulative_simpson for the comoving
distance integral (fastest + no interpolation error).

Units
-----
- H(z)  :  km / s / Mpc   (physical)
- D_A(z):  Mpc             (physical)
- chi(z):  Mpc             (physical)

Parameters are the *physical* densities omega_X = Omega_X h^2 that the
emulators already carry, plus h and (optionally) mnu [eV].
"""

import jax
import jax.numpy as jnp
import quadax

__all__ = [
    "Esqr",
    "Hz",
    "dchioverdz",
    "chi",
    "angular_diameter_distance",
    "comoving_angular_diameter_distance",
    "hubble_distance",
    "volume_distance",
    "recombination_redshift_hu_sugiyama",
    "recombination_redshift_aizpuru2021",
    "sound_horizon",
    "theta_star",
    "sound_horizon_drag",
    "sound_horizon_drag_aubourg2014",
    "sound_horizon_drag_aubourg2014_neff",
    "sound_horizon_drag_brieden2022",
    "dm_over_rs",
    "dh_over_rs",
    "dv_over_rs",
    "Omega_m_of_z",
    "growth_factor",
    "growth_factor_approx",
    "growth_rate",
    "growth_rate_approx",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_KMS = 299792.458  # speed of light [km/s]
T_CMB = 2.7255
OMEGA_GAMMA_H2 = 2.4728e-5 * (T_CMB / 2.7255) ** 4
RADIATION_NEFF_COEFF = 0.22710731766


# ---------------------------------------------------------------------------
# Neutrino density fraction
# ---------------------------------------------------------------------------
def Omega_nu(mnu, h):
    r"""Neutrino density parameter today for non-relativistic neutrinos.

    .. math::
        \Omega_\nu = \frac{m_\nu}{93.14\,h^2}\;\mathrm{eV}

    Parameters
    ----------
    mnu : float
        Sum of neutrino masses [eV].
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    Omega_nu : float
    """
    return mnu / (93.14 * h**2)


# ---------------------------------------------------------------------------
# Friedmann equation
# ---------------------------------------------------------------------------
def Esqr(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""E^2(z) = H^2(z) / H_0^2  for flat nuLCDM (w=-1).

    .. math::
        E^2(z) = \Omega_m (1+z)^3 + \Omega_\Lambda

    with :math:`\Omega_m = (\omega_b + \omega_{cdm})/ h^2 + \Omega_\nu`
    and  :math:`\Omega_\Lambda = 1 - \Omega_m` (flat universe).

    Parameters
    ----------
    omega_b : float
        Physical baryon density :math:`\omega_b = \Omega_b h^2`.
    omega_cdm : float
        Physical CDM density :math:`\omega_{cdm} = \Omega_{cdm} h^2`.
    h : float
        Dimensionless Hubble parameter.
    z : float or array
        Redshift.
    mnu : float
        Sum of neutrino masses [eV].  Default 0.06.

    Returns
    -------
    E2 : same shape as *z*
    """
    Omega_m = (omega_b + omega_cdm) / h**2 + Omega_nu(mnu, h)
    Omega_L = 1.0 - Omega_m
    return Omega_m * (1.0 + z) ** 3 + Omega_L


def Hz(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""Hubble parameter H(z) in km/s/Mpc (physical).

    .. math::
        H(z) = H_0\,E(z) = 100\,h\,E(z)

    Parameters
    ----------
    omega_b, omega_cdm, h, z, mnu : see :func:`Esqr`.

    Returns
    -------
    H : same shape as *z*, in km/s/Mpc.
    """
    H0 = 100.0 * h
    return H0 * jnp.sqrt(Esqr(omega_b, omega_cdm, h, z, mnu))


# ---------------------------------------------------------------------------
# Comoving distance
# ---------------------------------------------------------------------------
def dchioverdz(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""d\chi / dz  in Mpc (physical).

    .. math::
        \frac{d\chi}{dz} = \frac{c}{H(z)}
    """
    return C_KMS / Hz(omega_b, omega_cdm, h, z, mnu)


def chi(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""Radial comoving distance \chi(z) in Mpc (physical).

    .. math::
        \chi(z) = \int_0^z \frac{c\,dz'}{H(z')}

    Uses ``quadax.cumulative_simpson`` on the supplied *z* array.
    The first element of *z* **must** be 0 (or very close to it).

    Parameters
    ----------
    omega_b, omega_cdm, h : float
    z : 1-d array
        Redshift grid, starting at 0.
    mnu : float

    Returns
    -------
    chi : array, same shape as *z*, in Mpc.
    """
    integrand = dchioverdz(omega_b, omega_cdm, h, z, mnu)
    result = quadax.cumulative_simpson(integrand, x=z)
    # cumulative_simpson returns n-1 points; prepend chi(0) = 0
    return jnp.concatenate([jnp.zeros(1, dtype=result.dtype), result])


def chi_single(omega_b, omega_cdm, h, z_target, mnu=0.06, n_points=512):
    r"""Comoving distance to a single redshift, in Mpc (physical).

    Builds a temporary integration grid [0, z_target] with *n_points*
    linearly-spaced samples and calls :func:`chi`.

    Parameters
    ----------
    z_target : float
        Target redshift (scalar).
    n_points : int
        Number of integration points.

    Returns
    -------
    chi : float, in Mpc.
    """
    z_grid = jnp.linspace(0.0, z_target, n_points)
    chi_grid = chi(omega_b, omega_cdm, h, z_grid, mnu)
    return chi_grid[-1]


def _vectorize_redshift_fn(fn, z):
    """Evaluate a scalar-redshift function on scalar or array input."""
    z = jnp.asarray(z)
    if z.ndim == 0:
        return fn(z)
    z_flat = jnp.ravel(z)
    values = jax.vmap(fn)(z_flat)
    return jnp.reshape(values, z.shape)


def _omega_bc(omega_b, omega_cdm):
    return omega_b + omega_cdm


def _omega_radiation(omega_b, omega_cdm, h, neff=3.046):
    del omega_b, omega_cdm, h
    return OMEGA_GAMMA_H2 * (1.0 + RADIATION_NEFF_COEFF * neff)


def _Esqr_theta_star_a(omega_b, omega_cdm, h, a, neff=3.046):
    omega_m = _omega_bc(omega_b, omega_cdm)
    omega_r = _omega_radiation(omega_b, omega_cdm, h, neff=neff)
    Omega_m = omega_m / h**2
    Omega_r = omega_r / h**2
    Omega_L = 1.0 - Omega_m - Omega_r
    return Omega_r / a**4 + Omega_m / a**3 + Omega_L


def _Hz_theta_star_a(omega_b, omega_cdm, h, a, neff=3.046):
    return 100.0 * h * jnp.sqrt(_Esqr_theta_star_a(omega_b, omega_cdm, h, a, neff=neff))


def _sound_speed_baryon_photon_a(omega_b, a):
    r"""Sound speed of the tightly-coupled baryon-photon fluid."""
    R = 3.0 * omega_b * a / (4.0 * OMEGA_GAMMA_H2)
    return C_KMS / jnp.sqrt(3.0 * (1.0 + R))


# ---------------------------------------------------------------------------
# Angular diameter distance
# ---------------------------------------------------------------------------
def angular_diameter_distance(omega_b, omega_cdm, h, z, mnu=0.06,
                              n_points=512):
    r"""Angular diameter distance D_A(z) in Mpc (physical), for flat universe.

    .. math::
        D_A(z) = \frac{\chi(z)}{1 + z}

    Parameters
    ----------
    omega_b, omega_cdm, h : float
    z : float (scalar)
        Redshift.
    mnu : float
    n_points : int
        Integration grid size.

    Returns
    -------
    D_A : float, in Mpc.
    """
    return chi_single(omega_b, omega_cdm, h, z, mnu, n_points) / (1.0 + z)


def comoving_angular_diameter_distance(omega_b, omega_cdm, h, z, mnu=0.06,
                                       n_points=512):
    r"""Comoving angular diameter distance :math:`D_M(z)` in Mpc.

    In this flat-background module,

    .. math::
        D_M(z) = (1 + z) D_A(z) = \chi(z).
    """
    return _vectorize_redshift_fn(
        lambda zi: chi_single(omega_b, omega_cdm, h, zi, mnu, n_points),
        z,
    )


def hubble_distance(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""Hubble distance :math:`D_H(z) = c / H(z)` in Mpc."""
    return dchioverdz(omega_b, omega_cdm, h, z, mnu)


def volume_distance(omega_b, omega_cdm, h, z, mnu=0.06, n_points=512):
    r"""Volume-averaged BAO distance :math:`D_V(z)` in Mpc.

    .. math::
        D_V(z) = \left[z D_M^2(z) D_H(z)\right]^{1/3}
    """
    dm = comoving_angular_diameter_distance(
        omega_b, omega_cdm, h, z, mnu=mnu, n_points=n_points
    )
    dh = hubble_distance(omega_b, omega_cdm, h, z, mnu=mnu)
    return jnp.cbrt(jnp.asarray(z) * dm**2 * dh)


def recombination_redshift_hu_sugiyama(omega_b, omega_cdm, h, mnu=0.06,
                                       neff=3.046):
    r"""Approximate photon-decoupling redshift :math:`z_*`.

    Uses the Hu & Sugiyama (1996) expression quoted as Eq. (A3) in
    Aizpuru, Arjona & Nesseris (2021).

    Notes
    -----
    This approximation depends on :math:`\omega_m = \omega_b + \omega_{cdm}`
    and ignores ``mnu``/``neff`` beyond API compatibility.
    """
    del h, mnu, neff
    omega_m = _omega_bc(omega_b, omega_cdm)
    g1 = 0.0783 * omega_b**(-0.238) / (1.0 + 39.5 * omega_b**0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b**1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b**(-0.738)) * (1.0 + g1 * omega_m**g2)


def recombination_redshift_aizpuru2021(omega_b, omega_cdm, h, mnu=0.06,
                                       neff=3.046):
    r"""Approximate photon-decoupling redshift :math:`z_*`.

    Uses Eq. (A4) of Aizpuru, Arjona & Nesseris (2021).

    Notes
    -----
    This approximation depends on :math:`\omega_m = \omega_b + \omega_{cdm}`
    and ignores ``mnu``/``neff`` beyond API compatibility.
    """
    del h, mnu, neff
    omega_m = _omega_bc(omega_b, omega_cdm)
    return (
        (391.672 * omega_m**(-0.372296) + 937.422 * omega_b**(-0.97966))
        / (omega_m**(-0.0192951) * omega_b**(-0.93681))
        + omega_m**(-0.731631)
    )


def _recombination_redshift(omega_b, omega_cdm, h, mnu=0.06, neff=3.046,
                            fit="aizpuru2021"):
    if fit == "hu_sugiyama":
        return recombination_redshift_hu_sugiyama(
            omega_b, omega_cdm, h, mnu=mnu, neff=neff
        )
    if fit == "aizpuru2021":
        return recombination_redshift_aizpuru2021(
            omega_b, omega_cdm, h, mnu=mnu, neff=neff
        )
    raise ValueError(f"Unknown recombination-redshift fit '{fit}'.")


def sound_horizon(omega_b, omega_cdm, h, z, mnu=0.06, neff=3.046,
                  n_points=4096, a_min=1.0e-8):
    r"""Approximate sound horizon :math:`r_s(z)` in Mpc.

    Computes

    .. math::
        r_s(z) = \int_0^{a(z)} \frac{c_s(a')}{a'^2 H(a')} da'

    using a homogeneous background with baryons, CDM, photons, standard
    radiation and Lambda. This is designed for approximating
    :math:`r_s(z_*)` and therefore does not model the full massive-neutrino
    transition; ``mnu`` is accepted for API compatibility but not used.
    """
    del mnu

    def _sound_horizon_single(z_target):
        a_eval = 1.0 / (1.0 + z_target)
        a_grid = jnp.linspace(a_min, a_eval, n_points)
        cs_grid = _sound_speed_baryon_photon_a(omega_b, a_grid)
        h_grid = _Hz_theta_star_a(omega_b, omega_cdm, h, a_grid, neff=neff)
        integrand = cs_grid / (a_grid**2 * h_grid)
        integral = quadax.cumulative_simpson(integrand, x=a_grid)
        return jnp.concatenate([jnp.zeros(1, dtype=integral.dtype), integral])[-1]

    return _vectorize_redshift_fn(_sound_horizon_single, z)


def _comoving_distance_theta_star(omega_b, omega_cdm, h, z, neff=3.046,
                                  n_points=4096):
    def _chi_single(z_target):
        a_eval = 1.0 / (1.0 + z_target)
        a_grid = jnp.linspace(a_eval, 1.0, n_points)
        h_grid = _Hz_theta_star_a(omega_b, omega_cdm, h, a_grid, neff=neff)
        integrand = C_KMS / (a_grid**2 * h_grid)
        integral = quadax.cumulative_simpson(integrand, x=a_grid)
        return jnp.concatenate([jnp.zeros(1, dtype=integral.dtype), integral])[-1]

    return _vectorize_redshift_fn(_chi_single, z)


def theta_star(omega_b, omega_cdm, h, mnu=0.06, neff=3.046,
               z_star_fit="aizpuru2021", n_points=4096, a_min=1.0e-8):
    r"""Approximate CMB acoustic angular scale :math:`\theta_*`.

    Evaluates

    .. math::
        \theta_* = \frac{r_s(z_*)}{D_M(z_*)}

    with :math:`z_*` from a fitting formula and :math:`r_s(z_*)`,
    :math:`D_M(z_*)` from numerical integrals in a JAX-differentiable
    homogeneous background including photons and relativistic species.
    """
    z_star = _recombination_redshift(
        omega_b, omega_cdm, h, mnu=mnu, neff=neff, fit=z_star_fit
    )
    rs_star = sound_horizon(
        omega_b, omega_cdm, h, z_star, mnu=mnu, neff=neff,
        n_points=n_points, a_min=a_min
    )
    dm_star = _comoving_distance_theta_star(
        omega_b, omega_cdm, h, z_star, neff=neff, n_points=n_points
    )
    return rs_star / dm_star


def sound_horizon_drag_aubourg2014(omega_b, omega_cdm, h, mnu=0.06,
                                   neff=3.046):
    r"""Approximate drag-epoch sound horizon :math:`r_d` in Mpc.

    Uses Eq. (16) of Aubourg et al. (2015), valid for standard-neutrino
    nuLCDM-like models near the Planck-preferred parameter region.
    """
    del h, neff  # present for API consistency with the rest of the background module
    omega_cb = omega_b + omega_cdm
    omega_nu = mnu / 93.14
    return (
        55.154
        * jnp.exp(-72.3 * (omega_nu + 0.0006) ** 2)
        / (omega_cb**0.25351 * omega_b**0.12807)
    )


def sound_horizon_drag_aubourg2014_neff(omega_b, omega_cdm, h, mnu=0.06,
                                        neff=3.046):
    r"""Approximate drag-epoch sound horizon :math:`r_d` in Mpc.

    Uses Eq. (17) of Aubourg et al. (2015), extending the Eq. (16)
    approximation to varying neutrino mass and effective number of
    relativistic species.

    Notes
    -----
    The paper quotes this fit as accurate to 0.119% for
    :math:`0 < \sum m_\nu < 0.6\,\mathrm{eV}` and :math:`3 < N_\mathrm{eff} < 5`.
    """
    del h  # present for API consistency with the rest of the background module
    omega_cb = omega_b + omega_cdm
    omega_nu = mnu / 93.14
    return (
        56.067
        * jnp.exp(-49.7 * (omega_nu + 0.002) ** 2)
        / (
            omega_cb**0.2436
            * omega_b**0.128876
            * (1.0 + (neff - 3.046) / 30.60)
        )
    )


def sound_horizon_drag_brieden2022(omega_b, omega_cdm, h, mnu=0.06,
                                   neff=3.046):
    r"""Approximate LCDM drag-epoch sound horizon :math:`r_d` in Mpc.

    Uses Eq. (3.4) of Brieden, Gil-Marin & Verde (2022) with fixed
    :math:`N_\mathrm{eff} = 3.04`.

    Notes
    -----
    This fit is LCDM-specific and ignores ``mnu``.
    """
    del h, mnu, neff  # present for API consistency with the rest of the module
    omega_m = omega_b + omega_cdm
    return 147.05 * (omega_m / 0.1432) ** (-0.23) * (omega_b / 0.02236) ** (-0.13)


def sound_horizon_drag(omega_b, omega_cdm, h, mnu=0.06, neff=3.046,
                       fit="aubourg2014"):
    r"""Approximate drag-epoch sound horizon :math:`r_d = r_s(z_d)` in Mpc.

    By default this uses Eq. (16) of Aubourg et al. (2015), a neutrino-aware
    analytic fit for standard-neutrino nuLCDM-like models near the
    Planck-preferred parameter region.

    Notes
    -----
    Available ``fit`` backends are ``"aubourg2014"`` (default),
    ``"aubourg2014_neff"``, and ``"brieden2022"``.
    """
    if fit == "aubourg2014":
        return sound_horizon_drag_aubourg2014(
            omega_b, omega_cdm, h, mnu=mnu, neff=neff
        )
    if fit == "aubourg2014_neff":
        return sound_horizon_drag_aubourg2014_neff(
            omega_b, omega_cdm, h, mnu=mnu, neff=neff
        )
    if fit == "brieden2022":
        return sound_horizon_drag_brieden2022(
            omega_b, omega_cdm, h, mnu=mnu, neff=neff
        )
    raise ValueError(f"Unknown sound horizon fit '{fit}'.")


def dm_over_rs(omega_b, omega_cdm, h, z, mnu=0.06, neff=3.046, n_points=512,
               fit="aubourg2014"):
    r"""Dimensionless BAO observable :math:`D_M(z) / r_d`."""
    dm = comoving_angular_diameter_distance(
        omega_b, omega_cdm, h, z, mnu=mnu, n_points=n_points
    )
    return dm / sound_horizon_drag(
        omega_b, omega_cdm, h, mnu=mnu, neff=neff, fit=fit
    )


def dh_over_rs(omega_b, omega_cdm, h, z, mnu=0.06, neff=3.046,
               fit="aubourg2014"):
    r"""Dimensionless BAO observable :math:`D_H(z) / r_d`."""
    dh = hubble_distance(omega_b, omega_cdm, h, z, mnu=mnu)
    return dh / sound_horizon_drag(
        omega_b, omega_cdm, h, mnu=mnu, neff=neff, fit=fit
    )


def dv_over_rs(omega_b, omega_cdm, h, z, mnu=0.06, neff=3.046, n_points=512,
               fit="aubourg2014"):
    r"""Dimensionless BAO observable :math:`D_V(z) / r_d`."""
    dv = volume_distance(omega_b, omega_cdm, h, z, mnu=mnu, n_points=n_points)
    return dv / sound_horizon_drag(
        omega_b, omega_cdm, h, mnu=mnu, neff=neff, fit=fit
    )


# ---------------------------------------------------------------------------
# Derived quantities useful for the Fisher pipeline
# ---------------------------------------------------------------------------
def Omega_m_of_z(omega_b, omega_cdm, h, z, mnu=0.06):
    r"""Matter density parameter at redshift z.

    .. math::
        \Omega_m(z) = \frac{\Omega_m (1+z)^3}{E^2(z)}
    """
    Omega_m0 = (omega_b + omega_cdm) / h**2 + Omega_nu(mnu, h)
    return Omega_m0 * (1.0 + z) ** 3 / Esqr(omega_b, omega_cdm, h, z, mnu)


def _growth_factor_unnormalized_a(omega_b, omega_cdm, h, a, mnu=0.00,
                                  n_points=512, a_min=1.0e-4):
    r"""Unnormalized linear growth factor at scale factor ``a``.

    Uses the Heath integral for flat LCDM-like backgrounds,

    .. math::
        D(a) \propto E(a)\int_0^a \frac{da'}{a'^3 E(a')^3}.

    Notes
    -----
    This gives the standard scale-independent linear growth solution for
    a pressureless matter + Lambda background. If ``mnu > 0``, neutrinos
    only enter through the homogeneous background density, so this remains
    a background-level approximation rather than the full scale-dependent
    massive-neutrino growth.
    """
    a_eval = jnp.maximum(a, a_min)
    a_grid = jnp.linspace(a_min, a_eval, n_points)
    z_grid = 1.0 / a_grid - 1.0
    e_grid = jnp.sqrt(Esqr(omega_b, omega_cdm, h, z_grid, mnu))
    integrand = 1.0 / (a_grid**3 * e_grid**3)
    integral = quadax.cumulative_simpson(integrand, x=a_grid)
    integral_to_a = jnp.concatenate([jnp.zeros(1, dtype=integral.dtype), integral])[-1]
    z_eval = 1.0 / a_eval - 1.0
    e_eval = jnp.sqrt(Esqr(omega_b, omega_cdm, h, z_eval, mnu))
    omega_m0 = (omega_b + omega_cdm) / h**2 + Omega_nu(mnu, h)
    return 2.5 * omega_m0 * e_eval * integral_to_a


def growth_factor(omega_b, omega_cdm, h, z, mnu=0.00, n_points=512,
                  a_min=1.0e-4):
    r"""Linear growth factor :math:`D_+(z)` normalized to unity at ``z = 0``.

    Parameters
    ----------
    omega_b, omega_cdm, h, z, mnu : see :func:`Esqr`.
    n_points : int
        Number of scale-factor samples used in the growth integral.
    a_min : float
        Lower limit of the scale-factor integration.

    Returns
    -------
    Dplus : float or array
        Linear growth factor normalized such that ``Dplus(z=0) = 1``.
    """
    d_norm = _growth_factor_unnormalized_a(
        omega_b, omega_cdm, h, 1.0, mnu=mnu, n_points=n_points, a_min=a_min
    )

    def _single(z_single):
        a = 1.0 / (1.0 + z_single)
        return _growth_factor_unnormalized_a(
            omega_b, omega_cdm, h, a, mnu=mnu, n_points=n_points, a_min=a_min
        ) / d_norm

    z_arr = jnp.asarray(z)
    if z_arr.ndim == 0:
        return _single(z_arr)
    return jax.vmap(_single)(z_arr)


def growth_rate(omega_b, omega_cdm, h, z, mnu=0.00, n_points=512,
                a_min=1.0e-4):
    r"""Scale-independent linear growth rate :math:`f(z) = d\ln D / d\ln a`.

    Uses the same flat-LCDM Heath integral as :func:`growth_factor`.
    As with :func:`growth_factor`, ``mnu > 0`` only affects the background
    expansion, not the full scale-dependent neutrino growth.
    """
    omega_m0 = (omega_b + omega_cdm) / h**2 + Omega_nu(mnu, h)

    def _single(z_single):
        a = jnp.maximum(1.0 / (1.0 + z_single), a_min)
        z_eval = 1.0 / a - 1.0
        e_eval = jnp.sqrt(Esqr(omega_b, omega_cdm, h, z_eval, mnu))
        omega_m_a = Omega_m_of_z(omega_b, omega_cdm, h, z_eval, mnu)
        a_grid = jnp.linspace(a_min, a, n_points)
        z_grid = 1.0 / a_grid - 1.0
        e_grid = jnp.sqrt(Esqr(omega_b, omega_cdm, h, z_grid, mnu))
        integrand = 1.0 / (a_grid**3 * e_grid**3)
        integral = quadax.cumulative_simpson(integrand, x=a_grid)
        integral_to_a = jnp.concatenate([jnp.zeros(1, dtype=integral.dtype), integral])[-1]
        return -1.5 * omega_m_a + 1.0 / (a**2 * e_eval**3 * integral_to_a)

    z_arr = jnp.asarray(z)
    if z_arr.ndim == 0:
        return _single(z_arr)
    return jax.vmap(_single)(z_arr)


def growth_factor_approx(omega_b, omega_cdm, h, z, mnu=0.00, gamma=0.55,
                         n_points=512, a_min=1.0e-4):
    r"""Approximate linear growth factor from :math:`f(z)=\Omega_m(z)^\gamma`.

    The growth factor is normalized to unity at ``z = 0`` and obtained from

    .. math::
        \ln D(a) = -\int_a^1 f(a')\, d\ln a'.

    As with :func:`growth_rate_approx`, this is a scale-independent
    approximation.
    """

    def _single(z_single):
        a = jnp.maximum(1.0 / (1.0 + z_single), a_min)
        a_grid = jnp.linspace(a, 1.0, n_points)
        z_grid = 1.0 / a_grid - 1.0
        integrand = growth_rate_approx(
            omega_b, omega_cdm, h, z_grid, mnu=mnu, gamma=gamma
        ) / a_grid
        integral = quadax.cumulative_simpson(integrand, x=a_grid)
        return jnp.exp(-jnp.concatenate([jnp.zeros(1, dtype=integral.dtype), integral])[-1])

    z_arr = jnp.asarray(z)
    if z_arr.ndim == 0:
        return _single(z_arr)
    return jax.vmap(_single)(z_arr)


def growth_rate_approx(omega_b, omega_cdm, h, z, mnu=0.00, gamma=0.55):
    r"""Approximate linear growth rate f(z) ~ Omega_m(z)^gamma.

    Parameters
    ----------
    gamma : float, optional
        Growth rate index.  Default 0.55 (LCDM).
    mnu : float, optional
        Sum of neutrino masses [eV].  Default 0.00 (neutrinos do not contribute towards clustering and growth).

    Returns
    -------
    f : float
    """
    return Omega_m_of_z(omega_b, omega_cdm, h, z, mnu) ** gamma
