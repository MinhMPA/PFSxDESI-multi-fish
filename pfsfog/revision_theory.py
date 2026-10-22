"""Fixed physical coordinates for the revision's differentiable EFT mean.

The legacy Fisher adapter uses different counterterm coordinates. New revision
responses should build their backend leaves through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import Lock
from types import MappingProxyType
from weakref import WeakValueDictionary

from jax.core import eval_context
import jax.numpy as jnp
import numpy as np
from ps_1loop_jax import background as bg
from ps_1loop_jax import PowerSpectrum1Loop

from .cosmo import (
    _CP_BARYON_DEFAULTS, _load_sigma8_emulator,
    make_growth_rate_func, make_plin_func,
)
from .eft_params import NUISANCE_NAMES as _LEGACY_NUISANCE_NAMES
from .cosmology_space import (
    CosmologySpec, TRAINING_BOUNDS, cosmology_from_coordinates,
)
from .reference_units import make_reference_plin_func


COSMO_NAMES = ("ln10_10_As", "Mnu", "Omegam")
NUISANCE_NAMES = tuple(name for name in _LEGACY_NUISANCE_NAMES if name != "c1")


def cosmology_from_theta(theta, reference):
    """Map ``(ln(10^10 A_s), Mnu, Omega_m)`` at fixed h, omega_b and n_s.

    ``reference`` is a parameter dictionary or a ``FiducialCosmology``. Its
    values are copied into a new dictionary, so the AP and amplitude reference
    can remain frozen while ``theta`` changes.
    """
    source = reference.params if hasattr(reference, "params") else reference
    ln_as, mnu, omega_m = theta
    h = source["h"]
    omega_b = source["omega_b"]
    return {
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_m * h**2 - omega_b - mnu / 93.14,
        "n_s": source["n_s"],
        "ln10_10_As": ln_as,
        "mnu": mnu,
        "Omega_m": omega_m,
    }


def make_sigma8_func():
    """Return pure JAX ``sigma8(z, cosmo_dict)`` from the released network.

    The emulator and its fixed output redshift grid are captured outside the
    differentiated function. Its parameters and interpolation remain traceable.
    """
    emulator = _load_sigma8_emulator()
    z_modes = jnp.asarray(emulator.modes)
    ab = _CP_BARYON_DEFAULTS["A_b"]
    eta = _CP_BARYON_DEFAULTS["eta_b"]
    log_t = _CP_BARYON_DEFAULTS["logT_AGN"]

    def sigma8(z, cosmo_dict):
        params = jnp.array([[
            cosmo_dict["omega_b"],
            cosmo_dict["omega_cdm"],
            cosmo_dict["ln10_10_As"],
            cosmo_dict["n_s"],
            cosmo_dict["h"],
            ab, eta, log_t,
            cosmo_dict["mnu"],
        ]])
        values = jnp.atleast_1d(emulator.predict(params).squeeze())
        return jnp.interp(z, z_modes, values)

    return sigma8


def fiducial_nuisance(fid):
    """Extract the 11 active coordinates from an existing EFT fiducial."""
    return jnp.asarray([getattr(fid, name) for name in NUISANCE_NAMES])


def nuisance_to_backend(q, sigma8, sigma8_ref, f, h, nbar, k_nl):
    """Convert fixed nuisance coordinates into one auto backend dictionary.

    The external labels ``c0,c2,c4`` denote Q multipoles, ``c_tilde`` denotes
    T, and ``bGamma3`` denotes G. Their fiducial reference uses A=1, where
    A=(sigma8/sigma8_ref)^2. Stochastic coordinates remain normalized by the
    fixed observed number density passed as ``nbar``.
    """
    (
        b1_s8, b2_s8sq, bg2_s8sq, g,
        q0, q2, q4, t, pshot, a0_stoch, a2_stoch,
    ) = q
    amplitude = (sigma8 / sigma8_ref) ** 2
    c0, c2, c4 = q0 / amplitude, q2 / amplitude, q4 / amplitude
    return {
        "h": h,
        "f": f,
        "bias": {
            "b1": b1_s8 / sigma8,
            "b2": b2_s8sq / sigma8**2,
            "bG2": bg2_s8sq / sigma8**2,
            "bGamma3": g / amplitude,
        },
        "ctr": {
            "c0": c0 - f * c2 / 3 + 3 * f**2 * c4 / 35,
            "c2": c2 - 6 * f * c4 / 7,
            "c4": c4,
            "cfog": t / amplitude**2,
        },
        "stoch": {
            "P_shot": pshot,
            "a0": a0_stoch,
            "a2": a2_stoch,
        },
        "k_nl": k_nl,
        "ndens": nbar,
    }


def ap_coordinates(k, mu, alpha_perp, alpha_para):
    """Map fixed observed Fourier coordinates to true coordinates and volume.

    The returned ``k_true`` and ``mu_true`` have shape ``(Nk, Nmu)``.
    The Jacobian converts physical-volume mean power into observed units.
    """
    k = jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64))
    mu = jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64))
    fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
    mu_true = mu * (alpha_perp / alpha_para) / fac
    k_true = k[:, None] * fac[None, :] / alpha_perp
    jacobian = alpha_perp**2 * alpha_para
    return k_true, jnp.broadcast_to(mu_true[None, :], k_true.shape), jacobian


def _ap_grid_definition(n_k, n_mu):
    """Validate counts before lookup and describe the exact realized grids."""
    if isinstance(n_k, (bool, np.bool_)) or isinstance(n_mu, (bool, np.bool_)):
        raise TypeError("AP interpolation grid sizes must be integers")
    n_k, n_mu = index(n_k), index(n_mu)
    if n_k < 4 or n_mu < 4:
        raise ValueError("AP interpolation grids require at least four nodes")
    dtype = np.dtype(jnp.result_type(float)).str
    return (("geomspace", .002, .5, n_k, dtype),
            ("linspace", 0., 1., n_mu, dtype))


@dataclass(frozen=True, eq=False, repr=False)
class APInterpolationBackend:
    """Separate AP table; all loop evaluations use the original FFT backend.

    The inherited backend AP implementation is smooth at identity, but its
    default k table is also the coarse loop FFT grid. Refining this adapter
    must never alter that grid. Bounds cover the accepted .01--.20 observed
    range and its derivative neighborhoods; outside queries return NaN.
    """

    # Keep identity hashing for static JAX arguments, and support weak values
    # without allowing shared associations to be reassigned through __dict__.
    __slots__ = ("backend", "_kn", "_mu", "__weakref__")
    backend: object
    _kn: object
    _mu: object

    def __init__(self, backend, n_k=2048, n_mu=101):
        k_grid, mu_grid = _ap_grid_definition(n_k, n_mu)
        object.__setattr__(self, "backend", backend)
        # A cold public response may be constructed inside a JAX transform.
        # Shared static constants must outlive that trace as concrete arrays.
        # Use the normal eager grid kernels, without constant-folding changes
        # to their floating-point rounding.
        with eval_context():
            object.__setattr__(self, "_kn", jnp.geomspace(
                k_grid[1], k_grid[2], k_grid[3], dtype=np.dtype(k_grid[4])))
            object.__setattr__(self, "_mu", jnp.linspace(
                mu_grid[1], mu_grid[2], mu_grid[3], dtype=np.dtype(mu_grid[4])))

    def get_pkmu_pair(self, *args, **kwargs):
        return self.backend.get_pkmu_pair(*args, **kwargs)

    def _get_pkmu_pair_ref_interp(self, k, mu, alpha_perp, alpha_para,
                                  pk_data, params, add_stochasticity=False):
        value = PowerSpectrum1Loop._get_pkmu_pair_ref_interp(
            self, k, mu, alpha_perp, alpha_para, pk_data, params,
            add_stochasticity=add_stochasticity)
        kt, mt, _ = ap_coordinates(k, mu, alpha_perp, alpha_para)
        # Positive AP scales map every valid observed mu into [0,1]
        # analytically. At mu=1 XLA can round mu_true to 1+2e-16; testing
        # that rounded endpoint would reject a valid physical coordinate.
        # Validate the observed angular domain instead, without broadening it.
        observed_mu = jnp.atleast_1d(mu)[None, :]
        inside = ((kt >= self._kn[0]) & (kt <= self._kn[-1])
                  & (observed_mu >= 0) & (observed_mu <= 1)
                  & (alpha_perp > 0) & (alpha_para > 0) & jnp.isfinite(mt))
        return jnp.where(inside, value, jnp.nan)


_AP_BACKENDS = WeakValueDictionary()
_AP_BACKENDS_LOCK = Lock()


def _ap_interpolation_backend(backend, n_k, n_mu):
    """Reuse a live immutable adapter for an exact backend/grid context.

    Only the adapter's static constants are reused. Physical spectra and
    parameters remain dynamic inputs. Weak values add no registry ownership;
    the explicit association check also guards against stale identity tokens.
    Backends themselves must remain configured and frozen during a study.
    """
    grids = _ap_grid_definition(n_k, n_mu)
    key = (id(backend), grids)
    with _AP_BACKENDS_LOCK:
        adapter = _AP_BACKENDS.get(key)
        if adapter is None or adapter.backend is not backend:
            adapter = APInterpolationBackend(backend, grids[0][3], grids[1][3])
            _AP_BACKENDS[key] = adapter
        return adapter


class PairResponse:
    """Differentiable physical auto or cross mean with a frozen AP reference.

    An explicit space uses fixed h_ref Fourier and dimensional nuisance units,
    with physical-Mpc distance ratios for AP. Backend h behavior in that unit
    convention is audited only for do_irres=False. None retains baseline3.
    """

    def __init__(self, ps, z, reference, nbar_a, nbar_b, k_nl, ap_nk=2048, ap_nmu=101,
                 *, space: CosmologySpec | None = None):
        if space is not None:
            lower, upper = TRAINING_BOUNDS["redshift"]
            if not np.isfinite(z) or not lower <= z <= upper:
                raise ValueError(f"redshift outside [{lower}, {upper}]")
        self.space = space
        source = reference.params if hasattr(reference, "params") else reference
        self.reference = MappingProxyType({key: float(value) for key, value in source.items()})
        self.ps = ps
        # ELF already evaluates the full mean at true coordinates. The old
        # ps-only table remains available for historical diagnostics and toys.
        self.ap_backend = (ps if ap_nk is None or getattr(ps, "native_full_mean_ap", False)
                           else _ap_interpolation_backend(ps, ap_nk, ap_nmu))
        self.z = float(z)
        self.nbar_a = float(nbar_a)
        self.nbar_b = None if nbar_b is None else float(nbar_b)
        self.k_nl = float(k_nl)
        self.plin = (make_plin_func() if space is None
                     else make_reference_plin_func(space.h_ref))
        self.sigma8 = make_sigma8_func()
        self.growth_rate = make_growth_rate_func()
        ref = self.reference
        self.sigma8_ref = self.sigma8(self.z, ref)
        self.da_ref = bg.angular_diameter_distance(
            ref["omega_b"], ref["omega_cdm"], ref["h"], self.z, ref["mnu"])
        self.hubble_ref = bg.Hz(
            ref["omega_b"], ref["omega_cdm"], ref["h"], self.z, ref["mnu"])

    def _cosmology(self, theta):
        if self.space is None:
            return cosmology_from_theta(theta, self.reference)
        return cosmology_from_coordinates(theta, self.reference, self.space)

    def ap_alphas(self, theta):
        cosmo = self._cosmology(theta)
        da = bg.angular_diameter_distance(
            cosmo["omega_b"], cosmo["omega_cdm"], cosmo["h"], self.z, cosmo["mnu"])
        hubble = bg.Hz(
            cosmo["omega_b"], cosmo["omega_cdm"], cosmo["h"], self.z, cosmo["mnu"])
        return da / self.da_ref, self.hubble_ref / hubble

    def pkmu(self, k, mu, theta, q_a, q_b=None, q_cross=None, *, ap=True):
        k = jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64))
        # These plane-parallel RSD spectra are even in mu. The backend's
        # interpolation table covers [0, 1]; signed covariance quadrature
        # nodes must not be extrapolated beyond its angular domain.
        mu = jnp.abs(jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64)))
        cosmo = self._cosmology(theta)
        sigma8 = self.sigma8(self.z, cosmo)
        f = self.growth_rate(self.z, cosmo)
        params = nuisance_to_backend(
            q_a, sigma8, self.sigma8_ref, f, cosmo["h"], self.nbar_a, self.k_nl)
        is_cross = q_b is not None
        if is_cross:
            if self.nbar_b is None:
                raise ValueError("nbar_b is required for a cross response")
            second = nuisance_to_backend(
                q_b, sigma8, self.sigma8_ref, f, cosmo["h"], self.nbar_b, self.k_nl)
            params = {**params, "bias2": second["bias"], "ctr2": second["ctr"]}
        elif q_cross is not None:
            raise ValueError("q_cross requires q_b")

        alpha_perp, alpha_para = self.ap_alphas(theta) if ap else (1.0, 1.0)
        pk_data = self.plin(self.z, cosmo)
        result = self.ap_backend._get_pkmu_pair_ref_interp(
            k, mu, alpha_perp, alpha_para, pk_data, params,
            add_stochasticity=not is_cross)
        if is_cross and q_cross is not None:
            kt, mt, jacobian = ap_coordinates(k, mu, alpha_perp, alpha_para)
            cross_noise = (q_cross[0] + q_cross[1] * (kt * mt / self.k_nl)**2)
            result = result + cross_noise / (jnp.sqrt(self.nbar_a * self.nbar_b) * jacobian)
        return result

    def multipoles(self, k, theta, q_a, q_b=None, q_cross=None, *, ap=True, n_mu=32):
        if n_mu < 3:
            raise ValueError("n_mu must be at least 3")
        nodes, weights = np.polynomial.legendre.leggauss(n_mu)
        mu = jnp.asarray((nodes + 1) / 2, dtype=jnp.float64)
        weights = jnp.asarray(weights / 2, dtype=jnp.float64)
        pkmu = self.pkmu(k, mu, theta, q_a, q_b, q_cross, ap=ap)
        legendre = jnp.stack((
            jnp.ones_like(mu),
            (3 * mu**2 - 1) / 2,
            (35 * mu**4 - 30 * mu**2 + 3) / 8,
        ))
        return jnp.einsum("km,lm,m,l->kl", pkmu, legendre, weights,
                          jnp.asarray([1., 5., 9.], dtype=jnp.float64))


def make_pair_response(ps, z, reference, nbar_a, nbar_b=None, k_nl=0.7, *,
                       ap_nk=2048, ap_nmu=101, space: CosmologySpec | None = None):
    """Create an auto or cross response for one tracer pair and redshift."""
    return PairResponse(ps, z, reference, nbar_a, nbar_b, k_nl, ap_nk, ap_nmu,
                        space=space)
