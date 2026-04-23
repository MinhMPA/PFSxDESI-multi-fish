"""Cosmological background and linear power spectrum.

Two-tier backend: cosmopower-jax (default) or clax (fallback) for P_lin
and sigma8. Background quantities (H, D, f, chi, D_A) always come from
ps_1loop_jax.background.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ps_1loop_jax import background as bg

# ---------------------------------------------------------------------------
# Fiducial cosmology — Planck 2018 TT,TE,EE+lowE best-fit
# ---------------------------------------------------------------------------

FIDUCIAL: dict[str, float] = {
    "h": 0.6736,
    "omega_b": 0.02237,       # Ωb h²
    "omega_cdm": 0.12,        # Ωcdm h²
    "n_s": 0.9649,
    "ln10_10_As": 3.0445224377,
    "mnu": 0.06,              # eV, single massive neutrino
}

# Derived
FIDUCIAL["Omega_m"] = (
    (FIDUCIAL["omega_b"] + FIDUCIAL["omega_cdm"] + FIDUCIAL["mnu"] / 93.14)
    / FIDUCIAL["h"] ** 2
)

# cosmopower-jax baryon-feedback nuisance (irrelevant for linear Pk,
# but the emulator requires them).  Use training-range midpoints.
_CP_BARYON_DEFAULTS = {"A_b": 3.13, "eta_b": 0.603, "logT_AGN": 7.8}

# Path to Jense 2024 ΛCDM+Mν networks
_JENSE_DIR = Path("/Users/nguyenmn/cosmopower-jax-for-pfs/cosmology/jense2024")
_JENSE_MODEL = "jense_2023_camb_mnu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

C_KMS = 299792.458  # km/s


def _omega_m(omega_b: float, omega_cdm: float, mnu: float) -> float:
    return omega_b + omega_cdm + mnu / 93.14


# ---------------------------------------------------------------------------
# cosmopower-jax emulator loaders (cached singletons)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pklin_emulator():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
    net = _JENSE_DIR / _JENSE_MODEL / "networks" / f"{_JENSE_MODEL}_Pk_lin.npz"
    return CPJ(probe="custom_log", filepath=str(net))


@lru_cache(maxsize=1)
def _load_sigma8_emulator():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
    net = _JENSE_DIR / _JENSE_MODEL / "networks" / f"{_JENSE_MODEL}_sigma8.npz"
    return CPJ(probe="custom_log", filepath=str(net))


# ---------------------------------------------------------------------------
# Pure-function factories for JAX-traceable P_lin and f(z)
# ---------------------------------------------------------------------------


def make_plin_func(backend: str = "cosmopower"):
    """Return a pure JAX function: pkdata_fn(z, cosmo_dict) → pk_data dict.

    The emulator is loaded once (outside the JAX trace); the returned
    closure is fully JAX-traceable w.r.t. all entries of cosmo_dict.
    Returns ``{"k": k_h, "pk": pk_h}`` on the emulator's native k-grid,
    matching the format expected by ``ps_1loop_jax.get_pk_ell()``.
    """
    if backend != "cosmopower":
        raise NotImplementedError(f"make_plin_func only supports cosmopower, got {backend}")

    emu = _load_pklin_emulator()
    k_modes = jnp.array(emu.modes)  # emulator's native k grid in 1/Mpc

    # Baryon-feedback nuisance values (fixed)
    Ab = _CP_BARYON_DEFAULTS["A_b"]
    eta = _CP_BARYON_DEFAULTS["eta_b"]
    logT = _CP_BARYON_DEFAULTS["logT_AGN"]

    def pkdata_fn(z, cosmo_dict):
        param_vec = jnp.array([[
            cosmo_dict["omega_b"],
            cosmo_dict["omega_cdm"],
            cosmo_dict["ln10_10_As"],
            cosmo_dict["n_s"],
            cosmo_dict["h"],
            z,
            Ab, eta, logT,
            cosmo_dict["mnu"],
        ]])
        pk_mpc3 = emu.predict(param_vec)
        h = cosmo_dict["h"]
        k_h = k_modes / h
        pk_h = jnp.atleast_1d(pk_mpc3.squeeze()) * h**3
        return {"k": k_h, "pk": pk_h}

    return pkdata_fn


def make_growth_rate_func():
    """Return a pure JAX function: f_fn(z, cosmo_dict) → f(z).

    Uses the exact Heath-integral growth rate from ps_1loop_jax.background,
    consistent with FiducialCosmology.f(). Fully JAX-differentiable.
    """
    from ps_1loop_jax import background as bg

    def f_fn(z, cosmo_dict):
        return bg.growth_rate(
            cosmo_dict["omega_b"],
            cosmo_dict["omega_cdm"],
            cosmo_dict["h"],
            z,
            cosmo_dict.get("mnu", 0.06),
        )

    return f_fn


# ---------------------------------------------------------------------------
# FiducialCosmology
# ---------------------------------------------------------------------------

class FiducialCosmology:
    """Cosmological quantities at a fixed set of parameters.

    Parameters
    ----------
    params : dict
        Cosmological parameters.  Keys: h, omega_b, omega_cdm, n_s,
        ln10_10_As, mnu.  Defaults to ``FIDUCIAL``.
    backend : str
        ``"cosmopower"`` (fast emulator) or ``"clax"`` (Boltzmann solver).
    """

    def __init__(
        self,
        params: dict[str, float] | None = None,
        backend: str = "cosmopower",
    ) -> None:
        self.params = dict(params or FIDUCIAL)
        self.backend = backend

        self._h = self.params["h"]
        self._omega_b = self.params["omega_b"]
        self._omega_cdm = self.params["omega_cdm"]
        self._mnu = self.params["mnu"]
        self._n_s = self.params["n_s"]
        self._ln10_10_As = self.params["ln10_10_As"]
        self._logA = self._ln10_10_As  # cosmopower-jax uses the same convention

        if backend == "clax":
            self._init_clax()

    # -- clax initialization ------------------------------------------------

    def _init_clax(self) -> None:
        import clax
        from ps_1loop_jax.clax_adapter import make_clax_pk_data, make_clax_background_data

        self._clax_params = clax.CosmoParams(
            h=self._h,
            omega_b=self._omega_b,
            omega_cdm=self._omega_cdm,
            n_s=self._n_s,
            ln10A_s=self._logA,
            m_ncdm=self._mnu,
        )
        self._make_clax_pk_data = make_clax_pk_data
        self._make_clax_bg_data = make_clax_background_data

    # -- background (always ps_1loop_jax.background) ------------------------

    def H(self, z: float) -> float:
        """Hubble parameter H(z) in km/s/Mpc."""
        return bg.Hz(self._omega_b, self._omega_cdm, self._h, z, self._mnu)

    def D_A(self, z: float) -> float:
        """Angular diameter distance in Mpc/h."""
        da_mpc = bg.angular_diameter_distance(
            self._omega_b, self._omega_cdm, self._h, z, self._mnu,
        )
        return da_mpc * self._h  # Mpc → Mpc/h

    def chi(self, z: float) -> float:
        """Comoving distance in Mpc/h."""
        chi_mpc = bg.chi_single(
            self._omega_b, self._omega_cdm, self._h, z, self._mnu,
        )
        return chi_mpc * self._h  # Mpc → Mpc/h

    def D(self, z: float) -> float:
        """Linear growth factor D(z), normalised to D(0)=1."""
        d_z = bg.growth_factor(
            self._omega_b, self._omega_cdm, self._h, z, self._mnu,
        )
        d_0 = bg.growth_factor(
            self._omega_b, self._omega_cdm, self._h, 0.0, self._mnu,
        )
        return d_z / d_0

    def f(self, z: float) -> float:
        """Logarithmic growth rate f(z) = d ln D / d ln a."""
        return bg.growth_rate(
            self._omega_b, self._omega_cdm, self._h, z, self._mnu,
        )

    # -- P_lin and sigma8 ---------------------------------------------------

    def sigma8(self, z: float) -> float:
        """Linear matter fluctuation amplitude σ8(z)."""
        if self.backend == "cosmopower":
            return self._sigma8_cosmopower(z)
        return self._sigma8_clax(z)

    def fsigma8(self, z: float) -> float:
        return float(self.f(z) * self.sigma8(z))

    def Plin(self, k: np.ndarray, z: float) -> jnp.ndarray:
        """Linear matter power spectrum P(k,z) in (Mpc/h)³ at k in h/Mpc."""
        if self.backend == "cosmopower":
            return self._plin_cosmopower(k, z)
        return self._plin_clax(k, z)

    def pk_data(self, z: float) -> dict[str, jnp.ndarray]:
        """Return ``{'k': ..., 'pk': ...}`` for ``PowerSpectrum1Loop``."""
        if self.backend == "cosmopower":
            return self._pk_data_cosmopower(z)
        return self._pk_data_clax(z)

    # -- cosmopower-jax internals -------------------------------------------

    def _cp_params_pklin(self, z: float) -> jnp.ndarray:
        """Parameter vector for the P_lin emulator (single sample)."""
        return jnp.array([[
            self._omega_b, self._omega_cdm, self._logA, self._n_s,
            self._h, z,
            _CP_BARYON_DEFAULTS["A_b"],
            _CP_BARYON_DEFAULTS["eta_b"],
            _CP_BARYON_DEFAULTS["logT_AGN"],
            self._mnu,
        ]])

    def _cp_params_sigma8(self) -> jnp.ndarray:
        """Parameter vector for the σ8 emulator (no z — z is the output)."""
        return jnp.array([[
            self._omega_b, self._omega_cdm, self._logA, self._n_s,
            self._h,
            _CP_BARYON_DEFAULTS["A_b"],
            _CP_BARYON_DEFAULTS["eta_b"],
            _CP_BARYON_DEFAULTS["logT_AGN"],
            self._mnu,
        ]])

    def _plin_cosmopower(self, k: np.ndarray, z: float) -> jnp.ndarray:
        """P_lin(k, z) via cosmopower-jax, returned in (Mpc/h)³ at k h/Mpc."""
        emu = _load_pklin_emulator()
        # Emulator outputs P(k) in Mpc³ on its own k grid (Mpc⁻¹).
        pk_mpc3 = emu.predict(self._cp_params_pklin(z))  # (1, 1000)
        k_mpc = jnp.array(emu.modes)  # Mpc⁻¹

        # Convert to h-units: k_h = k_mpc / h,  P_h = P_mpc * h³
        k_h = k_mpc / self._h
        pk_h = pk_mpc3 * self._h**3  # (1, 1000) or (1000,)
        pk_h = jnp.atleast_1d(pk_h.squeeze())

        # Interpolate onto requested k grid
        k = jnp.atleast_1d(jnp.asarray(k, dtype=float))
        return jnp.interp(k, k_h, pk_h)

    def _pk_data_cosmopower(self, z: float) -> dict[str, jnp.ndarray]:
        """Return pk_data dict on the emulator's native k grid."""
        emu = _load_pklin_emulator()
        pk_mpc3 = emu.predict(self._cp_params_pklin(z))
        k_mpc = jnp.array(emu.modes)
        k_h = k_mpc / self._h
        pk_h = jnp.atleast_1d(pk_mpc3.squeeze()) * self._h**3
        return {"k": k_h, "pk": pk_h}

    def _sigma8_cosmopower(self, z: float) -> float:
        """σ8(z) from the cosmopower-jax emulator."""
        emu = _load_sigma8_emulator()
        s8_all = emu.predict(self._cp_params_sigma8())  # (1000,) over z
        z_modes = jnp.array(emu.modes)
        s8_all = jnp.atleast_1d(s8_all.squeeze())
        return float(jnp.interp(z, z_modes, s8_all))

    # -- clax internals -----------------------------------------------------

    def _plin_clax(self, k: np.ndarray, z: float) -> jnp.ndarray:
        k = jnp.atleast_1d(jnp.asarray(k, dtype=float))
        pd = self._pk_data_clax(z)
        return jnp.interp(k, pd["k"], pd["pk"])

    def _pk_data_clax(self, z: float) -> dict[str, jnp.ndarray]:
        return self._make_clax_pk_data(self._clax_params, z)

    def _sigma8_clax(self, z: float) -> float:
        pd = self._pk_data_clax(z)
        k, pk = pd["k"], pd["pk"]
        R = 8.0  # Mpc/h
        x = k * R
        W = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        integrand = k**2 * pk * W**2 / (2.0 * jnp.pi**2)
        return float(jnp.sqrt(jnp.trapezoid(integrand, k)))

    # -- volume helpers (used by surveys.py) --------------------------------

    def comoving_volume_element(self, z: float) -> float:
        """dV/dz/dΩ in (Mpc/h)³/sr."""
        chi_val = self.chi(z)
        H_val = self.H(z)
        return float(C_KMS * chi_val**2 / (H_val * self._h))  # (Mpc/h)³/sr
