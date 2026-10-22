"""ELF EPT in the forecast's existing physical nuisance coordinates.

Leading counterterms contract common tracer fields (CT-A). The caller still
supplies the same linear power, growth, biases, observed densities and AP
distance ratios. Cross noise
is supplied separately by PairResponse and never inferred from auto noise.
"""
from __future__ import annotations

from functools import partial
from operator import index

import elf
import jax
import jax.numpy as jnp
import numpy as np
from elf.fftlog import HankelGrids

from .counterterm_fields import pair_counterterms


class _ImmutableConfiguration:
    """A compiled calculation keeps the configuration with which it was built."""

    def __setattr__(self, name, value):
        if getattr(self, "_configuration_frozen", False):
            raise AttributeError("ELF configuration is immutable; construct a new engine")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_configuration_frozen", False):
            raise AttributeError("ELF configuration is immutable; construct a new engine")
        object.__delattr__(self, name)

    def _freeze_configuration(self):
        for name, value in tuple(vars(self).items()):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            elif isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        object.__setattr__(self, "_configuration_frozen", True)


class _FixedHankelGrids(_ImmutableConfiguration, HankelGrids):
    def __init__(self, source):
        for name, value in vars(source).items():
            setattr(self, name, value)
        self._freeze_configuration()


class _FixedEPT(_ImmutableConfiguration, elf.EPT):
    """Unchanged ELF equations with construction-only settings and grids."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hankel = _FixedHankelGrids(self._hankel)
        self._freeze_configuration()


def parameters_to_elf(params, *, add_stochasticity=False):
    """Contract raw physical field coefficients, retaining the tied FoG shape."""
    cross = "bias2" in params
    if cross and add_stochasticity:
        raise ValueError("cross stochasticity must be supplied explicitly by the pair response")
    if ("ctr2" in params) != cross:
        raise ValueError("a cross spectrum requires both bias2 and ctr2")
    bias_a = jnp.asarray([params["bias"][key] for key in ("b1", "b2", "bG2", "bGamma3")])
    bias_b = (jnp.asarray([params["bias2"][key] for key in ("b1", "b2", "bG2", "bGamma3")])
              if cross else bias_a)
    ca = params["ctr"]
    cb = params["ctr2"] if cross else ca
    ctr = pair_counterterms(ca, cb, bias_a[0], bias_b[0], params["f"])
    if add_stochasticity:
        st = params["stoch"]
        nbar, knl = params["ndens"], params["k_nl"]
        stoch = jnp.asarray([st["P_shot"]/nbar, st["a0"]/(nbar*knl**2),
                             st["a2"]/(nbar*knl**2)])
    else:
        stoch = jnp.zeros(3, dtype=ctr.dtype)
    biases = {"bias_a": bias_a, "bias_b": bias_b} if cross else {"bias": bias_a}
    return elf.make_params(f=params["f"], h=params["h"], ctr=ctr, stoch=stoch, **biases)


class ELFPowerSpectrum(_ImmutableConfiguration):
    """IR-off Eulerian angular spectrum with the accepted nuisance model.

    The compatibility methods deliberately retain the old argument order so
    cosmology and covariance consumers do not need a second physical model.
    AP uses ELF's intrinsic cubic interpolation, never the old 2D AP table.
    Instances are configured once, then treated as static for differentiation.
    """

    native_full_mean_ap = True
    do_irres = False
    subtract_k0_limit = True

    def __init__(self, *, nfft=512, kmin_fft=1.e-5, kmax_fft=100.,
                 method="hankel", do_irres=False, subtract_k0_limit=True):
        if do_irres or not subtract_k0_limit:
            raise ValueError("ELF forecast requires IR off and the zero-lag constant subtracted")
        if isinstance(nfft, (bool, np.bool_)) or index(nfft) < 16 or nfft % 2:
            raise ValueError("nfft must be an even integer >= 16")
        if not (np.isfinite(kmin_fft) and np.isfinite(kmax_fft) and 0 < kmin_fft < kmax_fft):
            raise ValueError("loop support must be finite and positive with kmin < kmax")
        if method not in ("hankel", "matrix"):
            raise ValueError("ELF method must be hankel or matrix")
        if not jax.config.jax_enable_x64:
            raise RuntimeError("ELF forecast requires JAX float64")
        self._nfft, self._kmin, self._kmax = int(nfft), float(kmin_fft), float(kmax_fft)
        self.method = method
        # The matrix representation has a narrower documented query domain.
        self.query_bounds = ((max(self._kmin, .003), min(self._kmax, 3.))
                             if method == "matrix" else (self._kmin, self._kmax))
        self.model = _FixedEPT(nfft=self._nfft, kmin_fft=self._kmin, kmax_fft=self._kmax,
                             method=method, do_irres=False, counterterm_base="linear",
                             subtract_k0_const=True, ngauss=32)
        self._freeze_configuration()

    def numerical_settings(self):
        """Actual integration and interpolation choices, without old AP labels."""
        from dataclasses import asdict
        return {"engine": "elf.EPT", "method": self.model.method,
                "do_irres": self.model.do_irres,
                "counterterm_base": self.model.counterterm_base,
                "counterterm_prescription": "CT-A-shared-field-leading-v1",
                "subtract_k0_const": self.model.subtract_k0_const,
                "fft_n": self.model._nfft, "fft_kmin": self.model._kmin,
                "fft_kmax": self.model._kmax,
                "pad_mode": self.model.pad_mode,
                "fftlog_settings": asdict(self.model.fftlog_settings),
                "ap_method": "elf-full-mean-cubic-logk-v1",
                "ap_query_bounds": list(self.query_bounds),
                "ap_interpolation_nk": None, "ap_interpolation_nmu": None,
                "ap_refinement_nk": None, "multipole_default_nmu": 32}

    @partial(jax.jit, static_argnames=("self", "add_stochasticity"))
    def get_pkmu_pair_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params,
                         add_stochasticity=False):
        k = jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64))
        mu = jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64))
        data = jnp.stack((pk_data["k"], pk_data["pk"]))
        par = parameters_to_elf(params, add_stochasticity=add_stochasticity)
        power = self.model.get_pkmu(k, mu, data, par,
                                    alpha_perp=alpha_perp, alpha_para=alpha_para)
        # Support checking does not alter any valid Fourier coordinate.
        ktrue = k[:, None]*jnp.sqrt((1-mu[None, :]**2)/alpha_perp**2
                                    + mu[None, :]**2/alpha_para**2)
        valid = ((alpha_perp > 0) & (alpha_para > 0)
                 & jnp.isfinite(alpha_perp) & jnp.isfinite(alpha_para)
                 & (k[:, None] > 0) & (jnp.abs(mu[None, :]) <= 1)
                 & (ktrue >= self.query_bounds[0]) & (ktrue <= self.query_bounds[1]))
        return jnp.where(valid, power, jnp.nan)

    # Compatibility name only: this uses ELF's full-mean AP, not a 2D table.
    _get_pkmu_pair_ref_interp = get_pkmu_pair_ref

    def get_pkmu_pair(self, k, mu, pk_data, params, add_stochasticity=False):
        return self.get_pkmu_pair_ref(k, mu, 1., 1., pk_data, params,
                                     add_stochasticity=add_stochasticity)

    def get_pkmu(self, k, mu, pk_data, params):
        return self.get_pkmu_pair(k, mu, pk_data, params, add_stochasticity=True)

    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        return self.get_pkmu_pair_ref(k, mu, alpha_perp, alpha_para, pk_data, params,
                                     add_stochasticity=True)

    def get_pk_ell_pair_ref(self, k, l, alpha_perp, alpha_para, pk_data, params,
                           num=32, add_stochasticity=False):
        if isinstance(num, (bool, np.bool_)) or index(num) < 7:
            raise ValueError("multipole projection requires at least seven angular nodes")
        ell = jnp.asarray(l)
        if ell.ndim != 0 or not jnp.issubdtype(ell.dtype, jnp.integer):
            raise ValueError("multipole order must be an integer scalar")
        if not isinstance(l, jax.core.Tracer):
            order = index(l)
            if order not in (0, 2, 4, 6, 8):
                raise ValueError("supported multipoles are 0,2,4,6,8")
            if 2*num-1 < 8+order:
                raise ValueError(f"ell={order} requires at least {(8+order)//2+1} angular nodes")
        nodes, weights = np.polynomial.legendre.leggauss(num)
        mu = (nodes+1)/2
        orders = np.array([0, 2, 4, 6, 8])
        projections = np.stack([weights*np.polynomial.legendre.Legendre.basis(int(order))(mu)
                                 *(2*order+1)/2 for order in orders])
        # All five polynomials are fixed. Selection remains valid when ell is
        # a traced argument of an external compiled spectrum or derivative.
        projected_weights = jnp.sum(jnp.where(
            (ell == jnp.asarray(orders))[:, None], jnp.asarray(projections), 0.), axis=0)
        power = self.get_pkmu_pair_ref(k, jnp.asarray(mu), alpha_perp, alpha_para,
                                       pk_data, params, add_stochasticity=add_stochasticity)
        valid = (ell >= 0) & (ell <= 8) & (ell % 2 == 0) & (2*num-1 >= 8+ell)
        # At identity AP the highest intrinsic power is mu^8. This minimum
        # ensures polynomial exactness; distorted AP still needs convergence.
        return jnp.where(valid, jnp.sum(power*projected_weights[None, :], axis=1), jnp.nan)

    def get_pk_ell_pair(self, k, l, pk_data, params, num=32, add_stochasticity=False):
        return self.get_pk_ell_pair_ref(k, l, 1., 1., pk_data, params, num=num,
                                       add_stochasticity=add_stochasticity)

    def get_pk_ell(self, k, l, pk_data, params, num=32):
        return self.get_pk_ell_pair(k, l, pk_data, params, num=num, add_stochasticity=True)

    def get_pk_ell_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=32):
        return self.get_pk_ell_pair_ref(k, l, alpha_perp, alpha_para, pk_data, params,
                                       num=num, add_stochasticity=True)
