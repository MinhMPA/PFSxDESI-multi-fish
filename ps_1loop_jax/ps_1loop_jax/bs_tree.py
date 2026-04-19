import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import quadax

from . import ir_resum
from . import pt_kernels
from . import utils_bispectrum
from .utils_loop import get_pk
from .utils_math import legendre


class BispectrumTree:
    """
    Tree-level galaxy bispectrum in redshift space.

    The public API mirrors the power-spectrum module:
    deterministic tree-level terms, stochasticity, optional IR resummation,
    optional Alcock-Paczynski remapping, and multipoles obtained by direct
    angular integration.
    """

    def __init__(
        self,
        do_irres=False,
        do_AP=False,
        rbao=110.0,
        ks=0.2,
        k_nl_rsd=0.3,
        kmin_fft=1e-5,
        kmax_fft=1e3,
        nfft=256,
    ):
        self.do_irres = do_irres
        self.do_AP = do_AP
        self.rbao = rbao
        self.ks = ks
        self.k_nl_rsd = k_nl_rsd
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft

    def _validate_triangle(self, k1, k2, k3):
        k1_np, k2_np, k3_np = np.broadcast_arrays(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            np.asarray(k3, dtype=float),
        )

        if np.any(k1_np <= 0) or np.any(k2_np <= 0) or np.any(k3_np <= 0):
            raise ValueError("All triangle side lengths must be strictly positive.")

        valid = (k1_np + k2_np >= k3_np) & (k1_np + k3_np >= k2_np) & (k2_np + k3_np >= k1_np)
        if not np.all(valid):
            raise ValueError("Input wavenumbers do not satisfy the triangle inequality.")

    def _get_bias(self, params):
        if "bias" not in params:
            raise KeyError("params['bias'] is required.")

        bias_in = params["bias"]
        if "b1" not in bias_in:
            raise KeyError("params['bias']['b1'] is required.")

        return {
            "b1": jnp.asarray(bias_in["b1"], dtype=float),
            "b2": jnp.asarray(bias_in.get("b2", 0.0), dtype=float),
            "bG2": jnp.asarray(bias_in.get("bG2", 0.0), dtype=float),
        }

    def _get_ctr(self, params):
        ctr_in = params.get("ctr", {})
        c1 = ctr_in.get("c1", params.get("c1", 0.0))
        return {"c1": jnp.asarray(c1, dtype=float)}

    def _get_stoch(self, params):
        stoch_in = params.get("stoch", {})

        if "P_shot" in stoch_in:
            pshot = stoch_in["P_shot"]
        elif "Pshot" in stoch_in:
            pshot = stoch_in["Pshot"]
        elif "P_shot" in params:
            pshot = params["P_shot"]
        else:
            pshot = params.get("Pshot", 0.0)

        if "B_shot" in stoch_in:
            bshot = stoch_in["B_shot"]
        elif "Bshot" in stoch_in:
            bshot = stoch_in["Bshot"]
        elif "B_shot" in params:
            bshot = params["B_shot"]
        else:
            bshot = params.get("Bshot", 0.0)

        if "A_shot" in stoch_in:
            ashot = stoch_in["A_shot"]
        elif "Ashot" in stoch_in:
            ashot = stoch_in["Ashot"]
        elif "A_shot" in params:
            ashot = params["A_shot"]
        else:
            ashot = params.get("Ashot", 0.0)

        stoch = {
            "P_shot": jnp.asarray(pshot, dtype=float),
            "B_shot": jnp.asarray(bshot, dtype=float),
            "A_shot": jnp.asarray(ashot, dtype=float),
        }
        return stoch

    def _get_ndens(self, params, stoch):
        has_explicit_stoch = (
            "stoch" in params
            or "P_shot" in params
            or "Pshot" in params
            or "B_shot" in params
            or "Bshot" in params
            or "A_shot" in params
            or "Ashot" in params
        )
        has_nonzero_stoch = any(np.any(np.asarray(stoch[name]) != 0.0) for name in stoch)

        if not has_explicit_stoch:
            return None
        if "ndens" in params:
            return jnp.asarray(params["ndens"], dtype=float)
        if has_nonzero_stoch:
            raise KeyError("params['ndens'] is required when bispectrum stochasticity is enabled.")
        return None

    def _get_pk_eval(self, k, pk_data):
        k = jnp.asarray(k, dtype=float)
        pk = get_pk(jnp.ravel(k), pk_data, kmin=self._kmin, kmax=self._kmax)
        return pk.reshape(k.shape)

    def _get_pk_nw_data(self, pk_data, params):
        if not self.do_irres:
            return None
        if "h" not in params:
            raise KeyError("params['h'] is required when do_irres=True.")
        return ir_resum.get_pk_nw_data(
            pk_data,
            params["h"],
            khmin=7e-5,
            khmax=7.0,
            kmin_interp=self._kmin,
            kmax_interp=self._kmax,
        )

    def _get_pk_tree_rsd(self, k, mu, pk_data, pk_nw_data, f):
        pk = self._get_pk_eval(k, pk_data)
        if pk_nw_data is None:
            return pk

        pk_nw = self._get_pk_eval(k, pk_nw_data)
        pk_w = pk - pk_nw

        sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        dsigma2 = ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
        sigma2_tot = (1.0 + mu**2 * f * (2.0 + f)) * sigma2 + f**2 * mu**2 * (mu**2 - 1.0) * dsigma2
        damp_fac = k**2 * sigma2_tot
        return pk_nw + jnp.exp(-damp_fac) * pk_w

    def _get_z1_fog(self, k, mu, bias, ctr, f):
        return pt_kernels.Z1(bias, f, mu) - ctr["c1"] * mu**2 * (k / self.k_nl_rsd) ** 2

    def _get_leg_factors(self, k1, k2, k3, mu1, mu2, mu3, pk_data, pk_nw_data, params):
        f = jnp.asarray(params["f"], dtype=float)
        bias = self._get_bias(params)
        ctr = self._get_ctr(params)

        pk1 = self._get_pk_tree_rsd(k1, mu1, pk_data, pk_nw_data, f)
        pk2 = self._get_pk_tree_rsd(k2, mu2, pk_data, pk_nw_data, f)
        pk3 = self._get_pk_tree_rsd(k3, mu3, pk_data, pk_nw_data, f)

        z1_1 = self._get_z1_fog(k1, mu1, bias, ctr, f)
        z1_2 = self._get_z1_fog(k2, mu2, bias, ctr, f)
        z1_3 = self._get_z1_fog(k3, mu3, bias, ctr, f)

        return pk1, pk2, pk3, z1_1, z1_2, z1_3

    def _get_bkmuphi_tree_spt_from_geometry(
        self,
        k1,
        k2,
        k3,
        mu1,
        mu2,
        mu3,
        mu12,
        mu13,
        mu23,
        pk_data,
        pk_nw_data,
        params,
    ):
        f = jnp.asarray(params["f"], dtype=float)
        bias = self._get_bias(params)

        pk1, pk2, pk3, z1_1, z1_2, z1_3 = self._get_leg_factors(
            k1, k2, k3, mu1, mu2, mu3, pk_data, pk_nw_data, params
        )

        z2_12 = pt_kernels.Z2(k1, k2, bias, f, mu1, mu2, mu12)
        z2_13 = pt_kernels.Z2(k1, k3, bias, f, mu1, mu3, mu13)
        z2_23 = pt_kernels.Z2(k2, k3, bias, f, mu2, mu3, mu23)

        return 2.0 * (
            z1_1 * z1_2 * z2_12 * pk1 * pk2
            + z1_1 * z1_3 * z2_13 * pk1 * pk3
            + z1_2 * z1_3 * z2_23 * pk2 * pk3
        )

    def _get_bkmuphi_stoch_from_geometry(
        self,
        k1,
        k2,
        k3,
        mu1,
        mu2,
        mu3,
        pk_data,
        pk_nw_data,
        params,
    ):
        stoch = self._get_stoch(params)
        ndens = self._get_ndens(params, stoch)
        if ndens is None:
            return 0.0

        f = jnp.asarray(params["f"], dtype=float)
        bias = self._get_bias(params)
        beta = f / bias["b1"]
        pshot = stoch["P_shot"]
        bshot = stoch["B_shot"]
        ashot = stoch["A_shot"]

        pk1 = bias["b1"] ** 2 * self._get_pk_eval(k1, pk_data)
        pk2 = bias["b1"] ** 2 * self._get_pk_eval(k2, pk_data)
        pk3 = bias["b1"] ** 2 * self._get_pk_eval(k3, pk_data)

        bk_linear = (
            pk1 * (1.0 + beta * mu1**2) * (bshot + pshot * beta * mu1**2)
            + pk2 * (1.0 + beta * mu2**2) * (bshot + pshot * beta * mu2**2)
            + pk3 * (1.0 + beta * mu3**2) * (bshot + pshot * beta * mu3**2)
        ) / ndens

        return bk_linear + ashot / ndens**2

    def _get_bkmuphi_tree_from_geometry(
        self,
        k1,
        k2,
        k3,
        mu1,
        mu2,
        mu3,
        mu12,
        mu13,
        mu23,
        pk_data,
        pk_nw_data,
        params,
    ):
        bk_tree = self._get_bkmuphi_tree_spt_from_geometry(
            k1,
            k2,
            k3,
            mu1,
            mu2,
            mu3,
            mu12,
            mu13,
            mu23,
            pk_data,
            pk_nw_data,
            params,
        )
        bk_stoch = self._get_bkmuphi_stoch_from_geometry(
            k1,
            k2,
            k3,
            mu1,
            mu2,
            mu3,
            pk_data,
            pk_nw_data,
            params,
        )
        return bk_tree + bk_stoch

    def _get_triangle_geometry(self, k1, k2, k3, mu1, phi):
        mu12 = utils_bispectrum.get_muij(k1, k2, k3)
        mu13 = utils_bispectrum.get_muij(k1, k3, k2)
        mu23 = utils_bispectrum.get_muij(k2, k3, k1)
        mu2, mu3 = utils_bispectrum.get_mu2_mu3(k1, k2, k3, mu1, phi)
        return mu2, mu3, mu12, mu13, mu23

    def get_bk_tree_spt(self, k1, k2, k3, mu1, phi, pk_data, params):
        k1 = jnp.asarray(k1, dtype=float)
        k2 = jnp.asarray(k2, dtype=float)
        k3 = jnp.asarray(k3, dtype=float)
        mu1 = jnp.asarray(mu1, dtype=float)
        phi = jnp.asarray(phi, dtype=float)

        self._validate_triangle(k1, k2, k3)
        pk_nw_data = self._get_pk_nw_data(pk_data, params)
        mu2, mu3, mu12, mu13, mu23 = self._get_triangle_geometry(k1, k2, k3, mu1, phi)

        return self._get_bkmuphi_tree_spt_from_geometry(
            k1, k2, k3, mu1, mu2, mu3, mu12, mu13, mu23, pk_data, pk_nw_data, params
        )

    def get_bkmuphi_stoch(self, k1, k2, k3, mu1, phi, pk_data, params):
        k1 = jnp.asarray(k1, dtype=float)
        k2 = jnp.asarray(k2, dtype=float)
        k3 = jnp.asarray(k3, dtype=float)
        mu1 = jnp.asarray(mu1, dtype=float)
        phi = jnp.asarray(phi, dtype=float)

        self._validate_triangle(k1, k2, k3)
        pk_nw_data = self._get_pk_nw_data(pk_data, params)
        mu2, mu3, _, _, _ = self._get_triangle_geometry(k1, k2, k3, mu1, phi)

        return self._get_bkmuphi_stoch_from_geometry(
            k1, k2, k3, mu1, mu2, mu3, pk_data, pk_nw_data, params
        )

    def get_bk_tree(self, k1, k2, k3, mu1, phi, pk_data, params):
        k1 = jnp.asarray(k1, dtype=float)
        k2 = jnp.asarray(k2, dtype=float)
        k3 = jnp.asarray(k3, dtype=float)
        mu1 = jnp.asarray(mu1, dtype=float)
        phi = jnp.asarray(phi, dtype=float)

        self._validate_triangle(k1, k2, k3)
        pk_nw_data = self._get_pk_nw_data(pk_data, params)
        mu2, mu3, mu12, mu13, mu23 = self._get_triangle_geometry(k1, k2, k3, mu1, phi)

        return self._get_bkmuphi_tree_from_geometry(
            k1, k2, k3, mu1, mu2, mu3, mu12, mu13, mu23, pk_data, pk_nw_data, params
        )

    def get_bk_tree_ref(
        self,
        k1,
        k2,
        k3,
        mu1,
        phi,
        alpha_perp,
        alpha_para,
        pk_data,
        params,
    ):
        k1 = jnp.asarray(k1, dtype=float)
        k2 = jnp.asarray(k2, dtype=float)
        k3 = jnp.asarray(k3, dtype=float)
        mu1 = jnp.asarray(mu1, dtype=float)
        phi = jnp.asarray(phi, dtype=float)

        self._validate_triangle(k1, k2, k3)
        pk_nw_data = self._get_pk_nw_data(pk_data, params)

        (
            k1_true,
            k2_true,
            k3_true,
            mu1_true,
            mu2_true,
            mu3_true,
            mu12_true,
            mu13_true,
            mu23_true,
        ) = utils_bispectrum.get_ap_ref_triangle(
            k1, k2, k3, mu1, phi, alpha_perp, alpha_para
        )

        bk_true = self._get_bkmuphi_tree_from_geometry(
            k1_true,
            k2_true,
            k3_true,
            mu1_true,
            mu2_true,
            mu3_true,
            mu12_true,
            mu13_true,
            mu23_true,
            pk_data,
            pk_nw_data,
            params,
        )

        return bk_true / (alpha_perp**4 * alpha_para**2)

    def get_bk(self, k1, k2, k3, mu1, phi, pk_data, params, alpha_perp=None, alpha_para=None):
        if self.do_AP or (alpha_perp is not None and alpha_para is not None):
            if alpha_perp is None or alpha_para is None:
                raise ValueError("Both alpha_perp and alpha_para are required for AP remapping.")
            return self.get_bk_tree_ref(
                k1, k2, k3, mu1, phi, alpha_perp, alpha_para, pk_data, params
            )
        return self.get_bk_tree(k1, k2, k3, mu1, phi, pk_data, params)

    def get_bk_ell(self, k1, k2, k3, ell, pk_data, params, num_mu=65, num_phi=65):
        mu1 = jnp.linspace(-1.0, 1.0, num_mu)
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, num_phi)

        bk = self.get_bk_tree(
            jnp.asarray(k1, dtype=float)[..., None, None],
            jnp.asarray(k2, dtype=float)[..., None, None],
            jnp.asarray(k3, dtype=float)[..., None, None],
            mu1[None, :, None],
            phi[None, None, :],
            pk_data,
            params,
        )

        bk_phi = quadax.simpson(bk, x=phi, axis=-1) / (2.0 * jnp.pi)
        return 0.5 * (2 * ell + 1) * quadax.simpson(bk_phi * legendre(ell, mu1), x=mu1, axis=-1)

    def get_bk_ell_ref(
        self,
        k1,
        k2,
        k3,
        ell,
        alpha_perp,
        alpha_para,
        pk_data,
        params,
        num_mu=65,
        num_phi=65,
    ):
        mu1 = jnp.linspace(-1.0, 1.0, num_mu)
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, num_phi)

        bk = self.get_bk_tree_ref(
            jnp.asarray(k1, dtype=float)[..., None, None],
            jnp.asarray(k2, dtype=float)[..., None, None],
            jnp.asarray(k3, dtype=float)[..., None, None],
            mu1[None, :, None],
            phi[None, None, :],
            alpha_perp,
            alpha_para,
            pk_data,
            params,
        )

        bk_phi = quadax.simpson(bk, x=phi, axis=-1) / (2.0 * jnp.pi)
        return 0.5 * (2 * ell + 1) * quadax.simpson(bk_phi * legendre(ell, mu1), x=mu1, axis=-1)

    def get_bk0_tree(self, k1, k2, k3, pk_data, params, num_mu=65, num_phi=65):
        return self.get_bk_ell(k1, k2, k3, 0, pk_data, params, num_mu=num_mu, num_phi=num_phi)

    def get_bk0_tree_ref(
        self,
        k1,
        k2,
        k3,
        alpha_perp,
        alpha_para,
        pk_data,
        params,
        num_mu=65,
        num_phi=65,
    ):
        return self.get_bk_ell_ref(
            k1,
            k2,
            k3,
            0,
            alpha_perp,
            alpha_para,
            pk_data,
            params,
            num_mu=num_mu,
            num_phi=num_phi,
        )

    def get_bk0(
        self,
        k1,
        k2,
        k3,
        pk_data,
        params,
        alpha_perp=None,
        alpha_para=None,
        num_mu=65,
        num_phi=65,
    ):
        if self.do_AP or (alpha_perp is not None and alpha_para is not None):
            if alpha_perp is None or alpha_para is None:
                raise ValueError("Both alpha_perp and alpha_para are required for AP remapping.")
            return self.get_bk0_tree_ref(
                k1,
                k2,
                k3,
                alpha_perp,
                alpha_para,
                pk_data,
                params,
                num_mu=num_mu,
                num_phi=num_phi,
            )
        return self.get_bk0_tree(
            k1,
            k2,
            k3,
            pk_data,
            params,
            num_mu=num_mu,
            num_phi=num_phi,
        )

    # Backward-compatible aliases for the placeholder names used in the draft.
    get_bkmuphi_tree = get_bk_tree
    get_bkmuphi = get_bk
