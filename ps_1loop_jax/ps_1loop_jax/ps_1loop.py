import os
import glob, re

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import quadax
import interpax

from .power_law_decomp import PowerLawDecomp
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int
from .utils_math import legendre
from . import ir_resum


class PowerSpectrum1Loop:

    def __init__(self, 
                 do_irres=True,
                 rbao=110.,
                 ks=0.2,
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 ):
        
        # set up the FFTLog-based power-law decomposition
        self.config_fft = {
            'pk_lin nu=-0.3': {'nu':-0.3, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_lin nu=-0.7': {'nu':-0.7, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_lin nu=-1.6': {'nu':-1.6, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
        }
        self._set_power_law_decomp(self.config_fft)
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._kn = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        # store the names of 1-loop terms calculated with the FFTLog-based method
        self.name_pk_terms = ['22_dd','13_dd','I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        self.name_pkmu_terms = {}
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M22_*.txt')
        self.name_pkmu_terms['22'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M13_*.txt')
        self.name_pkmu_terms['13'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_terms['tot'] = self.name_pkmu_terms['22'] + self.name_pkmu_terms['13']

        self.mat = {}
        self.matrix = {}
        # precompute the PT matrices
        self._set_matrix(self.name_pk_terms + self.name_pkmu_terms['tot'])

        self.do_irres = do_irres # flag to perform the IR resummation
        self.rbao = rbao
        self.ks = ks
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

    def _is_pair_spectrum(self, params):
        return self.cross or ('bias2' in params) or ('ctr2' in params)

    def _get_bias_pair(self, params):
        bias1 = params['bias']
        if self._is_pair_spectrum(params):
            bias2 = params['bias2']
        else:
            bias2 = bias1
        return bias1, bias2

    def _get_ctr_pair(self, params):
        ctr1 = params['ctr']
        if self._is_pair_spectrum(params):
            ctr2 = params['ctr2']
        else:
            ctr2 = ctr1
        return ctr1, ctr2

    def _make_pair_params(self, params, bias2, ctr2):
        pair_params = dict(params)
        pair_params['bias2'] = bias2
        pair_params['ctr2'] = ctr2
        return pair_params

    def _get_matter_bias(self, params):
        b1 = jnp.asarray(params['bias']['b1'], dtype=float)
        zero = jnp.zeros_like(b1)
        return {
            'b1': jnp.ones_like(b1),
            'b2': zero,
            'bG2': zero,
            'bGamma3': zero,
        }

    def _get_matter_ctr(self, params):
        c0 = jnp.asarray(params['ctr']['c0'], dtype=float)
        zero = jnp.zeros_like(c0)
        return {
            'c0': zero,
            'c2': zero,
            'c4': zero,
            'cfog': zero,
        }

    def _get_gm_pair_params(self, params):
        return self._make_pair_params(
            params,
            self._get_matter_bias(params),
            self._get_matter_ctr(params),
        )

    def _ap_is_identity(self, alpha_perp, alpha_para):
        try:
            return (float(alpha_perp) == 1.0) and (float(alpha_para) == 1.0)
        except Exception:
            return False

    def _set_power_law_decomp(self, config_fft):
        # set multiple instances of PowerLawDecomp class.
        self.decomp = {}
        for name, config in config_fft.items():
            nu = config['nu']
            kmin = config['kmin']
            kmax = config['kmax']
            nfft = config['nfft']
            self.decomp[name] = PowerLawDecomp(nu, kmin, kmax, nfft)

    def _set_matrix(self, names=[]):
        for name in names:
            matfile = glob.glob(os.path.dirname(__file__)+'/pt_matrix/*/*/%s.txt' % (name))[0]
            if '22' in name or 'I' in name or '12' in name:
                self.mat[name] = pt_matrix.PTMatrix22(matfile)
            elif '13' in name or 'F' in name:
                self.mat[name] = pt_matrix.PTMatrix13(matfile)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))
        
        # precompute the PT matrices for appropriate FFT settings.
        for name in names:
            if name in utils_loop.kernel_to_decomp_dict.keys():
                name_dec = utils_loop.kernel_to_decomp_dict[name]
            else:
                species_list = list(self.name_pkmu_terms.keys())
                species_list.remove('tot')
                for species in species_list:
                    if name in self.name_pkmu_terms[species]:
                        name_dec = utils_loop.kernel_to_decomp_dict[species]
                        break

            if '22' in name or 'I' in name or '12' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                nu_m2 = -0.5 * self.decomp[name_dec[1]].nu_m
                nu_m1, nu_m2 = jnp.meshgrid(nu_m1, nu_m2)
                self.matrix[name] = self.mat[name](nu_m1, nu_m2).T
            elif '13' in name or 'F' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                self.matrix[name] = self.mat[name](nu_m1)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))
    
    @partial(jit, static_argnames=['self'])
    def get_pk_dict(self, pk_data):
        pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)

        pk_dict = {}
        pk_dict['tree'] = pk_lin
        
        name_list = ['22_dd','13_dd']
        p_q, p_k, p_k0 = self.decomp['pk_lin nu=-0.3'].get_decomposed_data(pk_lin)

        for name in name_list:

            if '22' in name or 'I' in name:
                pk = self._kn**3 * jnp.diag(jnp.dot(p_q.T, jnp.dot(self.matrix[name], p_q)).real)
                if self.subtract_k0_limit:
                    pk_k0 = self._kmin**3 * jnp.dot(p_k0, jnp.dot(self.matrix[name], p_k0)).real
                    pk = pk - pk_k0

            elif '13' in name or 'F' in name:
                pk = self._kn**3 * p_k * jnp.dot(self.matrix[name], p_q).real
                if name == '13_dd':
                    pk_lin_int = get_pk_int(pk_data)
                    pk = pk - (61. / 315.) * self._kn**2 * p_k * pk_lin_int

            pk_dict[name] = pk

        name_list = ['I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        p_q, p_k, p_k0 = self.decomp['pk_lin nu=-1.6'].get_decomposed_data(pk_lin)

        for name in name_list:

            if '22' in name or 'I' in name:
                pk = self._kn**3 * jnp.diag(jnp.dot(p_q.T, jnp.dot(self.matrix[name], p_q)).real)
                if self.subtract_k0_limit:
                    pk_k0 = self._kmin**3 * jnp.dot(p_k0, jnp.dot(self.matrix[name], p_k0)).real
                    pk = pk - pk_k0

            elif '13' in name or 'F' in name:
                pk = self._kn**3 * p_k * jnp.dot(self.matrix[name], p_q).real
                if name == '13_dd':
                    pk_lin_int = get_pk_int(pk_data)
                    pk = pk - (61. / 315.) * self._kn**2 * p_k * pk_lin_int

            pk_dict[name] = pk

        return pk_dict

    @partial(jit, static_argnames=['self'])
    def get_pk_real(self, k, pk_data, params): # no IR resummation
        k = jnp.atleast_1d(k).astype(float)

        pk_dict = self.get_pk_dict(pk_data)

        bias = params['bias']
        ctr = params['ctr']
        stoch = params['stoch']
        k_nl = params['k_nl']
        ndens = params['ndens']

        pk_tree = bias['b1']**2 * pk_dict['tree']

        pk_1loop = bias['b1']**2 * (pk_dict['22_dd'] + pk_dict['13_dd']) \
            + bias['b1'] * bias['b2'] * pk_dict['I_d2'] \
            + 2 * bias['b1'] * bias['bG2'] * pk_dict['I_G2'] \
            + bias['b2']**2 / 4 * pk_dict['I_d2_d2'] \
            + bias['bG2']**2 * pk_dict['I_G2_G2'] \
            + bias['b2'] * bias['bG2'] * pk_dict['I_d2_G2'] \
            + 2 * bias['b1'] * bias['bG2'] * pk_dict['F_G2'] \
            + (4 / 5) * bias['b1'] * bias['bGamma3'] * pk_dict['F_G2']

        pk_ctr = - 2 * self._kn**2 * ctr['c0'] * pk_dict['tree']

        pk = interpax.interp1d(k, self._kn, pk_tree + pk_1loop + pk_ctr, method='cubic')
        pk_stoch = (stoch['P_shot'] + stoch['a0'] * (k / k_nl)**2) / ndens

        pk = pk + pk_stoch

        return pk

    @partial(jit, static_argnames=['self'])
    def get_pk_gm_real(self, k, pk_data, params):
        """Return real-space galaxy-matter cross-power spectrum with 1-loop corrections.

        This implements the galaxy-matter cross-spectrum following Eq. (2.12) of
        Chudaykin et al. (2020), arXiv:2004.10607.
        The cross-spectrum is linear in b1 (not quadratic like the galaxy auto-spectrum).

        Parameters
        ----------
        k : array_like
            Wavenumbers in units of h/Mpc
        pk_data : dict
            Dictionary containing linear power spectrum data with keys 'k' and 'pk'
        params : dict
            Dictionary containing cosmological and bias parameters:
            - bias: dict with 'b1', 'b2', 'bG2', 'bGamma3'
            - ctr: dict with counter-term parameters:
                * 'c0': EFT parameter (=c_s^2, sound speed squared) in (Mpc/h)^2
                * 'cs0': stochastic parameter (=R_*^2) in (Mpc/h)^2
                Note: 'c0' can be shared between P_gm (real) and P_gg (RSD monopole).
                Both represent c_s^2 and directly multiply k^2*P_lin.

        Returns
        -------
        pk_gm : array_like
            Galaxy-matter cross-power spectrum at wavenumbers k

        Notes
        -----
        The formula (Eq. 2.12 of arXiv:2004.10607) is:
        P_gm = b1*(P_lin + P_1loop_SPT) + (b2/2)*I_δ²
             + (b_G2 + 2/5*b_Γ3)*F_G2 + b_G2*I_G2
             - (cs0 + 2*c0*b1)*k²*P_lin

        Parameter definitions:
        - c0 = c_s² (sound speed squared), units: (Mpc/h)²
        - cs0 = R_*² (stochastic parameter), units: (Mpc/h)²

        For joint analyses, c0 can be shared with the RSD monopole
        counter-term as both represent the same physical quantity c_s².
        """
        k = jnp.atleast_1d(k).astype(float)

        pk_dict = self.get_pk_dict(pk_data)

        bias = params['bias']
        ctr = params['ctr']

        # Tree level: b1 * P_lin
        pk_tree = bias['b1'] * pk_dict['tree']

        # 1-loop matter: b1 * (P_22 + P_13) - linear in matter 1-loop SPT terms
        pk_1loop_matter = bias['b1'] * (pk_dict['22_dd'] + pk_dict['13_dd'])

        # 1-loop bias terms: second-order bias contributions
        # Note: (2/5) = 0.4
        pk_1loop_bias = (bias['b2'] / 2) * pk_dict['I_d2'] \
                      + bias['bG2'] * pk_dict['I_G2'] \
                      + (bias['bG2'] + (2./5.) * bias['bGamma3']) * pk_dict['F_G2']

        # Counter-term and stochastic term: -(cs0 + 2*c0*b1) * k² * P_lin
        # Note: c0 = c_s² (sound speed squared), cs0 = R_*² (stochastic parameter)
        # For joint analyses with P_gg multipoles:
        #   - c0 is the same parameter as RSD monopole c0 (both represent c_s²)
        #   - cs0 is typically independent (or set to 0) for galaxy-matter
        c0 = ctr.get('c0', 0.)
        cs0 = ctr.get('cs0', 0.)

        pk_ctr = - (cs0 + 2 * c0 * bias['b1']) * self._kn**2 * pk_dict['tree']

        # Total power spectrum on internal grid
        pk_total = pk_tree + pk_1loop_matter + pk_1loop_bias + pk_ctr

        # Interpolate to requested k values
        pk = interpax.interp1d(k, self._kn, pk_total, method='cubic')

        return pk

    @partial(jit, static_argnames=['self'])
    def get_pk_nw(self, k, pk_data, params):
        """Return de-wiggled linear matter power spectrum P_nw(k)."""
        k = jnp.atleast_1d(k).astype(float)

        h = params['h']
        pk_nw_data = ir_resum.get_pk_nw_data(
            pk_data,
            h,
            khmin=7e-5,
            khmax=7.,
            kmin_interp=self._kmin,
            kmax_interp=self._kmax,
        )
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)

        return pk_nw

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params):
        return self.get_pkmu_pair(k, mu, pk_data, params, add_stochasticity=True)

    @partial(jit, static_argnames=['self', 'add_stochasticity'])
    def get_pkmu_pair(self, k, mu, pk_data, params, add_stochasticity=False):
        if self.do_irres:
            # tree + 1-loop
            h = params['h']
            pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7., kmin_interp=self._kmin, kmax_interp=self._kmax)
            pkmu = self.get_pkmu_irres_LO_NLO(k, mu, pk_data, pk_nw_data, params)

            # counterterm
            pkmu_ctr_k2, pkmu_ctr_k4 = self.get_pkmu_ctrs(k, mu, pk_data, pk_nw_data, params)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        else:
            # tree + 1-loop
            pkmu_tree = self.get_pkmu_lin(k, mu, pk_data, params)
            pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, params)
            pkmu = pkmu_tree + pkmu_1loop

            # counterterm
            pkmu_ctr_k2, pkmu_ctr_k4 = self.get_pkmu_ctrs(k, mu, pk_data, {}, params)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4

        if add_stochasticity:
            if self._is_pair_spectrum(params):
                raise NotImplementedError('Additive stochasticity is only implemented for auto spectra.')
            pkmu = pkmu + self.get_pkmu_stoch(k, mu, params)
        
        return pkmu

    @partial(jit, static_argnames=['self', 'num'])
    def get_pk_ell(self, k, l, pk_data, params, num=256):
        return self.get_pk_ell_pair(k, l, pk_data, params, num=num, add_stochasticity=True)

    @partial(jit, static_argnames=['self', 'num', 'add_stochasticity'])
    def get_pk_ell_pair(self, k, l, pk_data, params, num=256, add_stochasticity=False):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu_pair(k, mu, pk_data, params, add_stochasticity=add_stochasticity)

        leg = (2*l+1) * legendre(l, mu)[None, :]  # broadcast over k
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        return self.get_pkmu_pair_ref(k, mu, alpha_perp, alpha_para, pk_data, params, add_stochasticity=True)

    def get_pkmu_pair_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params, add_stochasticity=False):
        if self._ap_is_identity(alpha_perp, alpha_para):
            return self.get_pkmu_pair(k, mu, pk_data, params, add_stochasticity=add_stochasticity)
        return self._get_pkmu_pair_ref_interp(k, mu, alpha_perp, alpha_para, pk_data, params, add_stochasticity=add_stochasticity)

    @partial(jit, static_argnames=['self', 'add_stochasticity'])
    def _get_pkmu_pair_ref_interp(self, k, mu, alpha_perp, alpha_para, pk_data, params, add_stochasticity=False):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu_true = mu * (alpha_perp / alpha_para) / fac
        k_true = k[:, None] * fac[None, :] / alpha_perp

        # spline interpolation
        pkmu_grid = self.get_pkmu_pair(self._kn, self._mu, pk_data, params, add_stochasticity=add_stochasticity)
        mu_grid = jnp.broadcast_to(mu_true[None, :], (len(k), len(mu)))
        pkmu = interpax.interp2d(jnp.ravel(k_true), jnp.ravel(mu_grid), self._kn, self._mu, pkmu_grid, extrap=True)
        pkmu = pkmu.reshape(len(k), len(mu)) / (alpha_perp**2 * alpha_para)

        return pkmu

    def get_pk_ell_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=256):
        return self.get_pk_ell_pair_ref(k, l, alpha_perp, alpha_para, pk_data, params, num=num, add_stochasticity=True)

    def get_pk_ell_pair_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=256, add_stochasticity=False):
        if self._ap_is_identity(alpha_perp, alpha_para):
            return self.get_pk_ell_pair(k, l, pk_data, params, num=num, add_stochasticity=add_stochasticity)
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu_pair_ref(k, mu, alpha_perp, alpha_para, pk_data, params, add_stochasticity=add_stochasticity)

        leg = (2*l+1) * legendre(l, mu)[None, :]  # broadcast over k
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    def get_pkmu_gm(self, k, mu, pk_data, params):
        return self.get_pkmu_pair(k, mu, pk_data, self._get_gm_pair_params(params))

    def get_pk_ell_gm(self, k, l, pk_data, params, num=256):
        return self.get_pk_ell_pair(k, l, pk_data, self._get_gm_pair_params(params), num=num)

    def get_pkmu_gm_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        return self.get_pkmu_pair_ref(k, mu, alpha_perp, alpha_para, pk_data, self._get_gm_pair_params(params))

    def get_pk_ell_gm_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=256):
        return self.get_pk_ell_pair_ref(k, l, alpha_perp, alpha_para, pk_data, self._get_gm_pair_params(params), num=num)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_lin(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        f = params['f']
        bias1, bias2 = self._get_bias_pair(params)

        Z1_1 = bias1['b1'] + f * mu**2
        Z1_2 = bias2['b1'] + f * mu**2
        pk_lin = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pkmu = pk_lin[:, None] * (Z1_1 * Z1_2)[None, :]

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict(self, pk_data):
        pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        p_q, p_k, p_k0 = self.decomp['pk_lin nu=-0.7'].get_decomposed_data(pk_lin)
        
        pk_dict = {}

        for name in self.name_pkmu_terms['22']:
            pk = self._kn**3 * jnp.diag(jnp.dot(p_q.T, jnp.dot(self.matrix[name], p_q)).real)
            if self.subtract_k0_limit:
                pk_k0 = self._kmin**3 * jnp.dot(p_k0, jnp.dot(self.matrix[name], p_k0)).real
                pk = pk - pk_k0
            pk_dict[name] = pk
        
        for name in self.name_pkmu_terms['13']:
            pk = self._kn**3 * p_k * jnp.dot(self.matrix[name], p_q).real
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data(self, pk_data, params):
        f = params['f']
        bias1, bias2 = self._get_bias_pair(params)

        # pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_dict = self.get_pkmu_dict(pk_data)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))
        for name in self.name_pkmu_terms['tot']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * (pk_dict[name][:, None] * (self._mu**nmu)[None, :])
            pkmu = pkmu + pkmu_term

        pkmu = pkmu + self.get_pkmu_13_UV(self._kn, self._mu, pk_data, params)

        return pkmu
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_13_UV(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        f = params['f']
        bias1, bias2 = self._get_bias_pair(params)

        Z1_g = bias1['b1'] + f * mu**2
        Z3_g_UV = - 61./315. * bias2['b1'] - 64./21. * bias2['bG2'] - 128./105. * bias2['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias2['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias2['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_1 = Z1_g * Z3_g_UV

        Z1_g = bias2['b1'] + f * mu**2
        Z3_g_UV = - 61./315. * bias1['b1'] - 64./21. * bias1['bG2'] - 128./105. * bias1['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias1['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias1['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_2 = Z1_g * Z3_g_UV

        Z1Z3_UV = (Z1Z3_UV_1 + Z1Z3_UV_2) / 2

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = get_pk_int(pk_data)

        pkmu_13 = (k**2 * pk * pk_int)[:, None] * Z1Z3_UV[None, :]

        return pkmu_13
    
    # @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        pkmu_data = self.get_pkmu_data(pk_data, params)

        k_grid = jnp.broadcast_to(k[:, None], (len(k), len(mu)))
        mu_grid = jnp.broadcast_to(mu[None, :], (len(k), len(mu)))
        pkmu = interpax.interp2d(jnp.ravel(k_grid), jnp.ravel(mu_grid), self._kn, self._mu, pkmu_data)
        pkmu = pkmu.reshape(len(k), len(mu))

        return pkmu

    # @partial(jax.jit, static_argnames=['self'])
    def _get_pkmu_irres_1loop_components(self, k, mu, pk_data, pk_nw_data, params):
        pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, params)
        pkmu_1loop_nw = self.get_pkmu_1loop(k, mu, pk_nw_data, params)
        pkmu_1loop_w = pkmu_1loop - pkmu_1loop_nw
        return pkmu_1loop_nw, pkmu_1loop_w
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_irres_LO_NLO(self, k, mu, pk_data, pk_nw_data, params):
        h = params['h']
        f = params['f']
        bias1, bias2 = self._get_bias_pair(params)

        pk_nw, pk_w, damp_fac = self._get_irres_components(k, mu, pk_data, pk_nw_data, f)

        # LO term
        Z1_1 = bias1['b1'] + f * mu**2
        Z1_2 = bias2['b1'] + f * mu**2
        Z1_factor = (Z1_1 * Z1_2)[None, :]
        pkmu_irres_tree = Z1_factor * (pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None] * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop_nw, pkmu_1loop_w = self._get_pkmu_irres_1loop_components(k, mu, pk_data, pk_nw_data, params)
        pkmu_irres_1loop = pkmu_1loop_nw + jnp.exp(-damp_fac) * pkmu_1loop_w

        pkmu = pkmu_irres_tree + pkmu_irres_1loop

        return pkmu
    
    # @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k2(self, k, mu, pk_data, pk_nw_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        h = params['h']
        f = params['f']
        ctr1, ctr2 = self._get_ctr_pair(params)

        ctr_k2_mu = (ctr1['c0'] + ctr2['c0']) / 2 \
                    + (ctr1['c2'] + ctr2['c2']) / 2 * f * mu**2 \
                    + (ctr1['c4'] + ctr2['c4']) / 2 * f**2 * mu**4

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, pk_nw_data, f)
            pkmu_ctr_k2 = - 2 * (k**2)[:, None] * ctr_k2_mu[None, :] * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k2 = - 2 * (k**2 * pk)[:, None] * ctr_k2_mu[None, :]

        return pkmu_ctr_k2
    
    # @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k4(self, k, mu, pk_data, pk_nw_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        h = params['h']
        f = params['f']
        bias1, bias2 = self._get_bias_pair(params)
        ctr1, ctr2 = self._get_ctr_pair(params)

        ctr_k4_mu = (ctr1['cfog'] + ctr2['cfog']) / 2 * f**4 * mu**4 \
                    * (bias1['b1'] + f * mu**2) * (bias2['b1'] + f * mu**2)

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, pk_nw_data, f)
            pkmu_ctr_k4 = - (k**4)[:, None] * ctr_k4_mu[None, :] * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k4 = - (k**4 * pk)[:, None] * ctr_k4_mu[None, :]

        return pkmu_ctr_k4

    # @partial(jit, static_argnames=['self'])
    def get_pkmu_ctrs(self, k, mu, pk_data, pk_nw_data, params):
        pkmu_ctr_k2 = self.get_pkmu_ctr_k2(k, mu, pk_data, pk_nw_data, params)
        pkmu_ctr_k4 = self.get_pkmu_ctr_k4(k, mu, pk_data, pk_nw_data, params)
        return pkmu_ctr_k2, pkmu_ctr_k4

    @partial(jit, static_argnames=['self'])
    def _get_pk_irres_rsd(self, k, mu, pk_data, pk_nw_data, f):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        pk_nw, pk_w, damp_fac = self._get_irres_components(k, mu, pk_data, pk_nw_data, f)

        pk = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
        return pk

    # @partial(jit, static_argnames=['self'])
    def _get_irres_components(self, k, mu, pk_data, pk_nw_data, f):
        # wiggly-non-wiggly decomposition
        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
        Sigma2_tot = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2

        damp_fac = (k**2)[:, None] * Sigma2_tot[None, :]

        return pk_nw, pk_w, damp_fac

    # NOTE: can be removed
    @partial(jit, static_argnames=['self'])
    def get_pkmu_stoch(self, k, mu, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        stoch = params['stoch']
        k_nl = params['k_nl']

        ndens = params['ndens']
        k_knl2 = ((k / k_nl)**2)[:, None]
        pkmu = stoch['P_shot'] \
            + stoch['a0'] * k_knl2 * jnp.ones_like(mu)[None, :] \
            + stoch['a2'] * k_knl2 * (mu**2)[None, :]
        pkmu = (1. / ndens) * pkmu

        return pkmu
