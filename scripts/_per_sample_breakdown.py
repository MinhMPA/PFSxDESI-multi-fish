#!/usr/bin/env python
"""Per-sample improvement breakdown: broad vs cross-cal for each DR2 sample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pfsfog.config import ForecastConfig
from pfsfog.cosmo import FiducialCosmology
from pfsfog.eft_params import NUISANCE_NAMES, COSMO_NAMES, broad_priors, tracer_fiducials
from pfsfog.ps1loop_adapter import fisher_to_ps1loop_auto, make_ps1loop_pkmu_func
from pfsfog.derivatives import dPell_dtheta_autodiff_all, dPell_d_cosmo_all
from pfsfog.covariance import single_tracer_cov
from pfsfog.fisher_full_area import full_area_fisher_per_zbin

from run_desi_multisample import (
    DR2_SAMPLES, SURVEY_CONSTRUCTORS, run_overlap_step1, average_calibrated_prior_diag
)

cfg = ForecastConfig.from_yaml("configs/default.yaml")
cosmo = FiducialCosmology(backend=cfg.cosmo_backend)
from ps_1loop_jax import PowerSpectrum1Loop
ps = PowerSpectrum1Loop(do_irres=False)

cfg.z_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
cal_per_tracer = run_overlap_step1(cfg, cosmo, ps, verbose=False)

ells = (0, 2, 4)
kmax = 0.20
k = np.arange(cfg.kmin, kmax + cfg.dk / 2, cfg.dk)

header = "{:<8s} {:>10s} {:>10s} {:>10s} {:>6s} {:>10s} {:>10s} {:>6s}".format(
    "Sample", "z", "s(fs8)_b", "s(fs8)_c", "D%", "s(Mnu)_b", "s(Mnu)_c", "D%")
print(header)
print("-" * 80)

for sample in DR2_SAMPLES:
    survey = SURVEY_CONSTRUCTORS[sample.tracer]()
    zlo, zhi = sample.z_range
    z_eff = 0.5 * (zlo + zhi)
    nbar = survey.nbar_eff(zlo, zhi)
    if nbar == 0:
        continue
    b1 = survey.b1_of_z(z_eff)
    V = survey.volume(zlo, zhi)
    s8 = cosmo.sigma8(z_eff)
    f_z = float(cosmo.f(z_eff))
    h = cosmo.params["h"]

    fid = tracer_fiducials(sample.tracer, b1, s8)
    params = fisher_to_ps1loop_auto(fid, s8, f_z, h, nbar)
    pk_data = cosmo.pk_data(z_eff)

    derivs = dPell_dtheta_autodiff_all(
        ps, jnp.array(k), pk_data, params, NUISANCE_NAMES, s8, ells)
    cd = dPell_d_cosmo_all(
        ps, jnp.array(k), pk_data, cosmo, params, z_eff, s8, ells)
    derivs.update(cd)

    pkmu = make_ps1loop_pkmu_func(ps, pk_data, params)
    cov = single_tracer_cov(pkmu, k, nbar, V, cfg.dk, ells)

    fr_b = full_area_fisher_per_zbin(
        derivs, cov, k, cfg.dk, broad_priors().prior_fisher_diag(),
        sample.z_range, kmax, ells)
    cal_diag = average_calibrated_prior_diag(
        cal_per_tracer.get(sample.tracer, {}), sample.overlap_zbins)
    fr_c = full_area_fisher_per_zbin(
        derivs, cov, k, cfg.dk, cal_diag,
        sample.z_range, kmax, ells)

    sb_f = fr_b.marginalized_sigma("fsigma8")
    sc_f = fr_c.marginalized_sigma("fsigma8")
    sb_m = fr_b.marginalized_sigma("Mnu")
    sc_m = fr_c.marginalized_sigma("Mnu")
    imp_f = (sb_f - sc_f) / sb_f * 100
    imp_m = (sb_m - sc_m) / sb_m * 100

    zstr = "[{:.1f},{:.1f}]".format(zlo, zhi)
    print("{:<8s} {:>10s} {:10.4e} {:10.4e} {:+5.1f}% {:10.4e} {:10.4e} {:+5.1f}%".format(
        sample.name, zstr, sb_f, sc_f, imp_f, sb_m, sc_m, imp_m))
