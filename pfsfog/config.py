"""Forecast configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ForecastConfig:
    """All tuneable knobs for the Fisher forecast."""

    # k-range
    kmin: float = 0.01          # h/Mpc
    kmax: float = 0.20          # h/Mpc  (overridden per scenario)
    dk: float = 0.005           # h/Mpc  bin width

    # z-bins for the overlap
    z_bins: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    )

    # Survey areas
    pfs_area_deg2: float = 1200.0
    desi_area_deg2: float = 14000.0
    overlap_area_deg2: float = 1200.0

    # PFS EFT scaling
    r_sigma_v: float = 0.75     # σ_v,PFS / σ_v,DESI

    # Backends
    cosmo_backend: str = "cosmopower"      # "cosmopower" or "clax"
    oneloop_backend: str = "ps_1loop_jax"  # "ps_1loop_jax" or "clax_ept"

    # Asymmetric kmax in the overlap (Phase A)
    kmax_desi_overlap: float = 0.20        # kmax for P^BB (DESI auto) in overlap
    kmax_pfs_overlap: float | None = None  # kmax for P^AA (PFS auto); None → auto
    kmax_cross_overlap: float | None = None  # kmax for P^AB; None → kmax_pfs

    # Shared catalog fraction for PFS-ELG × DESI-ELG cross-shot noise.
    # Only consulted when ``marginalize_cross_stoch=False`` — when
    # marginalization is enabled (the conservative default) the
    # fiducial cross-shot in the covariance is set to zero and the
    # cross-stochasticity is treated via free Fisher parameters
    # ``Pshot_cross`` / ``a2_cross`` (see Ebina & White 2024).
    f_shared_elg: float = 0.05

    # Cross-stochasticity treatment in the joint Fisher.
    # True (default): set fiducial cross-shot to zero in the covariance
    # and add free ``Pshot_cross`` + ``a2_cross`` parameters per cross-pair
    # per z-bin, marginalized via broad priors (CROSS_STOCH_PRIOR_SIGMA).
    # False: legacy behavior — use ``f_shared_elg`` as a fixed fiducial
    # cross-shot in the covariance, no cross-stochastic free parameters.
    marginalize_cross_stoch: bool = True

    # Nonlinear scale for stochasticity
    k_nl: float = 0.7           # h/Mpc

    # Output
    output_dir: str = "results"

    def compute_kmax_pfs(self) -> float:
        """kmax for PFS auto-spectrum in overlap.

        If not set, derived from kmax_DESI × r_σv^{-1}
        (EFT convergence: k_max ∝ c̃^{-1/4} ∝ σ_v^{-1},
        since c̃ ∝ σ_v⁴).
        """
        if self.kmax_pfs_overlap is not None:
            return self.kmax_pfs_overlap
        return self.kmax_desi_overlap * self.r_sigma_v ** (-1.0)

    def compute_kmax_cross(self) -> float:
        """kmax for cross-spectrum in overlap. Defaults to kmax_PFS."""
        if self.kmax_cross_overlap is not None:
            return self.kmax_cross_overlap
        return self.compute_kmax_pfs()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ForecastConfig:
        with open(path) as fh:
            d = yaml.safe_load(fh)
        if "z_bins" in d:
            d["z_bins"] = [tuple(zb) for zb in d["z_bins"]]
        return cls(**d)
