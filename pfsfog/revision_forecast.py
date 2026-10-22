"""Fixed-covariance likelihood assembly for the physical revision model."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from .covariance_mt_general import multi_tracer_cov_general
from .cosmology_space import make_cosmology_spec
from .ap_numerics import AP_GRIDS
from .eft_params import (CROSS_STOCH_NAMES, CROSS_STOCH_PRIOR_SIGMA,
                         broad_priors, tracer_fiducials)
from .fisher_core import fisher_from_shells
from .revision_theory import (COSMO_NAMES, NUISANCE_NAMES, cosmology_from_theta,
                              fiducial_nuisance, make_pair_response,
                              make_growth_rate_func, make_sigma8_func)
from .survey_geometry import ShellGeometry, support_cells


NOMINAL_BINS = ((.4, .6), (.6, .8), (.8, 1.), (1., 1.2),
                (1.2, 1.4), (1.4, 1.6), (1.6, 2.), (2., 2.1))
CASES = ("I", "A", "B", "D", "C")
_EMULATOR_LIMITS = {"omega_b": (.015, .030), "omega_cdm": (.09, .15),
                    "ln10_10_As": (2.5, 3.5), "n_s": (.85, 1.05),
                    "h": (.4, 1.), "mnu": (0., .5)}


def validate_reference_domain(reference, nominal_bins):
    """Check released Jense network inputs and the frozen physical basis."""
    source = reference.params if hasattr(reference, "params") else reference
    for name, (lower, upper) in _EMULATOR_LIMITS.items():
        value = source.get(name)
        if value is None or not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} outside released emulator domain [{lower}, {upper}]")
    actual_matter = (source["omega_b"] + source["omega_cdm"]
                     + source["mnu"] / 93.14) / source["h"]**2
    supplied_matter = source.get("Omega_m")
    if (supplied_matter is None or not np.isfinite(supplied_matter)
            or not np.isclose(supplied_matter, actual_matter, rtol=1e-10, atol=1e-12)):
        raise ValueError("inconsistent Omega_m and physical densities")
    if any(not (0. <= (lo + hi) / 2 <= 5.) for lo, hi in nominal_bins):
        raise ValueError("model redshift outside released emulator domain [0, 5]")


@dataclass(frozen=True)
class RevisionConfig:
    case: str
    kmin: float = .01
    kmax: float = .20
    dk: float = .005
    nuisance_factor: float = 1.
    exclude_elg1: bool = False
    k_nl: float = .7
    r_sigma_v: float = .75
    n_mu: int = 32
    ap_interpolation_nk: int = 2048
    ap_refinement_nk: tuple[int, int] = (4096, 8192)
    ap_interpolation_nmu: int = 101
    nominal_bins: tuple[tuple[float, float], ...] = NOMINAL_BINS
    cosmology_basis: str = "baseline3"

    def __post_init__(self):
        if self.cosmology_basis not in ("baseline3", "extended6"):
            raise ValueError("cosmology_basis must be baseline3 or extended6")
        if isinstance(self.ap_refinement_nk, list):
            object.__setattr__(self, "ap_refinement_nk", tuple(self.ap_refinement_nk))
        values = (self.kmin, self.kmax, self.dk, self.nuisance_factor,
                  self.k_nl, self.r_sigma_v)
        if (self.case not in CASES or not np.all(np.isfinite(values))
                or self.kmin < .01 or self.kmax > .20 or self.kmax <= self.kmin
                or self.dk <= 0 or self.nuisance_factor <= 0
                or self.k_nl <= 0 or self.r_sigma_v <= 0
                or not isinstance(self.n_mu, int) or self.n_mu < 3
                or not self.nominal_bins
                or any(b <= a or a < .4 or b > 2.1 for a, b in self.nominal_bins)):
            raise ValueError("revision configuration outside accepted domain")
        if (self.ap_interpolation_nk not in AP_GRIDS
                or not isinstance(self.ap_refinement_nk, tuple)
                or self.ap_refinement_nk !=
                   (2*self.ap_interpolation_nk, 4*self.ap_interpolation_nk)
                or self.ap_interpolation_nmu != 101):
            raise ValueError("AP interpolation/refinement configuration outside accepted sequence")


@dataclass(frozen=True)
class ShellGrid:
    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray


def make_shell_grid(kmin: float, kmax: float, dk: float) -> ShellGrid:
    if (not np.all(np.isfinite([kmin, kmax, dk])) or kmin <= 0
            or kmax <= kmin or dk <= 0):
        raise ValueError("invalid physical shell bounds or width")
    n_full = int(np.floor((kmax - kmin) / dk + 1e-12))
    edges = kmin + dk * np.arange(n_full + 1, dtype=float)
    edges[0] = kmin
    if edges[-1] < kmax - 1e-12 * max(1., kmax):
        edges = np.append(edges, kmax)
    else:
        edges[-1] = kmax
    widths = np.diff(edges)
    if np.any(widths <= 0) or edges[-1] > kmax:
        raise ValueError("invalid final shell")
    return ShellGrid(edges, (edges[:-1] + edges[1:]) / 2, widths)


def retained_pairs(case: str, tracer_names) -> list[tuple[str, str]]:
    if case not in CASES:
        raise ValueError(f"unknown revision case {case}")
    names = sorted(set(tracer_names))
    desi = [name for name in names if name.startswith("DESI-")]
    active = desi if case in ("I", "A", "B") else names
    autos = [(name, name) for name in active]
    if case in ("I", "A"):
        return autos
    return autos + [(a, b) for i, a in enumerate(active) for b in active[i + 1:]
                    if case != "D" or (a in desi and b in desi)]


@dataclass
class CellLikelihood:
    F_like: np.ndarray
    names: list[str]
    metadata: dict
    validation: dict
    derivatives: np.ndarray
    covariance: np.ndarray
    fiducial_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class RevisionLikelihood:
    config: RevisionConfig
    F_like: np.ndarray
    names: list[str]
    fiducial: np.ndarray
    parameters: list[dict]
    cells: list[CellLikelihood]
    bin_F_like: list[np.ndarray]
    grid: ShellGrid
    reference_cosmology: dict
    model: dict
    validation: dict

    @property
    def cosmo_names(self) -> tuple[str, ...]:
        return make_cosmology_spec(
            self.reference_cosmology,
            extended=self.config.cosmology_basis == "extended6",
            ap_nk=self.config.ap_interpolation_nk).names


@dataclass
class PriorResult:
    likelihood: RevisionLikelihood
    nuisance_factor: float
    parameters: list[dict]
    F_prior_cosmo: np.ndarray
    F_prior_nuisance: np.ndarray
    F_total: np.ndarray
    covariance: np.ndarray
    cov_cosmo_marg: np.ndarray
    cov_cosmo_nuisance_fixed: np.ndarray
    growth_z: np.ndarray
    growth_fiducial: np.ndarray
    growth_jacobian: np.ndarray
    growth_cov_marg: np.ndarray
    growth_cov_nuisance_fixed: np.ndarray
    validation: dict
    fisher_solver: str = 'scaled-cholesky-float64-v1'

    @property
    def F_like(self):
        return self.likelihood.F_like

    @property
    def names(self):
        return self.likelihood.names

    @property
    def fiducial(self):
        return self.likelihood.fiducial

    @property
    def grid(self):
        return self.likelihood.grid


def _bin_number(config: RevisionConfig, nominal_bin) -> int:
    try:
        return config.nominal_bins.index(tuple(nominal_bin))
    except ValueError as exc:
        raise ValueError(f"cell nominal bin {nominal_bin} absent from configuration") from exc


def _tracer_label(ibin, tracer, coordinate):
    return f"bin{ibin:02d}:{tracer}:{coordinate}"


def _cross_label(ibin, pair, coordinate):
    return f"bin{ibin:02d}:{pair[0]}|{pair[1]}:{coordinate}"


def _all_pairs(names):
    names = sorted(names)
    return [(a, a) for a in names] + [
        (a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def _cosmology_coordinates(config, reference):
    space = make_cosmology_spec(reference, extended=config.cosmology_basis == "extended6",
                                ap_nk=config.ap_interpolation_nk)
    keys = {"Mnu": "mnu", "Omegam": "Omega_m"}
    theta = np.asarray([reference[keys.get(name, name)] for name in space.names], dtype=float)
    return space, theta


def _scaled_spd(matrix, context):
    """Validate a positive matrix without changing its eigenvalues."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
        raise ValueError(f"nonfinite or nonsquare {context}")
    if np.any(np.diag(m) <= 0):
        raise ValueError(f"nonpositive diagonal in {context}")
    scale = np.max(np.abs(m))
    if np.max(np.abs(m - m.T)) > 1e-10 * scale:
        raise ValueError(f"asymmetric {context}")
    s = np.sqrt(np.diag(m))
    corr = (m + m.T) / (2 * s[:, None] * s[None, :])
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"non-positive-definite {context}") from exc
    residual = np.max(np.abs(np.einsum("ik,jk->ij", chol, chol, optimize=False) - corr))
    if not np.isfinite(residual) or residual > 1e-10:
        raise ValueError(f"scaled Cholesky residual {residual} in {context}")
    eigen = np.linalg.eigvalsh(corr)
    return {"min_scaled_eigenvalue": float(eigen[0]),
            "scaled_condition": float(eigen[-1] / eigen[0]),
            "scaled_residual": float(residual)}


def build_cell_likelihood(config: RevisionConfig, cell, region: str,
                          reference_fiducials, cosmo, ps) -> CellLikelihood:
    """Assemble one independent radial/footprint cell with fixed covariance."""
    if region not in ("overlap", "nonoverlap"):
        raise ValueError("region must be overlap or nonoverlap")
    ibin = _bin_number(config, cell.nominal_bin)
    names = cell.overlap_tracers if region == "overlap" else cell.nonoverlap_tracers
    names = tuple(sorted(n for n in names if config.case in ("C", "D") or n.startswith("DESI-")))
    volume = cell.overlap_volume if region == "overlap" else cell.nonoverlap_volume
    if volume <= 0 or not names:
        raise ValueError("cell requires positive volume and retained populations")
    for name in names:
        if name not in reference_fiducials or name not in cell.nbar or cell.nbar[name] <= 0:
            raise ValueError(f"missing fiducial or density for {name}")
    reference = cosmo.params if hasattr(cosmo, "params") else cosmo
    space, theta = _cosmology_coordinates(config, reference)
    theta = jnp.asarray(theta)
    n_cosmo = len(space.names)
    q = {name: fiducial_nuisance(reference_fiducials[name]) for name in names}
    grid = make_shell_grid(config.kmin, config.kmax, config.dk)
    all_pairs = _all_pairs(names)
    selected = retained_pairs(config.case, names)
    selected_cross = [pair for pair in selected if pair[0] != pair[1]]
    local_names = list(space.names)
    local_names += [_tracer_label(ibin, name, coord) for name in names for coord in NUISANCE_NAMES]
    local_names += [_cross_label(ibin, pair, coord) for pair in selected_cross for coord in CROSS_STOCH_NAMES]
    indexes = {name: i for i, name in enumerate(local_names)}
    responses = {pair: make_pair_response(
        ps, cell.model_redshift, reference, cell.nbar[pair[0]],
        None if pair[0] == pair[1] else cell.nbar[pair[1]], config.k_nl,
        ap_nk=config.ap_interpolation_nk, ap_nmu=config.ap_interpolation_nmu,
        **({"space": space} if config.cosmology_basis == "extended6" else {}))
        for pair in all_pairs}

    mu, _ = leggauss(config.n_mu)
    signals = {}
    for pair, response in responses.items():
        a, b = pair
        signals[pair] = np.asarray(response.pkmu(
            grid.centers, mu, theta, q[a], None if a == b else q[b],
            None if a == b else jnp.zeros(2)), dtype=float)
        if signals[pair].shape != (len(grid.centers), config.n_mu):
            raise ValueError(f"invalid field shape for {pair}")
    field_checks = []
    for ik, k in enumerate(grid.centers):
        for im, angle in enumerate(mu):
            field = np.array([[signals[tuple(sorted((a, b)))][ik, im]
                               + (1. / cell.nbar[a] if a == b else 0.)
                               for b in names] for a in names])
            field_checks.append(_scaled_spd(field, f"field cell={cell.radial_bin} k={k} mu={angle}"))
    # Every theoretical cross is supplied explicitly; the legacy covariance
    # helper's missing-key zero fallback is never used by this assembly.
    funcs = {pair: (lambda _k, _mu, value=value: value) for pair, value in signals.items()}
    full_cov = multi_tracer_cov_general(
        list(names), funcs, {name: cell.nbar[name] for name in names},
        grid.centers, volume, grid.widths, n_mu=config.n_mu)
    pair_index = {pair: i for i, pair in enumerate(all_pairs)}
    observed = [3 * pair_index[pair] + ell for pair in selected for ell in range(3)]
    cov = full_cov[:, observed][:, :, observed].copy()
    if config.case == "I":
        for ia in range(len(selected)):
            for ib in range(len(selected)):
                if ia != ib:
                    cov[:, 3*ia:3*ia+3, 3*ib:3*ib+3] = 0.
    covariance_checks = [_scaled_spd(cov[ik], f"multipole cell={cell.radial_bin} k={k}")
                         for ik, k in enumerate(grid.centers)]

    D = np.zeros((len(grid.centers), len(selected) * 3, len(local_names)))
    fiducial_mean = np.zeros((len(grid.centers), len(selected) * 3))
    for ipair, pair in enumerate(selected):
        a, b = pair
        is_cross = a != b
        response = responses[pair]
        packed = jnp.concatenate((theta, q[a], q[b] if is_cross else jnp.array([]),
                                  jnp.zeros(2) if is_cross else jnp.array([])))

        def mean(p):
            first = p[n_cosmo:n_cosmo + len(NUISANCE_NAMES)]
            if is_cross:
                second = p[n_cosmo + len(NUISANCE_NAMES):n_cosmo + 2*len(NUISANCE_NAMES)]
                stochastic = p[-2:]
                return response.multipoles(grid.centers, p[:n_cosmo], first, second,
                                           stochastic, n_mu=config.n_mu)
            return response.multipoles(grid.centers, p[:n_cosmo], first, n_mu=config.n_mu)

        fiducial_mean[:, 3*ipair:3*ipair+3] = np.asarray(mean(packed), dtype=float)
        jac = np.asarray(jax.jacfwd(mean)(packed))
        if jac.shape != (len(grid.centers), 3, len(packed)) or not np.all(np.isfinite(jac)):
            raise ValueError(f"invalid complete response for pair {pair}")
        if np.any(np.max(np.abs(jac[:, :, :n_cosmo]), axis=(0, 1)) == 0):
            raise ValueError(f"unimplemented cosmological response for pair {pair}")
        dest = list(range(n_cosmo))
        dest += [indexes[_tracer_label(ibin, a, coord)] for coord in NUISANCE_NAMES]
        if is_cross:
            dest += [indexes[_tracer_label(ibin, b, coord)] for coord in NUISANCE_NAMES]
            dest += [indexes[_cross_label(ibin, pair, coord)] for coord in CROSS_STOCH_NAMES]
        D[:, 3*ipair:3*ipair+3, dest] = np.moveaxis(jac, -1, 0).transpose(1, 2, 0)
    fisher = fisher_from_shells(D, cov, grid.centers)
    validation = {
        "field_min_scaled_eigenvalue": min(x["min_scaled_eigenvalue"] for x in field_checks),
        "field_max_scaled_condition": max(x["scaled_condition"] for x in field_checks),
        "covariance_min_scaled_eigenvalue": min(x["min_scaled_eigenvalue"] for x in covariance_checks),
        "covariance_max_scaled_condition": max(x["scaled_condition"] for x in covariance_checks),
        "max_scaled_residual": max(x["scaled_residual"] for x in field_checks + covariance_checks),
        "expected_shells": len(grid.centers), "processed_shells": len(grid.centers),
    }
    metadata = {"nominal_bin": tuple(cell.nominal_bin), "nominal_bin_index": ibin,
                "radial_bin": tuple(cell.radial_bin), "model_redshift": cell.model_redshift,
                "region": region, "volume": volume,
                "nbar": {name: cell.nbar[name] for name in names},
                "candidate_populations": tuple(cell.overlap_tracers if region == "overlap"
                                               else cell.nonoverlap_tracers),
                "active_populations": names, "theoretical_pairs": all_pairs,
                "retained_pairs": selected,
                "retained_observations": [(pair, ell) for pair in selected for ell in (0, 2, 4)],
                "local_names": local_names}
    return CellLikelihood(fisher, local_names, metadata, validation, D, cov,
                          fiducial_mean)


def build_revision_likelihood(config: RevisionConfig, survey_group, cosmo, ps) -> RevisionLikelihood:
    """Embed every retained support cell in one shared nominal-bin basis."""
    reference = dict(cosmo.params if hasattr(cosmo, "params") else cosmo)
    validate_reference_domain(reference, config.nominal_bins)
    space, theta = _cosmology_coordinates(config, reference)
    geometry = ShellGeometry.from_group(survey_group)
    prior_widths = broad_priors().sigma_dict()
    fiducials_by_bin = []
    all_cells = []
    for ibin, nominal in enumerate(config.nominal_bins):
        baseline = support_cells(survey_group, nominal, include_pfs=True,
                                 exclude_elg1=False, geometry=geometry)
        baseline_names = sorted({name for cell in baseline for name in cell.overlap_tracers})
        z = (nominal[0] + nominal[1]) / 2
        sigma8_z = float(cosmo.sigma8(z))
        if not np.isfinite(sigma8_z) or sigma8_z <= 0:
            raise ValueError(f"invalid fiducial sigma8 at z={z}")
        b1_ref = (survey_group.desi_tracers["DESI-ELG"].b1_of_z(z)
                  if "DESI-ELG" in baseline_names else 1.3)
        baseline_fids = {name: tracer_fiducials(
            name, survey_group.all_surveys[name].b1_of_z(z), sigma8_z,
            b1_ref=b1_ref, r_sigma_v=config.r_sigma_v)
            for name in baseline_names}
        fiducials_by_bin.append(baseline_fids)
        selected = support_cells(survey_group, nominal, include_pfs=True,
                                 exclude_elg1=config.exclude_elg1, geometry=geometry)
        for icell, cell in enumerate(selected):
            for region in ("overlap", "nonoverlap"):
                candidates = cell.overlap_tracers if region == "overlap" else cell.nonoverlap_tracers
                retained = retained_pairs(config.case, candidates)
                volume = cell.overlap_volume if region == "overlap" else cell.nonoverlap_volume
                if not retained or volume <= 0:
                    reason = "zero_volume" if volume <= 0 else "no_retained_population"
                    metadata = {
                        "cell_id": f"bin{ibin:02d}-radial{icell:02d}-{region}",
                        "nominal_bin": tuple(cell.nominal_bin), "nominal_bin_index": ibin,
                        "radial_bin": tuple(cell.radial_bin),
                        "model_redshift": cell.model_redshift,
                        "region": region, "volume": volume,
                        "nbar": {name: cell.nbar[name] for name in candidates},
                        "candidate_populations": tuple(candidates),
                        "active_populations": (), "theoretical_pairs": [],
                        "retained_pairs": [], "retained_observations": [],
                        "local_names": [],
                    }
                    validation = {
                        "reason": reason, "expected_shells": 0, "processed_shells": 0,
                        "field_min_scaled_eigenvalue": None,
                        "field_max_scaled_condition": None,
                        "covariance_min_scaled_eigenvalue": None,
                        "covariance_max_scaled_condition": None,
                        "max_scaled_residual": None,
                    }
                    all_cells.append(CellLikelihood(np.zeros((0, 0)), [], metadata,
                                                    validation, np.zeros((0, 0, 0)),
                                                    np.zeros((0, 0, 0))))
                    continue
                result = build_cell_likelihood(config, cell, region, baseline_fids, reference, ps)
                result.metadata["cell_id"] = f"bin{ibin:02d}-radial{icell:02d}-{region}"
                all_cells.append(result)

    names = list(space.names)
    parameters = [{"name": name, "role": "cosmo", "nominal_bin": None,
                   "tracers": [], "coordinate": name,
                   "unit": "eV" if name == "Mnu" else "",
                   "fiducial": float(value),
                   "prior_sigma": {"Mnu": 5., "Omegam": 1.}.get(name)}
                  for name, value in zip(space.names, theta)]
    for ibin, nominal in enumerate(config.nominal_bins):
        cells = [cell for cell in all_cells if cell.metadata["nominal_bin_index"] == ibin]
        tracers = sorted({name for cell in cells for name in cell.metadata["active_populations"]})
        crosses = sorted({pair for cell in cells for pair in cell.metadata["retained_pairs"]
                          if pair[0] != pair[1]})
        for tracer in tracers:
            fid = fiducials_by_bin[ibin][tracer]
            for coordinate in NUISANCE_NAMES:
                name = _tracer_label(ibin, tracer, coordinate)
                names.append(name)
                parameters.append({"name": name, "role": "tracer", "nominal_bin": ibin,
                                   "tracers": [tracer], "coordinate": coordinate,
                                   "unit": ("(Mpc/h)^2" if coordinate in ("c0", "c2", "c4")
                                            else "(Mpc/h)^4" if coordinate == "c_tilde" else ""),
                                   "fiducial": float(getattr(fid, coordinate)),
                                   "prior_sigma": prior_widths[coordinate]})
        for pair in crosses:
            for coordinate in CROSS_STOCH_NAMES:
                name = _cross_label(ibin, pair, coordinate)
                names.append(name)
                parameters.append({"name": name, "role": "cross", "nominal_bin": ibin,
                                   "tracers": list(pair), "coordinate": coordinate,
                                   "unit": "", "fiducial": 0.,
                                   "prior_sigma": CROSS_STOCH_PRIOR_SIGMA[coordinate]})
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("duplicate global parameter label")
    F_bins = [np.zeros((len(names), len(names))) for _ in config.nominal_bins]
    for cell in all_cells:
        local = [index[name] for name in cell.names]
        F_bins[cell.metadata["nominal_bin_index"]][np.ix_(local, local)] += cell.F_like
    F_like = np.sum(F_bins, axis=0)
    for i, record in enumerate(parameters):
        record["prior_only"] = bool(F_like[i, i] == 0 and record["prior_sigma"] is not None)
    model = {"backend": type(ps).__name__, "do_irres": getattr(ps, "do_irres", None),
             "subtract_k0_limit": getattr(ps, "subtract_k0_limit", None),
             "fft_kmin": getattr(ps, "_kmin", None),
             "fft_kmax": getattr(ps, "_kmax", None),
             "fft_n": getattr(ps, "_nfft", None),
             "ap_interpolation_nk": config.ap_interpolation_nk,
             "ap_refinement_nk": list(config.ap_refinement_nk),
             "ap_interpolation_nmu": config.ap_interpolation_nmu,
             "mean_n_mu": config.n_mu, "covariance_n_mu": config.n_mu,
             "fixed_observed_geometry": True, "fixed_covariance": True,
             "covariance_derivative_trace": False, "serial": True}
    if getattr(ps, "native_full_mean_ap", False):
        model.update(ps.numerical_settings())
    validation = {"cell_count": len(all_cells),
                  "expected_shells": sum(c.validation["expected_shells"] for c in all_cells),
                  "processed_shells": sum(c.validation["processed_shells"] for c in all_cells),
                  "min_field_scaled_eigenvalue": min(
                      (c.validation["field_min_scaled_eigenvalue"] for c in all_cells
                       if c.validation["field_min_scaled_eigenvalue"] is not None), default=None),
                  "min_covariance_scaled_eigenvalue": min(
                      (c.validation["covariance_min_scaled_eigenvalue"] for c in all_cells
                       if c.validation["covariance_min_scaled_eigenvalue"] is not None), default=None)}
    return RevisionLikelihood(config, F_like, names,
                              np.asarray([r["fiducial"] for r in parameters]),
                              parameters, all_cells, F_bins,
                              make_shell_grid(config.kmin, config.kmax, config.dk),
                              reference, model, validation)


def _scaled_inverse(matrix, labels, context, *, fisher_solver='scaled-cholesky-float64-v1'):
    from .fisher_precision import (DECIMAL_FISHER_SOLVER, decimal_inverse,
                                   fisher_solver_record)
    fisher_solver_record(fisher_solver)
    if fisher_solver == DECIMAL_FISHER_SOLVER:
        return decimal_inverse(matrix, labels, context)
    try:
        metrics = _scaled_spd(matrix, context)
        scales = np.sqrt(np.diag(matrix))
        corr = (matrix + matrix.T) / (2 * scales[:, None] * scales[None, :])
        chol = np.linalg.cholesky(corr)
        inverse_corr = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(labels))))
        residual = np.max(np.abs(np.einsum("ik,kj->ij", corr, inverse_corr, optimize=False)
                                 - np.eye(len(labels))))
        if not np.isfinite(residual) or residual > 1e-10:
            raise ValueError(f"scaled inverse residual {residual} in {context}")
        return inverse_corr / scales[:, None] / scales[None, :], {
            **metrics, "scaled_inverse_residual": float(residual)}
    except ValueError as exc:
        diagonal = np.diag(matrix)
        active = diagonal > 0
        if np.any(active):
            scaled = matrix[np.ix_(active, active)] / np.sqrt(diagonal[active])[:, None] / np.sqrt(diagonal[active])[None, :]
            eig, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
            suspect = [labels[i] for i in np.where(active)[0][np.argsort(-np.abs(vectors[:, 0]))[:5]]]
            if eig[0] > 1e-10:
                suspect = []
        else:
            suspect = []
        suspect += [labels[i] for i in np.where(~active)[0]]
        raise ValueError(f"{context}: unconstrained directions {suspect}: {exc}") from exc


def project_baseline_precision(likelihood: RevisionLikelihood, nuisance_factor: float, *,
                               fisher_solver='scaled-cholesky-float64-v1') -> dict:
    """Pure native3 prior/conditioning algebra, with no theory initialization."""
    if likelihood.config.cosmology_basis == "extended6":
        raise ValueError("extended6 requires explicit apply_cosmology_treatment")
    if not np.isfinite(nuisance_factor) or nuisance_factor <= 0:
        raise ValueError("nuisance width factor must be positive")
    if likelihood.names[:3] != list(COSMO_NAMES):
        raise ValueError("first three coordinates must be native cosmology")
    size = len(likelihood.names)
    if (len(set(likelihood.names)) != size
            or [r['name'] for r in likelihood.parameters] != likelihood.names
            or np.shape(likelihood.F_like) != (size, size)):
        raise ValueError('native3 parameter labels or Fisher shape mismatch')
    F_cosmo = np.zeros((size, size))
    F_cosmo[1, 1] = 1 / 5.**2
    F_cosmo[2, 2] = 1.
    F_nuisance = np.zeros((size, size))
    records = []
    for i, record in enumerate(likelihood.parameters):
        item = dict(record)
        if item["role"] != "cosmo" and item["prior_sigma"] is not None:
            item["prior_sigma"] *= nuisance_factor
            F_nuisance[i, i] = 1 / item["prior_sigma"]**2
        records.append(item)
    total = likelihood.F_like + F_cosmo + F_nuisance
    covariance, solve_diag = _scaled_inverse(total, likelihood.names, "global Fisher",
                                             fisher_solver=fisher_solver)
    fixed, fixed_diag = _scaled_inverse(total[:3, :3], likelihood.names[:3], "cosmology block",
                                        fisher_solver=fisher_solver)
    marginal = covariance[:3, :3]
    validation = {"global_solve": solve_diag, "cosmology_block_solve": fixed_diag,
                  "nuisance_factor": float(nuisance_factor),
                  "growth_rank_upper_bound": min(3, len(likelihood.config.nominal_bins)),
                  "prior_only": [record["name"] for record in records if record.get("prior_only")],
                  "q13_penalty_min": float(np.min(np.sqrt(np.diag(marginal) / np.diag(fixed))))}
    return dict(nuisance_factor=float(nuisance_factor), parameters=records,
                F_like=likelihood.F_like, F_prior_cosmo=F_cosmo, F_prior_nuisance=F_nuisance,
                F_total=total, covariance=covariance, cov_cosmo_marg=marginal,
                cov_cosmo_nuisance_fixed=fixed, validation=validation, fisher_solver=fisher_solver)


def apply_revision_priors(likelihood: RevisionLikelihood, nuisance_factor: float, *,
                          fisher_solver='scaled-cholesky-float64-v1') -> PriorResult:
    """Add once-only priors, invert full Fisher, and propagate physical growth."""
    precision = project_baseline_precision(likelihood, nuisance_factor, fisher_solver=fisher_solver)
    z = np.asarray([(a + b) / 2 for a, b in likelihood.config.nominal_bins])
    rate = make_growth_rate_func()
    sigma8 = make_sigma8_func()
    reference = likelihood.reference_cosmology

    def growth(theta):
        cosmo = cosmology_from_theta(theta, reference)
        return jnp.stack([rate(zi, cosmo) * sigma8(zi, cosmo) for zi in z])

    theta = jnp.asarray(likelihood.fiducial[:3])
    predicted = np.asarray(growth(theta), dtype=float)
    jac = np.asarray(jax.jacfwd(growth)(theta), dtype=float)
    if predicted.shape != (len(z),) or jac.shape != (len(z), 3) or not np.all(np.isfinite(jac)):
        raise ValueError("invalid growth predictions or Jacobian")
    return PriorResult(likelihood=likelihood,
                       **{k: v for k, v in precision.items() if k != 'F_like'},
                       growth_z=z, growth_fiducial=predicted, growth_jacobian=jac,
                       growth_cov_marg=jac @ precision['cov_cosmo_marg'] @ jac.T,
                       growth_cov_nuisance_fixed=jac @ precision['cov_cosmo_nuisance_fixed'] @ jac.T)
