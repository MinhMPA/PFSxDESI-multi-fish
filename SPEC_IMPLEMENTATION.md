# Implementation Spec: Data-Driven EFT Prior Calibration from PFS × DESI Overlap

**Target executor:** Claude Code
**Owner:** N.-M. Nguyen (Kavli IPMU)
**Scope:** 1–2 weekend project. Pure Python (NumPy / SciPy / CAMB).
**Self-contained:** This spec has no dependencies on earlier versions.

---

## 1. Scientific context and goal

### 1.1 The problem: EFT nuisance marginalization is expensive

EFT-based full-shape (FS) analyses of galaxy power spectrum
multipoles involve ~12 nuisance parameters per tracer per z-bin:

| Category | Parameters | Role |
|----------|-----------|------|
| Linear bias | b1σ8 | galaxy–matter relation |
| Nonlinear bias | b2σ8², bG2σ8² | one-loop corrections |
| Cubic tidal bias | bΓ3 | third-order tidal response |
| Leading counterterms | c0, c2, c4 | UV sensitivity of loops, [Mpc/h]² |
| NLO counterterm | c̃ | **Finger-of-God damping**, [Mpc/h]⁴ |
| Sub-leading counterterm | c1 | higher-derivative correction, [Mpc/h]² |
| Stochastic | Pshot, a0, a2 | departures from Poisson noise |

Marginalizing over these with conservative (broad) Gaussian priors
incurs a large cost on cosmological parameters (fσ8, Mν, Ωm).

**Evidence that tighter priors help enormously:**

- Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757):
  reparameterizing EFT priors (AP-rescaling, σ8-scaling) reduced
  projection effects from ~1.2σ to ~0.2σ in w0waCDM, enabling
  the first CMB+DESI(BAO+FS) dark energy constraints without SNe.

- Zhang, Bonici, Rocher, Percival + DESI (2025, arXiv:2504.10407):
  HOD-informed priors (normalizing flows trained on 320,000 mocks
  from AbacusSummit) applied to DESI DR1 FS improve σ(σ8) by
  **~23%** and σ(Ωm) by **~4%** over the baseline conservative
  priors, in ΛCDM with BAO+BBN.

- Chudaykin, Ivanov & Philcox (2026, arXiv:2602.18554):
  field-level simulation-based priors (SBPs) improve σ(H0) by
  **~40%** and σ(σ8) by **~50%** over conservative priors. The
  larger gains vs Zhang+ arise because field-level calibration
  breaks parameter degeneracies that persist at the power-spectrum
  level.

- Ivanov, Cuesta-Lazaro, Mishra-Sharma, Obuljen & Toomey (2024,
  arXiv:2402.13310): original SBP methodology on BOSS. SBPs reduce
  the posterior volume of bias parameters by an order of magnitude
  and improve fNL constraints by ~40%, equivalent to doubling the
  survey volume. Constraints on b1 do not improve (data already
  informative); the gain comes entirely from nonlinear bias and
  counterterm parameters.

- Ivanov, Obuljen, Cuesta-Lazaro & Toomey (2024, arXiv:2409.10609):
  extended SBPs to redshift space. Demonstrated sensitivity of
  full-shape analysis to priors on redshift-space counterterms
  (including FoG-associated c̃).

### 1.2 The opportunity: data-driven calibration from survey overlap

All existing informative-prior approaches rely on **HOD model
assumptions**. An independent, **data-driven** route to informative
EFT priors would:
- provide a model-independent cross-check of SBPs,
- avoid HOD-dependent systematic errors in the priors,
- be specific to the actual tracer population observed, not a
  simulated proxy.

We propose to obtain such priors from the ~1,200 deg² overlap
between PFS-ELG and DESI-ELG at 0.8 < z < 1.6. In this shared
volume, the joint analysis of three sets of power spectrum
multipoles — P_ℓ^{PFS}, P_ℓ^{DESI}, P_ℓ^{PFS×DESI} — constrains
the per-tracer EFT parameters more tightly than either
auto-spectrum alone. The calibrated DESI-ELG nuisance priors are
then **exported** to DESI's full 14,000 deg² footprint, where PFS
data are unavailable.

### 1.3 Why this works even with similar b1

PFS-ELG: b1(z) = 0.9 + 0.4z (Takada+ 2014).
DESI-ELG: b1(z) = 0.84/D(z) (DESI Collaboration 2016).

At z ~ 1, both give b1 ~ 1.3. The bias difference is small, so
standard multi-tracer sample-variance cancellation is modest.

But the calibration mechanism does **not** require different b1. It
requires different **nuisance parameters** — specifically different
FoG counterterms (c̃) and stochastic terms (Pshot, a0, a2). These
differ because:
- PFS goes ~1 mag deeper in [OII] flux → lower-mass host halos
  → lower virial velocity dispersion → smaller c̃.
- PFS has 2–3× higher nbar → different Pshot = 1/nbar baseline.
- Different satellite fractions → different sub-Poisson corrections
  (a0, a2).

The cross-power P^{PFS×DESI} has no auto-stochastic contribution
and a mixed FoG damping, providing independent leverage on
separating the two tracers' nuisance parameters.

### 1.4 What the forecast must show

Five analysis scenarios for DESI-ELG over 14,000 deg²:

| Label | EFT priors on DESI nuisance params | kmax | Source |
|-------|-------------------------------------|------|--------|
| `broad` | Conservative Gaussian (Chudaykin+ 2025 Table I) | 0.20 h/Mpc | baseline |
| `cross-cal` | From overlap PFS×DESI Fisher (this work) | 0.20 h/Mpc | data-driven |
| `cross-cal-ext` | From overlap Fisher | 0.25 h/Mpc | data-driven + extended kmax |
| `HOD-prior` | Benchmark from Zhang+ 2025 improvement ratios | 0.20 h/Mpc | simulation-based |
| `oracle` | Perfect (nuisance params fixed) | 0.25 h/Mpc | upper bound |

The `HOD-prior` scenario is **not** implemented as an actual prior
in the Fisher machinery. Instead, it is a **benchmark line** in the
figures: the published improvement ratios from Zhang+ 2025 (~4% on
Ωm, ~23% on σ8 over broad priors) are plotted as horizontal
reference lines on Figure 3, allowing direct visual comparison
with the cross-calibration results. This avoids the
apples-to-oranges problem of approximating correlated normalizing
flows as independent Gaussians.

Headline results:
- σ(fσ8), σ(Mν), σ(Ωm) for each scenario, per z-bin and combined.
- Calibration efficiency: (σ_broad − σ_cross-cal-ext) / (σ_broad − σ_oracle).
- Comparison: does cross-cal reach the Zhang+ HOD-prior benchmark?

### 1.5 Transferability assumption

The calibration transfers from overlap to full area iff DESI-ELG
in the overlap has the same HOD (and therefore same EFT parameters)
as DESI-ELG outside. Since DESI's target selection is spatially
uniform, this holds by construction. State as an explicit
assumption. Would break if imaging systematics caused spatial
variation in the ELG population, but this is separately controlled
in DESI's pipeline.

---

## 2. Survey specifications

All survey numbers are **hardcoded in the code** (not read from
user-supplied CSV files). The user can override via config.

### 2.1 PFS-ELG

Source: Takada+ 2014 (PASJ 66 R1); nbar from user-supplied table.
Linear bias: b1(z) = 0.9 + 0.4z.
Area: 1,200 deg².

| z_mid | nbar [(h⁻¹Mpc)⁻³] | b1   |
|-------|---------------------|------|
| 0.7   | 3.057e-4            | 1.18 |
| 0.9   | 9.609e-4            | 1.26 |
| 1.1   | 9.749e-4            | 1.34 |
| 1.3   | 6.542e-4            | 1.42 |
| 1.5   | 3.404e-4            | 1.50 |

### 2.2 DESI-ELG

Source: DESI Collaboration 2016 (arXiv:1611.00036), Table 2.3; Y5.
Linear bias: b1(z) = 0.84/D(z), where D(z) is the linear growth
factor normalized to D(0)=1.
Area: 14,000 deg².

| z_mid | nbar [(h⁻¹Mpc)⁻³] | b1   |
|-------|---------------------|------|
| 0.75  | 3.4e-4              | 1.18 |
| 0.85  | 4.0e-4              | 1.24 |
| 0.95  | 4.4e-4              | 1.30 |
| 1.05  | 3.9e-4              | 1.36 |
| 1.15  | 3.3e-4              | 1.43 |
| 1.25  | 2.7e-4              | 1.50 |
| 1.35  | 2.3e-4              | 1.57 |
| 1.45  | 1.7e-4              | 1.65 |
| 1.55  | 1.4e-4              | 1.74 |

### 2.3 Matched z-bins for the overlap

Rebin both surveys into 4 z-bins:
[0.8, 1.0], [1.0, 1.2], [1.2, 1.4], [1.4, 1.6].

For each bin, nbar is the average of the fine bins falling inside,
weighted by comoving volume per fine bin. b1 is evaluated at z_eff
(volume-weighted effective redshift of the bin).

Survey volume per bin:
V(z_min, z_max) = (area in sr) × ∫_{z_min}^{z_max} (c/H(z)) χ²(z) dz
where χ(z) is the comoving distance at Planck 2018 fiducial.

V_overlap uses 1,200 deg². V_DESI_full uses 14,000 deg².

### 2.4 Fiducial cosmology

Planck 2018 TT,TE,EE+lowE best-fit:
h = 0.6736, Ωm = 0.3153, Ωb h² = 0.02237, ns = 0.9649,
ln(10¹⁰ As) = 3.044, Mν = 0.06 eV (single massive neutrino).

Store in a module-level `FIDUCIAL` dict. No magic numbers in
function bodies.

---

## 3. EFT parameter setup

### 3.1 Parameterization convention

Follow Chudaykin, Ivanov & Philcox (2025, arXiv:2511.20757,
Table I) exactly. The EFT parameters are defined as follows:

**Sampled parameters** (enter the Fisher matrix as explicit rows/columns):
- `b1_sigma8` = b1 × σ8(z). Flat prior U[0, 3].
- `b2_sigma8sq` = b2 × σ8²(z). Gaussian prior N(0, 5²).
- `bG2_sigma8sq` = bG2 × σ8²(z). Gaussian prior N(0, 5²).

**Analytically marginalized parameters** (in a real MCMC, these are
marginalized analytically because they enter the likelihood
quadratically; in the Fisher forecast, we include them as explicit
parameters with their Gaussian priors added to the Fisher matrix,
which is mathematically equivalent):

| Parameter | Fiducial | Prior N(μ, σ²) | Units | Notes |
|-----------|----------|----------------|-------|-------|
| bΓ3 | 23/42 × (b1−1) | N(23/42(b1−1), 1²) | — | Rescaled by A_AP A_amp σ8⁴ |
| c0 | 0 | N(0, 30²) | [Mpc/h]² | Rescaled by A_AP A_amp |
| c2 | 30 | N(30, 30²) | [Mpc/h]² | Rescaled by A_AP A_amp |
| c4 | 0 | N(0, 30²) | [Mpc/h]² | Rescaled by A_AP A_amp |
| c̃ | 400 | N(400, 400²) | [Mpc/h]⁴ | **FoG counterterm.** Rescaled by A_AP A_amp |
| c1 | 0 | N(0, 5²) | [Mpc/h]² | Rescaled by A_AP A_amp |
| Pshot | 0 | N(0, 1²) | dimensionless | Relative to 1/nbar |
| a0 | 0 | N(0, 1²) | dimensionless | |
| a2 | 0 | N(0, 1²) | dimensionless | |

where A_AP = (H_fid/H)³ × (D_A,fid/D_A)² ≈ 1 at fiducial,
and A_amp = σ8²(z) / σ8,ref²(z) ≈ 1 at fiducial.

At the fiducial cosmology, A_AP = A_amp = 1 and the rescaling
is trivial. The rescaling matters only when computing derivatives
w.r.t. cosmological parameters (which shift H, D_A, σ8 away from
fiducial). The implementation must account for this.

### 3.2 DESI-ELG fiducials

Use the values from §3.1 directly. The fiducial values are the
prior means. b1 comes from §2.2.

### 3.3 PFS-ELG fiducials (scaled from DESI)

Since PFS-ELG mocks with EFT parameter fits do not yet exist,
scale from DESI fiducials using physically motivated prescriptions:

- `c̃_PFS = c̃_DESI × r_σv²` where r_σv ≡ σ_v,PFS / σ_v,DESI.
  Default r_σv = 0.75. Sensitivity range: r_σv ∈ {0.5, 0.6, 0.75,
  0.9, 1.0}. Rationale: PFS selects fainter [OII] emitters in
  lower-mass halos with lower virial velocities.

- `Pshot_PFS`: set by 1/nbar_PFS (Poisson baseline; fiducial
  deviation = 0).

- `c0, c2, c4, c1`: scale fiducials by b1_PFS / b1_DESI (leading
  counterterms scale roughly with bias).

- `b2, bG2, bΓ3`: use Lazeyras+ (2016) co-evolution relations
  from b1: b2 ≈ 0.412 − 2.143 b1 + 0.929 b1² + 0.008 b1³.

- `a0, a2`: same fiducial (0) and prior width as DESI.

### 3.4 Broad priors

The Gaussian prior widths in §3.1. These define the `broad`
scenario baseline.

### 3.5 HOD-prior benchmark (not a prior in the Fisher)

From Zhang, Bonici, Rocher, Percival + DESI (2025,
arXiv:2504.10407), applied to DESI DR1 in ΛCDM+BAO+BBN:
- σ(Ωm) improves by ~4% over baseline.
- σ(σ8) improves by ~23% over baseline.

From Chudaykin, Ivanov & Philcox (2026, arXiv:2602.18554),
field-level SBPs in ΛCDM+BAO+BBN:
- σ(H0) improves by ~40% over baseline.
- σ(σ8) improves by ~50% over baseline.

These are plotted as **reference lines** on figures, not
implemented as a Fisher prior. The Zhang+ numbers are the fairer
comparison (both operate at the power-spectrum level). The
Chudaykin+ numbers represent the field-level ceiling.

---

## 4. Theory model

### 4.1 User-supplied module (`pkmu_module.py`)

If the user supplies `data/pkmu_module.py`, the code imports it.
Required interface:

```python
def compute_pkmu(
    k: np.ndarray,       # (Nk,), h/Mpc
    mu: np.ndarray,      # (Nmu,), cosine of angle to LOS
    z: float,
    params: dict,        # EFT + cosmology params
) -> np.ndarray:
    """Return P(k, mu, z; params) with shape (Nk, Nmu)."""

def compute_multipoles(
    k: np.ndarray,
    z: float,
    params: dict,
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Return {ell: P_ell(k, z; params)} with shape (Nk,)."""

def compute_cross_multipoles(
    k: np.ndarray,
    z: float,
    params_A: dict,      # PFS params
    params_B: dict,      # DESI params
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Return {ell: P^{AB}_ell(k, z)} with shape (Nk,).

    Cross-power between two independent tracers.
    Cross-stochastic = 0.
    Cross-FoG damping = F_FoG(k,mu; c̃_A)^{1/2} × F_FoG(k,mu; c̃_B)^{1/2}.
    Cross-counterterms: at tree level,
      c0^{AB} = (c0_A b1_B + c0_B b1_A) / (b1_A + b1_B).
    """
```

### 4.2 Built-in fallback model (`builtin_pkmu.py`)

If `pkmu_module.py` is absent, the code uses a built-in model
sufficient for the Fisher forecast:

**Auto-power:**
```
P(k, μ, z) = D_FoG(k, μ; c̃) × [b1 + f μ²]² × P_lin(k, z)
             + c0 (k/k_norm)² P_lin
             + c2 (k/k_norm)² μ² P_lin
             + c4 (k/k_norm)² μ⁴ P_lin
             + (1/nbar)(1 + Pshot + a0 + a2 (k/k_norm)²)
```

where k_norm = 0.1 h/Mpc (Chudaykin+ convention),
f = f(z) is the logarithmic growth rate,
and

```
D_FoG(k, μ; c̃) = 1 / (1 + c̃ (k μ)⁴ / k_norm⁴)
```

is a Lorentzian-like FoG damping in the c̃ parameterization. This
captures the k⁴μ⁴ dependence that the NLO counterterm c̃ represents
in the EFT expansion.

**Cross-power (independent tracers):**
```
P^{AB}(k, μ, z) = D_FoG^{1/2}(k, μ; c̃_A) × D_FoG^{1/2}(k, μ; c̃_B)
                  × [b1_A + f μ²] [b1_B + f μ²] × P_lin(k, z)
                  + cross-counterterms (averaged)
```

Cross-stochastic = 0 for independent surveys.

**Multipoles** are obtained by Gauss-Legendre quadrature over μ:
```
P_ℓ(k) = (2ℓ+1)/2 ∫_{-1}^{1} P(k, μ) L_ℓ(μ) dμ
```
Use 20-point Gauss-Legendre (sufficient for ℓ ≤ 4).

**Limitations:** This model omits one-loop bias contributions
(b2, bG2 loop integrals). At the Fisher level, the missing terms
affect the absolute size of the power spectrum but not the
*derivatives* w.r.t. the nuisance parameters that drive the
calibration comparison. The built-in model is adequate for
weekend-1 results; the user's one-loop module refines them in
weekend 2.

---

## 5. Code structure

```
pfsfog/
├── __init__.py
├── config.py               # ForecastConfig dataclass + YAML loader
├── cosmo.py                # CAMB wrapper: H(z), D_A(z), D(z), f(z),
│                           #   sigma8(z), P_lin(k,z)
├── surveys.py              # PFS, DESI survey specs; SurveyPair;
│                           #   volume computation; z-binning
├── eft_params.py           # EFTFiducials, EFTPriors dataclasses;
│                           #   desi_elg_fiducials(), pfs_elg_fiducials(),
│                           #   broad_priors()
├── builtin_pkmu.py         # Built-in Kaiser+FoG+CT model (§4.2)
├── derivatives.py          # 5-point stencil numerical derivatives
│                           #   for auto and cross multipoles
├── covariance.py           # Gaussian multipole covariance matrix
│                           #   for single-tracer and multi-tracer
├── fisher.py               # Single-tracer Fisher matrix assembly
├── fisher_mt.py            # Multi-tracer Fisher (overlap volume)
├── prior_export.py         # Extract calibrated priors from overlap Fisher
├── fisher_full_area.py     # DESI full-area Fisher with imported priors
├── scenarios.py            # Define and run the 5 scenarios
├── plots.py                # All figures (serif, CM, ≥14pt)
└── cli.py                  # `python -m pfsfog run`

tests/
├── test_cosmo.py           # H(z), D(z), sigma8(z) against known values
├── test_builtin_pkmu.py    # P0 > 0, P2 sign flip, P4 small; multipole sum rule
├── test_derivatives.py     # convergence vs step size (decade sweep)
├── test_covariance.py      # positive definite; diagonal ≥ 0
├── test_fisher.py          # analytic Gaussian: P(k) = A k^n + 1/nbar
├── test_fisher_mt.py       # MT ≥ ST; identical tracers → 2×ST
├── test_prior_export.py    # exported σ_prior > 0; tighter than broad
├── test_scenarios.py       # ordering: σ_oracle ≤ σ_cross-cal-ext
│                           #           ≤ σ_cross-cal ≤ σ_broad
└── test_calibration_eff.py # efficiency ∈ [0, 1]

scripts/
├── run_all_scenarios.py    # end-to-end: overlap → export → full area
├── run_sensitivity.py      # vary r_σv, prior widths, kmax, overlap area
└── make_all_figures.py     # generate fig1–fig5

configs/
└── default.yaml            # default configuration
```

---

## 6. Component specifications

### 6.1 Cosmology (`cosmo.py`)

Single class `FiducialCosmology` wrapping CAMB:

```python
class FiducialCosmology:
    def __init__(self, params: dict = FIDUCIAL):
        """Initialize CAMB with fiducial params. Cache results."""
    def H(self, z: float) -> float:          # km/s/Mpc
    def D_A(self, z: float) -> float:        # Mpc/h
    def chi(self, z: float) -> float:        # comoving distance, Mpc/h
    def D(self, z: float) -> float:          # growth factor, D(0)=1
    def f(self, z: float) -> float:          # dlnD/dlna
    def sigma8(self, z: float) -> float:
    def fsigma8(self, z: float) -> float:
    def Plin(self, k: np.ndarray, z: float) -> np.ndarray:  # (Nk,)
```

### 6.2 Surveys (`surveys.py`)

```python
@dataclass
class Survey:
    name: str
    area_deg2: float
    z_bins: list[tuple[float, float]]  # [(zmin, zmax), ...]
    nbar_of_z: Callable[[float], float]
    b1_of_z: Callable[[float], float]

@dataclass
class SurveyPair:
    A: Survey                   # PFS
    B: Survey                   # DESI
    overlap_area_deg2: float    # 1,200

    def V_overlap(self, zmin: float, zmax: float,
                  cosmo: FiducialCosmology) -> float:
        """Comoving volume in overlap, h⁻³ Gpc³."""

    def V_full_B(self, zmin: float, zmax: float,
                 cosmo: FiducialCosmology) -> float:
        """Comoving volume of full DESI area, h⁻³ Gpc³."""

    def lever_arm(self, zmin: float, zmax: float,
                  cosmo: FiducialCosmology) -> float:
        """V_full_B / V_overlap ≈ 14000/1200 ≈ 12."""
```

### 6.3 Numerical derivatives (`derivatives.py`)

Five-point stencil:
```
f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h)
```

Step sizes per parameter from a config dict. Never hardcode steps
inside the Fisher routine.

Convergence test helper: sweep step h over a decade
(h_0/10, h_0/3, h_0, 3h_0, 10h_0) and verify the derivative is
stable to < 1%.

```python
def dPell_dtheta(
    pkmu_func: Callable,
    k: np.ndarray,
    z: float,
    fiducial_params: dict,
    param_name: str,
    step: float,
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Derivative of auto-power multipoles w.r.t. one parameter."""

def dPcross_dtheta(
    cross_func: Callable,
    k: np.ndarray,
    z: float,
    fiducial_params_A: dict,
    fiducial_params_B: dict,
    param_name: str,
    step: float,
    which_tracer: str,    # "A", "B", or "shared"
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Derivative of cross-power multipoles.

    For shared params (fσ8, Mν, Ωm): perturb in both A and B dicts.
    For tracer-A params (b1_A, c̃_A, ...): perturb only in A.
    For tracer-B params: perturb only in B.
    """
```

### 6.4 Covariance (`covariance.py`)

**Single-tracer Gaussian multipole covariance:**
```
Cov_{ℓℓ'}(k) = (2ℓ+1)(2ℓ'+1) / (2 N_modes(k))
               × ∫₋₁¹ dμ L_ℓ(μ) L_ℓ'(μ) [P(k,μ) + 1/nbar]²
```
where N_modes(k) = k² Δk V / (2π²), with Δk the bin width.

**Multi-tracer Gaussian covariance** for the 9-element observable
vector d(k) = {P_ℓ^AA, P_ℓ^BB, P_ℓ^AB} at each k:

```
Cov[(P_ℓ^{XY}), (P_{ℓ'}^{WZ})](k) =
    (2ℓ+1)(2ℓ'+1) / (2 N_modes(k))
    × ∫₋₁¹ dμ L_ℓ(μ) L_ℓ'(μ)
      × [P_tot^{XW}(k,μ) P_tot^{YZ}(k,μ)
         + P_tot^{XZ}(k,μ) P_tot^{YW}(k,μ)]
```
where P_tot^{XY}(k,μ) = P^{XY}(k,μ) + δ_{XY} / nbar_X.

This is a 9×9 matrix at each k (for ℓ ∈ {0,2,4} and
tracer pairs {AA, BB, AB}).

Use 20-point Gauss-Legendre for the μ integrals.

### 6.5 Single-tracer Fisher (`fisher.py`)

```
F_ab = ∫_{k_min}^{k_max} dk (k² V)/(2π²)
       × Σ_{ℓ,ℓ'} (∂P_ℓ/∂θ_a) [C⁻¹]_{ℓℓ'} (∂P_{ℓ'}/∂θ_b)
```

k_min = 0.01 h/Mpc, k_max from scenario, Δk = 0.005 h/Mpc
(trapezoidal integration over the k-grid).

```python
@dataclass
class FisherResult:
    F: np.ndarray                 # (Nparam, Nparam)
    param_names: list[str]
    z_bin: tuple[float, float]
    survey_name: str
    kmax: float

    def marginalized_sigma(self, name: str) -> float:
        """σ from (F⁻¹)_{ii}^{1/2}."""
    def conditional_sigma(self, name: str,
                          fixed: list[str]) -> float:
        """σ fixing other params."""
```

### 6.6 Multi-tracer Fisher in the overlap (`fisher_mt.py`)

Two tracers (A=PFS, B=DESI) in V_overlap(z).

Parameter vector per z-bin (27 params):
```
θ = [θ_cosmo, θ_PFS, θ_DESI]
  = [{fσ8, Mν, Ωm},
     {b1σ8, b2σ8², bG2σ8², bΓ3, c0, c2, c4, c̃, c1, Pshot, a0, a2}_PFS,
     {b1σ8, b2σ8², bG2σ8², bΓ3, c0, c2, c4, c̃, c1, Pshot, a0, a2}_DESI]
```

The Fisher matrix is:
```
F_ab = ∫_{k_min}^{k_max} dk (k² V_overlap)/(2π²)
       × [∂d/∂θ_a]ᵀ C⁻¹(k) [∂d/∂θ_b]
```
where d is the 9-element vector and C is the 9×9 covariance from
§6.4.

kmax used: 0.20 h/Mpc for both tracers in the overlap (same as
Chudaykin+ data analysis). No asymmetric kmax in the overlap
calibration step — the purpose is prior calibration, not
cosmological constraints from the overlap itself.

### 6.7 Prior export (`prior_export.py`)

From the overlap multi-tracer Fisher F_MT (27×27 per z-bin):

1. Add broad Gaussian priors to ALL parameters to regularize:
   F_regularized = F_MT + F_broad.
   F_broad is diagonal: 1/σ_broad² for each nuisance param
   (from §3.4), and a weak prior on cosmo params
   (e.g., σ(fσ8)_prior = 10, σ(Ωm)_prior = 1, σ(Mν)_prior = 5 eV).

2. Invert: C = F_regularized⁻¹.

3. Extract calibrated σ for each DESI nuisance param:
   σ_cal(θ_i) = √(C[i, i]) for i ∈ DESI nuisance indices.

4. Package as `CalibratedPriors` dataclass.

```python
@dataclass
class CalibratedPriors:
    params: dict[str, float]    # {param_name: σ_calibrated}
    z_bin: tuple[float, float]
    source: str = "PFS×DESI overlap"
```

### 6.8 Full-area DESI Fisher (`fisher_full_area.py`)

Single-tracer Fisher for DESI-ELG over V_DESI_full(z):

```
F_total(z) = F_DESI_ST(z; V_full, kmax) + F_ext_prior(z)
```

where F_ext_prior is diagonal with entries 1/σ² per nuisance
param, where σ comes from the selected scenario:

- `broad`: σ from §3.4.
- `cross-cal` / `cross-cal-ext`: σ from CalibratedPriors.
- `oracle`: σ → 0 (equivalently, fix nuisance params: set
  their Fisher rows/cols to a large number, e.g., 1/ε² with
  ε = 1e-10).

Combine z-bins: F_combined = Σ_z F_total(z) for shared cosmo
params. Per-z nuisance params do not combine across z-bins
(they are independent).

### 6.9 Scenarios (`scenarios.py`)

```python
@dataclass
class Scenario:
    name: str
    prior_source: str      # "broad", "cross-cal", "oracle"
    kmax: float            # h/Mpc

SCENARIOS = [
    Scenario("broad",          "broad",     0.20),
    Scenario("cross-cal",      "cross-cal", 0.20),
    Scenario("cross-cal-ext",  "cross-cal", 0.25),
    Scenario("oracle",         "oracle",    0.25),
]

# HOD-prior benchmark (not a scenario — just reference lines)
HOD_BENCHMARK = {
    "source": "Zhang+ 2025 (arXiv:2504.10407)",
    "sigma8_improvement": 0.23,   # 23% tighter than broad
    "Omegam_improvement": 0.04,   # 4% tighter than broad
}
FIELD_LEVEL_BENCHMARK = {
    "source": "Chudaykin+ 2026 (arXiv:2602.18554)",
    "H0_improvement": 0.40,
    "sigma8_improvement": 0.50,
}
```

For each scenario × z-bin: compute F_total, marginalize,
report σ(fσ8), σ(Mν), σ(Ωm).

---

## 7. Figures

### Fig. 1 — Overlap calibration
Per z-bin, bar chart: σ(c̃_DESI), σ(c0_DESI), σ(Pshot_DESI)
from (i) DESI-only in 1,200 deg², (ii) PFS-only in 1,200 deg²,
(iii) PFS×DESI multi-tracer. Shows the cross-calibration
tightening per EFT nuisance parameter.

### Fig. 2 — Calibrated vs broad priors
Per DESI nuisance parameter (x-axis: parameter name),
two error bars per parameter: broad prior width vs
cross-calibrated width. Per z-bin (panels or colors).

### Fig. 3 — Full-area DESI constraints (money figure)
σ(fσ8) and σ(Mν) per z-bin and combined, for the 4 Fisher
scenarios (grouped bars). Overlay horizontal dashed lines at the
Zhang+ HOD-prior benchmark (23% improvement on σ8) and the
Chudaykin+ field-level benchmark (50% on σ8) as reference.

### Fig. 4 — Calibration efficiency
Per z-bin, line plot:
(σ_broad − σ_cross-cal-ext) / (σ_broad − σ_oracle).
Include ±1σ band from the r_σv sensitivity sweep.

### Fig. 5 — Sensitivity to σ_v ratio
σ(fσ8)_combined for `cross-cal-ext` as a function of
r_σv = σ_v,PFS / σ_v,DESI ∈ [0.5, 1.0].
Marks the minimum FoG difference needed for the calibration to
beat the broad baseline by >10%.

Style: serif font, Computer Modern, fontsize ≥ 14 pt, hyphenated
survey labels (`PFS-ELG`, `DESI-ELG`, `DESI-DR1`), no chartjunk,
no gridlines on bar charts, thin axis frames.

---

## 8. Engineering requirements

- `ruff check .` clean. Type hints on all public functions.
- `pytest -q` < 60 s.
- Tests (see code tree in §5 for full list):
  - Analytic Fisher regression (Gaussian P(k) case).
  - MT ≥ ST (information monotonicity).
  - Identical tracers → F_MT = 2 × F_ST.
  - Scenario ordering: σ_oracle ≤ σ_cross-cal-ext ≤ σ_cross-cal ≤ σ_broad.
  - Calibration efficiency ∈ [0, 1].
  - Exported priors: σ_cal > 0 and σ_cal < σ_broad for at least
    the parameters that differ between tracers (c̃, Pshot).
- No globals except `FIDUCIAL`.
- Timestamped output dirs: `results/YYYYMMDD_HHMMSS/`.
- Config snapshot: copy of YAML into output dir.
- numpy-style docstrings on all public functions.
- No notebooks as source of truth: scripts produce figures.

---

## 9. Milestones

| Day | Deliverable |
|-----|-------------|
| Sat AM | Skeleton, `cosmo.py`, `surveys.py`, `eft_params.py`, `builtin_pkmu.py`; all unit tests green |
| Sat PM | `derivatives.py`, `covariance.py`, `fisher.py`; single-tracer Fisher verified against analytic |
| Sun AM | `fisher_mt.py`, `prior_export.py`; overlap calibration working; Fig. 1 on built-in model |
| Sun PM | `fisher_full_area.py`, `scenarios.py`; Figs 2–3; `summary.csv` |
| Wk2 Sat | Figs 4–5 (sensitivity); plug in user's `pkmu_module.py` if available |
| Wk2 Sun | Robustness (vary kmax, r_σv, overlap area); finalize README + `NOTES.md` for writeup |

---

## 10. Out of scope

- Non-Gaussian covariances / super-sample covariance.
- Survey window function / integral constraint.
- MCMC / posterior sampling — Fisher only.
- C++ acceleration (pure Python; forecast runs in seconds).
- Fitting to real data.
- Bispectrum (power spectrum multipoles P0, P2, P4 only).
- z > 1.6 PFS-only bins (no DESI ELG overlap).
- Realistic SBP implementation (normalizing flows with
  correlations). The HOD-prior comparison uses published
  improvement ratios, not an in-house SBP.
- Cross-covariance between overlap and non-overlap DESI volumes
  (the overlap is 9% of DESI; the cross-covariance is negligible
  for a Fisher-level forecast).

---

## 11. Handoff to paper-writing stage

`results/summary.csv` columns:
```
scenario, kmax, z_bin_min, z_bin_max, param_name,
sigma_marginalized, sigma_broad_baseline, improvement_pct,
calibration_efficiency
```

`results/priors/cross_calibrated_z{zmin}_{zmax}.json`:
```json
{
  "z_bin": [0.8, 1.0],
  "source": "PFS×DESI overlap multi-tracer Fisher",
  "priors": {
    "c_tilde": {"sigma": 142.3, "units": "[Mpc/h]^4"},
    "c0": {"sigma": 18.7, "units": "[Mpc/h]^2"},
    ...
  }
}
```

These files, together with `results/figures/fig1–fig5.pdf` and a
user-written `NOTES.md`, are the inputs for the writeup spec.
