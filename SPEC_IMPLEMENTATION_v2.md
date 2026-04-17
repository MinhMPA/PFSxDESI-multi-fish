# Implementation Spec: Multi-Tracer FoG-Cleaned Fisher Forecast for PFS

**Target executor:** Claude Code
**Owner:** N.-M. Nguyen (Kavli IPMU)
**Scope:** 1–2 weekend project. Pure Python (NumPy / SciPy / CAMB). Clean, lean, maintainable — not a production pipeline.

---

## 1. Scientific goal

### 1.1 Core idea

A FoG-cleaned sub-sample of PFS-ELG galaxies (satellites removed via
HSC photometric pre-selection or fiber-collision–induced
down-weighting) has **different bias** and **different σ_FoG** from the
full sample. These are not merely "full sample with fewer galaxies" —
they are a **physically distinct tracer** sharing the same volume.

The multi-tracer technique (Seljak 2009; McDonald & Seljak 2009;
Hamaus, Seljak & Desjacques 2012) exploits the fact that two tracers
with different bias respond differently to the same underlying density
field. Sample-variance cancellation in the ratio of the two power
spectra yields tighter constraints on quantities that couple to bias
differently — particularly **fσ8** (via the degeneracy f/b) and
**Mν** (via scale-dependent growth suppression).

The new ingredient relative to Baleato Lizancos et al. (2025) — who
proposed FoG cleaning methods but analyzed each sub-sample in isolation
— is to **jointly analyze the full and cleaned samples as a
multi-tracer pair**, combining:

1. The kmax extension from reduced σ_FoG in the cleaned sample
   (Baleato Lizancos+ argument).
2. The sample-variance cancellation from the multi-tracer combination
   (Seljak 2009 argument).
3. The redshift-dependent amplification of both effects: at high z,
   FoG compress in comoving space, the satellite fraction drops, and
   k_NL shifts to higher k, so (1) gets stronger; simultaneously, the
   bias ratio b_full/b_clean evolves with satellite fraction, keeping
   (2) effective.

### 1.2 What the forecast must show

Four layers of improvement, each additive:

| Label | kmax | Tracers | What it isolates |
|-------|------|---------|------------------|
| `full` | kmax_full(z) | full sample only | baseline |
| `clean` | kmax_clean(z) | cleaned sample only | kmax extension alone |
| `MT-same-kmax` | kmax_full(z) for both | full + clean jointly | sample-variance cancellation alone |
| `MT-ext-kmax` | kmax_full for full, kmax_clean for clean | full + clean jointly | **both effects combined** |

The headline result is the ratio σ(θ)_full / σ(θ)_MT-ext-kmax as a
function of redshift.

### 1.3 Physical arguments for high-z advantage

1. Comoving FoG scale ~ σ_v(z) · (1+z) / H(z) shrinks at high z.
2. Halo mass function drops steeply → fewer high-M halos hosting
   severe FoG at high z → satellite fraction f_sat(z) decreases.
3. k_NL(z) shifts to larger k at higher z; EFT validity range extends.
4. **New (multi-tracer specific):** As f_sat drops, b_clean/b_full → 1
   but nbar_clean/nbar_full → 1 simultaneously. The forecast must
   quantify whether the multi-tracer gain persists or fades at
   high z (it depends on which effect wins: bias-ratio shrinkage
   suppresses sample-variance cancellation, but nbar recovery reduces
   the shot-noise cost).

---

## 2. User-supplied inputs (treat as black boxes)

The user will provide, in `data/`:

### 2.1 `nz_table.csv`
Columns: `zmin, zmax, zeff, nbar_full [h^3/Mpc^3], V_survey [h^-3 Gpc^3]`.
PFS ELG only. ~4 z-bins, 0.6 ≲ z ≲ 2.4.

### 2.2 `pkmu_module.py`
A Python module exposing **at minimum** two functions:

```python
def compute_multipoles(
    k: np.ndarray,              # shape (Nk,), in h/Mpc
    z: float,                   # effective redshift of the bin
    params: dict,               # cosmology + bias + RSD + EFT params
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Return {ell: P_ell(k, z; params)} with shape (Nk,) each.
    
    For multi-tracer: call twice with tracer-specific (b1, sigma_FoG,
    EFT params). The forecast code handles the cross-power and
    covariance assembly.
    """

def compute_cross_multipoles(
    k: np.ndarray,
    z: float,
    params_A: dict,             # tracer A (full sample)
    params_B: dict,             # tracer B (cleaned sample)
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Return {ell: P^{AB}_ell(k, z)} with shape (Nk,) each.
    
    Cross-power between tracers A and B. At tree level:
    P^{AB}(k, mu) = F_FoG(k mu sigma_A) F_FoG(k mu sigma_B)
                    * (b_A + f mu^2)(b_B + f mu^2) P_lin(k, z)
    plus EFT cross-counterterms.
    
    If this function is absent, the forecast code constructs the
    cross-power from the geometric mean of auto-powers with a
    user-specified cross-correlation coefficient r_cc (default 0.95),
    and logs a warning.
    """
```

Expected `params` keys per tracer (the module declares its own
free-parameter names):

- Cosmological (shared): `fsigma8`, `Mnu`, plus fixed (`As`, `ns`,
  `Omega_m`, `h`).
- Per-tracer: `b1`, `b2`, `bG2`, `c0`, `c2`, `c4`, `Pshot`, `a0`,
  `a2`, `sigma_FoG`.

The forecast code **must not assume a specific Pkmu functional form**;
it only calls the module functions and differentiates numerically.

---

## 3. Deliverables

### 3.1 Code (Python package `pfsfog/`)

```
pfsfog/
├── __init__.py
├── config.py           # dataclasses for all config
├── cosmo.py            # minimal CAMB wrapper: H(z), chi(z), sigma8(z), fsigma8(z)
├── samples.py          # Sub-sample model (full / cleaned); TracerPair dataclass
├── fog_models.py       # Three swappable FoG-cleaning backends (§4.2)
├── kmax.py             # kmax(z) prescriptions (§4.4)
├── derivatives.py      # Numerical d P_ell / d theta with adaptive step
├── fisher.py           # Single-tracer Fisher assembly
├── fisher_mt.py        # Multi-tracer Fisher assembly (§4.5 — the new module)
├── plots.py            # Matplotlib figures (serif, CM, large fonts)
└── cli.py              # Thin CLI: `python -m pfsfog run --config cfg.yaml`

tests/
├── test_cosmo.py
├── test_derivatives.py    # verify convergence vs step size
├── test_fisher.py         # verify against analytic Gaussian case
├── test_fisher_mt.py      # verify MT Fisher against Seljak 2009 analytic limit
├── test_fog_models.py
└── test_mt_vs_single.py   # verify MT ≥ single tracer (information monotonicity)

notebooks/
└── exploration.ipynb

scripts/
├── run_kmax_scan.py
├── run_subsample_comparison.py
└── run_multitracer.py
```

### 3.2 Results directory

```
results/
├── fisher_matrices/       # .npz per (analysis_mode, kmax_prescription, z-bin)
├── figures/
│   ├── fig1_kmax_scan.pdf
│   ├── fig2_mt_decomposition.pdf
│   ├── fig3_improvement_vs_z.pdf
│   └── fig4_kmax_prescription_comparison.pdf   # stretch
└── summary.csv            # marginalized σ(θ) per configuration
```

### 3.3 Figures (required)

**Figure 1 — kmax scan (single-tracer context).** For each z-bin,
plot marginalized σ(fσ8) and σ(Mν) vs kmax swept from 0.1 to 0.4
h/Mpc, for:
- full sample (single tracer)
- cleaned sample (single tracer)
- multi-tracer at matched kmax (both tracers at same kmax)

Shows where the cleaned sub-sample overtakes the full sample, and
where the multi-tracer combination does better than either alone.

**Figure 2 — multi-tracer decomposition.** At fiducial kmax(z), bar
chart of σ(fσ8) / σ(Mν) per z-bin for the four analysis modes in
the table (§1.2). This is the money figure: it decomposes the total
improvement into the kmax-extension contribution and the
sample-variance-cancellation contribution. Use grouped bars with
four colors.

**Figure 3 — improvement ratio vs redshift.** Continuous curves of
σ(fσ8)_baseline / σ(fσ8)_X for X ∈ {clean, MT-same-kmax,
MT-ext-kmax}, as a function of z. Right y-axis: f_sat(z) and
b_clean/b_full(z) to show which physical quantities drive the
redshift evolution.

**Figure 4 — kmax prescription comparison (stretch).** The kmax(z)
prescriptions plotted as curves on (z, kmax) axes, with a second
panel showing the resulting combined σ(fσ8) for the MT-ext-kmax
configuration.

All figures: serif font, Computer Modern, font size ≥ 14 pt,
hyphenated survey labels (`PFS-ELG`), no chartjunk.

---

## 4. Component specifications

### 4.1 Cosmology (`cosmo.py`)

Unchanged from previous spec. Single class `FiducialCosmology`
wrapping CAMB. Planck 2018 TT,TE,EE+lowE, `Mnu = 0.06 eV`. All
fiducial numbers in a module-level `FIDUCIAL` dict.

### 4.2 Sub-sample / FoG models (`samples.py`, `fog_models.py`)

Two sample types: `full` and `cleaned`, each a `Sample` dataclass
with fields `name`, `nbar_of_z`, `sigma_FoG_of_z`, `b1_of_z`,
`V_of_z`.

New: `TracerPair` dataclass bundling two `Sample` objects sharing the
same volume:

```python
@dataclass
class TracerPair:
    full: Sample
    clean: Sample
    
    def bias_ratio(self, z: float) -> float:
        """b1_clean / b1_full. Drives multi-tracer gain."""
        return self.clean.b1_of_z(z) / self.full.b1_of_z(z)
    
    def nbar_ratio(self, z: float) -> float:
        """nbar_clean / nbar_full. Shot-noise cost of cleaning."""
        return self.clean.nbar_of_z(z) / self.full.nbar_of_z(z)
```

Three swappable FoG backends implementing `FogBackend` protocol
(unchanged):

1. **`AnalyticFog`** — reduce σ_FoG by factor `r_sigma`; reduce nbar
   by `cleaning_frac`. **New:** also compute b1_clean from
   number-weighted central/satellite mix:
   `b1_clean = [(1-f_s^clean)*b1_cen + f_s^clean*b1_sat]`.
   This requires `b1_cen`, `b1_sat` in the config, with defaults
   derived from published ELG HOD fits.

2. **`HodFog`** — two-component model. Centrals: (b1_cen, σ_FoG_cen).
   Satellites: (b1_sat, σ_FoG_sat, f_sat(z)). Cleaning = remove
   satellites with efficiency η. Returns (nbar_clean, σ_FoG_clean,
   b1_clean) as functions of z. **Critical for multi-tracer:** the
   bias split is physical, not a free knob.

3. **`EmpiricalFog`** — reads user table with columns
   `zeff, sample_tag, nbar, sigma_FoG, b1`. Pure lookup.

### 4.3 Numerical derivatives (`derivatives.py`)

Five-point stencil. Step sizes from config dict with convergence test
helper. Unchanged except: must handle derivatives of cross-power
spectra w.r.t. shared and per-tracer parameters.

```python
def dPell_dtheta(
    pkmu_module,
    k: np.ndarray,
    z: float,
    fiducial_params: dict,
    param: str,
    step: float,
    ells: tuple,
) -> dict[int, np.ndarray]:
    """Single-tracer derivative."""

def dPcross_dtheta(
    pkmu_module,
    k: np.ndarray,
    z: float,
    fiducial_params_A: dict,
    fiducial_params_B: dict,
    param: str,
    step: float,
    which_tracer: str,   # "A", "B", or "shared"
    ells: tuple,
) -> dict[int, np.ndarray]:
    """Cross-power derivative.
    
    For shared params (fsigma8, Mnu): perturb in both A and B.
    For tracer-specific params (b1_A, sigma_FoG_A): perturb only in A.
    """
```

### 4.4 kmax prescriptions (`kmax.py`)

Unchanged. Three callables plus user-table option. The key change is
that `kmax` is now **per-tracer**: the full sample uses kmax_full(z),
the cleaned sample uses kmax_clean(z) ≥ kmax_full(z). The
multi-tracer Fisher must handle the asymmetric k-range (see §4.5).

### 4.5 Multi-tracer Fisher assembly (`fisher_mt.py`)

**This is the new module and the core of the project.**

Two tracers A (full) and B (cleaned) sharing volume V(z). The
observable vector at each k is:

```
d(k) = [P_0^AA(k), P_2^AA(k), P_4^AA(k),
         P_0^BB(k), P_2^BB(k), P_4^BB(k),
         P_0^AB(k), P_2^AB(k), P_4^AB(k)]
```

The parameter vector is partitioned:

```
θ = [θ_shared, θ_A, θ_B]
  = [{fsigma8, Mnu},
     {b1_A, σ_FoG_A, c0_A, c2_A, c4_A, Pshot_A, a0_A, a2_A},
     {b1_B, σ_FoG_B, c0_B, c2_B, c4_B, Pshot_B, a0_B, a2_B}]
```

The Gaussian covariance matrix for the observable vector at each k is
the 9×9 block matrix (for ℓ, ℓ' ∈ {0,2,4} and tracers X, Y ∈
{A, B}):

```
Cov[(P_ℓ^XY, P_ℓ'^WZ)](k) = 
    (2ℓ+1)(2ℓ'+1) / (2 N_modes(k))
    × ∫ dμ L_ℓ(μ) L_ℓ'(μ)
      × [P_tot^XW(k,μ) P_tot^YZ(k,μ) + P_tot^XZ(k,μ) P_tot^YW(k,μ)]
```

where `P_tot^XY(k,μ) = P^XY(k,μ) + δ_XY / nbar_X` (noise only on
auto-spectra).

**Asymmetric kmax handling.** The multi-tracer data vector has three
blocks: AA, BB, AB. When kmax_A < kmax_B:
- For k ≤ kmax_A: use all 9 elements (AA + BB + AB).
- For kmax_A < k ≤ kmax_B: use only 3 elements (BB auto-spectra).
  The full sample provides no information above kmax_A, and the
  cross-power is undefined above the shorter tracer's kmax.

Implement this by integrating the Fisher in two segments:

```python
# Segment 1: k_min to kmax_A — full multi-tracer
F_mt_low = integrate_fisher_mt(k_min, kmax_A, d=[AA, BB, AB], ...)

# Segment 2: kmax_A to kmax_B — single tracer B only
F_st_high = integrate_fisher_st(kmax_A, kmax_B, tracer=B, ...)

# Combine (zero-pad F_st_high for θ_A parameters)
F_total = F_mt_low + pad(F_st_high)
```

**Multi-tracer Fisher in compact form:**

```
F_ab = ∫ dk k²V/(2π²) × [∂d/∂θ_a]^T C^{-1}(k) [∂d/∂θ_b]
```

where `∂d/∂θ_a` is a 9-vector (or 3-vector in segment 2) and
`C^{-1}` is the corresponding inverse covariance block.

Expose:

```python
@dataclass
class MultiTracerFisherResult:
    F: np.ndarray
    param_names: list[str]
    param_partition: dict[str, list[str]]  # {"shared": [...], "A": [...], "B": [...]}
    z_bins: list[float]
    analysis_mode: str                     # "full", "clean", "MT-same-kmax", "MT-ext-kmax"
    kmax_A: dict[float, float]             # {z: kmax} for tracer A
    kmax_B: dict[float, float]             # {z: kmax} for tracer B

    def marginalized_sigma(self, name: str) -> float: ...
    def improvement_over(self, baseline: 'MultiTracerFisherResult', name: str) -> float: ...
```

### 4.6 Single-tracer Fisher (`fisher.py`)

Retained from previous spec, unchanged. Used for the `full` and
`clean` single-tracer baselines.

### 4.7 Configuration (`config.py`)

```yaml
cosmology:
  fiducial: planck18
parameters:
  shared: [fsigma8, Mnu]
  per_tracer: [b1, sigma_FoG, c0, c2, c4, Pshot, a0, a2]
  steps:
    fsigma8: 0.01
    Mnu: 0.02
    b1: 0.05
    sigma_FoG: 10.0     # km/s
    c0: 0.5
    c2: 0.5
    c4: 0.5
    Pshot: 100.0
    a0: 100.0
    a2: 100.0
samples:
  full: {source: "data/nz_table.csv"}
  cleaned:
    backend: hod            # HodFog is the default for Path B
    hod:
      b1_cen: {z0: 1.2, z1: 1.5, z2: 1.8, z3: 2.2}
      b1_sat: {z0: 1.0, z1: 1.3, z2: 1.6, z3: 2.0}
      sigma_v_cen: 70.0     # km/s
      sigma_v_sat: 450.0    # km/s
      f_sat: {z0: 0.25, z1: 0.20, z2: 0.15, z3: 0.10}
      eta: 0.7               # satellite removal efficiency
analysis_modes:
  - full                    # single tracer, kmax_full
  - clean                   # single tracer, kmax_clean
  - MT-same-kmax            # multi-tracer, both at kmax_full
  - MT-ext-kmax             # multi-tracer, tracer-specific kmax
kmax:
  prescriptions: [sigma_v, knl, fixed]
  sigma_v: {alpha: 1.0}
  knl: {beta: 0.5}
  fixed: {value: 0.20}
observables:
  ells: [0, 2, 4]
  k_min: 0.005
  k_max_grid: 0.45
  n_k: 200
output:
  dir: results/
```

---

## 5. Engineering requirements

- **Lint clean**: `ruff check .` passes. Type hints on all public
  functions.
- **Tests**: `pytest -q` runs in < 60 s. Include:
  - Gaussian-analytic Fisher regression test (single tracer).
  - Multi-tracer sanity: for two tracers with identical bias,
    MT Fisher = 2 × single-tracer Fisher (additive volumes).
  - Multi-tracer sanity: MT Fisher ≥ single-tracer Fisher for
    any bias split (information monotonicity).
  - Seljak (2009) analytic limit: for two tracers in the
    nbar → ∞ limit, verify that σ(f/b) from the ratio
    P_A/P_B matches the analytic expression.
- **No globals** except `FIDUCIAL`.
- **Reproducibility**: every CLI run writes
  `results/<timestamp>/config.yaml`.
- **Plotting style**: serif, CM, fontsize 14, hyphenated labels.
- **Docstrings**: numpy-style on all public functions.
- **No notebooks as source of truth**.

---

## 6. Milestones

| Day | Deliverable |
|-----|-------------|
| Sat AM | Skeleton, `cosmo.py`, `FIDUCIAL`, CAMB; `samples.py` with `TracerPair`; tests green |
| Sat PM | `derivatives.py` incl. cross-power derivatives; dummy `pkmu_module` for CI |
| Sun AM | `fisher.py` (single-tracer); `HodFog` backend; Figure 1 on dummy data |
| Sun PM | `fisher_mt.py` with asymmetric kmax; MT sanity tests; Figure 2 on dummy data |
| Wk2 Sat | Plug in user's `pkmu_module` + `nz_table.csv`; Figures 1–3 with real inputs |
| Wk2 Sun | Figure 4, `summary.csv`, robustness (AnalyticFog/EmpiricalFog), README |

---

## 7. Out of scope (explicitly)

- Non-Gaussian covariances / super-sample covariance.
- Survey window / integral constraint.
- BAO / RSD separation (full-shape analysis only).
- MCMC / posterior sampling — Fisher only.
- C++ acceleration.
- Fitting to real data.
- Cross-covariance between the full and cleaned samples from
  overlapping galaxies (the cleaned sample is a strict subset of
  the full sample; see §8 for why we ignore this at Fisher level
  and what the caveat is).

---

## 8. Important modeling note: overlapping samples

The cleaned sample is a **subset** of the full sample. In a real
analysis, this induces cross-covariance between the auto-spectra of
the two "tracers" beyond what the multi-tracer Gaussian covariance
captures.

At the Fisher level, the standard multi-tracer formalism (McDonald
& Seljak 2009; Hamaus, Seljak & Desjacques 2012) already accounts
for this through the cross-power P^{AB}: when tracer B is a subset
of tracer A, the cross-correlation coefficient r_cc → 1, and the
multi-tracer gain comes entirely from the **bias difference**, not
from independent sampling.

The forecast is correct to the extent that:
1. The cross-power P^{AB} is accurately modeled (guaranteed by the
   user-supplied `pkmu_module`).
2. The covariance is Gaussian.

In practice, non-Gaussian covariance from the overlap would
**reduce** the multi-tracer gain somewhat. This is a known caveat
and should be stated in the paper. The direction of the bias is
conservative: if the forecast shows an improvement, the real
improvement is a lower bound.

Flag this in `NOTES.md` and ensure the writeup spec references it.

---

## 9. Handoff to the paper-writing stage

The CLI writes `results/summary.csv` and figures in
`results/figures/`. The `summary.csv` must include columns:
`analysis_mode, kmax_prescription, z_bin, param_name,
sigma_marginalized, sigma_conditional, improvement_factor`.

These, together with `NOTES.md`, are the handoff artifacts for
`SPEC_WRITEUP.md`.
