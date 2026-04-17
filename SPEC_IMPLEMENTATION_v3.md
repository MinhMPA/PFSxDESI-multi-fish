# Implementation Spec: Cross-Calibration of High-k Nuisance Parameters from PFS × DESI Overlap

**Target executor:** Claude Code
**Owner:** N.-M. Nguyen (Kavli IPMU)
**Scope:** 1–2 weekend project. Pure Python (NumPy / SciPy / CAMB). Clean, lean, maintainable.

---

## 1. Scientific goal

### 1.1 Core idea

PFS and DESI observe overlapping ELG samples in ~1,200 deg² at
0.8 < z < 1.6. These are genuinely independent tracers of the same
cosmic density field — different magnitude limits, different [OII]
flux thresholds, different satellite fractions, different bias,
different σ_FoG.

In this overlap volume, the joint analysis of P^{PFS}_{auto},
P^{DESI}_{auto}, and P^{PFS×DESI}_{cross} over-determines the
per-tracer nuisance parameters (σ_FoG, EFT counterterms, stochastic
terms). The resulting tight constraints on these nuisance parameters
can then be **exported as informative priors** to DESI's full
14,000 deg² footprint, where PFS data are unavailable.

This "calibration laboratory" framing converts a modest
multi-tracer gain in 1,200 deg² into a large effective gain across
DESI's entire survey area, by reducing the marginalization penalty
from nuisance parameters at high kmax.

### 1.2 Three layers of the argument

1. **Multi-tracer in the overlap** (Seljak 2009 mechanism):
   sample-variance cancellation from the bias difference between
   PFS-ELG and DESI-ELG. Modest gain on its own (1,200 deg² is
   ~9% of DESI).

2. **σ_FoG cross-calibration** (new): the cross-power damping
   F_FoG(kμσ_PFS) × F_FoG(kμσ_DESI) plus the two auto-power
   dampings jointly constrain both σ_FoG values. This
   self-calibration from data — not from a phenomenological model
   — yields a tight σ_FoG prior for DESI-ELG.

3. **Prior export to full DESI area** (the lever arm): the
   calibrated σ_FoG (and counterterm) priors from the overlap
   propagate to DESI's 14,000 deg². Tighter nuisance priors →
   less marginalization penalty → higher effective kmax → better
   σ(fσ8) and σ(Mν) from the full survey.

### 1.3 What the forecast must show

Five analysis scenarios for DESI-ELG over 14,000 deg²:

| Label | σ_FoG prior | CT priors | kmax | Source |
|-------|-------------|-----------|------|--------|
| `baseline` | broad | broad | conservative | DESI alone, standard |
| `self-cal` | from overlap Fisher | from overlap Fisher | conservative | + overlap calibration |
| `self-cal-ext` | from overlap Fisher | from overlap Fisher | extended | + push kmax with calibrated priors |
| `oracle` | perfect (fixed) | perfect (fixed) | extended | upper bound: no nuisance cost |
| `overlap-only` | — | — | — | multi-tracer Fisher in 1,200 deg² only |

The headline results:
- σ(fσ8) and σ(Mν) for each scenario, per z-bin and combined.
- The **prior export ratio**: σ(fσ8)_baseline / σ(fσ8)_self-cal-ext.
  This quantifies how much the 1,200 deg² overlap improves DESI's
  full-area constraints.
- The **calibration efficiency**: how close self-cal-ext gets to
  oracle. This measures how much of the nuisance-parameter
  marginalization cost is recoverable from cross-calibration.

### 1.4 Transferability assumption

The calibration transfers from overlap to full area iff DESI-ELG
in the overlap have the same HOD as DESI-ELG outside. Since DESI's
target selection is uniform across the footprint, this holds by
construction. State this assumption explicitly; it is the one
thing a referee will ask about.

---

## 2. User-supplied inputs (treat as black boxes)

The user will provide, in `data/`:

### 2.1 `survey_table.csv`
Columns: `zmin, zmax, zeff, nbar_PFS, nbar_DESI, V_overlap, V_DESI_full`
Units: nbar in h³/Mpc³, V in h⁻³ Gpc³.
~4 z-bins over 0.8 ≤ z ≤ 1.6 (the PFS-DESI ELG overlap range).
Additional z-bins at 1.6 < z < 2.4 for PFS-only (no DESI ELG).

### 2.2 `pkmu_module.py`
Same contract as v2 spec. Must expose:

```python
def compute_multipoles(
    k: np.ndarray,
    z: float,
    params: dict,
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Auto-power multipoles for a single tracer."""

def compute_cross_multipoles(
    k: np.ndarray,
    z: float,
    params_A: dict,
    params_B: dict,
    ells: tuple = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Cross-power multipoles between tracers A and B.

    The cross-FoG damping is the product:
    F_FoG(kμσ_A) × F_FoG(kμσ_B)
    which is distinct from either auto-FoG.

    Cross-counterterms: at tree level, c0^{AB} = (c0^A + c0^B)/2
    (or a user-specified relation). Document the convention.
    """
```

Fallback: if `compute_cross_multipoles` is absent, construct
P^{AB}_ℓ from geometric mean of auto-powers with cross-correlation
coefficient r_cc (default 0.95), log a warning.

### 2.3 `tracer_properties.csv`
Per z-bin, per tracer: `zeff, tracer, b1, b2, bG2, sigma_FoG,
f_sat, c0, c2, c4, Pshot, a0, a2`. Fiducial values for PFS-ELG
and DESI-ELG.

Sources: DESI-ELG from DR1 full-shape HOD-informed fits
(Zhang, Bonici, D'Amico+ 2025). PFS-ELG from PFS survey paper
(Takada+ 2014) or user-supplied HOD modeling.

---

## 3. Deliverables

### 3.1 Code (Python package `pfsfog/`)

```
pfsfog/
├── __init__.py
├── config.py               # dataclasses for all config
├── cosmo.py                # minimal CAMB wrapper
├── surveys.py              # Survey and SurveyPair dataclasses
├── tracer_properties.py    # per-tracer fiducial params, bias, sigma_FoG
├── kmax.py                 # kmax prescriptions (§4.3)
├── derivatives.py          # numerical derivatives (auto + cross)
├── fisher.py               # single-tracer Fisher
├── fisher_mt.py            # multi-tracer Fisher in overlap volume
├── prior_export.py         # extract nuisance priors from overlap Fisher (§4.5)
├── fisher_full_area.py     # DESI full-area Fisher with imported priors (§4.6)
├── plots.py                # matplotlib figures
└── cli.py                  # thin CLI

tests/
├── test_cosmo.py
├── test_derivatives.py
├── test_fisher.py
├── test_fisher_mt.py
├── test_prior_export.py    # verify prior extraction is self-consistent
├── test_oracle.py          # oracle ≥ self-cal-ext ≥ baseline (monotonicity)
└── test_transferability.py # identical tracer → identical priors

scripts/
├── run_overlap_calibration.py
├── run_full_area_comparison.py
└── run_calibration_efficiency.py
```

### 3.2 Results directory

```
results/
├── fisher_matrices/
│   ├── overlap/             # per z-bin multi-tracer Fisher
│   ├── full_area/           # per scenario per z-bin
│   └── priors/              # extracted nuisance priors (.json)
├── figures/
│   ├── fig1_overlap_mt.pdf
│   ├── fig2_prior_export.pdf
│   ├── fig3_full_area_comparison.pdf
│   ├── fig4_calibration_efficiency.pdf
│   └── fig5_kmax_vs_z.pdf           # stretch
└── summary.csv
```

### 3.3 Figures (required)

**Figure 1 — Multi-tracer in the overlap.** For each z-bin in the
overlap (0.8–1.6), show marginalized σ(σ_FoG^DESI) and σ(c0^DESI)
from:
- DESI auto-power only (in 1,200 deg²)
- PFS auto-power only (in 1,200 deg²)
- PFS × DESI multi-tracer (auto + auto + cross)
Demonstrates that the cross-calibration tightens nuisance priors
well beyond what either survey achieves alone in the same volume.

**Figure 2 — Prior export: from overlap to full area.** Show the
marginalized σ(σ_FoG^DESI) prior extracted from the overlap
multi-tracer Fisher, compared against the broad prior used in the
DESI baseline. Visualize as error bars or a 1D posterior width
comparison per z-bin. This is the "calibration product."

**Figure 3 — Full-area DESI comparison (money figure).** Bar chart
of σ(fσ8) and σ(Mν) per z-bin and combined, for the five scenarios
in §1.3. Group bars by scenario, one panel per parameter. Shows
the cascade: baseline → self-cal → self-cal-ext → oracle.

**Figure 4 — Calibration efficiency.** Per z-bin, plot the ratio
[σ_baseline − σ_self-cal-ext] / [σ_baseline − σ_oracle].
This is the fraction of the nuisance-marginalization cost that is
recovered by cross-calibration. If close to 1, the overlap
provides nearly perfect nuisance calibration; if close to 0, the
nuisance parameters are not the bottleneck.

**Figure 5 — kmax(z) with and without calibration (stretch).**
Plot kmax_conservative(z) and kmax_extended(z) on (z, kmax) axes.
Show where the calibrated priors allow pushing to higher k.

All figures: serif, CM, fontsize ≥ 14 pt, hyphenated labels
(`PFS-ELG`, `DESI-ELG`).

---

## 4. Component specifications

### 4.1 Cosmology (`cosmo.py`)

Unchanged. Planck 2018 fiducial, Mnu = 0.06 eV.

### 4.2 Survey model (`surveys.py`)

```python
@dataclass
class Survey:
    name: str                     # "PFS-ELG" or "DESI-ELG"
    nbar_of_z: Callable           # h³/Mpc³
    b1_of_z: Callable
    sigma_FoG_of_z: Callable      # km/s
    V_of_z: Callable              # h⁻³ Gpc³
    z_range: tuple[float, float]

@dataclass
class SurveyPair:
    """Two surveys sharing an overlap volume."""
    A: Survey                     # PFS-ELG
    B: Survey                     # DESI-ELG
    V_overlap_of_z: Callable      # shared volume (1,200 deg²)
    V_B_full_of_z: Callable       # DESI total volume (14,000 deg²)
    z_overlap: tuple[float, float]  # (0.8, 1.6)

    def area_ratio(self) -> float:
        """V_B_full / V_overlap ≈ 12. The lever arm."""
        ...
```

### 4.3 kmax prescriptions (`kmax.py`)

Per-tracer kmax. Same three prescriptions as before, but now
evaluated with tracer-specific σ_FoG:

1. `kmax_sigma_v(tracer, z)`: α / σ_v_comoving(tracer, z).
2. `kmax_knl(z)`: β × k_NL(z). Shared — depends on matter, not tracer.
3. `kmax_fixed`: 0.20 h/Mpc reference.

Additionally, a **calibration-informed prescription**:
4. `kmax_calibrated(tracer, z, sigma_FoG_prior)`: same as (1) but
   uses the *calibrated* σ_FoG from the overlap Fisher, not the
   fiducial. This is the kmax used in the `self-cal-ext` scenario.

### 4.4 Multi-tracer Fisher in the overlap (`fisher_mt.py`)

Same formalism as v2 spec. Two tracers A (PFS) and B (DESI) in
V_overlap. Observable vector at each k:

```
d(k) = [P_0^AA, P_2^AA, P_4^AA,
         P_0^BB, P_2^BB, P_4^BB,
         P_0^AB, P_2^AB, P_4^AB]
```

Parameter vector:

```
θ = [θ_shared, θ_PFS, θ_DESI]
  = [{fsigma8, Mnu},
     {b1_P, σ_FoG_P, c0_P, c2_P, c4_P, Pshot_P, a0_P, a2_P},
     {b1_D, σ_FoG_D, c0_D, c2_D, c4_D, Pshot_D, a0_D, a2_D}]
```

Asymmetric kmax handling as in v2: below min(kmax_PFS, kmax_DESI)
use all 9 elements; between the two kmax values use only the
longer-reach tracer's auto-spectrum.

Cross-counterterm relation: at tree level,
c0^{AB} = (c0^A b1^B + c0^B b1^A) / (b1^A + b1^B).
User can override this via `pkmu_module.compute_cross_multipoles`.

### 4.5 Prior extraction (`prior_export.py`)

**This is the novel module.**

From the overlap multi-tracer Fisher F_overlap (Nparam × Nparam),
extract the marginalized posterior width on each DESI nuisance
parameter. These become Gaussian priors for the full-area analysis:

```python
@dataclass
class CalibratedPriors:
    """Nuisance priors derived from overlap cross-calibration."""
    params: dict[str, float]      # {param_name: sigma_prior}
    source_z_bins: list[float]
    source_volume: float          # V_overlap in h⁻³ Gpc³

    @classmethod
    def from_overlap_fisher(
        cls,
        F_overlap: MultiTracerFisherResult,
        target_tracer: str,       # "DESI"
        nuisance_params: list[str],
    ) -> 'CalibratedPriors':
        """Extract marginalized σ for DESI nuisance params.

        Procedure:
        1. From F_overlap, compute C = F⁻¹ (full parameter covariance).
        2. Extract diagonal elements for DESI nuisance params.
        3. σ_prior(param) = sqrt(C[param, param]).

        These are the marginalized constraints on DESI nuisance
        parameters from the overlap multi-tracer analysis, which
        already accounts for degeneracies with cosmological params
        and PFS nuisance params.
        """
        ...
```

**z-bin handling.** EFT nuisance parameters are per-z-bin. The
overlap provides calibrated priors for z-bins in [0.8, 1.6]. For
DESI z-bins outside this range (if any), no calibration is
available — use the broad baseline prior. State this clearly.

**Cross-z-bin correlations.** The simplest approach (and the one
for the weekend project) treats each z-bin independently. A more
sophisticated version would model the z-dependence of σ_FoG(z)
parametrically and calibrate the parameters. Flag this as a
possible extension in NOTES.md.

### 4.6 Full-area DESI Fisher with imported priors (`fisher_full_area.py`)

Single-tracer Fisher for DESI-ELG over V_DESI_full(z), with
the prior matrix added:

```
F_DESI_total(z) = F_DESI_auto(z; V_full, kmax) + F_prior(z)
```

where F_prior is a diagonal matrix with entries
1/σ_prior(param)² for each nuisance parameter, derived from
CalibratedPriors.

**Five scenarios implemented as configurations:**

```python
@dataclass
class FullAreaScenario:
    name: str
    kmax_prescription: str
    prior_source: str            # "broad", "overlap", "oracle"

SCENARIOS = [
    FullAreaScenario("baseline",      "fixed",      "broad"),
    FullAreaScenario("self-cal",      "fixed",      "overlap"),
    FullAreaScenario("self-cal-ext",  "calibrated", "overlap"),
    FullAreaScenario("oracle",        "calibrated", "oracle"),
    # overlap-only is handled separately by fisher_mt.py
]
```

For "broad" priors: σ_prior(σ_FoG) = 200 km/s, σ_prior(c0) = 10,
etc. — effectively uninformative. User-configurable in the YAML.

For "oracle" priors: fix all nuisance parameters (infinite prior
precision). This gives the theoretical maximum for each kmax.

Expose:

```python
@dataclass
class FullAreaResult:
    scenario: FullAreaScenario
    F: np.ndarray
    param_names: list[str]
    z_bins: list[float]

    def marginalized_sigma(self, name: str) -> float: ...
    def calibration_efficiency(self, baseline: 'FullAreaResult',
                                oracle: 'FullAreaResult',
                                param: str) -> float:
        """(σ_baseline - σ_self) / (σ_baseline - σ_oracle)."""
        ...
```

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
    sigma_FoG: 10.0
    c0: 0.5
    c2: 0.5
    c4: 0.5
    Pshot: 100.0
    a0: 100.0
    a2: 100.0
surveys:
  PFS:
    source: "data/survey_table.csv"
    tracer_properties: "data/tracer_properties.csv"
  DESI:
    source: "data/survey_table.csv"
    tracer_properties: "data/tracer_properties.csv"
    V_full_area: "data/desi_full_volume.csv"
overlap:
  z_range: [0.8, 1.6]
  area_deg2: 1200
priors:
  broad:
    sigma_FoG: 200.0
    c0: 10.0
    c2: 10.0
    c4: 10.0
    Pshot: 1000.0
    a0: 1000.0
    a2: 1000.0
scenarios:
  - baseline
  - self-cal
  - self-cal-ext
  - oracle
  - overlap-only
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
- **Tests** (< 60 s total):
  - Single-tracer Fisher against analytic Gaussian.
  - Multi-tracer: identical tracers → F_MT = 2 × F_ST.
  - Multi-tracer ≥ single-tracer (information monotonicity).
  - Prior export: extracted prior ≥ 0 (positive definite).
  - Scenario ordering: σ_oracle ≤ σ_self-cal-ext ≤ σ_self-cal ≤ σ_baseline.
  - Calibration efficiency ∈ [0, 1].
- **No globals** except `FIDUCIAL`.
- **Reproducibility**: timestamped output dirs with config snapshot.
- **Plotting style**: serif, CM, fontsize 14, hyphenated labels.
- **Docstrings**: numpy-style on all public functions.

---

## 6. Milestones

| Day | Deliverable |
|-----|-------------|
| Sat AM | Skeleton, `cosmo.py`, `surveys.py`, `SurveyPair`; tests green |
| Sat PM | `derivatives.py` (auto + cross); `fisher.py` + `fisher_mt.py`; dummy pkmu |
| Sun AM | `prior_export.py`; `fisher_full_area.py`; Figure 1 on dummy data |
| Sun PM | Figures 2–4 on dummy data; scenario comparison working end-to-end |
| Wk2 Sat | Plug in user's `pkmu_module` + survey tables; Figures 1–4 with real inputs |
| Wk2 Sun | Figure 5, `summary.csv`, robustness checks (vary η, prior widths), README |

---

## 7. Out of scope

- Non-Gaussian covariances / super-sample covariance.
- Survey window / integral constraint.
- BAO-only or template-fit analysis — full-shape only.
- MCMC / posterior sampling — Fisher only.
- C++ acceleration.
- Fitting to real data.
- Modeling the z-dependence of nuisance parameters across bins
  (each z-bin calibrated independently).
- PFS-only z-bins at z > 1.6 (no DESI ELG overlap; these enter
  only as additional single-tracer Fisher matrices if desired).

---

## 8. Important modeling notes

### 8.1 Transferability

The calibration transfer assumes DESI-ELG outside the overlap has
the same HOD (and therefore the same σ_FoG, counterterms) as
DESI-ELG inside. This holds because DESI's target selection is
spatially uniform. The assumption would break if there were
significant spatial variation in the ELG population (e.g., from
imaging systematics), but this is separately controlled in DESI's
pipeline. State as an explicit assumption in the paper.

### 8.2 What the cross-power actually constrains

The cross-power P^{AB}(k, μ) contains the product of FoG
suppressions F(kμσ_A) × F(kμσ_B). Combined with the two
auto-powers (each containing F² for one tracer), the system of
three spectra at each (k, μ) yields:

  P^{AA} ∝ F(kμσ_A)² × (b_A + fμ²)²
  P^{BB} ∝ F(kμσ_B)² × (b_B + fμ²)²
  P^{AB} ∝ F(kμσ_A) F(kμσ_B) × (b_A + fμ²)(b_B + fμ²)

At fixed k and μ, these three equations in two unknowns (σ_A, σ_B)
plus the shared f(z) and P_lin(k) over-determine the FoG
parameters. The Fisher matrix captures this algebraically via the
cross-derivatives.

### 8.3 Conservative direction

The forecast uses Gaussian covariance. Non-Gaussian contributions
(trispectrum, super-sample) would weaken the calibration — making
the overlap priors broader and the full-area improvement smaller.
The forecast is therefore an upper bound on the gain. State this.

---

## 9. Handoff

`results/summary.csv` columns: `scenario, kmax_prescription,
z_bin, param_name, sigma_marginalized, sigma_conditional,
improvement_vs_baseline, calibration_efficiency`.

Plus `results/priors/*.json` containing the extracted nuisance
priors per z-bin, usable as inputs to any downstream Fisher or
MCMC analysis.
