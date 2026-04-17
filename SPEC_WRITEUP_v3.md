# Writeup Spec: Cross-Calibrating High-k Nuisance Parameters from PFS × DESI Overlap

**Target executor:** Claude Opus 4.7 (chat interface, not Claude Code)
**Owner:** N.-M. Nguyen
**Combines with:** results/ directory produced by `SPEC_IMPLEMENTATION_v3.md`
**Intended venue:** Physical Review D (regular article, ~8–10 pages
two-column). Editor's Suggestion ambition.

---

## 1. Inputs Opus will receive

1. `SPEC_IMPLEMENTATION_v3.md`.
2. `results/summary.csv`.
3. `results/figures/fig1_overlap_mt.pdf` through `fig5_kmax_vs_z.pdf`.
4. `results/priors/*.json` (calibrated nuisance priors per z-bin).
5. `NOTES.md` — user bullet notes on:
   - the prior export ratio per z-bin and combined,
   - which nuisance parameter dominates the marginalization cost
     (σ_FoG? counterterms? stochastic?),
   - calibration efficiency: how close to oracle,
   - sensitivity to broad-prior width and to kmax prescription,
   - the redshift where the gain peaks,
   - references to cite.
6. Optional: `refs.bib`.

---

## 2. Paper goal and one-sentence thesis

> The ~1,200 deg² overlap between PFS and DESI ELG samples
> provides a multi-tracer calibration laboratory: joint analysis
> of auto- and cross-power spectra in the overlap volume
> constrains per-tracer FoG and EFT nuisance parameters
> [X]× more tightly than either survey alone, and exporting
> these calibrated priors to DESI's full 14,000 deg² footprint
> improves constraints on fσ8 by [Y]% and on Mν by [Z]%
> relative to the uncalibrated baseline.

(Fill [X], [Y], [Z] from `summary.csv`. Do not invent.)

---

## 3. Positioning relative to prior work

### What exists

**Multi-tracer technique:** Seljak (2009), McDonald & Seljak
(2009), Hamaus, Seljak & Desjacques (2012). Sample-variance
cancellation from distinct tracers in the same volume. Applied
to luminosity/color splits within a single survey (Blake+ 2013,
Ross+ 2014, Beutler+ 2016) and to DESI LRG × ELG (DESI DR1
combined-tracer BAO).

**FoG cleaning:** Baleato Lizancos, Seljak, Karamanis, Bonici &
Ferraro (2025, JCAP). Data-driven cleaning via quadrupole
zero-crossing and tSZ-based satellite identification. Analyzed
cleaned sub-samples in isolation.

**Multi-tracer with tracer-specific kmax:** Rubira & Conteddu
(2025, arXiv:2504.18245). Fisher forecast showing that
sub-samples with different FoG scales can use different kmax.
Applied to sub-splits within a single survey.

**Cross-survey Fisher forecasts:** Qin, Wang & Zhao (2025,
arXiv:2505.04275) forecast J-PAS + PFS but explicitly avoid the
overlap volume, summing Fisher matrices without cross-correlations.
DESI DR1 combined-tracer analysis uses LRG × ELG cross-power for
BAO, not full-shape EFT.

**tSZ-based splitting of DESI galaxies:** Rashkovetskyi,
Eisenstein+ (2025, OJAp). Demonstrated environment-dependent
clustering from tSZ splitting.

### What is new here

No prior work uses the overlap volume of two independent
spectroscopic surveys as a **calibration laboratory** for EFT
nuisance parameters, whose output is a set of informative priors
exported to the larger survey's full footprint.

The novelty is the **prior export mechanism**: the 1,200 deg²
overlap is not valuable primarily for the cosmological
information it contains (modest, 9% of DESI area), but for the
nuisance-parameter calibration it produces, which propagates to
14,000 deg². This is a qualitatively different use of
multi-tracer data than sample-variance cancellation.

Opus must frame the introduction to make this distinction sharp.
The paper is a **methods paper** demonstrating a new
cross-survey calibration strategy, with PFS × DESI as the
worked example.

---

## 4. Target structure

### §1. Introduction (~1 page)

Open with: the promise of high-kmax spectroscopic analyses is
limited not by statistical power but by nuisance-parameter
marginalization. EFT counterterms, stochastic terms, and FoG
parameters absorb information at small scales, inflating error
bars on fσ8 and Mν. Any external constraint on these nuisance
parameters directly improves cosmological constraints.

Introduce the calibration laboratory concept: when two surveys
with different galaxy selections observe the same volume, their
cross-power spectrum constrains per-tracer nuisance parameters
more tightly than either auto-spectrum alone. This calibration
is then portable to the larger survey's full footprint.

State: PFS and DESI provide a concrete realization — ~1,200 deg²
of overlap in 0.8 < z < 1.6, with ELGs selected at different
magnitude limits → different bias, different σ_FoG.

Distinguish from: (i) standard multi-tracer in a single survey
(Rubira & Conteddu 2025), (ii) cross-survey Fisher with no
cross-correlations (Qin+ 2025), (iii) FoG cleaning in isolation
(Baleato Lizancos+ 2025). Close with paper roadmap.

### §2. The calibration laboratory concept (~1.5 pages)

**§2.1 Why cross-power constrains nuisance parameters.**
Write the system of three power spectra (AA, BB, AB) and show
how the three FoG dampings over-determine σ_A and σ_B. Explicit
equations:

  P^{AA}(k,μ) ∝ F²(kμσ_A) [b_A + fμ²]² P_lin
  P^{BB}(k,μ) ∝ F²(kμσ_B) [b_B + fμ²]² P_lin
  P^{AB}(k,μ) ∝ F(kμσ_A) F(kμσ_B) [b_A + fμ²][b_B + fμ²] P_lin

Three equations, two FoG unknowns (plus f, biases — but those
are also multi-tracer–constrained). Analogous argument for
cross-counterterms.

**§2.2 The prior export mechanism.** Derive the full-area
Fisher matrix with imported priors:

  F_total(z) = F_DESI_auto(z; V_full, kmax) + F_prior(z)

where F_prior = diag(1/σ²_prior) for each calibrated nuisance
parameter. Explain that σ_prior comes from the *marginalized*
overlap Fisher, not the conditional — it already accounts for
degeneracies with cosmological parameters.

**§2.3 Transferability.** One paragraph stating that DESI's
spatially uniform target selection guarantees that ELGs in the
overlap have the same HOD as ELGs outside. This is what makes
the calibration portable. State as an assumption.

### §3. PFS and DESI ELG samples (~1 page)

Describe both surveys' ELG selections:
- DESI-ELG: [OII] flux limit, magnitude cut, b1 ≈ 0.84/D(z),
  nbar, σ_FoG. Cite DESI DR1 papers.
- PFS-ELG: deeper [OII], fainter magnitude limit, higher nbar,
  lower satellite fraction, lower σ_FoG. Cite Takada+ 2014 and
  PFS SSP.
- Overlap geometry: ~1,200 deg² of the PFS cosmology footprint
  within DESI's 14,000 deg². Shared z-range 0.8–1.6.

Table 1: per z-bin survey properties (nbar, V, b1, σ_FoG,
f_sat) for both tracers.

### §4. Fisher forecast setup (~1.5 pages)

**§4.1 Multi-tracer Fisher in the overlap.** Observable vector,
covariance, asymmetric kmax. Reference implementation spec §4.4.

**§4.2 Parameters.** Table 2: shared + per-tracer parameters,
fiducial values, derivative steps.

**§4.3 Five scenarios.** Define baseline, self-cal, self-cal-ext,
oracle, overlap-only. Table format. Explain that comparing
self-cal-ext to baseline isolates the full impact of
cross-calibration; comparing self-cal-ext to oracle measures
calibration efficiency.

**§4.4 kmax prescriptions.** Conservative (baseline) and
extended (self-cal-ext) kmax values per z-bin. Explain that the
extended kmax is justified *because* σ_FoG is now calibrated
rather than marginalized over a broad prior — the model is
the same, only the prior information changes.

### §5. Results (~2 pages)

**§5.1 Nuisance calibration in the overlap.** Anchor to Figure 1.
Report σ(σ_FoG^DESI) from: DESI-only, PFS-only, multi-tracer.
Quantify the tightening factor. Identify which nuisance parameter
benefits most from cross-calibration.

**§5.2 Exported priors.** Anchor to Figure 2. Show the calibrated
prior widths vs the broad baseline. Report the prior tightening
factor per z-bin. Note that the calibration is tightest at the
lowest overlap z-bin (largest V_overlap) and loosest at the
highest.

**§5.3 Full-area DESI improvement.** Anchor to Figure 3 (money
figure). Report σ(fσ8) and σ(Mν) for all five scenarios. State
the prior export ratio: σ(fσ8)_baseline / σ(fσ8)_self-cal-ext.
Decompose: how much comes from tighter nuisance priors at fixed
kmax (self-cal vs baseline) vs from extending kmax with calibrated
priors (self-cal-ext vs self-cal).

**§5.4 Calibration efficiency.** Anchor to Figure 4. Report the
fraction of nuisance-marginalization cost recovered. If close
to 1 → the overlap is nearly as good as perfect knowledge of
nuisance parameters. If close to 0 → nuisance parameters are
not the bottleneck (cosmological degeneracies dominate).

**§5.5 Sensitivity tests.** Short paragraph each:
- Sensitivity to broad-prior width (does the gain persist for
  moderately informative priors?).
- Sensitivity to kmax prescription.
- Sensitivity to PFS-ELG tracer properties (vary σ_FoG^PFS,
  b1^PFS by ±20%).

### §6. Discussion (~1 page)

- **The 1,200 deg² as infrastructure.** Frame the overlap not as
  a small fraction of DESI, but as a calibration resource whose
  value scales with the full survey area. The lever arm is
  V_full / V_overlap ≈ 12. Future surveys with larger overlaps
  (e.g., DESI-II × PFS, or Spec-S5 × MegaMapper) would have
  even larger lever arms.

- **Generalization beyond PFS × DESI.** The mechanism applies
  whenever two spectroscopic surveys with different target
  selections overlap. Euclid spectroscopic × DESI. Roman ×
  Spec-S5. The requirement is that the two surveys select
  tracers with measurably different σ_FoG or EFT parameters.

- **Operational implications.** Should PFS and DESI coordinate
  overlap? Currently the overlap is incidental. If the
  calibration gain is large, there is a case for deliberately
  maximizing the overlap area — or for DESI to re-observe a
  subset of PFS targets to verify transferability.

- **Connection to Baleato Lizancos+ cleaning.** The
  cross-calibration is complementary to single-survey FoG
  cleaning: Baleato Lizancos+ reduces σ_FoG by removing
  satellites; this paper constrains σ_FoG from the data
  itself. Both could be combined.

- **Caveats.** Gaussian covariance (conservative direction).
  Transferability assumption. No imaging systematics. No
  z-dependent nuisance modeling. No assembly bias.

### §7. Conclusions (~0.3 page)

Four takeaways:
1. The PFS × DESI overlap volume acts as a multi-tracer
   calibration laboratory for EFT nuisance parameters.
2. Cross-calibrated σ_FoG priors are [X]× tighter than
   uninformative priors.
3. Exporting these priors to DESI's full 14,000 deg² improves
   σ(fσ8) by [Y]% and σ(Mν) by [Z]%.
4. The calibration efficiency is [W]%, indicating that nuisance
   marginalization is [a major / a moderate / not the primary]
   bottleneck for DESI full-shape analyses at high kmax.

Plus: Acknowledgments (placeholder), Appendix A (derivative
convergence), Appendix B (Fisher verification: single-tracer
analytic + multi-tracer Seljak limit).

---

## 5. Writing style requirements

- Concise, technically precise academic prose. No padding.
- Direct, assertive framing.
- `DESI-ELG`, `PFS-ELG`, `DESI-DR1`, `HSC-Y3` with hyphens.
- No arbitrary LaTeX line breaks.
- `\cref{}` throughout. No bare `\ref{}`.
- No boilerplate.
- Equations numbered only if referenced.
- Tight logical flow.
- Every forecast number traces to `summary.csv` or `NOTES.md`.
  Missing → `% TODO(NMN):`.

---

## 6. Figures and tables

- **Fig. 1** (overlap calibration) → §5.1.
- **Fig. 2** (exported priors) → §5.2.
- **Fig. 3** (full-area comparison, money figure) → §5.3.
  Caption must state all five scenarios, z-bins, parameters.
- **Fig. 4** (calibration efficiency) → §5.4.
- **Fig. 5** (kmax with/without calibration) → §4.4 or §5.5.
- **Tab. 1** — survey properties per z-bin.
- **Tab. 2** — parameters, fiducials, steps.
- **Tab. 3** — headline numbers: σ per scenario per parameter,
  improvement ratios, calibration efficiencies.

All captions self-contained.

---

## 7. Deliverables from Opus

1. **`pfs_desi_crosscal_draft.tex`** — single-file PRD LaTeX
   (`revtex4-2`, `prd`, `twocolumn`, `amsmath`, `graphicx`,
   `cleveref`, `booktabs`).

2. **`outline_notes.md`** — meta-document: which summary.csv
   rows anchor each number, TODO items, suggested references.

---

## 8. Invocation protocol

Opus opens with:

> "Draft `pfs_desi_crosscal_draft.tex` per this spec. Before
> drafting, I need to resolve:
> (a) which summary.csv rows anchor the headline improvement
>     percentages in abstract/conclusions,
> (b) the calibration efficiency value — does it indicate
>     nuisance parameters are the dominant bottleneck?,
> (c) whether sensitivity tests (§5.5) revealed any
>     configurations where the gain disappears,
> (d) the peak-gain redshift from Figure 3,
> (e) any ambiguous references in NOTES.md."

Opus does not draft before resolution.

---

## 9. Out of scope

- Response-to-referee material.
- Cover letter.
- Supplementary material beyond Appendices A–B.
- Speculative forecasts for surveys beyond PFS × DESI
  (mention Euclid, Roman etc. only in Discussion §6).
- Simulation-based inference comparison.
- DESI z-bins outside the PFS overlap range (can be added
  as uncalibrated single-tracer Fisher if desired, but not
  the focus).
