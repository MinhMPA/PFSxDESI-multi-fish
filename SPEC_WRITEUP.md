# Writeup Spec: Data-Driven EFT Prior Calibration from Cross-Survey Overlap

**Target executor:** Claude Opus 4.7
**Owner:** N.-M. Nguyen (Kavli IPMU)
**Combines with:** `results/` from implementation spec
**Venue:** JCAP, regular article, ~15–20 pages single-column.
**Template:** User provides JCAP template files in `paper/`.
**Self-contained:** No dependencies on earlier spec versions.

---

## 1. Inputs

Attach all of the following when invoking this spec:

1. **Implementation spec** (for context on code and methodology).
2. **`results/summary.csv`** — marginalized σ per scenario × z-bin × parameter.
3. **`results/figures/fig1–fig5.pdf`** — the five figures.
4. **`results/priors/*.json`** — cross-calibrated nuisance priors.
5. **`NOTES.md`** — user's bullet notes on observed trends, surprises, and references.
6. Optional: **`refs.bib`**.

---

## 2. Thesis (one sentence)

> Where simulation-based priors on EFT nuisance parameters rely on
> HOD model assumptions (Zhang+ 2025; Chudaykin+ 2026), we propose
> a complementary data-driven calibration: the multi-tracer analysis
> of PFS × DESI ELGs in their ~1,200 deg² overlap volume constrains
> stochastic parameters (Pshot) 2–7× more tightly than conservative
> priors, and exporting these to DESI's full 14,000 deg² improves
> σ(fσ8) by 19–35% and σ(Mν) by 48–53%, recovering 38–63% of the
> gap to perfect nuisance knowledge.

Numbers from `results/summary.csv` (combined z-bins, cross-cal
and cross-cal-ext scenarios). Calibration efficiency from Fig 4.

---

## 3. Positioning relative to prior work

### Papers to cite and distinguish from:

**EFT prior methodology — the problem:**
- Chudaykin, Ivanov & Philcox (2025, 2511.20757): DESI DR1
  reanalysis with reparameterized EFT priors. Showed prior
  choice dominates projection effects. Our baseline uses their
  parameterization and conservative prior widths.
- Chudaykin, Ivanov & Nishimichi (2024, 2410.16358): systematic
  study of prior volume and scale-cut effects in EFT analyses.

**Simulation-based priors — the competing solution:**
- Ivanov, Cuesta-Lazaro, Mishra-Sharma, Obuljen & Toomey (2024,
  2402.13310): original SBP methodology on BOSS. Normalizing
  flows on 10,500 HOD mocks. Posterior volume shrinks by ~10×.
  fNL improves by ~40%.
- Ivanov, Obuljen, Cuesta-Lazaro & Toomey (2024, 2409.10609):
  extended SBPs to redshift space + cosmology dependence.
- Zhang, Bonici, D'Amico, Paradiso & Percival (2024, 2409.12937):
  HOD-informed priors via 320,000 mocks + normalizing flows.
  Methodology paper.
- Zhang, Bonici, Rocher, Percival + DESI (2025, 2504.10407):
  HOD-informed priors applied to DESI DR1. σ(σ8) improves ~23%,
  σ(Ωm) ~4% over baseline. **This is the fair comparison** for
  our work (both operate at the power-spectrum level).
- Chudaykin, Ivanov & Philcox (2026, 2602.18554): field-level
  SBPs on DESI DR1. σ(H0) improves ~40%, σ(σ8) ~50%. **This is
  the ceiling** (field-level breaks degeneracies inaccessible to
  two-point statistics).

**Multi-tracer with different kmax:**
- Rubira & Conteddu (2025, 2504.18245): multi-tracer Fisher with
  sub-samples having different FoG → different kmax per tracer.
  Applied within a single survey, not cross-survey.

**FoG cleaning:**
- Baleato Lizancos, Seljak, Karamanis, Bonici & Ferraro (2025,
  2501.10587): data-driven FoG cleaning via quadrupole
  zero-crossing and tSZ. Single-tracer analysis.

**Cross-survey Fisher:**
- Qin, Wang & Zhao (2025, 2505.04275): J-PAS + PFS Fisher
  forecast. Explicitly avoids the overlap volume.

### Novelty claim:

No prior work uses the overlap volume of two independent
spectroscopic surveys as a **calibration laboratory for EFT
nuisance parameters** whose output is exported as priors to the
full survey footprint. This is qualitatively distinct from both
(a) sample-variance cancellation and (b) simulation-based priors.
It provides a model-independent cross-check of HOD-derived
calibrations.

Two specific novel elements:
1. **Stochastic separation via cross-spectrum**: the zero-shot-noise
   property of P^{AB} is the primary calibration mechanism (not
   FoG separation as might be naively expected).
2. **Asymmetric kmax across surveys**: exploiting PFS's lower FoG
   to extend kmax_PFS beyond kmax_DESI in the overlap, applying
   the Rubira & Conteddu (2025) idea cross-survey for the first time.

---

## 4. Target structure

### §1. Introduction (~1 page)

The frontier of full-shape spectroscopic cosmology is limited by
EFT nuisance marginalization. Cite Chudaykin+ 2025 (prior
reparameterization enables CMB+DESI without SNe), Zhang+ 2025
(HOD-informed priors improve σ8 by 23%), and Chudaykin+ 2026
(field-level SBPs improve H0 by 40%). State the problem: all
existing informative-prior approaches depend on HOD assumptions.

Introduce the cross-calibration concept: when two surveys with
different target selections observe the same volume, their
cross-power spectrum constrains per-tracer nuisance parameters
model-independently. The calibration is then portable to the
larger survey's full area.

State: PFS × DESI overlap provides a concrete realization.
Close with paper roadmap.

### §2. Data-driven EFT prior calibration (~2 pages)

**§2.1 The marginalization cost.** Review the EFT parameter set
(Chudaykin+ 2025, Table I). Which parameters are analytically
marginalized and why their priors matter. Cite the Zhang+ 2025
result: HOD-informed priors improve σ(σ8) by 23% — this
quantifies the cost of conservative priors.

**§2.2 Cross-power constraints on nuisance parameters.** Write
the auto-power P^{AA}, P^{BB} and cross-power P^{AB} in terms
of EFT parameters. Key features:
- P^{AB} has no auto-stochastic term (Pshot, a0, a2 enter only
  in auto-spectra). This is the **primary calibration mechanism**:
  measuring P^{AB} alongside P^{AA} and P^{BB} cleanly separates
  shot noise from signal. The forecast confirms this — Pshot
  tightens by 2–7× while counterterms barely improve.
- Cross-counterterms are averaged: (ctr_A + ctr_B) / 2 (standard
  EFT prescription, not geometric mean of Lorentzian damping).
- Three spectra at each (k, ℓ) jointly constrain per-tracer
  nuisance parameters more tightly than either auto alone.

**§2.2.1 Asymmetric kmax.** PFS ELGs reside in lower-mass halos
with smaller virial velocities → smaller FoG counterterm c̃_PFS.
The EFT convergence scale k_max ∝ c̃^{−1/4}, so
kmax_PFS = kmax_DESI × r_σv^{−1/2} > kmax_DESI. At k > kmax_DESI,
only P^{AA} and P^{AB} are available (DESI auto-spectrum excluded).
The Fisher matrix sums two k-ranges:
- k ≤ kmax_DESI: full 9 observables (3 spectra × 3 multipoles)
- kmax_DESI < k ≤ kmax_PFS: 6 observables (AA + AB only)

This asymmetric kmax makes the r_σv sensitivity non-trivial:
lower r_σv → higher kmax_PFS → more modes → tighter c̃ constraint
→ additional ~5% improvement on σ(fσ8) beyond the stochastic
contribution (Fig 5). Cite Rubira & Conteddu (2025, 2504.18245)
for the single-survey precedent.

**§2.3 The prior export mechanism.** The overlap multi-tracer
Fisher → marginalized σ for each DESI nuisance parameter →
Gaussian prior → added to DESI full-area Fisher:
F_full = F_DESI(V_full, kmax) + F_cross-cal_prior.
Lever arm: V_full / V_overlap ≈ 12.

**§2.4 Comparison with simulation-based priors.** Frame as
complementary:
- SBPs: tight, but dependent on HOD model correctness.
- Cross-cal: model-independent, but limited by overlap volume.
- The two can be combined; cross-cal serves as a consistency
  check on SBPs.
- Fair comparison: Zhang+ 2025 (power-spectrum level, ~23% on
  σ8). Ceiling: Chudaykin+ 2026 (field-level, ~50% on σ8).

### §3. PFS and DESI ELG samples (~0.75 page)

Survey specs: areas, z-ranges, nbar(z), b1(z). Cite DESI
Collaboration (arXiv:2503.14738, desimodel forecast) and
Takada+ (2014). Table 1: survey properties per z-bin (nbar, b1, V)
for both tracers. n(z) read from fine-binned files (Δz=0.01),
volume-weighted into 4 matched z-bins [0.8,1.0]–[1.4,1.6].

Overlap geometry: ~1,200 deg² of PFS footprint within DESI's
14,000 deg². Shared z-range 0.8–1.6 (DESI ELGs extend to 1.6,
PFS ELGs to 2.4).

### §4. Fisher forecast setup (~1.5 pages)

**§4.1 EFT parameterization.** Follow Chudaykin+ 2025 exactly.
Table 2: fiducial values and prior widths per tracer.

**§4.2 PFS-ELG EFT fiducials.** Scaling from DESI: c̃ by σ_v
ratio (default 0.75), Pshot by nbar, counterterms by b1.
State the sensitivity parameter r_σv and range tested.

**§4.3 Multi-tracer Fisher.** Observable vector: P_ℓ(k) multipoles
(ℓ=0,2,4) for three spectra (AA, BB, AB). 9×9 Gaussian covariance
at each k, computed from one-loop P(k,μ) via Gauss-Legendre
μ-integration. Asymmetric k-range: 9 observables at k ≤ kmax_DESI,
6 observables (AA+AB) at kmax_DESI < k ≤ kmax_PFS. Derivatives
via JAX autodiff through ps_1loop_jax (one-loop EFTofLSS).

**§4.4 Four scenarios + benchmark.** Define broad, cross-cal,
cross-cal-ext, oracle. Define HOD-prior benchmark (Zhang+ 2025)
and field-level benchmark (Chudaykin+ 2026) as reference lines.

**§4.5 Transferability.** DESI's uniform target selection
guarantees ELGs in/outside overlap have same EFT parameters.

### §5. Results (~2 pages)

**§5.1 Nuisance calibration in the overlap.** → Fig. 1.
The dominant tightening comes from stochastic parameters:
Pshot tightens to 15–45% of its broad prior (2–7× improvement),
a0 to 15–45%. The counterterms (c̃, c0) tighten only to ~85–95%
of broad. This is the key finding: **the cross-spectrum's zero
stochastic contribution, not FoG separation, is the primary
calibration mechanism**. PFS-only (orange) does better than
DESI-only (blue) on c̃ thanks to asymmetric kmax.

**§5.2 Calibrated priors vs broad.** → Fig. 2.
Pshot is the standout (nearly order-of-magnitude improvement on
log scale). b2σ8² and bG2σ8² show modest tightening from one-loop
sensitivity. c1 is unchanged (zero power-spectrum derivative;
would become active with the bispectrum).

**§5.3 Full-area DESI constraints.** → Fig. 3 (money figure).

| Scenario | kmax | σ(fσ8) | Δ% | σ(Mν) [eV] | Δ% | σ(Ωm) | Δ% |
|----------|------|--------|------|------------|------|-------|------|
| broad | 0.20 | 0.0757 | — | 1.075 | — | 0.0499 | — |
| cross-cal | 0.20 | 0.0610 | −19% | 0.559 | −48% | 0.0401 | −19% |
| cross-cal-ext | 0.25 | 0.0493 | −35% | 0.505 | −53% | 0.0388 | −22% |
| oracle | 0.25 | 0.0100 | −87% | 0.123 | −89% | 0.0194 | −61% |

Cross-cal is competitive with the Zhang+ 2025 HOD-prior benchmark
(23% on σ8). Cross-cal-ext exceeds it. Mν benefits most because
it is sensitive to the amplitude parameters that stochastic priors
constrain. Compare to Chudaykin+ 2026 field-level ceiling (50%).

**§5.4 Calibration efficiency.** → Fig. 4.
Efficiency = (σ_broad − σ_xcal-ext) / (σ_broad − σ_oracle).
Ranges from 35% (Ωm) to 63% (Mν). Mν highest because most
sensitive to stochastic parameters. fσ8 rises with z (38→47%)
as PFS nbar advantage grows. Ωm flat (~35%) — constrained by
P(k) shape, not amplitude.

**§5.5 Sensitivity tests.** → Fig. 5.
Varying r_σv ∈ [0.5, 1.0]: σ(fσ8) ranges from 0.046 (39%
improvement at r_σv=0.5) to 0.050 (34% at r_σv=1.0). The curve
slopes downward, confirming asymmetric kmax adds real FoG
leverage. Even at r_σv = 1.0 (identical FoG), cross-calibration
gives 34% improvement from stochastic separation alone — this
is robust and does not depend on uncertain FoG modeling.

### §6. Discussion (~1 page)

- **Stochastic dominance.** The improvement comes primarily from
  Pshot/a0 calibration (zero-stochastic cross-spectrum), not FoG
  counterterm separation. This is a strength: the mechanism does
  not depend on uncertain FoG modeling (r_σv sensitivity is mild,
  34–39%). The asymmetric kmax adds ~5 percentage points from
  FoG leverage — a bonus, not the core mechanism.
- **Complementarity with SBPs.** Cross-cal is model-independent;
  SBPs are tighter but HOD-dependent. Optimal: combine both.
  Cross-cal as consistency check on HOD assumptions. The stochastic
  parameters that cross-cal constrains best (Pshot, a0) are also
  the hardest to predict from HOD models (they depend on
  sub-Poisson corrections from satellite fractions).
- **Survey coordination.** The 19–35% improvement on σ(fσ8) from
  1,200 deg² overlap is significant; maximizing PFS-DESI overlap
  should be a survey-coordination priority.
- **Generalization.** Euclid × DESI, Roman × Spec-S5,
  DESI-II × MegaMapper. Any pair of spectroscopic surveys with
  overlapping volume and different target selections.
- **Bispectrum extension.** The tree-level bispectrum would
  activate the c1 counterterm (enters Z1_FoG), break b1–b2–bG2
  degeneracies, and add B_shot/A_shot constraints. This would
  tighten the counterterm calibration and push efficiency closer
  to the oracle. Deferred to future work; the Pk-only story
  already demonstrates the core calibration mechanism.
- **Caveats.** Gaussian Fisher (conservative direction).
  Scaled PFS EFT fiducials (no PFS mock fits yet). No bispectrum.
  No window function. Approximate SBP comparison (published
  improvement ratios, not re-implemented).

### §7. Conclusions (~0.3 page)

Five takeaways:
1. EFT nuisance marginalization is the dominant cost in FS
   analyses (citing Chudaykin+ 2025; Zhang+ 2025).
2. Cross-survey overlap provides data-driven, model-independent
   calibration — primarily through the zero-stochastic cross-power
   spectrum, which cleanly separates shot noise from signal.
3. For PFS × DESI: σ(fσ8) improves by 19–35%, σ(Mν) by 48–53%,
   recovering 38–63% of the gap to perfect nuisance knowledge.
4. The improvement is robust to FoG modeling assumptions (r_σv
   sensitivity: 34–39%), because the stochastic calibration does
   not depend on FoG. The asymmetric kmax from PFS's lower FoG
   adds a further ~5% bonus.
5. Complementary to HOD-informed priors; competitive at the
   power-spectrum level (exceeds Zhang+ 2025 benchmark on fσ8);
   provides an independent, data-driven consistency check on
   simulation-based calibrations.

Acknowledgments (placeholder). Appendix A (derivative
convergence). Appendix B (Fisher code verification).

---

## 5. Writing style

- Concise, technically precise academic prose. No padding.
- Direct, assertive framing. "We forecast" not "It can be argued."
- Hyphenated survey labels: `DESI-ELG`, `PFS-ELG`, `DESI-DR1`.
- No arbitrary LaTeX line breaks. Paragraphs flow naturally.
- `\cref{}` throughout (`\usepackage{cleveref}`). No bare `\ref{}`.
- No boilerplate ("It is worth noting…", "As is well known…").
- Equations numbered only if referenced; otherwise `equation*`.
- Every forecast number traces to `summary.csv` or `NOTES.md`.
  Missing numbers → `% TODO(NMN):` comment.
- Tight logical flow: each paragraph's first sentence advances
  the argument.

---

## 6. Figures and tables

| Figure | Section | Content |
|--------|---------|---------|
| Fig. 1 | §5.1 | Overlap calibration: σ/σ_broad per param for DESI-only, PFS-only, MT (4 z-bin panels) |
| Fig. 2 | §5.2 | Calibrated vs broad prior widths (all 11 nuisance params, log scale, 4 z-bins as dots) |
| Fig. 3 | §5.3 | Full-area DESI σ(fσ8, Mν, Ωm) per scenario + HOD/field-level benchmark lines (money figure) |
| Fig. 4 | §5.4 | Calibration efficiency per z-bin (fσ8, Mν, Ωm as separate curves) |
| Fig. 5 | §5.5 | σ(fσ8) vs r_σv with broad baseline and 10% improvement lines |

| Table | Content |
|-------|---------|
| Tab. 1 | Survey properties per z-bin (nbar, b1, V) |
| Tab. 2 | EFT fiducials + prior widths (both tracers) |
| Tab. 3 | Headline: σ per scenario, improvement %, efficiency |

All figure captions self-contained. A reader should understand
each figure from its caption alone.

---

## 7. Deliverables

1. **`paper/pfs_desi_eft_crosscal.tex`** — single-file LaTeX
   using JCAP class (`jcappub`). Packages: `amsmath`, `graphicx`,
   `cleveref`, `booktabs`. Compiles with pdflatex → bibtex →
   pdflatex × 2.

2. **`outline_notes.md`** — which `summary.csv` rows anchor each
   number, `% TODO(NMN)` items, suggested references.

---

## 8. Invocation protocol

When the user pastes this spec, Opus opens with:

> "Before drafting, I need to resolve:
> (a) which summary.csv rows anchor headline [Y]%, [Z]%, [W]%,
> (b) calibration efficiency — high enough to claim nuisance
>     params are the bottleneck?,
> (c) how cross-cal compares to Zhang+ 2025 HOD-prior benchmark
>     (23% on σ8) — competitive, weaker, or complementary?,
> (d) sensitivity: minimum r_σv for >10% improvement,
> (e) any ambiguous references in NOTES.md."

Opus does not draft before resolution.

---

## 9. Out of scope

- Response-to-referee material or cover letter.
- Supplementary material beyond Appendices A–B.
- Surveys beyond PFS × DESI (mention in Discussion §6 only).
- Simulation-based inference comparison.
- Full SBP implementation or normalizing flow reproduction.
