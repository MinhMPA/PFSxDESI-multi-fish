# Writeup Spec: Data-Driven EFT Prior Calibration from Cross-Survey Overlap

**Target executor:** Claude Opus 4.7
**Owner:** N.-M. Nguyen (Kavli IPMU)
**Combines with:** `results/` from implementation spec
**Venue:** Physical Review D, regular article, ~8–10 pages two-column.
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
> per-tracer counterterms and stochastic parameters [X]× more
> tightly than conservative priors, and exporting these to DESI's
> full 14,000 deg² improves σ(fσ8) by [Y]% and σ(Mν) by [Z]%,
> recovering [W]% of the gap to perfect nuisance knowledge.

Fill [X], [Y], [Z], [W] from `summary.csv`. Never invent numbers.

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
  in auto-spectra).
- FoG damping in P^{AB} is the geometric mean of per-tracer
  dampings, not the square.
- Three spectra at each (k, ℓ) jointly constrain per-tracer
  nuisance parameters more tightly than either auto alone.

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
Collaboration (2016, 1611.00036) and Takada+ (2014). Table 1:
survey properties per z-bin (nbar, b1, V) for both tracers.

Overlap geometry: ~1,200 deg² of PFS footprint within DESI's
14,000 deg². Shared z-range 0.8–1.6 (DESI ELGs extend to 1.6,
PFS ELGs to 2.4).

### §4. Fisher forecast setup (~1.5 pages)

**§4.1 EFT parameterization.** Follow Chudaykin+ 2025 exactly.
Table 2: fiducial values and prior widths per tracer.

**§4.2 PFS-ELG EFT fiducials.** Scaling from DESI: c̃ by σ_v
ratio (default 0.75), Pshot by nbar, counterterms by b1.
State the sensitivity parameter r_σv and range tested.

**§4.3 Multi-tracer Fisher.** Observable vector (9 elements:
3 multipoles × 3 spectra), Gaussian covariance (9×9), k-range.

**§4.4 Four scenarios + benchmark.** Define broad, cross-cal,
cross-cal-ext, oracle. Define HOD-prior benchmark (Zhang+ 2025)
and field-level benchmark (Chudaykin+ 2026) as reference lines.

**§4.5 Transferability.** DESI's uniform target selection
guarantees ELGs in/outside overlap have same EFT parameters.

### §5. Results (~2 pages)

**§5.1 Nuisance calibration in the overlap.** → Fig. 1.
Which parameters tighten most? Expect c̃ and Pshot (most
different between tracers). b1σ8 should not change much.

**§5.2 Calibrated priors vs broad.** → Fig. 2.

**§5.3 Full-area DESI constraints.** → Fig. 3 (money figure).
Report σ(fσ8), σ(Mν), σ(Ωm) for all scenarios.
Compare to Zhang+ 2025 and Chudaykin+ 2026 benchmarks.

**§5.4 Calibration efficiency.** → Fig. 4.
(σ_broad − σ_cross-cal-ext) / (σ_broad − σ_oracle).

**§5.5 Sensitivity tests.** → Fig. 5.
Vary r_σv ∈ [0.5, 1.0]. Also: kmax ∈ {0.20, 0.25, 0.30};
overlap area ∈ {600, 1200, 2400} deg².

### §6. Discussion (~1 page)

- **Complementarity with SBPs.** Cross-cal is model-independent;
  SBPs are tighter but HOD-dependent. Optimal: combine both.
  Cross-cal as consistency check on HOD assumptions.
- **Survey coordination.** If gain is significant, case for
  maximizing PFS-DESI overlap deliberately.
- **Generalization.** Euclid × DESI, Roman × Spec-S5,
  DESI-II × MegaMapper.
- **Caveats.** Gaussian Fisher (conservative direction).
  Built-in model for PFS EFT fiducials. No bispectrum.
  No window function. Approximate SBP comparison.

### §7. Conclusions (~0.3 page)

Four takeaways:
1. EFT nuisance marginalization is the dominant cost in FS
   analyses (citing Chudaykin+ 2025; Zhang+ 2025).
2. Cross-survey overlap provides data-driven, model-independent
   calibration of these parameters.
3. For PFS × DESI: σ(fσ8) improves by [Y]%, σ(Mν) by [Z]%.
4. Complementary to HOD-informed priors; provides independent
   consistency check.

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
| Fig. 1 | §5.1 | Overlap calibration: per-param σ for DESI-only vs MT |
| Fig. 2 | §5.2 | Calibrated vs broad prior widths |
| Fig. 3 | §5.3 | Full-area DESI σ(fσ8,Mν) per scenario + benchmarks |
| Fig. 4 | §5.4 | Calibration efficiency per z-bin |
| Fig. 5 | §5.5 | Sensitivity to σ_v ratio |

| Table | Content |
|-------|---------|
| Tab. 1 | Survey properties per z-bin (nbar, b1, V) |
| Tab. 2 | EFT fiducials + prior widths (both tracers) |
| Tab. 3 | Headline: σ per scenario, improvement %, efficiency |

All figure captions self-contained. A reader should understand
each figure from its caption alone.

---

## 7. Deliverables

1. **`pfs_desi_eft_crosscal_draft.tex`** — single-file LaTeX
   (`revtex4-2`, `prd`, `twocolumn`, `amsmath`, `graphicx`,
   `cleveref`, `booktabs`). Compiles with pdflatex → bibtex →
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
