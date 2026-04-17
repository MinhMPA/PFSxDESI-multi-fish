# Writeup Spec: Multi-Tracer FoG-Cleaned Analysis for PFS (Paper Draft)

**Target executor:** Claude Opus 4.7 (chat interface, not Claude Code)
**Owner:** N.-M. Nguyen
**Combines with:** results/ directory produced by `SPEC_IMPLEMENTATION_v2.md`
**Intended venue:** Physical Review D (regular article, ~8–10 pages
two-column including figures). Editor's Suggestion ambition.

---

## 1. Inputs Opus will receive

When invoking this spec, attach:

1. `SPEC_IMPLEMENTATION_v2.md` (for context on what the code does).
2. `results/summary.csv` (marginalized σ per analysis_mode ×
   kmax_prescription × z-bin × parameter).
3. `results/figures/fig1_kmax_scan.pdf`,
   `fig2_mt_decomposition.pdf`,
   `fig3_improvement_vs_z.pdf`,
   `fig4_kmax_prescription_comparison.pdf`.
4. `NOTES.md` — free-form bullet notes by the user on:
   - observed σ ratios across the four analysis modes,
   - the redshift at which the multi-tracer gain peaks and why,
   - the relative importance of kmax extension vs sample-variance
     cancellation (read off from comparing MT-same-kmax vs
     MT-ext-kmax),
   - sensitivity to η and to the HOD parameters,
   - any surprises,
   - references to cite.
5. Optional: `refs.bib`.

---

## 2. Paper goal and one-sentence thesis

> By treating the full PFS-ELG sample and a FoG-cleaned sub-sample
> as a multi-tracer pair sharing the same volume, we show that the
> joint analysis combines two distinct sources of information gain —
> extended kmax from reduced velocity dispersion, and
> sample-variance cancellation from the bias difference between
> centrals-dominated and satellite-inclusive populations — yielding
> constraints on fσ8 and Mν that exceed what either sub-sample
> achieves alone by [X]–[Y]% across the PFS redshift range.

(Opus fills [X]–[Y] from `summary.csv`. Do not invent numbers.)

---

## 3. Positioning relative to prior work

This paper builds on two established ideas and combines them in a
way that has not been done:

**Baleato Lizancos, Seljak, Karamanis, Bonici & Ferraro (2025,
JCAP):** proposed data-driven FoG cleaning via quadrupole
zero-crossing and tSZ-based satellite identification; analyzed
cleaned sub-samples in isolation (single-tracer). Did not consider
multi-tracer combination of full + cleaned samples.

**Seljak (2009); McDonald & Seljak (2009); Hamaus, Seljak &
Desjacques (2012):** multi-tracer technique for sample-variance
cancellation. Applied to distinct galaxy populations (e.g., red vs
blue, luminosity-split). Never applied to a FoG-cleaned sub-sample
where the bias split arises from satellite removal and the second
tracer simultaneously enjoys an extended kmax.

**Rashkovetskyi, Eisenstein et al. (2025, OJAp):** demonstrated
tSZ-based splitting of DESI LRGs with distinct clustering per
sub-sample; focused on bias measurement and environment dependence,
not on the kmax × multi-tracer forecast.

**This paper's novelty:** the multi-tracer combination of full +
FoG-cleaned samples, with tracer-specific kmax. The two effects
(kmax extension and sample-variance cancellation) are not merely
additive — the same physical process (satellite removal) produces
both the reduced σ_FoG (enabling higher kmax) and the bias
difference (enabling sample-variance cancellation). The forecast
quantifies the joint gain and its redshift dependence for PFS-ELG.

Opus must frame the introduction and related-work discussion to
make this positioning crystal clear. The paper is not a
"PFS application note" — it is a method paper demonstrating a new
multi-tracer strategy enabled by FoG cleaning, with PFS as the
worked example.

---

## 4. Target structure

Standard PRD article, eight sections:

### §1. Introduction (~1 page)

Open with: the frontier of spectroscopic cosmology is pushing to
higher kmax, where the density of Fourier modes scales as k³ and
the information content grows rapidly. Two barriers limit kmax:
nonlinear gravitational evolution (addressed by EFT) and
non-perturbative FoG from satellite galaxies in massive halos
(not addressed by EFT — cite Baleato Lizancos+ 2025 for this
argument).

State the two-pronged strategy: (i) clean the sample to extend
kmax; (ii) use the full and cleaned samples jointly as a
multi-tracer pair. State that these gains compound because the
same physical process (satellite removal) drives both.

Distinguish from Baleato Lizancos+ (single-tracer cleaning) and
from standard multi-tracer (distinct populations, no kmax
asymmetry). Close with paper roadmap.

### §2. Physical arguments (~1.5 pages)

Two subsections:

**§2.1 Why kmax extends at high z.** The three arguments from the
implementation spec (FoG compression, HMF suppression, k_NL shift),
each with an explicit equation evaluated at PFS z-bins. Reference
Figure 4 or a table.

**§2.2 Why multi-tracer gains compound with FoG cleaning.**
Derive or cite the multi-tracer Fisher for two tracers. Show that
the gain depends on (i) the bias ratio b_A/b_B ≠ 1 and
(ii) the shot-noise levels of both tracers. When tracer B is a
satellite-depleted sub-sample:
- b_B/b_A ≠ 1 because b_cen ≠ (1-f_s)b_cen + f_s b_sat.
- nbar_B < nbar_A, but this is partially compensated by the
  extended kmax.
- At high z, f_sat drops, so b_B/b_A → 1 (weakening MT gain)
  but nbar_B/nbar_A → 1 (reducing shot-noise cost). The forecast
  reveals which effect dominates.

Include the key inequality from the equations doc (§1.2 of
implementation spec: when does the cleaning pay off?) adapted to
the multi-tracer case.

### §3. PFS-ELG sample and sub-sample construction (~1 page)

Describe the baseline ELG sample (cite PFS survey paper). Then
the cleaning strategies:

- HSC photometric pre-selection: morphology / color / photo-M★
  for satellite flagging. Cite HSC PDR3.
- Fiber-collision–induced natural down-weighting: reframe Cobra
  incompleteness as FoG suppression; PIP weights (Bianchi &
  Percival 2017) on large scales for unbiased recovery.
- Quadrupole zero-crossing diagnostic (Baleato Lizancos+ 2025)
  as post-hoc validation.
- tSZ-based identification (Baleato Lizancos+ 2025;
  Rashkovetskyi+ 2025) — mention as applicable at lower z where
  ACT/SO overlap with PFS.

Introduce the `TracerPair` concept: tracer A = full PFS-ELG,
tracer B = cleaned sub-sample. State that they share the same
volume. Discuss the HOD-based parameterization: (b1_cen, b1_sat,
σ_v_cen, σ_v_sat, f_sat(z), η).

### §4. Multi-tracer Fisher forecast setup (~1.5 pages)

**§4.1 Observable vector and covariance.** Write the 9-element
data vector d(k) = {P_ℓ^AA, P_ℓ^BB, P_ℓ^AB} for ℓ ∈ {0,2,4}.
Write the Gaussian covariance. State the asymmetric-kmax
integration scheme (two segments).

**§4.2 Parameters.** Table 1: shared parameters
{fσ8, Mν} + per-tracer nuisance parameters
{b1, σ_FoG, c0, c2, c4, Pshot, a0, a2} × 2 tracers. State
fiducial values and step sizes.

**§4.3 kmax prescriptions.** Three prescriptions; Table 2 gives
kmax_full(z) and kmax_clean(z) per z-bin per prescription.

**§4.4 Four analysis modes.** Define the four rows of the
comparison table from §1.2 of the implementation spec:
full, clean, MT-same-kmax, MT-ext-kmax. Emphasize that
comparing MT-same-kmax vs clean isolates the multi-tracer
contribution, and comparing MT-ext-kmax vs MT-same-kmax
isolates the kmax-extension contribution.

**§4.5 Overlapping-sample caveat.** One paragraph. The cleaned
sample is a subset of the full sample. The multi-tracer
formalism handles this through r_cc → 1 in the cross-power.
Non-Gaussian covariance from the overlap is not included;
its effect would reduce the multi-tracer gain, making the
forecast a lower bound on the true improvement.

Table 3: n(z), V(z), f_sat(z), σ_v,full(z), σ_v,clean(z),
b1_full(z), b1_clean(z), kmax_full(z), kmax_clean(z) per
z-bin.

### §5. Results (~2 pages)

**§5.1 Single-tracer kmax scan.** Anchor to Figure 1. Show the
crossover kmax*(z) where cleaned overtakes full. Report
kmax*(z) per bin.

**§5.2 Multi-tracer decomposition.** Anchor to Figure 2. For
each z-bin and for the combined constraint:
- Report σ(fσ8) and σ(Mν) for all four analysis modes.
- Decompose the total MT-ext-kmax improvement into:
  (a) kmax extension alone = clean / full,
  (b) sample-variance cancellation alone = MT-same-kmax / full,
  (c) combined = MT-ext-kmax / full.
  State whether (c) > (a) + (b) - 1 (i.e., super-additive) or
  sub-additive, and explain why.

**§5.3 Redshift evolution.** Anchor to Figure 3. Describe how
the improvement ratio evolves with z. Identify the redshift
where the multi-tracer gain peaks and connect it to the
evolution of f_sat(z) and b_clean/b_full(z).

**§5.4 Sensitivity to FoG-cleaning model.** Short paragraph.
Compare HodFog vs AnalyticFog vs EmpiricalFog. Demonstrate
robustness or flag where it breaks.

**§5.5 Sensitivity to η.** Short paragraph. Vary satellite
removal efficiency η ∈ {0.3, 0.5, 0.7, 0.9}. Report how
the optimal η balances nbar loss against σ_FoG reduction and
bias contrast.

### §6. Discussion (~1 page)

- **Implications for PFS survey strategy.** Frame as actionable:
  which target-selection and fiber-assignment choices maximize
  the multi-tracer gain? Recommend that PFS analyses define both
  the full and cleaned catalogs as standard data products.
- **Generalization.** State which other surveys benefit:
  DESI-II (higher density, similar fiber geometry),
  MUST/Spec-S5/MegaMapper (very high z, low f_sat → kmax gain
  dominates over MT gain), WST (different fiber technology).
  One paragraph each for the two regimes (MT-dominated at
  intermediate z, kmax-dominated at high z).
- **Connection to Baleato Lizancos+ cleaning diagnostics.**
  The quadrupole zero-crossing and tSZ-based cleaning are
  complementary to photometric pre-selection. In a real analysis,
  multiple cleaning indicators could be combined; the forecast
  framework presented here applies regardless of the specific
  cleaning method.
- **Caveats.** Gaussian covariance, no survey window, overlapping
  samples (direction of bias is conservative), phenomenological
  HOD, no assembly bias. State crisply; do not bury.

### §7. Conclusions (~0.3 page)

Four takeaways in prose:
1. FoG cleaning extends kmax, with the gain growing at high z.
2. The full + cleaned multi-tracer combination adds
   sample-variance cancellation on top of the kmax extension.
3. The two effects arise from the same physical process
   (satellite removal) and compound: the total improvement
   exceeds either individually.
4. For PFS-ELG, the combined gain on fσ8 is [X]% at z ~ [peak z],
   and on Mν is [Y]% from all bins combined.

Plus: Acknowledgments (placeholder), Appendix A (derivative
convergence), Appendix B (Fisher code verification against
analytic Gaussian and Seljak 2009 multi-tracer limit).

---

## 5. Writing style requirements

(Identical to previous spec — repeated for self-containedness.)

- Concise, technically precise academic prose. No padding.
- Direct, assertive framing.
- Specific survey references: `DESI-DR1`, `HSC-Y3`, `PFS-ELG`.
- No arbitrary LaTeX line breaks.
- `\cref{}` throughout (requires `cleveref`). No bare `\ref{}`.
- No boilerplate.
- Equations numbered only if referenced.
- Tight logical flow: each paragraph's first sentence advances
  the argument.
- Numbers with units and appropriate precision.
- Every forecast number must trace to `summary.csv` or `NOTES.md`.
  Missing numbers → `% TODO(NMN):` comment.

---

## 6. Figures and tables

- **Fig. 1** (kmax scan) → §5.1. Caption: samples, FoG backend,
  kmax grid, cosmology.
- **Fig. 2** (MT decomposition) → §5.2. Caption: four analysis modes,
  z-bins, combined. **This is the money figure.** The caption must
  explicitly state what each bar represents and how to read the
  decomposition.
- **Fig. 3** (improvement vs z) → §5.3. Caption: continuous curves,
  dual y-axis (σ ratio + f_sat/bias ratio).
- **Fig. 4** (kmax prescriptions) → §2.1 or §5.4.
- **Tab. 1** — fiducial parameters and steps.
- **Tab. 2** — kmax(z) per prescription per tracer.
- **Tab. 3** — sample properties per z-bin (the comprehensive table).

All figure captions self-contained.

---

## 7. Deliverables from Opus

Two files:

1. **`pfs_mt_fog_draft.tex`** — single-file PRD-style LaTeX source
   (`revtex4-2`, `prd`, `twocolumn`, `amsmath`, `graphicx`,
   `cleveref`, `booktabs`). Compiles with `pdflatex` → `bibtex` →
   `pdflatex` × 2.

2. **`outline_notes.md`** — meta-document listing:
   - which `summary.csv` rows anchor each quoted number,
   - any `% TODO(NMN)` items,
   - suggested additional references.

---

## 8. Invocation protocol

When the user pastes this spec, Opus opens with:

> "Draft `pfs_mt_fog_draft.tex` per this spec. Before drafting,
> I need to resolve up to 5 points:
> (a) which `summary.csv` rows anchor the headline improvement
>     factors in the abstract/conclusions,
> (b) which FoG backend (HodFog vs AnalyticFog) is the headline
>     result,
> (c) the peak-improvement redshift from Figure 3,
> (d) whether the MT gain is super-additive or sub-additive
>     (this determines the framing of §5.2),
> (e) any references from `NOTES.md` that are ambiguous."

Opus does not draft before these are resolved.

---

## 9. Out of scope for the draft

- Response-to-referee material.
- Cover letter.
- Supplementary material beyond Appendices A and B.
- Speculative forecasts for surveys not in the implementation spec.
- Comparison with simulation-based inference approaches.
