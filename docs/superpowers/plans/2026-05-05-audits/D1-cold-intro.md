# Cold re-read: abstract + §1 Introduction

## What is the paper proposing?

The paper proposes a joint multi-tracer, multi-survey Fisher forecast that data-drives EFT bias and stochastic-parameter priors from cross-spectra between bias-distinct galaxy populations, rather than from HOD-based simulation priors (SBP). Two channels are combined inside a single volume-partitioned Fisher, F = F_overlap(~1,200 deg²) + F_nonoverlap(~12,800 deg²) covering 0.4<z<2.1: an "internal-DESI" channel (LRG×ELG×QSO across the full 14,000 deg² DESI footprint) and a "PFS×DESI overlap" channel that adds PFS-ELG as a fourth tracer over the ~1,200 deg² PFS footprint at 0.6<z<1.6. Cosmological parameters (fσ8, Mν, Ωm) are shared across tracers and z-bins, EFT nuisance parameters are per-(tracer, z-bin), and residual cross-stochasticity between disjoint populations is marginalized as a free function following Ebina & White. The headline claims are 33/80/49% tightening of σ(fσ8, Mν, Ωm) from internal-DESI alone, with the overlap channel adding a further 9/24/9%.

## What is the dominant mechanism according to the abstract + introduction?

(ii). The mechanism is a single, simultaneously-marginalized joint multi-tracer Fisher; internal-DESI provides the bulk of the gain and PFS×DESI overlap adds on top. The abstract states explicitly that the two channels are "combined in a single volume-partitioned joint Fisher", and the introduction restates this with the formula F = F_overlap + F_nonoverlap "marginalized in one pass". The internal-DESI channel is named as the "dominant gain" both in the abstract and in §1.

## Drift and inconsistency

- **"Priors" framing vs. joint-Fisher mechanism.** The abstract opens with "We propose data-driven multi-tracer priors as a model-independent alternative" and the closing paragraph again says the approach "targets the same b₁ calibration as SBPs but draws it from cross-spectra... it thus provides a model-independent cross-check on the HOD assumptions underlying SBP-driven b₁σ8 shifts." This is residual prior-language, soft but present. The mechanism is not really "priors" — it is a single joint Fisher in which cross-spectra contribute information directly. No "export priors" / "Step 1 / Step 2" / "calibrated priors transferred" language survives, but the noun "priors" in "data-driven multi-tracer priors" is at odds with the actual single-pass joint Fisher described two clauses later.
- **Volume partition vs. PFS redshift coverage.** The intro claims F = F_overlap(1,200) + F_nonoverlap(12,800) "covers 0.4<z<2.1", but PFS only covers 0.6<z<1.6. The text does not say what happens to the 1,200 deg² overlap footprint at z<0.6 and z>1.6 — does F_overlap there reduce to a 3-tracer DESI block, or is it folded into F_nonoverlap? A reviewer will ask.
- **Compounding-cost claim is uncited.** Lines 181–183 state that the σ(fσ8) degradation grows to ~4.6× and σ(Mν) to ~10× when 6 DR2 samples are combined. No citation; if this is a derivation from Chudaykin:2024wlw it should say so, otherwise it reads as an unsupported claim sitting next to a cited claim.
- **Zhang et al. attribution.** Line 193 says "Zhang et al.~\cite{DESI:2025wzd}" but DESI:2025wzd is a DESI collaboration paper. Line 192 cites Zhang:2024thl as a separate work. The two cites may or may not be the same underlying analysis; as written, the reader cannot tell.
- **\github{} after "this repository".** The abstract ends with "Code is available at this repository \github{...}" which renders as "...this repository [icon]." The prose lacks an actual URL or anchor noun; "this repository" with a trailing icon is grammatically dangling.

## Tone and style

- No "headline", "main result", "novel", or "key finding" promotional words.
- "Dominant" appears four times in the abstract+intro ("dominant systematic cost", "dominant gain", "dominant driver", "dominant modeling choice"). Repetitive; vary or drop.
- No eponymous mechanism naming ("McDonald-Seljak" etc.) — only standard `\cite{McDonald:2008sh}` for the b₁σ8–fσ8 degeneracy attribution. OK.
- No floaty hedges ("very", "many", "significant" without numbers); claims are quantified.
- **5-cite stacks:**
  - Line 168: `\cite{Chudaykin:2025aux,Chudaykin:2025lww,Chudaykin:2025vdh,Ivanov:2026dvl,Chudaykin:2026nls}` — flagged in source by an explicit `% [CITE-NEEDED: individual attribution ...]` comment.
  - Line 206: `\cite{Alarcon:2016bkr,Mergulhao:2021kip,Mergulhao:2023zso,Rubira:2025scu,Qin:2025nkk}` — same pattern, same explicit `% [CITE-NEEDED]` marker.
  - Both violate the project rule of max 2 clustered citations without individual discussion. Author already knows; un-fixed.
- **4-cite topical stack:** Line 164, EFTofLSS foundational `\cite{Baumann:2010tm,Carrasco:2012cv,Carroll:2013oxa,Senatore:2014eva}`. Defensible as foundational, but borderline.
- **Multi-application paragraph (line 166)** lists 7 sub-topics each with 1–3 citations and zero discussion. Reads as a topic list rendered as prose; each citation is unearned per the project's anti-citation-stacking rule.

## Citations

- Two explicit `% [CITE-NEEDED: individual attribution ...]` markers (lines 169, 207) flag the 5-cite stacks above as known TODOs.
- `Zhao:2024eboss` (line 208) — citation key naming convention is unusual (suffix "eboss" rather than arXiv number); verify the bib entry exists and resolves to the intended Zhao et al. eBOSS multi-tracer paper.
- `DESI:2025wzd` (line 168 and line 193) cited twice for different claims — once as a DESI DR1 reanalysis collaboration paper, once attributed to "Zhang et al." for power-spectrum-level SBP. These are likely different works; the same key cannot legitimately serve both roles.
- The compounding-cost claim (~4.6× / ~10× degradation when 6 DR2 samples combined, lines 181–183) carries no citation; either supply one or rephrase to make clear it is a calculation in this work.
- Line 167–168 attribute "evidence for dynamical dark energy and... a preference for the normal neutrino mass hierarchy" to the DESI papers and reanalyses but offer no individual attribution among the 5 stacked reanalysis cites — a referee will want to know which paper made which finding.
- `Takada:2014pfs` cited (line 218) for PFS overlap area; verify this is the PFS design paper and the area number 1,200 deg² appears there.

## Verdict

**NEEDS-ANOTHER-LOOP.**

Specific edits required:

1. Resolve the two `% [CITE-NEEDED]` 5-cite stacks (lines 168, 206) by individually attributing each work — at minimum a 1-clause role per citation, per the project anti-stacking rule.
2. Add a citation (or explicit "we compute") to the ~4.6× / ~10× compounding-cost claim on lines 181–183.
3. Reconcile the double use of `DESI:2025wzd` (lines 168 vs. 193) and decide whether the "Zhang et al." attribution refers to `Zhang:2024thl` or to a separate DESI:2025wzd companion paper.
4. Resolve the volume-partition / redshift-coverage ambiguity: state what F_overlap looks like at z<0.6 and z>1.6, where PFS contributes no tracer.
5. Decide whether to keep the "data-driven multi-tracer priors" framing in the abstract opener. If the actual mechanism is a single joint Fisher, either (a) drop the noun "priors" in favor of "data-driven multi-tracer calibration" / "joint Fisher information" or (b) explicitly say "the joint Fisher *implies* effective priors on b₁ that we report for use in single-tracer pipelines." Currently the abstract reads as both at once.
6. Fix the dangling "this repository \github{}" sentence — replace with "Code is available at \github{URL}" or include the URL inline as text.
7. Trim the 7-topic application list on line 166: either keep two representative topics with discussed citations, or move the full list to a footnote.
8. Reduce the four-fold use of "dominant" — pick a different word for at least two of the four occurrences.
9. Verify the `Zhao:2024eboss` bib key resolves; if it is a placeholder, replace with the canonical arXiv-style key.
