# Cold re-read: §6 Conclusions + appendices

Scope: lines 1229–1606 of `paper/PFSxDESI-multi-fish.tex`
(§6 Conclusions, acknowledgments, App. A Validation, App. B EFT priors from
co-evolution).

## §6 Conclusions — item-by-item summary

1. The volume-partitioned joint Fisher
   `F = F_overlap(1,200 deg²) + F_nonoverlap(12,800 deg²)` combines the
   internal-DESI channel (full footprint) with the PFS×DESI overlap channel
   (1,200 deg²) in a single Fisher with cosmology shared globally and
   nuisance per (tracer, z-bin), and avoids the auto-spectra double-counting
   of a two-stage prior-export pipeline.
2. DESI alone, via bias-distinct LRG/ELG/QSO at 0.4 < z < 2.1, reaches
   σ(fσ8)=0.0145, σ(Mν)=0.141 eV, σ(Ωm)=0.0157 at kmax=0.20 h/Mpc with no
   cross-survey input.
3. Adding PFS to the 1,200 deg² overlap further tightens σ(fσ8) by 8.9%,
   σ(Mν) by 23.6%, σ(Ωm) by 9.4% — reaching 0.0132 / 0.108 eV / 0.0143;
   the Mν gain is driven by joint-Fisher tightening of b1σ8 for DESI-ELG
   via a fourth low-bias ELG partner (PFS-ELG) with distinct selection /
   shot noise.
4. The joint analysis is not a replacement for SBPs but a model-independent
   companion: multi-tracer cross-correlations constrain b1 and stochastic
   terms from data, while SBPs additionally tighten counterterms and
   k-dependent stochasticity; framed as an independent handle on whether
   SBP-driven σ8 shifts reflect new physics or HOD systematics.
5. Generalization to N overlapping spectroscopic surveys: the union
   partitions into up to 2^N − 1 disjoint regions, each carrying its own
   multi-tracer Fisher with shared cosmology — explicit examples:
   PFS×DESI×Euclid×Roman, plus literature pairs Euclid×DESI,
   Roman×Spec-S5, DESI-II×MegaMapper, J-PAS×PFS.

## (a) → (b) → (c) chain in §6

Verified against canonical `\cref{tab:headline}` (lines 909–943):

| Quantity | DESI-only joint (item 2) | DESI+PFS joint (item 3) | PFS-unique Δ% |
|---|---|---|---|
| σ(fσ8)  | 0.0145 ✓ | 0.0132 ✓ | −8.9 % ✓ |
| σ(Mν)   | 0.141 eV ✓ | 0.108 eV ✓ | −23.6 % ✓ |
| σ(Ωm)   | 0.0157 ✓ | 0.0143 ✓ | −9.4 % ✓ |

All six headline numbers and all three percentage gains match `tab:headline`
(§4.2, lines 915–917) and the prose in §4.2 (lines 832–837). No drift.

Minor note: the §4.2 figure caption (line 897–899) rounds these to
"9% / 24% / 9%" while the table and §6 carry one decimal. Not an
inconsistency, but be aware.

## Drift and inconsistency in §6 + appendices

- **Old-framing residue, prominent in §6 lead-in (line 1237):** "treating
  cross-survey calibration as part of a single Fisher … rather than as a
  separate prior-export step." Item 1 (line 1247–1248) repeats the
  contrast: "avoids the auto-spectra double-counting that a two-stage
  prior-export pipeline incurs." This is fine *as a once-only* contrast
  with prior literature, but combined with the §4.3 "legacy two-stage
  pipeline" wording (line 950), §4.3 "legacy single-tracer diagnostic"
  (line 1009), and §4.3 "legacy treatment" (lines 1019, 1034, 1042) the
  paper still carries a noticeable amount of old framing. The conclusions
  could simply state the construction without re-litigating the
  abandoned alternative.
- **Dash variant inconsistency.** "PFS--DESI" (em-/en-dash form) appears
  at lines 420, 554, 825, 1234. "PFS\(\times\)DESI" (channel notation)
  appears at lines 1244, 1285, 1289, 1472. These are *different concepts*
  (the geographic overlap region vs. the cross-spectrum channel), but the
  paper does not visually telegraph the distinction; a reader skimming §6
  sees both forms in adjacent paragraphs. Recommendation: keep
  "PFS--DESI overlap" for the footprint region and "PFS×DESI" for the
  cross-spectrum, and verify each instance.
- **σ digit count.** Item 2 quotes σ(Mν)=0.141 eV and σ(Ωm)=0.0157 with
  4 sig-fig precision; item 3 quotes σ(Mν)=0.108 eV and σ(Ωm)=0.0143
  consistently. Headline table identical. No drift.
- **Figure-vs-table rounding.** As above, fig. 6 caption uses 9/24/9%
  while §4.2 prose and §6 use 8.9/23.6/9.4%. Minor.
- **Item 1 self-praise vs. item 5 forward-look.** Item 1 (~10 lines) is
  almost entirely about *not* double-counting, i.e. negative framing.
  Item 5 then claims generalization in a completely different register.
  The five items as a set are not symmetric.

## SBP positioning (item 4)

Reads as **complementary HOD-dependent route**, not a competitor.
Specific phrases that lock this in:
- "not a replacement for simulation-based priors but a model-independent
  companion" (line 1268–1269);
- attributes b1 and stochastic terms to multi-tracer; counterterms and
  k-dependent stochasticity to SBPs (line 1271–1273);
- closes with the S8-tension framing — "an independent handle on whether
  SBP-driven σ8 shifts reflect new physics or HOD systematics"
  (line 1273–1275).

This is consistent with §2.5 and §5.3. No "alternative to SBPs" residue
in the conclusion paragraph itself.

## Generalization (item 5)

Concrete, not boilerplate. Item 5 names:
- the partition formula (2^N − 1 disjoint regions);
- a forward example (PFS×DESI×Euclid×Roman);
- four literature pairs cited via Alarcon:2016bkr and Qin:2025nkk
  (Euclid×DESI, Roman×Spec-S5, DESI-II×MegaMapper, J-PAS×PFS);
- the closing claim that "the same machinery handles any subset of
  pairwise or higher-order overlaps without reformulation."

This is a real, falsifiable claim, not handwaving. One mild concern:
the four literature pairs are bundled into a single citation each
(Alarcon:2016bkr for the first three, Qin:2025nkk for J-PAS×PFS), and
the conclusion does not summarize what those references established
about each pair — but this is appropriate density for a conclusion
section, since the pairs are discussed earlier (see §5.3, lines 1196–1206).

## Appendices

### App. A — Validation of the Fisher pipeline

Contents: derivative validation (§A.1), covariance Wigner-3j vs GL
quadrature (§A.2), 5 Fisher checks (§A.3) including the volume-partition
identity F(V_overlap)+F(V_nonoverlap)=F(V_full) to 1e-9.

- **Old-framing residues:** none in App. A. Good.
- **Canonical channel noun phrases:** "internal-DESI and PFS×DESI overlap
  channels" (line 1471–1472) — matches §6 wording.
- **Eponymous mechanisms:** Wigner-3j formula attributed to "Rubira &
  Conteddu" (line 1420–1421) with eq. 2.20 cited; cosmic-variance-free
  floor cited to McDonald:2008sh (line 1500). Both appropriate.
- **Note:** the SSC sub-leading caveat (line 1481–1485) cites
  Wadekar:2019rdu and Sugiyama:2019ike — same pair appears at line 434.
  Consistent attribution.

### App. B — EFT priors from co-evolution

- **Old-framing residues:** none.
- **Eponymous attribution:** Lazeyras et al. (lines 1566, 627–628) and
  Mergulhão et al. (line 1570 with both 2021kip and 2023zso, plus their
  Appendix B explicitly named) — both well-attributed and earn their
  citations.
- **Closing comparison:** "the 8–22% gain from adding the PFS overlap"
  (line 1597) — this is a *range across the three cosmological
  parameters* (8.9 / 23.6 / 9.4 % rounded as 8–22%? No: 8.9 to 23.6
  rounds to "9–24%", not "8–22%"). The "8–22%" range understates the
  Mν gain. Minor numeric drift.

## Tone and style

- Conclusion is mostly tight. No 5+-citation stacks in §6 or appendices
  (the largest is item 5's `\cite{Alarcon:2016bkr}` followed in the same
  sentence by `\cite{Qin:2025nkk}` — two cites, one each per scope).
  The 5-cite stack at intro line 206 is out of scope here.
- No promotional phrasing such as "unprecedented", "first-ever",
  "powerful". Item 1's "avoids the auto-spectra double-counting" is the
  most assertive line and is technically defensible (validated in
  §A.3 check 3).
- Acknowledgments contain the personal "two villages to build PFS and
  DESI" and "VJA red-eye flights" anecdotes (line 1302). Author's
  call.
- Item 5 uses "naturally", a soft hedge ("framework generalizes
  naturally"). Minor.

## Citations

- **Claims with attached \cite that still hold:**
  - Alarcon:2016bkr at line 1289 — supports prior multi-survey forecasts.
  - Qin:2025nkk at line 1291 — J-PAS×PFS synergies; matches usage at
    lines 206, 1206.
  - Wadekar:2019rdu, Sugiyama:2019ike (line 1483) — SSC sub-leading.
  - Rubira:2025scu (lines 1421, 1438) — Wigner-3j Eq. 2.20.
  - McDonald:2008sh (line 1500) — cosmic-variance-free floor.
  - Lazeyras:2015lgp, Mergulhao:2021kip, Mergulhao:2023zso (App. B) —
    all three correctly cited and discussed individually.
- **Missing citations:** none flagged in §6 or apps.
- **No \cite attached to claims that have been overtaken**, given the
  "complementary not competitor" framing of item 4.

## Verdict

**NEEDS-ANOTHER-LOOP** — small list of polish items, not a structural
problem.

Specific edits:
1. **Item 1, lines 1236–1237 and 1247–1248.** The "rather than as a
   separate prior-export step" preface and the "two-stage prior-export
   pipeline" coda re-litigate the abandoned alternative *twice* in one
   item. Pick one. Suggestion: drop the lead-in clause "treating
   cross-survey calibration as part of a single Fisher … rather than as
   a separate prior-export step" from the opening paragraph (line
   1234–1237) and keep only the item-1 closing sentence.
2. **App. B, line 1597.** Change "8–22% gain" to "9–24% gain" so the
   range matches the headline 8.9 / 23.6 / 9.4 % (or pick endpoints
   that reflect the actual computed values).
3. **Dash audit.** Skim §6 (and §4.2) once more and choose: "PFS--DESI
   overlap" for the footprint region; "PFS×DESI" for the cross-spectrum
   channel. Currently mixed within close proximity (lines 1234, 1244,
   1285, 1289).
4. **Optional — item 5 wording.** "generalizes naturally" → "extends
   directly" or simply "extends to N overlapping spectroscopic
   surveys." Removes a soft hedge.
5. **Optional — figure-vs-prose rounding.** Either round the figure-6
   caption (line 897–899) to one decimal, or accept the discrepancy
   as a caption-vs-table convention.

The five items are otherwise complete, the (a)→(b)→(c) chain is
internally consistent and matches `tab:headline` exactly, and SBP
positioning is correctly framed as complementary. Appendices are clean
on old-framing.
