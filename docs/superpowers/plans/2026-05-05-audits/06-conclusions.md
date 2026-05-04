# §6 Conclusions — audit report

Source: `paper/PFSxDESI-multi-fish.tex`, lines 1185–1253.

## Item-by-item audit

### Item 1 (lines 1200–1208) — Volume-partitioned joint Fisher
- Claim: \(F = F_{\rm overlap}(1{,}200\,\text{deg}^2) + F_{\rm nonoverlap}(12{,}800\,\text{deg}^2)\); cosmology shared globally, nuisance per (tracer, z-bin); the framework "avoids the auto-spectra double-counting inherent to a two-stage calibration-then-cosmology pipeline."
- Hierarchy position: PASS — framework definition correctly sits at rank 1.
- σ values: N/A (no σ given here, correct).
- Old-framing: PARTIAL FAIL — the phrase "two-stage calibration-then-cosmology pipeline" is invoked as a foil, which is appropriate and does not reintroduce the old framing, but the wording still surfaces "calibration-then-cosmology" prominently. Acceptable as contrast; flag for tone polishing only.
- Recommended edit: minor — rephrase to keep the contrast crisp without echoing the deprecated label twice (intro paragraph already says "rather than as a separate prior-export step").

### Item 2 (lines 1210–1216) — Multi-tracer information from bias-distinct DESI tracers
- Claim: bias-distinct LRG/ELG/QSO span a wide \(b_1\) range over \(0.4<z<2.1\); DESI alone reaches \(\sigma(f\sigma_8)=0.014\), \(\sigma(M_\nu)=0.14\,\text{eV}\), \(\sigma(\Omega_m)=0.016\) at \(\kmax=0.20\,h\,\text{Mpc}^{-1}\) without cross-survey input.
- Hierarchy position: PASS — multi-tracer dominance correctly precedes PFS-on-top.
- σ values: PASS — 0.014 / 0.14 eV / 0.016 matches the rounded DESI-only joint multi-tracer numbers (precise: 0.0145 / 0.141 / 0.0157). Spec allows the rounded form.
- Old-framing: PASS — no "exported priors" / "Step 1 → Step 2" language.
- Recommended edit: none required. Optional: add the more-precise σ in parentheses for symmetry with item 3 if item 3 keeps three sig figs.

### Item 3 (lines 1218–1225) — PFS-on-top in the 1,200 deg² overlap
- Claim: Adding PFS tightens \(\sigma(f\sigma_8)\) by 9%, \(\sigma(M_\nu)\) by 24%, \(\sigma(\Omega_m)\) by 9%, reaching 0.013 / 0.108 eV / 0.014; largest gain on \(M_\nu\), driven by \(b_1\) calibration of DESI-ELG via the low-bias PFS-ELG partner.
- Hierarchy position: PASS — PFS-on-top correctly at rank 3.
- σ values: PASS — 9 / 24 / 9 % matches spec; final σ = 0.013 / 0.108 / 0.014 matches rounded targets (precise 0.0132 / 0.108 / 0.0143). Internal consistency: 0.0145→0.0132 is 9.0%; 0.141→0.108 is 23.4% (rounds to 24% only with looser rounding — see Aggregate Concerns); 0.0157→0.0143 is 8.9%. The 24% claim is borderline; verify against the Fisher pipeline's exact ratio.
- Old-framing: PASS — describes the joint Fisher channel directly, no prior-export language.
- Recommended edit: confirm whether 24% should be 23% given the precise inputs; otherwise none.

### Item 4 (lines 1227–1234) — SBP complementarity
- Claim: the joint analysis is "not a replacement for simulation-based priors but a model-independent companion"; multi-tracer cross-correlations constrain \(b_1\) and stochastic terms from data, while SBPs additionally tighten counterterms and \(k\)-dependent stochasticity beyond multi-tracer reach. Frames the data-driven channel as an independent handle on whether SBP-driven \(\sigma_8\) shifts reflect new physics or HOD systematics.
- Hierarchy position: PASS — SBP positioning correctly at rank 4.
- σ values: N/A.
- Old-framing: PASS — explicitly disavows "alternative to SBPs" framing; treats SBPs as complementary HOD-dependent route with multi-tracer priors as the model-independent check, exactly as required.
- Recommended edit: none. Strongest item in the section.

### Item 5 (lines 1236–1252) — Generalization to N surveys
- Claim: union partitions into up to \(2^N-1\) disjoint regions, each carrying its own multi-tracer Fisher; all add in a shared parameter space. Concrete example: PFS×DESI×Euclid×Roman in partial overlaps. Cites prior pairwise discussions (Euclid×DESI, Roman×Spec-S5, DESI-II×MegaMapper, J-PAS×PFS) and asserts the same machinery handles arbitrary subset overlaps.
- Hierarchy position: PASS — generalization correctly at rank 5.
- σ values: N/A.
- Old-framing: PASS.
- Concrete vs. boilerplate: PASS — the \(2^N-1\) partition statement plus the four-survey example is concrete; the literature contrast names specific pairs.
- Recommended edit: none required. Optional: tighten by replacing "naturally accommodates" + "naturally" within ~7 lines (line 1243 and the implicit echo); diction-only.

## Aggregate concerns

- **24% vs. 23% on \(M_\nu\):** with precise inputs 0.141 → 0.108, the fractional improvement is 23.4%, which rounds to 23%, not 24%. Item 3 should either (a) cite the exact unrounded figure from the pipeline, or (b) state ≈24%. Cross-check against the source numbers in §5.
- **σ digit-count consistency:** item 2 uses two sig figs (0.014, 0.14, 0.016); item 3 uses two-or-three sig figs (0.013, 0.108, 0.014). Standardize. If the abstract or §5 already uses three-digit precise values, mirror them here.
- **No conflict** between "DESI alone 0.014" and "DESI+PFS 0.013": at the precise level (0.0145 → 0.0132) this is a 9.0% improvement, exactly matching item 3.
- **Framework label drift:** intro paragraph says "joint multi-tracer Fisher analysis"; item 1 says "volume-partitioned joint Fisher"; item 5 says "the framework". Acceptable variation, but the noun phrase "volume-partitioned joint Fisher" never reappears outside item 1 — consider re-using it once more in item 5 to anchor the generalization.

## Citation audit

- Line 1185 (`\cite{Ivanov:2023qzb,Bakx:2025pop}`): outside §6 proper (tail of §5); not in scope.
- Lines 1248–1250 (`\cite{Alarcon:2016bkr}` and `\cite{Qin:2025nkk}`):
  - `Alarcon:2016bkr` — supports the Euclid×DESI / Roman×Spec-S5 / DESI-II×MegaMapper cluster of prior multi-survey discussions. PASS if Alarcon et al. indeed covers Euclid×DESI overlap forecasting; otherwise the citation is being asked to carry three pairs at once. Verify against the bib entry.
  - `Qin:2025nkk` — supports J-PAS×PFS specifically. PASS provided the bib entry is the J-PAS×PFS forecast.
- §6 contains no other `\cite{}` calls.

## Tone and style

- "We have presented", "We summarize" — appropriate declarative voice.
- No "headline" / "main result" / "McDonald-Seljak" — PASS.
- Em-dash asides used (line 1220 "9% and \(\sigma(\Omega_m)\) by 9\%---reaching"; line 1248 "literature---Euclid"). Both are grammatical asides, consistent with house style.
- Body noun phrases: "joint multi-tracer Fisher" (line 1193), "multi-tracer self-calibration" (line 1203), "PFS–DESI cross-survey channel" (line 1204), "data-driven channel" (line 1233) — all align with the required vocabulary. "Internal-DESI channel" / "PFS×DESI overlap channel" do not appear verbatim, but the substance is conveyed via "DESI internal multi-tracer self-calibration" and "PFS-DESI cross-survey channel". Optional: align item 1 wording to the exact channel labels for consistency with §4–§5.
- "Step 1 / Step 2", "exported priors", "calibration-then-cosmology pipeline" — only the last appears, used as a deliberate foil in item 1; flag for review but not a hard fail.
- Hyphenation: "DESI-LRG", "DESI-ELG", "DESI-QSO", "PFS-ELG" — PASS.
- En-dash vs hyphen: "PFS--DESI" (line 1194, en-dash) and "PFS-DESI" (line 1204, hyphen) co-occur within seven lines. Inconsistent — pick one.

## Recommended edits (ranked)

1. **Lines 1204 vs 1194 — en-dash/hyphen inconsistency.**
   Current (1194): `PFS--DESI overlap`
   Current (1204): `the PFS-DESI cross-survey channel`
   Proposed: change line 1204 to `the PFS--DESI cross-survey channel`.
   Justification: a single section should not switch between en-dash and hyphen for the same compound.

2. **Line 1219 — verify 24% claim.**
   Current: `\(\sigma(\Mnu)\) by 24\%`
   Proposed (option A, if pipeline truly gives 23.4%): `\(\sigma(\Mnu)\) by 23\%`
   Proposed (option B, if rounded headline is preferred): `\(\sigma(\Mnu)\) by \(\sim\)24\%`
   Justification: 0.141 → 0.108 is 23.4%, which rounds down; the unqualified "24%" is borderline-wrong unless the underlying numbers differ from those advertised.

3. **Line 1213 vs line 1221 — σ precision parity.**
   Current (1213): `\(\sigma(f\se)=0.014\), \(\sigma(\Mnu)=0.14\,\text{eV}\), \(\sigma(\Om)=0.016\)`
   Current (1221): `\(\sigma(f\se)=0.013\), \(\sigma(\Mnu)=0.108\,\text{eV}\), \(\sigma(\Om)=0.014\)`
   Proposed: either round item 3 to two sig figs (0.013 / 0.11 eV / 0.014) or upgrade item 2 to three sig figs (0.0145 / 0.141 eV / 0.0157), matching whichever convention §5 uses.
   Justification: side-by-side σ should share digit count for direct comparison.

4. **Lines 1206–1208 — soften the "calibration-then-cosmology" foil.**
   Current: `The framework avoids the auto-spectra double-counting inherent to a two-stage calibration-then-cosmology pipeline.`
   Proposed: `The single-Fisher construction avoids the auto-spectra double-counting that a two-stage prior-export pipeline incurs.`
   Justification: the deprecated label "calibration-then-cosmology" appears nowhere else in the paper post-revision; using "two-stage prior-export" matches the language already used in the section's opening paragraph (line 1196).

5. **Line 1243 — diction de-duplication.**
   Current: `This naturally accommodates upcoming multi-survey configurations such as PFS\(\times\)DESI\(\times\)Euclid\(\times\)Roman in partial overlaps`
   Proposed: `This accommodates upcoming multi-survey configurations such as PFS\(\times\)DESI\(\times\)Euclid\(\times\)Roman in partial overlaps`
   Justification: "naturally" already appears in the preceding sentence ("generalizes naturally"); the second use is filler.

6. **Optional, line 1204 — explicit channel naming.**
   Current: `combines the DESI internal multi-tracer self-calibration over the full footprint with the PFS-DESI cross-survey channel in the overlap`
   Proposed: `combines the internal-DESI channel over the full footprint with the PFS\(\times\)DESI overlap channel in the 1{,}200~deg\(^2\) intersection`
   Justification: aligns item 1's vocabulary with the channel labels mandated for §§4–5 of the revised draft.
