# §5 Discussion — audit report

## Diagnostic-chain paragraph (§5.1) — read-only check

Paragraph: lines 1050-1079 (i Localization, ii Correlation collapse, iii Data dominance + Zhao eBOSS anchor).

- **Internal consistency (numbers match each other and the rest of the paper): PASS.** 33%/80%/49% on (σ(f), σ(Mν), σ(Ωm)) at line 1057 matches abstract line 127 and §4 line 805. Per-bin Δσ(Mν) of 67–85% is internal to the diagnostic and not contradicted elsewhere. Single-tracer-only z-bin assignment (z<0.8 LRG-only, z>1.6 QSO-only; multi-tracer at 0.8<z<1.6) is consistent with the survey-spec n(z) tables and the 8-bin coverage stated at line 1104. ρ values {+0.95, +0.82, +0.45} → {+0.39, −0.07, −0.07} are diagnostic-internal and not otherwise quoted. Zhao eBOSS 18–27% on σ8 anchor is internal to the citation, not contradicted.
- **Supports (a)→(b)→(c) hierarchy: PASS.** Localization isolates the gain to the multi-DESI-tracer bins (b); correlation collapse identifies cross-spectrum bias-ratio determination as the mechanism inside (b); data dominance shows the gain is not prior-driven; Zhao anchor empirically validates (b) at smaller scale. The paragraph is squarely about why (b) dominates — appropriate for §5.1 — and the surrounding prose attaches (c) as the residual PFS tightening.
- **Citations (Seljak:2008xr, McDonald:2008sh, Zhao:2024eboss, Mergulhao:2023zso, Rubira:2025scu) correctly attached: PASS.** Seljak:2008xr and McDonald:2008sh attach to the cosmic-variance-free bias-ratio statement (line 1033). Mergulhao:2023zso and Rubira:2025scu attach to the EFT one-loop k-dependent shape pinning individual b1 (line 1035). Zhao:2024eboss attaches to the eBOSS DR16 LRG×ELG empirical anchor (line 1074). Each citation is correctly placed.

**Verdict: no change recommended (guarded).**

## Old-framing residues elsewhere in §5

- **"Prior-export" prose:** none found in §5 body. The token "prior-export" appears at line 1197 (§6 outlook) — outside §5 scope.
- **"Multi-survey priors" as umbrella:** not found in §5. §5.3 uses "data-driven multi-tracer priors" consistently (lines 1136-1137).
- **"McDonald–Seljak mechanism":** not found in §5. The mechanism is referred to as "cosmic-variance cancellation in cross-spectrum bias ratios" / "cosmic-variance-free in the high-density limit" / "cosmic-variance-cancelled bias-ratio determination" — consistent with the required noun phrase.
- **"Headline" / "main result" prose:** "headline" appears once at line 1102 inside the §5.1 figure caption ("the 24% improvement on σ(Mν) reported in \cref{tab:headline}"). This is a `\cref{tab:headline}` table reference, not the prohibited "headline result" prose. The label `tab:headline` is structural and changing it would be invasive. **No flag** for prose; one minor edit option below to remove residual framing connotation.

## Two-channel narrative consistency

§5 reinforces (a)→(b)→(c). §5.1 explicitly identifies (b) as the dominant driver: "The DESI-only joint analysis already exploits this through its three bias-distinct tracers in the full 14,000 deg² footprint, collapsing the broad-prior ellipse" (lines 1035-1038), and (c) as supplementary: "Adding PFS-ELG in the 1,200 deg² overlap contributes a fourth, low-bias ELG sample … tighten b1 for each (DESI tracer, redshift bin) further" (lines 1039-1044). The figure caption (lines 1097-1101) makes the hierarchy explicit: "DESI-internal multi-tracer self-calibration is doing the heavy lifting … PFS contributes the residual tightening". The diagnostic chain in lines 1050-1079 explains *why* (b) dominates (cosmic-variance cancellation, ρ collapse, data dominance). Consistent throughout.

## SBP positioning (§5.3)

§5.3 correctly positions multi-tracer as the model-independent cross-check on SBPs, not the reverse. Line 1115-1118: "The latter [joint Fisher] is model-independent by construction and provides a cross-check on whether SBP-calibrated b1 values are consistent with what the data themselves imply." Line 1144-1147: "A future PFS×DESI joint multi-tracer analysis that reproduces this shift without HOD input would strengthen the new-physics interpretation; a null result would point to systematic biases in the galaxy–halo connection." Directionality is consistent — multi-tracer is the cross-check on SBP/HOD assumptions. No reversal flagged.

Note: the §5.3 paragraph beginning at line 1108 ("SBPs achieve a similar effect through a different route") is *physically* in §5.1 (lines 1108-1118, before the §5.2 header at line 1121). This is structural — content belongs in §5.3 — but it functions as a bridge from the §5.1 mechanism to §5.3 implications. Flag for consideration; not load-bearing for narrative correctness.

## Citation audit

Every `\cite{}` in §5 (lines 1012-1183):

- L1033 `Seljak:2008xr,McDonald:2008sh` — supports cosmic-variance-free bias ratio in high-density limit. Correct.
- L1035 `Mergulhao:2023zso,Rubira:2025scu` — supports k-dependent EFT one-loop shape pinning individual b1. Correct.
- L1074 `Zhao:2024eboss` — eBOSS DR16 LRG×ELG empirical anchor for σ(σ8) gain. Correct.
- L1109 `Chudaykin:2026nls` — field-level SBPs shrink b1 by 30–40%. Correct.
- L1140 `Chudaykin:2026nls` — SBPs shift σ8 downward by ~2σ. Correct (same paper, different result).
- L1156 `DESI:2024hhd,Chudaykin:2025lww` — DR1+CMB prefer normal hierarchy. Correct.
- L1162 `Alarcon:2016bkr` — generalization to overlapping spectroscopic survey pairs. Correct.
- L1166 `Qin:2025nkk` — J-PAS×PFS joint forecasts. Correct.
- L1167 `Ivanov:2023qzb,Bakx:2025pop` — tree-level bispectrum activates c1 and breaks b1–b2–bG2 degeneracies. Correct.
- L1177 `FernandezGarcia:2025rso` — Uchuu-LRG mock. Correct.
- L1178 `Vaisakh:inprep` — Uchuu-ELG. Correct.
- L1179 `Ishiyama:inprep` — Mucho Uchuu. Correct.
- L1180 `PFS-CO-mock:inprep` — PFS galaxy mocks. Correct.

All bibkeys exist in `PFSxDESI-multi-fish.bib`. No missing entries. No citation stacking violations (max 2 grouped, each sentence supplies context).

## Tone and style

- Declarative voice maintained throughout. No "headline", "main result", or "McDonald–Seljak mechanism" in prose.
- Em-dash asides present and consistent ("source: SBPs derive…" line 1113-1115; "—Euclid×DESI, Roman×Spec-S5…" line 1163-1164).
- Required noun phrases present: "joint multi-tracer Fisher" (1016, 1110, 1144), "data-driven multi-tracer priors" (1136-1137), "cosmic-variance cancellation" (1066-1067, 1093), "DESI-internal multi-tracer" (1077, 1097-1098).
- Minor: line 1098-1099 "doing the heavy lifting" is colloquial but inside a figure caption and consistent with the "(a)→(b)→(c)" hierarchy framing. Not a tone violation.
- §5.2 (lines 1121-1130) is short — three sentences. The "headline"-drop edit is clean: prose says "PFS-induced improvement is stable" rather than "headline result is robust". Consistent with the new framing.
- §5.3 uses "data-driven multi-tracer priors" rather than "multi-survey priors". Consistent.

No tone deviations flagged.

## Recommended edits (ranked)

### 1. Diagnostic-chain paragraph (lines 1050-1079)
**No change recommended (guarded).** All three read-only checks pass.

### 2. Optional — clarify "tab:headline" prose connotation in figure caption (line 1100-1102)
- **Location:** lines 1100-1102 (figure caption).
- **Current text:** `PFS contributes the residual tightening that yields the 24\% improvement on \(\sigma(\Mnu)\) reported in \cref{tab:headline}.`
- **Proposed change:** `PFS contributes the residual tightening that yields the 24\% improvement on \(\sigma(\Mnu)\) reported in \cref{tab:headline} (DESI MT \(\sigma(\Mnu)=0.141\,\text{eV}\) \(\to\) DESI+PFS \(0.108\,\text{eV}\)).`
- **Justification:** Anchors the percentage to the absolute σ values from the table and reduces the "headline" framing connotation by inlining the actual numbers. Optional, not required.

### 3. Optional — relocate the "SBPs achieve a similar effect" paragraph (lines 1108-1118)
- **Location:** lines 1108-1118 currently sit at the end of §5.1 before the §5.2 header.
- **Current text:** opens with `SBPs achieve a similar effect through a different route.`
- **Proposed change:** Move this paragraph to the top of §5.3 (before line 1136), since its content (SBP comparison, model-independence cross-check) is the §5.3 thesis. Replace its current location with a one-sentence pointer: `We return to the comparison with simulation-based priors, which target the same b1 calibration through an independent route, in \cref{sec:discussion} below.` (or omit entirely — §5.3 already opens with the SBP comparison).
- **Justification:** Structural — content overlaps with §5.3 opening (both invoke Chudaykin:2026nls and the model-independence argument). Consolidating avoids redundancy. Optional; current placement functions as a bridge and is not incorrect.

### 4. No other edits.

The body of §5 is consistent with the joint multi-tracer Fisher framing, the (a)→(b)→(c) hierarchy is reinforced throughout, all numbers match the rest of the paper, and all citations resolve and attach correctly.
