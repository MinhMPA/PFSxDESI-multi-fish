# §4 Results — audit report

## Old-framing residues

- **L756–759** ("As we show in \cref{sec:discussion}, the \(b_1\se\) calibration is the dominant driver of the cosmological improvement…") — "calibration" framing is a residue; under joint-Fisher framing the more accurate phrasing is "the joint-Fisher tightening of \(b_1\se\)" or "the multi-tracer leverage on \(b_1\se\)". Soft residue, low risk.
- **L793** ("We now combine the cross-survey overlap *calibration* with the DESI internal multi-tracer self-calibration…") — "cross-survey overlap calibration" reads as Stage-1 prior export. Should be "PFS×DESI overlap channel".
- **L832–842** ("captures most of the multi-tracer *calibration leverage*… further tightens the *calibration*…") — uses "calibration" three times where "leverage" / "constraining power" would fit the joint-Fisher framing. Soft residue.
- **L906–917** §4.3 *Calibration efficiency* — the section is explicitly labeled "legacy two-stage pipeline" diagnostic and is internally honest about that ("does not factorize per redshift bin; this figure remains a useful diagnostic"). The framing band-aid is in place but the section title "Calibration efficiency" still uses Stage-1 vocabulary — defensible as a diagnostic label.
- **L968–973, L979–989** Fig. sensitivity — "legacy single-tracer DESI-ELG configuration with multi-tracer-calibrated priors" — explicitly flagged as legacy diagnostic. Internally consistent.
- **L991–1008** §4.4 cross-stoch comparison — talks about "legacy treatment" vs "conservative prescription" and quotes Δ% deltas. Internally consistent under joint Fisher.
- No "Step 1 / Step 2" wording found.
- "headline" prose: **L888** caption ("Headline cosmology constraints from the joint multi-tracer Fisher analysis…") — survives in the table caption itself. The constraint says don't retitle `tab:headline` (label name fixed) but the noun phrase "Headline cosmology constraints" in the *caption text* should be removed.
- No body-text occurrences of "headline numbers" / "headline cosmology" outside the caption.

## Two-channel narrative consistency

- The (a)→(b)→(c) chain is **partially visible** in §4.2:
  - (a) DESI single-tracer broad: invoked only by reference to Fig. full_area legend ("Red: DESI single-tracer analysis with broad nuisance priors"). It is **not in Tab. headline** — Tab. headline lists only DESI-only joint and DESI+PFS joint, with Δ% computed against DESI-only joint. The (a)→(b) step (broad → DESI MT joint, 33/80/49%) is **invisible in the table**.
  - (b) DESI-only joint: present in Tab. headline row 1 with σ = 0.0145, 0.141, 0.0157.
  - (c) DESI+PFS joint: Tab. headline row 2 with σ = 0.0132, 0.108, 0.0143 and Δ% = -8.9 / -23.6 / -9.4.
- **Gap:** Tab. headline omits the broad-prior baseline (a). The body sentence at L832–839 mentions DESI-internal cross-correlations breaking the b1σ8 degeneracy but does not quote the 33/80/49% (a)→(b) numbers anywhere in §4. This is the largest narrative weakness.
- Δ% in Tab. headline: -8.9 / -23.6 / -9.4 — **correct** vs the spec.

## σ value consistency

| Quantity | Spec | In §4 | Status |
|---|---|---|---|
| DESI-only joint σ(fσ8) | 0.0145 | 0.0145 (L884) | OK |
| DESI-only joint σ(Mν) | 0.141 | 0.141 (L884) | OK |
| DESI-only joint σ(Ωm) | 0.0157 | 0.0157 (L884) | OK |
| DESI+PFS joint σ(fσ8) | 0.0132 | 0.0132 (L885) | OK |
| DESI+PFS joint σ(Mν) | 0.108 | 0.108 (L885) | OK |
| DESI+PFS joint σ(Ωm) | 0.0143 | 0.0143 (L885) | OK |
| PFS-unique Δ% | 8.9/23.6/9.4 | 8.9/23.6/9.4 (L885, L1002) | OK |
| DESI single-tracer broad σ | 0.0216, 0.688, 0.0306 | **absent** | Gap |
| DESI MT vs broad Δ% | 33/80/49 | **absent** | Gap |
| Body L805–806 | 9 / 24 / 9% | matches Δ% (rounded) | OK |
| Body L843–845 | +24 pp / +9 pp / +9 pp | OK |
| §4.4 legacy Δ% | 8.6/22.8/9.7 → 8.9/23.6/9.4 | quoted L1002 | OK (legacy nums are session-internal) |
| §4.4 broadening | 2.7 / 9.2 / 2.2% | L999–1000 | OK (session-internal) |

## SBP positioning

- **L845–847**: "For context, power-spectrum-level SBPs on DESI~DR1~\cite{DESI:2025wzd} achieved \(\sim\!23\%\) on \(\sigma(\se)\)." — "For context" frames it as a benchmark, not a competitor. OK.
- **L869–874** Fig. full_area caption: SBP lines explicitly described as "benchmarks". The caveat about field-level SBPs worsening the Mν bound is descriptive, not adversarial. OK.
- No passage frames SBP as a parallel/competing route. Positioning is "benchmark / reference".

## Citation audit

| Line | Citation | Support |
|---|---|---|
| L718 | DESI:2025zgx | Imaging-systematics control on DESI ELGs — appropriate. |
| L735 | Chudaykin:2025lww, DESI:2024jxi | Source of broad nuisance priors used in the comparison — appropriate. |
| L745 | Seljak:2008xr, McDonald:2008sh | Multi-tracer bias-ratio / cosmic-variance cancellation — canonical, appropriate. |
| L839 | Seljak:2008xr, McDonald:2008sh | Same as above, used to justify DESI-internal MT leverage — appropriate. |
| L846 | DESI:2025wzd | Power-spectrum SBP DR1 ~23% on σ8 — needs verification of the exact number; flag for cross-check but cite is appropriate in scope. |
| L870 | DESI:2025wzd | PSB benchmark dashed line — appropriate. |
| L871 | Chudaykin:2026nls | Field-level SBP benchmark dotted line — appropriate; verify Tab.~IV reference matches their paper structure. |
| L874 | (in caption) | Cites Chudaykin:2026nls Tab.~IV for Mν worsening — appropriate. |
| L899 | Ebina:2024ojt | Cross-stochasticity treatment — appropriate (cited as the source of the marginalization scheme). |

No missing-cite gaps flagged. No `\cite{}` stacks beyond two per cluster.

## Tone and style

- "remarkably insensitive" (L956) — mild editorializing but acceptable; declarative-emphatic.
- Em-dash asides used appropriately (L741–742, L754, L763, L767, L962, L968, L1004).
- No "main result" / eponymous names found.
- "headline" appears once in caption text (Tab. headline caption, L888) — to be removed.
- "Headline cosmology" / "headline numbers" — only the caption survives.
- "calibration efficiency" / "calibration leverage" — residual two-stage vocabulary in §4.3, partially defended as "diagnostic" framing.

σ(Mν) diagnostic chain: not found in §4 (this section discusses 24% PFS-unique improvement and 9.2% conservative-vs-legacy broadening, but no per-z-bin localization / kmax monotonic scan / ρ collapse / data-domination chain). Likely lives in §5.1 — out of scope for this audit. No softening proposed.

Two-channel noun phrases: "internal-DESI channel" / "PFS×DESI overlap channel" do **not appear** in §4 — body uses "DESI internal multi-tracer self-calibration" (L794), "PFS overlap" (L840), "DESI-only shell" (L798). These are content-equivalent but not the canonical channel-named phrasing.

## Recommended edits (ranked)

### 1. Tab. headline caption — strip "Headline cosmology"
- **Location:** L888
- **Current:** `\caption{Headline cosmology constraints from the joint multi-tracer`
- **Proposed:** `\caption{Cosmology constraints from the joint multi-tracer`
- **Justification:** Removes the last "headline" noun-phrase residue per session edit; label `tab:headline` itself is unchanged.

### 2. Add the broad-prior baseline row to Tab. headline (or quote the (a)→(b) numbers in body)
- **Location:** L883–886 (table body) or L805–807 (body sentence)
- **Current (body, L804–806):** `\Cref{fig:full_area} and \cref{tab:headline} present the cosmology forecasts. Adding the PFS overlap to the DESI-only joint baseline improves \(\sigma(f\se)\) by 9\%, \(\sigma(\Mnu)\) by 24\%, and \(\sigma(\Om)\) by 9\% at \(\kmax=0.20\iMpch\).`
- **Proposed (body):** `\Cref{fig:full_area} and \cref{tab:headline} present the cosmology forecasts. The DESI internal multi-tracer joint Fisher tightens the broad single-tracer baseline (\(\sigma(f\se,\Mnu,\Om)=0.0216, 0.688, 0.0306\)) by 33\%, 80\%, and 49\%; adding the PFS\(\times\)DESI overlap channel further improves \(\sigma(f\se)\) by 8.9\%, \(\sigma(\Mnu)\) by 23.6\%, and \(\sigma(\Om)\) by 9.4\% at \(\kmax=0.20\iMpch\).`
- **Justification:** Makes the (a)→(b)→(c) chain visible with all six σ values in the body, restoring the keystone narrative.

### 3. §4.2 opening — channel-named phrasing
- **Location:** L793–796
- **Current:** `We now combine the cross-survey overlap calibration with the DESI internal multi-tracer self-calibration over the full \(14{,}000\;\text{deg}^2\) footprint, in a single joint Fisher analysis.`
- **Proposed:** `We now combine the PFS\(\times\)DESI overlap channel with the internal-DESI channel over the full \(14{,}000\;\text{deg}^2\) footprint in a single joint Fisher analysis.`
- **Justification:** Replaces two-stage "calibration" vocabulary with the canonical channel nouns; removes Stage-1 residue.

### 4. §4.2 — replace "calibration leverage" with joint-Fisher vocabulary
- **Location:** L832–842
- **Current:** `The DESI-only joint Fisher already captures most of the multi-tracer calibration leverage available in the data… Adding PFS to the \(1{,}200\;\text{deg}^2\) overlap further tightens the calibration through a fourth, low-bias ELG sample…`
- **Proposed:** `The internal-DESI channel already captures most of the multi-tracer information available in the data… Adding PFS to the \(1{,}200\;\text{deg}^2\) overlap channel further tightens the joint Fisher through a fourth, low-bias ELG sample…`
- **Justification:** Brings the body in line with two-channel noun phrases; "information" / "joint Fisher" replace the Stage-1 "calibration leverage".

### 5. §4.1 — soften "calibration is the dominant driver"
- **Location:** L756–759
- **Current:** `As we show in \cref{sec:discussion}, the \(b_1\se\) calibration is the dominant driver of the cosmological improvement, because \(\Mnu\) is nearly perfectly degenerate with \(b_1\) in the single-tracer Fisher.`
- **Proposed:** `As we show in \cref{sec:discussion}, the joint-Fisher tightening of \(b_1\se\) is the dominant driver of the cosmological improvement, because \(\Mnu\) is nearly perfectly degenerate with \(b_1\) in the single-tracer Fisher.`
- **Justification:** Replaces "calibration" with "joint-Fisher tightening" — same physics, joint-Fisher framing.

### 6. §4.1 caption / opening — make "projection" framing crisper
- **Location:** L731–733 (already revised in this session)
- **Current:** `To make the multi-tracer information explicit, we project the joint Fisher onto each (DESI tracer, redshift bin) and compare the resulting per-z-bin nuisance \(\sigma\) values with the conservative broad priors`
- **Verdict:** OK as written — explicit projection language is in place. No change needed.

### 7. §4.3 — preserve "legacy two-stage pipeline" honesty
- **Location:** L909–917
- **Verdict:** Already correctly labeled as a legacy diagnostic with the disclaimer "does not factorize per redshift bin". Internally consistent. No change required.

### 8. §4.4 (numbered §4.3 "Robustness") — asymmetric kmax + cross-stoch
- **Location:** L947–1008
- **Verdict:** Asymmetric `\kmax^{\PFS}\propto \rsv^{-1}` correctly stated (L954); the gain-on-PFS-nuisance / marginal-on-cosmology decomposition (L962–973) is consistent. Cross-stoch sweep (L991–1008) reports broadening 2.7 / 9.2 / 2.2% and Δ% shift 8.6/22.8/9.7 → 8.9/23.6/9.4 — internally consistent. No change required.

### 9. (Optional) Tab. headline — add a third row for DESI single-tracer broad
- **Location:** L883–886
- **Current:** two-row table (DESI-only joint, DESI+PFS joint).
- **Proposed:** prepend `DESI broad single-tracer & 0.0216 & --- & 0.688 & --- & 0.0306 & --- \\` and update Δ% column convention (Δ% relative to broad baseline).
- **Justification:** Makes Tab. headline the keystone of the full (a)→(b)→(c) chain in one place. Larger structural change — flag for author decision.
