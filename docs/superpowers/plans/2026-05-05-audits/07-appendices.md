# Appendices — audit report

## Appendix structure

`\appendix` begins at line 1283. Two top-level appendix sections, with three subsections in the first:

- **App. A — Validation of the Fisher pipeline** (`app:Fisher_convergence`), lines 1285–1513
  - A.1 Jacobian matrix (`app:derivatives`), lines 1291–1328
  - A.2 Covariance matrix (`app:covariance`), lines 1332–1403
  - A.3 Fisher matrix (`app:fisher_checks`), lines 1407–1513 (includes the 5-check list and Fig. `fisher_info` at lines 1500–1513)
- **App. B — EFT priors from co-evolution** (`app:coevolution_priors`), lines 1515–1553

Bibliography starts at line 1558.

## Old-framing residues

The appendices are largely clean of the old prior-export framing. Specific residues:

- **Line 1511** (Fig. `fisher_info` caption): "this is the cost that data-driven multi-tracer priors aim to recover." This already reflects the recent edit from "multi-survey priors" → "data-driven multi-tracer priors". Consistent with body usage at lines 112, 1137, 1511. No further edit needed on the noun phrase, but see Recommended Edits — the framing "priors aim to recover" is mildly back-leaning toward the prior-export paradigm; the joint-Fisher framing would say the channel "recovers" or "absorbs" the loss directly.
- **Line 1437** ("our implementation of \cref{eq:volume_partition} adds the two region Fishers without double-counting"): neutral and correct under the joint-Fisher framing; no residue.
- No occurrences of "multi-survey priors", "McDonald-Seljak mechanism", "headline" or "main result" phrases anywhere in the appendices. (The token `tab:headline` occurs in the body but never in the appendix; not in scope.)
- No "prior-export" language inside the appendices. (The conclusions at line 1197 still say "without a separate prior-export step" — body, out of scope here.)

## Two-channel narrative consistency

The appendices are largely **neutral** with respect to the (a)→(b)→(c) chain. They focus on numerical validation, not narrative.

- **App. A.3 Check 3** (lines 1424–1443) explicitly verifies the volume partition `F(V_overlap) + F(V_nonoverlap) = F(V_full)` — this directly underpins the (a) internal-DESI / (b) PFS×DESI overlap channel decomposition and is *consistent* with the two-channel framing, but does not name the channels. Could be sharpened.
- **App. A.3 Check 4** (lines 1445–1450) "Scenario ordering" frames the comparison as `DESI+PFS ≤ DESI-only`, which is a two-channel statement in disguise. Acceptable, neutral.
- **App. A.3 Check 5** (lines 1451–1471) demonstrates the cosmic-variance-free floor in a 2-tracer Schur-marginalized linear-bias toy. This is the conceptual underpinning of "cosmic-variance cancellation in cross-spectrum bias ratios". The current text says "the multi-tracer error converges to the floor, confirming the cancellation of cosmic variance" (line 1483) — language is consistent with the approved noun phrase.
- **App. B** is purely about co-evolution priors and gravitational evolution; it correctly avoids touching the channel framing.

## SBP positioning

No SBP mentions in either appendix. App. B discusses Mergulhão et al.'s co-evolution analyses but never positions SBPs as parallel/competitor — they are referenced only as a methodological precedent for the same null result on co-evolution priors. Clean.

## Citation audit

Every `\cite{}` between lines 1283 and 1561, with one-line support check:

| Line | Cite key(s) | Support |
|------|-------------|---------|
| 1313 | Piras:2023aub, Jense:2024llt | cosmopower-jax emulator and CMB-twin networks — supports |
| 1380 | Rubira:2025scu | Wigner-3j decomposition Eq. 2.20 — supports (specific equation cited) |
| 1397 | Rubira:2025scu | same — supports |
| 1441 | Wadekar:2019rdu, Sugiyama:2019ike | SSC sub-leading correction — supports (paired, both relevant) |
| 1456 | McDonald:2008sh | cosmic-variance-free floor — supports (foundational for `2/N_modes` limit) |
| 1521 | Lazeyras:2015lgp | co-evolution relations for `b_2`, `b_G2` fiducials — supports |
| 1525 | Mergulhao:2021kip, Mergulhao:2023zso | tested whether imposing co-evolution improves constraints — supports (specifically Appendix B of those refs, cited correctly) |

No citation stacking >2 without individual discussion. No bare lists. All citations earn their place.

## Tone and style

- "Headline" / "main result" / "McDonald-Seljak mechanism": none.
- Em-dash asides used appropriately at lines 1305 (`evident---\(b_1\se\) and \(f\se\)`), 1421, 1429, 1510. Consistent with the declarative style.
- One mild deviation from "declarative, em-dash asides": line 1532 uses a colon-introduced enumeration before the table ("yields the following changes relative to the free-bias baseline with broad priors:"). Acceptable as a methods/results table preamble (allowed by the user's prose-only rule for tables) — keep.
- Numbered lists appear in App. A.3 (the 5 Fisher checks) — acceptable per the user's CLAUDE.md exception for validation/methods enumerations.
- Line 1483 caption ends "confirming the cancellation of cosmic variance." This is the *short* form. Body uses the longer "cosmic-variance cancellation in cross-spectrum bias ratios" (line 837). Suggest tightening the appendix caption to match the approved noun phrase exactly. See Recommended Edits.

## Recommended edits (ranked)

### 1. Fig. `ms09_convergence` caption — match approved noun phrase

- **Location:** line 1483 (caption of Fig. `fig:ms09_convergence`, lines 1473–1488)
- **Current text (verbatim):**
  > the multi-tracer error converges to the floor, confirming the
  > cancellation of cosmic variance.
- **Proposed change (verbatim):**
  > the multi-tracer error converges to the floor, confirming
  > cosmic-variance cancellation in cross-spectrum bias ratios.
- **Justification:** Aligns the caption with the approved body noun phrase (line 837) and replaces the bare "cancellation of cosmic variance" — which has historically been associated with the McDonald-Seljak label that the revision is removing — with the explicit mechanistic phrasing.

### 2. Fig. `fisher_info` caption — sharpen the joint-Fisher framing

- **Location:** lines 1510–1511 (caption of Fig. `fig:fisher_info`)
- **Current text (verbatim):**
  > For \(\Mnu\), the loss exceeds 3 orders of magnitude---this is
  > the cost that data-driven multi-tracer priors aim to recover.
- **Proposed change (verbatim):**
  > For \(\Mnu\), the loss exceeds 3 orders of magnitude---this is
  > the cost that the joint multi-tracer Fisher recovers via
  > data-driven multi-tracer priors on the nuisance sector.
- **Justification:** "priors aim to recover" still echoes the prior-export aspiration; the joint-Fisher framing recovers the information directly within a single likelihood. Keeps the approved noun phrase while making the mechanism unambiguous.

### 3. App. A.3 Check 3 — name the two channels

- **Location:** lines 1428–1437
- **Current text (verbatim):**
  > For the DESI-only configuration with all three
  > DESI tracers active in the bin \(z=[1.0,\,1.2]\), we verify the
  > volume partition
  > \(F(V_{\rm overlap})+F(V_{\rm nonoverlap})=F(V_{\rm full})\)
- **Proposed change (verbatim):**
  > For the DESI-only configuration with all three
  > DESI tracers active in the bin \(z=[1.0,\,1.2]\), we verify the
  > volume partition underlying the internal-DESI and
  > PFS\(\times\)DESI overlap channels,
  > \(F(V_{\rm overlap})+F(V_{\rm nonoverlap})=F(V_{\rm full})\),
- **Justification:** The volume partition identity *is* the mathematical statement of the two-channel decomposition. Naming the channels here ties the validation directly to the approved framing without adding new content.

### 4. App. A.3 Check 4 — channel-aware phrasing

- **Location:** lines 1445–1450
- **Current text (verbatim):**
  > \textbf{Scenario ordering.}
  >   The marginalized constraints obey
  >   \(\sigma_{\rm DESI+PFS}\leq\sigma_{\rm DESI\text{-}only}\)
  >   for all cosmological parameters: adding PFS to the joint Fisher
  >   can only tighten or preserve the constraints, never weaken them.
- **Proposed change (verbatim):**
  > \textbf{Scenario ordering.}
  >   The marginalized constraints obey
  >   \(\sigma_{\rm DESI+PFS}\leq\sigma_{\rm DESI\text{-}only}\)
  >   for all cosmological parameters: adding the PFS\(\times\)DESI
  >   overlap channel to the internal-DESI channel in the joint
  >   multi-tracer Fisher can only tighten or preserve the
  >   constraints, never weaken them.
- **Justification:** Replaces the bare "adding PFS to the joint Fisher" with the approved channel/Fisher noun phrases — consistent with the body and with edit #3 above.

### 5. App. A.1 — minor wording cleanup (optional)

- **Location:** line 1311
- **Current text (verbatim):**
  > every parameter agrees to better than \(10^{-9}\), confirming the
  > end-to-end autodiff implementation through both the
- **Proposed change:** none required; included only because "end-to-end autodiff implementation" is mildly hedge-prose. Leave as-is unless a wider polish pass is underway.
- **Justification:** Stylistic, not framing-related. Skip if not opening other style edits.

### Items not requiring edits

- App. B (lines 1515–1553) is fully consistent with the new framing. The closing sentence (line 1552–1553) "small compared to the 8--22\% gain from adding the PFS overlap to the DESI-only joint Fisher" is the correct framing and should be retained verbatim.
- Citations: all seven citation sites in the appendices are appropriately individual or paired and need no `[CITE-NEEDED: ...]` markers.
- The 5-check enumeration in App. A.3 is allowed by the user's prose-rule exception for methods/validation lists.
