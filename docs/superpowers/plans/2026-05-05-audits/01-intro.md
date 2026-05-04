# §1 Introduction — audit report

## Old-framing residues

- **L200–201** (closing the SBP paragraph): "Whether this shift reflects new physics or systematic biases absorbed from the HOD modeling remains an open question---one that an independent, data-driven calibration of the same parameters could help resolve." Reads as a setup for an SBP-replacement / prior-export move ("a data-driven calibration of the same parameters") rather than for the two-channel multi-tracer Fisher that follows. The pivot from SBPs to multi-tracer self-calibration is left implicit.

- **L205–207**: "we demonstrate that EFT nuisance parameters can be calibrated directly from observational data, without any HOD modeling input, by combining two complementary multi-tracer channels into a single joint Fisher analysis." The phrase "EFT nuisance parameters can be calibrated directly from observational data" still leads with the *prior-calibration* mechanism (the old framing's verb) rather than with degeneracy breaking inside a joint Fisher; calibration is then extracted as a *consequence* in the abstract (b1 prior tightened to σ≈0.13). The intro should mirror that ordering: degeneracy-breaking estimator first, b1 calibration as a derived diagnostic.

- **L221–226**: The volume-partitioned Fisher \(F = F_{\rm overlap} + F_{\rm nonoverlap}\) is presented correctly, but the sentence opens "We treat the two channels in a single volume-partitioned Fisher" without first stating that the joint Fisher *is* the estimator. Combined with the L205 framing this still reads as "calibrate then combine" rather than "joint multi-tracer Fisher with two channels."

- No occurrences of "export priors", "Step 1 / Step 2", "transferred", or "alternative to SBPs" in §1 itself; the explicit prior-export language has been removed. §2.1 (L399) explicitly disclaims the old framing, confirming that any residue in §1 is stylistic/ordering rather than substantive.

## Two-channel narrative consistency

The (a)→(b)→(c) chain is **partially present but mis-weighted**:

- **(a) single-tracer broad-prior baseline**: established L171–183 via the σ(fσ8) ×2.4 / σ(Mν) ×6.4 degradation numbers. Clean.

- **(b) DESI internal LRG×ELG×QSO multi-tracer self-calibration**: appears only at L208–211 as "The first channel is internal to DESI ... LRG×ELG×QSO multi-tracer cross-correlations break parameter degeneracies." It is **not previewed earlier** as the dominant mechanism, and the **33/80/49% gain numbers from the abstract are not stated in §1 at all**. The reader cannot tell from the introduction that channel (b) is the dominant gain.

- **(c) DESI+PFS overlap channel**: L211–215, parallel structure to (b), no quantitative attribution (+9/24/9% missing). Treated as the *symmetric* second channel rather than as the case-study extension.

- **L203** (the Zhao et al. paragraph) is the natural setup for (b) but pivots away from it: "This demonstrates the statistical value of multi-tracer information, but not the calibration of survey-independent EFT priors." The pivot reframes (b) as insufficient because it is single-survey, redirecting toward (c). This inverts the actual hierarchy in which (b) dominates and (c) augments.

- The §1 closing summary (L228–234) discusses only the PFS overlap (channel c) and the FoG asymmetric \kmax cut. Channel (b) gets no closing emphasis.

**Net**: §1 lays out two channels but presents them as co-equal/symmetric, and the surrounding rhetoric (L203 pivot, L228 closing) tilts toward (c) as the central novelty. This contradicts the abstract's quantitative weighting.

## SBP positioning

Under the new framing SBPs are a complementary HOD-dependent route; the abstract (L137–141) handles this correctly ("targets the same b1 calibration as SBPs but draws it from cross-spectra ... rather than HOD-modeled mocks; it thus provides a model-independent cross-check").

In §1:

- **L185–201** treats SBPs as the foil whose limitations motivate the paper. The closing line (L198–201) frames the work's contribution as resolving the SBP \se shift via "an independent, data-driven calibration of the same parameters." This is the old SBP-as-competitor positioning: SBPs are introduced as the only existing route, their HOD dependence is flagged, and the paper is implicitly offered as the alternative. The "complementary cross-check" framing of the abstract is missing from §1.

- **L205–207** "without any HOD modeling input" is anti-SBP rhetoric; under the new complementary-route framing it should be softened to a positive statement of the multi-tracer mechanism, with HOD-independence noted as a property rather than as the contrast axis.

- The Zhao et al. paragraph (L203) does not mention SBPs, but its placement immediately after the SBP paragraph and before "In this paper we demonstrate..." reinforces an SBP→multi-tracer-as-alternative reading.

## Citation audit

- **L164** `Baumann:2010tm,Carrasco:2012cv,Carroll:2013oxa,Senatore:2014eva` — EFTofLSS foundational refs. Supports the "built on EFTofLSS" claim. OK.
- **L165** `DAmico:2019fhj,Ivanov:2019pdj` — first BOSS full-shape EFT applications. OK.
- **L165** `Colas:2019ret,Ivanov:2019hqk,Chudaykin:2025lww` — neutrino-mass full-shape constraints. OK.
- **L166** `Cabass:2022wjy,DAmico:2022gki,Chudaykin:2025vdh` — PNG full-shape. OK. Note: 3 stacked, at the boundary of the anti-citation-stacking rule but acceptable as a topic enumeration.
- **L166** `Ivanov:2020ril,DAmico:2020ods,Smith:2020rxx` — early dark energy. OK.
- **L166** `DAmico:2020tty` — clustering DE. OK.
- **L166** `Tsedrik:2025cwc` — interacting DE. OK.
- **L166** `Rogers:2023ezo` — ULA. OK.
- **L167** `He:2023oke,He:2025npy` — dark-sector interactions. OK.
- **L168** `DESI:2024jxi,DESI:2024hhd,DESI:2025wzd` — DESI DR1 full-shape. OK.
- **L169** `Chudaykin:2025aux,Chudaykin:2025lww,Chudaykin:2025vdh,Ivanov:2026dvl,Chudaykin:2026nls` — independent reanalyses of DR1. **Stacked 5 citations without individual discussion** — violates the anti-citation-stacking rule. Flag for citation-rigor phase.
- **L174** `Chudaykin:2020aoj,Desjacques:2016bnm` — EFT one-loop nuisance counting. OK.
- **L179** `Chudaykin:2024wlw` — source of the ×2.4 and ×6.4 degradation factors for a single DESI-ELG. OK, quantitative attribution is correct.
- **L181** `\cref{sec:conclusions}` — internal forward ref for the ×4.6 / ×10 multi-sample number. OK; verify §6 actually contains those numbers.
- **L190** `Ivanov:2024hgq,Ivanov:2024xgb,Zhang:2024thl` — SBP literature (PS-level + FL). OK.
- **L191–193** `DESI:2025wzd` (Zhang et al.) and `Chudaykin:2026nls` — quantitative attribution of 23% and 50% SBP gains. OK.
- **L195** `Akitsu:2024lyt` — supports "SBP only as accurate as the HOD model." OK.
- **L197** `Chudaykin:2026nls` — supports the ~2σ \se downward shift. OK; same reference cited twice in the paragraph (acceptable).
- **L203** `Alarcon:2016bkr,Mergulhao:2021kip,Mergulhao:2023zso,Rubira:2025scu,Qin:2025nkk` — multi-tracer Fisher forecast literature. **Stacked 5 citations** — violates anti-citation-stacking. Under the **new framing** this paragraph is the natural place to establish that channel (b) — DESI internal multi-tracer — is the dominant lever, and these refs should be promoted to individual discussion (e.g., which one establishes degeneracy breaking in 3+ tracers, which one quantifies the gain, which one motivates cross-stochasticity treatment). Under the old framing they were a generic "multi-tracer is known to help" citation block; that role is now load-bearing and unmotivated.
- **L203** `Zhao:2024eboss` — eBOSS LRG×ELG, 18–27% σ(\se) gain. OK; quantitative attribution is correct. **However**, the surrounding sentence uses Zhao et al. to *deflate* same-survey multi-tracer ("but not the calibration of survey-independent EFT priors"); under the new framing, where channel (b) is itself a same-survey multi-tracer analysis at DESI scale, this deflation undermines the paper's own dominant channel. Citation is fine; framing is wrong.
- **L211** `McDonald:2008sh` — McDonald & Seljak multi-tracer. OK as a foundational ref. Note: paper-style guidance forbids eponymous mechanism names ("McDonald-Seljak"); the bare \cite{} is fine, but verify no inline "McDonald-Seljak effect" phrasing later.
- **L213** `Takada:2014pfs` — PFS Cosmology Program reference. OK.
- **L220** `Ebina:2024ojt` — Ebina & White cross-stochasticity treatment. OK; matches the abstract's positioning.
- **L232** `Rubira:2025scu` — supports asymmetric \kmax^PFS > \kmax^DESI. OK.
- **L233** `BaleatoLizancos:2025wdg` — data-driven FoG cleaning footnote. OK.

## Tone and style

- **L161–164** opening sentence is acceptable but contains a tautology: "have now become the new standard in analyses of galaxy clustering in spectroscopic surveys." Two "analyses ... analyses" within one clause; trim to "the standard in spectroscopic-survey clustering analyses."
- **L168–169** "More recently, ..." is a long sentence (L168–169 spans roughly 5 lines) chaining DESI DR1 results, the dynamical DE evidence, the neutrino hierarchy preference, and the prior-sensitivity argument. Split into two: one declarative on the DR1 results, one on σ(Mν) prior-sensitivity.
- **L174** correct use of em-dash for definitional aside ("---bias coefficients, counterterms, and stochastic amplitudes"). Matches house style.
- **L181–183** "(\cref{sec:conclusions}), illustrating the increasing cost ..." — the parenthetical ×4.6 / ×10 numbers should be stated inline, not deferred to the conclusions, since they are load-bearing for the motivation. Quantitative attribution is the house style; deferring is a stylistic miss.
- **L192–194** "improving \(\sigma(\se)\) by \(\sim\!23\%\); ... achieved \(\sim\!50\%\) with field-level priors." Good quantitative attribution.
- **L194–195** "These gains are substantial, but ..." — declarative, OK.
- **L198–201** "Notably, ... deepening the tension ... open question---one that an independent, data-driven calibration of the same parameters could help resolve." The "open question---one that ... could help resolve" is promotional/teleological; the body-paper standard is to state the mechanism declaratively and let the reader infer the implication.
- **L203** is a 6-line single sentence that bundles (i) a literature cite block, (ii) Zhao et al. quantitative result, and (iii) a same-survey-systematics caveat. Split into three sentences in line with house cadence.
- **L205** "In this paper we demonstrate that ..." — acceptable, though "demonstrate" is forecasting work; "show via Fisher forecasts" is more honest. The body paper distinguishes forecast claims from data claims; check that this distinction is preserved.
- **L211** "in the \(\sim\!1{,}200\;\text{deg}^2\) overlap with the Subaru Prime Focus Spectrograph (PFS) Cosmology Program" — PFS expansion is correct here, but the title and abstract use "PFS" without expansion; ensure first-use rule is met (it is, here).
- **L228–234** the closing paragraph emphasizes the PFS overlap and the FoG asymmetric \kmax. Under the new framing where channel (b) is dominant, the closing emphasis is mis-allocated.

## Recommended edits (ranked)

### Framing

- **Location L205–211** — Reorder to put the joint Fisher *estimator* before the *calibration* outcome, and to weight the two channels by their actual gain.
  - Current: "In this paper we demonstrate that EFT nuisance parameters can be calibrated directly from observational data, without any HOD modeling input, by combining two complementary multi-tracer channels into a single joint Fisher analysis. The first channel is internal to DESI: across its full \(14{,}000\;\text{deg}^2\) footprint, the LRG\,$\times$\,ELG\,$\times$\,QSO multi-tracer cross-correlations break parameter degeneracies~\cite{McDonald:2008sh}."
  - Proposed: "In this paper we forecast a joint multi-tracer Fisher that combines two channels of data-driven multi-tracer priors. The internal-DESI channel---LRG\,$\times$\,ELG\,$\times$\,QSO across the full \(14{,}000\;\text{deg}^2\) footprint---provides the dominant gain, breaking the \(b_1\sigma_8\)--\(f\se\) degeneracy intrinsic to single-tracer analyses~\cite{McDonald:2008sh} and tightening \(\sigma(f\se,\Mnu,\Omega_m)\) by \(33/80/49\%\) over the single-tracer broad-prior baseline."
  - Justification: Promotes the joint Fisher to the lead, attributes the dominant 33/80/49% gain to channel (b) inline, and demotes "calibrated directly from observational data" from claim to property.

- **Location L211–215** — Reframe channel (c) as augmentation rather than parallel.
  - Current: "The second is cross-survey: in the \(\sim\!1{,}200\;\text{deg}^2\) overlap with the Subaru Prime Focus Spectrograph (PFS) Cosmology Program~\cite{Takada:2014pfs}, an additional ELG sample with distinct selection and shot noise contributes auto- and cross-spectra that the DESI-internal analysis cannot access."
  - Proposed: "The PFS\,$\times$\,DESI overlap channel adds a fourth tracer over \(\sim\!1{,}200\;\text{deg}^2\)~\cite{Takada:2014pfs}: a PFS-ELG sample with selection and shot noise distinct from DESI-ELG contributes auto- and cross-spectra unavailable to the internal-DESI analysis, tightening \(\sigma(f\se,\Mnu,\Omega_m)\) by a further \(9/24/9\%\)."
  - Justification: Establishes channel (c) as additive on top of (b), with quantitative attribution matching the abstract.

- **Location L198–201** — Replace the SBP-as-competitor closing with the complementary-route positioning from the abstract.
  - Current: "Whether this shift reflects new physics or systematic biases absorbed from the HOD modeling remains an open question---one that an independent, data-driven calibration of the same parameters could help resolve."
  - Proposed: "Whether this shift reflects new physics or HOD-modeling systematics remains open. A data-driven calibration of the same \(b_1\) prior, drawn from cross-spectra between observed galaxy populations rather than HOD-modeled mocks, provides a model-independent cross-check on the SBP-driven \(\se\) shift."
  - Justification: Removes "could help resolve" promotional hedge; restates SBPs as complementary, matching abstract L137–141.

- **Location L203** — Split the Zhao paragraph and remove the "but not the calibration of survey-independent EFT priors" deflation, which undercuts channel (b).
  - Current: "Meanwhile, multi-tracer Fisher forecasts have demonstrated parameter degeneracy breaking beyond what single-tracer analyses can achieve~\cite{Alarcon:2016bkr,Mergulhao:2021kip,Mergulhao:2023zso,Rubira:2025scu,Qin:2025nkk}. In particular, Zhao et al.~\cite{Zhao:2024eboss} demonstrated this with eBOSS DR16 data by jointly analyzing LRGs and ELGs within a single survey, improving \(\sigma(\se)\) by 18--27\% depending on the treatment of stochastic terms in the LRG--ELG cross spectrum. This demonstrates the statistical value of multi-tracer information, but not the calibration of survey-independent EFT priors: same-survey analyses share observational systematics, and even for disjoint tracer catalogs with no shared-object shot noise, residual cross-stochasticity remains a modeling assumption."
  - Proposed: "Multi-tracer Fisher forecasts have established parameter-degeneracy breaking beyond the single-tracer reach [CITE-NEEDED: individual attribution for Alarcon:2016bkr, Mergulhao:2021kip, Mergulhao:2023zso, Rubira:2025scu, Qin:2025nkk]. Zhao et al.~\cite{Zhao:2024eboss} realized this with eBOSS DR16, jointly analyzing LRGs and ELGs and improving \(\sigma(\se)\) by 18--27\%, with the spread set by the treatment of LRG--ELG cross-stochasticity. The cross-stochasticity treatment is the dominant modeling choice: even for disjoint tracer catalogs with no shared-object shot noise, residual cross-stochasticity from halo-bias and occupation correlations is a free function~\cite{Ebina:2024ojt}."
  - Justification: Splits the 6-line sentence; removes the "not the calibration of survey-independent EFT priors" line that undercuts channel (b); promotes cross-stochasticity to the load-bearing modeling choice (matches §2.2 / §4.3); flags the 5-cite stack for individual treatment.

- **Location L228–234** — Rebalance the closing paragraph so channel (b) gets parity with the PFS-specific scale-cut discussion.
  - Current: "The PFS overlap covers \(0.6<z<1.6\) with up to 4 tracers (PFS-ELG plus DESI-LRG, DESI-ELG, DESI-QSO) and ten auto- and cross-spectra per redshift bin. The weaker finger-of-god (FoG) contamination expected in PFS samples motivates an asymmetric scale cut, \(\kmax^{\PFS}>\kmax^{\DESI}\)~\cite{Rubira:2025scu}\footnote{Data-driven FoG cleaning methods~\citep[e.g.,][]{BaleatoLizancos:2025wdg} offer a complementary route to extending the usable \(k\)-range.}."
  - Proposed: "The internal-DESI channel covers \(0.4<z<2.1\) with three tracers (LRG, ELG, QSO) and the full DESI footprint. The PFS\,$\times$\,DESI overlap channel covers \(0.6<z<1.6\) with up to four tracers (PFS-ELG plus DESI-LRG, DESI-ELG, DESI-QSO) and ten auto- and cross-spectra per redshift bin. Weaker FoG contamination in PFS samples motivates an asymmetric scale cut, \(\kmax^{\PFS}>\kmax^{\DESI}\)~\cite{Rubira:2025scu}\footnote{Data-driven FoG cleaning methods~\citep[e.g.,][]{BaleatoLizancos:2025wdg} offer a complementary route to extending the usable \(k\)-range.}."
  - Justification: Surfaces channel (b) coverage and tracer count alongside channel (c); the asymmetric \kmax remains as a PFS-specific detail.

### Tone

- **Location L181–183** — Move the multi-sample degradation numbers inline.
  - Current: "the degradation grows to \(\sim\!4.6\times\) and \(\sim\!10\times\) when 6 DR2 samples are combined (\cref{sec:conclusions}), illustrating the increasing cost of nuisance-parameter marginalization in joint multi-sample (not multi-tracer) analyses."
  - Proposed: "the degradation grows to \(\sim\!4.6\times\) on \(\sigma(f\se)\) and \(\sim\!10\times\) on \(\sigma(\Mnu)\) when 6 DR2 samples are combined, since each sample brings its own 12-parameter nuisance set. Multi-sample combination without multi-tracer information thus pays a compounding nuisance cost."
  - Justification: Quantitative attribution is named (which σ); removes the forward-ref deferral; clarifies the multi-sample vs multi-tracer distinction without parenthetical.

- **Location L168–169** — Split the long DR1 sentence.
  - Current: "More recently, the DESI collaboration~\cite{DESI:2024jxi,DESI:2024hhd,DESI:2025wzd} and independent reanalyses~\cite{Chudaykin:2025aux,Chudaykin:2025lww,Chudaykin:2025vdh,Ivanov:2026dvl,Chudaykin:2026nls} applied this framework to DESI~DR1 data, reporting evidence for dynamical dark energy and---when combined with CMB data---a preference for the normal neutrino mass hierarchy over the inverted hierarchy. Sharpening \(\sigma(\Mnu)\) from galaxy clustering is therefore a pressing goal, as the neutrino mass constraint is currently the most prior-sensitive quantity in EFT full-shape analyses."
  - Proposed: "The DESI collaboration~\cite{DESI:2024jxi,DESI:2024hhd,DESI:2025wzd} and independent reanalyses [CITE-NEEDED: individual attribution for Chudaykin:2025aux, Chudaykin:2025lww, Chudaykin:2025vdh, Ivanov:2026dvl, Chudaykin:2026nls] applied this framework to DESI~DR1, reporting evidence for dynamical dark energy and, in combination with CMB data, a preference for the normal neutrino mass hierarchy. Sharpening \(\sigma(\Mnu)\) from galaxy clustering is therefore a direct goal of next-generation full-shape analyses, since \(\Mnu\) is currently the most prior-sensitive parameter in the EFT pipeline."
  - Justification: Splits the long sentence; converts em-dash inserts into a comma clause; flags the 5-cite stack.

### Word-choice

- **Location L161–164** — Trim the tautology.
  - Current: "Full-shape analyses of the galaxy power spectrum, built on the effective field theory of large-scale structure (EFTofLSS)~\cite{...}, have now become the new standard in analyses of galaxy clustering in spectroscopic surveys."
  - Proposed: "Full-shape analyses of the galaxy power spectrum, built on the effective field theory of large-scale structure (EFTofLSS)~\cite{...}, are now standard in spectroscopic-survey clustering."
  - Justification: Removes "have now become the new standard in analyses of galaxy clustering" → "are now standard in"; eliminates "analyses ... analyses" repetition.

- **Location L205** — Soften "demonstrate" for a forecast.
  - Current: "In this paper we demonstrate that EFT nuisance parameters can be calibrated directly from observational data, ..."
  - Proposed: "In this paper we forecast that ..." (subsumed into the larger framing edit above, but flag separately for the case where the framing edit is not adopted).
  - Justification: Body paper distinguishes forecast claims from data claims; "demonstrate" overclaims for a Fisher analysis.
