# §2 Joint multi-tracer multi-survey Fisher analysis — audit report

## Old-framing residues

- **L275–276** "The question is how to tighten it without relying on simulation assumptions." — closes §2.1 by setting up an alternative-to-SBP rhetorical move; under the new framing the next section should pivot to *how multi-tracer over-determines nuisance*, not to "an alternative to simulation priors".
- **L282–284** "The multi-tracer analysis of overlapping surveys provides exactly this---it isolates bias and stochastic parameters from the cross-spectrum between independent galaxy populations." — "overlapping surveys" only references the PFS×DESI overlap channel. The DESI-internal LRG×ELG×QSO multi-tracer channel (which also "isolates bias and stochastic parameters from the cross-spectrum between independent galaxy populations" without any survey overlap) is invisible here.
- **L397** label `\label{sec:prior_export}` — leftover from the prior-export framing; mismatched with the new section title "Joint Fisher across the partitioned footprint".
- **L399–401** "Rather than calibrating priors in the overlap and exporting them to a separate cosmology Fisher, we work with a single joint multi-tracer Fisher across the full DESI footprint." — explicit negation of the old framing. Acceptable as a one-time clarification, but the apologetic *"rather than"* construction is a residue; cleaner to assert the new framing positively.
- **L445** "Cross-calibration trades this model dependence for a volume limitation" — "Cross-calibration" is old-framing vocabulary. The new framing wants "the joint multi-tracer Fisher" or "data-driven multi-tracer priors".
- **L446–447** "the overlap constrains \(b_1\), \(\Pshot\), and \(b_{\G_2}\) model-independently" — collapses both channels into "the overlap"; the internal-DESI channel constrains the same parameters in the non-overlap volume too.

## Two-channel narrative consistency

The (a)→(b)→(c) chain is **never explicitly introduced** in §2. §2.2 motivates the multi-tracer cross-spectrum mechanism in the abstract (any X×Y), and §2.4 introduces the *volume partition* (\(F^{\rm overlap} + F^{\rm nonoverlap}\)) — but the volume partition is not the channel structure. The two-channel structure the user wants is:

- channel (b) = DESI internal LRG×ELG×QSO multi-tracer, *available in both V_overlap and V_nonoverlap*;
- channel (c) = additional PFS×DESI cross-spectra, *available only in V_overlap*.

§2.4 (L407–410) lists tracers per region but does not name the two channels or quantify their incremental gains (33/80/49% then 9/24/9%). A reader leaving §2 has no mental model of the two-channel decomposition.

§2.2 conflates the two channels under a generic "overlapping surveys" header (L282), even though the DESI-internal multi-tracer channel does not require survey overlap. This is the central narrative gap.

## SBP positioning (§2.5)

**Mostly correct, with residue.** L449 explicitly says "The two are complementary"; L450–451 frames SBPs as "mutual consistency checks". This is the new-framing language and is appropriate.

However:
- L444–445 still uses "Cross-calibration trades this model dependence for a volume limitation" — old-framing phrasing.
- L446–448 implies a clean parameter split (overlap → \(b_1, \Pshot, b_{\G_2}\); SBPs → counterterms, \(k\)-dependent stochasticity). Under the new joint-Fisher framing, *both channels* contribute to all nuisance parameters at varying strengths — the "split" is a degeneracy-ranking statement, not a partition. Phrasing slightly drifts back toward parallel-route language.
- L452–456 ("fair comparison", \(\sim\!23\%\) vs \(\sim\!50\%\)) reads as competitive benchmarking, not complementarity. Acceptable but should be reframed as "where SBPs provide an independent cross-check at comparable precision".

## Citation audit

| Line | Cite | Claim under NEW framing | Verdict |
|------|------|---|---|
| 265 | `Chudaykin:2020aoj` | Analytic marginalization of linear/quadratic nuisance | Supports |
| 274 | `DESI:2025wzd` | 23% \(\sigma(\se)\) tightening from power-spectrum SBP | Supports |
| 274 | `Chudaykin:2026nls` | 50% from field-level SBP | Supports |
| 286 | `Seljak:2008xr` | Multi-tracer cosmic-variance cancellation | Supports |
| 286 | `McDonald:2008sh` | Cross-spectrum bias-ratio constraints | Supports |
| 286 | `Mergulhao:2021kip` | Multi-tracer EFT nuisance constraints | Supports |
| 286 | `Mergulhao:2023zso` | Multi-tracer EFT — follow-up | Supports; stacked with previous, see Tone |
| 286 | `Rubira:2025scu` | Multi-tracer EFT priors | Supports |
| 316 | `Ebina:2024ojt` | Cross-stochastic Lagrangian parameterization | Supports (newly added; central to §2.2) |
| 318 | `Mergulhao:2021kip` (repeat) | Halo-bias/occupation residual cross-stochasticity | Supports |
| 318 | `Mergulhao:2023zso` (repeat) | Same | Supports; second repeat — see Tone |
| 333 | `Ebina:2024ojt` (repeat) | Conservative \(P^{XY}_{\rm shot}=0, a^{XY}_2=0\) prescription | Supports |
| 361 | `Zhao:2024eboss` | \(a_0,a_2\) degeneracy with \(c_0,c_2\) | Supports — but verify the eBOSS reference actually demonstrates this counterterm/stochastic degeneracy at one loop. [CITE-NEEDED: confirm `Zhao:2024eboss` covers this exact degeneracy claim] |
| 361 | `Mergulhao:2021kip` (repeat) | \(b_{\G_2}\) constrained via \(\mu\)-dependence | Supports |
| 361 | `Mergulhao:2023zso` (repeat) | Same | Supports |
| 368 | `PFS-CO-target:inprep` | PFS-ELG halo-mass distribution | Supports; in-prep citation, acceptable for the survey-internal claim |
| 373 | `Rubira:2025scu` (repeat) | \(\ctilde\) sets perturbative \(k_{\max}\) | Supports — verify `Rubira:2025scu` actually derives the \(k_{\max}\propto \ctilde^{-1/4}\) scaling. [CITE-NEEDED: confirm reference contains the scaling, not just qualitative validity-range argument] |
| 414 | `Takada:2013bfn` | Disjoint-volume Fourier modes uncorrelated | Supports |
| 414 | `Li:2014sga` | SSC neglected in additive Fisher decomposition | Supports |
| 417 | `Wadekar:2019rdu` | Higher-order covariance terms sub-leading | Supports |
| 417 | `Sugiyama:2019ike` | Same | Supports |
| 441 | `Ivanov:2024hgq` | Power-spectrum SBP from HOD mocks | Supports |
| 441 | `Zhang:2024thl` | Same | Supports |
| 441 | `Chudaykin:2026nls` (repeat) | Field-level SBP | Supports |
| 452 | `DESI:2025wzd` (repeat) | 23% reference number | Supports |
| 455 | `Chudaykin:2026nls` (repeat) | 50% reference number | Supports |

**Old-framing-only citations:** none flagged exclusively for the old framing — all citations remain valid under the joint-Fisher framing. The two `[CITE-NEEDED]` flags above are about *claim strength*, not framing.

## Tone and style

- **L286** five clustered citations (`Seljak:2008xr,McDonald:2008sh,Mergulhao:2021kip,Mergulhao:2023zso,Rubira:2025scu`) violates the anti-citation-stacking rule (max 2 without individual discussion). The five distinct contributions deserve 1–2 sentences of differentiation, especially because Seljak and McDonald are foundational while Mergulhao and Rubira are EFT-specific.
- **L286, L361** — the user's style guide explicitly forbids eponymous mechanism names like "McDonald-Seljak"; the current text avoids the eponym, which is correct, but does not name the mechanism at all. A one-line definition ("cosmic-variance cancellation in cross-spectrum bias ratios") would help.
- **L283** "exactly this" — slightly promotional / colloquial; the arXiv:2604.25171 register would say "this directly".
- **L350** "The key observation is that" — promotional framing word; the target style prefers a declarative opener.
- **L382–393** the §2.3 text is appropriately demoted in *length* (one paragraph) but its tone still leads with a derivation ("\(\kmax^{\PFS}=\kmax^{\DESI}\times r_{\sigma_v}^{-1}\)") rather than with the operational convention. The user's brief asks for "briefly state the per-pair convention" — currently the convention is buried at L382–386 *after* the derivation.
- **L399** "Rather than calibrating priors..." opens §2.4 by negating the old framing. arXiv:2604.25171 register would simply assert the joint construction.
- **L427** "The volume partition is bookkeeping for the natural joint analysis" — "natural" is mildly promotional; "the resulting joint analysis" is more declarative.
- **L443** "achieve tighter constraints than diagonal Gaussian priors" — quantify ("23–50% on \(\sigma(\se)\)" already cited at L272–274; reuse the number here for consistency).
- **L449** em-dash usage in "complementary---and because they constrain..." is on-style.
- Minor: "MCMC" defined L263 but never used again in §2; the parenthetical can be deferred to §3 if MCMC is discussed there.

## Recommended edits (ranked)

### 1. Insert explicit two-channel framing at the §2.2 opening
- **Location:** L282–284
- **Current text:**
  > The multi-tracer analysis of overlapping surveys provides exactly this---it isolates bias and stochastic parameters from the cross-spectrum between independent galaxy populations.
- **Proposed change:**
  > Two channels relax the nuisance prior in the present analysis. The internal-DESI channel uses LRG\(\,\times\,\)ELG\(\,\times\,\)QSO cross-spectra over the full DESI footprint to isolate bias and stochastic parameters from cross-spectra between independent populations. The PFS\(\,\times\,\)DESI overlap channel adds PFS-ELG cross-spectra in the \(1{,}200\;\text{deg}^2\) footprint, contributing tracers with the lowest \(b_1\) in the joint analysis.
- **Justification:** The (a)→(b)→(c) chain currently has no anchor; this paragraph is the right place to name the two channels before §2.2 develops the cross-spectrum mathematics.

### 2. Replace label and opener of §2.4
- **Location:** L397, L399–401
- **Current text:**
  > \label{sec:prior_export}
  > Rather than calibrating priors in the overlap and exporting them to a separate cosmology Fisher, we work with a single joint multi-tracer Fisher across the full DESI footprint.
- **Proposed change:**
  > \label{sec:joint_fisher}
  > The joint multi-tracer Fisher operates across the full DESI footprint with the volume partitioned by tracer availability.
- **Justification:** Removes the negation of the old framing and the prior-export label residue; opens §2.4 with a declarative statement of the construction.

### 3. Reorder §2.3 to lead with the convention
- **Location:** L367–393
- **Current text:** §2.3 leads with the halo-mass / virial-velocity derivation and reaches the per-pair masking convention only at L382–386.
- **Proposed change:** Move the convention statement to the section opener: "Auto- and cross-spectra are masked per-pair: an observable contributes at \(k\)-bin \(k_i\) only when every tracer entering it satisfies \(k_i < k_{\max}^{\rm tracer}\). PFS-ELGs reside in lower-mass halos than DESI-ELGs/LRGs..." then derive \(k_{\max}^{\rm PFS}\) afterward.
- **Justification:** The user's brief states §2.3 should "briefly state the per-pair convention, not lead with it as a major methodological pillar". Currently the section reads as a derivation-first methodological pillar.

### 4. Remove old-framing vocabulary from §2.5
- **Location:** L444–448
- **Current text:**
  > Cross-calibration trades this model dependence for a volume limitation: the overlap constrains \(b_1\), \(\Pshot\), and \(b_{\G_2}\) model-independently, while SBPs tighten the counterterms and \(k\)-dependent stochasticity that the overlap volume cannot reach.
- **Proposed change:**
  > The joint multi-tracer Fisher trades model dependence for a volume limitation: the data-driven multi-tracer priors constrain \(b_1\), \(\Pshot\), and \(b_{\G_2}\) without HOD assumptions, while SBPs additionally constrain the counterterms and \(k\)-dependent stochasticity that the cross-spectrum geometry constrains less effectively.
- **Justification:** Replaces "Cross-calibration" / "the overlap" with new-framing nouns and softens the parameter-partition implication into a degeneracy-ranking statement.

### 5. Break the L286 five-citation cluster
- **Location:** L285–286
- **Current text:**
  > The approach exploits the different stochastic properties of distinct galaxy populations~\citep[e.g.,][]{Seljak:2008xr,McDonald:2008sh,Mergulhao:2021kip,Mergulhao:2023zso,Rubira:2025scu}.
- **Proposed change:**
  > The approach exploits the different stochastic properties of distinct galaxy populations: cosmic-variance cancellation in cross-spectrum bias ratios was developed in Refs.~\cite{Seljak:2008xr,McDonald:2008sh}, extended to the EFTofLSS multi-tracer setting in Refs.~\cite{Mergulhao:2021kip,Mergulhao:2023zso}, and applied to the joint constraint of bias and stochastic parameters in Ref.~\cite{Rubira:2025scu}.
- **Justification:** Anti-citation-stacking; also names the mechanism descriptively without using the forbidden eponym.

### 6. Drop promotional openers
- **Location:** L350, L427
- **Current text:** "The key observation is that..." (L350); "The volume partition is bookkeeping for the natural joint analysis" (L427).
- **Proposed change:** "Jointly fitting the auto-spectra ..." (L350); "The volume partition reflects tracer availability" (L427).
- **Justification:** arXiv:2604.25171 register is declarative; "key observation" and "natural" are promotional.

### 7. Quantify SBP tightening at first reference in §2.5
- **Location:** L443
- **Current text:**
  > SBPs~\cite{Ivanov:2024hgq,Zhang:2024thl,Chudaykin:2026nls} encode the full joint distribution of EFT parameters from HOD mocks and achieve tighter constraints than diagonal Gaussian priors,
- **Proposed change:**
  > SBPs~\cite{Ivanov:2024hgq,Zhang:2024thl,Chudaykin:2026nls} encode the full joint distribution of EFT parameters from HOD mocks, tightening \(\sigma(\se)\) by 23--50\% relative to broad Gaussian nuisance priors,
- **Justification:** Reuses the number cited at L272–274 for internal consistency and replaces a vague "tighter constraints" with the quantitative claim.

### 8. Soften §2.5 closing competitive framing
- **Location:** L452–456
- **Current text:**
  > The power-spectrum-level SBP of Zhang et al.~\cite{DESI:2025wzd} (\(\sim\!23\%\) on \(\sigma(\se)\)) is the fair comparison for our work; the field-level SBP of Chudaykin et al.~\cite{Chudaykin:2026nls} (\(\sim\!50\%\)) breaks degeneracies inaccessible to two-point statistics.
- **Proposed change:**
  > The power-spectrum-level SBP of Zhang et al.~\cite{DESI:2025wzd} (\(\sim\!23\%\) on \(\sigma(\se)\)) addresses the same observable as the present analysis and provides the closest cross-check; the field-level SBP of Chudaykin et al.~\cite{Chudaykin:2026nls} (\(\sim\!50\%\)) accesses information beyond two-point statistics and is complementary to both.
- **Justification:** Replaces "fair comparison" (competitive register) with complementarity language consistent with the new framing.
