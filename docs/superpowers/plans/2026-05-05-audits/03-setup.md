# §3 Fisher forecast setup — audit report

Scope: lines 452-712 of `paper/PFSxDESI-multi-fish.tex`. Audit only; no .tex edits.

## Old-framing residues

- **L464-468 (intro paragraph).** "we construct multi-tracer and single-tracer Fisher matrices" implies two parallel Fishers; under the new framing the single-tracer broad case is a baseline reference, not a Fisher we "construct" alongside the joint analysis. The phrase "EFT parameterization, the covariance and derivative computation, and the analysis scenarios" reads as the old recipe — there is no mention of the two channels (internal-DESI / PFS×DESI overlap) that the new §3.5 rests on.
- **L667-668 label.** `\label{sec:scenarios}` is fine, but the section title "Analysis scenarios and benchmarks" still suggests a multi-scenario menu. Under the new framing only two configurations are reported (DESI-only joint, DESI+PFS joint), with the single-tracer broad baseline implicit; the heading should not promise more.
- **L685.** `\cref{sec:prior_export}` — the label name is a vestige of the prior-export framing. The target subsection (L396-397) has been re-titled "Joint Fisher across the partitioned footprint" but kept the old label `sec:prior_export`. The cref still resolves, but the name leaks the old framing into §3.6's prose.
- **L691-701 itemize.** Old-framing scenarios ("broad / cross-cal / cross-cal-ext / HOD-prior / oracle") are gone — good. But the listed "two configurations" omit the (a) single-tracer broad baseline that the audit brief identifies as channel (a). Either the baseline belongs in this list or the framing text needs to acknowledge it explicitly as the reference against which (b) and (c) are reported.
- **`eqn:cross_shot`.** Searched the full .tex; no occurrence. The only surviving reference in §3 is `\cref{eqn:cross}` at L636, which resolves to `\label{eqn:cross}` at L306 — that label still exists. **No orphan reference remains.**
- **L702-706.** Horizontal SBP reference lines (Zhang 23%, Chudaykin 50%) are presented as "benchmarks". Under the new framing SBPs are a complementary route, not the calibration target — see SBP positioning below.

## Two-channel narrative consistency

§3.5 does not set up the (a)→(b)→(c) chain.

- L670-690 frames the joint Fisher as a single-pass construction with a volume partition (overlap + non-overlap). That is correct under the new framing but does not name the two information channels: the internal-DESI multi-tracer channel (LRG×ELG×QSO bias-ratio + RSD) operates over the full footprint, while the PFS×DESI overlap channel operates only in the 1,200 deg² where PFS adds a fourth tracer plus cross-spectra. The volume partition (`F_z^overlap + F_z^nonoverlap`) is a bookkeeping device; the channel decomposition is the physics story and is missing.
- L692-701 lists "DESI-only joint" and "DESI+PFS joint" but omits the single-tracer broad baseline (channel a). Without (a) the two listed configurations have nothing to be compared against — the 23%/50% SBP lines at L702-706 are the only benchmarks named, which mis-positions SBPs as the comparison anchor.
- The connection from §3.5 forward to §4 (where Figure of Merit is reported relative to single-tracer broad) is not set up here.

## SBP positioning

- **L702-706.** "We also overlay horizontal reference lines at the published improvement ratios from the PS-level SBP of Zhang et al.\ (23\% on σ(fσ8)) and the field-level SBP of Chudaykin et al.\ (50\% on σ(fσ8))." This treats SBPs as the calibration target the joint Fisher is benchmarked against. Under the new framing SBPs are a complementary, methodologically distinct route (simulation-based, not data-driven priors); they belong in §1 or §6 as parallel approaches, not as the headline benchmark for §3 results. The sentence as written invites the reader to read the joint Fisher as "trying to match SBP numbers" — exactly the mis-framing the revision is meant to remove.

## Citation audit

Each `\cite{}` in lines 452-712, one-line check.

| Line | Citation | Used as | Supports new framing? |
|---|---|---|---|
| 452 | DESI:2025wzd (Zhang) | PS-level SBP, σ_v table for LRG/QSO/ELG | Yes — but L452-456 sits at end of §2; positioning is OK. |
| 455 | Chudaykin:2026nls | field-level SBP comparator | Yes — but treated as primary benchmark at L703-706; should be reframed as complementary. |
| 478 | DESI:2016fyo | DESI instrument | Yes. |
| 480 | DESI:2025zgx | DESI ELG sample | Yes. |
| 489 | Takada:2014pfs | PFS instrument | Yes. |
| 493 | Orsi:2010obl | PFS-ELG bias model | Yes. |
| 538 | Planck:2018params | fiducial cosmology | Yes. |
| 546 | Chudaykin:2025aux, Chudaykin:2025lww | EFT parameterization | Yes. |
| 548 | DESI:2024hhd, Chudaykin:2025lww | analytic marginalization rationale | Yes. |
| 574 | Chudaykin:2025lww | EFT parameterization (table) | Yes. |
| 576 | Lazeyras:2015lgp | b2 co-evolution | Yes. |
| 582 | DESI:2025wzd | σ_v table | Yes. |
| 599 | PFS-CO-target:inprep | PFS halo-mass argument | Yes. |
| 604 | Lazeyras:2015lgp | co-evolution relation | Yes. |
| 608 | DESI:2025wzd | σ_v table | Yes. |
| 622 | Ivanov:2019pdj, DAmico:2019fhj | full-shape multipole observable | Yes. |
| 642 | Chudaykin:2020aoj | one-loop EFTofLSS model | Yes. |
| 643 | Simonovic:2017mhp | FFTLog decomposition | Yes. |
| 645 | Kobayashi:2026inprep | ps_1loop_jax library | Yes. |
| 655 | Piras:2023aub | cosmopower-jax | Yes. |
| 658 | Jense:2024llt | νΛCDM emulator networks | Yes. |
| 703 | DESI:2025wzd | SBP benchmark | Mis-positioned (see SBP section); no new \cite needed, just reframe. |
| 705 | Chudaykin:2026nls | SBP benchmark | Same as above. |
| 718 | DESI:2025zgx | imaging-systematics control | Yes. |

No missing citations flagged for §3 under the new framing. No `[CITE-NEEDED:]` insertions required.

## Tone and style

- **L464 "To forecast the constraints from the PFS--DESI overlap, we construct ..."** — opening still reads as overlap-centric. Under the new framing the forecast is over the full DESI footprint with the overlap as one channel; this is the wrong topic sentence.
- **L488 / L477 paragraph headings ("\paragraph{DESI.}", "\paragraph{PFS.}", "\paragraph{Overlap geometry.}")** — these are eponymous-flavored mini-headings. Tone is OK (declarative); flag only if narrative-prose-only rule is enforced.
- **L548 "Including them as explicit Fisher parameters with the same priors is equivalent."** — declarative, good.
- **L590 "Since PFS-ELG mocks with EFT parameter fits do not yet exist, we construct PFS fiducials by scaling..."** — good declarative tone with quantitative attribution.
- **L621 "The observables are power spectrum multipoles..."** — clean.
- **L661 "This provides exact derivatives for all 15 parameters without step-size tuning"** — quantitative attribution, good.
- **L670 "We forecast with a single joint multi-tracer Fisher..."** — declarative; this is the only sentence in §3.5 that announces the new framing. It is buried as the topic sentence of an itemize-followed paragraph and could be load-bearing.
- **L702 "We also overlay horizontal reference lines at the published improvement ratios..."** — see SBP critique; tone is fine, framing is not.
- No em-dash asides in §3.5; the brief asks for declarative + em-dash asides. §3.5 is currently bullet-heavy (L691-701 itemize) which works against the prose-flow goal stated in CLAUDE.md (no bulleted output in body sections).

## Recommended edits (ranked)

### 1. Rewrite §3.5 topic and bullets to name the three configurations

- **Location:** L670-701
- **Current text:**
  ```
  We forecast with a single joint multi-tracer Fisher
  over the full DESI footprint. At each redshift bin, the
  Fisher decomposes as
  \be
  \label{eq:volume_partition}
  F_z = F_z^{\rm overlap}(V_{\rm overlap})
      + F_z^{\rm nonoverlap}(V_{\rm nonoverlap}),
  \ee
  ...
  \begin{itemize}
  \item \textbf{DESI-only joint}: PFS-ELG removed everywhere; ...
  \item \textbf{DESI+PFS joint}: as above, plus the ...
  \end{itemize}
  ```
- **Proposed change:**
  ```
  We report three configurations, each a single Fisher analysis
  over the full \(14{,}000\;\text{deg}^2\) DESI footprint with
  \(\kmax=0.20\iMpch\). The single-tracer broad baseline
  forecasts each DESI tracer independently with the
  conservative broad priors of \cref{tab:eft_params}; this is
  the reference against which the joint analyses are reported.
  The DESI-only joint configuration combines DESI-LRG, DESI-ELG,
  and DESI-QSO into one multi-tracer Fisher across the full
  footprint, exposing the internal-DESI channel through
  bias-ratio and RSD information in the auto- and cross-spectra
  alone. The DESI+PFS joint configuration adds PFS-ELG within
  the \(1{,}200\;\text{deg}^2\) overlap, opening the
  PFS\(\times\)DESI overlap channel through the additional auto-
  and cross-spectra available where the four tracers coexist.
  Both joint configurations decompose at each redshift bin as
  \cref{eq:volume_partition}, \(F_z = F_z^{\rm
  overlap}(V_{\rm overlap}) + F_z^{\rm nonoverlap}(V_{\rm
  nonoverlap})\), with \(V_{\rm overlap}=1{,}200\;\text{deg}^2\)
  and \(V_{\rm nonoverlap}=12{,}800\;\text{deg}^2\) treated as
  independent under the Gaussian-covariance approximation
  (\cref{sec:prior_export} discusses the SSC approximation).
  Per-redshift Fishers are combined across all bins with
  shared cosmological parameters \((f\se,\,\Mnu,\,\Om)\) and
  per-(tracer, \(z\)-bin) nuisance parameters, marginalized
  in one pass.
  ```
- **Justification:** Names the three-channel chain (a→b→c) explicitly with the required body noun phrases ("internal-DESI channel", "PFS×DESI overlap channel", "joint multi-tracer Fisher"), eliminates the bullet list per the prose-only rule, and removes the SBP-as-benchmark framing from §3.5.

### 2. Demote the SBP horizontal-line sentence

- **Location:** L702-706
- **Current text:**
  ```
  We also overlay horizontal reference lines at the published
  improvement ratios from the PS-level SBP of Zhang et al.\
  \cite{DESI:2025wzd} (23\% on \(\sigma(\se)\)) and the field-level
  SBP of Chudaykin et al.\ \cite{Chudaykin:2026nls} (50\% on
  \(\sigma(\se)\)).
  ```
- **Proposed change:**
  ```
  In the results figures we overlay published improvement
  ratios from the PS-level SBP of Zhang et al.\
  \cite{DESI:2025wzd} (23\% on \(\sigma(\se)\)) and the
  field-level SBP of Chudaykin et al.\ \cite{Chudaykin:2026nls}
  (50\% on \(\sigma(\se)\)) as complementary reference points;
  the SBP route is methodologically distinct---simulation-based
  rather than data-driven multi-tracer priors---and we discuss
  the comparison in \cref{sec:discussion}.
  ```
- **Justification:** Reframes SBPs as a complementary route, not the §3 benchmark target; matches the new "complementary" positioning the brief calls for; em-dash aside; declarative.

### 3. Rewrite §3 intro to lead with the joint Fisher and the two channels

- **Location:** L464-468
- **Current text:**
  ```
  To forecast the constraints from the PFS--DESI overlap, we construct
  multi-tracer and single-tracer Fisher matrices using the one-loop
  EFTofLSS. We describe the survey specifications, the EFT
  parameterization, the covariance and derivative computation, and the
  analysis scenarios below.
  ```
- **Proposed change:**
  ```
  We forecast cosmological constraints from a joint multi-tracer
  Fisher over the full \(14{,}000\;\text{deg}^2\) DESI footprint,
  with two information channels combined in a single analysis:
  the internal-DESI channel from LRG, ELG, and QSO multi-tracer
  cross-correlations everywhere, and the PFS\(\times\)DESI
  overlap channel from the additional PFS-ELG auto- and
  cross-spectra in the \(1{,}200\;\text{deg}^2\) overlap. Below
  we specify the surveys, the EFT parameterization, the covariance
  and derivative computation, and the configurations we report.
  ```
- **Justification:** Topic sentence now matches the joint multi-tracer framing and introduces the two-channel decomposition that §3.5 will use; drops the parallel "multi-tracer and single-tracer" construction that misreads the single-tracer baseline as a co-equal forecast.

### 4. Rename `sec:prior_export` label

- **Location:** L397 (target) and L685, plus other `\cref{sec:prior_export}` callers (L1370, L993 area — verify scope outside §3).
- **Current text (L397):** `\label{sec:prior_export}`
- **Proposed change:** `\label{sec:joint_fisher_partition}` (and update all `\cref` sites)
- **Justification:** Label is a vestige of the prior-export framing; renaming aligns the cross-reference text with the new framing without changing user-visible content. Out of §3 scope only mechanically; left for the global pass.

### 5. Mention the data-driven multi-tracer priors framing in §3.4

- **Location:** L618-647 (§3.4 Multi-tracer Fisher matrix)
- **Current text:** Pure mechanics — covariance formula, μ-quadrature, FFTLog citation. No framing sentence connects the construction to the "data-driven multi-tracer priors" phrase the brief mandates.
- **Proposed change:** Add one sentence after L631 (closing of the covariance equation) such as:
  ```
  This covariance, evaluated jointly over auto- and
  cross-spectra of the four tracers, is what makes the
  per-bin nuisance constraints data-driven multi-tracer
  priors rather than externally imposed Gaussian widths.
  ```
- **Justification:** Inserts the canonical body noun phrase once in §3 so downstream sections (§4, §5) can use it without first introduction; declarative, no new citation needed.

### 6. Remove "and benchmarks" from §3.5 title

- **Location:** L667 `\subsection{Analysis scenarios and benchmarks}`
- **Current text:** `\subsection{Analysis scenarios and benchmarks}`
- **Proposed change:** `\subsection{Analysis configurations}`
- **Justification:** "Benchmarks" advertises external comparators (the SBP lines) as part of the §3 deliverable; under the new framing those are §6 material. "Configurations" matches the language in edit #1 and avoids the "scenarios" word that carries the broad/cross-cal/oracle baggage.

---

Notes:
- No orphan equation references in §3. `\cref{eqn:cross}` at L636 resolves.
- No `\cite{}` deletions or additions required for §3 under the new framing.
- §3.1-§3.3 (surveys, EFT params, PFS fiducials) need no narrative changes; they describe inputs that are framing-agnostic.
- §3.4 (multi-tracer Fisher) needs only the one-sentence insertion in edit #5.
- §3.6 (transferability) is fine as-is — it states the assumption that lets internal-DESI calibration extend across the footprint, which is the correct framing.
