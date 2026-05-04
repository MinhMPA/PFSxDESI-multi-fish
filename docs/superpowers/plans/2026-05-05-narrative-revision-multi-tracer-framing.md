# Narrative Revision: from "multi-survey priors" to "multi-tracer framework, multi-survey case study" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and revise `paper/PFSxDESI-multi-fish.tex` so the framing, narrative, and emphasis throughout the manuscript reflect the *current* analysis (joint multi-tracer Fisher with two channels — internal-DESI and PFS×DESI overlap) rather than the *original* analysis (export of calibrated priors from the overlap to the non-overlap, framed as a parallel to SBPs). Preserve the original tone and style of the previous arXiv version (arXiv:2604.25171).

**Architecture:** Five-phase workflow:
1. Phase A — Cold-read audits in parallel (8 subagents) over distinct sections to catalogue residues of the old framing, narrative drift, and any citation that no longer supports the surrounding claim under the new framing.
2. Phase B — Synthesis of the audit findings into a single ranked-edit list.
3. Phase C — Section-by-section edits in dispatched subagents, one per section, each producing a self-contained commit.
4. Phase D — Audit-the-audit cold re-read passes (4 subagents) on the revised manuscript to surface any residual drift.
5. Phase E — Citation-rigor pass with strict no-new-citation default; any candidate new citation passes a 2–3-cycle verification against the bib metadata, arXiv, and Semantic Scholar before landing.
6. Phase F — Final integrated read and consolidated commit message.

**Tech Stack:** LaTeX (JHEP class), BibTeX, pdflatex; git for commits; the `latex-paper-en` skill scripts for build/format/abstract checks; the `cosmo-paper-reviewer` skill (optional) for an independent end-to-end audit.

**Editing constraints:**
- Tone and style mirror arXiv:2604.25171. Do not introduce promotional language ("headline", "main result", "novel", "key finding"). Do not introduce eponymous mechanism names ("McDonald–Seljak mechanism").
- Two-channel narrative ordering: (a) DESI single-tracer broad-prior baseline → (b) DESI internal LRG×ELG×QSO multi-tracer self-calibration (the dominant gain, 33/80/49% on σ(fσ8, Mν, Ωm)) → (c) DESI+PFS joint adding the overlap channel (further +9/24/9%). This ordering must be consistent across abstract, results, discussion, conclusions.
- Title is `Multi-tracers, multi-surveys: a joint Fisher analysis of DESI+PFS`. Do not propose retitling. The title's two-half structure is the intended frame: *multi-tracer is the mechanism; multi-survey is the case-study extension*.
- Cross-stochasticity: brief mention only (one trailing clause) in the abstract; full discussion in §2.2 and §4.3 already established. Do not expand into the abstract.
- The σ(Mν) 80% gain has been validated by four diagnostics (per-z-bin localization, kmax monotonic scan, ρ collapse, data-domination); the §5.1 diagnostic-chain paragraph stays. Audit subagents must NOT propose softening this.
- Repo / bib filename `PFSxDESI-multi-fish` is FIXED. Section labels (e.g. `tab:headline`) are FIXED. Cross-spectrum math notation `P^{PFS×DESI-ELG}` is FIXED.

**Reference policy (strict):**
- Do not add new \cite{} entries unless the new emphasis genuinely *requires* one.
- Any candidate new citation must pass three checks before landing:
  1. The bibtex metadata (title, authors, journal, year, doi/eprint) is verified against the arXiv abstract page or Semantic Scholar API.
  2. The cited claim is verified against the paper's abstract or relevant section (paraphrased) — no hallucinated specifics.
  3. A second pass confirms the cited paper actually says the cited thing (i.e. not just that it is on the same topic).
- Prefer to remove a citation that no longer supports its surrounding claim under the new framing rather than substitute a new one.
- The existing bibliography of 80+ entries is the working set. Subagents have a strong prior against introducing new keys.

---

## Phase A — Cold-read audits in parallel (8 subagents)

Each subagent reads its assigned section of the current manuscript with a fresh context and produces an *audit report* in markdown. Reports go to `docs/superpowers/plans/2026-05-05-audits/<section>.md` so subsequent phases can cross-reference them. Subagents do not edit the .tex; they only diagnose.

**Context every audit subagent receives:**
- The full goal, the editing constraints, and the reference policy from the plan header.
- The two-channel narrative ordering (a) → (b) → (c).
- The previous arXiv version (`arXiv:2604.25171`) used the *single-stage prior export* framing under the umbrella term "multi-survey priors". The current analysis uses a *joint* Fisher across both channels. Audit reports must distinguish "old framing residues that need editing" from "passages that already reflect the new framing".
- The audit format below.

### Task A1: Audit §1 Introduction

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 154-238 (introduction)
- Create: `docs/superpowers/plans/2026-05-05-audits/01-intro.md`

- [ ] **Step 1: Dispatch audit subagent**

Use the `general-purpose` agent (no editing tools needed for read-only audits). Prompt template:

```
You are auditing the Introduction (§1) of a cosmology paper draft for a
narrative revision. The paper changed from a *prior-export* framing
(multi-survey calibration in the overlap, then export to the
non-overlap) to a *joint multi-tracer Fisher* framing (two channels
combined in one analysis: DESI internal multi-tracer + PFS×DESI
overlap).

Read paper/PFSxDESI-multi-fish.tex lines 154-238 and produce a
markdown audit report at
docs/superpowers/plans/2026-05-05-audits/01-intro.md with these
sections:

  ## Old-framing residues
  Quote each line/paragraph that still uses prior-export language
  ("export priors", "calibrated priors transferred", "Step 1
  calibration → Step 2 cosmology", "multi-survey priors as
  alternative to SBPs", etc.) and note line numbers.

  ## Two-channel narrative consistency
  Check whether the introduction lays out the (a)→(b)→(c) chain:
  single-tracer broad → DESI multi-tracer → DESI+PFS. Note any place
  where it skips the (b) step or treats (c) as the headline mechanism.

  ## SBP positioning
  Under the old framing, SBPs were the parallel/competitor. Under the
  new framing they are a complementary HOD-dependent route. Flag any
  passage that still treats SBP positioning as the central novelty.

  ## Citation audit
  For every \cite{} in this section, write a one-line confirmation
  that the citation supports the surrounding claim under the NEW
  framing. Flag any \cite{} that was load-bearing under the old
  framing but is now unmotivated.

  ## Tone and style
  Compare to the abstract of arXiv:2604.25171 (which uses direct,
  declarative sentences with em-dash asides and quantitative
  attribution). Flag deviations.

  ## Recommended edits (ranked)
  Bullet list of suggested edits, ranked by impact: framing > tone >
  word-choice. Each bullet: location (line range), current text
  (quote), proposed change, justification.

Do not edit the .tex. Do not propose new citations unless absolutely
necessary; if you do, mark them [NEEDS-VERIFICATION] for the citation
rigor phase.
```

- [ ] **Step 2: Wait for report. Verify file exists and is non-trivial.**

```bash
ls -la docs/superpowers/plans/2026-05-05-audits/01-intro.md
wc -l docs/superpowers/plans/2026-05-05-audits/01-intro.md
```

Expected: file exists, >50 lines.

### Task A2: Audit §2 Joint multi-tracer methodology

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 241-450 (§2 sections)
- Create: `docs/superpowers/plans/2026-05-05-audits/02-methodology.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same prompt template as A1, with section name "§2 Joint multi-tracer methodology" and line range "241-450". Special focus areas: §2.2 cross-power constraints (rewritten this session for cross-stoch); §2.3 asymmetric scale cuts (demoted this session); §2.4 joint Fisher across the partitioned footprint (was the lead under the old framing, audit current emphasis); §2.5 comparison with SBPs (was the *parallel-to-SBP* anchor under old framing, audit how it now positions SBPs as complementary).

- [ ] **Step 2: Verify report.**

### Task A3: Audit §3 Fisher forecast setup

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 452-712 (§3)
- Create: `docs/superpowers/plans/2026-05-05-audits/03-setup.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same template as A1. Particular focus on §3.5 Analysis scenarios and benchmarks: the old framing used "broad / cross-cal / oracle" scenarios from the two-stage pipeline; the new framing uses "DESI single-tracer broad / DESI-only joint / DESI+PFS joint". Audit any vestigial scenario language. §3.6 Multi-tracer Fisher matrix: reference to `\cref{eqn:cross_shot}` was removed; verify no orphans remain.

- [ ] **Step 2: Verify report.**

### Task A4: Audit §4 Results

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 714-1010 (§4)
- Create: `docs/superpowers/plans/2026-05-05-audits/04-results.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same template. Focus: §4.1 per-redshift nuisance (uses calibrated_vs_broad fig, ensure phrasing reflects joint-Fisher *projection* not *exported priors*); §4.2 full-area constraints + Tab. headline + Fig. full_area; §4.3 robustness (rsv + cross-stoch). Audit "the cosmology constraints in [Tab. headline]" wording — the noun phrase "cosmology constraints" should be load-bearing now, not "headline".

- [ ] **Step 2: Verify report.**

### Task A5: Audit §5 Discussion

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 1012-1183 (§5)
- Create: `docs/superpowers/plans/2026-05-05-audits/05-discussion.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same template. Focus: §5.1 Which parameters drive the improvement (contains the σ(Mν) diagnostic chain — DO NOT propose softening; audit only that the chain is internally consistent and supports the (a)→(b)→(c) hierarchy); §5.2 robustness; §5.3 complementarity with SBPs and Mν implications. Critical: §5.3 was rewritten in this session to use "data-driven multi-tracer priors and SBPs" rather than "multi-survey priors and SBPs"; verify the language is consistent.

- [ ] **Step 2: Verify report.**

### Task A6: Audit §6 Conclusions

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 1185-1230 (§6, 5 enumerated items)
- Create: `docs/superpowers/plans/2026-05-05-audits/06-conclusions.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same template. Focus on the 5-item enumeration: are the items in the right priority order (volume-partition framework, multi-tracer dominance, PFS-on-top, SBP complementarity, generalization)? Does any item still describe the old prior-export pipeline? Are σ values consistent with the rest of the paper (DESI-only 0.014/0.14/0.016; DESI+PFS 0.013/0.108/0.014)?

- [ ] **Step 2: Verify report.**

### Task A7: Audit appendices

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 1240-end (appendices)
- Create: `docs/superpowers/plans/2026-05-05-audits/07-appendices.md`

- [ ] **Step 1: Dispatch audit subagent (general-purpose)**

Same template. Focus on App. fisher_checks (validation appendix) and the EFT-priors-from-co-evolution appendix. The Fig. fisher_info caption was just edited from "multi-survey priors" to "data-driven multi-tracer priors" — verify the rest of the appendix is consistent.

- [ ] **Step 2: Verify report.**

### Task A8: Citation map

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` (all `\cite{}` invocations) and `paper/PFSxDESI-multi-fish.bib`
- Create: `docs/superpowers/plans/2026-05-05-audits/08-citations.md`

- [ ] **Step 1: Dispatch citation-audit subagent (general-purpose)**

Prompt:
```
Audit every \cite{} in paper/PFSxDESI-multi-fish.tex. Produce a
markdown table at docs/superpowers/plans/2026-05-05-audits/08-citations.md
with columns:

  | bib_key | line | surrounding_claim (≤20 words) | supports_claim (Y/N/Maybe) | recommendation |

For each row, "supports_claim" should reflect the NEW framing
(joint multi-tracer Fisher with internal-DESI + PFS×DESI channels;
multi-tracer priors as data-driven complement to SBPs).

Recommendation column values:
  - "keep": citation supports the claim, no action.
  - "promote": citation deserves more weight given the new framing
    (paper to add as a co-citation elsewhere).
  - "demote": citation was prominent under old framing, less
    central now; consider trimming a co-citation.
  - "drop": surrounding claim no longer holds; if the claim is being
    edited away in the revision, drop the citation with it.
  - "verify": you are not certain the citation supports the claim;
    needs verification.

After the table, write a short summary: total citations, count by
recommendation. Do NOT propose new citations; that is reserved for
Phase E with strict verification.
```

- [ ] **Step 2: Verify report has the full table.**

```bash
grep -c "^|" docs/superpowers/plans/2026-05-05-audits/08-citations.md
```

Expected: at least one row per `\cite{}` invocation in the .tex.

---

## Phase B — Synthesize audit findings into a unified edit plan

Run after Phase A is complete. Single subagent (or inline).

### Task B1: Consolidate audit reports into a ranked edit list

**Files:**
- Read: all 8 reports in `docs/superpowers/plans/2026-05-05-audits/`
- Create: `docs/superpowers/plans/2026-05-05-audits/00-synthesis.md`

- [ ] **Step 1: Dispatch synthesis subagent (general-purpose)**

Prompt:
```
Read all eight audit reports in
docs/superpowers/plans/2026-05-05-audits/01-intro.md through
07-appendices.md and 08-citations.md. Produce a unified edit plan at
docs/superpowers/plans/2026-05-05-audits/00-synthesis.md with three
sections:

  ## Cross-section drift
  Patterns that appeared in multiple sections (e.g. "multi-survey
  priors" still used as umbrella term in N places). One bullet per
  pattern, listing the affected sections.

  ## Per-section edit list
  For each section §1, §2, §3, §4, §5, §6, appendices, list the
  specific edits to apply, ranked by impact. Format:
    1. [framing] line 234: "X" → "Y" because ...
    2. [tone] line 567: ...
  Keep edits self-contained — each line/sentence change should be
  applicable independently of the others.

  ## Citations to remove or rebalance
  From 08-citations.md, list "drop" and "demote" recommendations and
  the corresponding tex line(s).

  ## Citations to verify or potentially add
  From 08-citations.md "verify" rows plus any "potentially add"
  notes from the section audits. Mark each [NEEDS-VERIFICATION].
  These will go through Phase E strict review.

Do NOT make edits. The output is a planning document.
```

- [ ] **Step 2: Verify synthesis report exists and covers all sections.**

```bash
grep -E "^### §|^## " docs/superpowers/plans/2026-05-05-audits/00-synthesis.md | head -20
```

Expected: section markers for §1 through §6 + appendices + cross-section + citations.

---

## Phase C — Apply edits, one section per subagent

Each subagent:
1. Reads the synthesis edit list for its section.
2. Applies edits to `paper/PFSxDESI-multi-fish.tex`.
3. Runs a quick `pdflatex -draftmode` to verify the build still works.
4. Commits the section's edits as a single commit with a thorough message.

Subagents run **sequentially** to avoid merge conflicts on the single .tex file. Order: §1 → §2 → §3 → §4 → §5 → §6 → appendices.

**Common steps for each Phase C task:**

- [ ] **Step 1: Dispatch the edit subagent**

Use the `general-purpose` agent (with editing tools). Prompt template:

```
You are revising the cosmology paper at paper/PFSxDESI-multi-fish.tex.
Read the synthesis edit list at
docs/superpowers/plans/2026-05-05-audits/00-synthesis.md and apply
the edits scoped to <SECTION_NAME> only.

Editing constraints (strict):
- Preserve the tone and style of arXiv:2604.25171: declarative
  sentences, em-dash for definitional asides, quantitative
  attribution, no promotional language ("headline", "main result",
  "novel", "key finding"), no eponymous mechanism names.
- Two-channel narrative ordering must be visible:
  (a) single-tracer broad → (b) DESI multi-tracer → (c) DESI+PFS.
- Do not change \label{}, \cite{} keys, math notation, equation
  numbers, or table/figure references.
- Do not propose new \cite{} entries. If a passage reads weak
  without a citation, mark with a TeX comment "% [CITE-NEEDED:
  <what the citation should support>]" and leave it for Phase E.
- Match the body paper's noun phrases: "data-driven multi-tracer
  priors" (umbrella), "internal-DESI channel" / "PFS×DESI overlap
  channel" (the two channels), "joint multi-tracer Fisher" (the
  estimator), "DESI multi-tracer self-calibration" (the
  internal-DESI mechanism in the abstract).

After applying the edits:
1. Run `cd paper && pdflatex -draftmode -interaction=nonstopmode
   PFSxDESI-multi-fish.tex > /dev/null 2>&1` and check
   PFSxDESI-multi-fish.log for "undefined" or "^!" lines.
2. Stage and commit just paper/PFSxDESI-multi-fish.tex with a
   message in this format:

     revise §<N> <section name>: <one-line summary of the framing change>

     <bullet list of the specific edits applied, each with old → new>

     Ref: docs/superpowers/plans/2026-05-05-audits/00-synthesis.md

3. Report back: list of edits applied, build status, commit hash.
```

- [ ] **Step 2: Verify the build still works.**

```bash
cd paper && pdflatex -draftmode -interaction=nonstopmode PFSxDESI-multi-fish.tex > /dev/null 2>&1 && grep -iE "undefined|^! " PFSxDESI-multi-fish.log | head -3
```

Expected: empty output (no undefined refs, no fatal errors).

- [ ] **Step 3: Verify a commit was made.**

```bash
git log --oneline -1 paper/PFSxDESI-multi-fish.tex
```

Expected: commit message starts with "revise §<N>".

### Task C1: Revise §1 Introduction
Apply Phase B synthesis edits scoped to §1.

### Task C2: Revise §2 Methodology
Apply Phase B synthesis edits scoped to §2 (all subsections).

### Task C3: Revise §3 Setup
Apply Phase B synthesis edits scoped to §3.

### Task C4: Revise §4 Results
Apply Phase B synthesis edits scoped to §4. Critical guard: do NOT touch the §4.3 σ(Mν) diagnostic chain paragraph — it is load-bearing for the user's skepticism response and was already validated by four diagnostics.

### Task C5: Revise §5 Discussion
Apply Phase B synthesis edits scoped to §5. Same guard on §5.1 diagnostic-chain paragraph.

### Task C6: Revise §6 Conclusions
Apply Phase B synthesis edits scoped to §6.

### Task C7: Revise Appendices
Apply Phase B synthesis edits scoped to appendices.

---

## Phase D — Audit the audit (cold re-read passes)

Four subagents read the *revised* manuscript with fresh context, simulating a reviewer who has not seen the prior session. Each is asked to flag residual drift, internal inconsistencies, and any leftover old-framing language.

### Task D1: Cold re-read of abstract + §1 introduction

**Files:**
- Read: `paper/PFSxDESI-multi-fish.tex` lines 106-240
- Create: `docs/superpowers/plans/2026-05-05-audits/D1-cold-intro.md`

- [ ] **Step 1: Dispatch subagent (general-purpose)**

```
Read the abstract and introduction (lines 106-240) of
paper/PFSxDESI-multi-fish.tex with fresh eyes — assume you have
NOT seen the prior audit reports, the synthesis, or the previous
arXiv version. Your job: a cold reviewer's read.

Produce docs/superpowers/plans/2026-05-05-audits/D1-cold-intro.md
with these sections:

  ## What is the paper proposing?
  In one paragraph, summarize what you understand the paper's
  proposed method to be, based ONLY on the text you read.

  ## What is the dominant mechanism according to the abstract +
  ## introduction?
  Identify whether the dominant mechanism is described as:
  (i) cross-survey export of priors,
  (ii) joint multi-tracer Fisher with internal-DESI doing the bulk +
       PFS×DESI overlap adding on top,
  (iii) something else.

  ## Drift and inconsistency
  List any sentences that contradict each other, residual
  prior-export language, or framing that does not match (ii) above.

  ## Tone and style
  Note any promotional language, eponymous mechanisms, hand-waves,
  or floaty hedges.

  ## Citations
  Note any citations that look like they were attached to a claim
  no longer present, or any place a citation seems missing.
```

- [ ] **Step 2: Verify report. If drift is flagged, queue follow-up edits.**

### Task D2: Cold re-read of §2 + §3

Same template, scope `lines 241-712`, output `D2-cold-method.md`.

### Task D3: Cold re-read of §4 + §5

Same template, scope `lines 714-1183`, output `D3-cold-results-disc.md`.

### Task D4: Cold re-read of §6 + appendices

Same template, scope `lines 1185-end`, output `D4-cold-conclusions-app.md`.

### Task D5: Iteration decision

- [ ] **Step 1: Read all four D-reports.**

- [ ] **Step 2: Decide:**
  - If all four report "dominant mechanism = (ii)" with no major drift, proceed to Phase E.
  - If any report flags major drift, queue a second Phase C iteration scoped to the flagged sections, then re-run Phase D for those sections only.

The user's instruction is "Loop over the revision. Reiterate. Audit and audit the audit." — so a second loop is expected and welcome. Cap at 3 loops to avoid infinite oscillation.

---

## Phase E — Citation rigor pass

Strict default: no new citations. Existing citations may be removed or rebalanced (Phase A8 + Phase B output). Any candidate new citation flagged `[CITE-NEEDED]` in Phase C, or `[NEEDS-VERIFICATION]` in the synthesis, goes through 2–3 verification cycles before landing.

### Task E1: Apply citation removals and rebalancing from synthesis

**Files:**
- Read: `docs/superpowers/plans/2026-05-05-audits/00-synthesis.md` (citations-to-remove section)
- Edit: `paper/PFSxDESI-multi-fish.tex`

- [ ] **Step 1: Dispatch subagent (general-purpose)**

```
Read the "Citations to remove or rebalance" section of
docs/superpowers/plans/2026-05-05-audits/00-synthesis.md and apply
each removal/rebalancing to paper/PFSxDESI-multi-fish.tex.

For every removal:
  - Verify the surrounding claim is still in the paper (it may have
    been edited away in Phase C, in which case the removal is moot).
  - If the claim is still there, remove only the unsupported
    citation key from the \cite{} list. Do not delete the
    surrounding sentence.

For every rebalance (e.g. "Seljak:2008xr deserves to be paired with
McDonald:2008sh wherever cosmic-variance cancellation is invoked"):
  - Find each \cite{McDonald:2008sh} not already paired and add the
    co-citation.

Run pdflatex+bibtex+pdflatex+pdflatex and verify no undefined
citations. Commit:

  refs: rebalance citations after narrative revision

  - drop K from <line>: claim removed in Phase C
  - pair J alongside K at <lines>: cosmic-variance cancellation
  - ...
```

- [ ] **Step 2: Verify build is clean.**

### Task E2: Verify any candidate new citations (gate)

Only run if Phase A/B/C surfaced any `[CITE-NEEDED]` or `[NEEDS-VERIFICATION]` flag. Otherwise skip.

**Files:**
- Read: any `[CITE-NEEDED]` markers in the .tex or `[NEEDS-VERIFICATION]` rows in the synthesis
- Edit: potentially `paper/PFSxDESI-multi-fish.tex` and `paper/PFSxDESI-multi-fish.bib`

- [ ] **Step 1: List all candidate new citations.**

```bash
grep -nE "\[CITE-NEEDED|NEEDS-VERIFICATION" paper/PFSxDESI-multi-fish.tex \
  docs/superpowers/plans/2026-05-05-audits/*.md
```

If the grep is empty → skip the rest of E2.

- [ ] **Step 2: For each candidate, dispatch a citation-verification subagent (general-purpose with WebFetch).**

```
You are verifying whether a candidate citation actually supports the
claim it would be attached to. Strict three-cycle protocol — fail any
cycle and the citation is rejected.

Candidate: <bib_key or arxiv_id>
Claim text (verbatim from the .tex): "<claim>"

Cycle 1 — metadata verification:
  - Fetch arXiv abstract page or Semantic Scholar entry for the
    candidate.
  - Confirm: title, authors, year, journal, doi/eprint.
  - If any mismatch from a hypothetical existing bib entry, reject.

Cycle 2 — claim grounding:
  - Read the abstract of the candidate paper.
  - Does the abstract support the claim, or is the claim mentioned
    only in passing in the introduction/conclusion?
  - If the abstract doesn't support the claim, reject.

Cycle 3 — false-positive check:
  - Phrase the claim as a question. Ask whether the candidate paper
    is the BEST citation for it, or whether one of the existing 80+
    bib entries is better.
  - If a better existing citation is available, reject and use the
    existing one instead.

Output: ACCEPT (with one-line justification) or REJECT (with reason).
If ACCEPT, produce the bibtex entry exactly matching the verified
metadata.
```

- [ ] **Step 3: For each ACCEPT, add the bib entry and the \cite{} in the .tex.**

```bash
cd paper && pdflatex -draftmode... && bibtex... && pdflatex... && grep -iE "undefined|^! " PFSxDESI-multi-fish.log | head -3
```

Expected: clean.

- [ ] **Step 4: Commit.**

---

## Phase F — Final integrated read and consolidated commit

### Task F1: Final cold read

**Files:**
- Read: full `paper/PFSxDESI-multi-fish.tex`
- Create: `docs/superpowers/plans/2026-05-05-audits/F1-final-read.md`

- [ ] **Step 1: Dispatch one final subagent (general-purpose).**

```
Read paper/PFSxDESI-multi-fish.tex end-to-end with fresh context.
Produce a one-page summary at
docs/superpowers/plans/2026-05-05-audits/F1-final-read.md addressing:

  ## Coherence
  Does the abstract → introduction → methodology → results → discussion
  → conclusions form one coherent narrative? List any cross-section
  contradictions.

  ## (a) → (b) → (c) chain visibility
  Confirm the chain is visible in the abstract, results, and
  conclusions with consistent percentages (33/80/49% for (b) over (a);
  9/24/9% for (c) over (b)).

  ## Old-framing residues
  Any remaining "multi-survey priors" as umbrella term, "prior export",
  "Step 1 calibration → Step 2 cosmology", "headline", or eponymous
  mechanism names ("McDonald-Seljak").

  ## Citation hygiene
  Any \cite{} attached to claims that no longer hold, or any place
  that reads incomplete without a citation.

  ## Verdict
  PASS / NEEDS-ANOTHER-LOOP. If the latter, list specific sections.
```

- [ ] **Step 2: If verdict is PASS, proceed to F2. If NEEDS-ANOTHER-LOOP, queue another Phase C/D pass, then re-run F1.**

### Task F2: Run latex-paper-en compile + format checks

- [ ] **Step 1: Compile + bib + format checks.**

```bash
cd paper && rm -f PFSxDESI-multi-fish.{aux,bbl,blg,log,out,toc}
pdflatex -draftmode -interaction=nonstopmode PFSxDESI-multi-fish.tex > /dev/null 2>&1
bibtex PFSxDESI-multi-fish > /dev/null 2>&1
pdflatex -draftmode -interaction=nonstopmode PFSxDESI-multi-fish.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode PFSxDESI-multi-fish.tex 2>&1 | tail -3
grep -iE "undefined|^! " PFSxDESI-multi-fish.log | head -5
```

Expected: PDF builds, no undefined refs, no fatal errors.

- [ ] **Step 2: Run latex-paper-en format and references check.**

```bash
python3 /Users/nguyenmn/.claude/skills/latex-paper-en/scripts/check_format.py paper/PFSxDESI-multi-fish.tex
python3 /Users/nguyenmn/.claude/skills/latex-paper-en/scripts/check_references.py paper/PFSxDESI-multi-fish.tex
python3 /Users/nguyenmn/.claude/skills/latex-paper-en/scripts/verify_bib.py paper/PFSxDESI-multi-fish.bib --tex paper/PFSxDESI-multi-fish.tex
```

Expected: format PASS, no new bib lints (the pre-existing journal-field warnings on Cabass:2022wjy etc are out of scope and remain).

### Task F3: Update SESSION_STATE.md

- [ ] **Step 1: Edit SESSION_STATE.md to record this revision.**

Append a "## 2026-05-05 narrative revision" section summarizing what changed in this work-stream:
- What old framing was removed (prior export, multi-survey priors as umbrella).
- What new framing replaced it (joint multi-tracer Fisher, two channels).
- σ values unchanged.
- No new citations (or list any that landed via Phase E2).
- Reference to this plan file.

- [ ] **Step 2: Commit.**

```bash
git add SESSION_STATE.md
git commit -m "docs: log 2026-05-05 narrative revision in SESSION_STATE.md"
```

### Task F4: Final verification and handoff

- [ ] **Step 1: Quick sanity checks.**

```bash
git log --oneline | head -15
git status
```

Expected: clean working tree, new commits visible (one per Phase C section, one Phase E1 refs commit, F3 SESSION_STATE.md commit).

- [ ] **Step 2: Report to user with summary.**

Include:
- Number of Phase C edit commits.
- Number of citations dropped, demoted, paired (Phase E1 stats).
- Number of citations added (Phase E2 stats; should be 0 or very small).
- Any Phase D loops that were triggered.
- Final F1 verdict.

---

## Files modified (expected scope)

- `paper/PFSxDESI-multi-fish.tex` — narrative revision throughout. Probably 50–150 line changes spread across §1-§6 + appendices.
- `paper/PFSxDESI-multi-fish.bib` — only if Phase E2 lands an ACCEPTed new citation. Strong prior against this.
- `SESSION_STATE.md` — log the revision.
- `docs/superpowers/plans/2026-05-05-audits/*.md` — audit reports (8 from A, 1 synthesis, 4 from D, 1 from F1; ~14 markdown files). These are documentation, retained for the next session.

## Self-review

**Spec coverage:** the user's six concrete asks are each implemented:
1. "Adapt and update framing, narrative, discussion thoroughly" → Phase A audits + Phase B synthesis + Phase C edits cover all sections.
2. "Keep tone and style of the original" → editing constraints in plan header + Phase A6 tone check + Phase D cold reads.
3. "Change only narrative and emphasis" → Phase C subagents are explicitly told not to touch labels, cite-keys, math, equations, or scientific content.
4. "Orchestrate sub-agents in parallel and iterative tasks" → Phase A 8-way parallel, Phase D 4-way parallel; Phase C sequential; Phase D5 supports up to 3 loops.
5. "Loop over the revision, reiterate, audit and audit the audit" → Phase D explicitly is "audit the audit"; D5 supports re-iteration; F1 is the final cold audit.
6. "References must back up discussion; no hallucinated refs" → Phase A8 audits every existing citation; Phase E2 has a strict 3-cycle verification protocol gating any new citation; default policy is no new citations.

**Placeholder scan:** every Phase A/D task has a fully specified subagent prompt; no "TBD" or "fill in details". Phase C's per-section tasks share a single template by design (DRY) and are explicitly enumerated C1-C7.

**Type / signature consistency:** the plan uses consistent terminology throughout — "internal-DESI channel" / "PFS×DESI overlap channel" / "data-driven multi-tracer priors" / "joint multi-tracer Fisher". These match the noun phrases already in the abstract and §5 of the manuscript.
