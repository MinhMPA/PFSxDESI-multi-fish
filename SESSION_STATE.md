# Session state — for resuming Claude Code on another machine

> Read this file at the start of a new Claude Code session in this repo
> to pick up where the previous session left off. Pair with the
> existing `CLAUDE.md` (project-level instructions) and
> `pfsfog/README.md` (architecture).

Last updated: **2026-05-05** at commit `d3cb1ba` on branch `master`.

---

## How to resume

On the other machine:

```bash
git pull origin master   # pull all the local commits (see "what shipped" below)
```

then in the new Claude Code session paste:

> Read `SESSION_STATE.md` and `CLAUDE.md`, then summarize the recent
> direction so I can confirm we're on the same page before continuing.

---

## What shipped in this work-stream (chronological)

All on `master`, none pushed to remote yet. Run `git log --oneline
4230e1f..HEAD` to see the full chain. The major branches that landed:

| Branch (now merged) | What it added |
|---|---|
| `perf/jit-derivatives` | JIT-compile + vectorize the auto- and cross-spectrum nuisance derivatives via `jax.jacfwd` over a flat 12-vector. ~30× per-call speedup on each. |
| (loose commits on master) | JIT cosmology derivatives (`_dPell_d_OmMnu_jit`, `_dPell_d_fsigma8_jit`); enable the JAX persistent disk cache in the parent kernel as well as workers. |
| `perf/jit-cosmo-derivatives` | Wired the JIT'd cosmology kernel into `build_zbin_fisher`. Combined with the prior steps, per-z-bin warm-cache build time dropped from **26–72 s** (pre-JIT) to **2.66–5.03 s** (≈12–15× cumulative speedup). |
| `feat/asymmetric-kmax` | Reinstated the per-pair `kmax` convention (`kmax_PFS = kmax_DESI / r_sigma_v`) inside the joint Fisher. Implemented as masked-submatrix Fisher accumulation in `_assemble_fisher_with_cosmo` (`pair_kmax` arg). Found that the cosmology impact is sub-percent; gain accrues mostly to PFS-ELG nuisance σ at high z. |
| `feat/marginalize-cross-stoch` | Per **Ebina & White (2024, arXiv:2401.13166)**, every cross-pair carries two free stochastic parameters (`Pshot_cross`, `a2_cross`) marginalized via broad priors with fiducial 0. Controlled by `cfg.marginalize_cross_stoch` (default **True**). |
| Notebook + draft updates | Regenerated all figures, restructured abstract (DESI internal multi-tracer leads, PFS layered on top), demoted asymmetric kmax, added Ebina-White citation, flipped title to "**DESI+PFS**" (was "PFS×DESI"), added §5.1 σ(Mν) diagnostic chain, replaced "McDonald–Seljak mechanism" everywhere with technical language ("cosmic-variance cancellation in cross-spectrum bias ratios"), removed every promotional use of "headline". |

## Current σ values (DESI+PFS joint, default config)

| Parameter | DESI single-tracer broad | DESI multi-tracer joint | DESI+PFS joint | PFS-unique Δ% |
|---|---|---|---|---|
| σ(fσ8) | 2.16e-2 | 1.45e-2 | **1.32e-2** | −9% |
| σ(Mν) [eV] | 0.688 | 0.141 | **0.108** | −24% |
| σ(Ωm) | 3.06e-2 | 1.57e-2 | **1.43e-2** | −9% |

DESI multi-tracer self-calibration alone improves over the single-tracer
broad baseline by **33 / 80 / 49 %** on (fσ8, Mν, Ωm). PFS-unique
improvement on top: 9 / 24 / 9 %.

The 80 % σ(Mν) gain has been validated by four independent diagnostics
(`/tmp/sanity_mnu.py`): per-z-bin localization (gain only in bins with
≥2 DESI tracers), kmax monotonic scan, ρ(Mν, b₁·σ8) correlation
collapse from 0.95 to 0.39, and data-domination check
(σ(Mν) changes <1 % when the cosmology prior is loosened).

## User preferences (style & language)

These have been calibrated through multiple rounds in this session.
Apply them on every future edit unless the user explicitly redirects:

- **No promotional language.** Avoid "headline", "main result", "key
  finding", "novel". Use direct technical descriptors instead
  (e.g. "cosmology forecasts", "values", "the constraint").
- **No eponymous mechanism names.** Replace with technical
  description. Specifically: never write "McDonald–Seljak
  mechanism"; write "cosmic-variance cancellation in cross-spectrum
  bias ratios" (and cite both `Seljak:2008xr` and `McDonald:2008sh`
  where appropriate).
- **Mirror the rhythm of arXiv:2604.25171** (a previous arXiv version
  of this paper). Direct, declarative sentences. Em-dash for
  definitional asides. Quantitative attribution where possible
  ("X % of the gain"). Conservative hedges only where genuinely
  needed.
- **Multi-tracer narrative ordering is (a) → (b) → (c)**:
  (a) DESI single-tracer broad-prior baseline,
  (b) DESI internal multi-tracer self-calibration (the dominant gain),
  (c) DESI+PFS joint (PFS layered on top).
  This ordering shows up in the abstract, results, and conclusions —
  keep it consistent.
- **Cross-stochasticity discussion is brief in the abstract**, full in
  §2.2 and §4.3. The abstract carries one trailing clause:
  "after marginalizing over residual cross-population stochasticity
  [Ebina:2024ojt]".
- **Title is "Multi-tracers, multi-surveys: a joint Fisher analysis
  of DESI+PFS"**. Body uses "DESI+PFS joint" / "DESI-only joint" for
  scenarios; "PFS–DESI overlap" stays for the geographic region
  (smaller-first); cross-spectrum math notation `P^{PFS×DESI-ELG}`
  stays (tracer-pair convention).
- **Repo / bib filename `PFSxDESI-multi-fish` does NOT change**
  (keeps GitHub URL stable).

## Codebase invariants worth knowing

- `cfg.marginalize_cross_stoch=True` is the default; `False` runs the
  legacy fixed-fiducial path for backward compat / regression checks.
- `cfg.r_sigma_v=0.75` controls the asymmetric kmax. Symmetric is
  recovered by setting it to `1.0` (or by setting
  `cfg.kmax_pfs_overlap=cfg.kmax_desi_overlap` explicitly).
- The JAX persistent cache lives at `.cache/jax/` (override via
  `PFSFOG_JAX_CACHE_DIR`). It's enabled in `pfsfog/__init__.py` so
  every kernel that imports pfsfog gets it. **Caveat:** the
  cosmology JIT kernel (`_dPell_d_OmMnu_jit`) is not persisted to
  disk because cosmopower-jax embeds a host callback in its predict
  path that JAX's persistent cache refuses to serialize. Per kernel
  session this costs ~24 s once, then in-memory cache takes over.
- All published σ values were computed with `pfsfog/configs/default.yaml`
  and are reproduced exactly by `python scripts/run_joint_fisher.py
  --parallel --n-workers 8` from a clean kernel.
- Tests: `pytest tests/ -q` for the default suite (137 tests),
  `pytest tests/ -m slow -v` for the 2 parallel-vs-sequential
  equivalence tests. The notebook is regenerated by
  `jupyter nbconvert --to notebook --execute --inplace
  notebooks/figures.ipynb`.

## Pending / open items

- **Push to remote.** All commits are local. Run `git push origin
  master` when ready.
- **Notebook is up to date** as of commit `3fc912f` on master; nothing
  in `pfsfog/` or `scripts/` has changed since, so no re-execution is
  needed unless you change derivative code.
- **arXiv submission.** Draft is at 26 pages, builds clean, but has
  not yet been re-uploaded. The previous arXiv version is at
  `arXiv:2604.25171` (cited in the user's style-audit request as the
  tone reference).
- **Pre-existing bibliography lints** (missing `journal` field on
  `Cabass:2022wjy`, `Tsedrik:2025cwc`, `Akitsu:2024lyt`,
  `DESI:2016fyo`, `Bakx:2025pop`, `Rogers:2023ezo`, `Piras:2023aub`,
  `FernandezGarcia:2025rso`). Not introduced in this work-stream;
  flag if a reviewer complains.

## 2026-05-05 narrative revision (subagent-driven)

A multi-phase narrative revision swept the manuscript on 2026-05-05.
The plan and audit reports live at
`docs/superpowers/plans/2026-05-05-narrative-revision-multi-tracer-framing.md`
and `docs/superpowers/plans/2026-05-05-audits/`.

What changed:

- **Framing.** The paper had carried residual language from an earlier
  *prior-export* framing (multi-survey calibration in the overlap →
  export to the non-overlap, parallel to SBPs). The revision retired
  those residues across §1, §2, §3, §4, §6, and the appendices and
  installed a *joint multi-tracer Fisher* framing with two channels:
  (a) DESI single-tracer broad-prior baseline, (b) the internal-DESI
  channel (LRG×ELG×QSO across 14,000 deg², 33/80/49% improvement over
  (a)), and (c) the PFS×DESI overlap channel adding 8.9/23.6/9.4% on
  top of (b). The chain is now visible in the abstract, results table,
  and conclusions with consistent percentages.
- **Canonical noun phrases** (used everywhere the body talks about the
  mechanism): "data-driven multi-tracer priors" (umbrella),
  "internal-DESI channel" / "PFS×DESI overlap channel" (the two
  channels), "joint multi-tracer Fisher" (the estimator),
  "cosmic-variance cancellation in cross-spectrum bias ratios"
  (the technical mechanism — never "McDonald-Seljak mechanism").
- **§3.5 scenarios** now describes all three of (a)/(b)/(c) in flowing
  prose (was only (b)/(c) as an itemize).
- **Tab. headline** gained an (a) baseline row so the chain is visible
  in one place.
- **§3.7** retitled from "Transferability assumption" to an
  HOD-uniformity assumption framing, and extended from DESI-ELG alone
  to all three DESI tracers.
- **σ(Mν) diagnostic chain (§5.1)** untouched (GUARDED). It now
  serves as the load-bearing response to the σ(Mν)=80% reviewer
  question with three internal checks (per-z-bin localization,
  ρ collapse, data dominance) plus a Zhao+ eBOSS empirical anchor.
- **Citation rebalancing.** No new bib entries added; existing keys
  re-paired. Two 5-cite stacks in §1 were split into per-work prose
  attribution (DESI-DR1 reanalyses; multi-tracer Fisher forecasts).
  Seljak:2008xr is now consistently paired with McDonald:2008sh
  wherever cosmic-variance cancellation is invoked.
- **Promotional language removed** throughout: "headline", "main
  result", "novel", and floaty hedges. The label `\label{tab:headline}`
  is preserved (cref renders as "Tab. 1"; word never reaches the
  reader).

σ values are unchanged from the pre-revision state: DESI-only joint
σ(fσ8, Mν, Ωm) = 0.0145, 0.141, 0.0157; DESI+PFS joint
σ = 0.0132, 0.108, 0.0143. Numbers were not touched in this revision —
only the framing and noun phrases around them.

PDF: 27 pages (one more than pre-revision because of the 5-cite stack
splits in §1), `pdflatex` clean, `verify_bib` clean for entries used
by the revised text, latex-paper-en `check_format` PASS.

Commits applied this revision:
- `1ecb92d` revise §1 Introduction
- `9f2f27d` revise §2 Methodology (including `sec:prior_export` rename)
- `868d9ab` revise §3 Setup
- `7bec74a` revise §4 Results
- (no §5 commit — Phase A audit returned PASS; Phase D D3 PASS)
- `828939b` revise §6 Conclusions
- `2641034` revise appendices
- `109b373` polish: cold-read findings (D1+D2+D3+D4)
- `d3cb1ba` refs: rebalance citations after narrative revision
Plus this SESSION_STATE.md log commit.

## Quick orientation files

- `pfsfog/README.md` — architecture, conventions, cross-shot prescription.
- `CLAUDE.md` — project-level Claude Code instructions.
- `paper/PFSxDESI-multi-fish.tex` — the draft.
- `notebooks/figures.ipynb` — figure source + diagnostic outputs.
- `scripts/run_joint_fisher.py` — top-level driver.
- `tests/test_fisher_joint.py` — invariants pinned by tests, including
  `test_pfs_kmax_zero_recovers_desi_only`,
  `test_assemble_kmax_mask_inert_when_uniform`,
  `test_marginalize_cross_stoch_false_recovers_legacy`.
