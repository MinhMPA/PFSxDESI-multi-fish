# Cold re-read: §2 Methodology + §3 Setup

## What is the paper's method?

The paper forecasts cosmological constraints (fσ8, Mν, Ωm) from a single joint multi-tracer Fisher analysis run over the full 14,000 deg² DESI footprint. The volume is partitioned into a 1,200 deg² PFS×DESI overlap (carrying up to four tracers: PFS-ELG, DESI-LRG, DESI-ELG, DESI-QSO with their auto- and cross-spectra) and a 12,800 deg² DESI-only complement (three DESI tracers). At each redshift bin the per-bin Fisher decomposes additively as F_z = F_z^overlap(V_overlap) + F_z^nonoverlap(V_nonoverlap) under the Gaussian covariance / no-SSC approximation. Bias, stochastic, and counterterm parameters are 12 nuisance parameters per (tracer, z-bin) following Chudaykin–Ivanov–Philcox; auto- and cross-spectra over-determine bias and constant shot-noise terms, providing data-driven multi-tracer constraints on EFT nuisance parameters as a substitute/complement to simulation-based priors (SBPs). The analysis uses ps_1loop_jax + cosmopower-jax for derivatives via JAX autodiff, asymmetric kmax (PFS extends to ~0.27 h/Mpc via σ_v scaling), and reports three configurations: single-tracer broad baseline, DESI-only joint, DESI+PFS joint.

## Two-channel narrative consistency

The paper consistently presents the narrative as a single joint Fisher with **two information channels**, not as a three-stage chain (broad → DESI internal → DESI+PFS overlap). The audit-prescribed "(a) single-tracer broad → (b) DESI internal multi-tracer → (c) DESI+PFS overlap chain" is realized as **comparison configurations**, not as a sequential pipeline. Specifically:

- §2.2: "Two channels relax the nuisance prior in the present analysis. The internal-DESI channel uses LRG×ELG×QSO cross-spectra over the full DESI footprint... The PFS×DESI overlap channel adds PFS-ELG cross-spectra in the 1,200 deg² footprint" (lines 291–298). This is the two-channel framing — both channels are simultaneous, not sequential.
- §3 intro: "two information channels combined in a single analysis: the internal-DESI channel from LRG, ELG, and QSO multi-tracer cross-correlations everywhere, and the PFS×DESI overlap channel from the additional PFS-ELG auto- and cross-spectra in the 1,200 deg² overlap" (lines 486–490).
- §3.5 Analysis configurations: "We report three configurations, each a single Fisher analysis over the full 14,000 deg² DESI footprint" (lines 698–699). The single-tracer broad baseline, DESI-only joint, and DESI+PFS joint are *configurations to compare*, all of which are themselves single-pass joint Fishers — not stages of a chain.

§2 and §3 are internally consistent on this. The framing is "single joint Fisher with channels" throughout. There is no conflation; if anything, §3.5's "three configurations" maps cleanly onto the (a)/(b)/(c) the audit asks about, with the explicit caveat that they are independent baselines rather than a chained calibration-and-transfer.

## SBP positioning

SBPs are presented as a **methodologically distinct, complementary route** that constrains the same parameters via different mechanisms — not as a competitor that the multi-tracer analysis replaces. From §2.5 (lines 458–476):

> "SBPs encode the full joint distribution of EFT parameters from HOD mocks, tightening σ(s8) by 23–50% relative to broad Gaussian nuisance priors, but their accuracy is bounded by the fidelity of the galaxy–halo connection model. The joint multi-tracer Fisher trades model dependence for a volume limitation: the data-driven multi-tracer priors constrain b1, Pshot, and b_G2 without HOD assumptions, while SBPs additionally constrain the counterterms and k-dependent stochasticity that the cross-spectrum geometry constrains less effectively. The two are complementary—and because they constrain the same parameters through independent mechanisms, they serve as mutual consistency checks."

And §3.5 (lines 728–734): "the SBP route is methodologically distinct—simulation-based rather than data-driven multi-tracer priors". The framing is balanced: SBPs win on coverage (counterterms, k-dependent stochasticity); multi-tracer wins on model-independence; the two are mutual consistency checks. Good.

## Drift and inconsistency

- **`sec:prior_export` label**: not present. Grep returns no match. Closest is `sec:transferability` (line 738), and there's also `sec:joint_fisher` (line 415) and `sec:sbp_comparison` (line 456). No old-framing label residue in §2–§3.
- **"Step 1 / Step 2"**: not present anywhere in §2–§3.
- **"export priors" / "calibrated priors transferred"**: not present in §2–§3. The word "calibration" appears once in §3.7 (line 740) as "The calibration transfers from overlap to full area if and only if DESI-ELG in the overlap has the same HOD..." — this is a weak residue. In the joint-Fisher framing, there is no separate "calibration" step that transfers; the joint Fisher just *uses* the overlap as part of the volume partition. The transferability assumption is really a *uniformity* assumption on DESI-ELG HOD across the footprint, which the joint Fisher needs because per-(tracer, z-bin) nuisance parameters are shared between overlap and non-overlap regions of the same DESI tracer. The current wording ("calibration transfers", "calibrated and in-situ posteriors") still smells like the older two-step framing. Recommend rewording §3.7 to "uniformity assumption" or "HOD homogeneity assumption" to align with the §3.5 single-Fisher framing.
- **"alternative to SBPs"**: not present. SBPs are framed as complementary throughout.
- **"multi-survey priors"**: grep does not find this phrase in §2–§3. Clean.
- **Terminology consistency between §2 and §3**: §2 talks about "data-driven multi-tracer priors" (line 463–464) versus §3.4 "data-driven multi-tracer priors rather than externally imposed Gaussian widths" (lines 666–667). Consistent. However, §3.4 says "what makes the per-bin nuisance constraints data-driven multi-tracer priors" (lines 665–667), which uses "multi-tracer priors" as a noun describing the *outputs* of the joint Fisher. This is fine but slightly different from the §2 usage where "priors" is the input concept being relaxed. Not a blocker.

## Tone and style

- **Promotional language**: low. No "novel", "powerful", "state-of-the-art", "first-of-its-kind". Phrases like "data-drive", "tighten", "isolates ... far more tightly" (line 374–375) are quantitative claims tied to the multi-tracer geometry rather than promotional adjectives.
- **Eponymous mechanisms**: none — no "PFS-DESI synergy mechanism" or similar.
- **5+-citation stacks**: the largest cluster is `\cite{Seljak:2008xr,McDonald:2008sh}` then `\cite{Mergulhao:2021kip,Mergulhao:2023zso}` then `\cite{Rubira:2025scu}` (lines 301–304) — clusters of 2 each, individually attributed in the surrounding sentence ("developed in", "extended to", "applied to"). Compliant with the anti-stacking rule. §2.1 has `\cite{DESI:2025wzd,Chudaykin:2026nls}` (line 283), 2 cites with implicit roles (PS-level vs field-level SBP, which is clarified in §2.5). Acceptable.
- **Abstract-style hand-waves**: §3.4 line 664–667 ("This covariance, evaluated jointly over auto- and cross-spectra of the four tracers, is what makes the per-bin nuisance constraints data-driven multi-tracer priors rather than externally imposed Gaussian widths.") is borderline editorial — it's true but reads like a slogan in the middle of a Methods passage. Could be tightened or moved to §2. Minor.

## Citations

- Line 379: "benefit less from the cross spectrum~\cite{Zhao:2024eboss}" — the surname should be checked; the cite key suggests Zhao+ 2024 eBOSS, but no quantitative claim is made attributing the comparison directly to that paper. The sentence asserts a degeneracy property (a_0/a_2 ↔ c_0/c_2 both ∝ k²); the citation backs the empirical "benefit less" rather than the analytic degeneracy. This is fine but the connection is implicit.
- Line 379–380: "The one-loop bias structure additionally constrains b_G2 through its distinct μ-dependence in the loop integrals~\cite{Mergulhao:2021kip,Mergulhao:2023zso}" — this is a non-trivial claim (b_G2 separability via μ-anisotropy of loop kernels). The Mergulhão references do support multi-tracer EFT constraints on b_G2; appropriate.
- Line 517: "we assume the linear bias to be b_1(z)=0.9+0.4z~\cite{Orsi:2010obl}" — Orsi+2010 is the standard reference for PFS-style ELG bias parameterization. OK.
- Line 597–608 (table caption): heavy attribution to Lazeyras+2015, Chudaykin+2025, Zhang+ DESI:2025wzd. All are load-bearing for the fiducial values stated. OK.
- §3.7 (transferability) cites only DESI:2025zgx for imaging-systematics control. The claim "DESI target selection is spatially uniform by construction" might warrant a second cite (DESI ELG target selection paper, Raichoor+ or DESI:2024hhd). Minor gap.
- §2.4 line 431 cites Takada:2013bfn, Li:2014sga for "Fourier modes in disjoint volumes are uncorrelated and the two Fishers add exactly". These references are about super-sample covariance / volume partitioning; appropriate.
- §3.4: \cite{Chudaykin:2020aoj} for "one-loop EFTofLSS model" and \cite{Simonovic:2017mhp} for "FFTLog decomposition". Standard references. OK.

## Verdict

**PASS** with one small edit recommended.

Specific edit:
1. §3.7 (lines 737–748, `\subsection{Transferability assumption}`): the wording "The calibration transfers from overlap to full area..." and "calibrated and in-situ posteriors" still carries the older two-step framing residue (calibrate-then-export). In the §3.5 single-joint-Fisher framing, this is really an HOD-homogeneity assumption needed because per-tracer nuisance parameters are *shared* across overlap and non-overlap regions for DESI-ELG (and presumably for DESI-LRG, DESI-QSO). Recommend rewording to something like "Uniformity assumption: shared per-tracer nuisance parameters require that DESI-ELG (and DESI-LRG, DESI-QSO) have spatially uniform HOD across overlap and non-overlap regions; a violation would manifest as a detectable mismatch when the per-region Fishers are evaluated separately." Also — does the assumption apply only to DESI-ELG, or to all three DESI tracers? The current text only mentions DESI-ELG, which is suspicious.

Optional polish (non-blocking):
2. §3.4 line 664–667 ("This covariance ... is what makes the per-bin nuisance constraints data-driven multi-tracer priors rather than externally imposed Gaussian widths.") — slightly editorial for a Methods §; consider tightening or moving the sentiment to §2.2.
3. §3.7: consider adding a citation for spatially uniform DESI ELG target selection beyond just imaging-systematics control.

File: `/Users/nguyenmn/PFSxDESI-multi-fish/paper/PFSxDESI-multi-fish.tex` lines 255–751.
