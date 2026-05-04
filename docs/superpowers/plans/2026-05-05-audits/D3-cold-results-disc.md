# Cold re-read: §4 Results + §5 Discussion

## (a) → (b) → (c) chain visibility

The chain is visible and internally consistent across the prose, the headline figure caption, and `tab:headline`.

- (a) Single-tracer broad baseline: prose at L833 — "the broad single-tracer baseline (\(\sigma(f\se,\Mnu,\Om)=0.0216,\,0.688,\,0.0306\))" — and `tab:headline` row "DESI single-tracer broad: 0.0216 / 0.688 / 0.0306".
- (b) DESI multi-tracer joint: `tab:headline` row "DESI-only joint: 0.0145 / 0.141 / 0.0157" with `\Delta\%` = $-33$ / $-80$ / $-49$. Reproduced in prose at L834: "tightens ... by 33\%, 80\%, and 49\%". The §5.1 localization paragraph (L1097) repeats "33\%/80\%/49\% improvement on \((\sigma(f\se),\sigma(\Mnu),\sigma(\Om))\)" — consistent.
- (c) DESI+PFS joint: `tab:headline` row "DESI+PFS joint: 0.0132 / 0.108 / 0.0143" with `\Delta\%` = $-8.9$ / $-23.6$ / $-9.4$. Prose at L835 reports "8.9\%, ... 23.6\%, ... 9.4\%". Caption of `fig:full_area` rounds to "9\% / 24\% / 9\%" (L897-899). Discussion §5.1 cites "24\%" (L1085, L1140). §5.3 also cites "24\%" (L1192).

**Inconsistency / drift to flag:**
- L874 reads "\(\sigma(\Mnu)\) (\(+24\,\text{pp}\))" while `tab:headline` and L836 carry the precise "23.6\%". The 24\% in the figure caption and §5 prose is a clean rounding, but the "+24 pp" at L874 is in §4.2 *adjacent to* the precise "23.6\%" two lines earlier — the abrupt rounding inside the same paragraph reads as residual drift. Same paragraph also has "+9 pp / +9 pp" rounded vs the table's 8.9 / 9.4. Minor; recommend either rounding consistently throughout or footnoting.
- L1042 quotes the legacy cross-stochastic numbers "8.6/22.8/9.7\%" vs conservative "8.9/23.6/9.4\%". This is internally consistent with `tab:headline`.

## σ(Mν) diagnostic-chain paragraph (§5.1)

Quoted verbatim (L1090-1119):

> The diagnostic chain is direct in our Fisher results.
> \emph{(i) Localization.} Restricting the per-z-bin Fisher to a
> single bin reveals that
> \(\sigma(\Mnu)_{\rm DESI\,\,joint} =
>   \sigma(\Mnu)_{\rm broad\,\,single\text{-}tracer}\)
> to numerical precision in the four bins where only one DESI
> tracer is active (\(z<0.8\) LRG-only and \(z>1.6\) QSO-only); the
> 33\%/80\%/49\% improvement on \((\sigma(f\se),\sigma(\Mnu),\sigma(\Om))\)
> appears \emph{exclusively} in the four bins at \(0.8<z<1.6\)
> where two or three bias-distinct DESI tracers coexist (per-bin
> \(\Delta\sigma(\Mnu)\) of 67\%--85\%).
> \emph{(ii) Correlation collapse.} The Fisher correlation
> \(\rho(\Mnu, b_1\se)\) at \(z\!=\![0.8,1.0]\) drops from
> \(\{+0.95, +0.82, +0.45\}\) for \(\{\)LRG, ELG, QSO\(\}\) in the
> broad single-tracer baseline to \(\{+0.39, -0.07, -0.07\}\) in
> the DESI-only joint scenario, confirming that the
> cosmic-variance-cancelled bias-ratio determination from
> cross-spectra is the operative mechanism rather than any
> broadband-shape gain.
> \emph{(iii) Data dominance.} Loosening the Gaussian cosmology
> prior on \(\Mnu\) from \(\sigma_{\rm prior}=5\,\text{eV}\) to
> \(10^6\,\text{eV}\) changes \(\sigma(\Mnu)\) by \({<}1\%\) in
> both the broad single-tracer and DESI joint scenarios---the
> constraint is fully data-driven.

**Internal consistency check:**
- (i) Localization: claim that the gain occurs "exclusively" in bins with ≥2 DESI tracers — consistent with the 4 bins at $0.8<z<1.6$ stated.
- (ii) Correlation collapse: 0.95 → 0.39 for LRG (the dominant tracer of the prompt's "~0.95 → ~0.39" check). The audit prompt quoted this exactly. ELG (0.82 → −0.07) and QSO (0.45 → −0.07) likewise consistent with the broader "collapse" framing.
- (iii) Data dominance: prior 5 eV → 10⁶ eV, <1% change. Matches the prompt.

The paragraph is internally consistent and self-supporting. **GUARDED** — left untouched.

## Drift and inconsistency in §4 + §5

- **Old-framing residue (L874):** "+24 pp / +9 pp / +9 pp" mixes round numbers with the precise 23.6/8.9/9.4 in the same paragraph. Mild drift; not contradictory.
- **`tab:headline` cref label** is fine per the prompt.
- **Headline content of `tab:headline`** is internally clean: rows, $\Delta\%$ definition, footnote on cross-stochastic treatment all align with the §4 prose and the §4.4 sensitivity numbers.
- **Channel noun phrases:** the canonical pair is "internal-DESI channel" and "PFS×DESI overlap channel". Used consistently at L821-822, L832, L863, L933-934. No competing phrasings ("intra-DESI", "cross-survey channel") detected in §4-§5.
- **§4.3 "Calibration efficiency" paragraph (L959-966):** "neutrino mass constraint is most sensitive to the amplitude-related stochastic parameters that the joint multi-tracer calibration tightens." This framing rests on stochastic-parameter tightening as the $\Mnu$ driver, but §5.1 makes a cleaner case that the dominant driver is $b_1\sigma_8$ via cosmic-variance cancellation in bias ratios. Two adjacent sections give two different proximate causes for the $\Mnu$ gain — minor old-framing residue worth aligning.
- **§4.4 closing (L1043-1048):** "The largest impact is on $\sigma(\Mnu)$ because the $(k\mu)^2$ cross-stochastic term shares the broadband-shape geometry with the neutrino-mass signature." — Internally fine, but the broadband-shape framing is the very mechanism §5.1(ii) claims is *not* operative ("rather than any broadband-shape gain"). The two are about different parameters (cross-stochastic broadband vs $b_1$ ratio cancellation), but a careful reader will collide them. Worth either a clarifying parenthetical in §4.4 or one in §5.1.

## SBP positioning (§5.3)

L1156-1158: "The latter is model-independent by construction and provides a cross-check on whether SBP-calibrated \(b_1\) values are consistent with what the data themselves imply." — directionality correct: multi-tracer is the model-independent check on SBP/HOD.

L1183-1187: "A future PFS\(\times\)DESI joint multi-tracer analysis that reproduces this shift without HOD input would strengthen the new-physics interpretation; a null result would point to systematic biases in the galaxy--halo connection." — directionality also correct: multi-tracer is the arbiter.

L1189-1192: "The optimal strategy combines both approaches---multi-tracer calibration for \(b_1\) and the stochastic terms; SBPs for the counterterms and \(k\)-dependent stochasticity that the multi-tracer covariance cannot reach." — assigns scope to each, no inversion.

**SBP positioning is correct throughout §5.3.**

## Tone and style

- No promotional language detected ("revolutionize", "unprecedented", "first-ever" etc. — absent).
- **Eponymous-mechanism flag:** "McDonald-Seljak" not used as a bare label; cited via `\cite{Seljak:2008xr,McDonald:2008sh}` (L773, L870, L1073). Acceptable.
- **Citation stacks:** all 2-cite groupings (Seljak+McDonald, Mergulhao+Rubira, Ivanov+Bakx, FernandezGarcia+Vaisakh+Ishiyama+PFS-CO-mock at L1218-1220). The 4-cite mock-suite stack at L1218-1220 is a bibliographic enumeration, not a "many works have been proposed [1-5]" pattern; acceptable since each is named.
- **Hand-waves:** L1062 "The dominant driver is the calibration of the linear bias \(b_1\se\)." — backed by the diagnostic-chain paragraph below. L1148 "SBPs achieve a similar effect through a different route." — adequately specified ("simulated galaxy-halo connections" vs "cross-spectra between observed galaxy populations"). L1063-1066 quotes the precise correlation "$-0.99$" — quantified.
- Minor: L996 "The cosmology constraints turn out to be remarkably insensitive to both." — "remarkably" is a soft promotional word; the <0.4% / <0.5% numbers in the next paragraph carry the claim, so "remarkably" is removable for tone parity with the rest of the section.

## Citations

- L763 `\cite{Chudaykin:2025lww,DESI:2024jxi}` for "broad priors" — load-bearing on prior choice. Fine.
- L773, L870, L1073 `\cite{Seljak:2008xr,McDonald:2008sh}` for cosmic-variance cancellation in bias ratios — correctly attached to the mechanism claim.
- L877 `\cite{DESI:2025wzd}` for "$\sim\!23\%$ on $\sigma(\se)$" — specific number tied to a specific reference. Fine.
- L902 `\cite{Chudaykin:2026nls}` for the field-level SBP comparison line — fine.
- L939 `\cite{Ebina:2024ojt}` for cross-stochastic broad-prior treatment — appropriate.
- L1075 `\cite{Mergulhao:2023zso,Rubira:2025scu}` for "$k$-dependent EFT shape information at one-loop pins down the individual $b_1\se$ values" — load-bearing; acceptable.
- L1114 `\cite{Zhao:2024eboss}` for empirical eBOSS DR16 LRG×ELG 18-27% on $\sigma(\se)$ — concrete, cited.
- L1149 and L1180 both cite `Chudaykin:2026nls` for distinct field-level SBP claims (b₁ posterior shrinkage; downward σ8 shift). Both are load-bearing.
- L1196 `\cite{DESI:2024hhd,Chudaykin:2025lww}` for normal-vs-inverted neutrino hierarchy — fine.
- L1202 `\cite{Alarcon:2016bkr}` for "any pair of overlapping spectroscopic surveys" generalization — single citation supporting a broad claim; acceptable as a representative methodology reference.
- L1206 `\cite{Qin:2025nkk}` for J-PAS×PFS — fine.
- L1207-1208 `\cite{Ivanov:2023qzb,Bakx:2025pop}` for tree-level bispectrum activating $c_1$ and breaking $b_1$-$b_2$-$b_{\G_2}$ — these are also re-cited at L1224 for "non-Gaussian covariance ... modify the information budget". The same pair carrying two distinct technical claims in adjacent paragraphs is acceptable but borderline; consider differentiating by attaching one cite each to its more specific claim, or naming the contribution.
- **No missing-citation site detected** for claims that would normally demand attribution. The "PFS galaxy mocks `\cite{PFS-CO-mock:inprep}`" and similar `:inprep` keys (L1218-1221) are placeholders — flag for bibliography sweep before submission.

## Verdict

**PASS** with minor polish recommended. The (a)→(b)→(c) chain is consistent; the $\sigma(\Mnu)$ diagnostic paragraph is rigorous and self-contained; the SBP directionality is correct everywhere in §5.3; tone is largely sober.

Specific minor edits worth queuing (none blocking):

1. L874: harmonize "+24 pp / +9 pp / +9 pp" with the precise "23.6 / 8.9 / 9.4\%" already used two lines above; either round throughout that paragraph or quote precise figures.
2. L959-961 (§4.3 intro): align the proximate cause framing with §5.1 — i.e., $\Mnu$ gain comes from $b_1\sigma_8$ degeneracy break (cosmic-variance cancellation in bias ratios), not "amplitude-related stochastic parameters". The $b_1$ framing is the one §5.1 commits to.
3. L1043-1048 (§4.4 closing) vs L1107-1108 (§5.1(ii)): the "broadband-shape geometry" cross-stochastic argument and the "rather than any broadband-shape gain" $b_1$ argument are about different parameters but read as a tension; one parenthetical in either place would resolve it.
4. L996: drop "remarkably" — the numbers carry the claim.
5. L1207-1208 vs L1224: differentiate the dual citation of `Ivanov:2023qzb,Bakx:2025pop` by attaching each to its more specific role, or naming the contribution.
6. Bibliography pre-submission: resolve `:inprep` keys (L1218-1221) before submission.

Files referenced:
- `/Users/nguyenmn/PFSxDESI-multi-fish/paper/PFSxDESI-multi-fish.tex` (L752-1228 audited).
