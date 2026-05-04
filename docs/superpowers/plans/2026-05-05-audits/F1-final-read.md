# Final cold read — full manuscript

## Coherence
The arc abstract → intro → methods → results → discussion → conclusions reads as one narrative: 12-EFT-nuisance bottleneck → SBPs as one route, multi-tracer Fisher as a model-independent alternative → two-channel framework (internal-DESI + PFS×DESI overlap) in a volume-partitioned Fisher → headline numbers → b1σ8 calibration as the driver → S8/Mν cross-check implications. No cross-section contradictions found. Minor friction points only:
- Abstract reports the PFS-induced gain as "9%, 24%, 9%" (rounded), the intro as "9/24/9%" and "33/80/49%", and Tab. headline / conclusions as "8.9%, 23.6%, 9.4%" / "33%, 80%, 49%". Roundings are mutually consistent but readers will notice the asymmetric rounding (8.9 → 9 but 23.6 → 24 not 23). Defensible.
- §5.3 "Robustness" subsection on `f_shared` partly duplicates §4.4 sensitivity prose without adding new content; not a contradiction, but lightly redundant.

## (a) → (b) → (c) chain visibility
Visible end-to-end:
- Abstract: (a)→(b) "33%, 80%, 49% over single-tracer broad-prior baseline"; (b)→(c) "further 9%, 24%, 9%". Chain explicit.
- Intro (lines 208–225): same 33/80/49% and 9/24/9% with channel labels.
- Results §4.2 + Tab. headline: (a) σ=0.0216/0.688/0.0306; (b) Δ% = -33/-80/-49 (DESI-only joint); (c) Δ% = -8.9/-23.6/-9.4 (DESI+PFS joint). Δ% column footnote clearly states each row is "improvement over the preceding row".
- Conclusions item 2 (DESI alone reach) and item 3 (8.9/23.6/9.4 for adding PFS): consistent.
- Fig. full_area caption: "9%, 24%, 9%" rounded, matches abstract.

No drift in identity of channels/configurations. Percentage drift is just rounding (one-decimal vs integer); nothing reads as a different number.

## Old-framing residues
- "prior export" / "prior-export": one residue at line 1251, conclusions item 1 — "the auto-spectra double-counting that a two-stage prior-export pipeline incurs". This is intentional contrast against the legacy framing, but the term itself is the old language. Fine if read as historical-contrast, mildly jarring otherwise.
- "two-stage" / "legacy": appears at §4.3 (line 951 "legacy two-stage pipeline"), §4.4 (line 1011 "legacy single-tracer diagnostic"), §4.4 (line 1036 "legacy treatment"). Each is positioned as a deliberate diagnostic contrast — not an old framing residue, but the word "legacy" is used four times and reads slightly defensive.
- "Step 1 calibration → Step 2 cosmology": no residues.
- "multi-survey priors" as umbrella term: not present; the manuscript uses "data-driven multi-tracer priors" / "joint multi-tracer Fisher" consistently.
- "headline" in prose: none. Only the `tab:headline` cref label, as expected.
- Eponymous "McDonald-Seljak" / "MS09" mechanism names: no prose use. Only `\cite{Seljak:2008xr,McDonald:2008sh}` (always paired, never singled out as a named mechanism), and one figure label `fig:ms09_convergence` (internal label, never rendered).

## Citation hygiene
- No 5+-cite stacks. The only ≥4-stack is `\cite{Baumann:2010tm,Carrasco:2012cv,Carroll:2013oxa,Senatore:2014eva}` for the EFTofLSS foundation (line 164), which is appropriate boilerplate.
- 3-stack at line 165–168: the Chudaykin–Ivanov–Philcox / DESI / Ivanov citation cluster is unpacked sentence-by-sentence with named contributions — earns each citation.
- The `\cite{Seljak:2008xr,McDonald:2008sh}` pair recurs ~5 times. Each instance attaches to the specific cosmic-variance-cancellation/bias-ratio claim; not a stack.
- Possibly incomplete-without-citation: line 199–200 "deepening the tension with CMB measurements of S8" has no `\cite` for the CMB S8 measurement. Reader can chase via Chudaykin:2026nls earlier in the sentence; minor.
- Line 1271 conclusions item 4 "S8 tension" has no citation; assumed common knowledge but a single anchor citation would tighten it.
- All `\cite{...inprep}` keys (PFS-CO-target, PFS-CO-mock, Vaisakh, Ishiyama, Kobayashi:2026inprep) are flagged appropriately by name; no zombie claims.

## Tone and style
Generally tight and assertive. Specific items:
- Line 1305 acknowledgments: "heartening and uplifting conversations between the two VJA red-eye flights on which this work was begun and finished" — colorful personal note. Acceptable in acknowledgments.
- Line 1307 "It takes two villages to build the PFS and DESI surveys" — light idiom, fine.
- §5.3 first sentence "The PFS-induced improvement is stable across the natural variations of the analysis." — slightly promotional opener for a robustness section; the supporting sentence then quantifies, so it is anchored.
- Line 1140 "The DESI-internal multi-tracer self-calibration is doing the heavy lifting on the b1–Mν degeneracy" — "doing the heavy lifting" is mildly colloquial relative to surrounding prose, but the sentence concludes with a quantitative claim (24% from PFS), so it earns its keep.
- No floaty hedges like "may potentially", "could perhaps". No abstract closing — conclusions end on a concrete N-survey extension, not a sweeping claim.

## σ(Mν) diagnostic-chain paragraph (§5.1)
The (i)/(ii)/(iii) paragraph at lines 1092–1121 is internally consistent and well-anchored:
- (i) Localization: matches Tab. headline 33/80/49% headline numbers; states the 4 single-tracer bins reproduce the broad-baseline σ(Mν), and the four multi-tracer bins (0.8<z<1.6) carry the entire improvement, with per-bin Δσ(Mν) of 67–85%. The "exclusively" wording is supported by "to numerical precision".
- (ii) Correlation collapse: ρ(Mν, b1σ8) at z=[0.8,1.0] from {+0.95, +0.82, +0.45} (LRG, ELG, QSO) → {+0.39, -0.07, -0.07}. Internally consistent; numbers tie to the cross-spectrum bias-ratio mechanism named earlier.
- (iii) Data dominance: prior loosening 5 eV → 10⁶ eV with <1% change in σ(Mν) cleanly establishes data-drivenness; ties to the broad-vs-joint comparison in Tab. headline.
- The closing eBOSS DR16 cross-reference (Zhao et al. 18–27% on σ(σ8)) is qualitative ("consistent with our DESI-internal multi-tracer baseline once the larger DR2 footprint and three bias-distinct tracers are accounted for") — appropriate hedge.
GUARDED, no softening proposed.

## Verdict
PASS.

Tiny optional polish (NOT blocking):
- The two roundings of the PFS-induced gain (9/24/9 in abstract+fig, 8.9/23.6/9.4 in tab+conclusions) are both present; consider committing to one of them in the abstract for symmetry, or accepting current rounding as deliberate (precise where the number is the headline, rounded where it is narrative).
- "legacy" appears 4× across §4.3–§4.4; consider one rename to e.g. "diagnostic single-tracer pipeline" if the defensive read is a concern.
- §5.3 is a thin paragraph; could be folded into §5.2 closing, but standing alone is also fine.

Manuscript is reviewer-ready.
