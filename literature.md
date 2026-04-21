# Multi-tracer approaches for EFT parameter degeneracy breaking and calibration

## Scope of the question

The question was whether a **multi-tracer analysis using galaxy sample A and galaxy sample B from two different spectroscopic surveys** has already been considered as a way to break degeneracies among effective-field-theory (EFT) nuisance parameters, bias parameters, and cosmological parameters, or to calibrate such parameters more robustly.

The short answer is:

- **Yes in theory**: the multi-tracer EFT literature explicitly shows that using more than one tracer can reduce degeneracies among bias/EFT parameters and cosmological parameters.
- **Yes in same-survey data analyses**: this has already been demonstrated in practice for overlapping tracer populations such as LRGs and ELGs in eBOSS.
- **Yes in adjacent cross-survey / combined-tracer analyses**: recent DESI analyses and forecasts further support the same logic, although they are not all full EFT full-shape analyses.
- **But not yet as a mature standard pipeline** for *real-data, full-shape EFT analyses combining two galaxy samples from two distinct spectroscopic surveys* in the overlap volume.

## Main conclusion

A cross-survey multi-tracer EFT analysis is scientifically well motivated and is now clearly supported by the literature. The main gain is **not only** ordinary cosmic-variance cancellation. In the non-linear and EFT setting, the cross-spectrum and the enlarged tracer basis can make the parameter basis more orthogonal, which helps to break degeneracies among nuisance parameters and cosmology.

Formally, if the tracer overdensities are expanded as

\[
\delta_A(\mathbf{x},z) = \sum_O b_O^A(z)\, O(\mathbf{x},z) + \epsilon_A(\mathbf{x},z),
\]

\[
\delta_B(\mathbf{x},z) = \sum_O b_O^B(z)\, O(\mathbf{x},z) + \epsilon_B(\mathbf{x},z),
\]

then the auto- and cross-power spectra are schematically

\[
P_{AA}(k) = \sum_{O_\alpha,O_\beta} b_{O_\alpha}^A b_{O_\beta}^A\, P_{O_\alpha O_\beta}(k)
+ P^{AA}_{\rm c.t.}(k) + P^{AA}_{\rm stoch}(k),
\]

\[
P_{BB}(k) = \sum_{O_\alpha,O_\beta} b_{O_\alpha}^B b_{O_\beta}^B\, P_{O_\alpha O_\beta}(k)
+ P^{BB}_{\rm c.t.}(k) + P^{BB}_{\rm stoch}(k),
\]

\[
P_{AB}(k) = \sum_{O_\alpha,O_\beta} b_{O_\alpha}^A b_{O_\beta}^B\, P_{O_\alpha O_\beta}(k)
+ P^{AB}_{\rm c.t.}(k) + P^{AB}_{\rm stoch}(k).
\]

The cross-spectrum \(P_{AB}(k)\) carries direct information on **relative biasing**, **cross-stochasticity**, and survey-common structure. That is the structural reason multi-tracer analyses can help calibrate nuisance directions and reduce degeneracies.

## What the literature already establishes

## 1. EFT theory papers explicitly argue for degeneracy breaking

### Mergulhão et al. (2022): EFTofLSS + multi-tracer

A central reference is:

- Thiago Mergulhão, Henrique Rubira, Rodrigo Voivodic, L. Raul Abramo,
  **“The Effective Field Theory of Large-Scale Structure and Multi-tracer”**,
  JCAP 04 (2022) 021.
  - arXiv: https://arxiv.org/abs/2108.11363
  - DOI: https://doi.org/10.1088/1475-7516/2022/04/021

This paper is directly on point. It studies the perturbative bias expansion together with the multi-tracer technique and finds that the multi-tracer analysis gives stronger constraints on bias-expansion parameters and reduces degeneracies with cosmological parameters. In their setup, some cosmological parameters improve by about \(60\%\) relative to single-tracer fits, and bias-parameter uncertainties are typically about a factor of two smaller.

This is the clearest theoretical statement that the gain is **not only** cosmic-variance cancellation. The multi-tracer basis itself helps decorrelate parameters.

## 2. Same-survey spectroscopic EFT full-shape analyses already demonstrate the effect on real data

### Zhao et al. (2024): eBOSS DR16 LRG + ELG multitracer EFTofLSS

A direct observational implementation is:

- Ruiyang Zhao et al.,
  **“A multitracer analysis for the eBOSS galaxy sample based on the effective field theory of large-scale structure”**,
  MNRAS 532 (2024) 2018–2045.
  - arXiv: https://arxiv.org/abs/2308.06206
  - Journal page: https://academic.oup.com/mnras/article/532/2/2018/7693747
  - DOI: https://doi.org/10.1093/mnras/stae1452

This is the strongest existing real-data example in the exact spirit of the question. They jointly analyze eBOSS DR16 LRG and ELG samples with an EFTofLSS full-shape likelihood in Fourier space. Their mock tests show that with comparable footprints the multi-tracer analysis improves cosmological constraints by about \(10\%-35\%\), depending on assumptions about cross-stochasticity. In particular, the \(\sigma_8\) constraint improves by about \(18\%\), or about \(27\%\) if cross-stochastic terms are fixed to zero.

This paper also highlights a practical point that matters for any cross-survey implementation: prior choice, survey windows, and volume-projection effects materially affect the inference. That is a warning that cross-survey analyses are feasible but technically more demanding.

## 3. More recent theory sharpens the non-linear multi-tracer picture

### Rubira & Conteddu (2025): beyond linear theory

A more recent theory paper is:

- Henrique Rubira, Francesco Conteddu,
  **“Multi-tracer beyond linear theory”**,
  arXiv: https://arxiv.org/abs/2504.18245

This paper extends the non-linear multi-tracer discussion with Fisher analyses. The main message is that multi-tracer gains extend beyond the textbook linear-theory picture. The multi-tracer parameter basis becomes more diagonal than the single-tracer one, which effectively breaks degeneracies among cosmological and bias coefficients. It also emphasizes that the cross-spectrum contributes important information once non-linear bias terms are included.

This reinforces the interpretation that multi-tracer analyses can function as a nuisance-calibration tool in addition to a variance-reduction tool.

## 4. Earlier spectroscopic multi-tracer forecasts already showed the general mechanism

### Alarcon, Eriksen & Gaztañaga (2018)

A useful pre-EFT forecasting reference is:

- Alex Alarcon, Martin Eriksen, Enrique Gaztañaga,
  **“Cosmological constraints from multiple tracers in spectroscopic surveys”**,
  MNRAS 473 (2018) 1444–1460.
  - Journal page: https://academic.oup.com/mnras/article/473/2/1444/4222619
  - DOI: https://doi.org/10.1093/mnras/stx2446

This is not an EFTofLSS full-shape analysis, but it clearly shows the general multi-tracer mechanism in spectroscopic surveys. The improvement increases with bias contrast, and overlapping bins / overlapping tracers help further.

It is relevant because it establishes the core logic behind why two spectroscopic samples in the same volume should help disentangle growth, bias, and nuisance structure.

## What is closest to the specific “two different spectroscopic surveys” scenario?

The literature is still thinner here.

## 5. Combined-tracer DESI BAO analysis

A very relevant recent paper is:

- D. Valcin et al.,
  **“Combined tracer analysis for DESI 2024 BAO”**,
  arXiv: https://arxiv.org/abs/2508.05467

This is **not** a full EFT full-shape parameter-degeneracy paper. It is a combined-tracer BAO analysis for DESI DR1, merging overlapping LRG and ELG samples in \(0.8 < z < 1.1\). The paper reports an improvement of about \(11\%\) for \(\alpha_{\rm iso}\) and about \(7\%\) for \(\alpha_{\rm AP}\).

Why it matters here:

- it is a concrete, modern demonstration that overlapping tracers can be combined operationally in a survey pipeline;
- it shows that bias-informed tracer combination yields measurable gains on real data;
- it strengthens the case that cross-sample calibration strategies are now practical.

Why it does **not** fully answer the original question:

- it is BAO-focused rather than a full EFTofLSS nuisance-calibration analysis;
- it does not establish a mature cross-survey EFT pipeline.

## 6. DESI cross-correlations for local PNG

Another useful adjacent reference is:

- A. J. Rosado-Marín et al.,
  **“Local primordial non-Gaussianity using cross-correlations of DESI tracers”**,
  arXiv: https://arxiv.org/abs/2604.05213

This paper uses auto- and cross-correlations of DESI DR1 tracers, including LRGs, QSOs, and ELGs, to constrain \(f_{\rm NL}^{\rm loc}\). It reports that the LRG–QSO cross-correlation robustly improves the DESI DR1 \(f_{\rm NL}^{\rm loc}\) constraint by about \(9\%\), while the ELG contribution does not yet show a clear gain in the current data.

Again, this is **not** an EFT degeneracy-breaking paper in the strict sense. But it is highly relevant because it shows that cross-correlations among distinct tracers can be used as a robustness and systematics-control tool, precisely the role one would want in a future cross-survey EFT calibration program.

## Bottom line on the original question

The literature supports the following assessment.

### Already established
- Multi-tracer EFTofLSS analyses **do** break degeneracies among bias/EFT parameters and cosmological parameters.
- This has been shown both in theory and in real-data same-survey analyses.
- Cross-spectra are a key part of the information gain.

### Partially established
- Combined-tracer analyses in modern surveys like DESI are now operational.
- Cross-correlations are being used to improve robustness and reduce susceptibility to tracer-specific systematics.

### Still largely open
- A **published, standard, real-data, full-shape EFTofLSS analysis that combines galaxy sample A from spectroscopic survey 1 and galaxy sample B from spectroscopic survey 2 in the overlap volume** still does not appear to be a mature, widely adopted result.

So the scientifically honest answer is:

\[
\textbf{Yes, the idea has clearly been considered and is strongly motivated, but it is not yet a mature standard cross-survey EFT analysis.}
\]

## Why cross-survey multi-tracer EFT is attractive

A cross-survey implementation could do more than a same-survey split:

1. **Relative-bias calibration**  
   If the two surveys target different galaxy populations, then \(b_1^A\), \(b_2^A\), \(b_{G_2}^A\), \(b_{\Gamma_3}^A\), etc. can differ enough to help isolate EFT operator combinations.

2. **Systematics separation**  
   Survey-specific angular systematics, fibre assignment, redshift failures, completeness corrections, and selection functions are not shared. The cross-spectrum is comparatively less sensitive to non-common systematics.

3. **Different effective \(k\)-reach / FoG structure**  
   The two samples may have different Finger-of-God scales, number densities, and redshift precision, which can help stabilize the nuisance model.

4. **Overlap-volume self-calibration**  
   One can fit auto-spectra and the cross-spectrum jointly:
   \[
   \mathcal{L} = \mathcal{L}\!\left(P_{AA}, P_{BB}, P_{AB}\right),
   \]
   which constrains both common cosmology and tracer-dependent nuisance parameters.

## What makes it hard in practice

The reason this is not already routine is technical, not conceptual.

A real cross-survey EFT likelihood would need at least:

- accurate joint treatment of survey masks and window functions;
- consistent random catalogs or an equivalent forward model;
- explicit modeling of cross-shot-noise / cross-stochasticity;
- robust handling of redshift-range mismatch and effective-redshift mismatch;
- tracer-dependent EFT priors and possibly different \(k_{\max}\) choices;
- propagation of survey-specific systematics into the joint covariance.

That is a significant but tractable analysis problem.

## Practical take-away for future work

If the goal is **EFT parameter degeneracy breaking and calibration**, then the most promising path is likely:

1. choose two overlapping tracer populations with materially different bias structure;
2. build a joint likelihood for \(P_{AA}\), \(P_{BB}\), and \(P_{AB}\);
3. allow independent tracer nuisance parameters but shared cosmology;
4. test on mocks with realistic survey windows and cross-systematics;
5. quantify gain through posterior decorrelation, not only through fractional error shrinkage.

That would move beyond asking whether multi-tracer “helps” and directly measure **which EFT directions become identifiable only in the joint analysis**.

## References

1. Thiago Mergulhão, Henrique Rubira, Rodrigo Voivodic, L. Raul Abramo,  
   **The Effective Field Theory of Large-Scale Structure and Multi-tracer**  
   JCAP 04 (2022) 021  
   arXiv: https://arxiv.org/abs/2108.11363  
   DOI: https://doi.org/10.1088/1475-7516/2022/04/021

2. Ruiyang Zhao et al.,  
   **A multitracer analysis for the eBOSS galaxy sample based on the effective field theory of large-scale structure**  
   MNRAS 532 (2024) 2018–2045  
   arXiv: https://arxiv.org/abs/2308.06206  
   Journal: https://academic.oup.com/mnras/article/532/2/2018/7693747  
   DOI: https://doi.org/10.1093/mnras/stae1452

3. Henrique Rubira, Francesco Conteddu,  
   **Multi-tracer beyond linear theory**  
   arXiv: https://arxiv.org/abs/2504.18245

4. Alex Alarcon, Martin Eriksen, Enrique Gaztañaga,  
   **Cosmological constraints from multiple tracers in spectroscopic surveys**  
   MNRAS 473 (2018) 1444–1460  
   Journal: https://academic.oup.com/mnras/article/473/2/1444/4222619  
   DOI: https://doi.org/10.1093/mnras/stx2446

5. D. Valcin et al.,  
   **Combined tracer analysis for DESI 2024 BAO**  
   arXiv: https://arxiv.org/abs/2508.05467

6. A. J. Rosado-Marín et al.,  
   **Local primordial non-Gaussianity using cross-correlations of DESI tracers**  
   arXiv: https://arxiv.org/abs/2604.05213
