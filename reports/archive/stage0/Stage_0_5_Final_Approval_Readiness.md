# Stage 0.5: Final Pseudo-Covariance, Residual-Normalization, and Statistical Corrections
## Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals

---

### 1. Executive Summary

This document establishes **Stage 0.5: Final Pseudo-Covariance, Residual-Normalization, and Statistical Corrections** for the research project *Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals*. It is a narrowly scoped correction round addressing four specific mathematical and statistical issues identified in the independent scientific review of Stage 0.4.

#### Core Revisions Delivered in Stage 0.5

1. **Complete Pseudo-Covariance Treatment (Section 3–4):** Derived the cross pseudo-covariance $P_{12}(k) = \mathbb{E}[X_{m,k} X_{m+1,k}]$ and individual pseudo-covariance $P_{11}(k) = \mathbb{E}[X_{m,k}^2]$ explicitly for white Gaussian noise with the discrete Tukey window. Showed that these are DFT-like sums over window products that do not generically vanish for finite tapered transforms. Gave the complete periodogram covariance $\operatorname{Cov}(|Z_1|^2, |Z_2|^2) = |C_{12}|^2 + |P_{12}|^2$ and the full normalized correlation $\rho_P$ including pseudo-covariance contributions. Clarified that the simpler result $\rho_P = r_w^2$ holds only when pseudo-covariance terms are negligible (the proper-complex condition), and that this must be verified empirically.

2. **Filtered Noise Residual Normalization (Section 5):** Corrected the residual normalization to use $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ for the forward-backward Butterworth conditioning filter. Specified the exact processing order (generate → filter → trim → window → FFT → normalize) and updated the normalized residual definition. Defined the $\pm 3\%$ validation tolerance precisely as applying to ensemble-averaged normalized power, band-averaged over smooth spectral regions.

3. **Refined Phase-Marginalization Conditions (Section 6):** Replaced the overly strong "if and only if" statement with a properly scoped "sufficient condition" for single-$I_0$ Bessel marginalization. Added a four-part physical-waveform validation protocol: (a) phase grid evaluation, (b) norm variation measurement, (c) phase dependence of data overlap, (d) direct $I_0$ vs. numerical integration comparison — with the numerical comparison designated as the decisive test.

4. **Correct Odd/Even Median Formula (Section 7):** Explicitly separated the odd-$K$ ($K = 2L+1$) and even-$K$ ($K = 2L$) finite-sample median correction formulas. Noted that the alternating harmonic series representation equals the odd-$K$ case only, and that both baseline configurations ($K = 15$, $K = 255$) are odd-$K$ cases.

#### Material Preserved from Stage 0.4

All correct mathematical material from Stage 0.4 is preserved, including:

- The cross-covariance derivation $C_{12}(k) = (-1)^k r_w \sigma^2 M S_2$ for white noise.
- The exact discrete Tukey window definitions and implementation mandates.
- The spectral convolution formula for windowed coloured-noise expectations.
- The exact Hilbert quadrature family and its mathematical properties.
- The correct discrete Parseval relation.
- The complete experimental protocol and baseline parameter conventions.

---

### 2. Stage 0.4 Correction Log

```
+-----------------------------------------------------------------------------------------------------------------------------------+
|                                           Record of Stage 0.4 to Stage 0.5 Corrections                                            |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
| No.| Review Issue                     | Defective Formulation in Stage 0.4        | Reconciled Stage 0.5 Resolution                |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
| 1  | Pseudo-Covariance Assumed Zero   | Stated E[Z_1 Z_2] approx 0 away from DC  | Derived P_12(k), P_11(k), P_22(k) explicitly. |
|    |                                  | and Nyquist without derivation. Concluded | Gave full periodogram covariance including     |
|    |                                  | rho_P = r_w^2 as unconditional result.    | |P_12|^2 term. Clarified rho_P = r_w^2 valid  |
|    |                                  |                                           | only under proper-complex assumption.          |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
| 2  | Filtered Residual Normalization  | Validation protocol applied bandpass      | Derived S_filtered(f) = |H(f)|^4 S_n(f) for   |
|    |                                  | filter but normalized using unfiltered    | filtfilt. Updated residual definition and      |
|    |                                  | S_n(f) in denominator.                    | processing order explicitly.                   |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
| 3  | Phase Marginalization Claimed    | Stated I_0 marginalization holds "if and  | Replaced with "sufficient condition." Added    |
|    | "If and Only If"                 | only if" norm invariance + orthogonality  | 4-part validation. Numerical I_0 comparison    |
|    |                                  | hold. No decisive validation proposed.    | designated as decisive test.                   |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
| 4  | Odd/Even Median Formula          | Presented C(K) = sum_{j=1}^{K} (-1)^{j+1}| Separated odd K=2L+1 and even K=2L formulas.  |
|    |                                  | /j as generic formula for all K.          | Noted alternating harmonic = odd K only.       |
|    |                                  |                                           | Both K=15 and K=255 confirmed as odd cases.    |
+----+----------------------------------+-------------------------------------------+------------------------------------------------+
```

---

### 3. Complete Covariance and Pseudo-Covariance Derivation

#### 3.1 Setup and Definitions

Let $n_j$ ($j \in \mathbb{Z}$) be a discrete zero-mean stationary real Gaussian noise sequence with autocorrelation $R_n[p] = \mathbb{E}[n_j n_{j+p}]$. We partition the series into segments of length $M$ with shift $D = M/2$ (50% overlap). The windowed DFT coefficients for segment $m$ and adjacent segment $m+1$ are:

$$X_{m,k} = \sum_{j=0}^{M-1} w_j \, n_{j+mD} \, e^{-2\pi i jk/M}$$

$$X_{m+1,k} = \sum_{l=0}^{M-1} w_l \, n_{l+(m+1)D} \, e^{-2\pi i lk/M}$$

where $w_j$ is the real-valued discrete Tukey window ($\alpha = 0.25$, $M = 32{,}768$).

Because $n_j$ is real and Gaussian, the pair $(X_{m,k}, X_{m+1,k})$ are jointly complex Gaussian with zero mean. Their complete second-order statistics are specified by four quantities:

- **Covariance:** $C_{12}(k) \equiv \mathbb{E}[X_{m,k} \, X_{m+1,k}^*]$
- **Pseudo-covariance (relation):** $P_{12}(k) \equiv \mathbb{E}[X_{m,k} \, X_{m+1,k}]$
- **Individual variance:** $C_{aa}(k) \equiv \mathbb{E}[|X_{a,k}|^2]$ (same for $a = m$ or $m+1$ by stationarity)
- **Individual pseudo-covariance:** $P_{aa}(k) \equiv \mathbb{E}[X_{a,k}^2]$

A complex Gaussian random variable $Z$ is called **proper** (or circularly symmetric) if and only if $\mathbb{E}[Z^2] = 0$. Properness is equivalent to $\operatorname{Re}(Z)$ and $\operatorname{Im}(Z)$ having equal variances and being uncorrelated. A key point: even though the underlying noise $n_j$ is real and stationary, the *finite windowed* DFT coefficients $X_{m,k}$ are not automatically proper for all $k$.

#### 3.2 Cross-Covariance $C_{12}(k)$ for White Noise (Preserved from Stage 0.4)

For white noise with variance $\sigma^2$, $R_n[p] = \sigma^2 \delta_{p,0}$. The derivation from Stage 0.4 yields:

$$C_{12}(k) = \mathbb{E}[X_{m,k} \, X_{m+1,k}^*] = \sigma^2 (-1)^k \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2}$$

The individual variance is:

$$C_{aa}(k) = \mathbb{E}[|X_{a,k}|^2] = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2$$

The normalized cross-correlation coefficient is:

$$\gamma_k = \frac{C_{12}(k)}{C_{aa}(k)} = (-1)^k r_w, \qquad r_w \equiv \frac{\sum_{l=0}^{M/2-1} w_l \, w_{l+M/2}}{\sum_{j=0}^{M-1} w_j^2}$$

**This result is exact for white noise and holds at every frequency bin $k$.**

#### 3.3 Cross Pseudo-Covariance $P_{12}(k)$ for White Noise

We now derive $P_{12}(k) = \mathbb{E}[X_{m,k} \, X_{m+1,k}]$ — the quantity Stage 0.4 assumed to vanish.

$$P_{12}(k) = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} w_j \, w_l \, \mathbb{E}[n_{j+mD} \, n_{l+(m+1)D}] \, e^{-2\pi i(j+l)k/M}$$

For white noise, $\mathbb{E}[n_{j+mD} \, n_{l+(m+1)D}] = \sigma^2 \delta_{j-l,D} = \sigma^2 \delta_{j, l+D}$.

This constrains $j = l + D = l + M/2$, which requires $l + M/2 \le M - 1$, i.e., $l \le M/2 - 1$. Setting $j = l + M/2$:

$$P_{12}(k) = \sigma^2 \sum_{l=0}^{M/2-1} w_{l+M/2} \, w_l \, e^{-2\pi i(l + M/2 + l)k/M}$$

$$= \sigma^2 \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2} \, e^{-2\pi i(2l + M/2)k/M}$$

Factoring the constant phase:

$$= \sigma^2 \, e^{-i\pi k} \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2} \, e^{-4\pi i lk/M}$$

$$= (-1)^k \, \sigma^2 \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2} \, e^{-4\pi i lk/M}$$

**Equivalently**, substituting $q = l + D = l + M/2$ (so $l = q - D$, $q$ runs from $D$ to $M - 1$):

$$P_{12}(k) = \sigma^2 \sum_{q=D}^{M-1} w_q \, w_{q-D} \, e^{-2\pi i(2q - D)k/M}$$

which matches the form stated in the reviewer's request.

**Key observation:** $P_{12}(k)$ involves the same window-overlap product $w_l \, w_{l+M/2}$ as $C_{12}(k)$, but with an oscillatory DFT kernel $e^{-4\pi ilk/M}$ rather than a constant. Therefore:

- $P_{12}(k)$ is **not generically zero**.
- It vanishes only if the oscillatory sum cancels, which depends on the window shape and frequency bin $k$.
- At $k = 0$: $P_{12}(0) = (-1)^0 \sigma^2 \sum_{l=0}^{M/2-1} w_l w_{l+M/2} = C_{12}(0)$, so the pseudo-covariance equals the covariance.
- For large $k$ away from $k = 0$ and $k = M/2$, the oscillatory kernel causes substantial cancellation for smooth (slowly varying) windows. The Tukey window product $w_l w_{l+M/2}$ is a smooth, compactly supported function of $l$, so its DFT at high frequencies is exponentially suppressed.
- At $k = M/2$: $e^{-4\pi i l(M/2)/M} = e^{-2\pi i l} = 1$, so $P_{12}(M/2) = (-1)^{M/2} \sigma^2 \sum_l w_l w_{l+M/2} = (-1)^{M/2} C_{12}(0)$.

#### 3.4 Individual Pseudo-Covariance $P_{11}(k)$

$$P_{11}(k) = \mathbb{E}[X_{m,k}^2] = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} w_j \, w_l \, \mathbb{E}[n_{j+mD} \, n_{l+mD}] \, e^{-2\pi i(j+l)k/M}$$

For white noise, $\mathbb{E}[n_{j+mD} n_{l+mD}] = \sigma^2 \delta_{j,l}$:

$$P_{11}(k) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 \, e^{-4\pi ijk/M}$$

This is the DFT of $w_j^2$ evaluated at frequency $2k$:

$$P_{11}(k) = \sigma^2 \, \widehat{w^2}[2k]$$

where $\widehat{w^2}[n] \equiv \sum_{j=0}^{M-1} w_j^2 e^{-2\pi ijn/M}$.

**Properties of $P_{11}(k)$:**

- At $k = 0$: $P_{11}(0) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2 = C_{11}(0)$. The pseudo-covariance equals the variance; the DC coefficient is purely real.
- At $k = M/2$: $P_{11}(M/2) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 e^{-2\pi ij} = \sigma^2 M S_2 = C_{11}(M/2)$. Same conclusion; the Nyquist coefficient is also purely real.
- For general interior $k$ ($0 < k < M/2$): $P_{11}(k)$ is the DFT of the squared window at bin $2k$. Because $w_j^2$ is a smooth, bell-shaped function symmetric about $j = (M-1)/2$, its DFT concentrates power near $k = 0$. For $k \gg 1$, the DFT falls off rapidly. The Tukey $\alpha = 0.25$ window has a flat central region ($w_j = 1$ for $M_{\mathrm{taper}} \le j < M - M_{\mathrm{taper}}$), so $w_j^2 = w_j$, and the DFT of $w_j^2$ matches the DFT of $w_j$ itself. The main lobe width of $\widehat{w^2}$ is approximately $M / M_{\mathrm{taper}} = 1/(\alpha/2) = 8$ bins.

Therefore, for interior bins well separated from DC and Nyquist ($k \gg 8$ and $M/2 - k \gg 8$), we expect $|P_{11}(k)| \ll C_{11}(k) = \sigma^2 M S_2$. The pseudo-covariance is strongly suppressed but not identically zero.

#### 3.5 Individual Pseudo-Covariance $P_{22}(k)$

By stationarity and the fact that both segments use the identical window:

$$P_{22}(k) = \mathbb{E}[X_{m+1,k}^2] = \sigma^2 \sum_{j=0}^{M-1} w_j^2 \, e^{-4\pi ijk/M} = P_{11}(k)$$

This is exact: $P_{22}(k) = P_{11}(k)$ for all $k$.

---

### 4. Periodogram Covariance and Correlation Conditions

#### 4.1 Complete Isserlis/Wick Identity

Let $Z_1 = X_{m,k}$ and $Z_2 = X_{m+1,k}$. These are jointly zero-mean complex Gaussian. The complete Isserlis (Wick) theorem for complex Gaussians gives:

$$\operatorname{Cov}(|Z_1|^2, |Z_2|^2) = \mathbb{E}[|Z_1|^2 |Z_2|^2] - \mathbb{E}[|Z_1|^2]\mathbb{E}[|Z_2|^2] = |C_{12}|^2 + |P_{12}|^2$$

where $C_{12} = \mathbb{E}[Z_1 Z_2^*]$ is the covariance and $P_{12} = \mathbb{E}[Z_1 Z_2]$ is the pseudo-covariance.

Similarly, for the individual periodogram variance:

$$\operatorname{Var}(|Z_a|^2) = C_{aa}^2 + |P_{aa}|^2$$

**Derivation of the variance identity:** Setting $Z_1 = Z_2 = Z_a$:

$$\mathbb{E}[|Z_a|^4] - (\mathbb{E}[|Z_a|^2])^2 = |\mathbb{E}[Z_a Z_a^*]|^2 + |\mathbb{E}[Z_a^2]|^2 = C_{aa}^2 + |P_{aa}|^2$$

Note that $C_{aa} = \mathbb{E}[|Z_a|^2]$ is real and positive, hence $|C_{aa}|^2 = C_{aa}^2$.

#### 4.2 Full Periodogram Correlation

The normalized periodogram correlation is:

$$\rho_P(k) = \frac{\operatorname{Cov}(|Z_1|^2, |Z_2|^2)}{\sqrt{\operatorname{Var}(|Z_1|^2) \operatorname{Var}(|Z_2|^2)}} = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{\sqrt{\left(C_{11}^2 + |P_{11}(k)|^2\right)\left(C_{22}^2 + |P_{22}(k)|^2\right)}}$$

Since $C_{11} = C_{22}$ and $P_{11} = P_{22}$ (by stationarity and identical windows), this simplifies to:

$$\rho_P(k) = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{C_{11}^2 + |P_{11}(k)|^2}$$

#### 4.3 Proper-Complex Limit and the $r_w^2$ Result

**Definition:** A complex random variable $Z$ is *proper* (circularly symmetric) if $P = \mathbb{E}[Z^2] = 0$.

If the windowed DFT coefficients are proper at bin $k$ — i.e., $P_{11}(k) \approx 0$, $P_{22}(k) \approx 0$, $P_{12}(k) \approx 0$ — then:

$$\rho_P(k) \approx \frac{|C_{12}(k)|^2}{C_{11}^2} = \frac{r_w^2 (\sigma^2 M S_2)^2}{(\sigma^2 M S_2)^2} = r_w^2$$

**This is the result stated in Stage 0.4. It is valid only under the proper-complex assumption.**

For interior bins ($k \gg 8$ and $M/2 - k \gg 8$) with $M = 32{,}768$ and the Tukey $\alpha = 0.25$ window, we expect $|P_{11}(k)| / C_{11} \sim \mathcal{O}(M_{\mathrm{taper}}^{-1})$ or smaller, because the DFT of $w_j^2$ at bin $2k$ (with $2k$ far from $0$ and $M$) is strongly suppressed by the smoothness of $w_j^2$. Therefore, $\rho_P(k) \approx r_w^2$ should be an excellent approximation for the vast majority of analysis-band bins.

**However:**
- Near DC ($k \lesssim 8$) and Nyquist ($k \approx M/2$), pseudo-covariance terms are comparable to covariance terms, and $\rho_P(k) \ne r_w^2$.
- We do not have a proof that $\rho_P = r_w^2$ exactly for all $k$. It is an **approximation** whose accuracy depends on the window shape and bin index.
- The claim must not be stated as an unconditional exact result for the finite Tukey-window implementation.

#### 4.4 Coloured Noise Extension

For coloured noise with autocorrelation $R_n[p] \ne \sigma^2 \delta_{p,0}$, both $C_{12}(k)$ and $P_{12}(k)$ become frequency-dependent in a non-trivial way. The cross-covariance becomes:

$$C_{12}(k) = \sum_{p} R_n[p] \, e^{-2\pi i pk/M} \left(\sum_l w_{l+p+D} \, w_l\right) \cdot (-1)^k$$

The pseudo-covariance acquires a similar spectral convolution structure. Both quantities depend on the noise spectrum, and the periodogram correlation is fully frequency-dependent.

**Implementation and validation requirement (Stage 1.2):** The Monte Carlo pipeline must empirically estimate $\rho_P(k)$ from the actual discrete window and noise-generation pipeline. The comparison must include:

1. The measured adjacent-segment periodogram correlation $\hat{\rho}_P(k)$ across frequency.
2. The analytical prediction from $|C_{12}|^2 / C_{11}^2$ (the proper-complex approximation $r_w^2$).
3. The analytical prediction including pseudo-covariance: $(|C_{12}|^2 + |P_{12}|^2) / (C_{11}^2 + |P_{11}|^2)$.
4. Identification of frequency bins where the pseudo-covariance correction is significant (expected: near DC and Nyquist only).

Do not hard-code $r_w^2$ as the expected exact periodogram correlation.

---

### 5. Filtered, Windowed Coloured-Noise Residual Derivation

#### 5.1 Forward-Backward Filter Transfer Function

The data conditioning pipeline applies an 8th-order Butterworth bandpass filter via `scipy.signal.filtfilt`, which passes the data forward and then backward through the filter. If the one-pass transfer function is $H(f)$, then:

- Forward pass: output spectrum $\propto H(f) \tilde{n}(f)$.
- Backward pass (time-reversed forward filter): equivalent to filtering by $H^*(f)$ in the frequency domain.
- Combined transfer function: $H_{\mathrm{fb}}(f) = H(f) H^*(f) = |H(f)|^2$.

The filtered noise PSD, for a stationary input and away from finite-record boundary effects, is:

$$S_{\mathrm{filtered}}(f) = |H_{\mathrm{fb}}(f)|^2 \, S_n(f) = |H(f)|^4 \, S_n(f)$$

**Explanation of the $|H(f)|^4$ factor:** The forward pass multiplies the amplitude spectrum by $|H(f)|$. The backward pass multiplies the amplitude spectrum by $|H(f)|$ again. The PSD, being proportional to the squared amplitude, acquires a factor $|H(f)|^2$ from each pass, giving $|H(f)|^4$ total. For the 8th-order Butterworth, the effective magnitude roll-off is $|H(f)|^4 \propto f^{-32}$ far outside the passband (equivalent to a 16th-order magnitude response).

#### 5.2 Windowed Filtered Noise Expectation

Following the spectral convolution from Stage 0.4 Section 5, the windowed filtered noise has:

$$\mathbb{E}\left[|\tilde{n}_{w,\mathrm{filtered}}(f)|^2\right] = \frac{1}{2} \int_{-\infty}^{\infty} |\tilde{w}(f - f')|^2 \, S_{\mathrm{filtered}}(|f'|) \, df'$$

Under the **locally flat PSD approximation** (valid when $S_{\mathrm{filtered}}(f)$ varies slowly over the main lobe width $\Delta f_{\mathrm{lobe}} \approx 0.5 \, \mathrm{Hz}$):

$$\mathbb{E}\left[|\tilde{n}_{w,\mathrm{filtered}}(f)|^2\right] \approx \frac{1}{2} S_{\mathrm{filtered}}(f) \, T \, S_2 = \frac{1}{2} |H(f)|^4 \, S_n(f) \, T \, S_2$$

#### 5.3 Updated Normalized Residual Definition

The correctly normalized residual for the filtered, windowed noise at frequency bin $f_k$ is:

$$r(f_k) = \frac{\tilde{d}_{w,\mathrm{filtered}}(f_k) - \tilde{h}(f_k)}{\sqrt{\frac{1}{2} S_{\mathrm{filtered}}(f_k) \, T \, S_2}} = \frac{\tilde{d}_{w,\mathrm{filtered}}(f_k) - \tilde{h}(f_k)}{\sqrt{\frac{1}{2} |H(f_k)|^4 \, S_n(f_k) \, T \, S_2}}$$

Under stationary Gaussian noise with accurate $S_n(f_k)$ and $H(f_k)$, the real and imaginary parts of $r(f_k)$ should each have unit variance, and $\mathbb{E}[|r(f_k)|^2] = 2$ in the two-component (real + imaginary) sense, or equivalently $\mathbb{E}[|r(f_k)|^2 / 2] = 1$ for the mean normalized power per component.

#### 5.4 Exact Processing Order

The synthetic validation protocol (Stage 1.2) must follow this exact processing order:

1. **Generate** stationary coloured Gaussian noise with known PSD $S_n(f)$ (analytical aLIGO ZDHP curve).
2. **Filter** with the same 8th-order forward-backward Butterworth bandpass used in the analysis pipeline.
3. **Trim** the same finite-record buffer ($4 \, \mathrm{s}$ from each end) to remove filter boundary transients.
4. **Window** with the baseline discrete Tukey window ($\alpha = 0.25$, $M = 32{,}768$).
5. **FFT** via `numpy.fft.rfft`.
6. **Normalize** using $S_{\mathrm{filtered}}(f_k) = |H(f_k)|^4 S_n(f_k)$ and $S_2$ computed from the discrete window.

If any variation of this order is used (e.g., filtering after windowing, or different trim lengths), the corresponding expectation must be re-derived.

#### 5.5 Boundary Transient Caveat

The relation $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ assumes stationary, infinite-record filtering. For finite records, the forward-backward filter introduces startup transients at both ends of the segment. These transients are non-stationary and their effect is **not** captured by the transfer-function prediction. The 4-second trimming buffer is designed to exclude these transients, but the adequacy of this buffer is itself a proposed validation (unit impulse test in Stage 1.2), not a guaranteed result.

#### 5.6 Precise Definition of the $\pm 3\%$ Validation Tolerance

The $\pm 3\%$ tolerance applies to the **ensemble-averaged normalized power**, computed as follows:

1. Generate $N_{\mathrm{trials}} = 1{,}000$ independent noise realizations.
2. For each realization, compute the normalized residual $r(f_k)$ at each frequency bin.
3. Compute the ensemble average $\bar{P}(f_k) = \frac{1}{N_{\mathrm{trials}}} \sum_{i=1}^{N_{\mathrm{trials}}} |r_i(f_k)|^2 / 2$.
4. Band-average $\bar{P}(f_k)$ over smooth spectral regions (e.g., 10 Hz bands within $[40, 800] \, \mathrm{Hz}$, excluding known lines).
5. The band-averaged normalized power must satisfy $|\bar{P}_{\mathrm{band}} - 1| < 0.03$.

This tolerance does **not** apply to individual realizations (which have large $\chi^2_2$ fluctuations) or to individual frequency bins near steep spectral features.

---

### 6. Refined Phase-Marginalization Conditions

#### 6.1 Exact Mathematical Quadrature Family (Preserved from Stage 0.4)

The exact Hilbert quadrature construction $\tilde{h}_\perp(f) = -i \operatorname{sgn}(f) \tilde{h}_0(f)$ yields a one-parameter family:

$$h(t; \psi) = h_0(t) \cos\psi + h_\perp(t) \sin\psi$$

For this family, the following properties hold **exactly** (proven in Stage 0.4 Section 6.1):

1. **Norm invariance:** $(h(\psi) | h(\psi)) = \sigma_h^2$ for all $\psi$.
2. **Orthogonality:** $(h_0 | h_\perp) = 0$.
3. **Bessel marginalization:** Under uniform prior $\pi(\psi) = 1/(2\pi)$, the phase-marginalized likelihood evaluates to $I_0(|z|)$ exactly.

#### 6.2 Fundamental Likelihood-Level Condition

The analytical phase marginalization via a single Bessel $I_0$ function requires that the phase-dependent log likelihood can be written as:

$$\ln \mathcal{L}(\phi) = A \cos(m\phi - \alpha) + B$$

where $A \ge 0$, $\alpha$, and $B$ are quantities that do not depend on $\phi$, and $m$ is a positive integer (for the dominant quadrupole mode, $m = 2$).

Under a uniform prior $\pi(\phi) = 1/(2\pi)$, integration gives:

$$\int_0^{2\pi} \exp\left[A \cos(m\phi - \alpha)\right] \frac{d\phi}{2\pi} = I_0(A)$$

This identity follows from the integral representation of the modified Bessel function.

#### 6.3 A Sufficient Construction

A **sufficient** (not necessary) construction ensuring the single-$I_0$ form is a waveform family:

$$h(\phi) = u \cos(m\phi) + v \sin(m\phi)$$

with the **isometry and orthogonality conditions:**

$$(u | u) = (v | v), \qquad (u | v) = 0$$

Under these conditions:

- The template norm $(h(\phi) | h(\phi)) = (u|u) \cos^2(m\phi) + (v|v) \sin^2(m\phi) + 2(u|v) \cos(m\phi) \sin(m\phi) = (u|u)$ is phase-independent.
- The data-template overlap $(d | h(\phi)) = (d|u) \cos(m\phi) + (d|v) \sin(m\phi)$, which is a single sinusoid in $m\phi$.
- Therefore $\ln \mathcal{L}(\phi) = A \cos(m\phi - \alpha) + B$ with $A = \sqrt{(d|u)^2 + (d|v)^2}$.

**This construction is not claimed to be the only possible mathematical structure yielding a single-$I_0$ integral.** Other likelihood structures might also reduce to $I_0$ under different algebraic decompositions. The sufficient-condition framing avoids the overly strong "if and only if" claim.

#### 6.4 Physical Waveform Applicability

For a physical compact binary waveform $h(t; \theta, \phi_c)$ with coalescence orbital phase $\phi_c$:

1. **Dominant quadrupole mode:** Under the stationary-phase approximation, $\tilde{h}(f; \phi_c) \approx \tilde{h}_0(f) e^{-2i\phi_c}$ for $f > 0$. This implies $m = 2$ and the waveform can be written as $h(\phi_c) = u \cos(2\phi_c) + v \sin(2\phi_c)$ with $u$ and $v$ being the Hilbert pair. For the exact frequency-domain quadrature, $(u|u) = (v|v)$ and $(u|v) = 0$ exactly. Therefore, $I_0$ marginalization is exact for the mathematical quadrature family.

2. **Physical approximation quality:** For actual `TaylorF2` waveforms with finite bandwidth $[f_{\min}, f_{\max}]$, the SPA relation $\tilde{h}(f; \phi_c) = \tilde{h}_0(f) e^{-2i\phi_c}$ is an excellent approximation but not exactly the same as the Hilbert quadrature construction. Differences arise from counter-rotating terms, time-domain truncation at ISCO, and windowing. These differences must be quantified.

3. **Breakdown:** Higher harmonics $(\ell, m) \ne (2,2)$ introduce multi-harmonic phase dependence; spin precession introduces amplitude modulation. Neither satisfies the single-sinusoid condition.

#### 6.5 Four-Part Physical Waveform Validation Protocol (Stage 1.4)

Before enabling analytical phase marginalization for any physical waveform approximant, the following four tests must all pass:

| Test | Description | Procedure | Pass Criterion |
|------|-------------|-----------|----------------|
| (a) Phase grid | Generate waveforms over a dense phase grid | Compute $h(\phi_c)$ for $\phi_c \in \{0, \pi/100, 2\pi/100, \dots, 99\pi/100\}$ (100 points) | All waveforms successfully generated |
| (b) Norm variation | Measure phase dependence of template norm | $\Delta_{\mathrm{norm}} = \max_{\phi_c} \frac{|(h(\phi_c)|h(\phi_c)) - \sigma_h^2|}{\sigma_h^2}$ | $\Delta_{\mathrm{norm}} < 10^{-5}$ |
| (c) Overlap phase dependence | Measure phase dependence of data-template overlap | For fixed synthetic data $d$, compute $(d|h(\phi_c))$ over the phase grid; fit to $A\cos(2\phi_c - \alpha) + \epsilon(\phi_c)$. Compute $\max|\epsilon| / A$. | $\max|\epsilon|/A < 10^{-4}$ |
| (d) $I_0$ vs. numerical integration | Decisive comparison | Compute $\ln \mathcal{L}_{\mathrm{marg}}^{I_0}$ via the analytical Bessel formula and $\ln \mathcal{L}_{\mathrm{marg}}^{\mathrm{num}}$ via 1000-point trapezoidal quadrature over $\phi_c \in [0, \pi)$ | $|\ln \mathcal{L}_{\mathrm{marg}}^{I_0} - \ln \mathcal{L}_{\mathrm{marg}}^{\mathrm{num}}| < 10^{-4}$ at multiple SNR values |

**Test (d) is the decisive test.** If the numerical integration disagrees with the $I_0$ formula beyond the tolerance, the analytical marginalization must not be used for that approximant, regardless of what tests (a)–(c) show.

---

### 7. Correct Odd/Even Median Correction Formula

#### 7.1 Derivation Context

For $K$ independent exponentially distributed random variables $Y_1, \dots, Y_K$ with common mean $\mu$, the expected value of the sample median $M_K$ is:

$$\mathbb{E}[M_K] = \mu \, C(K)$$

where $C(K)$ is the finite-sample median correction factor.

#### 7.2 Odd $K = 2L + 1$

When $K$ is odd, the sample median is the $(L+1)$-th order statistic $Y_{(L+1)}$. From Rényi's representation of exponential order statistics:

$$\mathbb{E}[Y_{(L+1)}] = \mu \sum_{j=1}^{L+1} \frac{1}{K + 1 - j} = \mu \sum_{j=L+1}^{2L+1} \frac{1}{j}$$

Therefore:

$$C(K) = \sum_{j=L+1}^{2L+1} \frac{1}{j}, \qquad K = 2L + 1$$

**Equivalence to the alternating harmonic series:** By a known identity, this equals:

$$C(K) = \sum_{j=1}^{K} \frac{(-1)^{j+1}}{j}$$

This equivalence holds **only for odd $K$**.

#### 7.3 Even $K = 2L$

When $K$ is even, the sample median is conventionally defined as the average of the two central order statistics:

$$M_K = \frac{Y_{(L)} + Y_{(L+1)}}{2}$$

The expected value is:

$$\mathbb{E}[M_K] = \frac{\mu}{2}\left[\sum_{j=L}^{2L} \frac{1}{j} + \sum_{j=L+1}^{2L} \frac{1}{j}\right] = \mu \left[\sum_{j=L+1}^{2L} \frac{1}{j} + \frac{1}{2L}\right]$$

**Derivation:** $\mathbb{E}[Y_{(L)}] = \mu \sum_{j=L}^{2L} \frac{1}{j}$ and $\mathbb{E}[Y_{(L+1)}] = \mu \sum_{j=L+1}^{2L} \frac{1}{j}$. Averaging:

$$\mathbb{E}\left[\frac{Y_{(L)} + Y_{(L+1)}}{2}\right] = \frac{\mu}{2}\left(\sum_{j=L}^{2L} \frac{1}{j} + \sum_{j=L+1}^{2L} \frac{1}{j}\right) = \frac{\mu}{2}\left(\frac{1}{L} + 2\sum_{j=L+1}^{2L} \frac{1}{j}\right) = \mu\left(\frac{1}{2L} + \sum_{j=L+1}^{2L} \frac{1}{j}\right)$$

Therefore:

$$C(K) = \sum_{j=L+1}^{2L} \frac{1}{j} + \frac{1}{2L}, \qquad K = 2L$$

The alternating harmonic series $\sum_{j=1}^{K} (-1)^{j+1}/j$ does **not** equal this expression for even $K$.

#### 7.4 Baseline Numerical Values

Both baseline configurations use **odd** $K$:

| Configuration | $K$ | $L$ | Parity | $C(K) = \sum_{j=L+1}^{2L+1} 1/j$ | Numerical Value |
|---------------|-----|-----|--------|-------------------------------------|-----------------|
| On-source PSD ($T_{\mathrm{off}} = 64 \, \mathrm{s}$, $T_{\mathrm{seg}} = 8 \, \mathrm{s}$) | 15 | 7 | Odd | $\sum_{j=8}^{15} 1/j$ | 0.725372 |
| GW150914 PSD ($T_{\mathrm{off}} = 1024 \, \mathrm{s}$, $T_{\mathrm{seg}} = 8 \, \mathrm{s}$) | 255 | 127 | Odd | $\sum_{j=128}^{255} 1/j$ | 0.695104 |

Both values are computed from the odd-$K$ formula. The asymptotic limit $C(K) \to \ln 2 \approx 0.693147$ as $K \to \infty$.

#### 7.5 Applicability Assumptions

The finite-sample correction factor $C(K)$ is an **exact analytical result** derived under the strict assumption of **independent exponential periodograms**. For overlapping Welch segments, the periodograms are correlated (Section 3–4), and $C(K)$ does not fully account for the residual bias and variance. The empirical overlap calibration factor $\kappa_{\mathrm{overlap}}$ will be determined by Monte Carlo simulation in Stage 1.2.

---

### 8. Updated Baseline Parameter Table

```
+---------------------------------------------------------------------------------------------------------------------------------+
|                                    Final Baseline Parameters and Mathematical Conventions                                        |
+--------------------------------------+----------------------------------------------------------------------------------------+
| Parameter / Convention               | Explicit Specification                                                                 |
+--------------------------------------+----------------------------------------------------------------------------------------+
| Analysis Segment Duration            | T_analysis = 8.0 seconds                                                              |
| Sampling Frequency & Time Step       | f_s = 4096 Hz, Delta t = 2.44140625e-4 s (N = 32,768 points)                           |
| Discrete Frequency Resolution        | Delta f = 0.125 Hz                                                                     |
| Active Analysis Bandwidth            | [f_min, f_max] = [30.0, 1024.0] Hz (k_min = 240, k_max = 8192)                         |
| Continuous Fourier Transform         | x_tilde(f) = int x(t) exp(-2*pi*i*f*t) dt                                              |
| Discrete FFT Mapping                 | x_tilde(f_k) approx Delta t * X_k = (1/f_s) * X_k (unnormalized rfft)                   |
| Noise-Weighted Inner Product         | (a|b) = (4/[N*f_s]) Re sum_{k=240}^{8192} A_k* B_k / S_n(f_k) (dimensionless)         |
| Exact Frequency Quadrature           | h_perp_tilde(f) = -i sgn(f) h_0_tilde(f)                                               |
| Phase Extraction Formula             | 2 phi_c_hat = atan2(y, x), y = -Im[z], x = Re[z]                                       |
| On-Source Conditioning Window        | 8.0s Tukey alpha=0.25 (1.0s taper, 6.0s flat top)                                      |
| Window Power Normalization           | S_2 = (1/M) sum w_j^2 (discrete; asymptotic: 27/32 = 0.84375)                          |
| Conditioning Filter                  | 8th-order Butterworth filtfilt; effective |H|^4 magnitude response                      |
| Filtered PSD                         | S_filtered(f) = |H(f)|^4 S_n(f)                                                        |
| Normalized Residual Definition       | r(f_k) = [d_w(f_k) - h(f_k)] / sqrt(0.5 * S_filtered(f_k) * T * S_2)                  |
| Primary Welch PSD Configuration      | T_seg = 8.0s (Delta f = 0.125 Hz), 50% overlap, Tukey alpha=0.25                       |
| Periodogram Correlation (approx.)    | rho_P approx r_w^2 for interior bins (proper-complex approx.); not unconditional exact  |
| Finite-Sample Median Bias: Odd K     | C(K) = sum_{j=L+1}^{2L+1} 1/j  for K=2L+1                                              |
| Finite-Sample Median Bias: Even K    | C(K) = sum_{j=L+1}^{2L} 1/j + 1/(2L)  for K=2L                                         |
| C(15)                                | 0.725372 (odd K, L=7)                                                                   |
| C(255)                               | 0.695104 (odd K, L=127)                                                                 |
| Phase Marginalization Condition      | Sufficient: h(phi)=u cos(m*phi)+v sin(m*phi), (u|u)=(v|v), (u|v)=0                     |
| Phase Marg. Decisive Validation      | I_0 formula vs numerical quadrature comparison (Stage 1.4)                              |
| GW150914 Coalescence GPS Timestamp   | t_c = 1126259462.42 GPS seconds (2015-09-14 09:50:45.42 UTC)                            |
| GW150914 On-Source GPS Segment       | GPS [1126259458.0, 1126259466.0]                                                        |
| GW150914 Off-Source PSD Segment      | GPS [1126258418.0, 1126259442.0] (1024s, K=255 segments)                                |
| GW150914 Guard Interval              | GPS [1126259442.0, 1126259458.0] (16.0s, zero overlap)                                  |
+--------------------------------------+----------------------------------------------------------------------------------------+
```

---

### 9. Updated Validation Matrix

```
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                            Validation Matrix                                                                       |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Component / Domain       | Expected Analytical Property         | Proposed Numerical Test     | Numerical Tolerance   | Failure Interpretation     |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Discrete Parseval Balance| dt sum |x_j|^2 == df dt^2 [|X_0|^2  | Double-precision white      | Relative error        | Incorrect dt, df, or bin   |
| (Stage 1.1)              | + 2 sum |X_k|^2 + |X_Nyq|^2]         | noise time series           | < 1e-12               | weighting in Parseval sum. |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Complex Inner Product    | Conjugate symmetry on C^M:           | Random complex arrays A, B; | Absolute difference   | Missing complex conjugate  |
| (Stage 1.3)              | <A, B>_W == <B, A>_W*                | evaluate <A, B>_W           | < 1e-14               | in first argument.         |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Real Inner Product       | Strict symmetry on R^N:              | Random real arrays a, b;    | Absolute difference   | Asymmetric implementation  |
| (Stage 1.3)              | (a|b) == (b|a); (a|a) > 0 for a!=0  | evaluate (a|b)              | < 1e-14               | or real-part omission.     |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Finite-Sample Median     | Unbiased finite-sample correction:   | 10^4 independent exponential| Relative bias error   | Neglected finite-sample    |
| Welch (Stage 1.2)        | E[M_K / C(K)] == S_n                 | realizations for K=15, 255  | < 1.0%                | correction factor C(K).    |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Pseudo-Covariance        | |P_11(k)| / C_11 suppressed for      | Compute P_11(k), P_12(k)   | Ratio < 0.01 for     | Finite-window properness   |
| Magnitude (Stage 1.2)    | interior bins (k >> 8, M/2-k >> 8)  | from discrete Tukey window; | k in [240, 8192]      | assumption invalid for     |
|                          |                                      | compare with C_11, C_12     |                       | analysis band.             |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Periodogram Correlation  | rho_P(k) approx r_w^2 for interior  | 10^4 white-noise            | |rho_P - r_w^2| < 0.01| Pseudo-covariance or       |
| (Stage 1.2)              | bins under proper-complex approx.    | realizations; measure       | for k in [240, 8192]  | coloured-noise effects     |
|                          |                                      | empirical rho_P(k)          |                       | non-negligible.            |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Overlap Monte Carlo      | Calibrated variance inflation factor | 10^4 Gaussian realizations  | Relative SE approx    | Overlap covariance         |
| (Stage 1.2)              | kappa_overlap(K, alpha)              | with 50% overlap Tukey 0.25 | 1.4% (bootstrap check)| miscalculated.             |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Filtered Residual        | Ensemble mean |r(f_k)|^2/2 = 1      | 1000 coloured-noise         | Band-averaged         | Incorrect S_filtered or    |
| Normalization (Stage 1.2)| under correct S_filtered(f)          | realizations; process via   | |mean - 1| < 0.03     | boundary transient         |
|                          |                                      | filter->trim->window->FFT   | in [40,800] Hz smooth | contamination.             |
|                          |                                      | ->normalize with |H|^4*S_n  | spectral regions      |                            |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Quadrature FFT Sign      | Positive-frequency relation:         | Compute rfft of analytic    | Max relative error    | Sign error; used +i        |
| (Stage 1.4)              | h_perp_tilde(f) == -i * h_0_tilde(f)| sinusoids h_0 and h_perp    | < 1e-12               | instead of -i.             |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Phase Recovery & Branch  | Exact phase extraction:              | Inject monochromatic wave   | Absolute error        | Quadrature sign inversion  |
| (Stage 1.4)              | 2*phi_rec = atan2(y, x)              | with trial phi in [0, pi)   | < 1e-5 (modulo 2pi)  | or atan2 argument swap.    |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Fixed-Time SNR Statistics| Fixed-time SNR follows Rayleigh:     | Filter 10^4 Gaussian noise  | Kolmogorov-Smirnov    | Inappropriate Gaussian     |
| (Stage 1.4)              | p(rho) = rho * exp(-rho^2 / 2)      | realizations at fixed t_0   | p-value > 0.05        | assumption used.           |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Physical Waveform Norm   | Norm variation under phase rotation: | TaylorF2 waveforms over     | Delta_norm < 1e-5     | Physical waveform violates |
| Invariance (Stage 1.4)   | |<h(phi)|h(phi)> - sigma_h^2|/sigma_h^2 100-point phi_c grid in [0,pi)                       | single-harmonic structure. |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Phase Marginalization    | I_0 formula matches numerical        | Compare ln L_marg^{I_0}    | |difference| < 1e-4   | Physical waveform does not |
| I_0 Comparison (Stage1.4)| quadrature over phase                | against 1000-point trap.    | at multiple SNR values| satisfy single-sinusoid    |
|                          |                                      | quadrature over phi_c       |                       | likelihood condition.      |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| TaylorF2 PN Phase        | Frequency derivatives match PN       | Compare pure-Python TaylorF2| Overlap match > 0.999 | Typographical error in     |
| (Stage 1.5)              | expansion coefficients (30-1000 Hz)  | against analytical PN series| on [30, 1000] Hz      | Post-Newtonian coefficients|
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Bessel Phase Marg.       | Exponentially scaled Bessel identity | Evaluate ln I_0(u) via      | Relative difference   | Numerical overflow in I_0  |
| (Stage 1.6)              | ln I_0(u) = u + ln I_0e(u)           | i0e across u in [0, 1000]   | < 1e-12               | or improper scaling.       |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
| Stable Time Marg.        | Log-sum-exp sum over Bessel values   | Compare stable log-sum-exp  | Relative difference   | Improper offset scalar     |
| (Stage 1.6)              | matches numerical integral over t_c  | against fine time-grid quad.| < 1e-4                | used in log-sum-exp.       |
+--------------------------+--------------------------------------+-----------------------------+-----------------------+----------------------------+
```

---

### 10. Remaining Assumptions and Scientific Limitations

1. **GW150914 Off-Source Segment Local Stationarity:** While archival catalog records verify continuous science-mode operation, empirical verification of data-quality channels, glitch absence, and noise stationarity within GPS $[1126258418, 1126259442]$ will occur upon data ingestion in Stage 1.

2. **Empirical Overlap Variance Calibration:** The exact value of $\kappa_{\mathrm{overlap}}$ for $50\%$ overlapping Tukey segments under coloured noise will be calibrated numerically in Stage 1.2 rather than assumed analytically.

3. **Pseudo-Covariance Suppression:** The expectation that $|P_{11}(k)| / C_{11} \ll 1$ for interior analysis-band bins is based on the smoothness of $w_j^2$ and the resulting rapid DFT decay. This has not been numerically verified. Stage 1.2 will compute $P_{11}(k)$ and $P_{12}(k)$ from the actual discrete window and confirm suppression in the band $[30, 1024] \, \mathrm{Hz}$.

4. **Filtered PSD Stationarity:** The relation $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ assumes infinite-record stationary filtering. Boundary transient effects on finite records are a separate validation concern.

5. **Filter Edge Transient Decay:** The $4\text{-second}$ buffer for Butterworth forward-backward filtering will be tested on unit impulses in Stage 1.2 to verify attenuation below $10^{-6}$.

6. **Physical Waveform Phase Marginalization:** The validity of analytical $I_0$ marginalization for `TaylorF2` is an assumption based on the dominant-quadrupole SPA, not a proven result. The four-part validation in Stage 1.4 (especially the decisive $I_0$ vs. numerical integration comparison) is required before enabling this feature.

7. **Physical Waveform Approximant Limits:** Pure-Python `TaylorF2` terminates at $f_{\mathrm{ISCO}}$ and is valid for inspiral testing; high-mass IMR parameter estimation requires `IMRPhenomD` accessed via WSL2.

8. **Bilby $I_0$ Implementation:** The statement that Bilby evaluates phase marginalization via `scipy.special.i0e` reflects published methodology (Ashton et al. 2019). The exact source-code line reference is an unverified implementation expectation to be confirmed in Stage 1.

---

### 11. Final Approval-Readiness Statement

* **Document Generated:** [`Stage_0_5_Final_Approval_Readiness.md`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/Stage_0_5_Final_Approval_Readiness.md)
* **Implementation Status:** **Not started.**
* **Execution Boundary Compliance:**
  * No production source code has been written.
  * No Python packages have been installed on the Windows host or WSL2 subsystem.
  * No strain data have been downloaded from GWOSC.
  * No computational tests have been run.
  * Stage 1.1 has not been initiated.
  * The local workspace contains only design documentation.
* **Corrections Delivered in Stage 0.5:**
  1. Complete pseudo-covariance derivation ($P_{12}$, $P_{11}$, $P_{22}$) with proper-complex qualification.
  2. Filtered residual normalization with $S_{\mathrm{filtered}} = |H|^4 S_n$ and explicit processing order.
  3. Refined phase-marginalization sufficient condition with four-part validation protocol.
  4. Separated odd/even median correction formulas.
  5. Updated all consistency tables.
* **Readiness Assessment:**
  Stage 0.5 addresses all four mathematical and statistical issues identified in the independent review of Stage 0.4. The pseudo-covariance treatment is now complete and properly qualified. The residual normalization accounts for the conditioning filter. The phase-marginalization condition is properly scoped as sufficient rather than biconditional, with a decisive numerical validation protocol. The median correction formula distinguishes odd and even cases.
  **This document is ready for independent scientific review. The independent reviewer will decide whether to approve Stage 1.1.**
