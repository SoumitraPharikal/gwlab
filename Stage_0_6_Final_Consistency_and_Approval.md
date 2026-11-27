# Stage 0.6: Final Consistency and Approval Corrections
## Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals

---

### 1. Executive Summary

This document establishes **Stage 0.6: Final Consistency and Approval Corrections** for the research project *Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals*. It is a narrowly scoped correction round addressing three specific issues identified in the independent scientific review of Stage 0.5.

#### Core Revisions Delivered in Stage 0.6

1. **Residual Normalization Convention (Section 5):** Resolved the inconsistent mixing of complex-power and component-variance conventions. Selected **Option A (unit complex-power)** as the single authoritative convention: $\mathbb{E}[|r|^2] = 1$ with $\operatorname{Var}(\operatorname{Re} r) = \operatorname{Var}(\operatorname{Im} r) = 1/2$ under proper-complex assumptions. Clarified that data and template must undergo identical conditioning operations (filter → trim → window → FFT) for the residual to have the stated noise statistics. Updated the baseline parameter table, validation matrix, and all downstream references for full consistency.

2. **Pseudo-Covariance Fourier-Decay Claims (Section 3):** Removed the unsupported claim that pseudo-covariance sums are "exponentially suppressed" and the unproven $\mathcal{O}(M_{\mathrm{taper}}^{-1})$ scaling estimate. Preserved the exact discrete expressions for $P_{11}(k)$, $P_{12}(k)$, and $P_{22}(k)$. The expected suppression of pseudo-covariance in the analysis band is now stated as a **qualitative expectation and numerical hypothesis** to be verified in Stage 1.2, not an established result.

3. **Butterworth Filter Order and Roll-Off (Section 5):** Distinguished the Butterworth prototype order $N$ from the actual transfer-function order returned by `scipy.signal.butter` for a bandpass design. Removed the specific $f^{-32}$ asymptotic roll-off claim, which conflated prototype order with bandpass implementation order. The relation $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ is retained, with $H(f)$ defined as the actual one-pass transfer function whose order must be confirmed against the SciPy implementation in Stage 1.2.

#### Material Preserved from Stage 0.5

All correct mathematical material from Stage 0.5 is preserved:

- The white-noise cross-covariance derivation $C_{12}(k) = (-1)^k r_w \sigma^2 M S_2$.
- The exact discrete pseudo-covariance expressions for $P_{12}(k)$, $P_{11}(k)$, and $P_{22}(k)$.
- The Isserlis/Wick periodogram covariance identity $\operatorname{Cov}(|Z_1|^2, |Z_2|^2) = |C_{12}|^2 + |P_{12}|^2$.
- The full periodogram correlation $\rho_P$ including pseudo-covariance terms.
- The proper-complex qualification: $\rho_P \approx r_w^2$ is an approximation, not an unconditional exact result.
- The spectral convolution formula for windowed coloured-noise expectations.
- The refined phase-marginalization sufficient condition and four-part validation protocol.
- The separated odd/even median correction formulas.
- All GPS conventions, segment boundaries, and the $8.0$-second analysis baseline.

---

### 2. Stage 0.5 Correction Log

```
+----------------------------------------------------------------------------------------------------------------------------------+
|                                          Record of Stage 0.5 to Stage 0.6 Corrections                                            |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
| No.| Review Issue                     | Defective Formulation in Stage 0.5        | Reconciled Stage 0.6 Resolution               |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
| 1  | Residual Normalization           | Defined r with denominator sqrt(S/2 T S_2)| Selected Option A (unit complex-power):       |
|    | Convention Inconsistency         | but simultaneously claimed Var(Re r) = 1  | E[|r|^2] = 1, Var(Re r) = Var(Im r) = 1/2.   |
|    |                                  | and E[|r|^2] = 2. These are incompatible. | All downstream references updated.            |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
| 2  | Template Conditioning            | Residual written as d_w - h without       | Specified that both data and template undergo  |
|    |                                  | specifying template conditioning.         | identical filter->trim->window->FFT chain.     |
|    |                                  |                                           | Frequency-domain-only templates require        |
|    |                                  |                                           | equivalent treatment, documented separately.   |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
| 3  | Pseudo-Covariance Fourier Decay  | Claimed P_11(k) is "exponentially          | Removed unsupported exponential-decay claim    |
|    |                                  | suppressed" and scaled as                 | and O(M_taper^{-1}) estimate. Retained exact   |
|    |                                  | O(M_taper^{-1}) without proof.            | discrete sums. Suppression stated as           |
|    |                                  |                                           | hypothesis for Stage 1.2 verification.         |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
| 4  | Butterworth Prototype vs.        | Stated f^{-32} asymptotic roll-off        | Distinguished prototype order N from bandpass  |
|    | Bandpass Order                   | without distinguishing prototype order    | transfer-function order. Removed specific      |
|    |                                  | from bandpass implementation order.       | f^{-32} claim. Actual order to be confirmed    |
|    |                                  |                                           | against SciPy implementation in Stage 1.2.     |
+----+----------------------------------+-------------------------------------------+-----------------------------------------------+
```

---

### 3. Complete Covariance and Pseudo-Covariance Derivation

#### 3.1 Setup and Definitions

Let $n_j$ ($j \in \mathbb{Z}$) be a discrete zero-mean stationary real Gaussian noise sequence with autocorrelation $R_n[p] = \mathbb{E}[n_j n_{j+p}]$. We partition the series into segments of length $M$ with shift $D = M/2$ (50% overlap). The windowed DFT coefficients for segment $m$ and adjacent segment $m+1$ are:

$$X_{m,k} = \sum_{j=0}^{M-1} w_j \, n_{j+mD} \, e^{-2\pi i jk/M}$$

$$X_{m+1,k} = \sum_{l=0}^{M-1} w_l \, n_{l+(m+1)D} \, e^{-2\pi i lk/M}$$

where $w_j$ is the real-valued discrete Tukey window ($\alpha = 0.25$, $M = 32{,}768$, $M_{\mathrm{taper}} = \lfloor \alpha M / 2 \rfloor = 4096$).

Because $n_j$ is real and Gaussian, the pair $(X_{m,k}, X_{m+1,k})$ are jointly complex Gaussian with zero mean. Their complete second-order statistics are specified by four quantities:

- **Covariance:** $C_{12}(k) \equiv \mathbb{E}[X_{m,k} \, X_{m+1,k}^*]$
- **Pseudo-covariance (relation):** $P_{12}(k) \equiv \mathbb{E}[X_{m,k} \, X_{m+1,k}]$
- **Individual variance:** $C_{aa}(k) \equiv \mathbb{E}[|X_{a,k}|^2]$ (identical for $a = m$ or $m+1$ by stationarity)
- **Individual pseudo-covariance:** $P_{aa}(k) \equiv \mathbb{E}[X_{a,k}^2]$

**Properness.** A complex Gaussian random variable $Z$ is called **proper** (circularly symmetric) if and only if $\mathbb{E}[Z^2] = 0$, equivalently if $\operatorname{Re}(Z)$ and $\operatorname{Im}(Z)$ have equal variances and zero mutual covariance. Even though the underlying noise $n_j$ is real and stationary, the *finite windowed* DFT coefficients $X_{m,k}$ are not automatically proper for all $k$: properness depends on the window shape and the frequency bin.

#### 3.2 Cross-Covariance $C_{12}(k)$ for White Noise (Preserved from Stage 0.4–0.5)

For white noise with variance $\sigma^2$, $R_n[p] = \sigma^2 \delta_{p,0}$. The derivation yields:

$$C_{12}(k) = \mathbb{E}[X_{m,k} \, X_{m+1,k}^*] = \sigma^2 (-1)^k \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2}$$

The individual variance is:

$$C_{aa}(k) = \mathbb{E}[|X_{a,k}|^2] = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2$$

The normalized cross-correlation coefficient is:

$$\gamma_k = \frac{C_{12}(k)}{C_{aa}(k)} = (-1)^k r_w, \qquad r_w \equiv \frac{\sum_{l=0}^{M/2-1} w_l \, w_{l+M/2}}{\sum_{j=0}^{M-1} w_j^2}$$

**This result is exact for white noise and holds at every frequency bin $k$.**

#### 3.3 Cross Pseudo-Covariance $P_{12}(k)$ for White Noise

$$P_{12}(k) = \mathbb{E}[X_{m,k} \, X_{m+1,k}] = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} w_j \, w_l \, \mathbb{E}[n_{j+mD} \, n_{l+(m+1)D}] \, e^{-2\pi i(j+l)k/M}$$

For white noise, $\mathbb{E}[n_{j+mD} \, n_{l+(m+1)D}] = \sigma^2 \delta_{j, l+D}$, constraining $j = l + M/2$ with $0 \le l \le M/2 - 1$:

$$P_{12}(k) = \sigma^2 \sum_{l=0}^{M/2-1} w_{l+M/2} \, w_l \, e^{-2\pi i(2l + M/2)k/M} = (-1)^k \, \sigma^2 \sum_{l=0}^{M/2-1} w_l \, w_{l+M/2} \, e^{-4\pi i lk/M}$$

Equivalently, substituting $q = l + D$ (so $q$ runs from $D$ to $M - 1$):

$$P_{12}(k) = \sigma^2 \sum_{q=D}^{M-1} w_q \, w_{q-D} \, e^{-2\pi i(2q - D)k/M}$$

**Key observation:** $P_{12}(k)$ involves the same window-overlap product $w_l w_{l+M/2}$ as $C_{12}(k)$, but with an oscillatory DFT kernel $e^{-4\pi ilk/M}$:

- At $k = 0$: $P_{12}(0) = \sigma^2 \sum_{l=0}^{M/2-1} w_l w_{l+M/2} = C_{12}(0)$.
- At $k = M/2$: $e^{-4\pi i l(M/2)/M} = e^{-2\pi i l} = 1$ for all integer $l$, so $P_{12}(M/2) = (-1)^{M/2} C_{12}(0)$.
- For $0 < k < M/2$: $P_{12}(k)$ is a finite oscillatory sum. Its magnitude depends on the specific window shape and the frequency bin $k$; it is **not generically zero**.

#### 3.4 Individual Pseudo-Covariance $P_{11}(k)$

$$P_{11}(k) = \mathbb{E}[X_{m,k}^2] = \sigma^2 \sum_{j=0}^{M-1} w_j^2 \, e^{-4\pi ijk/M}$$

This is the DFT of the sequence $w_j^2$ evaluated at frequency index $2k$:

$$P_{11}(k) = \sigma^2 \, \widehat{w^2}[2k], \qquad \widehat{w^2}[n] \equiv \sum_{j=0}^{M-1} w_j^2 \, e^{-2\pi ijn/M}$$

**Properties of $P_{11}(k)$:**

- At $k = 0$: $P_{11}(0) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2 = C_{11}$. The pseudo-covariance equals the variance.
- At $k = M/2$: $P_{11}(M/2) = \sigma^2 \sum_{j=0}^{M-1} w_j^2 \, e^{-2\pi ij} = \sigma^2 M S_2 = C_{11}$.
- For general interior $k$ ($0 < k < M/2$): $P_{11}(k)$ is the DFT of $w_j^2$ at bin $2k$. Because $w_j^2$ is a smooth, symmetric, bell-shaped function of $j$ concentrated near the center of the segment, its DFT is concentrated near $k = 0$ and we expect $|\widehat{w^2}[2k]|$ to be small for $2k$ well separated from $0$ and $M$. However:
  - The Tukey window is only **piecewise smooth** (the cosine taper sections join the flat central region with continuous first derivatives but discontinuous second derivatives). The rate at which $|\widehat{w^2}[2k]|$ decays with $k$ depends on these smoothness properties and the periodic extension of the finite sequence. We do not claim any specific decay rate (exponential or algebraic) without a mathematical proof or numerical computation.
  - The magnitude $|P_{11}(k)|$ relative to $C_{11}$ at each analysis-band frequency bin is a **quantitative question** that must be resolved numerically from the exact discrete window in Stage 1.2.

**What we can state qualitatively:** The sequence $w_j^2$ varies slowly over the segment (it is unity over the central $75\%$ and smoothly tapered at the edges), so its DFT concentrates energy near $k = 0$. For bins well inside the analysis band ($k_{\min} = 240$ to $k_{\max} = 8192$, corresponding to $30$–$1024 \, \mathrm{Hz}$), the DFT bin index $2k$ ranges from $480$ to $16{,}384$, which is far from both $0$ and $M = 32{,}768$. We therefore **expect** the ratio $|P_{11}(k)| / C_{11}$ to be small in this range, but this remains a **hypothesis** until numerically verified.

#### 3.5 Individual Pseudo-Covariance $P_{22}(k)$

By stationarity and identical windows:

$$P_{22}(k) = P_{11}(k) \quad \text{for all } k$$

This is exact.

#### 3.6 Coloured Noise Extension

For coloured noise with autocorrelation $R_n[p] \ne \sigma^2 \delta_{p,0}$, the cross-covariance becomes:

$$C_{12}(k) = (-1)^k \sum_p R_n[p] \, e^{-2\pi ipk/M} \left(\sum_l w_{l+p+D} \, w_l\right)$$

Both the covariance and pseudo-covariance acquire spectral-convolution structure and are explicitly frequency-dependent. In the coloured-noise case, both $C_{12}(k)$ and $P_{12}(k)$ may need consideration.

---

### 4. Periodogram Covariance and Correlation Conditions

#### 4.1 Complete Isserlis/Wick Identity

Let $Z_1 = X_{m,k}$ and $Z_2 = X_{m+1,k}$. These are jointly zero-mean complex Gaussian. By the Isserlis (Wick) theorem for complex Gaussians:

$$\operatorname{Cov}(|Z_1|^2, |Z_2|^2) = |C_{12}|^2 + |P_{12}|^2$$

For the individual periodogram variance:

$$\operatorname{Var}(|Z_a|^2) = C_{aa}^2 + |P_{aa}|^2$$

**Derivation of the variance identity:** $\mathbb{E}[|Z_a|^4] - (\mathbb{E}[|Z_a|^2])^2 = |\mathbb{E}[Z_a Z_a^*]|^2 + |\mathbb{E}[Z_a^2]|^2 = C_{aa}^2 + |P_{aa}|^2$, where $C_{aa}$ is real and positive.

#### 4.2 Full Periodogram Correlation

$$\rho_P(k) = \frac{|C_{12}(k)|^2 + |P_{12}(k)|^2}{C_{11}^2 + |P_{11}(k)|^2}$$

using $C_{11} = C_{22}$ and $P_{11} = P_{22}$ (stationarity and identical windows).

#### 4.3 Proper-Complex Limit and the $r_w^2$ Result

If the windowed DFT coefficients are proper at bin $k$ — i.e., $P_{11}(k) \approx 0$, $P_{12}(k) \approx 0$ — then:

$$\rho_P(k) \approx \frac{|C_{12}(k)|^2}{C_{11}^2} = r_w^2$$

**This approximation is expected to hold for analysis-band bins** (where the DFT of $w_j^2$ and the oscillatory overlap-product sum are both expected to be small relative to the zero-frequency values), **but this is a hypothesis, not a proven result.** The conditions under which it holds are:

1. The individual pseudo-covariance $|P_{11}(k)|$ is negligible compared to $C_{11}$.
2. The cross pseudo-covariance $|P_{12}(k)|$ is negligible compared to $|C_{12}(k)|$.

Both conditions depend on the specific window shape and the frequency bin. Neither has been numerically verified for the baseline Tukey $\alpha = 0.25$ window.

**Implementation requirement (Stage 1.2):** Empirically estimate $\rho_P(k)$ from the actual discrete window and noise pipeline. The comparison must include:

1. The measured adjacent-segment periodogram correlation $\hat{\rho}_P(k)$ across frequency.
2. The proper-complex prediction $r_w^2$.
3. The full analytical prediction $(|C_{12}|^2 + |P_{12}|^2) / (C_{11}^2 + |P_{11}|^2)$ using the exact discrete sums.
4. Identification of frequency bins where the pseudo-covariance correction is non-negligible.

Do not hard-code $r_w^2$ as the expected exact periodogram correlation.

---

### 5. Filtered, Windowed Coloured-Noise Residual Derivation

#### 5.1 Butterworth Filter Specification

The data conditioning pipeline applies a Butterworth bandpass filter via `scipy.signal.filtfilt` (forward-backward zero-phase filtering).

**Distinction of orders and responses:**

1. **Prototype order $N$:** The parameter passed to `scipy.signal.butter(N, Wn, btype='bandpass')`. For this project, $N = 8$.

2. **Bandpass transfer-function order:** For a bandpass design, `scipy.signal.butter` constructs the filter by transforming a lowpass prototype of order $N$ to a bandpass system. The resulting digital transfer function generally has order $2N$ (i.e., $2N$ poles), because the lowpass-to-bandpass transformation doubles the prototype order. For $N = 8$, this yields a transfer function of order $16$. **However, this must be confirmed against the actual `scipy` implementation**, as implementation details (e.g., second-order sections via `output='sos'`) may affect the representation. The actual transfer-function order and its asymptotic roll-off slope at each band edge are **implementation details to be verified in Stage 1.2**, not assumed from the prototype order alone.

3. **One-pass transfer function $H(f)$:** The frequency response of the filter as returned by `scipy.signal.butter`, evaluated at discrete frequencies. This is a complex-valued function of frequency.

4. **Forward-backward (filtfilt) amplitude response:** $|H_{\mathrm{fb}}(f)| = |H(f)|^2$. The phase response is exactly zero (by construction of `filtfilt`).

5. **PSD response:** For stationary input noise with PSD $S_n(f)$, the output PSD of `filtfilt` is:

$$S_{\mathrm{filtered}}(f) = |H_{\mathrm{fb}}(f)|^2 \, S_n(f) = |H(f)|^4 \, S_n(f)$$

This relation assumes stationary, infinite-record (or sufficiently long-record) filtering. It holds away from finite-record boundary effects.

**Asymptotic roll-off:** The roll-off rate (in dB/octave) of $|H(f)|$ near each band edge depends on the actual transfer-function order at that edge. For the PSD response $|H(f)|^4 S_n(f)$, the roll-off is four times the one-pass amplitude roll-off (in dB). The specific numerical roll-off rates must be confirmed against the implemented SciPy filter design in Stage 1.2, rather than inferred from the prototype order $N = 8$ alone.

#### 5.2 Windowed Filtered Noise Expectation

Following the spectral convolution from Stage 0.4 Section 5, the windowed filtered noise has:

$$\mathbb{E}\left[|\tilde{n}_{w,\mathrm{filtered}}(f)|^2\right] = \frac{1}{2} \int_{-\infty}^{\infty} |\tilde{w}(f - f')|^2 \, S_{\mathrm{filtered}}(|f'|) \, df'$$

This is **exact** (under the assumption that the noise is stationary after filtering and that the windowed segment is well within the filter's settled response).

Under the **locally flat PSD approximation** — valid when $S_{\mathrm{filtered}}(f)$ varies slowly over the main lobe width of $|\tilde{w}|^2$ ($\Delta f_{\mathrm{lobe}} \approx 1/(\alpha T) = 0.5 \, \mathrm{Hz}$) — this simplifies to:

$$\mathbb{E}\left[|\tilde{n}_{w,\mathrm{filtered}}(f)|^2\right] \approx \frac{1}{2} \, S_{\mathrm{filtered}}(f) \, T \, S_2$$

#### 5.3 Template and Data Conditioning

For the residual to have the expected noise statistics, the data and the template must undergo **identical conditioning operations**. The residual numerator is defined as:

$$\tilde{r}_{\mathrm{num}}(f_k) = \tilde{d}_{w,\mathrm{filtered}}(f_k) - \tilde{h}_{w,\mathrm{filtered}}(f_k)$$

where both $d$ and $h$ have been processed through the same chain:

1. **Time-domain signal** — raw detector strain $d(t)$ or time-domain template waveform $h(t; \theta)$.
2. **Forward-backward filter** — the identical `filtfilt` call with the same Butterworth coefficients.
3. **Trim** — discard the same 4-second buffer from each end to excise filter boundary transients.
4. **Window** — apply the identical discrete Tukey window ($\alpha = 0.25$, $M = 32{,}768$).
5. **FFT** — compute `numpy.fft.rfft` on both windowed, filtered, trimmed signals.

If the template waveform is generated directly in the frequency domain (e.g., `TaylorF2` evaluated as $\tilde{h}(f)$), it must first be transformed to the time domain via inverse FFT before entering step 2, or an equivalent frequency-domain conditioning pipeline must be designed and its normalization derived separately. In the latter case, the effective frequency-domain operation is multiplication by $|H(f)|^2$ (the filtfilt transfer function) followed by convolution with $\tilde{w}(f)$ (the windowing), which is a non-trivial operation. The time-domain route is simpler and recommended for the initial implementation.

**Under correct conditioning**, the residual $\tilde{r}_{\mathrm{num}}(f_k) = \tilde{n}_{w,\mathrm{filtered}}(f_k)$ is the windowed, filtered noise, and its statistics are governed by the derivation above.

#### 5.4 Normalized Residual: Option A (Unit Complex-Power Convention)

**Selected convention.** We adopt the unit complex-power normalization:

$$r(f_k) = \frac{\tilde{r}_{\mathrm{num}}(f_k)}{\sqrt{\frac{1}{2} \, S_{\mathrm{filtered}}(f_k) \, T \, S_2}}$$

where $S_{\mathrm{filtered}}(f_k) = |H(f_k)|^4 \, S_n(f_k)$.

**Statistical properties under proper-complex Gaussian assumptions:**

Under the locally flat PSD approximation and if $r(f_k)$ is proper-complex Gaussian at bin $k$:

$$\mathbb{E}[|r(f_k)|^2] = 1$$

$$\operatorname{Var}(\operatorname{Re} r(f_k)) = \operatorname{Var}(\operatorname{Im} r(f_k)) = \frac{1}{2}$$

$$\operatorname{Cov}(\operatorname{Re} r(f_k), \operatorname{Im} r(f_k)) = 0$$

**Justification of the $\mathbb{E}[|r|^2] = 1$ value:** The denominator is $\sqrt{\mathbb{E}[|\tilde{n}_{w,\mathrm{filtered}}(f_k)|^2]}$ under the locally flat approximation, so by construction the normalized squared modulus has unit expectation.

**Assumptions required for equal-variance, uncorrelated real/imaginary parts:**

The real and imaginary parts of $r(f_k)$ have equal variance ($1/2$ each) and are uncorrelated if and only if $r(f_k)$ is proper, i.e., $\mathbb{E}[r(f_k)^2] = 0$. This requires that the underlying windowed DFT coefficient $X_k$ is proper: $P_{aa}(k) = \mathbb{E}[X_k^2] = 0$. As derived in Section 3.4, $P_{aa}(k) = \sigma^2 \widehat{w^2}[2k]$, which is not identically zero for all $k$. Therefore:

- **For analysis-band bins where $|P_{aa}(k)| \ll C_{aa}$:** The proper-complex approximation holds, and the real/imaginary variances are approximately $1/2$ each. Whether this condition is satisfied for all bins in $[30, 1024] \, \mathrm{Hz}$ is a hypothesis to be verified in Stage 1.2.
- **For bins near DC or Nyquist:** The coefficient is not proper ($P_{aa} \approx C_{aa}$), and the real and imaginary parts do not have equal variance. These bins lie outside the analysis band.
- **For an individual realization:** The value $|r(f_k)|^2$ is a single draw from a $\chi^2$-like distribution with considerable variance. The unit-mean property $\mathbb{E}[|r|^2] = 1$ is an **ensemble** statement, not a per-realization guarantee.

**Explicit statement of what the ±3% validation tests:** The tolerance applies to the **ensemble-averaged normalized power** computed as:

1. Generate $N_{\mathrm{trials}} = 1{,}000$ independent noise realizations.
2. For each realization $i$, process through filter → trim → window → FFT → normalize to obtain $r_i(f_k)$.
3. Compute the ensemble average $\bar{P}(f_k) = \frac{1}{N_{\mathrm{trials}}} \sum_{i=1}^{N_{\mathrm{trials}}} |r_i(f_k)|^2$.
4. Band-average $\bar{P}(f_k)$ over smooth spectral regions (e.g., 10 Hz bands within $[40, 800] \, \mathrm{Hz}$, excluding known spectral lines and steep gradients).
5. The band-averaged value must satisfy $|\bar{P}_{\mathrm{band}} - 1| < 0.03$.

This tolerance applies to the ensemble-and-band-averaged quantity, not to individual realizations or individual frequency bins.

#### 5.5 Exact Processing Order for Validation

The synthetic validation protocol (Stage 1.2) must follow this exact order:

1. **Generate** stationary coloured Gaussian noise with known PSD $S_n(f)$ (analytical aLIGO ZDHP curve).
2. **Filter** with the same forward-backward Butterworth bandpass used in the analysis pipeline.
3. **Trim** the same 4-second buffer from each end.
4. **Window** with the baseline discrete Tukey window ($\alpha = 0.25$, $M = 32{,}768$).
5. **FFT** via `numpy.fft.rfft`.
6. **Normalize** using $S_{\mathrm{filtered}}(f_k) = |H(f_k)|^4 \, S_n(f_k)$ and $S_2$ computed from the discrete window.
7. **Verify** $\bar{P}(f_k) \approx 1$ across the analysis band (ensemble average).

If any variation of this order is used, the corresponding expectation must be re-derived.

#### 5.6 Boundary Transient Caveat

The relation $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ assumes stationary, infinite-record filtering. For finite records, the forward-backward filter introduces startup transients at both ends of the segment. These transients are non-stationary and are **not** captured by the transfer-function prediction. The 4-second trimming buffer is designed to exclude these transients, but the adequacy of this buffer is itself a **proposed validation** (unit impulse test in Stage 1.2), not an established result.

---

### 6. Refined Phase-Marginalization Conditions

#### 6.1 Exact Mathematical Quadrature Family (Preserved)

The exact Hilbert quadrature construction $\tilde{h}_\perp(f) = -i \operatorname{sgn}(f) \tilde{h}_0(f)$ yields a one-parameter family:

$$h(t; \psi) = h_0(t) \cos\psi + h_\perp(t) \sin\psi$$

For this family, the following properties hold **exactly** (proven in Stage 0.4 Section 6.1):

1. **Norm invariance:** $(h(\psi) | h(\psi)) = \sigma_h^2$ for all $\psi$.
2. **Orthogonality:** $(h_0 | h_\perp) = 0$.
3. **Bessel marginalization:** Under uniform prior $\pi(\psi) = 1/(2\pi)$, the phase-marginalized likelihood evaluates to $I_0(|z|)$ exactly.

#### 6.2 Fundamental Likelihood-Level Condition

Analytical phase marginalization via a single Bessel $I_0$ function requires that the phase-dependent log likelihood can be written as:

$$\ln \mathcal{L}(\phi) = A \cos(m\phi - \alpha) + B$$

where $A \ge 0$, $\alpha$, and $B$ do not depend on $\phi$, and $m$ is a positive integer.

Under uniform prior $\pi(\phi) = 1/(2\pi)$:

$$\int_0^{2\pi} \exp\left[A \cos(m\phi - \alpha)\right] \frac{d\phi}{2\pi} = I_0(A)$$

#### 6.3 A Sufficient Construction

A **sufficient** (not necessary) construction ensuring the single-$I_0$ form is:

$$h(\phi) = u \cos(m\phi) + v \sin(m\phi)$$

with the isometry and orthogonality conditions:

$$(u | u) = (v | v), \qquad (u | v) = 0$$

Under these conditions:

- The template norm $(h(\phi)|h(\phi)) = (u|u)$ is phase-independent.
- The data-template overlap $(d|h(\phi)) = (d|u)\cos(m\phi) + (d|v)\sin(m\phi)$ is a single sinusoid in $m\phi$.
- Therefore $\ln \mathcal{L}(\phi) = A\cos(m\phi - \alpha) + B$ with $A = \sqrt{(d|u)^2 + (d|v)^2}$.

**This construction is not claimed to be the only possible mathematical structure yielding a single-$I_0$ integral.**

#### 6.4 Physical Waveform Applicability

For a physical compact binary waveform $h(t; \theta, \phi_c)$ with coalescence orbital phase $\phi_c$:

1. **Dominant quadrupole mode:** Under the stationary-phase approximation, $\tilde{h}(f; \phi_c) \approx \tilde{h}_0(f) e^{-2i\phi_c}$ for $f > 0$, giving $m = 2$.

2. **Approximation quality:** For `TaylorF2` waveforms with finite bandwidth, the SPA relation is an excellent approximation but not identical to the exact Hilbert quadrature construction. Differences arise from counter-rotating terms, time-domain truncation at ISCO, and conditioning.

3. **Breakdown:** Higher harmonics $(\ell, m) \ne (2,2)$ and spin precession violate the single-sinusoid condition.

#### 6.5 Four-Part Physical Waveform Validation Protocol (Stage 1.4)

| Test | Description | Procedure | Pass Criterion |
|------|-------------|-----------|----------------|
| (a) Phase grid | Generate waveforms over dense phase grid | $h(\phi_c)$ for 100 values in $[0, \pi)$ | All waveforms generated |
| (b) Norm variation | Measure $\Delta_{\mathrm{norm}} = \max_{\phi_c} |(h(\phi_c)|h(\phi_c)) - \sigma_h^2| / \sigma_h^2$ | Evaluate inner product at each phase | $\Delta_{\mathrm{norm}} < 10^{-5}$ |
| (c) Overlap phase dependence | Fit $(d|h(\phi_c))$ to $A\cos(2\phi_c - \alpha) + \epsilon(\phi_c)$ | Fixed synthetic $d$; compute residual $\epsilon$ | $\max|\epsilon|/A < 10^{-4}$ |
| (d) $I_0$ vs. numerical integration (**decisive**) | Compare $\ln \mathcal{L}_{\mathrm{marg}}^{I_0}$ vs. $\ln \mathcal{L}_{\mathrm{marg}}^{\mathrm{num}}$ | 1000-point trapezoidal quadrature over $\phi_c \in [0, \pi)$ | $|\text{difference}| < 10^{-4}$ at multiple SNR |

**Test (d) is the decisive test.** If the numerical integration disagrees with the $I_0$ formula beyond tolerance, analytical marginalization must not be used for that approximant.

---

### 7. Correct Odd/Even Median Correction Formula

#### 7.1 Odd $K = 2L + 1$

$$C(K) = \sum_{j=L+1}^{2L+1} \frac{1}{j}, \qquad K = 2L + 1$$

Equivalent to the alternating harmonic series $\sum_{j=1}^{K} (-1)^{j+1}/j$. **This equivalence holds only for odd $K$.**

#### 7.2 Even $K = 2L$

$$C(K) = \sum_{j=L+1}^{2L} \frac{1}{j} + \frac{1}{2L}, \qquad K = 2L$$

The alternating harmonic series does **not** equal this expression for even $K$.

#### 7.3 Baseline Numerical Values

| Configuration | $K$ | $L$ | Parity | Formula | $C(K)$ |
|---|---|---|---|---|---|
| Synthetic PSD ($T_{\mathrm{off}} = 64 \, \mathrm{s}$, $T_{\mathrm{seg}} = 8 \, \mathrm{s}$) | 15 | 7 | Odd | $\sum_{j=8}^{15} 1/j$ | 0.725372 |
| GW150914 PSD ($T_{\mathrm{off}} = 1024 \, \mathrm{s}$, $T_{\mathrm{seg}} = 8 \, \mathrm{s}$) | 255 | 127 | Odd | $\sum_{j=128}^{255} 1/j$ | 0.695104 |

Asymptotic limit: $C(K) \to \ln 2 \approx 0.693147$ as $K \to \infty$.

#### 7.4 Applicability

$C(K)$ is exact under the strict assumption of **independent exponential periodograms**. For overlapping Welch segments, periodograms are correlated (Section 3–4), and $C(K)$ does not fully capture the residual bias and variance. The empirical overlap calibration factor $\kappa_{\mathrm{overlap}}$ will be determined by Monte Carlo simulation in Stage 1.2.

---

### 8. Updated Baseline Parameter Table

```
+----------------------------------------------------------------------------------------------------------------------------------+
|                                    Final Baseline Parameters and Mathematical Conventions                                         |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| Parameter / Convention               | Explicit Specification                                                                  |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| Analysis Segment Duration            | T = 8.0 seconds                                                                        |
| Sampling Frequency & Time Step       | f_s = 4096 Hz, dt = 1/f_s = 2.44140625e-4 s, N = 32768 points                          |
| Discrete Frequency Resolution        | df = 0.125 Hz                                                                           |
| Active Analysis Bandwidth            | [f_min, f_max] = [30.0, 1024.0] Hz (k_min=240, k_max=8192)                              |
| Continuous Fourier Transform         | x_tilde(f) = int x(t) exp(-2*pi*i*f*t) dt                                               |
| Discrete FFT Mapping                 | x_tilde(f_k) approx dt * X_k = (1/f_s)*X_k  (unnormalized rfft)                         |
| Noise-Weighted Inner Product         | (a|b) = (4/[N*f_s]) Re sum_{k=240}^{8192} A_k* B_k / S_n(f_k)  (dimensionless)         |
| Exact Frequency Quadrature           | h_perp_tilde(f) = -i sgn(f) h_0_tilde(f)                                                |
| Phase Extraction Formula             | 2 phi_c_hat = atan2(y, x),  y = -Im[z],  x = Re[z]                                     |
| On-Source Conditioning Window        | 8.0s Tukey alpha=0.25 (1.0s taper, 6.0s flat top), M=32768                              |
| Window Power Normalization           | S_2 = (1/M) sum w_j^2  (discrete; asymptotic: 27/32 = 0.84375)                         |
| Conditioning Filter                  | Butterworth bandpass, prototype order N=8, applied via filtfilt                          |
|                                      | Actual transfer-function order to be confirmed against SciPy implementation              |
|                                      | One-pass transfer function: H(f); filtfilt response: |H(f)|^2                           |
| Filtered PSD                         | S_filtered(f) = |H(f)|^4 S_n(f)  (H = one-pass transfer function)                       |
| Residual Convention                  | Option A (unit complex-power): r = numerator / sqrt(S_filtered*T*S_2 / 2)               |
|                                      | E[|r|^2] = 1; Var(Re r) = Var(Im r) = 1/2 (under proper-complex approx.)               |
| Residual Numerator                   | r_num(f_k) = d_w_filtered(f_k) - h_w_filtered(f_k)                                      |
|                                      | Data and template undergo identical conditioning: filter->trim->window->FFT              |
| Residual Validation Tolerance        | Ensemble-and-band-averaged |r|^2: |mean - 1| < 0.03 in [40,800] Hz smooth regions       |
| Primary Welch PSD Configuration      | T_seg=8.0s (df=0.125 Hz), 50% overlap, Tukey alpha=0.25                                 |
| Periodogram Correlation              | rho_P approx r_w^2 for interior bins (proper-complex approx.; not unconditional exact)  |
| Pseudo-Covariance Suppression        | |P_11(k)|/C_11 expected small in analysis band; hypothesis, not proven                   |
| Median Bias: Odd K=2L+1              | C(K) = sum_{j=L+1}^{2L+1} 1/j                                                          |
| Median Bias: Even K=2L               | C(K) = sum_{j=L+1}^{2L} 1/j + 1/(2L)                                                   |
| C(15)                                | 0.725372 (odd, L=7)                                                                     |
| C(255)                               | 0.695104 (odd, L=127)                                                                   |
| Phase Marginalization                | Sufficient condition: h(phi)=u cos(m phi)+v sin(m phi), (u|u)=(v|v), (u|v)=0            |
| Phase Marg. Decisive Validation      | I_0 formula vs numerical quadrature comparison (Stage 1.4)                               |
| GW150914 Coalescence GPS Timestamp   | t_c = 1126259462.42 GPS seconds (2015-09-14 09:50:45.42 UTC)                             |
| GW150914 On-Source GPS Segment       | [1126259458.0, 1126259466.0]                                                             |
| GW150914 Off-Source PSD Segment      | [1126258418.0, 1126259442.0] (1024s, K=255 segments)                                    |
| GW150914 Guard Interval              | [1126259442.0, 1126259458.0] (16.0s, zero overlap)                                      |
+--------------------------------------+-----------------------------------------------------------------------------------------+
```

---

### 9. Updated Validation Matrix

```
+-----------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                              Validation Matrix                                                                       |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Component / Domain         | Expected Analytical Property         | Proposed Numerical Test       | Numerical Tolerance   | Failure Interpretation     |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Discrete Parseval Balance  | dt sum |x_j|^2 == df dt^2            | Double-precision white-noise  | Relative error        | Incorrect dt, df, or bin   |
| (Stage 1.1)                | [|X_0|^2 + 2 sum |X_k|^2 + |X_Nyq|^2]| time series                  | < 1e-12               | weighting.                 |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Complex Inner Product      | <A, B>_W == <B, A>_W*                | Random complex arrays A, B    | Absolute difference   | Missing conjugate in       |
| (Stage 1.3)                |                                      |                               | < 1e-14               | first argument.            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Real Inner Product         | (a|b)==(b|a); (a|a)>0 for a!=0       | Random real arrays a, b       | Absolute difference   | Asymmetric implementation  |
| (Stage 1.3)                |                                      |                               | < 1e-14               | or real-part omission.     |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Finite-Sample Median Welch | E[M_K / C(K)] == S_n                 | 10^4 independent exponential  | Relative bias         | Neglected C(K) or wrong    |
| (Stage 1.2)                |                                      | realizations, K=15, K=255     | < 1.0%                | parity formula.            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Pseudo-Covariance Magnitude| |P_11(k)|/C_11 and |P_12(k)|/|C_12| | Compute exact discrete sums   | To be determined      | Proper-complex approx.     |
| (Stage 1.2)                | expected small in analysis band      | from implemented Tukey window;| empirically from      | invalid; rho_P != r_w^2.  |
|                            | (hypothesis, not proven)             | compare with C_11, C_12       | discrete window       |                            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Periodogram Correlation    | rho_P(k) approx r_w^2 for interior  | 10^4 white-noise realizations;| |rho_P - r_w^2| < 0.01| Pseudo-covariance or       |
| (Stage 1.2)                | bins (proper-complex approx.)        | measure empirical rho_P(k);   | for k in [240, 8192]  | coloured-noise effects     |
|                            |                                      | compare analytical prediction |                       | non-negligible.            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Overlap Monte Carlo        | Calibrated kappa_overlap(K, alpha)   | 10^4 Gaussian realizations    | Relative SE approx    | Overlap covariance         |
| (Stage 1.2)                |                                      | with 50% overlap Tukey 0.25   | 1.4% (bootstrap)      | miscalculated.             |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Butterworth Transfer Fxn   | Actual order and roll-off match      | Evaluate scipy.signal.butter  | Confirmed order and   | Prototype-to-bandpass      |
| Verification (Stage 1.2)   | design specification                 | output; compute |H(f)|        | roll-off documented   | order mismatch.            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Filtered Residual          | Ensemble mean |r(f_k)|^2 = 1        | 1000 coloured-noise           | Band-averaged         | Incorrect S_filtered,      |
| Normalization (Stage 1.2)  | (Option A: unit complex-power)       | realizations; process via     | |mean - 1| < 0.03     | wrong convention, or       |
|                            |                                      | filter->trim->window->FFT    | in [40,800] Hz smooth | boundary transient         |
|                            |                                      | ->normalize with |H|^4*S_n   | spectral regions      | contamination.             |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Quadrature FFT Sign        | h_perp(f) == -i * h_0(f)  (f>0)     | rfft of analytic sinusoids    | Max relative error    | Sign error in quadrature.  |
| (Stage 1.4)                |                                      |                               | < 1e-12               |                            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Phase Recovery & Branch    | 2 phi_rec = atan2(y, x)              | Inject monochromatic wave     | Absolute error        | atan2 argument swap or     |
| (Stage 1.4)                |                                      | with trial phi in [0, pi)     | < 1e-5 (mod 2pi)     | quadrature sign error.     |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Fixed-Time SNR Statistics  | p(rho) = rho exp(-rho^2/2)           | 10^4 Gaussian noise           | KS p-value > 0.05     | Wrong noise distribution   |
| (Stage 1.4)                | (Rayleigh)                           | realizations at fixed t_0     |                       | assumption.                |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Physical Waveform Norm     | Delta_norm < 1e-5 across phase grid  | TaylorF2 over 100-point       | Delta_norm < 1e-5     | Waveform violates single-  |
| Invariance (Stage 1.4)     |                                      | phi_c grid in [0, pi)         |                       | harmonic structure.        |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Phase Marg. I_0 vs. Num.   | I_0 formula matches numerical        | ln L_marg^{I_0} vs 1000-pt   | |difference| < 1e-4   | Physical waveform fails    |
| Integration (Stage 1.4)    | quadrature (DECISIVE TEST)           | trapezoidal over phi_c        | at multiple SNR       | single-sinusoid condition. |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| TaylorF2 PN Phase          | PN coefficients correct on [30,1000] | Compare pure-Python TaylorF2  | Overlap > 0.999       | PN coefficient error.      |
| (Stage 1.5)                |                                      | against analytical PN series  |                       |                            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Bessel Phase Marg.         | ln I_0(u) = u + ln I_0e(u)           | Evaluate via i0e across       | Relative diff < 1e-12 | Overflow or scaling error. |
| (Stage 1.6)                |                                      | u in [0, 1000]                |                       |                            |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
| Stable Time Marg.          | Log-sum-exp matches numerical        | Compare against fine time-    | Relative diff < 1e-4  | Improper offset in         |
| (Stage 1.6)                | integral over t_c                    | grid quadrature               |                       | log-sum-exp.               |
+----------------------------+--------------------------------------+-------------------------------+-----------------------+----------------------------+
```

---

### 10. Remaining Assumptions and Scientific Limitations

1. **GW150914 Off-Source Segment Local Stationarity:** Empirical verification of data-quality channels, glitch absence, and noise stationarity within GPS $[1126258418, 1126259442]$ will occur upon data ingestion in Stage 1.

2. **Empirical Overlap Variance Calibration:** The exact value of $\kappa_{\mathrm{overlap}}$ for $50\%$ overlapping Tukey segments under coloured noise will be calibrated numerically in Stage 1.2 rather than assumed analytically.

3. **Pseudo-Covariance Suppression in Analysis Band:** The expectation that $|P_{11}(k)| / C_{11} \ll 1$ and $|P_{12}(k)| / |C_{12}(k)| \ll 1$ for all analysis-band bins ($k \in [240, 8192]$) is a **qualitative hypothesis** based on the slowly varying nature of $w_j^2$. No specific decay rate has been proven. Stage 1.2 will compute the exact discrete sums and determine the actual magnitudes.

4. **Proper-Complex Approximation for Periodogram Correlation:** The result $\rho_P \approx r_w^2$ depends on the pseudo-covariance terms being negligible. This is expected for analysis-band interior bins but has not been verified. Stage 1.2 will test this.

5. **Butterworth Transfer-Function Order:** The actual transfer-function order and asymptotic roll-off of the SciPy bandpass Butterworth design (prototype order $N = 8$) have not been verified. The expected bandpass order is $2N = 16$, but this must be confirmed against the implemented `scipy.signal.butter` output in Stage 1.2.

6. **Filtered PSD Stationarity:** $S_{\mathrm{filtered}}(f) = |H(f)|^4 S_n(f)$ assumes infinite-record stationary filtering. Boundary transient effects on finite records are a separate validation concern.

7. **Filter Edge Transient Decay:** The 4-second trimming buffer adequacy will be tested on unit impulses in Stage 1.2 (target: attenuation below $10^{-6}$).

8. **Locally Flat PSD Approximation:** The simplification $\mathbb{E}[|\tilde{n}_w|^2] \approx \frac{1}{2} S_{\mathrm{filtered}} T S_2$ breaks down near narrow spectral features and steep spectral gradients. The $\pm 3\%$ tolerance applies only to smooth spectral regions.

9. **Physical Waveform Phase Marginalization:** The validity of analytical $I_0$ marginalization for `TaylorF2` is an assumption based on the dominant-quadrupole SPA. The four-part validation (Stage 1.4), especially the decisive $I_0$ vs. numerical integration comparison, is required before enabling this feature.

10. **Template Conditioning for Frequency-Domain Waveforms:** If `TaylorF2` is generated directly in the frequency domain, an equivalent frequency-domain conditioning procedure must be designed and validated to ensure the residual normalization remains correct. The recommended initial approach is time-domain generation followed by the same conditioning chain as the data.

11. **Physical Waveform Approximant Limits:** Pure-Python `TaylorF2` terminates at $f_{\mathrm{ISCO}}$; high-mass IMR parameter estimation requires `IMRPhenomD` via WSL2.

12. **Bilby $I_0$ Implementation:** The claim that Bilby uses `scipy.special.i0e` for phase marginalization (Ashton et al. 2019) is an unverified implementation expectation to be confirmed in Stage 1.

---

### 11. Final Approval-Readiness Statement

* **Document:** [`Stage_0_6_Final_Consistency_and_Approval.md`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/Stage_0_6_Final_Consistency_and_Approval.md)
* **Implementation Status:** **Not started.**
* **Execution Boundary Compliance:**
  * No production source code has been written.
  * No Python packages have been installed on the Windows host or WSL2 subsystem.
  * No strain data have been downloaded from GWOSC.
  * No computational tests have been run.
  * Stage 1.1 has not been initiated.
  * The local workspace contains only design documentation.
* **Corrections Delivered in Stage 0.6:**
  1. Resolved residual normalization inconsistency: selected Option A (unit complex-power, $\mathbb{E}[|r|^2] = 1$), with explicit assumptions and consistent downstream references.
  2. Specified template conditioning: both data and template must undergo identical filter → trim → window → FFT chain.
  3. Removed unsupported exponential-decay and $\mathcal{O}(M_{\mathrm{taper}}^{-1})$ claims for pseudo-covariance sums; stated suppression as a numerical hypothesis.
  4. Distinguished Butterworth prototype order from bandpass transfer-function order; removed specific $f^{-32}$ roll-off claim; flagged actual order for implementation-stage verification.
  5. Performed full consistency sweep: all tables, tolerances, and validation criteria now reference the unit complex-power convention with $\mathbb{E}[|r|^2] = 1$.
* **Remaining Assumptions Requiring Implementation-Stage Validation:**
  1. Pseudo-covariance suppression in the analysis band (Stage 1.2).
  2. $\rho_P \approx r_w^2$ for analysis-band bins (Stage 1.2).
  3. Butterworth actual transfer-function order (Stage 1.2).
  4. Filter boundary transient decay within 4-second buffer (Stage 1.2).
  5. Locally flat PSD approximation accuracy in smooth spectral regions (Stage 1.2).
  6. Physical waveform $I_0$ marginalization validity (Stage 1.4).
  7. Frequency-domain template conditioning equivalence (Stage 1.2–1.4).
* **Readiness Assessment:**
  This document is ready for independent scientific review. The independent reviewer will decide whether to approve Stage 1.1.
