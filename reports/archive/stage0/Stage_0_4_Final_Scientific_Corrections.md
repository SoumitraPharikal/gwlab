# Stage 0.4: Final Scientific Corrections and Approval Readiness
## Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals

---

### 1. Executive Summary

This document establishes **Stage 0.4: Final Scientific Corrections and Approval Readiness** for the research project *Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals*. It rigorously addresses the five remaining mathematical and statistical issues raised in the independent scientific review of Stage 0.3.

#### Core Revisions Delivered in Stage 0.4
1. **Frequency-Dependent Overlap Covariance:** Derived the exact cross-covariance of windowed discrete Fourier transform coefficients between adjacent $50\%$-overlapping segments. Proved that for white noise, the covariance has alternating phase $(-1)^k r_w$, where $r_w$ is the window-overlap product sum. For coloured noise, the covariance is explicitly frequency-dependent due to convolution with the noise spectrum. Withdrew all claims of a universal analytical variance-inflation formula for the median-Welch estimator, establishing numerical Monte Carlo calibration on the implemented pipeline as the only defensible methodology.
2. **Exact Finite Discrete Tukey Window Quantities:** Distinguished continuous-window integrals, asymptotic limits, and exact finite discrete sums for $M = 32,768$ samples. Mandated that the implementation calculate $S_2$ and $r_w$ directly from discrete window samples rather than hard-coding continuous approximations.
3. **Coloured-Noise Windowed Residual Normalization:** Derived the exact frequency-domain expectation of windowed noise residuals via spectral convolution with the window energy transform $|\tilde{w}(f)|^2$. Formalized the locally flat PSD approximation that yields the simplified $S_2$ factor and specified synthetic validation to quantify deviations near lines and steep spectral gradients.
4. **Strict Separation of Exact Quadrature vs. Physical Waveform Likelihood:** Separated the exact mathematical Hilbert quadrature family $\tilde{h}_{\perp}(f) = -i \operatorname{sgn}(f) \tilde{h}_0(f)$ from physical phase-shifted waveforms. Formulated the general mathematical condition for exact analytical Bessel $I_0$ phase marginalization and documented why physical waveforms with sub-dominant multipoles or rapid amplitude variations deviate from this condition.
5. **Correct Discrete Parseval Formulation:** Formulated the exact discrete Parseval energy balance for unnormalized `rfft` arrays, explicitly accounting for interior bin weights ($2\times$) versus DC and Nyquist bins ($1\times$), fully consistent with dimensional analysis.
6. **Unified $8.0\text{-Second}$ Baseline and Clarifications:** Maintained complete consistency with the $T_{\mathrm{analysis}} = 8.0 \, \mathrm{s}$ baseline ($\Delta f = 0.125 \, \mathrm{Hz}$), clarified GPS timestamp conventions ($1126259462.42$ is GPS seconds), clarified Butterworth filter orders (8th-order forward-backward produces an effective 16th-order magnitude response with zero phase lag), and labeled unverified source details accordingly.

---

### 2. Stage 0.3 Correction Log

```
+-----------------------------------------------------------------------------------------------------------------------------------------------+
|                                                Record of Stage 0.3 to Stage 0.4 Corrections                                                   |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| No.| Review Issue                   | Defective Formulation in Stage 0.3     | Reconciled Stage 0.4 Resolution                                |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 1  | Frequency-Dependent Overlap    | Equated r_w = 4/9 directly to complex  | Derived Cov(X_{m,k}, X_{m+1,k}^*) = (-1)^k r_w for white noise.|
|    | Covariance                     | Fourier correlation at all frequencies.| Proved frequency dependence for coloured noise via spectral    |
|    |                                |                                        | convolution. Replaced analytical variance with empirical MC.   |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 2  | Discrete vs. Continuous Window | Treated r_w = 4/9 and S_2 = 27/32 as   | Separated continuous integrals from exact discrete sums for    |
|    | Normalization                  | exact discrete values without finite   | M = 32,768. Mandated discrete runtime sum in implementation.   |
|    |                                | sum proof.                             |                                                                |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 3  | Coloured-Noise Residual        | Stated E[|r_w|^2] = 0.5 * S_n * T * S_2| Derived convolution E[|n_w|^2] = 0.5 * int |w_tilde|^2 S_n df. |
|    | Normalization                  | as universally exact.                  | Formalized locally flat approximation; added line test protocol|
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 4  | Exact Quadrature vs. Physical  | Conflated Hilbert frequency-domain     | Formally separated exact Hilbert quadrature from physical chirp|
|    | Waveform Phase Marginalization | quadrature with physical phase-shifted | waveforms. Formulated general harmonic condition for Bessel I_0|
|    |                                | waveform.                              | marginalization; identified higher-mode breakdown.             |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 5  | Incomplete Parseval Relation   | Stated sum |x|^2 dt = sum |X|^2 df     | Replaced with exact unnormalized rfft identity:                |
|    |                                | missing dt^2 and factor-of-2 weights.  | dt sum |x_j|^2 = df dt^2 [|X_0|^2 + 2 sum |X_k|^2 + |X_Nyq|^2].|
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 6  | GW150914 Time Convention Label | Labeled 1126259462.42 as UTC.          | Corrected label to GPS seconds (1126259462.42 GPS).            |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 7  | Butterworth Filter Definition  | Described as 8th-order zero-phase      | Clarified: 8th-order filtfilt yields an effective 16th-order   |
|    |                                | without specifying filtfilt doubling.  | magnitude roll-off with zero net phase lag.                    |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 8  | Monte Carlo Variance Error     | Quoted 1.41% without deriving normal-  | Derived SE(s^2)/sigma^2 = sqrt(2/N_trials); noted non-Gaussian |
|    |                                | sampling assumptions.                  | kurtosis effects and proposed empirical bootstrap check.       |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
| 9  | Bilby Implementation Source    | Asserted Bilby i0e implementation      | Designated specific source line as an unverified implementation|
|    | Verification                   | verified.                              | expectation to be audited in Stage 1.                          |
+----+--------------------------------+----------------------------------------+----------------------------------------------------------------+
```

---

### 3. Frequency-Dependent Overlap Covariance Derivation

#### 3.1 Mathematical Formulation of Adjacent Windowed Fourier Coefficients
Let a discrete time series of zero-mean stationary Gaussian noise $n_j$ ($j \in \mathbb{Z}$) have autocorrelation sequence $R_n[p] = \mathbb{E}[n_j n_{j+p}]$.
We partition the series into segments of length $M$ with shift $D = M/2$ ($50\%$ overlap).
The samples in segment $m$ and adjacent segment $m+1$ are:
$$n_j^{(m)} = n_{j + m D}, \quad n_l^{(m+1)} = n_{l + (m+1)D} = n_{l + m D + D}, \quad j, l \in \{0, 1, \dots, M-1\}$$
Applying the real-valued discrete window $w_j$, the unnormalized discrete Fourier transform coefficients at frequency bin $k$ are:
$$X_{m, k} = \sum_{j=0}^{M-1} w_j n_{j + m D} e^{-2\pi i j k / M}$$
$$X_{m+1, k} = \sum_{l=0}^{M-1} w_l n_{l + m D + D} e^{-2\pi i l k / M}$$
The cross-covariance between Fourier coefficients of adjacent segments at the same frequency bin $k$ is:
$$\operatorname{Cov}\left( X_{m, k}, X_{m+1, k} \right) \equiv \mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right] = \sum_{j=0}^{M-1} \sum_{l=0}^{M-1} w_j w_l \mathbb{E}\left[ n_{j + m D} n_{l + m D + D} \right] e^{-2\pi i (j - l) k / M}$$
By stationarity, $\mathbb{E}[n_{j + m D} n_{l + m D + D}] = R_n[j - l - D]$.
Making the substitution $p = j - l - D$ (so that $j - l = p + D$):
$$\mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right] = e^{-2\pi i D k / M} \sum_{p} R_n[p] e^{-2\pi i p k / M} \left( \sum_{l} w_{l + p + D} w_l \right)$$
For $50\%$ overlap, $D = M/2$, so the geometric phase factor is:
$$e^{-2\pi i (M/2) k / M} = e^{-i \pi k} = (-1)^k$$

#### 3.2 Evaluation for White Gaussian Noise
For white Gaussian noise with variance $\sigma^2$, $R_n[p] = \sigma^2 \delta_{p, 0}$.
The double sum collapses to $p = 0$:
$$\mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right] = \sigma^2 (-1)^k \sum_{l=0}^{M/2 - 1} w_{l + M/2} w_l$$
The variance of a single segment's Fourier coefficient is:
$$\operatorname{Var}\left( X_{m, k} \right) = \mathbb{E}\left[ |X_{m, k}|^2 \right] = \sigma^2 \sum_{j=0}^{M-1} w_j^2 = \sigma^2 M S_2$$
The cross-correlation coefficient of adjacent complex Fourier coefficients is therefore:
$$\gamma_k \equiv \frac{\mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right]}{\sqrt{\operatorname{Var}(X_{m, k}) \operatorname{Var}(X_{m+1, k})}} = (-1)^k \frac{\sum_{l=0}^{M/2 - 1} w_l w_{l + M/2}}{\sum_{j=0}^{M-1} w_j^2} = (-1)^k r_w$$
where $r_w$ is the normalized discrete overlap-product sum.
* **Frequency-Bin Sign Modulation:** Even frequency bins ($k$ even) have positive correlation $+r_w$, while odd frequency bins ($k$ odd) have negative correlation $-r_w$.
* **Global Time vs. Local Time Conventions:** If the discrete Fourier transform is defined using global time indices $t = j + m D$, an additional phase factor $e^{-2\pi i (m D) k / M} = (-1)^{m k}$ enters, modifying the local phase. In either convention, the cross-covariance is non-trivially phase-modulated.

#### 3.3 Evaluation for Coloured Noise
For coloured noise, the autocorrelation $R_n[p] = \int_{-f_{\mathrm{Nyq}}}^{f_{\mathrm{Nyq}}} S_n^{(2)}(f) e^{2\pi i f p \Delta t} \, df$ does not collapse to a delta function.
The cross-covariance becomes:
$$\mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right] = (-1)^k \int_{-f_{\mathrm{Nyq}}}^{f_{\mathrm{Nyq}}} S_n^{(2)}(f) \left[ \sum_{p} e^{-2\pi i p (f - f_k) \Delta t} \sum_{l} w_{l + p + M/2} w_l \right] df$$
Because $S_n^{(2)}(f)$ varies across frequency, the cross-covariance $\mathbb{E}[X_{m, k} X_{m+1, k}^*]$ is **explicitly frequency-dependent**.
Near steep spectral features (e.g., the seismic wall or suspension violin resonances), the covariance deviates substantially from $(-1)^k r_w$.

#### 3.4 Periodogram Covariance and Pseudo-Covariance Terms
Let $Z_1 = X_{m, k}$ and $Z_2 = X_{m+1, k}$.
Because $n_j$ is a real zero-mean Gaussian process, $(Z_1, Z_2)$ are jointly zero-mean complex Gaussian variables.
By Isserlis' theorem (Wick's theorem) for complex Gaussian variables, the covariance of their squared moduli is:
$$\operatorname{Cov}\left( |Z_1|^2, |Z_2|^2 \right) = \left| \mathbb{E}\left[ Z_1 Z_2^* \right] \right|^2 + \left| \mathbb{E}\left[ Z_1 Z_2 \right] \right|^2$$
where $\mathbb{E}[Z_1 Z_2]$ is the complex **pseudo-covariance** (relation matrix).
For stationary noise, away from DC ($k=0$) and Nyquist ($k=M/2$), the pseudo-covariance vanishes: $\mathbb{E}[Z_1 Z_2] \approx 0$.
Therefore, for white noise:
$$\operatorname{Cov}\left( \hat{P}_m(f_k), \hat{P}_{m+1}(f_k) \right) \propto \left| \mathbb{E}\left[ X_{m, k} X_{m+1, k}^* \right] \right|^2 = \left| (-1)^k r_w \sigma^2 M S_2 \right|^2 = r_w^2 \left( \sigma^2 M S_2 \right)^2$$
Dividing by the individual periodogram variances $\operatorname{Var}(\hat{P}) = (\mathbb{E}[\hat{P}])^2$:
$$\rho_P = \frac{\operatorname{Cov}(\hat{P}_m, \hat{P}_{m+1})}{\sqrt{\operatorname{Var}(\hat{P}_m)\operatorname{Var}(\hat{P}_{m+1})}} = r_w^2$$
* **Summary on Exactness:**
  * For white noise, the periodogram correlation equals $r_w^2$ exactly away from DC and Nyquist.
  * For coloured noise, the periodogram correlation is frequency-dependent.
  * The quantity $4/9$ represents strictly the **asymptotic zero-frequency overlap-product sum** for a continuous Tukey $\alpha=0.25$ window.
  * Because the sample median is a non-linear order statistic, there is no universal analytical formula linking $\rho_P$ to the variance of the median-Welch estimator. Calibration must be performed numerically.

---

### 4. Exact Discrete Tukey-Window Definitions

#### 4.1 Distinction of Window Normalization Concepts
We distinguish three mathematical representations:
1. **Continuous Integral Normalization:** Evaluates continuous functions $w(t)$ on $t \in [0, T]$:
   $$S_2^{\mathrm{cont}} \equiv \frac{1}{T} \int_0^T w^2(t) \, dt = 1 - \frac{5}{8}\alpha = \frac{27}{32} = 0.84375 \quad (\text{for } \alpha = 0.25)$$
   $$r_w^{\mathrm{cont}} \equiv \frac{\frac{1}{T} \int_0^{T/2} w(t) w(t + T/2) \, dt}{S_2^{\mathrm{cont}}} = \frac{3/8}{27/32} = \frac{4}{9} \approx 0.444444$$
2. **Asymptotic Limit ($M \to \infty$):** As $M \to \infty$, discrete Riemann sums converge to the continuous integrals at rate $\mathcal{O}(M^{-2})$ due to the smooth cosine roll-off.
3. **Exact Finite Discrete Sums ($M = 32,768$):** For a sampled time series of length $M$ with sampling interval $\Delta t$, the discrete window samples $w_j$ ($j = 0, \dots, M-1$) are defined as:
   $$w_j = \begin{cases}
   \frac{1}{2}\left[ 1 - \cos\left( \frac{\pi (j + 0.5)}{M_{\mathrm{taper}}} \right) \right], & 0 \le j < M_{\mathrm{taper}} \\
   1.0, & M_{\mathrm{taper}} \le j < M - M_{\mathrm{taper}} \\
   \frac{1}{2}\left[ 1 - \cos\left( \frac{\pi (M - 0.5 - j)}{M_{\mathrm{taper}}} \right) \right], & M - M_{\mathrm{taper}} \le j < M
   \end{cases}$$
   where $M = 32,768$, $\alpha = 0.25$, and $M_{\mathrm{taper}} = \lfloor \alpha M / 2 \rfloor = 4096$.

#### 4.2 Exact Finite Discrete Expressions
The implementation must evaluate the exact finite sums directly on the discrete array:
$$S_2^{\mathrm{exact}} \equiv \frac{1}{M} \sum_{j=0}^{M-1} w_j^2$$
$$r_w^{\mathrm{exact}} \equiv \frac{\sum_{j=0}^{M/2 - 1} w_j w_{j + M/2}}{\sum_{j=0}^{M-1} w_j^2}$$
* **Implementation Mandate:** The codebase in Stage 1 will not hard-code $0.84375$ or $4/9$. It will compute $S_2^{\mathrm{exact}}$ and $r_w^{\mathrm{exact}}$ at initialization via `np.sum(w**2) / M` and `np.sum(w[:M//2] * w[M//2:]) / np.sum(w**2)`.

---

### 5. Coloured-Noise Windowed Residual Derivation

#### 5.1 General Frequency-Domain Expectation via Spectral Convolution
Let $n(t)$ be zero-mean wide-sense stationary noise with one-sided power spectral density $S_n(f)$ ($f \ge 0$), corresponding to two-sided PSD $S_n^{(2)}(f) = \frac{1}{2} S_n(|f|)$.
Let $w(t)$ be a real-valued tapering window of duration $T$. The windowed noise process is:
$$n_w(t) = w(t) n(t)$$
Its continuous Fourier transform is the frequency-domain convolution:
$$\tilde{n}_w(f) = \int_{-\infty}^{\infty} \tilde{w}(f - f') \tilde{n}(f') \, df'$$
Evaluating the ensemble expectation of $|\tilde{n}_w(f)|^2$:
$$\mathbb{E}\left[ |\tilde{n}_w(f)|^2 \right] = \int_{-\infty}^{\infty} df' \int_{-\infty}^{\infty} df'' \, \tilde{w}(f - f') \tilde{w}^*(f - f'') \mathbb{E}\left[ \tilde{n}(f') \tilde{n}^*(f'') \right]$$
For stationary noise, distinct Fourier modes are uncorrelated: $\mathbb{E}[\tilde{n}(f') \tilde{n}^*(f'')] = S_n^{(2)}(f') \delta(f' - f'') = \frac{1}{2} S_n(|f'|) \delta(f' - f'')$.
Substituting the delta function:
$$\mathbb{E}\left[ |\tilde{n}_w(f)|^2 \right] = \frac{1}{2} \int_{-\infty}^{\infty} |\tilde{w}(f - f')|^2 S_n(|f'|) \, df'$$
This equation is **exact**: the expected power in the windowed noise spectrum at frequency $f$ is a convolution of the true physical noise PSD with the window's spectral energy density $|\tilde{w}(\nu)|^2$.

#### 5.2 The Locally Flat PSD Approximation
The window's spectral window $|\tilde{w}(\nu)|^2$ has a narrow main lobe of characteristic width $\Delta f_{\mathrm{lobe}} \approx \frac{1}{\alpha T}$. For $T = 8.0 \, \mathrm{s}$ and $\alpha = 0.25$, $\Delta f_{\mathrm{lobe}} \approx 0.5 \, \mathrm{Hz}$.
If the physical PSD $S_n(f')$ varies slowly over the bandwidth of the main lobe, we can expand $S_n(|f'|)$ in a Taylor series around $f$:
$$S_n(|f'|) = S_n(f) + S_n'(f)(|f'| - f) + \frac{1}{2} S_n''(f)(|f'| - f)^2 + \dots$$
Integrating term by term:
$$\mathbb{E}\left[ |\tilde{n}_w(f)|^2 \right] \approx \frac{1}{2} S_n(f) \int_{-\infty}^{\infty} |\tilde{w}(f - f')|^2 \, df' + \mathcal{O}\left( S_n''(f) \Delta f_{\mathrm{lobe}}^2 \right)$$
By Parseval's theorem on the window function:
$$\int_{-\infty}^{\infty} |\tilde{w}(\nu)|^2 \, d\nu = \int_0^T w^2(t) \, dt = T S_2$$
Therefore, under the **locally flat PSD approximation**:
$$\mathbb{E}\left[ |\tilde{n}_w(f)|^2 \right] \approx \frac{1}{2} S_n(f) T S_2$$

#### 5.3 Breakdown Regions and Validation Protocol
1. **Breakdown near Narrow Spectral Features:**
   At suspension violin resonances ($\sim 500 \, \mathrm{Hz}$, quality factor $Q \sim 10^5$, width $\Delta f_{\mathrm{line}} \ll 0.1 \, \mathrm{Hz}$) or power grid harmonics ($60 \, \mathrm{Hz}$), the PSD varies by orders of magnitude over $\Delta f_{\mathrm{lobe}}$. Power leaks from the sharp peak into adjacent bins, causing $\mathbb{E}[|\tilde{n}_w(f)|^2] \gg \frac{1}{2} S_n(f) T S_2$ in the line skirts.
2. **Breakdown near the Low-Frequency Seismic Wall:**
   Below $\sim 30 \, \mathrm{Hz}$, the noise rises steeply ($S_n(f) \propto f^{-8}$ to $f^{-4}$). Spectral leakage from sub-band seismic noise inflates the residual variance in the lowest analysis bins.
3. **Synthetic Coloured Noise Validation Protocol (Stage 1.2):**
   * Generate 1,000 realizations of synthetic Gaussian noise colored by the analytical aLIGO ZDHP curve.
   * Window with the baseline 8-second Tukey $\alpha=0.25$ window, bandpass filter, and compute normalized residuals:
     $$r(f_k) = \frac{\tilde{n}_w(f_k)}{\sqrt{\frac{1}{2} S_n(f_k) T S_2}}$$
   * Evaluate the empirical variance $\widehat{\operatorname{Var}}(r(f_k))$ across frequency. In broadband smooth regions ($40 - 800 \, \mathrm{Hz}$), the variance must match $1.0$ within $\pm 3\%$. In steep gradient regions, deviations will be mapped and used to establish frequency masks.

---

### 6. Exact Quadrature Family vs. Physical Waveform Likelihood

#### 6.1 Exact Mathematical Quadrature Family
Let $\tilde{h}_0(f)$ be a reference frequency-domain template defined over $[f_{\min}, f_{\max}]$.
We construct the orthogonal quadrature template directly in the frequency domain via:
$$\tilde{h}_{\perp}(f) \equiv -i \operatorname{sgn}(f) \tilde{h}_0(f)$$
Define the one-parameter template family parametrized by phase angle $\psi \in [0, 2\pi)$:
$$h(t; \psi) \equiv h_0(t) \cos\psi + h_{\perp}(t) \sin\psi$$
In the frequency domain for $f > 0$:
$$\tilde{h}(f; \psi) = \tilde{h}_0(f) \cos\psi - i \tilde{h}_0(f) \sin\psi = \tilde{h}_0(f) e^{-i\psi}$$
1. **Exact Equal Norms:**
   $$(h(\psi) | h(\psi)) = 4 \int_{f_{\min}}^{f_{\max}} \frac{|\tilde{h}(f; \psi)|^2}{S_n(f)} \, df = 4 \int_{f_{\min}}^{f_{\max}} \frac{|\tilde{h}_0(f) e^{-i\psi}|^2}{S_n(f)} \, df = 4 \int_{f_{\min}}^{f_{\max}} \frac{|\tilde{h}_0(f)|^2}{S_n(f)} \, df = \sigma_h^2$$
   The template norm is **strictly invariant under $\psi$ for all $\psi \in [0, 2\pi)$**.
2. **Exact Orthogonality:**
   $$(h_0 | h_{\perp}) = 4 \operatorname{Re} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{h}_0^*(f) [-i \tilde{h}_0(f)]}{S_n(f)} \, df = 4 \operatorname{Re}\left[ -i \int_{f_{\min}}^{f_{\max}} \frac{|\tilde{h}_0(f)|^2}{S_n(f)} \, df \right] \equiv 0$$
3. **Exact Analytical Bessel $I_0$ Marginalization:**
   The log-likelihood expanded over $\psi$ is:
   $$\ln \mathcal{L}(d | \theta, \psi) = (d | h_0) \cos\psi + (d | h_{\perp}) \sin\psi - \frac{1}{2}\sigma_h^2 - \frac{1}{2}(d|d) - \frac{1}{2}\ln\det(2\pi \mathbf{C})$$
   $$= |z| \cos(\psi - \hat{\psi}) - \frac{1}{2}\sigma_h^2 - \frac{1}{2}(d|d) - \frac{1}{2}\ln\det(2\pi \mathbf{C})$$
   where $z = (d | h_0) - i (d | h_{\perp})$.
   Under a uniform prior $\pi(\psi) = \frac{1}{2\pi}$, the marginal likelihood is:
   $$\mathcal{L}_{\mathrm{marg}}(d | \theta) = \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2 \right]}{\sqrt{\det(2\pi \mathbf{C})}} \int_0^{2\pi} \exp\left[ |z| \cos(\psi - \hat{\psi}) \right] \frac{d\psi}{2\pi} = \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2 \right]}{\sqrt{\det(2\pi \mathbf{C})}} I_0(|z|)$$
   This result is **exact** for the mathematical quadrature family.

#### 6.2 Physical Phase-Dependent Waveform Family
Now consider a physical compact binary coalescence waveform $h(t; \theta, \phi_c)$ generated by a relativistic approximant (e.g., `IMRPhenomD`, `TaylorF2`, or `SEOBNRv4`), where $\phi_c$ is the physical coalescence orbital phase.
1. **Physical Waveform Fourier Relation:**
   For a finite-duration physical chirp, advancing $\phi_c$ rotates the orbital phase $\Phi_{\mathrm{orb}}(t) \to \Phi_{\mathrm{orb}}(t) + \phi_c$.
   For the dominant quadrupole mode, this shifts the gravitational-wave phase by $2\phi_c$.
   Under the stationary-phase approximation, the positive-frequency waveform satisfies $\tilde{h}(f; \phi_c) \approx \tilde{h}_0(f) e^{-2i\phi_c}$.
   However, this relation is an **approximation**:
   * Counter-rotating terms $e^{-i \Phi_{\mathrm{gw}}(t)}$ and abrupt time-domain termination at ISCO introduce $\mathcal{O}(1/f)$ diffraction corrections.
   * If the template is windowed or conditioned in the time domain, window edge tapers break exact norm invariance.
2. **General Mathematical Condition for Exact Phase Marginalization:**
   Analytic phase marginalization via a single Bessel function $I_0$ holds if and only if the physical waveform family satisfies:
   $$h(t; \phi_c) = u(t) \cos(m \phi_c) + v(t) \sin(m \phi_c)$$
   with the **orthogonality and isometry conditions**:
   $$(u | u) = (v | v) \quad \text{and} \quad (u | v) = 0$$
   for a single harmonic integer $m \in \mathbb{N}_{\ge 1}$ (for the quadrupole mode, $m = 2$).
3. **Breakdown Scenarios:**
   * **Higher Harmonics:** If sub-dominant multipoles $(\ell, m) \ne (2, 2)$ are present (`IMRPhenomXHM`), the waveform decomposes as $\sum_m [u_m(t) \cos(m\phi_c) + v_m(t) \sin(m\phi_c)]$. The norm $(h | h)$ acquires explicit $\phi_c$ dependence, and the likelihood integral involves cross-harmonic interference terms $\int \exp[\sum A_m \cos(m\phi_c - \alpha_m)] d\phi_c$, which cannot be reduced to $I_0$.
   * **Spin Precession:** In precessing binaries, spin-orbit coupling modulates the polarization amplitudes dynamically, violating the simple single-harmonic phase relationship.
* **Numerical Validation Test (Stage 1.4):**
  We define a dedicated unit test in Stage 1.4: generate physical `TaylorF2` waveforms across a 100-point grid of $\phi_c \in [0, \pi)$ and compute:
  $$\Delta_{\mathrm{norm}} \equiv \max_{\phi_c} \frac{|(h(\phi_c) | h(\phi_c)) - \sigma_h^2|}{\sigma_h^2}$$
  The relative norm variation must satisfy $\Delta_{\mathrm{norm}} < 10^{-5}$ across the analysis band before enabling analytical phase marginalization on that model.

---

### 7. Correct Discrete Parseval Relation

#### 7.1 Derivation from the Discrete Fourier Transform
Let $x = [x_0, x_1, \dots, x_{N-1}]^T \in \mathbb{R}^N$ be a real-valued discrete time series sampled at interval $\Delta t$ over duration $T = N \Delta t$.
The unnormalized two-sided discrete Fourier transform is:
$$Y_k \equiv \sum_{j=0}^{N-1} x_j e^{-2\pi i j k / N}, \quad k \in \{0, 1, \dots, N-1\}$$
By standard discrete orthogonality of complex exponentials ($\sum_{j=0}^{N-1} e^{-2\pi i j (k - k') / N} = N \delta_{k, k'}$):
$$\sum_{j=0}^{N-1} |x_j|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |Y_k|^2$$
Multiplying both sides by $\Delta t$:
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \frac{\Delta t}{N} \sum_{k=0}^{N-1} |Y_k|^2$$
Using $\Delta f = \frac{1}{T} = \frac{1}{N \Delta t}$, we have $\frac{\Delta t}{N} = \frac{\Delta t^2}{N \Delta t} = \Delta f \Delta t^2$.
Therefore:
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \Delta f \Delta t^2 \sum_{k=0}^{N-1} |Y_k|^2$$

#### 7.2 Folding for Real Signals (`rfft`)
For real-valued $x_j \in \mathbb{R}$, Hermitian symmetry holds: $Y_{N-k} = Y_k^* \implies |Y_{N-k}|^2 = |Y_k|^2$.
Let $X_k$ be the output of `numpy.fft.rfft` ($k = 0, \dots, N/2$), which computes $X_k = Y_k$.
Splitting the two-sided sum:
$$\sum_{k=0}^{N-1} |Y_k|^2 = |Y_0|^2 + \sum_{k=1}^{N/2 - 1} |Y_k|^2 + |Y_{N/2}|^2 + \sum_{k=N/2 + 1}^{N-1} |Y_k|^2$$
Using the substitution $k' = N - k$ in the final sum:
$$\sum_{k=N/2 + 1}^{N-1} |Y_k|^2 = \sum_{k'=1}^{N/2 - 1} |Y_{N-k'}|^2 = \sum_{k'=1}^{N/2 - 1} |Y_{k'}|^2$$
Combining:
$$\sum_{k=0}^{N-1} |Y_k|^2 = |X_0|^2 + 2 \sum_{k=1}^{N/2 - 1} |X_k|^2 + |X_{N/2}|^2$$
Substituting into the energy conservation equation yields the **exact discrete Parseval relation**:
$$\Delta t \sum_{j=0}^{N-1} |x_j|^2 = \Delta f \Delta t^2 \left[ |X_0|^2 + 2 \sum_{k=1}^{N/2 - 1} |X_k|^2 + |X_{N/2}|^2 \right]$$

#### 7.3 Weight Explanation and Consistency
* **Weight of 2 on Interior Bins ($1 \le k < N/2$):** Each positive frequency bin $k$ represents two distinct conjugate modes in the two-sided spectrum ($+f_k$ and $-f_k$), each carrying energy $|X_k|^2 \Delta t^2$.
* **Weight of 1 on DC ($k = 0$) and Nyquist ($k = N/2$):** The DC frequency ($f = 0$) and Nyquist frequency ($f = f_s/2$) are strictly non-degenerate in the two-sided spectrum; they have no negative counterparts.
* **Consistency with Inner Product:** In the inner product $(a | b) = \frac{4}{N f_s} \operatorname{Re} \sum_{k=k_{\min}}^{k_{\max}} \frac{A_k^* B_k}{S_n(f_k)}$, the analysis band strictly satisfies $0 < k_{\min} \le k \le k_{\max} < N/2$. All active bins are interior bins, so the folding factor of 2 is absorbed into the prefactor ($2 \times 2 = 4$).

---

### 8. Additional Statistical and Implementation Clarifications

1. **Separation of Finite-Sample $C(K)$ and Overlap Calibration:**
   The finite-sample correction factor $C(K) = \sum_{j=1}^K \frac{(-1)^{j+1}}{j}$ is an exact analytical result derived from Rényi's order statistics under the strict assumption of **independent exponential periodograms**. For overlapping Welch segments, $C(K)$ does not fully capture the residual bias. The empirical bias and variance inflation factor $\kappa_{\mathrm{overlap}}$ will be calibrated via numerical simulation in Stage 1.2.
2. **Monte Carlo Variance Uncertainty Assumptions:**
   The relative standard error $\operatorname{SE}(s^2) / \sigma^2 \approx \sqrt{2 / N_{\mathrm{trials}}}$ ($\approx 1.41\%$ for $N_{\mathrm{trials}} = 10,000$) assumes that the sample variance is computed from independent, approximately normally distributed summary statistics. Because sample medians of exponential variables converge to normality via the central limit theorem, this formula provides a sound asymptotic benchmark. In Stage 1.2, we will cross-check this standard error via non-parametric bootstrap resampling over the $10,000$ simulation runs.
3. **Decimation vs. Interpolation:**
   * *Decimation:* For integer multiples (e.g., $T_{\mathrm{seg}} = 16.0 \, \mathrm{s}$ decimated to $T_{\mathrm{analysis}} = 8.0 \, \mathrm{s}$), selecting every 2nd bin is an **exact sub-grid extraction**.
   * *Interpolation:* For mismatched grids (e.g., $T_{\mathrm{seg}} = 4.0 \, \mathrm{s}$ or external curves), monotonic cubic Hermite interpolation (PCHIP) in $\log S_n - \log f$ is an **approximation** that smooths broadband noise but cannot reconstruct sub-bin spectral features.
4. **Bilby Implementation Claim Status:**
   The statement that Bilby evaluates phase marginalization via `scipy.special.i0e` reflects published methodology (Ashton et al. 2019). The exact line reference in the installed source code is classified as an **unverified implementation expectation** to be confirmed when Bilby is audited in the WSL2 environment.
5. **Three-Run Nested-Sampling Agreement Criterion:**
   The heuristic that three independent sampler seeds agree within $0.2 \sigma_{\mathrm{posterior}}$ is a **practical screening diagnostic**, not a universal mathematical proof of convergence. In Experiment D, convergence will be reinforced by comparing evidence values against the theoretical entropy bound $\sigma_{\ln\mathcal{Z}} \approx \sqrt{H / n_{\mathrm{live}}}$ and inspecting posterior quantile stability.
6. **GW150914 Time Convention Label:**
   The value $1126259462.42$ is **GPS seconds** (elapsed seconds since the GPS epoch 1980-01-06 00:00:00 UTC). The UTC timestamp is 2015-09-14 09:50:45.42 UTC.
7. **Butterworth Filter Order Definition:**
   The filtering protocol specifies an **8th-order forward-backward zero-phase digital Butterworth filter** (`scipy.signal.filtfilt`). Because `filtfilt` passes data forward and then backward, the filter's transfer function is $|H(f)|^2$, producing an **effective 16th-order magnitude roll-off** ($96 \, \mathrm{dB/octave}$) with identically **zero phase distortion**.
8. **Four-Second Buffer Transient Status:**
   The $4\text{-second}$ trimming buffer is a **proposed design buffer**. In Stage 1.2, filter impulse-response tests will inject unit impulses and verify that filter transient amplitudes decay by $> 120 \, \mathrm{dB}$ before reaching the un-trimmed analysis segment.
9. **Scope of 100-Injection Coverage Validation (Exp D.2):**
   Experiment D.2 ($N_{\mathrm{inj}} = 100$) is an **exploratory calibration test**. Its statistical resolution is limited by the $10\%$ binomial confidence band at median. In addition to 1D P-P plots, simulation-based calibration (SBC) rank histograms will be computed to test for prior-posterior rank uniformity.

---

### 9. Revised Experimental Protocol

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                            Structured Experimental Design Matrix                                              |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| Exp.  | Scope / Category         | Independent Variable         | Controlled Variables      | Primary Metric        | Priority|
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| A     | Baseline Synthetic       | Distance-tuned SNR:          | Injected M_c=30, q=0.8;   | Recovered rho_max vs  | Core    |
|       | Recovery                 | rho* in {10, 20, 35}         | zero spins; ZDHP PSD      | rho*; parameter bias  |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| B     | PSD Sensitivity          | Factorial matrix (12 runs):  | Fixed injection (rho=20); | Credible interval     | Core    |
|       | (Factorial Design)       | Estimator (Mean vs Median);  | identical conditioning;   | width ratio           |         |
|       |                          | T_off (64, 1024s); T_seg(4,8,16) analytical ZDHP reference| W_90(PSD)/W_90(true)  |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| C.1   | Waveform Mismatch        | Inspiral-only (TaylorF2) vs. | Total mass M = 70 M_sun;  | Systematic parameter  | Core    |
|       | (Inspiral vs. IMR)       | IMR (IMRPhenomD)             | zero-noise + noisy        | bias; SNR loss        |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| C.3   | Same-Approximant Control | Control: Inject IMRPhenomD,  | Identical parameters      | Parameter bias        | Core    |
|       | Baseline                 | Recover IMRPhenomD           | as C.1                    | consistent with noise |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| D.1   | Pipeline Execution Pilot | 10 independent prior-draw    | Reduced 2D space (M_c, q);| Execution stability;  | Core    |
|       | (Not full coverage)      | injections into noise        | analytical ZDHP PSD       | quantile calculation  |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| E     | Real Detector Data       | Calibrated H1/L1 strain;     | Fixed 8s segment; 1024s   | Posterior medians vs. | Core    |
|       | (GW150914 Validation)    | empirical Welch PSD          | off-source PSD; guard 16s | published GWTC-1      |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| C.2   | Higher Modes Mismatch    | Inject IMRPhenomXHM,         | Inclination iota = 60 deg;| Bias in mass ratio q  | Ext.    |
|       | (Sub-dominant Harmonics) | Recover IMRPhenomD ((2,2) onl| zero-noise + noisy        | and distance d_L      |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
| D.2   | Exploratory Coverage     | 100 independent prior-draw   | Reduced 3D space (M_c,q,t)| SBC rank histograms & | Ext.    |
|       | Validation (P-P Test)    | injections into noise        | analytical ZDHP PSD       | KS test on quantiles  |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
```

---

### 10. Final Baseline Parameters and Mathematical Conventions

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                      Final Baseline Parameters and Mathematical Conventions                                    |
+------------------------------------+------------------------------------------------------------------------------------------+
| Parameter / Convention             | Explicit Specification                                                                   |
+------------------------------------+------------------------------------------------------------------------------------------+
| Analysis Segment Duration          | T_analysis = 8.0 seconds                                                                 |
| Sampling Frequency & Time Step     | f_s = 4096 Hz, Delta t = 2.44140625e-4 seconds (N = 32,768 points)                        |
| Discrete Frequency Resolution      | Delta f_analysis = 0.125 Hz                                                              |
| Active Analysis Bandwidth          | [f_min, f_max] = [30.0, 1024.0] Hz (k_min = 240, k_max = 8192)                           |
| Continuous Fourier Transform       | x_tilde(f) = int x(t) exp(-2*pi*i*f*t) dt                                                |
| Discrete FFT Mapping               | x_tilde(f_k) approx Delta t * X_k = (1/f_s) * X_k (unnormalized rfft)                     |
| Noise-Weighted Inner Product       | (a|b) = (4 / [N * f_s]) * Re sum_{k=240}^{8192} [A_k* * B_k / S_n(f_k)] (dimensionless) |
| Exact Frequency Quadrature         | h_perp_tilde(f) = -i * sgn(f) * h_0_tilde(f)                                             |
| Phase Extraction Formula           | 2*phi_c_hat = atan2(y, x), where y = -Im[z], x = Re[z]                                   |
| On-Source Conditioning Window      | 8.0s Tukey window with alpha = 0.25 (1.0s taper, 6.0s flat top)                           |
| Window Power Normalization         | S_2 = (1/M) sum w_j^2 (calculated from discrete samples; asymptotic: 27/32 = 0.84375)   |
| Normalized Residual Definition     | r(f_k) = [d_w_tilde(f_k) - h_tilde(f_k)] / sqrt(0.5 * S_n(f_k) * T_analysis * S_2)       |
| Primary Welch PSD Configuration    | T_seg = 8.0s (Delta f = 0.125 Hz native match), 50% overlap, Tukey alpha = 0.25           |
| Finite-Sample Median Bias Factor   | C(K) = sum_{j=L+1}^{2L+1} (1/j) for odd K = 2L + 1 (C(15) = 0.725372, C(255) = 0.695104)  |
| GW150914 Coalescence GPS Timestamp | t_c = 1126259462.42 GPS seconds (2015-09-14 09:50:45.42 UTC)                             |
| GW150914 On-Source GPS Segment     | GPS [1126259458.0, 1126259466.0]                                                         |
| GW150914 Off-Source PSD Segment    | GPS [1126258418.0, 1126259442.0] (1024s duration, K = 255 segments)                      |
| GW150914 Guard Interval            | GPS [1126259442.0, 1126259458.0] (16.0s duration, zero overlap)                          |
+------------------------------------+------------------------------------------------------------------------------------------+
```

---

### 11. Updated Validation Matrix

```
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                               Validation Matrix                                                                    |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Component / Domain       | Expected Analytical Property        | Proposed Numerical Test     | Numerical Tolerance   | Failure Interpretation     |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Discrete Parseval Balance| dt sum |x_j|^2 == df dt^2 [|X_0|^2  | Double-precision white      | Relative error        | Incorrect dt, df, or bin   |
| (Stage 1.1)              | + 2 sum |X_k|^2 + |X_Nyq|^2]        | noise time series           | < 1e-12               | weighting in Parseval sum. |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Complex Inner Product    | Conjugate symmetry on C^M:          | Random complex arrays A, B; | Absolute difference   | Missing complex conjugate  |
| (Stage 1.3)              | <A, B>_W == <B, A>_W*               | evaluate <A, B>_W           | < 1e-14               | in first argument.         |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Real Inner Product       | Strict symmetry on R^N:             | Random real arrays a, b;    | Absolute difference   | Asymmetric implementation  |
| (Stage 1.3)              | (a|b) == (b|a); (a|a) > 0 for a!=0  | evaluate (a|b)              | < 1e-14               | or real-part omission.     |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Finite-Sample Median     | Unbiased finite-sample correction:  | 10^4 independent exponential| Relative bias error   | Neglected finite-sample    |
| Welch (Stage 1.2)        | E[M_K / C(K)] == S_n                | realizations for K=15, 255  | < 1.0%                | correction factor C(K).    |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Overlap Monte Carlo      | Calibrated variance inflation factor| 10^4 Gaussian realizations  | Relative SE approx    | Overlap covariance         |
| (Stage 1.2)              | kappa_overlap(K, alpha)             | with 50% overlap Tukey 0.25 | 1.4% (bootstrap check)| miscalculated.             |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Quadrature FFT Sign      | Positive-frequency relation:        | Compute rfft of analytic    | Max relative error    | Sign error; used +i        |
| (Stage 1.4)              | h_perp_tilde(f) == -i * h_0_tilde(f)| sinusoids h_0 and h_perp    | < 1e-12               | instead of -i.             |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Phase Recovery & Branch  | Exact phase extraction:             | Inject monochromatic wave   | Absolute error        | Quadrature sign inversion  |
| (Stage 1.4)              | 2*phi_rec = atan2(y, x)             | with trial phi in [0, pi)   | < 1e-5 (modulo 2pi)   | or atan2 argument swap.    |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Fixed-Time SNR Statistics| Fixed-time SNR follows Rayleigh:    | Filter 10^4 Gaussian noise  | Kolmogorov-Smirnov    | Inappropriate Gaussian     |
| (Stage 1.4)              | p(rho) = rho * exp(-rho^2 / 2)      | realizations at fixed t_0   | p-value > 0.05        | assumption used.           |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| TaylorF2 PN Phase        | Frequency derivatives match PN      | Compare pure-Python TaylorF2| Overlap match > 0.999 | Typographical error in     |
| (Stage 1.5)              | expansion coefficients (30-1000 Hz) | against analytical PN series| on [30, 1000] Hz      | Post-Newtonian coefficients|
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Bessel Phase Marg.       | Exponentially scaled Bessel identity| Evaluate ln I_0(u) via      | Relative difference   | Numerical overflow in I_0  |
| (Stage 1.6)              | ln I_0(u) = u + ln I_0e(u)          | i0e across u in [0, 1000]   | < 1e-12               | or improper scaling.       |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Stable Time Marg.        | Log-sum-exp sum over Bessel values  | Compare stable log-sum-exp  | Relative difference   | Improper offset scalar     |
| (Stage 1.6)              | matches numerical integral over t_c | against fine time-grid quad.| < 1e-4                | used in log-sum-exp.       |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
```

---

### 12. Unresolved Assumptions and Scientific Limitations

1. **GW150914 Off-Source Segment Local Stationarity:** While archival catalog records verify continuous science-mode operation, empirical verification of data-quality channels, glitch absence, and noise stationarity within GPS $[1126258418, 1126259442]$ will occur upon data ingestion in Stage 1.
2. **Empirical Overlap Variance Calibration:** The exact value of $\kappa_{\mathrm{overlap}}$ for $50\%$ overlapping Tukey segments under coloured noise will be calibrated numerically in Stage 1.2 rather than assumed analytically.
3. **Filter Edge Transient Decay:** The $4\text{-second}$ buffer for Butterworth forward-backward filtering will be tested on unit impulses in Stage 1.2 to verify attenuation below $10^{-6}$.
4. **Physical Waveform Approximant Limits:** Pure-Python `TaylorF2` terminates at $f_{\mathrm{ISCO}}$ and is valid for inspiral testing; high-mass IMR parameter estimation requires `IMRPhenomD` accessed via WSL2.

---

### 13. Final Approval-Readiness Statement

* **Document Generated:** [`Stage_0_4_Final_Scientific_Corrections.md`](file:///c:/Users/Soumitra/Desktop/GW-matched-filtering-and-baysian-interference/Stage_0_4_Final_Scientific_Corrections.md)
* **Implementation Status:** **Not started.**
* **Execution Boundary Compliance:**
  * No production source code has been written.
  * No Python packages have been installed on the Windows host or WSL2 subsystem.
  * No strain data have been downloaded from GWOSC.
  * No computational tests have been run.
  * Stage 1.1 has not been initiated.
  * The local workspace contains only design documentation.
* **Approval Recommendation:**
  Stage 0.4 resolves all five substantive mathematical and statistical concerns and incorporates all required clarifications. The mathematical foundations, exact discrete definitions, spectral convolution models, and unified $8.0\text{-second}$ experimental protocols are complete, rigorous, and self-consistent.
  **Stage 0 is now ready for formal independent scientific approval to authorize Stage 1.1.**
