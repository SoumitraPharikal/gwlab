# Work Package 3: Mathematical Specification and Pre-Implementation Scientific Audit
## Robust Matched Filtering, Time-Series Correlation, Phase Maximization, and Detection Statistics
### Revision 6 — Final Targeted Scientific Closure Pass

**Project:** Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals  
**Role:** Antigravity Implementation Agent  
**Independent Scientific Reviewer:** ChatGPT  
**Current Branch:** `milestone-1.4a`  
**Current Accepted Checkpoint:** `ff91df71d3c57a64313a3537f523808f789f59d3`  
**Execution Mode:** STRICTLY SPECIFICATION, SOURCE INSPECTION, AND MATHEMATICAL AUDIT — NO IMPLEMENTATION  
**Date:** 2026-11-09  

---

## 1. Executive Summary & Revision 6 Change Log

`[MATHEMATICALLY DERIVED]` `[PROPOSED]`
This report establishes the complete, authoritative mathematical specification and pre-implementation software architecture for **Work Package 3 (WP3): Matched Filtering, Time-Lagged Complex Correlation, Phase Maximization, and Detection Statistics**, finalized in Revision 6 to achieve complete scientific and numerical closure.

Milestones WP0 (Fourier conventions, scaling, Parseval identities), WP1 (frequency grids, active observation masks, PSD engine), and WP2 (TaylorF2 3.5PN frequency-domain waveform engine) are fully implemented, verified across 200 unit tests, and committed at checkpoint `ff91df7`. WP3 synthesizes detector data $d(t)$, one-sided noise spectral density $S_n(f)$, and inspiral template strain $\tilde{h}(f)$ into optimal signal-to-noise ratio (SNR) time series $\rho(t)$, phase-maximized detection statistics $\rho_{\mathrm{max}}(t)$, and time-frequency signal-consistency diagnostics $\chi^2(t)$.

### 1.1 Revision 6 Change Log
The following targeted corrections have been applied in Revision 6:
1. **Allen Chi-Square Distribution and Glitch Characterization:** Corrected the statistical characterization of the Allen $\chi^2$ statistic. Formulated the **zero-projected-mean condition** ($\mathbb{E}[\Delta \mathbf{z}] = \mathbf{0}$) as the fundamental statistical condition for the central $\chi^2_{2q-2}$ law under the Gaussian covariance model and rank assumptions, identifying exact sub-band morphology matching as a sufficient physical condition. Explicitly established that deterministic waveform mismatch or unmodeled deterministic signal components producing a nonzero projected deterministic mean result in a non-central $\chi^2$ distribution ($\lambda > 0$). Strictly clarified that **non-Gaussian instrumental glitches do NOT generally follow a central or non-central $\chi^2$ distribution**, because glitches violate the underlying Gaussian noise model itself. Their residual-statistic distributions depend on the specific glitch process and must be empirically characterized (e.g. via time slides or morphology models). Arbitrary glitch-contaminated data is strictly not assigned a central or non-central $\chi^2$ law.
2. **Authoritative WP1 Observation Grid vs. Padded Discretizations:** Explicitly distinguished four distinct operations: (i) the original, frozen WP1 observation grid with fixed $\Delta f = 1/(N\Delta t)$, frequencies $f_k = k \Delta f$, and static active mask $\mathcal{K}_+$; (ii) padded grids used for discrete sequence correlation; (iii) PSD evaluation on refined frequency grids; and (iv) mapping computed results back to observation-time samples. Established that **the direct finite frequency sum on the frozen WP1 grid is the authoritative reference definition of the matched-filter statistic**. Clarified that padding in the frequency domain changes the frequency discretization and alters the active-bin inner product. Explicitly **deferred padded-grid implementations** and circular boundary certification, preserving the frozen WP1 inner product as authoritative.
3. **Formal Methodological Classification:** Updated all sections, tables, and readiness gates to rigorously separate:
   - `[EXISTING & SOURCE-VERIFIED]`: Confirmed present in repository source code and unit tests.
   - `[MATHEMATICALLY DERIVED]`: Rigorously deduced from first principles and frozen conventions.
   - `[PROPOSED]`: Recommended design, signature, or algorithm for WP3 implementation.
   - `[NUMERICALLY TESTED]`: Confirmed by automated regression suites.
   - `[DEFERRED]`: Deliberately postponed to post-implementation benchmarking or downstream stages (e.g., WP4, Stage 2).
   - `[UNRESOLVED]`: Requires project-level governance decision prior to implementation.

---

## 2. Baseline and Interface Inventory

### 2.1 Git State and Working Tree Audit
`[EXISTING & SOURCE-VERIFIED]`
- **Active Branch:** `milestone-1.4a`
- **Current HEAD Commit:** `ff91df71d3c57a64313a3537f523808f789f59d3`
- **Commit Subject:** `Implement WP0-WP2 Fourier conventions, PSD grid, and TaylorF2 waveform`
- **Parent Baseline:** `eafe68ae31c3e6e5cf25a62a7be6f0e5bf6a4b08` (`master`).
- **Source Tree Integrity:** Exactly 0 lines modified in `src/` and `tests/`.
- **Working Tree:** Clean, with only the report in `reports/`.
- **Automated Test Suite Status:** `200 passed in 5.54s` (157 baseline regression tests + 43 WP0–WP2 milestone tests).

### 2.2 Committed Public Interfaces Consumed by WP3
`[EXISTING & SOURCE-VERIFIED]`
Direct inspection of `src/gw_inference/` verifies the following exact public interfaces:
- [`FrequencyGrid`](../../../src/gw_inference/grid.py#L20-L110): Full frequency grid $f_k = k \Delta f$ ($k = 0, \dots, \lfloor N/2 \rfloor$), `create_active_mask(f_start, f_end)` with DC and even-N Nyquist strictly excluded.
- [`continuous_rfft`](../../../src/gw_inference/conventions.py#L126-L154): Continuous Fourier transform approximation $\tilde{x}(f_k) = \Delta t \cdot \operatorname{rfft}(x)_k$.
- [`PSDEngine`](../../../src/gw_inference/psd.py#L932-L1115): Log-linear PSD interpolator; `get_inverse_psd_on_active_bins` returning $1 / S_n(f_k)$ on $\mathcal{K}_+$ (requires mandatory $f_0$).
- [`taylorf2_fd_waveform`](../../../src/gw_inference/waveform.py#L258-L329): Generates $\tilde{h}(f_k) = \mathcal{A} f^{-7/6} e^{-i \Psi_{\mathrm{TF2}}^+(f_k)}$ for $f_k \le f_{\mathrm{term}} = \min(f_{\mathrm{end}}, f_{\mathrm{ISCO}})$, and identically 0 for $f_k > f_{\mathrm{term}}$.
- [`apply_coalescence_time_shift`](../../../src/gw_inference/waveform.py#L331-L361): Evaluates $\tilde{h}(f) e^{-i 2\pi f t_c}$.

---

## 3. Full Mathematical Derivations

### 3.1 Noise-Weighted Inner Product and Discretization
`[MATHEMATICALLY DERIVED]`

#### 3.1.1 Continuous Real Inner Product
Let $n(t)$ be a zero-mean, wide-sense stationary, real-valued Gaussian noise process. Under the continuous Fourier transform convention $\tilde{x}(f) = \int_{-\infty}^{\infty} x(t) e^{-i 2\pi f t} dt$, the two-sided power spectral density $S_n^{(2)}(f)$ is defined by $\mathbb{E}[\tilde{n}(f) \tilde{n}^*(f')] = S_n^{(2)}(f) \delta(f - f')$.  
For real-valued noise, $\tilde{n}(-f) = \tilde{n}^*(f)$ and $S_n^{(2)}(-f) = S_n^{(2)}(f)$. The one-sided power spectral density $S_n(f)$ is defined on $f \ge 0$ by $S_n(f) = 2 S_n^{(2)}(f)$.  
The log-likelihood ratio for real Gaussian noise establishes the real noise-weighted inner product:
$$(a|b) = 4 \operatorname{Re} \int_0^{\infty} \frac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} df$$

#### 3.1.2 Origin of the Factor of 4 and Dimensional Analysis
- One factor of 2 arises from the one-sided PSD definition: $\frac{1}{S_n^{(2)}(f)} = \frac{2}{S_n(f)}$.
- A second factor of 2 arises from folding negative frequencies $(-\infty, 0)$ into $(0, \infty)$ for real time series ($2 \times 2 = 4$).
- Units: $[\tilde{a}] = [\tilde{b}] = \mathrm{strain} \cdot \mathrm{s} = \mathrm{s}$, $[S_n] = \mathrm{s}$, $[df] = \mathrm{s}^{-1}$. The integral dimension is $[(a|b)] = \mathrm{s}^{-1} \cdot (\mathrm{s}^2 / \mathrm{s}) = 1$ (strictly dimensionless).

#### 3.1.3 Discrete Active-Band Inner Product
Discretizing on grid $f_k = k \Delta f$ with $\Delta f = 1/(N \Delta t)$ across active mask $\mathcal{K}_+$:
$$(a|b)_{\mathcal{K}_+} = 4 \Delta f \operatorname{Re} \sum_{k \in \mathcal{K}_+} \frac{\tilde{a}^*(f_k) \tilde{b}(f_k)}{S_n(f_k)}$$
Under continuous scaling $\tilde{a}(f_k) \simeq \Delta t A_k$ and $\tilde{b}(f_k) \simeq \Delta t B_k$, where $A_k, B_k$ are unnormalized DFT coefficients:
$$(a|b)_{\mathcal{K}_+} = 4 \Delta f \Delta t^2 \operatorname{Re} \sum_{k \in \mathcal{K}_+} \frac{A_k^* B_k}{S_n(f_k)} = \frac{4 \Delta t}{N} \operatorname{Re} \sum_{k \in \mathcal{K}_+} \frac{A_k^* B_k}{S_n(f_k)}$$

#### 3.1.4 Endpoint Exclusion, Odd/Even Grids, and Mask Invariance
1. **DC ($k = 0$):** Real-valued in DFT with folding weight 1, not 2; contains zero gravitational radiation while noise diverges. Strictly excluded ($\mathrm{mask}[0] = \mathrm{False}$).
2. **Nyquist ($k = N/2$ for even $N$):** Real-valued in DFT with folding weight 1, not 2. Strictly excluded ($\mathrm{mask}[N/2] = \mathrm{False}$) for even $N$ to prevent doubling variance.
3. **Odd $N$:** No Nyquist bin exists; $k = (N-1)/2$ is an interior bin and is retained if within $[f_{\mathrm{start}}, f_{\mathrm{end}}]$.
4. **Mask Invariance:** $\mathcal{K}_+$ is fixed by the detector band $[f_{\mathrm{start}}, f_{\mathrm{end}}]$ and is completely independent of template masses or $f_{\mathrm{term}}$. Waveform termination is enforced within $\tilde{h}(f_k)$ by zeroing bins above $f_{\mathrm{term}}$.

#### 3.1.5 Active-Band Whittle/Circulant Approximation Baseline
`[MATHEMATICALLY DERIVED]`
The noise-weighted inner product $(a|b)_{\mathcal{K}_+}$ is defined and evaluated strictly as a **positive-frequency active-band Whittle/circulant approximation**, not an exact arbitrary finite Toeplitz likelihood. For finite discrete segments with non-periodic signals and edge effects, the exact time-domain inverse covariance matrix contains off-diagonal elements that are not captured by a diagonal frequency-domain weighting.

---

### 3.2 Matched Filter Derivation & Categorization
`[MATHEMATICALLY DERIVED]`

Under the Whittle approximation, the log-likelihood ratio $\ln \Lambda(d) = (d|h)_{\mathcal{K}_+} - \frac{1}{2} (h|h)_{\mathcal{K}_+}$ establishes $(d|h)_{\mathcal{K}_+}$ as the optimal linear detection statistic in colored Gaussian noise.

WP3 defines five distinct mathematical quantities:
1. **Template Norm:** $\sigma_h \equiv \sqrt{(h|h)_{\mathcal{K}_+}} = \sqrt{4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{|\tilde{h}(f_k)|^2}{S_n(f_k)}}$
2. **Optimal SNR:** $\rho_{\mathrm{opt}} \equiv \sigma_h = \sqrt{(h|h)_{\mathcal{K}_+}}$
3. **Unnormalized Real Filter Output:** $x = (d | h)_{\mathcal{K}_+}$
4. **Normalized Real Filter SNR (Fixed Phase/Lag):** $\hat{\rho} = (d | h)_{\mathcal{K}_+} / \sigma_h \sim \mathcal{N}(0, 1)$ in pure noise
5. **Complex Matched Filter Time Series:** $z(t) = 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}(f_k) \tilde{h}^*(f_k)}{S_n(f_k)} e^{i 2\pi f_k t}$

---

### 3.3 Authoritative WP1 Observation Grid vs. Padded Discretizations
`[MATHEMATICALLY DERIVED]` `[PROPOSED]`

#### 3.3.1 Rigorous Separation of Grids and Operations
To prevent conflating distinct mathematical operations, WP3 formally distinguishes four separate structures:
1. **The Authoritative WP1 Observation Grid:**
   The physical observation baseline defined by segment length $N$, sampling interval $\Delta t$, duration $T = N \Delta t$, frequency resolution $\Delta f = 1/T$, frequencies $f_k = k \Delta f$ ($k = 0, \dots, \lfloor N/2 \rfloor$), and static active mask $\mathcal{K}_+ \subset \{1, \dots, \lfloor (N-1)/2 \rfloor\}$.  
   **Authoritative Ground-Truth Definition:** The matched-filter statistic $(d|h)_{\mathcal{K}_+}(t)$ is defined strictly by the **direct finite frequency sum on the frozen WP1 grid**:
   $$(d|h)_{\mathcal{K}_+}(t) \equiv 4 \Delta f \operatorname{Re} \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}(f_k) \tilde{h}^*(f_k)}{S_n(f_k)} e^{i 2\pi f_k t}$$
   This definition requires zero padding, introduces no altered frequency bins, and strictly preserves the WP1 inner product and likelihood contracts.
2. **Padded Correlation Grids ($N_{\mathrm{pad}} > N$):**
   Evaluating a correlation via FFT on a padded length $N_{\mathrm{pad}} > N$ refines the frequency resolution to $\Delta f_{\mathrm{pad}} = 1 / (N_{\mathrm{pad}} \Delta t) < \Delta f$.
   - **Crucial Distinction:** Zero-padding in time alters the frequency-domain discretization. It introduces new intermediate frequency bins $f_{k'} = k' \Delta f_{\mathrm{pad}}$ that do not coincide with the WP1 grid bins $f_k = k \Delta f$.
   - **Inner Product Invariance:** Re-evaluating $S_n(f)$ and summing across a padded grid defines a **different discrete inner product** whose mathematical equivalence to the frozen WP1 inner product is not established a priori.
   - **Governance Decision:** The WP1 observation-grid convention remains authoritative. Padded-grid correlation is NOT assumed equivalent to the WP1 statistic and its implementation is **DEFERRED**.
3. **PSD Evaluation on Padded Grids:**
   Interpolating the continuous-spectrum model $S_n(f)$ onto refined frequency points $\Delta f_{\mathrm{pad}}$ evaluates the continuum model at different nodes, but does not prove convergence to the discrete WP1 inner product.
4. **Mapping Back to Observation-Time Samples:**
   In circular correlation on the WP1 grid, the $N$-point IFFT evaluates the exact WP1 sum at the observation sample times $t_m = m \Delta t$ ($m = 0, \dots, N-1$).

#### 3.3.2 Non-Compact Filter Support and Boundary Guard Deferral
`[MATHEMATICALLY DERIVED]` `[DEFERRED]`
In the stationary phase approximation (SPA), the TaylorF2 template coalesces at $t_c = 0$, with the physical inspiral on negative times $t \in [-\tau_{\mathrm{chirp}}, 0]$, where $\tau_{\mathrm{chirp}} = \tau(f_{\mathrm{start}}) - \tau(f_{\mathrm{term}})$.
The noise-weighted filter kernel is $q(t) = h(-t) * w(t)$, where $w(t) = \mathcal{F}^{-1}\{1 / S_n(f)\}$.
Because $1 / S_n(f)$ contains continuous analytical roll-offs governed by $f_0$ (which is unresolved under open Decision D8), $w(t)$ is **infinitely supported (non-compact)**.
Therefore:
- An operational threshold (such as $|w(t)| < 10^{-4} \max |w|$) is an empirical diagnostic, not a mathematical proof of zero boundary error.
- **Circular Boundary Certification is DEFERRED:** The validity of circular correlation boundary guards is deferred pending post-implementation numerical convergence studies comparing circular correlation with direct summation and padded references.
- **Operational Diagnostic:** Pre-implementation testing applies `valid_mask` with $\tau_{\mathrm{filter, diag}} = 0.2\text{ s}$ strictly as an uncertified operational heuristic.

---

### 3.4 Full Computational FFT/IFFT Construction
`[MATHEMATICALLY DERIVED]`

#### 3.4.1 Discrete Analytic Signal Representation
The complex matched filter time series is:
$$z(t) = 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}_k \tilde{h}_k^*}{S_n(f_k)} e^{i 2\pi f_k t} = x(t) + i y(t)$$
where $x(t) = (d | h_0)_t$ is the in-phase matched filter, and $y(t) = (d | h_{\pi/2})_t$ is the quadrature matched filter.
1. **Insufficiency of Real `irfft`:** NumPy's `irfft(A, n=N)` accepts a one-sided spectrum and computes exclusively the real-valued time series $\frac{1}{N} [A_0 + 2 \operatorname{Re}\sum A_k e^{i 2\pi k m / N} + \dots]$. It completely discards the imaginary quadrature $y(t)$. To obtain the phase-maximized SNR $\rho_{\mathrm{max}}(t) = |z(t)|/\sigma_h$, `irfft` alone is insufficient unless executed twice with orthogonal templates.
2. **Full Complex `ifft` Embedding:**
   Construct an $N$-point complex frequency array $Z \in \mathbb{C}^N$ initialized to zero:
   $$Z_k = \begin{cases} 4 \Delta f \frac{\tilde{d}_k \tilde{h}_k^*}{S_n(f_k)} = \frac{4 \Delta t}{N} \frac{D_k H_k^*}{S_n(f_k)}, & k \in \mathcal{K}_+ \\ 0, & \text{otherwise} \quad (k \notin \mathcal{K}_+, \text{ including DC, Nyquist, and negative frequencies } k \ge \lceil N/2 \rceil) \end{cases}$$
   NumPy's $N$-point `ifft` evaluates:
   $$\operatorname{ifft}(Z)_m = \frac{1}{N} \sum_{k=0}^{N-1} Z_k e^{i 2\pi k m / N} = \frac{1}{N} z(t_m) \implies z(t_m) = N \cdot \operatorname{ifft}(Z)_m$$
   In terms of unnormalized DFT coefficients $\tilde{d}_k = \Delta t D_k, \tilde{h}_k = \Delta t H_k$:
   $$z(t_m) = 4 \Delta t \cdot \operatorname{ifft}\left( \left\{ \frac{D_k H_k^*}{S_n(f_k)} \right\}_{k \in \mathcal{K}_+} \right)_m$$
   This construction preserves both quadratures, has exact mathematical scaling, and avoids redundant transform calls.

#### 3.4.2 Reproducible Lag-Sign Numerical Examples
Let $N = 16, \Delta t = 0.05\text{ s}, T = 0.8\text{ s}, f_s = 20\text{ Hz}, \Delta f = 1.25\text{ Hz}$.  
One-sided grid: $f_k = k \times 1.25\text{ Hz}$ for $k = 0, \dots, 8$.  
Let active band $\mathcal{K}_+ = \{2, 3, 4\}$ ($2.5, 3.75, 5.0\text{ Hz}$), unit noise $S_n(f_k) = 1.0$, and unit template $\tilde{h}_k = 1.0$ on $\mathcal{K}_+$.
- **Zero Lag ($t_c = 0\text{ s}$):** Data $\tilde{d}_k = 1.0$; $Z_k = 4(1.25)(1.0) = 5.0$ for $k \in \{2, 3, 4\}$. IFFT peak at index $m = 0$; signed lag $\tau[0] = 0.0\text{ s}$; value $z[0] = 16 \times \frac{1}{16}(5.0 + 5.0 + 5.0) = 15.0$. Direct sum $= 15.0$.
- **Positive Lag ($t_c = +0.15\text{ s} = +3 \Delta t$):** Data $\tilde{d}_k = e^{-i 2\pi f_k (0.15)} = e^{-i 2\pi k \cdot 3 / 16}$. $Z_k = 5.0 e^{-i 2\pi k \cdot 3 / 16}$. Peak occurs at index $m = 3$; signed lag $\tau[3] = +0.15\text{ s}$; value $z[3] = 15.0$. Direct sum $= 15.0$.
- **Negative Lag ($t_c = -0.15\text{ s} = -3 \Delta t$):** Data $\tilde{d}_k = e^{+i 2\pi k \cdot 3 / 16}$. $Z_k = 5.0 e^{+i 2\pi k \cdot 3 / 16}$. Peak occurs at circular index $m = 16 - 3 = 13$; signed lag $\tau[13] = (13 - 16)\Delta t = -0.15\text{ s}$; value $z[13] = 15.0$. Direct sum $= 15.0$.
- **Odd-$N$ Negative Lag ($N = 15, \Delta t = 0.05\text{ s}, t_c = -0.10\text{ s} = -2 \Delta t$):** Peak at circular index $m = 15 - 2 = 13$; signed lag $\tau[13] = (13 - 15)\Delta t = -0.10\text{ s}$.

---

### 3.5 Complex Output Normalization & Statistics
`[MATHEMATICALLY DERIVED]`

Let $d(t) = n(t)$ be stationary Gaussian noise with $\mathbb{E}[|\tilde{d}_k|^2] = \frac{S_n(f_k)}{2 \Delta f}$ and $\mathbb{E}[\tilde{d}_k^2] = 0$.
The complex matched filter output is $z(t) = 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{d}_k \tilde{h}_k^*}{S_n(f_k)} e^{i 2\pi f_k t} = x(t) + i y(t)$.
$$\mathbb{E}[|z(t)|^2] = 2 \sigma_h^2, \qquad \mathbb{E}[z(t)^2] = 0$$
$$\implies \operatorname{Var}(x(t)) = \sigma_h^2, \quad \operatorname{Var}(y(t)) = \sigma_h^2, \quad \operatorname{Cov}(x(t), y(t)) = 0$$
Under the adopted authoritative convention:
$$\rho(t) \equiv \frac{z(t)}{\sigma_h} = \hat{\rho}(t) + i \hat{\rho}_{\pi/2}(t)$$
1. **Real Quadratures:** $\hat{\rho}(t) \sim \mathcal{N}(0, 1)$ and $\hat{\rho}_{\pi/2}(t) \sim \mathcal{N}(0, 1)$ are independent standard normal variables.
2. **Phase-Maximized SNR:** $\rho_{\mathrm{max}}(t) \equiv |\rho(t)| = \frac{|z(t)|}{\sigma_h} \sim \operatorname{Rayleigh}(\text{scale } 1)$, with:
   $$\mathbb{E}[\rho_{\mathrm{max}}] = \sqrt{\frac{\pi}{2}} \approx 1.2533, \qquad \operatorname{Var}(\rho_{\mathrm{max}}) = 2 - \frac{\pi}{2} \approx 0.4292$$
3. **Squared Magnitude:** $\rho_{\mathrm{max}}^2(t) \sim \chi^2_2 \equiv \operatorname{Exponential}(\text{scale } 2)$, with $\mathbb{E}[\rho_{\mathrm{max}}^2] = 2, \operatorname{Var}(\rho_{\mathrm{max}}^2) = 4$.

---

### 3.6 Parameter Mismatch Formulation
`[MATHEMATICALLY DERIVED]`

Let detector data contain signal $s(t)$ and Gaussian noise: $d(t) = s(t) + n(t)$. The filter is evaluated with template $h(t)$.  
Define the distinct norms:
$$\sigma_s \equiv \sqrt{(s|s)}_{\mathcal{K}_+} = \rho_{\mathrm{opt}, s}, \qquad \sigma_h \equiv \sqrt{(h|h)}_{\mathcal{K}_+}$$
The noise-weighted complex cross-correlation is:
$$\langle s | h \rangle_{\mathcal{K}_+}(\tau) \equiv 4 \Delta f \sum_{k \in \mathcal{K}_+} \frac{\tilde{s}(f_k) \tilde{h}_0^*(f_k)}{S_n(f_k)} e^{i 2\pi f_k \tau}$$
Define normalized complex overlap:
$$\mathcal{O}_{\mathbb{C}}(s, h)(\tau) \equiv \frac{\langle s | h \rangle_{\mathcal{K}_+}(\tau)}{\sigma_s \sigma_h}, \qquad |\mathcal{O}_{\mathbb{C}}(s, h)(\tau)| \le 1$$
In the absence of noise ($d = s$), normalized complex SNR is:
$$\rho(t) = \frac{\langle s | h \rangle_{\mathcal{K}_+}(t)}{\sigma_h} = \sigma_s \mathcal{O}_{\mathbb{C}}(s, h)(t)$$
Phase-maximized SNR is:
$$\rho_{\mathrm{max}}(t) = |\rho(t)| = \sigma_s |\mathcal{O}_{\mathbb{C}}(s, h)(t)| = \rho_{\mathrm{opt}, s} \cdot |\mathcal{O}_{\mathbb{C}}(s, h)(t)|$$
Peak SNR over all lags is:
$$\rho_{\mathrm{peak}} = \rho_{\mathrm{opt}, s} \cdot \max_\tau |\mathcal{O}_{\mathbb{C}}(s, h)(\tau)| \equiv \rho_{\mathrm{opt}, s} \cdot \mathcal{M}(s, h)$$
where $\mathcal{M}(s, h) \in [0, 1]$ is the match, without assuming $\sigma_s = \sigma_h$.

---

### 3.7 Allen Time-Frequency $\chi^2$: Statistical Foundations and Glitch Clarification
`[MATHEMATICALLY DERIVED]`

#### 3.7.1 Active Sub-Band Partition and Degenerate-Bin Handling
Partition active band $\mathcal{K}_+$ into $p$ nominal contiguous sub-bands $\mathcal{K}_{+, 1}, \dots, \mathcal{K}_{+, p}$.  
The template power in sub-band $l$ is $\sigma_{h, l}^2 = 4 \Delta f \sum_{k \in \mathcal{K}_{+, l}} \frac{|\tilde{h}(f_k)|^2}{S_n(f_k)}$.  
To prevent division by zero, define the relative power threshold $\epsilon_{\mathrm{tol}} = 10^{-12}$.  
Define active index set $\mathcal{P} \equiv \{l \in \{1, \dots, p\} : \sigma_{h, l}^2 > \epsilon_{\mathrm{tol}} \sigma_h^2\}$ with $q \equiv |\mathcal{P}|$.
1. **Case $q = p$:** Standard evaluation with $\nu = 2p - 2$.
2. **Case $2 \le q < p$:** Zero-power bins are merged into adjacent non-zero bins, preserving disjoint frequency support. Weights re-scale to $\tilde{w}_l = \tilde{\sigma}_{h, l}^2 / \sigma_h^2$ ($\sum_{l=1}^q \tilde{w}_l = 1$). The degrees of freedom are strictly $\nu = 2q - 2$.
3. **Degenerate Case $q < 2$ ($q = 1$ or $q = 0$):** All power is in $\le 1$ bin; residuals vanish identically ($\Delta z_1 = 0$) with $\nu = 0$. The Allen $\chi^2$ statistic is **mathematically undefined and unavailable**. The API sets `chisq = None`, `chisq_dof = None`, and `reweighted_snr = rho_max`.

#### 3.7.2 Assumptions for the Central $\chi^2_{2q - 2}$ Distribution
`[MATHEMATICALLY DERIVED]`
Under the specified Gaussian covariance model and rank assumptions, the central distribution $\chi^2 \sim \chi^2_{2q - 2}$ holds under the following fundamental statistical conditions:
1. **Gaussian Stationary Noise Covariance:** Detector noise $n(t)$ is stationary Gaussian noise, so sub-band noise outputs $z_l$ are circular complex Gaussians satisfying the diagonal covariance model $\mathbb{E}[(z_l - \mathbb{E}[z_l]) (z_m - \mathbb{E}[z_m])^*] = 2 \tilde{\sigma}_{h, l}^2 \delta_{lm}$.
2. **Strictly Positive Sub-Band Powers and Rank:** All $q \ge 2$ retained sub-bands satisfy $\tilde{\sigma}_{h, l}^2 > 0$. The orthogonal projection matrix onto the residual subspace, $\mathbf{P} = \mathbf{I}_q - \mathbf{v} \mathbf{v}^\dagger$ (where $v_l = \tilde{\sigma}_{h, l} / \sigma_h$), has rank $q - 1$, yielding strictly $\nu = 2(q - 1) = 2q - 2$ effective degrees of freedom.
3. **Zero Projected Mean (Fundamental Statistical Condition):** The projected residual vector has zero deterministic mean:
   $$\mathbb{E}[\Delta \mathbf{z}(t)] = \mathbf{0} \iff \mathbb{E}[z_l(t)] = \tilde{w}_l \mathbb{E}[z(t)] \quad \text{for all } l \in \{1, \dots, q\}$$
   such that the deterministic signal expectation vector $\mathbb{E}[\mathbf{z}(t)]$ lies entirely in the null space of the residual projection operator $\mathbf{P}$, ensuring non-centrality parameter $\lambda = 0$.
   - **Sufficient Physical Condition:** Exact signal-template morphology matching across all retained sub-bands ($s(t) = A h(t - \tau) e^{i \phi_0}$) is a **sufficient physical condition** to yield this zero-projected-mean result ($\mathbb{E}[\Delta \mathbf{z}(t)] = \mathbf{0}$). It is not automatically necessary in every possible case: a deterministic mismatch whose mean residual vector lies entirely in the statistic's null space may also yield zero projected mean.

#### 3.7.3 Non-Central Distribution vs. Arbitrary Glitches
`[MATHEMATICALLY DERIVED]`
1. **Deterministic Waveform Mismatch and Non-Central Distribution:** Under the specified Gaussian covariance model, if the data contains a deterministic signal component $s(t)$ (arising from physical waveform mismatch, unmodeled physics such as precession, eccentricity, or higher-order modes, or calibration errors) such that the projected residual vector has a **nonzero deterministic mean**:
   $$\mathbb{E}[\Delta \mathbf{z}(t)] = \boldsymbol{\mu}_{\Delta \mathbf{z}}(t) \ne \mathbf{0}$$
   (i.e. the deterministic signal response does not lie entirely in the statistic's null space), then $\chi^2(t)$ follows a **non-central $\chi^2$ distribution**:
   $$\chi^2(t) \sim \chi^2_{2q - 2}(\lambda(t)), \qquad \lambda(t) = \sum_{l=1}^q \frac{|\mathbb{E}[z_l(t)] - \tilde{w}_l \mathbb{E}[z(t)]|^2}{\tilde{\sigma}_{h, l}^2} > 0$$
   with expectation value $\mathbb{E}[\chi^2(t)] = 2q - 2 + \lambda(t)$.
2. **Non-Gaussian Instrumental Glitches:** Non-Gaussian detector artifacts (blips, scattered light, whistles) violate the Gaussian noise model itself. **Glitches do NOT generally follow a central or non-central $\chi^2$ distribution.** Their residual distributions depend on the specific non-Gaussian glitch process and must be modeled or empirically characterized separately (e.g. via time slides or glitch morphology studies). We strictly do not assign a central or non-central $\chi^2$ distribution to arbitrary glitch-contaminated data.

#### 3.7.4 Consistent Reweighted SNR Formulation
$$\tilde{\rho}(t) = \begin{cases} \rho_{\mathrm{max}}(t), & \chi_r^2(t) \le 1 \\ \frac{\rho_{\mathrm{max}}(t)}{\left[ \frac{1 + (\chi_r^2(t))^3}{2} \right]^{1/6}}, & \chi_r^2(t) > 1 \end{cases}, \qquad \chi_r^2(t) \equiv \frac{\chi^2(t)}{2q - 2}$$
When $q < 2$, $\tilde{\rho}(t) = \rho_{\mathrm{max}}(t)$ unconditionally.

---

### 3.8 Explicit PSD Scaling Invariance
`[MATHEMATICALLY DERIVED]`

Hold time-domain data $d(t)$ and template $h(t)$ strictly fixed. Scale the one-sided PSD by constant $c > 0$:
$$S_n'(f) = c \cdot S_n(f)$$
$$z'(t) = \frac{z(t)}{c}, \qquad \sigma_h' = \frac{\sigma_h}{\sqrt{c}}, \qquad \rho'(t) = \frac{\rho(t)}{\sqrt{c}}, \qquad \rho_{\mathrm{max}}'(t) = \frac{\rho_{\mathrm{max}}(t)}{\sqrt{c}}, \qquad (\chi^2)' = \frac{\chi^2}{c}$$

---

### 3.9 Time-Scaled DFT Injection Reconstruction
`[MATHEMATICALLY DERIVED]`

In `gw_inference/conventions.py` (L153), continuous Fourier transform scaling is $\tilde{X}_k = \Delta t \cdot \operatorname{rfft}(x)_k = \Delta t H_k$.  
Because NumPy's `irfft(H, n=N)` inverts `rfft(x)` exactly:
$$x(t_j) = \operatorname{irfft}\left( \frac{\tilde{X}}{\Delta t}, n=N \right)_j = \frac{1}{\Delta t} \operatorname{irfft}(\tilde{X}, n=N)_j = N \Delta f \cdot \operatorname{irfft}(\tilde{X}, n=N)_j$$
The exact algebraic round-trip identity holds to float64 precision ($< 10^{-15}$ relative error):
$$\frac{1}{\Delta t} \operatorname{irfft}(\texttt{continuous\_rfft}(x, \Delta t), n=N) \equiv x$$

---

## 4. Proposed Numerical API Signatures

`[PROPOSED]`

```python
"""Proposed WP3 API Specification (src/gw_inference/filter.py)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from gw_inference.grid import FrequencyGrid
from gw_inference.psd import PSDEngine


@dataclass(frozen=True)
class MatchedFilterResult:
    """Immutable container for matched-filter outputs and search diagnostics.

    Attributes
    ----------
    snr_time_series : np.ndarray
        1D complex128 array rho(t_j) = z(t_j) / sigma_h of length N.
        Re[rho] is 0-phase SNR; Im[rho] is pi/2-phase SNR.
    snr_max : np.ndarray
        1D float64 array of phase-maximized SNR |rho(t_j)| of length N.
    phase_max : np.ndarray
        1D float64 array of best-fit coalescence phase in radians in (-pi, pi].
    signed_lags : np.ndarray
        1D float64 array of signed physical time lags in seconds [-T/2, T/2).
    circular_lags : np.ndarray
        1D float64 array of circular time lags in seconds [0, T).
    valid_mask : np.ndarray
        1D bool array indicating valid lags under the operational boundary guard.
    sigma_h : float
        Scalar template norm (standard deviation) in active band.
    optimal_snr : float
        Optimal SNR rho_opt = sigma_h.
    peak_snr : float
        Maximum phase-maximized SNR over valid uncorrupted lags.
    peak_time : float
        Physical time lag in seconds corresponding to peak_snr.
    peak_phase : float
        Best-fit coalescence phase in radians corresponding to peak_snr.
    chisq : Optional[np.ndarray]
        1D float64 array of binned Allen chi^2 time series, if requested and q >= 2.
    chisq_dof : Optional[int]
        Effective degrees of freedom nu = 2q - 2, if chisq available.
    reweighted_snr : Optional[np.ndarray]
        1D float64 array of reweighted SNR time series, if chisq requested.
    """
    snr_time_series: np.ndarray
    snr_max: np.ndarray
    phase_max: np.ndarray
    signed_lags: np.ndarray
    circular_lags: np.ndarray
    valid_mask: np.ndarray
    sigma_h: float
    optimal_snr: float
    peak_snr: float
    peak_time: float
    peak_phase: float
    chisq: Optional[np.ndarray] = None
    chisq_dof: Optional[int] = None
    reweighted_snr: Optional[np.ndarray] = None


def compute_template_norm(
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    delta_f: float,
) -> float:
    """Compute active-band template norm sigma_h = sqrt((h|h)_K+)."""
    ...


def matched_filter_core(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    n_samples: int,
    delta_f: float,
) -> Tuple[np.ndarray, float]:
    """Execute complex matched-filter correlation via N-point analytic IFFT."""
    ...


def compute_signed_lags(n_samples: int, delta_t: float) -> np.ndarray:
    """Generate signed physical time lag array for even or odd N."""
    ...


def matched_filter(
    data: np.ndarray,
    template_fd: np.ndarray,
    grid: FrequencyGrid,
    psd_engine: PSDEngine,
    mask: np.ndarray,
    f_start: float,
    f_0: float,
    tau_chirp: Optional[float] = None,
    tau_filter_diag: float = 0.2,
    compute_chisq: bool = False,
    chisq_bins: int = 16,
) -> MatchedFilterResult:
    """High-level authoritative matched-filtering entry point."""
    ...


def direct_matched_filter_at_lag(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    freqs: np.ndarray,
    delta_f: float,
    time_lag: float,
) -> complex:
    """Evaluate complex matched filter at a specified lag via direct sum."""
    ...


def compute_chisq_binned(
    data_fd: np.ndarray,
    template_fd: np.ndarray,
    inv_psd_active: np.ndarray,
    active_indices: np.ndarray,
    freqs: np.ndarray,
    n_samples: int,
    delta_f: float,
    n_bins: int = 16,
    power_tol: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Compute Allen binned chi^2 with adaptive partition merging for degenerate bins."""
    ...
```

---

## 5. Revised Validation Matrix (M1–M12) & M7 Acceptance Procedure

`[PROPOSED]`

### 5.1 Test M7: Formal Marginal Statistical Acceptance Procedure Specification
`[PROPOSED]`
- **Null Hypothesis ($H_0$):** Stationary zero-mean Gaussian noise colored by $S_n(f)$. At fixed lag $t_0$, normalized quadratures satisfy $X = \hat{\rho}(t_0) \sim \mathcal{N}(0, 1)$ and $Y = \hat{\rho}_{\pi/2}(t_0) \sim \mathcal{N}(0, 1)$ independently, and magnitude $R = \sqrt{X^2 + Y^2} \sim \operatorname{Rayleigh}(1)$.
- **Monte Carlo Specification:** $M = 10,000$ independent realizations generated via `np.random.default_rng(seed=123456789)`.
- **Status Classification:** The numerical bounds below are **theoretical marginal acceptance intervals** ($1 - \alpha = 0.99$, two-sided $\alpha/2 = 0.005$). They are **not** a joint six-dimensional 99% confidence region (by Bonferroni bound, joint coverage is $\ge 94\%$). They are specified strictly as pre-implementation test criteria, not completed empirical measurements.

```
+---------------------------+-----------------------------------+-----------------------+---------------------------------------+
| Test Statistic            | Formula                           | Null Distribution     | Theoretical 99% Marginal Acceptance   |
+---------------------------+-----------------------------------+-----------------------+---------------------------------------+
| Real Quadrature Mean      | \bar{X} = (1/M) \sum X_i          | N(0, 1/M), SE = 0.01  | [-0.0258, +0.0258]                    |
| Imaginary Quadrature Mean | \bar{Y} = (1/M) \sum Y_i          | N(0, 1/M), SE = 0.01  | [-0.0258, +0.0258]                    |
| Real Quadrature Variance  | S_X^2 = (1/(M-1)) \sum (X_i-\bar{X})^2 | \chi^2_{9999}/9999   | [0.9639, 1.0368] (exact \chi^2 quant.)|
| Imaginary Quadrature Var. | S_Y^2 = (1/(M-1)) \sum (Y_i-\bar{Y})^2 | \chi^2_{9999}/9999   | [0.9639, 1.0368] (exact \chi^2 quant.)|
| Real-Imag. Covariance     | C_XY = (1/(M-1)) \sum (X_i-\bar{X})(Y_i-\bar{Y}) | Var \approx 1/M | [-0.0258, +0.0258] (approx. normal)   |
| Rayleigh Magnitude Mean   | \bar{R} = (1/M) \sum R_i          | Var=(2-\pi/2)/M       | [1.2364, 1.2702] (approx. normal)     |
+---------------------------+-----------------------------------+-----------------------+---------------------------------------+
```
**Acceptance Rule:** Test M7 passes if and only if each sample statistic falls strictly within its respective theoretical 99% marginal acceptance interval.

### 5.2 Complete Validation Matrix (M1–M12)

```
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| ID | Test Description                 | Mathematical Construction                     | Analytical Expectation | Justified Numerical Tolerance         |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M1 | Direct Sum vs. IFFT Equivalence  | Compare matched_filter_core against direct sum| Identical complex      | Relative discrepancy                  |
|    | at Signed Lags                   | at tau = 0, +10 dt, and -10 dt                | values at grid points  | |z_fft - z_dir| / |z_dir| < 1.0e-12   |
|    |                                  |                                               |                        | [Deterministic; float64 roundoff]     |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M2 | Zero-Lag Time-Domain Injection   | d(t) = (1/dt) * irfft(h_tilde, n=N);          | Peak at lag 0;         | |rho_max(0) - rho_opt| / rho_opt      |
|    | Recovery via Exact Reconstruction| filter with exact template h_tilde            | rho_max = rho_opt;     | < 1.0e-12;                            |
|    |                                  |                                               | phi_hat(0) = 0.0 rad   | |phi_hat(0)| < 1.0e-12 rad            |
|    |                                  |                                               |                        | [Deterministic; round-trip identity]  |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M3 | Cyclic vs. Linear Time Shift     | 1. Cyclic shift: d_j = d_orig[(j-k)%N]        | 1. Cyclic: peak at k;  | 1. Peak index exact; SNR error < 1e-12|
|    | Recovery                         | 2. Linear shift: padded signal inside segment | 2. Linear: peak at t_s | 2. SNR loss < 1.0e-4 (finite-band)    |
|    |                                  |                                               | without wrap loss      | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M4 | Coalescence Phase Sweep & Sign   | Inject signals with phi_0 in {0, pi/4, pi/2,  | rho_max = rho_opt;     | |rho_max - rho_opt| / rho_opt < 1.0e-12|
|    | Verification                     | 3pi/4, pi, 3pi/2}; template has phi_c = 0.0   | phi_hat == phi_0 mod 2pi| |phi_hat - phi_0| < 1.0e-12 rad       |
|    |                                  |                                               |                        | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M5 | Template Normalization Invariance| Scale template by alpha in {0.1, 10.0, 10^5}; | (h_norm|h_norm) = 1.0; | |sigma_{h_norm} - 1.0| < 1.0e-14;     |
|    |                                  | h_norm = h / sigma_h; re-evaluate filter      | rho_max invariant      | |rho_alpha - rho_orig| < 1.0e-14      |
|    |                                  |                                               |                        | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M6 | Odd/Even-N Grid Parity & Signed  | Evaluate filter on even N (16384) and odd N   | Both parities evaluate | 0 errors, 0 NaN/Inf;                  |
|    | Lag Range Verification           | (16385); check signed lag bounds and Nyquist  | cleanly; Nyquist absent| signed lag bounds match theoretical   |
|    |                                  |                                               | on odd N               | expressions exactly [Deterministic]   |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M7 | Multi-Parameter Gaussian Noise   | N_trials = 10,000 independent noise trials;   | Quadratures ~ N(0, 1); | All 6 theoretical 99% marginal        |
|    | Ensemble Acceptance Testing      | seed = 123456789; test quadratures and mag    | Cov(Re, Im) == 0;      | acceptance intervals satisfied        |
|    |                                  | independently                                 | rho_max ~ Rayleigh(1)  | [Statistical; specified in §5.1]      |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M8 | Parameter Mismatch with Separate | Inject signal s with norm sigma_s; filter with| rho_max = rho_opt,s *  | |rho_max / sigma_s - M(s, h)|         |
|    | Signal and Template Norms        | template h with norm sigma_h; compute match   | M(s, h)                | < 1.0e-4                              |
|    |                                  | M(s, h) = max_tau |O_C(s, h)|                 |                        | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M9 | PSD Scaling Homogeneity on Fixed | Scale S_n -> c * S_n (c > 0) holding data and | z scales as 1/c;       | |z_c - z_orig / c| / |z_orig| < 1e-14;|
|    | Data and Template                | template strictly fixed                       | sigma_h as 1/sqrt(c);  | |rho_c - rho_orig / sqrt(c)| < 1e-14  |
|    |                                  |                                               | rho_max as 1/sqrt(c)   | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M10| Malformed Input Rejection Suite  | Pass NaN/Inf arrays, non-positive PSD,        | Raises descriptive     | Specific ValueError or TypeError;     |
|    |                                  | zero-norm template, missing mandatory f_0,    | exceptions             | no silent numerical corruption        |
|    |                                  | or mismatched array dimensions                |                        | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M11| Direct-Sum vs. Circular Boundary | Compare circular filter with direct ground-   | Difference bounded in  | |z_circ - z_direct| / sigma_h < 1e-3  |
|    | Verification Benchmark           | truth sum on valid interior band              | interior valid band    | in certified interval [Proposed]      |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
| M12| Active Mask Invariance Across    | Filter data across light and heavy systems    | K_+ bitwise identical; | Mask array unchanged;                 |
|    | Distinct Waveform Cutoffs        | (f_ISCO = 73.3 Hz vs. 1570 Hz)                | waveform zeroed above  | h_tilde == 0 for f > f_term           |
|    |                                  |                                               | f_term                 | [Deterministic]                       |
+----+----------------------------------+-----------------------------------------------+------------------------+---------------------------------------+
```

---

## 6. Updated Scientific Risk Register

`[MATHEMATICALLY DERIVED]` `[PROPOSED]`

```
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| ID | Failure Mode                          | Observable Symptom                    | Independent Diagnostic                | Prevention / Mitigation Architecture  |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R1 | Factor-of-Two Error in Discrete       | Recovered SNR is off by sqrt(2)       | Test M2: verify rho_max == rho_opt    | Rigorous derivation in §3.1: prefactor |
|    | Inner Product Prefactor               | or 2; (h|h) off by factor of 2 or 4   | to float precision                    | 4*delta_f*delta_t^2 = 4*delta_t/N.    |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R2 | Inverted Complex Conjugation          | Filter correlates with reversed phase;| Test M4: sweep phi_0; verify recovered| Enforce d_tilde * h_tilde^* in inner  |
|    | in Template Kernel                    | time lag has inverted sign or shape   | phase hat{phi}_c == phi_0 mod 2pi     | product definition; verify test M4.   |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R3 | Inverted Time-Lag Sign in FFT Kernel  | Peak SNR occurs at T - t_0 instead    | Test M1/M3: test signed lags +/- 10 dt| Analytical derivation in §3.4 shows   |
|    | (Using forward FFT instead of IFFT)   | of +t_0 (time-reversed lag)           | verify peak is at +10 dt, not -10 dt  | exponent is +i*2*pi*f*t -> use IFFT.  |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R4 | Altering Authoritative WP1 Inner      | Discrepancy in likelihood or SNR      | Test M1: compare FFT matched filter   | Direct finite frequency sum on WP1    |
|    | Product via Padded Frequency Grids    | when changing padded transform size   | against direct ground-truth sum on WP1| grid is authoritative reference.      |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R5 | Confusing Raw DFT and Continuous-     | Injection reconstruction off by       | Test M2/M4: test round-trip identity  | Reconstruct x(t) = (1/dt)*irfft(X)    |
|    | Scaled Transform Conventions          | factor of dt or N                     | (1/dt)*irfft(continuous_rfft(x)) == x | with rigorous dimensional checks.     |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R6 | Endpoint Overcounting (DC & Nyquist)  | Noise quadrature variance != 1.0;     | Test M7: verify noise variance of real| Active mask K_+ strictly excludes DC  |
|    | on One-Sided Grid                     | statistical distribution biased       | quadrature is in [0.9639, 1.0368]     | (k=0) and even-N Nyquist (k=N/2).     |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R7 | Dynamically Mutating Active Mask      | Normalization changes with mass;      | Test M12: verify K_+ is static and    | K_+ defined solely by [f_start, f_end]|
|    | Dependent on Waveform Cutoff          | inner products across params invalid  | identical across different m1, m2     | ISCO cutoff applied inside h~(f_k).   |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R8 | Division by Zero in Degenerate        | NaN or crash in Allen chi^2 when      | Test M10: inject template with tiny   | Define active set P = {l : p_l > tol};|
|    | Sub-Bands (Empty Template Power)      | bandwidth is small or power is zero   | bandwidth; assert chisq == None       | set chisq = None when q < 2.          |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R9 | Spurious Penalty on Clean Signals     | Clean inspirals with chi_r^2 <= 1     | Test Allen chi^2 formulation on clean | Adopt standard Allen (2005) piecewise |
|    | in Reweighted SNR                     | suffer artificial SNR reduction       | injection; verify rho_reweighted ==   | formula: rho_tilde = rho for chi_r<=1.|
|    |                                       |                                       | rho_max when chi_r^2 <= 1             |                                       |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
| R10| Misinterpreting Glitch Statistics as  | False confidence in chi^2 thresholds  | Document non-Gaussian glitch residual | Group E compliance: defer global FAP  |
|    | Following a Non-Central Chi^2 Law     | for arbitrary non-Gaussian transients | characterization to empirical studies | calibration to empirical background.  |
+----+---------------------------------------+---------------------------------------+---------------------------------------+---------------------------------------+
```

---

## 7. Issue Classification Table

`[EXISTING & SOURCE-VERIFIED]` `[MATHEMATICALLY DERIVED]` `[PROPOSED]`

```
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| No.| Reviewer Issue / Governance Topic                             | Status     | Resolution Summary / Governance Action                    |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 1  | WP1 grid authority vs. padded-grid inner products             | RESOLVED   | Established direct finite sum on WP1 grid as authoritative|
|    |                                                               |            | reference definition; deferred padded-grid implementation |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 2  | Circular boundary convergence certification                   | DEFERRED   | Deferred quantitative circular certification pending      |
|    |                                                               |            | empirical benchmarks against direct WP1 ground-truth sum  |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 3  | Allen chi-square glitch vs. mismatch distribution             | RESOLVED   | Clarified non-central chi^2 applies to deterministic      |
|    |                                                               |            | mismatch; non-Gaussian glitches not assigned chi^2 law    |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 4  | Allen chi-square degenerate-bin handling (q < 2)              | RESOLVED   | Defined active set P = {l : sigma_l^2 > tol}; proved      |
|    |                                                               |            | dof=2q-2; set chisq=None when q < 2 (not silent zero)     |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 5  | M7 theoretical marginal acceptance procedure specification    | RESOLVED   | Formulated marginal 99% acceptance intervals with exact   |
|    |                                                               |            | \chi^2_{9999} quantiles; distinguished from empirical runs|
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 6  | Full computational FFT/IFFT lag construction                  | RESOLVED   | Specified positive-frequency complex N-point IFFT         |
|    |                                                               |            | embedding; proved insufficiency of real irfft; N=15,16    |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 7  | Parameter mismatch relation with separate norms               | RESOLVED   | Proved rho_max = rho_opt,s * |O_C(s,h)| with separate     |
|    |                                                               |            | signal norm sigma_s and template norm sigma_h             |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 8  | Fixed-data, fixed-template PSD scaling transformation         | RESOLVED   | Verified z -> z/c, sigma_h -> sigma_h/sqrt(c), and        |
|    |                                                               |            | rho_max -> rho_max/sqrt(c) under S_n -> c*S_n (c > 0)     |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 9  | Whittle/Toeplitz wording                                      | RESOLVED   | Formulated strictly as active-band Whittle/circulant      |
|    |                                                               |            | approximation under frozen project baseline               |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 10 | Decision D8 (f_0 low-frequency PSD continuation scale)        | UNRESOLVED | Decision D8 remains open; f_0 is strictly mandatory with  |
|    |                                                               |            | no default across all proposed public signatures          |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 11 | Whittle likelihood for Bayesian MCMC parameter estimation     | DEFERRED   | Deferred to Work Package 4 (WP4)                          |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
| 12 | Bank-wide metric template placement and time-slide FAR        | DEFERRED   | Deferred to Stage 2 (Detection Pipeline)                  |
+----+---------------------------------------------------------------+------------+-----------------------------------------------------------+
```

---

## 8. Pre-Implementation Readiness Verdict

```
+---------------------------------------------------------------------------------------+-----------+
| Inspection & Specification Gate Criteria                                              | Status    |
+---------------------------------------------------------------------------------------+-----------+
| 1. Active branch confirmed as milestone-1.4a and working tree clean                   | VERIFIED  |
| 2. Baseline commit confirmed as ff91df71d3c57a64313a3537f523808f789f59d3             | VERIFIED  |
| 3. All 200 WP0–WP2 unit tests confirmed passing with zero failures/warnings           | VERIFIED  |
| 4. Authoritative WP1 observation grid preserved; direct finite sum is ground truth    | COMPLETED |
| 5. Circular boundary convergence benchmark properly classified as deferred           | COMPLETED |
| 6. Allen chi^2 non-centrality restricted to mismatch; glitches not assigned chi^2 law | COMPLETED |
| 7. Degenerate Allen chi^2 defined with active set P; chisq unavailable when q < 2      | COMPLETED |
| 8. M7 theoretical marginal 99% acceptance intervals formulated with exact quantiles   | COMPLETED |
| 9. Computational N-point analytic IFFT embedding specified with exact 4*dt/N scaling  | COMPLETED |
| 10. Parameter mismatch relation derived with separate signal and template norms        | COMPLETED |
| 11. Fixed-data PSD scaling transformation verified analytically                       | COMPLETED |
| 12. Zero implementation code, test code, or commits executed in this turn            | COMPLETED |
+---------------------------------------------------------------------------------------+-----------+
```

### Pre-Implementation Readiness Verdict
**REVISION 6 IS READY FOR INDEPENDENT SCIENTIFIC REVIEW.**

All targeted corrections and qualifications from independent scientific review have been incorporated. The specification preserves the frozen WP1 observation grid as authoritative; properly classifies padded implementations and circular boundary certification as deferred; correctly bounds non-central $\chi^2$ claims away from arbitrary non-Gaussian glitches; and maintains strict report-only discipline.

**Implementation Authorization:** NOT GRANTED (Awaiting Independent Reviewer Sign-Off).
