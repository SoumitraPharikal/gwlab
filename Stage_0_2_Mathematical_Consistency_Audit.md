# Stage 0.2: Mathematical Consistency and Experimental Protocol Audit
## Robust Matched Filtering and Bayesian Parameter Estimation for Compact Binary Gravitational-Wave Signals

---

### 1. Correct GW150914 Data Interval and Conditioning Design

#### 1.1 Authoritative Event Parameters
The parameters for event GW150914 have been cross-referenced against the Gravitational Wave Open Science Center (GWOSC) and official discovery publications (Abbott et al. 2016, *Phys. Rev. Lett.* 116, 061102; Abbott et al. 2016, *Phys. Rev. Lett.* 116, 241102):
* **Official Event Identifier:** GW150914
* **Catalog Release:** GWTC-1-confident / O1 Cleaned Strain (Calibration Version C02)
* **Authoritative Geocentric Coalescence GPS Time:** $t_c = 1126259462.42$ UTC (2015-09-14 09:50:45.42 UTC)
* **Active Interferometer Network:** LIGO Hanford (H1) and LIGO Livingston (L1)
* **Official GWOSC Archive URL:** `https://gwosc.org/events/GW150914/` (Access Date: 2026-11-02)
* **Official 4096-Second Strain File Specification:**
  * Hanford: `H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5`
  * Livingston: `L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5`
  * Official GPS Start Timestamp: $1126257415.0$
  * Official GPS End Timestamp: $1126261511.0$ ($1126257415 + 4096$)
  * Native Sampling Rate: $4096 \, \mathrm{Hz}$ ($f_s = 4096 \, \mathrm{Hz}, \Delta t = 2.44140625 \times 10^{-4} \, \mathrm{s}$)

#### 1.2 Resolution of the Overlap Error and Timeline Specification
In the Stage 0.1 submission, the off-source interval was erroneously stated as GPS $[1126258438, 1126259462]$ and the on-source segment as GPS $[1126259458, 1126259466]$, resulting in a four-second overlap ($[1126259458, 1126259462]$) containing the physical signal. This would have caused signal power to contaminate the PSD estimate, biasing the noise metric and inducing self-subtraction.

The corrected timeline establishes strictly disjoint intervals separated by a verified guard interval:

```
                                  GW150914 TIMELINE SPECIFICATION
  GPS Time:
  1126257415             1126258418                     1126259442             1126259458      1126259462.42   1126259466
       |----------------------|------------------------------|----------------------|----------------*-------------|
       |<- Pre-Lock (1003s) ->|<-- Off-Source PSD (1024s) -->|<- Guard Int. (16s) ->|<-- On-Source Segment (8s) -->|
                              GPS: [1126258418, 1126259442]   GPS: [1126259442, 1126259458]  GPS: [1126259458, 1126259466]
```

* **Off-Source PSD Interval:** GPS $[1126258418.0, 1126259442.0]$
  * Duration: Exactly $1024.0 \, \mathrm{s}$ ($N = 4,194,304$ samples).
  * Data Availability: Resides entirely within the archival lock stretch ($1126258418 > 1126257415$).
* **Guard Interval:** GPS $[1126259442.0, 1126259458.0]$
  * Duration: Exactly $16.0 \, \mathrm{s}$ ($N = 65,536$ samples).
  * Scientific Role: Excludes any transient signal power and ensures that filter impulse-response tails from time-domain conditioning do not leak into the off-source noise sample.
* **On-Source Analysis Segment:** GPS $[1126259458.0, 1126259466.0]$
  * Duration: Exactly $8.0 \, \mathrm{s}$ ($N = 32,768$ samples).
  * Coalescence Placement: Event trigger occurs at $t = 4.42 \, \mathrm{s}$ into the segment ($1126259462.42$).
  * Overlap with Off-Source: Exactly $0.0 \, \mathrm{s}$ (separated by $16.0 \, \mathrm{s}$ of unanalyzed strain).

#### 1.3 Signal Duration and Tukey Flat-Top Containment
Under the leading-order 0PN post-Newtonian approximation, the time $\tau(f_{\min})$ required for a quasi-circular compact binary to sweep from frequency $f_{\min}$ to merger is:
$$\tau(f_{\min}) = \frac{5}{256} \frac{c^5}{(G \mathcal{M}_{\mathrm{det}})^{5/3}} (\pi f_{\min})^{-8/3}$$
For GW150914 with detector-frame chirp mass $\mathcal{M}_{\mathrm{det}} \approx 31.0 \, M_\odot$ ($G \mathcal{M} / c^3 \approx 1.527 \times 10^{-4} \, \mathrm{s}$):
* From $f_{\min} = 30.0 \, \mathrm{Hz}$: $\tau(30 \, \mathrm{Hz}) \approx 0.23 \, \mathrm{s}$.
* From $f_{\min} = 20.0 \, \mathrm{Hz}$: $\tau(20 \, \mathrm{Hz}) \approx 0.69 \, \mathrm{s}$.

Applying an 8-second Tukey window with roll-off $\alpha = 0.25$:
* Boundary taper regions: $t \in [0.0, 1.0]\text{ s}$ and $t \in [7.0, 8.0]\text{ s}$ ($1.0\text{ s}$ cosine roll-off at each boundary).
* Flat-top region ($w(t) \equiv 1.0$): $t \in [1.0, 7.0]\text{ s}$ (GPS $[1126259459.0, 1126259465.0]$, duration: $6.0\text{ s}$).
* Physical Signal Containment: Coalescence sits at $t = 4.42\text{ s}$. The start of detectable inspiral from $20\text{ Hz}$ occurs at $t \approx 4.42 - 0.69 = 3.73\text{ s}$, which is $2.73\text{ s}$ after the leading taper terminates. Ringdown terminates within $0.05\text{ s}$ after merger ($t \approx 4.47\text{ s}$), which is $2.53\text{ s}$ before the trailing taper begins.
* **Sufficiency of the 16-Second Guard Interval:** The $16\text{-second}$ guard interval exceeds the physical duration from $20\text{ Hz}$ ($0.69\text{ s}$) by a factor of 23. It completely isolates off-source data from the signal.

#### 1.4 Distinction Between Verified Metadata and Unverified Strain Properties
* **Verified from Official Metadata:** Data files exist in the GWOSC repository; H1 and L1 were in nominal science lock; start and end timestamps match official archival releases.
* **Unverified (Provisional Status):** The absence of local instrumental glitches, microseismic noise non-stationarity, and line dropouts within the specific $[1126258418, 1126259442]$ interval cannot be established from metadata alone. Direct inspection of the strain time series and data-quality channels during Stage 1 is required before finalizing this interval.

---

### 2. Derivation of Phase Quadrature Conventions from First Principles

#### 2.1 Orbital Phase vs. Gravitational-Wave Phase
Let a compact binary undergo quasi-circular inspiral. To eliminate phase-origin ambiguities:
1. Define the reference orbital phase evolution $\phi(t)$ relative to coalescence time $t_c$:
   $$\phi(t) \equiv \int_{t_c}^{t} \Omega_{\mathrm{orb}}(t') \, dt', \quad \text{such that } \phi(t_c) = 0$$
   Because the orbital frequency $\Omega_{\mathrm{orb}}(t) > 0$, $\phi(t)$ is negative during inspiral ($t < t_c$) and increases monotonically to $\phi(t_c) = 0$.
2. Let $\phi_c \in [0, \pi)$ denote the orbital phase of the binary at coalescence: $\Phi_{\mathrm{orb}}(t_c) = \phi_c$. The total orbital phase is:
   $$\Phi_{\mathrm{orb}}(t) = \phi(t) + \phi_c$$
3. For the dominant quadrupole mode $(\ell, |m|) = (2, 2)$, mass-quadrupole radiation generates two wave cycles per orbital revolution. The gravitational-wave phase is:
   $$\Phi_{\mathrm{gw}}(t) = 2 \Phi_{\mathrm{orb}}(t) = 2\phi(t) + 2\phi_c$$
   The gravitational-wave phase at coalescence is $\phi_{c, \mathrm{gw}} \equiv 2\phi_c \in [0, 2\pi)$.

#### 2.2 Time-Domain Waveform and Quadrature Templates
Projecting onto an interferometric detector, the measured strain for the $(2, 2)$ mode with coalescence phase lag $-2\phi_c$ is:
$$h(t; \phi_c) = A(t) \cos\left( 2\phi(t) - 2\phi_c \right)$$
where $A(t) > 0$ is the slowly varying post-Newtonian amplitude envelope.
Expanding via the cosine angle-subtraction formula:
$$h(t; \phi_c) = A(t) \cos(2\phi(t)) \cos(2\phi_c) + A(t) \sin(2\phi(t)) \sin(2\phi_c)$$
We define the two real physical quadrature templates:
* **Reference Template ($0$-phase quadrature, $\phi_c = 0$):**
  $$h_0(t) \equiv h(t; \phi_c = 0) = A(t) \cos(2\phi(t))$$
* **Orthogonal Quadrature Template ($\pi/2$ GW phase lag, $\phi_c = \pi/4$):**
  $$h_{\pi/2}(t) \equiv h(t; \phi_c = \pi/4) = A(t) \cos\left( 2\phi(t) - \frac{\pi}{2} \right) = A(t) \sin(2\phi(t))$$
* **Factor of Two:** Shifting the gravitational-wave phase by $\pi/2$ corresponds to an orbital phase advance of $\Delta \phi_c = \pi/4$.
The exact time-domain decomposition is:
$$h(t; \phi_c) = h_0(t) \cos(2\phi_c) + h_{\pi/2}(t) \sin(2\phi_c)$$

#### 2.3 Positive-Frequency Fourier Transform Relationship
Under the continuous Fourier transform convention:
$$\tilde{x}(f) = \int_{-\infty}^{\infty} x(t) e^{-2\pi i f t} \, dt$$
Let $\theta(t) \equiv 2\phi(t)$ be the monotonic gravitational-wave phase, with instantaneous frequency $f(t) = \frac{1}{2\pi} \dot{\theta}(t) > 0$.
Writing the templates using Euler's identity:
$$h_0(t) = A(t) \frac{e^{i\theta(t)} + e^{-i\theta(t)}}{2}$$
$$h_{\pi/2}(t) = A(t) \frac{e^{i\theta(t)} - e^{-i\theta(t)}}{2i} = -i A(t) \frac{e^{i\theta(t)} - e^{-i\theta(t)}}{2}$$
In the stationary-phase approximation (SPA) for positive physical frequencies $f > 0$:
$$\mathcal{F}\left\{ A(t) e^{i\theta(t)} \right\} = \int_{-\infty}^{\infty} A(t) e^{i(\theta(t) - 2\pi f t)} \, dt$$
The stationary phase condition is $\frac{d}{dt}(\theta(t) - 2\pi f t) = 0 \implies \dot{\theta}(t) = 2\pi f > 0$, which possesses a valid physical stationary point during inspiral.
For the counter-rotating term $e^{-i\theta(t)}$:
$$\frac{d}{dt}(-\theta(t) - 2\pi f t) = -\dot{\theta}(t) - 2\pi f < 0 \quad \text{for all } f > 0$$
This term has no stationary point on the real axis for $f > 0$ and integrates to negligible value.
Therefore, for positive frequencies $f > 0$:
$$\tilde{h}_0(f) = \frac{1}{2} \mathcal{F}\left\{ A(t) e^{i\theta(t)} \right\}_{f>0}$$
$$\tilde{h}_{\pi/2}(f) = -\frac{i}{2} \mathcal{F}\left\{ A(t) e^{i\theta(t)} \right\}_{f>0} = -i \left[ \frac{1}{2} \mathcal{F}\left\{ A(t) e^{i\theta(t)} \right\}_{f>0} \right] = -i \tilde{h}_0(f)$$
**Rigorous Result:** The relationship is strictly:
$$\tilde{h}_{\pi/2}(f) = -i \tilde{h}_0(f), \quad \text{for } f > 0$$
The $+i$ formulation is mathematically invalid under this Fourier convention and is permanently removed.

#### 2.4 Matched-Filter Statistic and Phase Recovery
Let $d(t)$ be detector strain. The real noise-weighted inner products with the quadratures are:
$$x(t) \equiv (d | h_0(t - \cdot)) = 4 \operatorname{Re} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{d}(f) \tilde{h}_0^*(f)}{S_n(f)} e^{2\pi i f t} \, df$$
$$y(t) \equiv (d | h_{\pi/2}(t - \cdot)) = 4 \operatorname{Re} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{d}(f) \tilde{h}_{\pi/2}^*(f)}{S_n(f)} e^{2\pi i f t} \, df$$
Substituting $\tilde{h}_{\pi/2}^*(f) = (-i \tilde{h}_0(f))^* = +i \tilde{h}_0^*(f)$:
$$y(t) = 4 \operatorname{Re} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{d}(f) [i \tilde{h}_0^*(f)]}{S_n(f)} e^{2\pi i f t} \, df = -4 \operatorname{Im} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{d}(f) \tilde{h}_0^*(f)}{S_n(f)} e^{2\pi i f t} \, df$$
Define the complex matched-filter time series $z(t)$ as:
$$z(t) \equiv 4 \int_{f_{\min}}^{f_{\max}} \frac{\tilde{d}(f) \tilde{h}_0^*(f)}{S_n(f)} e^{2\pi i f t} \, df$$
Then:
$$\operatorname{Re}[z(t)] = x(t)$$
$$\operatorname{Im}[z(t)] = -y(t) \implies y(t) = -\operatorname{Im}[z(t)]$$
For arbitrary coalescence phase $\phi_c$:
$$(d | h(t, \phi_c)) = x(t) \cos(2\phi_c) + y(t) \sin(2\phi_c) = \operatorname{Re}[z(t)] \cos(2\phi_c) - \operatorname{Im}[z(t)] \sin(2\phi_c) = \operatorname{Re}\left[ z(t) e^{2i\phi_c} \right]$$
Writing $z(t) = |z(t)| e^{i \alpha(t)}$ where $\alpha(t) = \operatorname{atan2}(\operatorname{Im}[z(t)], \operatorname{Re}[z(t)])$:
$$(d | h(t, \phi_c)) = |z(t)| \cos\left( \alpha(t) + 2\phi_c \right)$$
Maximizing analytically over coalescence phase:
$$\max_{\phi_c} (d | h(t, \phi_c)) = |z(t)|$$
The maximum occurs when $\alpha(t) + 2\hat{\phi}_c = 0 \pmod{2\pi}$:
$$2\hat{\phi}_c = -\alpha(t) = -\operatorname{atan2}(\operatorname{Im}[z(t)], \operatorname{Re}[z(t)]) = \operatorname{atan2}(-\operatorname{Im}[z(t)], \operatorname{Re}[z(t)]) = \operatorname{atan2}(y(t), x(t))$$
**Exact Interpretation:**
* $\operatorname{atan2}(\operatorname{Im}[z], \operatorname{Re}[z])$ evaluates $-\phi_{c, \mathrm{gw}} = -2\phi_c$.
* $\operatorname{atan2}(y, x) = \operatorname{atan2}(-\operatorname{Im}[z], \operatorname{Re}[z])$ evaluates $+2\phi_c = \phi_{c, \mathrm{gw}}$ (the gravitational-wave phase at coalescence).
* The recovered orbital coalescence phase is:
  $$\hat{\phi}_c = \frac{1}{2} \operatorname{atan2}(y(t), x(t)) \pmod{\pi}$$

#### 2.5 Finite-Duration Analytical Sinusoidal Cross-Check
To cross-check without infinite delta-functions, consider a finite-duration sinusoidal pulse over duration $T$, where $f_0 T = N_0 \in \mathbb{N}_{\ge 2}$ (integer cycles), embedded in white noise $S_n(f) = S_0$:
$$h_0(t) = A \cos(2\pi f_0 t), \quad t \in [0, T]$$
Its continuous Fourier transform is:
$$\tilde{h}_0(f) = \frac{A T}{2} \left[ e^{-i\pi(f - f_0)T} \operatorname{sinc}(\pi(f - f_0)T) + e^{-i\pi(f + f_0)T} \operatorname{sinc}(\pi(f + f_0)T) \right]$$
At the bin center $f = f_0$, since $f_0 T = N_0$:
$$\tilde{h}_0(f_0) = \frac{A T}{2} [1 + 0] = \frac{A T}{2}$$
Let observed data be phase-shifted by known physical parameter $\phi_{\mathrm{true}}$:
$$d(t) = A \cos(2\pi f_0 t - 2\phi_{\mathrm{true}}), \quad t \in [0, T]$$
At $f = f_0$:
$$\tilde{d}(f_0) = \frac{A T}{2} e^{-2i\phi_{\mathrm{true}}}$$
Evaluating the inner product Riemann sum with bin spacing $\Delta f = 1/T$:
$$z(0) = 4 \frac{\tilde{d}(f_0) \tilde{h}_0^*(f_0)}{S_0} \Delta f = \frac{4}{S_0} \left( \frac{A T}{2} e^{-2i\phi_{\mathrm{true}}} \right) \left( \frac{A T}{2} \right) \frac{1}{T} = \frac{A^2 T}{S_0} e^{-2i\phi_{\mathrm{true}}}$$
Extracting components:
$$x(0) = \operatorname{Re}[z(0)] = \frac{A^2 T}{S_0} \cos(2\phi_{\mathrm{true}})$$
$$y(0) = -\operatorname{Im}[z(0)] = \frac{A^2 T}{S_0} \sin(2\phi_{\mathrm{true}})$$
Recovering the phase:
$$2\hat{\phi}_c = \operatorname{atan2}(y(0), x(0)) = \operatorname{atan2}\left( \frac{A^2 T}{S_0} \sin(2\phi_{\mathrm{true}}), \frac{A^2 T}{S_0} \cos(2\phi_{\mathrm{true}}) \right) = 2\phi_{\mathrm{true}}$$
This confirms that the recovered phase matches $2\phi_{\mathrm{true}}$ identically for all $\phi_{\mathrm{true}} \in [0, \pi)$ without delta-function squaring artifacts.

---

### 3. Rigorous Inner-Product Definitions

We separate the **complex weighted frequency-domain product** from the **real physical inner product**.

#### 3.1 Complex Weighted Frequency-Domain Product on $\mathbb{C}^M$
Let $M = k_{\max} - k_{\min} + 1$ be the number of frequency bins in $[f_{\min}, f_{\max}]$.
Let $A, B \in \mathbb{C}^M$ represent one-sided discrete Fourier arrays.
The complex weighted inner product is defined as:
$$\langle A, B \rangle_W \equiv \frac{4}{N f_s} \sum_{k=k_{\min}}^{k_{\max}} \frac{A_k^* B_k}{S_n(f_k)}$$
* **Vector Space:** $\mathbb{C}^M$ over the complex field $\mathbb{C}$.
* **Conjugation & Linearity:** Conjugate symmetric ($\langle B, A \rangle_W = \langle A, B \rangle_W^*$), linear in the second argument, antilinear in the first argument.
* **Positivity:** $\langle A, A \rangle_W = \frac{4}{N f_s} \sum_k \frac{|A_k|^2}{S_n(f_k)} > 0$ for all $A \ne \mathbf{0}$.

#### 3.2 Physical Real Inner Product on $\mathbb{R}^N$
In gravitational-wave data analysis, physical time-series strain signals $a(t), b(t)$ are real-valued vectors in $\mathbb{R}^N$.
The physical noise-weighted inner product is defined as:
$$(a | b) \equiv \operatorname{Re} \langle A, B \rangle_W = \frac{4}{N f_s} \operatorname{Re} \sum_{k=k_{\min}}^{k_{\max}} \frac{A_k^* B_k}{S_n(f_k)}$$
* **Vector Space:** The real Hilbert space $\mathbb{R}^N$.
* **Symmetry:** Strictly symmetric: $(b | a) = \operatorname{Re}[\langle B, A \rangle_W] = \operatorname{Re}[\langle A, B \rangle_W^*] = \operatorname{Re}[\langle A, B \rangle_W] = (a | b)$.
* **Bilinearity & Positivity:** Bilinear over $\mathbb{R}$; $(a | a) > 0$ for all $a \ne \mathbf{0}$.

#### 3.3 Continuous-to-Discrete Scaling Reconciliation
Continuous definition:
$$(a | b)_{\mathrm{cont}} = 4 \operatorname{Re} \int_{f_{\min}}^{f_{\max}} \frac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} \, df$$
Approximating via Riemann sum with bin spacing $\Delta f = 1/T = f_s / N$:
$$(a | b)_{\mathrm{cont}} \approx 4 \Delta f \operatorname{Re} \sum_{k=k_{\min}}^{k_{\max}} \frac{\tilde{a}^*(f_k) \tilde{b}(f_k)}{S_n(f_k)}$$
Under the discrete Fourier transform scaling $\tilde{a}(f_k) \approx \Delta t A_k = \frac{1}{f_s} A_k$ and $\tilde{b}(f_k) \approx \Delta t B_k = \frac{1}{f_s} B_k$:
$$4 \Delta f \operatorname{Re} \sum_k \frac{(\Delta t A_k)^* (\Delta t B_k)}{S_n(f_k)} = 4 \left( \frac{f_s}{N} \right) \left( \frac{1}{f_s^2} \right) \operatorname{Re} \sum_k \frac{A_k^* B_k}{S_n(f_k)} = \frac{4}{N f_s} \operatorname{Re} \sum_{k=k_{\min}}^{k_{\max}} \frac{A_k^* B_k}{S_n(f_k)}$$
The factor $\frac{4}{N f_s}$ is the exact discrete equivalent of the continuous integral.

---

### 4. Median-Welch PSD Statistics and Overlap Calibration

#### 4.1 Distribution of Periodogram Ordinates
Under stationary Gaussian noise, each discrete Fourier coefficient $X_{m, k} = X_{R} + i X_{I}$ has independent real and imaginary parts distributed as:
$$X_R, X_I \sim \mathcal{N}\left( 0, \frac{M S_2}{4 \Delta t} S_n(f_k) \right)$$
where $S_2 = \frac{1}{M} \sum_{j=0}^{M-1} w_j^2$.
The modified periodogram ordinate is:
$$\hat{P}_m(f_k) = \frac{2 \Delta t}{M S_2} |X_{m, k}|^2 = \frac{2 \Delta t}{M S_2} (X_R^2 + X_I^2)$$
The sum $X_R^2 + X_I^2$ is the sum of squares of two independent Gaussian variables with equal variance $\sigma_0^2 = \frac{M S_2}{4 \Delta t} S_n(f_k)$.
Therefore, the scaled quantity $U = \frac{2 \Delta t}{M S_2 \sigma_0^2} (X_R^2 + X_I^2) = \frac{2 \hat{P}_m(f_k)}{S_n(f_k)}$ follows a chi-squared distribution with 2 degrees of freedom ($\chi_2^2$).
The probability density of an exponential distribution with rate $\lambda = 1 / S_n(f_k)$ and mean $\mu = S_n(f_k)$ is:
$$p(P) = \frac{1}{S_n(f_k)} \exp\left( -\frac{P}{S_n(f_k)} \right), \quad P \ge 0$$
The cumulative distribution function is:
$$F(P) = 1 - \exp\left( -\frac{P}{S_n(f_k)} \right)$$

#### 4.2 Population Median
The true population median $m_{\mathrm{pop}}$ satisfies $F(m_{\mathrm{pop}}) = 0.5$:
$$1 - \exp\left( -\frac{m_{\mathrm{pop}}}{S_n(f_k)} \right) = 0.5 \implies m_{\mathrm{pop}} = S_n(f_k) \ln 2$$

#### 4.3 Asymptotic Variance for Independent Segments
For $K$ independent periodogram realizations, the sample median $M_K$ is asymptotically normally distributed:
$$M_K \sim \mathcal{N}\left( m_{\mathrm{pop}}, \sigma_{\mathrm{med}}^2 \right), \quad \text{where } \sigma_{\mathrm{med}}^2 = \frac{1}{K} \frac{p(1 - p)}{[f(m_{\mathrm{pop}})]^2}$$
For the median ($p = 0.5$): $p(1 - p) = 0.25$.
Evaluating the density at $m_{\mathrm{pop}}$:
$$f(m_{\mathrm{pop}}) = \frac{1}{S_n(f_k)} e^{-\ln 2} = \frac{1}{2 S_n(f_k)}$$
$$[f(m_{\mathrm{pop}})]^2 = \frac{1}{4 S_n^2(f_k)}$$
Substituting:
$$\sigma_{\mathrm{med}}^2 = \frac{1}{K} \frac{0.25}{1 / (4 S_n^2(f_k))} = \frac{S_n^2(f_k)}{K}$$
Applying the scale factor $1/\ln 2$ to obtain the unbiased estimator $\hat{S}_n = M_K / \ln 2$:
$$\operatorname{Var}[\hat{S}_n^{\mathrm{indep}}] = \frac{\sigma_{\mathrm{med}}^2}{(\ln 2)^2} = \frac{1}{(\ln 2)^2} \frac{S_n^2(f_k)}{K}$$
Numerical check:
$$(\ln 2)^2 \approx (0.69314718)^2 \approx 0.48045301$$
$$\frac{1}{(\ln 2)^2} \approx 2.08136898 \approx \frac{2.0814}{K} S_n^2(f_k)$$

#### 4.4 Exact Finite-Sample Median Bias Correction $C(K)$
For a finite number of independent exponential segments $K$, the sample median exhibits finite-sample bias:
* **For odd $K = 2L + 1$:** The sample median is order statistic $P_{(L+1)}$. By Rényi's representation of exponential order statistics:
  $$C(K) \equiv \frac{\mathbb{E}[M_K]}{S_n(f_k)} = \sum_{j=L+1}^{2L+1} \frac{1}{j} = \sum_{j=1}^K \frac{(-1)^{j+1}}{j}$$
  * $K = 1$: $C(1) = 1.0$.
  * $K = 15$: $C(15) \approx 0.7187$ (differs from $\ln 2 \approx 0.6931$ by $+3.68\%$).
  * $K = 255$: $C(255) \approx 0.6951$ (differs from $\ln 2$ by $+0.28\%$).
* **For even $K = 2L$:** The sample median is $M_K = \frac{1}{2}(P_{(L)} + P_{(L+1)})$. Its expected value is:
  $$C(K) \equiv \frac{\mathbb{E}[M_K]}{S_n(f_k)} = \sum_{j=L+1}^{2L} \frac{1}{j} + \frac{1}{2L}$$
The unbiased estimator is:
$$\hat{S}_n(f_k) = \frac{M_K(f_k)}{C(K)}$$

#### 4.5 Statistical Dependencies under 50% Overlap
When segments overlap by $50\%$ ($D = M/2$), consecutive periodograms are statistically dependent. We distinguish:
1. **Window-Amplitude Overlap Correlation:** $r_w = \frac{\sum_{j=0}^{M/2-1} w_j w_{j+M/2}}{\sum_{j=0}^{M-1} w_j^2}$. (For Tukey $\alpha=0.25$, $r_w \approx 0.43$).
2. **Complex Fourier Coefficient Correlation:** $\gamma_k = r_w$ for all interior frequencies.
3. **Periodogram Correlation:** $\rho_P = |\gamma_k|^2 = r_w^2 \approx (0.43)^2 \approx 0.185$.
4. **Sample-Median Covariance:** The sample median is a non-linear rank statistic. The joint distribution of overlapping periodograms is a correlated multivariate exponential distribution. Linear-mean effective sample size formulas do not apply to sample medians.

#### 4.6 Empirical Monte Carlo Calibration Protocol
Rather than asserting unverified analytical overlap formulas, the variance inflation factor $\kappa_{\mathrm{overlap}}(K, \alpha)$ will be calibrated empirically in Stage 1.2:
* **Protocol:**
  * Sample rate: $f_s = 4096 \, \mathrm{Hz}$. Segment duration: $T_{\mathrm{seg}} = 8.0 \, \mathrm{s}$ ($M = 32,768$ samples).
  * Window: Tukey $\alpha = 0.25$. Overlap: $50\%$ ($D = 16,384$ samples). Segment count: $K = 255$.
  * Noise model: Synthesized stationary Gaussian white noise with known variance $\sigma^2 = 1.0$ ($S_n^{\mathrm{true}} = 2/f_s$).
  * Replications: $N_{\mathrm{trials}} = 10,000$ independent realizations.
* **Outputs:**
  * Measured bias: $\hat{b} = \frac{1}{N_{\mathrm{trials}}} \sum_{i=1}^{N_{\mathrm{trials}}} \hat{S}_{n, i}(f) - S_n^{\mathrm{true}}$, with standard error $\sigma_b \le 0.02\%$.
  * Measured variance inflation factor:
    $$\kappa_{\mathrm{overlap}}(K, \alpha) \equiv \frac{K \operatorname{Var}[\hat{S}_n(f)]}{(S_n^{\mathrm{true}})^2}$$

---

### 5. Resolution of the PSD Frequency-Grid Design

#### 5.1 Grid Relationships for Considered Segment Durations
For an on-source analysis segment of duration $T_{\mathrm{analysis}} = 4.0 \, \mathrm{s}$, the discrete frequency spacing is:
$$\Delta f_{\mathrm{analysis}} = \frac{1}{T_{\mathrm{analysis}}} = 0.25 \, \mathrm{Hz}$$
The frequency grids for the three candidate Welch segment durations relate as follows:
1. **$T_{\mathrm{seg}} = 4.0 \, \mathrm{s}$ ($\Delta f_{\mathrm{PSD}} = 0.25 \, \mathrm{Hz}$):**
   * Ratio: $\Delta f_{\mathrm{PSD}} = \Delta f_{\mathrm{analysis}}$ identically.
   * Method: Direct 1-to-1 bin correspondence. Zero interpolation error.
2. **$T_{\mathrm{seg}} = 8.0 \, \mathrm{s}$ ($\Delta f_{\mathrm{PSD}} = 0.125 \, \mathrm{Hz}$):**
   * Ratio: $\Delta f_{\mathrm{analysis}} = 2 \Delta f_{\mathrm{PSD}}$ (integer multiple $m = 2$).
   * Method: Exact integer decimation. Selecting every 2nd bin ($k_{\mathrm{PSD}} = 2 k_{\mathrm{analysis}}$) provides an exact matching grid without interpolation.
3. **$T_{\mathrm{seg}} = 2.0 \, \mathrm{s}$ ($\Delta f_{\mathrm{PSD}} = 0.50 \, \mathrm{Hz}$):**
   * Ratio: $\Delta f_{\mathrm{PSD}} = 2 \Delta f_{\mathrm{analysis}}$ (coarser grid).
   * Method: Requires interpolation to evaluate at intermediate $0.25 \, \mathrm{Hz}$ bins.

#### 5.2 Baseline Selection and Interpolation Protocol
* **Primary Baseline:** $T_{\mathrm{seg}} = 8.0 \, \mathrm{s}$ (with integer decimation to $0.25 \, \mathrm{Hz}$) and $T_{\mathrm{seg}} = 4.0 \, \mathrm{s}$ (native compatible grid). Direct bin selection is enforced.
* **Secondary Experimental Track (Interpolation Testing):**
  When testing mismatched grids ($T_{\mathrm{seg}} = 2.0 \, \mathrm{s}$) or external tabulated curves:
  * Algorithm: Monotonic cubic Hermite interpolation (PCHIP) in $\log S_n$ vs. $\log f$ space.
  * Boundaries: Strictly bounded to $[f_{\min}, f_{\max}]$. Extrapolation is forbidden.
  * Limitation: Interpolation smooths broadband noise but cannot resolve spectral lines narrower than $\Delta f_{\mathrm{PSD}}$.

---

### 6. Bayesian Phase and Time Marginalization

#### 6.1 Likelihood Formulation and Marginalization Roles
Under stationary Gaussian noise, the likelihood is:
$$\mathcal{L}(d | \theta, t_c, \phi_c) = \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2(\theta) \right]}{\sqrt{\det(2\pi \mathbf{C})}} \exp\left[ |z(t_c)| \cos\left( 2\phi_c - 2\hat{\phi}_c(t_c) \right) \right]$$
We distinguish four distinct operations:
1. **Maximization over Phase:**
   $$\max_{\phi_c} \ln \mathcal{L} = |z(t_c)| - \frac{1}{2}\sigma_h^2(\theta) - \frac{1}{2}(d|d) + \text{const}$$
2. **Analytical Marginalization over Uniform Phase Prior $\pi(\phi_c) = \frac{1}{2\pi}$ on $[0, 2\pi)$:**
   $$\mathcal{L}_{\mathrm{phase-marg}}(d | \theta, t_c) = \int_0^{2\pi} \mathcal{L}(d | \theta, t_c, \phi_c) \frac{d\phi_c}{2\pi} = \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2(\theta) \right]}{\sqrt{\det(2\pi \mathbf{C})}} I_0(|z(t_c)|)$$
   where $I_0(u) = \frac{1}{2\pi} \int_0^{2\pi} e^{u \cos\psi} d\psi$ is the modified Bessel function of the first kind.
3. **Maximization over Coalescence Time:**
   $$\max_{t_c} \ln \mathcal{L}_{\mathrm{phase-marg}} = \ln I_0(\max_t |z(t)|) - \frac{1}{2}\sigma_h^2(\theta) + \text{const}$$
4. **Marginalization over Uniform Time Prior $\pi(t_c) = \frac{1}{\Delta t_c}$ on $[t_{\min}, t_{\max}]$:**
   $$\mathcal{L}_{\mathrm{marg}}(d | \theta) = \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2(\theta) \right]}{\Delta t_c \sqrt{\det(2\pi \mathbf{C})}} \int_{t_{\min}}^{t_{\max}} I_0(|z(t)|) \, dt \approx \frac{\exp\left[ -\frac{1}{2}(d|d) - \frac{1}{2}\sigma_h^2(\theta) \right]}{\Delta t_c \sqrt{\det(2\pi \mathbf{C})}} \Delta t \sum_{j \in \mathcal{J}} I_0(|z(t_j)|)$$
   where $\mathcal{J} = \{j : t_{\min} \le j \Delta t \le t_{\max}\}$.

#### 6.2 Numerical Stability and Scaled Bessel Functions
To evaluate $I_0(u)$ without floating-point overflow for large arguments ($u > 700$), we use the exponentially scaled modified Bessel function $I_{0e}(u) \equiv e^{-u} I_0(u)$ (`scipy.special.i0e`):
$$\ln I_0(u) = u + \ln(I_{0e}(u))$$
Let $v_j \equiv \ln I_0(|z(t_j)|) = |z(t_j)| + \ln I_{0e}(|z(t_j)|)$.
Define the stable offset $a$:
$$a \equiv \max_{j \in \mathcal{J}} v_j = \max_{j \in \mathcal{J}} \left[ \ln I_0(|z(t_j)|) \right]$$
The discrete sum is evaluated via log-sum-exp:
$$\ln\left( \sum_{j \in \mathcal{J}} I_0(|z(t_j)|) \right) = a + \ln\left( \sum_{j \in \mathcal{J}} \exp(v_j - a) \right)$$
Because $v_j - a \le 0$, the exponentials are bounded in $(0, 1]$, eliminating overflow.

#### 6.3 Waveform Assumptions and Higher-Mode Breakdown
* **Applicability:** Exact for waveforms dominated by the $(\ell, |m|) = (2, 2)$ quadrupole mode where phase enters via $\cos(2\phi_c)$.
* **Higher-Mode Breakdown:** If sub-dominant multipoles $(\ell, m) \ne (2, 2)$ are included (`IMRPhenomXHM`), different modes depend on $e^{i m \phi_c}$. The integral over $\phi_c$ cannot be evaluated as a single $I_0$ Bessel function and requires 1D numerical quadrature.

---

### 7. Injection SNR Calibration Protocol

In Stage 0.1, distances of 700, 350, and 200 Mpc were listed without derivation. These are designated as **preliminary estimates**.

#### 7.1 Exact Distance-Tuning Protocol
The optimal SNR is defined as:
$$\rho_{\mathrm{opt}} \equiv \sqrt{(h | h)} = \left[ 4 \Delta f \sum_{k=k_{\min}}^{k_{\max}} \frac{|\tilde{h}(f_k; d_L)|^2}{S_n(f_k)} \right]^{1/2}$$
Because gravitational-wave strain scales inversely with distance ($\tilde{h}(f) \propto 1/d_L$), the target SNRs $\rho^* \in \{10.0, 20.0, 35.0\}$ are achieved via exact numerical scaling:
1. Specify binary parameters $(\mathcal{M}, q, \chi_{1z}, \chi_{2z}, \iota, \theta, \phi, \psi)$.
2. Generate waveform at fiducial distance $d_{L, 0} = 100.0 \, \mathrm{Mpc}$.
3. Compute fiducial SNR $\rho_0 = \sqrt{(h(d_{L, 0}) | h(d_{L, 0}))}$.
4. Scale distance to achieve exact target SNR:
   $$d_L^* = d_{L, 0} \left( \frac{\rho_0}{\rho^*} \right)$$

---

### 8. Noise-Only SNR Statistics and Calibrated Detection Thresholds

#### 8.1 Statistical Regimes
1. **Fixed-Time, Fixed-Phase Quadrature:**
   $$X = \frac{(n | h_0)}{\sigma_h} \sim \mathcal{N}(0, 1)$$
2. **Fixed-Time, Phase-Maximized SNR:**
   $$\rho(t_0) = \frac{\sqrt{(n|h_0)^2 + (n|h_{\pi/2})^2}}{\sigma_h} \sim \operatorname{Rayleigh}(\sigma = 1)$$
   $$p(\rho) = \rho e^{-\rho^2 / 2}, \quad \mathbb{E}[\rho] = \sqrt{\frac{\pi}{2}} \approx 1.2533, \quad \operatorname{Var}[\rho] = \frac{4 - \pi}{2} \approx 0.4292$$
3. **Time-Maximized Peak SNR ($\rho_{\max} = \max_{j} \rho(t_j)$):**
   The formula $N_{\mathrm{eff}} \approx T \Delta f_{\mathrm{BW}}$ is a heuristic approximation. Matched-filter time series have non-zero temporal correlation $\operatorname{sinc}(\pi \Delta f_{\mathrm{BW}} \tau)$.
   * **Protocol:** Rather than using an analytical Gumbel formula as an absolute gate, empirical detection thresholds will be determined in Stage 1.4 by filtering $10,000$ independent Gaussian noise realizations through the pipeline to construct the empirical survival function $P(\rho_{\max} > \rho_{\mathrm{thresh}})$.

---

### 9. Nested-Sampling Posterior Uncertainty and Calibration Limits

#### 9.1 Distinctions in Nested-Sampling Uncertainties
1. **Raw Sample Count ($N_{\mathrm{samples}}$):** Total number of dead points recorded.
2. **Kish Effective Sample Size ($N_{\mathrm{eff}}$):**
   $$N_{\mathrm{eff}} \equiv \frac{1}{\sum_{i=1}^{N_{\mathrm{samples}}} p_i^2}$$
   where $p_i$ are normalized posterior weights.
3. **Quantile Uncertainty:**
   $$\operatorname{Var}[\hat{q}_p] \approx \frac{p(1 - p)}{N_{\mathrm{eff}} [f(q_p)]^2} + \sigma_{\mathrm{shrinkage}}^2(p)$$
4. **Evidence Uncertainty:** $\sigma_{\ln\mathcal{Z}}^2 \approx H / n_{\mathrm{live}}$, where $H$ is the information gain (entropy) in nats.
5. **Bootstrap & Seed Protocols:** Quantile errors will be evaluated via 500 weighted bootstrap resamples (`dynesty.utils.resample_equal`) and confirmed via 3 independent random-seed executions.

#### 9.2 Limitations of 100 Injections for Coverage Validation
* **Statistical Power of 100 Injections (Exp D.2):**
  For $N_{\mathrm{inj}} = 100$, the $95\%$ binomial confidence interval on the empirical cumulative distribution at median $p = 0.5$ has width:
  $$\Delta p_{95\%} \approx \pm 1.96 \sqrt{\frac{0.5 \times 0.5}{100}} = \pm 0.098 \approx \pm 10\%$$
  This test detects gross pipeline errors (e.g., prior mis-scaling or factor-of-two likelihood bugs), but cannot detect subtle $\sim 2-3\%$ calibration drifts.
* **Separation of Concepts:**
  * *Algorithm Verification:* Unit tests on toy Gaussian models.
  * *Simulation-Based Calibration (SBC):* Uniformity of prior-predictive 1D marginal quantiles.
  * *Joint Credible Coverage:* 1D marginal uniformity does not prove joint multidimensional coverage.

---

### 10. Whittle Likelihood Assumptions and Residual Diagnostics

#### 10.1 Mathematical Assumptions
The Whittle likelihood assumes:
1. Noise $n(t)$ is wide-sense stationary and Gaussian.
2. The discrete Fourier transform coefficients are mutually uncorrelated: $\mathbb{E}[\tilde{n}(f_k) \tilde{n}^*(f_{k'})] = \frac{1}{2} S_n(f_k) \delta_{k, k'} T$.
3. Data boundaries are periodic.

#### 10.2 Diagnostic Protocol
Finite windowing and leakage introduce non-zero correlations across adjacent frequency bins. Statistical tests will be treated as **calibrated continuous diagnostics** rather than universal binary gates:
* Normalized residual: $r(f_k) \equiv \frac{\tilde{d}(f_k) - \tilde{h}(f_k; \hat{\theta})}{\sqrt{\frac{1}{2} S_n(f_k) T}}$.
* Autocorrelation metric: $\hat{\rho}_r(1) = \frac{\sum_k r^*(f_k) r(f_{k+1})}{\sum_k |r(f_k)|^2}$.
* In Stage 1.2, $\hat{\rho}_r(1)$ will be calibrated on 1,000 simulated Gaussian noise realizations conditioned through the pipeline to determine the empirical $95\%$ reference interval.

---

### 11. Coherent Baseline Data Conditioning Protocol

1. **Pre-Filtering:** Zero-phase 8th-order Butterworth bandpass filter from $20.0 \, \mathrm{Hz}$ to $1024.0 \, \mathrm{Hz}$ applied to a $16\text{-second}$ stretch (GPS $[1126259454.0, 1126259470.0]$).
2. **Trimming:** Trim outer $4\text{ seconds}$ from each side to remove filter transients, leaving the $8\text{-second}$ segment (GPS $[1126259458.0, 1126259466.0]$).
3. **Windowing:** Apply an 8-second Tukey window with roll-off $\alpha = 0.25$ ($1.0\text{ s}$ cosine taper on each boundary, leaving a $6.0\text{-second}$ flat top).
4. **Consistency:** In matched filtering and likelihood evaluations, templates are generated over the identical frequency grid, and likelihood integration is strictly restricted to $[30.0, 1024.0] \, \mathrm{Hz}$.

---

### 12. Environment and Software Audit

* **Host System:** Microsoft Windows 11 Home Single Language (64-bit, Build 10.0.26300).
  * Host Python: Python 3.13.5 (installed packages: `numpy` 2.1.2, `scipy` 1.16.0, `matplotlib` 3.10.3, `h5py` 3.14.0, `pytest` 9.2.0).
  * GW Stack Status: `lalsuite`, `pycbc`, `bilby`, and `dynesty` are **not installed** on host.
* **Secondary Subsystem:** WSL2 Ubuntu 24.04.2 LTS (Linux kernel 5.15, Python 3.12.3).
  * Package Managers: `conda` and `mamba` are currently **not installed** in default PATH.
* **Execution Boundary:** No software will be installed in Stage 0. The host environment remains pristine.

---

### 13. Revised Experimental Plan (Experiments A–E)

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
|       |                          | T_off (64, 1024s); T_seg (2,4,8) analytical ZDHP reference | W_90(PSD)/W_90(true)  |         |
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
| D.2   | Statistical Coverage     | 100 independent prior-draw   | Reduced 3D space (M_c,q,t)| Kolmogorov-Smirnov    | Ext.    |
|       | Validation (P-P Test)    | injections into noise        | analytical ZDHP PSD       | p-value on quantiles  |         |
+-------+--------------------------+------------------------------+---------------------------+-----------------------+---------+
```

---

### 14. Validation Matrix

```
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                               Validation Matrix                                                                    |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Component / Domain       | Expected Analytical Property        | Proposed Numerical Test     | Numerical Tolerance   | Failure Interpretation     |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Discrete Parseval Balance| sum |x_j|^2 dt == sum |X_k|^2 df    | Double-precision white      | Relative error        | Incorrect dt or df scaling |
| (Stage 1.1)              | for discrete Fourier transform      | noise time series           | < 1e-12               | in Fourier transform.      |
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
| Overlap Monte Carlo      | Calibrated variance inflation factor| 10^4 Gaussian realizations  | Monte Carlo standard  | Overlap covariance         |
| (Stage 1.2)              | kappa_overlap(K, alpha)             | with 50% overlap Tukey 0.25 | error < 0.05%         | miscalculated.             |
+--------------------------+-------------------------------------+-----------------------------+-----------------------+----------------------------+
| Quadrature FFT Sign      | Positive-frequency relation:        | Compute rfft of analytic    | Max relative error    | Sign error; used +i        |
| (Stage 1.4)              | h_pi/2(f) == -i * h_0(f) for f > 0  | sinusoids h_0 and h_pi/2    | < 1e-12               | instead of -i.             |
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

### 15. Deliverable Summary and Status Report

#### A. Concise List of Changes from Stage 0.1
1. **GW150914 Intervals:** Eliminated the 4-second overlap. Formulated a 1024-second off-source PSD interval ($[1126258418, 1126259442]$) separated by a 16-second guard interval ($[1126259442, 1126259458]$) from the 8-second on-source segment ($[1126259458, 1126259466]$). Corrected file start timestamp to GPS $1126257415$.
2. **Phase Quadrature:** Formally proved $\tilde{h}_{\pi/2}(f) = -i \tilde{h}_0(f)$ for $f > 0$. Defined accumulated orbital phase $\phi(t)$ such that $\phi(t_c) = 0$, explicitly separating orbital evolution from coalescence phase $\phi_c$. Replaced the delta-function cross-check with a finite-duration sinusoidal derivation.
3. **Inner Products:** Separated the complex Hermitian inner product on $\mathbb{C}^M$ from the physical real inner product on $\mathbb{R}^N$.
4. **Median-Welch Statistics:** Withdrew unverified overlap effective sample size claims. Established an empirical Monte Carlo calibration protocol (Approach B) while preserving the exact independent-segment baseline $\operatorname{Var}[\hat{S}_n] \approx \frac{2.0813}{K} S_n^2$ and finite-sample factor $C(K)$.
5. **PSD Frequency Grids:** Selected compatible integer-multiple segment lengths ($T_{\mathrm{seg}} = 8.0 \, \mathrm{s}$ with decimation, $T_{\mathrm{seg}} = 4.0 \, \mathrm{s}$ native) as primary baselines. Restricted PCHIP log-log interpolation to mismatched experimental tracks without extrapolation.
6. **Bayesian Marginalization:** Replaced the invalid scalar offset $u_{\max}$ with the exact maximum log-Bessel value $a = \max_j \ln I_0(u_j)$. Formulated numerically stable evaluation via exponentially scaled Bessel functions ($I_{0e}$).
7. **Injection SNR Calibration:** Formulated an exact inverse-distance scaling protocol $d_L^* = d_{L, 0}(\rho_0 / \rho^*)$ to hit target SNRs precisely.
8. **Noise Statistics:** Replaced analytical Gumbel acceptance gates with empirical Monte Carlo noise survival calibrations.
9. **Posterior Coverage:** Corrected quantile error formulas, accounted for nested-sampling stochastic shrinkage, and highlighted the $10\%$ binomial resolution limit of 100 injections.
10. **Residual Diagnostics:** Replaced universal binary rejection gates with simulation-calibrated diagnostic reference intervals.

#### B. Explicit List of Unresolved Scientific Decisions
1. **Local Stationarity of GW150914 Off-Source Segment:** Archival files are verified to be in nominal science lock, but local stationarity and absence of glitches within GPS $[1126258418, 1126259442]$ remain unverified until direct time-series inspection in Stage 1.
2. **Empirical Overlap Inflation Factor $\kappa_{\mathrm{overlap}}$:** Numerical value under Tukey $\alpha=0.25$ will be established via Monte Carlo simulation in Stage 1.2 rather than assumed analytically.
3. **Sampler Convergence on Degenerate Topologies:** The exact scaling of live points ($n_{\mathrm{live}} = 500$ vs. $1000$) for production runs will be validated via the proposed resolution-doubling experiment in Stage 1.6.

#### C. Pre-Implementation Confirmation Statement
* **Implementation Status:** **Not started.**
* No production source code has been written.
* No packages have been installed on either the Windows host or WSL2 subsystem.
* No strain data files have been downloaded from GWOSC.
* No computational tests have been launched.
* Stage 1.1 has not been initiated.
* The workspace remains clean.

#### D. Recommendation for Next Stage
Stage 0.2 is complete, mathematically self-consistent, and ready for independent scientific review. Upon formal authorization by the independent reviewer (ChatGPT), the project will advance to **Stage 1.1: Discrete Fourier Conventions, Scaling, and Synthetic Time-Series Generation**.

---
*End of Stage 0.2 Specification Document.*
