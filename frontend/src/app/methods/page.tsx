"use client";

import React from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import MathBlock from "@/components/MathBlock";
import { ShieldCheck } from "lucide-react";
import Link from "next/link";

export default function MethodsPage() {
  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Methods and provenance"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <ShieldCheck size={13} />
            Scientific Governance
          </span>
        }
        subtitle="Mathematical derivations, algorithmic standards, and data provenance audit trail."
        description="Frozen conventions of the gw-inference scientific engine, physical constants, likelihoods, and governance boundaries."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ScientificPanel title="Physical Units & Coordinate Standards">
            <div className="space-y-3 text-xs text-[var(--foreground-secondary)] leading-relaxed">
              <p>
                All calculations in <code className="font-mono">gw-inference</code> adhere to
                SI base units internally (m, kg, s) and standard astrophysical units for display:
              </p>
              <ul className="space-y-1.5 font-mono text-[11px]">
                <li>&bull; Solar mass: M_sun = 1.9884099 × 10³⁰ kg</li>
                <li>&bull; Speed of light: c = 299,792,458 m/s</li>
                <li>&bull; Gravitational constant: G = 6.67430 × 10⁻¹¹ m³/(kg·s²)</li>
                <li>&bull; Megaparsec: 1 Mpc = 3.085677581491367 × 10²² m</li>
                <li>&bull; Frozen ISCO constant: C_ISCO = 4397.17 M_sun·Hz</li>
              </ul>
            </div>
          </ScientificPanel>

          <ScientificPanel title="TaylorF2 Stationary-Phase Approximant">
            <div className="space-y-3 text-xs text-[var(--foreground-secondary)] leading-relaxed">
              <p>
                The waveform engine implements the restricted TaylorF2 frequency-domain
                approximant for non-spinning, quasi-circular binary inspirals:
              </p>
              <ul className="space-y-1.5 text-xs">
                <li>
                  &bull; <strong>Amplitude (0PN):</strong>{" "}
                  <MathBlock
                    math="|\tilde{h}(f)| = \sqrt{\frac{5}{24}}\,\pi^{-2/3}\left(\frac{G\mathcal{M}_c}{c^3}\right)^{5/6}\frac{c}{d_L}\,f^{-7/6}"
                    display={false}
                    className="text-[11px]"
                    ariaLabel="TaylorF2 amplitude formula"
                  />{" "}
                  (<span className="font-mono text-[11px]">strain/Hz</span>)
                </li>
                <li>&bull; <strong>Phase:</strong> 3.5PN post-Newtonian expansion via SPA</li>
                <li>
                  &bull; <strong>ISCO Cutoff:</strong>{" "}
                  <MathBlock
                    math="f_{\mathrm{ISCO}} = 4397.17 / (M_{\mathrm{total}} / \mathrm{M}_\odot)\;\mathrm{Hz}"
                    display={false}
                    className="text-[11px]"
                    ariaLabel="ISCO frequency formula"
                  />{" "}
                  (inclusive <span className="font-mono text-[11px]">f ≤ f_ISCO</span>)
                </li>
                <li>
                  &bull; <strong>Coalescence Shift:</strong>{" "}
                  <MathBlock math="\exp(-2\pi i f t_c)" display={false} className="text-[11px]" />
                </li>
              </ul>
            </div>
          </ScientificPanel>
        </div>

        <ScientificPanel title="Frozen PSD, Matched Filtering, Likelihood & Prior Conventions">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-[var(--foreground-secondary)] leading-relaxed">
            <div className="space-y-2">
              <div className="font-semibold text-[var(--foreground)]">
                Implemented &amp; Frozen Core Conventions
              </div>
              <ul className="space-y-1.5">
                <li>
                  &bull; <strong>Active Band:</strong>{" "}
                  <span className="font-mono text-[11px]">K_+ ⊂ [20.0, 512.0] Hz</span> (excluding
                  DC and Nyquist).
                </li>
                <li>
                  &bull; <strong>PSD Proxy &amp; Welch Setup:</strong> Analytical{" "}
                  <span className="font-mono text-[11px]">aligo_representative</span> proxy (
                  <span className="font-mono text-[11px]">strain²/Hz</span>); Welch PSD with{" "}
                  <span className="font-mono text-[11px]">4.0 s</span> segments, 50% overlap,
                  Tukey <span className="font-mono text-[11px]">α = 0.25</span> (K = 31 segments
                  over <span className="font-mono text-[11px]">64.0 s</span>).
                </li>
                <li>
                  &bull; <strong>Matched Filter:</strong>{" "}
                  <MathBlock
                    math="(d \mid h)_{K_+} = 4\,\Delta f\,\mathrm{Re}\sum_{k \in K_+} \tilde{d}_k \tilde{h}_k^* / S_n(f_k)"
                    display={false}
                    className="text-[11px]"
                  />
                  , <MathBlock math="\rho_{\mathrm{opt}} = \sqrt{(h \mid h)_{K_+}}" display={false} className="text-[11px]" />.
                </li>
              </ul>
            </div>
            <div className="space-y-2">
              <div className="font-semibold text-[var(--foreground)]">
                Inference Likelihood, Priors &amp; Scope Boundaries
              </div>
              <ul className="space-y-1.5">
                <li>
                  &bull; <strong>Production Likelihood (Model 2):</strong> Plug-in Gaussian Whittle
                  log-likelihood using the frozen off-source Welch PSD.
                </li>
                <li>
                  &bull; <strong>4D Uniform Prior (Run A):</strong>{" "}
                  <span className="font-mono text-[11px]">Mc ∈ [20, 40] M_sun</span>,{" "}
                  <span className="font-mono text-[11px]">η ∈ [0.10, 0.25]</span>,{" "}
                  <span className="font-mono text-[11px]">tc ∈ [1.9, 2.1] s</span>,{" "}
                  <span className="font-mono text-[11px]">φc ∈ [0, 2π) rad</span>.
                </li>
                <li>
                  &bull; <strong>Proposed / Frozen Extensions:</strong> Model 3 Student-t likelihood
                  is frozen (not approved for production); Run B{" "}
                  <span className="font-mono text-[11px]">(M_total, η, tc, φc)</span> prior
                  transform is implemented in <code className="font-mono">gw_inference</code> but
                  unexecuted.
                </li>
              </ul>
            </div>
          </div>
        </ScientificPanel>

        <ScientificPanel title="External Real-Strain Validation (GWOSC GW150914 v3)">
          <div className="space-y-4 text-xs text-[var(--foreground-secondary)] leading-relaxed">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-[3px] bg-[var(--surface-subtle)] border border-[var(--border)] text-[11px] font-medium text-[var(--foreground)]">
              External Real-Strain Check &mdash; Distinct from Synthetic Inference Demonstrations
            </div>

            <p>
              While GWLab&apos;s interactive pages and saved Bayesian posteriors operate on synthetic
              data and archived runs, the conditioning, off-source Welch PSD, and{" "}
              <code className="font-mono">gw_inference.filter.matched_filter</code> pipeline were
              separately checked against public GWOSC{" "}
              <span className="font-mono">GWTC-1-confident v3</span> 32-second H1/L1 strain for{" "}
              <strong>GW150914</strong> (GPS <span className="font-mono">1126259462.4</span>,
              4096 Hz downsampled with anti-aliasing to 2048 Hz;{" "}
              <span className="font-mono">Δf = 0.25 Hz</span>; 1969 active bins in{" "}
              <span className="font-mono">[20, 512] Hz</span>; 191 nonzero TaylorF2 support bins in{" "}
              <span className="font-mono">[20, 67.5] Hz</span>).
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 rounded-[4px] bg-[var(--surface-subtle)] border border-[var(--border)]">
                <div className="text-[11px] text-[var(--foreground-muted)]">H1 Baseline (Fixed 35+30 M_sun)</div>
                <div className="font-mono text-base font-semibold text-[var(--foreground)] mt-0.5 tabular-nums">
                  SNR 8.937598
                </div>
                <div className="font-mono text-[11px] text-[var(--foreground-secondary)] mt-0.5">
                  +0.0718 s &bull; DQmask 127 &bull; Injmask 31
                </div>
              </div>

              <div className="p-3 rounded-[4px] bg-[var(--surface-subtle)] border border-[var(--border)]">
                <div className="text-[11px] text-[var(--foreground-muted)]">L1 Baseline (Fixed 35+30 M_sun)</div>
                <div className="font-mono text-base font-semibold text-[var(--foreground)] mt-0.5 tabular-nums">
                  SNR 5.545914
                </div>
                <div className="font-mono text-[11px] text-[var(--foreground-secondary)] mt-0.5">
                  +0.0610 s &bull; DQmask 127 &bull; Injmask 23
                </div>
              </div>

              <div className="p-3 rounded-[4px] bg-[var(--surface-subtle)] border border-[var(--border)]">
                <div className="text-[11px] text-[var(--foreground-muted)]">
                  L1 5-Bin Exclusion Diagnostic ([52.25, 53.25] Hz)
                </div>
                <div className="font-mono text-base font-semibold text-[var(--foreground)] mt-0.5 tabular-nums">
                  SNR 5.039313 (&minus;9.1347%)
                </div>
                <div className="font-mono text-[11px] text-[var(--foreground-secondary)] mt-0.5">
                  +0.0605 s &bull; 5 / 191 support bins (2.6178%)
                </div>
              </div>
            </div>

            <ul className="space-y-1.5">
              <li>
                &bull; <strong>Consistent Normalization:</strong> Both baseline and L1 exclusion runs
                apply the exact same mask to the matched-filter numerator and template norm{" "}
                <span className="font-mono">σ_h</span>.
              </li>
              <li>
                &bull; <strong>L1 CW Hardware-Injection Qualification:</strong>{" "}
                <span className="font-mono">L1 NO_CW_HW_INJ</span> is false over the event
                interval (<span className="font-mono">Injmask = 23</span>); L1 is not confirmed
                injection-free. O1 Pulsar 5 (<span className="font-mono">52.8083 Hz</span>) is a
                known O1 CW frequency inside <span className="font-mono">[20, 67.5] Hz</span>, but
                the 5-bin exclusion is a sensitivity diagnostic only&mdash;it also removes real
                inspiral power and does not establish that a hardware injection caused the SNR
                change.
              </li>
              <li>
                &bull; <strong>Scope Limitations:</strong> Known-event check with a fixed,
                non-optimized TaylorF2 inspiral-only template (<span className="font-mono">f_ISCO = 67.65 Hz</span>);
                not a blind search, not a full IMR analysis, and not full real-data Bayesian
                parameter estimation.
              </li>
            </ul>
          </div>
        </ScientificPanel>

        <ScientificPanel title="Governance & Verification Principles">
          <div className="space-y-3 text-xs text-[var(--foreground-secondary)] leading-relaxed max-w-3xl">
            <p>
              GWLab presentation layers operate strictly in a{" "}
              <strong>read-only archive and deterministic simulation mode</strong>. No Markov chain
              or nested samplers are executed during interactive browsing, preventing accidental
              computational budget depletion and preserving audited historical ledgers.
            </p>
            <div className="pt-2">
              <Link
                href="/"
                className="inline-flex items-center gap-2 text-xs font-sans text-[var(--accent)] hover:underline"
              >
                &larr; Return to Overview
              </Link>
            </div>
          </div>
        </ScientificPanel>
      </div>
    </div>
  );
}
