"use client";

import React, { useState, useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import NumericalReadout from "@/components/NumericalReadout";
import MathBlock from "@/components/MathBlock";
import WaveformPlot, { WaveformTab } from "@/components/WaveformPlot";
import {
  generateWaveform,
  fetchChirpTime,
  WaveformResponseData,
  ChirpTimeResponseData,
} from "@/lib/api";
import { formatFloat } from "@/lib/format";
import { RotateCcw, Sliders, Activity, Info } from "lucide-react";

// Stage 6C Injection Benchmark Defaults
const STAGE6C_BENCHMARK = {
  mc_msun: 28.095555795460438,
  eta: 0.24710059171597634,
  distance_mpc: 410.0,
  tc_s: 0.0,
  phic_rad: 0.0,
};

export default function WaveformsPage() {
  // Input parameters
  const [mc, setMc] = useState<number>(STAGE6C_BENCHMARK.mc_msun);
  const [eta, setEta] = useState<number>(STAGE6C_BENCHMARK.eta);
  const [distance, setDistance] = useState<number>(STAGE6C_BENCHMARK.distance_mpc);
  const [tc, setTc] = useState<number>(STAGE6C_BENCHMARK.tc_s);
  const [phic, setPhic] = useState<number>(STAGE6C_BENCHMARK.phic_rad);

  // Active plot tab
  const [activeTab, setActiveTab] = useState<WaveformTab>("amplitude");

  // Output data & states
  const [data, setData] = useState<WaveformResponseData | null>(null);
  const [chirpData, setChirpData] = useState<ChirpTimeResponseData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch waveform whenever physical parameters change
  useEffect(() => {
    let stale = false;

    const runEvaluation = async () => {
      setLoading(true);
      setError(null);
      try {
        const [wfRes, chirpRes] = await Promise.all([
          generateWaveform({
            mc_msun: mc,
            eta: eta,
            distance_mpc: distance,
            tc_s: tc,
            phic_rad: phic,
            duration_s: 4.0,
            sampling_rate_hz: 2048.0,
            f_end_hz: 512.0,
          }),
          fetchChirpTime({
            mc_msun: mc,
            eta: eta,
          }),
        ]);
        if (stale) return;
        setData(wfRes);
        setChirpData(chirpRes);
      } catch (err) {
        if (stale) return;
        const message =
          err instanceof Error && err.message
            ? err.message
            : "Failed to generate waveform.";
        setError(message);
      } finally {
        if (!stale) setLoading(false);
      }
    };

    runEvaluation();

    return () => {
      stale = true;
    };
  }, [mc, eta, distance, tc, phic]);

  const handleReset = () => {
    setMc(STAGE6C_BENCHMARK.mc_msun);
    setEta(STAGE6C_BENCHMARK.eta);
    setDistance(STAGE6C_BENCHMARK.distance_mpc);
    setTc(STAGE6C_BENCHMARK.tc_s);
    setPhic(STAGE6C_BENCHMARK.phic_rad);
  };

  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Waveforms"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <Activity size={13} />
            Deterministic Physics Engine
          </span>
        }
        subtitle="TaylorF2 3.5PN frequency-domain strain synthesis and parameter space exploration."
        description="Explores stationary-phase approximation gravitational waveforms from binary inspirals up to the Schwarzschild innermost stable circular orbit (ISCO)."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        {/* Main Workstation Layout: 2 Columns (Controls + Readouts on Left/Top, Plot Dominant on Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column (4 cols on lg) */}
          <div className="lg:col-span-4 space-y-5">
            <ScientificPanel
              title="Physical Parameters"
              subtitle="Stationary-phase approximation inputs"
              action={
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1 px-2 py-1 text-xs font-sans rounded-[3px] text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-subtle)] transition-colors border border-[var(--border)]"
                  title="Reset to Stage 6C Injection Benchmark"
                >
                  <RotateCcw size={12} />
                  <span>Reset</span>
                </button>
              }
            >
              <div className="space-y-4">
                {/* Chirp Mass Mc */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label htmlFor="mc-slider" className="font-sans font-medium text-[var(--foreground)]">
                      Chirp Mass <em>M<sub>c</sub></em>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(mc, 4)} M<sub>☉</sub>
                    </span>
                  </div>
                  <input
                    id="mc-slider"
                    type="range"
                    min="10.0"
                    max="50.0"
                    step="0.05"
                    value={mc}
                    onChange={(e) => setMc(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-[var(--muted-light)]">
                    <span>10.0</span>
                    <span>28.0956 (benchmark)</span>
                    <span>50.0</span>
                  </div>
                </div>

                {/* Symmetric Mass Ratio eta */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label htmlFor="eta-slider" className="font-sans font-medium text-[var(--foreground)]">
                      Symmetric Mass Ratio &eta;
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(eta, 4)}
                    </span>
                  </div>
                  <input
                    id="eta-slider"
                    type="range"
                    min="0.05"
                    max="0.25"
                    step="0.001"
                    value={eta}
                    onChange={(e) => setEta(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-[var(--muted-light)]">
                    <span>0.05 (extreme q)</span>
                    <span>0.2471</span>
                    <span>0.2500 (equal mass)</span>
                  </div>
                </div>

                {/* Luminosity Distance dL */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label htmlFor="dist-slider" className="font-sans font-medium text-[var(--foreground)]">
                      Luminosity Distance <em>d<sub>L</sub></em>
                    </label>
                    <span className="font-mono text-[var(--foreground)] font-semibold">
                      {formatFloat(distance, 1)} Mpc
                    </span>
                  </div>
                  <input
                    id="dist-slider"
                    type="range"
                    min="50.0"
                    max="1500.0"
                    step="10.0"
                    value={distance}
                    onChange={(e) => setDistance(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-[var(--muted-light)]">
                    <span>50.0</span>
                    <span>410.0 (benchmark)</span>
                    <span>1500.0</span>
                  </div>
                </div>

                {/* Coalescence Time Shift tc */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label htmlFor="tc-slider" className="font-sans font-medium text-[var(--foreground)]">
                      Arrival Shift <em>t<sub>c</sub></em>
                    </label>
                    <span className="font-mono text-[var(--foreground)] font-semibold">
                      {formatFloat(tc, 3)} s
                    </span>
                  </div>
                  <input
                    id="tc-slider"
                    type="range"
                    min="-0.1"
                    max="0.1"
                    step="0.002"
                    value={tc}
                    onChange={(e) => setTc(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-[var(--muted-light)]">
                    <span>−0.10</span>
                    <span>0.000</span>
                    <span>+0.10</span>
                  </div>
                </div>

                {/* Coalescence Phase phic */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label htmlFor="phic-slider" className="font-sans font-medium text-[var(--foreground)]">
                      Coalescence Phase &phi;<sub>c</sub>
                    </label>
                    <span className="font-mono text-[var(--foreground)] font-semibold">
                      {formatFloat(phic, 2)} rad
                    </span>
                  </div>
                  <input
                    id="phic-slider"
                    type="range"
                    min="0"
                    max={6.283}
                    step="0.05"
                    value={phic}
                    onChange={(e) => setPhic(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-[var(--muted-light)]">
                    <span>0.00</span>
                    <span>&pi; (3.14)</span>
                    <span>2&pi; (6.28)</span>
                  </div>
                </div>
              </div>
            </ScientificPanel>

            {/* Derived Parameters Grid */}
            <ScientificPanel title="Derived Binary Mechanics">
              <div className="grid grid-cols-2 gap-3">
                <NumericalReadout
                  label="Primary mass m₁"
                  value={data ? formatFloat(data.m1_msun, 2) : "35.59"}
                  unit="M☉"
                />
                <NumericalReadout
                  label="Secondary mass m₂"
                  value={data ? formatFloat(data.m2_msun, 2) : "30.41"}
                  unit="M☉"
                />
                <NumericalReadout
                  label="Total mass M"
                  value={data ? formatFloat(data.m_total_msun, 2) : "66.00"}
                  unit="M☉"
                />
                <NumericalReadout
                  label="Mass ratio q"
                  value={data ? formatFloat(data.q, 3) : "1.170"}
                  subtext="m₁ / m₂"
                />
                <NumericalReadout
                  label="ISCO frequency f_ISCO"
                  value={data ? formatFloat(data.f_isco, 1) : "66.6"}
                  unit="Hz"
                  className="col-span-2"
                  subtext="Schwarzschild innermost stable circular orbit cutoff"
                />
              </div>
            </ScientificPanel>
          </div>

          {/* Plot Center (8 cols on lg) */}
          <div className="lg:col-span-8 space-y-5">
            <WaveformPlot
              data={data}
              chirpData={chirpData}
              activeTab={activeTab}
              onTabChange={setActiveTab}
              loading={loading}
              error={error}
            />

            {/* Scientific Formulation & Mathematical Foundation */}
            <ScientificPanel
              title="Mathematical Formulation & 3.5PN Expansion"
              subtitle="Post-Newtonian stationary-phase approximation in the frequency domain"
            >
              <div className="space-y-4 text-xs text-[var(--foreground-secondary)] leading-relaxed">
                <div>
                  <p className="font-sans font-medium text-[var(--foreground)] mb-1">
                    Frequency-Domain Gravitational Strain:
                  </p>
                  <MathBlock math="\tilde{h}(f) = \mathcal{A}(f)\,e^{-i\,\Psi(f)}" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                  <div>
                    <p className="font-sans font-medium text-[var(--foreground)] mb-1">
                      Newtonian Amplitude Prefactor:
                    </p>
                    <MathBlock math="\mathcal{A}(f) = \sqrt{\frac{5}{24}}\,\frac{G^{5/6}\mathcal{M}_c^{5/6}\,c^{-3/2}}{\pi^{2/3}\,d_L}\,f^{-7/6}" />
                  </div>

                  <div>
                    <p className="font-sans font-medium text-[var(--foreground)] mb-1">
                      Newtonian Chirp Time to Merger:
                    </p>
                    <MathBlock math="\tau(f) = \frac{5}{256}\left(\frac{G\mathcal{M}_c}{c^3}\right)^{-5/3}(\pi f)^{-8/3}" />
                  </div>
                </div>

                <div className="pt-2 border-t border-[var(--border-subtle)]">
                  <p className="font-sans font-medium text-[var(--foreground)] mb-1">
                    3.5PN Post-Newtonian Fourier Phase:
                  </p>
                  <MathBlock math="\Psi(f) = 2\pi f t_c - \phi_c - \frac{\pi}{4} + \frac{3}{128\eta v^5}\sum_{k=0}^{7} \alpha_k(\eta)\,v^k, \quad v = (\pi G M f / c^3)^{1/3}" />
                  <p className="text-[11px] text-[var(--muted)] mt-1">
                    The expansion incorporates PN coefficients up to 3.5PN including logarithmic corrections at 2.5PN and 3.0PN:
                    &alpha;<sub>0</sub> = 1, &alpha;<sub>2</sub> = 3715/756 + 55&eta;/9, &alpha;<sub>3</sub> = &minus;16&pi;, etc.
                  </p>
                </div>
              </div>
            </ScientificPanel>
          </div>
        </div>
      </div>
    </div>
  );
}
