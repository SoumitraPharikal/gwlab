"use client";

import React, { useState, useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import NumericalReadout from "@/components/NumericalReadout";
import MathBlock from "@/components/MathBlock";
import MatchedFilterPlot, { MatchedFilterTab } from "@/components/MatchedFilterPlot";
import { evaluateMatchedFilter, MatchedFilterResponseData } from "@/lib/api";
import { formatFloat } from "@/lib/format";
import { Activity, RotateCcw, Sliders, Info } from "lucide-react";

const REALIZATION_01_DEFAULTS = {
  mc_msun: 28.095555795460438,
  eta: 0.24710059171597634,
  distance_mpc: 410.0,
  tc_s: 2.0,
  phic_rad: 0.0,
  delta_mc_msun: 0.0,
  psd_model: "aligo_representative" as const,
  f_min_hz: 20.0,
  f_max_hz: 512.0,
  seed: 7151,
};

export default function MatchedFilteringPage() {
  const [mcMsun, setMcMsun] = useState<number>(REALIZATION_01_DEFAULTS.mc_msun);
  const [distanceMpc, setDistanceMpc] = useState<number>(REALIZATION_01_DEFAULTS.distance_mpc);
  const [tcS, setTcS] = useState<number>(REALIZATION_01_DEFAULTS.tc_s);
  const [deltaMc, setDeltaMc] = useState<number>(REALIZATION_01_DEFAULTS.delta_mc_msun);
  const [psdModel, setPsdModel] = useState<"aligo_representative" | "flat">(
    REALIZATION_01_DEFAULTS.psd_model
  );
  const [fMinHz, setFMinHz] = useState<number>(REALIZATION_01_DEFAULTS.f_min_hz);
  const [fMaxHz, setFMaxHz] = useState<number>(REALIZATION_01_DEFAULTS.f_max_hz);
  const [seed, setSeed] = useState<number>(REALIZATION_01_DEFAULTS.seed);

  const [activeTab, setActiveTab] = useState<MatchedFilterTab>("snr_full");
  const [data, setData] = useState<MatchedFilterResponseData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    const runFilter = async () => {
      if (fMinHz >= fMaxHz) {
        setError("Lower frequency f_min must be strictly less than upper frequency f_max.");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await evaluateMatchedFilter({
          mc_msun: mcMsun,
          eta: REALIZATION_01_DEFAULTS.eta,
          distance_mpc: distanceMpc,
          tc_s: tcS,
          phic_rad: REALIZATION_01_DEFAULTS.phic_rad,
          delta_mc_msun: deltaMc,
          psd_model: psdModel,
          f_min_hz: fMinHz,
          f_max_hz: fMaxHz,
          seed,
        });
        if (!stale) setData(res);
      } catch (err) {
        if (!stale) {
          setError(err instanceof Error ? err.message : "Matched filter evaluation failed.");
        }
      } finally {
        if (!stale) setLoading(false);
      }
    };

    runFilter();
    return () => {
      stale = true;
    };
  }, [mcMsun, distanceMpc, tcS, deltaMc, psdModel, fMinHz, fMaxHz, seed]);

  const handleReset = () => {
    setMcMsun(REALIZATION_01_DEFAULTS.mc_msun);
    setDistanceMpc(REALIZATION_01_DEFAULTS.distance_mpc);
    setTcS(REALIZATION_01_DEFAULTS.tc_s);
    setDeltaMc(REALIZATION_01_DEFAULTS.delta_mc_msun);
    setPsdModel(REALIZATION_01_DEFAULTS.psd_model);
    setFMinHz(REALIZATION_01_DEFAULTS.f_min_hz);
    setFMaxHz(REALIZATION_01_DEFAULTS.f_max_hz);
    setSeed(REALIZATION_01_DEFAULTS.seed);
  };

  const meta = data?.metadata;

  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Matched filtering"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <Activity size={13} />
            Interactive Detection Filter
          </span>
        }
        subtitle="Noise-weighted inner product, phase-maximized SNR time series, and template matching."
        description="Evaluate the active-band matched filter ρ(t) = |z(t)| / σₕ on synthetic TaylorF2 injections in stationary colored Gaussian noise."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column */}
          <div className="lg:col-span-4 space-y-5">
            <ScientificPanel
              title="Injection & Filter Controls"
              subtitle="Synthetic signal + colored noise parameters"
              action={
                <button
                  type="button"
                  onClick={handleReset}
                  className="flex items-center gap-1 px-2 py-1 text-xs font-sans rounded-[3px] text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-subtle)] transition-colors border border-[var(--border)]"
                  title="Reset to Realization 01 canonical defaults"
                >
                  <RotateCcw size={12} />
                  <span>Reset</span>
                </button>
              }
            >
              <div className="space-y-4 text-xs">
                {/* Chirp Mass */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <label htmlFor="mf-mc" className="font-sans font-medium text-[var(--foreground)]">
                      Injected Chirp Mass <em>M<sub>c</sub></em>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(mcMsun, 4)} M<sub>☉</sub>
                    </span>
                  </div>
                  <input
                    id="mf-mc"
                    type="range"
                    min="12"
                    max="45"
                    step="0.1"
                    value={mcMsun}
                    onChange={(e) => setMcMsun(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                {/* Distance */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <label htmlFor="mf-dl" className="font-sans font-medium text-[var(--foreground)]">
                      Luminosity Distance <em>d<sub>L</sub></em>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(distanceMpc, 0)} Mpc
                    </span>
                  </div>
                  <input
                    id="mf-dl"
                    type="range"
                    min="100"
                    max="1200"
                    step="10"
                    value={distanceMpc}
                    onChange={(e) => setDistanceMpc(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                {/* Injected tc */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <label htmlFor="mf-tc" className="font-sans font-medium text-[var(--foreground)]">
                      Injected Event Time <em>t<sub>c</sub></em>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(tcS, 2)} s
                    </span>
                  </div>
                  <input
                    id="mf-tc"
                    type="range"
                    min="0.5"
                    max="3.5"
                    step="0.05"
                    value={tcS}
                    onChange={(e) => setTcS(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                {/* Template Mismatch */}
                <div className="space-y-1 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex justify-between">
                    <label htmlFor="mf-dmc" className="font-sans font-medium text-[var(--foreground)]">
                      Template Mismatch &Delta;<em>M<sub>c</sub></em>
                    </label>
                    <span className="font-mono text-[var(--foreground)] font-semibold">
                      {deltaMc >= 0 ? "+" : ""}
                      {formatFloat(deltaMc, 2)} M<sub>☉</sub>
                    </span>
                  </div>
                  <input
                    id="mf-dmc"
                    type="range"
                    min="-2.0"
                    max="2.0"
                    step="0.05"
                    value={deltaMc}
                    onChange={(e) => setDeltaMc(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                {/* Active Band & PSD */}
                <div className="space-y-2.5 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 font-sans font-medium text-[var(--foreground)]">
                    <Sliders size={13} className="text-[var(--accent)]" />
                    <span>Active Band &amp; PSD Proxy</span>
                  </div>

                  <select
                    aria-label="PSD Proxy Model"
                    value={psdModel}
                    onChange={(e) =>
                      setPsdModel(e.target.value as "aligo_representative" | "flat")
                    }
                    className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs font-mono"
                  >
                    <option value="aligo_representative">aligo_representative (PSD proxy)</option>
                    <option value="flat">flat (1e-46 strain²/Hz)</option>
                  </select>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <div className="flex justify-between text-[11px] text-[var(--muted)]">
                        <span>f_min</span>
                        <span className="font-mono">{fMinHz} Hz</span>
                      </div>
                      <input
                        aria-label="Lower Frequency"
                        type="range"
                        min="10"
                        max="100"
                        step="5"
                        value={fMinHz}
                        onChange={(e) => setFMinHz(parseFloat(e.target.value))}
                        className="w-full accent-[var(--accent)] cursor-pointer"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-[11px] text-[var(--muted)]">
                        <span>f_max</span>
                        <span className="font-mono">{fMaxHz} Hz</span>
                      </div>
                      <input
                        aria-label="Upper Frequency"
                        type="range"
                        min="128"
                        max="1024"
                        step="16"
                        value={fMaxHz}
                        onChange={(e) => setFMaxHz(parseFloat(e.target.value))}
                        className="w-full accent-[var(--accent)] cursor-pointer"
                      />
                    </div>
                  </div>
                </div>

                {/* Synthetic Seed */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex justify-between items-center">
                    <label htmlFor="mf-seed" className="font-sans font-medium text-[var(--foreground)]">
                      Synthetic Noise Seed
                    </label>
                    <div className="flex gap-1">
                      {[7151, 42].map((preset) => (
                        <button
                          key={preset}
                          type="button"
                          onClick={() => setSeed(preset)}
                          className={`px-1.5 py-0.5 text-[10px] font-mono rounded-[2px] border ${
                            seed === preset
                              ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--surface-subtle)]"
                              : "border-[var(--border)] text-[var(--muted)]"
                          }`}
                        >
                          {preset}
                        </button>
                      ))}
                    </div>
                  </div>
                  <input
                    id="mf-seed"
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seed}
                    onChange={(e) => {
                      const val = parseInt(e.target.value, 10);
                      if (!Number.isNaN(val) && val >= 0) setSeed(val);
                    }}
                    className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1 text-xs font-mono"
                  />
                </div>
              </div>
            </ScientificPanel>

            {/* Mathematical Definitions & Provenance */}
            <ScientificPanel
              title="Matched-Filter Conventions"
              subtitle="Frozen K₊ inner product & synthetic provenance"
            >
              <div className="space-y-2.5 text-xs text-[var(--foreground-secondary)] leading-relaxed">
                <MathBlock
                  math="\rho_{\mathrm{opt}} = \sigma_h = \sqrt{(h \mid h)_{K_+}}, \qquad \rho(t) = \frac{|z(t)|}{\sigma_h}"
                  display={true}
                  className="my-1 text-xs"
                  ariaLabel="Optimal SNR and matched filter SNR time series"
                />
                <div className="flex items-start gap-2 text-[11px] text-[var(--muted)]">
                  <Info size={14} className="shrink-0 mt-0.5 text-[var(--accent)]" />
                  <p>
                    Evaluated on a <strong>synthetic injection + synthetic colored Gaussian noise</strong>{" "}
                    under the analytical <span className="font-mono">aligo_representative</span> PSD
                    proxy (not real H1 telemetry). Archived Realization 01 optimal SNR reference is{" "}
                    <span className="font-mono">45.3153</span>; displayed values reflect your
                    active configuration.
                  </p>
                </div>
              </div>
            </ScientificPanel>
          </div>

          {/* Plot & Readouts Column */}
          <div className="lg:col-span-8 space-y-5">
            <ScientificPanel
              title="Matched-Filter Output & Spectral Comparison"
              subtitle={
                meta
                  ? `${meta.psd_model_label} · Band [${meta.f_min_hz}, ${meta.f_max_hz}] Hz (${meta.n_active_bins} bins) · Seed ${meta.seed}`
                  : "Loading matched filter..."
              }
            >
              <MatchedFilterPlot
                data={data}
                activeTab={activeTab}
                onTabChange={setActiveTab}
                loading={loading}
                error={error}
              />
            </ScientificPanel>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <NumericalReadout
                label="Optimal SNR ρ_opt"
                value={meta ? formatFloat(meta.optimal_snr, 2) : "—"}
                subtext="Template norm σ_h"
              />
              <NumericalReadout
                label="Recovered Peak SNR"
                value={meta ? formatFloat(meta.recovered_peak_snr, 2) : "—"}
                subtext={meta ? `Seed ${meta.seed} realization` : undefined}
              />
              <NumericalReadout
                label="Recovered Peak t_c"
                value={meta ? formatFloat(meta.recovered_peak_time_s, 4) : "—"}
                unit="s"
                subtext={
                  meta
                    ? `Δt = ${meta.time_offset_s >= 0 ? "+" : ""}${formatFloat(meta.time_offset_s * 1000, 2)} ms`
                    : undefined
                }
              />
              <NumericalReadout
                label="Recovered Phase φ_c"
                value={meta ? formatFloat(meta.recovered_peak_phase_rad, 3) : "—"}
                unit="rad"
                subtext={meta ? `f_ISCO = ${formatFloat(meta.f_isco_tmpl_hz, 1)} Hz` : undefined}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
