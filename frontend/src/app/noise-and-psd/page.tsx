"use client";

import React, { useState, useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import NumericalReadout from "@/components/NumericalReadout";
import MathBlock from "@/components/MathBlock";
import NoisePsdPlot, { NoisePsdTab } from "@/components/NoisePsdPlot";
import { analyzeNoiseAndPsd, NoiseAnalysisResponseData } from "@/lib/api";
import { formatFloat, formatScientific } from "@/lib/format";
import { Activity, RotateCcw, Sliders, Info } from "lucide-react";

const FROZEN_STAGE5B_DEFAULTS = {
  psd_model: "aligo_representative" as const,
  f_min_hz: 20.0,
  f_max_hz: 512.0,
  duration_obs_s: 64.0,
  segment_duration_s: 4.0,
  overlap_fraction: 0.5,
  tukey_alpha: 0.25,
  seed: 7101,
};

export default function NoiseAndPsdPage() {
  const [psdModel, setPsdModel] = useState<"aligo_representative" | "flat">(
    FROZEN_STAGE5B_DEFAULTS.psd_model
  );
  const [fMinHz, setFMinHz] = useState<number>(FROZEN_STAGE5B_DEFAULTS.f_min_hz);
  const [fMaxHz, setFMaxHz] = useState<number>(FROZEN_STAGE5B_DEFAULTS.f_max_hz);
  const [segmentDurationS, setSegmentDurationS] = useState<number>(
    FROZEN_STAGE5B_DEFAULTS.segment_duration_s
  );
  const [overlapFraction, setOverlapFraction] = useState<number>(
    FROZEN_STAGE5B_DEFAULTS.overlap_fraction
  );
  const [tukeyAlpha, setTukeyAlpha] = useState<number>(FROZEN_STAGE5B_DEFAULTS.tukey_alpha);
  const [seed, setSeed] = useState<number>(FROZEN_STAGE5B_DEFAULTS.seed);

  const [activeTab, setActiveTab] = useState<NoisePsdTab>("asd");
  const [data, setData] = useState<NoiseAnalysisResponseData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    const runAnalysis = async () => {
      if (fMinHz >= fMaxHz) {
        setError("Lower frequency f_min must be strictly less than upper frequency f_max.");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await analyzeNoiseAndPsd({
          psd_model: psdModel,
          f_min_hz: fMinHz,
          f_max_hz: fMaxHz,
          sampling_rate_hz: 2048.0,
          duration_obs_s: FROZEN_STAGE5B_DEFAULTS.duration_obs_s,
          segment_duration_s: segmentDurationS,
          overlap_fraction: overlapFraction,
          tukey_alpha: tukeyAlpha,
          seed,
        });
        if (!stale) setData(res);
      } catch (err) {
        if (!stale) {
          setError(err instanceof Error ? err.message : "Failed to compute Noise & PSD.");
        }
      } finally {
        if (!stale) setLoading(false);
      }
    };

    runAnalysis();
    return () => {
      stale = true;
    };
  }, [psdModel, fMinHz, fMaxHz, segmentDurationS, overlapFraction, tukeyAlpha, seed]);

  const handleReset = () => {
    setPsdModel(FROZEN_STAGE5B_DEFAULTS.psd_model);
    setFMinHz(FROZEN_STAGE5B_DEFAULTS.f_min_hz);
    setFMaxHz(FROZEN_STAGE5B_DEFAULTS.f_max_hz);
    setSegmentDurationS(FROZEN_STAGE5B_DEFAULTS.segment_duration_s);
    setOverlapFraction(FROZEN_STAGE5B_DEFAULTS.overlap_fraction);
    setTukeyAlpha(FROZEN_STAGE5B_DEFAULTS.tukey_alpha);
    setSeed(FROZEN_STAGE5B_DEFAULTS.seed);
  };

  const meta = data?.metadata;

  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Noise and PSD"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <Activity size={13} />
            Interactive Spectral Engine
          </span>
        }
        subtitle="Analytical PSD proxy comparison, deterministic synthetic colored noise, and Welch estimation."
        description="Explore one-sided Power Spectral Density Sₙ(f) [strain²/Hz], Amplitude Spectral Density √Sₙ(f) [strain/√Hz], and Tukey-windowed Welch averages on synthetic Gaussian noise realizations."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column */}
          <div className="lg:col-span-4 space-y-5">
            <ScientificPanel
              title="PSD & Welch Controls"
              subtitle="Frozen Stage 5B defaults (64 s @ 2048 Hz)"
              action={
                <button
                  type="button"
                  onClick={handleReset}
                  className="flex items-center gap-1 px-2 py-1 text-xs font-sans rounded-[3px] text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-subtle)] transition-colors border border-[var(--border)]"
                  title="Reset to frozen Stage 5B benchmark configuration"
                >
                  <RotateCcw size={12} />
                  <span>Reset</span>
                </button>
              }
            >
              <div className="space-y-4 text-xs">
                {/* PSD Model Proxy */}
                <div className="space-y-1.5">
                  <label
                    htmlFor="psd-model-select"
                    className="block font-sans font-medium text-[var(--foreground)]"
                  >
                    Analytical PSD Model Proxy
                  </label>
                  <select
                    id="psd-model-select"
                    value={psdModel}
                    onChange={(e) =>
                      setPsdModel(e.target.value as "aligo_representative" | "flat")
                    }
                    className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5 text-xs font-mono text-[var(--foreground)]"
                  >
                    <option value="aligo_representative">
                      aligo_representative (aLIGO-like proxy)
                    </option>
                    <option value="flat">flat (White noise 1e-46 strain²/Hz)</option>
                  </select>
                </div>

                {/* Frequency Band Controls */}
                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <label
                      htmlFor="fmin-slider"
                      className="font-sans font-medium text-[var(--foreground)]"
                    >
                      Lower Frequency <em>f</em>
                      <sub>min</sub>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(fMinHz, 1)} Hz
                    </span>
                  </div>
                  <input
                    id="fmin-slider"
                    type="range"
                    min="5"
                    max="100"
                    step="1"
                    value={fMinHz}
                    onChange={(e) => setFMinHz(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <label
                      htmlFor="fmax-slider"
                      className="font-sans font-medium text-[var(--foreground)]"
                    >
                      Upper Frequency <em>f</em>
                      <sub>max</sub>
                    </label>
                    <span className="font-mono text-[var(--accent)] font-semibold">
                      {formatFloat(fMaxHz, 1)} Hz
                    </span>
                  </div>
                  <input
                    id="fmax-slider"
                    type="range"
                    min="128"
                    max="1024"
                    step="16"
                    value={fMaxHz}
                    onChange={(e) => setFMaxHz(parseFloat(e.target.value))}
                    className="w-full accent-[var(--accent)] cursor-pointer"
                  />
                </div>

                {/* Synthetic Noise Seed */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex justify-between items-center">
                    <label
                      htmlFor="seed-input"
                      className="font-sans font-medium text-[var(--foreground)]"
                    >
                      Synthetic Noise PRNG Seed
                    </label>
                    <div className="flex gap-1">
                      {[7101, 42].map((preset) => (
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
                    id="seed-input"
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seed}
                    onChange={(e) => {
                      const val = parseInt(e.target.value, 10);
                      if (!Number.isNaN(val) && val >= 0) setSeed(val);
                    }}
                    className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5 text-xs font-mono text-[var(--foreground)]"
                  />
                </div>

                {/* Welch Controls */}
                <div className="space-y-3 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 font-sans font-medium text-[var(--foreground)]">
                    <Sliders size={13} className="text-[var(--accent)]" />
                    <span>Welch Averaging Parameters</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label
                        htmlFor="seg-select"
                        className="block text-[11px] text-[var(--muted)] mb-1"
                      >
                        Segment Length
                      </label>
                      <select
                        id="seg-select"
                        value={segmentDurationS}
                        onChange={(e) => setSegmentDurationS(parseFloat(e.target.value))}
                        className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs font-mono"
                      >
                        <option value={2}>2.0 s</option>
                        <option value={4}>4.0 s</option>
                        <option value={8}>8.0 s</option>
                      </select>
                    </div>
                    <div>
                      <label
                        htmlFor="overlap-select"
                        className="block text-[11px] text-[var(--muted)] mb-1"
                      >
                        Segment Overlap
                      </label>
                      <select
                        id="overlap-select"
                        value={overlapFraction}
                        onChange={(e) => setOverlapFraction(parseFloat(e.target.value))}
                        className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs font-mono"
                      >
                        <option value={0}>0% (indep.)</option>
                        <option value={0.25}>25%</option>
                        <option value={0.5}>50% (frozen)</option>
                      </select>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <label htmlFor="tukey-slider" className="text-[11px] text-[var(--muted)]">
                        Tukey Window Taper &alpha;
                      </label>
                      <span className="font-mono text-[var(--foreground)] font-medium">
                        {formatFloat(tukeyAlpha, 2)}
                      </span>
                    </div>
                    <input
                      id="tukey-slider"
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.05"
                      value={tukeyAlpha}
                      onChange={(e) => setTukeyAlpha(parseFloat(e.target.value))}
                      className="w-full accent-[var(--accent)] cursor-pointer"
                    />
                  </div>
                </div>
              </div>
            </ScientificPanel>

            {/* Scientific Provenance & Qualification */}
            <ScientificPanel
              title="Provenance & Qualifications"
              subtitle="Model boundaries and Welch DOF assumptions"
            >
              <div className="space-y-2.5 text-xs text-[var(--foreground-secondary)] leading-relaxed">
                <div className="flex items-start gap-2">
                  <Info size={14} className="shrink-0 mt-0.5 text-[var(--accent)]" />
                  <p>
                    <strong>{meta ? meta.psd_model_label : "Analytical aLIGO-like PSD proxy"}:</strong>{" "}
                    {meta
                      ? meta.psd_provenance
                      : "Custom, regularized representative noise model for controlled numerical benchmarking; not an official design-sensitivity curve or measured H1 spectrum."}
                  </p>
                </div>
                <div className="rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-2.5 space-y-1">
                  <MathBlock
                    math={
                      meta
                        ? `\\nu_{\\mathrm{eff}} = \\frac{2K}{\\kappa_{\\mathrm{overlap}}} = \\frac{${2 * meta.n_segments}}{${formatFloat(meta.kappa_overlap, 4)}} \\approx ${formatFloat(meta.nu_eff, 4)}`
                        : "\\nu_{\\mathrm{eff}} = \\frac{2K}{\\kappa_{\\mathrm{overlap}}} = \\frac{62}{1157/837} \\approx 44.8522"
                    }
                    display={true}
                    className="my-0.5 text-xs"
                    ariaLabel="Effective Welch degrees of freedom formula"
                  />
                  <p className="text-[11px] text-[var(--muted)]">
                    Variance-matched effective DOF approximation for this Welch setup; not an exact
                    predictive or Bayesian DOF for overlapping tapered colored-noise PSD estimates.
                  </p>
                </div>
              </div>
            </ScientificPanel>
          </div>

          {/* Plot & Readouts Column */}
          <div className="lg:col-span-8 space-y-5">
            <ScientificPanel
              title="Spectral Density & Synthetic Noise Visualization"
              subtitle={
                meta
                  ? `${meta.psd_model_label} vs Welch estimate (${meta.n_segments} segments, Δf = ${meta.delta_f_hz} Hz, seed = ${meta.seed})`
                  : "Loading spectral analysis..."
              }
            >
              <NoisePsdPlot
                data={data}
                activeTab={activeTab}
                onTabChange={setActiveTab}
                loading={loading}
                error={error}
              />
            </ScientificPanel>

            {/* Welch & Noise Readouts */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <NumericalReadout
                label="Welch Segments K"
                value={meta ? String(meta.n_segments) : "—"}
                subtext={meta ? `${meta.duration_obs_s} s / ${meta.segment_duration_s} s` : undefined}
              />
              <NumericalReadout
                label="Inflation κ_overlap"
                value={meta ? formatFloat(meta.kappa_overlap, 4) : "—"}
                subtext={meta ? `Tukey α = ${formatFloat(meta.tukey_alpha, 2)}` : undefined}
              />
              <NumericalReadout
                label="Effective DOF ν_eff"
                value={meta ? formatFloat(meta.nu_eff, 2) : "—"}
                subtext="2K / κ_overlap"
              />
              <NumericalReadout
                label="Synthetic Noise σₙ"
                value={meta ? formatScientific(meta.noise_std, { sigFigs: 2 }) : "—"}
                subtext={meta ? `${meta.n_active_bins} active bins` : undefined}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
