"use client";

import React, { useState, useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import NumericalReadout from "@/components/NumericalReadout";
import MathBlock from "@/components/MathBlock";
import { Marginal1DPlot, Joint2DPlot } from "@/components/PosteriorPlot";
import {
  fetchInferencePosterior,
  PosteriorExplorerResponseData,
  PosteriorParamKey,
} from "@/lib/api";
import { formatFloat } from "@/lib/format";
import { Activity, AlertTriangle, Info } from "lucide-react";

const PARAM_OPTIONS: { key: PosteriorParamKey; label: string }[] = [
  { key: "mc", label: "Chirp Mass M_c [M_sun]" },
  { key: "eta", label: "Symmetric Mass Ratio η [dimless]" },
  { key: "tc", label: "Coalescence Time t_c [s]" },
  { key: "phic", label: "Coalescence Phase φ_c [rad]" },
];

export default function InferencePage() {
  const [paramX, setParamX] = useState<PosteriorParamKey>("mc");
  const [paramY, setParamY] = useState<PosteriorParamKey>("eta");
  const [data, setData] = useState<PosteriorExplorerResponseData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stale = false;
    const loadPosterior = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchInferencePosterior({
          param_x: paramX,
          param_y: paramY,
          bins_1d: 40,
          bins_2d: 36,
        });
        if (!stale) setData(res);
      } catch (err) {
        if (!stale) {
          setError(err instanceof Error ? err.message : "Failed to load posterior archive.");
        }
      } finally {
        if (!stale) setLoading(false);
      }
    };

    loadPosterior();
    return () => {
      stale = true;
    };
  }, [paramX, paramY]);

  const handleParamXChange = (nextX: PosteriorParamKey) => {
    if (nextX === paramY) {
      setParamY(paramX);
    }
    setParamX(nextX);
  };

  const handleParamYChange = (nextY: PosteriorParamKey) => {
    if (nextY === paramX) {
      setParamX(paramY);
    }
    setParamY(nextY);
  };

  const meta = data?.metadata;

  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Inference"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <Activity size={13} />
            Saved Posterior Explorer
          </span>
        }
        subtitle="Bayesian parameter estimation, weighted nested-sampling marginals, and posterior diagnostics."
        description="Interactive exploration of the archived Stage 6C Run A 4D posterior (Mc [M☉], η [dimensionless], tc [s], φc [rad]) from Synthetic Realization 01."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        {error && (
          <div className="rounded-sm border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-600 dark:text-rose-400 font-sans">
            {error}
          </div>
        )}

        {/* Top Readouts */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <NumericalReadout
            label="Log-Evidence ln Z"
            value={meta ? formatFloat(meta.log_evidence, 4) : "—"}
            unit="nats"
            subtext={
              meta
                ? `± ${formatFloat(meta.log_evidence_err, 4)} internal (Model 2 norm)`
                : undefined
            }
          />
          <NumericalReadout
            label="Weighted Dead Points"
            value={meta ? meta.n_samples.toLocaleString() : "—"}
            subtext={meta ? `Kish N_eff = ${formatFloat(meta.kish_n_eff, 1)}` : undefined}
          />
          <NumericalReadout
            label="Likelihood Calls"
            value={meta ? meta.guard_evaluations_completed.toLocaleString() : "—"}
            subtext={meta ? `${meta.sampler}` : undefined}
          />
          <NumericalReadout
            label="Posterior mass in upper 1% of η prior range"
            value={meta ? `${formatFloat(meta.eta_top1_pct, 2)}%` : "—"}
            subtext="η ∈ [0.2485, 0.2500] of [0.10, 0.25]"
          />
        </div>

        {/* Main Explorer Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: 2D Joint Density Explorer + Metadata */}
          <div className="lg:col-span-6 space-y-5">
            <ScientificPanel
              title="2D Joint Marginal Density"
              subtitle={
                meta
                  ? `Weighted 36×36 density from ${meta.run_id} (Seed ${meta.seed})`
                  : "Loading 2D joint posterior..."
              }
            >
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div>
                    <label
                      htmlFor="posterior-x-select"
                      className="block font-sans font-medium text-[var(--foreground)] mb-1"
                    >
                      Horizontal Axis
                    </label>
                    <select
                      id="posterior-x-select"
                      value={paramX}
                      onChange={(e) => handleParamXChange(e.target.value as PosteriorParamKey)}
                      className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5 text-xs font-mono"
                    >
                      {PARAM_OPTIONS.map((opt) => (
                        <option key={opt.key} value={opt.key}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label
                      htmlFor="posterior-y-select"
                      className="block font-sans font-medium text-[var(--foreground)] mb-1"
                    >
                      Vertical Axis
                    </label>
                    <select
                      id="posterior-y-select"
                      value={paramY}
                      onChange={(e) => handleParamYChange(e.target.value as PosteriorParamKey)}
                      className="w-full rounded-[3px] border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5 text-xs font-mono"
                    >
                      {PARAM_OPTIONS.map((opt) => (
                        <option key={opt.key} value={opt.key}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {loading && !data ? (
                  <div className="h-64 flex items-center justify-center text-xs font-mono text-[var(--muted)]">
                    Loading weighted posterior archive...
                  </div>
                ) : data ? (
                  <Joint2DPlot joint={data.joint_2d} />
                ) : null}
              </div>
            </ScientificPanel>

            {/* Source Run Metadata & Explicit Qualifications */}
            <ScientificPanel
              title="Source Run Provenance & Limitations"
              subtitle="Read-only archive inspection (no live sampler execution)"
            >
              <div className="space-y-3 text-xs text-[var(--foreground-secondary)] leading-relaxed">
                <div className="rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-3 font-mono text-[11px] space-y-1">
                  <div className="break-all">
                    &bull; Archive: {meta ? meta.artifact_npz : "experiments/wp6_stage6c_anchor/run_a_anchor_samples.npz"}
                  </div>
                  <div>
                    &bull; Run: {meta ? `${meta.run_id} · ${meta.realization} · Seed ${meta.seed}` : "run_a_anchor · Synthetic Realization 01 · Seed 8101"}
                  </div>
                  <div>
                    &bull; Likelihood &amp; Prior: {meta ? `${meta.likelihood_model} · ${meta.prior_model}` : "Model 2 (Plug-in Gaussian Whittle) · 4D Uniform Prior"}
                  </div>
                </div>

                <div className="flex items-start gap-2 rounded-xs border-l-2 border-[var(--accent-border)] bg-[var(--surface-subtle)] p-3 text-[11px]">
                  <AlertTriangle size={14} className="shrink-0 mt-0.5 text-[var(--accent)]" />
                  <div className="space-y-1.5">
                    <p>
                      {meta
                        ? meta.qualification
                        : "Saved synthetic single-realization posterior (Realization 01, Seed 8101), not a real detector event. Stage 6D reported between-seed evidence sensitivity against seed 8201 of |Δln Z| = 2.9720 nats. This archive does not establish full convergence, boundary-artifact freedom, or empirical coverage calibration."}
                    </p>
                    <p>
                      <strong>Boundary-concentration diagnostic:</strong>{" "}
                      {meta ? formatFloat(meta.eta_top1_pct, 2) : "38.57"}% of posterior mass lies in
                      the upper 1% of the <span className="font-mono">η ∈ [0.10, 0.25]</span> prior
                      range (<span className="font-mono">[0.2485, 0.2500]</span>). This is a
                      descriptive boundary diagnostic; the existing single-seed campaign does not
                      establish boundary-artifact freedom. Moreover,{" "}
                      <span className="font-mono">ln Z</span> includes the frozen Model 2 Whittle
                      likelihood normalization over <span className="font-mono">[20, 512] Hz</span>{" "}
                      on Realization 01 and is comparable only under identical data, band, and
                      normalization conventions.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-2 text-[11px] text-[var(--muted)]">
                  <Info size={14} className="shrink-0 mt-0.5 text-[var(--accent)]" />
                  <p>
                    {meta
                      ? meta.binning_note
                      : "Weighted histograms are visualization choices over weighted nested-sampling dead points, not resampled unweighted draws."}
                  </p>
                </div>
              </div>
            </ScientificPanel>
          </div>

          {/* Right: 4x 1D Marginal Plots + Summary Table */}
          <div className="lg:col-span-6 space-y-5">
            <ScientificPanel
              title="1D Weighted Marginal Posteriors"
              subtitle="Solid black = weighted median; shaded = 90% CI; dashed blue = injected truth"
            >
              {loading && !data ? (
                <div className="h-64 flex items-center justify-center text-xs font-mono text-[var(--muted)]">
                  Computing 1D weighted marginals...
                </div>
              ) : data ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {data.parameter_order.map((key) => (
                    <Marginal1DPlot key={key} marginal={data.marginals_1d[key]} />
                  ))}
                </div>
              ) : null}
            </ScientificPanel>

            {/* Weighted Parameter Summary Table */}
            <ScientificPanel
              title="Weighted Posterior Summaries"
              subtitle="Authoritative Stage 6C Run A quantiles"
            >
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)] text-[var(--muted)] font-sans uppercase tracking-wider text-[10px]">
                      <th className="py-2 pr-2 font-normal">Parameter</th>
                      <th className="py-2 px-2 font-normal text-right">Injected</th>
                      <th className="py-2 px-2 font-normal text-right">Median</th>
                      <th className="py-2 pl-2 font-normal text-right">90% CI [p5, p95]</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)] font-mono text-[11px]">
                    {data &&
                      data.parameter_order.map((key) => {
                        const m = data.marginals_1d[key];
                        const dec = key === "tc" || key === "eta" ? 4 : 4;
                        return (
                          <tr key={key}>
                            <td className="py-2 pr-2 font-sans text-[var(--foreground)]">
                              {m.symbol}{" "}
                              <span className="text-[10px] text-[var(--muted)]">[{m.unit}]</span>
                            </td>
                            <td className="py-2 px-2 text-right text-[var(--muted)]">
                              {formatFloat(m.injected, dec)}
                            </td>
                            <td className="py-2 px-2 text-right font-semibold text-[var(--foreground)]">
                              {formatFloat(m.summary.median, dec)}
                            </td>
                            <td className="py-2 pl-2 text-right text-[var(--foreground-secondary)]">
                              [{formatFloat(m.summary.p5, dec)}, {formatFloat(m.summary.p95, dec)}]
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
              <div className="mt-3 pt-2 border-t border-[var(--border-subtle)]">
                <MathBlock
                  math="p(\boldsymbol{\theta} \mid d) = \frac{\mathcal{L}_{\mathrm{Model\,2}}(d \mid \boldsymbol{\theta})\,\Pi(\boldsymbol{\theta})}{\mathcal{Z}}, \qquad \ln\mathcal{Z} = 201646.1413 \pm 0.2374\,\mathrm{nats}"
                  display={true}
                  className="my-0.5 text-xs"
                  ariaLabel="Bayes theorem for Stage 6C Run A"
                />
              </div>
            </ScientificPanel>
          </div>
        </div>
      </div>
    </div>
  );
}
