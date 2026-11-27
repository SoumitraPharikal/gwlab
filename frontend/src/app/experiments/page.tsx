"use client";

import React, { useState, useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import ScientificPanel from "@/components/ScientificPanel";
import NumericalReadout from "@/components/NumericalReadout";
import {
  fetchExperimentsExplorer,
  ExperimentsExplorerResponseData,
  CampaignStageStatus,
} from "@/lib/api";
import { formatFloat } from "@/lib/format";
import { Activity, AlertTriangle, CheckCircle2, Clock, PauseCircle, FileText } from "lucide-react";

type StatusFilter = "all" | "completed" | "non_completed";

function StatusBadge({ status }: { status: CampaignStageStatus }) {
  if (status === "Completed / Historical") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[3px] text-[11px] font-sans font-medium bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/30">
        <CheckCircle2 size={11} />
        {status}
      </span>
    );
  }
  if (status === "Frozen / Specified") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[3px] text-[11px] font-sans font-medium bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/30">
        <PauseCircle size={11} />
        {status}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[3px] text-[11px] font-sans font-medium bg-[var(--surface-subtle)] text-[var(--muted)] border border-[var(--border)]">
      <Clock size={11} />
      {status}
    </span>
  );
}

export default function ExperimentsPage() {
  const [data, setData] = useState<ExperimentsExplorerResponseData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<StatusFilter>("all");

  useEffect(() => {
    let stale = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchExperimentsExplorer();
        if (!stale) setData(res);
      } catch (err) {
        if (!stale) {
          setError(err instanceof Error ? err.message : "Failed to load experiment artifacts.");
        }
      } finally {
        if (!stale) setLoading(false);
      }
    };
    load();
    return () => {
      stale = true;
    };
  }, []);

  const comp = data?.multi_seed_comparison;
  const r1 = comp?.runs[0];
  const r2 = comp?.runs[1];

  const filteredStages = (data?.campaign_stages ?? []).filter((stage) => {
    if (filter === "completed") return stage.status === "Completed / Historical";
    if (filter === "non_completed") return stage.status !== "Completed / Historical";
    return true;
  });

  return (
    <div className="flex-1 pb-16">
      <PageHeader
        title="Experiments"
        badge={
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[3px] text-xs font-sans font-medium bg-[var(--surface-subtle)] text-[var(--accent)] border border-[var(--border)]">
            <Activity size={13} />
            Read-Only Saved Archive
          </span>
        }
        subtitle="Saved experiment registry, Stage 5G.2 two-seed comparison, and campaign status."
        description="Read-only explorer of completed historical runs (Stage 5G.2, Stage 6C Run A, Stage 6D closeout) alongside frozen, proposed, and unexecuted work on Synthetic Realization 01."
      />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 mt-6 space-y-6">
        {error && (
          <div className="rounded-sm border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-600 dark:text-rose-400 font-sans">
            {error}
          </div>
        )}

        {/* Section A: Two-Seed Sensitivity Diagnostic */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <NumericalReadout
            label="Evidence Difference |Δln Z|"
            value={comp ? formatFloat(comp.delta_logz, 4) : "—"}
            unit="nats"
            subtext="Seed 8101 vs Seed 8201"
          />
          <NumericalReadout
            label="Quadrature Uncertainty σ_Δ"
            value={comp ? formatFloat(comp.sigma_delta, 4) : "—"}
            unit="nats"
            subtext="√(σ₁² + σ₂²) internal"
          />
          <NumericalReadout
            label="Discrepancy / Uncertainty"
            value={
              comp && comp.discrepancy_ratio !== null
                ? formatFloat(comp.discrepancy_ratio, 2)
                : "—"
            }
            subtext="2-seed sensitivity ratio"
          />
        </div>

        <ScientificPanel
          title="Stage 5G.2 — Two-Seed Evidence & Posterior Sensitivity"
          subtitle="Paired Model 2 (Plug-in Gaussian Whittle) runs on Synthetic Realization 01"
        >
          {loading && !data ? (
            <div className="h-36 flex items-center justify-center text-xs font-mono text-[var(--muted)]">
              Loading saved Stage 5G.2 artifacts...
            </div>
          ) : r1 && r2 ? (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)] text-[var(--muted)] font-sans uppercase tracking-wider text-[10px]">
                      <th className="py-2 pr-3 font-normal">Metric</th>
                      <th className="py-2 px-3 font-normal text-right">
                        {r1.run_id} (Seed {r1.sampler_seed})
                      </th>
                      <th className="py-2 pl-3 font-normal text-right">
                        {r2.run_id} (Seed {r2.sampler_seed})
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)] font-mono text-[11px]">
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Model &amp; Realization
                      </td>
                      <td className="py-2 px-3 text-right">
                        {r1.model} &middot; {r1.realization}
                      </td>
                      <td className="py-2 pl-3 text-right">
                        {r2.model} &middot; {r2.realization}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Log-Evidence ln Z [nats]
                      </td>
                      <td className="py-2 px-3 text-right font-semibold text-[var(--foreground)]">
                        {formatFloat(r1.log_evidence, 4)} &plusmn; {formatFloat(r1.log_evidence_err, 4)}
                      </td>
                      <td className="py-2 pl-3 text-right font-semibold text-[var(--foreground)]">
                        {formatFloat(r2.log_evidence, 4)} &plusmn; {formatFloat(r2.log_evidence_err, 4)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Likelihood Evaluations
                      </td>
                      <td className="py-2 px-3 text-right">
                        {r1.n_evaluations !== null ? r1.n_evaluations.toLocaleString() : "N/A"}
                      </td>
                      <td className="py-2 pl-3 text-right">
                        {r2.n_evaluations !== null ? r2.n_evaluations.toLocaleString() : "N/A"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Weighted Dead Points / Kish N_eff
                      </td>
                      <td className="py-2 px-3 text-right">
                        {r1.n_total_samples !== null ? r1.n_total_samples.toLocaleString() : "N/A"} /{" "}
                        {r1.n_eff !== null ? formatFloat(r1.n_eff, 1) : "N/A"}
                      </td>
                      <td className="py-2 pl-3 text-right">
                        {r2.n_total_samples !== null ? r2.n_total_samples.toLocaleString() : "N/A"} /{" "}
                        {r2.n_eff !== null ? formatFloat(r2.n_eff, 1) : "N/A"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Medians (M_c [M_☉], η, t_c [s])
                      </td>
                      <td className="py-2 px-3 text-right">
                        {r1.mc_median !== null ? formatFloat(r1.mc_median, 4) : "N/A"},{" "}
                        {r1.eta_median !== null ? formatFloat(r1.eta_median, 4) : "N/A"},{" "}
                        {r1.tc_median !== null ? formatFloat(r1.tc_median, 4) : "N/A"}
                      </td>
                      <td className="py-2 pl-3 text-right">
                        {r2.mc_median !== null ? formatFloat(r2.mc_median, 4) : "N/A"},{" "}
                        {r2.eta_median !== null ? formatFloat(r2.eta_median, 4) : "N/A"},{" "}
                        {r2.tc_median !== null ? formatFloat(r2.tc_median, 4) : "N/A"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-3 font-sans text-[var(--foreground-secondary)]">
                        Source Artifact
                      </td>
                      <td className="py-2 px-3 text-right text-[10px] text-[var(--muted)] break-all">
                        {r1.artifact_path}
                      </td>
                      <td className="py-2 pl-3 text-right text-[10px] text-[var(--muted)] break-all">
                        {r2.artifact_path}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="flex items-start gap-2 rounded-xs border-l-2 border-[var(--accent-border)] bg-[var(--surface-subtle)] p-3 text-xs text-[var(--foreground-secondary)]">
                <AlertTriangle size={14} className="shrink-0 mt-0.5 text-[var(--accent)]" />
                <p>{comp?.qualification}</p>
              </div>
            </div>
          ) : null}
        </ScientificPanel>

        {/* Section B: Campaign Timeline & Status */}
        <ScientificPanel
          title="Campaign Timeline & Execution Status"
          subtitle="Strict separation between completed historical runs and frozen, proposed, or unexecuted work"
          action={
            <div className="inline-flex flex-wrap rounded-[3px] border border-[var(--border)] bg-[var(--surface-subtle)] p-0.5 text-xs">
              {(
                [
                  { id: "all", label: "All" },
                  { id: "completed", label: "Completed" },
                  { id: "non_completed", label: "Frozen / Proposed / Unexecuted" },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setFilter(tab.id)}
                  className={`px-2 py-0.5 rounded-[2px] font-sans transition-colors ${
                    filter === tab.id
                      ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-2xs"
                      : "text-[var(--muted)] hover:text-[var(--foreground)]"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          }
        >
          <div className="space-y-3">
            {filteredStages.map((stage) => (
              <div
                key={stage.stage_id}
                className="rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-3.5 space-y-1.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="font-sans font-semibold text-xs text-[var(--foreground)]">
                    {stage.title}
                  </div>
                  <StatusBadge status={stage.status} />
                </div>
                <div className="text-[11px] font-mono text-[var(--muted)]">
                  {stage.model} &middot; {stage.realization}
                </div>
                <p className="text-xs text-[var(--foreground-secondary)] leading-relaxed">
                  {stage.summary}
                </p>
                <div className="flex items-start gap-1.5 text-[10px] font-mono text-[var(--muted-light)] break-all pt-0.5">
                  <FileText size={11} className="shrink-0 mt-0.5" />
                  <span>{stage.artifact_path}</span>
                </div>
              </div>
            ))}
          </div>
        </ScientificPanel>

        {/* Section C: Provenance & Limitations */}
        <ScientificPanel
          title="Archive Provenance & Scientific Limitations"
          subtitle="Read-only display of saved JSON artifacts (no live sampler execution)"
        >
          <ul className="space-y-1.5 text-xs text-[var(--foreground-secondary)] leading-relaxed">
            {(data?.limitations ?? []).map((item, idx) => (
              <li key={idx}>&bull; {item}</li>
            ))}
          </ul>
        </ScientificPanel>
      </div>
    </div>
  );
}
