"use client";

import React, { useMemo, useState, useRef } from "react";
import { NoiseAnalysisResponseData } from "@/lib/api";
import { formatFloat, formatScientific } from "@/lib/format";
import { Activity, BarChart3, Waves } from "lucide-react";

export type NoisePsdTab = "asd" | "psd" | "timeseries";

interface NoisePsdPlotProps {
  data: NoiseAnalysisResponseData | null;
  activeTab: NoisePsdTab;
  onTabChange: (tab: NoisePsdTab) => void;
  loading?: boolean;
  error?: string | null;
  className?: string;
}

interface PlotSeriesPoint {
  xVal: number;
  yAnalytic: number;
  yWelch?: number;
  px: number;
  pyAnalytic: number;
  pyWelch?: number;
}

export function NoisePsdPlot({
  data,
  activeTab,
  onTabChange,
  loading = false,
  error = null,
  className = "",
}: NoisePsdPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 800;
  const height = 430;
  const margin = { top: 28, right: 30, bottom: 52, left: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const computed = useMemo(() => {
    if (!data) return null;

    if (activeTab === "asd" || activeTab === "psd") {
      const freqs = data.frequencies;
      const analytic = activeTab === "asd" ? data.asd_analytic : data.psd_analytic;
      const welch = activeTab === "asd" ? data.asd_welch : data.psd_welch;
      const valid = freqs
        .map((f, i) => ({ f, a: analytic[i], w: welch[i] }))
        .filter((p) => p.f > 0 && p.a > 0 && p.w > 0);
      if (valid.length === 0) return null;

      const minF = Math.min(...valid.map((p) => p.f));
      const maxF = Math.max(...valid.map((p) => p.f));
      const allY = valid.flatMap((p) => [p.a, p.w]);
      const minY = Math.min(...allY) * 0.85;
      const maxY = Math.max(...allY) * 1.18;

      const logMinF = Math.log10(minF);
      const logMaxF = Math.log10(maxF);
      const logMinY = Math.log10(minY);
      const logMaxY = Math.log10(maxY);

      const toX = (f: number) =>
        margin.left + ((Math.log10(f) - logMinF) / Math.max(1e-9, logMaxF - logMinF)) * plotWidth;
      const toY = (v: number) =>
        margin.top +
        plotHeight -
        ((Math.log10(v) - logMinY) / Math.max(1e-9, logMaxY - logMinY)) * plotHeight;

      const pathAnalytic = valid
        .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.f).toFixed(1)} ${toY(p.a).toFixed(1)}`)
        .join(" ");
      const pathWelch = valid
        .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.f).toFixed(1)} ${toY(p.w).toFixed(1)}`)
        .join(" ");

      const xTicks = [10, 20, 50, 100, 200, 500, 1000].filter((f) => f >= minF && f <= maxF);
      const yTicks: number[] = [];
      for (let exp = Math.ceil(logMinY); exp <= Math.floor(logMaxY); exp++) {
        yTicks.push(Math.pow(10, exp));
      }

      const points: PlotSeriesPoint[] = valid.map((p) => ({
        xVal: p.f,
        yAnalytic: p.a,
        yWelch: p.w,
        px: toX(p.f),
        pyAnalytic: toY(p.a),
        pyWelch: toY(p.w),
      }));

      return {
        mode: activeTab,
        toX,
        toY,
        xTicks,
        yTicks,
        pathAnalytic,
        pathWelch,
        points,
        xLabel: "Frequency f [Hz] (log scale)",
        yLabel:
          activeTab === "asd"
            ? "ASD √Sₙ(f) [strain/√Hz]"
            : "PSD Sₙ(f) [strain²/Hz]",
      };
    }

    // Time-series preview
    const times = data.time_preview_s;
    const strains = data.strain_preview;
    if (times.length === 0 || strains.length !== times.length) return null;

    const minT = times[0];
    const maxT = times[times.length - 1];
    const maxAbs = Math.max(...strains.map((v) => Math.abs(v)), 1e-30) * 1.12;

    const toX = (t: number) =>
      margin.left + ((t - minT) / Math.max(1e-9, maxT - minT)) * plotWidth;
    const toY = (s: number) =>
      margin.top + plotHeight - ((s + maxAbs) / (2 * maxAbs)) * plotHeight;

    const pathSeries = times
      .map((t, i) => `${i === 0 ? "M" : "L"} ${toX(t).toFixed(1)} ${toY(strains[i]).toFixed(1)}`)
      .join(" ");

    const xTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((t) => t >= minT && t <= maxT + 1e-6);
    const yTicks = [-maxAbs * 0.75, 0, maxAbs * 0.75];

    const points: PlotSeriesPoint[] = times.map((t, i) => ({
      xVal: t,
      yAnalytic: strains[i],
      px: toX(t),
      pyAnalytic: toY(strains[i]),
    }));

    return {
      mode: "timeseries" as const,
      toX,
      toY,
      xTicks,
      yTicks,
      pathSeries,
      points,
      xLabel: "Time t [s] (first 0.5 s preview)",
      yLabel: "Synthetic Strain n(t) [dimensionless]",
    };
  }, [data, activeTab, margin.left, margin.top, plotWidth, plotHeight]);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || !computed || computed.points.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = width / rect.width;
    const mouseX = (e.clientX - rect.left) * scaleX;
    if (mouseX < margin.left || mouseX > width - margin.right) {
      setHoverIdx(null);
      return;
    }

    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < computed.points.length; i++) {
      const dist = Math.abs(computed.points[i].px - mouseX);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    setHoverIdx(bestIdx);
  };

  const activePoint = hoverIdx !== null && computed ? computed.points[hoverIdx] : null;

  const tabs: { id: NoisePsdTab; label: string; icon: React.ReactNode }[] = [
    { id: "asd", label: "ASD (√Sₙ)", icon: <Activity size={13} /> },
    { id: "psd", label: "PSD (Sₙ)", icon: <BarChart3 size={13} /> },
    { id: "timeseries", label: "Synthetic Noise n(t)", icon: <Waves size={13} /> },
  ];

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Tab Selector + Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex flex-wrap rounded-[3px] border border-[var(--border)] bg-[var(--surface-subtle)] p-0.5">
          {tabs.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => {
                  onTabChange(tab.id);
                  setHoverIdx(null);
                }}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-sans rounded-[2px] transition-colors ${
                  active
                    ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-2xs"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {computed && computed.mode !== "timeseries" && (
          <div className="flex items-center gap-4 text-[11px] font-sans text-[var(--foreground-secondary)]">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block w-3.5 h-0.5 bg-[var(--accent)]" />
              Analytical PSD Proxy
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block w-3.5 h-0.5 bg-[#3b82f6]" />
              Welch Estimate (Synthetic)
            </span>
          </div>
        )}
      </div>

      {/* Error State */}
      {error && (
        <div className="rounded-sm border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-400 font-sans">
          {error}
        </div>
      )}

      {/* Plot Canvas */}
      <div className="relative rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-2">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--surface)]/60 backdrop-blur-[1px] text-xs font-mono text-[var(--muted)]">
            Computing Welch PSD &amp; synthetic noise...
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block select-none"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIdx(null)}
          role="img"
          aria-label="Noise and Power Spectral Density interactive plot"
        >
          {/* Plot Background */}
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            fill="var(--surface)"
            stroke="var(--border)"
            strokeWidth={1}
          />

          {computed && (
            <>
              {/* Vertical Grid Lines & X Ticks */}
              {computed.xTicks.map((tick) => {
                const x = computed.toX(tick);
                return (
                  <g key={`x-${tick}`}>
                    <line
                      x1={x}
                      y1={margin.top}
                      x2={x}
                      y2={margin.top + plotHeight}
                      stroke="var(--border-subtle)"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={x}
                      y={margin.top + plotHeight + 18}
                      textAnchor="middle"
                      className="fill-[var(--muted)] text-[10px] font-mono"
                    >
                      {computed.mode === "timeseries" ? formatFloat(tick, 1) : tick}
                    </text>
                  </g>
                );
              })}

              {/* Horizontal Grid Lines & Y Ticks */}
              {computed.yTicks.map((tick, idx) => {
                const y = computed.toY(tick);
                if (y < margin.top || y > margin.top + plotHeight) return null;
                return (
                  <g key={`y-${idx}`}>
                    <line
                      x1={margin.left}
                      y1={y}
                      x2={margin.left + plotWidth}
                      y2={y}
                      stroke="var(--border-subtle)"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={margin.left - 8}
                      y={y + 3}
                      textAnchor="end"
                      className="fill-[var(--muted)] text-[10px] font-mono"
                    >
                      {tick === 0 ? "0" : formatScientific(tick, { sigFigs: 1 })}
                    </text>
                  </g>
                );
              })}

              {/* Data Curves */}
              {computed.mode === "timeseries" ? (
                <path
                  d={computed.pathSeries}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={1.3}
                  opacity={0.9}
                />
              ) : (
                <>
                  <path
                    d={computed.pathWelch}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth={1.4}
                    opacity={0.78}
                  />
                  <path
                    d={computed.pathAnalytic}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth={2.2}
                  />
                </>
              )}

              {/* Axis Labels */}
              <text
                x={margin.left + plotWidth / 2}
                y={height - 10}
                textAnchor="middle"
                className="fill-[var(--foreground-secondary)] text-[11px] font-sans"
              >
                {computed.xLabel}
              </text>
              <text
                x={16}
                y={margin.top + plotHeight / 2}
                textAnchor="middle"
                transform={`rotate(-90 16 ${margin.top + plotHeight / 2})`}
                className="fill-[var(--foreground-secondary)] text-[11px] font-sans"
              >
                {computed.yLabel}
              </text>

              {/* Hover Crosshair */}
              {activePoint && (
                <g>
                  <line
                    x1={activePoint.px}
                    y1={margin.top}
                    x2={activePoint.px}
                    y2={margin.top + plotHeight}
                    stroke="var(--muted)"
                    strokeWidth={1}
                    strokeDasharray="2 2"
                  />
                  <circle
                    cx={activePoint.px}
                    cy={activePoint.pyAnalytic}
                    r={3.5}
                    fill={computed.mode === "timeseries" ? "#3b82f6" : "var(--accent)"}
                  />
                  {activePoint.pyWelch !== undefined && (
                    <circle cx={activePoint.px} cy={activePoint.pyWelch} r={3.5} fill="#3b82f6" />
                  )}
                </g>
              )}
            </>
          )}
        </svg>

        {/* Live Cursor Readout Bar */}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-2 py-1 text-[11px] font-mono text-[var(--muted)] border-t border-[var(--border-subtle)]">
          {activePoint && computed ? (
            computed.mode === "timeseries" ? (
              <span>
                t = {formatFloat(activePoint.xVal, 4)} s &middot; n(t) ={" "}
                {formatScientific(activePoint.yAnalytic, { sigFigs: 3 })}
              </span>
            ) : (
              <span>
                f = {formatFloat(activePoint.xVal, 2)} Hz &middot; Analytical ={" "}
                {formatScientific(activePoint.yAnalytic, { sigFigs: 3 })} &middot; Welch ={" "}
                {formatScientific(activePoint.yWelch ?? 0, { sigFigs: 3 })}
              </span>
            )
          ) : (
            <span>Hover over plot to inspect exact bin values.</span>
          )}
          <span className="font-sans text-[10px] text-[var(--muted-light)]">
            Synthetic realization &middot; Not measured H1 telemetry
          </span>
        </div>
      </div>
    </div>
  );
}

export default NoisePsdPlot;
