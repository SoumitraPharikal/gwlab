"use client";

import React, { useMemo, useState, useRef } from "react";
import { MatchedFilterResponseData } from "@/lib/api";
import { formatFloat, formatScientific } from "@/lib/format";
import { Activity, ZoomIn, BarChart3 } from "lucide-react";

export type MatchedFilterTab = "snr_full" | "snr_zoom" | "spectrum";

interface MatchedFilterPlotProps {
  data: MatchedFilterResponseData | null;
  activeTab: MatchedFilterTab;
  onTabChange: (tab: MatchedFilterTab) => void;
  loading?: boolean;
  error?: string | null;
  className?: string;
}

interface PlotPoint {
  xVal: number;
  y1: number;
  y2?: number;
  px: number;
  py1: number;
  py2?: number;
}

export function MatchedFilterPlot({
  data,
  activeTab,
  onTabChange,
  loading = false,
  error = null,
  className = "",
}: MatchedFilterPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 800;
  const height = 430;
  const margin = { top: 28, right: 30, bottom: 52, left: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const computed = useMemo(() => {
    if (!data) return null;

    if (activeTab === "snr_full" || activeTab === "snr_zoom") {
      const times = activeTab === "snr_full" ? data.snr_time_s : data.zoom_time_s;
      const absSnr = activeTab === "snr_full" ? data.snr_abs : data.zoom_abs;
      const realSnr = activeTab === "snr_full" ? data.snr_real : data.zoom_real;
      if (times.length === 0) return null;

      const minT = times[0];
      const maxT = times[times.length - 1];
      const maxVal = Math.max(...absSnr, 1.0) * 1.12;
      const minVal = activeTab === "snr_zoom" ? -maxVal : 0.0;

      const toX = (t: number) =>
        margin.left + ((t - minT) / Math.max(1e-9, maxT - minT)) * plotWidth;
      const toY = (v: number) =>
        margin.top + plotHeight - ((v - minVal) / Math.max(1e-9, maxVal - minVal)) * plotHeight;

      const pathAbs = times
        .map((t, i) => `${i === 0 ? "M" : "L"} ${toX(t).toFixed(1)} ${toY(absSnr[i]).toFixed(1)}`)
        .join(" ");
      const pathReal = times
        .map((t, i) => `${i === 0 ? "M" : "L"} ${toX(t).toFixed(1)} ${toY(realSnr[i]).toFixed(1)}`)
        .join(" ");

      const nXTicks = 5;
      const xTicks = Array.from({ length: nXTicks }, (_, i) => minT + (i / (nXTicks - 1)) * (maxT - minT));
      const yStep = maxVal > 30 ? 10 : maxVal > 12 ? 5 : 2;
      const yTicks: number[] = [];
      for (let v = Math.ceil(minVal / yStep) * yStep; v <= maxVal; v += yStep) {
        yTicks.push(v);
      }

      const injX =
        data.metadata.injected_tc_s >= minT && data.metadata.injected_tc_s <= maxT
          ? toX(data.metadata.injected_tc_s)
          : null;
      const peakX =
        data.metadata.recovered_peak_time_s >= minT && data.metadata.recovered_peak_time_s <= maxT
          ? toX(data.metadata.recovered_peak_time_s)
          : null;

      const points: PlotPoint[] = times.map((t, i) => ({
        xVal: t,
        y1: absSnr[i],
        y2: realSnr[i],
        px: toX(t),
        py1: toY(absSnr[i]),
        py2: toY(realSnr[i]),
      }));

      return {
        mode: activeTab,
        toX,
        toY,
        xTicks,
        yTicks,
        pathPrimary: pathAbs,
        pathSecondary: activeTab === "snr_zoom" ? pathReal : undefined,
        injX,
        peakX,
        points,
        xLabel:
          activeTab === "snr_full"
            ? "Time Lag t [s] (full 4.0 s segment)"
            : "Time Lag t [s] (zoomed around peak)",
        yLabel: "Matched-Filter SNR [dimensionless]",
      };
    }

    // Frequency-domain data & template spectrum
    const freqs = data.frequencies;
    const dAmp = data.data_amp;
    const tAmp = data.template_amp;
    const valid = freqs
      .map((f, i) => ({ f, d: dAmp[i], t: tAmp[i] }))
      .filter((p) => p.f > 0 && p.d > 0 && p.t > 0);
    if (valid.length === 0) return null;

    const minF = Math.min(...valid.map((p) => p.f));
    const maxF = Math.max(...valid.map((p) => p.f));
    const allY = valid.flatMap((p) => [p.d, p.t]);
    const minY = Math.min(...allY) * 0.8;
    const maxY = Math.max(...allY) * 1.25;

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

    const pathData = valid
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.f).toFixed(1)} ${toY(p.d).toFixed(1)}`)
      .join(" ");
    const pathTemplate = valid
      .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.f).toFixed(1)} ${toY(p.t).toFixed(1)}`)
      .join(" ");

    const xTicks = [10, 20, 50, 100, 200, 500, 1000].filter((f) => f >= minF && f <= maxF);
    const yTicks: number[] = [];
    for (let exp = Math.ceil(logMinY); exp <= Math.floor(logMaxY); exp++) {
      yTicks.push(Math.pow(10, exp));
    }

    const points: PlotPoint[] = valid.map((p) => ({
      xVal: p.f,
      y1: p.t,
      y2: p.d,
      px: toX(p.f),
      py1: toY(p.t),
      py2: toY(p.d),
    }));

    return {
      mode: "spectrum" as const,
      toX,
      toY,
      xTicks,
      yTicks,
      pathPrimary: pathTemplate,
      pathSecondary: pathData,
      injX: null,
      peakX: null,
      points,
      xLabel: "Frequency f [Hz] (log scale)",
      yLabel: "Fourier Strain |h~(f)|, |d~(f)| [strain/Hz]",
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

  const tabs: { id: MatchedFilterTab; label: string; icon: React.ReactNode }[] = [
    { id: "snr_full", label: "SNR Time Series |ρ(t)|", icon: <Activity size={13} /> },
    { id: "snr_zoom", label: "Zoomed Merger Peak", icon: <ZoomIn size={13} /> },
    { id: "spectrum", label: "Data & Template |d~(f)|", icon: <BarChart3 size={13} /> },
  ];

  return (
    <div className={`space-y-3 ${className}`}>
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

        {computed && (
          <div className="flex flex-wrap items-center gap-3 text-[11px] font-sans text-[var(--foreground-secondary)]">
            {computed.mode === "spectrum" ? (
              <>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3.5 h-0.5 bg-[var(--accent)]" />
                  Template |h~(f)|
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3.5 h-0.5 bg-[#3b82f6]" />
                  Synthetic Data |d~(f)|
                </span>
              </>
            ) : (
              <>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3.5 h-0.5 bg-[var(--accent)]" />
                  |ρ(t)| Envelope
                </span>
                {computed.mode === "snr_zoom" && (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block w-3.5 h-0.5 bg-[#3b82f6]" />
                    Re[ρ(t)] (0-phase)
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-sm border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-400 font-sans">
          {error}
        </div>
      )}

      <div className="relative rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-2">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--surface)]/60 backdrop-blur-[1px] text-xs font-mono text-[var(--muted)]">
            Evaluating matched filter...
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block select-none"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIdx(null)}
          role="img"
          aria-label="Matched filtering SNR and spectral comparison plot"
        >
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
              {computed.xTicks.map((tick, idx) => {
                const x = computed.toX(tick);
                return (
                  <g key={`x-${idx}`}>
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
                      {computed.mode === "spectrum"
                        ? tick
                        : formatFloat(tick, computed.mode === "snr_zoom" ? 2 : 1)}
                    </text>
                  </g>
                );
              })}

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
                      {computed.mode === "spectrum"
                        ? formatScientific(tick, { sigFigs: 1 })
                        : formatFloat(tick, 0)}
                    </text>
                  </g>
                );
              })}

              {computed.injX !== null && (
                <line
                  x1={computed.injX}
                  y1={margin.top}
                  x2={computed.injX}
                  y2={margin.top + plotHeight}
                  stroke="var(--muted)"
                  strokeWidth={1}
                  strokeDasharray="4 3"
                />
              )}

              {computed.pathSecondary && (
                <path
                  d={computed.pathSecondary}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={1.25}
                  opacity={0.75}
                />
              )}

              <path
                d={computed.pathPrimary}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={2}
              />

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
                  <circle cx={activePoint.px} cy={activePoint.py1} r={3.5} fill="var(--accent)" />
                  {activePoint.py2 !== undefined && (
                    <circle cx={activePoint.px} cy={activePoint.py2} r={3.5} fill="#3b82f6" />
                  )}
                </g>
              )}
            </>
          )}
        </svg>

        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-2 py-1 text-[11px] font-mono text-[var(--muted)] border-t border-[var(--border-subtle)]">
          {activePoint && computed ? (
            computed.mode === "spectrum" ? (
              <span>
                f = {formatFloat(activePoint.xVal, 2)} Hz &middot; |h~(f)| ={" "}
                {formatScientific(activePoint.y1, { sigFigs: 3 })} &middot; |d~(f)| ={" "}
                {formatScientific(activePoint.y2 ?? 0, { sigFigs: 3 })}
              </span>
            ) : (
              <span>
                t = {formatFloat(activePoint.xVal, 4)} s &middot; |ρ(t)| ={" "}
                {formatFloat(activePoint.y1, 2)}
                {computed.mode === "snr_zoom"
                  ? ` · Re[ρ(t)] = ${formatFloat(activePoint.y2 ?? 0, 2)}`
                  : ""}
              </span>
            )
          ) : (
            <span>Hover over plot to inspect exact matched-filter values.</span>
          )}
          <span className="font-sans text-[10px] text-[var(--muted-light)]">
            Synthetic injection + synthetic noise
          </span>
        </div>
      </div>
    </div>
  );
}

export default MatchedFilterPlot;
