"use client";

import React, { useMemo, useState, useRef } from "react";
import { WaveformResponseData, ChirpTimeResponseData } from "@/lib/api";
import { formatFloat, formatScientific } from "@/lib/format";
import { Activity, Clock, Compass, Layers } from "lucide-react";

export type WaveformTab = "amplitude" | "phase" | "chirptime" | "quadratures";

export interface PlotPoint {
  f: number;
  v1: number;
  v2?: number;
  x: number;
  y: number;
}

export interface PlotComputation {
  type: WaveformTab;
  pathStr?: string;
  pathN?: string;
  pathPN?: string;
  pathR?: string;
  pathI?: string;
  toX: (f: number) => number;
  toY: (v: number) => number;
  minF: number;
  maxF: number;
  minA?: number;
  maxA?: number;
  minP?: number;
  maxP?: number;
  maxTau?: number;
  maxAbs?: number;
  fTicks: number[];
  aTicks?: number[];
  pTicks?: number[];
  tTicks?: number[];
  vTicks?: number[];
  points: PlotPoint[];
  fIscoX: number | null;
}

interface WaveformPlotProps {
  data: WaveformResponseData | null;
  chirpData: ChirpTimeResponseData | null;
  activeTab: WaveformTab;
  onTabChange: (tab: WaveformTab) => void;
  loading?: boolean;
  error?: string | null;
  className?: string;
}

export function WaveformPlot({
  data,
  chirpData,
  activeTab,
  onTabChange,
  loading = false,
  error = null,
  className = "",
}: WaveformPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverData, setHoverData] = useState<{
    x: number;
    y: number;
    freq: number;
    val1: number;
    val2?: number;
  } | null>(null);

  // SVG dimensions
  const width = 800;
  const height = 440;
  const margin = { top: 30, right: 35, bottom: 50, left: 75 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Active dataset depending on tab
  const plotComputation: PlotComputation | null = useMemo(() => {
    if (!data && !chirpData) return null;

    if (activeTab === "amplitude" && data && data.frequencies.length > 0) {
      const freqs = data.frequencies;
      const amps = data.amplitude;
      const valid = freqs.map((f, i) => ({ f, a: amps[i] })).filter((pt) => pt.f > 0 && pt.a > 0);
      if (valid.length === 0) return null;

      const minF = Math.min(...valid.map((p) => p.f));
      const maxF = Math.max(...valid.map((p) => p.f));
      const minA = Math.min(...valid.map((p) => p.a));
      const maxA = Math.max(...valid.map((p) => p.a));

      const logMinF = Math.log10(minF);
      const logMaxF = Math.log10(maxF);
      const logMinA = Math.log10(minA);
      const logMaxA = Math.log10(maxA);

      const toX = (f: number) =>
        margin.left + ((Math.log10(f) - logMinF) / (logMaxF - logMinF)) * plotWidth;
      const toY = (a: number) =>
        margin.top + plotHeight - ((Math.log10(a) - logMinA) / (logMaxA - logMinA)) * plotHeight;

      const pathStr = valid.reduce((acc, pt, idx) => {
        const x = toX(pt.f);
        const y = toY(pt.a);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      // Generate grid ticks strictly within [minA, maxA]
      const fTicks = [20, 50, 100, 200, 500].filter((f) => f >= minF && f <= maxF);
      const aTicks = [];
      for (let exp = Math.ceil(logMinA); exp <= Math.floor(logMaxA); exp++) {
        aTicks.push(Math.pow(10, exp));
      }

      return {
        type: "amplitude",
        pathStr,
        toX,
        toY,
        minF,
        maxF,
        minA,
        maxA,
        fTicks,
        aTicks,
        points: valid.map((p) => ({ f: p.f, v1: p.a, x: toX(p.f), y: toY(p.a) })),
        fIscoX: data.f_isco ? toX(data.f_isco) : null,
      };
    }

    if (activeTab === "phase" && data && data.frequencies.length > 0) {
      const freqs = data.frequencies;
      const uphase = data.unwrapped_phase;
      const valid = freqs.map((f, i) => ({ f, p: uphase[i] })).filter((pt) => pt.f > 0);
      if (valid.length === 0) return null;

      const minF = Math.min(...valid.map((p) => p.f));
      const maxF = Math.max(...valid.map((p) => p.f));
      const minP = Math.min(...valid.map((p) => p.p));
      const maxP = Math.max(...valid.map((p) => p.p));

      const toX = (f: number) => margin.left + ((f - minF) / (maxF - minF)) * plotWidth;
      const toY = (p: number) =>
        margin.top + plotHeight - ((p - minP) / (maxP - minP || 1)) * plotHeight;

      const pathStr = valid.reduce((acc, pt, idx) => {
        const x = toX(pt.f);
        const y = toY(pt.p);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      const fTicks = [50, 100, 200, 300, 400, 500].filter((f) => f >= minF && f <= maxF);
      const pStep = (maxP - minP) / 5;
      const pTicks = Array.from({ length: 6 }, (_, i) => minP + i * pStep);

      return {
        type: "phase",
        pathStr,
        toX,
        toY,
        minF,
        maxF,
        minP,
        maxP,
        fTicks,
        pTicks,
        points: valid.map((p) => ({ f: p.f, v1: p.p, x: toX(p.f), y: toY(p.p) })),
        fIscoX: data.f_isco ? toX(data.f_isco) : null,
      };
    }

    if (activeTab === "chirptime" && chirpData && chirpData.frequencies.length > 0) {
      const freqs = chirpData.frequencies;
      const tauN = chirpData.tau_newtonian;
      const tauPN = chirpData.tau_35pn;

      const minF = Math.min(...freqs);
      const maxF = Math.max(...freqs);
      const maxTau = Math.max(
        ...tauN.filter((v) => isFinite(v)),
        ...tauPN.filter((v) => isFinite(v))
      );
      const minTau = 0.0;

      const toX = (f: number) => margin.left + ((f - minF) / (maxF - minF)) * plotWidth;
      const toY = (t: number) =>
        margin.top + plotHeight - (Math.max(0, t) / (maxTau || 1)) * plotHeight;

      const pathN = freqs.reduce((acc, f, idx) => {
        const x = toX(f);
        const y = toY(tauN[idx]);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      const pathPN = freqs.reduce((acc, f, idx) => {
        const x = toX(f);
        const y = toY(tauPN[idx]);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      const fTicks = [50, 100, 200, 300, 400, 500].filter((f) => f >= minF && f <= maxF);
      const tStep = maxTau / 5;
      const tTicks = Array.from({ length: 6 }, (_, i) => i * tStep);

      return {
        type: "chirptime",
        pathN,
        pathPN,
        toX,
        toY,
        minF,
        maxF,
        maxTau,
        fTicks,
        tTicks,
        points: freqs.map((f, i) => ({
          f,
          v1: tauPN[i],
          v2: tauN[i],
          x: toX(f),
          y: toY(tauPN[i]),
        })),
        fIscoX: data?.f_isco ? toX(data.f_isco) : null,
      };
    }

    if (activeTab === "quadratures" && data && data.frequencies.length > 0) {
      const freqs = data.frequencies;
      const real = data.real;
      const imag = data.imag;

      const minF = Math.min(...freqs);
      const maxF = Math.max(...freqs);
      const maxAbs = Math.max(
        ...real.map((v) => Math.abs(v)),
        ...imag.map((v) => Math.abs(v))
      );

      const toX = (f: number) => margin.left + ((f - minF) / (maxF - minF)) * plotWidth;
      const toY = (v: number) =>
        margin.top + plotHeight / 2 - (v / (maxAbs || 1e-22)) * (plotHeight / 2);

      const pathR = freqs.reduce((acc, f, idx) => {
        const x = toX(f);
        const y = toY(real[idx]);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      const pathI = freqs.reduce((acc, f, idx) => {
        const x = toX(f);
        const y = toY(imag[idx]);
        return idx === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }, "");

      const fTicks = [50, 100, 200, 300, 400, 500].filter((f) => f >= minF && f <= maxF);
      const vTicks = [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];

      return {
        type: "quadratures",
        pathR,
        pathI,
        toX,
        toY,
        minF,
        maxF,
        maxAbs,
        fTicks,
        vTicks,
        points: freqs.map((f, i) => ({
          f,
          v1: real[i],
          v2: imag[i],
          x: toX(f),
          y: toY(real[i]),
        })),
        fIscoX: data.f_isco ? toX(data.f_isco) : null,
      };
    }

    return null;
  }, [data, chirpData, activeTab, plotWidth, plotHeight, margin.left, margin.top]);

  // Handle mouse move for interactive tooltip
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!plotComputation || !plotComputation.points || plotComputation.points.length === 0) return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = ((e.clientX - rect.left) / rect.width) * width;
    if (mouseX < margin.left || mouseX > width - margin.right) {
      setHoverData(null);
      return;
    }

    // Find closest point
    let closest = plotComputation.points[0];
    let minDist = Math.abs(closest.x - mouseX);

    for (let i = 1; i < plotComputation.points.length; i++) {
      const pt = plotComputation.points[i];
      const dist = Math.abs(pt.x - mouseX);
      if (dist < minDist) {
        minDist = dist;
        closest = pt;
      }
    }

    setHoverData({
      x: closest.x,
      y: closest.y,
      freq: closest.f,
      val1: closest.v1,
      val2: closest.v2,
    });
  };

  const handleMouseLeave = () => {
    setHoverData(null);
  };

  return (
    <div className={`rounded-sm border border-[var(--border)] bg-[var(--surface)] shadow-xs overflow-hidden ${className}`}>
      {/* Plot Tabs Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-[var(--border-subtle)] bg-[var(--surface-subtle)] px-4 py-2 gap-2">
        <div className="flex flex-wrap items-center gap-1">
          <button
            onClick={() => onTabChange("amplitude")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-sans rounded-[3px] transition-colors ${
              activeTab === "amplitude"
                ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-xs border border-[var(--border)]"
                : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
            }`}
          >
            <Activity size={14} className="text-[var(--accent)]" />
            <span>Strain Amplitude</span>
          </button>

          <button
            onClick={() => onTabChange("phase")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-sans rounded-[3px] transition-colors ${
              activeTab === "phase"
                ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-xs border border-[var(--border)]"
                : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
            }`}
          >
            <Compass size={14} className="text-[var(--accent)]" />
            <span>Fourier Phase</span>
          </button>

          <button
            onClick={() => onTabChange("chirptime")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-sans rounded-[3px] transition-colors ${
              activeTab === "chirptime"
                ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-xs border border-[var(--border)]"
                : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
            }`}
          >
            <Clock size={14} className="text-[var(--accent)]" />
            <span>Chirp Time &tau;(f)</span>
          </button>

          <button
            onClick={() => onTabChange("quadratures")}
            className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-sans rounded-[3px] transition-colors ${
              activeTab === "quadratures"
                ? "bg-[var(--surface)] text-[var(--foreground)] font-medium shadow-xs border border-[var(--border)]"
                : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
            }`}
          >
            <Layers size={14} className="text-[var(--accent)]" />
            <span>Quadratures</span>
          </button>
        </div>

        {/* ISCO cutoff badge */}
        {data && data.f_isco > 0 && (
          <div className="text-[11px] font-mono text-[var(--muted)]">
            f<sub>ISCO</sub> = {data.f_isco.toFixed(1)}&thinsp;Hz
          </div>
        )}
      </div>

      {/* SVG Canvas Area */}
      <div className="relative p-2 bg-[var(--surface)]">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--surface)]/70 backdrop-blur-xs text-xs font-mono text-[var(--muted)]">
            Evaluating TaylorF2 3.5PN...
          </div>
        )}

        {error && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--surface)]/90 text-xs text-[var(--status-error)] p-4 text-center">
            {error}
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto select-none overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Plot Background Area */}
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            fill="var(--surface-subtle)"
            stroke="var(--border)"
            strokeWidth="1"
          />

          {plotComputation && (
            <>
              {/* Frequency Grid Lines */}
              {plotComputation.fTicks.map((tick) => {
                const x = plotComputation.toX(tick);
                return (
                  <g key={`f-${tick}`}>
                    <line
                      x1={x}
                      y1={margin.top}
                      x2={x}
                      y2={margin.top + plotHeight}
                      stroke="var(--border-subtle)"
                      strokeWidth="1"
                      strokeDasharray="2 3"
                    />
                    <text
                      x={x}
                      y={margin.top + plotHeight + 16}
                      textAnchor="middle"
                      className="text-[11px] font-mono fill-[var(--muted)]"
                    >
                      {tick}
                    </text>
                  </g>
                );
              })}

              {/* ISCO Cutoff Line */}
              {plotComputation.fIscoX &&
                plotComputation.fIscoX >= margin.left &&
                plotComputation.fIscoX <= margin.left + plotWidth && (
                  <g>
                    <line
                      x1={plotComputation.fIscoX}
                      y1={margin.top}
                      x2={plotComputation.fIscoX}
                      y2={margin.top + plotHeight}
                      stroke="var(--muted)"
                      strokeWidth="1"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={plotComputation.fIscoX}
                      y={margin.top - 6}
                      textAnchor="middle"
                      className="text-[10px] font-mono fill-[var(--muted)]"
                    >
                      f_ISCO
                    </text>
                  </g>
                )}

              {/* Amplitude Tab Specific Elements */}
              {plotComputation.type === "amplitude" && (
                <>
                  {plotComputation.aTicks?.map((tick) => {
                    const y = plotComputation.toY(tick);
                    return (
                      <g key={`a-${tick}`}>
                        <line
                          x1={margin.left}
                          y1={y}
                          x2={margin.left + plotWidth}
                          y2={y}
                          stroke="var(--border-subtle)"
                          strokeWidth="1"
                          strokeDasharray="2 3"
                        />
                        <text
                          x={margin.left - 8}
                          y={y + 4}
                          textAnchor="end"
                          className="text-[10px] font-mono fill-[var(--muted)]"
                        >
                          10^{Math.round(Math.log10(tick))}
                        </text>
                      </g>
                    );
                  })}
                  {/* Strain Amplitude Curve */}
                  <path
                    d={plotComputation.pathStr}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </>
              )}

              {/* Phase Tab Specific Elements */}
              {plotComputation.type === "phase" && (
                <>
                  {plotComputation.pTicks?.map((tick, idx) => {
                    const y = plotComputation.toY(tick);
                    return (
                      <g key={`p-${idx}`}>
                        <line
                          x1={margin.left}
                          y1={y}
                          x2={margin.left + plotWidth}
                          y2={y}
                          stroke="var(--border-subtle)"
                          strokeWidth="1"
                          strokeDasharray="2 3"
                        />
                        <text
                          x={margin.left - 8}
                          y={y + 4}
                          textAnchor="end"
                          className="text-[10px] font-mono fill-[var(--muted)]"
                        >
                          {formatFloat(tick, 1)}
                        </text>
                      </g>
                    );
                  })}
                  {/* Phase Curve */}
                  <path
                    d={plotComputation.pathStr}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </>
              )}

              {/* Chirp Time Tab Specific Elements */}
              {plotComputation.type === "chirptime" && (
                <>
                  {plotComputation.tTicks?.map((tick, idx) => {
                    const y = plotComputation.toY(tick);
                    return (
                      <g key={`t-${idx}`}>
                        <line
                          x1={margin.left}
                          y1={y}
                          x2={margin.left + plotWidth}
                          y2={y}
                          stroke="var(--border-subtle)"
                          strokeWidth="1"
                          strokeDasharray="2 3"
                        />
                        <text
                          x={margin.left - 8}
                          y={y + 4}
                          textAnchor="end"
                          className="text-[10px] font-mono fill-[var(--muted)]"
                        >
                          {formatFloat(tick, 2)}
                        </text>
                      </g>
                    );
                  })}
                  {/* Newtonian Curve (Dashed) */}
                  <path
                    d={plotComputation.pathN}
                    fill="none"
                    stroke="var(--foreground-secondary)"
                    strokeWidth="1.25"
                    strokeDasharray="4 3"
                  />
                  {/* 3.5PN Curve (Solid) */}
                  <path
                    d={plotComputation.pathPN}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                  />
                </>
              )}

              {/* Quadratures Tab Specific Elements */}
              {plotComputation.type === "quadratures" && (
                <>
                  {plotComputation.vTicks?.map((tick, idx) => {
                    const y = plotComputation.toY(tick);
                    return (
                      <g key={`q-${idx}`}>
                        <line
                          x1={margin.left}
                          y1={y}
                          x2={margin.left + plotWidth}
                          y2={y}
                          stroke={tick === 0 ? "var(--border)" : "var(--border-subtle)"}
                          strokeWidth={tick === 0 ? "1.25" : "1"}
                          strokeDasharray={tick === 0 ? "none" : "2 3"}
                        />
                        <text
                          x={margin.left - 8}
                          y={y + 4}
                          textAnchor="end"
                          className="text-[10px] font-mono fill-[var(--muted)]"
                        >
                          {formatScientific(tick, { sigFigs: 1 })}
                        </text>
                      </g>
                    );
                  })}
                  {/* Real Part (Solid) */}
                  <path
                    d={plotComputation.pathR}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.25"
                    opacity="0.9"
                  />
                  {/* Imag Part (Subtle) */}
                  <path
                    d={plotComputation.pathI}
                    fill="none"
                    stroke="var(--foreground-secondary)"
                    strokeWidth="1.25"
                    opacity="0.75"
                  />
                </>
              )}

              {/* Interactive Hover Crosshair & Tooltip */}
              {hoverData && (
                <g>
                  <line
                    x1={hoverData.x}
                    y1={margin.top}
                    x2={hoverData.x}
                    y2={margin.top + plotHeight}
                    stroke="var(--foreground)"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                    opacity="0.6"
                  />
                  <circle
                    cx={hoverData.x}
                    cy={hoverData.y}
                    r="4"
                    fill="var(--accent)"
                    stroke="var(--surface)"
                    strokeWidth="2"
                  />
                  {/* Readout Pill */}
                  <rect
                    x={Math.min(hoverData.x + 8, width - 180)}
                    y={Math.max(hoverData.y - 36, margin.top + 4)}
                    width="170"
                    height="32"
                    rx="3"
                    fill="var(--surface)"
                    stroke="var(--border)"
                    strokeWidth="1"
                    className="shadow-sm"
                  />
                  <text
                    x={Math.min(hoverData.x + 16, width - 172)}
                    y={Math.max(hoverData.y - 22, margin.top + 18)}
                    className="text-[11px] font-mono fill-[var(--foreground)]"
                  >
                    f = {formatFloat(hoverData.freq, 1)}&thinsp;Hz
                  </text>
                  <text
                    x={Math.min(hoverData.x + 16, width - 172)}
                    y={Math.max(hoverData.y - 10, margin.top + 30)}
                    className="text-[10px] font-mono fill-[var(--accent)] font-semibold"
                  >
                    {activeTab === "amplitude" && `|h~| = ${formatScientific(hoverData.val1)}`}
                    {activeTab === "phase" && `\u03A8 = ${formatFloat(hoverData.val1, 2)} rad`}
                    {activeTab === "chirptime" && `\u03C4 = ${formatFloat(hoverData.val1, 3)} s (3.5PN)`}
                    {activeTab === "quadratures" && `Re = ${formatScientific(hoverData.val1)}`}
                  </text>
                </g>
              )}
            </>
          )}

          {/* Axes Titles */}
          <text
            x={margin.left + plotWidth / 2}
            y={height - 12}
            textAnchor="middle"
            className="text-[12px] font-sans fill-[var(--foreground-secondary)]"
          >
            Frequency f (Hz)
          </text>
          <text
            x={-(margin.top + plotHeight / 2)}
            y={20}
            transform="rotate(-90)"
            textAnchor="middle"
            className="text-[12px] font-sans fill-[var(--foreground-secondary)]"
          >
            {activeTab === "amplitude" && "Strain Amplitude |h~(f)| (Hz\u207B\u00B9)"}
            {activeTab === "phase" && "Fourier Phase \u03A8(f) (rad)"}
            {activeTab === "chirptime" && "Time to Coalescence \u03C4(f) (s)"}
            {activeTab === "quadratures" && "Quadratures Re[h~], Im[h~]"}
          </text>
        </svg>
      </div>

      {/* Plot Caption / Legend Footer */}
      <div className="flex flex-wrap items-center justify-between border-t border-[var(--border-subtle)] bg-[var(--surface-subtle)] px-4 py-2 text-xs text-[var(--muted)] font-sans">
        <div>
          {activeTab === "amplitude" && (
            <span>
              Log-log Fourier-domain strain amplitude with characteristic <code className="font-mono text-[11px]">f &minus;7/6</code> inspiral power-law scaling.
              TaylorF2 models the inspiral only, up to the adopted Schwarzschild ISCO cutoff; it does not model merger or ringdown.
            </span>
          )}
          {activeTab === "phase" && (
            <span>
              Unwrapped 3.5PN post-Newtonian Fourier phase trajectory <code className="font-mono text-[11px]">&Psi;(f)</code>.
            </span>
          )}
          {activeTab === "chirptime" && (
            <span className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[var(--accent)] inline-block"></span>
                <span>3.5PN &tau;(f)</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[var(--foreground-secondary)] inline-block border-t border-dashed"></span>
                <span>Newtonian &tau;<sub>0</sub>(f)</span>
              </span>
            </span>
          )}
          {activeTab === "quadratures" && (
            <span className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[var(--accent)] inline-block"></span>
                <span>Re[h~]</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[var(--foreground-secondary)] inline-block"></span>
                <span>Im[h~]</span>
              </span>
            </span>
          )}
        </div>
        <div className="text-[11px] font-mono text-[var(--muted-light)]">
          Pure deterministic evaluation &middot; Zero sampler overhead
        </div>
      </div>
    </div>
  );
}

export default WaveformPlot;
