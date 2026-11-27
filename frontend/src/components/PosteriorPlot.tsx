"use client";

import React, { useState } from "react";
import { PosteriorMarginal1D, PosteriorJoint2D } from "@/lib/api";
import { formatFloat } from "@/lib/format";

interface Marginal1DPlotProps {
  marginal: PosteriorMarginal1D;
}

export function Marginal1DPlot({ marginal }: Marginal1DPlotProps) {
  const width = 380;
  const height = 220;
  const margin = { top: 18, right: 16, bottom: 38, left: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const [rMin, rMax] = marginal.support_range;
  const maxD = Math.max(...marginal.density, 1e-12) * 1.12;

  const toX = (v: number) =>
    margin.left + ((v - rMin) / Math.max(1e-12, rMax - rMin)) * plotW;
  const toY = (d: number) => margin.top + plotH - (d / maxD) * plotH;

  const xTicks = [0, 0.25, 0.5, 0.75, 1.0].map((f) => rMin + f * (rMax - rMin));
  const decimals = marginal.key === "tc" || marginal.key === "eta" ? 4 : 2;

  const p5X = toX(Math.max(rMin, Math.min(rMax, marginal.summary.p5)));
  const p95X = toX(Math.max(rMin, Math.min(rMax, marginal.summary.p95)));
  const medX = toX(Math.max(rMin, Math.min(rMax, marginal.summary.median)));
  const injX =
    marginal.injected >= rMin && marginal.injected <= rMax
      ? toX(marginal.injected)
      : null;

  return (
    <div className="rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-2.5 space-y-1.5">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="font-sans font-semibold text-[var(--foreground)]">
          {marginal.label}
        </span>
        <span className="font-mono text-[11px] text-[var(--muted)]">
          [{marginal.unit}]
        </span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto block select-none"
        role="img"
        aria-label={`1D weighted marginal posterior for ${marginal.label}`}
      >
        <rect
          x={margin.left}
          y={margin.top}
          width={plotW}
          height={plotH}
          fill="var(--surface)"
          stroke="var(--border)"
          strokeWidth={1}
        />

        {/* 90% Credible Interval Shading */}
        <rect
          x={Math.min(p5X, p95X)}
          y={margin.top}
          width={Math.max(1, Math.abs(p95X - p5X))}
          height={plotH}
          fill="var(--accent)"
          opacity={0.08}
        />

        {/* Weighted Histogram Bars */}
        {marginal.density.map((d, idx) => {
          const x0 = toX(marginal.bin_edges[idx]);
          const x1 = toX(marginal.bin_edges[idx + 1]);
          const barW = Math.max(1, x1 - x0 - 0.5);
          const yTop = toY(d);
          const barH = Math.max(0, margin.top + plotH - yTop);
          return (
            <rect
              key={idx}
              x={x0}
              y={yTop}
              width={barW}
              height={barH}
              fill="var(--accent)"
              opacity={0.72}
            />
          );
        })}

        {/* Median & Injected Lines */}
        <line
          x1={medX}
          y1={margin.top}
          x2={medX}
          y2={margin.top + plotH}
          stroke="var(--foreground)"
          strokeWidth={1.5}
        />
        {injX !== null && (
          <line
            x1={injX}
            y1={margin.top}
            x2={injX}
            y2={margin.top + plotH}
            stroke="#3b82f6"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
        )}

        {/* X Axis Ticks */}
        {xTicks.map((t, i) => {
          const x = toX(t);
          return (
            <text
              key={i}
              x={x}
              y={margin.top + plotH + 14}
              textAnchor="middle"
              className="fill-[var(--muted)] text-[9px] font-mono"
            >
              {formatFloat(t, decimals)}
            </text>
          );
        })}

        <text
          x={margin.left + plotW / 2}
          y={height - 4}
          textAnchor="middle"
          className="fill-[var(--foreground-secondary)] text-[10px] font-sans"
        >
          {marginal.symbol} [{marginal.unit}]
        </text>
        <text
          x={12}
          y={margin.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${margin.top + plotH / 2})`}
          className="fill-[var(--foreground-secondary)] text-[10px] font-sans"
        >
          Weighted Density
        </text>
      </svg>

      <div className="flex flex-wrap justify-between gap-2 text-[10px] font-mono text-[var(--muted)] px-1">
        <span>
          Med: {formatFloat(marginal.summary.median, decimals)}
        </span>
        <span>
          90% CI: [{formatFloat(marginal.summary.p5, decimals)},{" "}
          {formatFloat(marginal.summary.p95, decimals)}]
        </span>
      </div>
    </div>
  );
}

interface Joint2DPlotProps {
  joint: PosteriorJoint2D;
}

export function Joint2DPlot({ joint }: Joint2DPlotProps) {
  const [hoverCell, setHoverCell] = useState<{
    ix: number;
    iy: number;
    val: number;
  } | null>(null);

  const width = 640;
  const height = 420;
  const margin = { top: 24, right: 24, bottom: 50, left: 74 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const nx = joint.x_centers.length;
  const ny = joint.y_centers.length;
  const xMin = joint.x_edges[0];
  const xMax = joint.x_edges[nx];
  const yMin = joint.y_edges[0];
  const yMax = joint.y_edges[ny];

  const toX = (v: number) =>
    margin.left + ((v - xMin) / Math.max(1e-12, xMax - xMin)) * plotW;
  const toY = (v: number) =>
    margin.top + plotH - ((v - yMin) / Math.max(1e-12, yMax - yMin)) * plotH;

  const xTicks = [0, 0.25, 0.5, 0.75, 1.0].map((f) => xMin + f * (xMax - xMin));
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map((f) => yMin + f * (yMax - yMin));
  const decX = joint.param_x === "tc" || joint.param_x === "eta" ? 4 : 2;
  const decY = joint.param_y === "tc" || joint.param_y === "eta" ? 4 : 2;

  const injX =
    joint.injected_x >= xMin && joint.injected_x <= xMax ? toX(joint.injected_x) : null;
  const injY =
    joint.injected_y >= yMin && joint.injected_y <= yMax ? toY(joint.injected_y) : null;

  return (
    <div className="space-y-2">
      <div className="rounded-sm border border-[var(--border-subtle)] bg-[var(--surface-subtle)] p-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block select-none"
          onMouseLeave={() => setHoverCell(null)}
          role="img"
          aria-label={`2D weighted joint posterior for ${joint.x_label} vs ${joint.y_label}`}
        >
          <rect
            x={margin.left}
            y={margin.top}
            width={plotW}
            height={plotH}
            fill="var(--surface)"
            stroke="var(--border)"
            strokeWidth={1}
          />

          {/* 2D Weighted Density Heatmap Cells */}
          {joint.grid_normalized.map((row, ix) =>
            row.map((val, iy) => {
              if (val <= 0.002) return null;
              const x0 = toX(joint.x_edges[ix]);
              const x1 = toX(joint.x_edges[ix + 1]);
              const y1 = toY(joint.y_edges[iy]);
              const y0 = toY(joint.y_edges[iy + 1]);
              return (
                <rect
                  key={`${ix}-${iy}`}
                  x={x0}
                  y={y0}
                  width={Math.max(1, x1 - x0)}
                  height={Math.max(1, y1 - y0)}
                  fill="var(--accent)"
                  opacity={Math.min(0.95, Math.max(0.06, Math.pow(val, 0.75)))}
                  onMouseEnter={() => setHoverCell({ ix, iy, val })}
                />
              );
            })
          )}

          {/* Injected Truth Crosshair */}
          {injX !== null && (
            <line
              x1={injX}
              y1={margin.top}
              x2={injX}
              y2={margin.top + plotH}
              stroke="#3b82f6"
              strokeWidth={1.2}
              strokeDasharray="4 3"
            />
          )}
          {injY !== null && (
            <line
              x1={margin.left}
              y1={injY}
              x2={margin.left + plotW}
              y2={injY}
              stroke="#3b82f6"
              strokeWidth={1.2}
              strokeDasharray="4 3"
            />
          )}
          {injX !== null && injY !== null && (
            <circle cx={injX} cy={injY} r={4} fill="#3b82f6" stroke="#ffffff" strokeWidth={1.2} />
          )}

          {/* Axis Ticks */}
          {xTicks.map((t, i) => (
            <text
              key={`x-${i}`}
              x={toX(t)}
              y={margin.top + plotH + 16}
              textAnchor="middle"
              className="fill-[var(--muted)] text-[10px] font-mono"
            >
              {formatFloat(t, decX)}
            </text>
          ))}
          {yTicks.map((t, i) => (
            <text
              key={`y-${i}`}
              x={margin.left - 8}
              y={toY(t) + 3}
              textAnchor="end"
              className="fill-[var(--muted)] text-[10px] font-mono"
            >
              {formatFloat(t, decY)}
            </text>
          ))}

          {/* Axis Labels */}
          <text
            x={margin.left + plotW / 2}
            y={height - 10}
            textAnchor="middle"
            className="fill-[var(--foreground-secondary)] text-[11px] font-sans"
          >
            {joint.x_label} [{joint.x_unit}]
          </text>
          <text
            x={16}
            y={margin.top + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90 16 ${margin.top + plotH / 2})`}
            className="fill-[var(--foreground-secondary)] text-[11px] font-sans"
          >
            {joint.y_label} [{joint.y_unit}]
          </text>
        </svg>

        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-2 py-1 text-[11px] font-mono text-[var(--muted)] border-t border-[var(--border-subtle)]">
          {hoverCell ? (
            <span>
              {joint.param_x} = {formatFloat(joint.x_centers[hoverCell.ix], decX)} &middot;{" "}
              {joint.param_y} = {formatFloat(joint.y_centers[hoverCell.iy], decY)} &middot;
              Rel. Density = {formatFloat(hoverCell.val, 3)}
            </span>
          ) : (
            <span>Hover over density cells to inspect coordinates (blue dot = injected value).</span>
          )}
          <span className="font-sans text-[10px] text-[var(--muted-light)]">
            Weighted nested-sampling dead points
          </span>
        </div>
      </div>
    </div>
  );
}
