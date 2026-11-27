# GWLab — Phase 4B.1 Runtime Verification Report

**Date:** 2026-11-25  
**Repository:** `C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference`  
**Frontend:** `frontend/` (Next.js 16.4.6 production build in `.next/`)  
**Status:** **PASS WITH QUALIFICATIONS**

---

## 1. Process & Port Diagnosis and Action Taken

- Inspected TCP port `3000` in `Listen` state using PowerShell (`Get-NetTCPConnection -LocalPort 3000 -State Listen`).
- **Finding:** Port `3000` was free (`PORT_3000_FREE`) at the time of inspection (no stale GWLab or third-party process was listening).
- **Action Taken:** Verified `.next/` existed without rebuilding or modifying any files, and proceeded directly to start the production server on port `3000`.

---

## 2. Server URL and Startup Result

- **Command:** `Set-Location "C:\Users\Soumitra\Desktop\GW-matched-filtering-and-baysian-interference\frontend"; npx next start -p 3000`
- **Server URL:** `http://localhost:3000`
- **Startup Result:** Started cleanly as a persistent background daemon (`task-81`) and served HTTP `200 OK` on `/` and `/waveforms`.
- **Backend API Observed:** Frontend requests to `http://localhost:8000/api/status`, `http://localhost:8000/api/benchmarks/overview`, `http://localhost:8000/api/waveforms/generate`, and `http://localhost:8000/api/waveforms/chirp-time` also resolved with HTTP `200 OK`.

---

## 3. Routes Actually Opened and Inspected

- `http://localhost:3000/` (Desktop `1440x900` and Mobile `390x844`)
- `http://localhost:3000/waveforms` (Desktop `1440x900` and Mobile `390x844`)

---

## 4. Browser Console and Network Errors

- **Network Requests:** `0` HTTP `4xx`/`5xx` errors and `0` missing JavaScript chunks on both `/` and `/waveforms`. All static JS/CSS/font assets, Next.js `_rsc` prefetches, and backend `/api/*` calls resolved with HTTP `200 OK`.
- **Console / Page Errors:**
  - `http://localhost:3000/`: `0` console or page errors.
  - `http://localhost:3000/waveforms`: `0` console or page errors on initial load, tab switching (`Strain Amplitude`, `Fourier Phase`, `Chirp Time τ(f)`, `Quadratures`), and keyboard slider interaction. *(Note: a synthetic `IndexSizeError` occurred only when Playwright's `.fill()` was invoked directly on `<input type="range">`, not during normal user/keyboard interaction.)*

---

## 5. Responsive Viewport Checks

- **Desktop (`1440x900`):**
  - `/`: `html` and `body` width `1424.8px <= 1440px`, max child right edge `847.2px`, `0` overflowing elements.
  - `/waveforms`: `html` and `body` width `1424.8px <= 1440px`, max child right edge `1424.8px`, `0` overflowing elements. Inline `<svg>` waveform chart rendered at `785.08 × 431.79 px`.
- **Mobile (`390x844`):**
  - `/`: `html` and `body` width `375.2px <= 390px`; no page-level horizontal scroll (`pageScrolledHorizontally = false`). Minor internal clipping on card footer `"Source: ..."` lines (`right` up to `527.8px`) and internal scroll on the Provenance Matrix table (`maxRight = 770.7px`).
  - `/waveforms`: `html` and `body` width `375.2px <= 390px`; no page-level horizontal scroll (`pageScrolledHorizontally = false`). Inline `<svg>` waveform chart rendered at `325.60 × 179.07 px`. Minor internal visual clipping observed in screenshot `final_step_31.png`:
    - The chart tab strip (`381.15px`) clips `"Quadratures"` to `"Qua"` at `390px` viewport width.
    - The top y-axis tick (`10^-19`) slightly overlaps `f_ISCO = 67.6 Hz` above the chart area.

---

## 6. Interactive & Scientific Presentation Checks (`/waveforms`)

- **Parameter Controls & Plot Updates:**
  - **Initial state:** `Chirp Mass Mc = 28.0956 M☉`, `Symmetric Mass Ratio η = 0.2471`, `Luminosity Distance dL = 410 Mpc`, `Arrival Shift tc = 0 s`, `Coalescence Phase φc = 0 rad`. Derived mechanics: `m₁ = 36 M☉`, `m₂ = 29 M☉`, `M = 65 M☉`, `q = 1.241`, `f_ISCO = 67.6 Hz`. Chart SVG path length `3776`.
  - **Updated to `Mc = 15 M☉`, `dL = 800 Mpc`:** Derived mechanics updated to `m₁ = 19.22 M☉`, `m₂ = 15.48 M☉`, `M = 34.7 M☉`, `q = 1.241`, `f_ISCO = 126.7 Hz`. Chart SVG path updated to length `7079` with updated x-ticks (`["20", "50", "100"]`) and y-ticks (`["10^-23", "10^-22", "10^-21"]`).
  - **Keyboard slider adjustment (`Mc = 27.6 M☉`, `dL = 460 Mpc`):** Derived mechanics updated to `m₁ = 35.37 M☉`, `m₂ = 28.49 M☉`, `M = 63.85 M☉`, `f_ISCO = 68.9 Hz`.
- **Amplitude Unit:** Verified SVG y-axis label reads `"Strain Amplitude |h~(f)| (Hz⁻¹)"` (`Hz⁻¹`, not `Hz⁻¹/²`) both before and after parameter changes.
- **TaylorF2 Inspiral-Only / Schwarzschild ISCO Description:**
  - Header description: `"Explores stationary-phase approximation gravitational waveforms from binary inspirals up to the Schwarzschild innermost stable circular orbit (ISCO)."`
  - Derived mechanics sublabel: `"Schwarzschild innermost stable circular orbit cutoff"`
  - Plot caption: `"Log-log Fourier-domain strain amplitude with characteristic f −7/6 inspiral power-law scaling. TaylorF2 models the inspiral only, up to the adopted Schwarzschild ISCO cutoff; it does not model merger or ringdown."`
- **Units Consistency:** `M☉`, `Mpc`, `s`, `rad`, `Hz`, and `Hz⁻¹` remained visible and consistent across all parameter updates.

---

## 7. Exact Status

**PASS WITH QUALIFICATIONS**

All runtime hydration, chunk loading, API connectivity, interactive parameter updates, desktop layout, and scientific presentation checks passed with zero source edits. Qualifications are limited to minor mobile-only (`390x844`) visual clipping inside `/waveforms` (the `"Quadratures"` tab label clipping and the `10^-19` top y-axis tick slightly overlapping `f_ISCO = 67.6 Hz`) and card footer source-line clipping on `/`.

---

## 8. Remaining Blocker & Smallest Next Action

- **Remaining Blocker:** None blocking Phase 4B.1 runtime verification.
- **Smallest Next Action:** If addressing the mobile visual qualifications before Phase 4B.2, allow the `/waveforms` chart tab strip to wrap/scroll cleanly at `< 400px` and add top padding/offset to the SVG plot so the top y-axis tick does not overlap `f_ISCO`; otherwise, proceed to Phase 4B.2 when ready.
