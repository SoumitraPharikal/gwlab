# GWLab Phase 4B.1 — Frontend Runtime Failure Diagnosis

**Date:** 2026-11-22
**Auditor:** OpenCode implementation agent (read-only)
**Status:** Read-only diagnosis; no source changes, no process restarts, no Git operations.

## 1. Exact failing URL / chunk

- Page: `http://127.0.0.1:3000/waveforms`
- Failing resource: `http://127.0.0.1:3000/_next/static/chunks/0k_o-tk0ur_6q.js`
- Response: HTTP 500 (Internal Server Error) on that chunk.
- Console error observed: `Failed to load resource: the server responded with a status of 500 (Internal Server Error)` for the above URL.

## 2. Server stack trace

Not captured — the OpenCode shell tool cannot spawn `pwsh.exe` ("Access is denied, os error 5"), so the running server's terminal output and process details could not be inspected directly. The diagnosis below is inferred from filesystem evidence (see §4).

## 3. Root cause

**Stale running server serving a build whose chunk names no longer exist on disk.**

Evidence:
- The failing chunk `0k_o-tk0ur_6q.js` is **not present** in `frontend/.next/static/chunks/` of the current on-disk build.
- The current on-disk build (BUILD_ID `1HSwW5E9TUOSQI1dq6sbd`) contains a different set of chunk filenames (e.g. `05df5xkqy7gok.js`, `0c1ad7rcpjghx.js`, …).
- Therefore the process listening on port 3000 is an older server instance whose in-memory/manifest references chunks from a previous build that has since been overwritten by the successful `npm run build` (which completed with 10/10 static pages).

In other words: the running server and the on-disk `.next` build are out of sync. The server is not serving the current build.

## 4. Affected routes

- `/waveforms` confirmed failing (chunk 500 → no hydration → empty main content).
- Likely all client-hydrated routes are affected (`/`, `/overview`, etc.) since they share the stale chunk manifest, but only `/waveforms` was observed. Server-rendered shell (navbar/footer) still loads.

## 5. Whether the production build remains valid

**Yes.** The most recent `npm run build` completed successfully:
- Compiled in 4.1s, TypeScript passed, 10/10 static pages generated.
- The on-disk `.next` contains the corrected `WaveformPlot.tsx` (Hz⁻¹ label) source mapping and a consistent chunk set.
- Caveat: that build predates the later Overview/Waveforms text edits and the ESLint `useEffect` refactor, so a fresh rebuild is needed to include those.

## 6. Minimal recommended corrective action

1. Stop the stale frontend server process on port 3000 (identify via OS process manager; do not kill unrelated processes).
2. Rebuild to incorporate all pending source edits:
   `cd frontend && npm run build`
3. Start a fresh production server:
   `cd frontend && npx next start -p 3000`
4. Verify `http://127.0.0.1:3000/waveforms` loads without 500s and that the amplitude tab shows `Hz⁻¹`.
5. Ensure the FastAPI backend is running on 8000 (it is, per prior session: uvicorn PID 38636).

Do not delete `.next` or reinstall dependencies; a rebuild + fresh server start is sufficient.

## 7. Exact commands to reproduce and verify

```powershell
# Reproduce failure
# (open http://127.0.0.1:3000/waveforms in browser; observe 500 on chunk 0k_o-tk0ur_6q.js)

# Verify current build validity
cd frontend
npm run build

# Fresh server start (after stopping stale process)
npx next start -p 3000

# Verify
# http://127.0.0.1:3000/waveforms  -> no 500s, amplitude label Hz^-1
```

## 8. Items that could not be verified

- The running process's stdout/stderr (shell tool blocked).
- Whether the stale server is `next dev` or `next start`.
- Exact process ID / command line of the stale server.

## Conclusion

Root cause identified as **stale server / build mismatch** (server referencing chunks absent from the current `.next`). The current production build is valid. No source-code changes required for this diagnosis. Supervisor review requested before any fix.