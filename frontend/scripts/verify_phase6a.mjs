import { chromium } from "playwright";

const BASE_URL = process.env.GWLAB_FRONTEND_URL || "http://localhost:3000";

const VIEWPORTS = [
  { name: "desktop_1440x900", width: 1440, height: 900 },
  { name: "tablet_768x1024", width: 768, height: 1024 },
  { name: "mobile_390x844", width: 390, height: 844 },
];

const ROUTES = [
  {
    path: "/",
    id: "overview",
    requiredTexts: ["201646.1413", "8.92", "Stage 6D"],
  },
  {
    path: "/waveforms",
    id: "waveforms",
    requiredTexts: ["TaylorF2", "ISCO", "Hz"],
  },
  {
    path: "/noise-and-psd",
    id: "noise_psd",
    requiredTexts: ["Analytical aLIGO-like PSD proxy", "Hz", "Welch"],
  },
  {
    path: "/matched-filtering",
    id: "matched_filtering",
    requiredTexts: ["SNR", "Hz"],
  },
  {
    path: "/inference",
    id: "inference",
    requiredTexts: ["201646.14", "upper 1% of"],
  },
  {
    path: "/experiments",
    id: "experiments",
    requiredTexts: ["base_01_plug_inst_s1", "Stage 6C Run B"],
  },
  {
    path: "/methods",
    id: "methods",
    requiredTexts: ["Whittle"],
  },
];

async function assertNoOverflow(page, label, failures) {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  if (metrics.scrollWidth > metrics.innerWidth + 1) {
    failures.push(
      `Overflow on ${label}: scrollWidth=${metrics.scrollWidth} > innerWidth=${metrics.innerWidth}`
    );
  }
}

async function run() {
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  const consoleErrors = [];
  const pageErrors = [];
  const failedApiRequests = [];
  const failures = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(`${page.url()}: ${msg.text()}`);
    }
  });
  page.on("pageerror", (err) => {
    pageErrors.push(`${page.url()}: ${err.message}`);
  });
  page.on("requestfailed", (req) => {
    if (req.url().includes(":8000/api/")) {
      failedApiRequests.push(`FAILED ${req.method()} ${req.url()} (${req.failure()?.errorText})`);
    }
  });
  page.on("response", (res) => {
    if (res.url().includes(":8000/api/") && res.status() >= 400) {
      failedApiRequests.push(`HTTP ${res.status()} ${res.url()}`);
    }
  });

  // 1. Route x Viewport matrix
  for (const route of ROUTES) {
    for (const vp of VIEWPORTS) {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      const resp = await page.goto(`${BASE_URL}${route.path}`, { waitUntil: "networkidle" });
      if (!resp || resp.status() >= 400) {
        failures.push(`Route ${route.path} (${vp.name}) returned HTTP ${resp?.status()}`);
        continue;
      }
      await page.waitForTimeout(350);

      await assertNoOverflow(page, `${route.path} [${vp.name}]`, failures);

      const bodyText = await page.locator("body").innerText();
      for (const reqText of route.requiredTexts) {
        if (!bodyText.includes(reqText)) {
          failures.push(`Missing expected label "${reqText}" on ${route.path} [${vp.name}]`);
        }
      }
    }
  }

  // 2. Interactive controls check at desktop (1440x900)
  await page.setViewportSize({ width: 1440, height: 900 });

  // Waveforms tabs
  await page.goto(`${BASE_URL}/waveforms`, { waitUntil: "networkidle" });
  for (const tab of ["Fourier Phase", "Chirp Time", "Quadratures"]) {
    await page.getByRole("button", { name: new RegExp(tab, "i") }).click();
    await page.waitForTimeout(200);
  }

  // Noise & PSD model toggle
  await page.goto(`${BASE_URL}/noise-and-psd`, { waitUntil: "networkidle" });
  const flatBtn = page.getByRole("button", { name: /Flat/i });
  if (await flatBtn.count()) {
    await flatBtn.first().click();
    await page.waitForTimeout(500);
  }

  // Matched filtering view/reset check
  await page.goto(`${BASE_URL}/matched-filtering`, { waitUntil: "networkidle" });
  const specBtn = page.getByRole("button", { name: /Spectrum/i });
  if (await specBtn.count()) {
    await specBtn.first().click();
    await page.waitForTimeout(250);
  }

  // Inference parameter selector check
  await page.goto(`${BASE_URL}/inference`, { waitUntil: "networkidle" });
  const selects = page.locator("select");
  if ((await selects.count()) >= 2) {
    await selects.nth(1).selectOption("tc");
    await page.waitForTimeout(500);
  }

  // Experiments campaign filter tabs
  await page.goto(`${BASE_URL}/experiments`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /^Completed/i }).click();
  await page.waitForTimeout(200);
  const completedText = await page.locator("body").innerText();
  if (!completedText.includes("Stage 5G.2")) {
    failures.push("Experiments filter 'Completed' did not retain Stage 5G.2");
  }
  await page.getByRole("button", { name: /Frozen \/ Proposed \/ Unexecuted/i }).click();
  await page.waitForTimeout(200);
  const unexecText = await page.locator("body").innerText();
  if (!unexecText.includes("Stage 6C Run B")) {
    failures.push("Experiments filter 'Frozen / Proposed / Unexecuted' did not show Stage 6C Run B");
  }

  await browser.close();

  console.log("=== GWLab Phase 6A Playwright Regression Summary ===");
  console.log(`Routes tested: ${ROUTES.length} x ${VIEWPORTS.length} viewports = ${ROUTES.length * VIEWPORTS.length} checks`);
  console.log(`Console errors: ${consoleErrors.length}`);
  console.log(`Page errors: ${pageErrors.length}`);
  console.log(`Failed API requests: ${failedApiRequests.length}`);
  console.log(`Assertion failures: ${failures.length}`);

  if (consoleErrors.length) console.error("Console errors:", consoleErrors);
  if (pageErrors.length) console.error("Page errors:", pageErrors);
  if (failedApiRequests.length) console.error("Failed API requests:", failedApiRequests);
  if (failures.length) console.error("Assertion failures:", failures);

  if (
    consoleErrors.length > 0 ||
    pageErrors.length > 0 ||
    failedApiRequests.length > 0 ||
    failures.length > 0
  ) {
    process.exit(1);
  }
}

run().catch((err) => {
  console.error("Phase 6A Playwright runner crashed:", err);
  process.exit(1);
});
