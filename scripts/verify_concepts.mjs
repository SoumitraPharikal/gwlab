import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUT_DIR = path.resolve(__dirname, "../reports/screenshots/concepts");
fs.mkdirSync(OUT_DIR, { recursive: true });

const BASE_URL = "http://localhost:3005";

const VIEWPORTS = [
  { name: "desktop_1440x900", width: 1440, height: 900 },
  { name: "tablet_1024x768", width: 1024, height: 768 },
  { name: "mobile_390x844", width: 390, height: 844 },
];

const PAGES = [
  { path: "/index.html", id: "portal_index" },
  { path: "/comparison.html", id: "comparison_sheet" },
  { path: "/concept-a/index.html", id: "concept_a_hero" },
  { path: "/concept-a/waveforms.html", id: "concept_a_waveforms" },
  { path: "/concept-b/index.html", id: "concept_b_hero" },
  { path: "/concept-b/waveforms.html", id: "concept_b_waveforms" },
  { path: "/concept-c/index.html", id: "concept_c_hero" },
  { path: "/concept-c/waveforms.html", id: "concept_c_waveforms" },
];

async function main() {
  console.log("=== Launching Playwright Chromium for GWLab Phase 0 Visual Verification ===");
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  const auditReport = {
    timestamp: new Date().toISOString(),
    totalScreenshots: 0,
    errors: [],
    overflows: [],
    interactiveTests: [],
    screenshots: [],
  };

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      console.error(`[Console Error] [${msg.location().url}]: ${msg.text()}`);
      auditReport.errors.push({ url: msg.location().url, message: msg.text() });
    }
  });

  page.on("pageerror", (err) => {
    console.error(`[Page Error] ${err.message}`);
    auditReport.errors.push({ error: err.message });
  });

  for (const p of PAGES) {
    console.log(`\nAuditing Page: ${p.path} (${p.id})`);

    for (const vp of VIEWPORTS) {
      await page.setViewportSize({ width: vp.width, height: vp.height });

      for (const theme of ["light", "dark"]) {
        const targetUrl = `${BASE_URL}${p.path}`;
        await page.goto(targetUrl, { waitUntil: "networkidle" });
        await page.evaluate((th) => {
          document.documentElement.setAttribute("data-theme", th);
        }, theme);
        await page.waitForTimeout(250);

        // Check horizontal overflow
        const overflow = await page.evaluate(() => {
          return {
            scrollWidth: document.documentElement.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            hasOverflow: document.documentElement.scrollWidth > window.innerWidth,
          };
        });

        if (overflow.hasOverflow) {
          console.warn(`[OVERFLOW] ${p.id} at ${vp.name} [${theme}]: scroll=${overflow.scrollWidth} > client=${overflow.clientWidth}`);
          auditReport.overflows.push({
            page: p.id,
            viewport: vp.name,
            theme,
            scrollWidth: overflow.scrollWidth,
            clientWidth: overflow.clientWidth,
          });
        }

        const shotFilename = `${p.id}_${vp.name}_${theme}.png`;
        const shotPath = path.join(OUT_DIR, shotFilename);
        await page.screenshot({ path: shotPath, fullPage: vp.width <= 768 });
        auditReport.screenshots.push(shotFilename);
        auditReport.totalScreenshots++;
      }
    }
  }

  // --- Interactive Slider & Trace Verification on Concept B Waveforms ---
  console.log("\n=== Testing Concept B Waveforms Interactive Slider Trace Update ===");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${BASE_URL}/concept-b/waveforms.html`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));

  // Check initial calculation values
  const initBVal = await page.evaluate(() => {
    return {
      mcText: document.getElementById("val-mc")?.textContent,
      iscoText: document.getElementById("calc-isco")?.textContent,
      iscoLineX: document.getElementById("isco-marker-line")?.getAttribute("x1"),
    };
  });
  console.log("Concept B initial readouts:", initBVal);

  // Trigger slider adjustment to Mc = 32.50
  await page.evaluate(() => {
    const slider = document.getElementById("slider-mc");
    slider.value = "32.50";
    slider.dispatchEvent(new Event("input"));
  });
  await page.waitForTimeout(200);

  const updatedBVal = await page.evaluate(() => {
    return {
      mcText: document.getElementById("val-mc")?.textContent,
      iscoText: document.getElementById("calc-isco")?.textContent,
      iscoLineX: document.getElementById("isco-marker-line")?.getAttribute("x1"),
    };
  });
  console.log("Concept B updated readouts:", updatedBVal);

  const bPassed = updatedBVal.mcText.includes("32.50") && updatedBVal.iscoText !== initBVal.iscoText;
  auditReport.interactiveTests.push({
    test: "Concept B Waveform Slider Dynamic Trace Update",
    passed: bPassed,
    initial: initBVal,
    updated: updatedBVal,
  });

  const bShotPath = path.join(OUT_DIR, "concept_b_waveforms_interactive_mc325_dark.png");
  await page.screenshot({ path: bShotPath });
  auditReport.screenshots.push("concept_b_waveforms_interactive_mc325_dark.png");
  auditReport.totalScreenshots++;

  // --- Interactive Slider & Trace Verification on Concept C Waveforms ---
  console.log("\n=== Testing Concept C Waveforms Interactive Slider Update ===");
  await page.goto(`${BASE_URL}/concept-c/waveforms.html`, { waitUntil: "networkidle" });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));

  const initCVal = await page.evaluate(() => {
    return {
      mcText: document.getElementById("atlas-val-mc")?.textContent,
      riscoText: document.getElementById("atlas-calc-risco")?.textContent,
      fiscoText: document.getElementById("atlas-calc-fisco")?.textContent,
    };
  });
  console.log("Concept C initial readouts:", initCVal);

  await page.evaluate(() => {
    const slider = document.getElementById("atlas-slider-mc");
    slider.value = "30.00";
    slider.dispatchEvent(new Event("input"));
  });
  await page.waitForTimeout(200);

  const updatedCVal = await page.evaluate(() => {
    return {
      mcText: document.getElementById("atlas-val-mc")?.textContent,
      riscoText: document.getElementById("atlas-calc-risco")?.textContent,
      fiscoText: document.getElementById("atlas-calc-fisco")?.textContent,
    };
  });
  console.log("Concept C updated readouts:", updatedCVal);

  const cPassed = updatedCVal.mcText.includes("30.00") && updatedCVal.fiscoText !== initCVal.fiscoText;
  auditReport.interactiveTests.push({
    test: "Concept C Atlas Slider Geometric Metric Update",
    passed: cPassed,
    initial: initCVal,
    updated: updatedCVal,
  });

  const cShotPath = path.join(OUT_DIR, "concept_c_waveforms_interactive_mc300_light.png");
  await page.screenshot({ path: cShotPath });
  auditReport.screenshots.push("concept_c_waveforms_interactive_mc300_light.png");
  auditReport.totalScreenshots++;

  // Save audit report JSON
  fs.writeFileSync(
    path.join(OUT_DIR, "verification_results.json"),
    JSON.stringify(auditReport, null, 2),
    "utf-8"
  );

  console.log("\n=== Phase 0 Visual Verification Complete ===");
  console.log(`Total Screenshots Captured: ${auditReport.totalScreenshots}`);
  console.log(`Total Console / Page Errors: ${auditReport.errors.length}`);
  console.log(`Total Horizontal Overflows: ${auditReport.overflows.length}`);
  console.log(`Interactive Tests: ${auditReport.interactiveTests.filter(t => t.passed).length}/${auditReport.interactiveTests.length} Passed`);

  await browser.close();
}

main().catch((err) => {
  console.error("FATAL in verification runner:", err);
  process.exit(1);
});
