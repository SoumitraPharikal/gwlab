import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUT_DIR = path.resolve(__dirname, "../../reports/screenshots/phase4a");
fs.mkdirSync(OUT_DIR, { recursive: true });

const VIEWPORTS = [
  { name: "desktop_1440x900", width: 1440, height: 900 },
  { name: "tablet_1024x768", width: 1024, height: 768 },
  { name: "mobile_390x844", width: 390, height: 844 },
];

const ROUTES = [
  { path: "/", id: "overview" },
  { path: "/waveforms", id: "waveforms" },
  { path: "/noise-and-psd", id: "noise_psd" },
  { path: "/matched-filtering", id: "matched_filtering" },
  { path: "/inference", id: "inference" },
  { path: "/experiments", id: "experiments" },
  { path: "/methods", id: "methods" },
];

async function main() {
  console.log("=== Launching Chromium for GWLab Visual Verification ===");
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  const results = {
    errors: [],
    overflows: [],
    screenshots: [],
  };

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      console.error(`[Console Error] ${msg.text()}`);
      results.errors.push(msg.text());
    }
  });

  page.on("pageerror", (err) => {
    console.error(`[Page Error] ${err.message}`);
    results.errors.push(err.message);
  });

  for (const route of ROUTES) {
    console.log(`\nTesting Route: ${route.path} (${route.id})`);

    for (const vp of VIEWPORTS) {
      await page.setViewportSize({ width: vp.width, height: vp.height });

      // Light mode test
      await page.goto(`http://localhost:3000${route.path}`, { waitUntil: "networkidle" });
      await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
      await page.waitForTimeout(400);

      // Check horizontal overflow
      const hasOverflowLight = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });

      if (hasOverflowLight) {
        console.warn(`[OVERFLOW] Horizontal overflow detected on ${route.path} at ${vp.width}px (Light)`);
        results.overflows.push(`${route.id}_${vp.name}_light`);
      }

      const lightShotPath = path.join(OUT_DIR, `${route.id}_${vp.name}_light.png`);
      await page.screenshot({ path: lightShotPath, fullPage: vp.width <= 768 });
      results.screenshots.push(lightShotPath);

      // Dark mode test
      await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
      await page.waitForTimeout(400);

      const hasOverflowDark = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });

      if (hasOverflowDark) {
        console.warn(`[OVERFLOW] Horizontal overflow detected on ${route.path} at ${vp.width}px (Dark)`);
        results.overflows.push(`${route.id}_${vp.name}_dark`);
      }

      const darkShotPath = path.join(OUT_DIR, `${route.id}_${vp.name}_dark.png`);
      await page.screenshot({ path: darkShotPath, fullPage: vp.width <= 768 });
      results.screenshots.push(darkShotPath);
    }
  }

  // Interactive Tab Testing on Waveforms at 1440x900
  console.log("\n=== Testing Waveform Plot Interactive Tabs ===");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("http://localhost:3000/waveforms", { waitUntil: "networkidle" });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));

  // Click Fourier Phase Tab
  await page.click("text=Fourier Phase");
  await page.waitForTimeout(400);
  const phaseShot = path.join(OUT_DIR, "waveforms_tab_phase_1440x900.png");
  await page.screenshot({ path: phaseShot });
  results.screenshots.push(phaseShot);

  // Click Chirp Time Tab
  await page.click("text=Chirp Time");
  await page.waitForTimeout(400);
  const chirpShot = path.join(OUT_DIR, "waveforms_tab_chirptime_1440x900.png");
  await page.screenshot({ path: chirpShot });
  results.screenshots.push(chirpShot);

  // Click Quadratures Tab
  await page.click("text=Quadratures");
  await page.waitForTimeout(400);
  const quadShot = path.join(OUT_DIR, "waveforms_tab_quadratures_1440x900.png");
  await page.screenshot({ path: quadShot });
  results.screenshots.push(quadShot);

  await browser.close();

  console.log("\n=== VISUAL VERIFICATION SUMMARY ===");
  console.log(`Total Screenshots Captured: ${results.screenshots.length}`);
  console.log(`Uncaught Console Errors: ${results.errors.length}`);
  console.log(`Horizontal Overflows: ${results.overflows.length}`);

  if (results.overflows.length > 0) {
    console.warn("Overflows on:", results.overflows);
  }

  return results;
}

main().catch((err) => {
  console.error("Verification failed:", err);
  process.exit(1);
});
