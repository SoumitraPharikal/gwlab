import { chromium } from "../frontend/node_modules/playwright/index.mjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUT_DIR = path.resolve(__dirname, "../reports/screenshots/phase4b2");
fs.mkdirSync(OUT_DIR, { recursive: true });

const TARGET_URL = "http://localhost:3000/noise-and-psd";

const VIEWPORTS = [
  { name: "desktop_1440x900", width: 1440, height: 900 },
  { name: "mobile_390x844", width: 390, height: 844 },
];

async function runAudit() {
  console.log("=== Starting Playwright Audit for /noise-and-psd ===");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const auditResults = {
    timestamp: new Date().toISOString(),
    url: TARGET_URL,
    consoleMessages: [],
    pageErrors: [],
    failedRequests: [],
    viewports: {},
  };

  page.on("console", (msg) => {
    const entry = { type: msg.type(), text: msg.text() };
    auditResults.consoleMessages.push(entry);
    if (msg.type() === "error") {
      console.error(`[Console Error]: ${msg.text()}`);
    }
  });

  page.on("pageerror", (err) => {
    console.error(`[Page Error]: ${err.message}`);
    auditResults.pageErrors.push(err.message);
  });

  page.on("requestfailed", (req) => {
    console.warn(`[Failed Request]: ${req.url()} - ${req.failure()?.errorText}`);
    auditResults.failedRequests.push({ url: req.url(), error: req.failure()?.errorText });
  });

  for (const vp of VIEWPORTS) {
    console.log(`\nAuditing Viewport: ${vp.name} (${vp.width}x${vp.height})`);
    await page.setViewportSize({ width: vp.width, height: vp.height });
    auditResults.viewports[vp.name] = {};

    for (const theme of ["light", "dark"]) {
      console.log(`  Testing Theme: ${theme}`);
      await page.goto(TARGET_URL, { waitUntil: "networkidle" });
      await page.evaluate((th) => {
        document.documentElement.setAttribute("data-theme", th);
        if (th === "dark") {
          document.documentElement.classList.add("dark");
        } else {
          document.documentElement.classList.remove("dark");
        }
      }, theme);
      await page.waitForTimeout(300);

      // Evaluate layout metrics & elements
      const metrics = await page.evaluate(() => {
        const docEl = document.documentElement;
        const body = document.body;
        const h1 = document.querySelector("h1");
        const badge = document.querySelector("header span, [class*='badge'], h1 + div, header div");
        const cards = Array.from(document.querySelectorAll("section, article, div[class*='border']")).map(el => ({
          tagName: el.tagName,
          className: el.className,
          textSnippet: el.innerText.slice(0, 80),
        }));
        const links = Array.from(document.querySelectorAll("a")).map(a => ({
          href: a.getAttribute("href"),
          text: a.innerText.trim(),
        }));
        const controls = Array.from(document.querySelectorAll("input, button, select, [role='slider']")).map(c => ({
          tag: c.tagName,
          type: c.getAttribute("type") || "button",
          ariaLabel: c.getAttribute("aria-label"),
          text: c.innerText?.trim() || "",
        }));

        return {
          scrollWidth: docEl.scrollWidth,
          clientWidth: docEl.clientWidth,
          hasHorizontalOverflow: docEl.scrollWidth > window.innerWidth,
          h1Title: h1?.innerText || null,
          linksCount: links.length,
          links,
          controlsCount: controls.length,
          controls,
          bodyTextSnippet: body.innerText.slice(0, 500),
        };
      });

      auditResults.viewports[vp.name][theme] = metrics;

      const shotPath = path.join(OUT_DIR, `noise_psd_${vp.name}_${theme}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  Saved screenshot: ${shotPath}`);
      console.log(`  Overflow check: ${metrics.hasHorizontalOverflow ? "FAIL (Overflow detected!)" : "PASS (No overflow)"}`);
    }
  }

  // Interactive Test: Theme toggle button in Navbar
  console.log("\nTesting Theme Toggle Interaction in Navbar...");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(TARGET_URL, { waitUntil: "networkidle" });

  const themeToggle = await page.$("button[aria-label*='theme' i], button[aria-label*='mode' i], nav button:has(svg)");
  let themeToggleWorks = false;
  if (themeToggle) {
    const initialTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    await themeToggle.click();
    await page.waitForTimeout(200);
    const newTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    themeToggleWorks = initialTheme !== newTheme;
    console.log(`  Theme toggle clicked: ${initialTheme} -> ${newTheme} (Result: ${themeToggleWorks})`);
  } else {
    console.log("  No specific theme toggle button identified by selector.");
  }
  auditResults.themeToggleWorks = themeToggleWorks;

  // Interaction Test: Navigate link to /waveforms
  const waveformsLink = await page.$("a[href='/waveforms']");
  let linkNavigationWorks = false;
  if (waveformsLink) {
    console.log("  Found link to /waveforms. Testing navigation...");
    await Promise.all([
      page.waitForNavigation({ waitUntil: "networkidle" }),
      waveformsLink.click(),
    ]);
    linkNavigationWorks = page.url().includes("/waveforms");
    console.log(`  Navigated to: ${page.url()} (Success: ${linkNavigationWorks})`);
  }
  auditResults.linkNavigationWorks = linkNavigationWorks;

  await browser.close();

  const reportJsonPath = path.join(OUT_DIR, "audit_summary.json");
  fs.writeFileSync(reportJsonPath, JSON.stringify(auditResults, null, 2));
  console.log(`\nAudit summary saved to ${reportJsonPath}`);
}

runAudit().catch(err => {
  console.error("Audit failed:", err);
  process.exit(1);
});
