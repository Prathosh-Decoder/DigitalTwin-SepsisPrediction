const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const dashboardUrl = process.env.DASHBOARD_URL || "http://127.0.0.1:8710";
const outputPath = path.resolve(process.argv[2] || "icu_digital_twin_demo.webm");
const videoDirectory = path.join(path.dirname(outputPath), ".playwright-video");

async function run() {
  fs.mkdirSync(videoDirectory, { recursive: true });
  const browserCandidates = [
    process.env.CHROMIUM_PATH,
    chromium.executablePath(),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ].filter(Boolean);
  const executablePath = browserCandidates.find((candidate) => fs.existsSync(candidate));
  const browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    recordVideo: { dir: videoDirectory, size: { width: 1440, height: 1000 } },
  });
  const page = await context.newPage();
  const video = page.video();

  await page.goto(dashboardUrl, { waitUntil: "networkidle" });
  await page.getByText("Models online").waitFor({ timeout: 60_000 });
  await page.locator(".bed").first().waitFor({ timeout: 60_000 });

  const pauseButton = page.getByRole("button", { name: "Pause" });
  if (await pauseButton.count()) await pauseButton.click();

  await page.locator("#hour").evaluate((input) => {
    input.value = "24";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.getByText("Hour 24", { exact: true }).waitFor({ timeout: 30_000 });
  await page.getByRole("button", { name: /ICU-04/ }).click();
  await page.locator("#detail-content").waitFor({ state: "visible", timeout: 30_000 });
  await page.locator("#observation").waitFor({ state: "visible", timeout: 30_000 });
  await page.evaluate(() => window.scrollTo(0, 0));

  // Establish the stable baseline, then advance through forecast, active alert,
  // watch, and stable states for patient 21 (ICU hours 24-35).
  await page.waitForTimeout(1_500);
  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForTimeout(11_500);
  await page.getByRole("button", { name: "Pause" }).click();
  await page.waitForTimeout(1_500);

  await context.close();
  await browser.close();
  const recordedPath = await video.path();
  fs.copyFileSync(recordedPath, outputPath);
  fs.rmSync(videoDirectory, { recursive: true, force: true });
  process.stdout.write(`${outputPath}\n`);
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
