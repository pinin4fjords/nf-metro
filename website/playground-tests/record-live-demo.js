#!/usr/bin/env node
// Records the docs/live.mdx demo clip (website/public/assets/live_demo.mp4).
//
// Lives in this package (rather than scripts/) to reuse its @playwright/test
// devDependency and already-cached Chromium download - it is not itself an
// e2e test.
//
// Usage (from the repo root, with an nf-metro + Nextflow env active - see
// the `live-demo-video` skill):
//   node website/playground-tests/record-live-demo.js <output-dir>
//
// Produces <output-dir>/*.webm. Convert that to the committed mp4 with
// scripts/convert_demo_video.sh (see the skill for why: Playwright only
// records webm).
const { chromium } = require("@playwright/test");
const { spawn, spawnSync } = require("child_process");
const http = require("http");
const path = require("path");

const PORT = 8790;
const SERVE_URL = `http://localhost:${PORT}/`;
const STATE_URL = `http://localhost:${PORT}/state`;
const MAP = "examples/live/pipeline.mmd";
const WORKFLOW = "examples/live/workflow/main.nf";
const WORKFLOW_CONFIG = "examples/live/workflow/nextflow.config";

const OUT_DIR = process.argv[2];
const REPO_ROOT = process.argv[3] || process.cwd();
if (!OUT_DIR) {
  console.error("usage: record-live-demo.js <output-dir> [repo-root]");
  process.exit(1);
}

function fetchState() {
  return new Promise((resolve, reject) => {
    http
      .get(STATE_URL, (res) => {
        let data = "";
        res.on("data", (c) => (data += c));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(e);
          }
        });
      })
      .on("error", reject);
  });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForServer(deadline) {
  while (Date.now() < deadline) {
    try {
      await fetchState();
      return true;
    } catch (e) {
      await sleep(300);
    }
  }
  return false;
}

(async () => {
  const server = spawn("nf-metro", ["serve", MAP, "--port", String(PORT)], {
    cwd: REPO_ROOT,
    stdio: "inherit",
  });

  if (!(await waitForServer(Date.now() + 15_000))) {
    console.error("nf-metro serve did not come up in time");
    server.kill();
    process.exit(1);
  }

  const browser = await chromium.launch();
  const context = await browser.newContext({
    // Matches the previous clip's framing; keep in sync with the <video>
    // element's aspect ratio in docs/live.mdx if either changes.
    viewport: { width: 980, height: 640 },
    // nf-metro serve bakes one concrete light/dark mode into the SVG rather
    // than shipping light-dark() for it, independent of the page chrome
    // (which does follow prefers-color-scheme) - force dark so the two match
    // instead of pairing a dark toolbar with a light map card.
    colorScheme: "dark",
    recordVideo: { dir: OUT_DIR, size: { width: 980, height: 640 } },
  });
  const page = await context.newPage();
  // The page holds an open SSE connection to /stream, so "networkidle" never
  // fires - wait for the map SVG instead.
  await page.goto(SERVE_URL, { waitUntil: "load" });
  await page.waitForSelector("svg");

  // Let the idle map sit on screen for a beat before the run starts.
  await sleep(1500);

  const nextflow = spawn(
    "nextflow",
    [
      "run",
      WORKFLOW,
      "-c",
      WORKFLOW_CONFIG,
      "-with-weblog",
      `http://localhost:${PORT}/events`,
    ],
    { cwd: REPO_ROOT, stdio: "inherit" },
  );

  const deadline = Date.now() + 120_000;
  let finished = false;
  while (Date.now() < deadline) {
    await sleep(1000);
    try {
      const state = await fetchState();
      // The state schema calls this "complete", not "completed".
      if (
        state.run &&
        (state.run.state === "complete" || state.run.state === "error")
      ) {
        finished = true;
        break;
      }
    } catch (e) {
      // server between requests - keep polling
    }
  }
  if (!finished) {
    console.error("Timed out waiting for the run to finish");
  }

  // Hold on the finished state for a couple of seconds before cutting.
  await sleep(2500);

  await context.close();
  await browser.close();
  nextflow.kill();
  server.kill();

  console.log(`Recorded to ${path.resolve(OUT_DIR)}`);
})();
