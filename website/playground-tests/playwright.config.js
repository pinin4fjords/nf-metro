const { defineConfig, devices } = require("@playwright/test");
const path = require("path");

// Pyodide cold-start (WASM download + micropip install) dominates wall-clock,
// so timeouts are generous and tests run serially against one shared page.
module.exports = defineConfig({
  testDir: __dirname,
  fullyParallel: false,
  workers: 1,
  timeout: 200_000,
  expect: { timeout: 160_000 },
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: "http://127.0.0.1:8765",
    acceptDownloads: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "python3 -m http.server 8765",
    // Serve from public/playground/ so wheels/ and examples.json are reachable
    // at their relative paths (app.js resolves them against the page URL).
    // harness.html is a standalone test fixture; the deployed site uses the
    // Astro page at website/src/pages/playground.astro instead.
    cwd: path.join(__dirname, "..", "public", "playground"),
    url: "http://127.0.0.1:8765/harness.html",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
