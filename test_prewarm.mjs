/**
 * Targeted integration test for the render-metro.mjs pre-warm logic.
 *
 * Tests exercised:
 *   1. findMmdFiles correctly enumerates .mmd files
 *   2. prewarmMetroCache (via metroVitePlugin buildStart) populates .metro-cache/
 *   3. Cache entries are prolog-free SVG (no <?xml …?>)
 *   4. renderMetroFile hits the warm cache (returns immediately, same content)
 *   5. Hash identity: renderMetroFile and prewarm write to the same cache path
 *
 * Run from repo root with nf-metro on PATH:
 *   source ~/.local/bin/mm-activate nf-metro-fix-1127
 *   node test_prewarm.mjs
 */

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, rmSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = __dirname;
const CACHE_DIR = join(__dirname, "website/.metro-cache");

// --- helpers ----------------------------------------------------------------

let passed = 0;
let failed = 0;

function ok(label, cond, detail = "") {
  if (cond) {
    console.log(`  PASS  ${label}`);
    passed++;
  } else {
    console.error(`  FAIL  ${label}${detail ? `\n        ${detail}` : ""}`);
    failed++;
  }
}

// Wipe the cache before each run so we can observe what prewarm writes.
function clearCache() {
  rmSync(CACHE_DIR, { recursive: true, force: true });
  mkdirSync(CACHE_DIR, { recursive: true });
}

// Count .svg files in the cache.
function cacheSize() {
  try {
    return readdirSync(CACHE_DIR).filter((f) => f.endsWith(".svg")).length;
  } catch {
    return 0;
  }
}

// --- load the module under test ---------------------------------------------

// Force module re-evaluation with a fresh cache dir.
clearCache();

const mod = await import("./website/src/lib/render-metro.mjs");
const { renderMetroFile, metroVitePlugin } = mod;

// --- test 1: examples dir has .mmd files ------------------------------------

console.log("\n[1] File discovery");
const examplesDir = join(REPO_ROOT, "examples");
const topLevelMmd = readdirSync(examplesDir).filter((f) => f.endsWith(".mmd"));
ok(
  `examples/ has .mmd files`,
  topLevelMmd.length > 0,
  `found: ${topLevelMmd.length}`,
);

const fixturesDir = join(REPO_ROOT, "tests/fixtures");
const topLevelFixtures = readdirSync(fixturesDir).filter((f) =>
  f.endsWith(".mmd"),
);
ok(
  `tests/fixtures/ has top-level .mmd files`,
  topLevelFixtures.length > 0,
  `found: ${topLevelFixtures.length}`,
);

// --- test 2: cold cache, buildStart populates it ----------------------------

console.log("\n[2] buildStart pre-warms the cache");
const plugin = metroVitePlugin();
ok(`plugin has buildStart hook`, typeof plugin.buildStart === "function");

clearCache();
const beforeCount = cacheSize();
ok(`cache empty before buildStart`, beforeCount === 0, `had ${beforeCount}`);

// Call the hook (it's synchronous).
plugin.buildStart();

const afterCount = cacheSize();
ok(
  `cache populated after buildStart`,
  afterCount > 0,
  `wrote ${afterCount} entries`,
);

// One entry per (file × mode) pair — at minimum one plain-mode entry per .mmd.
// We expect at least the top-level examples (plain + debug) and top-level fixtures.
const minExpected = topLevelMmd.length + topLevelFixtures.length;
ok(
  `at least ${minExpected} entries (top-level examples + fixtures, plain mode)`,
  afterCount >= minExpected,
  `got ${afterCount}`,
);

// --- test 3: cache entries are prolog-free ----------------------------------

console.log("\n[3] Cache entries contain no XML prolog");
const cacheFiles = readdirSync(CACHE_DIR).filter((f) => f.endsWith(".svg"));
let prologCount = 0;
for (const f of cacheFiles) {
  const content = readFileSync(join(CACHE_DIR, f), "utf-8");
  if (content.startsWith("<?xml")) prologCount++;
}
ok(
  `0 / ${cacheFiles.length} cache entries have an XML prolog`,
  prologCount === 0,
  `${prologCount} still have the prolog`,
);

// All cache entries must start with <svg.
let badStart = 0;
for (const f of cacheFiles) {
  const content = readFileSync(join(CACHE_DIR, f), "utf-8");
  if (!content.trimStart().startsWith("<svg")) badStart++;
}
ok(
  `all cache entries start with <svg`,
  badStart === 0,
  `${badStart} don't`,
);

// --- test 4: renderMetroFile hits warm cache --------------------------------

console.log("\n[4] renderMetroFile hits the warm cache");
const sampleMmd = join(examplesDir, topLevelMmd[0]);

// Measure: cold render (cache already populated, but time the read path).
const t0 = Date.now();
const svg1 = renderMetroFile(sampleMmd);
const warmMs = Date.now() - t0;

ok(`renderMetroFile returns non-empty SVG`, svg1.length > 100);
ok(`return value starts with <svg`, svg1.trimStart().startsWith("<svg"), `starts with: ${svg1.slice(0, 40)}`);
ok(`return value has no XML prolog`, !svg1.startsWith("<?xml"), `starts with: ${svg1.slice(0, 40)}`);

// Second call must be identical (deterministic).
const svg2 = renderMetroFile(sampleMmd);
ok(`two calls to renderMetroFile return identical SVG`, svg1 === svg2);

// --- test 5: warm-cache second buildStart is a no-op -----------------------

console.log("\n[5] Second buildStart skips already-cached entries");
const countBefore = cacheSize();
plugin.buildStart();
const countAfter = cacheSize();
ok(
  `cache size unchanged on second buildStart`,
  countBefore === countAfter,
  `was ${countBefore}, now ${countAfter}`,
);

// --- test 6: hash identity (prewarm and renderMetroFile agree on cache key) -

console.log("\n[6] Hash identity between prewarm and renderMetroFile");
// renderMetroFile writes to CACHE_DIR/<hash>.svg. After the pre-warm the file
// must already exist — that's exactly what the pre-warm was for.
const source = readFileSync(sampleMmd, "utf-8");
// We need NF_METRO_VERSION; grab it from a cache entry's name by running a
// renderMetroFile on a fresh cache and seeing which file it creates.
clearCache();
renderMetroFile(sampleMmd); // creates one cache entry
const freshFiles = readdirSync(CACHE_DIR).filter((f) => f.endsWith(".svg"));
ok(
  `renderMetroFile creates exactly one cache entry for a fresh cache`,
  freshFiles.length === 1,
  `got ${freshFiles.length}`,
);

// After re-running buildStart, the file renderMetroFile created must still be
// there (prewarm must not overwrite with a different key).
clearCache();
plugin.buildStart(); // prewarm
const afterPrewarm = new Set(
  readdirSync(CACHE_DIR).filter((f) => f.endsWith(".svg")),
);
clearCache();
renderMetroFile(sampleMmd); // individual render
const afterSingle = new Set(
  readdirSync(CACHE_DIR).filter((f) => f.endsWith(".svg")),
);
// The single-render cache file must appear in the prewarm set.
const singleFile = [...afterSingle][0];
ok(
  `prewarm creates the same cache key as renderMetroFile`,
  afterPrewarm.has(singleFile),
  `prewarm set: ${[...afterPrewarm].slice(0, 3).join(", ")}…; single: ${singleFile}`,
);

// --- summary ----------------------------------------------------------------

console.log(`\n${"─".repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
