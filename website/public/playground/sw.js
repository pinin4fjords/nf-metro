"use strict";

// Keep in sync with PYODIDE_VERSION in app.js. Bumping this constant
// changes the cache name, which causes the activate handler to evict all
// assets from the previous version.
const PYODIDE_VERSION = "v0.27.2";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`;
const CACHE_NAME = `nfm-playground-${PYODIDE_VERSION}`;

function shouldCache(url) {
  // Cache Pyodide CDN assets and the local wheel file.
  // wheels/index.json is always fetched fresh by app.js (cache: "no-store")
  // so we leave it out of the SW cache entirely.
  if (url.startsWith(PYODIDE_BASE)) return true;
  try {
    const path = new URL(url).pathname;
    if (/\/wheels\/[^/]+\.whl$/.test(path)) return true;
  } catch (_) {}
  return false;
}

self.addEventListener("install", (event) => {
  // Activate immediately without waiting for existing clients to close.
  self.skipWaiting();
  // Precache the Pyodide entry script so the next navigation serves it from
  // the SW cache without a CDN round-trip. The WASM and other large assets are
  // cached on their first fetch once the SW is controlling the page.
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      try {
        await cache.add(`${PYODIDE_BASE}pyodide.js`);
      } catch (_) {
        // Don't fail install when the CDN is unreachable (offline / first-visit).
      }
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((k) => k.startsWith("nfm-playground-") && k !== CACHE_NAME)
          .map((k) => caches.delete(k)),
      );
      // Take control of all open clients so the cache is active on first
      // load without requiring a page reload.
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const url = event.request.url;
  if (!shouldCache(url)) return;

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(event.request);
      if (cached) return cached;
      const response = await fetch(event.request);
      if (response.ok) {
        cache.put(event.request, response.clone());
      }
      return response;
    })(),
  );
});
