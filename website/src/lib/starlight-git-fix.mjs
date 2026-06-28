/**
 * Vite plugin that fixes Starlight's `lastUpdated` feature when the content
 * collection directory is a symlink.
 *
 * Starlight pre-computes git commit dates at build time by running:
 *   git log --name-status -- <docsPath>
 * where <docsPath> = website/src/content/docs (a symlink to ../../docs).
 * Git tracks files at docs/, not through the symlink, so the scan returns
 * nothing and every page gets undefined for lastUpdated.
 *
 * We intercept the `virtual:starlight/git-info` module before Starlight
 * builds it, resolve the symlink to the real path, run the same git scan
 * against that, and return paths relative to the real docs directory so
 * Starlight's `endsWith` lookup works regardless of which form entry.filePath
 * takes at runtime.
 */

import { spawnSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const symlinkDocsPath = fileURLToPath(
  new URL("../content/docs", import.meta.url),
);

function getRepoRoot(directory) {
  const result = spawnSync("git", ["rev-parse", "--show-toplevel"], {
    cwd: directory,
    encoding: "utf-8",
  });
  if (result.error) return directory;
  try {
    return realpathSync(result.stdout.trim());
  } catch {
    return directory;
  }
}

function buildGitDateMap() {
  let realDocsPath;
  try {
    realDocsPath = realpathSync(symlinkDocsPath);
  } catch {
    return [];
  }

  const repoRoot = getRepoRoot(realDocsPath);
  const gitLog = spawnSync(
    "git",
    ["log", "--format=t:%ct", "--name-status", "--", realDocsPath],
    { cwd: repoRoot, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 },
  );
  if (gitLog.error) return [];

  let runningDate = Date.now();
  const latestDates = new Map();
  for (const line of gitLog.stdout.split("\n")) {
    if (line.startsWith("t:")) {
      runningDate = parseInt(line.slice(2), 10) * 1000;
    }
    const tab = line.lastIndexOf("\t");
    if (tab === -1) continue;
    const file = line.slice(tab + 1);
    latestDates.set(file, Math.max(latestDates.get(file) ?? 0, runningDate));
  }

  return [...latestDates.entries()].map(([file, date]) => {
    const fullPath = resolve(repoRoot, file);
    // Path relative to real docs dir; Starlight uses endsWith() for lookup
    // against entry.filePath, so this short form works for both symlinked
    // and real absolute paths.
    const relPath = relative(realDocsPath, fullPath).replace(/\\/g, "/");
    return [relPath, date];
  });
}

export function starlightGitFix() {
  return {
    name: "nfm-starlight-git-fix",
    enforce: "pre",
    resolveId(id) {
      if (id === "virtual:starlight/git-info")
        return "\0virtual:nfm/git-info";
    },
    load(id) {
      if (id !== "\0virtual:nfm/git-info") return;
      const data = buildGitDateMap();
      return `
const data = ${JSON.stringify(data)};
export const getNewestCommitDate = (file) => {
  const entry = data.find(([path]) => file.endsWith(path));
  if (!entry) throw new Error('No git history for "' + file + '"');
  return new Date(entry[1]);
};
`;
    },
  };
}
