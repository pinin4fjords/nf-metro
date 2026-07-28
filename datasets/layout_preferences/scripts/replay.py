"""Replay the geometry corpus across historical revisions.

Each shard owns a private worktree, checks out one SHA at a time, and runs the
version-independent extractor against that revision's engine AND that
revision's own fixture files. Feature definitions therefore come from today
while both engine and input come from the past, which is what makes a
within-pair comparison meaningful.

History is walked newest-first so that if the engine API drifts out from under
the extractor, the failures land on the oldest and least relevant revisions
rather than silently thinning the recent ones.

Usage:
    python replay.py --shard 0 --shards 6 --shas shas_needed.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

S = Path(__file__).parent
OUT = S / "geometry"
MAIN = Path("/Users/jonathan.manning/projects/nf-metro")


def run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 300
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def order_newest_first(shas: list[str]) -> list[str]:
    """Sort by commit time descending; unknown SHAs sink to the end."""
    dated = []
    for sha in shas:
        r = run(["git", "-C", str(MAIN), "show", "-s", "--format=%ct", sha])
        try:
            dated.append((int(r.stdout.strip()), sha))
        except ValueError:
            dated.append((0, sha))
    return [s for _, s in sorted(dated, reverse=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--shas", type=Path, required=True)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    wt = Path(f"/Users/jonathan.manning/projects/nf-metro-geom{args.shard}")
    if not wt.exists():
        r = run(
            ["git", "-C", str(MAIN), "worktree", "add", str(wt), "--detach", "HEAD"]
        )
        if r.returncode:
            print(
                f"shard {args.shard}: worktree failed: {r.stderr[:300]}",
                file=sys.stderr,
            )
            return

    all_shas = order_newest_first(
        [s.strip() for s in args.shas.read_text().splitlines() if s.strip()]
    )
    mine = [s for i, s in enumerate(all_shas) if i % args.shards == args.shard]
    print(f"shard {args.shard}: {len(mine)} SHAs", flush=True)

    done = skipped = failed = 0
    for i, sha in enumerate(mine, 1):
        dest = OUT / f"{sha}.json"
        if dest.exists():
            skipped += 1
            continue
        co = run(
            ["git", "-C", str(wt), "checkout", "--force", "--detach", sha], timeout=180
        )
        if co.returncode:
            dest.write_text(json.dumps({"sha": sha, "error": "checkout_failed"}))
            failed += 1
            continue
        ex = run(
            [
                sys.executable,
                str(S / "extract_features.py"),
                "--worktree",
                str(wt),
                "--sha",
                sha,
                "--out",
                str(dest),
            ],
            timeout=900,
        )
        if ex.returncode or not dest.exists():
            dest.write_text(
                json.dumps(
                    {"sha": sha, "error": "extract_failed", "stderr": ex.stderr[-600:]}
                )
            )
            failed += 1
        else:
            done += 1
        if i % 10 == 0:
            print(
                f"shard {args.shard}: {i}/{len(mine)} "
                f"(ok {done}, skip {skipped}, fail {failed})",
                flush=True,
            )

    print(
        f"shard {args.shard} COMPLETE: ok {done}, skipped {skipped}, failed {failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
