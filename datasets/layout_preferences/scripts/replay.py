"""Replay the geometry corpus across historical revisions, in shards.

Each shard owns a private worktree and walks its slice of the SHA list, one
checkout and one extraction at a time (``revisions.geometry_at``). History is
walked newest-first so that if the engine API drifts out from under the
extractor, the failures land on the oldest and least relevant revisions rather
than silently thinning the recent ones.

A failing revision is recorded as a failure marker in the geometry cache, so a
re-run resumes rather than retrying revisions that cannot be measured.

Usage:
    python replay.py --shard 0 --shards 6 --shas shas_needed.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revisions import (
    CACHE,
    REPO_ROOT,
    GeometryError,
    ensure_worktree,
    geometry_at,
    git,
)

OUT = CACHE


def order_newest_first(shas: list[str]) -> list[str]:
    """Sort by commit time descending; unknown SHAs sink to the end."""
    dated = []
    for sha in shas:
        try:
            dated.append(
                (int(git("show", "-s", "--format=%ct", sha).stdout.strip()), sha)
            )
        except ValueError:
            dated.append((0, sha))
    return [s for _, s in sorted(dated, reverse=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--shas", type=Path, required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    wt = ensure_worktree(REPO_ROOT.parent / f"{REPO_ROOT.name}-geom{args.shard}")

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
        try:
            geometry_at(sha, worktree=wt)
        except GeometryError as exc:
            dest.write_text(json.dumps({"sha": sha, "error": str(exc)[:600]}))
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
