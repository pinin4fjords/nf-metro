"""Join geometry vectors onto history labels to produce preference pairs.

``pair_rules`` holds the emission rules; this script supplies the labels mined
from history and the replayed geometry, then reports what the join produced.

Usage:
    python build_dataset.py                # join geometry onto labels
    python build_dataset.py --from-pairs   # directional signal from the committed pairs
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pair_rules import emit_anchors, emit_pairs
from terms import SENTINEL, UNDEFINED

S = Path(__file__).parent
PAIRS = S.parent / "dataset_pairs.jsonl"
ANCHORS = S.parent / "dataset_anchors.jsonl"

MIN_MOVED = 8
"""Moved pairs a feature needs before the report will state a percentage for it."""

FLAG_MARGIN = 0.18
"""Distance from chance at which the grouped reading is called discriminative."""


def load_geometry() -> dict[str, dict]:
    geo: dict[str, dict] = {}
    for f in (S / "geometry").glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "fixtures" in d:
            geo[d["sha"]] = d
    return geo


@dataclass(frozen=True)
class Signal:
    """How often one feature moved down across the pairs that moved it at all."""

    feature: str
    raw: float
    grouped: float
    n_pairs: int
    n_fixtures: int

    @property
    def discriminative(self) -> bool:
        return abs(self.grouped - 0.5) > FLAG_MARGIN


def feature_keys(pairs: list[dict]) -> list[str]:
    for p in pairs:
        if p.get("features_before"):
            return sorted(p["features_before"])
    return []


def moves_by_fixture(pairs: list[dict], key: str) -> dict[str, list[float]]:
    """Per-fixture deltas on ``key``, keeping only the pairs that moved it."""
    out: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        before, after = p.get("features_before"), p.get("features_after")
        if not before or not after or key not in before or key not in after:
            continue
        if key in SENTINEL and UNDEFINED in (before[key], after[key]):
            continue
        delta = after[key] - before[key]
        if abs(delta) > 1e-6:
            out[p["fixture"]].append(delta)
    return out


def directional_signal(pairs: list[dict]) -> list[Signal]:
    """Share of moves that DECREASED each feature, counted two ways.

    ``raw`` weights a fixture by how many pairs it contributes, so one map
    appearing in fifty pairs can state a corpus-wide trend by itself, and the
    extent features are the ones most exposed to it. ``grouped`` gives every
    fixture one vote, and is the figure to triage features on.
    """
    signals = []
    for key in feature_keys(pairs):
        by_fixture = moves_by_fixture(pairs, key)
        deltas = [d for ds in by_fixture.values() for d in ds]
        if len(deltas) < MIN_MOVED:
            continue
        down = [sum(1 for d in ds if d < 0) / len(ds) for ds in by_fixture.values()]
        signals.append(
            Signal(
                feature=key,
                raw=sum(1 for d in deltas if d < 0) / len(deltas),
                grouped=statistics.fmean(down),
                n_pairs=len(deltas),
                n_fixtures=len(by_fixture),
            )
        )
    return signals


def print_directional_signal(pairs: list[dict]) -> None:
    print("\n=== directional signal check (issue-fix pairs) ===")
    print("share of the moves that DECREASED the feature.")
    print("`raw` counts every pair, so a fixture in many pairs speaks many times;")
    print("`grouped` gives each fixture one vote. Triage on grouped.")
    print(f"  {'feature':28s} {'raw':>7} {'grouped':>8} {'pairs':>6} {'fixtures':>9}")
    for s in directional_signal(pairs):
        flag = "  <-- discriminative" if s.discriminative else ""
        print(
            f"  {s.feature:28s} {s.raw * 100:6.1f}% {s.grouped * 100:7.1f}%"
            f" {s.n_pairs:6d} {s.n_fixtures:9d}{flag}"
        )


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def directional_pairs(pairs: list[dict]) -> list[dict]:
    return [
        p for p in pairs if p["label"] == "after_better" and p["kind"] == "preference"
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-pairs",
        action="store_true",
        help="re-print the signal check from the committed pairs, skipping the "
        "join (which needs the uncommitted per-revision geometry)",
    )
    if ap.parse_args().from_pairs:
        print_directional_signal(directional_pairs(load_pairs(PAIRS)))
        return

    labels = json.load(open(S / "labels.json"))
    xfail_path = S / "labels_xfail.json"
    if xfail_path.exists():
        labels += json.load(open(xfail_path))
    geo = load_geometry()
    print(f"geometry revisions available: {len(geo)}")

    pairs: list[dict] = []
    anchors: list[dict] = []
    stats: Counter = Counter()
    drop: Counter = Counter()

    for row in labels:
        if row["source"] in ("open_bug", "xfail_known_bad"):
            out = emit_anchors(row, geo.get(row["sha_before"]))
            anchors += out.rows
        else:
            out = emit_pairs(
                row, geo.get(row["sha_before"]), geo.get(row.get("sha_after"))
            )
            pairs += out.rows
        stats += out.stats
        drop += out.drops

    with PAIRS.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    with ANCHORS.open("w") as fh:
        for a in anchors:
            fh.write(json.dumps(a) + "\n")

    print("\n=== emitted ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:34s} {v}")
    print("\n=== dropped ===")
    for k, v in sorted(drop.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {v}")

    strong = directional_pairs(pairs)
    print(f"\ntotal pairs: {len(pairs)}   directional (after_better): {len(strong)}")
    print(
        f"abort transitions: {sum(1 for p in pairs if p['kind'] == 'abort_transition')}"
    )
    print(f"anchors: {len(anchors)}")
    print(f"distinct fixtures in pairs: {len({p['fixture'] for p in pairs})}")
    print(f"features per vector: {len(feature_keys(pairs))}")

    if strong:
        print_directional_signal(strong)


if __name__ == "__main__":
    main()
