"""Join geometry vectors onto history labels to produce preference pairs.

A pair is only emitted when the comparison is actually about the engine:

  * the fixture resolves at both revisions
  * its ``.mmd`` content hash is IDENTICAL at both, so the geometry moved
    because the engine moved and not because the map was rewritten
  * both revisions measured through the same routing entrypoint
  * the feature vectors actually differ

Abort transitions are emitted as their own class. A fixture that laid out at
one revision and raises at the other is the cleanest label in the corpus: no
human judgement is involved and the direction is unambiguous.

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

S = Path(__file__).parent
PAIRS = S.parent / "dataset_pairs.jsonl"

MIN_MOVED = 8
"""Moved pairs a feature needs before the report will state a percentage for it."""

FLAG_MARGIN = 0.18
"""Distance from chance at which the grouped reading is called discriminative."""

SENTINEL = frozenset({"min_marker_gap", "min_station_distance"})
"""Features whose extractor emits ``-1.0`` for "no such measurement exists".

A map with no foreign line near any marker has no minimum gap. Read as the
number -1 the sentinel would be the tightest clearance in the corpus, inverting
the feature, so a pair undefined at either revision contributes no movement.
Mirrors ``fit_objective.SENTINEL``.
"""


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


def vecs_differ(a: dict, b: dict) -> bool:
    return any(abs(a[k] - b[k]) > 1e-6 for k in a if k in b)


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
        if key in SENTINEL and -1.0 in (before[key], after[key]):
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
        before, after = row["sha_before"], row.get("sha_after")

        if row["source"] in ("open_bug", "xfail_known_bad"):
            gb = geo.get(before)
            if not gb:
                drop["open_bug_no_geometry"] += 1
                continue
            for fx in row["fixtures"]:
                rec = gb["fixtures"].get(fx)
                if not rec or rec.get("status") != "ok":
                    drop["open_bug_fixture_missing"] += 1
                    continue
                anchors.append(
                    {
                        "kind": "defective",
                        "source": row["source"],
                        "fixture": fx,
                        "sha": before,
                        "issue": row.get("issue"),
                        "check": row.get("check"),
                        "features": rec["features"],
                        "confidence": row["confidence"],
                        "title": row["title"],
                    }
                )
                stats[f"anchor_{row['source']}"] += 1
            continue

        gb, ga = geo.get(before), geo.get(after)
        if not gb or not ga:
            drop[f"{row['source']}_missing_geometry"] += 1
            continue
        if gb.get("routing_entrypoint") != ga.get("routing_entrypoint"):
            drop["entrypoint_straddle"] += 1
            continue

        named = set(row["fixtures"])
        shared = set(gb["fixtures"]) & set(ga["fixtures"])
        scope_fixtures = (named & shared) if named else shared
        if named and not (named & shared):
            drop[f"{row['source']}_named_fixture_absent"] += 1

        for fx in sorted(scope_fixtures):
            rb, ra = gb["fixtures"][fx], ga["fixtures"][fx]
            if rb.get("input_sha1") != ra.get("input_sha1"):
                drop["input_changed"] += 1
                continue

            ok_b = rb.get("status") == "ok"
            ok_a = ra.get("status") == "ok"

            if ok_b != ok_a:
                pairs.append(
                    {
                        "kind": "abort_transition",
                        "fixture": fx,
                        "source": row["source"],
                        "pr": row.get("pr"),
                        "issue": row.get("issue"),
                        "sha_before": before,
                        "sha_after": after,
                        "features_before": rb["features"] if ok_b else None,
                        "features_after": ra["features"] if ok_a else None,
                        "error": (ra if not ok_a else rb).get("error"),
                        "label": "after_worse" if ok_b else "after_better",
                        "confidence": "certain",
                        "title": row["title"],
                    }
                )
                stats["abort_transition"] += 1
                continue

            if not (ok_b and ok_a):
                drop["both_abort"] += 1
                continue
            if not vecs_differ(rb["features"], ra["features"]):
                drop["geometry_identical"] += 1
                continue

            # An issue-fix row carries direction but no trustworthy subject, so
            # every fixture the merge actually moved inherits the direction, and
            # a prose-named fixture that also moved is marked as corroborated.
            geometry_derived = row.get("attribution") == "geometry_derived"
            is_named = fx in named or (
                geometry_derived and fx in set(row.get("fixtures_named_in_issue") or [])
            )
            directional = row["source"] in ("issue_fix", "xfail_cleared") and (
                is_named or geometry_derived
            )
            pairs.append(
                {
                    "kind": "preference",
                    "fixture": fx,
                    "source": row["source"],
                    "corroborated_by_issue_text": bool(
                        geometry_derived
                        and fx in set(row.get("fixtures_named_in_issue") or [])
                    ),
                    "pr": row.get("pr"),
                    "issue": row.get("issue"),
                    "sha_before": before,
                    "sha_after": after,
                    "features_before": rb["features"],
                    "features_after": ra["features"],
                    "check": row.get("check"),
                    "label": "after_better" if directional else "after_not_worse",
                    "scope": "fixture" if is_named else "pr_set",
                    "confidence": row["confidence"] if is_named else "weak",
                    "title": row["title"],
                }
            )
            stats[f"pair_{row['source']}_{'named' if is_named else 'set'}"] += 1

    with (S / "dataset_pairs.jsonl").open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    with (S / "dataset_anchors.jsonl").open("w") as fh:
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
