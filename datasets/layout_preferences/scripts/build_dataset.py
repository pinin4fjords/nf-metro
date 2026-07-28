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
    python build_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

S = Path(__file__).parent
FEATURE_KEYS: list[str] = []


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


def main() -> None:
    labels = json.load(open(S / "labels.json"))
    xfail_path = S / "labels_xfail.json"
    if xfail_path.exists():
        labels += json.load(open(xfail_path))
    geo = load_geometry()
    print(f"geometry revisions available: {len(geo)}")

    global FEATURE_KEYS
    for d in geo.values():
        for rec in d["fixtures"].values():
            if rec.get("status") == "ok":
                FEATURE_KEYS = sorted(rec["features"])
                break
        if FEATURE_KEYS:
            break

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

    strong = [
        p for p in pairs if p["label"] == "after_better" and p["kind"] == "preference"
    ]
    print(f"\ntotal pairs: {len(pairs)}   directional (after_better): {len(strong)}")
    print(
        f"abort transitions: {sum(1 for p in pairs if p['kind'] == 'abort_transition')}"
    )
    print(f"anchors: {len(anchors)}")
    print(f"distinct fixtures in pairs: {len({p['fixture'] for p in pairs})}")
    print(f"features per vector: {len(FEATURE_KEYS)}")

    if strong:
        print("\n=== directional signal check (issue-fix pairs) ===")
        print("fraction of pairs where the feature DECREASED after the fix:")
        for k in FEATURE_KEYS:
            d = [
                p["features_after"][k] - p["features_before"][k]
                for p in strong
                if p["features_before"] and p["features_after"]
            ]
            moved = [x for x in d if abs(x) > 1e-6]
            if len(moved) < 8:
                continue
            down = sum(1 for x in moved if x < 0) / len(moved)
            flag = "  <-- discriminative" if abs(down - 0.5) > 0.18 else ""
            print(f"  {k:28s} {down * 100:5.1f}%  (n={len(moved):3d}){flag}")


if __name__ == "__main__":
    main()
