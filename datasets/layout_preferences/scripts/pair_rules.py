"""When two revisions of one fixture form a usable preference pair.

A label row says "something about the after side was better"; these rules decide
which fixtures that claim may legitimately be attached to, and at what strength.
A pair is only emitted when the comparison is actually about the engine:

  * the fixture resolves at both revisions
  * its ``.mmd`` content hash is IDENTICAL at both, so the geometry moved
    because the engine moved and not because the map was rewritten
  * both revisions measured through the same routing entrypoint
  * the feature vectors actually differ

Abort transitions are emitted as their own class. A fixture that laid out at one
revision and raises at the other is the cleanest label in the corpus: no human
judgement is involved and the direction is unambiguous.

The historical join (``build_dataset.py``) and forward capture
(``capture_pr.py``) both run through here, so the two corpora carry the same
schema and the same emission strengths and can be concatenated without a
reconciliation step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

LAYOUT_LABELS = frozenset({"layout", "routing", "render", "bug"})
"""Issue labels that make a closed issue a claim about render quality."""

DIRECTIONAL_SOURCES = ("issue_fix", "xfail_cleared")
"""Label sources that assert a direction rather than mere non-blockingness.

Everything else contributes ``after_not_worse``: a merged PR ratifies its diff
as a whole, so treating its renders as individually preferred would fit weights
to reviewer impatience.
"""


@dataclass
class Emission:
    """Rows produced from one label row, with the tallies that explain them."""

    rows: list[dict] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    drops: Counter = field(default_factory=Counter)


def vecs_differ(a: dict, b: dict) -> bool:
    return any(abs(a[k] - b[k]) > 1e-6 for k in a if k in b)


def emit_anchors(row: dict, before: dict | None) -> Emission:
    """One-sided negatives: the named fixtures were defective at ``sha_before``."""
    out = Emission()
    if not before:
        out.drops["open_bug_no_geometry"] += 1
        return out
    for fx in row["fixtures"]:
        rec = before["fixtures"].get(fx)
        if not rec or rec.get("status") != "ok":
            out.drops["open_bug_fixture_missing"] += 1
            continue
        out.rows.append(
            {
                "kind": "defective",
                "source": row["source"],
                "fixture": fx,
                "sha": row["sha_before"],
                "issue": row.get("issue"),
                "check": row.get("check"),
                "features": rec["features"],
                "confidence": row["confidence"],
                "title": row["title"],
            }
        )
        out.stats[f"anchor_{row['source']}"] += 1
    return out


def emit_pairs(row: dict, before: dict | None, after: dict | None) -> Emission:
    """Pairs for every fixture the two revisions moved, at the row's strength."""
    out = Emission()
    if not before or not after:
        out.drops[f"{row['source']}_missing_geometry"] += 1
        return out
    if before.get("routing_entrypoint") != after.get("routing_entrypoint"):
        out.drops["entrypoint_straddle"] += 1
        return out

    named = set(row["fixtures"])
    shared = set(before["fixtures"]) & set(after["fixtures"])
    scope_fixtures = (named & shared) if named else shared
    if named and not (named & shared):
        out.drops[f"{row['source']}_named_fixture_absent"] += 1

    for fx in sorted(scope_fixtures):
        rb, ra = before["fixtures"][fx], after["fixtures"][fx]
        if rb.get("input_sha1") != ra.get("input_sha1"):
            out.drops["input_changed"] += 1
            continue

        ok_b = rb.get("status") == "ok"
        ok_a = ra.get("status") == "ok"

        if ok_b != ok_a:
            out.rows.append(
                {
                    "kind": "abort_transition",
                    "fixture": fx,
                    "source": row["source"],
                    "pr": row.get("pr"),
                    "issue": row.get("issue"),
                    "sha_before": row["sha_before"],
                    "sha_after": row["sha_after"],
                    "features_before": rb["features"] if ok_b else None,
                    "features_after": ra["features"] if ok_a else None,
                    "error": (ra if not ok_a else rb).get("error"),
                    "label": "after_worse" if ok_b else "after_better",
                    "confidence": "certain",
                    "title": row["title"],
                }
            )
            out.stats["abort_transition"] += 1
            continue

        if not (ok_b and ok_a):
            out.drops["both_abort"] += 1
            continue
        if not vecs_differ(rb["features"], ra["features"]):
            out.drops["geometry_identical"] += 1
            continue

        # An issue-fix row carries direction but no trustworthy subject, so
        # every fixture the merge actually moved inherits the direction, and
        # a prose-named fixture that also moved is marked as corroborated.
        geometry_derived = row.get("attribution") == "geometry_derived"
        named_in_issue = set(row.get("fixtures_named_in_issue") or [])
        is_named = fx in named or (geometry_derived and fx in named_in_issue)
        directional = row["source"] in DIRECTIONAL_SOURCES and (
            is_named or geometry_derived
        )
        out.rows.append(
            {
                "kind": "preference",
                "fixture": fx,
                "source": row["source"],
                "corroborated_by_issue_text": bool(
                    geometry_derived and fx in named_in_issue
                ),
                "pr": row.get("pr"),
                "issue": row.get("issue"),
                "sha_before": row["sha_before"],
                "sha_after": row["sha_after"],
                "features_before": rb["features"],
                "features_after": ra["features"],
                "check": row.get("check"),
                "label": "after_better" if directional else "after_not_worse",
                "scope": "fixture" if is_named else "pr_set",
                "confidence": row["confidence"] if is_named else "weak",
                "title": row["title"],
            }
        )
        out.stats[f"pair_{row['source']}_{'named' if is_named else 'set'}"] += 1
    return out
