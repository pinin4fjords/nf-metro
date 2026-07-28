"""Guards on the layout-preference capture pipeline and its committed ledger.

Two things are protected here. The emission rules decide which comparisons are
about the engine and at what strength each preference is asserted, so a change
that quietly promoted a set-level ratification to a per-render positive would
poison the fit rather than break anything visible. And the committed forward
ledger is append-only data that no other test reads, so its schema and its
scope discipline are asserted directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "datasets" / "layout_preferences"
sys.path.insert(0, str(DATASET / "scripts"))

import capture_pr as cap  # noqa: E402
from pair_rules import emit_anchors, emit_pairs  # noqa: E402

FEATURES = {"crossings": 2.0, "bends_per_route": 3.0}
MOVED = {"crossings": 1.0, "bends_per_route": 2.0}

LABELS = {"after_better", "after_not_worse", "after_worse"}
SCOPES = {"fixture", "pr_set", None}
VERDICT_SCOPES = {"fixture", "pr", None}


def fixture_rec(features=FEATURES, *, input_sha1="in0", status="ok", error=None):
    rec = {"input_sha1": input_sha1, "status": status}
    if features is not None:
        rec["features"] = features
    if error:
        rec["error"] = error
    return rec


def revision(sha, fixtures, entrypoint="route_edges_centred"):
    return {"sha": sha, "routing_entrypoint": entrypoint, "fixtures": fixtures}


def label_row(source="pr_signoff", **over):
    row = {
        "source": source,
        "pr": 42,
        "fixtures": [],
        "sha_before": "before0",
        "sha_after": "after0",
        "confidence": "medium",
        "title": "t",
    }
    row.update(over)
    return row


def merged_pr(issues=(), number=42):
    return cap.MergedPR(
        number=number,
        title="fix: something",
        merged_at="2026-07-28T00:00:00Z",
        sha_before="before0",
        sha_after="after0",
        issues=tuple(issues),
    )


def issue(number=7, labels=("layout",), body=""):
    return cap.ClosedIssue(
        number=number, title="[bug]: a defect", body=body, labels=labels
    )


# --------------------------------------------------------------------------- #
# Emission rules
# --------------------------------------------------------------------------- #


def test_an_edited_mmd_is_not_a_preference():
    """A rewritten map is indistinguishable from an engine change, so it drops."""
    out = emit_pairs(
        label_row(),
        revision("b", {"fan": fixture_rec(input_sha1="in0")}),
        revision("a", {"fan": fixture_rec(MOVED, input_sha1="in1")}),
    )
    assert out.rows == []
    assert out.drops["input_changed"] == 1


def test_identical_geometry_yields_no_pair():
    out = emit_pairs(
        label_row(),
        revision("b", {"fan": fixture_rec()}),
        revision("a", {"fan": fixture_rec()}),
    )
    assert out.rows == []
    assert out.drops["geometry_identical"] == 1


def test_geometry_from_two_routing_entrypoints_is_not_comparable():
    out = emit_pairs(
        label_row(),
        revision("b", {"fan": fixture_rec()}, entrypoint="route_edges"),
        revision("a", {"fan": fixture_rec(MOVED)}),
    )
    assert out.rows == []
    assert out.drops["entrypoint_straddle"] == 1


@pytest.mark.parametrize(
    ("before_ok", "after_ok", "label"),
    [(True, False, "after_worse"), (False, True, "after_better")],
)
def test_abort_transition_is_certain_and_directional(before_ok, after_ok, label):
    out = emit_pairs(
        label_row(),
        revision(
            "b",
            {
                "fan": fixture_rec()
                if before_ok
                else fixture_rec(None, status="error", error="CurveInvariantError")
            },
        ),
        revision(
            "a",
            {
                "fan": fixture_rec(MOVED)
                if after_ok
                else fixture_rec(None, status="error", error="CurveInvariantError")
            },
        ),
    )
    (row,) = out.rows
    assert row["kind"] == "abort_transition"
    assert (row["label"], row["confidence"]) == (label, "certain")


def test_a_pr_signoff_is_never_a_per_render_positive():
    """A merged diff ratifies its renders as a set, never individually (#1586)."""
    out = emit_pairs(
        label_row(),
        revision("b", {"fan": fixture_rec(), "fold": fixture_rec()}),
        revision("a", {"fan": fixture_rec(MOVED), "fold": fixture_rec(MOVED)}),
    )
    assert len(out.rows) == 2
    for row in out.rows:
        assert row["label"] == "after_not_worse"
        assert row["scope"] == "pr_set"
        assert row["confidence"] == "weak"


def test_issue_fix_direction_reaches_every_moved_fixture():
    out = emit_pairs(
        label_row(
            "issue_fix",
            attribution="geometry_derived",
            fixtures_named_in_issue=["fold"],
            issue=7,
        ),
        revision("b", {"fan": fixture_rec(), "fold": fixture_rec()}),
        revision("a", {"fan": fixture_rec(MOVED), "fold": fixture_rec(MOVED)}),
    )
    by_fixture = {row["fixture"]: row for row in out.rows}
    assert {row["label"] for row in out.rows} == {"after_better"}
    assert by_fixture["fold"]["corroborated_by_issue_text"] is True
    assert by_fixture["fan"]["corroborated_by_issue_text"] is False


def test_an_anchor_needs_measurable_geometry():
    ok = emit_anchors(
        label_row("xfail_known_bad", fixtures=["fan"], check="_XFAIL_A"),
        revision("b", {"fan": fixture_rec()}),
    )
    assert ok.rows[0]["check"] == "_XFAIL_A"
    aborted = emit_anchors(
        label_row("xfail_known_bad", fixtures=["fan"], check="_XFAIL_A"),
        revision("b", {"fan": fixture_rec(None, status="error")}),
    )
    assert aborted.rows == []
    assert aborted.drops["open_bug_fixture_missing"] == 1


# --------------------------------------------------------------------------- #
# What a merged PR claims
# --------------------------------------------------------------------------- #


def test_a_directional_pr_does_not_also_ratify_itself():
    rows = cap.label_rows(merged_pr([issue()]), [], {"fan"})
    assert [row["source"] for row in rows] == ["issue_fix"]
    assert rows[0]["attribution"] == "geometry_derived"


def test_an_issue_without_a_layout_label_leaves_only_a_signoff():
    rows = cap.label_rows(merged_pr([issue(labels=("documentation",))]), [], {"fan"})
    assert [row["source"] for row in rows] == ["pr_signoff"]


def test_prose_naming_only_corroborates():
    rows = cap.label_rows(
        merged_pr([issue(body="fan_out_wrap looks wrong")]), [], {"fan_out_wrap"}
    )
    assert rows[0]["fixtures"] == []
    assert rows[0]["fixtures_named_in_issue"] == ["fan_out_wrap"]


def test_xfail_churn_becomes_a_named_check_row():
    events = [
        cap.XfailEvent("fan", "_XFAIL_BBOX_TOP_PAD", added=False),
        cap.XfailEvent("fold", "_XFAIL_LABEL_AT_STATION_X", added=True),
    ]
    rows = {row["source"]: row for row in cap.label_rows(merged_pr(), events, {"fan"})}
    assert rows["xfail_cleared"]["check"] == "_XFAIL_BBOX_TOP_PAD"
    assert rows["xfail_cleared"]["confidence"] == "high"
    known_bad = rows["xfail_known_bad"]
    assert known_bad["sha_after"] is None
    assert known_bad["sha_before"] == "after0", "an added entry anchors the merge"


def test_registry_entries_reads_both_assignment_forms():
    source = (
        '_XFAIL_ONE: dict[str, str] = {"fan": "reason"}\n'
        '_XFAIL_TWO = {"fold": "reason", "wrap": "reason"}\n'
        'OTHER = {"ignored": "not a registry"}\n'
    )
    assert cap.registry_entries(source) == {
        ("_XFAIL_ONE", "fan"),
        ("_XFAIL_TWO", "fold"),
        ("_XFAIL_TWO", "wrap"),
    }


def test_the_named_check_supersedes_the_geometry_derived_row():
    weak = {
        "kind": "preference",
        "fixture": "fan",
        "sha_before": "b",
        "sha_after": "a",
        "scope": "pr_set",
        "confidence": "weak",
        "check": None,
    }
    strong = {**weak, "scope": "fixture", "confidence": "high", "check": "_XFAIL_A"}
    assert cap.strongest_per_comparison([weak, strong]) == [strong]
    assert cap.strongest_per_comparison([strong, weak]) == [strong]


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


def test_a_verdict_on_an_unchanged_render_is_refused():
    with pytest.raises(cap.CaptureError, match="did not move"):
        cap.check_fixture_verdicts({"fold"}, {"fan"})


def test_a_set_level_verdict_never_sets_a_per_render_label():
    rows = [
        {
            "kind": "preference",
            "fixture": "fan",
            "label": "after_not_worse",
            "scope": "pr_set",
            "confidence": "weak",
        }
    ]
    cap.apply_verdicts(rows, fixture_verdicts={}, pr_verdict="improvement")
    assert rows[0]["verdict_scope"] == "pr"
    assert rows[0]["label"] == "after_not_worse"
    assert rows[0]["scope"] == "pr_set"


def test_a_per_render_verdict_sets_the_label_it_asserts():
    rows = [
        {
            "kind": "preference",
            "fixture": "fan",
            "label": "after_not_worse",
            "scope": "pr_set",
            "confidence": "weak",
        },
        {
            "kind": "preference",
            "fixture": "fold",
            "label": "after_not_worse",
            "scope": "pr_set",
            "confidence": "weak",
        },
    ]
    cap.apply_verdicts(
        rows, fixture_verdicts={"fan": "detrimental"}, pr_verdict="neutral"
    )
    fan, fold = rows
    assert (fan["label"], fan["scope"], fan["verdict_scope"]) == (
        "after_worse",
        "fixture",
        "fixture",
    )
    assert (fold["label"], fold["verdict_scope"]) == ("after_not_worse", "pr")


def test_a_verdict_cannot_soften_an_abort_transition():
    rows = [
        {
            "kind": "abort_transition",
            "fixture": "fan",
            "label": "after_better",
            "confidence": "certain",
        }
    ]
    cap.apply_verdicts(rows, fixture_verdicts={"fan": "neutral"}, pr_verdict=None)
    assert (rows[0]["label"], rows[0]["confidence"]) == ("after_better", "certain")


def test_a_merged_pr_cannot_be_detrimental_as_a_whole():
    """A merged detrimental render has to be named, not asserted set-wide."""
    with pytest.raises(cap.CaptureError, match="--fixture-verdict"):
        cap.main(["1606", "--pr-verdict", "detrimental"])


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def test_the_ledger_is_append_only(tmp_path):
    path = tmp_path / "forward_pairs.jsonl"
    cap.append_jsonl(path, [{"pr": 1}])
    first = path.read_bytes()
    cap.append_jsonl(path, [{"pr": 2}])
    assert path.read_bytes().startswith(first)
    assert [row["pr"] for row in cap.read_jsonl(path)] == [1, 2]


def test_a_captured_pr_is_not_captured_twice(monkeypatch, tmp_path):
    log = tmp_path / "forward_log.jsonl"
    monkeypatch.setattr(cap, "LOG", log)
    cap.append_jsonl(
        log, [{"kind": "captured", "pr": 1606}, {"kind": "verdict", "pr": 1606}]
    )
    assert cap.examined_prs() == {1606: "captured"}
    with pytest.raises(cap.CaptureError, match="already examined"):
        cap.main(["1606"])


def test_a_pr_that_moved_nothing_is_still_recorded_as_examined(monkeypatch, tmp_path):
    log = tmp_path / "forward_log.jsonl"
    monkeypatch.setattr(cap, "LOG", log)
    cap.append_jsonl(log, [{"kind": "no_capture", "pr": 1601, "reason": "docs only"}])
    assert cap.examined_prs() == {1601: "no_capture"}


def test_a_sweep_will_not_guess_where_to_start(monkeypatch, tmp_path):
    monkeypatch.setattr(cap, "LOG", tmp_path / "empty.jsonl")
    with pytest.raises(cap.CaptureError, match="--since"):
        cap.main(["--sweep"])


def test_a_sweep_cannot_carry_one_verdict_across_many_prs():
    with pytest.raises(cap.CaptureError, match="cannot apply"):
        cap.main(["--sweep", "--pr-verdict", "improvement"])


# --------------------------------------------------------------------------- #
# The committed ledger
# --------------------------------------------------------------------------- #


def committed(name):
    path = DATASET / name
    return cap.read_jsonl(path) if path.exists() else []


def test_committed_pairs_carry_an_explicit_verdict_scope():
    rows = committed("forward_pairs.jsonl")
    if not rows:
        pytest.skip("no forward captures yet")
    for row in rows:
        assert row["kind"] in {"preference", "abort_transition"}
        assert row["label"] in LABELS
        assert row.get("scope") in SCOPES
        assert "verdict" in row and "verdict_scope" in row, "scope must be explicit"
        assert row["verdict_scope"] in VERDICT_SCOPES
        assert row["features_before"] or row["features_after"]


def test_a_set_level_ratification_is_never_stored_as_a_per_render_positive():
    for row in committed("forward_pairs.jsonl"):
        if row["verdict_scope"] == "pr":
            assert row["scope"] != "fixture"
            assert row["label"] != "after_better" or row["source"] != "pr_signoff"


def test_every_captured_pr_has_a_log_entry():
    logged = {row["pr"] for row in committed("forward_log.jsonl")}
    for name in ("forward_pairs.jsonl", "forward_anchors.jsonl"):
        for row in committed(name):
            assert row["pr"] in logged, f"{name}: PR #{row['pr']} has no log row"


def test_committed_rows_share_the_historical_pair_schema():
    """Phase 2 concatenates the two corpora, so the columns have to agree."""
    forward = committed("forward_pairs.jsonl")
    if not forward:
        pytest.skip("no forward captures yet")
    historical = json.loads(
        (DATASET / "dataset_pairs.jsonl").read_text().splitlines()[0]
    )
    for row in forward:
        if row["kind"] != "preference":
            continue
        missing = set(historical) - set(row)
        assert not missing, f"forward row lacks {sorted(missing)}"
        assert set(row["features_before"]) == set(historical["features_before"])
