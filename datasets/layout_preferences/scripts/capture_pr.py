#!/usr/bin/env python3
"""Capture the review signal a merged PR produced, into the forward ledger.

Every merged engine PR is a preference claim: the reviewer looked at the changed
renders and let them through. The CI render-diff renders that claim to an HTML
page which is then deleted, which is why the historical corpus had to be
recovered by replaying ~900 revisions (#1583). This records it as data.

Why capture from the merged commits rather than from CI's render-diff:

* ``.github/workflows/pr-cleanup.yml`` removes ``_pr/<N>/`` when the PR closes,
  which is the exact moment the verdict becomes final. Anything persisted there
  is deleted on merge.
* The render-diff's "before" is the PR's base SHA. #1583 measured that as the
  wrong pairing for this repo -- PRs stack on chained base branches, so 50 of 94
  fix PRs showed zero geometry change against it. ``mergeCommit^1`` is the
  target-branch tip immediately before the merge, so the pair is a true
  main-to-main delta.
* The diff page reports the ten display metrics of ``tests/layout_metrics.py``;
  the fit consumes the 31 features of ``extract_features.py``. Persisting the
  former would leave Phase 2 a reconstruction step regardless.

**Capture is back-fillable, which is what makes a local script sound here.** The
inputs are commits, and commits persist: a PR missed today can be captured next
month with an identical result. CI capture has the opposite property, where
missing the window loses the data permanently to pruning. ``--sweep`` exploits
this by finding every merged PR the ledger has not examined yet, so the ledger
heals itself and capture is a periodic chore rather than a per-PR obligation.

Only the verdict needs a human, and its **scope** is carried by which flag
supplies it: ``--pr-verdict`` records a set-level ratification, whereas
``--fixture-verdict`` asserts something about one named render. Conflating the
two cost the #1586 fit 5.2 points, so a fixture-level verdict naming a render
the merge did not move is a hard error rather than a quietly demoted row.

Usage:
    python capture_pr.py 1606 --pr-verdict neutral
    python capture_pr.py 1606 --fixture-verdict fan_out_wrap=improvement
    python capture_pr.py --sweep                    # every PR not yet examined
    python capture_pr.py --sweep --since 1600
    python capture_pr.py 1606 --verdict-only --fixture-verdict x=detrimental
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

from pair_rules import LAYOUT_LABELS, Emission, emit_anchors, emit_pairs
from revisions import (
    REPO_ROOT,
    GeometryError,
    MissingRevision,
    ensure_local,
    ensure_worktree,
    geometry_at,
    git,
    remove_worktree,
    rev_parse,
    touches_engine,
)

S = Path(__file__).resolve().parent
DATASET = S.parent
PAIRS = DATASET / "forward_pairs.jsonl"
ANCHORS = DATASET / "forward_anchors.jsonl"
LOG = DATASET / "forward_log.jsonl"

VERDICT_LABEL = {
    "improvement": "after_better",
    "neutral": "after_not_worse",
    "detrimental": "after_worse",
}

CONFIDENCE_RANK = {"weak": 0, "triage": 0, "medium": 1, "high": 2, "certain": 3}

GENERIC_STEMS = frozenset({"pipeline"})
"""Fixture stems too generic to recognise in prose."""


class CaptureError(RuntimeError):
    """Capture cannot proceed, or would record a claim that is not true."""


# --------------------------------------------------------------------------- #
# GitHub + git reads
# --------------------------------------------------------------------------- #


def gh_json(*args: str) -> dict | list:
    r = subprocess.run(
        ["gh", *args, "--repo", "seqeralabs/nf-metro"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode:
        raise CaptureError(f"gh {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout)


@dataclass(frozen=True)
class ClosedIssue:
    number: int
    title: str
    body: str
    labels: tuple[str, ...]

    @property
    def is_layout(self) -> bool:
        return bool(LAYOUT_LABELS & set(self.labels))


@dataclass(frozen=True)
class MergedPR:
    number: int
    title: str
    merged_at: str
    sha_before: str
    sha_after: str
    issues: tuple[ClosedIssue, ...]

    @property
    def layout_issues(self) -> list[ClosedIssue]:
        return [i for i in self.issues if i.is_layout]


@dataclass(frozen=True)
class XfailEvent:
    """One ``_XFAIL_*`` registry entry appearing or disappearing across a merge."""

    fixture: str
    check: str
    added: bool


def fetch_issue(number: int) -> ClosedIssue:
    d = gh_json("issue", "view", str(number), "--json", "number,title,body,labels")
    assert isinstance(d, dict)
    return ClosedIssue(
        number=d["number"],
        title=d["title"] or "",
        body=d["body"] or "",
        labels=tuple(lbl["name"] for lbl in d["labels"]),
    )


def fetch_pr(number: int) -> MergedPR:
    """Resolve a merged PR to the commit pair the dataset pairs on."""
    d = gh_json(
        "pr",
        "view",
        str(number),
        "--json",
        "number,title,mergedAt,mergeCommit,closingIssuesReferences",
    )
    assert isinstance(d, dict)
    merge = (d.get("mergeCommit") or {}).get("oid")
    if not d.get("mergedAt") or not merge:
        raise CaptureError(
            f"PR #{number} is not merged, so it has no merge commit to pair on. "
            "Only merged PRs carry a sign-off."
        )
    ensure_local(merge)
    parent = rev_parse(f"{merge}^1")
    if not parent:
        raise CaptureError(f"PR #{number}: {merge[:9]}^1 does not resolve")
    return MergedPR(
        number=d["number"],
        title=d["title"] or "",
        merged_at=d["mergedAt"],
        sha_before=parent,
        sha_after=merge,
        issues=tuple(
            fetch_issue(r["number"]) for r in d.get("closingIssuesReferences") or []
        ),
    )


def registry_entries(source: str) -> set[tuple[str, str]]:
    """``(check, fixture)`` for the ``_XFAIL_*`` registries in one test module.

    The registries are module-level dicts keyed by fixture path, read with the
    AST because importing a historical test module would need that revision's
    whole test environment. Only the keys are read: a registry whose reasons are
    shared module constants or implicitly concatenated strings is not a literal,
    and evaluating the whole dict would discard it entirely.

    Keys are normalised to the stem the geometry records are keyed by, since a
    registry may key by bare stem, by file name, or by repo-relative path.
    """
    entries: set[tuple[str, str]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return entries
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        check = next((n for n in names if n.startswith("_XFAIL_")), None)
        if check is None or not isinstance(value, ast.Dict):
            continue
        entries.update(
            (check, PurePosixPath(key.value).stem)
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    return entries


@lru_cache(maxsize=None)
def xfail_entries(sha: str) -> frozenset[tuple[str, str]]:
    """Every ``_XFAIL_*`` registry entry in the test suite at ``sha``.

    ``git grep`` narrows the read to the few modules that carry a registry;
    reading the whole of ``tests/`` out of the tree instead costs about as long
    as measuring the entire fixture corpus.
    """
    listed = git("grep", "-l", "_XFAIL_", sha, "--", "tests").stdout.splitlines()
    entries: set[tuple[str, str]] = set()
    for line in listed:
        _, _, path = line.partition(":")
        if path.endswith(".py"):
            entries |= registry_entries(git("show", f"{sha}:{path}").stdout)
    return frozenset(entries)


def xfail_events(before: str, after: str) -> list[XfailEvent]:
    """Registry churn across a merge, newest state minus oldest."""
    was, now = xfail_entries(before), xfail_entries(after)
    events = [XfailEvent(fx, check, added=False) for check, fx in sorted(was - now)]
    events += [XfailEvent(fx, check, added=True) for check, fx in sorted(now - was)]
    return events


# --------------------------------------------------------------------------- #
# Label rows
# --------------------------------------------------------------------------- #


def mentioned_stems(text: str, stems: set[str]) -> list[str]:
    """Fixture stems named in prose, for corroboration only.

    #1583 established that prose cannot attribute a label: an issue names the
    motivating real-world pipeline map while the fix moves synthetic topology
    fixtures. The subject comes from the geometry; this only marks the rows
    where the two happen to agree.
    """
    lowered = text.lower()
    return sorted(s for s in stems if s not in GENERIC_STEMS and s.lower() in lowered)


def label_rows(pr: MergedPR, events: list[XfailEvent], stems: set[str]) -> list[dict]:
    """The claims a merged PR makes, before geometry decides their subject.

    A PR closing a layout-labelled issue asserts a direction, so it yields an
    ``issue_fix`` row; one that does not yields the weaker ``pr_signoff``. It
    yields one or the other and never both: emitting a set-level duplicate of a
    row that is already directional would count the same comparison twice, at
    two strengths.
    """
    issues = pr.layout_issues
    common = {
        "pr": pr.number,
        "sha_before": pr.sha_before,
        "sha_after": pr.sha_after,
        "pairing": "merge_first_parent",
        "capture": "forward",
    }
    if issues:
        rows = [
            {
                **common,
                "source": "issue_fix",
                "issue": issues[0].number,
                "issues": [i.number for i in issues],
                # The subject is whatever the merge moved; prose only corroborates.
                "fixtures": [],
                "attribution": "geometry_derived",
                "fixtures_named_in_issue": mentioned_stems(
                    " ".join(f"{i.title} {i.body}" for i in issues), stems
                ),
                "claim": "after_better_than_before",
                "scope": "fixture",
                "confidence": "medium",
                "title": issues[0].title[:140],
            }
        ]
    else:
        rows = [
            {
                **common,
                "source": "pr_signoff",
                "fixtures": [],
                "claim": "no_change_was_blocking",
                "scope": "pr",
                "confidence": "medium",
                "title": pr.title[:140],
            }
        ]

    for ev in events:
        churn = {
            **common,
            "fixtures": [ev.fixture],
            "check": ev.check,
            "scope": "fixture",
            "confidence": "high",
        }
        if ev.added:
            # An added entry names a defect the merge acknowledged rather than
            # fixed, so it is a one-sided anchor on the revision that carries
            # the entry, and an anchor is measured at its `sha_before`.
            churn |= {
                "source": "xfail_known_bad",
                "sha_before": pr.sha_after,
                "sha_after": None,
                "claim": "before_is_defective",
                "title": f"xfail added: {ev.check} :: {pr.title}"[:140],
            }
        else:
            churn |= {
                "source": "xfail_cleared",
                "claim": "after_better_than_before",
                "title": f"xfail cleared: {ev.check} :: {pr.title}"[:140],
            }
        rows.append(churn)
    return rows


def strongest_per_comparison(rows: list[dict]) -> list[dict]:
    """One row per (fixture, before, after) comparison, keeping the strongest.

    A merge can both close an issue and clear an xfail for the same fixture. The
    xfail row names the violated invariant, so it supersedes the weaker
    geometry-derived one instead of training on the same comparison twice.
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["fixture"], row["sha_before"], row.get("sha_after"), row["kind"])
        rank = (
            row.get("scope") == "fixture",
            CONFIDENCE_RANK.get(row["confidence"], 0),
            row.get("check") is not None,
        )
        incumbent = best.get(key)
        if incumbent is None or rank > incumbent["_rank"]:
            best[key] = {"_rank": rank, "row": row}
    return [entry["row"] for entry in best.values()]


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


@dataclass
class Capture:
    """Everything one PR contributes to the ledger."""

    pr: MergedPR
    pairs: list[dict] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    drops: Counter = field(default_factory=Counter)

    @property
    def moved(self) -> set[str]:
        return {row["fixture"] for row in self.pairs}


def capture_rows(
    pr: MergedPR,
    before: dict,
    after: dict,
    events: list[XfailEvent],
) -> Capture:
    """Join geometry onto a PR's claims through the shared emission rules."""
    stems = set(after["fixtures"]) | set(before["fixtures"])
    out = Capture(pr=pr)
    for row in label_rows(pr, events, stems):
        emission: Emission
        if row["source"] == "xfail_known_bad":
            emission = emit_anchors(row, after)
            out.anchors += emission.rows
        else:
            emission = emit_pairs(row, before, after)
            out.pairs += emission.rows
        out.stats += emission.stats
        out.drops += emission.drops
    out.pairs = strongest_per_comparison(out.pairs)
    return out


def check_fixture_verdicts(named: set[str], moved: set[str]) -> None:
    """Refuse a per-render verdict on a render this merge did not move."""
    unknown = sorted(named - moved)
    if not unknown:
        return
    listed = ", ".join(sorted(moved)[:12]) or "(none)"
    more = "" if len(moved) <= 12 else f" (+{len(moved) - 12} more)"
    raise CaptureError(
        f"fixture-level verdict names {', '.join(unknown)}, which this merge did "
        "not move. A per-render verdict on an unchanged render is a set-level "
        "ratification wearing a per-render label, which is the failure mode this "
        "ledger exists to prevent. Use --pr-verdict for a set-level sign-off.\n"
        f"moved here: {listed}{more}"
    )


def apply_verdicts(
    rows: list[dict], *, fixture_verdicts: dict[str, str], pr_verdict: str | None
) -> None:
    """Record the human judgement on each row, at the scope it was given.

    A fixture-level verdict sets the row's label; a PR-level one never does,
    because a merged diff ratifies its renders as a set. An abort transition
    keeps its own label and certainty either way: whether a fixture lays out at
    all is not a matter of taste.
    """
    for row in rows:
        verdict = fixture_verdicts.get(row["fixture"])
        row["verdict"] = verdict or pr_verdict
        row["verdict_scope"] = "fixture" if verdict else ("pr" if pr_verdict else None)
        if verdict and row["kind"] != "abort_transition":
            row["label"] = VERDICT_LABEL[verdict]
            row["scope"] = "fixture"
            row["confidence"] = "high"


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows. The ledger is append-only: rows carry inline feature vectors,
    so a rewrite would churn the whole file for every capture."""
    if not rows:
        return
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def no_capture_row(number: int, reason: str, pr: MergedPR | None = None) -> dict:
    """A PR that was examined and yielded nothing, with why.

    The commit fields stay present-but-null when the PR never resolved to a
    commit pair, so every ``no_capture`` row in the log has one shape.
    """
    return {
        "kind": "no_capture",
        "pr": number,
        "merged_at": pr.merged_at if pr else None,
        "sha_before": pr.sha_before if pr else None,
        "sha_after": pr.sha_after if pr else None,
        "reason": reason,
    }


def examined_prs() -> dict[int, str]:
    """PR number -> log row kind, for every PR capture has already looked at."""
    return {
        row["pr"]: row["kind"]
        for row in read_jsonl(LOG)
        if row["kind"] in ("captured", "no_capture")
    }


def merged_prs_since(number: int, limit: int) -> list[int]:
    data = gh_json(
        "pr",
        "list",
        "--state",
        "merged",
        "--base",
        "main",
        "--limit",
        str(limit),
        "--json",
        "number",
    )
    assert isinstance(data, list)
    return sorted(p["number"] for p in data if p["number"] > number)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_fixture_verdicts(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        fixture, _, verdict = spec.partition("=")
        if verdict not in VERDICT_LABEL:
            raise CaptureError(
                f"--fixture-verdict {spec!r} must be FIXTURE=<"
                f"{'|'.join(VERDICT_LABEL)}>"
            )
        out[fixture] = verdict
    return out


def capture_one(
    number: int,
    *,
    worktree: Path,
    fixture_verdicts: dict[str, str],
    pr_verdict: str | None,
    dry_run: bool,
) -> None:
    pr = fetch_pr(number)
    if not touches_engine(pr.sha_before, pr.sha_after):
        print(f"#{number}: no engine change, nothing to measure")
        if not dry_run:
            reason = "diff does not reach the layout/render/parser engine"
            append_jsonl(LOG, [no_capture_row(number, reason, pr)])
        return

    before = geometry_at(pr.sha_before, worktree=worktree)
    after = geometry_at(pr.sha_after, worktree=worktree)
    events = xfail_events(pr.sha_before, pr.sha_after)
    out = capture_rows(pr, before, after, events)
    check_fixture_verdicts(set(fixture_verdicts), out.moved)
    apply_verdicts(out.pairs, fixture_verdicts=fixture_verdicts, pr_verdict=pr_verdict)

    directional = sum(1 for r in out.pairs if r["label"] == "after_better")
    print(
        f"#{number}: {len(out.pairs)} pairs ({directional} directional), "
        f"{len(out.anchors)} anchors, {len(events)} xfail events"
    )
    if out.drops:
        print(
            "   dropped: " + ", ".join(f"{k}={v}" for k, v in out.drops.most_common())
        )
    if dry_run:
        return

    append_jsonl(PAIRS, out.pairs)
    append_jsonl(ANCHORS, out.anchors)
    append_jsonl(
        LOG,
        [
            {
                "kind": "captured",
                "pr": number,
                "merged_at": pr.merged_at,
                "sha_before": pr.sha_before,
                "sha_after": pr.sha_after,
                "issues": [i.number for i in pr.layout_issues],
                "pairs": len(out.pairs),
                "directional": directional,
                "anchors": len(out.anchors),
                "xfail_events": [
                    {"fixture": e.fixture, "check": e.check, "added": e.added}
                    for e in events
                ],
                "verdict": pr_verdict,
                "verdict_scope": "pr" if pr_verdict else None,
                "fixture_verdicts": fixture_verdicts or None,
                "drops": dict(out.drops),
            }
        ],
    )


def record_late_verdict(
    number: int, *, fixture_verdicts: dict[str, str], pr_verdict: str | None
) -> None:
    """Attach a verdict to an already-captured PR, without rewriting its rows.

    Phase 2 applies these over the pair rows they name; the pair rows themselves
    are never edited, so the ledger stays append-only.
    """
    captured = {row["fixture"] for row in read_jsonl(PAIRS) if row.get("pr") == number}
    if not captured:
        raise CaptureError(
            f"PR #{number} has no captured pairs, so there is nothing to verdict. "
            "Capture it first."
        )
    check_fixture_verdicts(set(fixture_verdicts), captured)
    rows = [
        {
            "kind": "verdict",
            "pr": number,
            "fixture": fixture,
            "verdict": verdict,
            "verdict_scope": "fixture",
        }
        for fixture, verdict in sorted(fixture_verdicts.items())
    ]
    if pr_verdict:
        rows.append(
            {
                "kind": "verdict",
                "pr": number,
                "fixture": None,
                "verdict": pr_verdict,
                "verdict_scope": "pr",
            }
        )
    append_jsonl(LOG, rows)
    print(f"#{number}: recorded {len(rows)} verdict rows")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prs", nargs="*", type=int, help="merged PR numbers to capture")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="capture every merged PR the ledger has not examined yet",
    )
    ap.add_argument(
        "--since",
        type=int,
        help="sweep from this PR number exclusive (default: the highest already "
        "examined; required when the ledger is empty)",
    )
    ap.add_argument("--limit", type=int, default=100, help="PRs a sweep may consider")
    ap.add_argument(
        "--pr-verdict",
        choices=sorted(VERDICT_LABEL),
        help="set-level sign-off: no changed render was blocking. Never sets a "
        "per-render label",
    )
    ap.add_argument(
        "--fixture-verdict",
        action="append",
        default=[],
        metavar="FIXTURE=VERDICT",
        help="per-render judgement, repeatable. Errors if the merge did not move "
        "that render",
    )
    ap.add_argument(
        "--verdict-only",
        action="store_true",
        help="record a verdict against an already-captured PR, replaying nothing",
    )
    ap.add_argument("--dry-run", action="store_true", help="report, append nothing")
    ap.add_argument(
        "--worktree",
        type=Path,
        default=REPO_ROOT.parent / f"{REPO_ROOT.name}-capture",
        help="scratch worktree used to check out the two revisions",
    )
    ap.add_argument(
        "--keep-worktree",
        action="store_true",
        help="do not remove the scratch worktree",
    )
    args = ap.parse_args(argv)

    fixture_verdicts = parse_fixture_verdicts(args.fixture_verdict)
    if args.pr_verdict == "detrimental":
        raise CaptureError(
            "a merged PR cannot carry a set-level 'detrimental' verdict: if a "
            "changed render was detrimental, name it with --fixture-verdict so "
            "the negative attaches to the render it is about."
        )
    if args.sweep and (fixture_verdicts or args.pr_verdict):
        raise CaptureError(
            "a sweep spans many PRs, so one verdict cannot apply to it. Sweep to "
            "capture the mechanical rows, then attach verdicts per PR with "
            "--verdict-only."
        )
    if args.sweep and args.prs:
        raise CaptureError("--sweep takes no PR numbers")
    if args.verdict_only:
        if len(args.prs) != 1 or not (fixture_verdicts or args.pr_verdict):
            raise CaptureError("--verdict-only needs one PR number and a verdict")
        record_late_verdict(
            args.prs[0], fixture_verdicts=fixture_verdicts, pr_verdict=args.pr_verdict
        )
        return

    examined = examined_prs()
    if args.sweep:
        since = args.since if args.since is not None else max(examined, default=None)
        if since is None:
            raise CaptureError(
                "the ledger is empty, so a sweep has no starting point. Pass "
                "--since <PR> to bound it; project history before that is #1583's "
                "backfill, not this script's job."
            )
        targets = [n for n in merged_prs_since(since, args.limit) if n not in examined]
        print(f"sweep: {len(targets)} merged PRs after #{since} not yet examined")
    else:
        if not args.prs:
            raise CaptureError("give one or more PR numbers, or --sweep")
        already = [n for n in args.prs if n in examined]
        if already:
            raise CaptureError(
                f"already examined: {already}. The ledger is append-only; use "
                "--verdict-only to attach a verdict to a captured PR."
            )
        targets = args.prs

    if not targets:
        return
    worktree = ensure_worktree(args.worktree)
    try:
        for number in targets:
            try:
                capture_one(
                    number,
                    worktree=worktree,
                    fixture_verdicts=fixture_verdicts,
                    pr_verdict=args.pr_verdict,
                    dry_run=args.dry_run,
                )
            except (CaptureError, GeometryError) as exc:
                if not args.sweep:
                    raise
                print(f"#{number}: skipped -- {exc}", file=sys.stderr)
                # An unreachable commit can never be measured, so the sweep
                # settles it; anything else may just have failed this once and
                # is left for the next sweep to retry.
                if isinstance(exc, MissingRevision) and not args.dry_run:
                    append_jsonl(LOG, [no_capture_row(number, str(exc))])
    finally:
        if not args.keep_worktree:
            remove_worktree(worktree)


if __name__ == "__main__":
    try:
        main()
    except (CaptureError, GeometryError) as exc:
        sys.exit(f"error: {exc}")
