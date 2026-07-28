"""Assemble every recoverable render preference label from project history.

Emits one row per (fixture, sha_before, sha_after) preference claim, tagged with
its source and confidence, plus the union of SHAs the geometry replay must
visit. Nothing here renders anything; it only decides what is worth rendering.

Sources, strongest first:

  A  issue_fix   A COMPLETED layout/routing/render/bug issue names fixture F and
                 was closed by merged PR P. F at P.base is DETRIMENTAL; F at
                 P.merge is IMPROVEMENT. Per-fixture and precisely attributed.
  B  pr_signoff  A merged PR touching the engine. Every render it changed was
                 judged non-blocking. Set-level weak positive.
  C  transcript  A human sign-off recovered from session transcripts. Upgrades
                 the confidence of the matching B row.
  D  pr_rejected A closed-unmerged PR touching the engine: candidate rejected
                 geometry, but closure reason is not machine-readable, so these
                 are emitted for triage rather than trusted.
  E  open_bug    An OPEN layout/routing bug naming fixture F: F at HEAD is
                 DETRIMENTAL. No "after" side exists yet, so it is a one-sided
                 anchor rather than a pair.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

S = Path(__file__).parent
REPO = Path("/Users/jonathan.manning/projects/nf-metro")
ENGINE_PATHS = ("src/nf_metro/layout", "src/nf_metro/render", "src/nf_metro/parser")
LAYOUT_LABELS = {"layout", "routing", "render", "bug"}


def stem_regex() -> re.Pattern[str]:
    stems = [
        s.strip()
        for s in (S / "stems.txt").read_text().splitlines()
        if s.strip() and s.strip() != "pipeline"
    ]
    return re.compile(
        r"(?<![\w-])("
        + "|".join(re.escape(s) for s in sorted(stems, key=len, reverse=True))
        + r")(?![\w-])",
        re.I,
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout


def touches_engine(base: str, merge: str) -> bool:
    """Whether a PR's own diff reaches the layout/render/parser code."""
    files = git("diff", "--name-only", f"{base}..{merge}")
    if not files.strip():
        return False
    return any(p in files for p in ENGINE_PATHS)


def main() -> None:
    rx = stem_regex()
    issues = json.load(open(S / "issues.json"))
    prs = {p["number"]: p for p in json.load(open(S / "prs.json"))}
    links = {
        i["number"]: [
            r["number"] for r in (i.get("closedByPullRequestsReferences") or [])
        ]
        for i in json.load(open(S / "issue_links.json"))
    }
    shas = json.load(open(S / "pr_shas.json"))
    transcript_prs = {
        r["pr_at_verdict"]
        for r in map(json.loads, open(S / "verdicts.jsonl"))
        if r.get("scope") == "pr" and r.get("pr_at_verdict")
    }

    rows: list[dict] = []
    needed: set[str] = set()
    stats: Counter = Counter()

    # --- A: issue-fix pairs -------------------------------------------------
    for iss in issues:
        if iss["stateReason"] != "COMPLETED":
            continue
        if not (LAYOUT_LABELS & {lbl["name"] for lbl in iss["labels"]}):
            continue
        fixtures = sorted({m.group(1).lower() for m in rx.finditer(iss["body"] or "")})
        if not fixtures:
            continue
        for pr_num in links.get(iss["number"], []):
            sha = shas.get(str(pr_num))
            if not sha or not sha.get("mergeCommit"):
                continue
            base, merge = sha["baseRefOid"], sha["mergeCommit"]
            if not touches_engine(base, merge):
                stats["A_skipped_non_engine"] += 1
                continue
            needed.update((base, merge))
            rows.append(
                {
                    "source": "issue_fix",
                    "issue": iss["number"],
                    "pr": pr_num,
                    "fixtures": fixtures,
                    "sha_before": base,
                    "sha_after": merge,
                    "claim": "after_better_than_before",
                    "scope": "fixture",
                    "confidence": "high" if len(fixtures) == 1 else "medium",
                    "title": iss["title"][:140],
                }
            )
            stats["A_issue_fix"] += 1

    # --- B / C / D: PR-level ----------------------------------------------
    for num, pr in prs.items():
        sha = shas.get(str(num))
        if not sha:
            continue
        merged = bool(pr["mergedAt"]) and sha.get("mergeCommit")
        if merged:
            base, after = sha["baseRefOid"], sha["mergeCommit"]
        elif pr["closedAt"] and sha.get("headRefOid"):
            base, after = sha["baseRefOid"], sha["headRefOid"]
        else:
            continue
        if not touches_engine(base, after):
            stats["B_skipped_non_engine"] += 1
            continue
        needed.update((base, after))
        if merged:
            rows.append(
                {
                    "source": "pr_signoff",
                    "pr": num,
                    "fixtures": [],
                    "sha_before": base,
                    "sha_after": after,
                    "claim": "no_change_was_blocking",
                    "scope": "pr",
                    "confidence": "high" if num in transcript_prs else "medium",
                    "transcript_confirmed": num in transcript_prs,
                    "title": pr["title"][:140],
                }
            )
            stats["B_pr_signoff"] += 1
            stats["C_transcript_confirmed"] += num in transcript_prs
        else:
            rows.append(
                {
                    "source": "pr_rejected",
                    "pr": num,
                    "fixtures": [],
                    "sha_before": base,
                    "sha_after": after,
                    "claim": "candidate_rejected_needs_triage",
                    "scope": "pr",
                    "confidence": "triage",
                    "title": pr["title"][:140],
                }
            )
            stats["D_pr_rejected"] += 1

    # --- E: open bugs as one-sided negatives at HEAD ----------------------
    head = git("rev-parse", "HEAD").strip()
    for iss in issues:
        if iss["state"] != "OPEN":
            continue
        if not (LAYOUT_LABELS & {lbl["name"] for lbl in iss["labels"]}):
            continue
        fixtures = sorted({m.group(1).lower() for m in rx.finditer(iss["body"] or "")})
        if not fixtures:
            continue
        needed.add(head)
        rows.append(
            {
                "source": "open_bug",
                "issue": iss["number"],
                "fixtures": fixtures,
                "sha_before": head,
                "sha_after": None,
                "claim": "before_is_defective",
                "scope": "fixture",
                "confidence": "high" if len(fixtures) == 1 else "medium",
                "title": iss["title"][:140],
            }
        )
        stats["E_open_bug"] += 1

    (S / "labels.json").write_text(json.dumps(rows, indent=1))
    (S / "shas_needed.txt").write_text("\n".join(sorted(needed)))

    print("=== label sources ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")
    print(f"\nrows: {len(rows)}")
    print(f"distinct SHAs to render: {len(needed)}")
    by_src = Counter(r["source"] for r in rows)
    print("by source:", dict(by_src))
    fix_rows = [r for r in rows if r["scope"] == "fixture"]
    print(
        f"fixture-scope rows: {len(fix_rows)}  "
        f"distinct fixtures: {len({f for r in fix_rows for f in r['fixtures']})}"
    )
    per = defaultdict(int)
    for r in fix_rows:
        for f in r["fixtures"]:
            per[f] += 1
    print("most-labelled fixtures:", sorted(per.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()
