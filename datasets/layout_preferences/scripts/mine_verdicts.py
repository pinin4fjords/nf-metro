"""Mine render verdicts out of Claude Code transcripts into a label ledger.

Phase 0 bootstrap for the learned-objective programme: recover the
improvement/neutral/detrimental judgements already given on rendered fixtures
during past sessions, with enough provenance (session, branch, PR, timestamp)
to later join each label onto a (base_sha, head_sha, fixture) geometry pair.

Human turns are GOLD labels. Assistant self-classifications are recorded
separately and marked SILVER: the project's own record is that agents
over-claim "benign" on visual deltas, so they cannot be trusted as ground
truth without confirmation.

Usage:
    python mine_verdicts.py --stems stems.txt --out verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_DIRS = [
    Path.home() / ".claude/projects/-Users-jonathan-manning-projects-nf-metro",
    Path.home() / ".claude/projects/-Users-jonathan-manning-projects-nf-metro-plugin",
]

# Stems too generic to treat as a fixture mention.
STEM_STOPLIST = {"pipeline"}

# Verdict cues. A turn carrying both polarities classifies as "mixed" rather
# than resolving to either: praise plus a named defect is a residual report,
# not a clean preference, and only a human can split it.
NEGATIVE = re.compile(
    r"\b(?:worse|wrong|broken|breaks?|regress(?:ion|ed|es)?|detrimental|"
    r"still (?:bad|wrong|broken|kink|crossing|there|happening)|"
    r"not (?:right|fixed|good|better)|ugly|awful|terrible|nasty|"
    r"kink(?:s|ed|ing)?|crossing|overlap(?:s|ping)?|collid|clash|"
    r"dog-?leg|strike[sd]?|breeze|off-?track|doubles? back|backtrack|"
    r"too (?:close|tight|far|steep|wide)|cramped|squish(?:ed|ing)?|"
    r"misaligned|asymmetric|"
    r"lost the|no longer|spurious|overshoot|wobble|doesn'?t (?:work|look right))\b",
    re.I,
)

# Bulk sign-off on an entire render-diff. Carries a verdict but no fixture: it
# ratifies every render the PR changed, so it expands into set-level weak
# positives via that PR's changed-fixture manifest.
PR_APPROVAL = re.compile(
    r"\b(?:merge and clean up|push everything|looks good,? (?:merge|push)|lgtm|"
    r"ship it|merge it|go ahead and merge)\b",
    re.I,
)

# Turns that direct work rather than judge output. High-frequency false
# positives: they name fixtures and carry evaluative words but assert nothing.
IMPERATIVE = re.compile(
    r"^\s*(?:check|refine|figure out|how would|can you|please|file|open|"
    r"the following is|i'?d like|i want|let'?s|run |render |fix )",
    re.I,
)
POSITIVE = re.compile(
    r"\b(?:better|fixed|improve(?:d|ment|s)?|resolved|"
    r"looks? (?:good|great|right|correct)|"
    r"that'?s (?:right|it|better|good)|nice|lovely|perfect|clean(?:er)?|"
    r"much better|good now|correct now|happy with)\b",
    re.I,
)
NEUTRAL = re.compile(
    r"\b(?:neutral|benign|no (?:visual )?(?:change|difference|regression)|"
    r"byte-?identical|unchanged|acceptable|fine|no-?op|harmless|indifferent)\b",
    re.I,
)

# Turns that are about a render at all. Without one of these the fixture
# mention is probably incidental (a file path in a command, say).
VISUAL_CONTEXT = re.compile(
    r"\b(?:render|rendered|diff|preview|map|layout|svg|png|looks?|see|eyeball|"
    r"visual|image|station|line|port|section|route|edge|label|curve)\b",
    re.I,
)

# Harness-injected blocks that ride along on a human turn. A genuine verdict
# frequently arrives with a reminder attached, so the blocks are excised and the
# surrounding prose kept.
INJECTED_BLOCK = re.compile(
    r"<(system-reminder|local-command-stdout|local-command-caveat|task-notification|"
    r"command-message|command-name|command-args)>.*?</\1>",
    re.I | re.S,
)
INJECTED_OPEN = re.compile(
    r"<(?:system-reminder|local-command-stdout|command-message|command-name|"
    r"command-args|task-notification)>.*",
    re.I | re.S,
)

# How many preceding assistant turns can supply the fixture a human verdict
# refers to. Verdicts are overwhelmingly anaphoric ("that one is worse"), so
# without this window the human-tier yield collapses to near zero.
CONTEXT_WINDOW = 3


def load_stems(path: Path) -> list[str]:
    stems = [
        s.strip()
        for s in path.read_text().splitlines()
        if s.strip() and s.strip() not in STEM_STOPLIST
    ]
    # Longest-first so `fold_double_wide` wins over `fold_double`.
    return sorted(set(stems), key=len, reverse=True)


def build_stem_regex(stems: list[str]) -> re.Pattern[str]:
    return re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(s) for s in stems) + r")(?![\w-])",
        re.I,
    )


def text_of(message: object) -> str:
    """Flatten a transcript message to its human/assistant prose only."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def classify(text: str) -> str | None:
    neg, pos, neu = NEGATIVE.search(text), POSITIVE.search(text), NEUTRAL.search(text)
    if neg and pos:
        return "mixed"
    if neg:
        return "detrimental"
    if pos:
        return "improvement"
    if neu:
        return "neutral"
    return None


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]


def _pr_at(pr_links: list[tuple[str, int]], ts: str | None) -> int | None:
    """The PR under discussion at a moment: the newest link at or before it."""
    candidates = [pr for lts, pr in pr_links if not ts or lts <= ts]
    return candidates[-1] if candidates else (pr_links[0][1] if pr_links else None)


def scan_session(path: Path, stem_re: re.Pattern[str]) -> tuple[list[dict], dict]:
    """Return (records, session_meta) for one transcript."""
    meta: dict = {
        "session_id": path.stem,
        "pr_numbers": [],
        "issue_refs": [],
        "branches": [],
        "first_ts": None,
        "last_ts": None,
        "human_turns": 0,
    }
    records: list[dict] = []
    recent_context: list[set[str]] = []
    pr_links: list[tuple[str, int]] = []

    for line in path.open(errors="replace"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rtype = rec.get("type")

        if rtype == "pr-link":
            pr = rec.get("prNumber")
            if pr:
                pr_links.append((rec.get("timestamp") or "", pr))
                if pr not in meta["pr_numbers"]:
                    meta["pr_numbers"].append(pr)
            continue

        if rtype not in ("user", "assistant"):
            continue

        ts = rec.get("timestamp")
        if ts:
            meta["first_ts"] = meta["first_ts"] or ts
            meta["last_ts"] = ts
        branch = rec.get("gitBranch")
        if branch and branch not in meta["branches"]:
            meta["branches"].append(branch)

        raw = text_of(rec.get("message"))
        if not raw:
            continue

        for m in re.finditer(
            r"<command-name>/?([\w-]+)</command-name>\s*<command-args>([^<]*)</command-args>",
            raw,
        ):
            if m.group(1) in ("fix-issue", "review", "gh") and m.group(2).strip():
                ref = m.group(2).strip()
                if ref not in meta["issue_refs"]:
                    meta["issue_refs"].append(ref)

        body = INJECTED_OPEN.sub("", INJECTED_BLOCK.sub("", raw)).strip()
        if not body:
            continue

        if rtype == "assistant":
            if rec.get("isSidechain"):
                continue
            recent_context.append({m.group(1).lower() for m in stem_re.finditer(body)})
            del recent_context[:-CONTEXT_WINDOW]
            records.extend(_sentence_labels(body, stem_re, path.stem, ts, branch))
            continue

        # Human turn.
        meta["human_turns"] += 1
        if len(body) > 2000:
            continue
        is_approval = bool(PR_APPROVAL.search(body))
        if not is_approval and not VISUAL_CONTEXT.search(body):
            continue
        verdict = classify(body)
        if verdict is None:
            continue
        explicit = {m.group(1).lower() for m in stem_re.finditer(body)}

        if is_approval and not explicit:
            records.append(
                {
                    "session_id": path.stem,
                    "timestamp": ts,
                    "git_branch": branch,
                    "speaker": "human",
                    "tier": "gold",
                    "scope": "pr",
                    "verdict": "non_detrimental",
                    "fixtures": [],
                    "pr_at_verdict": _pr_at(pr_links, ts),
                    "confidence": "high",
                    "quote": body[:500],
                }
            )
            continue

        if IMPERATIVE.match(body):
            continue

        inherited: set[str] = set()
        if not explicit:
            for fixtures in reversed(recent_context):
                inherited |= fixtures
                if inherited:
                    break
        if not (explicit or inherited):
            continue
        records.append(
            {
                "session_id": path.stem,
                "timestamp": ts,
                "git_branch": branch,
                "speaker": "human",
                "tier": "gold",
                "scope": "fixture",
                "verdict": verdict,
                "fixtures": sorted(explicit or inherited),
                "fixture_source": "explicit" if explicit else "context",
                "pr_at_verdict": _pr_at(pr_links, ts),
                "confidence": "high"
                if explicit and len(explicit) == 1
                else "needs_review",
                "quote": body[:500],
            }
        )

    return records, meta


def _sentence_labels(
    body: str,
    stem_re: re.Pattern[str],
    session: str,
    ts: str | None,
    branch: str | None,
) -> list[dict]:
    """Assistant self-classifications, scoped to single sentences for precision."""
    out = []
    for sent in sentences(body):
        if len(sent) > 600:
            continue
        hits = {m.group(1).lower() for m in stem_re.finditer(sent)}
        if not hits or not VISUAL_CONTEXT.search(sent):
            continue
        verdict = classify(sent)
        if verdict is None:
            continue
        out.append(
            {
                "session_id": session,
                "timestamp": ts,
                "git_branch": branch,
                "speaker": "assistant",
                "tier": "silver",
                "scope": "fixture",
                "verdict": verdict,
                "fixtures": sorted(hits),
                "fixture_source": "explicit",
                "confidence": "needs_review",
                "quote": sent.strip()[:400],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stem_re = build_stem_regex(load_stems(args.stems))

    all_records: list[dict] = []
    sessions: list[dict] = []
    files = sorted(f for d in PROJECT_DIRS if d.is_dir() for f in d.glob("*.jsonl"))

    for i, path in enumerate(files, 1):
        recs, meta = scan_session(path, stem_re)
        meta["labels_found"] = len(recs)
        sessions.append(meta)
        # Stamp session-level join keys onto each label.
        for r in recs:
            r["pr_numbers"] = meta["pr_numbers"]
            r["issue_refs"] = meta["issue_refs"]
        all_records.extend(recs)
        if i % 25 == 0:
            print(
                f"  {i}/{len(files)} transcripts, {len(all_records)} labels", flush=True
            )

    all_records.sort(key=lambda r: (r["timestamp"] or "", r["session_id"]))
    with args.out.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    args.out.with_suffix(".sessions.json").write_text(json.dumps(sessions, indent=1))

    gold = [r for r in all_records if r["tier"] == "gold"]
    pr_level = [r for r in gold if r["scope"] == "pr"]
    fixture_level = [r for r in gold if r["scope"] == "fixture"]

    print(
        f"\n=== {len(all_records)} candidate labels from {len(files)} transcripts ==="
    )
    print(
        f"gold, PR-scope sign-offs:  {len(pr_level)}  "
        f"over {len({r['pr_at_verdict'] for r in pr_level if r['pr_at_verdict']})} PRs"
    )
    print(f"gold, fixture-scope:       {len(fixture_level)}")
    print(f"silver (assistant):        {len(all_records) - len(gold)}")
    print(
        "\nfixture-scope verdict mix:",
        dict(Counter(r["verdict"] for r in fixture_level)),
    )
    print("attribution:", dict(Counter(r["fixture_source"] for r in fixture_level)))
    print(
        "distinct fixtures named:",
        len({f for r in fixture_level for f in r["fixtures"]}),
    )
    print("sessions with a PR link:", sum(1 for s in sessions if s["pr_numbers"]))


if __name__ == "__main__":
    main()
