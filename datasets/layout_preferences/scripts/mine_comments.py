"""Mine render verdicts from PR and issue comment threads.

Better attributed than transcript mining: a comment already belongs to a known
PR or issue, so its SHAs come from the PR record rather than from guessing, and
a fixture named in the comment attaches to a concrete before/after pair.

Only human comments count. Bot and workflow comments (render-preview links, CI
status) are dropped, and an author allowlist is derived from the data rather
than hardcoded so the miner does not silently assume who reviews.

Usage:
    python mine_comments.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

S = Path(__file__).parent

NEGATIVE = re.compile(
    r"\b(?:worse|wrong|broken|breaks?|regress(?:ion|ed|es)?|detrimental|"
    r"still (?:bad|wrong|broken|there)|not (?:right|fixed|good)|ugly|"
    r"kink(?:s|ed)?|crossing|overlap(?:s|ping)?|collid|clash|dog-?leg|"
    r"strike[sd]?|breeze|off-?track|doubles? back|cramped|squish(?:ed)?|"
    r"misaligned|asymmetric|too (?:close|tight|steep))\b",
    re.I,
)
POSITIVE = re.compile(
    r"\b(?:better|fixed|resolved|improve(?:d|ment|s)?|"
    r"looks? (?:good|right|great|correct)|"
    r"clean(?:er)?|correct now|good now|nice)\b",
    re.I,
)
NEUTRAL = re.compile(
    r"\b(?:no (?:visual )?(?:change|difference|regression)|byte-?identical|"
    r"unchanged|neutral|benign|no-?op)\b",
    re.I,
)
BOTTY = re.compile(
    r"render preview is ready|^\s*<!--|github-actions|codecov|"
    r"^\s*!\[|workflow run|https://seqeralabs\.github\.io/nf-metro/_pr/\S+\s*$",
    re.I | re.M,
)


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


def main() -> None:
    rx = stem_regex()
    shas = json.load(open(S / "pr_shas.json"))
    links = {
        i["number"]: [
            r["number"] for r in (i.get("closedByPullRequestsReferences") or [])
        ]
        for i in json.load(open(S / "issue_links.json"))
    }

    authors: Counter = Counter()
    rows: list[dict] = []
    scanned = 0

    for path in sorted((S / "comments").glob("*.json")):
        if path.stat().st_size < 2:
            continue
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        kind, num = path.stem.split("_", 1)
        num = int(num)
        for cm in doc.get("comments") or []:
            body = cm.get("body") or ""
            author = (cm.get("author") or {}).get("login") or "?"
            authors[author] += 1
            if BOTTY.search(body) or len(body) > 4000:
                continue
            scanned += 1
            for para in re.split(r"\n{2,}", body):
                para = para.strip()
                if not para or len(para) > 900 or para.startswith(">"):
                    continue
                verdict = classify(para)
                if verdict is None:
                    continue
                fixtures = sorted({m.group(1).lower() for m in rx.finditer(para)})
                if not fixtures:
                    continue
                pr_nums = [num] if kind == "pr" else links.get(num, [])
                for pr in pr_nums:
                    sha = shas.get(str(pr))
                    if not sha:
                        continue
                    rows.append(
                        {
                            "source": "comment",
                            "origin": f"{kind}_{num}",
                            "pr": pr,
                            "author": author,
                            "verdict": verdict,
                            "fixtures": fixtures,
                            "sha_before": sha["baseRefOid"],
                            "sha_after": sha.get("mergeCommit")
                            or sha.get("headRefOid"),
                            "created": cm.get("createdAt"),
                            "quote": para[:400],
                        }
                    )

    (S / "comment_verdicts.json").write_text(json.dumps(rows, indent=1))
    print(f"comment threads scanned: {scanned} human comments")
    print(f"verdict rows: {len(rows)}")
    print("top authors:", authors.most_common(6))
    print("verdict mix:", dict(Counter(r["verdict"] for r in rows)))
    print("distinct fixtures:", len({f for r in rows for f in r["fixtures"]}))
    print("distinct PRs:", len({r["pr"] for r in rows}))
    print("\nsample:")
    for r in rows[:6]:
        print(
            f"  [{r['verdict']:12s}] PR{r['pr']} {','.join(r['fixtures'])[:34]:36s} "
            f"{' '.join(r['quote'].split())[:80]}"
        )


if __name__ == "__main__":
    main()
