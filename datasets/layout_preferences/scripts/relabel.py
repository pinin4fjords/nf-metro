"""Rebuild PR-derived labels with correct commit pairing and attribution.

Two corrections over the first pass, both established by measurement:

**Pairing.** The "before" side is ``mergeCommit^1``, not ``baseRefOid``. This
project stacks PRs on chained base branches, so a PR's base branch frequently
already contains its parent PR's work; measured against ``baseRefOid``, 50 of
94 fix PRs showed zero geometry change anywhere in the corpus. ``mergeCommit^1``
is the target-branch tip immediately before the merge, so the pair is a true
main-to-main delta.

**Attribution.** A fixture named in an issue body is usually the motivating
real-world pipeline map, while the fix PR moves synthetic topology fixtures. The
two vocabularies rarely intersect, so prose naming cannot carry the label. The
issue link supplies the DIRECTION ("something was defective before this
merged") and the geometry supplies the SUBJECT (whichever fixtures actually
moved). Where a prose-named fixture does turn out to be one that moved, that row
is promoted to a higher confidence tier.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from revisions import REPO_ROOT

S = Path(__file__).parent
REPO = REPO_ROOT


def rev(spec: str) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", spec],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out or None


def main() -> None:
    old = json.load(open(S / "labels.json"))
    shas = json.load(open(S / "pr_shas.json"))

    fixed: list[dict] = []
    stats: Counter = Counter()

    for row in old:
        if row["source"] not in ("issue_fix", "pr_signoff", "pr_rejected"):
            fixed.append(row)
            continue
        pr = row.get("pr")
        merge = (shas.get(str(pr)) or {}).get("mergeCommit")
        if not merge:
            # Unmerged PR: head is the only "after" that exists.
            fixed.append(row)
            stats["kept_unmerged"] += 1
            continue
        parent = rev(merge + "^1")
        if not parent:
            fixed.append(row)
            stats["no_first_parent"] += 1
            continue
        new = dict(row)
        new["sha_before"] = parent
        new["sha_after"] = merge
        new["pairing"] = "merge_first_parent"
        if row["source"] == "issue_fix":
            # Direction comes from the issue; subject comes from the geometry.
            new["fixtures_named_in_issue"] = row["fixtures"]
            new["fixtures"] = []
            new["attribution"] = "geometry_derived"
        fixed.append(new)
        stats[f"repaired_{row['source']}"] += 1

    (S / "labels.json").write_text(json.dumps(fixed, indent=1))

    need = {r["sha_before"] for r in fixed if r.get("sha_before")}
    need |= {r["sha_after"] for r in fixed if r.get("sha_after")}
    (S / "shas_needed_v2.txt").write_text("\n".join(sorted(need)))

    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")
    print(f"\nrows: {len(fixed)}   distinct SHAs referenced: {len(need)}")


if __name__ == "__main__":
    main()
