#!/usr/bin/env python3
"""Shadow-mode disagreement log: iter2's score against forward-capture verdicts.

Replays ``fit_objective.py``'s own pair classification over
``forward_pairs.jsonl`` -- PRs merged and verdicted through #1584's capture
pipeline, which post-date the iter2 fit. ``dataset_pairs.jsonl`` (the fit's own
training data) cannot supply independent evidence about it: cross-validation
over that corpus already reports iter2's residuals there, so replaying it again
would only restate a result already measured, not test the score against
anything it has not seen.

**forward_pairs.jsonl starts empty.** Forward capture (#1584) only landed in
PR #1607, and a directional pair needs a human ``--fixture-verdict`` recorded
after capture, which nothing has received yet. This script has nothing to log
until real PRs accumulate one -- that is expected, not a bug, and is not
evidence the score works or doesn't.

Usage:
    python shadow_log.py [--pairs FILE] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORWARD_PAIRS = HERE.parent / "forward_pairs.jsonl"
DISAGREEMENT_LOG = HERE.parent / "shadow_disagreements.jsonl"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_objective  # noqa: E402
import scored_objective  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=FORWARD_PAIRS)
    ap.add_argument("--out", type=Path, default=DISAGREEMENT_LOG)
    args = ap.parse_args()

    if not args.pairs.exists():
        print(
            f"{args.pairs} does not exist yet -- forward capture has produced "
            "no verdicted directional pairs. Nothing to log."
        )
        return

    directional, _weak, _skipped = fit_objective.load(args.pairs)
    weights = scored_objective.load_weights()["weights"]

    agree = disagree = abstain = 0
    disagreements: list[dict] = []
    for pair in directional:
        delta = sum(weights.get(k, 0.0) * pair.delta.get(k, 0.0) for k in weights)
        if abs(delta) < fit_objective.TIE:
            abstain += 1
            continue
        if delta < 0:
            agree += 1
            continue
        disagree += 1
        disagreements.append(
            {
                "fixture": pair.fixture,
                "source": pair.source,
                "label": "after_better",
                "learned_score_delta": delta,
            }
        )

    with args.out.open("w") as f:
        for row in disagreements:
            f.write(json.dumps(row) + "\n")

    print(
        f"directional pairs: {len(directional)}  agree: {agree}  "
        f"disagree: {disagree}  abstain: {abstain}"
    )
    print(f"disagreement log written to {args.out} ({len(disagreements)} rows)")


if __name__ == "__main__":
    main()
