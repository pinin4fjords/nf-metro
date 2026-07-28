# Layout preference dataset

Labelled render-quality preferences recovered from this repository's own
history, for fitting the layout objective described in #881.

The project has no machine notion of "better render", only "byte-identical to
the last approved one". Fitting one needs preference pairs, and they turned out
to be recoverable from artifacts that were never written as a dataset.

## Files

| file                        | records | what it is                                                         |
| --------------------------- | ------- | ------------------------------------------------------------------ |
| `dataset_pairs.jsonl`       | 1598    | the dataset. One preference claim per line, feature vectors inline |
| `dataset_anchors.jsonl`     | 305     | one-sided negatives: fixture F was defective at revision X         |
| `labels.json`               | 778     | assembled labels before geometry was joined on                     |
| `labels_xfail.json`         | 319     | labels derived from `_XFAIL_*` registry churn                      |
| `xfail_events_engine.json`  | 319     | raw registry churn: entry, check, commit, direction                |
| `verdict_ledger_by_pr.json` | 108     | per-PR sign-off counts recovered from session transcripts          |
| `dataset_report.txt`        | -       | emitted/dropped counts and the per-feature directional check       |
| `scripts/`                  | 7       | the generating pipeline                                            |

## Pair schema

```json
{ "kind": "preference",
  "fixture": "convergence_fold_diamond",
  "source": "issue_fix", "pr": 1538, "issue": 1542,
  "sha_before": "…", "sha_after": "…",
  "features_before": { "crossings": 3.0, … },
  "features_after":  { "crossings": 1.0, … },
  "label": "after_better", "scope": "pr_set", "confidence": "weak",
  "corroborated_by_issue_text": false, "check": null, "title": "[bug]: …" }
```

`kind` is `preference` or `abort_transition`. An abort transition (laid out at
one revision, raised at the other) involves no human judgement and has an
unambiguous direction, making it the most reliable row type in the corpus.

`label` is `after_better` (187 rows) or `after_not_worse` (1411).

## Label hierarchy

Treat these as different strengths of evidence, not interchangeable rows.

| source                              | rows | strength                                                              |
| ----------------------------------- | ---- | --------------------------------------------------------------------- |
| `xfail_cleared` / `xfail_known_bad` | 319  | strongest: the violated invariant is **named** in `check`             |
| `issue_fix`                         | 174  | directional, per-fixture                                              |
| `pr_signoff`                        | 1012 | **set-level only**: means "no changed render in this PR was blocking" |
| `pr_rejected`                       | 401  | needs triage; PR closure reason is not machine-readable               |
| `open_bug`                          | 28   | one-sided anchor at HEAD                                              |

**Weighting a `pr_signoff` row as a per-render positive is the main way a fit
can silently go wrong.** A merged PR ratifies the diff as a whole, not each
render individually.

## Construction rules

A pair is only emitted when the comparison is genuinely about the engine:

- the fixture resolves at both revisions
- its `.mmd` content hash is **identical** at both, so geometry moved because
  the engine moved and not because the map was rewritten
- both revisions measured through the same routing entrypoint
- the feature vectors actually differ

Each revision is replayed with **its own** engine and **its own** fixture
files, but a single fixed set of feature definitions, so a feature's meaning
cannot drift underneath the dataset.

## Two constraints that were established by measurement

**The "before" side is `mergeCommit^1`, not `baseRefOid`.** This project stacks
PRs on chained base branches, so a PR's base frequently already contains its
parent PR's work. Measured against `baseRefOid`, 50 of 94 fix PRs showed zero
geometry change anywhere in the corpus.

**Issue text cannot attribute a label to a fixture.** A fixture named in an
issue body is usually the motivating real-world pipeline map, while the fix PR
moves synthetic topology fixtures; the vocabularies barely intersect. The issue
link therefore supplies the _direction_ and the geometry supplies the _subject_.
Only 20 of 174 issue-fix pairs have a prose-named fixture that also moved; those
carry `corroborated_by_issue_text`.

Applying both took directional pairs from 31 to 187.

## Sources that were checked and rejected

Recorded so they are not re-attempted:

- **PR/issue comment threads**: 9 usable rows from 217 human comments (a further
  546 comments are bot output). The classifier also mislabelled the most
  consequential verdict in the history, the PR #353 solver shelve, as an
  improvement.
- **Session transcript prose**: 20 fixture-level verdicts. Human verdicts are
  anaphoric ("that one is worse") with the fixture named in the preceding
  assistant turn, and PR-level sign-offs dominate per-fixture judgements by
  roughly 30 to 1.

Structured artifacts carry the labels; prose does not. The `_XFAIL_*` registry
history alone outyields every prose channel by an order of magnitude and
attaches a named defect class to each row.

## Regenerating

```bash
python scripts/mine_verdicts.py --stems <stems> --out verdicts.jsonl
python scripts/assemble_labels.py      # GitHub history -> labels.json
python scripts/relabel.py              # mergeCommit^1 pairing + attribution
python scripts/replay.py --shard N --shards 6 --shas shas_needed.txt
python scripts/build_dataset.py        # join geometry onto labels
```

`replay.py` shards across throwaway worktrees; 971 revisions take roughly 30
minutes on 6 shards. The per-revision geometry vectors it produces (~15MB
compressed, ~104MB raw) are deliberately **not** committed: they are
intermediate, they would add a fresh binary blob on every feature iteration, and
`dataset_pairs.jsonl` already carries the vectors a model needs inline.

## Known gap

27 of the 431 referenced SHAs no longer resolve, affecting 238 pairs (47 of them
directional). Those commits were replaced by the 2026-07-27 history rewrite and
predate the available commit map. Training is unaffected because feature vectors
are stored inline; only the provenance link is dangling for those rows. Anchors
are unaffected.

## Headline finding

`crossings` shows **no directional signal** (54.8% of directional pairs, n=42),
yet it carries the second-heaviest weight in `scripts/optimize_layout.py`'s
authored objective. `ink_density` (51.8%) and `detour_mean` (58.1%) are flat
too. What does carry signal is bend and corner quality: `lone_diagonals` 90%,
`bends_per_route` and `corners_total` 89.5%, `non_45_segments` 82%,
`min_marker_gap` increasing in 85% of pairs.

The authored weights load onto features that do not discriminate and omit those
that do, which is a measured explanation for the "naive minimisation makes
layouts worse" result in the auto-layout investigation.

These are univariate counts over correlated pairs, not a fit. Fixture-grouped
cross-validation in #1586 is what decides viability.
