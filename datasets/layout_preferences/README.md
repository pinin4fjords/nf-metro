# Layout preference dataset

Labelled render-quality preferences recovered from this repository's own
history, for fitting the layout objective described in #881.

The project has no machine notion of "better render", only "byte-identical to
the last approved one". Fitting one needs preference pairs, and they turned out
to be recoverable from artifacts that were never written as a dataset.

## Files

| file                        | records | what it is                                                         |
| --------------------------- | ------- | ------------------------------------------------------------------ |
| `dataset_pairs.jsonl`       | 1585    | the dataset. One preference claim per line, feature vectors inline |
| `dataset_anchors.jsonl`     | 305     | one-sided negatives: fixture F was defective at revision X         |
| `labels.json`               | 778     | assembled labels before geometry was joined on                     |
| `labels_xfail.json`         | 319     | labels derived from `_XFAIL_*` registry churn                      |
| `xfail_events_engine.json`  | 319     | raw registry churn: entry, check, commit, direction                |
| `verdict_ledger_by_pr.json` | 108     | per-PR sign-off counts recovered from session transcripts          |
| `dataset_report.txt`        | -       | emitted/dropped counts and the per-feature directional check       |
| `fit_report.txt`            | -       | the fitted model, its cross-validated gate result and its controls |
| `scripts/`                  | 8       | the generating pipeline, plus `fit_objective.py`                   |

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

`label` is `after_better` (192 rows) or `after_not_worse` (1393).

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

### What the identical-hash rule buys beyond isolation

Because both sides of a pair are the same map, size cannot act as a confound.
`n_stations`, `n_routes`, `n_sections` and `n_ports` have exactly zero delta in
all 192 directional pairs, so no fitted weight can be learning "large maps get
fixed" in place of a quality signal.

The same fact makes them useless in a **pairwise** design matrix: a feature with
zero delta carries no information about the preference. Eight of the 31 columns
are inert on pairs and should be dropped when fitting on them:

- `n_stations`, `n_routes`, `n_sections`, `n_ports` — zero delta by construction
- `stations_per_route`, `ports_per_section` — ratios of the above, so also fixed
- `marker_strikes`, `marker_strikes_per_station` — too sparse to move in a pair

That leaves 23 usable columns against 192 pairs, about 8 observations per
parameter. All eight stay meaningful for the one-sided anchors, which are
cross-fixture and where size genuinely varies.

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

## Fitted model and gate result

`scripts/fit_objective.py` fits a Bradley-Terry model over feature deltas and
runs the #1586 gate. `fit_report.txt` is its committed output; regenerate with:

```bash
python scripts/fit_objective.py --out fit_report.txt
```

Agreement is measured over held-out pairs under 5-fold cross-validation grouped
by whole fixture. An arm whose features all have zero delta across a pair has no
opinion about it; those abstentions are counted as half a hit in `pooled` and
excluded from `decided`, because an objective reading sparse features scores near
50% by saying nothing rather than by disagreeing.

| arm                | features                    | weights     | pooled | decided | coverage |
| ------------------ | --------------------------- | ----------- | ------ | ------- | -------- |
| `authored`         | the authored objective's    | hand-binned | 54.7%  | 63.2%   | 35.4%    |
| `refit`            | the authored objective's    | fitted      | 54.7%  | 63.2%   | 35.4%    |
| `iter1`            | discriminative, raw counts  | fitted      | 50.0%  | 50.0%   | 62.5%    |
| `iter2`            | discriminative, normalised  | fitted      | 68.5%  | 69.6%   | 94.3%    |
| `only_bend_family` | 3 bend/turn terms (control) | fitted      | 54.9%  | 82.8%   | 15.1%    |
| `only_path_len`    | `path_len_per_route` alone  | fitted      | 53.9%  | 55.3%   | 73.4%    |

**Fitting the weights is worth nothing on its own.** `refit` sees exactly the
information the authored objective sees and reproduces its accuracy to the
decimal: 22 wins, 22 losses, paired sign test p=1.0. The mechanism is in the
report's feature-movement histogram - 124 of 192 pairs move none of the authored
features, and 40 of the 68 that do move exactly one. When a single term moves,
its weight cannot change the predicted direction, so there is nothing for a fit
to improve.

**Choosing the features is worth 13.8 points.** `iter2` reaches 68.5% pooled
against `authored`'s 54.7% (98 wins, 50 losses, p=0.0001), and still wins
70.6% vs 63.2% when restricted to the 68 pairs `authored` does have an opinion
on, so the margin is not merely coverage. What separates it from the failed
`iter1` is normalisation: `lone_diagonals_per_route` and `non_45_frac` move on
nearly every pair where the same defects counted raw sit at zero.

This is the result #881 predicted and #1602 half-measured: the value is in the
features, not the weights.

### Verdict

- **Primary gate (promote to a measurement, #1587): pass.** The fitted objective
  beats the hand-binned weights by 13.8 points pooled, 7.4 on coverage-matched
  pairs, p=0.0001.
- **Secondary gate (promote to an optimisation target, #1588 / #1589): fail.**
  The absolute bar is 85% held-out agreement; `iter2` reaches 68.5% pooled and
  69.6% decided.

Two findings say the secondary bar should stay shut even if agreement later
climbs. `iter2` carries **negative** fitted weights on `bbox_h` and
`path_len_per_route`, so an optimiser minimising it would inflate the drawing
without bound - the PR #353 failure mode exactly. And four of its eight weights
flip sign between folds, so their magnitudes are artefacts of which fixtures were
held out rather than statements about layout.

### Findings that outlast the gate

- **`marker_crowding` is inert on this corpus.** It holds one of the three
  heaviest authored weights (3.0) and has zero delta on 189 of 192 directional
  pairs, because `min_marker_gap` is undefined on 164 of them and beyond one lane
  pitch on most of the rest.
- **The univariate percentages above are inflated by fixture repetition.**
  `path_len_per_route` reads 67% directional univariately, but 55.3% under
  fixture-grouped cross-validation, and the `only_path_len` control _loses_ to
  the authored objective head-to-head (46.3%). Triage features fixture-grouped,
  not univariately.
- **The bend family is precise but narrow.** `only_bend_family` is right on 82.8%
  of pairs, but only speaks to 15.1% of them. It corroborates the bend/corner
  headline while showing why those three terms cannot carry an objective alone.
- **`crossings` earns a fitted weight of ~0** in every arm that includes it,
  against the 0.5 it is authored with, matching its 54.8% univariate coin-flip.
- **The weak-label warning is confirmed empirically.** Adding `pr_signoff` rows
  to training at a tenth of a directional row's weight costs `iter2` 5.2 points.
  Set-level ratifications are not per-render positives.
