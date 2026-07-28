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
| `iter2_weights.json`        | -       | the fitted weights the render-diff reads (see below)               |
| `feature_parity_report.txt` | -       | render-diff vs `extract_features.py` feature-source parity check   |
| `scripts/`                  | 14      | the generating pipeline, plus `fit_objective.py`                   |

The tables above are the historical backfill, which is a one-off.
[Forward capture](#forward-capture) is how the same signal keeps arriving, into
`forward_pairs.jsonl`, `forward_anchors.jsonl` and `forward_log.jsonl`.

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
python scripts/build_dataset.py --from-pairs   # signal check alone, no geometry needed
```

`--from-pairs` re-prints the directional signal check from the committed
`dataset_pairs.jsonl`, which is what the bottom section of `dataset_report.txt`
is. It exists so the report's percentages can be recomputed without the 30-minute
replay, since the vectors they are derived from are committed inline.

`replay.py` shards across throwaway worktrees; 971 revisions take roughly 30
minutes on 6 shards. The per-revision geometry vectors it produces (~15MB
compressed, ~104MB raw) are deliberately **not** committed: they are
intermediate, they would add a fresh binary blob on every feature iteration, and
`dataset_pairs.jsonl` already carries the vectors a model needs inline.

`scripts/pair_rules.py` holds the emission rules the join applies and
`scripts/revisions.py` the per-revision measurement. Forward capture goes
through both, so the two corpora cannot drift apart.

## Forward capture

Everything above reconstructs the past. `scripts/capture_pr.py` records the same
signal as it is produced, so no future phase has to reconstruct anything.

| file                    | what it is                                                     |
| ----------------------- | -------------------------------------------------------------- |
| `forward_pairs.jsonl`   | pairs, in `dataset_pairs.jsonl`'s schema                       |
| `forward_anchors.jsonl` | one-sided negatives, in `dataset_anchors.jsonl`'s schema       |
| `forward_log.jsonl`     | one row per PR examined, plus verdicts recorded after the fact |

Phase 2 reads `dataset_pairs.jsonl` and `forward_pairs.jsonl` together. Both are
emitted by `pair_rules.py` over the same 31 features, so concatenating them needs
no join, no key mapping and no reconstruction step.

### Why this is not captured in CI

The render-diff already computes a per-fixture before/after vector, so persisting
_that_ looks like the cheaper answer. Three properties rule it out:

- `.github/workflows/pr-cleanup.yml` deletes `_pr/<N>/` when the PR closes, which
  is the moment the verdict becomes final. That is why only a handful of preview
  directories survive on `gh-pages` at any time.
- Its "before" is the PR's base SHA, which is the pairing measured as wrong for
  this repo (see [Two constraints](#two-constraints-that-were-established-by-measurement)).
- It reports the ten display metrics of `tests/layout_metrics.py`, not the 31
  features the fit consumes, so a consumer would still have to reconstruct.

### Back-fillability is what makes a local script sound

Capture replays commits, and commits persist. A PR missed today can be captured
next month with a byte-identical result, so nothing depends on remembering at the
right moment. CI capture has the opposite property: miss the window and pruning
destroys the data permanently.

`--sweep` turns that into a self-healing chore. It finds every merged PR the log
has not examined yet and captures them in one pass:

```bash
python scripts/capture_pr.py --sweep
```

Consecutive merges to `main` share a commit -- one merge's `mergeCommit` is the
next one's `mergeCommit^1` -- so a sweep of _k_ PRs costs about _k + 1_
extractions rather than _2k_. One extraction is ~11s for the whole 327-fixture
corpus.

A PR that yields nothing (its diff never reaches the engine, or its commits were
made unreachable by the 2026-07-27 history rewrite) still gets a `no_capture` row
naming the reason, so the log records what was considered rather than only what
succeeded, and a sweep never re-examines it.

### What needs a human, and what does not

| recorded                                      | how             |
| --------------------------------------------- | --------------- |
| `mergeCommit^1` / `mergeCommit` pairing       | mechanical      |
| which fixtures actually moved geometry        | mechanical      |
| `.mmd` content hash on both sides per fixture | mechanical      |
| `_XFAIL_*` entries added or removed, by check | mechanical      |
| abort transitions                             | mechanical      |
| the verdict, and its **scope**                | one human field |

### Scope is carried by the flag, not by a convention

```bash
python scripts/capture_pr.py 1606 --pr-verdict neutral
python scripts/capture_pr.py 1606 --fixture-verdict fan_out_wrap=improvement
```

`--pr-verdict` records a set-level ratification and **never** sets a per-render
label. `--fixture-verdict` asserts something about one named render, and naming a
render the merge did not move is a hard error: a per-render verdict on an
unchanged render is a set-level ratification in disguise, and mixing the two cost
the #1586 fit 5.2 points. A merged PR cannot carry a set-level `detrimental`
verdict either -- name the render, or the negative has nothing to attach to.

A verdict formed after capture is appended to `forward_log.jsonl` as a `verdict`
row with `--verdict-only`, which Phase 2 applies over the pair rows it names.

### Append-only

Pair rows carry their feature vectors inline at roughly 2.2KB each, so the
ledgers are only ever appended to: a rewrite would churn the whole file for every
capture, while appending keeps each diff proportional to the new work. Nothing in
the pipeline edits a row that has been written, which is also why a late verdict
is a new row rather than an edit.

## Known gap

27 of the 431 referenced SHAs no longer resolve, affecting 238 pairs (47 of them
directional). Those commits were replaced by the 2026-07-27 history rewrite and
predate the available commit map. Training is unaffected because feature vectors
are stored inline; only the provenance link is dangling for those rows. Anchors
are unaffected.

## Headline finding

Every percentage here is fixture-grouped: each fixture gets one vote, so a map
that appears in many pairs cannot state a corpus-wide trend on its own.

`crossings` shows **no directional signal** (44.9% agreement over 42 pairs and 25
fixtures, the largest sample in the corpus), which is why it sits in the lowest
weight bin of `scripts/optimize_layout.py`'s objective.
`detour_mean` (53.4%, n=148) and `detour_max` (54.1%) are flat too. What carries
signal is bend and corner quality: `bends_per_route` and `corners_total` 94.8%,
`lone_diagonals` 93.8%, `non_45_segments` 83.3%, `turn_angle_per_route` 75.4%.

The authored weights load onto features that do not discriminate and omit those
that do, which is a measured explanation for the "naive minimisation makes
layouts worse" result in the auto-layout investigation.

Grouping is not a uniform haircut on the raw per-pair numbers, which
`dataset_report.txt` prints alongside: it sharpens the bend family (89.5% ->
94.8%), pushes `crossings` from just above chance to just below, and drops
`bbox_w` out of discriminative range altogether. These are still univariate
counts over correlated pairs, not a fit. Fixture-grouped cross-validation in
#1586 is what decides viability.

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

- **`marker_crowding` is inert on this corpus.** It has zero delta on 189 of 192
  directional pairs: `min_marker_gap` is undefined on 25 of them, and unchanged
  or beyond one lane pitch on the rest. Three moved pairs cannot measure a term,
  so it is weighted as unmeasured rather than as measured-and-agreeing. Re-binning
  it left every score in the table above unchanged to the decimal, which is what
  "inert" means quantitatively.
- **Raw per-pair percentages are inflated by fixture repetition.**
  `path_len_per_route` reads 67% directional per pair but 59% grouped by fixture,
  and the `only_path_len` control _loses_ to the authored objective head-to-head
  (46.3%) at 55.3% under cross-validation. Triage features fixture-grouped;
  `dataset_report.txt` prints both.
- **The bend family is precise but narrow.** `only_bend_family` is right on 82.8%
  of pairs, but only speaks to 15.1% of them. It corroborates the bend/corner
  headline while showing why those three terms cannot carry an objective alone.
- **`crossings` earns a fitted weight of ~0** in every arm that includes it,
  matching its 44.9% grouped agreement, so it is authored at the floor. The
  fitted sign is slightly negative and must not be read as a reward: minimising a
  negative weight would instruct a search to _add_ crossings.
- **The weak-label warning is confirmed empirically.** Adding `pr_signoff` rows
  to training at a tenth of a directional row's weight costs `iter2` 5.2 points.
  Set-level ratifications are not per-render positives.

## Reporting the learned score in the render-diff

`iter2_weights.json` is the single owned statement of the fitted weights,
written by:

```bash
python scripts/fit_objective.py --dump-weights iter2 --weights-out ../iter2_weights.json
```

`scripts/scored_objective.py` is the only reader: it loads that artifact and
turns a base/PR feature-vector pair into the weighted delta the render-diff
prints in its "Learned score (shadow)" column. Nothing else holds a copy of
the weights, so a re-fit (as the forward capture ledger grows) never leaves a
second copy to drift out of sync.

**Shadow mode, not a gate.** The render-diff reports the delta and orders
changed renders by it (worst first), but nothing is filtered or hidden: every
changed render still needs a human look regardless of what the score says.
That is what makes ordering by it safe at 68.5%/69.6% agreement rather than
needing the 85% bar #1587 and #1588/#1589 reserve for an actual optimisation
target -- a wrong order costs a render being read third instead of first, in a
set that averages a handful of renders per PR; nothing is ever skipped.

### Feature-source parity

The render-diff scores a render's actual _drawn_ geometry (`drawn_polylines`,
offsets applied); `extract_features.py` -- what produced the fitted weights'
training data -- resolves its own routing pass (`route_edges_centred`) instead.
The two are meant to measure the same thing by a different decomposition, so
`scripts/check_feature_parity.py` measures the gap between them across the
whole fixture corpus rather than assuming it:

```bash
python scripts/check_feature_parity.py --out ../feature_parity_report.txt
```

The check found one real bug rather than only decomposition noise:
`extract_features.py`'s bend/turn-angle scan read a route's raw waypoints
without dropping consecutive duplicates, so a route with an incidental
zero-length step (invisible in the render) counted as a corner. Fixed via
`dedupe_consecutive`; afterwards the two heaviest-weighted features
(`lone_diagonals_per_route` at 12.3, `bends_per_route` at 3.0 -- together most
of the objective's weight mass) match exactly across all 319 valid corpus
fixtures, and the worst-case weighted-score divergence dropped from 3.06 to
1.42 (mean 0.10). `tests/test_learned_score_parity.py` locks that bound so a
future engine change that reopens the gap fails loudly instead of quietly
decalibrating the reported score.

### Shadow-mode disagreement log

```bash
python scripts/shadow_log.py --pairs ../forward_pairs.jsonl --out ../shadow_disagreements.jsonl
```

Replays iter2 over `forward_pairs.jsonl` -- PRs merged and verdicted through
[forward capture](#forward-capture), which post-date the fit -- and writes
every directional pair where the learned score's predicted direction
contradicts the recorded human verdict. `dataset_pairs.jsonl` is deliberately
**not** the input here: it is the fit's own training data, and cross-validation
over it already reports iter2's residuals, so replaying it again would add no
independent evidence.

`forward_pairs.jsonl` starts empty: forward capture only landed in #1584/PR
#1607, and a directional pair needs a human `--fixture-verdict` recorded after
capture, which nothing has received yet as of this writing. The log has
nothing to show until real PRs accumulate one; that reflects how new the
capture pipeline is, not a claim the score has been validated against fresh
data. Its output feeds the next feature iteration for #1585.
