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
| `safe_weights.json`         | -       | the constrained fit, the only weights safe to minimise             |
| `safety_report.txt`         | -       | the structural check and the growth probe behind that claim        |
| `feature_parity_report.txt` | -       | render-diff vs `extract_features.py` feature-source parity check   |
| `scripts/`                  | 16      | the generating pipeline, plus `fit_objective.py`                   |

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

The constrained arms and the safety report regenerate with:

```bash
python scripts/fit_objective.py --dump-weights safe --weights-out ../safe_weights.json
python scripts/check_objective_safety.py --out ../safety_report.txt
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
- **Secondary gate (promote to an optimisation target, #1588 / #1589): fail**, on
  the grounds below rather than on the 85% bar. A safe objective does exist and
  is committed; it adds nothing over what already ships.

## Is any of it safe to minimise?

A score is safe to descend when no input can improve it without bound. `iter2`
is not: minimising it inflates the drawing forever, which is the PR #353 failure
mode. That is the one property a human reviewing every candidate cannot
compensate for, because it degenerates _every_ candidate rather than
occasionally offering a bad one, so it is the question #1588 exists to settle.

### The safety property, and why it is structural

If every feature is non-negative and every weight is non-negative, the score is
a non-negative combination of non-negative terms: bounded below by zero, and
monotone non-decreasing in every feature. No input can drive it arbitrarily low.
That is a one-line proof, and it needs the _features_ to cooperate, which two of
them do not:

| feature          | verdict          | why no weight on it is safe                                                                       |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| `min_marker_gap` | `terms.ANTITONE` | more clearance is better and clearance is always wideable, so its useful weight is negative       |
| `aspect_log`     | `terms.SIGNED`   | `log10(w/h)`, so a negative weight buys an arbitrarily tall drawing and a positive one a wide one |

`scripts/terms.py` holds `ADMISSIBILITY`, a verdict for every feature the
extractor emits, so a feature added without being classified cannot slip into a
minimisable objective unexamined. `min_marker_gap` has a repair -
`marker_crowding`, a one-sided penalty saturating at one lane pitch - and
`aspect_log` does not, so it stays out.

Only features that are **unbounded above** actually need their weight pinned: a
negative weight on a term confined to `[0, 1]` costs a finite amount. Both arms
are fitted, so a collapse cannot be blamed on constraining more than safety
requires:

| arm        | pinned                        | pooled | decided | coverage |
| ---------- | ----------------------------- | ------ | ------- | -------- |
| `iter2`    | nothing                       | 68.5%  | 69.6%   | 94.3%    |
| `safe`     | all 8 weights                 | 54.4%  | 79.3%   | 15.1%    |
| `safe_min` | the 6 unbounded-above weights | 54.4%  | 79.3%   | 15.1%    |

`safe` and `safe_min` are identical to the decimal, so the constraint is not the
variable. Excluding the growth features outright, #1588's third option, is the
same model again: the constraint lands on the boundary at **exactly zero** for
`bbox_h`, `path_len_per_route` and `crossings`, and a weight pinned at zero
predicts identically to an absent feature.

### 1. Is it safe? Demonstrated, not asserted

`x_spacing` and `y_spacing` are ordinary knobs - a CLI flag and a `%%metro`
directive - so a uniformly inflated drawing is a _reachable_ input.
`scripts/check_objective_safety.py` lays five fixtures out at 1x, 2x, 4x and 8x
the default spacing through the real engine and rescores them:

```bash
python scripts/check_objective_safety.py --out ../safety_report.txt
```

| arm     | score fell under growth | `genomic_pipeline.mmd` 1x -> 8x |
| ------- | ----------------------- | ------------------------------- |
| `iter2` | 5 of 5 fixtures         | 8.53 -> **-13.90**              |
| `safe`  | 0 of 5 fixtures         | 18.54 -> 19.43                  |

The growth terms are linear in the multiple, so nothing about 8x ends `iter2`'s
fall. `safety_report.txt` is the committed output and
`tests/test_objective_safety.py` locks both directions - the unconstrained arm is
kept as a live counter-example, so the probe is known to be capable of catching
an unsafe objective rather than merely passing a safe one.

The constraint fixed the fold instability too, as a side effect rather than by
design: `safe` has an empty `sign_flips_across_folds`, against three for `iter2`.
A weight that cannot cross zero cannot change sign between folds.

### 2. What constraining cost: all of the reach

`iter2` loses 14.1 points pooled and 79.2 points of coverage. The mechanism is
that **reach and correct direction sit on disjoint sets of features**. Over the
192 directional pairs, for every admissible feature:

| feature                    | moves on | agrees ("more is worse") |
| -------------------------- | -------- | ------------------------ |
| `detour_mean`              | 77.6%    | 50.7%                    |
| `path_len_per_route`       | 73.4%    | **33.8%**                |
| `bbox_h`                   | 41.7%    | **26.0%**                |
| `crossings`                | 21.9%    | 48.0%                    |
| `turn_angle_per_route`     | 15.1%    | 69.6%                    |
| `bends_per_route`          | 9.9%     | 93.8%                    |
| `lone_diagonals_per_route` | 5.2%     | 87.5%                    |
| `marker_crowding`          | 1.6%     | 100.0%                   |

Every feature with reach above 20% either points the wrong way or is a coin
flip; every feature that points the right way moves on under 16% of pairs.
`iter2`'s 94.3% coverage was **entirely** the growth terms: the four bend-family
terms it shares with `safe` move on 29 of 192 pairs between them, and 15.1% is
exactly the reach of `only_bend_family`. Pinning the growth weights therefore
does not degrade the model so much as delete most of it.

**Why the reachy terms point the wrong way.** All 192 directional rows are defect
repairs - 181 issue fixes and 11 xfail-registry clearings - so the `after` side
is always a map whose defect the engine had just learned to avoid. Repairs cost
space: a curve gets its full radius of runway, a port gets its own lane, two
sections are pushed apart. So on this corpus the preferred layout is usually the
longer and taller one. That is a true statement about repairs, not a statement
that space is good, and a pairwise fit cannot tell the two apart. It is also why
this is **not** a proof that no safe objective exists: it is a measurement that
_this corpus, in this regime,_ cannot fit one.

### 3. Does it beat the incumbent? No

The incumbent is the hand-binned `optimize_layout.WEIGHTS`, since that is what
ships. Simulating what a search driven by each arm would offer over the 192 pairs

- `surfaced` = it ranks the human-preferred arrangement first, `wasted` = it
  ranks the rejected one first and costs a review, `silent` = it abstains and the
  engine's own output stands:

| arm                 | surfaced | wasted | silent | useful:wasted |
| ------------------- | -------- | ------ | ------ | ------------- |
| greedy (status quo) | 0        | 0      | 192    | -             |
| `authored`          | 43       | 25     | 124    | 1.72:1        |
| `iter2`             | 126      | 55     | 11     | 2.29:1        |
| `safe`              | 23       | 6      | 163    | 3.83:1        |
| `only_bend_family`  | 24       | 5      | 163    | **4.80:1**    |

`safe` is the most _precise_ candidate arm, and it is beaten on its own terms by
a three-feature control with no fit at all - which is what `CONTROL_SETS` exists
to detect. Its 54.4% pooled is also 0.3 points _below_ `authored` (22 wins, 22
losses, sign test p=1.0), so on the gating comparison it is indistinguishable
from the weights already in the tree while surfacing half as many improvements
(23 against 43).

`wasted` is the only column that costs anything, and the 960 ratified-neutral
`pr_signoff` rows deliberately have no equivalent. Those rows mean "no changed
render in this PR was blocking", so an arm preferring the `before` side there
would have declined a change that turned out fine - a non-improvement, not a
harm. `fit_report.txt` reports them separately for that reason: missing a real
improvement and declining an acceptable one are not the same cost, and adding
the second into a waste column would overstate the price of a cautious arm.

So: **#1588's second stop condition.** A safe objective exists, is committed, and
adds nothing over the top weight bin of the objective `optimize_layout.py`
already has. Closed on those grounds rather than for missing a threshold - and
#1589 stays shut, because there is nothing here for a search to descend that the
authored weights do not already express.

### What none of this measures

The corpus has **no candidate sets**. Every row is two arrangements of one map
produced by two engine revisions, and every number above is about ordering that
pair the way a human did. A search would instead rank many arrangements
generated at one revision, most of them unlike anything in this corpus, and
would be free to seek out whichever region of the score it likes best. A good
useful:wasted ratio is therefore evidence about ordering two known layouts and
**not** evidence that ranking generated alternatives will work. Whatever a future
phase concludes, it needs candidate-set data to conclude it from.

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
- **An absent measurement is the cleanest state, not the worst.**
  `min_marker_gap` emits `-1.0` for "no line comes near a marker it does not
  serve", which 104 of the 278 fixtures carrying a vector are in. Fed to the
  crowding formula as a gap it reads 1.025 - past the term's ceiling and above
  every real value - so a non-negative weight would penalise precisely the
  fixtures with the most room. `terms.marker_crowding` masks the sentinel to zero
  and clamps to `[0, 1]`; every consumer reads that one definition.
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
