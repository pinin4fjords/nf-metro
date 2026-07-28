# Layout preference dataset (frozen record)

Evidence from a closed programme. Nothing here is live and nothing consumes it.
It is kept only so the conclusions it supports can be checked rather than taken
on trust.

The programme asked whether the layout quality objective could be fitted from
the project's own review history instead of written by hand. #881 holds the
question and the answer. In short:

- **Fitting the weights adds nothing.** A refit on the objective's own features
  produced no improvement at all: 22 wins, 22 losses, p=1.0. 124 of 192
  directional pairs move none of those features, and 40 of the remaining 68 move
  exactly one, where no weight can change which side wins. The objective was
  never mis-weighted; it was silent on two thirds of the evidence.
- **Widening and normalising the features adds a lot**, 13.8 points held-out
  over the hand-binned weights. Those four features now ship in
  `tests/layout_metrics.py` and appear on every render-diff.
- **A safe objective exists and adds nothing over what already ships.** 54.4%
  against the incumbent's 54.7%, and a three-feature bend control with no fit at
  all beat it on precision.
- **More data of this kind would not change that.** Every row is a defect
  repair, and repairs buy space, so the features that move often point the wrong
  way while the features pointing the right way barely move. Ranking generated
  alternatives needs a different distribution, which capturing fixes does not
  produce at any volume.

## Files

| file                    | what it is                                                                   |
| ----------------------- | ---------------------------------------------------------------------------- |
| `dataset_pairs.jsonl`   | 1585 preference pairs, 192 directional, feature vectors inline on both sides |
| `dataset_anchors.jsonl` | 305 one-sided negatives, 277 naming the violated invariant                   |
| `dataset_report.txt`    | per-feature movement and agreement across the pairs                          |
| `fit_report.txt`        | the fitted arms, their weights, and the gate result                          |
| `safety_report.txt`     | the constrained arms, and the growth demonstration through the real engine   |

## What was removed

The tooling that built and consumed this data is gone: the feature extractor,
the historical replay, the label assembly, the fitter, the safety prober, the
render-diff score column, the forward-capture ledger and the prose miners. None
had a live consumer once the programme closed, and a rebuild path for a dataset
whose own conclusion is "this shape of data cannot answer the question" is cruft.

It is all in git history. `git log --diff-filter=D -- datasets/layout_preferences/scripts`
finds the deletion, and the commit before it holds the working state.

Two things survived into live code because they earned it: the four
discriminative metrics in `tests/layout_metrics.py`, and the measured weights in
`scripts/optimize_layout.py`.

## One standing conclusion

The bar for any future attempt is **safety, not accuracy**. A wrong ranking is
cheap, because a human reviews every render and rejects a poor candidate. An
objective that can be gamed without limit is not, because every candidate it
offers degenerates: the fitted weights put negative coefficients on drawing
height and path length, so minimising that score inflates the drawing without
bound.

Do not inherit an accuracy threshold from earlier drafts of #1586. The 85%
figure was justified by a cost that does not exist under human review.
