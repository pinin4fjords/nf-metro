#!/usr/bin/env python3
"""Fit a pairwise layout objective and gate it against the hand-binned weights.

The question this answers is narrow: does *fitting* weights add anything beyond
reading the univariate directional measurements and binning the weights by hand,
as `scripts/optimize_layout.py` currently does?

Four candidate arms are compared on identical fixture-grouped splits:

| arm        | features                 | weights     |
| ---------- | ------------------------ | ----------- |
| `authored` | the authored objective's | hand-binned |
| `refit`    | the authored objective's | fitted      |
| `iter1`    | discriminative subset    | fitted      |
| `iter2`    | second subset            | fitted      |

`refit` is the arm that isolates the question. It sees exactly the information
the authored objective sees, so any gap between it and `authored` is
attributable to the weights alone, and any gap between it and `iter1`/`iter2`
is attributable to the feature choice.

`CONTROL_SETS` adds deliberately impoverished arms alongside these. They exist
to be beaten: a candidate whose margin a one-feature control reproduces has not
measured layout quality.

Usage:

    python scripts/fit_objective.py --out ../fit_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset_pairs.jsonl"

Y_SPACING = 40.0
"""One lane pitch, mirroring ``nf_metro.layout.constants.Y_SPACING``.

Hardcoded so this script stays importable without the package, matching the
rest of the generating pipeline.  ``marker_crowding`` is the only feature that
needs it.
"""

INERT = frozenset(
    {
        # Both sides of a pair are the same map, so size is fixed by construction.
        "n_stations",
        "n_routes",
        "n_sections",
        "n_ports",
        # Ratios of the above.
        "stations_per_route",
        "ports_per_section",
        # Too sparse to move within a pair.
        "marker_strikes",
        "marker_strikes_per_station",
    }
)

SENTINEL = frozenset({"min_marker_gap", "min_station_distance"})
"""Features whose extractor emits ``-1.0`` for "no such measurement exists".

A map with no foreign line near any marker has no minimum gap.  Treating the
sentinel as the number -1 would read as the tightest possible clearance, which
inverts the feature, so a pair is given a zero delta on that feature instead:
undefined on either side means the feature says nothing about this preference.
"""


AUTHORED_WEIGHTS = {
    "lone_diagonals": 3.0,
    "bends_per_route": 3.0,
    "marker_crowding": 2.0,
    "turn_angle_per_route": 2.0,
    "crossings": 0.25,
    "near_horizontal": 0.5,
    "lane_gap_excess": 0.25,
}
"""``optimize_layout.WEIGHTS`` expressed over the dataset's feature names.

Two of the nine authored terms have no counterpart in the pair vectors and are
dropped from **both** sides of the comparison, so neither model can use them:

- `label_strikes` (2.0) needs render-time label boxes the replay never captured
- `wasted_canvas` (0.5) needs the rendered canvas size, likewise absent

Four more are analogues rather than identities: `optimize_layout` reads
`crossings`, `near_horizontal`, `single_diagonals` and `excessive_gaps` off
validator verdict counts, while the dataset measures the same defects
geometrically.  `lane_gap_excess` is the loosest of the four -- a worst-run
magnitude in pixels standing in for a count of `excessive_column_gap` verdicts
-- so the authored arm is also scored without it, as `authored_no_gap`.
`bends_per_route`, `corners_total` and `turn_angle_per_route` share the
extractor's 5-degree turn floor with `tests/layout_metrics.py` and are direct.
"""

SCALE_MISMATCHED = "lane_gap_excess"

FEATURE_SETS = {
    # The discriminative features from `dataset_report.txt`, one per signal.
    # `corners_total` is excluded for the reason `optimize_layout` gives: it is
    # `bends_per_route` times the route count, so fitting both would re-count
    # one signal in proportion to map size.
    "iter1": [
        "lone_diagonals",
        "bends_per_route",
        "turn_angle_per_route",
        "non_45_segments",
        "min_marker_gap",
        "aspect_log",
    ],
    # Second iteration: per-route normalisations in place of raw counts, the
    # extent term the report flags as increasing under fixes, and the two
    # features the authored objective leans on that the report finds flat.
    "iter2": [
        "lone_diagonals_per_route",
        "bends_per_route",
        "turn_angle_per_route",
        "non_45_frac",
        "min_marker_gap",
        "bbox_h",
        "path_len_per_route",
        "crossings",
    ],
}

CONTROL_SETS = {
    # Controls, not candidates. Each is too impoverished to be a real objective,
    # so if one matches a candidate arm's agreement then that arm's margin is not
    # evidence about layout quality.
    #
    # `only_path_len` is the one that matters: `dataset_report.txt` has
    # `path_len_per_route` rising in 67% of directional pairs, which is on its
    # own enough to look like a fitted model succeeding.
    "only_bend_family": ["lone_diagonals", "bends_per_route", "turn_angle_per_route"],
    "only_path_len": ["path_len_per_route"],
    "only_bbox_h": ["bbox_h"],
}

FAMILY_RULES = (
    ("fan", ("fan",)),
    ("fold", ("fold", "serpentine", "wrap")),
    ("tb", ("tb_", "_tb", "to_tb")),
    ("merge", ("merge", "converg", "junction", "diamond", "sink", "collector")),
    ("port", ("port", "entry", "exit")),
)

REAL_PIPELINES = (
    "differentialabundance",
    "da_pipeline",
    "genomeassembly",
    "genomic_pipeline",
    "rnaseq",
    "riboseq",
    "variant",
    "seqinspector",
    "funcprofiler",
    "longread",
)

N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def derive(vec: dict[str, float]) -> dict[str, float | None]:
    """Copy a feature vector, masking sentinels and adding derived columns."""
    out: dict[str, float | None] = {
        k: (None if k in SENTINEL and v == -1.0 else v) for k, v in vec.items()
    }
    gap = out.get("min_marker_gap")
    # A penalty for tight clearance that never pays for loose clearance, so the
    # term cannot be minimised by spreading the map out.  Mirrors
    # `optimize_layout.marker_crowding`.
    out["marker_crowding"] = (
        None if gap is None else max(0.0, Y_SPACING - gap) / Y_SPACING
    )
    return out


class Pair:
    """One directional preference: the "after" side was judged better."""

    __slots__ = ("fixture", "source", "delta")

    def __init__(
        self,
        fixture: str,
        source: str,
        before: dict[str, float],
        after: dict[str, float],
    ) -> None:
        self.fixture = fixture
        self.source = source
        b, a = derive(before), derive(after)
        self.delta: dict[str, float] = {}
        for key in b:
            if key in INERT:
                continue
            bv, av = b[key], a[key]
            self.delta[key] = 0.0 if bv is None or av is None else av - bv

    def families(self) -> list[str]:
        name = self.fixture.lower()
        hits = [fam for fam, keys in FAMILY_RULES if any(k in name for k in keys)]
        return hits or ["unclassified"]

    def is_real_pipeline(self) -> bool:
        return any(p in self.fixture.lower() for p in REAL_PIPELINES)


def load(path: Path) -> tuple[list[Pair], list[Pair], Counter]:
    """Read the dataset into directional pairs and weak same-or-better pairs.

    Returns ``(directional, weak, skipped)``.  A row is only usable pairwise if
    both sides carry geometry; an abort transition has geometry on one side by
    definition and so cannot supply a delta.
    """
    directional: list[Pair] = []
    weak: list[Pair] = []
    skipped: Counter = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        before, after = row.get("features_before"), row.get("features_after")
        if not before or not after:
            skipped[f"{row['label']}/no_geometry/{row['kind']}"] += 1
            continue
        pair = Pair(row["fixture"], row["source"], before, after)
        if row["label"] == "after_better":
            directional.append(pair)
        elif row["source"] == "pr_signoff":
            # A set-level ratification: "no changed render in this PR was
            # blocking".  Usable only as a weak same-or-better constraint.
            weak.append(pair)
        else:
            # `pr_rejected` closure reasons are not machine-readable, so these
            # rows carry no usable direction.
            skipped[f"{row['label']}/{row['source']}"] += 1
    return directional, weak, skipped


def fold_of(pairs: list[Pair], n_folds: int) -> dict[str, int]:
    """Assign whole fixtures to folds, largest first, to balance fold sizes.

    Grouping by fixture is load-bearing: the same fixture appears on both sides
    of many pairs, so a random split over pairs would leak.
    """
    counts = Counter(p.fixture for p in pairs)
    order = sorted(counts, key=lambda f: (-counts[f], f))
    sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for fixture in order:
        target = min(range(n_folds), key=lambda i: (sizes[i], i))
        assignment[fixture] = target
        sizes[target] += counts[fixture]
    return assignment


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


TIE = 1e-12
"""Below this, an arm's score is unmoved across a pair and it has no opinion."""


def score_delta(weights: dict[str, float], pair: Pair) -> float:
    """Change in objective across a pair. Negative means "after is better"."""
    return sum(w * pair.delta.get(k, 0.0) for k, w in weights.items())


def hit(delta: float) -> float:
    """Credit for one prediction: 1 correct, 0 wrong, and half for an abstention.

    Scoring an abstention at half keeps an arm from being either rewarded or
    punished for staying silent, which matters because the arms under test differ
    enormously in how often they speak at all.
    """
    if abs(delta) < TIE:
        return 0.5
    return 1.0 if delta < 0 else 0.0


def tally(deltas: Sequence[float]) -> tuple[float, int]:
    """Mean credit over predictions, and how many of them were abstentions."""
    if not deltas:
        return float("nan"), 0
    ties = sum(1 for d in deltas if abs(d) < TIE)
    return sum(hit(d) for d in deltas) / len(deltas), ties


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small symmetric positive-definite system by Gaussian elimination."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            return [0.0] * n
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            if factor:
                for k in range(col, n + 1):
                    aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for col in reversed(range(n)):
        total = aug[col][n] - sum(aug[col][k] * out[k] for k in range(col + 1, n))
        out[col] = total / aug[col][col]
    return out


def fit(
    rows_in: list[tuple[Pair, float]],
    keys: list[str],
    *,
    l2: float = 1.0,
    iters: int = 50,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Fit a Bradley-Terry / logistic model over feature deltas.

    The model is deliberately antisymmetric: no intercept, and features are
    scaled but never centred.  Centring a delta would let the model prefer
    "after" regardless of geometry, which is exactly the bias the pairwise
    design exists to exclude.

    Ridge-penalised Newton steps, since a handful of features over a couple of
    hundred rows makes the exact Hessian cheaper than tuning a step size.
    """
    scales = []
    for key in keys:
        values = [p.delta.get(key, 0.0) for p, _ in rows_in]
        rms = math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0
        scales.append(rms if rms > TIE else 1.0)

    rows = [
        ([p.delta.get(k, 0.0) / s for k, s in zip(keys, scales)], weight)
        for p, weight in rows_in
    ]
    total = sum(w for _, w in rows) or 1.0
    n = len(keys)

    w = [0.0] * n
    for _ in range(iters):
        grad = [l2 * wj / total for wj in w]
        hess = [[(l2 / total if i == j else 0.0) for j in range(n)] for i in range(n)]
        for x, weight in rows:
            # The objective falls when -w.x > 0, so z is the logit for "better".
            z = -sum(wj * xj for wj, xj in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            g = weight * (1.0 - p) / total
            h = weight * p * (1.0 - p) / total
            for i, xi in enumerate(x):
                if not xi:
                    continue
                grad[i] += g * xi
                for j in range(i, n):
                    hess[i][j] += h * xi * x[j]
        for i in range(n):
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = _solve(hess, grad)
        w = [wj - sj for wj, sj in zip(w, step)]
        if max(abs(s) for s in step) < tol:
            break

    return {k: wj / s for k, wj, s in zip(keys, w, scales)}


def rescale_to(weights: dict[str, float], anchor: str) -> dict[str, float]:
    """Rescale so ``anchor`` reads 3.0, matching the authored weights' units.

    Only the ratios of a logistic model's weights are identified against a
    pairwise objective, because scaling them all by one positive constant
    reproduces every prediction exactly.  An anchor is therefore needed before
    they can be read against `AUTHORED_WEIGHTS` at all.
    """
    pivot = weights.get(anchor)
    if not pivot:
        return dict(weights)
    factor = 3.0 / pivot
    return {k: v * factor for k, v in weights.items()}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class Arm:
    """One objective under test, with its cross-validated predictions."""

    def __init__(self) -> None:
        self.predictions: list[tuple[Pair, float]] = []
        self.fold_scores: list[float] = []
        self.weights: dict[str, float] = {}
        self.fold_weights: list[dict[str, float]] = []

    def record(self, pairs: list[Pair], weights: dict[str, float]) -> None:
        deltas = [score_delta(weights, pair) for pair in pairs]
        self.predictions.extend(zip(pairs, deltas))
        self.fold_scores.append(tally(deltas)[0])
        self.fold_weights.append(weights)

    def pooled(self) -> tuple[float, int]:
        return tally([d for _, d in self.predictions])

    def decided(self) -> tuple[float, float]:
        """Agreement over pairs this arm has an opinion on, and its coverage.

        Pooled agreement mixes two different things: how often an arm is right,
        and how often it says anything at all.  An arm reading only sparse
        features scores near 50% because it abstains, not because it disagrees,
        so the two have to be separated before any margin can be interpreted.
        """
        if not self.predictions:
            return float("nan"), float("nan")
        decided = [d for _, d in self.predictions if abs(d) >= TIE]
        coverage = len(decided) / len(self.predictions)
        return tally(decided)[0], coverage

    def by_pair(self) -> dict[int, float]:
        return {id(p): d for p, d in self.predictions}

    def subset(self, predicate: Callable[[Pair], bool]) -> tuple[float, int]:
        rows = [d for p, d in self.predictions if predicate(p)]
        return tally(rows)[0], len(rows)

    def weight_spread(self) -> dict[str, tuple[float, float]]:
        """Per-feature min/max of the fitted weight across folds.

        A weight whose sign flips between folds is not a finding about layout,
        it is an artefact of which fixtures happened to be held out.
        """
        spread: dict[str, tuple[float, float]] = {}
        for key in self.weights:
            values = [fw.get(key, 0.0) for fw in self.fold_weights]
            if values:
                spread[key] = (min(values), max(values))
        return spread


def cross_validate(
    directional: list[Pair],
    weak: list[Pair],
    *,
    weak_weight: float = 0.0,
) -> dict[str, Arm]:
    """Run every arm over the same fixture-grouped folds."""
    folds = fold_of(directional, N_FOLDS)
    authored_keys = list(AUTHORED_WEIGHTS)
    no_gap = {k: v for k, v in AUTHORED_WEIGHTS.items() if k != SCALE_MISMATCHED}

    fitted_sets = {**FEATURE_SETS, **CONTROL_SETS}
    arms = {
        name: Arm() for name in ("authored", "authored_no_gap", "refit", *fitted_sets)
    }

    for fold in range(N_FOLDS):
        train = [p for p in directional if folds[p.fixture] != fold]
        test = [p for p in directional if folds[p.fixture] == fold]
        if not test:
            continue
        arms["authored"].record(test, AUTHORED_WEIGHTS)
        arms["authored_no_gap"].record(test, no_gap)
        train_rows = [(p, 1.0) for p in train]
        if weak_weight > 0:
            # Weak rows are grouped by the same fixture assignment, so a held-out
            # fixture stays held out in every arm that consumes them.
            train_rows += [
                (p, weak_weight) for p in weak if folds.get(p.fixture, -1) != fold
            ]
        arms["refit"].record(test, fit(train_rows, authored_keys))
        for name, keys in fitted_sets.items():
            arms[name].record(test, fit(train_rows, keys))

    # Weights for inspection come from a fit on everything; the fold weights
    # kept on each arm are what the spread check reads.
    full = [(p, 1.0) for p in directional]
    if weak_weight > 0:
        full += [(p, weak_weight) for p in weak]
    arms["authored"].weights = dict(AUTHORED_WEIGHTS)
    arms["authored_no_gap"].weights = dict(no_gap)
    arms["refit"].weights = fit(full, authored_keys)
    for name, keys in fitted_sets.items():
        arms[name].weights = fit(full, keys)
    return arms


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def pct(value: float) -> str:
    return "   n/a" if value != value else f"{value * 100:5.1f}%"


def binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value at p=0.5, for the paired sign test."""
    if trials == 0:
        return float("nan")
    tail = sum(math.comb(trials, k) for k in range(successes + 1))
    return min(1.0, 2 * tail / 2**trials)


def report(
    directional: list[Pair],
    weak: list[Pair],
    skipped: Counter,
    arms: dict[str, Arm],
    weak_arms: dict[str, Arm],
) -> str:
    out: list[str] = []
    add = out.append

    add("=== inputs ===")
    add(f"directional pairs usable pairwise: {len(directional)}")
    add(f"distinct fixtures:                 {len({p.fixture for p in directional})}")
    add(f"weak pr_signoff rows available:    {len(weak)}")
    for key, count in sorted(skipped.items()):
        add(f"  skipped {key}: {count}")
    zeroed = Counter()
    for pair in directional:
        for key in SENTINEL | {"marker_crowding"}:
            if pair.delta.get(key, 0.0) == 0.0:
                zeroed[key] += 1
    add("  zero-delta (incl. sentinel-masked) counts:")
    for key, count in sorted(zeroed.items()):
        add(f"    {key}: {count}/{len(directional)}")

    candidates = ("authored", "authored_no_gap", "refit", *FEATURE_SETS)

    add("")
    add(f"=== fixture-grouped {N_FOLDS}-fold agreement ===")
    add("pooled counts an abstention as half a hit; decided excludes abstentions")
    add(f"{'arm':<18}{'pooled':>8}{'decided':>9}{'coverage':>10}  per fold (pooled)")
    baseline_pooled, _ = arms["authored"].pooled()
    for name in (*candidates, *CONTROL_SETS):
        arm = arms[name]
        pooled, _ = arm.pooled()
        acc, coverage = arm.decided()
        folds = " ".join(pct(s) for s in arm.fold_scores)
        marker = "  [control]" if name in CONTROL_SETS else ""
        add(f"{name:<18}{pct(pooled)}{pct(acc):>9}{pct(coverage):>10}  {folds}{marker}")

    add("")
    add("=== margin over the hand-binned authored objective ===")
    for name in ("refit", *FEATURE_SETS, *CONTROL_SETS):
        pooled, _ = arms[name].pooled()
        delta = (pooled - baseline_pooled) * 100
        marker = "  [control]" if name in CONTROL_SETS else ""
        add(
            f"{name:<18}{pct(pooled)}  vs {pct(baseline_pooled)}"
            f"   {delta:+5.1f} pp{marker}"
        )

    add("")
    add("=== head-to-head where the authored objective has an opinion ===")
    add("restricted to pairs `authored` does not abstain on, so coverage cannot")
    add("flatter either side")
    ref = arms["authored"].by_pair()
    decided_ids = {i for i, d in ref.items() if abs(d) >= TIE}
    for name in (*candidates, *CONTROL_SETS):
        arm = arms[name]
        rows = [d for i, d in arm.by_pair().items() if i in decided_ids]
        marker = "  [control]" if name in CONTROL_SETS else ""
        add(f"{name:<18}{pct(tally(rows)[0])}  (n={len(rows)}){marker}")

    add("")
    add("=== paired sign test against `authored`, over all held-out pairs ===")
    add("counts only the pairs the two arms disagree on, so the shared")
    add("abstentions and shared hits cannot manufacture a margin")
    for name in ("refit", *FEATURE_SETS, *CONTROL_SETS):
        wins = losses = 0
        for i, delta in arms[name].by_pair().items():
            mine, theirs = hit(delta), hit(ref[i])
            if mine > theirs:
                wins += 1
            elif mine < theirs:
                losses += 1
        n = wins + losses
        p = binomial_two_sided(min(wins, losses), n) if n else float("nan")
        marker = "  [control]" if name in CONTROL_SETS else ""
        add(
            f"{name:<18}wins {wins:>3}  losses {losses:>3}  n={n:>3}  p={p:.4f}{marker}"
        )

    add("")
    add("=== how many features move within a single pair ===")
    add("a sparse objective abstains; when it does speak, usually one term moves,")
    add("so the weight on that term cannot change the predicted direction")
    for label, keys in (
        ("authored (7 features)", list(AUTHORED_WEIGHTS)),
        ("iter2 (8 features)", FEATURE_SETS["iter2"]),
    ):
        hist = Counter(
            sum(1 for k in keys if abs(p.delta.get(k, 0.0)) > TIE) for p in directional
        )
        spread = "  ".join(f"{n}:{hist[n]}" for n in sorted(hist))
        add(f"  {label:<24}{spread}")

    add("")
    add("=== fitted weights (rescaled so bends_per_route = 3.0) ===")
    add("positive = more of this is worse, matching the authored convention")
    for name in ("refit", *FEATURE_SETS, *CONTROL_SETS):
        arm = arms[name]
        add(f"-- {name}")
        scaled = rescale_to(arm.weights, "bends_per_route")
        spread = arm.weight_spread()
        for key, value in sorted(scaled.items(), key=lambda kv: -abs(kv[1])):
            authored = AUTHORED_WEIGHTS.get(key)
            ref = f"authored {authored:.2f}" if authored is not None else "not authored"
            lo, hi = spread.get(key, (0.0, 0.0))
            flips = " SIGN FLIPS ACROSS FOLDS" if lo * hi < 0 else ""
            add(f"   {key:<26}{value:+7.2f}   ({ref}){flips}")

    add("")
    add("=== per-family agreement (families overlap; n = held-out pairs) ===")
    families = [fam for fam, _ in FAMILY_RULES] + ["unclassified"]
    header = f"{'arm':<18}" + "".join(f"{f:>16}" for f in families)
    add(header)
    for name in ("authored", "refit", *FEATURE_SETS, *CONTROL_SETS):
        arm = arms[name]
        cells = []
        for fam in families:
            score, n = arm.subset(lambda p, fam=fam: fam in p.families())
            cells.append(f"{pct(score)} (n={n})".rjust(16))
        add(f"{name:<18}" + "".join(cells))

    add("")
    add("=== synthetic topology fixtures vs real pipeline maps ===")
    add(f"{'arm':<18}{'synthetic':>18}{'real':>18}")
    for name in ("authored", "refit", *FEATURE_SETS, *CONTROL_SETS):
        arm = arms[name]
        syn, n_syn = arm.subset(lambda p: not p.is_real_pipeline())
        real, n_real = arm.subset(lambda p: p.is_real_pipeline())
        left = f"{pct(syn)} (n={n_syn})"
        right = f"{pct(real)} (n={n_real})"
        add(f"{name:<18}{left:>18}{right:>18}")

    add("")
    add("=== ablation: weak pr_signoff rows added to training ===")
    add("set-level ratifications, weighted 0.1 against a directional row's 1.0")
    add(f"{'arm':<18}{'without':>10}{'with':>10}{'delta':>9}")
    for name in ("refit", *FEATURE_SETS):
        base, _ = arms[name].pooled()
        aug, _ = weak_arms[name].pooled()
        add(f"{name:<18}{pct(base)}{pct(aug)}{(aug - base) * 100:+8.1f} pp")

    return "\n".join(out) + "\n"


ANCHOR_FEATURE = "bends_per_route"
"""Feature the committed weights are rescaled against; see ``rescale_to``."""


def weights_artifact(arm: Arm, name: str) -> dict:
    """The single owned statement of one arm's fitted weights, for consumers.

    Rescaled to ``ANCHOR_FEATURE`` = 3.0, matching the units every number in
    this module's ``report()`` is already printed in. Only ratios between a
    Bradley-Terry model's weights are identified, so this rescaling changes no
    prediction; it exists purely so a consumer reads the same numbers a human
    reviewing ``fit_report.txt`` does.

    Carries the gate result and the sign-flip warning as data, not just prose,
    so a consumer (the render-diff scorecard) cannot quote the weights without
    also seeing why per-feature contributions are not meant to be surfaced.
    """
    pooled, _ = arm.pooled()
    decided, coverage = arm.decided()
    scaled = rescale_to(arm.weights, ANCHOR_FEATURE)
    spread = arm.weight_spread()
    flips = sorted(k for k, (lo, hi) in spread.items() if lo * hi < 0)
    return {
        "arm": name,
        "anchor_feature": ANCHOR_FEATURE,
        "anchor_value": 3.0,
        "weights": scaled,
        "gate": {"pooled": pooled, "decided": decided, "coverage": coverage},
        "sign_flips_across_folds": flips,
        "note": (
            "Fitted to predict a pairwise preference direction, not to be "
            "minimised -- see #1587/#1588/#1589. Report the aggregate "
            "weighted delta only; per-feature contributions are not "
            "meaningful when sign_flips_across_folds is non-empty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dump-weights",
        metavar="ARM",
        default=None,
        help="Write ARM's fitted weights as JSON to --weights-out instead of "
        "(or alongside) the text report, e.g. --dump-weights iter2.",
    )
    parser.add_argument("--weights-out", type=Path, default=None)
    args = parser.parse_args()

    directional, weak, skipped = load(args.pairs)
    arms = cross_validate(directional, weak, weak_weight=0.0)
    weak_arms = cross_validate(directional, weak, weak_weight=0.1)
    text = report(directional, weak, skipped, arms, weak_arms)
    print(text, end="")
    if args.out:
        args.out.write_text(text)

    if args.dump_weights:
        if args.dump_weights not in arms:
            raise SystemExit(f"unknown arm: {args.dump_weights!r}")
        artifact = weights_artifact(arms[args.dump_weights], args.dump_weights)
        payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        if args.weights_out:
            args.weights_out.write_text(payload)
        else:
            print(payload, end="")


if __name__ == "__main__":
    main()
