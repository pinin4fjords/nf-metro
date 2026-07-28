"""Lock the claims that rest on the committed layout-preference dataset.

The verdict written up under "Fitted model and gate result" in
`datasets/layout_preferences/README.md` rests on numbers, and the dataset it
rests on is regenerable. These assertions fail if a regeneration moves any of the
claims the verdict is built from, so the write-up cannot quietly drift out of
agreement with the data.

The same dataset also decides which weight bin each term of the advisory
objective belongs in, and how `dataset_report.txt` reports a feature's
directional signal, so those are pinned here too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "datasets" / "layout_preferences"

sys.path.insert(0, str(REPO / "scripts"))
from optimize_layout import WEIGHTS as ADVISORY_WEIGHTS  # noqa: E402

# `optimize_layout` reads four terms off validator verdict counts that the
# dataset measures geometrically instead, so a name-for-name comparison of the
# two weight tables needs the translation. `fit_objective.AUTHORED_WEIGHTS`
# documents the pairing.
ADVISORY_TO_DATASET = {
    "single_diagonals": "lone_diagonals",
    "excessive_gaps": "lane_gap_excess",
}
# Neither the rendered canvas size nor render-time label boxes were captured by
# the replay, so these two terms have no counterpart in the pair vectors.
NOT_IN_DATASET = frozenset({"label_strikes", "wasted_canvas"})


def _dataset_key(advisory_term: str) -> str:
    return ADVISORY_TO_DATASET.get(advisory_term, advisory_term)


def _load_dataset_script(stem: str):
    """Import a dataset script by path; `datasets/` is not an importable package."""
    path = DATASET_DIR / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fitter():
    return _load_dataset_script("fit_objective")


@pytest.fixture(scope="module")
def dataset():
    fitter = _load_fitter()
    directional, weak, _ = fitter.load(DATASET_DIR / "dataset_pairs.jsonl")
    assert len(directional) == 192, "dataset changed shape; re-read the verdict"
    return directional, weak


@pytest.fixture(scope="module")
def directional(dataset):
    return dataset[0]


@pytest.fixture(scope="module")
def arms(dataset):
    return _load_fitter().cross_validate(*dataset, weak_weight=0.0)


def pooled(arms: dict, name: str) -> float:
    score, _ = arms[name].pooled()
    return score


def test_fitting_the_authored_features_changes_nothing(arms):
    """Refitting weights over the authored feature set is a wash.

    This is the finding the gate turns on: the authored features are too sparse
    for a weight to change any prediction, so the margin has to come from
    choosing different features instead.
    """
    margin = pooled(arms, "refit") - pooled(arms, "authored")
    assert abs(margin) < 0.01, f"refit now differs from authored by {margin:+.3f}"


def test_fitted_objective_beats_the_hand_binned_weights(arms):
    """The primary gate: a margin over `optimize_layout.WEIGHTS`."""
    margin = pooled(arms, "iter2") - pooled(arms, "authored")
    assert margin > 0.10, f"iter2's margin over authored fell to {margin:+.3f}"


def test_fitted_objective_still_misses_the_optimisation_bar(arms):
    """The secondary gate stays failed, keeping #1588 / #1589 shut.

    A failure here means agreement has climbed past the 85% bar, which is a
    reason to revisit the verdict rather than to relax this assertion.
    """
    assert pooled(arms, "iter2") < 0.85, "iter2 now clears 85%; re-gate the target"


@pytest.mark.parametrize("control", ["only_path_len", "only_bbox_h"])
def test_degenerate_controls_do_not_explain_the_margin(arms, control):
    """A single extent-like feature must not reproduce the fitted arm's margin.

    If one of these ever beat the authored objective, the margin above would be
    "fixes enlarge the drawing" wearing a fitted model's clothes.
    """
    margin = pooled(arms, control) - pooled(arms, "authored")
    assert margin < 0.02, f"control {control} now beats authored by {margin:+.3f}"


def test_the_top_weight_bin_holds_only_terms_the_corpus_measured(directional):
    """The heaviest weight is reserved for terms the corpus can actually measure.

    A term that barely moves across the pairs has no measured agreement to
    justify a weight with, however plausible it looks, so promoting one into the
    top bin has to fail here. The bar is the minimum sample the report itself
    will print a percentage for.
    """
    weights = ADVISORY_WEIGHTS
    floor = _load_dataset_script("build_dataset").MIN_MOVED
    heaviest = max(weights.values())

    for term, weight in weights.items():
        if weight < heaviest or term in NOT_IN_DATASET:
            continue
        key = _dataset_key(term)
        moved = sum(1 for p in directional if abs(p.delta.get(key, 0.0)) > 1e-6)
        assert moved >= floor, (
            f"{term} carries the heaviest weight ({weight}) but moves on only "
            f"{moved} of {len(directional)} pairs, below the {floor} the report "
            f"needs to state a percentage"
        )


def _synthetic_pair(fixture: str, before: float, after: float, key: str) -> dict:
    return {
        "fixture": fixture,
        "features_before": {key: before},
        "features_after": {key: after},
    }


def test_one_repetitive_fixture_cannot_carry_a_corpus_wide_percentage():
    """Every fixture gets one vote in the grouped figure.

    A raw percentage weights a fixture by how many pairs it contributes, so a
    single map appearing in many pairs can state a trend on its own. Here ten
    pairs of one map disagree with five other maps, and only the raw reading
    calls that a majority.
    """
    report = _load_dataset_script("build_dataset")
    pairs = [_synthetic_pair("loud", 2.0, 1.0, "f") for _ in range(10)]
    pairs += [_synthetic_pair(f"quiet_{i}", 1.0, 2.0, "f") for i in range(5)]

    (signal,) = report.directional_signal(pairs)

    assert (signal.n_pairs, signal.n_fixtures) == (15, 6)
    assert signal.raw == pytest.approx(10 / 15)
    assert signal.grouped == pytest.approx(1 / 6)


def test_an_undefined_measurement_is_not_read_as_the_tightest_one():
    """A `-1.0` gap means no foreign line came near any marker, so a pair whose
    gap is undefined at either revision says nothing about the preference. Read
    as the number -1 it would be the tightest clearance in the corpus, so a map
    that stopped crowding markers would count as having started."""
    report = _load_dataset_script("build_dataset")
    key = "min_marker_gap"
    widened = [_synthetic_pair(f"real_{i}", 30.0, 40.0, key) for i in range(8)]
    undefined = [_synthetic_pair(f"gone_{i}", 12.0, -1.0, key) for i in range(20)]

    (signal,) = report.directional_signal(widened + undefined)

    assert signal.n_pairs == 8
    assert signal.raw == 0.0


def test_both_dataset_scripts_mask_the_same_sentinel_features():
    """The two scripts are standalone by design, so the sentinel list is stated
    twice. A feature masked in one and read as the number -1 in the other would
    give the report and the fit opposite readings of the same pair."""
    assert _load_dataset_script("build_dataset").SENTINEL == _load_fitter().SENTINEL


def test_the_fit_scores_the_weights_the_advisory_objective_actually_uses():
    """`AUTHORED_WEIGHTS` is the advisory table restated over dataset feature
    names, so the arm the gate compares against has to be the live one. Left to
    drift, the fit would report a margin over weights nobody uses."""
    assert _load_fitter().AUTHORED_WEIGHTS == {
        _dataset_key(term): weight
        for term, weight in ADVISORY_WEIGHTS.items()
        if term not in NOT_IN_DATASET
    }
