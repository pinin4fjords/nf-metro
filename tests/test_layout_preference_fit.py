"""Lock the layout-objective gate verdict against the committed preference dataset.

The verdict written up under "Fitted model and gate result" in
`datasets/layout_preferences/README.md` rests on numbers, and the dataset it
rests on is regenerable. These assertions fail if a
regeneration moves any of the four claims the verdict is built from, so the
write-up cannot quietly drift out of agreement with the data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "layout_preferences"


def _load_fitter():
    """Import the fit script by path; `datasets/` is not an importable package."""
    path = DATASET_DIR / "scripts" / "fit_objective.py"
    spec = importlib.util.spec_from_file_location("fit_objective", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms():
    fitter = _load_fitter()
    directional, weak, _ = fitter.load(DATASET_DIR / "dataset_pairs.jsonl")
    assert len(directional) == 192, "dataset changed shape; re-read the verdict"
    return fitter.cross_validate(directional, weak, weak_weight=0.0)


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
