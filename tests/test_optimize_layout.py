"""Tests for the advisory auto-layout minimiser (``scripts/optimize_layout``)."""

from __future__ import annotations

import sys
from pathlib import Path

from conftest import parse_and_layout
from layout_metrics import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from optimize_layout import (  # noqa: E402
    WEIGHTS,
    inject_direction,
    marker_crowding,
    objective,
    strip_layout_directives,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _score(text: str) -> float:
    return objective(compute_metrics(parse_and_layout(text)))


def test_objective_prefers_parallel_tracks_over_a_collapsed_column() -> None:
    """The known false positive: on hlatyping, forcing the HLA Typing section to
    TB collapses two parallel processing tracks into one column and loops both
    lines back out of its foot, which reads far worse than the default LR
    arrangement.  The objective has to rank the default the better of the two,
    or every suggestion it makes is suspect.
    """
    auto = strip_layout_directives((EXAMPLES / "hlatyping.mmd").read_text())
    parallel_tracks = _score(auto)
    collapsed_column = _score(inject_direction(auto, "hla_typing", "TB"))
    assert parallel_tracks < collapsed_column


def test_the_weight_bins_keep_the_three_states_of_knowledge_apart() -> None:
    """The bins rank evidence, not plausibility.

    A metric measured to track human judgement outranks one the corpus could not
    measure, which in turn outranks one measured and found to carry no signal: a
    measured null is stronger grounds for ignoring a metric than never having
    measured it. Collapsing the last two bins together is what let route shape
    and marker crowding be weighted as if they had the same standing.
    """
    tracks_judgement = ("single_diagonals", "bends_per_route", "turn_angle_per_route")
    unmeasurable = (
        "label_strikes",
        "marker_crowding",
        "near_horizontal",
        "wasted_canvas",
    )
    measured_null = ("crossings", "excessive_gaps")
    assert set(WEIGHTS) == {*tracks_judgement, *unmeasurable, *measured_null}

    heaviest_null = max(WEIGHTS[m] for m in measured_null)
    assert min(WEIGHTS[m] for m in tracks_judgement) > heaviest_null
    assert min(WEIGHTS[m] for m in unmeasurable) > heaviest_null
    assert max(WEIGHTS[m] for m in unmeasurable) < max(
        WEIGHTS[m] for m in tracks_judgement
    ), "an unmeasured metric may not be weighted level with the best-measured one"

    assert WEIGHTS["bends_per_route"] > WEIGHTS["turn_angle_per_route"]
    assert "corners_total" not in WEIGHTS


def test_marker_crowding_penalises_tightness_without_rewarding_sprawl() -> None:
    """Clearance beyond one lane pitch is free, so no arrangement can lower the
    objective by pushing its content further apart."""
    assert marker_crowding(None) == 0.0
    assert marker_crowding(0.0) == 1.0
    assert marker_crowding(20.0) == 0.5
    assert marker_crowding(40.0) == 0.0
    assert marker_crowding(400.0) == 0.0


def test_objective_skips_a_metric_the_map_cannot_define() -> None:
    """An undefined clearance contributes nothing rather than crashing the search."""
    defined = {key: 0.0 for key in WEIGHTS}
    del defined["marker_crowding"]
    assert objective({**defined, "marker_clearance": None}) == 0.0
    assert objective({**defined, "marker_clearance": 0.0}) == WEIGHTS["marker_crowding"]
