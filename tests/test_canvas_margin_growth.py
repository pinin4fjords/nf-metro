"""The canvas holds ink drawn outside the section-box envelope.

A route that runs left of, or above, every section box is outside the envelope
the boxes were placed with, so the margin that envelope leaves says nothing
about how much room the route has.  The canvas grows past the rightmost and
bottommost ink already; these tests hold the same for the two margins whose
edge sits at the coordinate origin, where the ink has nowhere to go and is
drawn against -- or beyond -- the canvas edge instead.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph
from nf_metro.parser.model import MetroGraph, PermissiveGuardWarning
from nf_metro.render.constants import CANVAS_PADDING
from nf_metro.render.svg import RenderPlan, build_render_plan
from nf_metro.themes import resolve_theme

ROOT = Path(__file__).parents[1]

# The two arms of the defect, plus corpus fixtures that escape the envelope on
# each side, so the invariant is exercised beyond its own reproducers.
LEFT_ESCAPING = (
    "examples/topologies/wrap_return_canvas_margin.mmd",
    "examples/topologies/inter_row_wrap_clearance.mmd",
    "examples/diagonal_labels.mmd",
)
TOP_ESCAPING = (
    "examples/topologies/lr_perp_top_exit_perp_entry.mmd",
    "examples/topologies/tb_lr_exit_left.mmd",
    "examples/topologies/cross_col_top_entry.mmd",
)


def _plan(path: Path, *, strict: bool = False) -> RenderPlan:
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    graph.strict = strict
    return build_render_plan(graph, resolve_theme(None, graph))


def _box_envelope(graph: MetroGraph) -> tuple[float, float]:
    """Top-left corner of the union of the drawn section boxes."""
    boxes = [section for section in graph.sections.values() if section.bbox_w > 0]
    return (
        min(section.bbox_x for section in boxes),
        min(section.bbox_y for section in boxes),
    )


def _drawn_ink_origin(plan: RenderPlan) -> tuple[float, float]:
    """Leftmost and topmost coordinate any route or station is drawn at."""
    points = [point for polyline in plan.route_polylines for point in polyline]
    stations = [
        station
        for station in plan.graph.stations.values()
        if not station.is_port and not station.is_hidden
    ]
    return (
        min([point[0] for point in points] + [station.x for station in stations]),
        min([point[1] for point in points] + [station.y for station in stations]),
    )


@pytest.mark.parametrize("name", LEFT_ESCAPING)
def test_ink_left_of_the_box_envelope_keeps_the_canvas_margin(name: str) -> None:
    path = ROOT / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = _plan(path)
    envelope_x, _ = _box_envelope(plan.graph)
    ink_x, _ = _drawn_ink_origin(plan)
    assert ink_x < envelope_x, f"{name} draws nothing left of its box envelope"
    assert ink_x >= CANVAS_PADDING, (
        f"{name} draws ink at x={ink_x:.1f}, inside the {CANVAS_PADDING:.0f}px "
        f"margin the canvas leaves past its rightmost ink"
    )


@pytest.mark.parametrize("name", TOP_ESCAPING)
def test_ink_above_the_box_envelope_keeps_the_canvas_margin(name: str) -> None:
    path = ROOT / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = _plan(path)
    _, envelope_y = _box_envelope(plan.graph)
    _, ink_y = _drawn_ink_origin(plan)
    assert ink_y < envelope_y, f"{name} draws nothing above its box envelope"
    assert ink_y >= CANVAS_PADDING, (
        f"{name} draws ink at y={ink_y:.1f}, inside the {CANVAS_PADDING:.0f}px "
        f"margin the canvas leaves past its bottommost ink"
    )


@pytest.mark.parametrize("name", LEFT_ESCAPING[:1] + TOP_ESCAPING[:1])
def test_a_canvas_margin_corridor_holds_its_claim_on_both_arms(name: str) -> None:
    """The strict canvas-corridor guard passes, not merely warns.

    The reproducers each seat a whole bundle in a canvas margin, which is the
    arrangement the guard refuses when the margin cannot hold it.
    """
    path = ROOT / name
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plan(path, strict=True)
    downgraded = [
        item for item in caught if issubclass(item.category, PermissiveGuardWarning)
    ]
    assert not downgraded, [str(item.message) for item in downgraded]
