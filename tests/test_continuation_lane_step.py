"""A trunk that cannot keep its hand-over lane steps once, mid-section.

The lane a bundle leaves its exit port on is the lane the receiving section is
asked to hold.  Where a station part-way in has already given that lane to a
line of its own, the continuation cannot hold it all the way and the difference
has to be drawn somewhere.  Drawn against the entry port it tilts every run
from the junction inwards; drawn across the whole connector it paints a slope
too gentle to read as a turn and too steep to read as level.  The house shape
puts it on the one leg that cannot be levelled, as a flat runway, a 45-degree
diagonal of exactly the lane difference, then a flat run into the station that
claimed the lane.

``continuation_lane_step`` is the map that cannot be levelled: two lines cross
into a section whose hub, two stations in, starts two more lines on the lanes
they arrive on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import MIN_STRAIGHT_EDGE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import RoutedPath, apply_route_offsets
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "topologies" / "continuation_lane_step.mmd"

STEPPING_EDGE = ("deseq2", "results_hub")
LEVEL_EDGES = (
    ("salmon_quant", "quant__exit_right_0"),
    ("quant__exit_right_0", "__junction_3"),
    ("__junction_3", "diff__entry_left_1"),
    ("diff__entry_left_1", "deseq2"),
)
TRUNK_LINES = ("counts", "norm")
_TOLERANCE = 0.5


def _drawn_routes() -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph)
    offsets = dict(compute_station_offsets(graph))
    routes: list[RoutedPath] = route_edges(graph, station_offsets=offsets)
    return {
        (route.edge.source, route.edge.target, route.line_id): apply_route_offsets(
            route, offsets
        )
        for route in routes
    }


@pytest.mark.parametrize("line_id", TRUNK_LINES)
@pytest.mark.parametrize("edge", LEVEL_EDGES, ids=lambda item: f"{item[0]}-{item[1]}")
def test_trunk_holds_its_hand_over_lane_up_to_the_step(
    edge: tuple[str, str], line_id: str
) -> None:
    """Every run from the source station to the last levelable one draws flat."""
    points = _drawn_routes()[(*edge, line_id)]
    laterals = {round(y, 3) for _x, y in points}
    assert len(laterals) == 1, f"{edge} ({line_id}) is not level: {points}"


@pytest.mark.parametrize("line_id", TRUNK_LINES)
def test_trunk_steps_onto_the_hub_lane_with_a_runway_either_side(
    line_id: str,
) -> None:
    """The one unlevelable leg draws flat, a 45-degree diagonal, flat."""
    points = _drawn_routes()[(*STEPPING_EDGE, line_id)]
    assert len(points) == 4, points
    lead = points[1][0] - points[0][0]
    diagonal_x = points[2][0] - points[1][0]
    diagonal_y = points[2][1] - points[1][1]
    tail = points[3][0] - points[2][0]
    assert points[0][1] == pytest.approx(points[1][1])
    assert points[2][1] == pytest.approx(points[3][1])
    assert lead >= MIN_STRAIGHT_EDGE - _TOLERANCE
    assert tail >= MIN_STRAIGHT_EDGE - _TOLERANCE
    assert diagonal_x == pytest.approx(abs(diagonal_y))
    assert diagonal_y > 0
