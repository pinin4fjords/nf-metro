"""An unlevelable lane hand-over is drawn as a step, not smeared into a slant.

A connector between two stations on one row draws level when both ends carry the
line on the same lane.  Where they cannot -- the receiving section already
carries the line above, so the lane its feeder hands the line over on is taken
-- the difference still has to be drawn.  Spreading it over the whole connector
paints a shallow slope: too gentle to read as a turn, too steep to read as
level, and carrying a chevron pointing off-axis.  The house shape is the step:
a flat runway, a 45-degree diagonal of exactly the lane difference, then a flat
runway into the port.

``funcprofiler_upstream`` is the fixture that cannot be levelled: its receiving
section carries the handed-over line eight lanes above the lane its feeder
holds it on.  The rest are maps whose same-row hand-overs change lane for their
own reasons, so the invariant is exercised beyond one topology.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import (
    COORD_TOLERANCE,
    CURVE_RADIUS,
    DIAGONAL_RUN,
    MIN_STRAIGHT_EDGE,
)
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.centrelines import _lane_change_step
from nf_metro.layout.routing.common import RoutedPath, apply_route_offsets
from nf_metro.layout.routing.context import _build_routing_context, _get_offset
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent

FIXTURES = [
    "examples/topologies/funcprofiler_upstream.mmd",
    "tests/fixtures/hash_seed_determinism/seed_41.mmd",
    "tests/fixtures/hash_seed_determinism/seed_77.mmd",
]
FIXTURE_IDS = [Path(f).stem for f in FIXTURES]

STEP_FIXTURE = "examples/topologies/funcprofiler_upstream.mmd"
STEP_EDGE = ("__junction_6", "profiling__entry_left_3", "db")

_TOLERANCE = 0.5


def _lane_changing_row_connectors(
    fixture: str,
) -> list[tuple[RoutedPath, list[tuple[float, float]], float]]:
    """Same-row connectors that have to change lane, with room to step.

    Selects on the settled offsets rather than on the drawn shape, so the same
    connectors are found whether they step or slant.  A route that doubles back
    is a merge overshoot rather than a run between the two ports, and a run
    shorter than two runways plus the lane difference has nowhere to put the
    diagonal.
    """
    graph = parse_metro_mermaid((ROOT / fixture).read_text(), max_station_columns=15)
    compute_layout(graph)
    offsets = dict(compute_station_offsets(graph))
    selected = []
    for route in route_edges(graph, station_offsets=offsets):
        source = graph.stations.get(route.edge.source)
        target = graph.stations.get(route.edge.target)
        if not route.is_inter_section or source is None or target is None:
            continue
        if abs(source.y - target.y) > _TOLERANCE:
            continue
        delta = abs(
            offsets.get((route.edge.source, route.line_id), 0.0)
            - offsets.get((route.edge.target, route.line_id), 0.0)
        )
        if delta <= _TOLERANCE:
            continue
        points = apply_route_offsets(route, offsets)
        span = points[-1][0] - points[0][0]
        sign = 1.0 if span >= 0 else -1.0
        if any(
            (right[0] - left[0]) * sign < -_TOLERANCE
            for left, right in zip(points, points[1:])
        ):
            continue
        if abs(span) + _TOLERANCE < 2 * MIN_STRAIGHT_EDGE + delta:
            continue
        selected.append((route, points, delta))
    return selected


@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_IDS)
def test_row_connector_changes_lane_on_a_diagonal(fixture: str) -> None:
    """Every segment of such a connector is axis-aligned or exactly 45 degrees."""
    selected = _lane_changing_row_connectors(fixture)
    assert selected, f"{fixture} no longer exercises a same-row lane change"
    slanted = [
        f"{route.edge.source}->{route.edge.target} ({route.line_id}): {points}"
        for route, points, _delta in selected
        for left, right in zip(points, points[1:])
        if abs(left[0] - right[0]) > _TOLERANCE
        and abs(left[1] - right[1]) > _TOLERANCE
        and abs(abs(left[0] - right[0]) - abs(left[1] - right[1])) > _TOLERANCE
    ]
    assert not slanted, "\n".join(slanted)


@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_IDS)
def test_row_connector_keeps_a_runway_either_side_of_the_diagonal(
    fixture: str,
) -> None:
    """The diagonal covers the whole lane change and no more, flat either side.

    Restricted to connectors whose only lateral travel is the lane change: a
    route that swings further than that is reaching a channel, and the segment
    it changes lane on answers to that channel rather than to the two ports.
    """
    problems = []
    for route, points, delta in _lane_changing_row_connectors(fixture):
        laterals = [y for _x, y in points]
        if max(laterals) - min(laterals) > delta + _TOLERANCE:
            continue
        name = f"{route.edge.source}->{route.edge.target} ({route.line_id})"
        diagonals = [
            (left, right)
            for left, right in zip(points, points[1:])
            if abs(left[1] - right[1]) > _TOLERANCE
        ]
        if len(diagonals) != 1:
            problems.append(f"{name}: {len(diagonals)} lane-changing segments")
            continue
        start, end = diagonals[0]
        lead = abs(start[0] - points[0][0])
        tail = abs(points[-1][0] - end[0])
        if min(lead, tail) + _TOLERANCE < MIN_STRAIGHT_EDGE:
            problems.append(f"{name}: runways {lead} and {tail} either side")
    assert not problems, "\n".join(problems)


def test_over_constrained_hand_over_steps_at_the_port() -> None:
    """The unlevelable hand-over draws its step hard against the port."""
    selected = _lane_changing_row_connectors(STEP_FIXTURE)
    assert len(selected) == 1
    route, points, delta = selected[0]
    assert (route.edge.source, route.edge.target, route.line_id) == STEP_EDGE
    assert len(points) == 4, points
    lead = points[1][0] - points[0][0]
    diagonal = points[2][0] - points[1][0]
    tail = points[3][0] - points[2][0]
    assert lead >= MIN_STRAIGHT_EDGE
    assert diagonal == pytest.approx(delta)
    assert tail == pytest.approx(MIN_STRAIGHT_EDGE)
    assert points[0][1] == points[1][1]
    assert points[2][1] == points[3][1]


def test_lane_change_step_declines_a_run_too_short_for_two_runways() -> None:
    """A run shorter than two runways plus the lane difference draws no step.

    ``_lane_change_step`` hands off to :func:`route_lane_transition`, which
    lays a flat runway, the diagonal, and a second flat runway end to end; a
    run too short to fit all three has nowhere to put them.  This calls the
    step builder directly on the port-bound edge that does step, with its run
    shortened below that floor, isolating the guard from the placement that
    normally keeps every real run above it.
    """
    graph = parse_metro_mermaid(
        (ROOT / STEP_FIXTURE).read_text(), max_station_columns=15
    )
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    ctx = _build_routing_context(graph, DIAGONAL_RUN, CURVE_RADIUS, offsets)
    edge = ctx.edge_by_key[STEP_EDGE]
    assert edge.target in ctx.graph.ports
    p_src = (graph.stations[edge.source].x, graph.stations[edge.source].y)
    diagonal_run = abs(
        _get_offset(ctx, edge.target, edge.line_id)
        - _get_offset(ctx, edge.source, edge.line_id)
    )
    short_run = 2 * MIN_STRAIGHT_EDGE + diagonal_run - COORD_TOLERANCE - 1
    p_tgt = (p_src[0] + short_run, p_src[1])
    assert _lane_change_step(edge, ctx, p_src, p_tgt) is None
