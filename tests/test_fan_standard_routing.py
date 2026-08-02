"""Fan semantics preserve the standard routing and curve contracts."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import is_orthogonal_turn
from nf_metro.layout.routing.corners import resolve_curve_radii
from nf_metro.layout.routing.invariants import (
    FanOpeningSubfloorRadiusViolation,
    assert_render_curve_invariants,
    check_bundle_order_preserved,
    check_concentric_bundle_corners,
    check_distinct_fan_opening_corners_concentric,
    check_fan_opening_turn_runway,
    check_planned_vertical_fan_opening,
    check_seam_segments_meet_at_port,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES = Path("examples/topologies")


def _layout(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_metro_mermaid((TOPOLOGIES / f"{name}.mmd").read_text())
        compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    return graph, route_edges(graph, station_offsets=offsets), offsets


def _junction_routes(routes, junction_id: str):
    incoming = {
        route.line_id: route for route in routes if route.edge.target == junction_id
    }
    outgoing = {
        route.line_id: route for route in routes if route.edge.source == junction_id
    }
    return incoming, outgoing


def _arc_centre(route, corner_index: int) -> tuple[float, float]:
    points = route.points
    corner = points[corner_index]
    before = points[corner_index - 1]
    after = points[corner_index + 1]
    assert is_orthogonal_turn(before, corner, after)
    radii = resolve_curve_radii(points, route.curve_radii)
    radius = radii[corner_index - 1]
    incoming = (
        (corner[0] - before[0]) / abs(corner[0] - before[0])
        if corner[0] != before[0]
        else 0.0,
        (corner[1] - before[1]) / abs(corner[1] - before[1])
        if corner[1] != before[1]
        else 0.0,
    )
    outgoing = (
        (after[0] - corner[0]) / abs(after[0] - corner[0])
        if after[0] != corner[0]
        else 0.0,
        (after[1] - corner[1]) / abs(after[1] - corner[1])
        if after[1] != corner[1]
        else 0.0,
    )
    return (
        corner[0] + radius * (outgoing[0] - incoming[0]),
        corner[1] + radius * (outgoing[1] - incoming[1]),
    )


def test_bottom_exit_divergence_keeps_curve_runway() -> None:
    """A vertical fan trunk reaches its first horizontal turn through a real run."""
    graph, routes, offsets = _layout("bottom_exit_junction_collinear_top_entry")
    branch = next(
        route
        for route in routes
        if route.edge.source.startswith("__junction")
        and route.edge.target.startswith("second__entry")
    )
    first, turn = branch.points[:2]
    assert turn[0] == pytest.approx(first[0])
    assert abs(turn[1] - first[1]) >= CURVE_RADIUS
    assert branch.curve_radii
    assert not check_fan_opening_turn_runway(graph, routes, offsets)

    branch.points[0] = branch.points[1]
    assert check_fan_opening_turn_runway(graph, routes, offsets)


def test_stacked_right_entry_fan_continues_the_incoming_lanes() -> None:
    """The standard wrap owns a vertical lead-in from each bottom-exit lane."""
    graph, routes, offsets = _layout("bottom_exit_stacked_right_entry_fan")
    junction_id = next(iter(graph.junction_ids))
    incoming, outgoing = _junction_routes(routes, junction_id)

    assert incoming.keys() == outgoing.keys() == {"upper", "lower"}
    for line_id in incoming:
        assert outgoing[line_id].points[0] == pytest.approx(
            incoming[line_id].points[-1]
        )
        first, turn = outgoing[line_id].points[:2]
        assert turn[0] == pytest.approx(first[0])
        assert abs(turn[1] - first[1]) >= CURVE_RADIUS
    assert not check_planned_vertical_fan_opening(graph, routes, offsets)

    outgoing["lower"].points[0] = (
        outgoing["lower"].points[0][0] + 4,
        outgoing["lower"].points[0][1],
    )
    assert check_planned_vertical_fan_opening(graph, routes, offsets)


def test_stacked_right_multiline_branch_keeps_lane_order_through_landing() -> None:
    graph, routes, offsets = _layout("bottom_exit_stacked_right_entry_multiline_branch")
    upper = {
        route.line_id: route
        for route in routes
        if route.fan_route_emitter is not None
        and route.edge.target.startswith("upper_target__entry_right")
    }

    assert upper.keys() == {"upper_a", "upper_b"}
    assert upper["upper_b"].points[2][0] < upper["upper_a"].points[2][0]
    assert upper["upper_b"].points[-1][1] < upper["upper_a"].points[-1][1]
    assert not check_seam_segments_meet_at_port(graph, routes, offsets)
    assert not check_bundle_order_preserved(routes)
    assert not check_concentric_bundle_corners(graph, routes, offsets)
    assert_render_curve_invariants(graph, routes, offsets)


def test_cross_family_fan_opening_corners_are_concentric() -> None:
    """Different route families share the standard fan corner geometry."""
    graph, routes, offsets = _layout("seed72_cross_family_fan")
    junction_id = next(iter(graph.junction_ids))
    by_line = {
        route.line_id: route
        for route in routes
        if route.edge.source == junction_id and len(route.points) >= 3
    }

    normal = by_line["normal"]
    exempt = by_line["exempt"]
    assert normal.curve_radii is not None
    assert exempt.curve_radii is not None
    assert min(normal.curve_radii[:2] + exempt.curve_radii[:2]) >= CURVE_RADIUS
    assert _arc_centre(normal, 1) == pytest.approx(_arc_centre(exempt, 1))
    assert _arc_centre(normal, 2) == pytest.approx(_arc_centre(exempt, 2))
    assert not check_distinct_fan_opening_corners_concentric(graph, routes, offsets)

    for route in (normal, exempt):
        route.curve_radii[0] -= OFFSET_STEP
        route.curve_radii[1] -= OFFSET_STEP
    assert any(
        isinstance(violation, FanOpeningSubfloorRadiusViolation)
        for violation in check_distinct_fan_opening_corners_concentric(
            graph, routes, offsets
        )
    )
