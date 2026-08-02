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
    FanOpeningFailure,
    FanOpeningGeometryViolation,
    NonConcentricCornerViolation,
    StarvedTurnViolation,
    assert_render_curve_invariants,
    check_bundle_order_preserved,
    check_concentric_bundle_corners,
    check_fan_opening_geometry,
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
    assert not check_fan_opening_geometry(graph, routes, offsets)

    branch.points[0] = branch.points[1]
    assert check_fan_opening_geometry(graph, routes, offsets)


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
    assert not check_fan_opening_geometry(graph, routes, offsets)

    outgoing["lower"].points[0] = (
        outgoing["lower"].points[0][0] + 4,
        outgoing["lower"].points[0][1],
    )
    assert check_fan_opening_geometry(graph, routes, offsets)


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
    assert normal.curve_radii[:2] == [
        CURVE_RADIUS + OFFSET_STEP,
        CURVE_RADIUS,
    ]
    assert exempt.curve_radii[:2] == [
        CURVE_RADIUS,
        CURVE_RADIUS + OFFSET_STEP,
    ]
    assert _arc_centre(normal, 1) == pytest.approx(_arc_centre(exempt, 1))
    assert _arc_centre(normal, 2) == pytest.approx(_arc_centre(exempt, 2))
    assert not check_fan_opening_geometry(graph, routes, offsets)

    for route in (normal, exempt):
        route.curve_radii[0] -= OFFSET_STEP
        route.curve_radii[1] -= OFFSET_STEP
    assert any(
        isinstance(violation, StarvedTurnViolation)
        for violation in check_fan_opening_geometry(graph, routes, offsets)
    )


@pytest.mark.parametrize(
    "corruption, expected_type, expected_failure",
    [
        ("collapsed", FanOpeningGeometryViolation, FanOpeningFailure.DEGENERATE),
        ("diagonal", FanOpeningGeometryViolation, FanOpeningFailure.NON_CARDINAL),
        ("same_rank", NonConcentricCornerViolation, None),
        ("same_status", NonConcentricCornerViolation, None),
    ],
)
def test_cross_family_fan_corruption_cannot_hide_from_semantic_invariant(
    corruption: str,
    expected_type: type,
    expected_failure: FanOpeningFailure | None,
) -> None:
    graph, routes, offsets = _layout("seed72_cross_family_fan")
    junction_id = next(iter(graph.junction_ids))
    by_line = {
        route.line_id: route
        for route in routes
        if route.edge.source == junction_id and len(route.points) >= 4
    }
    normal = by_line["normal"]
    exempt = by_line["exempt"]

    if corruption == "collapsed":
        normal.points[0] = normal.points[1]
    elif corruption == "diagonal":
        normal.points[1] = (normal.points[1][0], normal.points[1][1] + 3.0)
    else:
        assert normal.curve_radii is not None
        normal.curve_radii[0] -= OFFSET_STEP
        normal.curve_radii[1] -= OFFSET_STEP
        if corruption == "same_rank":
            plan = graph.fan_plan_query.planned_for_fork(junction_id)
            assert plan is not None
            normal_branch = next(
                branch for branch in plan.branches if "normal" in branch.line_ids
            )
            exempt_branch = next(
                branch for branch in plan.branches if "exempt" in branch.line_ids
            )
            object.__setattr__(
                normal_branch, "opening_rank", exempt_branch.opening_rank
            )
        else:
            normal.normalize_exempt = exempt.normalize_exempt

    violations = check_fan_opening_geometry(graph, routes, offsets)
    matches = [
        violation for violation in violations if isinstance(violation, expected_type)
    ]
    assert matches
    if expected_failure is not None:
        assert any(violation.failure is expected_failure for violation in matches)


@pytest.mark.parametrize(
    ("corruption", "failure"),
    [
        ("collapsed", FanOpeningFailure.DEGENERATE),
        ("diagonal", FanOpeningFailure.NON_CARDINAL),
    ],
)
def test_cross_family_fan_family_survives_when_every_first_run_is_malformed(
    corruption: str, failure: FanOpeningFailure
) -> None:
    graph, routes, offsets = _layout("seed72_cross_family_fan")
    junction_id = next(iter(graph.junction_ids))
    branches = [
        route
        for route in routes
        if route.edge.source == junction_id and route.line_id in {"normal", "exempt"}
    ]
    for route in branches:
        if corruption == "collapsed":
            route.points[0] = route.points[1]
        else:
            route.points[0] = (route.points[0][0], route.points[0][1] + 3.0)

    violations = check_fan_opening_geometry(graph, routes, offsets)
    malformed = [
        violation
        for violation in violations
        if isinstance(violation, FanOpeningGeometryViolation)
        and violation.failure is failure
    ]
    assert {violation.line_id for violation in malformed} == {"normal", "exempt"}


@pytest.mark.parametrize(
    "fixture",
    [
        "seed72_cross_family_fan",
        "bottom_exit_junction_collinear_top_entry",
        "bottom_exit_stacked_right_entry_fan",
        "bypass_gap2_rightward_overflow",
        "fanout_bundle_plus_spurs",
        "asymmetric_tree",
        "dogleg_twoline_fanout",
        "fan_bypass_nesting",
        "fanout_intersection_shared_channel",
        "same_line_fan_distinct_descent",
        "wide_fan_out",
    ],
)
def test_semantic_fan_opening_invariant_accepts_clean_families(fixture: str) -> None:
    graph, routes, offsets = _layout(fixture)
    assert not check_fan_opening_geometry(graph, routes, offsets)
