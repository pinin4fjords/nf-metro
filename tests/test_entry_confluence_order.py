"""Opposing feeders retain one lane order through an entry confluence."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import port_peeloff_tail
from nf_metro.layout.routing.corners import resolve_curve_radii
from nf_metro.layout.routing.invariants import (
    _segments_properly_cross,
    check_opposing_entry_confluence_order,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "topologies" / "leftward_up_exit_turn_order.mmd"


def _feeders():
    graph = parse_metro_mermaid(FIXTURE.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    feeders = {
        route.line_id: route
        for route in routes
        if route.edge.target == "source__entry_right_4"
        and route.line_id in {"feed_a", "feed_b"}
    }
    assert feeders.keys() == {"feed_a", "feed_b"}
    return graph, offsets, routes, feeders


def _proper_crossings(first, second) -> list[tuple[float, float]]:
    return [
        crossing
        for start_a, end_a in zip(first.points, first.points[1:])
        for start_b, end_b in zip(second.points, second.points[1:])
        if (
            crossing := _segments_properly_cross(
                start_a,
                end_a,
                start_b,
                end_b,
            )
        )
        is not None
    ]


def _last_arc_centre(route) -> tuple[float, float]:
    radii = resolve_curve_radii(route.points, route.curve_radii)
    corner = route.points[-2]
    radius = radii[-1]
    return corner[0] - radius, corner[1] - radius


def test_opposing_feeders_keep_port_order_through_the_confluence_corner() -> None:
    graph, _offsets, routes, feeders = _feeders()
    tails = {line_id: port_peeloff_tail(route) for line_id, route in feeders.items()}
    assert all(tail is not None for tail in tails.values())
    port_order = sorted(tails, key=lambda line_id: tails[line_id].port_y)
    channel_order = sorted(tails, key=lambda line_id: tails[line_id].peel_x)

    assert port_order == channel_order
    assert _last_arc_centre(feeders["feed_a"]) == pytest.approx(
        _last_arc_centre(feeders["feed_b"])
    )
    assert not _proper_crossings(feeders["feed_a"], feeders["feed_b"])
    assert not check_opposing_entry_confluence_order(graph, routes)


def test_double_crossing_at_an_entry_confluence_is_rejected() -> None:
    graph, _offsets, routes, feeders = _feeders()
    first_x = feeders["feed_a"].points[-3][0]
    second_x = feeders["feed_b"].points[-3][0]
    for point_index in (-3, -2):
        first_y = feeders["feed_a"].points[point_index][1]
        second_y = feeders["feed_b"].points[point_index][1]
        feeders["feed_a"].points[point_index] = (second_x, first_y)
        feeders["feed_b"].points[point_index] = (first_x, second_y)

    violations = check_opposing_entry_confluence_order(graph, routes)

    assert {violation.line_id for violation in violations} == {"feed_a", "feed_b"}
    assert all(violation.pair_crossings == 2 for violation in violations)
