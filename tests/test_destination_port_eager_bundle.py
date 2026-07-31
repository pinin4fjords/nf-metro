from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import (
    COORD_TOLERANCE,
    CURVE_RADIUS,
    graph_offset_step,
)
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import iter_horizontal_trunks
from nf_metro.layout.routing.invariants import check_peeloff_concentric
from nf_metro.parser.mermaid import parse_metro_mermaid

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FIXTURE = EXAMPLES / "topologies" / "packed_cell_right_exit_left_entry_wrap.mmd"


def _routes_into(port_id: str):
    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph, validate=True)
    offsets = compute_station_offsets(graph)
    routes = [
        route
        for route in route_edges(graph, station_offsets=offsets)
        if route.edge.target == port_id
    ]
    return graph, routes


def _tail(route):
    points = route.points
    assert len(points) >= 4
    trunk = list(iter_horizontal_trunks(route))[-1][1]
    trunk_start, trunk_end, approach_end, port_end = points[-4:]
    assert trunk_start[1] == pytest.approx(trunk_end[1])
    assert trunk_end[0] == pytest.approx(approach_end[0])
    assert approach_end[1] == pytest.approx(port_end[1])
    return {
        "trunk": trunk,
        "trunk_y": trunk.y,
        "peel_x": approach_end[0],
        "port_y": port_end[1],
        "trunk_sign": 1 if trunk_end[0] > trunk_start[0] else -1,
        "vertical_sign": 1 if approach_end[1] > trunk_end[1] else -1,
        "port_lead_sign": 1 if port_end[0] > approach_end[0] else -1,
    }


@pytest.mark.parametrize(
    ("port_id", "line_ids"),
    [
        ("qc__entry_left_6", {"reference", "short"}),
        ("polish__entry_left_7", {"assembled", "short"}),
    ],
)
def test_same_destination_port_bundles_at_earliest_shared_corridor(
    port_id: str, line_ids: set[str]
) -> None:
    """Distinct lines sharing a destination-facing trunk overlap form one bundle."""
    graph, routes = _routes_into(port_id)
    assert {route.line_id for route in routes} == line_ids
    tails = {route.line_id: _tail(route) for route in routes}

    overlap_lo = max(tail["trunk"].x_lo for tail in tails.values())
    overlap_hi = min(tail["trunk"].x_hi for tail in tails.values())
    assert overlap_hi - overlap_lo >= 2 * CURVE_RADIUS

    trunk_ys = sorted(tail["trunk_y"] for tail in tails.values())
    step = graph_offset_step(graph)
    assert trunk_ys[-1] - trunk_ys[0] == pytest.approx(
        (len(trunk_ys) - 1) * step,
        abs=COORD_TOLERANCE,
    )
    assert all(
        b - a == pytest.approx(step, abs=COORD_TOLERANCE)
        for a, b in zip(trunk_ys, trunk_ys[1:])
    )


@pytest.mark.parametrize(
    "port_id",
    ["qc__entry_left_6", "polish__entry_left_7"],
)
def test_same_destination_port_tail_keeps_three_axis_order(port_id: str) -> None:
    """Trunk, approach, and port orders follow the tail's turn parity."""
    _graph, routes = _routes_into(port_id)
    tails = {route.line_id: _tail(route) for route in routes}

    shapes = {
        (
            tail["trunk_sign"],
            tail["vertical_sign"],
            tail["port_lead_sign"],
        )
        for tail in tails.values()
    }
    assert len(shapes) == 1
    trunk_sign, vertical_sign, port_lead_sign = shapes.pop()

    trunk_order = sorted(tails, key=lambda line_id: tails[line_id]["trunk_y"])
    peel_order = sorted(tails, key=lambda line_id: tails[line_id]["peel_x"])
    port_order = sorted(tails, key=lambda line_id: tails[line_id]["port_y"])

    expected_peel = (
        trunk_order if -vertical_sign == trunk_sign else list(reversed(trunk_order))
    )
    expected_port = (
        trunk_order if port_lead_sign == trunk_sign else list(reversed(trunk_order))
    )
    assert peel_order == expected_peel
    assert port_order == expected_port


def test_destination_tail_runtime_guard_is_not_vacuous(monkeypatch) -> None:
    """The guard reports loose tails when eager trunk seating is disabled."""
    from nf_metro.layout.routing import normalize

    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    monkeypatch.setattr(
        normalize,
        "_bundle_same_destination_tails",
        lambda routes, ctx: None,
    )
    routes = route_edges(graph, station_offsets=offsets)

    violations = check_peeloff_concentric(graph, routes)
    messages = [violation.message() for violation in violations]
    assert any("qc__entry_left_6" in message for message in messages)
    assert any("polish__entry_left_7" in message for message in messages)
