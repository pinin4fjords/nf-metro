"""Opposing feeders retain one lane order through an entry confluence."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from nf_metro.layout.constants import CURVE_RADIUS, DIAGONAL_RUN
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing import normalize as routing_normalize
from nf_metro.layout.routing.common import (
    OpposingEntryConfluence,
    PeeloffTail,
    RoutedPath,
    iter_opposing_entry_confluences,
    opposing_entry_confluence_slots,
    port_peeloff_tail,
)
from nf_metro.layout.routing.context import _build_routing_context
from nf_metro.layout.routing.corners import resolve_curve_radii
from nf_metro.layout.routing.invariants import (
    _segments_properly_cross,
    check_opposing_entry_confluence_order,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge, MetroGraph, Port, PortSide, Section

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "topologies" / "leftward_up_exit_turn_order.mmd"
PARTIAL_PORT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "opposing_entry_partial_port_line.mmd"
)


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


def test_partial_port_line_keeps_opposing_pair_out_of_compatibility_ownership() -> None:
    graph = parse_metro_mermaid(PARTIAL_PORT_FIXTURE.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_layout(graph, validate=True)
    routes = route_edges(graph, station_offsets=compute_station_offsets(graph))
    port_id = graph.sections["target"].entry_ports[0]
    shaped_lines = {
        route.line_id
        for route in routes
        if route.edge.target == port_id and port_peeloff_tail(route) is not None
    }

    assert shaped_lines == {"feed_a", "feed_b"}
    assert set(graph.station_lines(port_id)) == {"direct", "feed_a", "feed_b"}
    assert not any(
        bundle.port_id == port_id
        for bundle in iter_opposing_entry_confluences(routes, graph, 4.0)
    )


def _candidate_graph(*line_ids: str) -> MetroGraph:
    graph = MetroGraph()
    graph.ports["entry"] = Port(
        id="entry",
        section_id="target",
        side=PortSide.LEFT,
        is_entry=True,
    )
    graph.edges = [Edge(f"source_{line_id}", "entry", line_id) for line_id in line_ids]
    return graph


def _candidate_route(
    line_id: str,
    *,
    trunk_start_x: float,
    trunk_y: float,
    peel_x: float,
    port_y: float,
) -> RoutedPath:
    return RoutedPath(
        edge=Edge(f"source_{line_id}", "entry", line_id),
        line_id=line_id,
        points=[
            (trunk_start_x, trunk_y),
            (peel_x, trunk_y),
            (peel_x, port_y),
            (220.0, port_y),
        ],
        is_inter_section=True,
    )


def _candidate_bundle(
    *,
    second_trunk_y: float = 4.0,
    second_peel_x: float = 104.0,
    second_port_y: float = 104.0,
) -> tuple[MetroGraph, list[RoutedPath]]:
    graph = _candidate_graph("a", "b")
    routes = [
        _candidate_route(
            "a",
            trunk_start_x=0.0,
            trunk_y=0.0,
            peel_x=100.0,
            port_y=100.0,
        ),
        _candidate_route(
            "b",
            trunk_start_x=200.0,
            trunk_y=second_trunk_y,
            peel_x=second_peel_x,
            port_y=second_port_y,
        ),
    ]
    return graph, routes


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param({"second_port_y": 100.0}, id="duplicate-port-slot"),
        pytest.param({"second_port_y": 108.0}, id="noncontiguous-port-band"),
        pytest.param({"second_peel_x": 108.0}, id="noncontiguous-channel-band"),
        pytest.param(
            {"second_trunk_y": 84.0},
            id="insufficient-common-approach",
        ),
    ],
)
def test_invalid_opposing_candidate_is_rejected(candidate: dict[str, float]) -> None:
    graph, routes = _candidate_bundle(**candidate)

    assert not list(iter_opposing_entry_confluences(routes, graph, 4.0))


def test_complete_opposing_candidate_is_recognised() -> None:
    graph, routes = _candidate_bundle()

    bundles = list(iter_opposing_entry_confluences(routes, graph, 4.0))

    assert len(bundles) == 1
    assert isinstance(bundles[0], OpposingEntryConfluence)
    assert bundles[0].per_line == {
        "a": PeeloffTail(0.0, 0.0, 100.0, 100.0, 1, 1, 1),
        "b": PeeloffTail(200.0, 4.0, 104.0, 104.0, -1, 1, 1),
    }


def _swap_confluence_channels(feeders: dict[str, RoutedPath]) -> None:
    first_x = feeders["feed_a"].points[-3][0]
    second_x = feeders["feed_b"].points[-3][0]
    for point_index in (-3, -2):
        feeders["feed_a"].points[point_index] = (
            second_x,
            feeders["feed_a"].points[point_index][1],
        )
        feeders["feed_b"].points[point_index] = (
            first_x,
            feeders["feed_b"].points[point_index][1],
        )


def test_planner_owned_opposing_group_is_left_atomic() -> None:
    graph, offsets, routes, feeders = _feeders()
    _swap_confluence_channels(feeders)
    feeders["feed_a"].fan_route_emitter = "planned"
    before = {line_id: list(route.points) for line_id, route in feeders.items()}
    ctx = _build_routing_context(graph, DIAGONAL_RUN, CURVE_RADIUS, offsets)

    routing_normalize._stagger_convergent_distinct_lines(routes, ctx)

    assert {line_id: route.points for line_id, route in feeders.items()} == before


def test_obstructed_opposing_group_is_left_atomic() -> None:
    graph, offsets, routes, feeders = _feeders()
    _swap_confluence_channels(feeders)
    bundle = next(iter(iter_opposing_entry_confluences(routes, graph, 4.0)))
    slots = opposing_entry_confluence_slots(bundle, graph, 4.0)
    blocked_x = slots["feed_a"].peel_x
    graph.sections["obstacle"] = Section(
        id="obstacle",
        name="Obstacle",
        bbox_x=blocked_x - 5.0,
        bbox_y=180.0,
        bbox_w=10.0,
        bbox_h=130.0,
    )
    before = {line_id: list(route.points) for line_id, route in feeders.items()}
    ctx = _build_routing_context(graph, DIAGONAL_RUN, CURVE_RADIUS, offsets)

    routing_normalize._stagger_convergent_distinct_lines(routes, ctx)

    assert {line_id: route.points for line_id, route in feeders.items()} == before


def test_opposing_group_with_an_extra_mapped_approach_uses_generic_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, offsets, routes, feeders = _feeders()
    port_id = feeders["feed_a"].edge.target
    extra = replace(
        feeders["feed_a"],
        edge=Edge(
            feeders["feed_a"].edge.source,
            "source_a",
            feeders["feed_a"].line_id,
        ),
        points=list(feeders["feed_a"].points),
    )
    routes.append(extra)
    ctx = _build_routing_context(graph, DIAGONAL_RUN, CURVE_RADIUS, offsets)
    ctx.merge.entry_port_for["source_a"] = port_id

    def fail_if_compatibility_claims_group(*_args, **_kwargs):
        pytest.fail("partial destination group reached compatibility slotting")

    monkeypatch.setattr(
        routing_normalize,
        "opposing_entry_confluence_slots",
        fail_if_compatibility_claims_group,
    )

    routing_normalize._stagger_convergent_distinct_lines(routes, ctx)
