"""A fan's lane order is read from each leg's own approach, not the bundle's.

A gap corridor collects every leg that descends through it, so one bundle can
hold a divergence fan alongside an unrelated leg that reaches the gap from the
far side.  The approach-weave term that orders the fan models a lead-in
travelling *towards* its channel over the lanes seated before it; a leg
arriving from the other side makes no such weave and says nothing about how the
fan it happens to share the gap with has to stack.

Read per bundle instead of per leg, one such foreign leg silences the term for
everyone, and the fan falls back to its incoming-x order.  For a rightward run
turning down that is the mirror of the order the corner needs: the shallowest
approach rides the outside of the bend and turns off at the largest x, so the
lanes read bottom-approach-first from the left.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.layout.routing.normalize import _distinct_line_order, _VChannel
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge

FROZEN = Path(__file__).parent / "fixtures" / "hash_seed_determinism"

_FAN_SOURCE = "__junction_27"
_FAN_TARGET = "s10__entry_right_19"


def _descent(
    line_id: str,
    target: str,
    *,
    source_x: float,
    approach_y: float,
    channel_x: float,
    deep_y: float,
) -> _VChannel:
    """One leg approaching horizontally at *approach_y*, then descending."""
    points = [
        (source_x, approach_y),
        (channel_x, approach_y),
        (channel_x, deep_y),
        (channel_x - 100.0, deep_y),
    ]
    route = RoutedPath(
        Edge("src", target, line_id), line_id, points, is_inter_section=True
    )
    return _VChannel(route, 1, channel_x, approach_y, deep_y, True)


def _fan() -> list[_VChannel]:
    """Four legs off one junction, overlaid on the channel they will nest in."""
    return [
        _descent(
            f"l{rank}",
            f"t{rank}",
            source_x=100.0,
            approach_y=6.0 + 4.0 * rank,
            channel_x=150.0,
            deep_y=200.0,
        )
        for rank in range(4)
    ]


def _foreign_leg() -> _VChannel:
    """A leg reaching the same gap from the right, sharing the fan's line l1."""
    return _descent(
        "l1", "far", source_x=300.0, approach_y=150.0, channel_x=250.0, deep_y=200.0
    )


def test_fan_lanes_mirror_their_approach_order() -> None:
    """The reference: the deepest approach takes the leftmost lane."""
    assert _distinct_line_order(_fan()) == ["l3", "l2", "l1", "l0"]


def test_a_foreign_leg_in_the_gap_leaves_the_fan_order_alone() -> None:
    """A far-side leg carries no weave, so it casts no vote on the fan."""
    assert _distinct_line_order([*_fan(), _foreign_leg()]) == ["l3", "l2", "l1", "l0"]


def test_seed_41_fan_turns_down_in_its_approach_order() -> None:
    """The corpus case: junction 27's fan shares its gap with a far-side leg.

    Its two same-target legs meet at one corner, where the lane order the corner
    is built from has to match the lanes their approaches arrive on.
    """
    graph = parse_metro_mermaid((FROZEN / "seed_41.mmd").read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    observed = observe_route_edges(graph, station_offsets=offsets)

    legs = {
        route.line_id: route.points
        for route in observed.routes
        if (route.edge.source, route.edge.target) == (_FAN_SOURCE, _FAN_TARGET)
    }
    assert set(legs) == {"l2", "l3"}
    # points[0] is the fan stub, points[1] the corner the descent turns on.
    approach = {line_id: points[0][1] for line_id, points in legs.items()}
    channel = {line_id: points[1][0] for line_id, points in legs.items()}
    deep = {line_id: points[2][1] for line_id, points in legs.items()}

    assert approach["l2"] < approach["l3"]
    assert channel["l3"] < channel["l2"]
    assert deep["l3"] < deep["l2"]
