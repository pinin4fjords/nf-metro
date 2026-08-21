"""Diagonally translated fan openings keep one arc centre at both corners.

A fan whose branches descend one lane apart in X *and* traverse one band apart
in Y translates its whole opening diagonally.  The two flanking turns then nest
in opposite senses: the branch that is innermost leaving the junction is
outermost turning onto its traverse.  Sizing the traverse's incoming corner from
a rank magnitude rather than from the branch's signed displacement inverts one of
those nests, pinching the bundle through the bend.
"""

from __future__ import annotations

import pytest

from nf_metro.layout.constants import CURVE_RADIUS
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.layout.routing.corners import _corner_travel_units
from nf_metro.layout.routing.invariants import (
    _translated_corner,
    check_fan_opening_geometry,
)
from nf_metro.layout.routing.normalize import _flanking_reference_radii
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge

FORK_ID = "__junction_5"

_TEMPLATE = """%%metro title: diagonal fan opening
%%metro line: a | A | #8453d7
%%metro line: b | B | #6ef362
%%metro line: d | D | #dde6c4
%%metro line: e | E | #156075
%%metro grid: src | 0,0
%%metro grid: t0 | 1,0
%%metro grid: t1 | 1,{t1_row}
%%metro grid: t2 | 2,{t2_row}
%%metro grid: t3 | 1,3

graph LR
    subgraph src [Source]
        s_in[Source input]
        s_hub[Source hub]
        s_in -->|a| s_hub
    end
    subgraph t0 [t0]
        t0_a[t0 a]
        t0_b[t0 b]
        t0_a -->|b| t0_b
    end
    subgraph t1 [t1]
        t1_a[t1 a]
        t1_b[t1 b]
        t1_a -->|b| t1_b
    end
    subgraph t2 [t2]
        %%metro entry: right | d
        t2_a[t2 a]
        t2_b[t2 b]
        t2_a -->|d| t2_b
    end
    subgraph t3 [t3]
        %%metro entry: right | e
        t3_a[t3 a]
        t3_b[t3 b]
        t3_a -->|e| t3_b
    end

    s_hub -->|b| t0_a
    s_hub -->|b| t1_a
    s_hub -->|d| t2_a
    s_hub -->|e| t3_a
"""


def _branch(routes: list[RoutedPath], line_id: str) -> RoutedPath:
    """The named line's first branch off the shared fork."""
    return next(
        route
        for route in routes
        if route.edge.source == FORK_ID and route.line_id == line_id
    )


@pytest.mark.parametrize(
    ("t1_row", "t2_row"),
    [(1, 2), (2, 1)],
    ids=["t2-below-t1", "t2-above-t1"],
)
def test_diagonal_fan_opening_corners_share_one_arc_centre(
    t1_row: int, t2_row: int
) -> None:
    """Both flanking turns of a diagonally translated opening stay concentric."""
    source = _TEMPLATE.format(t1_row=t1_row, t2_row=t2_row)
    graph = parse_metro_mermaid(source)
    compute_layout(graph, validate=True)

    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    assert check_fan_opening_geometry(graph, routes, offsets) == []

    inner = _branch(routes, "d")
    outer = _branch(routes, "e")
    assert inner.curve_radii is not None
    assert outer.curve_radii is not None

    # The premise: one opening translated in both axes, so a rank magnitude and
    # a signed displacement disagree about which branch is inside each turn.
    corner_delta = (
        outer.points[1][0] - inner.points[1][0],
        outer.points[1][1] - inner.points[1][1],
    )
    channel_delta = (
        outer.points[2][0] - inner.points[2][0],
        outer.points[2][1] - inner.points[2][1],
    )
    assert corner_delta == channel_delta
    assert corner_delta[0] != 0.0
    assert corner_delta[1] != 0.0

    for corner_index in (1, 2):
        incoming, outgoing = _corner_travel_units(
            inner.points[corner_index - 1],
            inner.points[corner_index],
            inner.points[corner_index + 1],
        )
        translated = _translated_corner(
            inner.points[corner_index],
            inner.curve_radii[corner_index - 1],
            outer.points[corner_index],
            outer.curve_radii[corner_index - 1],
            incoming,
            outgoing,
        )
        assert translated is not None, f"corner {corner_index} is not translated"
        assert translated[2] < 1e-6, (
            f"corner {corner_index} arc centres {translated[0]} and {translated[1]} "
            f"are {translated[2]:.1f}px apart"
        )


def _reseat_route(points: list[tuple[float, float]]) -> RoutedPath:
    """A route carrying one corner radius slot per interior waypoint."""
    return RoutedPath(
        edge=Edge("start", "end", "a"),
        line_id="a",
        points=points,
        curve_radii=[CURVE_RADIUS] * max(len(points) - 2, 0),
    )


def test_flanking_references_read_each_side_from_its_own_corner() -> None:
    """An interior leg's two corners each take the displacement facing them.

    The leg is inside one turn and outside the other, so the incoming corner is
    sized from the incoming displacement and the outgoing corner from the
    outgoing one.
    """
    route = _reseat_route([(50.0, 0.0), (150.0, 0.0), (150.0, 60.0), (300.0, 60.0)])

    incoming, outgoing = _flanking_reference_radii(
        ((route, 1, 180.0, 0, (-14.0, 8.0)),), CURVE_RADIUS
    )

    assert incoming == pytest.approx(CURVE_RADIUS + 14.0)
    assert outgoing == pytest.approx(CURVE_RADIUS + 8.0)


def test_flanking_references_leave_a_first_leg_incoming_side_at_the_floor() -> None:
    """The route's opening leg has no incoming corner to size.

    Its incoming radius slot is ``idx - 1``, which for the opening leg indexes
    from the end of the route and lands on the *closing* corner.  Sizing the
    incoming reference from that unrelated corner inflates it by a displacement
    belonging to the other end of the route.
    """
    route = _reseat_route([(100.0, 0.0), (100.0, 50.0), (400.0, 50.0)])

    incoming, outgoing = _flanking_reference_radii(
        ((route, 0, 130.0, 0, (24.0, 6.0)),), CURVE_RADIUS
    )

    assert incoming == pytest.approx(CURVE_RADIUS)
    assert outgoing == pytest.approx(CURVE_RADIUS + 6.0)
