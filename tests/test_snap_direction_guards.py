"""Snap and reseat movers preserve run direction and corridor identity.

Unit coverage for the direction-preserving mover guards: a coincide snap
never reverses a flanking horizontal run, the pre-freeze seat partitions
same-rank runs into band-overlap components and clamps a pinned component's
straggler only at the nesting pitch, and a landing-opening reseat measures
its runway signed along the approach, refusing a far-side seat unless the
caller carries the join.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nf_metro.layout.constants import CURVE_RADIUS
from nf_metro.layout.route_plan import (
    ConvergenceLanding,
    DemandAxis,
    turn_handedness,
)
from nf_metro.layout.routing import member_geometry, normalize
from nf_metro.layout.routing.common import Direction, RoutedPath
from nf_metro.layout.routing.convergences import _reseat_landing_opening
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.normalize import (
    _snap_reverses_adjacent_run,
    _VChannel,
)
from nf_metro.layout.routing.reserved_bands import (
    ReservedBand,
    ReservedCorridors,
)
from nf_metro.parser.model import Edge
from nf_metro.parser.route_topology import ResolvedEdge


def _channel(points: list[tuple[float, float]], idx: int) -> _VChannel:
    route = RoutedPath(Edge("a", "b", "line"), "line", points, is_inter_section=True)
    lo, hi = sorted((points[idx][1], points[idx + 1][1]))
    return _VChannel(
        route, idx, points[idx][0], lo, hi, points[idx + 1][1] > points[idx][1]
    )


def test_snap_keeping_a_tail_direction_is_allowed() -> None:
    # Riser at x=100 with a leftward tail to the join at x=80: a snap to 90
    # shortens the tail but keeps it leftward.
    ch = _channel([(160.0, 0.0), (100.0, 0.0), (100.0, 50.0), (80.0, 50.0)], 1)
    assert not _snap_reverses_adjacent_run(ch, 90.0)


def test_snap_across_a_tail_endpoint_is_a_reversal() -> None:
    # The same tail flips to rightward when the riser crosses the join.
    ch = _channel([(160.0, 0.0), (100.0, 0.0), (100.0, 50.0), (80.0, 50.0)], 1)
    assert _snap_reverses_adjacent_run(ch, 60.0)


def test_snap_reversing_the_incoming_run_is_a_reversal() -> None:
    # The incoming run travels rightward 160->200; a snap left of its start
    # would flip it.
    ch = _channel([(160.0, 0.0), (200.0, 0.0), (200.0, 50.0), (240.0, 50.0)], 1)
    assert _snap_reverses_adjacent_run(ch, 140.0)


def test_band_overlap_components_split_disjoint_corridors() -> None:
    def item(lo: float, hi: float, coordinate: float):
        route = RoutedPath(
            Edge(f"s{lo}", f"t{hi}", "line"),
            "line",
            [(0.0, coordinate), (10.0, coordinate)],
        )
        return (route, 1, coordinate, ReservedBand(lo, hi))

    disjoint = [item(0.0, 40.0, 10.0), item(100.0, 140.0, 110.0)]
    assert [len(c) for c in member_geometry._band_overlap_components(disjoint)] == [
        1,
        1,
    ]

    nested = [item(0.0, 40.0, 10.0), item(30.0, 70.0, 50.0), item(60.0, 90.0, 80.0)]
    assert [len(c) for c in member_geometry._band_overlap_components(nested)] == [3]


def _landing(join_x: float, opening_x: float) -> ConvergenceLanding:
    return ConvergenceLanding(
        member_id="member",
        edge=ResolvedEdge("src", "tgt", "line"),
        source_junction_id="src",
        approach_axis=DemandAxis.X,
        approach_direction=Direction.L,
        source_column=0,
        source_row=0,
        lane_rank=0,
        order=0,
        join_point=(join_x, 100.0),
        corner_handedness=turn_handedness(Direction.U, Direction.L),
        minimum_runway=opening_x - join_x,
        opening_turn_coordinate=opening_x,
        opening_turn_segment=((opening_x, 200.0), (opening_x, 100.0)),
        bypass=False,
        long_haul=False,
        multiple_row=False,
    )


def test_reseat_on_the_approach_side_keeps_the_join() -> None:
    landing = _landing(join_x=80.0, opening_x=100.0)
    reseated = _reseat_landing_opening(landing, 120.0, CURVE_RADIUS)
    assert reseated is not None
    assert reseated.join_point == (80.0, 100.0)
    assert reseated.minimum_runway == pytest.approx(40.0)
    assert reseated.opening_turn_coordinate == 120.0


def test_reseat_across_the_join_is_refused_without_carry() -> None:
    landing = _landing(join_x=80.0, opening_x=100.0)
    assert _reseat_landing_opening(landing, 60.0, CURVE_RADIUS) is None


def test_reseat_across_the_join_carries_it_downstream_when_opted_in() -> None:
    landing = _landing(join_x=80.0, opening_x=100.0)
    reseated = _reseat_landing_opening(landing, 60.0, CURVE_RADIUS, carry_join=True)
    assert reseated is not None
    assert reseated.opening_turn_coordinate == 60.0
    assert reseated.minimum_runway == pytest.approx(CURVE_RADIUS)
    assert reseated.join_point == (60.0 - CURVE_RADIUS, 100.0)


def _seat_candidate(
    name: str, coordinate: float, band: ReservedBand
) -> tuple[member_geometry._MemberCandidate, tuple]:
    route = RoutedPath(
        Edge(f"{name}-src", f"{name}-tgt", name),
        name,
        [
            (300.0, coordinate - 60.0),
            (200.0, coordinate - 60.0),
            (200.0, coordinate),
            (100.0, coordinate),
            (100.0, coordinate + 60.0),
            (60.0, coordinate + 60.0),
        ],
    )
    candidate = member_geometry._MemberCandidate(
        route,
        RouteFamilyId.STANDARD_L_SHAPE,
        f"system-{name}",
        f"carrier-{name}",
        (f"connector-{name}",),
    )
    key = (route.edge.source, route.edge.target, name, 2)
    return candidate, (key, band)


def test_pinned_component_clamps_stragglers_alone() -> None:
    # One run below the band and one above it veto any rigid travel; each is
    # clamped into the band alone, landing a full pitch apart.
    band = ReservedBand(110.0, 130.0)
    first, first_claim = _seat_candidate("one", 100.0, band)
    second, second_claim = _seat_candidate("two", 140.0, band)
    second = member_geometry._MemberCandidate(
        second.route,
        second.family_id,
        first.system_id,
        second.carrier_id,
        second.connector_ids,
    )
    ctx = SimpleNamespace(
        reserved_bands=ReservedCorridors(per_claim=dict([first_claim, second_claim])),
        offset_step=4.0,
    )
    member_geometry._seat_claimed_segments_before_freeze((first, second), ctx)
    assert first.route.points[2][1] == pytest.approx(110.0)
    assert second.route.points[2][1] == pytest.approx(130.0)


def _fan_pair(
    tail_points: list[tuple[float, float]],
) -> tuple[RoutedPath, RoutedPath]:
    mover = RoutedPath(
        Edge("junction", "target", "line"),
        "line",
        [(100.0, 50.0), (60.0, 50.0), (60.0, 90.0)],
        is_inter_section=True,
    )
    upstream = RoutedPath(
        Edge("port", "junction", "line"), "line", tail_points, is_inter_section=True
    )
    return mover, upstream


def test_fanned_leg_carries_its_paired_upstream_tail() -> None:
    # The horizontal tail ends on the fanned leg's drawn start, so it is one
    # continuous stroke and both tail waypoints follow the lane change.
    mover, upstream = _fan_pair([(140.0, 30.0), (120.0, 50.0), (100.0, 50.0)])
    start_drawn = mover.points[0]
    normalize._carry_fanned_upstream_tails(mover, [upstream], {}, start_drawn, 8.0)
    assert upstream.points[-2:] == [(120.0, 58.0), (100.0, 58.0)]


def test_fanned_leg_stretches_a_vertical_upstream_tail() -> None:
    # A vertical tail reaches the new lane through its existing corner: only
    # its endpoint moves.
    mover, upstream = _fan_pair([(140.0, 30.0), (100.0, 30.0), (100.0, 50.0)])
    start_drawn = mover.points[0]
    normalize._carry_fanned_upstream_tails(mover, [upstream], {}, start_drawn, 8.0)
    assert upstream.points[-2:] == [(100.0, 30.0), (100.0, 58.0)]


def test_fanned_leg_leaves_a_foreign_seam_alone() -> None:
    # A tail ending away from the fanned leg's start is another pass's seam.
    mover, upstream = _fan_pair([(140.0, 30.0), (120.0, 30.0), (104.0, 30.0)])
    start_drawn = mover.points[0]
    normalize._carry_fanned_upstream_tails(mover, [upstream], {}, start_drawn, 8.0)
    assert upstream.points[-2:] == [(120.0, 30.0), (104.0, 30.0)]
