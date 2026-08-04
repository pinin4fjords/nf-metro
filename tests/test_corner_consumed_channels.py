"""Rounded corners and straight corridor ownership share one boundary."""

from __future__ import annotations

import pytest

from nf_metro.layout.constants import CURVE_RADIUS
from nf_metro.layout.route_reservations import (
    ColumnGapRegion,
    CorridorOrientation,
    _maximal_axis_segments,
    _segment_owns_gap_corridor,
)
from nf_metro.layout.routing import invariants
from nf_metro.layout.routing.common import Edge, RoutedPath
from nf_metro.layout.routing.corners import axis_segment_has_straight_run


def _corner_route(vertical_length: float) -> RoutedPath:
    return RoutedPath(
        edge=Edge(source="a", target="b", line_id="x"),
        line_id="x",
        points=[
            (-CURVE_RADIUS, 0.0),
            (0.0, 0.0),
            (0.0, vertical_length),
        ],
        is_inter_section=True,
        curve_radii=[CURVE_RADIUS],
    )


@pytest.mark.parametrize(
    ("vertical_length", "has_straight_run"),
    [
        (CURVE_RADIUS, False),
        (CURVE_RADIUS + 1.0, True),
    ],
)
def test_effective_corner_radius_is_the_straight_run_boundary(
    vertical_length: float,
    has_straight_run: bool,
) -> None:
    route = _corner_route(vertical_length)
    assert (
        axis_segment_has_straight_run(route.points, route.curve_radii, 1, 2)
        is has_straight_run
    )


@pytest.mark.parametrize(
    ("vertical_length", "reserved"),
    [
        (CURVE_RADIUS, False),
        (CURVE_RADIUS + 1.0, True),
    ],
)
def test_reservation_discovery_owns_only_a_remaining_straight_run(
    vertical_length: float,
    reserved: bool,
) -> None:
    route = _corner_route(vertical_length)
    segment = next(
        segment
        for segment in _maximal_axis_segments(route.points)
        if segment.orientation is CorridorOrientation.VERTICAL
    )
    assert (
        _segment_owns_gap_corridor(
            segment,
            ColumnGapRegion(0, 1),
            route.points,
            route.curve_radii,
        )
        is reserved
    )


@pytest.mark.parametrize(
    ("vertical_length", "violation_count"),
    [
        (CURVE_RADIUS, 0),
        (CURVE_RADIUS + 1.0, 1),
    ],
)
def test_gap_materialization_owns_only_a_remaining_straight_run(
    monkeypatch: pytest.MonkeyPatch,
    vertical_length: float,
    violation_count: int,
) -> None:
    route = _corner_route(vertical_length)
    monkeypatch.setattr(invariants, "gap_lookup_geometry", lambda _graph: object())
    monkeypatch.setattr(
        invariants,
        "gap_lo_for_x",
        lambda *_args, **_kwargs: (0, None),
    )
    violations = invariants.check_gap_channels_materialized(None, [route])  # type: ignore[arg-type]
    assert len(violations) == violation_count
