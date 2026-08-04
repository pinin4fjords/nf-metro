"""Cardinal orientation helpers shared by route planners and emitters."""

from __future__ import annotations

from typing import TypeAlias

from nf_metro.layout.route_plan import DemandAxis, turn_handedness
from nf_metro.layout.routing.common import Direction, right_normal_axis_sign

__all__ = (
    "direction_axis",
    "direction_vector",
    "get_point_coordinate",
    "heading_for_axis_delta",
    "lateral_axis",
    "lateral_order_sign",
    "turn_handedness",
)

Point: TypeAlias = tuple[float, float]
Vector: TypeAlias = tuple[int, int]


def direction_axis(direction: Direction) -> DemandAxis:
    """Return the canvas axis along which *direction* travels."""
    if direction in (Direction.R, Direction.L):
        return DemandAxis.X
    return DemandAxis.Y


def lateral_axis(direction: Direction) -> DemandAxis:
    """Return the canvas axis perpendicular to *direction*."""
    if direction_axis(direction) is DemandAxis.X:
        return DemandAxis.Y
    return DemandAxis.X


def direction_vector(direction: Direction) -> Vector:
    """Return the unit screen-space vector for *direction*."""
    return {
        Direction.R: (1, 0),
        Direction.L: (-1, 0),
        Direction.U: (0, -1),
        Direction.D: (0, 1),
    }[direction]


def heading_for_axis_delta(axis: DemandAxis, delta: float) -> Direction:
    """Return the cardinal heading for a signed displacement on *axis*."""
    if axis is DemandAxis.X:
        return Direction.R if delta > 0 else Direction.L
    return Direction.D if delta > 0 else Direction.U


def lateral_order_sign(direction: Direction) -> int:
    """Return the screen-axis sign of the right-hand normal to *direction*."""
    return right_normal_axis_sign(direction)


def get_point_coordinate(point: Point, axis: DemandAxis) -> float:
    """Read one X or Y coordinate from *point*."""
    if axis is DemandAxis.X:
        return point[0]
    if axis is DemandAxis.Y:
        return point[1]
    raise ValueError("a point coordinate requires the X or Y axis")
