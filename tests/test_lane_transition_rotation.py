"""Lane-transition templates rotate without changing horizontal geometry."""

from __future__ import annotations

import pytest

from nf_metro.layout.routing.centrelines import route_lane_transition
from nf_metro.layout.routing.common import Direction, OffsetRegime
from nf_metro.parser.model import Edge

EDGE = Edge("source", "target", "line")


@pytest.mark.parametrize(
    ("direction", "source", "target", "expected"),
    (
        (
            Direction.R,
            (0.0, 0.0),
            (30.0, 0.0),
            [(0.0, 0.0), (5.0, 0.0), (9.0, 4.0), (30.0, 4.0)],
        ),
        (
            Direction.L,
            (30.0, 0.0),
            (0.0, 0.0),
            [(30.0, 0.0), (25.0, 0.0), (21.0, 4.0), (0.0, 4.0)],
        ),
        (
            Direction.D,
            (0.0, 0.0),
            (0.0, 30.0),
            [(0.0, 0.0), (0.0, 5.0), (4.0, 9.0), (4.0, 30.0)],
        ),
        (
            Direction.U,
            (0.0, 30.0),
            (0.0, 0.0),
            [(0.0, 30.0), (0.0, 25.0), (4.0, 21.0), (4.0, 0.0)],
        ),
    ),
)
def test_lane_transition_rotates_by_run_direction(
    direction: Direction,
    source: tuple[float, float],
    target: tuple[float, float],
    expected: list[tuple[float, float]],
) -> None:
    route = route_lane_transition(
        EDGE,
        source,
        target,
        source_offset=0.0,
        target_offset=4.0,
        run_direction=direction,
        source_runway=5.0,
        target_runway=5.0,
        diagonal_run=4.0,
        place_at_source=True,
        is_inter_section=False,
    )

    assert route.points == expected
    assert route.offset_regime is OffsetRegime.BAKED
    assert route.normalize_exempt is True
    assert route.is_inter_section is False


@pytest.mark.parametrize(
    ("direction", "source", "target", "expected"),
    (
        (
            Direction.R,
            (0.0, 0.0),
            (30.0, 0.0),
            [(0.0, 0.0), (21.0, 0.0), (25.0, 4.0), (30.0, 4.0)],
        ),
        (
            Direction.L,
            (30.0, 0.0),
            (0.0, 0.0),
            [(30.0, 0.0), (9.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
        ),
        (
            Direction.D,
            (0.0, 0.0),
            (0.0, 30.0),
            [(0.0, 0.0), (0.0, 21.0), (4.0, 25.0), (4.0, 30.0)],
        ),
        (
            Direction.U,
            (0.0, 30.0),
            (0.0, 0.0),
            [(0.0, 30.0), (0.0, 9.0), (4.0, 5.0), (4.0, 0.0)],
        ),
    ),
)
def test_lane_transition_can_place_diagonal_at_target(
    direction: Direction,
    source: tuple[float, float],
    target: tuple[float, float],
    expected: list[tuple[float, float]],
) -> None:
    route = route_lane_transition(
        EDGE,
        source,
        target,
        source_offset=0.0,
        target_offset=4.0,
        run_direction=direction,
        source_runway=5.0,
        target_runway=5.0,
        diagonal_run=4.0,
        place_at_source=False,
        is_inter_section=True,
    )

    assert route.points == expected
    assert route.is_inter_section is True


@pytest.mark.parametrize(
    ("direction", "source", "target", "source_offset", "target_offset", "kwargs"),
    (
        (
            Direction.R,
            (0.0, 0.0),
            (30.0, 0.0),
            0.0,
            4.0,
            {"source_runway": 0.0, "target_runway": 5.0, "diagonal_run": 4.0},
        ),
        (
            Direction.D,
            (0.0, 0.0),
            (0.0, 30.0),
            0.0,
            4.0,
            {"source_runway": 5.0, "target_runway": 0.0, "diagonal_run": 4.0},
        ),
        (
            Direction.L,
            (30.0, 0.0),
            (0.0, 0.0),
            0.0,
            4.0,
            {"source_runway": 5.0, "target_runway": 5.0, "diagonal_run": 0.0},
        ),
        (
            Direction.U,
            (0.0, 30.0),
            (0.0, 0.0),
            0.0,
            6.0,
            {"source_runway": 5.0, "target_runway": 5.0, "diagonal_run": 4.0},
        ),
        (
            Direction.R,
            (0.0, 0.0),
            (12.0, 0.0),
            0.0,
            4.0,
            {"source_runway": 5.0, "target_runway": 5.0, "diagonal_run": 4.0},
        ),
        (
            Direction.D,
            (0.0, 30.0),
            (0.0, 0.0),
            0.0,
            4.0,
            {"source_runway": 5.0, "target_runway": 5.0, "diagonal_run": 4.0},
        ),
    ),
)
def test_lane_transition_rejects_inconsistent_geometry(
    direction: Direction,
    source: tuple[float, float],
    target: tuple[float, float],
    source_offset: float,
    target_offset: float,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(
        ValueError, match="lane-transition template inputs are inconsistent"
    ):
        route_lane_transition(
            EDGE,
            source,
            target,
            source_offset=source_offset,
            target_offset=target_offset,
            run_direction=direction,
            place_at_source=True,
            is_inter_section=False,
            **kwargs,
        )
