"""Corridor cohorts allocate from realised geometry and frozen witnesses."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

from nf_metro.layout.routing.corridor_cohorts import (
    CorridorAllocationFailureReason,
    CorridorAllocationProblem,
    CorridorAllocationStatus,
    CorridorEquality,
    CorridorLane,
    CorridorObstacle,
    solve_corridor_cohorts,
)


def _lane(
    member_id: str,
    corridor_owner_id: str,
    endpoint_owner_id: str,
    boundary_coordinate: float,
    planned_coordinate: float,
    *,
    span: tuple[float, float],
    line_rank: int,
    root_rank: int = 0,
) -> CorridorLane:
    return CorridorLane(
        member_id=member_id,
        cohort_id=corridor_owner_id,
        endpoint_owner_id=endpoint_owner_id,
        boundary_coordinate=boundary_coordinate,
        planned_coordinate=planned_coordinate,
        span_start=span[0],
        span_end=span[1],
        semantic_rank=(root_rank, line_rank),
    )


def _obstacle(
    obstacle_id: str,
    order_coordinate: float,
    realised_coordinate: float,
    *,
    span: tuple[float, float],
    semantic_rank: tuple[int, ...],
) -> CorridorObstacle:
    return CorridorObstacle(
        obstacle_id=obstacle_id,
        order_coordinate=order_coordinate,
        realised_coordinate=realised_coordinate,
        span_start=span[0],
        span_end=span[1],
        semantic_rank=semantic_rank,
    )


def _allocations(problem: CorridorAllocationProblem) -> dict[str, float]:
    result = solve_corridor_cohorts(problem)
    assert result.status is CorridorAllocationStatus.PLANNED
    assert result.reason is None
    return dict(result.allocations)


def _seed77_problem(
    lanes: tuple[CorridorLane, ...] | None = None,
) -> CorridorAllocationProblem:
    if lanes is None:
        lanes = (
            _lane(
                "s9:l1",
                "corridor:s9",
                "endpoint:s9",
                478.0,
                558.0,
                span=(5.0, 15.0),
                line_rank=1,
                root_rank=0,
            ),
            _lane(
                "s9:l3",
                "corridor:s9",
                "endpoint:s9",
                482.0,
                554.0,
                span=(5.0, 15.0),
                line_rank=3,
                root_rank=0,
            ),
            _lane(
                "s17:l0",
                "corridor:s17",
                "endpoint:s17",
                478.0,
                550.0,
                span=(0.0, 10.0),
                line_rank=0,
                root_rank=1,
            ),
            _lane(
                "s17:l3",
                "corridor:s17",
                "endpoint:s17",
                482.0,
                554.0,
                span=(20.0, 30.0),
                line_rank=3,
                root_rank=1,
            ),
        )
    return CorridorAllocationProblem(
        lanes=lanes,
        obstacles=(
            _obstacle(
                "realised:s10",
                550.0,
                554.0,
                span=(0.0, 10.0),
                semantic_rank=(2, 0),
            ),
        ),
        clearance=4.0,
    )


SEED77_EXPECTED = {
    "s9:l1": 542.0,
    "s9:l3": 546.0,
    "s17:l0": 550.0,
    "s17:l3": 554.0,
}


def test_seed77_shape_uses_boundary_offsets_and_realised_obstacle() -> None:
    assert _allocations(_seed77_problem()) == SEED77_EXPECTED


def test_seed77_all_lane_permutations_have_identical_allocations() -> None:
    lanes = _seed77_problem().lanes

    for order in permutations(lanes):
        assert _allocations(_seed77_problem(order)) == SEED77_EXPECTED


def test_allocation_is_invariant_under_coordinate_translation() -> None:
    offset = 80.0
    problem = _seed77_problem()
    translated = replace(
        problem,
        lanes=tuple(
            replace(
                lane,
                boundary_coordinate=lane.boundary_coordinate + offset,
                planned_coordinate=lane.planned_coordinate + offset,
            )
            for lane in problem.lanes
        ),
        obstacles=tuple(
            replace(
                obstacle,
                order_coordinate=obstacle.order_coordinate + offset,
                realised_coordinate=obstacle.realised_coordinate + offset,
            )
            for obstacle in problem.obstacles
        ),
    )

    assert _allocations(translated) == {
        member_id: coordinate + offset
        for member_id, coordinate in SEED77_EXPECTED.items()
    }


def test_planned_coordinates_do_not_reorder_boundary_witnesses() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "early",
                "corridor:ordered",
                "endpoint:ordered",
                10.0,
                104.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "late",
                "corridor:ordered",
                "endpoint:ordered",
                14.0,
                100.0,
                span=(0.0, 10.0),
                line_rank=1,
            ),
        )
    )

    allocations = _allocations(problem)

    assert allocations["late"] - allocations["early"] == 4.0


def test_same_line_label_does_not_join_distinct_corridor_owners() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "left:l3",
                "corridor:left",
                "endpoint:left",
                10.0,
                100.0,
                span=(0.0, 10.0),
                line_rank=3,
                root_rank=0,
            ),
            _lane(
                "right:l3",
                "corridor:right",
                "endpoint:right",
                10.0,
                200.0,
                span=(20.0, 30.0),
                line_rank=3,
                root_rank=1,
            ),
        )
    )

    assert _allocations(problem) == {"left:l3": 100.0, "right:l3": 200.0}


def test_explicit_equalities_do_not_collide_with_their_own_members() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "left",
                "corridor:left",
                "endpoint:left",
                10.0,
                10.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "right",
                "corridor:right",
                "endpoint:right",
                10.0,
                10.0,
                span=(0.0, 10.0),
                line_rank=1,
            ),
        ),
        equalities=(CorridorEquality("network:coincident", "left", "right", 0.0),),
    )

    assert _allocations(problem) == {"left": 10.0, "right": 10.0}


def test_cohort_chooses_one_obstacle_side_for_every_lane() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "low",
                "corridor:paired",
                "endpoint:paired",
                0.0,
                8.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "high",
                "corridor:paired",
                "endpoint:paired",
                4.0,
                12.0,
                span=(0.0, 10.0),
                line_rank=1,
            ),
        ),
        obstacles=(
            _obstacle(
                "obstacle:middle",
                10.0,
                10.0,
                span=(0.0, 10.0),
                semantic_rank=(1, 0),
            ),
        ),
        clearance=4.0,
    )

    assert _allocations(problem) == {"low": 2.0, "high": 6.0}


def test_realised_obstacle_motion_cannot_flip_its_planned_order_side() -> None:
    lane = _lane(
        "lane",
        "corridor:lane",
        "endpoint:lane",
        20.0,
        20.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=1,
    )
    before = _obstacle(
        "obstacle",
        20.0,
        18.0,
        span=(0.0, 10.0),
        semantic_rank=(0, 0),
    )
    after = replace(before, realised_coordinate=22.0)

    before_allocation = _allocations(
        CorridorAllocationProblem(lanes=(lane,), obstacles=(before,))
    )
    after_allocation = _allocations(
        CorridorAllocationProblem(lanes=(lane,), obstacles=(after,))
    )

    assert before_allocation == {"lane": 22.0}
    assert after_allocation == {"lane": 26.0}


def test_one_cohort_cannot_span_distinct_endpoint_owners() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "a",
                "corridor:shared",
                "endpoint:a",
                10.0,
                10.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "b",
                "corridor:shared",
                "endpoint:b",
                14.0,
                14.0,
                span=(20.0, 30.0),
                line_rank=1,
            ),
        )
    )

    result = solve_corridor_cohorts(problem)

    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.INVALID
    assert result.allocations == ()
    assert result.blocking_member_ids == ("a", "b")
    assert result.blocking_endpoint_owner_ids == ("endpoint:a", "endpoint:b")


def test_empty_lane_or_obstacle_semantic_rank_is_invalid() -> None:
    rankless_lane = CorridorLane(
        member_id="rankless",
        cohort_id="corridor:rankless",
        endpoint_owner_id="endpoint:rankless",
        boundary_coordinate=10.0,
        planned_coordinate=10.0,
        span_start=0.0,
        span_end=10.0,
        semantic_rank=(),
    )
    lane_result = solve_corridor_cohorts(
        CorridorAllocationProblem(lanes=(rankless_lane,))
    )

    assert lane_result.status is CorridorAllocationStatus.FAILURE
    assert lane_result.reason is CorridorAllocationFailureReason.INVALID
    assert lane_result.blocking_member_ids == ("rankless",)

    ranked_lane = replace(rankless_lane, semantic_rank=(0, 0))
    rankless_obstacle = _obstacle(
        "rankless-obstacle",
        10.0,
        10.0,
        span=(0.0, 10.0),
        semantic_rank=(),
    )
    obstacle_result = solve_corridor_cohorts(
        CorridorAllocationProblem(lanes=(ranked_lane,), obstacles=(rankless_obstacle,))
    )

    assert obstacle_result.status is CorridorAllocationStatus.FAILURE
    assert obstacle_result.reason is CorridorAllocationFailureReason.INVALID
    assert obstacle_result.blocking_obstacle_ids == ("rankless-obstacle",)


def test_spans_touching_at_one_endpoint_do_not_overlap() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "before",
                "corridor:before",
                "endpoint:before",
                10.0,
                10.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "after",
                "corridor:after",
                "endpoint:after",
                10.0,
                10.0,
                span=(10.0, 20.0),
                line_rank=1,
            ),
        )
    )

    assert _allocations(problem) == {"before": 10.0, "after": 10.0}


def test_exact_obstacle_tie_is_equivariant_under_axis_reflection() -> None:
    lane = _lane(
        "lane",
        "corridor:lane",
        "endpoint:lane",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=1,
    )
    obstacle = _obstacle(
        "obstacle",
        10.0,
        10.0,
        span=(0.0, 10.0),
        semantic_rank=(0, 0),
    )
    forward = _allocations(
        CorridorAllocationProblem(lanes=(lane,), obstacles=(obstacle,), axis_sign=1)
    )
    reflected_lane = CorridorLane(
        member_id=lane.member_id,
        cohort_id=lane.cohort_id,
        endpoint_owner_id=lane.endpoint_owner_id,
        boundary_coordinate=-lane.boundary_coordinate,
        planned_coordinate=-lane.planned_coordinate,
        span_start=lane.span_start,
        span_end=lane.span_end,
        semantic_rank=lane.semantic_rank,
    )
    reflected_obstacle = _obstacle(
        obstacle.obstacle_id,
        -obstacle.order_coordinate,
        -obstacle.realised_coordinate,
        span=(obstacle.span_start, obstacle.span_end),
        semantic_rank=obstacle.semantic_rank,
    )

    reflected = _allocations(
        CorridorAllocationProblem(
            lanes=(reflected_lane,), obstacles=(reflected_obstacle,), axis_sign=-1
        )
    )

    assert reflected == {
        member_id: -coordinate for member_id, coordinate in forward.items()
    }


def test_incomplete_witnesses_return_whole_problem_to_compatibility() -> None:
    problem = CorridorAllocationProblem(
        lanes=(
            _lane(
                "known",
                "corridor:partial",
                "endpoint:partial",
                10.0,
                10.0,
                span=(0.0, 10.0),
                line_rank=1,
            ),
        ),
        witnesses_complete=False,
    )

    result = solve_corridor_cohorts(problem)

    assert result.status is CorridorAllocationStatus.COMPATIBILITY
    assert result.reason is None
    assert result.allocations == ()


def test_contradictory_equalities_are_attributed_without_mutating_input() -> None:
    lanes = (
        _lane(
            "a",
            "corridor:a",
            "endpoint:a",
            10.0,
            10.0,
            span=(0.0, 10.0),
            line_rank=1,
        ),
        _lane(
            "b",
            "corridor:b",
            "endpoint:b",
            14.0,
            14.0,
            span=(0.0, 10.0),
            line_rank=2,
        ),
    )
    equalities = (
        CorridorEquality("network:ab", "a", "b", 2.0),
        CorridorEquality("network:ab", "a", "b", 4.0),
    )
    problem = CorridorAllocationProblem(lanes=lanes, equalities=equalities)
    before = (problem.lanes, problem.obstacles, problem.equalities)

    result = solve_corridor_cohorts(problem)

    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.CONTRADICTION
    assert result.allocations == ()
    assert result.blocking_member_ids == ("a", "b")
    assert result.blocking_obstacle_ids == ()
    assert result.blocking_equality_owner_ids == ("network:ab",)
    assert (problem.lanes, problem.obstacles, problem.equalities) == before
