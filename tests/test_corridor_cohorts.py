"""Corridor cohorts allocate from realised geometry and frozen witnesses."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
from itertools import permutations
from types import SimpleNamespace

import pytest

from nf_metro.layout.route_reservations import (
    ColumnGapRegion,
    CorridorOrientation,
    RowGapRegion,
)
from nf_metro.layout.routing.common import Direction
from nf_metro.layout.routing.corridor_cohort_integration import (
    CorridorCohortClaimRole,
    CorridorCohortCompilationError,
    CorridorCohortLedger,
    CorridorCohortLedgerClaim,
    CorridorCohortTarget,
    CorridorScalarOwnerKind,
    CorridorScalarRequest,
    CorridorScalarVariable,
    _destination_claim_axis_sign,
    build_corridor_footprint_witnesses,
    compile_corridor_cohort_plan,
)
from nf_metro.layout.routing.corridor_cohorts import (
    CorridorAllocationFailureReason,
    CorridorAllocationProblem,
    CorridorAllocationStatus,
    CorridorCoordinateDomain,
    CorridorDirectedSeparation,
    CorridorEquality,
    CorridorFixedEquality,
    CorridorForbiddenInterval,
    CorridorLane,
    CorridorObstacle,
    CorridorSeparation,
    solve_corridor_cohorts,
)
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.parser.model import PortSide


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


def test_fixed_equalities_pin_a_complete_rigid_bundle() -> None:
    lanes = tuple(
        _lane(
            f"lane:{rank}",
            "corridor",
            "endpoint",
            -4.0 * rank,
            12.0 - 4.0 * rank,
            span=(0.0, 10.0),
            line_rank=rank,
        )
        for rank in range(3)
    )
    obstacles = tuple(
        _obstacle(
            f"fixed:{rank}",
            12.0 - 4.0 * rank,
            12.0 - 4.0 * rank,
            span=(0.0, 10.0),
            semantic_rank=(rank,),
        )
        for rank in range(3)
    )
    problem = CorridorAllocationProblem(
        lanes=lanes,
        obstacles=obstacles,
        separations=tuple(
            CorridorSeparation(
                lane.member_id,
                obstacle.obstacle_id,
                (
                    0.0
                    if lane.member_id.split(":")[-1]
                    == obstacle.obstacle_id.split(":")[-1]
                    else 4.0
                ),
            )
            for lane in lanes
            for obstacle in obstacles
        ),
        fixed_equalities=tuple(
            CorridorFixedEquality(
                f"reservation-lane:{rank}",
                f"lane:{rank}",
                f"fixed:{rank}",
            )
            for rank in range(3)
        ),
    )

    expected = {
        "lane:0": 12.0,
        "lane:1": 8.0,
        "lane:2": 4.0,
    }
    assert _allocations(replace(problem, fixed_equalities=())) == expected
    assert _allocations(problem) == expected


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


def test_directed_separation_can_reverse_default_semantic_order() -> None:
    lanes = (
        _lane(
            "semantic-first",
            "corridor:first",
            "endpoint:first",
            0.0,
            10.0,
            span=(0.0, 10.0),
            line_rank=0,
        ),
        _lane(
            "semantic-second",
            "corridor:second",
            "endpoint:second",
            0.0,
            20.0,
            span=(0.0, 10.0),
            line_rank=1,
        ),
    )
    problem = CorridorAllocationProblem(
        lanes,
        directed_separations=(
            CorridorDirectedSeparation(
                "controlled-lead-order",
                "semantic-second",
                "semantic-first",
                4.0,
            ),
        ),
    )

    allocations = _allocations(problem)

    assert allocations["semantic-first"] - allocations["semantic-second"] == 4.0


def test_directed_separation_is_invariant_under_input_permutations() -> None:
    lanes = (
        _lane(
            "a",
            "corridor:a",
            "endpoint:a",
            0.0,
            30.0,
            span=(0.0, 10.0),
            line_rank=0,
        ),
        _lane(
            "b",
            "corridor:b",
            "endpoint:b",
            0.0,
            20.0,
            span=(10.0, 20.0),
            line_rank=1,
        ),
        _lane(
            "c",
            "corridor:c",
            "endpoint:c",
            0.0,
            10.0,
            span=(20.0, 30.0),
            line_rank=2,
        ),
    )
    separations = (
        CorridorDirectedSeparation("order:ba", "b", "a", 4.0),
        CorridorDirectedSeparation("order:cb", "c", "b", 4.0),
    )
    expected = None
    for lane_order in permutations(lanes):
        for separation_order in permutations(separations):
            allocations = _allocations(
                CorridorAllocationProblem(
                    lane_order,
                    directed_separations=separation_order,
                )
            )
            expected = allocations if expected is None else expected
            assert allocations == expected


def test_directed_separation_cycle_fails_with_owner_provenance() -> None:
    lanes = tuple(
        _lane(
            member_id,
            f"corridor:{member_id}",
            f"endpoint:{member_id}",
            0.0,
            coordinate,
            span=(rank * 10.0, (rank + 1) * 10.0),
            line_rank=rank,
        )
        for rank, (member_id, coordinate) in enumerate(
            (("a", 0.0), ("b", 10.0), ("c", 20.0))
        )
    )
    problem = CorridorAllocationProblem(
        lanes,
        directed_separations=(
            CorridorDirectedSeparation("order:ab", "a", "b", 4.0),
            CorridorDirectedSeparation("order:bc", "b", "c", 4.0),
            CorridorDirectedSeparation("order:ca", "c", "a", 4.0),
        ),
    )

    result = solve_corridor_cohorts(problem)

    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.INFEASIBLE
    assert result.allocations == ()
    assert result.blocking_member_ids == ("a", "b", "c")
    assert result.blocking_order_owner_ids == ("order:ab", "order:bc", "order:ca")
    assert result.blocking_endpoint_owner_ids == (
        "endpoint:a",
        "endpoint:b",
        "endpoint:c",
    )


def test_forbidden_coordinate_interval_chooses_one_deterministic_side() -> None:
    lane = _lane(
        "movable",
        "corridor:movable",
        "endpoint:movable",
        0.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=1,
    )
    problem = CorridorAllocationProblem(
        (lane,),
        forbidden_intervals=(
            CorridorForbiddenInterval(
                "movable",
                "fixed-perpendicular-footprint",
                8.0,
                12.0,
                (0, 0),
            ),
        ),
    )

    assert _allocations(problem) == {"movable": 12.0}


def test_forbidden_coordinate_interval_is_equivariant_under_axis_reflection() -> None:
    lane = _lane(
        "movable",
        "corridor:movable",
        "endpoint:movable",
        0.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=1,
    )
    interval = CorridorForbiddenInterval(
        "movable",
        "fixed-perpendicular-footprint",
        8.0,
        12.0,
        (0, 0),
    )
    reflected_lane = replace(
        lane,
        boundary_coordinate=-lane.boundary_coordinate,
        planned_coordinate=-lane.planned_coordinate,
    )
    reflected_interval = replace(
        interval,
        minimum_coordinate=-interval.maximum_coordinate,
        maximum_coordinate=-interval.minimum_coordinate,
    )

    original = _allocations(
        CorridorAllocationProblem((lane,), forbidden_intervals=(interval,))
    )
    reflected = _allocations(
        CorridorAllocationProblem(
            (reflected_lane,),
            forbidden_intervals=(reflected_interval,),
            axis_sign=-1,
        )
    )

    assert reflected == {
        member_id: -coordinate for member_id, coordinate in original.items()
    }


def test_infeasible_forbidden_interval_reports_its_named_obstacle() -> None:
    lane = _lane(
        "movable",
        "corridor:movable",
        "endpoint:movable",
        0.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
    )
    problem = CorridorAllocationProblem(
        (lane,),
        domains=(
            CorridorCoordinateDomain(
                "movable",
                minimum_coordinate=9.0,
                maximum_coordinate=11.0,
            ),
        ),
        forbidden_intervals=(
            CorridorForbiddenInterval(
                "movable",
                "fixed-perpendicular-footprint",
                8.0,
                12.0,
                (0, 0),
            ),
        ),
    )

    result = solve_corridor_cohorts(problem)

    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.INFEASIBLE
    assert result.blocking_member_ids == ("movable",)
    assert result.blocking_obstacle_ids == ("fixed-perpendicular-footprint",)


def test_opposite_running_members_are_not_implicitly_bundled_by_ordering() -> None:
    problem = CorridorAllocationProblem(
        (
            _lane(
                "right-running",
                "corridor:direction:R",
                "endpoint:right-running",
                0.0,
                0.0,
                span=(0.0, 10.0),
                line_rank=0,
            ),
            _lane(
                "left-running",
                "corridor:direction:L",
                "endpoint:left-running",
                0.0,
                10.0,
                span=(10.0, 20.0),
                line_rank=1,
            ),
        ),
        directed_separations=(
            CorridorDirectedSeparation(
                "counter-running-order",
                "right-running",
                "left-running",
                4.0,
            ),
        ),
    )

    allocations = _allocations(problem)

    assert allocations["left-running"] - allocations["right-running"] >= 4.0
    assert allocations["left-running"] != allocations["right-running"]


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


def test_realised_obstacle_coordinate_is_the_final_clearance_constraint() -> None:
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
    assert after_allocation == {"lane": 18.0}


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


def test_fixed_obstacle_side_is_chosen_by_semantic_order() -> None:
    lane = _lane(
        "lane",
        "corridor:lane",
        "endpoint:lane",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=0,
    )
    obstacle = _obstacle(
        "obstacle",
        10.0,
        10.0,
        span=(0.0, 10.0),
        semantic_rank=(1, 0),
    )

    assert _allocations(
        CorridorAllocationProblem(lanes=(lane,), obstacles=(obstacle,))
    ) == {"lane": 6.0}


def test_exact_clearance_boundary_preserves_preferred_coordinate() -> None:
    lane = _lane(
        "lane",
        "corridor:lane",
        "endpoint:lane",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
    )
    obstacle = _obstacle(
        "obstacle",
        14.0,
        14.0,
        span=(0.0, 10.0),
        semantic_rank=(1, 0),
    )

    assert _allocations(
        CorridorAllocationProblem(lanes=(lane,), obstacles=(obstacle,))
    ) == {"lane": 10.0}


def test_root_dag_resolves_obstacle_sides_as_one_plan() -> None:
    lanes = (
        _lane(
            "early",
            "corridor:early",
            "endpoint:early",
            0.0,
            10.0,
            span=(0.0, 10.0),
            line_rank=0,
            root_rank=0,
        ),
        _lane(
            "middle",
            "corridor:middle",
            "endpoint:middle",
            0.0,
            10.0,
            span=(5.0, 15.0),
            line_rank=0,
            root_rank=1,
        ),
        _lane(
            "late",
            "corridor:late",
            "endpoint:late",
            0.0,
            10.0,
            span=(10.0, 20.0),
            line_rank=0,
            root_rank=2,
        ),
    )
    obstacle = _obstacle(
        "fixed",
        14.0,
        14.0,
        span=(0.0, 20.0),
        semantic_rank=(3, 0),
    )

    problem = CorridorAllocationProblem(lanes=lanes, obstacles=(obstacle,))

    assert _allocations(problem) == {"early": 2.0, "middle": 6.0, "late": 10.0}
    for order in permutations(lanes):
        assert _allocations(replace(problem, lanes=order)) == {
            "early": 2.0,
            "middle": 6.0,
            "late": 10.0,
        }


def test_pair_separation_is_permutation_and_reflection_invariant() -> None:
    lanes = (
        _lane(
            "early",
            "corridor:early",
            "endpoint:early",
            0.0,
            0.0,
            span=(0.0, 10.0),
            line_rank=0,
            root_rank=0,
        ),
        _lane(
            "late",
            "corridor:late",
            "endpoint:late",
            4.0,
            4.0,
            span=(0.0, 10.0),
            line_rank=0,
            root_rank=1,
        ),
    )
    separation = CorridorSeparation("early", "late", 12.0)
    expected = {"early": -8.0, "late": 4.0}

    for order in permutations(lanes):
        assert (
            _allocations(
                CorridorAllocationProblem(
                    lanes=order,
                    separations=(separation,),
                )
            )
            == expected
        )

    exact = tuple(
        replace(lane, planned_coordinate=expected[lane.member_id]) for lane in lanes
    )
    assert (
        _allocations(CorridorAllocationProblem(lanes=exact, separations=(separation,)))
        == expected
    )

    reflected = tuple(
        replace(
            lane,
            boundary_coordinate=-lane.boundary_coordinate,
            planned_coordinate=-lane.planned_coordinate,
        )
        for lane in lanes
    )
    assert _allocations(
        CorridorAllocationProblem(
            lanes=reflected,
            separations=(separation,),
            axis_sign=-1,
        )
    ) == {member_id: -coordinate for member_id, coordinate in expected.items()}


def test_pair_separation_controls_fixed_obstacle_clearance() -> None:
    lane = _lane(
        "lane",
        "corridor:lane",
        "endpoint:lane",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
        root_rank=0,
    )
    obstacle = _obstacle(
        "obstacle",
        14.0,
        14.0,
        span=(0.0, 10.0),
        semantic_rank=(1, 0),
    )
    separation = CorridorSeparation("lane", "obstacle", 12.0)

    assert _allocations(
        CorridorAllocationProblem(
            lanes=(lane,),
            obstacles=(obstacle,),
            separations=(separation,),
        )
    ) == {"lane": 2.0}
    assert _allocations(
        CorridorAllocationProblem(
            lanes=(replace(lane, planned_coordinate=2.0),),
            obstacles=(obstacle,),
            separations=(separation,),
        )
    ) == {"lane": 2.0}


def test_zero_pair_separation_allows_same_track_coordinates() -> None:
    lanes = (
        _lane(
            "first",
            "corridor:first",
            "endpoint:first",
            10.0,
            10.0,
            span=(0.0, 10.0),
            line_rank=0,
            root_rank=0,
        ),
        _lane(
            "second",
            "corridor:second",
            "endpoint:second",
            10.0,
            10.0,
            span=(0.0, 10.0),
            line_rank=0,
            root_rank=1,
        ),
    )

    assert _allocations(
        CorridorAllocationProblem(
            lanes=lanes,
            separations=(CorridorSeparation("first", "second", 0.0),),
        )
    ) == {"first": 10.0, "second": 10.0}


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


def test_coordinate_domains_constrain_the_joint_solve_and_attribute_conflicts() -> None:
    lane = _lane(
        "movable",
        "cohort",
        "endpoint",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
    )
    bounded = solve_corridor_cohorts(
        CorridorAllocationProblem(
            (lane,),
            domains=(CorridorCoordinateDomain("movable", maximum_coordinate=8.0),),
        )
    )

    assert bounded.status is CorridorAllocationStatus.PLANNED
    assert bounded.allocations == (("movable", 8.0),)

    conflicting = solve_corridor_cohorts(
        CorridorAllocationProblem(
            (lane,),
            domains=(
                CorridorCoordinateDomain(
                    "movable",
                    minimum_coordinate=9.0,
                    obstacle_ids=("left-lead",),
                ),
                CorridorCoordinateDomain(
                    "movable",
                    maximum_coordinate=8.0,
                    obstacle_ids=("right-lead",),
                ),
            ),
        )
    )

    assert conflicting.status is CorridorAllocationStatus.FAILURE
    assert conflicting.reason is CorridorAllocationFailureReason.INFEASIBLE
    assert conflicting.blocking_member_ids == ("movable",)
    assert conflicting.blocking_obstacle_ids == ("left-lead", "right-lead")


def test_coordinate_domains_report_order_propagation_conflicts() -> None:
    lanes = (
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
            5.0,
            5.0,
            span=(0.0, 10.0),
            line_rank=1,
        ),
    )

    domains = (
        CorridorCoordinateDomain("before", minimum_coordinate=10.0),
        CorridorCoordinateDomain(
            "after",
            maximum_coordinate=5.0,
            obstacle_ids=("positive-side-blocker",),
        ),
        CorridorCoordinateDomain(
            "after",
            maximum_coordinate=100.0,
            obstacle_ids=("inactive-blocker",),
        ),
    )
    results = tuple(
        solve_corridor_cohorts(
            CorridorAllocationProblem(
                ordered_lanes,
                domains=domains,
                coordinate_axis=1,
            )
        )
        for ordered_lanes in (lanes, tuple(reversed(lanes)))
    )

    assert results[0] == results[1]
    result = results[0]
    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.INFEASIBLE
    assert result.allocations == ()
    assert result.clearance_shortfall is not None
    assert result.clearance_shortfall.claim_ids == ("before",)
    assert result.clearance_shortfall.blocking_obstacle_ids == (
        "positive-side-blocker",
    )
    assert result.clearance_shortfall.deficit == 9.0
    assert result.clearance_shortfall.axis == 1
    assert result.clearance_shortfall.required_shift_sign == 1


def test_non_boundary_domain_conflict_has_no_clearance_shortfall() -> None:
    lane = _lane(
        "movable",
        "cohort",
        "endpoint",
        10.0,
        10.0,
        span=(0.0, 10.0),
        line_rank=0,
    )

    result = solve_corridor_cohorts(
        CorridorAllocationProblem(
            (lane,),
            domains=(
                CorridorCoordinateDomain("movable", minimum_coordinate=10.0),
                CorridorCoordinateDomain("movable", maximum_coordinate=5.0),
            ),
        )
    )

    assert result.status is CorridorAllocationStatus.FAILURE
    assert result.reason is CorridorAllocationFailureReason.INFEASIBLE
    assert result.clearance_shortfall is None


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


def _ledger_claim(
    member_id: str,
    *,
    claim_rank: int,
    coordinate_rank: int,
    endpoint_cohort_id: str | None,
    direction: Direction = Direction.R,
    reservation_id: str | None = None,
    lane_rank: int = 0,
    network_id: str | None = None,
    complete: bool = True,
    line_id: str = "line",
    orientation: CorridorOrientation = CorridorOrientation.HORIZONTAL,
    destination_boundary_carrier: bool | None = None,
    destination_boundary_axis_sign: int | None = None,
) -> CorridorCohortLedgerClaim:
    edge_key = (f"{member_id}:source", f"{member_id}:target", line_id)
    return CorridorCohortLedgerClaim(
        claim_id=f"claim:{claim_rank}",
        reservation_id=reservation_id or f"reservation:{claim_rank}",
        reservation_rank=claim_rank,
        claim_rank=0,
        region=(
            RowGapRegion(0, 1)
            if orientation is CorridorOrientation.HORIZONTAL
            else ColumnGapRegion(0, 1)
        ),
        orientation=orientation,
        direction=direction,
        lane_rank=lane_rank,
        member_id=member_id,
        member_geometry_plan_id=f"plan:{member_id}",
        edge_key=edge_key,
        family_id=RouteFamilyId.SAME_Y_STRAIGHT,
        connector_ids=(f"connector:{member_id}",),
        segment_rank=0,
        path_rank=claim_rank,
        endpoint_cohort_id=endpoint_cohort_id,
        endpoint_network_rank=(
            coordinate_rank if endpoint_cohort_id is not None else None
        ),
        destination_boundary_carrier=(
            endpoint_cohort_id is not None
            if destination_boundary_carrier is None
            else destination_boundary_carrier
        ),
        destination_boundary_axis_sign=(
            1
            if endpoint_cohort_id is not None and destination_boundary_axis_sign is None
            else destination_boundary_axis_sign
        ),
        network_id=network_id,
        reservation_complete=complete,
    )


def _target(
    claim: CorridorCohortLedgerClaim,
    coordinate: float,
    *,
    mutable: bool,
) -> CorridorCohortTarget:
    assert claim.edge_key is not None
    assert claim.family_id is not None
    if claim.orientation is CorridorOrientation.HORIZONTAL:
        start, end = (
            ((0.0, coordinate), (10.0, coordinate))
            if claim.direction is Direction.R
            else ((10.0, coordinate), (0.0, coordinate))
        )
        lead_start, lead_end = (
            ((10.0, coordinate + 10.0), (20.0, coordinate + 10.0))
            if claim.direction is Direction.R
            else ((0.0, coordinate + 10.0), (-10.0, coordinate + 10.0))
        )
        endpoint_lane_axis = 1
    else:
        start, end = (
            ((coordinate, 0.0), (coordinate, 10.0))
            if claim.direction is Direction.D
            else ((coordinate, 10.0), (coordinate, 0.0))
        )
        lead_start, lead_end = (
            ((coordinate + 10.0, 10.0), (coordinate + 10.0, 20.0))
            if claim.direction is Direction.D
            else ((coordinate + 10.0, 0.0), (coordinate + 10.0, -10.0))
        )
        endpoint_lane_axis = 0
    endpoint = claim.endpoint_cohort_id is not None
    route = SimpleNamespace(
        edge=SimpleNamespace(
            source=claim.edge_key[0],
            target=claim.edge_key[1],
        ),
        line_id=claim.edge_key[2],
        points=[start, end, lead_start, lead_end] if endpoint else [start, end],
        curve_radii=None,
        exit_shared_opening_points=(),
        route_system_owned_segment_ranks=(),
        convergence_owned_segment_ranks=(),
        fan_route_emitter=None,
        fan_plan_id=None,
        exit_turn_axis_id=None,
        exit_turn_plan_id=None,
        exit_turn_segment_rank=None,
        exit_lane_transition_plan_id=None,
        concentric_corner_offsets_by_segment={},
        concentric_corner_bases_by_segment={},
    )
    return CorridorCohortTarget(
        claim.member_id,
        f"plan:{claim.member_id}",
        claim.edge_key,
        claim.family_id,
        claim.connector_ids,
        route,
        mutable,
        endpoint_lane_axis if endpoint else None,
        (
            lead_start[endpoint_lane_axis] + (claim.endpoint_network_rank or 0) * 4.0
            if endpoint
            else None
        ),
        claim.network_id,
    )


def _ledger(
    claims: tuple[CorridorCohortLedgerClaim, ...],
) -> CorridorCohortLedger:
    endpoint_members: dict[str, set[str]] = {}
    for claim in claims:
        if claim.endpoint_cohort_id is not None:
            endpoint_members.setdefault(claim.endpoint_cohort_id, set()).add(
                claim.member_id
            )
    return CorridorCohortLedger(
        claims,
        tuple(
            (cohort_id, frozenset(member_ids))
            for cohort_id, member_ids in sorted(endpoint_members.items())
        ),
        frozenset(claim.member_id for claim in claims),
        frozenset(),
        4.0,
    )


def _footprint_target(
    member_id: str,
    line_id: str,
    points: list[tuple[float, float]],
    *,
    legal_crossing_segment_ranks: frozenset[int] = frozenset(),
) -> CorridorCohortTarget:
    edge_key = (f"{member_id}:source", f"{member_id}:target", line_id)
    return CorridorCohortTarget(
        member_id,
        f"plan:{member_id}",
        edge_key,
        RouteFamilyId.MERGE_TRUNK,
        (f"connector:{member_id}",),
        SimpleNamespace(
            edge=SimpleNamespace(source=edge_key[0], target=edge_key[1]),
            line_id=line_id,
            points=points,
        ),
        False,
        legal_crossing_segment_ranks=legal_crossing_segment_ranks,
    )


def test_footprint_witnesses_publish_typed_scalar_ownership_deterministically() -> None:
    target = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 4.0), (10.0, 4.0), (10.0, 20.0)],
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        target.member_id,
        target.edge_key,
        target.connector_ids,
        1,
        0,
        10.0,
    )

    witnesses = build_corridor_footprint_witnesses((target,), (variable,))

    assert tuple(item.segment_rank for item in witnesses) == (0, 1)
    assert witnesses[0].end_variable_id == variable.variable_id
    assert witnesses[1].coordinate_variable_id == variable.variable_id
    assert witnesses[1].owner_id == target.member_geometry_plan_id


def test_scalar_request_compiles_fixed_dogleg_from_one_snapshot() -> None:
    scalar = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    fixed = _footprint_target(
        "fixed",
        "branch",
        [(10.0, -10.0), (10.0, 9.0), (20.0, 9.0)],
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        scalar.member_id,
        scalar.edge_key,
        scalar.connector_ids,
        0,
        1,
        5.0,
    )
    request = CorridorScalarRequest(
        variable,
        5.0,
        CorridorCoordinateDomain(variable.variable_id, -100.0, 100.0),
    )

    plan = compile_corridor_cohort_plan(
        _ledger(()),
        (fixed, scalar),
        scalar_requests=(request,),
    )

    assert len(plan.scalar_grants) == 1
    problem = next(
        problem
        for component in plan.components
        for problem in component.problems
        if problem.lanes[0].member_id == variable.variable_id
    )
    assert len(problem.forbidden_intervals) == 1
    forbidden = problem.forbidden_intervals[0]
    assert not (
        forbidden.minimum_coordinate
        < plan.scalar_grants[0].coordinate
        < forbidden.maximum_coordinate
    )


def test_legal_perpendicular_crossing_emits_no_forbidden_interval() -> None:
    scalar = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    crossing = _footprint_target(
        "crossing",
        "branch",
        [(10.0, -10.0), (10.0, 10.0), (20.0, 10.0)],
        legal_crossing_segment_ranks=frozenset((0,)),
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        scalar.member_id,
        scalar.edge_key,
        scalar.connector_ids,
        0,
        1,
        5.0,
    )
    request = CorridorScalarRequest(
        variable,
        5.0,
        CorridorCoordinateDomain(variable.variable_id, -100.0, 100.0),
    )

    plan = compile_corridor_cohort_plan(
        _ledger(()),
        (crossing, scalar),
        scalar_requests=(request,),
    )

    assert plan.scalar_grants[0].coordinate == 5.0
    assert all(
        not problem.forbidden_intervals
        for component in plan.components
        for problem in component.problems
    )


def test_far_fixed_dogleg_is_a_crossing_not_a_clearance_obstacle() -> None:
    scalar = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    fixed = _footprint_target(
        "fixed",
        "branch",
        [(10.0, -10.0), (10.0, 30.0), (20.0, 30.0)],
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        scalar.member_id,
        scalar.edge_key,
        scalar.connector_ids,
        0,
        1,
        5.0,
    )
    request = CorridorScalarRequest(
        variable,
        5.0,
        CorridorCoordinateDomain(variable.variable_id, -100.0, 100.0),
    )

    plan = compile_corridor_cohort_plan(
        _ledger(()),
        (fixed, scalar),
        scalar_requests=(request,),
    )

    assert plan.scalar_grants[0].coordinate == 5.0
    assert all(
        not problem.forbidden_intervals
        for component in plan.components
        for problem in component.problems
    )


def test_fixed_dogleg_one_step_beyond_domain_has_typed_shortfall() -> None:
    scalar = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    fixed = _footprint_target(
        "fixed",
        "branch",
        [(10.0, -10.0), (10.0, 9.0), (20.0, 9.0)],
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        scalar.member_id,
        scalar.edge_key,
        scalar.connector_ids,
        0,
        1,
        5.0,
    )
    request = CorridorScalarRequest(
        variable,
        5.0,
        CorridorCoordinateDomain(variable.variable_id, 0.0, 9.0),
    )

    with pytest.raises(CorridorCohortCompilationError) as raised:
        compile_corridor_cohort_plan(
            _ledger(()),
            (fixed, scalar),
            scalar_requests=(request,),
        )

    (failure,) = raised.value.failures
    assert failure.clearance_shortfall is not None
    assert failure.clearance_shortfall.deficit == 4.0
    assert failure.clearance_shortfall.required_shift_sign == 1
    assert failure.clearance_shortfall.claim_ids == (variable.variable_id,)
    assert tuple(item.edge_key for item in failure.blocking_obstacles) == (
        fixed.edge_key,
    )


def test_scalar_compilation_is_request_and_target_permutation_invariant() -> None:
    first = _footprint_target(
        "scalar-a",
        "first",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    second = _footprint_target(
        "scalar-b",
        "second",
        [(0.0, 9.0), (20.0, 9.0)],
    )
    fixed = _footprint_target(
        "fixed",
        "branch",
        [(10.0, -10.0), (10.0, 2.0), (20.0, 2.0)],
    )

    def request(target: CorridorCohortTarget, coordinate: float):
        variable = CorridorScalarVariable(
            f"variable:{target.member_id}",
            CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
            f"convergence:{target.member_id}",
            target.member_id,
            target.edge_key,
            target.connector_ids,
            0,
            1,
            coordinate,
        )
        return CorridorScalarRequest(
            variable,
            coordinate,
            CorridorCoordinateDomain(variable.variable_id, -100.0, 100.0),
        )

    requests = (request(first, 5.0), request(second, 9.0))
    expected = compile_corridor_cohort_plan(
        _ledger(()),
        (fixed, first, second),
        scalar_requests=requests,
    )

    observed = compile_corridor_cohort_plan(
        _ledger(()),
        (second, first, fixed),
        scalar_requests=tuple(reversed(requests)),
    )

    assert observed == expected


def test_nonintersecting_scalar_footprint_keeps_its_preference() -> None:
    scalar = _footprint_target(
        "scalar",
        "trunk",
        [(0.0, 5.0), (20.0, 5.0)],
    )
    fixed = _footprint_target(
        "fixed",
        "branch",
        [(30.0, -10.0), (30.0, 10.0), (40.0, 10.0)],
    )
    variable = CorridorScalarVariable(
        "variable:scalar",
        CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
        "convergence:scalar",
        scalar.member_id,
        scalar.edge_key,
        scalar.connector_ids,
        0,
        1,
        5.0,
    )
    request = CorridorScalarRequest(
        variable,
        5.0,
        CorridorCoordinateDomain(variable.variable_id, -100.0, 100.0),
    )

    plan = compile_corridor_cohort_plan(
        _ledger(()),
        (fixed, scalar),
        scalar_requests=(request,),
    )

    assert plan.scalar_grants[0].coordinate == 5.0
    assert all(
        not problem.forbidden_intervals
        for component in plan.components
        for problem in component.problems
    )


def test_semantic_ledger_claims_store_no_observed_geometry() -> None:
    field_names = {field.name for field in fields(CorridorCohortLedgerClaim)}

    assert "coordinate" not in field_names
    assert "longitudinal_start" not in field_names
    assert "longitudinal_end" not in field_names


def test_compiler_derives_current_obstacle_geometry_and_never_patches_fixed() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        line_id="other",
    )
    movable_target = _target(movable, 10.0, mutable=True)
    fixed_target = _target(fixed, 12.0, mutable=False)
    movable_target.route.curve_radii = [10.0]
    fixed_before = tuple(fixed_target.route.points)

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)), (movable_target, fixed_target)
    )

    assert plan.allocations[0].coordinate == 8.0
    assert movable_target.route.points[:2] == [(0.0, 8.0), (10.0, 8.0)]
    assert movable_target.route.curve_radii == [10.0]
    assert tuple(fixed_target.route.points) == fixed_before


def test_partial_concentric_corner_evidence_fails_before_mutation() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
    )
    movable_target = _target(movable, 10.0, mutable=True)
    movable_target.route.curve_radii = [10.0]
    movable_target.route.concentric_corner_offsets_by_segment = {0: (None, 2.0)}
    fixed_target = _target(fixed, 12.0, mutable=False)
    before = tuple(movable_target.route.points)

    with pytest.raises(CorridorCohortCompilationError, match="source snapshot"):
        compile_corridor_cohort_plan(
            _ledger((movable, fixed)), (movable_target, fixed_target)
        )

    assert tuple(movable_target.route.points) == before


def test_finalized_missing_landing_fails_before_publication() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        line_id="other",
    )
    movable_target = _target(movable, 10.0, mutable=True)
    fixed_target = _target(fixed, 12.0, mutable=False)
    movable_target.route.points[:] = [(0.0, 10.0), (10.0, 10.0), (10.0, 20.0)]
    movable_target = replace(
        movable_target,
        endpoint_lane_axis=0,
        endpoint_lane_coordinate=10.0,
    )
    before = tuple(movable_target.route.points)
    ledger = replace(
        _ledger((movable, fixed)),
        finalized_owned_segments=frozenset(
            {
                (movable.member_id, movable.edge_key, 0),
                (movable.member_id, movable.edge_key, 2),
            }
        ),
    )

    with pytest.raises(
        CorridorCohortCompilationError,
        match="realization changed ownership",
    ):
        compile_corridor_cohort_plan(ledger, (movable_target, fixed_target))

    assert tuple(movable_target.route.points) == before
    assert movable_target.route.route_system_owned_segment_ranks == ()


def test_published_allocation_span_uses_complete_atomic_patch() -> None:
    movable = replace(
        _ledger_claim(
            "movable",
            claim_rank=0,
            coordinate_rank=0,
            endpoint_cohort_id="endpoint",
            direction=Direction.D,
            orientation=CorridorOrientation.VERTICAL,
        ),
        segment_rank=1,
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        direction=Direction.D,
        line_id="other",
        orientation=CorridorOrientation.VERTICAL,
    )
    movable_target = _target(movable, 10.0, mutable=True)
    movable_target.route.points[:] = [
        (0.0, 10.0),
        (10.0, 10.0),
        (10.0, 20.0),
        (20.0, 20.0),
    ]
    movable_target = replace(
        movable_target,
        endpoint_lane_axis=1,
        endpoint_lane_coordinate=24.0,
    )
    fixed_target = _target(fixed, 12.0, mutable=False)
    fixed_target.route.points[:] = [(12.0, 21.0), (12.0, 23.0)]

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (movable_target, fixed_target),
    )

    allocation = plan.allocations[0]
    assert allocation.coordinate == 8.0
    assert (allocation.longitudinal_start, allocation.longitudinal_end) == (
        10.0,
        24.0,
    )
    assert movable_target.route.points[1:3] == [(8.0, 10.0), (8.0, 24.0)]


def test_semantic_incompleteness_is_compatibility_but_binding_loss_is_failure() -> None:
    incomplete = _ledger_claim(
        "incomplete",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        complete=False,
    )
    incomplete_target = _target(incomplete, 10.0, mutable=True)

    plan = compile_corridor_cohort_plan(_ledger((incomplete,)), (incomplete_target,))

    assert plan.allocations == ()
    assert plan.components[0].status is CorridorAllocationStatus.COMPATIBILITY

    complete = replace(incomplete, reservation_complete=True)
    with pytest.raises(
        CorridorCohortCompilationError, match="incomplete eligible frame"
    ):
        compile_corridor_cohort_plan(_ledger((complete,)), ())


def test_stale_member_geometry_plan_identity_fails_before_mutation() -> None:
    claim = _ledger_claim(
        "member",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    target = _target(claim, 10.0, mutable=True)
    target = replace(target, member_geometry_plan_id="plan:stale")
    before = tuple(target.route.points)

    with pytest.raises(CorridorCohortCompilationError, match="current route"):
        compile_corridor_cohort_plan(_ledger((claim,)), (target,))

    assert tuple(target.route.points) == before


def test_immutable_continuation_is_fixed_instead_of_promoted_to_equality() -> None:
    leader = _ledger_claim(
        "leader",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:movable",
        network_id="network",
    )
    follower = _ledger_claim(
        "follower",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:fixed",
        network_id="network",
    )

    plan = compile_corridor_cohort_plan(
        _ledger((leader, follower)),
        (
            _target(leader, 10.0, mutable=True),
            _target(follower, 10.0, mutable=False),
        ),
    )

    assert dict(plan.components[0].claim_roles)[follower.claim_id] is (
        CorridorCohortClaimRole.FIXED
    )
    assert all(
        follower.claim_id not in (equality.left_member_id, equality.right_member_id)
        for problem in plan.components[0].problems
        for equality in problem.equalities
    )


def test_same_network_lane_matches_its_fixed_continuation() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:movable",
        network_id="network",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:fixed",
        network_id="network",
    )

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (
            _target(movable, 10.0, mutable=True),
            _target(fixed, 12.0, mutable=False),
        ),
    )

    problem = plan.components[0].problems[0]
    assert len(problem.fixed_equalities) == 1
    assert problem.fixed_equalities[0].member_id == movable.claim_id
    assert problem.fixed_equalities[0].obstacle_id == fixed.claim_id
    assert plan.allocations[0].coordinate == 12.0


def test_opposite_running_peer_is_not_a_fixed_equality() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:shared",
        network_id="network",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:shared",
        network_id="network",
        direction=Direction.L,
    )

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (
            _target(movable, 10.0, mutable=True),
            _target(fixed, 12.0, mutable=False),
        ),
    )

    assert all(not problem.fixed_equalities for problem in plan.components[0].problems)


def test_coordinate_proximity_without_a_shared_network_is_not_a_fixed_equality() -> (
    None
):
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:movable",
        network_id="network:movable",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:fixed",
        network_id="network:fixed",
    )

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (
            _target(movable, 10.0, mutable=True),
            _target(fixed, 10.0, mutable=False),
        ),
    )

    assert all(not problem.fixed_equalities for problem in plan.components[0].problems)


def test_overlapping_same_lane_without_a_shared_terminal_is_not_a_fixed_equality() -> (
    None
):
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:movable",
        network_id="network",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:fixed",
        network_id="network",
    )
    movable_target = _target(movable, 10.0, mutable=True)
    fixed_target = _target(fixed, 10.0, mutable=False)
    fixed_target.route.points[:] = [(5.0, 10.0), (15.0, 10.0)]

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (movable_target, fixed_target),
    )

    assert all(not problem.fixed_equalities for problem in plan.components[0].problems)


def test_same_reservation_lane_matches_a_fixed_branch() -> None:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        reservation_id="reservation:shared",
        network_id="network",
    )
    fixed = _ledger_claim(
        "fixed",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        reservation_id="reservation:shared",
        network_id="network",
    )
    movable_target = _target(movable, 10.0, mutable=True)
    fixed_target = _target(fixed, 12.0, mutable=False)
    fixed_target.route.points[:] = [(5.0, 12.0), (15.0, 12.0)]

    plan = compile_corridor_cohort_plan(
        _ledger((movable, fixed)),
        (movable_target, fixed_target),
    )

    problem = plan.components[0].problems[0]
    assert len(problem.fixed_equalities) == 1
    assert plan.allocations[0].coordinate == 12.0


def test_existing_semantic_segment_owner_has_terminal_fixed_precedence() -> None:
    claim = _ledger_claim(
        "owned",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    target = _target(claim, 10.0, mutable=True)
    target.route.convergence_owned_segment_ranks = (0,)

    plan = compile_corridor_cohort_plan(_ledger((claim,)), (target,))

    assert plan.allocations == ()
    assert plan.components[0].claim_roles == (
        (claim.claim_id, CorridorCohortClaimRole.FIXED),
    )


def test_complete_non_endpoint_claim_is_movable_beside_terminal_fixed_claim() -> None:
    movable = _ledger_claim(
        "regular",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id=None,
    )
    terminal_fixed = _ledger_claim(
        "terminal",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="other",
    )
    movable_target = _target(movable, 10.0, mutable=True)
    terminal_target = _target(terminal_fixed, 10.0, mutable=True)
    terminal_target.route.exit_turn_plan_id = "exit-turn"
    terminal_target.route.exit_turn_segment_rank = 0

    plan = compile_corridor_cohort_plan(
        _ledger((movable, terminal_fixed)),
        (movable_target, terminal_target),
    )

    assert dict(plan.components[0].claim_roles) == {
        movable.claim_id: CorridorCohortClaimRole.MOVABLE,
        terminal_fixed.claim_id: CorridorCohortClaimRole.FIXED,
    }
    assert [item.member_id for item in plan.allocations] == [movable.member_id]
    assert movable_target.route.route_system_owned_segment_ranks == (0,)
    assert terminal_target.route.route_system_owned_segment_ranks == ()


def test_opposite_running_non_endpoint_claims_are_never_bundled() -> None:
    rightward = _ledger_claim(
        "rightward",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        direction=Direction.R,
    )
    leftward = _ledger_claim(
        "leftward",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        direction=Direction.L,
        line_id="other",
    )
    terminal_fixed = _ledger_claim(
        "terminal",
        claim_rank=2,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="anchor",
    )
    rightward_target = _target(rightward, 10.0, mutable=True)
    leftward_target = _target(leftward, 10.0, mutable=True)
    terminal_target = _target(terminal_fixed, 30.0, mutable=True)
    terminal_target.route.exit_turn_plan_id = "exit-turn"
    terminal_target.route.exit_turn_segment_rank = 0

    plan = compile_corridor_cohort_plan(
        _ledger((rightward, leftward, terminal_fixed)),
        (rightward_target, leftward_target, terminal_target),
    )

    problem = plan.components[0].problems[0]
    assert len(problem.lanes) == 2
    assert len({lane.cohort_id for lane in problem.lanes}) == 2
    assert problem.equalities == ()
    assert any(
        {separation.left_member_id, separation.right_member_id}
        == {rightward.claim_id, leftward.claim_id}
        for separation in problem.separations
    )
    assert rightward_target.route.route_system_owned_segment_ranks == (0,)
    assert leftward_target.route.route_system_owned_segment_ranks == (0,)


def test_endpoint_cohort_takes_only_the_exit_turn_continuation_flank() -> None:
    claim = _ledger_claim(
        "endpoint",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
    )
    adjacent = _target(claim, 10.0, mutable=True)
    adjacent.route.exit_turn_plan_id = "exit-turn"
    adjacent.route.exit_turn_segment_rank = 1

    adjacent_plan = compile_corridor_cohort_plan(_ledger((claim,)), (adjacent,))

    assert adjacent_plan.allocations
    assert adjacent_plan.components[0].claim_roles == (
        (claim.claim_id, CorridorCohortClaimRole.MOVABLE),
    )

    exact = _target(claim, 10.0, mutable=True)
    exact.route.exit_turn_plan_id = "exit-turn"
    exact.route.exit_turn_segment_rank = 0

    exact_plan = compile_corridor_cohort_plan(_ledger((claim,)), (exact,))

    assert exact_plan.allocations == ()
    assert exact_plan.components[0].claim_roles == (
        (claim.claim_id, CorridorCohortClaimRole.FIXED),
    )

    incomplete_claim = replace(claim, reservation_complete=False)
    incomplete = _target(incomplete_claim, 10.0, mutable=True)
    incomplete.route.exit_turn_plan_id = "exit-turn"
    incomplete.route.exit_turn_segment_rank = 1

    incomplete_plan = compile_corridor_cohort_plan(
        _ledger((incomplete_claim,)),
        (incomplete,),
    )

    assert incomplete_plan.allocations == ()
    assert incomplete_plan.components[0].claim_roles == (
        (claim.claim_id, CorridorCohortClaimRole.FIXED),
    )


@pytest.mark.parametrize(
    ("side", "direction", "expected"),
    (
        (PortSide.LEFT, Direction.R, 1),
        (PortSide.LEFT, Direction.L, -1),
        (PortSide.LEFT, Direction.U, 1),
        (PortSide.LEFT, Direction.D, -1),
        (PortSide.RIGHT, Direction.R, -1),
        (PortSide.RIGHT, Direction.L, 1),
        (PortSide.RIGHT, Direction.U, -1),
        (PortSide.RIGHT, Direction.D, 1),
        (PortSide.TOP, Direction.R, -1),
        (PortSide.TOP, Direction.L, 1),
        (PortSide.TOP, Direction.U, -1),
        (PortSide.TOP, Direction.D, 1),
        (PortSide.BOTTOM, Direction.R, 1),
        (PortSide.BOTTOM, Direction.L, -1),
        (PortSide.BOTTOM, Direction.U, 1),
        (PortSide.BOTTOM, Direction.D, -1),
    ),
)
def test_destination_claim_axis_sign_rotates_the_port_arrival_frame(
    side: PortSide,
    direction: Direction,
    expected: int,
) -> None:
    assert _destination_claim_axis_sign(side, direction) == expected


@pytest.mark.parametrize(
    ("axis_sign", "orientation", "carrier"),
    (
        (1, CorridorOrientation.HORIZONTAL, True),
        (1, CorridorOrientation.VERTICAL, True),
        (1, CorridorOrientation.HORIZONTAL, False),
        (-1, CorridorOrientation.HORIZONTAL, False),
        (1, CorridorOrientation.VERTICAL, False),
        (-1, CorridorOrientation.VERTICAL, False),
    ),
)
def test_endpoint_boundary_delta_uses_frozen_semantic_frame(
    axis_sign: int,
    orientation: CorridorOrientation,
    carrier: bool,
) -> None:
    direction = (
        Direction.R if orientation is CorridorOrientation.HORIZONTAL else Direction.D
    )
    rank_zero = _ledger_claim(
        "rank-zero",
        claim_rank=10,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        direction=direction,
        line_id="first",
        orientation=orientation,
        destination_boundary_carrier=carrier,
        destination_boundary_axis_sign=axis_sign,
    )
    rank_one = _ledger_claim(
        "rank-one",
        claim_rank=0,
        coordinate_rank=1,
        endpoint_cohort_id="endpoint",
        direction=direction,
        line_id="second",
        orientation=orientation,
        destination_boundary_carrier=carrier,
        destination_boundary_axis_sign=axis_sign,
    )

    for order in permutations((rank_zero, rank_one)):
        targets = tuple(_target(claim, 10.0, mutable=True) for claim in reversed(order))
        plan = compile_corridor_cohort_plan(_ledger(order), targets)
        allocations = {item.claim_id: item.coordinate for item in plan.allocations}
        assert allocations[rank_one.claim_id] - allocations[
            rank_zero.claim_id
        ] == pytest.approx(axis_sign * 4.0)


def test_physical_cohort_compresses_absent_endpoint_network_ranks() -> None:
    rank_zero = _ledger_claim(
        "rank-zero",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="first",
    )
    rank_two = _ledger_claim(
        "rank-two",
        claim_rank=1,
        coordinate_rank=2,
        endpoint_cohort_id="endpoint",
        line_id="second",
    )

    plan = compile_corridor_cohort_plan(
        _ledger((rank_zero, rank_two)),
        (_target(rank_zero, 10.0, mutable=True), _target(rank_two, 10.0, mutable=True)),
    )
    allocations = {item.claim_id: item.coordinate for item in plan.allocations}

    assert allocations[rank_two.claim_id] - allocations[
        rank_zero.claim_id
    ] == pytest.approx(4.0)


def test_endpoint_slots_bind_only_the_current_eligible_members() -> None:
    rank_zero = _ledger_claim(
        "rank-zero",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="first",
    )
    absent = _ledger_claim(
        "absent",
        claim_rank=1,
        coordinate_rank=1,
        endpoint_cohort_id="endpoint",
        complete=False,
        line_id="absent",
    )
    rank_two = _ledger_claim(
        "rank-two",
        claim_rank=2,
        coordinate_rank=2,
        endpoint_cohort_id="endpoint",
        line_id="second",
    )
    rank_zero_target = replace(
        _target(rank_zero, 10.0, mutable=True),
        endpoint_lane_coordinate=30.0,
    )
    rank_two_target = replace(
        _target(rank_two, 10.0, mutable=True),
        endpoint_lane_coordinate=10.0,
    )
    expected = {
        rank_zero.edge_key: 10.0,
        rank_two.edge_key: 30.0,
    }

    for claim_order in permutations((rank_zero, absent, rank_two)):
        ledger = replace(
            _ledger(claim_order),
            eligible_member_ids=frozenset((rank_zero.member_id, rank_two.member_id)),
        )
        for target_order in (
            (rank_zero_target, rank_two_target),
            (rank_two_target, rank_zero_target),
        ):
            targets = tuple(copy.deepcopy(target) for target in target_order)
            plan = compile_corridor_cohort_plan(
                ledger,
                targets,
            )
            assert {
                item.edge_key: item.coordinate for item in plan.landings
            } == expected


def test_endpoint_slots_ignore_members_eligible_only_in_other_corridors() -> None:
    rank_zero = _ledger_claim(
        "rank-zero",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="first",
    )
    rank_two = _ledger_claim(
        "rank-two",
        claim_rank=1,
        coordinate_rank=2,
        endpoint_cohort_id="endpoint",
        line_id="second",
    )
    unrelated = _ledger_claim(
        "unrelated",
        claim_rank=2,
        coordinate_rank=1,
        endpoint_cohort_id=None,
        line_id="other",
    )
    ledger = replace(
        _ledger((rank_zero, rank_two, unrelated)),
        endpoint_members=(
            (
                "endpoint",
                frozenset(
                    (rank_zero.member_id, unrelated.member_id, rank_two.member_id)
                ),
            ),
        ),
    )
    rank_zero_target = replace(
        _target(rank_zero, 10.0, mutable=True),
        endpoint_lane_coordinate=30.0,
    )
    rank_two_target = replace(
        _target(rank_two, 10.0, mutable=True),
        endpoint_lane_coordinate=10.0,
    )

    plan = compile_corridor_cohort_plan(
        ledger,
        (
            _target(unrelated, 100.0, mutable=True),
            rank_two_target,
            rank_zero_target,
        ),
    )

    assert {item.edge_key: item.coordinate for item in plan.landings} == {
        rank_zero.edge_key: 10.0,
        rank_two.edge_key: 30.0,
    }


def test_endpoint_slots_fail_closed_when_a_claimed_member_has_no_target() -> None:
    rank_zero = _ledger_claim(
        "rank-zero",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        line_id="first",
    )
    missing = _ledger_claim(
        "missing",
        claim_rank=1,
        coordinate_rank=1,
        endpoint_cohort_id="endpoint",
        line_id="second",
    )

    with pytest.raises(
        CorridorCohortCompilationError,
        match="incomplete eligible frame",
    ):
        compile_corridor_cohort_plan(
            _ledger((rank_zero, missing)),
            (_target(rank_zero, 10.0, mutable=True),),
        )


def _landing_obstacle_case(
    *, obstacle_target_id: str, obstacle_network_id: str
) -> tuple[
    CorridorCohortLedger,
    CorridorCohortTarget,
    CorridorCohortTarget,
]:
    movable = _ledger_claim(
        "movable",
        claim_rank=0,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        direction=Direction.U,
        network_id="shared-network",
        orientation=CorridorOrientation.VERTICAL,
    )
    movable = replace(
        movable,
        edge_key=(movable.edge_key[0], "shared-target", movable.edge_key[2]),
    )
    movable_target = _target(movable, 10.0, mutable=True)
    movable_target.route.points = [(10.0, 20.0), (10.0, 10.0), (0.0, 10.0)]
    movable_target = replace(
        movable_target,
        endpoint_lane_axis=1,
        endpoint_lane_coordinate=10.0,
    )

    obstacle = _ledger_claim(
        "obstacle",
        claim_rank=1,
        coordinate_rank=0,
        endpoint_cohort_id=None,
        direction=Direction.L,
        line_id="other",
        network_id=obstacle_network_id,
    )
    obstacle = replace(
        obstacle,
        edge_key=(obstacle.edge_key[0], obstacle_target_id, obstacle.edge_key[2]),
    )
    obstacle_target = _target(obstacle, 10.0, mutable=False)
    obstacle_target.route.points = [(20.0, 10.0), (0.0, 10.0)]
    return _ledger((movable,)), movable_target, obstacle_target


def test_same_endpoint_network_has_no_snapshot_landing_constraint() -> None:
    ledger, movable, obstacle = _landing_obstacle_case(
        obstacle_target_id="shared-target",
        obstacle_network_id="shared-network",
    )

    plan = compile_corridor_cohort_plan(ledger, (obstacle, movable))

    assert plan.allocations[0].coordinate == pytest.approx(10.0)
    assert all(
        obstacle_id.startswith("member-footprint-endpoint-order|")
        for component in plan.components
        for problem in component.problems
        for domain in problem.domains
        for obstacle_id in domain.obstacle_ids
    )


def test_unrelated_route_has_no_snapshot_landing_constraint() -> None:
    ledger, movable, obstacle = _landing_obstacle_case(
        obstacle_target_id="other-target",
        obstacle_network_id="other-network",
    )

    plan = compile_corridor_cohort_plan(ledger, (obstacle, movable))

    assert plan.allocations[0].coordinate == pytest.approx(10.0)
    assert all(
        obstacle_id.startswith("member-footprint-endpoint-order|")
        for component in plan.components
        for problem in component.problems
        for domain in problem.domains
        for obstacle_id in domain.obstacle_ids
    )


def test_same_line_counter_running_lanes_have_distinct_owners_and_clearance() -> None:
    right = _ledger_claim(
        "right",
        claim_rank=10,
        coordinate_rank=0,
        endpoint_cohort_id="endpoint",
        direction=Direction.R,
        network_id="shared-network",
    )
    left = _ledger_claim(
        "left",
        claim_rank=0,
        coordinate_rank=1,
        endpoint_cohort_id="endpoint",
        direction=Direction.L,
        network_id="shared-network",
    )
    plan = compile_corridor_cohort_plan(
        _ledger((right, left)),
        (_target(right, 10.0, mutable=True), _target(left, 12.0, mutable=True)),
    )

    assert len(plan.components[0].problems) == 1
    cohort_owners = {
        lane.cohort_id
        for problem in plan.components[0].problems
        for lane in problem.lanes
    }
    allocations = {item.claim_id: item.coordinate for item in plan.allocations}
    assert len(cohort_owners) == 2
    assert plan.components[0].problems[0].separations == (
        CorridorSeparation(right.claim_id, left.claim_id, 11.0),
    )
    assert abs(allocations[right.claim_id] - allocations[left.claim_id]) >= 11.0
    assert allocations[right.claim_id] < allocations[left.claim_id]
    assert all(
        role is CorridorCohortClaimRole.MOVABLE
        for _claim_id, role in plan.components[0].claim_roles
    )
