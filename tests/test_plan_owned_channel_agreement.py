"""Planner-owned geometry has one freeze rule and one final realization."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import nf_metro.layout.routing.corridor_cohort_integration as cohort_integration
import nf_metro.layout.routing.invariants as routing_invariants
import nf_metro.layout.routing.member_geometry as member_geometry
import nf_metro.layout.routing.normalize as normalize
from nf_metro.api import prepare_graph, render_string, resolve_theme
from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP
from nf_metro.layout.routing.common import RoutedPath, planner_owns_segment
from nf_metro.parser.model import Edge
from nf_metro.parser.route_topology import ResolvedEdge
from nf_metro.render.svg import build_observed_render_plan

SEED_15 = Path(__file__).parent / "fixtures" / "hash_seed_determinism" / "seed_15.mmd"


def _route(**attributes) -> RoutedPath:
    route = RoutedPath(
        Edge("source", "target", "line"),
        "line",
        [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (30.0, 20.0),
        ],
    )
    for name, value in attributes.items():
        setattr(route, name, value)
    return route


@pytest.mark.parametrize(
    ("attributes", "owned_ranks"),
    (
        pytest.param({}, frozenset(), id="unplanned"),
        pytest.param(
            {"route_system_owned_segment_ranks": (2,)},
            frozenset({1, 2, 3}),
            id="member-boundary",
        ),
        pytest.param(
            {"convergence_owned_segment_ranks": (2,)},
            frozenset({1, 2, 3}),
            id="convergence-boundary",
        ),
        pytest.param(
            {"exit_shared_opening_points": ((0.0, 0.0), (10.0, 0.0))},
            frozenset({0, 1}),
            id="shared-opening",
        ),
        pytest.param(
            {"fan_plan_id": "fan-plan"},
            frozenset(range(5)),
            id="fan-plan",
        ),
        pytest.param(
            {"fan_route_emitter": "fan-emitter"},
            frozenset(range(5)),
            id="fan-emitter",
        ),
        pytest.param(
            {"exit_lane_transition_plan_id": "transition-plan"},
            frozenset(range(5)),
            id="lane-transition",
        ),
        pytest.param(
            {
                "exit_turn_plan_id": "exit-plan",
                "exit_turn_axis_id": "exit-axis",
                "exit_turn_segment_rank": 2,
            },
            frozenset({1, 2, 3}),
            id="exit-turn-boundary",
        ),
    ),
)
def test_every_consumer_reads_the_canonical_ownership_matrix(
    attributes: dict[str, object], owned_ranks: frozenset[int]
) -> None:
    route = _route(**attributes)
    observed = frozenset(
        rank
        for rank in range(len(route.points) - 1)
        if planner_owns_segment(route, rank)
    )

    assert observed == owned_ranks
    assert all(
        normalize._planner_owns_channel(SimpleNamespace(route=route, idx=rank))
        == (rank in owned_ranks)
        for rank in range(len(route.points) - 1)
    )
    assert member_geometry.planner_owns_segment is planner_owns_segment
    assert cohort_integration.planner_owns_segment is planner_owns_segment
    assert routing_invariants.planner_owns_segment is planner_owns_segment


def test_allocation_can_explicitly_relinquish_one_pending_exit_turn() -> None:
    route = _route(
        exit_turn_plan_id="pending-exit",
        exit_turn_axis_id="pending-axis",
        exit_turn_segment_rank=2,
    )

    assert {rank for rank in range(5) if planner_owns_segment(route, rank)} == {1, 2, 3}
    assert not any(
        planner_owns_segment(
            route,
            rank,
            relinquished_exit_turn_plan_ids=frozenset({"pending-exit"}),
        )
        for rank in range(5)
    )


@pytest.fixture(scope="module")
def seed_15_production():
    graph = prepare_graph(SEED_15.read_text(), source_dir=str(SEED_15.parent))
    return build_observed_render_plan(graph, resolve_theme(None, graph))


def _render_route(observed, edge: ResolvedEdge):
    return next(
        (route, points)
        for route, points in zip(
            observed.plan.routes, observed.plan.route_polylines, strict=True
        )
        if (route.edge.source, route.edge.target, route.line_id)
        == (edge.source, edge.target, edge.line_id)
    )


def test_seed_15_member_plan_and_final_route_publish_one_channel(
    seed_15_production,
) -> None:
    edge = ResolvedEdge("__junction_24", "s6__entry_right_14", "l1")
    member = next(
        plan
        for plan in seed_15_production.route_plan.member_geometry_plans
        if plan.edge == edge
    )
    route, _points = _render_route(seed_15_production, edge)

    assert tuple(route.points) == member.points
    assert member.owned_segment_ranks
    assert all(
        tuple(route.points[channel.segment_rank : channel.segment_rank + 2])
        == (channel.start, channel.end)
        for channel in member.gap_channels
    )


def test_seed_15_entry_tails_keep_distinct_planned_columns(
    seed_15_production,
) -> None:
    l0, _ = _render_route(
        seed_15_production,
        ResolvedEdge("__junction_22", "s5__entry_right_16", "l0"),
    )
    l2, _ = _render_route(
        seed_15_production,
        ResolvedEdge("__junction_23", "s5__entry_right_16", "l2"),
    )

    assert abs(l0.points[-2][0] - l2.points[-2][0]) == pytest.approx(OFFSET_STEP)


def test_seed_15_relocated_trunk_keeps_formed_corners(seed_15_production) -> None:
    route, _ = _render_route(
        seed_15_production,
        ResolvedEdge("__junction_24", "__merge_12", "l2"),
    )

    assert route.curve_radii
    assert min(route.curve_radii) >= CURVE_RADIUS


def test_seed_15_render_emits_no_geometry_warnings() -> None:
    """The corridor stack sits inside every reserved band it claims."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_string(SEED_15.read_text())
    geometry = [
        str(item.message)
        for item in caught
        if type(item.message).__name__ == "PermissiveGuardWarning"
    ]
    assert geometry == []
