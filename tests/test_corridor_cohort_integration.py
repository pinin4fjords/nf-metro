"""Planning-seam coverage for canonical destination-corridor cohorts."""

from __future__ import annotations

import copy
import inspect
import warnings
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import nf_metro.layout.routing.core as routing_core
import nf_metro.layout.routing.corridor_cohort_integration as cohort_integration
import nf_metro.layout.routing.member_geometry as member_geometry_routing
import nf_metro.layout.routing.planning as routing_planning
import nf_metro.render.svg as render_svg_module
from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import COORD_TOLERANCE, CURVE_RADIUS
from nf_metro.layout.geometry import cotravelling_lane_clearance
from nf_metro.layout.route_reservations import (
    CorridorOrientation,
    RouteReservationId,
    RouteReservationLane,
)
from nf_metro.layout.routing.common import Direction, right_normal_axis_sign
from nf_metro.layout.routing.context import port_arrival_order, port_lane_coord
from nf_metro.layout.routing.corridor_cohort_integration import (
    CorridorCohortClaimRole,
    CorridorCohortCompilationError,
    CorridorCohortLedgerClaim,
    build_corridor_cohort_ledger,
    compile_corridor_cohort_plan,
)
from nf_metro.layout.routing.corridor_cohorts import CorridorAllocationStatus
from nf_metro.layout.routing.member_geometry import (
    fresh_member_route,
    validate_member_geometry_emission,
)
from nf_metro.render.plan import freeze_render_value
from nf_metro.render.svg import build_observed_render_plan

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED77 = REPO_ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_77.mmd"
SEED15 = REPO_ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_15.mmd"
OPPOSING_BYPASS = REPO_ROOT / "examples" / "topologies" / "opposing_bypass_corridor.mmd"

S9_L1 = ("__junction_39", "s9__entry_right_25", "l1")
S9_L3 = ("__junction_41", "s9__entry_right_25", "l3")
S17_L0 = ("__junction_39", "s17__entry_right_26", "l0")
S17_L3 = ("__junction_43", "s17__entry_right_26", "l3")
S10_L0 = ("__junction_41", "s10__entry_right_29", "l0")
S16_L2 = ("__junction_35", "s16__entry_right_18", "l2")
S16_L1 = ("s15__exit_left_13", "s16__entry_right_18", "l1")
TARGET_KEYS = frozenset((S9_L1, S9_L3, S17_L0, S17_L3))
SEED15_LANDING = ("__junction_22", "s5__entry_right_16", "l0")
SEED15_FIXED_LEAD = ("__junction_24", "__merge_12", "l2")


def _edge_key(plan):
    return plan.edge.source, plan.edge.target, plan.edge.line_id


def _graph_edge(graph, edge_key):
    return next(
        edge
        for edge in graph.edges
        if (edge.source, edge.target, edge.line_id) == edge_key
    )


def _capture_settled_seed77(
    monkeypatch: pytest.MonkeyPatch, *, capture_compilation: bool | str = False
):
    captures = []
    compilation_calls = []
    active_prior_plans = []
    original_route_edges = routing_core._route_edges
    original_planning = routing_core.prepare_route_system_planning
    original_compile = member_geometry_routing.compile_corridor_cohort_plan

    if capture_compilation:

        def capture_compile(ledger, targets, *, scalar_requests=()):
            source_targets = copy.deepcopy(tuple(targets))
            plan = original_compile(ledger, targets, scalar_requests=scalar_requests)
            realized_targets = copy.deepcopy(tuple(targets))
            compilation_calls.append((ledger, source_targets, plan, realized_targets))
            return plan

        monkeypatch.setattr(
            member_geometry_routing,
            "compile_corridor_cohort_plan",
            capture_compile,
        )

    def capture_route_call(*args, **kwargs):
        active_prior_plans.append(kwargs.get("reservations"))
        try:
            return original_route_edges(*args, **kwargs)
        finally:
            active_prior_plans.pop()

    def capture_planning(graph, ctx, **kwargs):
        planning = original_planning(graph, ctx, **kwargs)
        if active_prior_plans and active_prior_plans[-1] is not None:
            captures.append((planning, active_prior_plans[-1], ctx.reserved_bands, ctx))
        return planning

    monkeypatch.setattr(routing_core, "_route_edges", capture_route_call)
    monkeypatch.setattr(routing_core, "prepare_route_system_planning", capture_planning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(SEED77.read_text(), source_dir=str(SEED77.parent))
        observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    granting = tuple(
        capture
        for capture in captures
        if capture[0].corridor_cohorts is not None
        and capture[0].corridor_cohorts.allocations
    )
    assert granting
    planning, prior_plan, reserved_corridors, ctx = granting[0]
    result = graph, planning, prior_plan, reserved_corridors, ctx
    if capture_compilation:
        granting_calls = tuple(
            call
            for call in compilation_calls
            if call[0].finalized_owned_segments is None and call[2].allocations
        )
        assert granting_calls
        if capture_compilation == "all":
            return (
                *result,
                granting_calls[0][:3],
                observed.route_plan,
                compilation_calls,
            )
        return (*result, granting_calls[0][:3])
    return result


def _cohort_plan(graph, planning, prior_plan, reserved_corridors):
    del graph, prior_plan, reserved_corridors
    plan = planning.corridor_cohorts
    assert plan is not None
    return plan


def _edge_by_member_id(prior_plan):
    return {
        str(member.id): (
            member.source.station_id,
            member.target.station_id,
            member.line_id,
        )
        for member in prior_plan.members
    }


def _claim_edge_by_id(prior_plan):
    edge_by_member = _edge_by_member_id(prior_plan)
    return {
        f"{reservation.id}|claim:{claim_rank}": edge_by_member[str(claim.member_id)]
        for reservation in prior_plan.reservations
        for claim_rank, claim in enumerate(reservation.claims)
    }


def _target_components(plan, prior_plan):
    edge_by_claim = _claim_edge_by_id(prior_plan)
    return tuple(
        component
        for component in plan.components
        if TARGET_KEYS
        & {
            edge_by_claim[claim_id]
            for claim_id, _role in component.claim_roles
            if claim_id in edge_by_claim
        }
    )


def _shared_seed77_reservation(prior_plan):
    edge_by_member = _edge_by_member_id(prior_plan)
    expected = TARGET_KEYS | {S10_L0}
    return next(
        reservation
        for reservation in prior_plan.reservations
        if {edge_by_member[str(claim.member_id)] for claim in reservation.claims}
        == expected
    )


def _without_s10_witness(prior_plan, reserved_corridors):
    edge_by_member = _edge_by_member_id(prior_plan)
    shared = _shared_seed77_reservation(prior_plan)
    kept_ranks = tuple(
        rank
        for rank, claim in enumerate(shared.claims)
        if edge_by_member[str(claim.member_id)] != S10_L0
    )
    old_to_new = {old: new for new, old in enumerate(kept_ranks)}
    claims = tuple(shared.claims[rank] for rank in kept_ranks)
    lanes = tuple(
        RouteReservationLane(
            tuple(old_to_new[rank] for rank in lane.claim_indices if rank in old_to_new)
        )
        for lane in shared.lanes
        if any(rank in old_to_new for rank in lane.claim_indices)
    )
    incomplete = replace(
        shared,
        claimant_member_ids=tuple(claim.member_id for claim in claims),
        claims=claims,
        lanes=lanes,
        lane_count=len(lanes),
    )
    prior_plan = replace(
        prior_plan,
        reservations=tuple(
            incomplete if item.id == shared.id else item
            for item in prior_plan.reservations
        ),
    )
    reserved_corridors = replace(
        reserved_corridors,
        planned_order_coordinates=MappingProxyType(
            {
                key: value
                for key, value in reserved_corridors.planned_order_coordinates.items()
                if key != (*S10_L0, 2)
            }
        ),
    )
    return prior_plan, reserved_corridors


def _with_ambiguous_shared_reservation(prior_plan):
    shared = _shared_seed77_reservation(prior_plan)
    s10_member_id = next(
        member_id
        for member_id, edge in _edge_by_member_id(prior_plan).items()
        if edge == S10_L0
    )
    conflicting = replace(
        shared,
        id=RouteReservationId(f"{shared.id}:ambiguous"),
        claims=tuple(
            replace(claim, allocation_coordinate=claim.allocation_coordinate + 4.0)
            if str(claim.member_id) == s10_member_id
            else claim
            for claim in shared.claims
        ),
    )
    return replace(
        prior_plan,
        reservations=(*prior_plan.reservations, conflicting),
    )


def _target_by_identity(targets):
    return {(target.member_id, target.edge_key): target for target in targets}


def _allocation_key(allocation):
    return (
        allocation.member_id,
        allocation.edge_key,
        allocation.segment_rank,
        allocation.axis,
    )


def _assert_opposite_direction_allocations_clear(ledger, plan) -> None:
    direction_by_claim = {claim.claim_id: claim.direction for claim in ledger.claims}
    allocations = plan.allocations
    for rank, left in enumerate(allocations):
        for right in allocations[rank + 1 :]:
            if (
                left.axis != right.axis
                or direction_by_claim[left.claim_id]
                is direction_by_claim[right.claim_id]
                or max(left.longitudinal_start, right.longitudinal_start)
                >= min(left.longitudinal_end, right.longitudinal_end)
            ):
                continue
            required = cotravelling_lane_clearance(
                same_line=left.edge_key[2] == right.edge_key[2],
                counter_running=True,
                curve_radius=CURVE_RADIUS,
            )
            assert abs(left.coordinate - right.coordinate) >= required


def _assert_allocation_segment(plan, allocation) -> None:
    start, end = plan.points[allocation.segment_rank : allocation.segment_rank + 2]
    assert (start[allocation.axis], end[allocation.axis]) == pytest.approx(
        (allocation.coordinate, allocation.coordinate)
    )
    assert sorted((start[1 - allocation.axis], end[1 - allocation.axis])) == (
        pytest.approx((allocation.longitudinal_start, allocation.longitudinal_end))
    )


def test_seed77_closes_one_snapshot_into_complete_atomic_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        graph,
        planning,
        prior_plan,
        reserved_corridors,
        _ctx,
        (ledger, source_targets, compiled_plan),
        _published_plan,
        compilation_calls,
    ) = _capture_settled_seed77(monkeypatch, capture_compilation="all")
    plan = _cohort_plan(graph, planning, prior_plan, reserved_corridors)

    assert compiled_plan is plan
    assert planning.member_geometry.corridor_cohorts is plan
    assert len(ledger.claims) == 59
    classified_claim_ids = {
        claim_id
        for component in plan.components
        for claim_id, _role in component.claim_roles
    }
    assert sum(len(component.claim_roles) for component in plan.components) == 44
    assert len(classified_claim_ids) == 44
    assert sum(len(component.problems) for component in plan.components) == 9
    assert len(plan.allocations) == 25
    assert len({allocation.claim_id for allocation in plan.allocations}) == 25
    assert len(plan.protected_segments) == len(set(plan.protected_segments))
    assert all(
        component.allocations == () and component.protected_segments == ()
        for component in plan.components
        if component.status is not CorridorAllocationStatus.PLANNED
    )
    physical_claim_ids = tuple(
        claim_id
        for component in plan.components
        for problem in component.problems
        for claim_id in (
            *(lane.member_id for lane in problem.lanes),
            *(obstacle.obstacle_id for obstacle in problem.obstacles),
        )
    )
    roles_by_claim_id = {
        claim_id: role
        for component in plan.components
        for claim_id, role in component.claim_roles
    }
    assert set(physical_claim_ids) == classified_claim_ids
    assert set(roles_by_claim_id) == classified_claim_ids
    assert all(
        len(component.problems) == len(component.results)
        for component in plan.components
    )
    source_by_identity = _target_by_identity(compilation_calls[0][3])
    for allocation in plan.allocations:
        route = source_by_identity[(allocation.member_id, allocation.edge_key)].route
        start, end = route.points[allocation.segment_rank : allocation.segment_rank + 2]
        longitudinal = sorted((start[1 - allocation.axis], end[1 - allocation.axis]))
        assert longitudinal == pytest.approx(
            (allocation.longitudinal_start, allocation.longitudinal_end)
        )

    target = {
        allocation.edge_key: allocation.coordinate
        for allocation in plan.allocations
        if allocation.edge_key in TARGET_KEYS
        and allocation.segment_rank == 3
        and allocation.axis == 0
    }
    assert set(target) == TARGET_KEYS
    assert target[S9_L1] > target[S9_L3]
    assert target[S17_L0] > target[S17_L3]
    assert all(
        component.status is CorridorAllocationStatus.PLANNED
        for component in _target_components(plan, prior_plan)
    )


def test_no_solver_owner_contains_opposite_running_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, planning, prior_plan, reserved_corridors, ctx = _capture_settled_seed77(
        monkeypatch
    )
    plan = _cohort_plan(graph, planning, prior_plan, reserved_corridors)
    scaffold = planning.exit_turns.scaffold
    assert scaffold is not None
    ledger = build_corridor_cohort_ledger(
        graph,
        scaffold,
        prior_plan,
        station_offsets=ctx.station_offsets or {},
    )
    direction_by_claim = {claim.claim_id: claim.direction for claim in ledger.claims}

    for component in plan.components:
        for problem in component.problems:
            directions_by_cohort = {}
            for lane in problem.lanes:
                directions_by_cohort.setdefault(lane.cohort_id, set()).add(
                    direction_by_claim[lane.member_id]
                )
            assert all(
                len(directions) == 1 for directions in directions_by_cohort.values()
            )

            directions_by_equality_owner = {}
            for equality in problem.equalities:
                directions_by_equality_owner.setdefault(
                    equality.owner_id, set()
                ).update(
                    (
                        direction_by_claim[equality.left_member_id],
                        direction_by_claim[equality.right_member_id],
                    )
                )
            assert all(
                len(directions) == 1
                for directions in directions_by_equality_owner.values()
            )


def test_direction_views_preserve_cross_view_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, planning, prior_plan, reserved_corridors, ctx = _capture_settled_seed77(
        monkeypatch
    )
    plan = _cohort_plan(graph, planning, prior_plan, reserved_corridors)
    scaffold = planning.exit_turns.scaffold
    assert scaffold is not None
    ledger = build_corridor_cohort_ledger(
        graph,
        scaffold,
        prior_plan,
        station_offsets=ctx.station_offsets or {},
    )
    direction_by_claim = {claim.claim_id: claim.direction for claim in ledger.claims}
    final_coordinates = {
        allocation.claim_id: allocation.coordinate for allocation in plan.allocations
    }
    opposite_overlaps = {}
    for component in plan.components:
        for problem in component.problems:
            for lane in problem.lanes:
                for obstacle in problem.obstacles:
                    if direction_by_claim[lane.member_id] is direction_by_claim[
                        obstacle.obstacle_id
                    ] or max(lane.span_start, obstacle.span_start) >= min(
                        lane.span_end, obstacle.span_end
                    ):
                        continue
                    pair = frozenset((lane.member_id, obstacle.obstacle_id))
                    opposite_overlaps[pair] = abs(
                        final_coordinates[lane.member_id]
                        - final_coordinates.get(
                            obstacle.obstacle_id,
                            obstacle.realised_coordinate,
                        )
                    )

    assert len(final_coordinates) == len(plan.allocations) == 25
    assert opposite_overlaps
    assert min(opposite_overlaps.values()) >= 4.0
    assert all(
        direction_by_claim[equality.left_member_id]
        is direction_by_claim[equality.right_member_id]
        for component in plan.components
        for problem in component.problems
        for equality in problem.equalities
    )
    _assert_opposite_direction_allocations_clear(ledger, plan)


def test_destination_endpoint_rank_controls_vertical_peel_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        graph,
        _planning,
        _prior_plan,
        _reserved_corridors,
        ctx,
        (ledger, _source_targets, plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    claims = {claim.claim_id: claim for claim in ledger.claims}
    vertical_allocations = {
        allocation.edge_key: allocation
        for allocation in plan.allocations
        if allocation.segment_rank == 3
        and allocation.axis == 0
        and allocation.edge_key in {S16_L2, S16_L1, S9_L1, S9_L3, S17_L0, S17_L3}
    }

    assert set(vertical_allocations) == {
        S16_L2,
        S16_L1,
        S9_L1,
        S9_L3,
        S17_L0,
        S17_L3,
    }
    for port_id, edge_keys, lead_direction in (
        ("s16__entry_right_18", (S16_L2, S16_L1), Direction.L),
        ("s9__entry_right_25", (S9_L1, S9_L3), Direction.L),
        ("s17__entry_right_26", (S17_L0, S17_L3), Direction.L),
    ):
        port = graph.stations[port_id]
        canonical_order = port_arrival_order(
            graph,
            port,
            ctx.station_offsets,
        )
        ordered_edges = sorted(
            edge_keys,
            key=lambda edge_key: (
                claims[vertical_allocations[edge_key].claim_id].endpoint_network_rank
            ),
        )
        assert [edge_key[2] for edge_key in ordered_edges] == [
            line_id
            for line_id in canonical_order
            if line_id in {key[2] for key in edge_keys}
        ]
        left, right = ordered_edges
        port_delta = port_lane_coord(
            graph,
            port,
            right[2],
            ctx.station_offsets,
        ) - port_lane_coord(
            graph,
            port,
            left[2],
            ctx.station_offsets,
        )
        carrier_direction = claims[vertical_allocations[left].claim_id].direction
        carrier_delta = (
            vertical_allocations[right].coordinate
            - vertical_allocations[left].coordinate
        )
        turn_sign = right_normal_axis_sign(carrier_direction) / right_normal_axis_sign(
            lead_direction
        )
        assert port_delta > 0
        assert carrier_delta * turn_sign > 0

    allocation_claim_ids = {allocation.claim_id for allocation in plan.allocations}
    roles = {
        claim_id: role
        for component in plan.components
        for claim_id, role in component.claim_roles
    }
    destination_edges = {
        S16_L2,
        S16_L1,
        S9_L1,
        S9_L3,
        S17_L0,
        S17_L3,
    }
    carrier_claims = tuple(
        claim
        for claim in ledger.claims
        if claim.edge_key in destination_edges and claim.destination_boundary_carrier
    )
    source_claims = tuple(
        claim
        for claim in ledger.claims
        if claim.edge_key in destination_edges and claim.segment_rank == 1
    )
    assert {(claim.edge_key, claim.segment_rank) for claim in carrier_claims} == {
        (edge_key, 3) for edge_key in destination_edges
    }
    assert source_claims
    assert all(not claim.destination_boundary_carrier for claim in source_claims)
    assert all(claim.claim_id not in allocation_claim_ids for claim in source_claims)
    assert all(
        roles[claim.claim_id] is CorridorCohortClaimRole.FIXED
        if claim.claim_id in roles
        else claim.endpoint_cohort_id is None
        for claim in source_claims
    )


@pytest.mark.parametrize("evidence", ("missing", "ambiguous"))
def test_incomplete_target_evidence_is_atomic_compatibility_without_ownership(
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    (
        graph,
        planning,
        prior_plan,
        reserved_corridors,
        ctx,
        (_ledger, source_targets, _compiled_plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    if evidence == "missing":
        prior_plan, reserved_corridors = _without_s10_witness(
            prior_plan, reserved_corridors
        )
    else:
        prior_plan = _with_ambiguous_shared_reservation(prior_plan)
    scaffold = planning.exit_turns.scaffold
    assert scaffold is not None

    ledger = build_corridor_cohort_ledger(
        graph,
        scaffold,
        prior_plan,
        station_offsets=ctx.station_offsets or {},
    )
    plan = compile_corridor_cohort_plan(ledger, copy.deepcopy(source_targets))
    components = _target_components(plan, prior_plan)

    assert components
    assert all(
        component.status is CorridorAllocationStatus.COMPATIBILITY
        for component in components
    )
    assert all(component.allocations == () for component in components)
    assert all(component.protected_segments == () for component in components)
    target_members = {
        member_id
        for member_id, edge in _edge_by_member_id(prior_plan).items()
        if edge in TARGET_KEYS
    }
    assert not target_members & {
        member_id for member_id, _rank in plan.protected_segments
    }


def test_ownerless_endpoint_cohort_is_explicit_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, planning, prior_plan, reserved_corridors, _ctx = _capture_settled_seed77(
        monkeypatch
    )
    plan = _cohort_plan(graph, planning, prior_plan, reserved_corridors)

    ownerless = tuple(
        component
        for component in plan.components
        if component.endpoint_cohort_ids
        and not component.claim_roles
        and not component.problems
    )
    assert ownerless
    assert all(
        component.status is CorridorAllocationStatus.COMPATIBILITY
        and component.allocations == ()
        and component.protected_segments == ()
        for component in ownerless
    )


def test_finalized_endpoint_landings_use_the_canonical_current_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _graph,
        _planning,
        _prior_plan,
        _reserved_corridors,
        _ctx,
        (_ledger, _source_targets, _compiled_plan),
        published_plan,
        compilation_calls,
    ) = _capture_settled_seed77(monkeypatch, capture_compilation="all")
    published_by_edge = {
        _edge_key(plan): plan for plan in published_plan.member_geometry_plans
    }
    _final_ledger, final_source_targets, final_plan, _realized_targets = (
        compilation_calls[-1]
    )
    final_targets = _target_by_identity(final_source_targets)
    landing_edges = {landing.edge_key for landing in final_plan.landings}

    assert {edge[1] for edge in landing_edges} >= {
        "s9__entry_right_25",
        "s16__entry_right_18",
        "s19__entry_right_24",
    }
    assert landing_edges <= set(published_by_edge)
    for landing in final_plan.landings:
        target = final_targets[(landing.member_id, landing.edge_key)]
        assert target.endpoint_lane_axis == landing.axis
        assert target.endpoint_lane_coordinate == pytest.approx(landing.coordinate)
        plan = published_by_edge[landing.edge_key]
        assert plan.points[-1][landing.axis] == pytest.approx(landing.coordinate)
        assert plan.points[-2][landing.axis] == pytest.approx(landing.coordinate)


def test_fixed_endpoint_landing_is_planned_clear_of_an_unclaimed_fixed_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilations = []
    original_compile = member_geometry_routing.compile_corridor_cohort_plan

    def capture_compile(ledger, targets, *, scalar_requests=()):
        plan = original_compile(ledger, targets, scalar_requests=scalar_requests)
        compilations.append((ledger, plan, copy.deepcopy(tuple(targets))))
        return plan

    monkeypatch.setattr(
        member_geometry_routing,
        "compile_corridor_cohort_plan",
        capture_compile,
    )
    graph = prepare_graph(SEED15.read_text(), source_dir=str(SEED15.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    route_plan = observed.route_plan
    assert route_plan is not None
    fresh_compilations = tuple(
        capture
        for capture in compilations
        if capture[0].finalized_owned_segments is None
    )
    assert len(fresh_compilations) == 1
    _ledger, cohort_plan, realized_targets = fresh_compilations[0]
    carrier = next(
        allocation
        for allocation in cohort_plan.allocations
        if allocation.edge_key == SEED15_LANDING
        and allocation.segment_rank == 3
        and allocation.axis == 0
    )
    landing = next(
        item
        for item in cohort_plan.landings
        if item.edge_key == SEED15_LANDING and item.segment_rank == 4 and item.axis == 1
    )
    published = next(
        item
        for item in route_plan.member_geometry_plans
        if _edge_key(item) == SEED15_LANDING
    )
    fixed_target = next(
        target for target in realized_targets if target.edge_key == SEED15_FIXED_LEAD
    )
    fixed = fixed_target.route
    landing_segment = published.points[landing.segment_rank : landing.segment_rank + 2]
    fixed_lead = tuple(fixed.points[:2])

    assert published.points[carrier.segment_rank][carrier.axis] == pytest.approx(
        carrier.coordinate
    )
    assert published.points[carrier.segment_rank + 1][carrier.axis] == pytest.approx(
        carrier.coordinate
    )
    assert landing_segment[0][landing.axis] == pytest.approx(landing.coordinate)
    assert landing_segment[1][landing.axis] == pytest.approx(landing.coordinate)
    assert fixed_lead[0][1] == pytest.approx(fixed_lead[1][1])
    assert landing_segment[0][1] == pytest.approx(landing_segment[1][1])
    assert landing_segment[0][1] == pytest.approx(fixed_lead[0][1])
    landing_span = sorted(point[0] for point in landing_segment)
    fixed_span = sorted(point[0] for point in fixed_lead)
    assert landing_span[1] + COORD_TOLERANCE <= fixed_span[0]


def _capture_seed15_fresh_compilation(monkeypatch: pytest.MonkeyPatch):
    captures = []
    original_compile = member_geometry_routing.compile_corridor_cohort_plan

    def capture_compile(ledger, targets, *, scalar_requests=()):
        source_targets = copy.deepcopy(tuple(targets))
        plan = original_compile(ledger, targets, scalar_requests=scalar_requests)
        if ledger.finalized_owned_segments is None:
            captures.append((ledger, source_targets, plan))
        return plan

    monkeypatch.setattr(
        member_geometry_routing,
        "compile_corridor_cohort_plan",
        capture_compile,
    )
    monkeypatch.setattr(
        render_svg_module,
        "assert_render_curve_invariants",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        render_svg_module,
        "assert_render_layout_invariants",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        render_svg_module,
        "assert_reservations_are_settled",
        lambda *_args, **_kwargs: None,
    )
    graph = prepare_graph(SEED15.read_text(), source_dir=str(SEED15.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_observed_render_plan(graph, resolve_theme(None, graph))
    assert len(captures) == 1
    return captures[0]


def test_fixed_predecessor_order_witness_orients_the_endpoint_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, source_targets, _plan = _capture_seed15_fresh_compilation(monkeypatch)
    target_source = "s8__exit_left_8"
    target_name = "s10__entry_right_19"

    def snapshot(candidate_ledger, candidate_targets):
        plan = compile_corridor_cohort_plan(candidate_ledger, candidate_targets)
        allocations = tuple(
            sorted(
                (
                    item.edge_key[2],
                    item.segment_rank,
                    item.axis,
                    item.coordinate,
                )
                for item in plan.allocations
                if item.edge_key[:2] == (target_source, target_name)
            )
        )
        landings = tuple(
            sorted(
                (item.edge_key[2], item.segment_rank, item.axis, item.coordinate)
                for item in plan.landings
                if item.edge_key[:2] == (target_source, target_name)
            )
        )
        return allocations, landings

    variants = (
        snapshot(ledger, copy.deepcopy(source_targets)),
        snapshot(
            replace(ledger, claims=tuple(reversed(ledger.claims))),
            copy.deepcopy(source_targets),
        ),
        snapshot(ledger, copy.deepcopy(tuple(reversed(source_targets)))),
    )
    assert all(candidate == variants[0] for candidate in variants[1:])
    allocations, landings = variants[0]
    carrier_y = {
        line_id: coordinate
        for line_id, segment_rank, axis, coordinate in allocations
        if segment_rank == 2 and axis == 1
    }
    landing_y = {
        line_id: coordinate
        for line_id, _segment_rank, axis, coordinate in landings
        if axis == 1
    }
    targets_by_edge = {target.edge_key: target for target in source_targets}
    incoming_x = {
        line_id: targets_by_edge[(target_source, target_name, line_id)].route.points[1][
            0
        ]
        for line_id in ("l0", "l1")
    }

    assert incoming_x["l0"] > incoming_x["l1"]
    assert carrier_y["l0"] > carrier_y["l1"]
    assert landing_y["l0"] > landing_y["l1"]


def test_conflicting_fixed_predecessor_order_witnesses_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, source_targets, _plan = _capture_seed15_fresh_compilation(monkeypatch)
    predecessor_claims = tuple(
        claim
        for claim in ledger.claims
        if claim.edge_key is not None
        and claim.edge_key[:2] == ("s8__exit_left_8", "s10__entry_right_19")
        and claim.segment_rank == 1
        and claim.endpoint_cohort_id is None
    )
    assert {claim.edge_key[2] for claim in predecessor_claims} == {"l0", "l1"}
    conflicting_claims = tuple(
        replace(
            claim,
            claim_id=f"{claim.claim_id}|conflicting-witness",
            reservation_id="conflicting-fixed-predecessor-order",
            reservation_rank=len(ledger.claims),
            lane_rank=0 if claim.edge_key[2] == "l0" else 1,
        )
        for claim in predecessor_claims
    )
    conflicting_ledger = replace(
        ledger,
        claims=ledger.claims + conflicting_claims,
    )

    with pytest.raises(
        CorridorCohortCompilationError,
        match="conflicting fixed predecessor order witnesses",
    ):
        compile_corridor_cohort_plan(
            conflicting_ledger,
            copy.deepcopy(source_targets),
        )


def test_curve_guard_observes_the_mandatory_canonical_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    top_crossings: dict[str, set[float]] = {}
    original_compile = member_geometry_routing.compile_corridor_cohort_plan
    original_guard = render_svg_module.assert_render_curve_invariants

    def capture_compile(ledger, targets, *, scalar_requests=()):
        for target in targets:
            if target.edge_key[1] != "orf_calling__entry_top_9":
                continue
            assert target.endpoint_lane_axis == 0
            assert target.endpoint_lane_coordinate is not None
            top_crossings.setdefault(target.edge_key[2], set()).add(
                target.endpoint_lane_coordinate
            )
        plan = original_compile(ledger, targets, scalar_requests=scalar_requests)
        events.append(
            "fresh-compile"
            if ledger.finalized_owned_segments is None
            else "replay-compile"
        )
        return plan

    def capture_guard(graph, routes, station_offsets):
        events.append("curve-guard")
        return original_guard(graph, routes, station_offsets)

    monkeypatch.setattr(
        member_geometry_routing,
        "compile_corridor_cohort_plan",
        capture_compile,
    )
    monkeypatch.setattr(
        render_svg_module,
        "assert_render_curve_invariants",
        capture_guard,
    )
    graph = prepare_graph(
        OPPOSING_BYPASS.read_text(), source_dir=str(OPPOSING_BYPASS.parent)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_observed_render_plan(graph, resolve_theme(None, graph))

    first_guard = events.index("curve-guard")
    assert "fresh-compile" in events[:first_guard]
    assert "replay-compile" in events[:first_guard]
    assert top_crossings == {"ribo": {692.0}, "rnaseq": {696.0}}


def test_aperture_grant_has_one_ordered_planning_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_compile = member_geometry_routing.compile_corridor_cohort_plan
    original_settle = render_svg_module.settle_route_envelopes
    original_transition = render_svg_module._assert_clearance_grant_transition
    original_guard = render_svg_module.assert_render_curve_invariants

    def capture_compile(ledger, targets, *, scalar_requests=()):
        plan = original_compile(ledger, targets, scalar_requests=scalar_requests)
        events.append(
            "cohort-intent"
            if ledger.finalized_owned_segments is None
            else "cohort-final"
        )
        return plan

    def capture_settle(graph, plan, *args, **kwargs):
        kinds = {item.kind for item in plan.boundary_clearance_requirements}
        if render_svg_module.BoundaryClearanceRequirementKind.GENERAL in kinds:
            events.append("general-grant")
        if (
            render_svg_module.BoundaryClearanceRequirementKind.CORRIDOR_COHORT_APERTURE
            in kinds
        ):
            events.append("aperture-grant")
        return original_settle(graph, plan, *args, **kwargs)

    def capture_transition(*args, **kwargs):
        original_transition(*args, **kwargs)
        events.append("aperture-transition")

    def capture_guard(*args, **kwargs):
        original_guard(*args, **kwargs)
        events.append("final-curve-guard")

    monkeypatch.setattr(
        member_geometry_routing,
        "compile_corridor_cohort_plan",
        capture_compile,
    )
    monkeypatch.setattr(render_svg_module, "settle_route_envelopes", capture_settle)
    monkeypatch.setattr(
        render_svg_module,
        "_assert_clearance_grant_transition",
        capture_transition,
    )
    monkeypatch.setattr(
        render_svg_module,
        "assert_render_curve_invariants",
        capture_guard,
    )

    graph = prepare_graph(SEED77.read_text(), source_dir=str(SEED77.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_observed_render_plan(graph, resolve_theme(None, graph))

    assert events.count("general-grant") <= 1
    assert events.count("aperture-grant") == 1
    assert events.count("aperture-transition") == 1
    assert events.count("final-curve-guard") == 1
    intent = events.index("cohort-intent")
    grant = events.index("aperture-grant")
    final = events.index("cohort-final", grant)
    transition = events.index("aperture-transition")
    guard = events.index("final-curve-guard")
    assert intent < grant < final < transition < guard
    assert "cohort-intent" not in events[transition + 1 :]


def test_canonical_cohort_render_is_geometry_idempotent() -> None:
    def geometry(graph):
        return {
            **{
                f"section:{key}": (
                    section.bbox_x,
                    section.bbox_y,
                    section.bbox_w,
                    section.bbox_h,
                )
                for key, section in graph.sections.items()
            },
            **{
                f"station:{key}": (station.x, station.y)
                for key, station in graph.stations.items()
            },
            **{f"port:{key}": (port.x, port.y) for key, port in graph.ports.items()},
        }

    graph = prepare_graph(SEED77.read_text(), source_dir=str(SEED77.parent))
    theme = resolve_theme(None, graph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        once = render_svg_module._settled_render_graph(graph, theme)
        twice = render_svg_module._settled_render_graph(once, theme)

    assert geometry(twice) == geometry(once)


@pytest.mark.parametrize("mismatch", ("missing", "identity", "geometry"))
def test_compilation_is_typed_exact_and_rejects_the_whole_plan_atomically(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    (
        _graph,
        _planning,
        _prior_plan,
        _reserved_corridors,
        _ctx,
        (ledger, source_targets, _compiled_plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    targets = copy.deepcopy(source_targets)
    plan = compile_corridor_cohort_plan(ledger, targets)

    routes_by_identity = {
        (target.member_id, target.edge_key): target.route for target in targets
    }
    for allocation in plan.allocations:
        route = routes_by_identity[(allocation.member_id, allocation.edge_key)]
        assert route.points[allocation.segment_rank][allocation.axis] == (
            allocation.coordinate
        )
        assert route.points[allocation.segment_rank + 1][allocation.axis] == (
            allocation.coordinate
        )
        assert allocation.segment_rank in route.route_system_owned_segment_ranks

    rejected_targets = list(copy.deepcopy(source_targets))
    claim = next(
        item for item in ledger.claims if item.claim_id == plan.allocations[0].claim_id
    )
    target_rank = next(
        rank
        for rank, target in enumerate(rejected_targets)
        if target.member_id == claim.member_id and target.edge_key == claim.edge_key
    )
    if mismatch == "missing":
        del rejected_targets[target_rank]
    elif mismatch == "identity":
        target = rejected_targets[target_rank]
        rejected_targets[target_rank] = replace(
            target,
            member_geometry_plan_id=f"{target.member_geometry_plan_id}:wrong",
        )
    else:
        target = rejected_targets[target_rank]
        axis = int(claim.orientation is CorridorOrientation.HORIZONTAL)
        longitudinal_axis = 1 - axis
        start = target.route.points[claim.segment_rank]
        end = list(target.route.points[claim.segment_rank + 1])
        end[longitudinal_axis] = start[longitudinal_axis]
        target.route.points[claim.segment_rank + 1] = end[0], end[1]
    before = freeze_render_value(tuple(target.route for target in rejected_targets))

    with pytest.raises(CorridorCohortCompilationError):
        compile_corridor_cohort_plan(ledger, rejected_targets)

    assert freeze_render_value(tuple(target.route for target in rejected_targets)) == (
        before
    )


def test_coordinate_free_ledger_compiles_against_a_translated_current_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _graph,
        _planning,
        _prior_plan,
        _reserved_corridors,
        _ctx,
        (ledger, source_targets, _compiled_plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    forbidden = {"longitudinal_start", "longitudinal_end", "allocation_coordinate"}
    assert forbidden.isdisjoint(CorridorCohortLedgerClaim.__dataclass_fields__)

    baseline = compile_corridor_cohort_plan(ledger, copy.deepcopy(source_targets))
    shifted_targets = copy.deepcopy(source_targets)
    dx, dy = 37.0, 53.0
    translated_targets = []
    for target in shifted_targets:
        route = target.route
        route.points[:] = [(x + dx, y + dy) for x, y in route.points]
        route.exit_shared_opening_points = tuple(
            (x + dx, y + dy) for x, y in route.exit_shared_opening_points
        )
        lane_shift = dx if target.endpoint_lane_axis == 0 else dy
        translated_targets.append(
            replace(
                target,
                endpoint_lane_coordinate=(
                    None
                    if target.endpoint_lane_coordinate is None
                    else target.endpoint_lane_coordinate + lane_shift
                ),
            )
        )
    shifted_targets = tuple(translated_targets)
    shifted = compile_corridor_cohort_plan(ledger, shifted_targets)

    baseline_by_claim = {item.claim_id: item for item in baseline.allocations}
    shifted_by_claim = {item.claim_id: item for item in shifted.allocations}
    assert set(shifted_by_claim) == set(baseline_by_claim)
    for claim_id, original in baseline_by_claim.items():
        translated = shifted_by_claim[claim_id]
        lateral_shift = dx if original.axis == 0 else dy
        longitudinal_shift = dy if original.axis == 0 else dx
        assert translated.coordinate == pytest.approx(
            original.coordinate + lateral_shift
        )
        assert translated.longitudinal_start == pytest.approx(
            original.longitudinal_start + longitudinal_shift
        )
        assert translated.longitudinal_end == pytest.approx(
            original.longitudinal_end + longitudinal_shift
        )


def test_endpoint_network_ranks_and_landings_ignore_input_permutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        graph,
        planning,
        prior_plan,
        _reserved_corridors,
        ctx,
        (ledger, source_targets, _compiled_plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    scaffold = planning.exit_turns.scaffold
    assert scaffold is not None
    reversed_ledger = build_corridor_cohort_ledger(
        graph,
        scaffold,
        replace(prior_plan, reservations=tuple(reversed(prior_plan.reservations))),
        station_offsets=ctx.station_offsets or {},
    )

    def endpoint_ranks(candidate_ledger):
        return {
            claim.member_id: claim.endpoint_network_rank
            for claim in candidate_ledger.claims
            if claim.endpoint_cohort_id is not None
        }

    forbidden = {"longitudinal_start", "longitudinal_end", "allocation_coordinate"}
    assert forbidden.isdisjoint(CorridorCohortLedgerClaim.__dataclass_fields__)
    assert endpoint_ranks(reversed_ledger) == endpoint_ranks(ledger)

    variants = (
        compile_corridor_cohort_plan(ledger, copy.deepcopy(source_targets)),
        compile_corridor_cohort_plan(
            reversed_ledger,
            copy.deepcopy(source_targets),
        ),
        compile_corridor_cohort_plan(
            ledger,
            copy.deepcopy(tuple(reversed(source_targets))),
        ),
        compile_corridor_cohort_plan(
            reversed_ledger,
            copy.deepcopy(tuple(reversed(source_targets))),
        ),
    )
    landing_maps = [
        {
            _allocation_key(landing): (
                landing.coordinate,
                landing.longitudinal_start,
                landing.longitudinal_end,
            )
            for landing in variant.landings
        }
        for variant in variants
    ]
    assert all(candidate == landing_maps[0] for candidate in landing_maps[1:])


def test_finalized_ledger_realizes_exact_keys_in_the_current_coordinate_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _graph,
        _planning,
        _prior_plan,
        _reserved_corridors,
        _ctx,
        (ledger, source_targets, _compiled_plan),
    ) = _capture_settled_seed77(monkeypatch, capture_compilation=True)
    targets = copy.deepcopy(source_targets)
    granted = compile_corridor_cohort_plan(ledger, targets)
    finalized = replace(
        ledger,
        finalized_owned_segments=frozenset(
            (item.member_id, item.edge_key, item.segment_rank)
            for item in (*granted.allocations, *granted.landings)
        ),
    )
    dx, dy = 37.0, 53.0
    translated_targets = []
    for target in targets:
        route = target.route
        route.points[:] = [(x + dx, y + dy) for x, y in route.points]
        route.exit_shared_opening_points = tuple(
            (x + dx, y + dy) for x, y in route.exit_shared_opening_points
        )
        lane_shift = dx if target.endpoint_lane_axis == 0 else dy
        translated_targets.append(
            replace(
                target,
                endpoint_lane_coordinate=(
                    None
                    if target.endpoint_lane_coordinate is None
                    else target.endpoint_lane_coordinate + lane_shift
                ),
            )
        )
    targets = tuple(translated_targets)
    solve_calls = []
    original_solve = cohort_integration.solve_corridor_cohorts

    def observe_solve(problem):
        solve_calls.append(problem)
        return original_solve(problem)

    monkeypatch.setattr(cohort_integration, "solve_corridor_cohorts", observe_solve)
    replay = compile_corridor_cohort_plan(finalized, targets)

    assert solve_calls
    assert {_allocation_key(item) for item in replay.allocations} == {
        _allocation_key(item) for item in granted.allocations
    }
    assert {_allocation_key(item) for item in replay.landings} == {
        _allocation_key(item) for item in granted.landings
    }
    granted_by_key = {_allocation_key(item): item for item in granted.allocations}
    for item in replay.allocations:
        original = granted_by_key[_allocation_key(item)]
        lateral_shift = dx if item.axis == 0 else dy
        longitudinal_shift = dy if item.axis == 0 else dx
        assert item.coordinate == pytest.approx(original.coordinate + lateral_shift)
        assert item.longitudinal_start == pytest.approx(
            original.longitudinal_start + longitudinal_shift
        )
        assert item.longitudinal_end == pytest.approx(
            original.longitudinal_end + longitudinal_shift
        )
    granted_landings = {_allocation_key(item): item for item in granted.landings}
    for item in replay.landings:
        original = granted_landings[_allocation_key(item)]
        lane_shift = dx if item.axis == 0 else dy
        assert item.coordinate == pytest.approx(original.coordinate + lane_shift)
    _assert_opposite_direction_allocations_clear(finalized, replay)


def test_wireup_rebuilds_all_references_and_bypasses_legacy_rank_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_observations = []
    merge_observations = []
    events = []
    route_identity_by_id = {}
    route_refs = []

    original_compile = member_geometry_routing.compile_corridor_cohort_plan

    def observe_compile(ledger, targets, *, scalar_requests=()):
        route_refs.extend(target.route for target in targets)
        route_identity_by_id.update(
            (id(target.route), target.edge_key) for target in targets
        )
        events.append(
            (
                "compile",
                "fresh" if ledger.finalized_owned_segments is None else "replay",
                frozenset(id(target.route) for target in targets),
            )
        )
        return original_compile(ledger, targets, scalar_requests=scalar_requests)

    monkeypatch.setattr(
        member_geometry_routing,
        "compile_corridor_cohort_plan",
        observe_compile,
    )

    def install_spy(name):
        original = getattr(member_geometry_routing, name)

        def spy(routes, *args, **kwargs):
            route_refs.extend(routes)
            route_identity_by_id.update(
                (
                    id(route),
                    (route.edge.source, route.edge.target, route.line_id),
                )
                for route in routes
            )
            events.append(("pass", name, frozenset(id(route) for route in routes)))
            before = {
                (id(route), rank): tuple(route.points[rank : rank + 2])
                for route in routes
                for rank in route.route_system_owned_segment_ranks
                if rank + 1 < len(route.points)
            }
            result = original(routes, *args, **kwargs)
            routes_by_id = {id(route): route for route in routes}
            after = {
                key: tuple(routes_by_id[key[0]].points[key[1] : key[1] + 2])
                for key in before
            }
            if before:
                pass_observations.append((name, before, after))
            return result

        monkeypatch.setattr(member_geometry_routing, name, spy)

    original_merge = routing_planning._with_settled_exit_turns

    def capture_merge(execution, allocation, ctx):
        merged = original_merge(execution, allocation, ctx)
        if execution.corridor_cohorts is not None:
            merge_observations.append((execution, merged))
        return merged

    monkeypatch.setattr(routing_planning, "_with_settled_exit_turns", capture_merge)
    install_spy("_materialize_trunk_slots")
    install_spy("_hold_runs_in_corridor_clearance")
    install_spy("_separate_fused_cotravelling_runs")
    install_spy("_separate_declared_opposing_gap_bundles")
    install_spy("_dogleg_off_exempt_trunks")
    (
        graph,
        planning,
        prior_plan,
        reserved_corridors,
        _ctx,
        (initial_ledger, _source_targets, compiled_plan),
        published_plan,
        compilation_calls,
    ) = _capture_settled_seed77(monkeypatch, capture_compilation="all")
    plan = _cohort_plan(graph, planning, prior_plan, reserved_corridors)
    granting_compilation_rank = next(
        rank
        for rank, call in enumerate(compilation_calls)
        if call[0].finalized_owned_segments is None and call[2].allocations
    )
    granted_keys = {_allocation_key(item) for item in plan.allocations}
    granted_landing_keys = {_allocation_key(item) for item in plan.landings}
    persisted_ledger = published_plan.corridor_cohort_ledger

    assert compiled_plan is plan
    assert persisted_ledger is not None
    assert persisted_ledger.finalized_owned_segments is not None
    assert persisted_ledger.claims == initial_ledger.claims
    assert persisted_ledger.endpoint_members == initial_ledger.endpoint_members
    assert persisted_ledger.finalized_owned_segments == frozenset(
        (item.member_id, item.edge_key, item.segment_rank)
        for item in (*plan.allocations, *plan.landings)
    )
    assert granted_keys
    assert granted_landing_keys
    assert granting_compilation_rank == 0
    assert (
        sum(call[0].finalized_owned_segments is None for call in compilation_calls) == 1
    )
    assert (
        sum(
            call[0].finalized_owned_segments is None and bool(call[2].allocations)
            for call in compilation_calls
        )
        == 1
    )
    assert all(
        call[0].finalized_owned_segments is not None
        for call in compilation_calls[granting_compilation_rank + 1 :]
    )
    assert all(
        call[0] == persisted_ledger
        for call in compilation_calls[granting_compilation_rank + 1 :]
    )
    compile_events = tuple(event for event in events if event[0] == "compile")
    assert sum(event[1] == "fresh" for event in compile_events) == 1
    post_compile_legacy_routes = tuple(
        (
            _phase,
            _later_name,
            tuple(
                sorted(
                    route_identity_by_id[route_id]
                    for route_id in compiled_route_ids.intersection(pass_route_ids)
                )
            ),
        )
        for event_rank, (_kind, _phase, compiled_route_ids) in enumerate(events)
        if _kind == "compile"
        for later_kind, _later_name, pass_route_ids in events[event_rank + 1 :]
        if later_kind == "pass"
        if compiled_route_ids.intersection(pass_route_ids)
    )
    assert not post_compile_legacy_routes, post_compile_legacy_routes[0]
    for _ledger, replay_targets, replay_plan, realized_targets in compilation_calls[
        granting_compilation_rank + 1 :
    ]:
        assert {_allocation_key(item) for item in replay_plan.allocations} == (
            granted_keys
        )
        assert {_allocation_key(item) for item in replay_plan.landings} == (
            granted_landing_keys
        )
        replay_by_identity = _target_by_identity(replay_targets)
        realized_by_identity = _target_by_identity(realized_targets)
        problem_spans = {
            lane.member_id: (lane.span_start, lane.span_end)
            for component in replay_plan.components
            for problem in component.problems
            for lane in problem.lanes
        }
        for item in replay_plan.allocations:
            route = replay_by_identity[(item.member_id, item.edge_key)].route
            start, end = route.points[item.segment_rank : item.segment_rank + 2]
            assert sorted((start[1 - item.axis], end[1 - item.axis])) == (
                pytest.approx(problem_spans[item.claim_id])
            )
            _assert_allocation_segment(
                realized_by_identity[(item.member_id, item.edge_key)].route,
                item,
            )
        _assert_opposite_direction_allocations_clear(_ledger, replay_plan)
    assert planning.corridor_cohorts is plan
    assert planning.member_geometry.corridor_cohorts is plan
    assert not pass_observations
    assert {event[1] for event in events if event[0] == "pass"} == {
        "_materialize_trunk_slots",
        "_hold_runs_in_corridor_clearance",
        "_separate_fused_cotravelling_runs",
        "_separate_declared_opposing_gap_bundles",
        "_dogleg_off_exempt_trunks",
    }
    assert not merge_observations
    assert route_refs
    planned_by_edge = {_edge_key(item): item for item in planning.member_geometry.plans}
    for allocation in plan.allocations:
        member_plan = planned_by_edge[allocation.edge_key]
        assert allocation.segment_rank in (
            member_plan.corridor_cohort_owned_segment_ranks
        )
        _assert_allocation_segment(member_plan, allocation)

    by_edge = {_edge_key(item): item for item in planning.member_geometry.plans}
    target_plans = tuple(by_edge[key] for key in TARGET_KEYS)
    assert all(3 in item.corridor_cohort_owned_segment_ranks for item in target_plans)
    assert (
        not {item.member_id for item in target_plans}
        & planning.member_geometry.reconciled_member_ids
    )

    assert planning.route_systems is not None
    for system in planning.route_systems.systems:
        for member in system.members:
            if member.geometry_plan is None:
                continue
            canonical = planning.member_geometry.plan_for_edge(member.edge)
            assert member.geometry_plan is canonical
            assert str(canonical.id) in system.plan_ids

    routes = [
        fresh_member_route(
            member_plan,
            _graph_edge(graph, _edge_key(member_plan)),
        )
        for member_plan in planning.member_geometry.plans
    ]
    validate_member_geometry_emission(routes, planning.member_geometry)
    target_coordinates = {
        allocation.edge_key: allocation.coordinate
        for allocation in plan.allocations
        if allocation.edge_key in TARGET_KEYS
        and allocation.segment_rank == 3
        and allocation.axis == 0
    }
    assert set(target_coordinates) == TARGET_KEYS
    assert all(
        by_edge[key].points[3][0] == target_coordinates[key] for key in TARGET_KEYS
    )
    published_by_edge = {
        _edge_key(item): item for item in published_plan.member_geometry_plans
    }
    member_geometry_source = inspect.getsource(member_geometry_routing)
    assert "settle_shared_opening_trunk_conflicts" not in member_geometry_source
    assert "_plan_gap_channels" not in member_geometry_source
    assert all(
        3 in published_by_edge[key].corridor_cohort_owned_segment_ranks
        for key in TARGET_KEYS
    )
    matching_compilations = tuple(
        call
        for call in compilation_calls
        if {target.edge_key: tuple(target.route.points) for target in call[3]}.items()
        >= {
            edge_key: member_plan.points
            for edge_key, member_plan in published_by_edge.items()
        }.items()
    )
    assert len(matching_compilations) == 1
    published_compilation = matching_compilations[0]
    assert published_compilation[0].finalized_owned_segments is not None
    replay_targets = {
        target.edge_key: target.route for target in published_compilation[3]
    }
    assert all(
        published_by_edge[key].points[2:4] == tuple(replay_targets[key].points[2:4])
        for key in TARGET_KEYS
    )
    for allocation in published_compilation[2].allocations:
        member_plan = published_by_edge[allocation.edge_key]
        assert allocation.segment_rank in (
            member_plan.corridor_cohort_owned_segment_ranks
        )
        _assert_allocation_segment(member_plan, allocation)
