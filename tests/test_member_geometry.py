"""Non-convergence member geometry is planned once and emitted exactly."""

from pathlib import Path
from types import MappingProxyType

import pytest

import nf_metro.layout.routing.member_geometry as member_geometry
from nf_metro.api import prepare_graph
from nf_metro.layout.constants import CURVE_RADIUS, DIAGONAL_RUN
from nf_metro.layout.route_plan import (
    ConvergenceDisposition,
    EmissionMemberId,
    RouteMemberGapChannel,
    RouteMemberGeometryPlan,
    RouteMemberGeometryPlanId,
    RouteSystemDisposition,
    RouteSystemId,
    build_route_plan_query,
    build_route_semantic_scaffold,
)
from nf_metro.layout.routing.common import Direction, OffsetRegime
from nf_metro.layout.routing.context import _build_routing_context
from nf_metro.layout.routing.core import (
    _allocation_eligible_system_ids,
    _route_edges,
    observe_route_edges,
)
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.offsets import compute_station_offsets
from nf_metro.layout.routing.reserved_bands import build_reserved_corridors
from nf_metro.parser.model import Edge
from nf_metro.parser.route_topology import ResolvedEdge

ROOT = Path(__file__).parents[1]


def _observe(path: Path):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    return graph, observation


def _route_for_plan(observation, plan):
    return next(
        route
        for route in observation.routes
        if ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        == plan.edge
    )


def _assert_channels_equal_emission(observation, plan) -> None:
    route = _route_for_plan(observation, plan)
    assert route.route_system_disposition == "planned"
    assert str(plan.id) in route.route_plan_ids
    assert route.route_system_owned_segment_ranks == tuple(
        dict.fromkeys(channel.segment_rank for channel in plan.gap_channels)
    )
    assert tuple(route.gap_slots) == plan.gap_slots
    for channel in plan.gap_channels:
        assert tuple(route.points[channel.segment_rank : channel.segment_rank + 2]) == (
            channel.start,
            channel.end,
        )


def test_live_claim_index_exposes_only_eligible_prior_systems_in_order() -> None:
    failed = RouteSystemId("failed")
    survivor = RouteSystemId("survivor")
    future = RouteSystemId("future")
    claims = tuple(
        member_geometry.PreliminaryGapChannelClaim(
            system_id,
            coordinate,
            0.0,
            100.0,
            True,
            (0, 0),
            frozenset({line_id}),
        )
        for system_id, coordinate, line_id in (
            (failed, 100.0, "failed-line"),
            (survivor, 112.0, "survivor-line"),
            (future, 124.0, "future-line"),
        )
    )
    failures = MappingProxyType({failed: "canonical-template-declined-member"})
    eligible_claims = member_geometry._eligible_preliminary_gap_claims(claims, failures)
    visible = member_geometry._visible_claims_by_system_gap(
        eligible_claims, {failed: 0, survivor: 1, future: 2}
    )

    assert tuple(claim.system_id for claim in visible[(survivor, (0, 0))]) == (
        survivor,
    )
    assert tuple(claim.system_id for claim in visible[(future, (0, 0))]) == (
        survivor,
        future,
    )
    assert _allocation_eligible_system_ids(
        frozenset({failed, survivor, future}), frozenset(failures)
    ) == frozenset({survivor, future})


def test_compatibility_context_uses_and_restores_the_narrow_edge_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = ROOT / "examples" / "topologies" / "aligner_row_pinned_continuation.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    scaffold = build_route_semantic_scaffold(graph)
    assert scaffold is not None
    system_id = scaffold.ordered_system_ids[0]
    system_edges = tuple(
        edge
        for edge in scaffold.edge_order
        if scaffold.system_for_edge(edge) == system_id
    )
    assert system_edges
    ctx = _build_routing_context(
        graph,
        DIAGONAL_RUN,
        CURVE_RADIUS,
        compute_station_offsets(graph),
    )
    prior_edges = frozenset({("prior", "edge", "line")})
    ctx.compatibility_edges = prior_edges
    prior_systems = ctx.route_systems

    def stop(edge, current_ctx):
        assert current_ctx.is_compatibility_edge(edge)
        assert current_ctx.route_systems is prior_systems
        raise RuntimeError("stop after checking compatibility context")

    monkeypatch.setattr(member_geometry, "_route_compatibility_template", stop)
    with pytest.raises(RuntimeError, match="stop after checking"):
        member_geometry._append_compatibility_context(
            ctx, scaffold, system_id, system_edges
        )

    assert ctx.compatibility_edges == prior_edges
    assert ctx.route_systems is prior_systems


def test_exit_turn_channel_is_a_published_member_geometry_decision() -> None:
    _graph, observation = _observe(
        ROOT / "examples" / "topologies" / "exit_run_three_drop_columns.mmd"
    )
    plan = next(
        item
        for item in observation.plan.member_geometry_plans
        if item.edge == ResolvedEdge("__junction_9", "e__entry_left_5", "main")
    )

    assert len(plan.gap_channels) == 1
    assert plan.exit_turn_axis_id is not None
    system = next(
        item for item in observation.plan.systems if item.id == plan.system_id
    )
    assert plan.id in system.member_geometry_plan_ids
    _assert_channels_equal_emission(observation, plan)


def test_multi_gap_wrap_channels_are_planned_without_exit_or_fan_ownership() -> None:
    _graph, observation = _observe(
        ROOT / "examples" / "topologies" / "funcprofiler_upstream.mmd"
    )
    plans = tuple(
        item
        for item in observation.plan.member_geometry_plans
        if item.edge.source == "__junction_7"
        and item.edge.target == "Output__entry_left_5"
    )

    assert len(plans) == 7
    assert all(len(plan.gap_channels) == 2 for plan in plans)
    assert all(plan.exit_turn_axis_id is None for plan in plans)
    assert all(plan.fan_plan_id is None for plan in plans)
    for plan in plans:
        _assert_channels_equal_emission(observation, plan)


def test_trunk_slot_settles_before_adjacent_gap_channels_freeze() -> None:
    _graph, observation = _observe(
        ROOT / "examples" / "topologies" / "disjoint_sameline_trunks.mmd"
    )
    plan = next(
        item
        for item in observation.plan.member_geometry_plans
        if item.edge == ResolvedEdge("__junction_8", "secD__entry_left_6", "c")
    )
    route = _route_for_plan(observation, plan)

    assert plan.owned_segment_ranks == (1, 3)
    assert plan.points[2:4] == ((198.0, 200.0), (568.0, 200.0))
    assert tuple(route.points[2:4]) == plan.points[2:4]
    _assert_channels_equal_emission(observation, plan)


def test_one_segment_can_own_distinct_gap_row_claims() -> None:
    channels = (
        RouteMemberGapChannel(1, (20.0, 10.0), (20.0, 90.0), 0, 0, Direction.D),
        RouteMemberGapChannel(1, (20.0, 10.0), (20.0, 90.0), 0, 1, Direction.D),
    )
    plan = RouteMemberGeometryPlan(
        RouteMemberGeometryPlanId("plan"),
        RouteSystemId("system"),
        EmissionMemberId("member"),
        ResolvedEdge("source", "target", "line"),
        RouteFamilyId.BYPASS_FAMILY,
        ((0.0, 10.0), (20.0, 10.0), (20.0, 90.0)),
        None,
        OffsetRegime.BAKED,
        False,
        (),
        None,
        channels,
    )

    assert plan.gap_channels == channels
    route = member_geometry.fresh_member_route(plan, Edge("source", "target", "line"))
    assert plan.owned_segment_ranks == (1,)
    assert route.route_system_owned_segment_ranks == (1,)

    route.route_system_disposition = "planned"
    execution = member_geometry.MemberGeometryExecution(
        (plan,),
        MappingProxyType({}),
        MappingProxyType({plan.edge: plan}),
        MappingProxyType({plan.system_id: (plan,)}),
    )
    route.points[0] = (-10.0, 10.0)
    member_geometry.validate_member_geometry_emission([route], execution)
    route.points[2] = (20.0, 95.0)
    with pytest.raises(RuntimeError, match="channel geometry changed"):
        member_geometry.validate_member_geometry_emission([route], execution)

    with pytest.raises(ValueError, match="repeats a symbolic gap claim"):
        RouteMemberGeometryPlan(
            RouteMemberGeometryPlanId("duplicate"),
            RouteSystemId("system"),
            EmissionMemberId("member"),
            ResolvedEdge("source", "target", "line"),
            RouteFamilyId.BYPASS_FAMILY,
            ((0.0, 10.0), (20.0, 10.0), (20.0, 90.0)),
            None,
            OffsetRegime.BAKED,
            False,
            (),
            None,
            (channels[0], channels[0]),
        )


def test_reservation_reroute_keeps_identity_and_reuses_settled_template() -> None:
    graph, first = _observe(ROOT / "examples" / "genomeassembly.mmd")
    routes, _moves, second_plan = _route_edges(
        graph,
        DIAGONAL_RUN,
        CURVE_RADIUS,
        compute_station_offsets(graph),
        observe_plan=True,
        reservations=first.plan,
    )
    assert second_plan is not None
    first_by_id = {item.id: item for item in first.plan.member_geometry_plans}
    second_by_id = {item.id: item for item in second_plan.member_geometry_plans}
    shared = first_by_id.keys() & second_by_id.keys()

    assert shared
    corridors = build_reserved_corridors(graph, first.plan)
    for plan_id in shared:
        plan = second_by_id[plan_id]
        assert plan.consumed_reservation_ids == tuple(
            str(reservation.id)
            for reservation in first.plan.reservations
            if plan.member_id in reservation.claimant_member_ids
        )
        route = next(
            item
            for item in routes
            if ResolvedEdge(item.edge.source, item.edge.target, item.line_id)
            == plan.edge
        )
        for channel in plan.gap_channels:
            band = corridors.for_segment(
                plan.edge.source,
                plan.edge.target,
                plan.edge.line_id,
                channel.segment_rank,
            )
            if band is not None:
                assert band.lo <= channel.start[0] <= band.hi
            assert tuple(
                route.points[channel.segment_rank : channel.segment_rank + 2]
            ) == (channel.start, channel.end)


def test_failed_system_discards_member_geometry_before_compatibility_emission(
    monkeypatch,
) -> None:
    path = ROOT / "examples" / "topologies" / "funcprofiler_upstream.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    original = member_geometry._route_template
    failed = False

    def fail_first(edge, family_id, ctx):
        nonlocal failed
        if not failed:
            failed = True
            raise member_geometry.MemberGeometryDeclinedError("fixture decline")
        return original(edge, family_id, ctx)

    monkeypatch.setattr(member_geometry, "_route_template", fail_first)
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )

    assert failed
    failed_systems = tuple(
        system
        for system in observation.plan.systems
        if any(
            reason.owner == "member-geometry-plan"
            for reason in system.compatibility_reasons
        )
    )
    assert len(failed_systems) == 1
    assert failed_systems[0].disposition is RouteSystemDisposition.COMPATIBILITY
    assert not failed_systems[0].member_geometry_plan_ids
    assert all(
        plan.system_id != failed_systems[0].id
        for plan in observation.plan.member_geometry_plans
    )
    convergence_plans = tuple(
        plan
        for plan in observation.plan.convergence_plans
        if plan.system_id == failed_systems[0].id
    )
    assert convergence_plans
    assert all(
        plan.disposition is ConvergenceDisposition.LEGACY
        and not plan.endpoint_ownership
        and not plan.shared_reference_ids
        and not plan.demand_ids
        for plan in convergence_plans
    )
    build_route_plan_query(observation.plan)
