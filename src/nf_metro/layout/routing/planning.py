"""Shared route-system planning preparation before path emission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.route_plan import (
    EmissionMemberId,
    ExitTurnPlanId,
    RouteMemberGeometryPlan,
    RouteSystemDisposition,
    RouteSystemId,
)
from nf_metro.layout.routing import exit_turns as exit_turn_routing
from nf_metro.layout.routing.context import _RoutingCtx
from nf_metro.layout.routing.convergences import (
    ConvergencePlanExecution,
    build_convergence_plan_execution,
    empty_convergence_plan_execution,
    preliminary_member_gap_claims,
    restrict_convergence_execution,
    settle_global_convergence_execution,
    settle_preliminary_convergence_execution,
)
from nf_metro.layout.routing.exit_turns import ExitTurnExecution
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.inter_section_handlers import (
    classify_inter_section_family,
)
from nf_metro.layout.routing.member_geometry import (
    MemberGeometryExecution,
    build_member_geometry_execution,
    empty_member_geometry_execution,
)
from nf_metro.layout.routing.system_emission import (
    RouteSystemEmissionExecution,
    RouteSystemGeometryOwner,
    build_route_system_emission_execution,
    classify_route_system_dispositions,
)
from nf_metro.parser.model import MetroGraph
from nf_metro.parser.route_topology import ResolvedEdge


@dataclass(frozen=True, slots=True)
class RoutePlanningExecution:
    """Final no-emission planning state consumed by routing and diagnostics."""

    exit_turns: ExitTurnExecution
    convergences: ConvergencePlanExecution
    member_geometry: MemberGeometryExecution
    route_systems: RouteSystemEmissionExecution | None
    planned_system_ids: frozenset[RouteSystemId]
    exit_turn_dispositions: tuple[tuple[ExitTurnPlanId, str | None], ...] = ()
    """Every plan's frozen verdict, including systems whose record is restricted.

    Settlement re-routes across moved geometry, so a fresh planning pass can
    reach a different verdict on a plan sitting near a tolerance boundary.
    Replay reads the verdict from here, which is why it is captured before the
    published record is narrowed to planned systems."""


def _allocation_eligible_system_ids(
    preliminary_planned_ids: frozenset[RouteSystemId],
    member_failure_ids: frozenset[RouteSystemId],
) -> frozenset[RouteSystemId]:
    """Remove member-failed systems before shared geometry allocation."""
    return preliminary_planned_ids - member_failure_ids


def _allocated_points_in_source_frame(
    plan: RouteMemberGeometryPlan,
    allocated: RouteMemberGeometryPlan,
    ctx: _RoutingCtx,
) -> tuple[tuple[float, float], ...]:
    """Keep an allocated route's initial run on its frozen fork frame."""
    if plan.edge.source not in ctx.fork_stations or len(allocated.points) < 2:
        return allocated.points
    first, second = allocated.points[:2]
    if abs(first[0] - second[0]) <= COORD_TOLERANCE:
        secondary_axis = 0
    elif abs(first[1] - second[1]) <= COORD_TOLERANCE:
        secondary_axis = 1
    else:
        return allocated.points
    allocated_frame = first[secondary_axis]
    execution_frame = plan.points[0][secondary_axis]
    if abs(allocated_frame - execution_frame) <= COORD_TOLERANCE:
        return allocated.points
    points = list(allocated.points)
    for index, point in enumerate(points):
        if abs(point[secondary_axis] - allocated_frame) > COORD_TOLERANCE:
            break
        shifted = list(point)
        shifted[secondary_axis] = execution_frame
        points[index] = (shifted[0], shifted[1])
    return tuple(points)


def _with_settled_exit_turns(
    execution: MemberGeometryExecution,
    allocation: MemberGeometryExecution,
    ctx: _RoutingCtx,
) -> MemberGeometryExecution:
    """Apply each allocated source axis to the fully normalized member path."""
    from nf_metro.layout.routing.common import Direction, RoutedPath
    from nf_metro.layout.routing.exit_turns import planned_exit_turn_corner_offsets
    from nf_metro.layout.routing.normalize import _reseat_concentric_flanking

    allocated_by_edge = {plan.edge: plan for plan in allocation.plans}
    reconciled_targets = {
        ctx.merge.entry_port_for.get(plan.edge.target, plan.edge.target)
        for plan in execution.plans
        if (
            (plan.edge.source, plan.edge.target, plan.edge.line_id)
            in ctx.settled_exit_turns
            or plan.member_id in execution.reconciled_member_ids
            or plan.member_id in allocation.reconciled_member_ids
        )
    }
    plans = []
    for plan in execution.plans:
        settled = ctx.settled_exit_turns.get(
            (plan.edge.source, plan.edge.target, plan.edge.line_id)
        )
        rank = plan.exit_turn_segment_rank
        allocated = allocated_by_edge.get(plan.edge)
        if (
            settled is not None
            or plan.member_id in execution.reconciled_member_ids
            or plan.member_id in allocation.reconciled_member_ids
            or ctx.merge.entry_port_for.get(plan.edge.target, plan.edge.target)
            in reconciled_targets
        ) and allocated is not None:
            allocated_points = _allocated_points_in_source_frame(plan, allocated, ctx)
            gap_channels = tuple(
                replace(
                    channel,
                    start=allocated_points[channel.segment_rank],
                    end=allocated_points[channel.segment_rank + 1],
                )
                for channel in allocated.gap_channels
            )
            plans.append(
                replace(
                    plan,
                    points=allocated_points,
                    curve_radii=allocated.curve_radii,
                    gap_slots=allocated.gap_slots,
                    trunk_slot=allocated.trunk_slot,
                    gap_channels=gap_channels,
                    concentric_corner_offsets_by_segment=(
                        allocated.concentric_corner_offsets_by_segment
                    ),
                    concentric_corner_bases_by_segment=(
                        allocated.concentric_corner_bases_by_segment
                    ),
                )
            )
            continue
        if settled is None or rank is None:
            plans.append(plan)
            continue
        if plan.curve_radii is None or ctx.exit_turns is None:
            raise RuntimeError(
                f"settled exit turn {plan.id} has no explicit corner geometry"
            )
        membership = ctx.exit_turns.membership_for_edge(plan.edge)
        if membership is None:
            raise RuntimeError(f"settled exit turn {plan.id} lost its plan membership")
        corner_offsets = planned_exit_turn_corner_offsets(membership)
        if corner_offsets is None:
            raise RuntimeError(
                f"settled exit turn {plan.id} has no standard corner offsets"
            )
        curve_radii = list(plan.curve_radii)
        route = RoutedPath(
            ctx.edge_by_key[(plan.edge.source, plan.edge.target, plan.edge.line_id)],
            plan.edge.line_id,
            list(plan.points),
            curve_radii=curve_radii,
            concentric_corner_offsets_by_segment=dict(
                plan.concentric_corner_offsets_by_segment
            ),
            concentric_corner_bases_by_segment=dict(
                plan.concentric_corner_bases_by_segment
            ),
        )
        existing_offsets = route.concentric_corner_offsets_by_segment.get(rank)
        existing_bases = route.concentric_corner_bases_by_segment.get(rank)
        offset_out = (
            existing_offsets[1]
            if existing_offsets is not None and existing_offsets[1] is not None
            else 0.0
        )
        base_radius_out = (
            existing_bases[1]
            if existing_bases is not None and existing_bases[1] is not None
            else (curve_radii[rank] if rank < len(curve_radii) else ctx.curve_radius)
        )
        axis = 0 if settled.run_direction in {Direction.R, Direction.L} else 1
        lead = list(route.points[rank - 1])
        lead[axis] = settled.launch_coordinate
        route.points[rank - 1] = (lead[0], lead[1])
        _reseat_concentric_flanking(
            route,
            rank,
            settled.axis_coordinate,
            axis=axis,
            offset_in=corner_offsets[0],
            offset_out=offset_out,
            base_radius=ctx.curve_radius,
            base_radius_out=base_radius_out,
        )
        points = route.points
        gap_channels = tuple(
            replace(
                channel,
                start=points[channel.segment_rank],
                end=points[channel.segment_rank + 1],
            )
            for channel in plan.gap_channels
        )
        plans.append(
            replace(
                plan,
                points=tuple(points),
                curve_radii=tuple(curve_radii),
                gap_channels=gap_channels,
                concentric_corner_offsets_by_segment=tuple(
                    sorted(route.concentric_corner_offsets_by_segment.items())
                ),
                concentric_corner_bases_by_segment=tuple(
                    sorted(route.concentric_corner_bases_by_segment.items())
                ),
            )
        )
    frozen_plans = tuple(plans)
    return MemberGeometryExecution(
        frozen_plans,
        execution.failure_reasons,
        MappingProxyType({plan.edge: plan for plan in frozen_plans}),
        allocation.settled_exit_turns,
        execution.reconciled_member_ids | allocation.reconciled_member_ids,
    )


def prepare_route_system_planning(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    *,
    include_convergence_resources: bool,
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]] | None = None,
    allow_convergence_clearance_requirements: bool = False,
) -> RoutePlanningExecution:
    """Run the canonical planning phases without emitting production paths.

    Compatibility context is established immediately after preliminary atomic
    disposition.  Only planned systems then contribute convergence claims and
    member geometry to final shared allocation.  Resource publication happens
    after final disposition and follows ``include_convergence_resources``.
    """
    station_offsets = ctx.station_offsets
    initial_station_offsets = dict(station_offsets or {})
    provisional_exit_turns = exit_turn_routing.build_exit_turn_execution(
        graph,
        ctx,
        adopt_prior_dispositions=False,
    )
    scaffold = provisional_exit_turns.scaffold
    if scaffold is None:
        empty_members = empty_member_geometry_execution()
        empty_convergences = empty_convergence_plan_execution()
        ctx.convergences = empty_convergences.query
        ctx.route_systems = None
        return RoutePlanningExecution(
            provisional_exit_turns,
            empty_convergences,
            empty_members,
            None,
            frozenset(),
            tuple(
                (plan.id, plan.legacy_reason) for plan in provisional_exit_turns.plans
            ),
        )

    def prepare_member_geometry(
        exit_turns: ExitTurnExecution,
        pending_plan_ids: frozenset[ExitTurnPlanId],
        settled_plan_ids: frozenset[ExitTurnPlanId] = frozenset(),
    ) -> tuple[
        Mapping[ResolvedEdge, RouteFamilyId],
        ConvergencePlanExecution,
        frozenset[RouteSystemId],
        MemberGeometryExecution,
    ]:
        ctx.exit_turns = exit_turns.query
        family_by_edge = MappingProxyType(
            {
                edge: family
                for edge in scaffold.edge_order
                if (
                    family := classify_inter_section_family(
                        ctx.edge_by_key[(edge.source, edge.target, edge.line_id)],
                        graph.stations[edge.source],
                        graph.stations[edge.target],
                        ctx,
                    )
                )
                is not None
            }
        )
        convergences = build_convergence_plan_execution(
            graph,
            ctx,
            scaffold,
            exit_turn_plans=exit_turns.plans,
            fan_plans=graph.fan_plans,
            member_geometry=empty_member_geometry_execution(),
            include_resources=False,
            allow_clearance_requirements=(allow_convergence_clearance_requirements),
        )
        ctx.convergences = convergences.query
        preliminary = classify_route_system_dispositions(
            scaffold,
            exit_turn_plans=exit_turns.plans,
            fan_plans=graph.fan_plans,
            convergence_plans=convergences.plans,
        )
        complete_path_system_ids = frozenset(
            decision.system_id
            for decision in preliminary
            if decision.geometry_owner is RouteSystemGeometryOwner.MEMBER_GEOMETRY
            and decision.superseded_verdicts
        )
        planned_ids = frozenset(scaffold.ordered_system_ids)
        convergences = settle_preliminary_convergence_execution(
            convergences,
            graph,
            ctx,
            exit_turn_plans=exit_turns.plans,
            planned_system_ids=planned_ids,
        )
        ctx.convergences = convergences.query
        member_geometry = build_member_geometry_execution(
            graph,
            ctx,
            scaffold,
            family_by_edge=family_by_edge,
            convergence_plans=convergences.plans,
            complete_path_system_ids=complete_path_system_ids,
            preliminary_gap_claims=preliminary_member_gap_claims(
                convergences,
                graph,
                planned_ids,
            ),
            reservation_ids_by_member=reservation_ids_by_member,
            pending_exit_turn_plan_ids=pending_plan_ids,
            settled_exit_turn_plan_ids=settled_plan_ids,
        )
        return family_by_edge, convergences, planned_ids, member_geometry

    allocation_exit_turns, pending_plan_ids = (
        exit_turn_routing.promote_pending_gap_allocation(provisional_exit_turns)
    )
    if pending_plan_ids:
        _, _, _, allocation_geometry = prepare_member_geometry(
            allocation_exit_turns, pending_plan_ids
        )
        ctx.settled_exit_turns = allocation_geometry.settled_exit_turns
        if station_offsets is not None:
            station_offsets.clear()
            station_offsets.update(initial_station_offsets)
        ctx.station_offsets = station_offsets
        exit_turns = exit_turn_routing.build_exit_turn_execution(
            graph,
            ctx,
        )
        unresolved = tuple(
            plan
            for plan in exit_turns.plans
            if plan.legacy_reason == exit_turn_routing.GAP_ALLOCATION_PENDING
        )
        if unresolved:
            exit_turns = exit_turn_routing.decline_unsettled_gap_allocation(exit_turns)
        family_by_edge, convergences, preliminary_planned_ids, member_geometry = (
            prepare_member_geometry(
                exit_turns,
                frozenset(),
                pending_plan_ids,
            )
        )
        member_geometry = _with_settled_exit_turns(
            member_geometry,
            allocation_geometry,
            ctx,
        )
    elif ctx.prior_exit_turn_dispositions is not None:
        exit_turns = exit_turn_routing.build_exit_turn_execution(graph, ctx)
        family_by_edge, convergences, preliminary_planned_ids, member_geometry = (
            prepare_member_geometry(exit_turns, pending_plan_ids)
        )
    else:
        exit_turns = provisional_exit_turns
        family_by_edge, convergences, preliminary_planned_ids, member_geometry = (
            prepare_member_geometry(exit_turns, pending_plan_ids)
        )
    allocation_planned_ids = _allocation_eligible_system_ids(
        preliminary_planned_ids,
        frozenset(member_geometry.failure_reasons),
    )
    convergences = settle_global_convergence_execution(
        convergences,
        graph,
        ctx,
        exit_turn_plans=exit_turns.plans,
        member_geometry=member_geometry,
        planned_system_ids=allocation_planned_ids,
        include_resources=False,
        allow_clearance_requirements=allow_convergence_clearance_requirements,
    )
    ctx.convergences = convergences.query
    route_systems = build_route_system_emission_execution(
        scaffold,
        exit_turn_plans=exit_turns.plans,
        fan_plans=graph.fan_plans,
        convergence_plans=convergences.plans,
        reservation_ids_by_member=reservation_ids_by_member,
        family_by_edge=family_by_edge,
        member_geometry_plans=member_geometry.plans,
        member_geometry_failures=member_geometry.failure_reasons,
        require_member_geometry=True,
    )
    planned_system_ids = frozenset(
        system.system_id
        for system in route_systems.systems
        if system.disposition is RouteSystemDisposition.PLANNED
    )
    ctx.route_systems = route_systems
    convergences = restrict_convergence_execution(
        convergences,
        graph,
        planned_system_ids=planned_system_ids,
        include_resources=include_convergence_resources,
    )
    exit_turn_dispositions = tuple(
        (plan.id, plan.legacy_reason) for plan in exit_turns.plans
    )
    # Member templates consume exit-turn axes even when another planner owns the
    # complete system. Publishing only system-owned records would hide those
    # coordinates from the templates that must freeze them.
    emission_exit_turns = exit_turns.query
    exit_turns = exit_turns.restrict_to_systems(planned_system_ids)
    ctx.exit_turns = emission_exit_turns
    ctx.convergences = convergences.query.restrict_to_systems(planned_system_ids)
    return RoutePlanningExecution(
        exit_turns,
        convergences,
        member_geometry,
        route_systems,
        planned_system_ids,
        exit_turn_dispositions,
    )
