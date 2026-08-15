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
    HandlerGapChannel,
    build_convergence_plan_execution,
    empty_convergence_plan_execution,
    handler_gap_channels,
    overlaying_lane_obstacles,
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
    settle_shared_opening_trunk_conflicts,
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


def _publish_member_exit_axes(
    execution: ExitTurnExecution,
    member_geometry: MemberGeometryExecution,
) -> ExitTurnExecution:
    """Publish the exit axes carried by the final immutable member paths."""
    by_member = {plan.member_id: plan for plan in member_geometry.plans}
    replacements = {}
    for exit_plan in execution.plans:
        axes = []
        changed = False
        for axis in exit_plan.axes:
            coordinates = tuple(
                member.points[member.exit_turn_segment_rank][axis.axis.point_index]
                for member_id in axis.claimant_member_ids
                if (member := by_member.get(member_id)) is not None
                and member.exit_turn_axis_id == axis.id
                and member.exit_turn_segment_rank is not None
            )
            if not coordinates or any(
                abs(coordinate - coordinates[0]) > COORD_TOLERANCE
                for coordinate in coordinates[1:]
            ):
                axes.append(axis)
                continue
            published = replace(axis, coordinate=coordinates[0])
            axes.append(published)
            changed |= published != axis
        if changed:
            replacements[exit_plan.id] = replace(exit_plan, axes=tuple(axes))
    if not replacements:
        return execution
    query = execution.query.replacing_plans(replacements)
    return replace(execution, plans=query.plans, query=query)


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


def _reconciled_corner_inputs(
    plan: RouteMemberGeometryPlan,
    allocated: RouteMemberGeometryPlan,
    allocated_points: tuple[tuple[float, float], ...],
    allocation_changed_path: bool,
) -> tuple[
    tuple[float, ...] | None,
    tuple[tuple[int, tuple[float | None, float | None]], ...],
    tuple[tuple[int, tuple[float | None, float | None]], ...],
]:
    """Reconcile a member's settled and allocated corner inputs, per corner.

    An allocation that moved the path owns every corner outright.  On an
    unchanged path the settled record wins corner by corner, with one
    exception: a settled corner recording a family reference wider than the
    standard radius at zero displacement is a provisional handler radius
    laundered into a base (a real widened family records the displacement it
    nests at, and a coincide-unified corner shares its vertex), so where the
    allocation re-derived that corner on the standard base and its record
    resolves on the allocated geometry, the allocated inputs replace it.
    """
    from nf_metro.layout.constants import CURVE_RADIUS
    from nf_metro.layout.routing.corners import concentric_corner_radius_at

    if allocation_changed_path or plan.curve_radii is None:
        return (
            allocated.curve_radii,
            allocated.concentric_corner_offsets_by_segment,
            allocated.concentric_corner_bases_by_segment,
        )
    if allocated.curve_radii is None or len(allocated.curve_radii) != len(
        plan.curve_radii
    ):
        return (
            plan.curve_radii,
            plan.concentric_corner_offsets_by_segment,
            plan.concentric_corner_bases_by_segment,
        )
    allocated_offsets = dict(allocated.concentric_corner_offsets_by_segment)
    allocated_bases = dict(allocated.concentric_corner_bases_by_segment)
    plan_offsets = dict(plan.concentric_corner_offsets_by_segment)
    plan_bases = dict(plan.concentric_corner_bases_by_segment)
    merged_radii = list(plan.curve_radii)
    merged_offsets = dict(plan.concentric_corner_offsets_by_segment)
    merged_bases = dict(plan.concentric_corner_bases_by_segment)
    for i in range(len(merged_radii)):
        plan_offset_pair = plan_offsets.get(i + 1)
        plan_base_pair = plan_bases.get(i + 1)
        plan_offset = plan_offset_pair[0] if plan_offset_pair is not None else None
        plan_base = plan_base_pair[0] if plan_base_pair is not None else None
        if (
            plan_offset is None
            or plan_base is None
            or abs(plan_offset) > COORD_TOLERANCE
            or plan_base <= CURVE_RADIUS + COORD_TOLERANCE
        ):
            continue
        offset_pair = allocated_offsets.get(i + 1)
        base_pair = allocated_bases.get(i + 1)
        offset = offset_pair[0] if offset_pair is not None else None
        base = base_pair[0] if base_pair is not None else None
        if (
            offset is None
            or base is None
            or base > CURVE_RADIUS + COORD_TOLERANCE
            or i + 2 >= len(allocated_points)
        ):
            continue
        implied = concentric_corner_radius_at(
            allocated_points[i],
            allocated_points[i + 1],
            allocated_points[i + 2],
            offset,
            base,
        )
        if abs(implied - allocated.curve_radii[i]) > COORD_TOLERANCE:
            continue
        merged_radii[i] = allocated.curve_radii[i]
        for segment_rank, tuple_index in ((i, 1), (i + 1, 0)):
            source_pair = allocated_offsets.get(segment_rank)
            source_base = allocated_bases.get(segment_rank)
            if source_pair is None or source_base is None:
                continue
            offsets_pair = list(merged_offsets.get(segment_rank, (None, None)))
            bases_pair = list(merged_bases.get(segment_rank, (None, None)))
            offsets_pair[tuple_index] = source_pair[tuple_index]
            bases_pair[tuple_index] = source_base[tuple_index]
            merged_offsets[segment_rank] = (offsets_pair[0], offsets_pair[1])
            merged_bases[segment_rank] = (bases_pair[0], bases_pair[1])
    return (
        tuple(merged_radii),
        tuple(sorted(merged_offsets.items())),
        tuple(sorted(merged_bases.items())),
    )


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
        allocation_changed_path = (
            allocated is not None and allocated.points != plan.points
        )
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
            radii, offsets, bases = _reconciled_corner_inputs(
                plan, allocated, allocated_points, allocation_changed_path
            )
            plans.append(
                replace(
                    plan,
                    points=allocated_points,
                    curve_radii=radii,
                    gap_slots=allocated.gap_slots,
                    trunk_slot=allocated.trunk_slot,
                    gap_channels=gap_channels,
                    concentric_corner_offsets_by_segment=offsets,
                    concentric_corner_bases_by_segment=bases,
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


def _restore_initial_station_offsets(
    ctx: _RoutingCtx,
    station_offsets: dict[tuple[str, str], float] | None,
    initial: dict[tuple[str, str], float],
) -> None:
    """Return the shared offset map to the state a planning pass starts from."""
    if station_offsets is not None:
        station_offsets.clear()
        station_offsets.update(initial)
    ctx.station_offsets = station_offsets


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
        lane_obstacles: tuple[HandlerGapChannel, ...] = (),
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
            lane_obstacles=lane_obstacles,
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

    def seat_members(
        lane_obstacles: tuple[HandlerGapChannel, ...] = (),
    ) -> tuple[
        ExitTurnExecution,
        Mapping[ResolvedEdge, RouteFamilyId],
        ConvergencePlanExecution,
        frozenset[RouteSystemId],
        MemberGeometryExecution,
    ]:
        if pending_plan_ids:
            _, _, _, allocation_geometry = prepare_member_geometry(
                allocation_exit_turns, pending_plan_ids, frozenset(), lane_obstacles
            )
            ctx.settled_exit_turns = allocation_geometry.settled_exit_turns
            _restore_initial_station_offsets(
                ctx, station_offsets, initial_station_offsets
            )
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
                exit_turns = exit_turn_routing.decline_unsettled_gap_allocation(
                    exit_turns
                )
            family_by_edge, convergences, planned_ids, members = (
                prepare_member_geometry(
                    exit_turns,
                    frozenset(),
                    pending_plan_ids,
                    lane_obstacles,
                )
            )
            members = _with_settled_exit_turns(members, allocation_geometry, ctx)
            return exit_turns, family_by_edge, convergences, planned_ids, members
        if ctx.prior_exit_turn_dispositions is not None:
            exit_turns = exit_turn_routing.build_exit_turn_execution(graph, ctx)
        else:
            exit_turns = provisional_exit_turns
        family_by_edge, convergences, planned_ids, members = prepare_member_geometry(
            exit_turns, pending_plan_ids, frozenset(), lane_obstacles
        )
        return exit_turns, family_by_edge, convergences, planned_ids, members

    (
        exit_turns,
        family_by_edge,
        convergences,
        preliminary_planned_ids,
        member_geometry,
    ) = seat_members()
    # A landing column is spent before any handler member is frozen, so the
    # lanes handler-owned members hold are readable only from a seating that
    # has already produced them.
    lane_obstacles = overlaying_lane_obstacles(
        convergences.plans, graph, handler_gap_channels(member_geometry, graph)
    )
    if lane_obstacles:
        _restore_initial_station_offsets(ctx, station_offsets, initial_station_offsets)
        (
            exit_turns,
            family_by_edge,
            convergences,
            preliminary_planned_ids,
            member_geometry,
        ) = seat_members(lane_obstacles)
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
    member_geometry = settle_shared_opening_trunk_conflicts(
        member_geometry,
        convergences.plans,
        graph,
        curve_radius=ctx.curve_radius,
    )
    exit_turns = _publish_member_exit_axes(exit_turns, member_geometry)
    ctx.exit_turns = exit_turns.query
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
