"""Immutable non-convergence member templates shared by planning and emission."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from nf_metro.layout.constants import (
    BUNDLE_TO_BUNDLE_CLEARANCE,
    COORD_TOLERANCE,
    EDGE_TO_BUNDLE_CLEARANCE,
    MIN_CORRIDOR_Y_OVERLAP,
    OFFSET_STEP,
)
from nf_metro.layout.geometry import spans_share_corridor
from nf_metro.layout.route_plan import (
    EmissionMemberId,
    FanPlan,
    RouteMemberGapChannel,
    RouteMemberGeometryPlan,
    RouteMemberGeometryPlanId,
    RouteSemanticScaffold,
    RouteSystemDisposition,
    RouteSystemId,
    _ordered_unique,
)
from nf_metro.layout.routing.common import Direction, RoutedPath, column_gap_edges
from nf_metro.layout.routing.context import _RoutingCtx
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.inter_section_handlers import (
    _route_inter_section,
)
from nf_metro.layout.routing.normalize import (
    _locate_slot_channel,
    _materialize_gap_slots,
    _materialize_trunk_slots,
)
from nf_metro.parser.model import Edge, MetroGraph
from nf_metro.parser.route_topology import ResolvedEdge, semantic_route_id


class MemberGeometryDeclinedError(RuntimeError):
    """A classified non-convergence member produced no canonical template."""


@dataclass(frozen=True, slots=True)
class PreliminaryGapChannelClaim:
    """A convergence-owned leg that member allocation must account for."""

    system_id: RouteSystemId
    coordinate: float
    y_lo: float
    y_hi: float
    down: bool
    gap: tuple[int, int | None]
    line_ids: frozenset[str]


def _eligible_preliminary_gap_claims(
    claims: tuple[PreliminaryGapChannelClaim, ...],
    failure_reasons: Mapping[RouteSystemId, str],
) -> tuple[PreliminaryGapChannelClaim, ...]:
    """Drop convergence claims whose member system cannot own geometry."""
    return tuple(claim for claim in claims if claim.system_id not in failure_reasons)


def _claims_visible_to_system(
    claims: tuple[PreliminaryGapChannelClaim, ...],
    system_id: RouteSystemId,
    system_rank: Mapping[RouteSystemId, int],
) -> tuple[PreliminaryGapChannelClaim, ...]:
    """Expose own and prior claims in canonical system order."""
    return tuple(
        claim
        for claim in claims
        if system_rank[claim.system_id] <= system_rank[system_id]
    )


@dataclass(frozen=True, slots=True)
class _CompatibilitySystem:
    disposition: RouteSystemDisposition = RouteSystemDisposition.COMPATIBILITY


@dataclass(frozen=True, slots=True)
class _CompatibilitySystemLookup:
    edges: frozenset[ResolvedEdge]

    def system_for_edge(self, edge: Edge | ResolvedEdge) -> _CompatibilitySystem | None:
        resolved = (
            edge
            if isinstance(edge, ResolvedEdge)
            else ResolvedEdge(edge.source, edge.target, edge.line_id)
        )
        return _CompatibilitySystem() if resolved in self.edges else None


@dataclass(frozen=True, slots=True)
class MemberGeometryExecution:
    plans: tuple[RouteMemberGeometryPlan, ...]
    failure_reasons: Mapping[RouteSystemId, str]
    _by_edge: Mapping[ResolvedEdge, RouteMemberGeometryPlan]
    _by_system: Mapping[RouteSystemId, tuple[RouteMemberGeometryPlan, ...]]

    def plan_for_edge(
        self, edge: Edge | ResolvedEdge
    ) -> RouteMemberGeometryPlan | None:
        resolved = (
            edge
            if isinstance(edge, ResolvedEdge)
            else ResolvedEdge(edge.source, edge.target, edge.line_id)
        )
        return self._by_edge.get(resolved)

    def plans_for_system(
        self, system_id: RouteSystemId
    ) -> tuple[RouteMemberGeometryPlan, ...]:
        return self._by_system.get(system_id, ())

    def gap_channels_for_system(
        self,
        system_id: RouteSystemId,
        *,
        excluding: frozenset[ResolvedEdge] = frozenset(),
    ) -> tuple[tuple[frozenset[str], RouteMemberGapChannel], ...]:
        return tuple(
            (frozenset((plan.edge.line_id,)), channel)
            for plan in self.plans_for_system(system_id)
            if plan.edge not in excluding
            for channel in plan.gap_channels
        )

    def gap_channels(
        self, *, excluding: frozenset[ResolvedEdge] = frozenset()
    ) -> tuple[tuple[frozenset[str], RouteMemberGapChannel], ...]:
        """Every provisional member channel outside *excluding*."""
        return tuple(
            (frozenset((plan.edge.line_id,)), channel)
            for plan in self.plans
            if plan.edge not in excluding
            for channel in plan.gap_channels
        )


def empty_member_geometry_execution() -> MemberGeometryExecution:
    return MemberGeometryExecution(
        (), MappingProxyType({}), MappingProxyType({}), MappingProxyType({})
    )


def _convergence_member_edges(
    scaffold: RouteSemanticScaffold,
) -> frozenset[ResolvedEdge]:
    edges = {
        edge
        for view in scaffold.query.convergences
        for connector_id in view.group.connector_ids
        for path in scaffold.query.resolved_paths(connector_id)
        for edge in path
        if edge.target == view.junction_id or edge.source == view.junction_id
    }
    return frozenset(edges)


def _route_template(
    edge: Edge,
    family_id: RouteFamilyId,
    ctx: _RoutingCtx,
) -> RoutedPath:
    source, target = ctx.graph.edge_endpoints(edge)
    route = _route_inter_section(
        edge,
        source,
        target,
        ctx,
        planned_family_id=family_id,
    )
    if route is None:
        raise MemberGeometryDeclinedError("canonical family declined its member")
    return route


def _route_compatibility_template(edge: Edge, ctx: _RoutingCtx) -> RoutedPath:
    source, target = ctx.graph.edge_endpoints(edge)
    route = _route_inter_section(edge, source, target, ctx)
    if route is None:
        raise RuntimeError("compatibility inter-section member emitted no route")
    return route


def _append_compatibility_context(
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    system_id: RouteSystemId,
    system_edges: tuple[ResolvedEdge, ...],
) -> None:
    """Emit one compatibility system solely as ordered planning context."""
    prior_systems = ctx.route_systems
    prior_exit_turns = ctx.exit_turns
    prior_convergences = ctx.convergences
    other_systems = frozenset(scaffold.ordered_system_ids) - {system_id}
    ctx.route_systems = _CompatibilitySystemLookup(frozenset(system_edges))
    if ctx.exit_turns is not None:
        ctx.exit_turns = ctx.exit_turns.restrict_to_systems(other_systems)
    if ctx.convergences is not None:
        ctx.convergences = ctx.convergences.restrict_to_systems(other_systems)
    try:
        for resolved in system_edges:
            key = (resolved.source, resolved.target, resolved.line_id)
            if key in ctx.skip_edges:
                continue
            edge = ctx.edge_by_key[key]
            ctx.built_routes.append(_route_compatibility_template(edge, ctx))
    finally:
        ctx.route_systems = prior_systems
        ctx.exit_turns = prior_exit_turns
        ctx.convergences = prior_convergences


def _freeze_plan(
    scaffold: RouteSemanticScaffold,
    route: RoutedPath,
    family_id: RouteFamilyId,
    ctx: _RoutingCtx,
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]],
) -> RouteMemberGeometryPlan:
    resolved = ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
    member_id = scaffold.member_id_by_edge[resolved]
    connector_ids = _ordered_unique(
        item.connector_id for item in scaffold.refs_by_edge[resolved]
    )
    system_id = scaffold.system_for(connector_ids)
    channels: list[RouteMemberGapChannel] = []
    channel_claims: set[tuple[int, int, int | None, Direction]] = set()
    for slot in route.gap_slots:
        channel = _locate_slot_channel(route, slot, ctx.graph)
        if channel is None:
            continue
        claim = (channel.idx, slot.gap_lo_col, slot.row, slot.direction)
        if claim in channel_claims:
            continue
        channel_claims.add(claim)
        channels.append(
            RouteMemberGapChannel(
                channel.idx,
                route.points[channel.idx],
                route.points[channel.idx + 1],
                slot.gap_lo_col,
                slot.row,
                slot.direction,
            )
        )
    plan_id = RouteMemberGeometryPlanId(
        semantic_route_id(
            "route-member-geometry", system_id, member_id, family_id.value
        )
    )
    return RouteMemberGeometryPlan(
        plan_id,
        system_id,
        member_id,
        resolved,
        family_id,
        tuple(route.points),
        None if route.curve_radii is None else tuple(route.curve_radii),
        route.offset_regime,
        route.normalize_exempt,
        tuple(route.gap_slots),
        route.trunk_slot,
        tuple(channels),
        exit_turn_plan_id=route.exit_turn_plan_id,
        exit_turn_member_id=route.exit_turn_member_id,
        exit_turn_family_id=route.exit_turn_family_id,
        exit_turn_axis_id=route.exit_turn_axis_id,
        exit_turn_segment_rank=route.exit_turn_segment_rank,
        exit_lane_transition_plan_id=route.exit_lane_transition_plan_id,
        fan_plan_id=route.fan_plan_id,
        fan_route_emitter=route.fan_route_emitter,
        consumed_reservation_ids=reservation_ids_by_member.get(member_id, ()),
    )


def _allocate_preliminary_gap_claims(
    routes: list[RoutedPath],
    claims: tuple[PreliminaryGapChannelClaim, ...],
    ctx: _RoutingCtx,
    system_id_by_route: Mapping[int, RouteSystemId],
    system_rank: Mapping[RouteSystemId, int],
) -> None:
    """Seat mutable member channels around preliminary convergence claims."""
    from nf_metro.layout.routing.normalize import _set_vchannel_x

    def planner_owns_channel(route: RoutedPath, segment_rank: int) -> bool:
        if route.exit_lane_transition_plan_id is not None:
            return True
        if route.fan_plan_id is not None or route.fan_route_emitter is not None:
            return True
        owned_ranks = (
            *route.convergence_owned_segment_ranks,
            *(
                ()
                if route.exit_turn_segment_rank is None
                else (route.exit_turn_segment_rank,)
            ),
        )
        return any(abs(owned_rank - segment_rank) <= 1 for owned_rank in owned_ranks)

    if not claims:
        return
    materialized = tuple(
        (route, channel, slot, (slot.gap_lo_col, slot.row))
        for route in routes
        for slot in route.gap_slots
        if (channel := _locate_slot_channel(route, slot, ctx.graph)) is not None
    )
    seated: set[tuple[int, int]] = set()
    for route, channel, slot, gap in materialized:
        if (id(route), channel.idx) in seated:
            continue
        if planner_owns_channel(route, channel.idx):
            continue
        route_system_id = system_id_by_route[id(route)]
        matching = tuple(
            claim
            for claim in claims
            if claim.system_id == route_system_id
            and claim.gap == gap
            and route.line_id in claim.line_ids
            and spans_share_corridor(channel.y_lo, channel.y_hi, claim.y_lo, claim.y_hi)
        )
        coordinates = {claim.coordinate for claim in matching}
        if len(coordinates) != 1:
            continue
        target = next(iter(coordinates))
        if abs(channel.x - target) <= COORD_TOLERANCE:
            continue
        left, right = column_gap_edges(
            ctx.graph, slot.gap_lo_col, slot.gap_hi_col, row=slot.row
        )
        lo = left + EDGE_TO_BUNDLE_CLEARANCE
        hi = right - EDGE_TO_BUNDLE_CLEARANCE
        band = ctx.reserved_bands.for_segment(
            route.edge.source, route.edge.target, route.line_id, channel.idx
        )
        if band is not None:
            lo = max(lo, band.lo)
            hi = min(hi, band.hi)
        candidates = {target, lo, hi}
        candidates.update(
            route.points[rank][0] + sign * ctx.curve_radius
            for rank in (channel.idx - 1, channel.idx + 2)
            if 0 <= rank < len(route.points)
            for sign in (-1.0, 1.0)
        )

        def feasible_same_line(candidate: float) -> bool:
            return (
                (band is None or band.lo <= candidate <= band.hi)
                and lo - COORD_TOLERANCE <= candidate <= hi + COORD_TOLERANCE
                and all(
                    not (
                        0 <= rank < len(route.points)
                        and abs(route.points[rank][0] - candidate)
                        < ctx.curve_radius - COORD_TOLERANCE
                    )
                    for rank in (channel.idx - 1, channel.idx + 2)
                )
            )

        coordinate = next(
            (
                candidate
                for candidate in sorted(
                    candidates,
                    key=lambda item: (abs(item - target), abs(item - channel.x), item),
                )
                if feasible_same_line(candidate)
            ),
            None,
        )
        if coordinate is not None:
            _set_vchannel_x(channel, coordinate)
            seated.add((id(route), channel.idx))
    materialized = tuple(
        (route, channel, (slot.gap_lo_col, slot.row))
        for route in routes
        for slot in route.gap_slots
        if (channel := _locate_slot_channel(route, slot, ctx.graph)) is not None
    )
    effective_claims: list[PreliminaryGapChannelClaim] = []
    for claim in claims:
        shared_coordinates = {
            channel.x
            for route, channel, gap in materialized
            if system_id_by_route[id(route)] == claim.system_id
            and gap == claim.gap
            and route.line_id in claim.line_ids
            and spans_share_corridor(channel.y_lo, channel.y_hi, claim.y_lo, claim.y_hi)
        }
        effective_claims.append(
            replace(claim, coordinate=next(iter(shared_coordinates)))
            if len(shared_coordinates) == 1
            else claim
        )
    seen: set[tuple[int, int]] = set()
    for route in routes:
        route_system_id = system_id_by_route[id(route)]
        visible_claims = _claims_visible_to_system(
            tuple(effective_claims), route_system_id, system_rank
        )
        for slot in route.gap_slots:
            channel = _locate_slot_channel(route, slot, ctx.graph)
            if channel is None or (id(route), channel.idx) in seen:
                continue
            seen.add((id(route), channel.idx))
            if planner_owns_channel(route, channel.idx):
                continue
            gap = (slot.gap_lo_col, slot.row)
            obstacles = tuple(
                claim
                for claim in visible_claims
                if claim.gap == gap
                and spans_share_corridor(
                    channel.y_lo, channel.y_hi, claim.y_lo, claim.y_hi
                )
            )
            if not obstacles:
                continue

            def clearance(claim: PreliminaryGapChannelClaim) -> float:
                if route.line_id in claim.line_ids:
                    return 0.0
                overlap = min(channel.y_hi, claim.y_hi) - max(channel.y_lo, claim.y_lo)
                if overlap <= MIN_CORRIDOR_Y_OVERLAP:
                    return 0.0
                return (
                    BUNDLE_TO_BUNDLE_CLEARANCE
                    if channel.down is not claim.down
                    else OFFSET_STEP
                )

            ambiguous = tuple(
                claim
                for claim in obstacles
                if route.line_id in claim.line_ids
                and abs(channel.x - claim.coordinate) > COORD_TOLERANCE
            )
            crowded = tuple(
                claim
                for claim in obstacles
                if (required := clearance(claim)) > 0.0
                and abs(channel.x - claim.coordinate) < required - COORD_TOLERANCE
            )
            if not ambiguous and not crowded:
                continue
            left, right = column_gap_edges(
                ctx.graph, slot.gap_lo_col, slot.gap_hi_col, row=slot.row
            )
            lo = left + EDGE_TO_BUNDLE_CLEARANCE
            hi = right - EDGE_TO_BUNDLE_CLEARANCE
            band = ctx.reserved_bands.for_segment(
                route.edge.source, route.edge.target, route.line_id, channel.idx
            )
            if band is not None:
                lo = max(lo, band.lo)
                hi = min(hi, band.hi)
            candidates = {claim.coordinate for claim in ambiguous}
            candidates.update(
                claim.coordinate + sign * clearance(claim)
                for claim in crowded
                for sign in (-1.0, 1.0)
            )
            candidates.update({lo, hi})
            candidates.update(
                route.points[rank][0] + sign * ctx.curve_radius
                for rank in (channel.idx - 1, channel.idx + 2)
                if 0 <= rank < len(route.points)
                for sign in (-1.0, 1.0)
            )

            def feasible(candidate: float) -> bool:
                if band is not None and not band.lo <= candidate <= band.hi:
                    return False
                if candidate < lo - COORD_TOLERANCE or candidate > hi + COORD_TOLERANCE:
                    return False
                if any(
                    0 <= rank < len(route.points)
                    and abs(route.points[rank][0] - candidate)
                    < ctx.curve_radius - COORD_TOLERANCE
                    for rank in (channel.idx - 1, channel.idx + 2)
                ):
                    return False
                return all(
                    route.line_id not in claim.line_ids
                    or abs(candidate - claim.coordinate) <= COORD_TOLERANCE
                    for claim in obstacles
                )

            def rank(candidate: float) -> tuple[int, float, float, float]:
                residual = sum(
                    required > 0.0
                    and abs(candidate - claim.coordinate) < required - COORD_TOLERANCE
                    for claim in obstacles
                    for required in (clearance(claim),)
                )
                separation = min(
                    (
                        abs(candidate - claim.coordinate)
                        for claim in obstacles
                        if route.line_id not in claim.line_ids
                    ),
                    default=float("inf"),
                )
                return residual, -separation, abs(candidate - channel.x), candidate

            coordinate = next(
                (
                    candidate
                    for candidate in sorted(candidates, key=rank)
                    if feasible(candidate)
                ),
                None,
            )
            if coordinate is not None:
                _set_vchannel_x(channel, coordinate)


def build_member_geometry_execution(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    *,
    exit_turn_plans: tuple,
    fan_plans: tuple[FanPlan, ...],
    family_by_edge: Mapping[ResolvedEdge, RouteFamilyId],
    compatibility_system_ids: frozenset[RouteSystemId] = frozenset(),
    preliminary_gap_claims: tuple[PreliminaryGapChannelClaim, ...] = (),
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]] | None = None,
) -> MemberGeometryExecution:
    """Freeze each eligible non-convergence member's sole production template."""
    del exit_turn_plans, fan_plans
    convergence_edges = _convergence_member_edges(scaffold)
    reservation_ids = reservation_ids_by_member or {}
    candidates: list[tuple[RoutedPath, RouteFamilyId, RouteSystemId]] = []
    failures: dict[RouteSystemId, str] = {}
    edges_by_system: dict[RouteSystemId, list[ResolvedEdge]] = defaultdict(list)
    for resolved in scaffold.edge_order:
        connector_ids = _ordered_unique(
            item.connector_id for item in scaffold.refs_by_edge[resolved]
        )
        edges_by_system[scaffold.system_for(connector_ids)].append(resolved)
    built_start = len(ctx.built_routes)
    try:
        for system_id in scaffold.ordered_system_ids:
            system_edges = tuple(edges_by_system.get(system_id, ()))
            if system_id in compatibility_system_ids:
                _append_compatibility_context(ctx, scaffold, system_id, system_edges)
                continue
            system_start = len(ctx.built_routes)
            system_candidates: list[
                tuple[RoutedPath, RouteFamilyId, RouteSystemId]
            ] = []
            for resolved in system_edges:
                key = (resolved.source, resolved.target, resolved.line_id)
                if resolved in convergence_edges or key in ctx.skip_edges:
                    continue
                edge = ctx.edge_by_key.get(key)
                if edge is None:
                    failures[system_id] = "missing-emission-edge"
                    break
                family_id = family_by_edge.get(resolved)
                if family_id is None:
                    failures[system_id] = "missing-production-family"
                    break
                try:
                    route = _route_template(edge, family_id, ctx)
                except MemberGeometryDeclinedError:
                    failures[system_id] = "canonical-template-declined-member"
                    break
                system_candidates.append((route, family_id, system_id))
                ctx.built_routes.append(route)
            if system_id not in failures:
                candidates.extend(system_candidates)
                continue

            del ctx.built_routes[system_start:]
            _append_compatibility_context(ctx, scaffold, system_id, system_edges)

        candidate_routes = [route for route, _family, _system in candidates]
        _materialize_gap_slots(candidate_routes, ctx)
        _materialize_trunk_slots(candidate_routes, ctx)
        eligible_claims = _eligible_preliminary_gap_claims(
            preliminary_gap_claims,
            failures,
        )
        _allocate_preliminary_gap_claims(
            candidate_routes,
            eligible_claims,
            ctx,
            {id(route): system_id for route, _family, system_id in candidates},
            {
                system_id: rank
                for rank, system_id in enumerate(scaffold.ordered_system_ids)
            },
        )
        plans = tuple(
            _freeze_plan(scaffold, route, family_id, ctx, reservation_ids)
            for route, family_id, _system_id in candidates
            if scaffold.system_for(
                _ordered_unique(
                    item.connector_id
                    for item in scaffold.refs_by_edge[
                        ResolvedEdge(
                            route.edge.source, route.edge.target, route.line_id
                        )
                    ]
                )
            )
            not in failures
        )
    finally:
        del ctx.built_routes[built_start:]

    by_system: dict[RouteSystemId, list[RouteMemberGeometryPlan]] = defaultdict(list)
    for plan in plans:
        by_system[plan.system_id].append(plan)
    return MemberGeometryExecution(
        plans,
        MappingProxyType(failures),
        MappingProxyType({plan.edge: plan for plan in plans}),
        MappingProxyType({key: tuple(value) for key, value in by_system.items()}),
    )


def fresh_member_route(plan: RouteMemberGeometryPlan, edge: Edge) -> RoutedPath:
    """Materialise a normalizable path carrying its immutable owned segments."""
    route = RoutedPath(
        edge,
        edge.line_id,
        list(plan.points),
        is_inter_section=True,
        curve_radii=None if plan.curve_radii is None else list(plan.curve_radii),
        offset_regime=plan.offset_regime,
        normalize_exempt=plan.normalize_exempt,
        gap_slots=list(plan.gap_slots),
        trunk_slot=plan.trunk_slot,
        exit_turn_plan_id=plan.exit_turn_plan_id,
        exit_turn_member_id=plan.exit_turn_member_id,
        exit_turn_family_id=plan.exit_turn_family_id,
        exit_turn_axis_id=plan.exit_turn_axis_id,
        fan_plan_id=plan.fan_plan_id,
        fan_route_emitter=plan.fan_route_emitter,
        exit_turn_segment_rank=plan.exit_turn_segment_rank,
        exit_lane_transition_plan_id=plan.exit_lane_transition_plan_id,
        route_system_owned_segment_ranks=plan.owned_segment_ranks,
    )
    return route


def validate_member_geometry_emission(
    routes: list[RoutedPath], execution: MemberGeometryExecution
) -> None:
    """Require every emitted plan-owned channel to retain its exact geometry."""
    for route in routes:
        plan = execution.plan_for_edge(route.edge)
        if plan is None or route.route_system_disposition != "planned":
            continue
        if tuple(route.route_system_owned_segment_ranks) != plan.owned_segment_ranks:
            raise RuntimeError(f"member geometry plan {plan.id} lost channel ownership")
        for channel in plan.gap_channels:
            actual = tuple(
                route.points[channel.segment_rank : channel.segment_rank + 2]
            )
            if actual != (channel.start, channel.end):
                raise RuntimeError(
                    f"member geometry plan {plan.id} channel geometry changed"
                )
