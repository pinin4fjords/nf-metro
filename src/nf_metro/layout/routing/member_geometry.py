"""Immutable non-convergence member templates shared by planning and emission."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
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
    RouteMemberGapChannel,
    RouteMemberGeometryPlan,
    RouteMemberGeometryPlanId,
    RouteSemanticScaffold,
    RouteSystemId,
)
from nf_metro.layout.routing.common import (
    Direction,
    GapSlot,
    RoutedPath,
    column_gap_edges,
)
from nf_metro.layout.routing.context import _RoutingCtx
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.inter_section_handlers import (
    _route_inter_section,
)
from nf_metro.layout.routing.intra_handlers import (
    _route_entry_runway,
    _route_intra_section,
)
from nf_metro.layout.routing.normalize import (
    _locate_slot_channel,
    _materialize_gap_slots,
    _materialize_trunk_slots,
    _VChannel,
)
from nf_metro.layout.routing.reserved_bands import ReservedBand
from nf_metro.layout.routing.tb_handlers import _route_tb_section
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


@dataclass(frozen=True, slots=True)
class _MemberCandidate:
    route: RoutedPath
    family_id: RouteFamilyId
    system_id: RouteSystemId


@dataclass(frozen=True, slots=True)
class _MaterializedChannel:
    candidate: _MemberCandidate
    channel: _VChannel
    slot: GapSlot

    @property
    def gap(self) -> tuple[int, int | None]:
        return self.slot.gap_lo_col, self.slot.row

    @property
    def key(self) -> tuple[RouteSystemId, ResolvedEdge, int]:
        route = self.candidate.route
        return (
            self.candidate.system_id,
            ResolvedEdge(route.edge.source, route.edge.target, route.line_id),
            self.channel.idx,
        )


@dataclass(frozen=True, slots=True)
class _ChannelBounds:
    gap_lo: float
    gap_hi: float
    lo: float
    hi: float
    band: ReservedBand | None


def _eligible_preliminary_gap_claims(
    claims: tuple[PreliminaryGapChannelClaim, ...],
    failure_reasons: Mapping[RouteSystemId, str],
) -> tuple[PreliminaryGapChannelClaim, ...]:
    """Drop convergence claims whose member system cannot own geometry."""
    return tuple(claim for claim in claims if claim.system_id not in failure_reasons)


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
        route = _route_tb_section(edge, source, target, ctx)
    if route is None:
        route = _route_entry_runway(edge, source, target, ctx)
    if route is None:
        route = _route_intra_section(edge, source, target, ctx)
    if route is None:
        raise RuntimeError("compatibility member emitted no route")
    return route


def _append_compatibility_context(
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    system_id: RouteSystemId,
    system_edges: tuple[ResolvedEdge, ...],
) -> None:
    """Emit one compatibility system solely as ordered planning context."""
    prior_compatibility_edges = ctx.compatibility_edges
    prior_exit_turns = ctx.exit_turns
    prior_convergences = ctx.convergences
    other_systems = frozenset(scaffold.ordered_system_ids) - {system_id}
    ctx.compatibility_edges = frozenset(
        (edge.source, edge.target, edge.line_id) for edge in system_edges
    )
    if ctx.exit_turns is not None:
        ctx.exit_turns = ctx.exit_turns.restrict_to_systems(other_systems)
    if ctx.convergences is not None:
        ctx.convergences = ctx.convergences.restrict_to_systems(other_systems)
    try:
        for resolved in system_edges:
            key = (resolved.source, resolved.target, resolved.line_id)
            edge = ctx.edge_by_key[key]
            ctx.built_routes.append(_route_compatibility_template(edge, ctx))
    finally:
        ctx.compatibility_edges = prior_compatibility_edges
        ctx.exit_turns = prior_exit_turns
        ctx.convergences = prior_convergences


def _freeze_plan(
    scaffold: RouteSemanticScaffold,
    candidate: _MemberCandidate,
    ctx: _RoutingCtx,
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]],
) -> RouteMemberGeometryPlan:
    route = candidate.route
    family_id = candidate.family_id
    system_id = candidate.system_id
    resolved = ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
    member_id = scaffold.member_id_by_edge[resolved]
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


def _materialized_channels(
    candidates: Iterable[_MemberCandidate], ctx: _RoutingCtx
) -> tuple[_MaterializedChannel, ...]:
    return tuple(
        _MaterializedChannel(candidate, channel, slot)
        for candidate in candidates
        for slot in candidate.route.gap_slots
        if (channel := _locate_slot_channel(candidate.route, slot, ctx.graph))
        is not None
    )


def _index_claims(
    claims: Iterable[PreliminaryGapChannelClaim],
) -> Mapping[
    tuple[RouteSystemId, tuple[int, int | None]],
    tuple[PreliminaryGapChannelClaim, ...],
]:
    indexed: defaultdict[
        tuple[RouteSystemId, tuple[int, int | None]],
        list[PreliminaryGapChannelClaim],
    ] = defaultdict(list)
    for claim in claims:
        indexed[(claim.system_id, claim.gap)].append(claim)
    return MappingProxyType({key: tuple(value) for key, value in indexed.items()})


def _index_materialized_channels(
    channels: Iterable[_MaterializedChannel],
) -> Mapping[
    tuple[RouteSystemId, tuple[int, int | None]],
    tuple[_MaterializedChannel, ...],
]:
    indexed: defaultdict[
        tuple[RouteSystemId, tuple[int, int | None]], list[_MaterializedChannel]
    ] = defaultdict(list)
    for item in channels:
        indexed[(item.candidate.system_id, item.gap)].append(item)
    return MappingProxyType({key: tuple(value) for key, value in indexed.items()})


def _planner_owns_channel(route: RoutedPath, segment_rank: int) -> bool:
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


def _channel_bounds(item: _MaterializedChannel, ctx: _RoutingCtx) -> _ChannelBounds:
    route = item.candidate.route
    left, right = column_gap_edges(
        ctx.graph,
        item.slot.gap_lo_col,
        item.slot.gap_hi_col,
        row=item.slot.row,
    )
    lo = left + EDGE_TO_BUNDLE_CLEARANCE
    hi = right - EDGE_TO_BUNDLE_CLEARANCE
    band = ctx.reserved_bands.for_segment(
        route.edge.source, route.edge.target, route.line_id, item.channel.idx
    )
    if band is not None:
        lo = max(lo, band.lo)
        hi = min(hi, band.hi)
    return _ChannelBounds(left, right, lo, hi, band)


def _runway_candidates(
    item: _MaterializedChannel,
    bounds: _ChannelBounds,
    seeds: Iterable[float],
    ctx: _RoutingCtx,
) -> set[float]:
    route = item.candidate.route
    candidates = {*seeds, bounds.lo, bounds.hi}
    candidates.update(
        route.points[rank][0] + sign * ctx.curve_radius
        for rank in (item.channel.idx - 1, item.channel.idx + 2)
        if 0 <= rank < len(route.points)
        for sign in (-1.0, 1.0)
    )
    return candidates


def _candidate_clears_runway(
    item: _MaterializedChannel,
    bounds: _ChannelBounds,
    candidate: float,
    ctx: _RoutingCtx,
    *,
    shared_carrier: bool = False,
) -> bool:
    """Whether a candidate preserves its corridor and both corner runways.

    An exact same-line convergence claim already owns a physical carrier, so
    extending that stroke may use the whole gap.  It still cannot leave the
    gap, escape a reservation band, or starve either corner.
    """
    route = item.candidate.route
    if bounds.band is not None and not bounds.band.lo <= candidate <= bounds.band.hi:
        return False
    corridor_lo = bounds.gap_lo if shared_carrier else bounds.lo
    corridor_hi = bounds.gap_hi if shared_carrier else bounds.hi
    if (
        candidate < corridor_lo - COORD_TOLERANCE
        or candidate > corridor_hi + COORD_TOLERANCE
    ):
        return False
    return all(
        not (
            0 <= rank < len(route.points)
            and abs(route.points[rank][0] - candidate)
            < ctx.curve_radius - COORD_TOLERANCE
        )
        for rank in (item.channel.idx - 1, item.channel.idx + 2)
    )


def _align_same_line_channels(
    materialized: tuple[_MaterializedChannel, ...],
    claims_by_system_gap: Mapping[
        tuple[RouteSystemId, tuple[int, int | None]],
        tuple[PreliminaryGapChannelClaim, ...],
    ],
    ctx: _RoutingCtx,
) -> None:
    from nf_metro.layout.routing.normalize import _set_vchannel_x

    seated: set[tuple[RouteSystemId, ResolvedEdge, int]] = set()
    for item in materialized:
        route = item.candidate.route
        channel = item.channel
        if item.key in seated or _planner_owns_channel(route, channel.idx):
            continue
        matching = tuple(
            claim
            for claim in claims_by_system_gap.get(
                (item.candidate.system_id, item.gap), ()
            )
            if route.line_id in claim.line_ids
            and spans_share_corridor(channel.y_lo, channel.y_hi, claim.y_lo, claim.y_hi)
        )
        coordinates = {claim.coordinate for claim in matching}
        if len(coordinates) != 1:
            continue
        target = next(iter(coordinates))
        if abs(channel.x - target) <= COORD_TOLERANCE:
            continue
        bounds = _channel_bounds(item, ctx)
        candidates = _runway_candidates(item, bounds, (target,), ctx)
        coordinate = next(
            (
                candidate
                for candidate in sorted(
                    candidates,
                    key=lambda value: (
                        abs(value - target),
                        abs(value - channel.x),
                        value,
                    ),
                )
                if _candidate_clears_runway(
                    item,
                    bounds,
                    candidate,
                    ctx,
                    shared_carrier=abs(candidate - target) <= COORD_TOLERANCE,
                )
            ),
            None,
        )
        if coordinate is not None:
            _set_vchannel_x(channel, coordinate)
            seated.add(item.key)


def _effective_claims(
    claims: tuple[PreliminaryGapChannelClaim, ...],
    channels_by_system_gap: Mapping[
        tuple[RouteSystemId, tuple[int, int | None]],
        tuple[_MaterializedChannel, ...],
    ],
) -> tuple[PreliminaryGapChannelClaim, ...]:
    effective: list[PreliminaryGapChannelClaim] = []
    for claim in claims:
        shared_coordinates = {
            item.channel.x
            for item in channels_by_system_gap.get((claim.system_id, claim.gap), ())
            if item.candidate.route.line_id in claim.line_ids
            and spans_share_corridor(
                item.channel.y_lo,
                item.channel.y_hi,
                claim.y_lo,
                claim.y_hi,
            )
        }
        effective.append(
            replace(claim, coordinate=next(iter(shared_coordinates)))
            if len(shared_coordinates) == 1
            else claim
        )
    return tuple(effective)


def _visible_claims_by_system_gap(
    claims: tuple[PreliminaryGapChannelClaim, ...],
    system_rank: Mapping[RouteSystemId, int],
) -> Mapping[
    tuple[RouteSystemId, tuple[int, int | None]],
    tuple[PreliminaryGapChannelClaim, ...],
]:
    gaps = tuple(dict.fromkeys(claim.gap for claim in claims))
    return MappingProxyType(
        {
            (system_id, gap): tuple(
                claim
                for claim in claims
                if claim.gap == gap and system_rank[claim.system_id] <= rank
            )
            for system_id, rank in system_rank.items()
            for gap in gaps
        }
    )


def _claim_clearance(
    item: _MaterializedChannel, claim: PreliminaryGapChannelClaim
) -> float:
    route = item.candidate.route
    channel = item.channel
    if route.line_id in claim.line_ids:
        return 0.0
    overlap = min(channel.y_hi, claim.y_hi) - max(channel.y_lo, claim.y_lo)
    if overlap <= MIN_CORRIDOR_Y_OVERLAP:
        return 0.0
    return BUNDLE_TO_BUNDLE_CLEARANCE if channel.down is not claim.down else OFFSET_STEP


def _allocate_channel_around_claims(
    item: _MaterializedChannel,
    obstacles: tuple[PreliminaryGapChannelClaim, ...],
    ctx: _RoutingCtx,
) -> None:
    from nf_metro.layout.routing.normalize import _set_vchannel_x

    route = item.candidate.route
    channel = item.channel
    ambiguous = tuple(
        claim
        for claim in obstacles
        if route.line_id in claim.line_ids
        and abs(channel.x - claim.coordinate) > COORD_TOLERANCE
    )
    crowded = tuple(
        claim
        for claim in obstacles
        if (required := _claim_clearance(item, claim)) > 0.0
        and abs(channel.x - claim.coordinate) < required - COORD_TOLERANCE
    )
    if not ambiguous and not crowded:
        return
    bounds = _channel_bounds(item, ctx)
    seeds = {claim.coordinate for claim in ambiguous}
    seeds.update(
        claim.coordinate + sign * _claim_clearance(item, claim)
        for claim in crowded
        for sign in (-1.0, 1.0)
    )
    candidates = _runway_candidates(item, bounds, seeds, ctx)

    def feasible(candidate: float) -> bool:
        return _candidate_clears_runway(item, bounds, candidate, ctx) and all(
            route.line_id not in claim.line_ids
            or abs(candidate - claim.coordinate) <= COORD_TOLERANCE
            for claim in obstacles
        )

    def rank(candidate: float) -> tuple[int, float, float, float]:
        residual = sum(
            required > 0.0
            and abs(candidate - claim.coordinate) < required - COORD_TOLERANCE
            for claim in obstacles
            for required in (_claim_clearance(item, claim),)
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


def _allocate_preliminary_gap_claims(
    candidates: tuple[_MemberCandidate, ...],
    claims: tuple[PreliminaryGapChannelClaim, ...],
    ctx: _RoutingCtx,
    system_rank: Mapping[RouteSystemId, int],
) -> None:
    """Seat mutable member channels around preliminary convergence claims."""
    if not claims:
        return
    materialized = _materialized_channels(candidates, ctx)
    _align_same_line_channels(materialized, _index_claims(claims), ctx)

    materialized = _materialized_channels(candidates, ctx)
    effective_claims = _effective_claims(
        claims, _index_materialized_channels(materialized)
    )
    visible = _visible_claims_by_system_gap(effective_claims, system_rank)
    seen: set[tuple[RouteSystemId, ResolvedEdge, int]] = set()
    for item in materialized:
        if item.key in seen:
            continue
        seen.add(item.key)
        if _planner_owns_channel(item.candidate.route, item.channel.idx):
            continue
        obstacles = tuple(
            claim
            for claim in visible.get((item.candidate.system_id, item.gap), ())
            if spans_share_corridor(
                item.channel.y_lo,
                item.channel.y_hi,
                claim.y_lo,
                claim.y_hi,
            )
        )
        if obstacles:
            _allocate_channel_around_claims(item, obstacles, ctx)


def build_member_geometry_execution(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    *,
    family_by_edge: Mapping[ResolvedEdge, RouteFamilyId],
    compatibility_system_ids: frozenset[RouteSystemId] = frozenset(),
    preliminary_gap_claims: tuple[PreliminaryGapChannelClaim, ...] = (),
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]] | None = None,
) -> MemberGeometryExecution:
    """Freeze each eligible non-convergence member's sole production template."""
    convergence_edges = _convergence_member_edges(scaffold)
    reservation_ids = reservation_ids_by_member or {}
    candidates: list[_MemberCandidate] = []
    failures: dict[RouteSystemId, str] = {}
    edges_by_system: dict[RouteSystemId, list[ResolvedEdge]] = defaultdict(list)
    for resolved in scaffold.edge_order:
        edges_by_system[scaffold.system_for_edge(resolved)].append(resolved)
    built_start = len(ctx.built_routes)
    try:
        for system_id in scaffold.ordered_system_ids:
            system_edges = tuple(edges_by_system.get(system_id, ()))
            if system_id in compatibility_system_ids:
                _append_compatibility_context(ctx, scaffold, system_id, system_edges)
                continue
            system_start = len(ctx.built_routes)
            system_candidates: list[_MemberCandidate] = []
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
                system_candidates.append(_MemberCandidate(route, family_id, system_id))
                ctx.built_routes.append(route)
            if system_id not in failures:
                candidates.extend(system_candidates)
                continue

            del ctx.built_routes[system_start:]
            _append_compatibility_context(ctx, scaffold, system_id, system_edges)

        candidate_routes = [candidate.route for candidate in candidates]
        _materialize_gap_slots(candidate_routes, ctx)
        _materialize_trunk_slots(candidate_routes, ctx)
        eligible_claims = _eligible_preliminary_gap_claims(
            preliminary_gap_claims,
            failures,
        )
        _allocate_preliminary_gap_claims(
            tuple(candidates),
            eligible_claims,
            ctx,
            {
                system_id: rank
                for rank, system_id in enumerate(scaffold.ordered_system_ids)
            },
        )
        plans = tuple(
            _freeze_plan(scaffold, candidate, ctx, reservation_ids)
            for candidate in candidates
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
