"""Pre-emission convergence planning and template consumption."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

from nf_metro.layout.constants import (
    COORD_TOLERANCE,
    CURVE_RADIUS,
    SECTION_ROUTE_CLEARANCE,
)
from nf_metro.layout.envelope_settlement import (
    EnvelopeCapacityLimitation,
    EnvelopeCapacityProof,
    EnvelopeClaimAllocation,
)
from nf_metro.layout.geometry import (
    perpendicular_port_sides,
    point_to_polyline_distance,
    section_box,
)
from nf_metro.layout.route_plan import (
    BindingKind,
    ConvergenceContinuation,
    ConvergenceDisposition,
    ConvergenceEndpointOwnership,
    ConvergenceEndpointRole,
    ConvergenceLanding,
    ConvergencePlan,
    ConvergencePlanId,
    ConvergenceTrunkAxis,
    ConvergenceTrunkReason,
    CoordinateRegime,
    DemandAxis,
    DemandKind,
    EmissionMemberId,
    ExitTurnPlan,
    ExitTurnPlanId,
    FanPlan,
    FanPlanDisposition,
    FanPlanId,
    GridSpan,
    KeepOutClass,
    RoutePlanDiagnostic,
    RouteSemanticScaffold,
    RouteSystemId,
    SharedReference,
    SharedReferenceId,
    SharedReferenceKind,
    SymbolicDemand,
    TurnHandedness,
    _ordered_unique,
    _plan_provenance,
    convergence_resource_ids,
    grid_span_for_sections,
    reservation_decision_refs,
    turn_handedness,
)
from nf_metro.layout.routing.common import (
    Direction,
    HTrunkSeg,
    OffsetRegime,
    RoutedPath,
    apply_route_offsets,
    iter_horizontal_trunks,
)
from nf_metro.layout.routing.context import (
    _EdgeKey,
    _resolve_section_colrow,
    _RoutingCtx,
)
from nf_metro.layout.routing.orientation import direction_axis
from nf_metro.layout.routing.perp import convergence_perp_entry_lane_coordinate
from nf_metro.parser.model import Edge, MetroGraph, PortSide, Station
from nf_metro.parser.route_topology import (
    EndpointGroupId,
    ResolvedConvergenceView,
    ResolvedEdge,
    semantic_route_id,
)

if TYPE_CHECKING:
    from nf_metro.layout.routing.envelope_allocations import EnvelopeAllocationQuery


class ConvergenceInvariantError(RuntimeError):
    """A planned convergence template violated its immutable contract."""


class UnsupportedConvergenceError(ValueError):
    """Canonical templates cannot represent a complete convergence system."""


class ConvergencePlanningError(RuntimeError):
    """Semantic convergence membership is internally inconsistent."""


class ConvergenceAllocationNeed(str, Enum):
    """A whole-system conflict that a realised envelope can discharge."""

    SHARED_CHANNEL = "shared-channel"
    OPPOSING_OPENINGS = "opposing-openings"
    UNOWNED_CORRIDOR = "unowned-corridor"
    CHAINED_SAME_LINE = "chained-same-line"
    LOCAL_GEOMETRY = "local-geometry"


@dataclass(frozen=True, slots=True)
class ConvergenceAllocationConflict:
    need: ConvergenceAllocationNeed
    message: str
    member_ids: tuple[EmissionMemberId, ...] = ()


_ENVELOPE_ALLOCATION_NEEDS = frozenset(
    {
        ConvergenceAllocationNeed.SHARED_CHANNEL,
        ConvergenceAllocationNeed.OPPOSING_OPENINGS,
        ConvergenceAllocationNeed.UNOWNED_CORRIDOR,
        ConvergenceAllocationNeed.CHAINED_SAME_LINE,
    }
)


_PlanMembership: TypeAlias = tuple[
    tuple[tuple[ResolvedEdge, ...], ...],
    tuple[ResolvedEdge, ...],
    tuple[EmissionMemberId, ...],
]
_OpeningKey: TypeAlias = tuple[EmissionMemberId, DemandAxis, Direction]


@dataclass(frozen=True, slots=True)
class ConvergenceRouteMembership:
    plan: ConvergencePlan
    member_id: EmissionMemberId
    landing: ConvergenceLanding | None
    continuation: ConvergenceContinuation | None
    ownership: ConvergenceEndpointOwnership
    covering_edge: ResolvedEdge | None
    allocations: tuple[EnvelopeClaimAllocation, ...] = ()


@dataclass(frozen=True, slots=True)
class ConvergencePlanExecutionQuery:
    plans: tuple[ConvergencePlan, ...]
    _by_edge: Mapping[ResolvedEdge, ConvergenceRouteMembership]
    _envelope_allocations: EnvelopeAllocationQuery | None = None

    def membership_for_edge(
        self, edge: Edge | ResolvedEdge
    ) -> ConvergenceRouteMembership | None:
        key = (
            edge
            if isinstance(edge, ResolvedEdge)
            else ResolvedEdge(edge.source, edge.target, edge.line_id)
        )
        return self._by_edge.get(key)

    def covering_edge_for_edge(self, edge: Edge | ResolvedEdge) -> ResolvedEdge | None:
        membership = self.membership_for_edge(edge)
        return membership.covering_edge if membership is not None else None

    def project_point(
        self,
        member_id: EmissionMemberId,
        point_rank: int,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        if self._envelope_allocations is None:
            return point
        return self._envelope_allocations.project_point(member_id, point_rank, point)

    def project_segment(
        self,
        member_id: EmissionMemberId,
        segment_rank: int,
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self._envelope_allocations is None:
            return segment
        return self._envelope_allocations.project_segment(
            member_id, segment_rank, segment
        )


@dataclass(frozen=True, slots=True)
class ConvergencePlanExecution:
    plans: tuple[ConvergencePlan, ...]
    references: tuple[SharedReference, ...]
    demands: tuple[SymbolicDemand, ...]
    diagnostics: tuple[RoutePlanDiagnostic, ...]
    query: ConvergencePlanExecutionQuery


def empty_convergence_plan_execution() -> ConvergencePlanExecution:
    query = ConvergencePlanExecutionQuery((), MappingProxyType({}), None)
    return ConvergencePlanExecution((), (), (), (), query)


def _entry_lane_order(
    graph: MetroGraph,
    scaffold: RouteSemanticScaffold,
    view: ResolvedConvergenceView,
) -> tuple[str, ...]:
    entry_group = scaffold.query.endpoint_group_for_port(
        scaffold.query.entry_port(view.group.entry_group_id)
    )
    system_id = scaffold.system_for(view.group.connector_ids)
    lines = {
        scaffold.query.connector(connector_id).line_id
        for connector_id in entry_group.connector_ids
        if scaffold.system_for((connector_id,)) == system_id
    }
    return tuple(line_id for line_id in graph.lines if line_id in lines)


def _direction(a: tuple[float, float], b: tuple[float, float]) -> Direction:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) > abs(dy):
        return Direction.R if dx > 0 else Direction.L
    return Direction.D if dy > 0 else Direction.U


def _owned_paths(
    scaffold: RouteSemanticScaffold,
    view: ResolvedConvergenceView,
) -> tuple[tuple[ResolvedEdge, ...], ...]:
    junction_id = view.junction_id
    paths: list[tuple[ResolvedEdge, ...]] = []
    for connector_id in view.group.connector_ids:
        for path in scaffold.query.resolved_paths(connector_id):
            owned = tuple(
                edge
                for edge in path
                if edge.target == junction_id or edge.source == junction_id
            )
            if owned:
                paths.append(owned)
    return tuple(paths)


def _trunk_route(
    edge: Edge,
    ctx: _RoutingCtx,
) -> RoutedPath:
    from nf_metro.layout.routing.inter_section_handlers import (
        _build_inter_facts,
        _route_merge_trunk_feeder,
    )

    src, tgt = ctx.graph.edge_endpoints(edge)
    route = _route_merge_trunk_feeder(_build_inter_facts(edge, src, tgt, ctx))
    if route is None:
        raise UnsupportedConvergenceError("primary trunk template declined its member")
    _consume_exit_turn(route, ctx)
    _consume_envelope_allocation(route, ctx)
    return route


def _trial_route(edge: Edge, ctx: _RoutingCtx) -> RoutedPath:
    from nf_metro.layout.routing.inter_section_handlers import (
        _build_inter_facts,
        _match_inter_section_rule,
        _route_l_shape,
    )

    source, target = ctx.graph.edge_endpoints(edge)
    facts = _build_inter_facts(edge, source, target, ctx)
    rule = _match_inter_section_rule(facts)
    route = (
        rule.route(facts)
        if rule is not None
        else _route_l_shape(edge, source, target, facts.i, facts.n, ctx)
    )
    if route is None:
        raise UnsupportedConvergenceError("convergence template declined its member")
    _consume_exit_turn(route, ctx)
    _consume_envelope_allocation(route, ctx)
    return route


def _consume_exit_turn(route: RoutedPath, ctx: _RoutingCtx) -> None:
    from nf_metro.layout.routing.exit_turns import consume_exit_turn_route
    from nf_metro.layout.routing.inter_section_handlers import (
        classify_inter_section_family,
    )

    source, target = ctx.graph.edge_endpoints(route.edge)
    family = classify_inter_section_family(route.edge, source, target, ctx)
    if family is None:
        raise UnsupportedConvergenceError(
            "planned convergence member has no routing family"
        )
    consume_exit_turn_route(route, family, ctx)


def _consume_envelope_allocation(route: RoutedPath, ctx: _RoutingCtx) -> None:
    allocations = ctx.envelope_allocations
    if allocations is None:
        return
    member_id = allocations.member_for_route(route)
    if member_id is None:
        return
    available = []
    for allocation in allocations.allocations_for_member(member_id):
        start = allocation.segment_rank
        end = allocation.segment_end_rank + 1
        if start < 0 or end >= len(route.points) or start >= end:
            continue
        coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
        travel_rank = 1 - coordinate_rank
        points = route.points[start : end + 1]
        if any(
            abs(first[coordinate_rank] - second[coordinate_rank]) > COORD_TOLERANCE
            or abs(first[travel_rank] - second[travel_rank]) <= COORD_TOLERANCE
            for first, second in zip(points, points[1:])
        ):
            continue
        available.append(allocation)
    _apply_claim_allocations(route, tuple(available))
    for allocation in allocations.allocations_for_member(member_id):
        if (
            allocation.segment_rank != allocation.segment_end_rank
            or allocation.segment_rank != len(route.points) - 1
        ):
            continue
        coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
        point = list(route.points[allocation.segment_rank])
        point[coordinate_rank] = allocation.coordinate
        route.points[allocation.segment_rank] = (point[0], point[1])


def _consume_final_envelope_allocation(route: RoutedPath, ctx: _RoutingCtx) -> None:
    """Authenticate and record allocations after convergence shaping is complete."""
    if ctx.envelope_allocations is not None:
        ctx.envelope_allocations.consume(route)


def _seat_trial_opening(
    route: RoutedPath,
    member_id: EmissionMemberId,
    opening_coordinates: Mapping[_OpeningKey, float],
    graph: MetroGraph,
) -> None:
    from nf_metro.layout.routing.normalize import (
        _opening_fanout_descent,
        _seat_merge_feeder_opening,
    )

    opening = _opening_fanout_descent(route)
    if opening is not None:
        key = (
            member_id,
            DemandAxis.Y,
            Direction.D if opening.down else Direction.U,
        )
        allocated_coordinate = opening_coordinates.get(key)
        coordinate = (
            allocated_coordinate if allocated_coordinate is not None else opening.x
        )
        if allocated_coordinate is None:
            source_coordinate = route.points[0][0]
            requested_radius = (
                route.curve_radii[0]
                if route.curve_radii and route.curve_radii[0] is not None
                else CURVE_RADIUS
            )
            if opening.x > source_coordinate + COORD_TOLERANCE:
                coordinate = max(coordinate, source_coordinate + requested_radius)
            elif opening.x < source_coordinate - COORD_TOLERANCE:
                coordinate = min(coordinate, source_coordinate - requested_radius)
        if abs(coordinate - opening.x) > COORD_TOLERANCE:
            _seat_merge_feeder_opening(route, coordinate, graph, planned=True)


def _claim_allocations(
    member_id: EmissionMemberId,
    proofs: tuple[EnvelopeCapacityProof, ...],
) -> tuple[EnvelopeClaimAllocation, ...]:
    allocations = tuple(
        allocation
        for proof in proofs
        for reservation in proof.reservations
        for allocation in reservation.allocations
        if allocation.member_id == member_id
        and abs(allocation.coordinate - allocation.original_coordinate)
        > COORD_TOLERANCE
    )
    keys = tuple(
        (item.segment_rank, item.segment_end_rank, item.axis) for item in allocations
    )
    if len(set(keys)) != len(keys):
        raise UnsupportedConvergenceError(
            "settled envelope contains duplicate claim allocations"
        )
    return tuple(
        sorted(
            allocations,
            key=lambda item: (
                item.segment_rank,
                item.segment_end_rank,
                item.axis.value,
            ),
        )
    )


def _apply_claim_allocations(
    route: RoutedPath,
    allocations: tuple[EnvelopeClaimAllocation, ...],
) -> tuple[int, ...]:
    owned: list[int] = []
    for allocation in allocations:
        start = allocation.segment_rank
        end = allocation.segment_end_rank + 1
        if start < 0 or end >= len(route.points) or start >= end:
            raise UnsupportedConvergenceError(
                "settled envelope claim is outside its emitted route"
            )
        points = route.points[start : end + 1]
        coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
        travel_rank = 1 - coordinate_rank
        if any(
            abs(first[coordinate_rank] - second[coordinate_rank]) > COORD_TOLERANCE
            or abs(first[travel_rank] - second[travel_rank]) <= COORD_TOLERANCE
            for first, second in zip(points, points[1:])
        ):
            raise UnsupportedConvergenceError(
                "settled envelope claim disagrees with its emitted segment axis"
            )
        for rank in range(start, end + 1):
            point = list(route.points[rank])
            point[coordinate_rank] = allocation.coordinate
            route.points[rank] = (point[0], point[1])
        owned.extend(range(start, end))
    return _ordered_unique(owned)


def _iter_trunk_runs(route: RoutedPath) -> Iterator[tuple[int, HTrunkSeg]]:
    return iter_horizontal_trunks(route)


def _trunk_run(route: RoutedPath, expected_coordinate: float) -> HTrunkSeg:
    runs = tuple(segment for _rank, segment in _iter_trunk_runs(route))
    if not runs:
        raise UnsupportedConvergenceError(
            "primary trunk template emitted no shared run"
        )
    return min(runs, key=lambda segment: abs(segment.y - expected_coordinate))


def _extend_axis_segment_to_coordinates(
    route: RoutedPath,
    segment_rank: int,
    axis: DemandAxis,
    coordinates: tuple[float, ...],
) -> None:
    """Monotonically extend a routed segment and its two perpendicular flanks."""
    if not coordinates:
        return
    coordinate_rank = 0 if axis is DemandAxis.X else 1
    cross_rank = 1 - coordinate_rank
    start = route.points[segment_rank]
    end = route.points[segment_rank + 1]
    if (
        abs(start[cross_rank] - end[cross_rank]) > COORD_TOLERANCE
        or abs(start[coordinate_rank] - end[coordinate_rank]) <= COORD_TOLERANCE
        or segment_rank == 0
        or segment_rank + 2 >= len(route.points)
    ):
        raise ConvergenceInvariantError("planned trunk segment is not flank-bounded")
    low = min(start[coordinate_rank], end[coordinate_rank], *coordinates)
    high = max(start[coordinate_rank], end[coordinate_rank], *coordinates)
    start_coordinate, end_coordinate = (
        (low, high) if start[coordinate_rank] < end[coordinate_rank] else (high, low)
    )

    def replace_coordinate(
        point: tuple[float, float], coordinate: float
    ) -> tuple[float, float]:
        values = list(point)
        values[coordinate_rank] = coordinate
        return values[0], values[1]

    route.points[segment_rank - 1] = replace_coordinate(
        route.points[segment_rank - 1], start_coordinate
    )
    route.points[segment_rank] = replace_coordinate(start, start_coordinate)
    route.points[segment_rank + 1] = replace_coordinate(end, end_coordinate)
    route.points[segment_rank + 2] = replace_coordinate(
        route.points[segment_rank + 2], end_coordinate
    )


def _extend_trunk_to_upstream_exit_axes(
    trunk_route: RoutedPath,
    trial_routes: Mapping[ResolvedEdge, RoutedPath],
    expected_coordinate: float,
    exit_turn_plan_ids: tuple[ExitTurnPlanId, ...],
    ctx: _RoutingCtx,
) -> HTrunkSeg:
    query = ctx.exit_turns
    if query is None or not exit_turn_plan_ids:
        return _trunk_run(trunk_route, expected_coordinate)
    coordinates = tuple(
        membership.axis.coordinate
        for route in trial_routes.values()
        if (membership := query.membership_for_edge(route.edge)) is not None
        and membership.plan.id in exit_turn_plan_ids
        and membership.axis is not None
        and membership.axis.axis is DemandAxis.X
    )
    runs = tuple(_iter_trunk_runs(trunk_route))
    if not runs:
        raise UnsupportedConvergenceError(
            "primary trunk template emitted no shared run"
        )
    rank, run = min(runs, key=lambda item: abs(item[1].y - expected_coordinate))
    _extend_axis_segment_to_coordinates(
        trunk_route,
        rank,
        DemandAxis.X,
        coordinates,
    )
    return _trunk_run(trunk_route, expected_coordinate)


def _seat_trunk_on_foreign_corridor(
    route: RoutedPath,
    run: HTrunkSeg,
    primary_member_id: EmissionMemberId,
    convergence_member_ids: tuple[EmissionMemberId, ...],
    proofs: tuple[EnvelopeCapacityProof, ...],
) -> HTrunkSeg:
    direction = Direction.R if run.xb > run.xa else Direction.L
    member_ids = set(convergence_member_ids)
    coordinates = tuple(
        dict.fromkeys(
            allocation.coordinate
            for proof in proofs
            if proof.axis is DemandAxis.Y
            for reservation in proof.reservations
            if primary_member_id in reservation.claimant_member_ids
            and not any(
                allocation.member_id == primary_member_id
                for allocation in reservation.allocations
            )
            if reservation.direction is direction
            for allocation in reservation.allocations
            if allocation.member_id not in member_ids
        )
    )
    if not coordinates:
        return run
    if len(coordinates) != 1:
        raise UnsupportedConvergenceError(
            "settled convergence trunk has conflicting foreign corridor coordinates"
        )
    rank = next(rank for rank, candidate in _iter_trunk_runs(route) if candidate == run)
    from nf_metro.layout.routing.normalize import _set_htrunk_y

    _set_htrunk_y(route, rank, coordinates[0])
    return _trunk_run(route, coordinates[0])


def _axis_from_run(run: HTrunkSeg, route: RoutedPath) -> ConvergenceTrunkAxis:
    return ConvergenceTrunkAxis(
        axis=DemandAxis.X,
        coordinate=run.y,
        extent_start=run.x_lo,
        extent_end=run.x_hi,
        direction=Direction.R if run.xb > run.xa else Direction.L,
        source_flank_coordinate=run.before_y,
        target_flank_coordinate=run.after_y,
        source_endpoint_coordinate=route.points[0][0],
        target_endpoint_coordinate=route.points[-1][0],
    )


def _run_from_axis(axis: ConvergenceTrunkAxis) -> HTrunkSeg:
    if axis.axis is not DemandAxis.X:
        raise ConvergenceInvariantError("planned bypass trunk is not horizontal")
    xa, xb = (
        (axis.extent_start, axis.extent_end)
        if axis.direction is Direction.R
        else (axis.extent_end, axis.extent_start)
    )
    return HTrunkSeg(
        y=axis.coordinate,
        xa=xa,
        xb=xb,
        before_y=axis.source_flank_coordinate,
        after_y=axis.target_flank_coordinate,
    )


def _exit_turn_geometry(route: RoutedPath) -> tuple[tuple[float, float], ...] | None:
    if route.exit_lane_transition_plan_id is not None:
        return tuple(route.points)
    if route.exit_turn_segment_rank is None:
        return None
    rank = route.exit_turn_segment_rank
    return tuple(route.points[max(0, rank - 1) : rank + 1])


def _closest_point_on_polyline(
    point: tuple[float, float], points: list[tuple[float, float]]
) -> tuple[float, float]:
    px, py = point
    candidates: list[tuple[float, tuple[float, float]]] = []
    for start, end in zip(points, points[1:]):
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        proportion = (
            0.0
            if length_squared == 0.0
            else max(
                0.0,
                min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared),
            )
        )
        candidate = (ax + proportion * dx, ay + proportion * dy)
        candidates.append(
            (
                (px - candidate[0]) ** 2 + (py - candidate[1]) ** 2,
                candidate,
            )
        )
    if not candidates:
        raise UnsupportedConvergenceError("planned trunk has no drawable segment")
    return min(candidates, key=lambda item: item[0])[1]


def _connect_route_endpoint(route: RoutedPath, target: tuple[float, float]) -> None:
    endpoint = route.points[-1]
    if all(
        abs(actual - expected) <= COORD_TOLERANCE
        for actual, expected in zip(endpoint, target, strict=True)
    ):
        route.points[-1] = target
        return
    prior = route.points[-2]
    horizontal = abs(prior[1] - endpoint[1]) <= COORD_TOLERANCE
    vertical = abs(prior[0] - endpoint[0]) <= COORD_TOLERANCE
    if horizontal and abs(endpoint[1] - target[1]) <= COORD_TOLERANCE:
        route.points[-1] = target
        return
    if vertical and abs(endpoint[0] - target[0]) <= COORD_TOLERANCE:
        route.points[-1] = target
        return
    elbow = (target[0], endpoint[1]) if horizontal else (endpoint[0], target[1])
    if elbow != endpoint and elbow != target:
        route.points.append(elbow)
    route.points.append(target)


def _reset_convergence_curve_radii(
    route: RoutedPath,
    curve_radius: float,
    *,
    preserve_existing: bool = False,
) -> None:
    """Reconcile corner radii with convergence-emitted waypoint geometry."""
    corner_count = max(0, len(route.points) - 2)
    if preserve_existing:
        if route.curve_radii is None:
            return
        route.curve_radii = [
            *route.curve_radii[:corner_count],
            *(
                curve_radius
                for _rank in range(max(0, corner_count - len(route.curve_radii)))
            ),
        ]
        return
    exit_segment_rank = route.exit_turn_segment_rank
    if exit_segment_rank is None:
        route.curve_radii = None
        return
    exit_corner_rank = exit_segment_rank - 1
    if (
        route.curve_radii is None
        or exit_corner_rank < 0
        or exit_corner_rank >= len(route.curve_radii)
    ):
        raise ConvergenceInvariantError(
            "an exit-owned corner has no materialized planned radius"
        )
    if exit_corner_rank >= corner_count:
        raise ConvergenceInvariantError(
            "convergence emission removed an exit-owned corner"
        )
    exit_radius = route.curve_radii[exit_corner_rank]
    route.curve_radii = [curve_radius for _rank in range(corner_count)]
    route.curve_radii[exit_corner_rank] = exit_radius


def _set_convergence_owned_segment_ranks(
    route: RoutedPath,
    ranks: tuple[int, ...],
    *,
    extend: bool = False,
) -> None:
    """Record convergence construction ranks disjoint from the source exit."""
    exit_rank = route.exit_turn_segment_rank
    owned = tuple(rank for rank in ranks if rank != exit_rank)
    if extend:
        owned = route.convergence_owned_segment_ranks + owned
    route.convergence_owned_segment_ranks = _ordered_unique(owned)
    if exit_rank is not None and exit_rank in route.convergence_owned_segment_ranks:
        raise ConvergenceInvariantError(
            "exit and convergence plans claim the same emitted segment"
        )


def _consume_planned_continuation(
    route: RoutedPath,
    continuation: ConvergenceContinuation,
    ctx: _RoutingCtx,
) -> None:
    """Emit one continuation from its planned trunk point to its entry port."""
    _bake_route(route, ctx)
    start = continuation.start_point
    end = continuation.end_point
    port = ctx.graph.ports.get(continuation.entry_port_id)
    if port is None or route.edge.target != continuation.entry_port_id:
        raise ConvergenceInvariantError(
            "planned convergence continuation has no matching entry port"
        )

    horizontal_approach = port.side in {PortSide.LEFT, PortSide.RIGHT}
    expected_approach = {
        PortSide.LEFT: Direction.R,
        PortSide.RIGHT: Direction.L,
        PortSide.TOP: Direction.D,
        PortSide.BOTTOM: Direction.U,
    }[port.side]
    terminal = next(
        (
            (first, second)
            for first, second in reversed(tuple(zip(route.points, route.points[1:])))
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) > COORD_TOLERANCE
        ),
        None,
    )
    if terminal is None or any(
        abs(actual - expected) > COORD_TOLERANCE
        for actual, expected in zip(terminal[1], end, strict=True)
    ):
        raise ConvergenceInvariantError(
            "planned convergence continuation has no matching endpoint"
        )
    terminal_start, _terminal_end = terminal
    if _direction(terminal_start, end) is not expected_approach:
        raise ConvergenceInvariantError(
            "planned convergence continuation cannot preserve its entry runway"
        )

    projected = (start[0], end[1]) if horizontal_approach else (end[0], start[1])
    projected_runway = abs(projected[0] - end[0]) + abs(projected[1] - end[1])
    projected_approach = (
        _direction(projected, end) if projected_runway > COORD_TOLERANCE else None
    )
    candidates: tuple[tuple[float, float], ...]
    if projected_approach is expected_approach and (
        projected_runway >= ctx.curve_radius - COORD_TOLERANCE
    ):
        candidates = (start, projected, end)
    else:
        bridge = (
            (terminal_start[0], start[1])
            if horizontal_approach
            else (start[0], terminal_start[1])
        )
        candidates = (start, bridge, terminal_start, end)
    points: list[tuple[float, float]] = []
    for point in candidates:
        if not points or (
            abs(points[-1][0] - point[0]) + abs(points[-1][1] - point[1])
            > COORD_TOLERANCE
        ):
            points.append(point)
    if len(points) < 2 or any(
        abs(first[0] - second[0]) > COORD_TOLERANCE
        and abs(first[1] - second[1]) > COORD_TOLERANCE
        for first, second in zip(points, points[1:])
    ):
        raise ConvergenceInvariantError(
            "planned convergence continuation has no orthogonal emission"
        )
    route.points = points
    _reset_convergence_curve_radii(
        route,
        ctx.curve_radius,
        preserve_existing=True,
    )
    _set_convergence_owned_segment_ranks(
        route,
        tuple(range(len(points) - 1)),
    )
    from nf_metro.layout.routing.inter_section_handlers import (
        _declare_channel,
        _declare_trunk,
    )

    for first, second in zip(points, points[1:]):
        if (
            abs(first[0] - second[0]) <= COORD_TOLERANCE
            and abs(first[1] - second[1]) > COORD_TOLERANCE
        ):
            _declare_channel(route, ctx, first[0], _direction(first, second))
    _declare_trunk(route, ctx)


def _bake_route(route: RoutedPath, ctx: _RoutingCtx) -> None:
    if route.offset_regime is OffsetRegime.DEFERRED:
        route.points = apply_route_offsets(route, ctx.station_offsets or {})
        route.offset_regime = OffsetRegime.BAKED


def _landing_approach(
    route: RoutedPath, join_point: tuple[float, float]
) -> tuple[Direction, TurnHandedness | None, float] | None:
    for rank, (start, end) in enumerate(zip(route.points, route.points[1:])):
        runway = abs(start[0] - join_point[0]) + abs(start[1] - join_point[1])
        if (
            runway <= COORD_TOLERANCE
            or point_to_polyline_distance(join_point, (start, end)) > COORD_TOLERANCE
        ):
            continue
        approach = _direction(start, join_point)
        handedness = None
        if rank > 0:
            prior = route.points[rank - 1]
            if abs(prior[0] - start[0]) + abs(prior[1] - start[1]) > COORD_TOLERANCE:
                incoming = _direction(prior, start)
                if direction_axis(incoming) is not direction_axis(approach):
                    handedness = turn_handedness(incoming, approach)
        return approach, handedness, runway
    return None


def _landing_from_trial(
    *,
    plan_member_id: EmissionMemberId,
    edge: Edge,
    route: RoutedPath,
    source: Station,
    target: Station,
    run: HTrunkSeg | None,
    trunk_points: list[tuple[float, float]],
    is_trunk: bool,
    ctx: _RoutingCtx,
    lane_rank: int,
) -> ConvergenceLanding:
    if is_trunk:
        assert run is not None
        join_point = (run.xa, run.y)
    elif run is not None:
        from nf_metro.layout.routing.normalize import _land_feeder_on_run

        key = (edge.source, edge.target, edge.line_id)
        if key in ctx.merge.branch_edges:
            exit_turn_geometry = _exit_turn_geometry(route)
            _land_feeder_on_run(route, run, ctx)
            if exit_turn_geometry != _exit_turn_geometry(route):
                raise UnsupportedConvergenceError(
                    "convergence landing conflicts with an upstream exit turn"
                )
        _bake_route(route, ctx)
        join_point = _closest_point_on_polyline(route.points[-1], trunk_points)
        _connect_route_endpoint(route, join_point)
    else:
        _bake_route(route, ctx)
        join_point = _closest_point_on_polyline(route.points[-1], trunk_points)
        _connect_route_endpoint(route, join_point)
    approach = _landing_approach(route, join_point)
    if approach is None:
        raise UnsupportedConvergenceError("convergence landing has no approach")
    approach_direction, handedness, runway = approach
    source_column, source_row = _resolve_section_colrow(ctx.graph, source)
    target_column, target_row = _resolve_section_colrow(ctx.graph, target)
    from nf_metro.layout.routing.normalize import _opening_fanout_descent

    opening_turn = _opening_fanout_descent(route)
    opening_turn_segment = (
        (route.points[opening_turn.idx], route.points[opening_turn.idx + 1])
        if opening_turn is not None
        else None
    )
    column_span = (
        abs(target_column - source_column)
        if source_column is not None and target_column is not None
        else 0
    )
    return ConvergenceLanding(
        member_id=plan_member_id,
        edge=ResolvedEdge(edge.source, edge.target, edge.line_id),
        source_junction_id=edge.source,
        approach_axis=direction_axis(approach_direction),
        approach_direction=approach_direction,
        source_column=source_column,
        source_row=source_row,
        lane_rank=lane_rank,
        order=0,
        join_point=join_point,
        corner_handedness=handedness,
        minimum_runway=runway,
        opening_turn_coordinate=(opening_turn.x if opening_turn is not None else None),
        opening_turn_segment=opening_turn_segment,
        bypass=is_trunk
        or (edge.source, edge.target, edge.line_id) in ctx.merge.branch_edges,
        long_haul=column_span > 1,
        multiple_row=(
            source_row is not None
            and target_row is not None
            and source_row != target_row
        ),
    )


def _direct_axis_points(
    merge: tuple[float, float], entry: tuple[float, float]
) -> ConvergenceTrunkAxis:
    direction = _direction(merge, entry)
    if direction in {Direction.R, Direction.L}:
        start, end = sorted((merge[0], entry[0]))
        coordinate = merge[1]
    else:
        start, end = sorted((merge[1], entry[1]))
        coordinate = merge[0]
    return ConvergenceTrunkAxis(
        direction_axis(direction),
        coordinate,
        start,
        end,
        direction,
        coordinate,
        coordinate,
        merge[0] if direction in {Direction.R, Direction.L} else merge[1],
        entry[0] if direction in {Direction.R, Direction.L} else entry[1],
    )


def _direct_axis(merge: Station, entry: Station) -> ConvergenceTrunkAxis:
    return _direct_axis_points((merge.x, merge.y), (entry.x, entry.y))


def _shared_terminal_axis(
    routes: tuple[RoutedPath, ...],
    target_point: tuple[float, float],
) -> tuple[ConvergenceTrunkAxis, int]:
    segments: list[tuple[DemandAxis, float, float, float, Direction, int]] = []
    for rank, route in enumerate(routes):
        if len(route.points) < 2:
            continue
        start, end = route.points[-2:]
        if any(
            abs(actual - expected) > COORD_TOLERANCE
            for actual, expected in zip(end, target_point, strict=True)
        ):
            continue
        direction = _direction(start, end)
        axis = direction_axis(direction)
        if axis is DemandAxis.X:
            if abs(start[1] - end[1]) > COORD_TOLERANCE:
                continue
            coordinate = end[1]
            extent_start, extent_end = sorted((start[0], end[0]))
        else:
            if abs(start[0] - end[0]) > COORD_TOLERANCE:
                continue
            coordinate = end[0]
            extent_start, extent_end = sorted((start[1], end[1]))
        if extent_end - extent_start > COORD_TOLERANCE:
            segments.append(
                (
                    axis,
                    coordinate,
                    extent_start,
                    extent_end,
                    direction,
                    rank,
                )
            )
    if not segments:
        raise UnsupportedConvergenceError(
            "direct convergence has no emitted terminal approach"
        )
    axis, coordinate, extent_start, extent_end, direction, rank = max(
        segments, key=lambda item: item[3] - item[2]
    )
    source_longitudinal, target_longitudinal = (
        (extent_start, extent_end)
        if direction in {Direction.R, Direction.D}
        else (extent_end, extent_start)
    )
    return (
        ConvergenceTrunkAxis(
            axis,
            coordinate,
            extent_start,
            extent_end,
            direction,
            coordinate,
            coordinate,
            source_longitudinal,
            target_longitudinal,
        ),
        rank,
    )


def _axis_target_point(axis: ConvergenceTrunkAxis) -> tuple[float, float]:
    longitudinal = (
        axis.extent_end
        if axis.direction in {Direction.R, Direction.D}
        else axis.extent_start
    )
    return (
        (longitudinal, axis.coordinate)
        if axis.axis is DemandAxis.X
        else (axis.coordinate, longitudinal)
    )


def _axis_source_point(axis: ConvergenceTrunkAxis) -> tuple[float, float]:
    longitudinal = (
        axis.extent_start
        if axis.direction in {Direction.R, Direction.D}
        else axis.extent_end
    )
    return (
        (longitudinal, axis.coordinate)
        if axis.axis is DemandAxis.X
        else (axis.coordinate, longitudinal)
    )


def _plan_membership(
    scaffold: RouteSemanticScaffold,
    view: ResolvedConvergenceView,
) -> _PlanMembership:
    paths = _owned_paths(scaffold, view)
    edges = _ordered_unique(edge for path in paths for edge in path)
    edge_order = set(scaffold.edge_order)
    if any(edge not in edge_order for edge in edges):
        raise ConvergencePlanningError(
            "convergence membership is absent from emission order"
        )
    try:
        member_ids = tuple(scaffold.member_id_by_edge[edge] for edge in edges)
    except KeyError as error:
        raise ConvergencePlanningError(
            f"resolved convergence member {error.args[0]!r} is not routable"
        ) from error
    return paths, edges, member_ids


def _build_planned_convergence(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    view: ResolvedConvergenceView,
    membership: _PlanMembership,
    exit_turn_plan_ids: tuple[ExitTurnPlanId, ...],
    fan_plan_ids: tuple[FanPlanId, ...],
    opening_coordinates: Mapping[_OpeningKey, float] = MappingProxyType({}),
    envelope_proofs: tuple[EnvelopeCapacityProof, ...] = (),
) -> ConvergencePlan:
    group = view.group
    system_id = scaffold.system_for(group.connector_ids)
    paths, edges, member_ids = membership
    member_by_edge = dict(zip(edges, member_ids, strict=True))
    connector_ids_by_edge = {
        edge: _ordered_unique(
            reference.connector_id for reference in scaffold.refs_by_edge[edge]
        )
        for edge in edges
    }
    entry_port_id = scaffold.query.entry_port(group.entry_group_id)
    entry = graph.stations[entry_port_id]
    outgoing_edges = tuple(edge for edge in edges if edge.source == view.junction_id)
    incoming_edges = tuple(edge for edge in edges if edge.target == view.junction_id)
    if not incoming_edges or not outgoing_edges:
        raise ConvergencePlanningError(
            "convergence has incomplete feeder or continuation membership"
        )

    line_order = _entry_lane_order(graph, scaffold, view)
    if group.line_id not in line_order:
        raise ConvergencePlanningError(
            "convergence line is outside its target entry lane order"
        )
    lane_rank_by_line = {line_id: rank for rank, line_id in enumerate(line_order)}
    trunk_source_id = ctx.merge.trunk_source.get(view.junction_id)
    trial_routes: dict[ResolvedEdge, RoutedPath] = {}
    trunk_run = None
    if trunk_source_id is not None:
        trunk_edge_key = next(
            (
                edge
                for edge in incoming_edges
                if edge.source == trunk_source_id and edge.line_id == group.line_id
            ),
            None,
        )
        if trunk_edge_key is None:
            raise ConvergencePlanningError(
                "classified primary trunk is outside convergence membership"
            )
        edge = ctx.edge_by_key[
            (trunk_edge_key.source, trunk_edge_key.target, trunk_edge_key.line_id)
        ]
        trunk_route = _trunk_route(edge, ctx)
        _seat_trial_opening(
            trunk_route,
            member_by_edge[trunk_edge_key],
            opening_coordinates,
            graph,
        )
        _consume_envelope_allocation(trunk_route, ctx)
        trial_routes[trunk_edge_key] = trunk_route
        trunk_run = _trunk_run(
            trial_routes[trunk_edge_key], ctx.merge.trunk_by[view.junction_id]
        )
        trunk_axis = _axis_from_run(trunk_run, trial_routes[trunk_edge_key])
        primary_member_id = member_by_edge[trunk_edge_key]
        primary_reason = ConvergenceTrunkReason.LONGEST_BYPASS
    else:
        trunk_edge_key = outgoing_edges[0]
        edge = ctx.edge_by_key[
            (trunk_edge_key.source, trunk_edge_key.target, trunk_edge_key.line_id)
        ]
        trial_routes[trunk_edge_key] = _trial_route(edge, ctx)
        direct_route = trial_routes[trunk_edge_key]
        trunk_axis = _direct_axis_points(
            direct_route.points[0], direct_route.points[-1]
        )
        primary_member_id = member_by_edge[trunk_edge_key]
        primary_reason = ConvergenceTrunkReason.OUTGOING_CONTINUATION

    for edge_key in incoming_edges:
        edge = ctx.edge_by_key[(edge_key.source, edge_key.target, edge_key.line_id)]
        route = trial_routes.get(edge_key)
        if route is None:
            if (
                trunk_run is not None
                and (
                    edge.source,
                    edge.target,
                    edge.line_id,
                )
                in ctx.merge.branch_edges
            ):
                from nf_metro.layout.routing.inter_section_handlers import (
                    _build_inter_facts,
                    _route_merge_branch_feeder,
                )

                src, tgt = graph.edge_endpoints(edge)
                route = _route_merge_branch_feeder(
                    _build_inter_facts(edge, src, tgt, ctx)
                )
                if route is not None:
                    _consume_exit_turn(route, ctx)
                    _consume_envelope_allocation(route, ctx)
            else:
                route = _trial_route(edge, ctx)
        if route is None:
            raise UnsupportedConvergenceError(f"feeder template declined {edge_key!r}")
        _seat_trial_opening(
            route,
            member_by_edge[edge_key],
            opening_coordinates,
            graph,
        )
        _consume_envelope_allocation(route, ctx)
        trial_routes[edge_key] = route

    if trunk_run is not None:
        trunk_route = trial_routes[trunk_edge_key]
        trunk_run = _extend_trunk_to_upstream_exit_axes(
            trunk_route,
            trial_routes,
            ctx.merge.trunk_by[view.junction_id],
            exit_turn_plan_ids,
            ctx,
        )
        trunk_run = _seat_trunk_on_foreign_corridor(
            trunk_route,
            trunk_run,
            primary_member_id,
            member_ids,
            envelope_proofs,
        )
        trunk_axis = _axis_from_run(trunk_run, trunk_route)
        from nf_metro.layout.routing.normalize import (
            _merge_feeder_groups,
            _snap_merge_feeder_group,
        )

        exit_turn_geometry = {
            edge_key: _exit_turn_geometry(route)
            for edge_key, route in trial_routes.items()
        }
        trial_route_list = list(trial_routes.values())
        for feeder_group in _merge_feeder_groups(trial_route_list, ctx):
            _snap_merge_feeder_group(feeder_group, graph)
        for route in trial_route_list:
            _consume_envelope_allocation(route, ctx)
        if any(
            exit_turn_geometry[edge_key] != _exit_turn_geometry(route)
            for edge_key, route in trial_routes.items()
        ):
            raise UnsupportedConvergenceError(
                "convergence alignment conflicts with an upstream exit turn"
            )
    else:
        for edge_key in incoming_edges:
            _bake_route(trial_routes[edge_key], ctx)
            _consume_envelope_allocation(trial_routes[edge_key], ctx)
        try:
            trunk_axis, carrier_rank = _shared_terminal_axis(
                tuple(trial_routes[edge_key] for edge_key in incoming_edges),
                trial_routes[outgoing_edges[0]].points[-1],
            )
        except UnsupportedConvergenceError:
            outgoing_edge_key = outgoing_edges[0]
            direct_route = trial_routes[outgoing_edge_key]
            if (
                outgoing_edge_key.source,
                outgoing_edge_key.target,
                outgoing_edge_key.line_id,
            ) in ctx.skip_edges:
                for edge_key in incoming_edges:
                    _connect_route_endpoint(
                        trial_routes[edge_key], direct_route.points[-1]
                    )
                trunk_axis, carrier_rank = _shared_terminal_axis(
                    tuple(trial_routes[edge_key] for edge_key in incoming_edges),
                    direct_route.points[-1],
                )
                trunk_edge_key = incoming_edges[carrier_rank]
                primary_member_id = member_by_edge[trunk_edge_key]
                primary_reason = ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
            else:
                trunk_edge_key = outgoing_edge_key
                _bake_route(direct_route, ctx)
                trunk_axis = _direct_axis_points(
                    direct_route.points[0], direct_route.points[-1]
                )
                primary_member_id = member_by_edge[trunk_edge_key]
                primary_reason = ConvergenceTrunkReason.OUTGOING_CONTINUATION
        else:
            trunk_edge_key = incoming_edges[carrier_rank]
            primary_member_id = member_by_edge[trunk_edge_key]
            primary_reason = ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH

    landings: list[ConvergenceLanding] = []
    trunk_points = trial_routes[trunk_edge_key].points
    for edge_key in incoming_edges:
        edge = ctx.edge_by_key[(edge_key.source, edge_key.target, edge_key.line_id)]
        route = trial_routes[edge_key]
        is_trunk = edge_key == trunk_edge_key and trunk_run is not None
        landing = _landing_from_trial(
            plan_member_id=member_by_edge[edge_key],
            edge=edge,
            route=route,
            source=graph.stations[edge.source],
            target=entry,
            run=trunk_run,
            trunk_points=trunk_points,
            is_trunk=is_trunk,
            ctx=ctx,
            lane_rank=lane_rank_by_line[edge.line_id],
        )
        landings.append(landing)
    landings.sort(
        key=lambda item: (
            item.join_point[0]
            if trunk_axis.direction is Direction.R
            else -item.join_point[0],
            item.join_point[1]
            if trunk_axis.direction is Direction.D
            else -item.join_point[1],
            str(item.member_id),
        )
    )
    landings = [replace(item, order=rank) for rank, item in enumerate(landings)]

    continuations: list[ConvergenceContinuation] = []
    ownership: list[ConvergenceEndpointOwnership] = []
    landing_by_member = {item.member_id: item for item in landings}
    edge_by_member = dict(zip(member_ids, edges, strict=True))
    for edge_key, member_id in zip(edges, member_ids, strict=True):
        if edge_key.target == view.junction_id:
            landing = landing_by_member[member_id]
            role = (
                ConvergenceEndpointRole.TRUNK
                if member_id == primary_member_id
                and primary_reason
                in {
                    ConvergenceTrunkReason.LONGEST_BYPASS,
                    ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH,
                }
                else ConvergenceEndpointRole.FEEDER
            )
            endpoint = (
                trial_routes[trunk_edge_key].points[-1]
                if role is ConvergenceEndpointRole.TRUNK
                else landing.join_point
            )
            ownership.append(
                ConvergenceEndpointOwnership(
                    member_id=member_id,
                    edge=edge_key,
                    connector_ids=connector_ids_by_edge[edge_key],
                    role=role,
                    endpoint=endpoint,
                )
            )
            continue
        continuation_route = trial_routes.get(edge_key)
        if continuation_route is None:
            edge = ctx.edge_by_key[(edge_key.source, edge_key.target, edge_key.line_id)]
            continuation_route = _trial_route(edge, ctx)
            trial_routes[edge_key] = continuation_route
        end_point = continuation_route.points[-1]
        hop_start_point = continuation_route.points[0]
        start_point = (
            _axis_source_point(trunk_axis)
            if primary_reason
            in {
                ConvergenceTrunkReason.OUTGOING_CONTINUATION,
                ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH,
            }
            else _axis_target_point(trunk_axis)
        )
        feeder_at_start = any(
            all(
                abs(actual - expected) <= COORD_TOLERANCE
                for actual, expected in zip(
                    trial_routes[item].points[-1], hop_start_point, strict=True
                )
            )
            for item in incoming_edges
        )
        endpoint_carriers = tuple(
            item
            for item in incoming_edges
            if all(
                abs(actual - expected) <= COORD_TOLERANCE
                for actual, expected in zip(
                    trial_routes[item].points[-1], end_point, strict=True
                )
            )
        )
        carrier_edge = next(
            (
                item
                for item in endpoint_carriers
                if point_to_polyline_distance(start_point, trial_routes[item].points)
                <= COORD_TOLERANCE
            ),
            (
                endpoint_carriers[0]
                if endpoint_carriers
                and primary_reason is ConvergenceTrunkReason.OUTGOING_CONTINUATION
                and member_id == primary_member_id
                else None
            ),
        )
        immutable_binding = (
            ctx.envelope_allocations.immutable_binding_for(member_id)
            if ctx.envelope_allocations is not None
            else None
        )
        if immutable_binding is not None:
            covered_by = (
                immutable_binding.covering_member_id
                if immutable_binding.kind is not BindingKind.EMITTED
                else None
            )
        else:
            covered_by = (
                member_by_edge[carrier_edge]
                if carrier_edge is not None
                and (
                    not feeder_at_start
                    or (edge_key.source, edge_key.target, edge_key.line_id)
                    in ctx.skip_edges
                )
                else None
            )
        if covered_by is not None and covered_by not in member_ids:
            raise UnsupportedConvergenceError(
                "immutable continuation carrier is outside convergence membership"
            )
        if immutable_binding is not None and covered_by is not None:
            carrier_edge = edge_by_member[covered_by]
            carrier_route = trial_routes.get(carrier_edge)
            if carrier_route is None or carrier_edge not in incoming_edges:
                raise UnsupportedConvergenceError(
                    "immutable continuation carrier is not a convergence feeder"
                )
            carrier_axis, _rank = _shared_terminal_axis((carrier_route,), end_point)
            start_point = _axis_source_point(carrier_axis)
        if (
            carrier_edge is not None
            and covered_by is not None
            and point_to_polyline_distance(
                start_point, trial_routes[carrier_edge].points
            )
            > COORD_TOLERANCE
            and not (
                primary_reason is ConvergenceTrunkReason.OUTGOING_CONTINUATION
                and member_id == primary_member_id
            )
        ):
            raise UnsupportedConvergenceError(
                "covered continuation is absent from its carrier"
            )
        continuations.append(
            ConvergenceContinuation(
                member_id,
                edge_key,
                entry_port_id,
                lane_rank_by_line[edge_key.line_id],
                start_point,
                end_point,
                covered_by,
            )
        )
        ownership.append(
            ConvergenceEndpointOwnership(
                member_id=member_id,
                edge=edge_key,
                connector_ids=connector_ids_by_edge[edge_key],
                role=(
                    ConvergenceEndpointRole.COVERED_CONTINUATION
                    if covered_by is not None
                    else ConvergenceEndpointRole.CONTINUATION
                ),
                endpoint=end_point,
                covered_by_member_id=covered_by,
            )
        )

    if primary_reason is ConvergenceTrunkReason.OUTGOING_CONTINUATION:
        (continuation,) = continuations
        if continuation.covered_by_member_id is not None:
            primary_member_id = continuation.covered_by_member_id
            carrier_ownership = next(
                item for item in ownership if item.member_id == primary_member_id
            )
            carrier_route = trial_routes[carrier_ownership.edge]
            trunk_axis, _rank = _shared_terminal_axis(
                (carrier_route,), continuation.end_point
            )
            primary_reason = ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
            continuations = [
                replace(
                    continuation,
                    start_point=_axis_source_point(trunk_axis),
                )
            ]
            ownership = [
                replace(item, role=ConvergenceEndpointRole.TRUNK)
                if item.member_id == primary_member_id
                else item
                for item in ownership
            ]

    primary_ownership = next(
        item for item in ownership if item.member_id == primary_member_id
    )
    target_arm_continuation_ids = _unbound_target_arm_continuations(
        tuple(continuations), primary_member_id, trunk_axis, ctx
    )
    covered_continuation_ids = (
        set()
        if envelope_proofs
        else {
            item.member_id
            for item in continuations
            if item.member_id != primary_member_id
            and item.covered_by_member_id is None
            and (
                ctx.envelope_allocations is None
                or ctx.envelope_allocations.immutable_binding_for(item.member_id)
                is None
            )
            and all(
                abs(actual - expected) <= COORD_TOLERANCE
                for actual, expected in zip(
                    primary_ownership.endpoint, item.end_point, strict=True
                )
            )
            and _point_on_trunk_geometry(item.start_point, trunk_axis)
        }
    ) | target_arm_continuation_ids
    continuations = [
        replace(item, covered_by_member_id=primary_member_id)
        if item.member_id in covered_continuation_ids
        else item
        for item in continuations
    ]
    ownership = [
        replace(
            item,
            role=ConvergenceEndpointRole.COVERED_CONTINUATION,
            covered_by_member_id=primary_member_id,
        )
        if item.member_id in covered_continuation_ids
        else item
        for item in ownership
    ]
    trunk_axis, ownership = _cover_target_arm_continuation(
        tuple(continuations),
        target_arm_continuation_ids,
        primary_member_id,
        trunk_axis,
        ownership,
    )

    plan_id = ConvergencePlanId(
        semantic_route_id("convergence-plan", system_id, group.id)
    )
    reference_ids, demand_ids = convergence_resource_ids(plan_id)
    return ConvergencePlan(
        id=plan_id,
        system_id=system_id,
        convergence_ids=(group.id,),
        entry_group_ids=(group.entry_group_id,),
        merge_junction_ids=(view.junction_id,),
        target_entry_port_ids=(entry_port_id,),
        connector_ids=group.connector_ids,
        member_ids=member_ids,
        resolved_member_paths=paths,
        resolved_member_edges=edges,
        line_ids=(group.line_id,),
        upstream_exit_turn_plan_ids=exit_turn_plan_ids,
        upstream_fan_plan_ids=fan_plan_ids,
        primary_trunk_member_id=primary_member_id,
        primary_trunk_reason=primary_reason,
        trunk_axis=trunk_axis,
        landings=tuple(landings),
        outgoing_continuations=tuple(continuations),
        lane_order=line_order,
        endpoint_ownership=tuple(ownership),
        shared_reference_ids=reference_ids,
        demand_ids=demand_ids,
        foreign_reference_ids=(),
        disposition=ConvergenceDisposition.PLANNED,
        legacy_reason=None,
    )


def _legacy_plan(
    scaffold: RouteSemanticScaffold,
    view: ResolvedConvergenceView,
    membership: _PlanMembership,
    reason: str,
) -> ConvergencePlan:
    group = view.group
    paths, edges, member_ids = membership
    system_id = scaffold.system_for(group.connector_ids)
    return ConvergencePlan(
        id=ConvergencePlanId(
            semantic_route_id("convergence-plan", system_id, group.id)
        ),
        system_id=system_id,
        convergence_ids=(group.id,),
        entry_group_ids=(group.entry_group_id,),
        merge_junction_ids=(view.junction_id,),
        target_entry_port_ids=(scaffold.query.entry_port(group.entry_group_id),),
        connector_ids=group.connector_ids,
        member_ids=member_ids,
        resolved_member_paths=paths,
        resolved_member_edges=edges,
        line_ids=(group.line_id,),
        upstream_exit_turn_plan_ids=(),
        upstream_fan_plan_ids=(),
        primary_trunk_member_id=None,
        primary_trunk_reason=None,
        trunk_axis=None,
        landings=(),
        outgoing_continuations=(),
        lane_order=(),
        endpoint_ownership=(),
        shared_reference_ids=(),
        demand_ids=(),
        foreign_reference_ids=(),
        disposition=ConvergenceDisposition.LEGACY,
        legacy_reason=reason,
    )


def _plan_span(graph: MetroGraph, plan: ConvergencePlan) -> GridSpan:
    topology = graph.route_topology
    if topology is None:
        raise ValueError("convergence planning requires resolved route topology")
    connector_by_id = {item.id: item for item in topology.connectors}
    section_ids = _ordered_unique(
        section_id
        for connector_id in plan.connector_ids
        for section_id in (
            connector_by_id[connector_id].source_section,
            connector_by_id[connector_id].target_section,
        )
    )
    return grid_span_for_sections(graph, section_ids)


def _parallel_segments_conflict(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
    clearance: float,
) -> bool:
    (first_start, first_end), (second_start, second_end) = first, second
    first_horizontal = abs(first_start[1] - first_end[1]) <= COORD_TOLERANCE
    second_horizontal = abs(second_start[1] - second_end[1]) <= COORD_TOLERANCE
    first_vertical = abs(first_start[0] - first_end[0]) <= COORD_TOLERANCE
    second_vertical = abs(second_start[0] - second_end[0]) <= COORD_TOLERANCE
    if not (
        first_horizontal and second_horizontal or first_vertical and second_vertical
    ):
        return False
    if first_horizontal:
        separation = abs(first_start[1] - second_start[1])
        first_extent = sorted((first_start[0], first_end[0]))
        second_extent = sorted((second_start[0], second_end[0]))
    else:
        separation = abs(first_start[0] - second_start[0])
        first_extent = sorted((first_start[1], first_end[1]))
        second_extent = sorted((second_start[1], second_end[1]))
    overlap = min(first_extent[1], second_extent[1]) - max(
        first_extent[0], second_extent[0]
    )
    return separation < clearance and overlap > COORD_TOLERANCE


def _route_segments(
    route: RoutedPath,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(zip(route.points, route.points[1:]))


def _landing_cross_segment(
    landing: ConvergenceLanding,
    graph: MetroGraph,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if landing.corner_handedness is None:
        return None
    source = graph.stations[landing.source_junction_id]
    if landing.approach_axis is DemandAxis.X:
        runway_sign = 1.0 if landing.approach_direction is Direction.R else -1.0
        turn_coordinate = (
            landing.opening_turn_coordinate
            if landing.opening_turn_coordinate is not None
            else landing.join_point[0] - runway_sign * landing.minimum_runway
        )
        segment = (
            (turn_coordinate, source.y),
            (turn_coordinate, landing.join_point[1]),
        )
    else:
        runway_sign = 1.0 if landing.approach_direction is Direction.D else -1.0
        turn_coordinate = landing.join_point[1] - (runway_sign * landing.minimum_runway)
        segment = (
            (source.x, turn_coordinate),
            (landing.join_point[0], turn_coordinate),
        )
    if all(
        abs(start - end) <= COORD_TOLERANCE for start, end in zip(*segment, strict=True)
    ):
        return None
    return segment


def _opposing_landing_approach_members(
    plans: tuple[ConvergencePlan, ...], graph: MetroGraph
) -> tuple[EmissionMemberId, ...]:
    landing_crosses = tuple(
        (plan, landing, segment, _direction(*segment))
        for plan in plans
        for landing in plan.landings
        if (segment := _landing_cross_segment(landing, graph)) is not None
    )
    for rank, (first_plan, first, first_segment, first_direction) in enumerate(
        landing_crosses
    ):
        for second_plan, second, second_segment, second_direction in landing_crosses[
            rank + 1 :
        ]:
            if (
                first_plan.id != second_plan.id
                and first.edge.line_id == second.edge.line_id
                and first_direction is not second_direction
                and _parallel_segments_conflict(
                    first_segment, second_segment, COORD_TOLERANCE
                )
            ):
                return (first.member_id, second.member_id)
    return ()


def _move_trunk_flank(
    plan: ConvergencePlan,
    flank_rank: int,
    coordinate: float,
) -> ConvergencePlan:
    axis = plan.trunk_axis
    assert axis is not None
    source_flank = flank_rank == 1
    moves_start = source_flank == (axis.direction in {Direction.R, Direction.D})
    old_coordinate = axis.extent_start if moves_start else axis.extent_end
    new_axis = (
        replace(axis, extent_start=coordinate)
        if moves_start
        else replace(axis, extent_end=coordinate)
    )
    old_segment = _trunk_segments(axis)[flank_rank]

    landings: list[ConvergenceLanding] = []
    moved_join_by_member: dict[EmissionMemberId, tuple[float, float]] = {}
    for landing in plan.landings:
        if (
            point_to_polyline_distance(landing.join_point, old_segment)
            > COORD_TOLERANCE
        ):
            landings.append(landing)
            continue
        join_point = (
            (coordinate, landing.join_point[1])
            if axis.axis is DemandAxis.X
            else (landing.join_point[0], coordinate)
        )
        opening = landing.opening_turn_coordinate
        opening_segment = landing.opening_turn_segment
        if axis.axis is DemandAxis.X and landing.approach_axis is DemandAxis.Y:
            if opening is not None and abs(opening - old_coordinate) <= COORD_TOLERANCE:
                opening = coordinate
                assert opening_segment is not None
                opening_segment = (
                    (coordinate, opening_segment[0][1]),
                    (coordinate, opening_segment[1][1]),
                )
        moved = replace(
            landing,
            join_point=join_point,
            opening_turn_coordinate=opening,
            opening_turn_segment=opening_segment,
        )
        landings.append(moved)
        moved_join_by_member[landing.member_id] = join_point

    old_axis_point = (
        _axis_source_point(axis) if source_flank else _axis_target_point(axis)
    )
    new_axis_point = (
        _axis_source_point(new_axis) if source_flank else _axis_target_point(new_axis)
    )
    continuations = tuple(
        replace(item, start_point=new_axis_point)
        if all(
            abs(actual - expected) <= COORD_TOLERANCE
            for actual, expected in zip(item.start_point, old_axis_point, strict=True)
        )
        else item
        for item in plan.outgoing_continuations
    )
    ownership = tuple(
        replace(item, endpoint=moved_join_by_member[item.member_id])
        if item.member_id in moved_join_by_member
        and item.role is ConvergenceEndpointRole.FEEDER
        else item
        for item in plan.endpoint_ownership
    )
    return replace(
        plan,
        trunk_axis=new_axis,
        landings=tuple(landings),
        outgoing_continuations=continuations,
        endpoint_ownership=ownership,
    )


def _settle_landing_trunk_flanks(
    plans: tuple[ConvergencePlan, ...],
    graph: MetroGraph,
    curve_radius: float,
    fixed_openings: Mapping[_OpeningKey, float] = MappingProxyType({}),
) -> tuple[ConvergencePlan, ...]:
    settled = list(plans)
    clearance = curve_radius + COORD_TOLERANCE
    for landing_plan in plans:
        for landing in landing_plan.landings:
            landing_segment = _landing_cross_segment(landing, graph)
            if landing_segment is None:
                continue
            landing_direction = _direction(*landing_segment)
            landing_horizontal = (
                abs(landing_segment[0][1] - landing_segment[1][1]) <= COORD_TOLERANCE
            )
            landing_coordinate = (
                landing_segment[0][1] if landing_horizontal else landing_segment[0][0]
            )
            for plan_rank, trunk_plan in enumerate(tuple(settled)):
                axis = trunk_plan.trunk_axis
                if trunk_plan.id == landing_plan.id or axis is None:
                    continue
                for flank_rank in (1, 3):
                    flank = _trunk_segments(axis)[flank_rank]
                    if (
                        landing.edge.line_id not in trunk_plan.line_ids
                        or landing_direction is _direction(*flank)
                        or not _parallel_segments_conflict(
                            landing_segment, flank, curve_radius
                        )
                    ):
                        continue
                    flank_coordinate = (
                        flank[0][1] if landing_horizontal else flank[0][0]
                    )
                    if any(
                        item.opening_turn_coordinate is not None
                        and item.opening_turn_segment is not None
                        and abs(item.opening_turn_coordinate - flank_coordinate)
                        <= COORD_TOLERANCE
                        and (
                            item.member_id,
                            direction_axis(_direction(*item.opening_turn_segment)),
                            _direction(*item.opening_turn_segment),
                        )
                        in fixed_openings
                        for item in trunk_plan.landings
                    ):
                        continue
                    endpoint = (
                        axis.source_endpoint_coordinate
                        if flank_rank == 1
                        else axis.target_endpoint_coordinate
                    )
                    if (
                        endpoint is None
                        or abs(endpoint - flank_coordinate) <= clearance
                    ):
                        continue
                    toward_endpoint = 1.0 if endpoint > flank_coordinate else -1.0
                    coordinate = landing_coordinate + toward_endpoint * clearance
                    if (endpoint - coordinate) * toward_endpoint <= curve_radius:
                        continue
                    moved = _move_trunk_flank(trunk_plan, flank_rank, coordinate)
                    settled[plan_rank] = moved
                    trunk_plan = moved
                    axis = moved.trunk_axis
                    assert axis is not None
    return tuple(settled)


def _perp_entry_family_source(
    graph: MetroGraph, entry: Station, ctx: _RoutingCtx
) -> tuple[float, float]:
    port = graph.ports[entry.id]
    section = graph.section_for_port(port)
    if port.side is PortSide.TOP:
        from nf_metro.layout.routing.inter_section_handlers import (
            _top_entry_above_channel_y,
        )

        return entry.x, _top_entry_above_channel_y(ctx, section)
    if port.side is PortSide.BOTTOM:
        from nf_metro.layout.routing.inter_section_handlers import (
            _bottom_entry_below_channel_y,
        )

        return entry.x, _bottom_entry_below_channel_y(ctx, section)
    clearance = SECTION_ROUTE_CLEARANCE + ctx.curve_radius
    left, _top, right, _bottom = section_box(section)
    if port.side is PortSide.LEFT:
        return left - clearance, entry.y
    return right + clearance, entry.y


def _plan_perp_entry_terminal_families(
    plans: tuple[ConvergencePlan, ...], graph: MetroGraph, ctx: _RoutingCtx
) -> tuple[ConvergencePlan, ...]:
    """Plan one exterior terminal-axis family per perpendicular entry."""
    groups: defaultdict[str, list[tuple[int, ConvergencePlan]]] = defaultdict(list)
    for rank, plan in enumerate(plans):
        if (
            plan.owns_geometry
            and plan.primary_trunk_reason
            is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
            and len(plan.target_entry_port_ids) == 1
        ):
            groups[plan.target_entry_port_ids[0]].append((rank, plan))

    settled = list(plans)
    for entry_id, members in groups.items():
        entry = graph.stations[entry_id]
        port = graph.ports[entry_id]
        section = graph.section_for_port(port)
        if len(
            {line for _rank, plan in members for line in plan.line_ids}
        ) < 2 or port.side not in perpendicular_port_sides(section.direction):
            continue
        family_source = _perp_entry_family_source(graph, entry, ctx)
        expected_direction = {
            PortSide.TOP: Direction.D,
            PortSide.BOTTOM: Direction.U,
            PortSide.LEFT: Direction.R,
            PortSide.RIGHT: Direction.L,
        }[port.side]
        expected_axis = (
            DemandAxis.Y
            if port.side in {PortSide.TOP, PortSide.BOTTOM}
            else DemandAxis.X
        )
        family_line_order = tuple(
            line_id
            for _coordinate, line_id in sorted(
                (
                    convergence_perp_entry_lane_coordinate(ctx, entry_id, line_id),
                    line_id,
                )
                for _rank, member in members
                for line_id in member.line_ids
            )
            if _coordinate is not None
        )
        for plan_rank, plan in members:
            (line_id,) = plan.line_ids
            lane_rank = plan.lane_order.index(line_id)
            approach_step = max(ctx.offset_step, ctx.curve_radius + COORD_TOLERANCE)
            outward = {
                PortSide.TOP: (0.0, -1.0),
                PortSide.BOTTOM: (0.0, 1.0),
                PortSide.LEFT: (-1.0, 0.0),
                PortSide.RIGHT: (1.0, 0.0),
            }[port.side]
            line_source = (
                family_source[0] + outward[0] * lane_rank * approach_step,
                family_source[1] + outward[1] * lane_rank * approach_step,
            )
            lane_coordinate = convergence_perp_entry_lane_coordinate(
                ctx, entry_id, line_id
            )
            if lane_coordinate is None:
                raise ConvergencePlanningError(
                    "perpendicular entry terminal family has no lane coordinate"
                )
            target = (
                (lane_coordinate, entry.y)
                if expected_axis is DemandAxis.Y
                else (entry.x, lane_coordinate)
            )
            source = (
                (lane_coordinate, line_source[1])
                if expected_axis is DemandAxis.Y
                else (line_source[0], lane_coordinate)
            )
            axis = _direct_axis_points(source, target)
            if (
                axis.axis is not expected_axis
                or axis.direction is not expected_direction
            ):
                raise ConvergencePlanningError(
                    "perpendicular entry terminal family points toward the exterior"
                )

            landings = []
            moved_endpoints: dict[EmissionMemberId, tuple[float, float]] = {}
            for landing in plan.landings:
                primary = landing.member_id == plan.primary_trunk_member_id
                join = target if primary else source
                opening_segment = landing.opening_turn_segment
                if opening_segment is not None:
                    opening_start, opening_end = opening_segment
                    feeder = graph.stations[landing.source_junction_id]
                    feeder_offset = (ctx.station_offsets or {}).get(
                        (feeder.id, line_id), 0.0
                    )
                    opening_vertical = (
                        abs(opening_start[0] - opening_end[0]) <= COORD_TOLERANCE
                    )
                    opening_start = (
                        (opening_start[0], feeder.y + feeder_offset)
                        if opening_vertical
                        else (feeder.x - feeder_offset, opening_start[1])
                    )
                    opening_segment = (
                        opening_start,
                        (opening_end[0], source[1])
                        if expected_axis is DemandAxis.Y
                        else (source[0], opening_end[1]),
                    )
                if primary:
                    feeder = graph.stations[landing.source_junction_id]
                    transverse_start = (
                        (feeder.x, source[1])
                        if expected_axis is DemandAxis.Y
                        else (source[0], feeder.y)
                    )
                    transverse = _direction(transverse_start, source)
                    moved = replace(
                        landing,
                        join_point=join,
                        approach_axis=expected_axis,
                        approach_direction=expected_direction,
                        corner_handedness=turn_handedness(
                            transverse, expected_direction
                        ),
                        minimum_runway=abs(target[0] - source[0])
                        + abs(target[1] - source[1]),
                        opening_turn_segment=opening_segment,
                    )
                else:
                    approach_direction = landing.approach_direction
                    handedness = landing.corner_handedness
                    runway = landing.minimum_runway
                    if opening_segment is not None:
                        approach_direction = _direction(opening_segment[1], source)
                        handedness = turn_handedness(
                            _direction(*opening_segment), approach_direction
                        )
                        runway = ctx.curve_radius
                    moved = replace(
                        landing,
                        join_point=join,
                        approach_axis=direction_axis(approach_direction),
                        approach_direction=approach_direction,
                        corner_handedness=handedness,
                        minimum_runway=runway,
                        opening_turn_segment=opening_segment,
                    )
                landings.append(moved)
                moved_endpoints[landing.member_id] = join

            continuations = tuple(
                replace(
                    item,
                    lane_rank=family_line_order.index(item.edge.line_id),
                    start_point=source,
                    end_point=target,
                )
                for item in plan.outgoing_continuations
            )
            landings = [
                replace(
                    item,
                    lane_rank=family_line_order.index(item.edge.line_id),
                )
                for item in landings
            ]
            ownership = tuple(
                replace(
                    item,
                    endpoint=moved_endpoints.get(item.member_id, item.endpoint),
                )
                if item.role is ConvergenceEndpointRole.FEEDER
                else replace(item, endpoint=target)
                if item.role
                in {
                    ConvergenceEndpointRole.TRUNK,
                    ConvergenceEndpointRole.CONTINUATION,
                    ConvergenceEndpointRole.COVERED_CONTINUATION,
                }
                else item
                for item in plan.endpoint_ownership
            )
            settled[plan_rank] = replace(
                plan,
                trunk_axis=axis,
                landings=tuple(landings),
                outgoing_continuations=continuations,
                lane_order=family_line_order,
                endpoint_ownership=ownership,
            )
    return tuple(settled)


def _divergence_opening_coordinates(
    plans: tuple[ConvergencePlan, ...],
    graph: MetroGraph,
    fixed: Mapping[_OpeningKey, float],
) -> Mapping[_OpeningKey, float]:
    groups: defaultdict[
        tuple[RouteSystemId, str, Direction, str],
        list[tuple[int, int, ConvergenceLanding]],
    ] = defaultdict(list)
    for plan_rank, plan in enumerate(plans):
        if not plan.owns_geometry:
            continue
        for landing_rank, landing in enumerate(plan.landings):
            segment = landing.opening_turn_segment
            if segment is None:
                continue
            groups[
                plan.system_id,
                landing.source_junction_id,
                _direction(*segment),
                landing.edge.line_id,
            ].append((plan_rank, landing_rank, landing))

    coordinates: dict[_OpeningKey, float] = {}
    for (
        _system_id,
        source_id,
        opening_direction,
        _line_id,
    ), members in groups.items():
        if len(members) < 2:
            continue
        source_x = graph.stations[source_id].x
        axis = direction_axis(opening_direction)

        def effective(item: tuple[int, int, ConvergenceLanding]) -> float:
            landing = item[2]
            assert landing.opening_turn_coordinate is not None
            return fixed.get(
                (landing.member_id, axis, opening_direction),
                landing.opening_turn_coordinate,
            )

        anchor = min(
            members,
            key=lambda item: (
                abs(effective(item) - source_x),
                item[0],
                item[1],
            ),
        )
        coordinate = effective(anchor)
        coordinates.update(
            {
                (landing.member_id, axis, opening_direction): coordinate
                for _plan_rank, _landing_rank, landing in members
            }
        )
    return MappingProxyType(coordinates)


def _landing_trunk_flank_conflict(
    plans: tuple[ConvergencePlan, ...], graph: MetroGraph, curve_radius: float
) -> bool:
    return any(
        landing_plan.id != trunk_plan.id
        and landing.edge.line_id in trunk_plan.line_ids
        and _direction(*landing_segment) is not _direction(*flank)
        and _parallel_segments_conflict(landing_segment, flank, curve_radius)
        for landing_plan in plans
        for landing in landing_plan.landings
        if (landing_segment := _landing_cross_segment(landing, graph)) is not None
        for trunk_plan in plans
        if trunk_plan.trunk_axis is not None
        for rank, flank in enumerate(_trunk_segments(trunk_plan.trunk_axis))
        if rank in {1, 3}
    )


def _system_allocation_conflict(
    plans: tuple[ConvergencePlan, ...],
    system_edges: tuple[ResolvedEdge, ...],
    scaffold: RouteSemanticScaffold,
    ctx: _RoutingCtx,
) -> ConvergenceAllocationConflict | None:
    complete_pairwise_system = len(plans) == 2
    complete_isolated_system = len(plans) == 1
    opening_arms = tuple(
        (plan, landing, ctx.graph.stations[landing.source_junction_id])
        for plan in plans
        if plan.trunk_axis is not None
        for landing in plan.landings
        if landing.opening_turn_coordinate is not None
    )
    for rank, (first_plan, first, first_source) in enumerate(opening_arms):
        assert first_plan.trunk_axis is not None
        assert first.opening_turn_coordinate is not None
        for second_plan, second, second_source in opening_arms[rank + 1 :]:
            assert second_plan.trunk_axis is not None
            assert second.opening_turn_coordinate is not None
            if (
                first_plan.id == second_plan.id
                or first.edge.line_id != second.edge.line_id
                or first.source_junction_id != second.source_junction_id
                or first_plan.trunk_axis.axis is not second_plan.trunk_axis.axis
                or abs(first.opening_turn_coordinate - second.opening_turn_coordinate)
                > COORD_TOLERANCE
            ):
                continue
            if first_plan.trunk_axis.axis is DemandAxis.X:
                first_delta = first_plan.trunk_axis.coordinate - first_source.y
                second_delta = second_plan.trunk_axis.coordinate - second_source.y
            else:
                first_delta = first_plan.trunk_axis.coordinate - first_source.x
                second_delta = second_plan.trunk_axis.coordinate - second_source.x
            if first_delta * second_delta < 0:
                return ConvergenceAllocationConflict(
                    ConvergenceAllocationNeed.OPPOSING_OPENINGS,
                    "planned fan arms require opposing opening channels",
                    (first.member_id, second.member_id),
                )
            if (
                first_plan.line_ids == second_plan.line_ids
                and abs(
                    first_plan.trunk_axis.coordinate - second_plan.trunk_axis.coordinate
                )
                > ctx.offset_step + COORD_TOLERANCE
            ):
                return ConvergenceAllocationConflict(
                    ConvergenceAllocationNeed.CHAINED_SAME_LINE,
                    "chained same-line convergences require one shared system "
                    "settlement",
                    (first.member_id, second.member_id),
                )

    if opposing_members := _opposing_landing_approach_members(plans, ctx.graph):
        return ConvergenceAllocationConflict(
            ConvergenceAllocationNeed.OPPOSING_OPENINGS,
            "planned convergence feeder approaches require one shared channel decision",
            opposing_members,
        )

    trunks = tuple(
        (
            plan,
            segment,
            rank == 0,
            plan.trunk_axis.direction if rank == 0 else _direction(*segment),
        )
        for plan in plans
        if plan.trunk_axis is not None
        for rank, segment in enumerate(_trunk_segments(plan.trunk_axis))
        if rank in {0, 1, 3}
    )
    primary_source = {
        plan.id: next(
            ownership.edge.source
            for ownership in plan.endpoint_ownership
            if ownership.member_id == plan.primary_trunk_member_id
        )
        for plan in plans
        if plan.owns_geometry
    }
    for rank, (first_plan, first_segment, first_central, first_direction) in enumerate(
        trunks
    ):
        for second_plan, second_segment, second_central, second_direction in trunks[
            rank + 1 :
        ]:
            if first_plan.id == second_plan.id:
                continue
            if not _parallel_segments_conflict(
                first_segment, second_segment, ctx.curve_radius
            ):
                continue
            same_line = first_plan.line_ids == second_plan.line_ids
            if same_line and first_direction is not second_direction:
                assert first_plan.primary_trunk_member_id is not None
                assert second_plan.primary_trunk_member_id is not None
                return ConvergenceAllocationConflict(
                    ConvergenceAllocationNeed.SHARED_CHANNEL,
                    "planned convergence trunks require one shared channel decision",
                    (
                        first_plan.primary_trunk_member_id,
                        second_plan.primary_trunk_member_id,
                    ),
                )
            if (
                not first_central
                and not second_central
                and first_direction is second_direction
                and first_plan.entry_group_ids != second_plan.entry_group_ids
                and primary_source[first_plan.id] == primary_source[second_plan.id]
                and complete_pairwise_system
            ):
                first_horizontal = (
                    abs(first_segment[0][1] - first_segment[1][1]) <= COORD_TOLERANCE
                )
                separation = abs(
                    first_segment[0][1] - second_segment[0][1]
                    if first_horizontal
                    else first_segment[0][0] - second_segment[0][0]
                )
                if separation > COORD_TOLERANCE:
                    assert first_plan.primary_trunk_member_id is not None
                    assert second_plan.primary_trunk_member_id is not None
                    return ConvergenceAllocationConflict(
                        ConvergenceAllocationNeed.SHARED_CHANNEL,
                        "planned convergence trunks require one shared channel "
                        "decision",
                        (
                            first_plan.primary_trunk_member_id,
                            second_plan.primary_trunk_member_id,
                        ),
                    )

    if _landing_trunk_flank_conflict(plans, ctx.graph, ctx.curve_radius):
        return ConvergenceAllocationConflict(
            ConvergenceAllocationNeed.LOCAL_GEOMETRY,
            "planned convergence approaches and trunks have no settlement room",
        )

    owned_edges = {edge for plan in plans for edge in plan.resolved_member_edges}
    unowned_system_edges: list[ResolvedEdge] = []
    for edge_key in system_edges:
        if edge_key in owned_edges:
            continue
        unowned_system_edges.append(edge_key)
        candidate_trunks = tuple(
            (trunk_segment, planned_direction)
            for plan, trunk_segment, central, planned_direction in trunks
            if central
            and edge_key.line_id in plan.line_ids
            and edge_key.target in plan.target_entry_port_ids
        )
        if not candidate_trunks:
            continue
        edge = ctx.edge_by_key.get((edge_key.source, edge_key.target, edge_key.line_id))
        if edge is None:
            continue
        try:
            route = _trial_route(edge, ctx)
        except UnsupportedConvergenceError:
            continue
        _bake_route(route, ctx)
        for trunk_segment, planned_direction in candidate_trunks:
            for route_segment in _route_segments(route):
                if planned_direction is _direction(
                    *route_segment
                ) and _parallel_segments_conflict(
                    trunk_segment, route_segment, ctx.curve_radius
                ):
                    if _segments_overlap(trunk_segment, route_segment):
                        continue
                    return ConvergenceAllocationConflict(
                        ConvergenceAllocationNeed.UNOWNED_CORRIDOR,
                        "planned convergence corridor conflicts with unowned "
                        "route-system member",
                        _ordered_unique(
                            (
                                scaffold.member_id_by_edge[edge_key],
                                *(
                                    plan.primary_trunk_member_id
                                    for plan in plans
                                    if plan.primary_trunk_member_id is not None
                                    if edge_key.line_id in plan.line_ids
                                    and edge_key.target in plan.target_entry_port_ids
                                ),
                            )
                        ),
                    )
    if not complete_isolated_system:
        return None
    landing_sources = {
        landing.source_junction_id for plan in plans for landing in plan.landings
    }
    foreign_groups: defaultdict[
        tuple[str, tuple[EndpointGroupId, ...], tuple[EndpointGroupId, ...]],
        set[str],
    ] = defaultdict(set)
    for foreign_edge in unowned_system_edges:
        if foreign_edge.source not in landing_sources:
            continue
        connectors = tuple(
            scaffold.query.connector(ref.connector_id)
            for ref in scaffold.refs_by_edge[foreign_edge]
        )
        foreign_groups[
            (
                foreign_edge.source,
                _ordered_unique(item.exit_group_id for item in connectors),
                _ordered_unique(item.entry_group_id for item in connectors),
            )
        ].add(foreign_edge.line_id)
    if any(len(line_ids) > 1 for line_ids in foreign_groups.values()):
        return ConvergenceAllocationConflict(
            ConvergenceAllocationNeed.UNOWNED_CORRIDOR,
            "planned convergence corridor conflicts with unowned route-system members",
            _ordered_unique(
                scaffold.member_id_by_edge[item] for item in unowned_system_edges
            ),
        )
    return None


def _capacity_proofs_for_conflict(
    system_id: RouteSystemId,
    conflict: ConvergenceAllocationConflict,
    proofs: tuple[EnvelopeCapacityProof, ...],
) -> tuple[EnvelopeCapacityProof, ...]:
    if (
        conflict.need not in _ENVELOPE_ALLOCATION_NEEDS
        or not conflict.member_ids
        or not proofs
    ):
        return ()
    conflict_members = set(conflict.member_ids)
    return tuple(
        proof
        for proof in proofs
        if proof.available_width + COORD_TOLERANCE >= proof.required_width
        and system_id in proof.system_ids
        and conflict_members.issubset(proof.claimant_member_ids)
    )


def _allocated_opening_coordinates(
    plans: tuple[ConvergencePlan, ...],
    proofs: tuple[EnvelopeCapacityProof, ...],
) -> Mapping[_OpeningKey, float]:
    system_by_member = {
        member_id: plan.system_id for plan in plans for member_id in plan.member_ids
    }
    coordinates: dict[_OpeningKey, float] = {}
    for plan in plans:
        for landing in plan.landings:
            if landing.opening_turn_coordinate is None:
                continue
            assert landing.opening_turn_segment is not None
            opening_direction = _direction(*landing.opening_turn_segment)
            opening_axis = direction_axis(opening_direction)
            allocation_axis = (
                DemandAxis.X if opening_axis is DemandAxis.Y else DemandAxis.Y
            )
            direct = tuple(
                allocation
                for proof in proofs
                for reservation in proof.reservations
                if reservation.system_id == system_by_member[landing.member_id]
                and reservation.direction is opening_direction
                for allocation in reservation.allocations
                if allocation.member_id == landing.member_id
                and allocation.axis is allocation_axis
                and (
                    abs(
                        allocation.original_coordinate - landing.opening_turn_coordinate
                    )
                    <= COORD_TOLERANCE
                    or abs(allocation.coordinate - landing.opening_turn_coordinate)
                    <= COORD_TOLERANCE
                )
            )
            candidates = direct or tuple(
                allocation
                for proof in proofs
                for reservation in proof.reservations
                if reservation.system_id == system_by_member[landing.member_id]
                and reservation.direction is opening_direction
                for allocation in reservation.allocations
                if allocation.axis is allocation_axis
                if allocation.member_id in plan.member_ids
                if (
                    abs(
                        allocation.original_coordinate - landing.opening_turn_coordinate
                    )
                    <= COORD_TOLERANCE
                    or abs(allocation.coordinate - landing.opening_turn_coordinate)
                    <= COORD_TOLERANCE
                )
                and next(
                    edge.line_id
                    for member_id, edge in zip(
                        plan.member_ids, plan.resolved_member_edges, strict=True
                    )
                    if member_id == allocation.member_id
                )
                == landing.edge.line_id
            )
            coordinates_for_opening = tuple(
                dict.fromkeys(item.coordinate for item in candidates)
            )
            if len(coordinates_for_opening) == 1:
                coordinates[(landing.member_id, opening_axis, opening_direction)] = (
                    coordinates_for_opening[0]
                )
    return MappingProxyType(coordinates)


def _fork_pivot_references(
    graph: MetroGraph, plans: tuple[ConvergencePlan, ...]
) -> tuple[SharedReference, ...]:
    groups: defaultdict[
        tuple[RouteSystemId, str, Direction, str],
        list[tuple[ConvergencePlan, ConvergenceLanding]],
    ] = defaultdict(list)
    for plan in plans:
        if not plan.owns_geometry:
            continue
        for landing in plan.landings:
            if landing.opening_turn_segment is None:
                continue
            groups[
                plan.system_id,
                landing.source_junction_id,
                _direction(*landing.opening_turn_segment),
                landing.edge.line_id,
            ].append((plan, landing))

    assert graph.route_topology is not None
    provenance = _plan_provenance(graph, graph.route_topology.connectors)
    references: list[SharedReference] = []
    for (system_id, source_id, opening_direction, line_id), members in groups.items():
        if len(members) < 2:
            continue
        owner_plans = _ordered_unique(plan.id for plan, _landing in members)
        plans_by_id = {plan.id: plan for plan, _landing in members}
        connector_ids = _ordered_unique(
            connector_id
            for plan_id in owner_plans
            for connector_id in plans_by_id[plan_id].connector_ids
        )
        spans = tuple(_plan_span(graph, plans_by_id[item]) for item in owner_plans)
        span = GridSpan(
            min(item.min_column for item in spans),
            max(item.max_column for item in spans),
            min(item.min_row for item in spans),
            max(item.max_row for item in spans),
        )
        references.append(
            SharedReference(
                SharedReferenceId(
                    semantic_route_id(
                        "shared-reference",
                        system_id,
                        "fork-pivot",
                        source_id,
                        opening_direction.value,
                        line_id,
                    )
                ),
                system_id,
                SharedReferenceKind.FORK_PIVOT,
                tuple(landing.member_id for _plan, landing in members),
                CoordinateRegime.LAYOUT_CANVAS,
                reservation_decision_refs(provenance, connector_ids, span),
            )
        )
    return tuple(references)


def _resources(
    graph: MetroGraph,
    plans: tuple[ConvergencePlan, ...],
) -> tuple[tuple[SharedReference, ...], tuple[SymbolicDemand, ...]]:
    assert graph.route_topology is not None
    provenance = _plan_provenance(graph, graph.route_topology.connectors)
    references: list[SharedReference] = []
    demands: list[SymbolicDemand] = []
    for plan in plans:
        if not plan.owns_geometry:
            continue
        assert plan.trunk_axis is not None
        span = _plan_span(graph, plan)
        decision_refs = reservation_decision_refs(provenance, plan.connector_ids, span)
        for reference_id, kind in zip(
            plan.shared_reference_ids,
            (SharedReferenceKind.TRUNK, SharedReferenceKind.LANDING_SEQUENCE),
            strict=True,
        ):
            references.append(
                SharedReference(
                    reference_id,
                    plan.system_id,
                    kind,
                    plan.member_ids,
                    CoordinateRegime.LAYOUT_CANVAS,
                    decision_refs,
                )
            )
        demands.extend(
            (
                SymbolicDemand(
                    plan.demand_ids[0],
                    plan.system_id,
                    plan.member_ids,
                    DemandKind.LANES,
                    DemandAxis.Y
                    if plan.trunk_axis.axis is DemandAxis.X
                    else DemandAxis.X,
                    span,
                    len(plan.lane_order),
                    None,
                    None,
                    (plan.shared_reference_ids[0],),
                    (KeepOutClass.SECTION, KeepOutClass.MARKER),
                    decision_refs,
                ),
                SymbolicDemand(
                    plan.demand_ids[1],
                    plan.system_id,
                    tuple(item.member_id for item in plan.landings),
                    DemandKind.RUNWAY,
                    plan.trunk_axis.axis,
                    span,
                    len(plan.landings),
                    max(item.minimum_runway for item in plan.landings),
                    CoordinateRegime.LAYOUT_CANVAS,
                    plan.shared_reference_ids,
                    (KeepOutClass.SECTION, KeepOutClass.MARKER),
                    decision_refs,
                ),
            )
        )
    references.extend(_fork_pivot_references(graph, plans))
    return tuple(references), tuple(demands)


def _query(
    plans: tuple[ConvergencePlan, ...],
    proofs: tuple[EnvelopeCapacityProof, ...] = (),
    ctx: _RoutingCtx | None = None,
) -> ConvergencePlanExecutionQuery:
    by_edge: dict[ResolvedEdge, ConvergenceRouteMembership] = {}
    for plan in plans:
        if not plan.owns_geometry:
            continue
        landings = {item.member_id: item for item in plan.landings}
        continuations = {item.member_id: item for item in plan.outgoing_continuations}
        ownership_by_member = {item.member_id: item for item in plan.endpoint_ownership}
        for ownership in plan.endpoint_ownership:
            covering_ownership = (
                ownership_by_member[ownership.covered_by_member_id]
                if ownership.covered_by_member_id is not None
                else None
            )
            allocations = (
                ctx.envelope_allocations.allocations_for_member(ownership.member_id)
                if ctx is not None and ctx.envelope_allocations is not None
                else _claim_allocations(ownership.member_id, proofs)
            )
            membership = ConvergenceRouteMembership(
                plan,
                ownership.member_id,
                landings.get(ownership.member_id),
                continuations.get(ownership.member_id),
                ownership,
                covering_ownership.edge if covering_ownership is not None else None,
                allocations,
            )
            if ownership.edge in by_edge:
                raise ValueError("planned convergence edge has more than one owner")
            by_edge[ownership.edge] = membership
    return ConvergencePlanExecutionQuery(
        plans,
        MappingProxyType(by_edge),
        ctx.envelope_allocations if ctx is not None else None,
    )


def _claim_continuation_emissions(
    plans: tuple[ConvergencePlan, ...], ctx: _RoutingCtx
) -> None:
    ctx.skip_edges.difference_update(
        (item.edge.source, item.edge.target, item.edge.line_id)
        for plan in plans
        if plan.owns_geometry
        for item in plan.outgoing_continuations
        if item.covered_by_member_id is None
    )


def _restore_immutable_merge_skips(
    scaffold: RouteSemanticScaffold,
    ctx: _RoutingCtx,
) -> None:
    """Restore covered continuation membership before final planning."""
    allocations = ctx.envelope_allocations
    if allocations is None:
        return
    ctx.skip_edges.update(
        (edge.source, edge.target, edge.line_id)
        for edge, member_id in scaffold.member_id_by_edge.items()
        if (
            (binding := allocations.immutable_binding_for(member_id)) is not None
            and binding.kind is BindingKind.MERGE_SKIP
        )
    )


def build_convergence_plan_execution(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    *,
    exit_turn_plans: tuple[ExitTurnPlan, ...],
    fan_plans: tuple[FanPlan, ...],
    include_resources: bool = True,
    envelope_proofs: tuple[EnvelopeCapacityProof, ...] = (),
    envelope_limitations: tuple[EnvelopeCapacityLimitation, ...] = (),
) -> ConvergencePlanExecution:
    """Plan every semantic convergence atomically by route system."""
    _restore_immutable_merge_skips(scaffold, ctx)
    views_by_system: dict[RouteSystemId, list[ResolvedConvergenceView]] = defaultdict(
        list
    )
    for view in scaffold.query.convergences:
        views_by_system[scaffold.system_for(view.group.connector_ids)].append(view)
    edges_by_system: dict[RouteSystemId, list[ResolvedEdge]] = defaultdict(list)
    for edge in scaffold.edge_order:
        edge_connector_ids = _ordered_unique(
            item.connector_id for item in scaffold.refs_by_edge[edge]
        )
        edges_by_system[scaffold.system_for(edge_connector_ids)].append(edge)
    exit_turn_plans_by_system: dict[RouteSystemId, list[ExitTurnPlan]] = defaultdict(
        list
    )
    for exit_turn_plan in exit_turn_plans:
        exit_turn_plans_by_system[exit_turn_plan.system_id].append(exit_turn_plan)
    fan_plans_by_system: dict[RouteSystemId, list[FanPlan]] = defaultdict(list)
    for fan_plan in fan_plans:
        if fan_plan.system_id is not None:
            fan_plans_by_system[fan_plan.system_id].append(fan_plan)
    proofs_by_system: defaultdict[RouteSystemId, list[EnvelopeCapacityProof]] = (
        defaultdict(list)
    )
    for proof in envelope_proofs:
        for system_id in proof.system_ids:
            proofs_by_system[system_id].append(proof)
    limitations_by_system = {
        limitation.system_id: limitation for limitation in envelope_limitations
    }
    plans: list[ConvergencePlan] = []
    consumed_proofs: list[EnvelopeCapacityProof] = []
    diagnostics: list[RoutePlanDiagnostic] = []
    for system_id in scaffold.ordered_system_ids:
        views = views_by_system.get(system_id, [])
        if not views:
            continue
        connector_ids = set(
            connector_id for view in views for connector_id in view.group.connector_ids
        )
        memberships = tuple(_plan_membership(scaffold, view) for view in views)
        member_ids = {
            member_id
            for _paths, _edges, members in memberships
            for member_id in members
        }
        upstream_exit_plans = tuple(
            item
            for item in exit_turn_plans_by_system.get(system_id, ())
            if set(item.connector_ids) & connector_ids
            or set(item.member_ids) & member_ids
        )
        upstream_exit_ids = tuple(item.id for item in upstream_exit_plans)
        upstream_fan_ids = tuple(
            item.id
            for item in fan_plans_by_system.get(system_id, ())
            if (
                set(item.connector_ids) & connector_ids
                or set(item.member_ids) & member_ids
            )
        )
        unresolved_fan_ownership = any(
            item.id in upstream_fan_ids
            and item.disposition is FanPlanDisposition.LEGACY
            and item.legacy_reason == "overlapping-fan-ownership"
            for item in fan_plans_by_system.get(system_id, ())
        )
        try:
            system_proofs = tuple(proofs_by_system.get(system_id, ()))
            provisional_plans = tuple(
                _build_planned_convergence(
                    graph,
                    ctx,
                    scaffold,
                    view,
                    membership,
                    upstream_exit_ids,
                    upstream_fan_ids,
                    envelope_proofs=system_proofs,
                )
                for view, membership in zip(views, memberships, strict=True)
            )
            allocated_openings = _allocated_opening_coordinates(
                provisional_plans,
                system_proofs,
            )
            fork_openings = _divergence_opening_coordinates(
                provisional_plans,
                graph,
                allocated_openings,
            )
            opening_coordinates = MappingProxyType(
                {**allocated_openings, **fork_openings}
            )
            system_plans = (
                tuple(
                    _build_planned_convergence(
                        graph,
                        ctx,
                        scaffold,
                        view,
                        membership,
                        upstream_exit_ids,
                        upstream_fan_ids,
                        opening_coordinates,
                        system_proofs,
                    )
                    for view, membership in zip(views, memberships, strict=True)
                )
                if opening_coordinates or system_proofs
                else provisional_plans
            )
            system_plans = _plan_perp_entry_terminal_families(system_plans, graph, ctx)
            system_plans = _settle_landing_trunk_flanks(
                system_plans,
                graph,
                ctx.curve_radius,
                opening_coordinates,
            )
            allocation_conflict = _system_allocation_conflict(
                system_plans,
                tuple(edges_by_system.get(system_id, ())),
                scaffold,
                ctx,
            )
            capacity_limitation = limitations_by_system.get(system_id)
            if capacity_limitation is not None:
                pinned = ", ".join(capacity_limitation.pinned_section_ids) or "none"
                reservations = ", ".join(
                    str(item) for item in capacity_limitation.reservation_ids
                )
                allocation_conflict = ConvergenceAllocationConflict(
                    ConvergenceAllocationNeed.LOCAL_GEOMETRY,
                    "settled route envelope remains infeasible under authored "
                    f"grid commitments (pins {pinned}; reservations {reservations}; "
                    f"owner #{capacity_limitation.owner_issue})",
                )
            depends_on_fan_consolidation = (
                allocation_conflict is not None
                and allocation_conflict.need
                is ConvergenceAllocationNeed.UNOWNED_CORRIDOR
                and unresolved_fan_ownership
                and any(
                    ctx.merge.entry_port_for.get(continuation.edge.source)
                    == continuation.edge.target
                    and (
                        continuation.edge.source,
                        continuation.edge.target,
                        continuation.edge.line_id,
                    )
                    not in ctx.skip_edges
                    for plan in system_plans
                    for continuation in plan.outgoing_continuations
                )
            )
            if depends_on_fan_consolidation:
                allocation_conflict = ConvergenceAllocationConflict(
                    ConvergenceAllocationNeed.LOCAL_GEOMETRY,
                    "planned convergence corridor depends on unresolved "
                    "overlapping fan ownership (owner #1658)",
                )
            allocation_proofs = (
                _capacity_proofs_for_conflict(
                    system_id, allocation_conflict, system_proofs
                )
                if allocation_conflict is not None
                else system_proofs
                if opening_coordinates
                else ()
            )
            if allocation_conflict is not None and allocation_proofs:
                allocation_conflict = _system_allocation_conflict(
                    system_plans,
                    tuple(edges_by_system.get(system_id, ())),
                    scaffold,
                    ctx,
                )
                if allocation_conflict is not None and _capacity_proofs_for_conflict(
                    system_id,
                    allocation_conflict,
                    allocation_proofs,
                ):
                    allocation_conflict = None
            if allocation_conflict is not None:
                raise UnsupportedConvergenceError(allocation_conflict.message)
            consumed_proofs.extend(allocation_proofs)
        except UnsupportedConvergenceError as error:
            reason = str(error) or type(error).__name__
            system_plans = tuple(
                _legacy_plan(scaffold, view, membership, reason)
                for view, membership in zip(views, memberships, strict=True)
            )
            for item in system_plans:
                diagnostics.append(
                    RoutePlanDiagnostic(
                        None,
                        "convergence-plan-legacy",
                        f"convergence system {item.system_id} uses legacy routing: "
                        f"{reason}",
                        blocking=False,
                    )
                )
        plans.extend(system_plans)
    frozen_plans = tuple(plans)
    _claim_continuation_emissions(frozen_plans, ctx)
    references, demands = (
        _resources(graph, frozen_plans) if include_resources else ((), ())
    )
    return ConvergencePlanExecution(
        frozen_plans,
        references,
        demands,
        tuple(diagnostics),
        _query(frozen_plans, tuple(consumed_proofs), ctx),
    )


def convergence_failure(
    membership: ConvergenceRouteMembership,
    emitted_endpoint: tuple[float, float],
) -> str:
    plan = membership.plan
    connectors = ", ".join(str(item) for item in membership.ownership.connector_ids)
    expected = membership.ownership.endpoint
    return (
        f"convergence system {plan.system_id} connectors {connectors} member "
        f"{membership.member_id} planned join {expected} emitted endpoint "
        f"{emitted_endpoint}"
    )


def _point_on_trunk_geometry(
    point: tuple[float, float], axis: ConvergenceTrunkAxis
) -> bool:
    return any(
        point_to_polyline_distance(point, segment) <= COORD_TOLERANCE
        for segment in _trunk_segments(axis)
    )


def _unbound_target_arm_continuations(
    continuations: tuple[ConvergenceContinuation, ...],
    primary_member_id: EmissionMemberId,
    axis: ConvergenceTrunkAxis,
    ctx: _RoutingCtx,
) -> set[EmissionMemberId]:
    target_point = _axis_target_point(axis)
    target_arm = _trunk_segments(axis)[4]
    return {
        item.member_id
        for item in continuations
        if item.member_id != primary_member_id
        and item.covered_by_member_id is None
        and (
            ctx.envelope_allocations is None
            or ctx.envelope_allocations.immutable_binding_for(item.member_id) is None
        )
        and all(
            abs(actual - expected) <= COORD_TOLERANCE
            for actual, expected in zip(item.start_point, target_point, strict=True)
        )
        and point_to_polyline_distance(item.end_point, target_arm) <= COORD_TOLERANCE
    }


def _cover_target_arm_continuation(
    continuations: tuple[ConvergenceContinuation, ...],
    target_arm_continuation_ids: set[EmissionMemberId],
    primary_member_id: EmissionMemberId,
    axis: ConvergenceTrunkAxis,
    ownership: list[ConvergenceEndpointOwnership],
) -> tuple[ConvergenceTrunkAxis, list[ConvergenceEndpointOwnership]]:
    target_arm_continuations = tuple(
        item
        for item in continuations
        if item.member_id in target_arm_continuation_ids
        and item.covered_by_member_id == primary_member_id
    )
    if not target_arm_continuations:
        return axis, ownership
    if len(target_arm_continuations) != 1:
        raise UnsupportedConvergenceError(
            "planned trunk has conflicting target-arm continuations"
        )
    target_endpoint = target_arm_continuations[0].end_point
    axis = replace(
        axis,
        target_endpoint_coordinate=(
            target_endpoint[0] if axis.axis is DemandAxis.X else target_endpoint[1]
        ),
    )
    ownership = [
        replace(item, endpoint=target_endpoint)
        if item.member_id == primary_member_id
        else item
        for item in ownership
    ]
    return axis, ownership


def _route_covers_segment(
    route: RoutedPath,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> bool:
    if segment_start == segment_end:
        return (
            point_to_polyline_distance(segment_start, route.points) <= COORD_TOLERANCE
        )
    horizontal = abs(segment_start[1] - segment_end[1]) <= COORD_TOLERANCE
    vertical = abs(segment_start[0] - segment_end[0]) <= COORD_TOLERANCE
    if not horizontal and not vertical:
        return False
    coordinate = segment_start[1] if horizontal else segment_start[0]
    extent_start, extent_end = sorted(
        (segment_start[0], segment_end[0])
        if horizontal
        else (segment_start[1], segment_end[1])
    )
    intervals: list[tuple[float, float]] = []
    for start, end in zip(route.points, route.points[1:]):
        if horizontal:
            if (
                abs(start[1] - coordinate) > COORD_TOLERANCE
                or abs(end[1] - coordinate) > COORD_TOLERANCE
            ):
                continue
            interval = (min(start[0], end[0]), max(start[0], end[0]))
        else:
            if (
                abs(start[0] - coordinate) > COORD_TOLERANCE
                or abs(end[0] - coordinate) > COORD_TOLERANCE
            ):
                continue
            interval = (min(start[1], end[1]), max(start[1], end[1]))
        if interval[1] - interval[0] > COORD_TOLERANCE:
            intervals.append(interval)
    covered_until = extent_start
    for interval_start, interval_end in sorted(intervals):
        if interval_end < covered_until - COORD_TOLERANCE:
            continue
        if interval_start > covered_until + COORD_TOLERANCE:
            return False
        covered_until = max(covered_until, interval_end)
        if covered_until >= extent_end - COORD_TOLERANCE:
            return True
    return False


def _trunk_segments(
    axis: ConvergenceTrunkAxis,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    source_longitudinal, target_longitudinal = (
        (axis.extent_start, axis.extent_end)
        if axis.direction in {Direction.R, Direction.D}
        else (axis.extent_end, axis.extent_start)
    )
    source_endpoint = (
        axis.source_endpoint_coordinate
        if axis.source_endpoint_coordinate is not None
        else source_longitudinal
    )
    target_endpoint = (
        axis.target_endpoint_coordinate
        if axis.target_endpoint_coordinate is not None
        else target_longitudinal
    )
    if axis.axis is DemandAxis.X:
        return (
            (
                (axis.extent_start, axis.coordinate),
                (axis.extent_end, axis.coordinate),
            ),
            (
                (source_longitudinal, axis.coordinate),
                (source_longitudinal, axis.source_flank_coordinate),
            ),
            (
                (source_longitudinal, axis.source_flank_coordinate),
                (source_endpoint, axis.source_flank_coordinate),
            ),
            (
                (target_longitudinal, axis.coordinate),
                (target_longitudinal, axis.target_flank_coordinate),
            ),
            (
                (target_longitudinal, axis.target_flank_coordinate),
                (target_endpoint, axis.target_flank_coordinate),
            ),
        )
    return (
        (
            (axis.coordinate, axis.extent_start),
            (axis.coordinate, axis.extent_end),
        ),
        (
            (axis.coordinate, source_longitudinal),
            (axis.source_flank_coordinate, source_longitudinal),
        ),
        (
            (axis.source_flank_coordinate, source_longitudinal),
            (axis.source_flank_coordinate, source_endpoint),
        ),
        (
            (axis.coordinate, target_longitudinal),
            (axis.target_flank_coordinate, target_longitudinal),
        ),
        (
            (axis.target_flank_coordinate, target_longitudinal),
            (axis.target_flank_coordinate, target_endpoint),
        ),
    )


def _route_covers_trunk(route: RoutedPath, axis: ConvergenceTrunkAxis) -> bool:
    return all(
        _route_covers_segment(route, start, end) for start, end in _trunk_segments(axis)
    )


def _segments_overlap(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    (a, b), (c, d) = first, second
    first_horizontal = abs(a[1] - b[1]) <= COORD_TOLERANCE
    second_horizontal = abs(c[1] - d[1]) <= COORD_TOLERANCE
    if first_horizontal != second_horizontal:
        return False
    if first_horizontal:
        return (
            abs(a[1] - c[1]) <= COORD_TOLERANCE
            and min(a[0], b[0]) <= max(c[0], d[0]) + COORD_TOLERANCE
            and min(c[0], d[0]) <= max(a[0], b[0]) + COORD_TOLERANCE
        )
    return (
        abs(a[0] - c[0]) <= COORD_TOLERANCE
        and min(a[1], b[1]) <= max(c[1], d[1]) + COORD_TOLERANCE
        and min(c[1], d[1]) <= max(a[1], b[1]) + COORD_TOLERANCE
    )


def _trunk_segment_ranks(
    route: RoutedPath, axis: ConvergenceTrunkAxis
) -> tuple[int, ...]:
    planned = _trunk_segments(axis)
    return tuple(
        rank
        for rank, segment in enumerate(zip(route.points, route.points[1:]))
        if any(_segments_overlap(segment, item) for item in planned)
    )


def _seat_route_on_trunk_flanks(
    route: RoutedPath,
    axis: ConvergenceTrunkAxis,
    graph: MetroGraph,
    lane_offset: float,
    *,
    shared_terminal_reference_radius: float | None = None,
) -> None:
    central = _trunk_segments(axis)[0]
    central_horizontal = axis.axis is DemandAxis.X
    central_span = sorted(
        (
            central[0][0] if central_horizontal else central[0][1],
            central[1][0] if central_horizontal else central[1][1],
        )
    )
    central_candidates = []
    for rank, (start, end) in enumerate(zip(route.points, route.points[1:])):
        horizontal = abs(start[1] - end[1]) <= COORD_TOLERANCE
        vertical = abs(start[0] - end[0]) <= COORD_TOLERANCE
        if central_horizontal != horizontal or not (horizontal or vertical):
            continue
        span = sorted(
            (
                start[0] if horizontal else start[1],
                end[0] if horizontal else end[1],
            )
        )
        if all(
            abs(actual - expected) <= COORD_TOLERANCE
            for actual, expected in zip(span, central_span, strict=True)
        ):
            central_candidates.append(rank)
    if len(central_candidates) > 1:
        raise ConvergenceInvariantError(
            f"planned trunk axis {central} is not unique in member {route.edge!r}"
        )
    if central_candidates and central_horizontal:
        central_rank = central_candidates[0]
        from nf_metro.layout.routing.normalize import _set_htrunk_y

        _set_htrunk_y(route, central_rank, axis.coordinate)
    elif central_candidates:
        central_rank = central_candidates[0]
        from nf_metro.layout.routing.normalize import (
            _reconcile_moved_gap_slot,
            _set_vchannel_x,
            _VChannel,
        )

        start, end = route.points[central_rank : central_rank + 2]
        channel = _VChannel(
            route=route,
            idx=central_rank,
            x=start[0],
            y_lo=min(start[1], end[1]),
            y_hi=max(start[1], end[1]),
            down=end[1] > start[1],
        )
        _reconcile_moved_gap_slot(channel, axis.coordinate, graph)
        _set_vchannel_x(channel, axis.coordinate)

    planned_flanks = (_trunk_segments(axis)[1], _trunk_segments(axis)[3])
    source_offset = (
        lane_offset if axis.direction in {Direction.R, Direction.D} else -lane_offset
    )
    for flank_rank, planned in enumerate(planned_flanks):
        if all(
            abs(actual - expected) <= COORD_TOLERANCE
            for actual, expected in zip(*planned, strict=True)
        ):
            continue
        planned_horizontal = abs(planned[0][1] - planned[1][1]) <= COORD_TOLERANCE
        planned_span = sorted(
            (
                planned[0][0] if planned_horizontal else planned[0][1],
                planned[1][0] if planned_horizontal else planned[1][1],
            )
        )
        planned_coordinate = planned[0][1] if planned_horizontal else planned[0][0]
        candidates: list[tuple[float, int]] = []
        for rank, (start, end) in enumerate(zip(route.points, route.points[1:])):
            horizontal = abs(start[1] - end[1]) <= COORD_TOLERANCE
            vertical = abs(start[0] - end[0]) <= COORD_TOLERANCE
            if planned_horizontal != horizontal or not (horizontal or vertical):
                continue
            span = sorted(
                (
                    start[0] if horizontal else start[1],
                    end[0] if horizontal else end[1],
                )
            )
            if any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual, expected in zip(span, planned_span, strict=True)
            ):
                continue
            coordinate = start[1] if horizontal else start[0]
            candidates.append((abs(coordinate - planned_coordinate), rank))
        if not candidates:
            raise ConvergenceInvariantError(
                f"planned trunk flank {planned} is absent from member {route.edge!r}"
            )
        _distance, rank = min(candidates)
        start, end = route.points[rank : rank + 2]
        offset_in, offset_out = (
            (source_offset, 0.0) if flank_rank == 0 else (0.0, -source_offset)
        )
        if planned_horizontal:
            from nf_metro.layout.routing.normalize import _set_htrunk_y

            _set_htrunk_y(
                route,
                rank,
                planned_coordinate,
                offset_in=offset_in,
                offset_out=offset_out,
            )
        else:
            from nf_metro.layout.routing.normalize import (
                _reconcile_moved_gap_slot,
                _set_vchannel_x,
                _VChannel,
            )

            channel = _VChannel(
                route=route,
                idx=rank,
                x=start[0],
                y_lo=min(start[1], end[1]),
                y_hi=max(start[1], end[1]),
                down=end[1] > start[1],
            )
            _reconcile_moved_gap_slot(channel, planned_coordinate, graph)
            _set_vchannel_x(
                channel,
                planned_coordinate,
                offset_in,
                offset_out=offset_out,
            )

    source_arm, target_arm = _trunk_segments(axis)[2::2]
    for endpoint_rank, neighbour_rank, arm in (
        (0, 1, source_arm),
        (-1, -2, target_arm),
    ):
        if all(
            abs(actual - expected) <= COORD_TOLERANCE
            for actual, expected in zip(*arm, strict=True)
        ):
            continue
        endpoint = arm[1]
        neighbour = route.points[neighbour_rank]
        horizontal = abs(arm[0][1] - arm[1][1]) <= COORD_TOLERANCE
        if (
            horizontal
            and abs(neighbour[1] - endpoint[1]) > COORD_TOLERANCE
            or not horizontal
            and abs(neighbour[0] - endpoint[0]) > COORD_TOLERANCE
        ):
            raise ConvergenceInvariantError(
                f"planned trunk endpoint arm {arm} is absent from member {route.edge!r}"
            )
        route.points[endpoint_rank] = endpoint

    if shared_terminal_reference_radius is None or not central_candidates:
        return
    central_rank = central_candidates[0]
    adjacent_ranks = tuple(
        rank
        for rank in (central_rank - 1, central_rank + 1)
        if 0 <= rank < len(route.points) - 1
    )
    for rank in adjacent_ranks:
        start, end = route.points[rank : rank + 2]
        horizontal = abs(start[1] - end[1]) <= COORD_TOLERANCE
        if horizontal == central_horizontal:
            continue
        if horizontal:
            from nf_metro.layout.routing.normalize import _set_htrunk_y

            _set_htrunk_y(
                route,
                rank,
                start[1],
                offset_in=source_offset,
                offset_out=source_offset,
                base_radius=shared_terminal_reference_radius,
                base_radius_out=shared_terminal_reference_radius,
            )
        else:
            from nf_metro.layout.routing.normalize import _set_vchannel_x, _VChannel

            channel = _VChannel(
                route=route,
                idx=rank,
                x=start[0],
                y_lo=min(start[1], end[1]),
                y_hi=max(start[1], end[1]),
                down=end[1] > start[1],
            )
            _set_vchannel_x(
                channel,
                start[0],
                source_offset,
                offset_out=source_offset,
                base_radius=shared_terminal_reference_radius,
                base_radius_out=shared_terminal_reference_radius,
            )


def _extend_primary_trunk_to_axis(
    route: RoutedPath,
    axis: ConvergenceTrunkAxis,
) -> None:
    if axis.axis is not DemandAxis.X:
        return
    runs = tuple(iter_horizontal_trunks(route))
    if not runs:
        raise ConvergenceInvariantError("primary trunk emitted no shared run")
    rank, _run = min(runs, key=lambda item: abs(item[1].y - axis.coordinate))
    _extend_axis_segment_to_coordinates(
        route,
        rank,
        axis.axis,
        (axis.extent_start, axis.extent_end),
    )


def _assert_landing_geometry(
    route: RoutedPath,
    plan: ConvergencePlan,
    landing: ConvergenceLanding,
    query: ConvergencePlanExecutionQuery | None = None,
) -> None:
    join_point = (
        query.project_point(
            landing.member_id,
            len(route.points) - 1,
            landing.join_point,
        )
        if query is not None
        else landing.join_point
    )
    actual = _landing_approach(route, join_point)
    if actual is None:
        raise ConvergenceInvariantError(
            f"convergence system {plan.system_id} feeder {landing.member_id} "
            "has no emitted approach to its planned join"
        )
    direction, handedness, runway = actual
    if (
        direction is not landing.approach_direction
        or handedness is not landing.corner_handedness
        or runway < landing.minimum_runway - COORD_TOLERANCE
    ):
        raise ConvergenceInvariantError(
            f"convergence system {plan.system_id} feeder {landing.member_id} "
            f"planned {landing.approach_direction.value} approach, "
            f"{landing.corner_handedness} handedness, and "
            f"{landing.minimum_runway:g}px runway but emitted "
            f"{direction.value}, {handedness}, and {runway:g}px"
        )
    if landing.opening_turn_coordinate is not None:
        from nf_metro.layout.routing.normalize import _opening_fanout_descent

        opening = _opening_fanout_descent(route)
        emitted_segment = (
            (route.points[opening.idx], route.points[opening.idx + 1])
            if opening is not None
            else None
        )
        expected_segment = (
            query.project_segment(
                landing.member_id,
                opening.idx,
                landing.opening_turn_segment,
            )
            if query is not None
            and opening is not None
            and landing.opening_turn_segment is not None
            else landing.opening_turn_segment
            if opening is not None and landing.opening_turn_segment is not None
            else None
        )
        if (
            opening is None
            or expected_segment is None
            or emitted_segment is None
            or any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual_point, expected_point in zip(
                    emitted_segment, expected_segment, strict=True
                )
                for actual, expected in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise ConvergenceInvariantError(
                f"convergence system {plan.system_id} feeder {landing.member_id} "
                f"planned opening {expected_segment} but emitted "
                f"{emitted_segment}"
            )


def _seat_shared_terminal_source(
    route: RoutedPath, axis: ConvergenceTrunkAxis, ctx: _RoutingCtx
) -> None:
    """Project a provisional terminal approach onto its settled family source."""
    _bake_route(route, ctx)
    source = _axis_source_point(axis)
    if len(route.points) < 2:
        _connect_route_endpoint(route, source)
        return
    start = route.points[-2]
    end = route.points[-1]
    terminal_horizontal = abs(start[1] - end[1]) <= COORD_TOLERANCE
    terminal_vertical = abs(start[0] - end[0]) <= COORD_TOLERANCE
    if axis.axis is DemandAxis.Y and terminal_horizontal:
        route.points[-2] = (start[0], source[1])
        route.points[-1] = source
    elif axis.axis is DemandAxis.X and terminal_vertical:
        route.points[-2] = (source[0], start[1])
        route.points[-1] = source
    else:
        _connect_route_endpoint(route, source)
    _reset_convergence_curve_radii(route, ctx.curve_radius)


def _seat_shared_terminal_departure(
    route: RoutedPath,
    planned: tuple[tuple[float, float], tuple[float, float]],
    curve_radius: float,
) -> None:
    """Seat a family carrier on its same-line junction departure lane."""
    from nf_metro.layout.routing.normalize import _opening_fanout_descent

    opening = _opening_fanout_descent(route)
    if opening is None:
        raise ConvergenceInvariantError(
            "shared terminal family carrier has no opening descent"
        )
    start, end = planned
    route.points[opening.idx] = start
    if opening.idx > 0:
        prior = route.points[opening.idx - 1]
        route.points[opening.idx - 1] = (
            (prior[0], start[1])
            if abs(start[0] - end[0]) <= COORD_TOLERANCE
            else (start[0], prior[1])
        )
    _reset_convergence_curve_radii(
        route,
        curve_radius,
        preserve_existing=True,
    )


def consume_convergence_route(route: RoutedPath, ctx: _RoutingCtx) -> None:
    query = ctx.convergences
    if query is None:
        return
    membership = query.membership_for_edge(route.edge)
    if membership is None:
        return
    plan = membership.plan
    if not plan.owns_geometry:
        return
    _consume_envelope_allocation(route, ctx)
    route.convergence_plan_id = str(plan.id)
    route.convergence_member_id = str(membership.member_id)
    landing = membership.landing
    if landing is None:
        continuation = membership.continuation
        assert continuation is not None
        if continuation.covered_by_member_id is not None:
            immutable_binding = (
                ctx.envelope_allocations.immutable_binding_for(membership.member_id)
                if ctx.envelope_allocations is not None
                else None
            )
            if (
                immutable_binding is not None
                and immutable_binding.kind is BindingKind.COVERED_MERGE_HOP
            ):
                _consume_final_envelope_allocation(route, ctx)
                return
            raise ConvergenceInvariantError(
                "planned covered continuation reached convergence emission"
            )
        _consume_planned_continuation(route, continuation, ctx)
        if point_to_polyline_distance(
            continuation.start_point, route.points
        ) > COORD_TOLERANCE or any(
            abs(actual - expected) > COORD_TOLERANCE
            for actual, expected in zip(
                route.points[-1], continuation.end_point, strict=True
            )
        ):
            raise ConvergenceInvariantError(
                f"convergence system {plan.system_id} continuation member "
                f"{membership.member_id} differs from its planned endpoints"
            )
        if plan.primary_trunk_member_id == membership.member_id:
            assert plan.trunk_axis is not None
            _set_convergence_owned_segment_ranks(
                route,
                _trunk_segment_ranks(route, plan.trunk_axis),
                extend=True,
            )
        _consume_final_envelope_allocation(route, ctx)
        return
    opening_rank: int | None = None
    if landing.opening_turn_coordinate is not None:
        from nf_metro.layout.routing.normalize import (
            _opening_fanout_descent,
            _seat_merge_feeder_opening,
        )

        _seat_merge_feeder_opening(
            route,
            landing.opening_turn_coordinate,
            ctx.graph,
            planned=True,
        )
        if (
            plan.primary_trunk_reason is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
            and landing.opening_turn_segment is not None
        ):
            _seat_shared_terminal_departure(
                route,
                landing.opening_turn_segment,
                ctx.curve_radius,
            )
        opening = _opening_fanout_descent(route)
        if opening is None:
            raise ConvergenceInvariantError(
                f"convergence system {plan.system_id} feeder {landing.member_id} "
                "has no emitted opening turn"
            )
        opening_rank = opening.idx
    if plan.primary_trunk_member_id == membership.member_id:
        if plan.primary_trunk_reason is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH:
            assert plan.trunk_axis is not None
            _seat_shared_terminal_source(route, plan.trunk_axis, ctx)
            target = _axis_target_point(plan.trunk_axis)
            _connect_route_endpoint(route, target)
        assert plan.trunk_axis is not None
        if plan.primary_trunk_reason is ConvergenceTrunkReason.LONGEST_BYPASS:
            _extend_primary_trunk_to_axis(route, plan.trunk_axis)
        lane_rank = plan.lane_order.index(route.line_id)
        lane_offset = (len(plan.lane_order) - lane_rank - 1) * ctx.offset_step
        if (
            plan.primary_trunk_reason is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
            and route.curve_radii is None
        ):
            route.curve_radii = [
                ctx.curve_radius for _rank in range(max(0, len(route.points) - 2))
            ]
        _seat_route_on_trunk_flanks(
            route,
            plan.trunk_axis,
            ctx.graph,
            lane_offset,
            shared_terminal_reference_radius=(
                ctx.curve_radius + (len(plan.lane_order) - 1) * ctx.offset_step
                if plan.primary_trunk_reason
                is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH
                else None
            ),
        )
        _set_convergence_owned_segment_ranks(
            route,
            _trunk_segment_ranks(route, plan.trunk_axis)
            + (() if opening_rank is None else (opening_rank,)),
        )
        _consume_final_envelope_allocation(route, ctx)
        _assert_landing_geometry(route, plan, landing, query)
        return
    elif plan.primary_trunk_reason is ConvergenceTrunkReason.LONGEST_BYPASS:
        assert plan.trunk_axis is not None
        run = _run_from_axis(plan.trunk_axis)
        from nf_metro.layout.routing.normalize import _land_feeder_on_run

        key: _EdgeKey = (route.edge.source, route.edge.target, route.line_id)
        if key in ctx.merge.branch_edges:
            _land_feeder_on_run(route, run, ctx)
    if plan.primary_trunk_reason is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH:
        assert plan.trunk_axis is not None
        _seat_shared_terminal_source(route, plan.trunk_axis, ctx)
    else:
        _bake_route(route, ctx)
    _connect_route_endpoint(route, landing.join_point)
    _set_convergence_owned_segment_ranks(
        route,
        (len(route.points) - 2,) + (() if opening_rank is None else (opening_rank,)),
    )
    endpoint = route.points[-1]
    if any(
        abs(actual - expected) > COORD_TOLERANCE
        for actual, expected in zip(endpoint, landing.join_point, strict=True)
    ):
        raise ConvergenceInvariantError(convergence_failure(membership, endpoint))
    _consume_final_envelope_allocation(route, ctx)
    _assert_landing_geometry(route, plan, landing, query)


def validate_convergence_plans(
    routes: list[RoutedPath],
    execution: ConvergencePlanExecution,
) -> None:
    """Require every planned emitted feeder to retain its exact endpoint."""
    by_edge = {
        ResolvedEdge(route.edge.source, route.edge.target, route.line_id): route
        for route in routes
    }
    for plan in execution.plans:
        if not plan.owns_geometry:
            continue
        assert plan.trunk_axis is not None
        primary_ownership = next(
            item
            for item in plan.endpoint_ownership
            if item.member_id == plan.primary_trunk_member_id
        )
        trunk_route = by_edge.get(primary_ownership.edge)
        if trunk_route is None or not _route_covers_trunk(trunk_route, plan.trunk_axis):
            raise ConvergenceInvariantError(
                f"convergence system {plan.system_id} primary trunk member "
                f"{plan.primary_trunk_member_id} does not emit planned "
                f"{plan.trunk_axis.axis.value}-axis {plan.trunk_axis.coordinate} "
                f"over [{plan.trunk_axis.extent_start}, "
                f"{plan.trunk_axis.extent_end}]"
            )
        for landing in plan.landings:
            route = by_edge.get(landing.edge)
            if route is None:
                raise ConvergenceInvariantError(
                    f"convergence system {plan.system_id} lost member "
                    f"{landing.member_id}"
                )
            membership = execution.query.membership_for_edge(landing.edge)
            assert membership is not None
            endpoint = route.points[-1]
            expected_endpoint = execution.query.project_point(
                landing.member_id,
                len(route.points) - 1,
                landing.join_point,
            )
            if (
                not _point_on_trunk_geometry(landing.join_point, plan.trunk_axis)
                or point_to_polyline_distance(landing.join_point, trunk_route.points)
                > COORD_TOLERANCE
            ):
                raise ConvergenceInvariantError(
                    f"convergence system {plan.system_id} feeder "
                    f"{landing.member_id} joins outside its planned trunk axis"
                )
            if landing.member_id == plan.primary_trunk_member_id:
                _assert_landing_geometry(route, plan, landing, execution.query)
                continue
            if any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual, expected in zip(endpoint, expected_endpoint, strict=True)
            ):
                raise ConvergenceInvariantError(
                    convergence_failure(membership, endpoint)
                )
            _assert_landing_geometry(route, plan, landing, execution.query)
        ownership_by_member = {
            ownership.member_id: ownership for ownership in plan.endpoint_ownership
        }
        for continuation in plan.outgoing_continuations:
            membership = execution.query.membership_for_edge(continuation.edge)
            assert membership is not None
            if continuation.covered_by_member_id is not None:
                carrier = ownership_by_member[continuation.covered_by_member_id]
                route = by_edge.get(carrier.edge)
                if (
                    route is None
                    or point_to_polyline_distance(
                        continuation.start_point, route.points
                    )
                    > COORD_TOLERANCE
                    or point_to_polyline_distance(continuation.end_point, route.points)
                    > COORD_TOLERANCE
                ):
                    raise ConvergenceInvariantError(
                        f"convergence system {plan.system_id} covered continuation "
                        f"{continuation.member_id} is absent from its carrier"
                    )
                continue
            route = by_edge.get(continuation.edge)
            if (
                route is None
                or point_to_polyline_distance(continuation.start_point, route.points)
                > COORD_TOLERANCE
                or any(
                    abs(actual - expected) > COORD_TOLERANCE
                    for actual, expected in zip(
                        route.points[-1], continuation.end_point, strict=True
                    )
                )
            ):
                raise ConvergenceInvariantError(
                    f"convergence system {plan.system_id} continuation member "
                    f"{continuation.member_id} differs from its planned endpoints"
                )
        for ownership in plan.endpoint_ownership:
            if ownership.role not in {
                ConvergenceEndpointRole.TRUNK,
                ConvergenceEndpointRole.CONTINUATION,
            }:
                continue
            route = by_edge.get(ownership.edge)
            membership = execution.query.membership_for_edge(ownership.edge)
            if route is None or membership is None:
                raise ConvergenceInvariantError(
                    f"convergence system {plan.system_id} lost endpoint owner "
                    f"{ownership.member_id}"
                )
            endpoint = route.points[-1]
            expected_endpoint = execution.query.project_point(
                ownership.member_id,
                len(route.points) - 1,
                ownership.endpoint,
            )
            if any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual, expected in zip(endpoint, expected_endpoint, strict=True)
            ):
                raise ConvergenceInvariantError(
                    convergence_failure(membership, endpoint)
                )
