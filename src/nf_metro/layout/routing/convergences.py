"""Pre-emission convergence planning and template consumption."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.geometry import point_to_polyline_distance
from nf_metro.layout.route_plan import (
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
    DemandId,
    DemandKind,
    EmissionMemberId,
    ExitTurnPlan,
    ExitTurnPlanId,
    FanPlan,
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
from nf_metro.parser.model import Edge, MetroGraph, Station
from nf_metro.parser.route_topology import (
    ResolvedConvergenceView,
    ResolvedEdge,
    semantic_route_id,
)


class ConvergenceInvariantError(RuntimeError):
    """A planned convergence template violated its immutable contract."""


class UnsupportedConvergenceError(ValueError):
    """Canonical templates cannot represent a complete convergence system."""


class ConvergencePlanningError(RuntimeError):
    """Semantic convergence membership is internally inconsistent."""


_PlanMembership: TypeAlias = tuple[
    tuple[tuple[ResolvedEdge, ...], ...],
    tuple[ResolvedEdge, ...],
    tuple[EmissionMemberId, ...],
]


@dataclass(frozen=True, slots=True)
class ConvergenceRouteMembership:
    plan: ConvergencePlan
    member_id: EmissionMemberId
    landing: ConvergenceLanding | None
    continuation: ConvergenceContinuation | None
    ownership: ConvergenceEndpointOwnership
    covering_edge: ResolvedEdge | None


@dataclass(frozen=True, slots=True)
class ConvergencePlanExecutionQuery:
    plans: tuple[ConvergencePlan, ...]
    _by_edge: Mapping[ResolvedEdge, ConvergenceRouteMembership]

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


@dataclass(frozen=True, slots=True)
class ConvergencePlanExecution:
    plans: tuple[ConvergencePlan, ...]
    references: tuple[SharedReference, ...]
    demands: tuple[SymbolicDemand, ...]
    diagnostics: tuple[RoutePlanDiagnostic, ...]
    query: ConvergencePlanExecutionQuery


def empty_convergence_plan_execution() -> ConvergencePlanExecution:
    query = ConvergencePlanExecutionQuery((), MappingProxyType({}))
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


def _trunk_run(route: RoutedPath, expected_coordinate: float) -> HTrunkSeg:
    runs = tuple(segment for _rank, segment in iter_horizontal_trunks(route))
    if not runs:
        raise UnsupportedConvergenceError(
            "primary trunk template emitted no shared run"
        )
    return min(runs, key=lambda segment: abs(segment.y - expected_coordinate))


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
    from nf_metro.layout.routing.normalize import _initial_fanout_descent

    opening_turn = _initial_fanout_descent(route)
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
) -> ConvergencePlan:
    group = view.group
    system_id = scaffold.system_for(group.connector_ids)
    paths, edges, member_ids = membership
    member_by_edge = dict(zip(edges, member_ids, strict=True))
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
        trial_routes[trunk_edge_key] = _trunk_route(edge, ctx)
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
            else:
                route = _trial_route(edge, ctx)
        if route is None:
            raise UnsupportedConvergenceError(f"feeder template declined {edge_key!r}")
        trial_routes[edge_key] = route

    if trunk_run is not None:
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
                ConvergenceEndpointOwnership(member_id, edge_key, role, endpoint)
            )
            continue
        continuation_route = trial_routes.get(edge_key)
        if continuation_route is None:
            edge = ctx.edge_by_key[(edge_key.source, edge_key.target, edge_key.line_id)]
            continuation_route = _trial_route(edge, ctx)
            trial_routes[edge_key] = continuation_route
        start_point = (
            _axis_source_point(trunk_axis)
            if primary_reason
            in {
                ConvergenceTrunkReason.OUTGOING_CONTINUATION,
                ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH,
            }
            else _axis_target_point(trunk_axis)
        )
        end_point = continuation_route.points[-1]
        hop_start_point = continuation_route.points[0]
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
                member_id,
                edge_key,
                (
                    ConvergenceEndpointRole.COVERED_CONTINUATION
                    if covered_by is not None
                    else ConvergenceEndpointRole.CONTINUATION
                ),
                end_point,
                covered_by,
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

    plan_id = ConvergencePlanId(
        semantic_route_id("convergence-plan", system_id, group.id)
    )
    reference_ids = (
        SharedReferenceId(semantic_route_id("convergence-trunk", plan_id)),
        SharedReferenceId(semantic_route_id("convergence-landings", plan_id)),
    )
    demand_ids = (
        DemandId(semantic_route_id("convergence-band", plan_id)),
        DemandId(semantic_route_id("convergence-runway", plan_id)),
    )
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
    return tuple(references), tuple(demands)


def _query(plans: tuple[ConvergencePlan, ...]) -> ConvergencePlanExecutionQuery:
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
            membership = ConvergenceRouteMembership(
                plan,
                ownership.member_id,
                landings.get(ownership.member_id),
                continuations.get(ownership.member_id),
                ownership,
                covering_ownership.edge if covering_ownership is not None else None,
            )
            if ownership.edge in by_edge:
                raise ValueError("planned convergence edge has more than one owner")
            by_edge[ownership.edge] = membership
    return ConvergencePlanExecutionQuery(plans, MappingProxyType(by_edge))


def build_convergence_plan_execution(
    graph: MetroGraph,
    ctx: _RoutingCtx,
    scaffold: RouteSemanticScaffold,
    *,
    exit_turn_plans: tuple[ExitTurnPlan, ...],
    fan_plans: tuple[FanPlan, ...],
    include_resources: bool = True,
) -> ConvergencePlanExecution:
    """Plan every semantic convergence atomically by route system."""
    views_by_system: dict[RouteSystemId, list[ResolvedConvergenceView]] = defaultdict(
        list
    )
    for view in scaffold.query.convergences:
        views_by_system[scaffold.system_for(view.group.connector_ids)].append(view)
    plans: list[ConvergencePlan] = []
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
            for item in exit_turn_plans
            if set(item.connector_ids) & connector_ids
            or set(item.member_ids) & member_ids
        )
        upstream_exit_ids = tuple(item.id for item in upstream_exit_plans)
        upstream_fan_ids = tuple(
            item.id
            for item in fan_plans
            if item.system_id == system_id
            and (
                set(item.connector_ids) & connector_ids
                or set(item.member_ids) & member_ids
            )
        )
        try:
            system_plans = tuple(
                _build_planned_convergence(
                    graph,
                    ctx,
                    scaffold,
                    view,
                    membership,
                    upstream_exit_ids,
                    upstream_fan_ids,
                )
                for view, membership in zip(views, memberships, strict=True)
            )
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
    references, demands = (
        _resources(graph, frozen_plans) if include_resources else ((), ())
    )
    return ConvergencePlanExecution(
        frozen_plans,
        references,
        demands,
        tuple(diagnostics),
        _query(frozen_plans),
    )


def convergence_failure(
    membership: ConvergenceRouteMembership,
    emitted_endpoint: tuple[float, float],
) -> str:
    plan = membership.plan
    connector = plan.connector_ids[0]
    expected = membership.ownership.endpoint
    return (
        f"convergence system {plan.system_id} connector {connector} member "
        f"{membership.member_id} planned join {expected} emitted endpoint "
        f"{emitted_endpoint}"
    )


def _point_on_axis(point: tuple[float, float], axis: ConvergenceTrunkAxis) -> bool:
    transverse, longitudinal = (
        (point[1], point[0]) if axis.axis is DemandAxis.X else (point[0], point[1])
    )
    return (
        abs(transverse - axis.coordinate) <= COORD_TOLERANCE
        and axis.extent_start - COORD_TOLERANCE
        <= longitudinal
        <= axis.extent_end + COORD_TOLERANCE
    )


def _point_on_trunk_geometry(
    point: tuple[float, float], axis: ConvergenceTrunkAxis
) -> bool:
    if _point_on_axis(point, axis):
        return True
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
        return any(
            abs(point[0] - longitudinal) <= COORD_TOLERANCE
            and min(axis.coordinate, flank) - COORD_TOLERANCE
            <= point[1]
            <= max(axis.coordinate, flank) + COORD_TOLERANCE
            for longitudinal, flank in (
                (source_longitudinal, axis.source_flank_coordinate),
                (target_longitudinal, axis.target_flank_coordinate),
            )
        ) or any(
            abs(point[1] - flank) <= COORD_TOLERANCE
            and min(longitudinal, endpoint) - COORD_TOLERANCE
            <= point[0]
            <= max(longitudinal, endpoint) + COORD_TOLERANCE
            for longitudinal, flank, endpoint in (
                (
                    source_longitudinal,
                    axis.source_flank_coordinate,
                    source_endpoint,
                ),
                (
                    target_longitudinal,
                    axis.target_flank_coordinate,
                    target_endpoint,
                ),
            )
        )
    return any(
        abs(point[1] - longitudinal) <= COORD_TOLERANCE
        and min(axis.coordinate, flank) - COORD_TOLERANCE
        <= point[0]
        <= max(axis.coordinate, flank) + COORD_TOLERANCE
        for longitudinal, flank in (
            (source_longitudinal, axis.source_flank_coordinate),
            (target_longitudinal, axis.target_flank_coordinate),
        )
    ) or any(
        abs(point[0] - flank) <= COORD_TOLERANCE
        and min(longitudinal, endpoint) - COORD_TOLERANCE
        <= point[1]
        <= max(longitudinal, endpoint) + COORD_TOLERANCE
        for longitudinal, flank, endpoint in (
            (
                source_longitudinal,
                axis.source_flank_coordinate,
                source_endpoint,
            ),
            (
                target_longitudinal,
                axis.target_flank_coordinate,
                target_endpoint,
            ),
        )
    )


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


def _assert_landing_geometry(
    route: RoutedPath,
    plan: ConvergencePlan,
    landing: ConvergenceLanding,
) -> None:
    actual = _landing_approach(route, landing.join_point)
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
    route.convergence_plan_id = str(plan.id)
    route.convergence_member_id = str(membership.member_id)
    landing = membership.landing
    if landing is None:
        continuation = membership.continuation
        if continuation is not None and (
            continuation.covered_by_member_id is not None
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
                f"{membership.member_id} differs from its planned endpoints"
            )
        if plan.primary_trunk_member_id == membership.member_id:
            assert plan.trunk_axis is not None
            route.convergence_owned_segment_ranks = _trunk_segment_ranks(
                route, plan.trunk_axis
            )
        return
    if plan.primary_trunk_member_id == membership.member_id:
        if plan.primary_trunk_reason is ConvergenceTrunkReason.SHARED_TERMINAL_APPROACH:
            _bake_route(route, ctx)
            _connect_route_endpoint(route, landing.join_point)
        assert plan.trunk_axis is not None
        route.convergence_owned_segment_ranks = _trunk_segment_ranks(
            route, plan.trunk_axis
        )
        _assert_landing_geometry(route, plan, landing)
        return
    elif plan.primary_trunk_reason is ConvergenceTrunkReason.LONGEST_BYPASS:
        assert plan.trunk_axis is not None
        run = _run_from_axis(plan.trunk_axis)
        from nf_metro.layout.routing.normalize import (
            _land_feeder_on_run,
            _seat_merge_feeder_opening,
        )

        key: _EdgeKey = (route.edge.source, route.edge.target, route.line_id)
        if key in ctx.merge.branch_edges:
            if landing.opening_turn_coordinate is not None:
                _seat_merge_feeder_opening(
                    route,
                    landing.opening_turn_coordinate,
                    ctx.graph,
                )
            _land_feeder_on_run(route, run, ctx)
    _bake_route(route, ctx)
    _connect_route_endpoint(route, landing.join_point)
    route.convergence_owned_segment_ranks = (len(route.points) - 2,)
    endpoint = route.points[-1]
    if any(
        abs(actual - expected) > COORD_TOLERANCE
        for actual, expected in zip(endpoint, landing.join_point, strict=True)
    ):
        raise ConvergenceInvariantError(convergence_failure(membership, endpoint))
    _assert_landing_geometry(route, plan, landing)


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
            endpoint = route.points[-1]
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
                _assert_landing_geometry(route, plan, landing)
                continue
            if any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual, expected in zip(endpoint, landing.join_point, strict=True)
            ):
                membership = execution.query.membership_for_edge(landing.edge)
                assert membership is not None
                raise ConvergenceInvariantError(
                    convergence_failure(membership, endpoint)
                )
            _assert_landing_geometry(route, plan, landing)
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
            if any(
                abs(actual - expected) > COORD_TOLERANCE
                for actual, expected in zip(endpoint, ownership.endpoint, strict=True)
            ):
                raise ConvergenceInvariantError(
                    convergence_failure(membership, endpoint)
                )
