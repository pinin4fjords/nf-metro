"""Core edge routing: the main route_edges() dispatcher.

Routes edges as horizontal segments with 45-degree diagonal transitions.
The per-handler families and post-routing passes live in sibling modules
(context, *_handlers, normalize, postprocess) and are re-exported here for
backward-compatible ``routing.core`` imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nf_metro.layout.constants import (
    CURVE_RADIUS,
    DIAGONAL_RUN,
)
from nf_metro.layout.routing.common import (
    RoutedPath,
)
from nf_metro.layout.routing.context import (  # noqa: F401
    _build_routing_context,
    _classify_merge_edges,
    _compute_bypass_gap_indices,
    _compute_junction_fan_info,
    _compute_section_trunk_ys,
    _EdgeKey,
    _get_offset,
    _has_intervening_sections,
    _max_offset_at,
    _MergeRouting,
    _resolve_section_col,
    _resolve_section_colrow,
    _resolve_section_row,
    _RoutingCtx,
    _tb_x_offset,
    compute_junction_fan_info,
)
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.layout.routing.inter_section_handlers import (  # noqa: F401
    _build_right_entry_wrap_route,
    _corridor_descent_x,
    _corridor_is_viable,
    _fan_has_corridor_sibling,
    _fan_left_entry_descent_x,
    _gap_above_target_y,
    _has_around_section_sibling,
    _has_bypass_sibling_to_same_entry,
    _left_entry_descent_x,
    _right_entry_gap_above_is_clear,
    _route_around_section_below,
    _route_bottom_exit_junction,
    _route_bypass,
    _route_inter_row_gap_corridor,
    _route_inter_section,
    _route_l_shape,
    _route_left_entry_wrap,
    _route_left_exit_left_entry_drop,
    _route_merge_branch,
    _route_merge_trunk,
    _route_right_entry_around_below,
    _route_right_entry_via_gap_above,
    _route_right_entry_wrap,
    _route_tb_bottom_exit,
    _route_top_entry_l_shape,
    _v1_corner_x,
)
from nf_metro.layout.routing.intra_handlers import (  # noqa: F401
    _is_side_branch_ascent,
    _route_diagonal,
    _route_entry_runway,
    _route_intra_section,
)
from nf_metro.layout.routing.normalize import (  # noqa: F401
    _band_order_crossings,
    _bundle_divergent_distinct_descents,
    _bundle_divergent_distinct_traverses,
    _clamp_inter_row_band_top,
    _clear_channel_x_in_band,
    _clear_merge_trunk_opposite_arm,
    _coincide_fanout_opening_descents,
    _coincide_merge_fanout_pivots,
    _coincide_same_line_tracks,
    _coincident_trunk_slots,
    _collect_htrunks,
    _distinct_line_order,
    _dogleg_off_exempt_trunks,
    _drop_covered_merge_entry_hops,
    _final_port_approach,
    _gap_channel_base,
    _group_channel_trunks,
    _h_segment_crosses_other_section,
    _HTrunk,
    _inter_row_gap_band,
    _land_merge_feeders_on_trunk,
    _materialize_gap_slots,
    _materialize_trunk_slots,
    _nest_bypass_above_over_top_wrap,
    _plan_trunk_band,
    _reconcile_port_peeloff_risers,
    _restack_channel,
    _restack_htrunk,
    _restack_trunk_band,
    _round_junction_perp_peeloff,
    _separate_declared_opposing_gap_bundles,
    _separate_opposing_inter_row_trunks,
    _set_vchannel_x,
    _stagger_convergent_distinct_lines,
    _suboptimal_trunk_bands,
    _unify_coincident_corner_radii,
    _VChannel,
)
from nf_metro.layout.routing.postprocess import (  # noqa: F401
    _align_uncentered_siblings,
    _apply_diagonal_spread,
    _apply_station_moves,
    _BubbleCtx,
    _build_bubble_ctx,
    _center_bubble_stations,
    _clear_bypass_v_label_strikes,
    _collect_centering_candidates,
    _is_diagonal_route,
    _spread_diagonal_bundles,
    _StationMoveCandidate,
)
from nf_metro.layout.routing.tb_handlers import (  # noqa: F401
    _compute_diagonal_placement,
    _perp_entry_drop_delta,
    _route_perp_entry,
    _route_tb_diagonal,
    _route_tb_internal,
    _route_tb_lr_entry,
    _route_tb_lr_exit,
    _route_tb_section,
)
from nf_metro.parser.model import (
    LineSpread,
    MetroGraph,
)

if TYPE_CHECKING:
    from nf_metro.layout.envelope_settlement import (
        EnvelopeCapacityLimitation,
        EnvelopeCapacityProof,
        EnvelopeIdentityProjection,
    )
    from nf_metro.layout.route_plan import (
        EmissionBinding,
        RouteObservation,
        RoutePlan,
        RoutePlanObserver,
    )
    from nf_metro.layout.route_reservations import RouteReservation


def _route_edges(
    graph: MetroGraph,
    diagonal_run: float,
    curve_radius: float,
    station_offsets: dict[tuple[str, str], float] | None,
    *,
    observe_plan: bool,
    offset_step: float | None = None,
    envelope_proofs: tuple[EnvelopeCapacityProof, ...] = (),
    envelope_limitations: tuple[EnvelopeCapacityLimitation, ...] = (),
    envelope_reservations: tuple[RouteReservation, ...] = (),
    envelope_bindings: tuple[EmissionBinding, ...] = (),
    envelope_identity_projections: tuple[EnvelopeIdentityProjection, ...] = (),
) -> tuple[list[RoutedPath], dict[str, float], RoutePlan | None]:
    """Route all edges, returning the paths and the bubble-centring moves.

    Shared body behind :func:`route_edges` (pure) and
    :func:`route_edges_centred` (applies the moves).  The ``moves`` are the
    per-station X-targets the bubble-centring pass produced as ``{station_id:
    x}`` requests; the route points are adjusted in place either way.
    """
    observer: RoutePlanObserver | None
    if graph.line_spread is LineSpread.RAILS:
        from nf_metro.layout.routing.rail import route_rail_edges

        rail_graph_routes = route_rail_edges(graph)
        from nf_metro.layout.route_plan import build_route_semantic_scaffold

        scaffold = build_route_semantic_scaffold(
            graph,
            coupled_connector_groups=tuple(
                fan_plan.connector_ids
                for fan_plan in graph.fan_plans
                if fan_plan.connector_ids
            ),
        )
        if scaffold is not None and envelope_reservations:
            from nf_metro.layout.routing.envelope_allocations import (
                build_envelope_allocation_query,
            )

            envelope_allocations = build_envelope_allocation_query(
                envelope_proofs,
                scaffold.member_id_by_edge,
                envelope_reservations,
                envelope_bindings,
                envelope_limitations,
                envelope_identity_projections,
            )
            for route in rail_graph_routes:
                envelope_allocations.consume(route)
            envelope_allocations.assert_complete(rail_graph_routes)
        observer = None
        if observe_plan:
            from nf_metro.layout.route_plan import build_route_plan_observer

            observer = build_route_plan_observer(graph, None, scaffold=scaffold)
        if observer is not None:
            observer.record_rail_routes(rail_graph_routes)
        return (
            rail_graph_routes,
            {},
            observer.finish(rail_graph_routes) if observer is not None else None,
        )

    # Per-section rail mode: route each rail section's own edges with the
    # dedicated rail router (straight rails, no bundling) and let the normal
    # router handle every other edge.  An edge belongs to a rail section when
    # both endpoints sit in it and at most one of them is a boundary port: a
    # port-to-station leg carries the fan between the section's single gateway
    # and the rails, so the rail router owns it too; a port-to-port leg crosses
    # between sections and stays with the normal router.
    rail_routes: list[RoutedPath] = []
    rail_internal: set[tuple[str, str, str]] = set()
    if graph.has_rail_sections:
        from nf_metro.layout.routing.rail import route_rail_edges

        rail_edges = []
        for edge in graph.edges:
            src, tgt = graph.edge_endpoints(edge)
            if src.is_port and tgt.is_port:
                continue
            if (
                src.section_id == tgt.section_id
                and src.section_id is not None
                and graph.is_rail_section(src.section_id)
            ):
                rail_edges.append(edge)
                rail_internal.add((edge.source, edge.target, edge.line_id))
        rail_routes = route_rail_edges(graph, rail_edges, station_offsets)

    ctx = _build_routing_context(
        graph,
        diagonal_run,
        curve_radius,
        station_offsets,
        offset_step=offset_step,
    )
    from nf_metro.layout.route_plan import BindingKind, build_route_plan_observer
    from nf_metro.layout.routing.exit_turns import build_exit_turn_execution
    from nf_metro.parser.route_topology import ResolvedEdge

    execution = build_exit_turn_execution(graph, ctx)
    if execution.scaffold is not None:
        from nf_metro.layout.routing.envelope_allocations import (
            build_envelope_allocation_query,
        )

        ctx.envelope_allocations = build_envelope_allocation_query(
            envelope_proofs,
            execution.scaffold.member_id_by_edge,
            envelope_reservations,
            envelope_bindings,
            envelope_limitations,
            envelope_identity_projections,
        )
        from nf_metro.layout.routing.exit_turns import (
            materialize_exit_turn_envelope_axes,
        )

        execution = materialize_exit_turn_envelope_axes(
            execution, ctx.envelope_allocations
        )
    ctx.exit_turns = execution.query
    from nf_metro.layout.routing.convergences import (
        build_convergence_plan_execution,
        empty_convergence_plan_execution,
    )

    convergence_execution = (
        build_convergence_plan_execution(
            graph,
            ctx,
            execution.scaffold,
            exit_turn_plans=execution.plans,
            fan_plans=graph.fan_plans,
            include_resources=observe_plan,
            envelope_proofs=envelope_proofs,
            envelope_limitations=envelope_limitations,
        )
        if execution.scaffold is not None
        else empty_convergence_plan_execution()
    )
    ctx.convergences = convergence_execution.query
    observer = None
    if observe_plan:
        observer = build_route_plan_observer(
            graph,
            ctx,
            scaffold=execution.scaffold,
            exit_turn_plans=execution.plans,
            exit_turn_references=execution.references,
            exit_turn_demands=execution.demands,
            exit_turn_diagnostics=execution.diagnostics,
            convergence_plans=convergence_execution.plans,
            convergence_references=convergence_execution.references,
            convergence_demands=convergence_execution.demands,
            convergence_diagnostics=convergence_execution.diagnostics,
        )
    # Route into the context's own list so handlers can read the routes settled
    # so far (a wrap clearing an already-placed sibling channel); it grows as
    # edges route and is what every post-loop pass consumes.
    routes: list[RoutedPath] = ctx.built_routes
    routes.extend(rail_routes)

    for edge in graph.edges:
        resolved_edge = ResolvedEdge(edge.source, edge.target, edge.line_id)
        immutable_binding = (
            ctx.envelope_allocations.immutable_binding_for_edge(resolved_edge)
            if ctx.envelope_allocations is not None
            else None
        )
        requires_post_dispatch_coverage = (
            immutable_binding is not None
            and immutable_binding.kind is BindingKind.COVERED_MERGE_HOP
        )
        planned_covering_edge = (
            ctx.convergences.covering_edge_for_edge(edge)
            if ctx.convergences is not None
            else None
        )
        if planned_covering_edge is not None and not requires_post_dispatch_coverage:
            if observer is not None:
                observer.record_merge_skip(
                    (edge.source, edge.target, edge.line_id),
                    (
                        planned_covering_edge.source,
                        planned_covering_edge.target,
                        planned_covering_edge.line_id,
                    ),
                )
            continue
        if (
            edge.source,
            edge.target,
            edge.line_id,
        ) in ctx.skip_edges and not requires_post_dispatch_coverage:
            if observer is not None:
                observer.record_merge_skip(
                    (edge.source, edge.target, edge.line_id),
                    observer.covering_edge((edge.source, edge.target, edge.line_id)),
                )
            continue
        if (edge.source, edge.target, edge.line_id) in rail_internal:
            continue

        src, tgt = graph.edge_endpoints(edge)
        edge_key = (edge.source, edge.target, edge.line_id)
        observe_fallback = (
            observer is not None
            and (src.is_port or edge.source in ctx.junction_ids)
            and (tgt.is_port or edge.target in ctx.junction_ids)
        )

        # Try each routing handler in priority order.
        # The first handler that returns a RoutedPath wins.
        result = _route_inter_section(edge, src, tgt, ctx, observer=observer)
        if result is None:
            result = _route_tb_section(edge, src, tgt, ctx)
            if result is not None and observer is not None and observe_fallback:
                observer.record_dispatch(edge_key, RouteFamilyId.TB_SECTION_FALLBACK)
        if result is None:
            result = _route_entry_runway(edge, src, tgt, ctx)
            if result is not None and observer is not None and observe_fallback:
                observer.record_dispatch(edge_key, RouteFamilyId.ENTRY_RUNWAY_FALLBACK)
        if result is None:
            result = _route_intra_section(edge, src, tgt, ctx)
            if result is not None and observer is not None and observe_fallback:
                observer.record_dispatch(edge_key, RouteFamilyId.INTRA_SECTION_FALLBACK)

        if result is not None:
            routes.append(result)

    from nf_metro.layout.routing.exit_turns import (
        assert_exit_turn_snapshot,
        snapshot_exit_turn_segments,
        validate_exit_turn_plans,
    )

    planned_segments = snapshot_exit_turn_segments(routes, execution.plans)
    from nf_metro.layout.routing.envelope_allocations import assert_route_allocations

    moves = _center_bubble_stations(routes, graph)
    assert_route_allocations(routes, "bubble centring")
    assert_exit_turn_snapshot(routes, planned_segments, "bubble centring")
    _spread_diagonal_bundles(routes, ctx)
    assert_route_allocations(routes, "diagonal spreading")
    assert_exit_turn_snapshot(routes, planned_segments, "diagonal spreading")
    _materialize_gap_slots(routes, ctx)
    assert_route_allocations(routes, "gap-slot materialization")
    assert_exit_turn_snapshot(routes, planned_segments, "gap-slot materialization")
    _materialize_trunk_slots(routes, ctx)
    assert_route_allocations(routes, "trunk-slot materialization")
    assert_exit_turn_snapshot(routes, planned_segments, "trunk-slot materialization")
    # Counter-running flows that entered one inter-row gap from opposite rows
    # sit in different dip groups the trunk-slot pass never compares, so they
    # both centre on the gap and fold over each other; split them onto their
    # own direction-specific bands before the downstream coincidence passes read
    # the settled channels.
    _separate_opposing_inter_row_trunks(routes, ctx)
    assert_route_allocations(routes, "opposing-trunk separation")
    assert_exit_turn_snapshot(routes, planned_segments, "opposing-trunk separation")
    # Re-stack peel-off risers against the settled trunk depths, so each rises
    # on the concentric slot its post-repack depth earns.
    _reconcile_port_peeloff_risers(routes, ctx)
    assert_route_allocations(routes, "port-peeloff reconciliation")
    assert_exit_turn_snapshot(routes, planned_segments, "port-peeloff reconciliation")
    # Peel-off reconciliation can transpose riser order, so symmetric
    # divergences must be joined against those final columns.
    _separate_declared_opposing_gap_bundles(routes, ctx)
    assert_route_allocations(routes, "opposing-gap separation")
    assert_exit_turn_snapshot(routes, planned_segments, "opposing-gap separation")
    # A merge fan-out's branches leave one fork and turn off its lead-out
    # through a first corner each; fuse those corners onto one shared pivot
    # column so the fork opens as one stroke, before the same-line coincidence
    # pass reads the settled channels.
    _coincide_merge_fanout_pivots(routes, ctx)
    assert_route_allocations(routes, "merge-fanout pivoting")
    assert_exit_turn_snapshot(routes, planned_segments, "merge-fanout pivoting")
    # Coincidence runs after the trunk/gap channels are finalised: it snaps
    # same-line tracks onto a reference read from that final geometry (the
    # port-side track, the source-side track, the merge trunk's descent, and
    # the fan-out junction handoff tail), so a single line reads as one stroke.
    _coincide_same_line_tracks(routes, ctx)
    assert_route_allocations(routes, "same-line coincidence")
    assert_exit_turn_snapshot(routes, planned_segments, "same-line coincidence")
    # Settle every fan-out's opening-descent column in one pass: fuse each line's
    # same-source descents onto the source-nearest track and nest the distinct
    # lines one step apart until each turns off.  Runs after the coincidence pass
    # so a perpendicular drop already resolved onto the junction column stays
    # clear of an L-shaped sibling diverging to another column.
    _coincide_fanout_opening_descents(routes, ctx)
    assert_route_allocations(routes, "fanout opening coincidence")
    assert_exit_turn_snapshot(routes, planned_segments, "fanout opening coincidence")
    # Distinct lines fanning out share the corridor they turn onto; nest their
    # traverses one step apart so the bundle holds a constant width until each
    # line peels off, rather than running on independently-sized bands.
    _bundle_divergent_distinct_traverses(routes, ctx)
    assert_route_allocations(routes, "fanout traverse bundling")
    assert_exit_turn_snapshot(routes, planned_segments, "fanout traverse bundling")
    # A perpendicular branch dropped directly off a horizontal fan-out junction
    # trunk peels off at a hard 90; give its departure a lead-in so the corner
    # curves. Runs after coincidence settles the drop's port column.
    _round_junction_perp_peeloff(routes, ctx)
    assert_route_allocations(routes, "perpendicular peeloff rounding")
    assert_exit_turn_snapshot(
        routes, planned_segments, "perpendicular peeloff rounding"
    )
    # Distinct-line counterpart: spread any two different lines whose final port
    # descents were forced onto one channel (a shared gap left of a wide target).
    _stagger_convergent_distinct_lines(routes, ctx)
    assert_route_allocations(routes, "convergence staggering")
    assert_exit_turn_snapshot(routes, planned_segments, "convergence staggering")
    # A same-row over-top wrap to a RIGHT entry is pinned deep in the inter-row
    # gap by the target's header clearance; lift any longer-haul cross-row bypass
    # sharing that gap above the wrap's peak so the local wrap nests beneath it.
    _nest_bypass_above_over_top_wrap(routes, ctx)
    assert_route_allocations(routes, "bypass nesting")
    assert_exit_turn_snapshot(routes, planned_segments, "bypass nesting")
    _clear_bypass_v_label_strikes(routes, ctx)
    assert_route_allocations(routes, "bypass label clearance")
    assert_exit_turn_snapshot(routes, planned_segments, "bypass label clearance")
    # A merge fan-out's down-trunk and an opposite up-arm to a second merge can
    # settle onto one column over a shared Y span, folding the line back over
    # itself.  Slide the down-trunk's descent column a curve radius past the
    # up-arm through the concentric channel machinery so the two clear.  Reads
    # the settled columns, so it runs after the channel-settling passes.
    _clear_merge_trunk_opposite_arm(routes, ctx)
    assert_route_allocations(routes, "merge-arm clearance")
    assert_exit_turn_snapshot(routes, planned_segments, "merge-arm clearance")
    # Settle where each merge feeder meets its trunk -- on the trunk's own
    # centreline, at or before the corner it turns away on. Runs downstream of
    # every pass that moves a trunk channel or a feeder's descent column, since
    # it reads both from the settled geometry.
    _land_merge_feeders_on_trunk(routes, ctx)
    assert_route_allocations(routes, "merge-feeder landing")
    assert_exit_turn_snapshot(routes, planned_segments, "merge-feeder landing")
    # Same-line legs a coincidence pass fused onto one channel each kept their
    # handler's corner radius; unify every turn they share so the fused stroke
    # draws one arc rather than concentric duplicates.
    _unify_coincident_corner_radii(routes, ctx)
    assert_route_allocations(routes, "corner-radius unification")
    assert_exit_turn_snapshot(routes, planned_segments, "corner-radius unification")
    covered_merge_hops = _drop_covered_merge_entry_hops(
        routes, ctx, report_coverage=observer is not None
    )
    assert_route_allocations(routes, "covered merge-hop removal")
    assert_exit_turn_snapshot(routes, planned_segments, "covered merge-hop removal")
    if observer is not None:
        observer.record_covered_merge_hops(covered_merge_hops)
    if ctx.envelope_allocations is not None:
        ctx.envelope_allocations.assert_complete(routes)

    validate_exit_turn_plans(
        graph,
        routes,
        execution.plans,
        ctx.station_offsets or {},
    )
    from nf_metro.layout.fan_plans import validate_fan_route_emissions

    validate_fan_route_emissions(graph, routes, ctx.station_offsets)
    from nf_metro.layout.routing.convergences import validate_convergence_plans

    validate_convergence_plans(routes, convergence_execution)

    return routes, moves, observer.finish(routes) if observer is not None else None


def route_edges(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
    station_offsets: dict[tuple[str, str], float] | None = None,
    *,
    offset_step: float | None = None,
) -> list[RoutedPath]:
    """Route all edges with smooth direction changes.

    Detects cross-row edges (large Y gap relative to X gap) and routes
    them through a vertical connector at the fold edge.

    Routing is pure with respect to placement: it never moves stations.  The
    bubble-centring pass emits its per-station X-targets as move requests,
    which this entry point discards; :func:`route_edges_centred` is the variant
    that applies them.  Callers get a route they can inspect without perturbing
    ``graph.stations``.
    """
    routes, _moves, _plan = _route_edges(
        graph,
        diagonal_run,
        curve_radius,
        station_offsets,
        observe_plan=False,
        offset_step=offset_step,
    )
    return routes


def observe_route_edges(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
    station_offsets: dict[tuple[str, str], float] | None = None,
    *,
    offset_step: float | None = None,
    envelope_proofs: tuple[EnvelopeCapacityProof, ...] = (),
    envelope_limitations: tuple[EnvelopeCapacityLimitation, ...] = (),
    envelope_reservations: tuple[RouteReservation, ...] = (),
    envelope_bindings: tuple[EmissionBinding, ...] = (),
    envelope_identity_projections: tuple[EnvelopeIdentityProjection, ...] = (),
) -> RouteObservation:
    """Route once and return the context-local semantic observation."""
    from nf_metro.layout.route_plan import RouteObservation

    routes, _moves, plan = _route_edges(
        graph,
        diagonal_run,
        curve_radius,
        station_offsets,
        observe_plan=True,
        offset_step=offset_step,
        envelope_proofs=envelope_proofs,
        envelope_limitations=envelope_limitations,
        envelope_reservations=envelope_reservations,
        envelope_bindings=envelope_bindings,
        envelope_identity_projections=envelope_identity_projections,
    )
    assert plan is not None
    return RouteObservation(routes, plan)


def _settle_station_moves(graph: MetroGraph, moves: dict[str, float]) -> None:
    for sid, x in moves.items():
        graph.stations[sid].x = x


def route_edges_centred(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
    station_offsets: dict[tuple[str, str], float] | None = None,
    *,
    offset_step: float | None = None,
) -> list[RoutedPath]:
    """Route, then settle the bubble-centred markers onto ``graph.stations``.

    The drawn variant of :func:`route_edges`: it applies the centring move
    requests so any reader of marker / label geometry after routing (the SVG
    render, the label-overlap spacing search, the render-output strike guards)
    sees the markers on their centred flats.  Unlike :func:`route_edges` this
    is *not* placement-pure -- it is the single named home for that mutation.

    Inside a ``_restoring_layout_geometry`` scope the move is undone on exit,
    so a probe can inspect the drawn geometry without perturbing the settled
    layout.  Bisection / placement guards that must read the *un-centred*
    placement geometry call :func:`route_edges` directly instead.
    """
    routes, moves, _plan = _route_edges(
        graph,
        diagonal_run,
        curve_radius,
        station_offsets,
        observe_plan=False,
        offset_step=offset_step,
    )
    _settle_station_moves(graph, moves)
    return routes


def observe_route_edges_centred(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
    station_offsets: dict[tuple[str, str], float] | None = None,
    *,
    offset_step: float | None = None,
    envelope_proofs: tuple[EnvelopeCapacityProof, ...] = (),
    envelope_limitations: tuple[EnvelopeCapacityLimitation, ...] = (),
    envelope_reservations: tuple[RouteReservation, ...] = (),
    envelope_bindings: tuple[EmissionBinding, ...] = (),
    envelope_identity_projections: tuple[EnvelopeIdentityProjection, ...] = (),
) -> RouteObservation:
    """Route drawn geometry and return its context-local semantic observation."""
    from nf_metro.layout.route_plan import RouteObservation

    routes, moves, plan = _route_edges(
        graph,
        diagonal_run,
        curve_radius,
        station_offsets,
        observe_plan=True,
        offset_step=offset_step,
        envelope_proofs=envelope_proofs,
        envelope_limitations=envelope_limitations,
        envelope_reservations=envelope_reservations,
        envelope_bindings=envelope_bindings,
        envelope_identity_projections=envelope_identity_projections,
    )
    _settle_station_moves(graph, moves)
    assert plan is not None
    return RouteObservation(routes, plan)
