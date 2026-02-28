"""Core edge routing: the main route_edges() dispatcher.

Routes edges as horizontal segments with 45-degree diagonal transitions.
For folded layouts, cross-row edges route through the fold edge with a
clean vertical drop. Inter-section edges use L-shaped routing with
per-line bundle offsets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nf_metro.layout.constants import (
    BYPASS_CLEARANCE,
    BYPASS_NEST_STEP,
    COORD_TOLERANCE,
    COORD_TOLERANCE_FINE,
    CROSS_ROW_THRESHOLD,
    CURVE_RADIUS,
    DIAGONAL_RUN,
    FOLD_MARGIN,
    JUNCTION_MARGIN,
    MIN_STRAIGHT_EDGE,
    MIN_STRAIGHT_PORT,
    OFFSET_STEP,
)
from nf_metro.layout.labels import label_text_width
from nf_metro.layout.routing.common import (
    RoutedPath,
    adjacent_column_gap_x,
    bypass_bottom_y,
    compute_bundle_info,
    inter_column_channel_x,
)
from nf_metro.layout.routing.corners import (
    l_shape_radii,
    reversed_offset,
    tb_entry_corner,
    tb_exit_corner,
)
from nf_metro.parser.model import Edge, MetroGraph, PortSide, Station

# ---------------------------------------------------------------------------
# Routing context: pre-computed state shared by all handlers
# ---------------------------------------------------------------------------


@dataclass
class _RoutingCtx:
    """Pre-computed state shared by edge routing handlers."""

    graph: MetroGraph
    fold_x: float
    junction_ids: set[str]
    bottom_exit_junctions: set[str]
    bottom_exit_junction_ports: dict[str, str]
    offset_step: float
    fork_stations: set[str]
    join_stations: set[str]
    tb_sections: set[str]
    tb_right_entry: set[str]
    bundle_info: dict[tuple[str, str, str], tuple[int, int]]
    bypass_gap_idx: dict[tuple[str, str, str], tuple[int, int, int, int]]
    station_offsets: dict[tuple[str, str], float] | None
    diagonal_run: float
    curve_radius: float
    skip_edges: set[tuple[str, str, str]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_edges(
    graph: MetroGraph,
    diagonal_run: float = DIAGONAL_RUN,
    curve_radius: float = CURVE_RADIUS,
    station_offsets: dict[tuple[str, str], float] | None = None,
) -> list[RoutedPath]:
    """Route all edges with smooth direction changes.

    Detects cross-row edges (large Y gap relative to X gap) and routes
    them through a vertical connector at the fold edge.
    """
    ctx = _build_routing_context(graph, diagonal_run, curve_radius, station_offsets)
    routes: list[RoutedPath] = []

    for edge in graph.edges:
        if (edge.source, edge.target, edge.line_id) in ctx.skip_edges:
            continue

        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue

        # Try each routing handler in priority order.
        # The first handler that returns a RoutedPath wins.
        result = _route_inter_section(edge, src, tgt, ctx)
        if result is None:
            result = _route_tb_internal(edge, src, tgt, ctx)
        if result is None:
            result = _route_tb_lr_exit(edge, src, tgt, ctx)
        if result is None:
            result = _route_tb_lr_entry(edge, src, tgt, ctx)
        if result is None:
            result = _route_perp_entry(edge, src, tgt, ctx)
        if result is None:
            result = _route_intra_section(edge, src, tgt, ctx)

        if result is not None:
            routes.append(result)

    _center_bubble_stations(routes, graph)

    return routes


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _build_routing_context(
    graph: MetroGraph,
    diagonal_run: float,
    curve_radius: float,
    station_offsets: dict[tuple[str, str], float] | None,
) -> _RoutingCtx:
    """Pre-compute all shared state for edge routing."""
    junction_ids = set(graph.junctions)

    # Fold edge: max X across all stations
    all_x = [s.x for s in graph.stations.values()]
    fold_x = max(all_x) if all_x else 0

    # Junctions fed by BOTTOM exit ports
    bottom_exit_junctions: set[str] = set()
    bottom_exit_junction_ports: dict[str, str] = {}
    for e in graph.edges:
        if e.target in junction_ids:
            port = graph.ports.get(e.source)
            if port and not port.is_entry and port.side == PortSide.BOTTOM:
                bottom_exit_junctions.add(e.target)
                bottom_exit_junction_ports[e.target] = e.source

    # Fork/join stations
    fork_targets: dict[str, set[str]] = defaultdict(set)
    join_sources: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        fork_targets[e.source].add(e.target)
        join_sources[e.target].add(e.source)
    fork_stations = {sid for sid, tgts in fork_targets.items() if len(tgts) > 1}
    join_stations = {sid for sid, srcs in join_sources.items() if len(srcs) > 1}

    # TB sections and their entry sides
    tb_sections = {sid for sid, s in graph.sections.items() if s.direction == "TB"}
    tb_right_entry: set[str] = set()
    for port in graph.ports.values():
        if (
            port.is_entry
            and port.side == PortSide.RIGHT
            and port.section_id in tb_sections
        ):
            tb_right_entry.add(port.section_id)

    # Bundle assignments and bypass gap indices
    line_priority = {lid: i for i, lid in enumerate(graph.lines.keys())}
    bundle_info = compute_bundle_info(
        graph, junction_ids, line_priority, bottom_exit_junctions
    )
    bypass_gap_idx = _compute_bypass_gap_indices(graph, junction_ids, line_priority)

    return _RoutingCtx(
        graph=graph,
        fold_x=fold_x,
        junction_ids=junction_ids,
        bottom_exit_junctions=bottom_exit_junctions,
        bottom_exit_junction_ports=bottom_exit_junction_ports,
        offset_step=OFFSET_STEP,
        fork_stations=fork_stations,
        join_stations=join_stations,
        tb_sections=tb_sections,
        tb_right_entry=tb_right_entry,
        bundle_info=bundle_info,
        bypass_gap_idx=bypass_gap_idx,
        station_offsets=station_offsets,
        diagonal_run=diagonal_run,
        curve_radius=curve_radius,
    )


# ---------------------------------------------------------------------------
# Offset helpers
# ---------------------------------------------------------------------------


def _get_offset(ctx: _RoutingCtx, station_id: str, line_id: str) -> float:
    """Get the station offset for a (station, line) pair, defaulting to 0."""
    if ctx.station_offsets:
        return ctx.station_offsets.get((station_id, line_id), 0.0)
    return 0.0


def _max_offset_at(ctx: _RoutingCtx, station_id: str) -> float:
    """Get the maximum offset across all lines at a station."""
    if not ctx.station_offsets:
        return 0.0
    all_offs = [
        ctx.station_offsets.get((station_id, lid), 0.0)
        for lid in ctx.graph.station_lines(station_id)
    ]
    return max(all_offs) if all_offs else 0.0


def _tb_x_offset(
    ctx: _RoutingCtx, station_id: str, line_id: str, section_id: str
) -> float:
    """Compute the TB-aware X offset for a station.

    RIGHT-entry sections use non-reversed offsets; others use reversed.
    """
    off = _get_offset(ctx, station_id, line_id)
    if section_id in ctx.tb_right_entry:
        return off
    return reversed_offset(off, _max_offset_at(ctx, station_id))


# ---------------------------------------------------------------------------
# Handler 1: Inter-section edges (port-to-port / junction)
# ---------------------------------------------------------------------------


def _route_inter_section(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route edges between ports/junctions using L-shapes (no diagonals)."""
    graph = ctx.graph
    is_inter = (src.is_port or edge.source in ctx.junction_ids) and (
        tgt.is_port or edge.target in ctx.junction_ids
    )
    if not is_inter:
        return None

    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx
    dy = ty - sy

    i, n = ctx.bundle_info.get((edge.source, edge.target, edge.line_id), (0, 1))

    # Check for TB BOTTOM exit
    src_port = graph.ports.get(edge.source)
    src_is_tb_bottom = (
        src_port is not None
        and not src_port.is_entry
        and src_port.side == PortSide.BOTTOM
        and src.section_id in ctx.tb_sections
    )

    # Resolve section columns for bypass detection
    src_col = _resolve_section_col(graph, src, ctx.junction_ids)
    tgt_col = _resolve_section_col(graph, tgt, ctx.junction_ids)
    needs_bypass = (
        src_col is not None
        and tgt_col is not None
        and abs(tgt_col - src_col) > 1
        and _has_intervening_sections(graph, src_col, tgt_col)
    )

    if abs(dy) < COORD_TOLERANCE_FINE and not needs_bypass:
        # Same Y: straight horizontal
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (tx, ty)],
            is_inter_section=True,
        )

    if src_is_tb_bottom and ctx.station_offsets:
        return _route_tb_bottom_exit(edge, src, tgt, ctx)

    if abs(dx) < COORD_TOLERANCE:
        # Same X: straight vertical drop
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (tx, ty)],
            is_inter_section=True,
        )

    if edge.source in ctx.bottom_exit_junctions:
        return _route_bottom_exit_junction(edge, src, tgt, i, n, ctx)

    if needs_bypass:
        return _route_bypass(edge, src, tgt, i, src_col, tgt_col, ctx)

    # TOP entry port: vertical-first L-shape so line drops into the
    # section rather than routing along its top edge.
    tgt_port = graph.ports.get(edge.target)
    if tgt_port and tgt_port.is_entry and tgt_port.side == PortSide.TOP:
        return _route_top_entry_l_shape(edge, src, tgt, i, n, ctx)

    # Near-vertical: junction to same-column entry with tiny horizontal
    # offset (just the junction margin).  The standard L-shape would
    # place the vertical channel on the wrong side (toward the target,
    # which is back inside the section).  Instead, route the channel
    # further into the inter-column gap (away from the target) so the
    # line continues in the junction's natural direction before dropping.
    if (
        edge.source in ctx.junction_ids
        and abs(dx) <= JUNCTION_MARGIN + COORD_TOLERANCE
        and abs(dy) > abs(dx) * 3
    ):
        delta, r_first, r_second = l_shape_radii(
            i,
            n,
            going_down=(dy > 0),
            offset_step=ctx.offset_step,
            base_radius=ctx.curve_radius,
        )
        # Push channel away from target into the inter-column gap.
        if dx < 0:
            vx = sx + ctx.curve_radius + ctx.offset_step + delta
        else:
            vx = sx - ctx.curve_radius - ctx.offset_step + delta
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (vx, sy), (vx, ty), (tx, ty)],
            is_inter_section=True,
            curve_radii=[r_first, r_second],
        )

    # RIGHT entry port with source to the LEFT: wrap the vertical
    # channel around the right side of the target section so the route
    # goes over the top and in from the right, rather than cutting
    # horizontally through the section interior.
    if not tgt_port:
        tgt_port = graph.ports.get(edge.target)
    if (
        tgt_port
        and tgt_port.is_entry
        and tgt_port.side == PortSide.RIGHT
        and dx > 0
    ):
        return _route_right_entry_wrap(edge, src, tgt, i, n, ctx)

    # Standard L-shape
    return _route_l_shape(edge, src, tgt, i, n, ctx)


def _route_tb_bottom_exit(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath:
    """Vertical drop from TB BOTTOM exit with X offsets."""
    x_off = _tb_x_offset(ctx, edge.source, edge.line_id, src.section_id)
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(src.x + x_off, src.y), (tgt.x + x_off, tgt.y)],
        is_inter_section=True,
        offsets_applied=True,
    )


def _route_bottom_exit_junction(
    edge: Edge, src: Station, tgt: Station, i: int, n: int, ctx: _RoutingCtx
) -> RoutedPath:
    """Vertical-first L-shape from bottom exit junction."""
    exit_pid = ctx.bottom_exit_junction_ports[edge.source]
    if ctx.station_offsets:
        exit_src = ctx.graph.stations.get(exit_pid)
        sec_id = exit_src.section_id if exit_src else ""
        x_off = _tb_x_offset(ctx, exit_pid, edge.line_id, sec_id or "")
    else:
        x_off = ((n - 1) / 2 - i) * ctx.offset_step

    tgt_off = _get_offset(ctx, edge.target, edge.line_id)
    r = ctx.curve_radius + x_off
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (src.x + x_off, src.y),
            (src.x + x_off, tgt.y + tgt_off),
            (tgt.x, tgt.y + tgt_off),
        ],
        is_inter_section=True,
        curve_radii=[r],
        offsets_applied=True,
    )


def _route_bypass(
    edge: Edge,
    src: Station,
    tgt: Station,
    i: int,
    src_col: int,
    tgt_col: int,
    ctx: _RoutingCtx,
) -> RoutedPath:
    """U-shaped bypass route around intervening sections."""
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx
    graph = ctx.graph

    ekey = (edge.source, edge.target, edge.line_id)
    g1_j, g1_n, g2_j, g2_n = ctx.bypass_gap_idx.get(ekey, (0, 1, 0, 1))

    nest_idx = max(i, g2_j)
    nest_offset = nest_idx * BYPASS_NEST_STEP
    base_y = bypass_bottom_y(graph, src_col, tgt_col, BYPASS_CLEARANCE)
    by = base_y + nest_offset

    base_bypass_offset = ctx.curve_radius + ctx.offset_step
    gap1_extra = g1_j * ctx.offset_step
    gap2_extra = g2_j * ctx.offset_step

    if dx > 0:
        gap1_base = (
            adjacent_column_gap_x(graph, src_col, src_col + 1) - base_bypass_offset
        )
        gap1_limit = sx + ctx.curve_radius
        # When gap is too narrow, fan out from the limit toward gap center
        if gap1_base - (g1_n - 1) * ctx.offset_step < gap1_limit:
            gap1_x = gap1_limit + (g1_n - 1 - g1_j) * ctx.offset_step
        else:
            gap1_x = gap1_base - gap1_extra

        gap2_base = (
            adjacent_column_gap_x(graph, tgt_col - 1, tgt_col) + base_bypass_offset
        )
        gap2_limit = tx - ctx.curve_radius
        # When gap is too narrow, fan out from the limit toward gap center
        if gap2_base + (g2_n - 1) * ctx.offset_step > gap2_limit:
            gap2_x = gap2_limit - (g2_n - 1 - g2_j) * ctx.offset_step
        else:
            gap2_x = gap2_base + gap2_extra
    else:
        gap1_base = (
            adjacent_column_gap_x(graph, src_col - 1, src_col) + base_bypass_offset
        )
        gap1_limit = sx - ctx.curve_radius
        if gap1_base + (g1_n - 1) * ctx.offset_step > gap1_limit:
            gap1_x = gap1_limit - (g1_n - 1 - g1_j) * ctx.offset_step
        else:
            gap1_x = gap1_base + gap1_extra

        gap2_base = (
            adjacent_column_gap_x(graph, tgt_col, tgt_col + 1) - base_bypass_offset
        )
        gap2_limit = tx + ctx.curve_radius
        if gap2_base - (g2_n - 1) * ctx.offset_step < gap2_limit:
            gap2_x = gap2_limit + (g2_n - 1 - g2_j) * ctx.offset_step
        else:
            gap2_x = gap2_base - gap2_extra

    r_bypass = ctx.curve_radius + max(gap1_extra, gap2_extra)

    # Apply per-line offsets directly so the renderer doesn't have to
    # guess which waypoints belong to the source vs target side.
    # (When source and target share the same base Y, the midpoint
    # heuristic in the renderer breaks.)
    src_off = _get_offset(ctx, edge.source, edge.line_id)
    tgt_off = _get_offset(ctx, edge.target, edge.line_id)

    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (sx, sy + src_off),
            (gap1_x, sy + src_off),
            (gap1_x, by),
            (gap2_x, by),
            (gap2_x, ty + tgt_off),
            (tx, ty + tgt_off),
        ],
        is_inter_section=True,
        curve_radii=[r_bypass, r_bypass, r_bypass, r_bypass],
        offsets_applied=True,
    )


def _route_l_shape(
    edge: Edge, src: Station, tgt: Station, i: int, n: int, ctx: _RoutingCtx
) -> RoutedPath:
    """Standard L-shape inter-section route with concentric arcs."""
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx
    dy = ty - sy

    delta, r_first, r_second = l_shape_radii(
        i,
        n,
        going_down=(dy > 0),
        offset_step=ctx.offset_step,
        base_radius=ctx.curve_radius,
    )
    max_r = ctx.curve_radius + (n - 1) * ctx.offset_step
    mid_x = inter_column_channel_x(
        ctx.graph, src, tgt, sx, tx, dx, max_r, ctx.offset_step
    )
    vx = mid_x + delta
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (vx, sy), (vx, ty), (tx, ty)],
        is_inter_section=True,
        curve_radii=[r_first, r_second],
    )


def _route_top_entry_l_shape(
    edge: Edge, src: Station, tgt: Station, i: int, n: int, ctx: _RoutingCtx
) -> RoutedPath:
    """Vertical-first L-shape for TOP entry ports.

    Routes via a short horizontal lead-in so the transition from any
    preceding horizontal edge (e.g. exit -> junction) curves smoothly
    into the vertical drop::

        (sx,sy) -> (lx, sy) -> (lx, hy) -> (tx, hy) -> (tx, ty)

    The horizontal run sits in the inter-row gap just above the target
    section so the line drops cleanly into the TOP port, mirroring how
    LEFT entry ports receive a vertical run in the inter-column gap.
    """
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx
    dy = ty - sy

    delta, r_first, r_second = l_shape_radii(
        i,
        n,
        going_down=(dy > 0),
        offset_step=ctx.offset_step,
        base_radius=ctx.curve_radius,
    )

    # Compute Y for the horizontal channel in the inter-row gap.
    mid_y = _inter_row_channel_y(ctx.graph, src, tgt, sy, ty, dy, ctx.curve_radius)
    hy = mid_y + delta

    # Horizontal lead-in: a short run in the target's direction so the
    # corner from horizontal to vertical gets a proper curve instead of
    # a sharp right angle at the junction point.
    r_lead = ctx.curve_radius
    if abs(dx) > r_lead:
        lead_sign = 1.0 if dx > 0 else -1.0
        lx = sx + lead_sign * r_lead
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (lx, sy), (lx, hy), (tx, hy), (tx, ty)],
            is_inter_section=True,
            curve_radii=[r_lead, r_first, r_second],
        )

    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (sx, hy), (tx, hy), (tx, ty)],
        is_inter_section=True,
        curve_radii=[r_first, r_second],
    )


def _route_right_entry_wrap(
    edge: Edge, src: Station, tgt: Station, i: int, n: int, ctx: _RoutingCtx
) -> RoutedPath:
    """Route to a RIGHT entry port by wrapping around the right side.

    When the source is to the LEFT of a RIGHT entry port, the standard
    L-shape would cut horizontally through the target section.  Instead,
    drop into the inter-row gap, run horizontally past the target
    section's right edge, then drop into the RIGHT entry port::

        (sx,sy) -> (sx, hy) -> (vx, hy) -> (vx, ty) -> (tx, ty)

    This avoids crossing through intervening sections.
    """
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dy = ty - sy

    delta, r_first, r_second = l_shape_radii(
        i,
        n,
        going_down=(dy > 0),
        offset_step=ctx.offset_step,
        base_radius=ctx.curve_radius,
    )

    # Horizontal channel Y: in the inter-row gap above the target section.
    hy = _inter_row_channel_y(
        ctx.graph, src, tgt, sy, ty, dy, ctx.curve_radius
    )
    hy += delta

    # Vertical channel X: just past the entry port in the inter-section gap.
    vx = tx + ctx.curve_radius + ctx.offset_step + delta

    # Short horizontal lead-in so the first corner (horizontal-to-vertical)
    # gets a smooth curve instead of a sharp right angle at the junction.
    r_lead = ctx.curve_radius
    lx = sx + r_lead
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (lx, sy), (lx, hy), (vx, hy), (vx, ty), (tx, ty)],
        is_inter_section=True,
        curve_radii=[r_lead, r_first, r_first, r_second],
    )


def _inter_row_channel_y(
    graph: MetroGraph,
    src: Station,
    tgt: Station,
    sy: float,
    ty: float,
    dy: float,
    max_r: float,
) -> float:
    """Compute Y for a horizontal channel in an inter-row gap.

    Vertical equivalent of ``inter_column_channel_x``: places the
    channel in the inter-row gap, above the target section's header
    (number badge + label rendered above bbox_y).
    """
    # Section headers are rendered above bbox_y by approximately
    # 2 * circle_radius + y_offset (~26px).  Keep the channel above
    # this zone with a small margin.
    HEADER_CLEARANCE = 30.0

    # Resolve sections for junction stations (section_id is None for
    # junctions; trace through edges to find a connected port's section).
    src_sec = _resolve_section(graph, src)
    tgt_sec = _resolve_section(graph, tgt)

    if src_sec and tgt_sec and src_sec.grid_row != tgt_sec.grid_row:
        src_row = src_sec.grid_row
        tgt_row = tgt_sec.grid_row

        if dy > 0:
            # Going down: gap between bottom of source row and top of target row
            row_bottom = max(
                (
                    s.bbox_y + s.bbox_h
                    for s in graph.sections.values()
                    if s.grid_row == src_row and s.bbox_h > 0
                ),
                default=sy,
            )
            row_top = min(
                (
                    s.bbox_y
                    for s in graph.sections.values()
                    if s.grid_row == tgt_row and s.bbox_h > 0
                ),
                default=ty,
            )
            # Place above the header zone
            header_top = row_top - HEADER_CLEARANCE
            return (row_bottom + header_top) / 2
        else:
            # Going up: gap between top of source row and bottom of target row
            row_top = min(
                (
                    s.bbox_y
                    for s in graph.sections.values()
                    if s.grid_row == src_row and s.bbox_h > 0
                ),
                default=sy,
            )
            row_bottom = max(
                (
                    s.bbox_y + s.bbox_h
                    for s in graph.sections.values()
                    if s.grid_row == tgt_row and s.bbox_h > 0
                ),
                default=ty,
            )
            header_bottom = row_bottom + HEADER_CLEARANCE
            return (row_top + header_bottom) / 2

    # Fallback: place near target, clearing the header zone
    if dy > 0:
        return ty - HEADER_CLEARANCE - max_r
    else:
        return ty + HEADER_CLEARANCE + max_r


def _resolve_section(graph: MetroGraph, station: Station):
    """Resolve a station's section, tracing through junctions if needed.

    For junctions (section_id is None), traces edges to find a connected
    port's section.  Prefers exit-port connections (incoming edges) so
    the junction resolves to the *upstream* section.
    """
    if station.section_id:
        return graph.sections.get(station.section_id)
    # Junction: prefer the section connected via an incoming edge
    # (exit_port -> junction), i.e. the upstream section.
    for e in graph.edges:
        if e.target == station.id:
            other = graph.stations.get(e.source)
            if other and other.section_id:
                sec = graph.sections.get(other.section_id)
                if sec:
                    return sec
    # Fall back to outgoing edges
    for e in graph.edges:
        if e.source == station.id:
            other = graph.stations.get(e.target)
            if other and other.section_id:
                sec = graph.sections.get(other.section_id)
                if sec:
                    return sec
    return None


# ---------------------------------------------------------------------------
# Handler 2: TB section internal edges
# ---------------------------------------------------------------------------


def _route_tb_internal(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route internal edges within TB sections as vertical drops."""
    graph = ctx.graph
    src_sec = src.section_id
    tgt_sec = tgt.section_id

    tgt_exit_port = graph.ports.get(edge.target)
    tgt_is_bottom_exit = (
        tgt_exit_port is not None
        and not tgt_exit_port.is_entry
        and tgt_exit_port.side == PortSide.BOTTOM
    )
    if not (
        src_sec
        and src_sec == tgt_sec
        and src_sec in ctx.tb_sections
        and not src.is_port
        and (not tgt.is_port or tgt_is_bottom_exit)
    ):
        return None

    x_src = _tb_x_offset(ctx, edge.source, edge.line_id, src_sec)
    x_tgt = _tb_x_offset(ctx, edge.target, edge.line_id, src_sec)

    sx = src.x + x_src
    sy = src.y
    tx = tgt.x + x_tgt
    ty = tgt.y
    dx = tx - sx

    # Different X tracks: route with vertical runs + 45-degree diagonal
    if abs(dx) >= COORD_TOLERANCE:
        return _route_tb_diagonal(edge, sx, sy, tx, ty, ctx)

    # Same track: straight vertical drop
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (tx, ty)],
        offsets_applied=True,
    )


def _route_tb_diagonal(
    edge: Edge,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    ctx: _RoutingCtx,
) -> RoutedPath:
    """Route TB edges with vertical runs and a 45-degree diagonal transition.

    Mirrors ``_route_diagonal()`` but with axes swapped: vertical runs at
    source and target connected by a 45-degree diagonal that shifts between
    X tracks.
    """
    dy = ty - sy
    sign = 1.0 if dy > 0 else -1.0
    half_diag = ctx.diagonal_run / 2
    min_straight = MIN_STRAIGHT_EDGE

    # Bias diagonal toward fork/join stations
    is_fork = edge.source in ctx.fork_stations
    is_join = edge.target in ctx.join_stations
    if is_fork:
        mid_y = sy + sign * (min_straight + half_diag)
    elif is_join:
        mid_y = ty - sign * (min_straight + half_diag)
    else:
        mid_y = (sy + ty) / 2

    diag_start_y = mid_y - sign * half_diag
    diag_end_y = mid_y + sign * half_diag

    # Clamp to ensure minimum straight vertical runs at endpoints
    if sign > 0:
        diag_start_y = max(diag_start_y, sy + min_straight)
        diag_end_y = min(diag_end_y, ty - min_straight)
        if diag_end_y < diag_start_y:
            midpoint = (diag_start_y + diag_end_y) / 2
            diag_start_y = diag_end_y = midpoint
    else:
        diag_start_y = min(diag_start_y, sy - min_straight)
        diag_end_y = max(diag_end_y, ty + min_straight)
        if diag_end_y > diag_start_y:
            midpoint = (diag_start_y + diag_end_y) / 2
            diag_start_y = diag_end_y = midpoint

    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (sx, diag_start_y), (tx, diag_end_y), (tx, ty)],
        offsets_applied=True,
    )


# ---------------------------------------------------------------------------
# Handler 3: TB section LEFT/RIGHT exit
# ---------------------------------------------------------------------------


def _route_tb_lr_exit(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route internal station -> LEFT/RIGHT exit port in a TB section."""
    graph = ctx.graph
    tgt_port = graph.ports.get(edge.target)
    tgt_is_lr_exit = (
        tgt_port is not None
        and not tgt_port.is_entry
        and tgt_port.side in (PortSide.LEFT, PortSide.RIGHT)
    )
    if not (
        tgt_is_lr_exit
        and not src.is_port
        and src.section_id in ctx.tb_sections
        and src.section_id == tgt.section_id
    ):
        return None

    src_off = _get_offset(ctx, edge.source, edge.line_id)
    max_src_off = _max_offset_at(ctx, edge.source)

    vert_x_off, horiz_y_off, r = tb_exit_corner(
        src_off,
        max_src_off,
        exit_right=(tgt_port.side == PortSide.RIGHT),
        base_radius=ctx.curve_radius,
    )
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (src.x + vert_x_off, src.y),
            (src.x + vert_x_off, tgt.y + horiz_y_off),
            (tgt.x, tgt.y + horiz_y_off),
        ],
        offsets_applied=True,
        curve_radii=[r],
    )


# ---------------------------------------------------------------------------
# Handler 4: TB section LEFT/RIGHT entry
# ---------------------------------------------------------------------------


def _route_tb_lr_entry(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route LEFT/RIGHT entry port -> internal station in a TB section."""
    graph = ctx.graph
    src_port = graph.ports.get(edge.source)
    if not (
        src_port
        and src_port.side in (PortSide.LEFT, PortSide.RIGHT)
        and src_port.is_entry
        and not tgt.is_port
        and src.section_id in ctx.tb_sections
    ):
        return None

    src_off = _get_offset(ctx, edge.source, edge.line_id)
    tgt_off = _get_offset(ctx, edge.target, edge.line_id)
    max_tgt_off = _max_offset_at(ctx, edge.target)

    vert_x_off, r = tb_entry_corner(
        tgt_off,
        max_tgt_off,
        entry_right=(src_port.side == PortSide.RIGHT),
        base_radius=ctx.curve_radius,
    )
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (src.x, src.y + src_off),
            (tgt.x + vert_x_off, src.y + src_off),
            (tgt.x + vert_x_off, tgt.y),
        ],
        offsets_applied=True,
        curve_radii=[r],
    )


# ---------------------------------------------------------------------------
# Handler 5: Perpendicular (TOP/BOTTOM) port entry to internal station
# ---------------------------------------------------------------------------


def _route_perp_entry(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route TOP/BOTTOM port -> internal station with upstream merging."""
    graph = ctx.graph
    src_port = graph.ports.get(edge.source)
    if not (
        src_port
        and src_port.side in (PortSide.TOP, PortSide.BOTTOM)
        and not tgt.is_port
    ):
        return None

    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx

    src_off = _get_offset(ctx, edge.source, edge.line_id)
    tgt_off = _get_offset(ctx, edge.target, edge.line_id)

    # Try to merge with upstream inter-section edge
    upstream_st = _find_upstream_for_merge(edge, src, ctx)

    if upstream_st is not None:
        return _route_perp_entry_merged(
            edge, src, tgt, upstream_st, src_off, tgt_off, ctx
        )

    if abs(dx) < COORD_TOLERANCE:
        # Nearly same X: straight vertical drop
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx + src_off, sy), (tx, ty + tgt_off)],
            offsets_applied=True,
        )

    # L-shape: vertical drop then horizontal to station
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (sx + src_off, sy),
            (sx + src_off, ty + tgt_off),
            (tx, ty + tgt_off),
        ],
        offsets_applied=True,
        curve_radii=[ctx.curve_radius + src_off],
    )


def _find_upstream_for_merge(
    edge: Edge, src: Station, ctx: _RoutingCtx
) -> Station | None:
    """Find an upstream station to merge with for combined L-shape routing.

    Returns the upstream station if merging is appropriate, or None.
    Adds the upstream edge to skip_edges when merging.
    """
    if not ctx.station_offsets:
        return None

    graph = ctx.graph
    for e2 in graph.edges:
        if e2.target != edge.source or e2.line_id != edge.line_id:
            continue
        u = graph.stations.get(e2.source)
        if not u:
            continue
        # Don't merge with TB BOTTOM exits
        u_port = graph.ports.get(e2.source)
        if (
            u_port
            and not u_port.is_entry
            and u_port.side == PortSide.BOTTOM
            and u.section_id in ctx.tb_sections
        ):
            continue
        # Only merge when upstream is at the same Y as the entry port
        if abs(u.y - src.y) > COORD_TOLERANCE:
            continue
        ctx.skip_edges.add((e2.source, e2.target, e2.line_id))
        return u

    return None


def _route_perp_entry_merged(
    edge: Edge,
    src: Station,
    tgt: Station,
    upstream_st: Station,
    src_off: float,
    tgt_off: float,
    ctx: _RoutingCtx,
) -> RoutedPath:
    """Route a combined inter-section + perpendicular entry as one L-shape."""
    graph = ctx.graph
    tx, ty = tgt.x, tgt.y
    up_y_off = _get_offset(ctx, upstream_st.id, edge.line_id)

    if abs(upstream_st.x - src.x) < COORD_TOLERANCE:
        # Same X: 4-point combined route through inter-column channel
        mid_x = inter_column_channel_x(
            graph,
            upstream_st,
            tgt,
            upstream_st.x,
            tgt.x,
            tgt.x - upstream_st.x,
            ctx.curve_radius,
            ctx.offset_step,
        )
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[
                (upstream_st.x, upstream_st.y + up_y_off),
                (mid_x + src_off, upstream_st.y + up_y_off),
                (mid_x + src_off, ty + tgt_off),
                (tx, ty + tgt_off),
            ],
            offsets_applied=True,
            curve_radii=[ctx.curve_radius, ctx.curve_radius + src_off],
        )

    # Different X (cross-column entry): 3-point L-shape
    max_tgt_off = _max_offset_at(ctx, edge.target)
    rev_tgt_off = reversed_offset(tgt_off, max_tgt_off)
    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[
            (upstream_st.x, upstream_st.y + up_y_off),
            (tx + rev_tgt_off, upstream_st.y + up_y_off),
            (tx + rev_tgt_off, ty + tgt_off),
        ],
        offsets_applied=True,
        curve_radii=[ctx.curve_radius + rev_tgt_off],
    )


# ---------------------------------------------------------------------------
# Handler 6: Intra-section edges (diagonal transitions, folds, straights)
# ---------------------------------------------------------------------------


def _route_intra_section(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath | None:
    """Route intra-section edges: diagonals, fold routing, straight lines."""
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx
    dy = ty - sy

    # Cross-row fold edge (skip for intra-section RL edges)
    same_section = src.section_id and src.section_id == tgt.section_id
    if dx <= 0 and abs(dy) > CROSS_ROW_THRESHOLD and not same_section:
        fold_right = ctx.fold_x + FOLD_MARGIN
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (fold_right, sy), (fold_right, ty), (tx, ty)],
        )

    # Same track: straight line
    if abs(sy - ty) < COORD_TOLERANCE_FINE:
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (tx, ty)],
        )

    # Near-zero X gap: straight line
    if abs(dx) < COORD_TOLERANCE:
        return RoutedPath(
            edge=edge,
            line_id=edge.line_id,
            points=[(sx, sy), (tx, ty)],
        )

    # Different tracks: horizontal -> diagonal -> horizontal
    return _route_diagonal(edge, src, tgt, ctx)


def _route_diagonal(
    edge: Edge, src: Station, tgt: Station, ctx: _RoutingCtx
) -> RoutedPath:
    """Route with horizontal runs and a 45-degree diagonal transition."""
    sx, sy = src.x, src.y
    tx, ty = tgt.x, tgt.y
    dx = tx - sx

    sign = 1.0 if dx > 0 else -1.0
    half_diag = ctx.diagonal_run / 2

    # Minimum straight track at endpoints
    if src.is_port or tgt.is_port:
        min_straight = ctx.curve_radius + MIN_STRAIGHT_PORT
    else:
        min_straight = MIN_STRAIGHT_EDGE

    # Extend straight run past labels at fork/join stations, but only
    # when there is enough horizontal room.  If label clearance would
    # collapse the diagonal to a near-vertical line, fall back to the
    # base min_straight so a proper diagonal can still be drawn.
    src_min = min_straight
    tgt_min = min_straight
    if edge.source in ctx.fork_stations and src.label.strip():
        src_min = max(min_straight, label_text_width(src.label) / 2)
    if edge.target in ctx.join_stations and tgt.label.strip():
        tgt_min = max(min_straight, label_text_width(tgt.label) / 2)
    if src_min + tgt_min + ctx.diagonal_run > abs(dx):
        src_min = min_straight
        tgt_min = min_straight

    # Bias diagonal toward the convergence/divergence station so that
    # the visual fork/join is close to the topological fork/join.
    is_fork = edge.source in ctx.fork_stations
    is_join = edge.target in ctx.join_stations
    if is_fork:
        mid_x = sx + sign * (src_min + half_diag)
    elif is_join:
        mid_x = tx - sign * (tgt_min + half_diag)
    else:
        mid_x = (sx + tx) / 2

    diag_start_x = mid_x - sign * half_diag
    diag_end_x = mid_x + sign * half_diag

    # Clamp to ensure label clearance
    if sign > 0:
        diag_start_x = max(diag_start_x, sx + src_min)
        diag_end_x = min(diag_end_x, tx - tgt_min)
        if diag_end_x < diag_start_x:
            midpoint = (diag_start_x + diag_end_x) / 2
            diag_start_x = diag_end_x = midpoint
    else:
        diag_start_x = min(diag_start_x, sx - src_min)
        diag_end_x = max(diag_end_x, tx + tgt_min)
        if diag_end_x > diag_start_x:
            midpoint = (diag_start_x + diag_end_x) / 2
            diag_start_x = diag_end_x = midpoint

    return RoutedPath(
        edge=edge,
        line_id=edge.line_id,
        points=[(sx, sy), (diag_start_x, sy), (diag_end_x, ty), (tx, ty)],
    )


# ---------------------------------------------------------------------------
# Post-processing: centre bubble stations on their flat segments
# ---------------------------------------------------------------------------


def _center_bubble_stations(routes: list[RoutedPath], graph: MetroGraph) -> None:
    """Shift diagonals so bubble stations sit centred on their flat segments.

    A "bubble station" branches off the trunk at a different Y, with a
    diagonal on each side.  The fork/join bias in ``_route_diagonal``
    keeps diagonals symmetric at the shared station but leaves the bubble
    station off-centre.  This pass detects such stations and shifts both
    adjacent diagonals by the same amount to equalise the flat runs.

    Skips stations whose neighbouring diagonal endpoints are bundle
    convergence/divergence points (multiple diagonal routes arriving or
    departing), since shifting individual diagonals in a bundle would
    break the visual alignment of parallel return paths.
    """
    # Identify fork/join stations from the full graph (all edge types, not
    # just 4-point diagonals).  A fork has multiple targets, a join has
    # multiple sources.  These must NOT be recentred as bubble stations.
    all_sources: dict[str, set[str]] = defaultdict(set)
    all_targets: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        all_targets[edge.source].add(edge.target)
        all_sources[edge.target].add(edge.source)

    # Index 4-point diagonal routes by their source and target station IDs.
    # 4-point routes have the shape: [flat, diag_start, diag_end, flat].
    incoming: dict[str, list[RoutedPath]] = defaultdict(list)
    outgoing: dict[str, list[RoutedPath]] = defaultdict(list)

    # Also count how many diagonal routes converge on / diverge from each
    # station, to detect bundle convergence/divergence points.
    diag_in_count: dict[str, int] = defaultdict(int)
    diag_out_count: dict[str, int] = defaultdict(int)

    for rp in routes:
        if len(rp.points) != 4:
            continue
        # Verify it really has a diagonal (Y changes between points 1 and 2)
        if abs(rp.points[1][1] - rp.points[2][1]) < COORD_TOLERANCE_FINE:
            continue
        incoming[rp.edge.target].append(rp)
        outgoing[rp.edge.source].append(rp)
        diag_in_count[rp.edge.target] += 1
        diag_out_count[rp.edge.source] += 1

    for sid, station in graph.stations.items():
        if station.is_port:
            continue

        # Skip fork/join stations - they sit at the trunk, not on a bubble.
        if len(all_targets.get(sid, set())) > 1 or len(all_sources.get(sid, set())) > 1:
            continue

        in_routes = incoming.get(sid, [])
        out_routes = outgoing.get(sid, [])

        # Only handle simple bubble stations: exactly one diagonal in,
        # one diagonal out, both at the station's Y level.
        if len(in_routes) != 1 or len(out_routes) != 1:
            continue

        in_rp = in_routes[0]
        out_rp = out_routes[0]

        # Skip if either neighbouring endpoint is a bundle convergence or
        # divergence point.  Shifting one diagonal in a bundle of parallel
        # return paths would misalign them visually.
        if diag_in_count.get(out_rp.edge.target, 0) > 1:
            continue
        if diag_out_count.get(in_rp.edge.source, 0) > 1:
            continue

        # Incoming route: [..., (diag_end_x, stn_y), (stn_x, stn_y)]
        # The flat at station Y runs from diag_end_x to station.x
        in_diag_end_x = in_rp.points[2][0]

        # Outgoing route: [(stn_x, stn_y), (diag_start_x, stn_y), ...]
        # The flat at station Y runs from station.x to diag_start_x
        out_diag_start_x = out_rp.points[1][0]

        in_flat = station.x - in_diag_end_x
        out_flat = out_diag_start_x - station.x

        # Skip if flats are already balanced or negligible
        if abs(in_flat) < 1 or abs(out_flat) < 1:
            continue
        if abs(in_flat - out_flat) < 1:
            continue

        # Shift both diagonals by the same amount to equalise the flats.
        # Positive shift moves diagonals toward the station's right.
        shift = (in_flat - out_flat) / 2

        # Shift incoming diagonal
        in_rp.points[1] = (in_rp.points[1][0] + shift, in_rp.points[1][1])
        in_rp.points[2] = (in_rp.points[2][0] + shift, in_rp.points[2][1])

        # Shift outgoing diagonal
        out_rp.points[1] = (out_rp.points[1][0] + shift, out_rp.points[1][1])
        out_rp.points[2] = (out_rp.points[2][0] + shift, out_rp.points[2][1])


# ---------------------------------------------------------------------------
# Utility functions (unchanged)
# ---------------------------------------------------------------------------


def _resolve_section_col(
    graph: MetroGraph,
    station: Station,
    junction_ids: set[str],
) -> int | None:
    """Resolve the grid column for a port or junction station.

    Ports have a section_id directly.  Junctions need to trace through
    edges to find a connected port's section.
    """
    if station.section_id:
        sec = graph.sections.get(station.section_id)
        if sec and sec.grid_col >= 0:
            return sec.grid_col
        return None

    if station.id in junction_ids:
        for e in graph.edges:
            other_id = None
            if e.source == station.id:
                other_id = e.target
            elif e.target == station.id:
                other_id = e.source
            if other_id:
                other = graph.stations.get(other_id)
                if other and other.section_id:
                    sec = graph.sections.get(other.section_id)
                    if sec and sec.grid_col >= 0:
                        return sec.grid_col
    return None


def _has_intervening_sections(
    graph: MetroGraph,
    src_col: int,
    tgt_col: int,
) -> bool:
    """Check if any sections exist in columns strictly between src and tgt."""
    lo, hi = min(src_col, tgt_col), max(src_col, tgt_col)
    for s in graph.sections.values():
        if s.bbox_w > 0 and lo < s.grid_col < hi:
            return True
    return False


def _compute_bypass_gap_indices(
    graph: MetroGraph,
    junction_ids: set[str],
    line_priority: dict[str, int] | None = None,
) -> dict[tuple[str, str, str], tuple[int, int, int, int]]:
    """Assign per-gap indices for bypass routes sharing physical gaps.

    Bypass routes from different source columns can share the same
    physical gap (e.g., routes from cols 1->4 and 2->4 both use the
    gap between cols 3 and 4 for their gap2 vertical).  This function
    groups bypass routes by their gap1 and gap2 column pairs and
    assigns per-gap indices so each route gets a unique X offset.

    Returns
    -------
    dict mapping (source_id, target_id, line_id) to
    (gap1_idx, gap1_count, gap2_idx, gap2_count).
    """
    EdgeKey = tuple[str, str, str]
    bypass_edges: list[tuple[EdgeKey, int, int, float]] = []

    for edge in graph.edges:
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue

        is_inter = (src.is_port or edge.source in junction_ids) and (
            tgt.is_port or edge.target in junction_ids
        )
        if not is_inter:
            continue

        src_col = _resolve_section_col(graph, src, junction_ids)
        tgt_col = _resolve_section_col(graph, tgt, junction_ids)
        if (
            src_col is None
            or tgt_col is None
            or abs(tgt_col - src_col) <= 1
            or not _has_intervening_sections(graph, src_col, tgt_col)
        ):
            continue

        dx = tgt.x - src.x
        ekey: EdgeKey = (edge.source, edge.target, edge.line_id)
        bypass_edges.append((ekey, src_col, tgt_col, dx))

    gap1_groups: dict[tuple[int, int], list[tuple[EdgeKey, int]]] = defaultdict(list)
    gap2_groups: dict[tuple[int, int], list[tuple[EdgeKey, int, str]]] = defaultdict(
        list
    )

    for ekey, src_col, tgt_col, dx in bypass_edges:
        line_id = ekey[2]
        if dx > 0:
            gap1_pair = (src_col, src_col + 1)
            gap2_pair = (tgt_col - 1, tgt_col)
        else:
            gap1_pair = (src_col - 1, src_col)
            gap2_pair = (tgt_col, tgt_col + 1)
        gap1_groups[gap1_pair].append((ekey, src_col))
        gap2_groups[gap2_pair].append((ekey, src_col, line_id))

    gap1_idx: dict[EdgeKey, tuple[int, int]] = {}
    gap2_idx: dict[EdgeKey, tuple[int, int]] = {}

    for group in gap1_groups.values():
        group.sort(key=lambda x: x[1])
        n = len(group)
        for j, (ek, _) in enumerate(group):
            gap1_idx[ek] = (j, n)

    lp = line_priority or {}
    for group in gap2_groups.values():
        # Sort by line priority so the lowest-offset line (highest
        # priority) gets the outermost vertical channel.  This
        # prevents crossings when lines converge at an entry port
        # from different source columns.
        group.sort(key=lambda x: lp.get(x[2], 0))
        n = len(group)
        for j, (ek, _sc, _lid) in enumerate(group):
            gap2_idx[ek] = (j, n)

    result: dict[EdgeKey, tuple[int, int, int, int]] = {}
    all_keys = set(gap1_idx) | set(gap2_idx)
    for ek in all_keys:
        g1_j, g1_n = gap1_idx.get(ek, (0, 1))
        g2_j, g2_n = gap2_idx.get(ek, (0, 1))
        result[ek] = (g1_j, g1_n, g2_j, g2_n)

    return result
