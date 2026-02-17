"""Core edge routing: the main route_edges() dispatcher.

Routes edges as horizontal segments with 45-degree diagonal transitions.
For folded layouts, cross-row edges route through the fold edge with a
clean vertical drop. Inter-section edges use L-shaped routing with
per-line bundle offsets.
"""

from __future__ import annotations

from collections import defaultdict

from nf_metro.layout.constants import (
    CHAR_WIDTH,
    COORD_TOLERANCE,
    COORD_TOLERANCE_FINE,
    CROSS_ROW_THRESHOLD,
    CURVE_RADIUS,
    DIAGONAL_RUN,
    FOLD_MARGIN,
    MIN_STRAIGHT_EDGE,
    MIN_STRAIGHT_PORT,
    OFFSET_STEP,
)
from nf_metro.layout.routing.common import (
    RoutedPath,
    compute_bundle_info,
    inter_column_channel_x,
)
from nf_metro.parser.model import MetroGraph, PortSide


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
    routes: list[RoutedPath] = []

    # Detect fold: find the max X position (fold edge)
    all_x = [s.x for s in graph.stations.values()]
    fold_x = max(all_x) if all_x else 0

    junction_ids = set(graph.junctions)

    # Identify junctions fed by BOTTOM exit ports for vertical-first routing
    bottom_exit_junctions: set[str] = set()
    bottom_exit_junction_ports: dict[str, str] = {}  # junction -> exit port
    for e in graph.edges:
        if e.target in junction_ids:
            port = graph.ports.get(e.source)
            if port and not port.is_entry and port.side == PortSide.BOTTOM:
                bottom_exit_junctions.add(e.target)
                bottom_exit_junction_ports[e.target] = e.source

    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}
    offset_step = OFFSET_STEP

    # Pre-compute fork and join stations for diagonal bias and label clearance
    _fork_targets: dict[str, set[str]] = defaultdict(set)
    _join_sources: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        _fork_targets[e.source].add(e.target)
        _join_sources[e.target].add(e.source)
    fork_stations = {sid for sid, tgts in _fork_targets.items() if len(tgts) > 1}
    join_stations = {sid for sid, srcs in _join_sources.items() if len(srcs) > 1}

    # Identify TB sections for special routing
    tb_sections = {sid for sid, s in graph.sections.items() if s.direction == "TB"}

    # TB sections entered from the RIGHT side need non-reversed X offsets
    # on their internal vertical edges (to stay consistent with the
    # direction-aware entry/exit L-shapes).
    tb_right_entry: set[str] = set()
    for port in graph.ports.values():
        if (
            port.is_entry
            and port.side == PortSide.RIGHT
            and port.section_id in tb_sections
        ):
            tb_right_entry.add(port.section_id)

    # Pre-compute bundle assignments: groups inter-section edges that
    # share the same vertical channel so they get consistent per-line
    # X offsets instead of overlapping.
    bundle_info = compute_bundle_info(
        graph, junction_ids, line_priority, bottom_exit_junctions
    )

    # Edges absorbed into a combined inter-section + entry route
    skip_edges: set[tuple[str, str, str]] = set()

    for edge in graph.edges:
        if (edge.source, edge.target, edge.line_id) in skip_edges:
            continue

        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue

        sx, sy = src.x, src.y
        tx, ty = tgt.x, tgt.y
        dx = tx - sx
        dy = ty - sy

        # Inter-section routing: edges between ports/junctions use only
        # horizontal and vertical segments (L-shapes), no diagonals.
        is_inter_section = (src.is_port or edge.source in junction_ids) and (
            tgt.is_port or edge.target in junction_ids
        )
        if is_inter_section:
            i, n = bundle_info.get((edge.source, edge.target, edge.line_id), (0, 1))

            # Check for TB BOTTOM exit first: route as a near-vertical
            # drop with reversed X offsets even with a small X offset
            # (entry port may be at section right edge for RL targets).
            src_port_obj2 = graph.ports.get(edge.source)
            src_is_tb_bottom = (
                src_port_obj2 is not None
                and not src_port_obj2.is_entry
                and src_port_obj2.side == PortSide.BOTTOM
                and src.section_id in tb_sections
            )

            if abs(dy) < COORD_TOLERANCE_FINE:
                # Same Y: straight horizontal
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx, sy), (tx, ty)],
                        is_inter_section=True,
                    )
                )
            elif src_is_tb_bottom and station_offsets:
                # Vertical drop from TB BOTTOM exit with X offsets.
                # RIGHT-entry sections use non-reversed; others reversed.
                src_off = station_offsets.get((edge.source, edge.line_id), 0.0)
                if src.section_id in tb_right_entry:
                    x_off = src_off
                else:
                    all_offs = [
                        station_offsets.get((edge.source, lid), 0.0)
                        for lid in graph.station_lines(edge.source)
                    ]
                    max_off = max(all_offs) if all_offs else 0.0
                    x_off = max_off - src_off
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx + x_off, sy), (tx + x_off, ty)],
                        is_inter_section=True,
                        offsets_applied=True,
                    )
                )
            elif abs(dx) < COORD_TOLERANCE:
                # Same X: straight vertical drop
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx, sy), (tx, ty)],
                        is_inter_section=True,
                    )
                )
            elif edge.source in bottom_exit_junctions:
                # Vertical-first L-shape from bottom exit junction:
                # drop to target Y, then horizontal to target.
                # Use exit port's station offsets for X continuity
                # with the exit_port -> junction segment above.
                exit_pid = bottom_exit_junction_ports[edge.source]
                if station_offsets:
                    src_off = station_offsets.get((exit_pid, edge.line_id), 0.0)
                    exit_src = graph.stations.get(exit_pid)
                    if exit_src and exit_src.section_id in tb_right_entry:
                        x_off = src_off
                    else:
                        all_offs = [
                            station_offsets.get((exit_pid, lid), 0.0)
                            for lid in graph.station_lines(exit_pid)
                        ]
                        max_off = max(all_offs) if all_offs else 0.0
                        x_off = max_off - src_off
                else:
                    x_off = ((n - 1) / 2 - i) * offset_step
                # Manually apply target entry port Y offset (the
                # renderer can't do it because offsets_applied=True,
                # which is needed since the source X offsets are TB).
                tgt_off = (
                    station_offsets.get((edge.target, edge.line_id), 0.0)
                    if station_offsets
                    else 0.0
                )
                r = curve_radius + x_off
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[
                            (sx + x_off, sy),
                            (sx + x_off, ty + tgt_off),
                            (tx, ty + tgt_off),
                        ],
                        is_inter_section=True,
                        curve_radii=[r],
                        offsets_applied=True,
                    )
                )
            else:
                # L-shape: vertical bundle between source and target,
                # with per-line offsets for visual separation.
                # Direction-aware: preserve top-to-bottom ordering as
                # left-to-right when the bundle turns upward, and as
                # right-to-left when it turns downward.
                if dy < 0:
                    # Going UP: top line (i=0) -> leftmost
                    delta = (i - (n - 1) / 2) * offset_step
                    # Bottom corner (right-to-up): concentric arcs
                    r_first = curve_radius + i * offset_step
                    # Top corner (up-to-right): concentric arcs,
                    # leftmost (i=0) is outermost so gets largest radius
                    r_second = curve_radius + (n - 1 - i) * offset_step
                else:
                    # Going DOWN: top line (i=0) -> rightmost
                    delta = ((n - 1) / 2 - i) * offset_step
                    r_first = curve_radius + (n - 1 - i) * offset_step
                    r_second = curve_radius + i * offset_step

                # Place vertical channel in the inter-column gap so it
                # doesn't pass through sibling sections stacked in the
                # source's column.
                max_r = curve_radius + (n - 1) * offset_step
                mid_x = inter_column_channel_x(
                    graph, src, tgt, sx, tx, dx, max_r, offset_step
                )
                vx = mid_x + delta
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[
                            (sx, sy),
                            (vx, sy),
                            (vx, ty),
                            (tx, ty),
                        ],
                        is_inter_section=True,
                        curve_radii=[r_first, r_second],
                    )
                )
            continue

        # TB section internal edges: L-shaped elbows and vertical runs
        # with per-line offsets pre-applied to the correct axis.
        # Also includes edges to BOTTOM exit ports so the vertical run
        # continues straight down without an offset discontinuity.
        src_sec = src.section_id
        tgt_sec = tgt.section_id
        tgt_exit_port = graph.ports.get(edge.target)
        tgt_is_bottom_exit = (
            tgt_exit_port is not None
            and not tgt_exit_port.is_entry
            and tgt_exit_port.side == PortSide.BOTTOM
        )
        if (
            src_sec
            and src_sec == tgt_sec
            and src_sec in tb_sections
            and not src.is_port
            and (not tgt.is_port or tgt_is_bottom_exit)
        ):
            src_off = (
                station_offsets.get((edge.source, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            tgt_off = (
                station_offsets.get((edge.target, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            # TB sections: internal edges are vertical drops with X offsets.
            # LEFT-entry sections use reversed offsets; RIGHT-entry sections
            # use non-reversed offsets (matching their L-shape corners).
            if src_sec in tb_right_entry:
                x_src = src_off
                x_tgt = tgt_off
            else:

                def _reverse_off(station_id: str, off: float) -> float:
                    all_offs = (
                        [
                            station_offsets.get((station_id, lid), 0.0)
                            for lid in graph.station_lines(station_id)
                        ]
                        if station_offsets
                        else []
                    )
                    max_off = max(all_offs) if all_offs else 0.0
                    return max_off - off

                x_src = _reverse_off(edge.source, src_off)
                x_tgt = _reverse_off(edge.target, tgt_off)
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx + x_src, sy),
                        (tx + x_tgt, ty),
                    ],
                    offsets_applied=True,
                )
            )
            continue

        # Internal station -> LEFT/RIGHT exit port in a TB section:
        # L-shaped exit run (vertical drop then horizontal to exit port).
        # Lines go straight down from the last station, turn the corner,
        # and exit through the port at the return row level.
        tgt_port_obj = graph.ports.get(edge.target)
        tgt_is_lr_exit = (
            tgt_port_obj is not None
            and not tgt_port_obj.is_entry
            and tgt_port_obj.side in (PortSide.LEFT, PortSide.RIGHT)
        )
        if (
            tgt_is_lr_exit
            and not src.is_port
            and src.section_id in tb_sections
            and src.section_id == tgt.section_id
        ):
            src_off = (
                station_offsets.get((edge.source, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            # Reverse source offset for vertical segment (TB convention).
            # Skip reversal for already-reversed sections: their internal
            # offsets already account for the reversal, so applying it
            # again would double-reverse and create line crossings.
            all_src_offs = (
                [
                    station_offsets.get((edge.source, lid), 0.0)
                    for lid in graph.station_lines(edge.source)
                ]
                if station_offsets
                else []
            )
            max_src_off = max(all_src_offs) if all_src_offs else 0.0
            rev_src_off = max_src_off - src_off
            # Direction-aware X offset: LEFT port = DOWN-to-LEFT turn
            # (use reversed), RIGHT port = DOWN-to-RIGHT turn (use
            # non-reversed).  For reversed TB sections the station
            # offsets are already flipped, so the standard formula
            # produces the correct un-reversed exit ordering.
            if tgt_port_obj.side == PortSide.RIGHT:
                vert_x_off = src_off
            else:
                vert_x_off = rev_src_off
            # Concentric corner: the outermost vertical line gets the
            # largest curve radius, producing nested arcs.  horiz_y_off
            # uses rev_src_off for both sides so the radius formula
            # (curve_radius + rev_src_off) maps outer -> larger radius.
            horiz_y_off = rev_src_off
            # L-shape: vertical from station, curve, horizontal to port
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx + vert_x_off, sy),
                        (sx + vert_x_off, ty + horiz_y_off),
                        (tx, ty + horiz_y_off),
                    ],
                    offsets_applied=True,
                    curve_radii=[curve_radius + rev_src_off],
                )
            )
            continue

        # LEFT/RIGHT entry port -> internal station in a TB section:
        # L-shaped entry run (horizontal then curve then vertical drop).
        # The port sits above the first station so the turn happens
        # before the station, not at it.
        src_port_obj = graph.ports.get(edge.source)
        if (
            src_port_obj
            and src_port_obj.side in (PortSide.LEFT, PortSide.RIGHT)
            and src_port_obj.is_entry
            and not tgt.is_port
            and src.section_id in tb_sections
        ):
            src_off = (
                station_offsets.get((edge.source, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            tgt_off = (
                station_offsets.get((edge.target, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            # Reverse target offset for vertical segment (TB convention)
            all_tgt_offs = (
                [
                    station_offsets.get((edge.target, lid), 0.0)
                    for lid in graph.station_lines(edge.target)
                ]
                if station_offsets
                else []
            )
            max_tgt_off = max(all_tgt_offs) if all_tgt_offs else 0.0
            rev_tgt_off = max_tgt_off - tgt_off
            # Direction-aware X offset: LEFT port = RIGHT-to-DOWN turn
            # (use reversed), RIGHT port = LEFT-to-DOWN turn (use
            # non-reversed).
            if src_port_obj.side == PortSide.RIGHT:
                vert_x_off = tgt_off
            else:
                vert_x_off = rev_tgt_off

            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx, sy + src_off),
                        (tx + vert_x_off, sy + src_off),
                        (tx + vert_x_off, ty),
                    ],
                    offsets_applied=True,
                    curve_radii=[curve_radius + rev_tgt_off],
                )
            )
            continue

        # TOP/BOTTOM port -> internal station: L-shaped elbow
        # (e.g., vertical entry from above curving into an RL section).
        # Station offsets from compute_station_offsets already account
        # for reversed sections (fed by TB BOTTOM exits), so we use
        # src_off directly as the X offset on the vertical segment.
        if (
            src_port_obj
            and src_port_obj.side in (PortSide.TOP, PortSide.BOTTOM)
            and not tgt.is_port
        ):
            src_off = (
                station_offsets.get((edge.source, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )
            tgt_off = (
                station_offsets.get((edge.target, edge.line_id), 0.0)
                if station_offsets
                else 0.0
            )

            # Check for an upstream inter-section edge feeding this
            # entry port.  Combine into one L-shape with proper curves
            # instead of two disjoint segments with a right-angle seam.
            # Skip TB BOTTOM exit ports: their vertical drop should stay
            # separate so lines fall straight down between sections.
            upstream_st = None
            if station_offsets:
                for e2 in graph.edges:
                    if e2.target == edge.source and e2.line_id == edge.line_id:
                        u = graph.stations.get(e2.source)
                        if u:
                            # Don't merge with TB BOTTOM exits - keep the
                            # clean vertical drop as a separate segment.
                            u_port = graph.ports.get(e2.source)
                            if (
                                u_port
                                and not u_port.is_entry
                                and u_port.side == PortSide.BOTTOM
                                and u.section_id in tb_sections
                            ):
                                continue
                            # Only merge when the upstream source is at the
                            # same Y as the entry port.  Cross-column sources
                            # at different Y levels must stay separate so
                            # the inter-section L-shape converges them at the
                            # entry port Y first.
                            if abs(u.y - sy) > 1.0:
                                continue
                            upstream_st = u
                            skip_edges.add((e2.source, e2.target, e2.line_id))
                            break

            if upstream_st is not None:
                up_y_off = station_offsets.get((upstream_st.id, edge.line_id), 0.0)
                if abs(upstream_st.x - sx) < COORD_TOLERANCE:
                    # Same X: 4-point combined route through
                    # inter-column channel (vertical drop case).
                    mid_x = inter_column_channel_x(
                        graph,
                        upstream_st,
                        tgt,
                        upstream_st.x,
                        tgt.x,
                        tgt.x - upstream_st.x,
                        curve_radius,
                        offset_step,
                    )
                    routes.append(
                        RoutedPath(
                            edge=edge,
                            line_id=edge.line_id,
                            points=[
                                (upstream_st.x, upstream_st.y + up_y_off),
                                (mid_x + src_off, upstream_st.y + up_y_off),
                                (mid_x + src_off, ty + tgt_off),
                                (tx, ty + tgt_off),
                            ],
                            offsets_applied=True,
                            curve_radii=[
                                curve_radius,
                                curve_radius + src_off,
                            ],
                        )
                    )
                else:
                    # Different X (cross-column entry): 3-point L-shape
                    # horizontal from upstream -> curve -> vertical to
                    # target.  Reverse target offset for TB convention.
                    all_tgt_offs = (
                        [
                            station_offsets.get((edge.target, lid), 0.0)
                            for lid in graph.station_lines(edge.target)
                        ]
                        if station_offsets
                        else []
                    )
                    max_tgt_off = max(all_tgt_offs) if all_tgt_offs else 0.0
                    rev_tgt_off = max_tgt_off - tgt_off
                    routes.append(
                        RoutedPath(
                            edge=edge,
                            line_id=edge.line_id,
                            points=[
                                (upstream_st.x, upstream_st.y + up_y_off),
                                (tx + rev_tgt_off, upstream_st.y + up_y_off),
                                (tx + rev_tgt_off, ty + tgt_off),
                            ],
                            offsets_applied=True,
                            curve_radii=[curve_radius + rev_tgt_off],
                        )
                    )
            elif abs(dx) < COORD_TOLERANCE:
                # Nearly same X: straight vertical drop
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx + src_off, sy), (tx, ty + tgt_off)],
                        offsets_applied=True,
                    )
                )
            else:
                # L-shape: vertical drop then horizontal to station
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[
                            (sx + src_off, sy),
                            (sx + src_off, ty + tgt_off),
                            (tx, ty + tgt_off),
                        ],
                        offsets_applied=True,
                        curve_radii=[curve_radius + src_off],
                    )
                )
            continue

        # Detect cross-row edge: target is to the left (only in folded layouts)
        is_cross_row = dx <= 0 and abs(dy) > CROSS_ROW_THRESHOLD

        if is_cross_row:
            # Route through fold edge: horizontal to fold, vertical drop,
            # horizontal to target
            fold_margin = FOLD_MARGIN
            fold_right = fold_x + fold_margin
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx, sy),
                        (fold_right, sy),
                        (fold_right, ty),
                        (tx, ty),
                    ],
                )
            )
        elif abs(sy - ty) < COORD_TOLERANCE_FINE:
            # Same track: straight line
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                )
            )
        else:
            # Different tracks: horizontal, diagonal, horizontal
            if abs(dx) < COORD_TOLERANCE:
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx, sy), (tx, ty)],
                    )
                )
                continue

            sign = 1.0 if dx > 0 else -1.0
            half_diag = diagonal_run / 2

            # Minimum straight track at each station endpoint.
            # Port-adjacent edges need more room so stations sit on
            # visible straight track, not on a curve.
            if src.is_port or tgt.is_port:
                min_straight = curve_radius + MIN_STRAIGHT_PORT
            else:
                min_straight = MIN_STRAIGHT_EDGE

            # At fork/join stations, extend the straight run past the
            # label so diverging/converging diagonals don't cross
            # through the label text.
            src_min = min_straight
            tgt_min = min_straight
            if edge.source in fork_stations and src.label.strip():
                src_min = max(min_straight, len(src.label) * CHAR_WIDTH / 2)
            if edge.target in join_stations and tgt.label.strip():
                tgt_min = max(min_straight, len(tgt.label) * CHAR_WIDTH / 2)

            # Bias diagonal toward source at fork points so the
            # visual divergence happens near the fork, avoiding
            # diagonals that pass through intermediate stations.
            if edge.source in fork_stations:
                mid_x = sx + sign * (src_min + half_diag)
            else:
                mid_x = (sx + tx) / 2

            diag_start_x = mid_x - sign * half_diag
            diag_end_x = mid_x + sign * half_diag

            # Clamp to ensure label clearance at each station.
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

            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[
                        (sx, sy),
                        (diag_start_x, sy),
                        (diag_end_x, ty),
                        (tx, ty),
                    ],
                )
            )

    return routes
