"""Edge routing for metro map layout.

Routes edges as horizontal segments with 45-degree diagonal transitions.
For folded layouts, cross-row edges route through the fold edge with a
clean vertical drop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nf_metro.parser.model import Edge, MetroGraph, PortSide


@dataclass
class RoutedPath:
    """A routed path for an edge, consisting of (x, y) waypoints."""

    edge: Edge
    line_id: str
    points: list[tuple[float, float]]
    is_inter_section: bool = False
    curve_radii: list[float] | None = None
    offsets_applied: bool = False


def route_edges(
    graph: MetroGraph,
    diagonal_run: float = 30.0,
    curve_radius: float = 10.0,
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

    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}
    offset_step = 3.0

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
    bundle_info = _compute_bundle_info(graph, junction_ids, line_priority)

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

            if abs(dy) < 0.01:
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
            elif abs(dx) < 1.0:
                # Same X: straight vertical drop
                routes.append(
                    RoutedPath(
                        edge=edge,
                        line_id=edge.line_id,
                        points=[(sx, sy), (tx, ty)],
                        is_inter_section=True,
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
                mid_x = _inter_column_channel_x(
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

        # Internal station → LEFT/RIGHT exit port in a TB section:
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
            # Reverse source offset for vertical segment (TB convention)
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
            # non-reversed).
            if tgt_port_obj.side == PortSide.RIGHT:
                vert_x_off = src_off
            else:
                vert_x_off = rev_src_off
            # Concentric corner: the outermost vertical line gets the
            # largest curve radius, producing nested arcs.  horiz_y_off
            # uses rev_src_off for both sides so the radius formula
            # (curve_radius + rev_src_off) maps outer → larger radius.
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

        # LEFT/RIGHT entry port → internal station in a TB section:
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

        # TOP/BOTTOM port → internal station: L-shaped elbow
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

            # Check for an upstream inter-section edge at the same X
            # (e.g. junction → TOP entry where both sit in the column
            # gap).  Combine into one L-shape with proper curves
            # instead of two disjoint segments with a right-angle seam.
            upstream_st = None
            if station_offsets:
                for e2 in graph.edges:
                    if e2.target == edge.source and e2.line_id == edge.line_id:
                        u = graph.stations.get(e2.source)
                        if u and abs(u.x - sx) < 1.0:
                            upstream_st = u
                            skip_edges.add(
                                (e2.source, e2.target, e2.line_id)
                            )
                            break

            if upstream_st is not None:
                # Combined route: upstream station → horizontal →
                # vertical → horizontal → internal station.
                up_y_off = station_offsets.get(
                    (upstream_st.id, edge.line_id), 0.0
                )
                mid_x = _inter_column_channel_x(
                    graph, upstream_st, tgt, upstream_st.x, tgt.x,
                    tgt.x - upstream_st.x, curve_radius, offset_step,
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
            elif abs(dx) < 1.0:
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
        is_cross_row = dx <= 0 and abs(dy) > 80

        if is_cross_row:
            # Route through fold edge: horizontal to fold, vertical drop,
            # horizontal to target
            fold_margin = 30
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
        elif abs(sy - ty) < 0.01:
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
            if abs(dx) < 1.0:
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

            mid_x = (sx + tx) / 2
            diag_start_x = mid_x - sign * half_diag
            diag_end_x = mid_x + sign * half_diag

            # Ensure minimum straight track at each station.
            # Port-adjacent edges need more room so stations sit on
            # visible straight track, not on a curve.
            if src.is_port or tgt.is_port:
                min_straight = curve_radius + 5
            else:
                min_straight = 10
            if sign > 0:
                diag_start_x = max(diag_start_x, sx + min_straight)
                diag_end_x = min(diag_end_x, tx - min_straight)
            else:
                diag_start_x = min(diag_start_x, sx - min_straight)
                diag_end_x = max(diag_end_x, tx + min_straight)

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


def _compute_bundle_info(
    graph: MetroGraph,
    junction_ids: set[str],
    line_priority: dict[str, int],
) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Pre-compute bundle assignments for inter-section edges.

    Groups inter-section edges that share the same geometric corridor
    (same vertical channel position and direction) and assigns consistent
    per-line positions within each bundle. This ensures lines traveling
    between sections are visually parallel with proper spacing, rather
    than overlapping at the same X coordinate.

    Returns dict mapping (source_id, target_id, line_id) -> (index, count).
    """
    # Collect all inter-section edges with their geometry
    inter_edges: list[tuple[Edge, float, float, float, float]] = []
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

        inter_edges.append((edge, src.x, src.y, tgt.x, tgt.y))

    # Group by corridor: edges sharing the same vertical channel
    # Key: (route_type, rounded_channel_position, vertical_direction)
    corridor_groups: dict[
        tuple[str, int, int], list[tuple[Edge, float, float, float, float]]
    ] = defaultdict(list)

    for item in inter_edges:
        edge, sx, sy, tx, ty = item
        dx = tx - sx
        dy = ty - sy

        if abs(dy) < 0.01:
            continue  # Horizontal edges don't need bundling

        v_dir = 1 if dy > 0 else -1

        if abs(dx) < 1.0:
            # Vertical: group by shared X position
            key = ("V", round(sx), v_dir)
        else:
            # L-shaped: group by source X and horizontal direction.
            # The vertical channel is placed in the inter-column gap
            # near the source, so source X is the right grouping key.
            # Using midpoint (sx+tx)/2 fails for junction fan-outs
            # where targets are at different X positions but share
            # the same vertical channel.
            h_dir = 1 if dx > 0 else -1
            key = ("L", round(sx), v_dir, h_dir)

        corridor_groups[key].append(item)

    # Assign per-line positions within each corridor
    assignments: dict[tuple[str, str, str], tuple[int, int]] = {}

    for _key, group in corridor_groups.items():
        # Sort by spatial ordering so the bundle's visual position
        # is preserved around corners.
        source_ids = {e[0].source for e in group}
        if len(source_ids) == 1:
            # All edges share the same source port: sort by the
            # connected internal station's Y (consistent with
            # compute_station_offsets).
            exit_port_id = group[0][0].source
            port = graph.ports.get(exit_port_id)
            if port and not port.is_entry:
                source_y = _line_source_y_at_port(exit_port_id, graph)
                group.sort(
                    key=lambda e: (
                        source_y.get(e[0].line_id, 0),
                        line_priority.get(e[0].line_id, 999),
                    )
                )
            else:
                group.sort(key=lambda e: line_priority.get(e[0].line_id, 999))
        else:
            # Fan-in: edges from different source ports. Sort by
            # actual source Y position to preserve spatial ordering
            # around the L-shaped corner.
            group.sort(key=lambda e: (e[2], line_priority.get(e[0].line_id, 999)))

        n = len(group)
        for i, (edge, *_rest) in enumerate(group):
            assignments[(edge.source, edge.target, edge.line_id)] = (i, n)

    return assignments


def _inter_column_channel_x(
    graph: MetroGraph,
    src,
    tgt,
    sx: float,
    tx: float,
    dx: float,
    max_r: float,
    offset_step: float,
) -> float:
    """Compute the X position for a vertical channel in an L-shaped route.

    Places the channel in the gap between columns so it doesn't pass
    through sibling sections stacked in the source's column. Falls
    back to near-source placement when section info is unavailable.
    """
    src_sec = graph.sections.get(src.section_id) if src.section_id else None
    tgt_sec = graph.sections.get(tgt.section_id) if tgt.section_id else None

    if src_sec and tgt_sec and src_sec.grid_col != tgt_sec.grid_col:
        # Find the rightmost/leftmost edges of the source and target
        # columns (accounting for sibling sections that may be wider).
        src_col = src_sec.grid_col
        tgt_col = tgt_sec.grid_col

        if dx > 0:
            col_right = max(
                (
                    s.bbox_x + s.bbox_w
                    for s in graph.sections.values()
                    if s.grid_col == src_col and s.bbox_w > 0
                ),
                default=sx,
            )
            col_left = min(
                (
                    s.bbox_x
                    for s in graph.sections.values()
                    if s.grid_col == tgt_col and s.bbox_w > 0
                ),
                default=tx,
            )
            return (col_right + col_left) / 2
        else:
            col_left = min(
                (
                    s.bbox_x
                    for s in graph.sections.values()
                    if s.grid_col == src_col and s.bbox_w > 0
                ),
                default=sx,
            )
            col_right = max(
                (
                    s.bbox_x + s.bbox_w
                    for s in graph.sections.values()
                    if s.grid_col == tgt_col and s.bbox_w > 0
                ),
                default=tx,
            )
            return (col_left + col_right) / 2

    # Fallback: place near source
    if dx > 0:
        return sx + max_r + offset_step
    else:
        return sx - max_r - offset_step


def route_inter_section_edges(
    graph: MetroGraph,
    diagonal_run: float = 30.0,
    curve_radius: float = 10.0,
) -> list[RoutedPath]:
    """Route edges that cross section boundaries (port-to-port).

    These are edges where both source and target are ports (or one is a port).
    Uses the same routing style as regular edges.
    """
    routes: list[RoutedPath] = []

    for edge in graph.edges:
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue

        # Only route edges that cross section boundaries:
        # - Both endpoints in different sections, OR
        # - One endpoint is a junction (section_id=None, e.g. divergence point)
        src_sec = src.section_id
        tgt_sec = tgt.section_id
        is_junction_edge = src_sec is None or tgt_sec is None
        if not is_junction_edge and (src_sec == tgt_sec):
            continue

        sx, sy = src.x, src.y
        tx, ty = tgt.x, tgt.y
        dx = tx - sx
        dy = ty - sy

        if abs(dy) < 0.01:
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                )
            )
        elif abs(dx) < 1.0:
            routes.append(
                RoutedPath(
                    edge=edge,
                    line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                )
            )
        else:
            sign = 1.0 if dx > 0 else -1.0
            half_diag = diagonal_run / 2

            mid_x = (sx + tx) / 2
            diag_start_x = mid_x - sign * half_diag
            diag_end_x = mid_x + sign * half_diag

            # Ensure minimum straight track at each station so the
            # station sits on a visible horizontal segment, not a curve.
            min_straight = curve_radius + 15
            if sign > 0:
                diag_start_x = max(diag_start_x, sx + min_straight)
                diag_end_x = min(diag_end_x, tx - min_straight)
            else:
                diag_start_x = min(diag_start_x, sx - min_straight)
                diag_end_x = max(diag_end_x, tx + min_straight)

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


def compute_station_offsets(
    graph: MetroGraph,
    offset_step: float = 3.0,
) -> dict[tuple[str, str], float]:
    """Compute per-station Y offsets for each line.

    Each line gets a globally consistent offset based on its declaration
    order (priority). This ensures lines maintain their position within
    bundles across all sections - when a line splits off and later
    rejoins, it returns to its reserved slot rather than shifting.

    For reversed sections (fed by a TB section's BOTTOM exit), offsets
    are flipped so the bundle ordering matches the reversed spatial flow.

    Returns dict mapping (station_id, line_id) -> y_offset.
    """
    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}
    max_priority = len(line_order) - 1 if line_order else 0

    reversed_sections = _detect_reversed_sections(graph)

    offsets: dict[tuple[str, str], float] = {}
    for sid in graph.stations:
        lines = graph.station_lines(sid)
        if not lines:
            continue
        station = graph.stations[sid]
        reverse = station.section_id in reversed_sections
        for lid in lines:
            p = line_priority.get(lid, 0)
            if reverse:
                offsets[(sid, lid)] = (max_priority - p) * offset_step
            else:
                offsets[(sid, lid)] = p * offset_step

    # Set exit port offsets on TB sections with LEFT/RIGHT exits to
    # match the exit L-shape's horiz_y_off (= rev of internal station
    # offset).  This adapts automatically to the section's reversed
    # status: reversed sections produce standard exit ordering,
    # non-reversed sections produce reversed exit ordering.
    tb_sections = {sid for sid, s in graph.sections.items() if s.direction == "TB"}
    for port_id, port_obj in graph.ports.items():
        if port_obj.is_entry or port_obj.section_id not in tb_sections:
            continue
        if port_obj.side not in (PortSide.LEFT, PortSide.RIGHT):
            continue
        # Find offsets at the internal station feeding this exit port
        internal_offs: dict[str, float] = {}
        for edge in graph.edges:
            if edge.target == port_id:
                src_st = graph.stations.get(edge.source)
                if src_st and not src_st.is_port:
                    internal_offs[edge.line_id] = offsets.get(
                        (edge.source, edge.line_id), 0.0
                    )
        if internal_offs:
            max_int = max(internal_offs.values())
            for lid, ioff in internal_offs.items():
                offsets[(port_id, lid)] = max_int - ioff

    # Junctions have section_id=None so they get default line-priority
    # ordering above, which may not match the exit port feeding them.
    # Inherit offsets from the upstream exit port instead.
    for jid in graph.junctions:
        for edge in graph.edges:
            if edge.target == jid:
                src = graph.stations.get(edge.source)
                if src and src.is_port and not graph.ports.get(edge.source, None).is_entry:
                    # Copy exit port's offsets to the junction
                    for lid in graph.station_lines(jid):
                        port_off = offsets.get((edge.source, lid))
                        if port_off is not None:
                            offsets[(jid, lid)] = port_off
                    break

    return offsets


def _line_source_y_at_port(
    port_id: str,
    graph: MetroGraph,
) -> dict[str, float]:
    """Map line_id -> Y of connected internal station at an exit port.

    For an exit port, looks at edges going TO the port (station -> port)
    and returns the source station's Y position for each line.
    """
    line_y: dict[str, float] = {}
    for edge in graph.edges:
        if edge.target == port_id:
            src = graph.stations.get(edge.source)
            if src and not src.is_port:
                line_y[edge.line_id] = src.y
    return line_y


def _line_incoming_y_at_entry_port(
    port_id: str,
    graph: MetroGraph,
    exit_offsets: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Map line_id -> effective Y of incoming connection at an entry port.

    Uses the source station's Y + its already-computed station offset
    for the line, so the entry port ordering matches the bundle ordering
    from the source section.
    """
    line_y: dict[str, float] = {}
    for edge in graph.edges:
        if edge.target == port_id:
            src = graph.stations.get(edge.source)
            if src and src.is_port:
                src_off = exit_offsets.get((edge.source, edge.line_id), 0)
                line_y[edge.line_id] = src.y + src_off
    return line_y


def _detect_reversed_sections(graph: MetroGraph) -> set[str]:
    """Find sections where incoming bundle ordering is reversed.

    A section is "reversed" when it receives lines via a TB section's
    exit port that reverses the bundle ordering. This happens in two cases:

    1. TOP entry fed by a TB section's BOTTOM exit: the TB section reverses
       X offsets in the vertical bundle, so the downstream section must use
       reversed Y ordering to match.

    2. LEFT/RIGHT entry fed by a TB section's LEFT/RIGHT exit: the
       concentric corner routing reverses the bundle ordering (outermost
       vertical line becomes outermost horizontal line), so the downstream
       section must use reversed Y ordering to match.

    Reversal propagates: if a reversed section exits to another section
    on the same row, that downstream section is also reversed so bundle
    ordering stays consistent along the return row.
    """
    tb_sections = {sid for sid, s in graph.sections.items() if s.direction == "TB"}
    reversed_secs: set[str] = set()
    junction_ids = set(graph.junctions)

    # Phase 1a: detect sections directly fed by TB BOTTOM exits
    for sec_id, section in graph.sections.items():
        for port_id in section.entry_ports:
            port = graph.ports.get(port_id)
            if not port or port.side != PortSide.TOP:
                continue
            for edge in graph.edges:
                if edge.target == port_id:
                    src = graph.stations.get(edge.source)
                    if not src or not src.is_port:
                        continue
                    src_port = graph.ports.get(edge.source)
                    if (
                        src_port
                        and not src_port.is_entry
                        and src_port.side == PortSide.BOTTOM
                        and src.section_id in tb_sections
                    ):
                        reversed_secs.add(sec_id)

    # Build section adjacency from inter-section edges (used by
    # propagation phases below).
    sec_successors: dict[str, set[str]] = {}
    for edge in graph.edges:
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if not src or not tgt:
            continue
        if src.section_id and tgt.section_id and src.section_id != tgt.section_id:
            sec_successors.setdefault(src.section_id, set()).add(tgt.section_id)

    def _propagate_along_rows() -> bool:
        """Propagate reversal to same-row successors.

        Returns True if any new sections were added.
        """
        added_any = False
        changed = True
        while changed:
            changed = False
            for sec_id in list(reversed_secs):
                section = graph.sections.get(sec_id)
                if not section:
                    continue
                for succ_id in sec_successors.get(sec_id, set()):
                    if succ_id in reversed_secs:
                        continue
                    succ = graph.sections.get(succ_id)
                    if succ and succ.grid_row == section.grid_row:
                        reversed_secs.add(succ_id)
                        changed = True
                        added_any = True
        return added_any

    # Phase 1b + Phase 2: iteratively detect sections fed by TB
    # LEFT/RIGHT exits and propagate along rows.  The concentric
    # corner reverses the bundle ordering ONLY when the TB section
    # uses non-reversed internal offsets.  If the TB section is
    # itself already reversed (e.g. via propagation from an earlier
    # TB exit), its exit L-shape un-reverses back to standard, so
    # the downstream section should NOT be marked reversed.
    #
    # We iterate because a later TB section may become reversed
    # through row propagation from an earlier TB section's
    # downstream (e.g. calling -> hard_filter -> ... -> integration).
    def _is_tb_lr_exit_nonreversed(port_obj):
        """Check if port is an LR exit of a non-reversed TB section."""
        return (
            port_obj is not None
            and not port_obj.is_entry
            and port_obj.side in (PortSide.LEFT, PortSide.RIGHT)
            and port_obj.section_id in tb_sections
            and port_obj.section_id not in reversed_secs
        )

    # Process one TB exit at a time: add the downstream section,
    # propagate along rows (which may mark the next TB section as
    # reversed), then re-scan.  This ensures that propagation from
    # an earlier TB exit's downstream is visible when checking later
    # TB exits.
    stable = False
    while not stable:
        stable = True

        for sec_id, section in graph.sections.items():
            if sec_id in reversed_secs:
                continue
            added = False
            for port_id in section.entry_ports:
                if added:
                    break
                port = graph.ports.get(port_id)
                if not port or port.side not in (
                    PortSide.LEFT,
                    PortSide.RIGHT,
                ):
                    continue
                for edge in graph.edges:
                    if added:
                        break
                    if edge.target != port_id:
                        continue
                    src = graph.stations.get(edge.source)
                    if not src:
                        continue
                    matched = False
                    if edge.source in junction_ids:
                        # Look through junction to find upstream exit port
                        for e2 in graph.edges:
                            if e2.target == edge.source:
                                s2 = graph.stations.get(e2.source)
                                if not s2 or not s2.is_port:
                                    continue
                                s2_port = graph.ports.get(e2.source)
                                if _is_tb_lr_exit_nonreversed(s2_port):
                                    matched = True
                                    break
                    elif src.is_port:
                        src_port = graph.ports.get(edge.source)
                        matched = _is_tb_lr_exit_nonreversed(src_port)
                    if matched:
                        reversed_secs.add(sec_id)
                        _propagate_along_rows()
                        stable = False
                        added = True
            if added:
                break  # restart outer scan

    return reversed_secs
