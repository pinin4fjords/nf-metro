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

    # Pre-compute bundle assignments: groups inter-section edges that
    # share the same vertical channel so they get consistent per-line
    # X offsets instead of overlapping.
    bundle_info = _compute_bundle_info(graph, junction_ids, line_priority)

    for edge in graph.edges:
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
        is_inter_section = (
            (src.is_port or edge.source in junction_ids)
            and (tgt.is_port or edge.target in junction_ids)
        )
        if is_inter_section:
            i, n = bundle_info.get(
                (edge.source, edge.target, edge.line_id), (0, 1))

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
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                    is_inter_section=True,
                ))
            elif src_is_tb_bottom and station_offsets:
                # Vertical drop from TB BOTTOM exit with reversed X offsets.
                src_off = station_offsets.get(
                    (edge.source, edge.line_id), 0.0)
                all_offs = [
                    station_offsets.get((edge.source, lid), 0.0)
                    for lid in graph.station_lines(edge.source)
                ]
                max_off = max(all_offs) if all_offs else 0.0
                rev_off = max_off - src_off
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx + rev_off, sy), (tx + rev_off, ty)],
                    is_inter_section=True,
                    offsets_applied=True,
                ))
            elif abs(dx) < 1.0:
                # Same X: straight vertical drop
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                    is_inter_section=True,
                ))
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

                # Place vertical channel just far enough from source
                # for the curve to fit, so the turn happens close to
                # the source rather than midway to the target.
                max_r = curve_radius + (n - 1) * offset_step
                if dx > 0:
                    mid_x = sx + max_r + offset_step
                else:
                    mid_x = sx - max_r - offset_step
                vx = mid_x + delta
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx, sy),
                        (vx, sy),
                        (vx, ty),
                        (tx, ty),
                    ],
                    is_inter_section=True,
                    curve_radii=[r_first, r_second],
                ))
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
        if (src_sec and src_sec == tgt_sec and src_sec in tb_sections
                and not src.is_port
                and (not tgt.is_port or tgt_is_bottom_exit)):
            src_off = station_offsets.get((edge.source, edge.line_id), 0.0) if station_offsets else 0.0
            tgt_off = station_offsets.get((edge.target, edge.line_id), 0.0) if station_offsets else 0.0
            src_is_vert = src.layer > 0
            tgt_is_vert = tgt.layer > 0 or tgt_is_bottom_exit

            # For downward-turning elbows: reverse X offsets so that the
            # top line in the horizontal bundle becomes the rightmost in
            # the vertical bundle (concentric curves at the corner).
            def _reverse_off(station_id: str, off: float) -> float:
                all_offs = [
                    station_offsets.get((station_id, lid), 0.0)
                    for lid in graph.station_lines(station_id)
                ] if station_offsets else []
                max_off = max(all_offs) if all_offs else 0.0
                return max_off - off

            if not src_is_vert and tgt_is_vert:
                # L-shaped elbow: horizontal run then vertical drop.
                # Y offset on horizontal segment, reversed X offset on vertical.
                # Concentric curve radii: outer line (rightmost) gets largest radius.
                rev_tgt_off = _reverse_off(edge.target, tgt_off)
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx, sy + src_off),
                        (tx + rev_tgt_off, sy + src_off),
                        (tx + rev_tgt_off, ty),
                    ],
                    offsets_applied=True,
                    curve_radii=[curve_radius + rev_tgt_off],
                ))
            elif src_is_vert and tgt_is_vert:
                # Vertical run: straight line with reversed X offsets
                rev_src_off = _reverse_off(edge.source, src_off)
                rev_tgt_off = _reverse_off(edge.target, tgt_off)
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx + rev_src_off, sy),
                        (tx + rev_tgt_off, ty),
                    ],
                    offsets_applied=True,
                ))
            else:
                # Horizontal run within TB section (layer 0 to layer 0)
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx, sy + src_off),
                        (tx, ty + tgt_off),
                    ],
                    offsets_applied=True,
                ))
            continue

        # TOP/BOTTOM port → internal station: L-shaped elbow
        # (e.g., vertical entry from above curving into an RL section)
        src_port_obj = graph.ports.get(edge.source)
        if (src_port_obj
                and src_port_obj.side in (PortSide.TOP, PortSide.BOTTOM)
                and not tgt.is_port):
            src_off = station_offsets.get((edge.source, edge.line_id), 0.0) if station_offsets else 0.0
            tgt_off = station_offsets.get((edge.target, edge.line_id), 0.0) if station_offsets else 0.0

            # Check if the incoming chain uses reversed X offsets
            # (from a TB section's BOTTOM exit port above).
            uses_rev_x = False
            if src_port_obj.is_entry and station_offsets:
                for e2 in graph.edges:
                    if e2.target == edge.source:
                        upstream = graph.stations.get(e2.source)
                        if upstream and upstream.is_port:
                            up_port = graph.ports.get(e2.source)
                            if (up_port and not up_port.is_entry
                                    and up_port.side == PortSide.BOTTOM
                                    and upstream.section_id in tb_sections):
                                uses_rev_x = True
                                break

            if uses_rev_x:
                # Reversed X offsets on vertical segment to match
                # the TB section's convention above.
                all_src_offs = [
                    station_offsets.get((edge.source, lid), 0.0)
                    for lid in graph.station_lines(edge.source)
                ]
                max_src_off = max(all_src_offs) if all_src_offs else 0.0
                rev_src_off = max_src_off - src_off

                # Target Y offsets are used as-is (not reversed) because
                # _detect_reversed_sections already flips the line ordering
                # for stations in sections fed by TB exits. This keeps the
                # arrival offsets consistent with the departure offsets.

                if abs(dx) < 1.0:
                    # Nearly same X: straight vertical drop
                    routes.append(RoutedPath(
                        edge=edge, line_id=edge.line_id,
                        points=[(sx + rev_src_off, sy), (tx, ty + tgt_off)],
                        offsets_applied=True,
                    ))
                else:
                    # L-shape: vertical drop (reversed X) then horizontal.
                    # Concentric curves at the corner: outermost line
                    # gets largest radius.
                    routes.append(RoutedPath(
                        edge=edge, line_id=edge.line_id,
                        points=[
                            (sx + rev_src_off, sy),
                            (sx + rev_src_off, ty + tgt_off),
                            (tx, ty + tgt_off),
                        ],
                        offsets_applied=True,
                        curve_radii=[curve_radius + rev_src_off],
                    ))
            elif abs(dx) < 1.0:
                # Nearly vertical: straight drop with Y offsets
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx, sy + src_off), (tx, ty + tgt_off)],
                    offsets_applied=True,
                ))
            else:
                # L-shape: vertical drop then horizontal to station
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx, sy + src_off),
                        (sx, ty + tgt_off),
                        (tx, ty + tgt_off),
                    ],
                    offsets_applied=True,
                ))
            continue

        # Detect cross-row edge: target is to the left (only in folded layouts)
        is_cross_row = dx <= 0 and abs(dy) > 80

        if is_cross_row:
            # Route through fold edge: horizontal to fold, vertical drop,
            # horizontal to target
            fold_margin = 30
            fold_right = fold_x + fold_margin
            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[
                    (sx, sy),
                    (fold_right, sy),
                    (fold_right, ty),
                    (tx, ty),
                ],
            ))
        elif abs(sy - ty) < 0.01:
            # Same track: straight line
            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[(sx, sy), (tx, ty)],
            ))
        else:
            # Different tracks: horizontal, diagonal, horizontal
            if abs(dx) < 1.0:
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                ))
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

            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[
                    (sx, sy),
                    (diag_start_x, sy),
                    (diag_end_x, ty),
                    (tx, ty),
                ],
            ))

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

        is_inter = (
            (src.is_port or edge.source in junction_ids)
            and (tgt.is_port or edge.target in junction_ids)
        )
        if not is_inter:
            continue

        inter_edges.append((edge, src.x, src.y, tgt.x, tgt.y))

    # Group by corridor: edges sharing the same vertical channel
    # Key: (route_type, rounded_channel_position, vertical_direction)
    corridor_groups: dict[tuple[str, int, int], list[tuple[Edge, float, float, float, float]]] = defaultdict(list)

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
            # L-shaped: group by vertical channel midpoint
            mid_x = (sx + tx) / 2
            key = ("L", round(mid_x), v_dir)

        corridor_groups[key].append(item)

    # Assign per-line positions within each corridor
    assignments: dict[tuple[str, str, str], tuple[int, int]] = {}

    for _key, group in corridor_groups.items():
        # Sort by the exit port's source-station Y ordering (same as
        # compute_station_offsets uses for exit ports) so the bundle
        # ordering matches the exit port's station offsets.
        exit_port_id = group[0][0].source
        port = graph.ports.get(exit_port_id)
        if port and not port.is_entry:
            source_y = _line_source_y_at_port(exit_port_id, graph)
            group.sort(key=lambda e: (
                source_y.get(e[0].line_id, 0),
                line_priority.get(e[0].line_id, 999),
            ))
        else:
            group.sort(key=lambda e: line_priority.get(e[0].line_id, 999))

        n = len(group)
        for i, (edge, *_rest) in enumerate(group):
            assignments[(edge.source, edge.target, edge.line_id)] = (i, n)

    return assignments


def route_inter_section_edges(
    graph: MetroGraph,
    diagonal_run: float = 30.0,
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
            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[(sx, sy), (tx, ty)],
            ))
        elif abs(dx) < 1.0:
            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[(sx, sy), (tx, ty)],
            ))
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

            routes.append(RoutedPath(
                edge=edge, line_id=edge.line_id,
                points=[
                    (sx, sy),
                    (diag_start_x, sy),
                    (diag_end_x, ty),
                    (tx, ty),
                ],
            ))

    return routes


def compute_station_offsets(
    graph: MetroGraph,
    offset_step: float = 3.0,
) -> dict[tuple[str, str], float]:
    """Compute per-station Y offsets for each line.

    At each station, lines are stacked in definition order: the first
    line sits at offset 0, subsequent lines stack below. This keeps
    each line at a consistent position as it passes through a station,
    so transitions between bundles of different sizes are smooth.

    Port stations use physical Y ordering to preserve bundle consistency:
    - Exit ports: sorted by connected internal station Y
    - Entry ports: sorted by incoming connection's effective Y
    This ensures the bundle ordering doesn't change between sections.

    Returns dict mapping (station_id, line_id) -> y_offset.
    """
    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}

    offsets: dict[tuple[str, str], float] = {}

    # Phase 1: Exit ports - sort by connected internal station Y
    for sid in graph.stations:
        port = graph.ports.get(sid)
        if not port or port.is_entry:
            continue
        lines = graph.station_lines(sid)
        source_y = _line_source_y_at_port(sid, graph)
        lines.sort(key=lambda l: (source_y.get(l, 0), line_priority.get(l, 999)))
        for i, lid in enumerate(lines):
            offsets[(sid, lid)] = i * offset_step

    # Phase 2: Entry ports - sort by incoming source's effective Y
    # (source station Y + its offset for the line) so bundle ordering
    # is preserved through the turn into the section
    for sid in graph.stations:
        port = graph.ports.get(sid)
        if not port or not port.is_entry:
            continue
        lines = graph.station_lines(sid)
        incoming_y = _line_incoming_y_at_entry_port(sid, graph, offsets)
        lines.sort(key=lambda l: (incoming_y.get(l, 0), line_priority.get(l, 999)))
        for i, lid in enumerate(lines):
            offsets[(sid, lid)] = i * offset_step

    # Detect sections where incoming bundle has reversed line ordering
    # (TOP entry fed by a TB section's BOTTOM exit). Internal station
    # offsets must match so lines don't flip at the first station.
    reversed_sections = _detect_reversed_sections(graph)

    # Phase 3: All other stations - sort by line priority
    for sid in graph.stations:
        if (sid, graph.station_lines(sid)[0]) in offsets if graph.station_lines(sid) else False:
            continue
        lines = graph.station_lines(sid)
        station = graph.stations[sid]
        reverse = station.section_id in reversed_sections
        lines.sort(key=lambda l: line_priority.get(l, 999), reverse=reverse)
        for i, lid in enumerate(lines):
            offsets[(sid, lid)] = i * offset_step

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

    A section is "reversed" when it receives lines via a TOP entry port
    that connects upstream to a TB section's BOTTOM exit port. The TB
    section reverses X offsets in the vertical bundle, so the downstream
    section must use reversed Y ordering to match.
    """
    tb_sections = {sid for sid, s in graph.sections.items() if s.direction == "TB"}
    reversed_secs: set[str] = set()

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
                    if (src_port and not src_port.is_entry
                            and src_port.side == PortSide.BOTTOM
                            and src.section_id in tb_sections):
                        reversed_secs.add(sec_id)

    return reversed_secs
