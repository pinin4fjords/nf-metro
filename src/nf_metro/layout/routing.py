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


def route_edges(
    graph: MetroGraph,
    diagonal_run: float = 30.0,
    curve_radius: float = 10.0,
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

    # Pre-compute per-line index and count at each junction
    # for concentric arc radius calculations
    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}
    junction_line_info: dict[tuple[str, str], tuple[int, int]] = {}
    offset_step = 3.0
    for jid in junction_ids:
        lines_at = graph.station_lines(jid)
        lines_at.sort(key=lambda l: line_priority.get(l, 999))
        n = len(lines_at)
        for i, lid in enumerate(lines_at):
            junction_line_info[(jid, lid)] = (i, n)

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
            info = junction_line_info.get(
                (edge.source, edge.line_id),
                junction_line_info.get(
                    (edge.target, edge.line_id), None))
            i, n = info if info else (0, 1)

            if abs(dy) < 0.01:
                # Same Y: straight horizontal
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx, sy), (tx, ty)],
                    is_inter_section=True,
                ))
            elif abs(dx) < 1.0:
                # Same X: straight vertical - apply horizontal offset
                x_off = curve_radius + (2 * (n - 1) - i) * offset_step
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[(sx + x_off, sy), (tx + x_off, ty)],
                    is_inter_section=True,
                ))
            else:
                # L-shape: center vertical bundle between source and
                # target, with per-line offsets for visual separation.
                mid_x = (sx + tx) / 2
                delta = ((n - 1) / 2 - i) * offset_step
                vx = mid_x + delta
                # Concentric arcs: different radii at top vs bottom.
                r_top = curve_radius + (n - 1 - i) * offset_step
                r_bottom = curve_radius + i * offset_step
                routes.append(RoutedPath(
                    edge=edge, line_id=edge.line_id,
                    points=[
                        (sx, sy),
                        (vx, sy),
                        (vx, ty),
                        (tx, ty),
                    ],
                    is_inter_section=True,
                    curve_radii=[r_top, r_bottom],
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

            if sign > 0:
                diag_start_x = max(diag_start_x, sx + 10)
                diag_end_x = min(diag_end_x, tx - 10)
            else:
                diag_start_x = min(diag_start_x, sx - 10)
                diag_end_x = max(diag_end_x, tx + 10)

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

            if sign > 0:
                diag_start_x = max(diag_start_x, sx + 10)
                diag_end_x = min(diag_end_x, tx - 10)
            else:
                diag_start_x = min(diag_start_x, sx - 10)
                diag_end_x = max(diag_end_x, tx + 10)

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

    Returns dict mapping (station_id, line_id) -> y_offset.
    """
    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}

    offsets: dict[tuple[str, str], float] = {}
    for sid in graph.stations:
        lines = graph.station_lines(sid)
        lines.sort(key=lambda l: line_priority.get(l, 999))
        for i, lid in enumerate(lines):
            offsets[(sid, lid)] = i * offset_step

    return offsets
