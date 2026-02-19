"""Shared types and helper functions for edge routing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nf_metro.layout.constants import BYPASS_CLEARANCE, COORD_TOLERANCE, COORD_TOLERANCE_FINE
from nf_metro.parser.model import Edge, MetroGraph


@dataclass
class RoutedPath:
    """A routed path for an edge, consisting of (x, y) waypoints."""

    edge: Edge
    line_id: str
    points: list[tuple[float, float]]
    is_inter_section: bool = False
    curve_radii: list[float] | None = None
    offsets_applied: bool = False


def compute_bundle_info(
    graph: MetroGraph,
    junction_ids: set[str],
    line_priority: dict[str, int],
    bottom_exit_junctions: set[str] | None = None,
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
    corridor_groups: dict[tuple, list[tuple[Edge, float, float, float, float]]] = (
        defaultdict(list)
    )

    for item in inter_edges:
        edge, sx, sy, tx, ty = item
        dx = tx - sx
        dy = ty - sy

        if abs(dy) < COORD_TOLERANCE_FINE:
            continue  # Horizontal edges don't need bundling

        v_dir = 1 if dy > 0 else -1

        if abs(dx) < COORD_TOLERANCE:
            # Vertical: group by shared X position
            key = ("V", round(sx), v_dir)
        else:
            # L-shaped: group by the inter-column gap the vertical
            # channel will occupy.  Use (src_col, tgt_col) when
            # section info is available so that edges from different
            # ports in the same column share one bundle and get
            # proper offsets.  Fall back to round(sx) for junctions
            # or edges without section info.
            h_dir = 1 if dx > 0 else -1
            src_st = graph.stations.get(edge.source)
            tgt_st = graph.stations.get(edge.target)
            src_sec = (
                graph.sections.get(src_st.section_id)
                if src_st and src_st.section_id
                else None
            )
            tgt_sec = (
                graph.sections.get(tgt_st.section_id)
                if tgt_st and tgt_st.section_id
                else None
            )
            if src_sec and tgt_sec and src_sec.grid_col != tgt_sec.grid_col:
                col_key = (src_sec.grid_col, tgt_sec.grid_col)
            else:
                col_key = round(sx)
            key = ("L", col_key, v_dir, h_dir)

        corridor_groups[key].append(item)

    # Assign per-line positions within each corridor
    assignments: dict[tuple[str, str, str], tuple[int, int]] = {}

    for _key, group in corridor_groups.items():
        # Sort by spatial ordering so the bundle's visual position
        # is preserved around corners.
        source_ids = {e[0].source for e in group}
        if len(source_ids) == 1:
            exit_port_id = group[0][0].source
            if bottom_exit_junctions and exit_port_id in bottom_exit_junctions:
                # Vertical-first: longest drop (largest target Y) is
                # outermost (i=0) to prevent crossings at corners.
                group.sort(
                    key=lambda e: (
                        -e[4],
                        line_priority.get(e[0].line_id, 999),
                    )
                )
            elif (port := graph.ports.get(exit_port_id)) and not port.is_entry:
                source_y = line_source_y_at_port(exit_port_id, graph)
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


def inter_column_channel_x(
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


def line_source_y_at_port(
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


def adjacent_column_gap_x(
    graph: MetroGraph,
    col_a: int,
    col_b: int,
) -> float:
    """X midpoint between two adjacent columns.

    Finds the right edge of col_a and left edge of col_b (assuming
    col_a < col_b) and returns the midpoint.
    """
    lo, hi = min(col_a, col_b), max(col_a, col_b)
    right_of_lo = max(
        (
            s.bbox_x + s.bbox_w
            for s in graph.sections.values()
            if s.grid_col == lo and s.bbox_w > 0
        ),
        default=0.0,
    )
    left_of_hi = min(
        (
            s.bbox_x
            for s in graph.sections.values()
            if s.grid_col == hi and s.bbox_w > 0
        ),
        default=right_of_lo,
    )
    return (right_of_lo + left_of_hi) / 2


def bypass_bottom_y(
    graph: MetroGraph,
    src_col: int,
    tgt_col: int,
    clearance: float = BYPASS_CLEARANCE,
) -> float:
    """Max bottom Y of sections in columns strictly between src_col and tgt_col.

    Returns the bottom Y + clearance, providing the baseline for a bypass
    route that goes below the intervening sections.
    """
    lo, hi = min(src_col, tgt_col), max(src_col, tgt_col)
    max_bottom = 0.0
    for s in graph.sections.values():
        if s.bbox_w > 0 and lo < s.grid_col < hi:
            bottom = s.bbox_y + s.bbox_h
            if bottom > max_bottom:
                max_bottom = bottom
    # Also include source and target columns - the route goes below everything
    for s in graph.sections.values():
        if s.bbox_w > 0 and s.grid_col in (lo, hi):
            bottom = s.bbox_y + s.bbox_h
            if bottom > max_bottom:
                max_bottom = bottom
    return max_bottom + clearance


def line_incoming_y_at_entry_port(
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
