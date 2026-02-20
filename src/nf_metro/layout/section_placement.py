"""Section meta-graph layout: place sections on the canvas.

Builds a DAG of sections from inter-section edges, uses topological
layering for column assignment and row stacking within columns.
Grid overrides can pin sections to specific positions.
"""

from __future__ import annotations

__all__ = ["place_sections", "position_ports"]

from collections import defaultdict, deque

from nf_metro.layout.constants import (
    MIN_INTER_SECTION_GAP,
    MIN_PORT_STATION_GAP,
    PLACEMENT_X_GAP,
    PLACEMENT_Y_GAP,
    PORT_MIN_GAP,
)
from nf_metro.parser.model import MetroGraph, PortSide, Section


def _build_section_dag(
    graph: MetroGraph,
) -> set[tuple[str, str]]:
    """Build section dependency edges from graph edges, traversing junctions."""
    section_edges: set[tuple[str, str]] = set()

    junction_ids = set(graph.junctions)
    junction_targets: dict[str, set[str]] = defaultdict(set)
    junction_sources: dict[str, set[str]] = defaultdict(set)

    for edge in graph.edges:
        src_sec = graph.section_for_station(edge.source)
        tgt_sec = graph.section_for_station(edge.target)

        if edge.target in junction_ids and src_sec:
            junction_sources[edge.target].add(src_sec)
        elif edge.source in junction_ids and tgt_sec:
            junction_targets[edge.source].add(tgt_sec)
        elif src_sec and tgt_sec and src_sec != tgt_sec:
            section_edges.add((src_sec, tgt_sec))

    for jid in junction_ids:
        for src_sec in junction_sources.get(jid, set()):
            for tgt_sec in junction_targets.get(jid, set()):
                if src_sec != tgt_sec:
                    section_edges.add((src_sec, tgt_sec))

    return section_edges


def _assign_grid_layout(
    graph: MetroGraph,
    section_edges: set[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Assign grid columns and rows to each section.

    Returns (col_assign, row_assign) dicts mapping section IDs to positions.
    """
    section_ids = list(graph.sections.keys())

    # Topological layering (columns)
    in_degree: dict[str, int] = {sid: 0 for sid in section_ids}
    adj: dict[str, list[str]] = {sid: [] for sid in section_ids}
    for src, tgt in section_edges:
        adj[src].append(tgt)
        in_degree[tgt] += 1

    # BFS topological sort for column assignment
    col_assign: dict[str, int] = {}
    queue: deque[str] = deque()
    for sid in section_ids:
        if in_degree[sid] == 0:
            queue.append(sid)
            col_assign[sid] = 0

    while queue:
        sid = queue.popleft()
        for tgt in adj[sid]:
            new_col = col_assign[sid] + 1
            if tgt not in col_assign or new_col > col_assign[tgt]:
                col_assign[tgt] = new_col
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)

    # Handle any sections not reached (disconnected)
    for sid in section_ids:
        if sid not in col_assign:
            col_assign[sid] = 0

    # Apply grid overrides
    for sid, (col, row, rowspan, colspan) in graph.grid_overrides.items():
        if sid in graph.sections:
            graph.sections[sid].grid_col = col
            graph.sections[sid].grid_row = row
            graph.sections[sid].grid_row_span = rowspan
            graph.sections[sid].grid_col_span = colspan
            col_assign[sid] = col

    # Group sections by column
    col_groups: dict[int, list[str]] = defaultdict(list)
    for sid, col in col_assign.items():
        col_groups[col].append(sid)

    # Assign rows within each column (order by section number, then by id)
    row_assign: dict[str, int] = {}
    for col, sids in sorted(col_groups.items()):
        explicit = [
            (sid, graph.sections[sid].grid_row)
            for sid in sids
            if graph.sections[sid].grid_row >= 0
        ]
        auto = [sid for sid in sids if graph.sections[sid].grid_row < 0]

        auto.sort(key=lambda s: graph.sections[s].number)

        used_rows: set[int] = set()
        for sid, row in explicit:
            row_assign[sid] = row
            span = graph.sections[sid].grid_row_span
            for r in range(row, row + span):
                used_rows.add(r)

        next_row = 0
        for sid in auto:
            while next_row in used_rows:
                next_row += 1
            row_assign[sid] = next_row
            used_rows.add(next_row)
            next_row += 1

    return col_assign, row_assign


def _compute_section_offsets(
    graph: MetroGraph,
    col_assign: dict[str, int],
    row_assign: dict[str, int],
    section_x_gap: float,
    section_y_gap: float,
) -> tuple[int, int]:
    """Compute pixel offsets for each section from grid assignments.

    Returns (min_col, max_col) for use by downstream gap enforcement.
    """
    min_col = min(col_assign.values()) if col_assign else 0
    max_col = max(col_assign.values()) if col_assign else 0
    for sid in graph.sections:
        cspan = graph.sections[sid].grid_col_span
        col = col_assign.get(sid, 0)
        min_col = min(min_col, col)
        max_col = max(max_col, col + cspan - 1)

    # Max width per column (only from single-column sections)
    col_widths: dict[int, float] = defaultdict(float)
    for sid, section in graph.sections.items():
        if section.grid_col_span == 1:
            col = col_assign.get(sid, 0)
            col_widths[col] = max(col_widths[col], section.bbox_w)

    for c in range(min_col, max_col + 1):
        if c not in col_widths:
            col_widths[c] = 0.0

    # Expand columns if a spanning section exceeds spanned column widths
    for sid, section in graph.sections.items():
        cspan = section.grid_col_span
        if cspan <= 1:
            continue
        start_col = col_assign.get(sid, 0)
        spanned = sum(col_widths[c] for c in range(start_col, start_col + cspan))
        spanned += (cspan - 1) * section_x_gap
        if section.bbox_w > spanned:
            deficit = section.bbox_w - spanned
            col_widths[start_col + cspan - 1] += deficit

    # Cumulative x offsets
    col_offsets: dict[int, float] = {}
    cumulative_x = 0.0
    for col in range(min_col, max_col + 1):
        col_offsets[col] = cumulative_x
        cumulative_x += col_widths.get(col, 0) + section_x_gap

    # Global row heights (only single-row non-TB sections)
    max_row = max(row_assign.values()) if row_assign else 0
    for sid in graph.sections:
        span = graph.sections[sid].grid_row_span
        row = row_assign.get(sid, 0)
        max_row = max(max_row, row + span - 1)

    row_heights: dict[int, float] = defaultdict(float)
    for sid, section in graph.sections.items():
        if section.grid_row_span == 1 and section.direction != "TB":
            row = row_assign.get(sid, 0)
            row_heights[row] = max(row_heights[row], section.bbox_h)

    for r in range(max_row + 1):
        if r not in row_heights:
            row_heights[r] = 0.0

    # Expand rows if a spanning section exceeds spanned row heights
    for sid, section in graph.sections.items():
        rspan = section.grid_row_span
        if rspan <= 1:
            continue
        start_row = row_assign.get(sid, 0)
        spanned = sum(row_heights[r] for r in range(start_row, start_row + rspan))
        spanned += (rspan - 1) * section_y_gap
        if section.bbox_h > spanned:
            deficit = section.bbox_h - spanned
            row_heights[start_row + rspan - 1] += deficit

    # Cumulative y offsets per row
    row_offsets: dict[int, float] = {}
    cumulative_y = 0.0
    for r in range(max_row + 1):
        row_offsets[r] = cumulative_y
        cumulative_y += row_heights[r] + section_y_gap

    # TB fold sections visually span into the next row
    tb_sections = sorted(
        [
            (sid, section)
            for sid, section in graph.sections.items()
            if section.direction == "TB" and section.grid_row_span == 1
        ],
        key=lambda x: row_assign.get(x[0], 0),
    )
    for sid, section in tb_sections:
        row = row_assign.get(sid, 0)
        next_row = row + 1
        if next_row not in row_offsets:
            continue
        section.bbox_h += section_y_gap
        tb_bottom = row_offsets[row] + section.bbox_h
        next_row_bottom = row_offsets[next_row] + row_heights[next_row]
        if tb_bottom > next_row_bottom:
            delta = tb_bottom - next_row_bottom
            for r in range(next_row, max_row + 1):
                if r in row_offsets:
                    row_offsets[r] += delta
        next_row_bottom = row_offsets[next_row] + row_heights[next_row]
        section.bbox_h = next_row_bottom - row_offsets[row]

    # Right-align columns containing RL or TB sections
    right_align_cols: set[int] = set()
    for sid, section in graph.sections.items():
        if section.direction in ("RL", "TB") and section.grid_col_span == 1:
            right_align_cols.add(col_assign.get(sid, 0))

    # Set section offsets and adjust for spanning
    for sid, section in graph.sections.items():
        section.grid_col = col_assign.get(sid, 0)
        section.grid_row = row_assign.get(sid, 0)
        section.offset_x = col_offsets.get(section.grid_col, 0)
        section.offset_y = row_offsets.get(section.grid_row, 0)

        if section.grid_col_span == 1 and (
            section.direction in ("RL", "TB") or section.grid_col in right_align_cols
        ):
            col_w = col_widths.get(section.grid_col, 0)
            if col_w > section.bbox_w:
                section.offset_x += col_w - section.bbox_w

        rspan = section.grid_row_span
        if rspan > 1:
            start_row = section.grid_row
            spanned_height = sum(
                row_heights[r] for r in range(start_row, start_row + rspan)
            )
            spanned_height += (rspan - 1) * section_y_gap
            section.bbox_h = spanned_height

        cspan = section.grid_col_span
        if cspan > 1:
            start_col = section.grid_col
            spanned_width = sum(
                col_widths[c] for c in range(start_col, start_col + cspan)
            )
            spanned_width += (cspan - 1) * section_x_gap
            section.bbox_w = spanned_width

    return min_col, max_col


def place_sections(
    graph: MetroGraph,
    section_x_gap: float = PLACEMENT_X_GAP,
    section_y_gap: float = PLACEMENT_Y_GAP,
) -> None:
    """Place sections on the canvas by computing offsets.

    Builds a meta-graph of section dependencies, assigns columns
    via topological layering, assigns rows within columns, then
    computes pixel offsets for each section.
    """
    if not graph.sections:
        return

    section_edges = _build_section_dag(graph)
    col_assign, row_assign = _assign_grid_layout(graph, section_edges)
    min_col, max_col = _compute_section_offsets(
        graph, col_assign, row_assign, section_x_gap, section_y_gap
    )
    _enforce_min_column_gaps(graph, col_assign, min_col, max_col)


def _rows_overlap(a: Section, b: Section) -> bool:
    """Return True if two sections occupy overlapping grid rows."""
    a_start = a.grid_row
    a_end = a_start + a.grid_row_span - 1
    b_start = b.grid_row
    b_end = b_start + b.grid_row_span - 1
    return a_start <= b_end and b_start <= a_end


def _enforce_min_column_gaps(
    graph: MetroGraph,
    col_assign: dict[str, int],
    min_col: int,
    max_col: int,
    min_gap: float = MIN_INTER_SECTION_GAP,
) -> None:
    """Shift columns rightward so adjacent section bboxes are at least *min_gap* apart.

    Scans column pairs left-to-right.  For each pair, finds the narrowest
    physical gap between sections whose row ranges overlap.  Only row-
    overlapping pairs can have bypass routes between them, so sections in
    non-overlapping rows are ignored.  If the gap is too narrow, all
    sections in the right column and beyond are shifted rightward by the
    deficit.  Processing left-to-right makes shifts cumulative.
    """
    if max_col <= min_col:
        return

    # Group sections by their assigned column
    col_sections: dict[int, list[Section]] = defaultdict(list)
    for sid, section in graph.sections.items():
        col = col_assign.get(sid, 0)
        col_sections[col].append(section)

    for col in range(min_col, max_col):
        left_secs = col_sections.get(col, [])
        right_secs = col_sections.get(col + 1, [])
        if not left_secs or not right_secs:
            continue

        # Find the tightest gap among row-overlapping section pairs
        worst_gap: float | None = None
        for ls in left_secs:
            for rs in right_secs:
                if not _rows_overlap(ls, rs):
                    continue
                left_edge = ls.offset_x + ls.bbox_x + ls.bbox_w
                right_edge = rs.offset_x + rs.bbox_x
                gap = right_edge - left_edge
                if worst_gap is None or gap < worst_gap:
                    worst_gap = gap

        if worst_gap is None or worst_gap >= min_gap:
            continue

        deficit = min_gap - worst_gap
        # Shift all sections in columns > col rightward
        for shift_col in range(col + 1, max_col + 1):
            for s in col_sections.get(shift_col, []):
                s.offset_x += deficit


def position_ports(section: Section, graph: MetroGraph) -> None:
    """Position port stations on section boundaries.

    Entry ports go on the entry side, exit ports on the exit side.
    Port Y/X is aligned with the connected internal station where possible.
    Multiple ports on the same side are spaced evenly along the boundary.
    """
    # Group ports by side
    side_ports: dict[PortSide, list[str]] = defaultdict(list)
    for pid in section.entry_ports + section.exit_ports:
        port = graph.ports.get(pid)
        if port:
            side_ports[port.side].append(pid)

    for side, port_ids in side_ports.items():
        if side == PortSide.LEFT:
            _position_ports_on_boundary(
                port_ids, section.bbox_x, section, graph, fixed_axis="x"
            )
        elif side == PortSide.RIGHT:
            right_x = section.bbox_x + section.bbox_w
            _position_ports_on_boundary(
                port_ids, right_x, section, graph, fixed_axis="x"
            )
        elif side == PortSide.TOP:
            _position_ports_on_boundary(
                port_ids, section.bbox_y, section, graph, fixed_axis="y"
            )
        elif side == PortSide.BOTTOM:
            bottom_y = section.bbox_y + section.bbox_h
            _position_ports_on_boundary(
                port_ids, bottom_y, section, graph, fixed_axis="y"
            )

    # TB sections: move LEFT/RIGHT exit ports just below the last
    # internal station (not the section bottom) so lines don't detour
    # to the bottom when the successor is at a similar Y level.
    if section.direction == "TB":
        exit_set = set(section.exit_ports)
        internal_ids = (
            set(section.station_ids) - set(section.entry_ports) - exit_set
        )
        internal_ys = [
            graph.stations[sid].y
            for sid in internal_ids
            if sid in graph.stations and not graph.stations[sid].is_port
        ]
        last_y = max(internal_ys) if internal_ys else section.bbox_y + section.bbox_h
        target_y = last_y + MIN_PORT_STATION_GAP
        for pid in exit_set:
            port = graph.ports.get(pid)
            if port and port.side in (PortSide.LEFT, PortSide.RIGHT):
                station = graph.stations.get(pid)
                if station:
                    station.y = target_y
                port.y = target_y


def _position_ports_on_boundary(
    port_ids: list[str],
    fixed_coord: float,
    section: Section,
    graph: MetroGraph,
    fixed_axis: str,
) -> None:
    """Position ports along a section boundary.

    Args:
        fixed_axis: "x" for vertical boundaries (LEFT/RIGHT),
                    "y" for horizontal boundaries (TOP/BOTTOM).
    """
    if not port_ids:
        return

    # The "free" axis is the one ports can slide along
    free_axis = "y" if fixed_axis == "x" else "x"

    for pid in port_ids:
        station = graph.stations.get(pid)
        if not station:
            continue

        connected = _find_connected_internal_coord(pid, section, graph, free_axis)
        if free_axis == "y":
            default = section.bbox_y + section.bbox_h / 2
        else:
            default = section.bbox_x + section.bbox_w / 2

        setattr(station, fixed_axis, fixed_coord)
        setattr(station, free_axis, connected if connected is not None else default)

        port = graph.ports.get(pid)
        if port:
            port.x = station.x
            port.y = station.y

    if free_axis == "y":
        span_start = section.bbox_y
        span_end = section.bbox_y + section.bbox_h
    else:
        span_start = section.bbox_x
        span_end = section.bbox_x + section.bbox_w

    _spread_overlapping_ports(
        port_ids,
        graph,
        axis=free_axis,
        span_start=span_start,
        span_end=span_end,
    )


def _find_connected_internal_coord(
    port_id: str,
    section: Section,
    graph: MetroGraph,
    axis: str,
) -> float | None:
    """Find the coordinate to align a port with its connected internal stations.

    Returns the average X or Y (determined by *axis*) of all connected
    internal stations, or None if no connections found.
    """
    internal_ids = (
        set(section.station_ids) - set(section.entry_ports) - set(section.exit_ports)
    )
    vals: list[float] = []
    for edge in graph.edges:
        if edge.source == port_id and edge.target in internal_ids:
            vals.append(getattr(graph.stations[edge.target], axis))
        if edge.target == port_id and edge.source in internal_ids:
            vals.append(getattr(graph.stations[edge.source], axis))
    if vals:
        return sum(vals) / len(vals)
    return None


def _spread_overlapping_ports(
    port_ids: list[str],
    graph: MetroGraph,
    axis: str,
    span_start: float,
    span_end: float,
    min_gap: float = PORT_MIN_GAP,
) -> None:
    """Spread ports that ended up at the same position."""
    if len(port_ids) <= 1:
        return

    # Check for overlap
    positions: list[tuple[str, float]] = []
    for pid in port_ids:
        station = graph.stations.get(pid)
        if station:
            pos = station.y if axis == "y" else station.x
            positions.append((pid, pos))

    positions.sort(key=lambda p: p[1])

    # Check if any ports overlap
    needs_spread = False
    for i in range(1, len(positions)):
        if abs(positions[i][1] - positions[i - 1][1]) < min_gap:
            needs_spread = True
            break

    if not needs_spread:
        return

    # Evenly space ports along the span
    n = len(positions)
    margin = min_gap
    available = (span_end - span_start) - 2 * margin
    step = available / max(n - 1, 1)

    for i, (pid, _) in enumerate(positions):
        new_pos = span_start + margin + i * step
        station = graph.stations.get(pid)
        if station:
            if axis == "y":
                station.y = new_pos
            else:
                station.x = new_pos
            port = graph.ports.get(pid)
            if port:
                if axis == "y":
                    port.y = new_pos
                else:
                    port.x = new_pos
