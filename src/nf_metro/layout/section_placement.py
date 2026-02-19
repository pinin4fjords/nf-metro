"""Section meta-graph layout: place sections on the canvas.

Builds a DAG of sections from inter-section edges, uses topological
layering for column assignment and row stacking within columns.
Grid overrides can pin sections to specific positions.
"""

from __future__ import annotations

from collections import defaultdict, deque

from nf_metro.layout.constants import (
    MIN_INTER_SECTION_GAP,
    PLACEMENT_X_GAP,
    PLACEMENT_Y_GAP,
    PORT_MIN_GAP,
)
from nf_metro.parser.model import MetroGraph, PortSide, Section


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

    # Build section dependency DAG (traverse through junctions)
    section_edges: set[tuple[str, str]] = set()

    # Build adjacency from junctions to find which sections they connect
    junction_ids = set(graph.junctions)
    # junction -> set of sections reachable via outgoing edges
    junction_targets: dict[str, set[str]] = defaultdict(set)
    # junction -> set of sections feeding into it
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

    # Add edges through junctions: each source section -> each target section
    for jid in junction_ids:
        for src_sec in junction_sources.get(jid, set()):
            for tgt_sec in junction_targets.get(jid, set()):
                if src_sec != tgt_sec:
                    section_edges.add((src_sec, tgt_sec))

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
        # Respect explicit grid_row if set
        explicit = [
            (sid, graph.sections[sid].grid_row)
            for sid in sids
            if graph.sections[sid].grid_row >= 0
        ]
        auto = [sid for sid in sids if graph.sections[sid].grid_row < 0]

        # Sort auto sections by their number (definition order)
        auto.sort(key=lambda s: graph.sections[s].number)

        # Place explicit ones first, reserving all spanned rows
        used_rows: set[int] = set()
        for sid, row in explicit:
            row_assign[sid] = row
            span = graph.sections[sid].grid_row_span
            for r in range(row, row + span):
                used_rows.add(r)

        # Fill in auto sections
        next_row = 0
        for sid in auto:
            while next_row in used_rows:
                next_row += 1
            row_assign[sid] = next_row
            used_rows.add(next_row)
            next_row += 1

    # Compute pixel offsets
    min_col = min(col_assign.values()) if col_assign else 0
    max_col = max(col_assign.values()) if col_assign else 0
    # Account for columns occupied by spanning sections
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

    # Ensure all columns have an entry (handles negative grid columns)
    for c in range(min_col, max_col + 1):
        if c not in col_widths:
            col_widths[c] = 0.0

    # Expand columns if a spanning section's intrinsic width exceeds the
    # sum of its spanned columns. Distributes the extra to the last column.
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

    # Cumulative x offsets (columns are shared, handles negative grid columns)
    col_offsets: dict[int, float] = {}
    cumulative_x = 0.0
    for col in range(min_col, max_col + 1):
        col_offsets[col] = cumulative_x
        cumulative_x += col_widths.get(col, 0) + section_x_gap

    # Global row heights: max bbox_h per row across ALL columns
    # (only from single-row sections, since spanning sections derive height)
    max_row = max(row_assign.values()) if row_assign else 0
    # Account for rows occupied by spanning sections
    for sid in graph.sections:
        span = graph.sections[sid].grid_row_span
        row = row_assign.get(sid, 0)
        max_row = max(max_row, row + span - 1)

    row_heights: dict[int, float] = defaultdict(float)
    for sid, section in graph.sections.items():
        if section.grid_row_span == 1 and section.direction != "TB":
            row = row_assign.get(sid, 0)
            row_heights[row] = max(row_heights[row], section.bbox_h)

    # Ensure all rows have an entry (even if only occupied by spanning sections)
    for r in range(max_row + 1):
        if r not in row_heights:
            row_heights[r] = 0.0

    # Expand rows if a spanning section's intrinsic height exceeds spanned rows
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

    # TB fold sections (rowspan=1) visually span into the next row.
    # Push the next row down so its non-TB sections' bottom aligns with
    # the TB section's bottom, then extend the TB section's bbox to cover.
    # Process top-to-bottom so cascading TB sections accumulate correctly.
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
        # Add exit routing margin so lines can flow down from the last
        # station before curving out to the return row.
        section.bbox_h += section_y_gap
        tb_bottom = row_offsets[row] + section.bbox_h
        next_row_bottom = row_offsets[next_row] + row_heights[next_row]
        if tb_bottom > next_row_bottom:
            delta = tb_bottom - next_row_bottom
            for r in range(next_row, max_row + 1):
                if r in row_offsets:
                    row_offsets[r] += delta
        # Extend bbox to cover through the next row
        next_row_bottom = row_offsets[next_row] + row_heights[next_row]
        section.bbox_h = next_row_bottom - row_offsets[row]

    # Columns containing RL or TB sections should right-align all their
    # sections so stacked LR sections share a right edge with the RL/TB ones.
    right_align_cols: set[int] = set()
    for sid, section in graph.sections.items():
        if section.direction in ("RL", "TB") and section.grid_col_span == 1:
            right_align_cols.add(col_assign.get(sid, 0))

    # Set section offsets and adjust height for spanning sections
    for sid, section in graph.sections.items():
        section.grid_col = col_assign.get(sid, 0)
        section.grid_row = row_assign.get(sid, 0)
        section.offset_x = col_offsets.get(section.grid_col, 0)
        section.offset_y = row_offsets.get(section.grid_row, 0)

        # Right-align sections within columns that contain RL or TB sections,
        # so fold sections and post-fold RL sections share a right edge with
        # any LR sections stacked above them in the same column.
        if section.grid_col_span == 1 and (
            section.direction in ("RL", "TB") or section.grid_col in right_align_cols
        ):
            col_w = col_widths.get(section.grid_col, 0)
            if col_w > section.bbox_w:
                section.offset_x += col_w - section.bbox_w

        # For row-spanning sections, set bbox_h to cover all spanned rows + gaps
        rspan = section.grid_row_span
        if rspan > 1:
            start_row = section.grid_row
            spanned_height = sum(
                row_heights[r] for r in range(start_row, start_row + rspan)
            )
            spanned_height += (rspan - 1) * section_y_gap
            section.bbox_h = spanned_height

        # For col-spanning sections, set bbox_w to cover all spanned cols + gaps
        cspan = section.grid_col_span
        if cspan > 1:
            start_col = section.grid_col
            spanned_width = sum(
                col_widths[c] for c in range(start_col, start_col + cspan)
            )
            spanned_width += (cspan - 1) * section_x_gap
            section.bbox_w = spanned_width

    # Top-align sections within their row. No vertical centering -- this
    # keeps the top edges flush so horizontal lines between sections in
    # the same row don't need unnecessary vertical jogs.

    # Enforce minimum physical gap between adjacent columns so bypass
    # route corners have enough room for smooth curves.
    _enforce_min_column_gaps(graph, col_assign, min_col, max_col)


def _enforce_min_column_gaps(
    graph: MetroGraph,
    col_assign: dict[str, int],
    min_col: int,
    max_col: int,
    min_gap: float = MIN_INTER_SECTION_GAP,
) -> None:
    """Shift columns rightward so adjacent section bboxes are at least *min_gap* apart.

    Scans column pairs left-to-right.  For each pair, computes the actual
    physical gap between the rightmost bbox edge in the left column and the
    leftmost bbox edge in the right column (using pre-global-transform
    ``offset_x + bbox_x`` coordinates).  If the gap is too narrow, all
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

        # Rightmost edge of any section in the left column
        max_right_edge = max(s.offset_x + s.bbox_x + s.bbox_w for s in left_secs)
        # Leftmost edge of any section in the right column
        min_left_edge = min(s.offset_x + s.bbox_x for s in right_secs)

        actual_gap = min_left_edge - max_right_edge
        if actual_gap >= min_gap:
            continue

        deficit = min_gap - actual_gap
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
            x = section.bbox_x
            _position_ports_vertical(port_ids, x, section, graph)
        elif side == PortSide.RIGHT:
            x = section.bbox_x + section.bbox_w
            _position_ports_vertical(port_ids, x, section, graph)
        elif side == PortSide.TOP:
            y = section.bbox_y
            _position_ports_horizontal(port_ids, y, section, graph)
        elif side == PortSide.BOTTOM:
            y = section.bbox_y + section.bbox_h
            _position_ports_horizontal(port_ids, y, section, graph)

    # TB sections: move LEFT/RIGHT exit ports to the section bottom
    # so lines flow down from the last station then curve out.
    if section.direction == "TB":
        exit_set = set(section.exit_ports)
        for pid in exit_set:
            port = graph.ports.get(pid)
            if port and port.side in (PortSide.LEFT, PortSide.RIGHT):
                target_y = section.bbox_y + section.bbox_h
                station = graph.stations.get(pid)
                if station:
                    station.y = target_y
                port.y = target_y


def _position_ports_vertical(
    port_ids: list[str],
    x: float,
    section: Section,
    graph: MetroGraph,
) -> None:
    """Position ports along a vertical boundary (LEFT or RIGHT side)."""
    if not port_ids:
        return

    # Try to align each port with its connected internal station
    for pid in port_ids:
        station = graph.stations.get(pid)
        if not station:
            continue

        port = graph.ports.get(pid)

        # Find connected internal station
        connected_y = _find_connected_internal_y(pid, section, graph)
        station.x = x
        station.y = (
            connected_y
            if connected_y is not None
            else (section.bbox_y + section.bbox_h / 2)
        )

        # Update port data too
        if port:
            port.x = station.x
            port.y = station.y

    # If multiple ports are at the same Y, space them out
    _spread_overlapping_ports(
        port_ids,
        graph,
        axis="y",
        span_start=section.bbox_y,
        span_end=section.bbox_y + section.bbox_h,
    )


def _position_ports_horizontal(
    port_ids: list[str],
    y: float,
    section: Section,
    graph: MetroGraph,
) -> None:
    """Position ports along a horizontal boundary (TOP or BOTTOM side)."""
    if not port_ids:
        return

    for pid in port_ids:
        station = graph.stations.get(pid)
        if not station:
            continue

        connected_x = _find_connected_internal_x(pid, section, graph)
        station.x = (
            connected_x
            if connected_x is not None
            else (section.bbox_x + section.bbox_w / 2)
        )
        station.y = y

        port = graph.ports.get(pid)
        if port:
            port.x = station.x
            port.y = station.y

    _spread_overlapping_ports(
        port_ids,
        graph,
        axis="x",
        span_start=section.bbox_x,
        span_end=section.bbox_x + section.bbox_w,
    )


def _find_connected_internal_y(
    port_id: str,
    section: Section,
    graph: MetroGraph,
) -> float | None:
    """Find the Y coordinate to align a port with its connected internal stations.

    If the port connects to multiple stations (e.g. a shared entry port),
    returns the average Y of all connected stations.
    """
    internal_ids = (
        set(section.station_ids) - set(section.entry_ports) - set(section.exit_ports)
    )
    ys: list[float] = []
    for edge in graph.edges:
        if edge.source == port_id and edge.target in internal_ids:
            ys.append(graph.stations[edge.target].y)
        if edge.target == port_id and edge.source in internal_ids:
            ys.append(graph.stations[edge.source].y)
    if ys:
        return sum(ys) / len(ys)
    return None


def _find_connected_internal_x(
    port_id: str,
    section: Section,
    graph: MetroGraph,
) -> float | None:
    """Find the X coordinate to align a port with its connected internal stations.

    If the port connects to multiple stations, returns the average X.
    """
    internal_ids = (
        set(section.station_ids) - set(section.entry_ports) - set(section.exit_ports)
    )
    xs: list[float] = []
    for edge in graph.edges:
        if edge.source == port_id and edge.target in internal_ids:
            xs.append(graph.stations[edge.target].x)
        if edge.target == port_id and edge.source in internal_ids:
            xs.append(graph.stations[edge.source].x)
    if xs:
        return sum(xs) / len(xs)
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


def _expand_lone_rowspans(
    graph: MetroGraph,
    col_assign: dict[str, int],
    row_assign: dict[str, int],
) -> None:
    """Expand rowspan for sections that inflate a row but have empty rows below.

    When a section is the tallest in its row (among rowspan=1 sections) and
    its column has consecutive empty rows below, expanding its rowspan
    distributes its height across multiple rows. This reduces dead space
    for shorter sections sharing the original row.
    """
    # Build occupancy grid (col, row) -> section_id
    occupied: dict[tuple[int, int], str] = {}
    for sid, section in graph.sections.items():
        col = col_assign.get(sid, 0)
        row = row_assign.get(sid, 0)
        for c in range(col, col + section.grid_col_span):
            for r in range(row, row + section.grid_row_span):
                occupied[(c, r)] = sid

    max_row = 0
    for sid, section in graph.sections.items():
        row = row_assign.get(sid, 0)
        max_row = max(max_row, row + section.grid_row_span - 1)

    # Process sections top to bottom to avoid conflicts.
    # Skip TB sections: they are fold bridges whose vertical extent is
    # determined by their content. Expanding them would make the row too
    # short, placing the return-row sections above the fold exit.
    candidates = sorted(
        [
            (sid, graph.sections[sid])
            for sid in graph.sections
            if graph.sections[sid].grid_row_span == 1
            and graph.sections[sid].direction != "TB"
        ],
        key=lambda x: row_assign.get(x[0], 0),
    )

    for sid, section in candidates:
        col = col_assign.get(sid, 0)
        row = row_assign.get(sid, 0)

        # Find consecutive empty rows below in all columns this section occupies
        max_expand_row = row
        for r in range(row + 1, max_row + 1):
            all_empty = True
            for c in range(col, col + section.grid_col_span):
                if (c, r) in occupied:
                    all_empty = False
                    break
            if not all_empty:
                break
            max_expand_row = r

        potential_rowspan = max_expand_row - row + 1
        if potential_rowspan <= 1:
            continue

        # Only expand if this section is inflating its row: its bbox_h
        # exceeds the tallest other rowspan=1 section in the same row.
        max_other_h = 0.0
        for other_sid, other in graph.sections.items():
            if other_sid == sid:
                continue
            if row_assign.get(other_sid, -1) == row and other.grid_row_span == 1:
                max_other_h = max(max_other_h, other.bbox_h)

        if section.bbox_h <= max_other_h:
            continue

        # Expand rowspan
        section.grid_row_span = potential_rowspan
        for c in range(col, col + section.grid_col_span):
            for r in range(row, row + potential_rowspan):
                occupied[(c, r)] = sid
