"""Layout coordinator: combines layer assignment, ordering, and coordinate mapping.

Section-first layout: sections are laid out independently, then placed on a meta-graph.
"""

from __future__ import annotations

__all__ = ["compute_layout"]

from collections import Counter

from nf_metro.layout.constants import (
    ENTRY_INSET_LR,
    ENTRY_SHIFT_TB,
    ENTRY_SHIFT_TB_CROSS,
    EXIT_GAP_MULTIPLIER,
    FONT_HEIGHT,
    JUNCTION_MARGIN,
    LABEL_LINE_HEIGHT,
    LABEL_MARGIN,
    LABEL_OFFSET,
    LABEL_PAD,
    LINE_GAP,
    MIN_PORT_STATION_GAP,
    ROW_GAP,
    SECTION_GAP,
    SECTION_X_GAP,
    SECTION_X_PADDING,
    SECTION_Y_GAP,
    SECTION_Y_PADDING,
    STATION_ELBOW_TOLERANCE,
    TB_LINE_Y_OFFSET,
    TERMINUS_ICON_CLEARANCE,
    X_OFFSET,
    X_SPACING,
    Y_OFFSET,
    Y_SPACING,
)
from nf_metro.layout.labels import label_text_width
from nf_metro.layout.layers import assign_layers
from nf_metro.layout.ordering import assign_tracks
from nf_metro.parser.model import Edge, MetroGraph, PortSide, Section, Station


def compute_layout(
    graph: MetroGraph,
    x_spacing: float = X_SPACING,
    y_spacing: float = Y_SPACING,
    x_offset: float = X_OFFSET,
    y_offset: float = Y_OFFSET,
    row_gap: float = ROW_GAP,
    section_gap: float = SECTION_GAP,
    section_x_padding: float = SECTION_X_PADDING,
    section_y_padding: float = SECTION_Y_PADDING,
    section_x_gap: float = SECTION_X_GAP,
    section_y_gap: float = SECTION_Y_GAP,
) -> None:
    """Compute layout positions for all stations in the graph."""
    # Optionally reorder lines by section span before layout.
    # Must happen here (on the full graph) before section subgraphs are
    # built, since subgraphs share graph.lines via reference.
    if graph.line_order == "span" and graph.lines:
        from nf_metro.layout.ordering import _reorder_by_span

        new_order = _reorder_by_span(graph, list(graph.lines.keys()))
        graph.lines = {lid: graph.lines[lid] for lid in new_order}

    if not graph.sections:
        _compute_flat_layout(
            graph,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        return

    _compute_section_layout(
        graph,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        x_offset=x_offset,
        y_offset=y_offset,
        section_x_padding=section_x_padding,
        section_y_padding=section_y_padding,
        section_x_gap=section_x_gap,
        section_y_gap=section_y_gap,
    )


def _compute_flat_layout(
    graph: MetroGraph,
    x_spacing: float = X_SPACING,
    y_spacing: float = Y_SPACING,
    x_offset: float = X_OFFSET,
    y_offset: float = Y_OFFSET,
) -> None:
    """Flat layout for sectionless pipelines.

    Runs layer/track assignment directly on the full graph and maps
    to coordinates without section boxes or port routing.
    """
    layers = assign_layers(graph)
    tracks = assign_tracks(graph, layers)

    if not layers:
        return

    # When tracks is empty (e.g. no named lines), default all to track 0.
    if not tracks:
        tracks = {sid: 0 for sid in layers}

    unique_tracks = sorted(set(tracks.values()))
    track_rank = {t: i for i, t in enumerate(unique_tracks)}

    layer_extra = _compute_fork_join_gaps(graph, layers, tracks, x_spacing)

    for sid, station in graph.stations.items():
        station.layer = layers.get(sid, 0)
        station.track = tracks.get(sid, 0)
        station.x = (
            x_offset + station.layer * x_spacing + layer_extra.get(station.layer, 0)
        )
        station.y = y_offset + track_rank[station.track] * y_spacing


def _compute_section_layout(
    graph: MetroGraph,
    x_spacing: float = X_SPACING,
    y_spacing: float = Y_SPACING,
    x_offset: float = X_OFFSET,
    y_offset: float = Y_OFFSET,
    section_x_padding: float = SECTION_X_PADDING,
    section_y_padding: float = SECTION_Y_PADDING,
    section_x_gap: float = SECTION_X_GAP,
    section_y_gap: float = SECTION_Y_GAP,
) -> None:
    """Section-first layout pipeline.

    Phase 1: Parse & partition (already done by parser)
    Phase 2: Internal section layout (per section, real stations only)
    Phase 3: Section placement (meta-graph)
    Phase 4: Global coordinate mapping
    Phase 5: Port positioning on section boundaries
    """
    from nf_metro.layout.section_placement import place_sections, position_ports

    # Phase 2: Lay out each section independently (real stations only, no ports)
    section_subgraphs: dict[str, MetroGraph] = {}
    for sec_id, section in graph.sections.items():
        sub = _layout_single_section(
            graph, section, x_spacing, y_spacing, section_x_padding, section_y_padding
        )
        if sub is not None:
            section_subgraphs[sec_id] = sub

    # Phase 3: Place sections on the canvas
    place_sections(graph, section_x_gap, section_y_gap)

    # Phase 4: Translate local coords to global coords (real stations)
    for sec_id, section in graph.sections.items():
        sub = section_subgraphs.get(sec_id)
        if not sub:
            continue

        for sid, local_station in sub.stations.items():
            if sid in graph.stations:
                graph.stations[sid].layer = local_station.layer
                graph.stations[sid].track = local_station.track
                graph.stations[sid].x = local_station.x + section.offset_x + x_offset
                graph.stations[sid].y = local_station.y + section.offset_y + y_offset

        # Update section bbox to global coords
        section.bbox_x += section.offset_x + x_offset
        section.bbox_y += section.offset_y + y_offset

    # Phase 5: Position ports on section boundaries (after bbox is in global coords)
    for sec_id, section in graph.sections.items():
        position_ports(section, graph)

    # Phase 6: Position junction stations in the inter-section gap
    _position_junctions(graph)

    # Phase 7: Align LEFT/RIGHT entry ports with their incoming connection's Y
    # so inter-section horizontal runs are straight
    _align_entry_ports(graph)

    # Phase 7b: For non-fold LR/RL sections, pull exit-entry port pairs toward
    # the downstream section's stations so lines flow more directly across.
    _align_ports_to_downstream(graph)

    # Phase 7c: Ensure ports maintain at least y_spacing from terminus
    # stations in their section so file icons don't overlap routed lines.
    _space_ports_from_termini(graph, y_spacing)

    # Phase 8: Align LEFT/RIGHT exit ports on row-spanning (fold) sections
    # with their target's Y so the exit is at the return row level
    _align_exit_ports(graph)

    # Phase 9: Re-position junctions after exit port alignment.
    # Phase 8 may have moved exit ports on fold sections, so junctions
    # placed in Phase 6 need updating to match the new exit port Y.
    _position_junctions(graph)


def _layout_single_section(
    graph: MetroGraph,
    section: Section,
    x_spacing: float,
    y_spacing: float,
    section_x_padding: float,
    section_y_padding: float,
) -> MetroGraph | None:
    """Lay out a single section's internal stations and compute its bbox.

    Runs layer/track assignment on the section's real stations, applies
    direction-specific adjustments (RL mirror, TB label extent, entry shifts),
    and computes the section bounding box. Returns the section subgraph with
    positioned stations, or None if the section has no layoutable stations.
    """
    sub = _build_section_subgraph(graph, section)
    if not sub.stations:
        return None

    layers = assign_layers(sub)
    tracks = assign_tracks(sub, layers)

    if not layers:
        return None

    # Compact tracks so widely-spaced line priorities don't inflate
    # the vertical spread.  Gaps larger than LINE_GAP get capped so
    # distant line base tracks don't create excessive whitespace.
    unique_tracks = sorted(set(tracks.values()))
    track_rank: dict[float, float] = {}
    if unique_tracks:
        track_rank[unique_tracks[0]] = 0.0
        for idx in range(1, len(unique_tracks)):
            gap = unique_tracks[idx] - unique_tracks[idx - 1]
            track_rank[unique_tracks[idx]] = track_rank[unique_tracks[idx - 1]] + min(
                gap, LINE_GAP
            )

    # Detect fork/join layers and add extra spacing so stations
    # aren't too close to divergence/convergence points.
    section_sids = set(section.station_ids)
    layer_extra = _compute_fork_join_gaps(
        sub, layers, tracks, x_spacing, graph, section_sids
    )

    # Widen track spacing when multi-line labels need more vertical room
    effective_y_spacing = _multiline_track_spacing(sub, y_spacing)

    # Assign local coordinates based on section direction
    for sid, station in sub.stations.items():
        station.layer = layers.get(sid, 0)
        station.track = tracks.get(sid, 0)
        if section.direction == "TB":
            station.x = track_rank[station.track] * x_spacing
            station.y = station.layer * y_spacing + layer_extra.get(station.layer, 0)
        else:
            station.x = station.layer * x_spacing + layer_extra.get(station.layer, 0)
            station.y = track_rank[station.track] * effective_y_spacing

    # Normalize Y so minimum is 0 (raw tracks can be negative)
    _normalize_min(sub, axis="y")

    # RL: mirror X so layer 0 is rightmost
    if section.direction == "RL":
        _mirror_rl(sub)

    # Normalize local X so leftmost station is at x=0
    _normalize_min(sub, axis="x")

    # Ensure minimum inner extent so stations sit on visible track
    _enforce_min_extent(sub, section, x_spacing, y_spacing)

    # Compute section bounding box from real stations only.
    # Extra Y padding for multi-line labels (outermost stations' labels
    # extend beyond the normal padding).
    xs = [s.x for s in sub.stations.values()]
    ys = [s.y for s in sub.stations.values()]
    extra_label_h = _multiline_label_padding(sub)
    y_pad = section_y_padding + extra_label_h
    section.bbox_x = min(xs) - section_x_padding
    section.bbox_y = min(ys) - y_pad
    section.bbox_w = (max(xs) - min(xs)) + section_x_padding * 2
    section.bbox_h = (max(ys) - min(ys)) + y_pad * 2

    # Apply direction-specific bbox adjustments
    _adjust_tb_labels(sub, section, graph)
    _adjust_tb_entry_shifts(section, sub, graph, y_spacing)
    _adjust_lr_entry_inset(sub, section, graph, x_spacing)
    _adjust_lr_exit_gap(sub, section, graph, layers, x_spacing)
    _adjust_terminus_icon_clearance(sub, section, graph)

    return sub


def _multiline_track_spacing(sub: MetroGraph, y_spacing: float) -> float:
    """Return effective Y track spacing, widened for multi-line labels.

    When labels from adjacent tracks face each other (one below, one
    above due to layer alternation) the track gap must be large enough
    for both labels plus clearance.  Returns *y_spacing* unchanged when
    no multi-line labels are present.
    """
    max_text_h = FONT_HEIGHT
    for s in sub.stations.values():
        n = s.label.count("\n")
        if n > 0:
            h = FONT_HEIGHT + n * FONT_HEIGHT * LABEL_LINE_HEIGHT
            max_text_h = max(max_text_h, h)

    if max_text_h <= FONT_HEIGHT:
        return y_spacing  # no multi-line labels

    # Worst case: adjacent tracks with labels facing inward.
    # Each side needs label_offset + its text height.
    min_gap = LABEL_OFFSET + max_text_h + LABEL_OFFSET + FONT_HEIGHT + LABEL_MARGIN
    return max(y_spacing, min_gap)


def _multiline_label_padding(sub: MetroGraph) -> float:
    """Return extra bbox Y padding for the tallest multi-line label."""
    max_extra = 0.0
    for s in sub.stations.values():
        n = s.label.count("\n")
        if n > 0:
            extra = n * FONT_HEIGHT * LABEL_LINE_HEIGHT
            max_extra = max(max_extra, extra)
    return max_extra


def _normalize_min(sub: MetroGraph, axis: str) -> None:
    """Shift all stations so the minimum coordinate on the given axis is 0."""
    vals = [getattr(s, axis) for s in sub.stations.values()]
    if vals:
        min_val = min(vals)
        if min_val != 0:
            for s in sub.stations.values():
                setattr(s, axis, getattr(s, axis) - min_val)


def _mirror_rl(sub: MetroGraph) -> None:
    """Mirror X coordinates for RL sections so layer 0 is rightmost.

    Anchors on non-terminus stations so adding terminus layers
    extends leftward without shifting the entry point.
    """
    non_term = [
        s for s in sub.stations.values() if not (s.is_terminus and not s.label.strip())
    ]
    anchor_stations = non_term if non_term else list(sub.stations.values())
    max_x_val = max(s.x for s in anchor_stations)
    for s in sub.stations.values():
        s.x = max_x_val - s.x


def _enforce_min_extent(
    sub: MetroGraph,
    section: Section,
    x_spacing: float,
    y_spacing: float,
) -> None:
    """Ensure minimum inner extent so stations sit on visible track."""
    xs = [s.x for s in sub.stations.values()]
    ys = [s.y for s in sub.stations.values()]
    if section.direction == "TB":
        inner_h = max(ys) - min(ys)
        min_inner_h = y_spacing
        if inner_h < min_inner_h:
            shift = (min_inner_h - inner_h) / 2
            for station in sub.stations.values():
                station.y += shift
    else:
        inner_w = max(xs) - min(xs)
        min_inner_w = x_spacing
        if inner_w < min_inner_w:
            shift = (min_inner_w - inner_w) / 2
            for station in sub.stations.values():
                station.x += shift


def _adjust_tb_labels(
    sub: MetroGraph,
    section: Section,
    graph: MetroGraph,
) -> None:
    """TB sections: expand bbox and shift stations right so labels fit.

    Labels extend leftward from the station (text_anchor=end).
    """
    if section.direction != "TB":
        return

    xs = [s.x for s in sub.stations.values()]
    max_label_extent = 0.0
    for sid, s in sub.stations.items():
        if s.label.strip():
            n_lines = len(sub.station_lines(sid))
            offset_span = (n_lines - 1) * TB_LINE_Y_OFFSET
            extent = offset_span / 2 + 11 + label_text_width(s.label)
            max_label_extent = max(max_label_extent, extent)
    need_left = max_label_extent + LABEL_PAD
    have_left = min(xs) - section.bbox_x
    if need_left > have_left:
        extra = need_left - have_left
        for s in sub.stations.values():
            s.x += extra
        section.bbox_w += extra


def _adjust_tb_entry_shifts(
    section: Section,
    sub: MetroGraph,
    graph: MetroGraph,
    y_spacing: float,
) -> None:
    """Apply TB section entry shifts for perpendicular and cross-column entries."""
    if section.direction != "TB":
        return

    # Perpendicular entry: shift stations down so first station isn't
    # at the entry port (avoiding station-as-elbow).
    has_perp_entry = any(
        graph.ports[pid].side in (PortSide.LEFT, PortSide.RIGHT)
        for pid in section.entry_ports
        if pid in graph.ports
    )
    if has_perp_entry:
        entry_shift = y_spacing * ENTRY_SHIFT_TB
        for s in sub.stations.values():
            s.y += entry_shift
        section.bbox_h += entry_shift

    # Cross-column TOP entry: shift stations down for L-shape routing room.
    has_cross_col_top_entry = False
    for pid in section.entry_ports:
        port = graph.ports.get(pid)
        if not port or port.side != PortSide.TOP:
            continue
        for edge in graph.edges:
            if edge.target == pid:
                src = graph.stations.get(edge.source)
                if src and src.section_id:
                    src_sec = graph.sections.get(src.section_id)
                    if src_sec and src_sec.grid_col != section.grid_col:
                        has_cross_col_top_entry = True
                        break
        if has_cross_col_top_entry:
            break
    if has_cross_col_top_entry:
        entry_shift = y_spacing * ENTRY_SHIFT_TB_CROSS
        for s in sub.stations.values():
            s.y += entry_shift
        section.bbox_h += entry_shift


def _adjust_lr_entry_inset(
    sub: MetroGraph,
    section: Section,
    graph: MetroGraph,
    x_spacing: float,
) -> None:
    """LR/RL sections: add extra bbox width when entry has curves."""
    if section.direction not in ("LR", "RL"):
        return

    has_perp_entry = any(
        graph.ports[pid].side in (PortSide.TOP, PortSide.BOTTOM)
        for pid in section.entry_ports
        if pid in graph.ports
    )
    if has_perp_entry:
        entry_inset = x_spacing * ENTRY_INSET_LR
        section.bbox_w += entry_inset
        return

    # Flow-side entry that fans out to multiple internal stations at
    # different Y positions needs extra room for the diagonal transitions.
    for pid in section.entry_ports:
        if pid not in graph.ports:
            continue
        flow_side = PortSide.LEFT if section.direction == "LR" else PortSide.RIGHT
        if graph.ports[pid].side != flow_side:
            continue
        targets = {
            e.target
            for e in graph.edges
            if e.source == pid and e.target in section.station_ids
        }
        if len(targets) > 1:
            entry_inset = x_spacing * EXIT_GAP_MULTIPLIER
            # Shift stations away from the entry edge to make room
            for s in sub.stations.values():
                s.x += entry_inset
            section.bbox_w += entry_inset
            return


def _adjust_lr_exit_gap(
    sub: MetroGraph,
    section: Section,
    graph: MetroGraph,
    layers: dict[str, int],
    x_spacing: float,
) -> None:
    """LR/RL sections with flow-side exit: add label clearance gap."""
    if section.direction not in ("LR", "RL"):
        return

    flow_exit_side = PortSide.RIGHT if section.direction == "LR" else PortSide.LEFT
    has_flow_exit = any(
        graph.ports[pid].side == flow_exit_side
        for pid in section.exit_ports
        if pid in graph.ports
    )
    if not has_flow_exit or not layers:
        return

    exit_gap = x_spacing * EXIT_GAP_MULTIPLIER

    # For single-layer sections the asymmetry is very visible, so split the
    # gap between both sides to keep the station visually centered.  For
    # multi-layer sections the gap belongs entirely on the exit side.
    n_layers = len(set(layers.values()))
    center = n_layers <= 1

    if section.direction == "LR":
        if center:
            half_gap = exit_gap / 2
            for s in sub.stations.values():
                s.x += half_gap
        section.bbox_w += exit_gap
    else:
        shift = exit_gap / 2 if center else exit_gap
        for s in sub.stations.values():
            s.x += shift
        section.bbox_w += exit_gap


def _adjust_terminus_icon_clearance(
    sub: MetroGraph,
    section: Section,
    graph: MetroGraph,
) -> None:
    """Expand bbox when terminus file icons would be too close to the edge.

    Terminus stations display a file icon on their "outside" (flow-entry for
    sources, flow-exit for sinks).  The icon extends roughly 39px from the
    station center.  If SECTION_X_PADDING doesn't provide enough room, we
    grow the bbox on the affected side.
    """
    for station in sub.stations.values():
        if not station.is_terminus:
            continue

        # Determine source vs sink from the full graph's edges
        is_source = not any(e.target == station.id for e in graph.edges)

        section_dir = section.direction or "LR"

        # Icon is always placed horizontally (left or right of station),
        # even for TB/BT sections.
        if section_dir in ("LR", "TB"):
            icon_on_left = is_source
        else:  # RL, BT
            icon_on_left = not is_source

        if icon_on_left:
            clearance = station.x - section.bbox_x
            if clearance < TERMINUS_ICON_CLEARANCE:
                expand = TERMINUS_ICON_CLEARANCE - clearance
                section.bbox_x -= expand
                section.bbox_w += expand
        else:
            bbox_right = section.bbox_x + section.bbox_w
            clearance = bbox_right - station.x
            if clearance < TERMINUS_ICON_CLEARANCE:
                expand = TERMINUS_ICON_CLEARANCE - clearance
                section.bbox_w += expand


def _position_junctions(graph: MetroGraph) -> None:
    """Position junction stations at the midpoint of the inter-section gap.

    A junction is where bundled lines diverge to different downstream sections.
    It sits horizontally between the exit port and the entry ports, at the
    exit port's Y coordinate so lines travel straight from exit to junction.
    """
    for jid in graph.junctions:
        junction = graph.stations.get(jid)
        if not junction:
            continue

        # Find the exit port feeding this junction (source of edge to junction)
        exit_port_x: float | None = None
        exit_port_y: float | None = None
        exit_port_id: str | None = None
        entry_port_xs: list[float] = []

        for edge in graph.edges:
            if edge.target == jid:
                src = graph.stations.get(edge.source)
                if src and src.is_port:
                    exit_port_x = src.x
                    exit_port_y = src.y
                    exit_port_id = edge.source
            if edge.source == jid:
                tgt = graph.stations.get(edge.target)
                if tgt and tgt.is_port:
                    entry_port_xs.append(tgt.x)

        if exit_port_x is not None and exit_port_y is not None and entry_port_xs:
            margin = JUNCTION_MARGIN
            exit_port_obj = graph.ports.get(exit_port_id) if exit_port_id else None
            if exit_port_obj and exit_port_obj.side == PortSide.BOTTOM:
                # BOTTOM exit: position junction below (same X, Y + margin)
                # so lines continue their vertical drop.
                junction.x = exit_port_x
                junction.y = exit_port_y + margin
            else:
                # Position close to the exit port with a small margin,
                # rather than at the midpoint. This keeps the divergence
                # point near the source section so lines turn sooner.
                nearest_entry_x = min(entry_port_xs, key=lambda x: abs(x - exit_port_x))
                direction = 1.0 if nearest_entry_x > exit_port_x else -1.0
                junction.x = exit_port_x + direction * margin
                junction.y = exit_port_y


def _resolve_source_section_id(
    graph: MetroGraph, edge_source: str, junction_ids: set[str]
) -> str | None:
    """Resolve the section ID of an edge's source, tracing through junctions.

    For port stations, returns section_id directly. For junctions, follows
    edges backward to find the connected port's section.
    """
    src = graph.stations.get(edge_source)
    if not src:
        return None
    src_section_id = src.section_id
    if edge_source in junction_ids:
        for e2 in graph.edges:
            if e2.target == edge_source:
                s2 = graph.stations.get(e2.source)
                if s2 and s2.section_id:
                    src_section_id = s2.section_id
                    break
    return src_section_id


def _set_port_y(graph: MetroGraph, port_id: str, y: float) -> None:
    """Set the Y coordinate on both the station and port objects."""
    station = graph.stations.get(port_id)
    port = graph.ports.get(port_id)
    if station:
        station.y = y
    if port:
        port.y = y


def _set_port_x(graph: MetroGraph, port_id: str, x: float) -> None:
    """Set the X coordinate on both the station and port objects."""
    station = graph.stations.get(port_id)
    port = graph.ports.get(port_id)
    if station:
        station.x = x
    if port:
        port.x = x


def _align_entry_ports(graph: MetroGraph) -> None:
    """Align entry ports with their incoming connection's coordinates.

    LEFT/RIGHT ports: align Y for straight horizontal runs.
    TOP/BOTTOM ports: align X for vertical drops or Y for cross-column.
    """
    junction_ids = set(graph.junctions)

    for port_id, port in graph.ports.items():
        if not port.is_entry:
            continue

        entry_section = graph.sections.get(port.section_id)
        if not entry_section:
            continue

        if port.side in (PortSide.LEFT, PortSide.RIGHT):
            _align_lr_entry_port(graph, port_id, port, entry_section, junction_ids)
        elif port.side in (PortSide.TOP, PortSide.BOTTOM):
            _align_tb_entry_port(graph, port_id, port, entry_section, junction_ids)


def _align_lr_entry_port(
    graph: MetroGraph,
    port_id: str,
    port,
    entry_section: Section,
    junction_ids: set[str],
) -> None:
    """Align a LEFT/RIGHT entry port's Y with its incoming source."""
    for edge in graph.edges:
        if edge.target != port_id:
            continue
        src = graph.stations.get(edge.source)
        if not src or not (src.is_port or edge.source in junction_ids):
            continue

        src_section_id = _resolve_source_section_id(graph, edge.source, junction_ids)
        src_section = graph.sections.get(src_section_id) if src_section_id else None
        if not src_section:
            continue

        if entry_section.grid_row != src_section.grid_row:
            break

        # Skip alignment if source Y is outside entry section bbox
        entry_station = graph.stations.get(port_id)
        if entry_station:
            bbox_top = entry_section.bbox_y
            bbox_bot = entry_section.bbox_y + entry_section.bbox_h
            if not (bbox_top <= src.y <= bbox_bot):
                break

        target_y = src.y

        # Clamp for TB sections with perpendicular entry
        if entry_section.direction == "TB" and port.side in (
            PortSide.LEFT,
            PortSide.RIGHT,
        ):
            target_y = _clamp_tb_entry_port(
                graph,
                entry_section,
                target_y,
                edge,
                src,
                junction_ids,
            )

        _set_port_y(graph, port_id, target_y)
        break


def _align_tb_entry_port(
    graph: MetroGraph,
    port_id: str,
    port,
    entry_section: Section,
    junction_ids: set[str],
) -> None:
    """Align a TOP/BOTTOM entry port with its incoming sources."""
    # Collect all incoming sources
    sources: list[tuple[Station, str | None]] = []
    for edge in graph.edges:
        if edge.target != port_id:
            continue
        src = graph.stations.get(edge.source)
        if not src or not (src.is_port or edge.source in junction_ids):
            continue
        src_section_id = _resolve_source_section_id(graph, edge.source, junction_ids)
        sources.append((src, src_section_id))

    if not sources:
        return

    # Check if any source is cross-column
    my_cols = set(
        range(
            entry_section.grid_col,
            entry_section.grid_col + entry_section.grid_col_span,
        )
    )
    is_cross_column = False
    for _, src_sid in sources:
        src_sec = graph.sections.get(src_sid) if src_sid else None
        if src_sec:
            src_cols = set(
                range(src_sec.grid_col, src_sec.grid_col + src_sec.grid_col_span)
            )
            if not (src_cols & my_cols):
                is_cross_column = True
                break

    if is_cross_column:
        # Cross-column: set Y to the closest source level
        src_ys = [s.y for s, _ in sources]
        if port.side == PortSide.TOP:
            target_y = min(src_ys)
        else:
            target_y = max(src_ys)
        # Clamp within bbox
        target_y = max(target_y, entry_section.bbox_y)
        target_y = min(target_y, entry_section.bbox_y + entry_section.bbox_h)
        _set_port_y(graph, port_id, target_y)
        # Only nudge X for LR/RL sections where TOP/BOTTOM ports are perpendicular
        if entry_section.direction in ("LR", "RL"):
            _nudge_port_from_stations(port_id, entry_section, graph)
    else:
        # Same-column: align X with source for vertical drop
        src, _ = sources[0]
        _set_port_x(graph, port_id, src.x)


def _nudge_port_from_stations(
    port_id: str,
    section: Section,
    graph: MetroGraph,
    tolerance: float = STATION_ELBOW_TOLERANCE,
) -> None:
    """Nudge a TOP/BOTTOM port away from any internal station at the same X.

    Moves the port toward the entry side of the section so it doesn't
    visually pass through a station marker (station-as-elbow).
    """
    station = graph.stations.get(port_id)
    port = graph.ports.get(port_id)
    if not station or not port:
        return

    internal_ids = (
        set(section.station_ids) - set(section.entry_ports) - set(section.exit_ports)
    )
    internal_xs = [
        graph.stations[sid].x
        for sid in internal_ids
        if sid in graph.stations and not graph.stations[sid].is_port
    ]
    if not internal_xs:
        return

    # Check if port X coincides with any internal station X
    if not any(abs(station.x - ix) < tolerance for ix in internal_xs):
        return

    # Move port toward the entry side of the section
    # For LR: entry is left, so move port left (toward bbox_x)
    # For RL: entry is right, so move port right (toward bbox_x + bbox_w)
    if section.direction == "RL":
        new_x = max(internal_xs) + tolerance
        # Clamp within bbox
        new_x = min(new_x, section.bbox_x + section.bbox_w - tolerance)
    else:
        new_x = min(internal_xs) - tolerance
        # Clamp within bbox
        new_x = max(new_x, section.bbox_x + tolerance)

    station.x = new_x
    port.x = new_x


def _align_ports_to_downstream(graph: MetroGraph) -> None:
    """Pull exit-entry port pairs toward downstream station positions.

    After Phase 7 aligns entry ports to their source (exit port), the
    exit-entry pair may sit at a Y that is far from the downstream
    section's internal stations, forcing lines to detour vertically
    between sections.  This pass moves both ports toward the downstream
    section's average station Y when that would reduce the detour.

    Only applies to non-fold LR/RL sections without fan-out junctions
    (fold/TB sections are handled by ``_align_exit_ports``).
    """
    junction_ids = set(graph.junctions)

    for port_id, port in graph.ports.items():
        if port.is_entry:
            continue

        exit_section = graph.sections.get(port.section_id)
        if not exit_section:
            continue

        # Skip fold/TB sections (handled by Phase 8)
        if exit_section.grid_row_span > 1 or exit_section.direction == "TB":
            continue

        if port.side not in (PortSide.LEFT, PortSide.RIGHT):
            continue

        # Find the single target entry port (skip fan-out via junctions)
        target_entry_id: str | None = None
        for edge in graph.edges:
            if edge.source != port_id:
                continue
            if edge.target in junction_ids:
                # Fan-out to junction -- don't override
                target_entry_id = None
                break
            tgt = graph.stations.get(edge.target)
            if tgt and tgt.is_port:
                tgt_port = graph.ports.get(edge.target)
                if tgt_port and tgt_port.is_entry:
                    target_entry_id = edge.target
                    # Keep scanning to detect junctions on later edges

        if not target_entry_id:
            continue

        # Locate the downstream section and its internal stations
        entry_port_obj = graph.ports.get(target_entry_id)
        if not entry_port_obj:
            continue
        entry_section = graph.sections.get(entry_port_obj.section_id)
        if not entry_section:
            continue

        # Skip cross-row connections (different grid rows)
        if exit_section.grid_row != entry_section.grid_row:
            continue

        # Skip when entry port is perpendicular to its section's flow.
        # A LEFT port on a TB section must bend, so aligning it with an
        # internal station's Y would route the line through that station.
        _perp = False
        if entry_section.direction == "TB" and entry_port_obj.side in (
            PortSide.LEFT,
            PortSide.RIGHT,
        ):
            _perp = True
        elif entry_section.direction in ("LR", "RL") and entry_port_obj.side in (
            PortSide.TOP,
            PortSide.BOTTOM,
        ):
            _perp = True
        if _perp:
            continue

        internal_ids = (
            set(entry_section.station_ids)
            - set(entry_section.entry_ports)
            - set(entry_section.exit_ports)
        )
        downstream_ys: list[float] = []
        for edge in graph.edges:
            if edge.source == target_entry_id and edge.target in internal_ids:
                downstream_ys.append(graph.stations[edge.target].y)
        if not downstream_ys:
            continue

        if graph.diamond_style == "straight":
            # Snap to the Y that the most lines target, so the majority
            # of lines flow straight.  Ties broken by topmost (smallest Y).
            y_counts: Counter[float] = Counter(downstream_ys)
            target_y = min(y_counts, key=lambda y: (-y_counts[y], y))
        else:
            target_y = sum(downstream_ys) / len(downstream_ys)

        # Only move if target_y fits within both section bboxes
        exit_top = exit_section.bbox_y
        exit_bot = exit_section.bbox_y + exit_section.bbox_h
        if not (exit_top <= target_y <= exit_bot):
            continue

        entry_top = entry_section.bbox_y
        entry_bot = entry_section.bbox_y + entry_section.bbox_h
        if not (entry_top <= target_y <= entry_bot):
            continue

        _set_port_y(graph, port_id, target_y)
        _set_port_y(graph, target_entry_id, target_y)


def _align_exit_ports(graph: MetroGraph) -> None:
    """Align LEFT/RIGHT exit ports on fold sections with their target's Y.

    Applies to sections with grid_row_span > 1 OR TB direction (fold bridges).
    These have exit ports placed near the section bottom, but the target
    section's entry may be at a different Y. Aligning ensures a straight
    horizontal inter-section connection.
    """
    junction_ids = set(graph.junctions)

    for port_id, port in graph.ports.items():
        if port.is_entry:
            continue

        exit_section = graph.sections.get(port.section_id)
        if not exit_section:
            continue
        if exit_section.grid_row_span <= 1 and exit_section.direction != "TB":
            continue

        if port.side in (PortSide.LEFT, PortSide.RIGHT):
            _align_lr_exit_port(graph, port_id, port, exit_section, junction_ids)


def _align_lr_exit_port(
    graph: MetroGraph,
    port_id: str,
    port,
    exit_section: Section,
    junction_ids: set[str],
) -> None:
    """Align a LEFT/RIGHT exit port's Y with its target entry port."""
    for edge in graph.edges:
        if edge.source != port_id:
            continue
        tgt = graph.stations.get(edge.target)
        if not tgt:
            continue

        # Don't align with fan-out junctions
        if edge.target in junction_ids:
            break

        if not tgt.is_port:
            continue

        # Don't align with perpendicular target ports (cross-axis)
        tgt_port_obj = graph.ports.get(tgt.id)
        if tgt_port_obj and tgt_port_obj.side in (PortSide.TOP, PortSide.BOTTOM):
            break

        # Don't pull exit port outside its section bbox
        bbox_top = exit_section.bbox_y
        bbox_bot = exit_section.bbox_y + exit_section.bbox_h
        if not (bbox_top <= tgt.y <= bbox_bot):
            break

        if exit_section.direction == "TB":
            tgt_y = _resolve_tb_exit_y(graph, port, tgt, exit_section)
        else:
            tgt_y = tgt.y

        _set_port_y(graph, port_id, tgt_y)
        break


def _resolve_tb_exit_y(
    graph: MetroGraph,
    port,
    tgt: Station,
    exit_section: Section,
) -> float:
    """Resolve the Y coordinate for a TB section's exit port.

    Mirrors the entry-side gap: finds how far the perpendicular entry
    port sits above the first internal station, and places the exit port
    the same distance below the last internal station. Pushes the target
    section down if needed so the inter-section line is straight.
    """
    internal_ys = [
        graph.stations[sid].y
        for sid in exit_section.station_ids
        if sid in graph.stations and not graph.stations[sid].is_port
    ]
    last_y = max(internal_ys) if internal_ys else port.y
    first_y = min(internal_ys) if internal_ys else port.y

    # Mirror the entry-side gap (distance from entry port to first station)
    entry_gap = MIN_PORT_STATION_GAP
    for pid in exit_section.entry_ports:
        ep = graph.ports.get(pid)
        if ep and ep.side in (PortSide.LEFT, PortSide.RIGHT):
            entry_gap = max(entry_gap, first_y - graph.stations[pid].y)
            break

    min_exit_y = last_y + entry_gap
    if tgt.y >= min_exit_y:
        tgt_y = tgt.y
    else:
        # Push target section down to align with exit port
        tgt_y = min_exit_y
        delta = tgt_y - tgt.y

        tgt.y = tgt_y
        tgt_port = graph.ports.get(tgt.id)
        if tgt_port:
            tgt_port.y = tgt_y
            tgt_sec = graph.sections.get(tgt_port.section_id)
            if tgt_sec:
                for sid in tgt_sec.station_ids:
                    s = graph.stations.get(sid)
                    if s and s.id != tgt.id:
                        s.y += delta
                        p = graph.ports.get(sid)
                        if p:
                            p.y += delta
                tgt_sec.bbox_y += delta

    # Extend exit section bbox so padding below the exit port
    # mirrors the padding above the entry port.
    entry_port_y = None
    for pid in exit_section.entry_ports:
        ep = graph.ports.get(pid)
        if ep and ep.side in (PortSide.LEFT, PortSide.RIGHT):
            entry_port_y = graph.stations[pid].y
            break
    if entry_port_y is not None:
        top_pad = entry_port_y - exit_section.bbox_y
        desired_bot = tgt_y + top_pad
        current_bot = exit_section.bbox_y + exit_section.bbox_h
        if desired_bot > current_bot:
            exit_section.bbox_h = desired_bot - exit_section.bbox_y

    return tgt_y


def _clamp_tb_entry_port(
    graph: MetroGraph,
    entry_section: Section,
    target_y: float,
    edge: Edge,
    src: Station,
    junction_ids: set[str],
) -> float:
    """Clamp a TB section's perpendicular entry port above internal stations.

    The entry port must stay above the first internal station so the
    direction-change curve has room. When clamped, also pulls the source
    station/junction up to maintain a straight horizontal run.

    Returns the (possibly clamped) target_y.
    """
    internal_ids = (
        set(entry_section.station_ids)
        - set(entry_section.entry_ports)
        - set(entry_section.exit_ports)
    )
    internal_ys = [
        graph.stations[sid].y
        for sid in internal_ids
        if sid in graph.stations and not graph.stations[sid].is_port
    ]
    if not internal_ys:
        return target_y

    first_y = min(internal_ys)
    max_y = first_y - MIN_PORT_STATION_GAP
    if target_y <= max_y:
        return target_y

    # Prefer the topmost source-side station feeding the exit port
    # so that line exits horizontally.
    exit_pid = edge.source
    if edge.source in junction_ids:
        for e2 in graph.edges:
            if e2.target == edge.source:
                ep = graph.stations.get(e2.source)
                if ep and ep.is_port:
                    exit_pid = e2.source
                    break

    top_src_y = None
    for e3 in graph.edges:
        if e3.target == exit_pid:
            s3 = graph.stations.get(e3.source)
            if s3 and not s3.is_port and e3.source not in junction_ids:
                if top_src_y is None or s3.y < top_src_y:
                    top_src_y = s3.y

    if top_src_y is not None and top_src_y < max_y:
        target_y = top_src_y
    else:
        target_y = max_y

    # Pull source up to maintain straight horizontal run
    src.y = target_y
    if src.is_port and edge.source in graph.ports:
        graph.ports[edge.source].y = target_y
    # If source is a junction, also pull the exit port feeding it
    if edge.source in junction_ids:
        for e2 in graph.edges:
            if e2.target == edge.source:
                ep = graph.stations.get(e2.source)
                if ep and ep.is_port:
                    ep.y = target_y
                    if e2.source in graph.ports:
                        graph.ports[e2.source].y = target_y

    return target_y


def _space_ports_from_termini(
    graph: MetroGraph,
    y_spacing: float,
) -> None:
    """Push ports away from terminus stations so there is a full row gap.

    After port alignment, an entry or exit port may sit very close to a
    terminus station in the same section.  Lines routed from that port
    then overlap the terminus file icon.

    Only entry ports are checked against entry-side (source) termini, and
    exit ports against exit-side (sink) termini, to avoid displacing
    ports on the opposite side of the section.

    Exit ports on fold sections (grid_row_span > 1 or TB direction) are
    skipped because Phase 8 (_align_exit_ports) will overwrite them.
    """
    # Pre-compute edge adjacency (used to identify direct connections
    # and to propagate port moves across section boundaries).
    adjacency: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {}
    predecessors: dict[str, set[str]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)
        successors.setdefault(edge.source, set()).add(edge.target)
        predecessors.setdefault(edge.target, set()).add(edge.source)

    for section in graph.sections.values():
        entry_port_ids = set(section.entry_ports)
        exit_port_ids = set(section.exit_ports)
        all_port_ids = entry_port_ids | exit_port_ids
        real_sids = {s for s in section.station_ids if s not in all_port_ids}

        # Skip exit ports on fold sections -- Phase 8 will handle them.
        is_fold = section.grid_row_span > 1 or section.direction == "TB"

        # Classify termini by side.  A station with no in-section
        # predecessors is an entry-side (source) terminus; one with no
        # in-section successors is an exit-side (sink) terminus.  A
        # station can be both (isolated within the section), but we only
        # add it to entry_termini to avoid conflicting pushes from both
        # the entry and exit port passes.
        entry_termini: list[tuple[str, float]] = []
        exit_termini: list[tuple[str, float]] = []
        for sid in real_sids:
            st = graph.stations.get(sid)
            if not st or not st.is_terminus or st.is_port:
                continue
            preds = predecessors.get(sid, set())
            succs = successors.get(sid, set())
            is_source = not (preds & real_sids)
            is_sink = not (succs & real_sids)
            if is_source:
                entry_termini.append((sid, st.y))
            elif is_sink:
                # Only classify as exit terminus if not already an
                # entry terminus (avoids double-counting isolated nodes).
                exit_termini.append((sid, st.y))

        _push_ports_from_termini(
            graph,
            sorted(entry_port_ids),
            entry_termini,
            section,
            adjacency,
            predecessors,
            y_spacing,
        )
        if not is_fold:
            _push_ports_from_termini(
                graph,
                sorted(exit_port_ids),
                exit_termini,
                section,
                adjacency,
                predecessors,
                y_spacing,
            )


def _push_ports_from_termini(
    graph: MetroGraph,
    port_ids: list[str],
    termini: list[tuple[str, float]],
    section: Section,
    adjacency: dict[str, set[str]],
    predecessors: dict[str, set[str]],
    y_spacing: float,
) -> None:
    """Ensure *y_spacing* between each port and non-connected termini.

    The strategy depends on how the port connects across sections:

    - **Junction link** (fan-out): move the port and propagate through
      the junction to its *upstream* (predecessor) port only, keeping
      the exit-junction-entry chain straight without disturbing other
      fan-out targets.
    - **Direct port-to-port link** (no junction): moving the port would
      cascade to the other section and mis-align its internal stations.
      Instead, push the conflicting *terminus* away from the port.
    - **No cross-section link**: move the port freely.

    *port_ids* must be a sorted list so that results are deterministic
    when multiple ports in the same section conflict with the same
    terminus.
    """
    junction_ids = set(graph.junctions)
    section_port_set = set(port_ids)

    for pid in port_ids:
        port_st = graph.stations.get(pid)
        if not port_st:
            continue
        port_obj = graph.ports.get(pid)
        assert port_obj is not None, f"port {pid} missing from graph.ports"
        neighbours = adjacency.get(pid, set())

        # Classify cross-section connection type.
        has_junction = bool(neighbours & junction_ids)
        has_direct_port = False
        if not has_junction:
            for nb in neighbours:
                if nb in graph.ports and nb not in section_port_set:
                    has_direct_port = True
                    break

        # Collect all termini that are too close and not directly
        # connected to this port.
        conflict_ids: list[str] = []
        conflict_ys: list[float] = []
        for tid, ty in termini:
            if tid in neighbours:
                continue
            if abs(port_st.y - ty) < y_spacing:
                conflict_ids.append(tid)
                conflict_ys.append(ty)

        if not conflict_ys:
            continue

        if has_direct_port:
            # Move the terminus instead of the port so the
            # inter-section line stays straight.
            _push_termini_from_port(graph, conflict_ids, port_st.y, section, y_spacing)
            continue

        # Compute the single best Y that satisfies all conflicts.
        above_candidates = [ty - y_spacing for ty in conflict_ys]
        below_candidates = [ty + y_spacing for ty in conflict_ys]

        best_above = min(above_candidates)
        best_below = max(below_candidates)

        dist_above = abs(port_st.y - best_above)
        dist_below = abs(port_st.y - best_below)
        # Ties go above (smaller Y) to keep ports near the top.
        new_y = best_above if dist_above <= dist_below else best_below

        port_st.y = new_y
        port_obj.y = new_y

        # Propagate through junctions so inter-section lines stay straight.
        _propagate_through_junctions(
            graph,
            pid,
            new_y,
            neighbours,
            junction_ids,
            predecessors,
        )

        # Grow this section's bbox to contain the moved port.
        _expand_bbox_for_y(section, new_y)


def _propagate_through_junctions(
    graph: MetroGraph,
    origin_pid: str,
    new_y: float,
    neighbours: set[str],
    junction_ids: set[str],
    predecessors: dict[str, set[str]],
) -> None:
    """Move connected junctions and their upstream exit ports to *new_y*.

    Only propagates to the junction's upstream (predecessor) ports, not
    to other fan-out targets (entry ports to other sections).
    """
    for nb in neighbours:
        if nb not in junction_ids:
            continue
        nb_st = graph.stations.get(nb)
        if not nb_st:
            continue

        nb_st.y = new_y
        for jnb in predecessors.get(nb, set()):
            if jnb == origin_pid:
                continue
            jnb_st = graph.stations.get(jnb)
            if not jnb_st or not jnb_st.is_port:
                continue
            jnb_st.y = new_y
            jnb_obj = graph.ports.get(jnb)
            if jnb_obj:
                jnb_obj.y = new_y
                jnb_sec = graph.sections.get(jnb_obj.section_id)
                if jnb_sec:
                    _expand_bbox_for_y(jnb_sec, new_y)


def _push_termini_from_port(
    graph: MetroGraph,
    terminus_ids: list[str],
    port_y: float,
    section: Section,
    y_spacing: float,
) -> None:
    """Push terminus stations to the nearest station row that clears the port.

    Instead of placing the terminus at the arbitrary ``port_y ± y_spacing``,
    snap it to an existing station Y in the section that satisfies the
    minimum clearance.  This keeps the terminus aligned with an actual
    track row rather than floating at an unrelated Y coordinate.
    """
    # Collect existing station Y values in the section (excluding ports
    # and the termini being moved) as candidate snap targets.
    port_ids = set(section.entry_ports) | set(section.exit_ports)
    tid_set = set(terminus_ids)
    section_ys: set[float] = set()
    for sid in section.station_ids:
        if sid in port_ids or sid in tid_set:
            continue
        st = graph.stations.get(sid)
        if st and not st.is_port:
            section_ys.add(st.y)

    for tid in terminus_ids:
        t_st = graph.stations.get(tid)
        if not t_st:
            continue

        # Determine push direction
        going_down = t_st.y > port_y

        # Find the nearest section row Y that satisfies clearance
        candidates = sorted(
            (y for y in section_ys if abs(y - port_y) >= y_spacing),
            key=lambda y: abs(y - t_st.y),
        )
        if going_down:
            candidates = [y for y in candidates if y >= port_y + y_spacing]
        else:
            candidates = [y for y in candidates if y <= port_y - y_spacing]

        if candidates:
            new_y = candidates[0]
        else:
            # No existing row satisfies clearance; fall back to offset.
            new_y = (port_y + y_spacing) if going_down else (port_y - y_spacing)

        t_st.y = new_y
        _expand_bbox_for_y(section, new_y)


def _expand_bbox_for_y(section: Section, y: float) -> None:
    """Expand *section*'s bbox so *y* sits inside with padding."""
    pad = SECTION_Y_PADDING
    top = section.bbox_y
    bot = section.bbox_y + section.bbox_h
    if y - pad < top:
        section.bbox_h += top - (y - pad)
        section.bbox_y = y - pad
    elif y + pad > bot:
        section.bbox_h = (y + pad) - section.bbox_y


def _build_section_subgraph(graph: MetroGraph, section: Section) -> MetroGraph:
    """Build a temporary MetroGraph containing only a section's real stations and edges.

    Excludes port stations and any edges that touch ports. Ports are positioned
    separately on section boundaries after the internal layout is computed.
    """
    sub = MetroGraph()
    sub.lines = graph.lines  # Share line definitions
    sub.diamond_style = graph.diamond_style

    # Collect port IDs for this section
    port_ids = set(section.entry_ports) | set(section.exit_ports)

    # Add only real (non-port) stations belonging to this section
    real_station_ids: set[str] = set()
    for sid in section.station_ids:
        if sid in port_ids:
            continue
        if sid in graph.stations:
            station = graph.stations[sid]
            if station.is_port:
                continue
            sub.add_station(
                Station(
                    id=station.id,
                    label=station.label,
                    section_id=station.section_id,
                    is_port=False,
                    is_terminus=station.is_terminus,
                    terminus_label=station.terminus_label,
                )
            )
            real_station_ids.add(sid)

    # Add only edges between real stations (no port-touching edges)
    for edge in graph.edges:
        if edge.source in real_station_ids and edge.target in real_station_ids:
            sub.add_edge(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    line_id=edge.line_id,
                )
            )

    return sub


def _compute_fork_join_gaps(
    sub: MetroGraph,
    layers: dict[str, int],
    tracks: dict[str, float],
    x_spacing: float,
    full_graph: MetroGraph | None = None,
    section_station_ids: set[str] | None = None,
) -> dict[int, float]:
    """Compute extra X offset per layer at fork/join points.

    Adds a fractional gap after fork layers (where tracks diverge) and
    before join layers (where tracks converge) so labels aren't obscured
    by diagonal crossings.

    When full_graph and section_station_ids are provided, fork/join
    detection uses all edges within the section (including port-touching
    edges). This catches divergences where a station connects to both
    internal stations and exit ports.

    In single-track sections (all stations on the same Y), port-bound
    divergences are suppressed because there are no diagonal transitions
    and the extra spacing is purely wasteful.
    """
    from collections import defaultdict

    out_targets: dict[str, set[str]] = defaultdict(set)
    in_sources: dict[str, set[str]] = defaultdict(set)

    # Use full graph edges for fork/join detection when available,
    # so that edges to/from port stations are counted as divergences.
    if full_graph is not None and section_station_ids is not None:
        for edge in full_graph.edges:
            src_in = edge.source in section_station_ids
            tgt_in = edge.target in section_station_ids
            if src_in and tgt_in:
                out_targets[edge.source].add(edge.target)
                in_sources[edge.target].add(edge.source)
    else:
        for edge in sub.edges:
            out_targets[edge.source].add(edge.target)
            in_sources[edge.target].add(edge.source)

    # Only count forks/joins that span multiple tracks (requiring a
    # diagonal routing transition).  Same-track fan-outs (e.g. a station
    # connecting to both an internal successor and an exit port on the
    # same Y) don't need extra horizontal room.
    #
    # Port stations aren't in ``tracks`` (they're positioned later), so
    # treat them conservatively: if any participant is missing from
    # tracks, assume it may be on a different track and count the
    # fork/join.
    #
    # Exception for **forks** in single-track sections: exit-side ports
    # sit at the far section boundary, so the diagonal from the fork
    # station has ample horizontal room without extra layer spacing.
    # Join gaps are kept even in single-track sections because entry
    # ports are close to the first internal station, and the diagonal
    # from a different-Y entry needs the extra room.
    all_section_tracks = set(tracks.values())
    is_single_track = len(all_section_tracks) <= 1

    fork_layers: set[int] = set()
    for sid, targets in out_targets.items():
        if len(targets) > 1 and sid in layers:
            if any(t not in tracks for t in targets):
                if not is_single_track:
                    fork_layers.add(layers[sid])
            else:
                target_tracks = {tracks[t] for t in targets}
                if len(target_tracks) > 1:
                    fork_layers.add(layers[sid])

    join_layers: set[int] = set()
    for sid, sources in in_sources.items():
        if len(sources) > 1 and sid in layers:
            if any(s not in tracks for s in sources):
                join_layers.add(layers[sid])
            else:
                source_tracks = {tracks[s] for s in sources}
                if len(source_tracks) > 1:
                    join_layers.add(layers[sid])

    if not fork_layers and not join_layers:
        return {}

    max_layer = max(layers.values()) if layers else 0
    base_gap = x_spacing * EXIT_GAP_MULTIPLIER

    # Compute per-layer gap scaled by label width at fork/join stations.
    # The gap must be large enough that the diagonal transition starts
    # past the label text and still has room for the transition itself.
    layer_gap: dict[int, float] = {}
    for layer in fork_layers | join_layers:
        max_label_half = 0.0
        for sid, lyr in layers.items():
            if lyr == layer:
                station = sub.stations.get(sid)
                if station and station.label.strip():
                    label_half = label_text_width(station.label) / 2
                    max_label_half = max(max_label_half, label_half)
        layer_gap[layer] = max(base_gap, max_label_half)

    cumulative = 0.0
    layer_extra: dict[int, float] = {}
    for layer in range(max_layer + 1):
        # Add gap before join layers
        if layer in join_layers:
            cumulative += layer_gap.get(layer, base_gap)
        layer_extra[layer] = cumulative
        # Add gap after fork layers
        if layer in fork_layers:
            cumulative += layer_gap.get(layer, base_gap)

    return layer_extra
