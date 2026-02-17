"""Layout coordinator: combines layer assignment, ordering, and coordinate mapping.

Section-first layout: sections are laid out independently, then placed on a meta-graph.
"""

from __future__ import annotations

from nf_metro.layout.layers import assign_layers
from nf_metro.layout.ordering import assign_tracks
from nf_metro.parser.model import Edge, MetroGraph, PortSide, Section, Station


def compute_layout(
    graph: MetroGraph,
    x_spacing: float = 60.0,
    y_spacing: float = 40.0,
    x_offset: float = 80.0,
    y_offset: float = 120.0,
    row_gap: float = 120.0,
    section_gap: float = 3.0,
    section_x_padding: float = 50.0,
    section_y_padding: float = 35.0,
    section_x_gap: float = 50.0,
    section_y_gap: float = 40.0,
) -> None:
    """Compute layout positions for all stations in the graph."""
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


def _compute_section_layout(
    graph: MetroGraph,
    x_spacing: float = 60.0,
    y_spacing: float = 40.0,
    x_offset: float = 80.0,
    y_offset: float = 120.0,
    section_x_padding: float = 50.0,
    section_y_padding: float = 35.0,
    section_x_gap: float = 50.0,
    section_y_gap: float = 40.0,
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
        sub = _build_section_subgraph(graph, section)
        if not sub.stations:
            continue

        # Run standard layout on the sub-graph
        layers = assign_layers(sub)
        tracks = assign_tracks(sub, layers)

        if not layers:
            continue

        # Compact tracks to consecutive integers so widely-spaced
        # line priorities don't inflate the vertical spread.
        unique_tracks = sorted(set(tracks.values()))
        track_rank = {t: i for i, t in enumerate(unique_tracks)}

        # Detect fork/join layers and add extra spacing so stations
        # aren't too close to divergence/convergence points
        layer_extra = _compute_fork_join_gaps(sub, layers, x_spacing)

        for sid, station in sub.stations.items():
            station.layer = layers.get(sid, 0)
            station.track = tracks.get(sid, 0)
            if section.direction == "TB":
                # Top-to-bottom: layers map to Y, tracks map to X.
                # All stations stay in the same vertical column so
                # the flow is a clean straight-down path. The entry
                # port connects horizontally from the section edge.
                station.x = track_rank[station.track] * x_spacing
                station.y = station.layer * y_spacing + layer_extra.get(
                    station.layer, 0
                )
            else:
                station.x = station.layer * x_spacing + layer_extra.get(
                    station.layer, 0
                )
                station.y = track_rank[station.track] * y_spacing

        # Normalize Y so minimum is 0 (raw tracks can be negative)
        ys_all = [s.y for s in sub.stations.values()]
        if ys_all:
            min_y = min(ys_all)
            if min_y != 0:
                for s in sub.stations.values():
                    s.y -= min_y

        # RL: mirror X so layer 0 is rightmost.
        # Anchor on non-terminus stations so adding terminus layers
        # extends leftward without shifting the entry point.
        if section.direction == "RL":
            non_term = [
                s
                for s in sub.stations.values()
                if not (s.is_terminus and not s.label.strip())
            ]
            anchor_stations = non_term if non_term else list(sub.stations.values())
            max_x_val = max(s.x for s in anchor_stations)
            for s in sub.stations.values():
                s.x = max_x_val - s.x

        # Normalize local X so leftmost station is at x=0.
        # After RL mirror, terminus stations may have negative X; normalizing
        # ensures bbox_x is always at -padding, and extra width from terminus
        # goes into bbox_w (which feeds into grid column sizing).
        min_local_x = min(s.x for s in sub.stations.values())
        if min_local_x != 0:
            for s in sub.stations.values():
                s.x -= min_local_x

        # Ensure minimum inner extent so stations sit on visible track
        xs = [s.x for s in sub.stations.values()]
        ys = [s.y for s in sub.stations.values()]
        if section.direction == "TB":
            inner_h = max(ys) - min(ys)
            min_inner_h = y_spacing
            if inner_h < min_inner_h:
                shift = (min_inner_h - inner_h) / 2
                for station in sub.stations.values():
                    station.y += shift
                ys = [s.y for s in sub.stations.values()]
        else:
            inner_w = max(xs) - min(xs)
            min_inner_w = x_spacing
            if inner_w < min_inner_w:
                shift = (min_inner_w - inner_w) / 2
                for station in sub.stations.values():
                    station.x += shift
                xs = [s.x for s in sub.stations.values()]

        # Compute section bounding box from real stations only
        section.bbox_x = min(xs) - section_x_padding
        section.bbox_y = min(ys) - section_y_padding
        section.bbox_w = (max(xs) - min(xs)) + section_x_padding * 2
        section.bbox_h = (max(ys) - min(ys)) + section_y_padding * 2

        # TB sections: labels extend leftward from the station (text_anchor=end).
        # Expand bbox and shift stations right so labels fit within the section.
        if section.direction == "TB":
            char_width = 7.0
            label_pad = 6.0
            max_label_extent = 0.0
            for sid, s in sub.stations.items():
                if s.label.strip():
                    n_lines = len(sub.station_lines(sid))
                    offset_span = (n_lines - 1) * 3.0
                    extent = offset_span / 2 + 11 + len(s.label) * char_width
                    max_label_extent = max(max_label_extent, extent)
            need_left = max_label_extent + label_pad
            have_left = min(xs) - section.bbox_x
            if need_left > have_left:
                extra = need_left - have_left
                for s in sub.stations.values():
                    s.x += extra
                xs = [s.x for s in sub.stations.values()]
                section.bbox_w += extra

        # Direction-change offset: when a section has an entry port
        # perpendicular to its flow, shift stations in the flow direction
        # so the first station isn't at the entry port (avoiding the
        # station-as-elbow problem at direction changes).
        if section.direction == "TB":
            has_perp_entry = any(
                graph.ports[pid].side in (PortSide.LEFT, PortSide.RIGHT)
                for pid in section.entry_ports
                if pid in graph.ports
            )
            if has_perp_entry:
                entry_shift = y_spacing * 0.6
                for s in sub.stations.values():
                    s.y += entry_shift
                section.bbox_h += entry_shift

        # LR/RL sections with TOP/BOTTOM entry: add extra bbox width so
        # the grid allocates space for the vertical-to-horizontal curve.
        # Unlike TB sections, no station shift is needed here because the
        # entry port (on TOP/BOTTOM boundary) and the first station are
        # already at different positions along the perpendicular axis.
        if section.direction in ("LR", "RL"):
            has_perp_entry = any(
                graph.ports[pid].side in (PortSide.TOP, PortSide.BOTTOM)
                for pid in section.entry_ports
                if pid in graph.ports
            )
            if has_perp_entry:
                entry_inset = x_spacing * 0.3
                section.bbox_w += entry_inset

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

    # Phase 4b: Center content vertically in row-spanning non-TB sections.
    # Rowspan expansion (from _expand_lone_rowspans) stretches the bbox but
    # leaves stations at the top. Shift them down to center within the box.
    for sec_id, section in graph.sections.items():
        if section.grid_row_span <= 1 or section.direction == "TB":
            continue
        sub = section_subgraphs.get(sec_id)
        if not sub:
            continue
        ys = [graph.stations[sid].y for sid in sub.stations if sid in graph.stations]
        if not ys:
            continue
        content_h = (max(ys) - min(ys)) + 2 * section_y_padding
        if content_h >= section.bbox_h:
            continue
        content_top = min(ys) - section_y_padding
        shift = (section.bbox_h - content_h) / 2 - (content_top - section.bbox_y)
        for sid in sub.stations:
            if sid in graph.stations:
                graph.stations[sid].y += shift

    # Phase 5: Position ports on section boundaries (after bbox is in global coords)
    for sec_id, section in graph.sections.items():
        position_ports(section, graph)

    # Phase 6: Position junction stations in the inter-section gap
    _position_junctions(graph)

    # Phase 7: Align LEFT/RIGHT entry ports with their incoming connection's Y
    # so inter-section horizontal runs are straight
    _align_entry_ports(graph)

    # Phase 8: Align LEFT/RIGHT exit ports on row-spanning (fold) sections
    # with their target's Y so the exit is at the return row level
    _align_exit_ports(graph)

    # Phase 9: Re-position junctions after exit port alignment.
    # Phase 8 may have moved exit ports on fold sections, so junctions
    # placed in Phase 6 need updating to match the new exit port Y.
    _position_junctions(graph)



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
        entry_port_xs: list[float] = []

        for edge in graph.edges:
            if edge.target == jid:
                src = graph.stations.get(edge.source)
                if src and src.is_port:
                    exit_port_x = src.x
                    exit_port_y = src.y
            if edge.source == jid:
                tgt = graph.stations.get(edge.target)
                if tgt and tgt.is_port:
                    entry_port_xs.append(tgt.x)

        if exit_port_x is not None and exit_port_y is not None and entry_port_xs:
            # Position close to the exit port with a small margin,
            # rather than at the midpoint. This keeps the divergence
            # point near the source section so lines turn sooner.
            margin = 10.0
            nearest_entry_x = min(entry_port_xs, key=lambda x: abs(x - exit_port_x))
            direction = 1.0 if nearest_entry_x > exit_port_x else -1.0
            junction.x = exit_port_x + direction * margin
            junction.y = exit_port_y


def _align_entry_ports(graph: MetroGraph) -> None:
    """Align LEFT/RIGHT entry ports with their incoming connection's Y.

    Only aligns when the entry port's section is in the same grid row as
    the source's section, so horizontal runs between adjacent sections are
    straight. Ports in different rows keep their position for L-shaped routing.
    """
    junction_ids = set(graph.junctions)

    for port_id, port in graph.ports.items():
        if not port.is_entry:
            continue

        entry_section = graph.sections.get(port.section_id)
        if not entry_section:
            continue

        if port.side in (PortSide.LEFT, PortSide.RIGHT):
            # Align Y with incoming source so horizontal runs are straight
            for edge in graph.edges:
                if edge.target == port_id:
                    src = graph.stations.get(edge.source)
                    if not src or not (src.is_port or edge.source in junction_ids):
                        continue

                    src_section_id = src.section_id
                    if edge.source in junction_ids:
                        for e2 in graph.edges:
                            if e2.target == edge.source:
                                s2 = graph.stations.get(e2.source)
                                if s2 and s2.section_id:
                                    src_section_id = s2.section_id
                                    break

                    src_section = (
                        graph.sections.get(src_section_id) if src_section_id else None
                    )
                    if not src_section:
                        continue

                    if entry_section.grid_row == src_section.grid_row:
                        station = graph.stations.get(port_id)
                        if station:
                            station.y = src.y
                        port.y = src.y
                    break

        elif port.side in (PortSide.TOP, PortSide.BOTTOM):
            # Align X with incoming source so vertical drops are straight
            for edge in graph.edges:
                if edge.target == port_id:
                    src = graph.stations.get(edge.source)
                    if not src or not (src.is_port or edge.source in junction_ids):
                        continue
                    station = graph.stations.get(port_id)
                    if station:
                        station.x = src.x
                    port.x = src.x
                    break


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
            # Find the target of this exit port (could go through a junction)
            for edge in graph.edges:
                if edge.source == port_id:
                    tgt = graph.stations.get(edge.target)
                    if not tgt:
                        continue

                    # If target is a junction, follow through to entry port
                    if edge.target in junction_ids:
                        for e2 in graph.edges:
                            if e2.source == edge.target:
                                t2 = graph.stations.get(e2.target)
                                if t2 and t2.is_port:
                                    tgt = t2
                                    break

                    if tgt.is_port or edge.target in junction_ids:
                        station = graph.stations.get(port_id)
                        if station:
                            station.y = tgt.y
                        port.y = tgt.y
                    break



def _build_section_subgraph(graph: MetroGraph, section: Section) -> MetroGraph:
    """Build a temporary MetroGraph containing only a section's real stations and edges.

    Excludes port stations and any edges that touch ports. Ports are positioned
    separately on section boundaries after the internal layout is computed.
    """
    sub = MetroGraph()
    sub.lines = graph.lines  # Share line definitions

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
    x_spacing: float,
) -> dict[int, float]:
    """Compute extra X offset per layer at fork/join points.

    Adds a fractional gap after fork layers (where tracks diverge) and
    before join layers (where tracks converge) so labels aren't obscured
    by diagonal crossings.
    """
    from collections import defaultdict

    out_targets: dict[str, set[str]] = defaultdict(set)
    in_sources: dict[str, set[str]] = defaultdict(set)
    for edge in sub.edges:
        out_targets[edge.source].add(edge.target)
        in_sources[edge.target].add(edge.source)

    fork_layers = {
        layers[sid]
        for sid, targets in out_targets.items()
        if len(targets) > 1 and sid in layers
    }
    join_layers = {
        layers[sid]
        for sid, sources in in_sources.items()
        if len(sources) > 1 and sid in layers
    }

    if not fork_layers and not join_layers:
        return {}

    max_layer = max(layers.values()) if layers else 0
    gap = x_spacing * 0.4

    cumulative = 0.0
    layer_extra: dict[int, float] = {}
    for layer in range(max_layer + 1):
        # Add gap before join layers
        if layer in join_layers:
            cumulative += gap
        layer_extra[layer] = cumulative
        # Add gap after fork layers
        if layer in fork_layers:
            cumulative += gap

    return layer_extra
