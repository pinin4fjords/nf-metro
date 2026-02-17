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
    x_spacing: float = 60.0,
    y_spacing: float = 40.0,
    x_offset: float = 80.0,
    y_offset: float = 120.0,
) -> None:
    """Flat layout for sectionless pipelines.

    Runs layer/track assignment directly on the full graph and maps
    to coordinates without section boxes or port routing.
    """
    layers = assign_layers(graph)
    tracks = assign_tracks(graph, layers)

    if not layers:
        return

    unique_tracks = sorted(set(tracks.values()))
    track_rank = {t: i for i, t in enumerate(unique_tracks)}

    layer_extra = _compute_fork_join_gaps(graph, layers, x_spacing)

    for sid, station in graph.stations.items():
        station.layer = layers.get(sid, 0)
        station.track = tracks.get(sid, 0)
        station.x = x_offset + station.layer * x_spacing + layer_extra.get(
            station.layer, 0
        )
        station.y = y_offset + track_rank[station.track] * y_spacing


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
        # aren't too close to divergence/convergence points.
        # Pass full graph so port-touching edges count as forks/joins.
        section_sids = set(section.station_ids)
        layer_extra = _compute_fork_join_gaps(
            sub, layers, x_spacing, graph, section_sids
        )

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

        # TB sections with TOP entry from a cross-column source: shift
        # stations down so there's room for the L-shape entry routing
        # above the first station. Without this, the entry port sits at
        # the bbox top and lines from adjacent columns go all the way up.
        if section.direction == "TB":
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
                entry_shift = y_spacing * 1.0
                for s in sub.stations.values():
                    s.y += entry_shift
                section.bbox_h += entry_shift

        # LR/RL sections with TOP/BOTTOM entry: add extra bbox width so
        # the grid allocates space for the vertical-to-horizontal curve.
        if section.direction in ("LR", "RL"):
            has_perp_entry = any(
                graph.ports[pid].side in (PortSide.TOP, PortSide.BOTTOM)
                for pid in section.entry_ports
                if pid in graph.ports
            )
            if has_perp_entry:
                entry_inset = x_spacing * 0.3
                section.bbox_w += entry_inset

        # LR/RL sections with exit ports on the flow side: add clearance
        # so diagonal routing from the last station to the exit port doesn't
        # overlap station labels. Gap is proportional to the longest label at
        # the exit-side layer, mirroring _compute_fork_join_gaps logic.
        if section.direction in ("LR", "RL"):
            flow_exit_side = (
                PortSide.RIGHT if section.direction == "LR" else PortSide.LEFT
            )
            has_flow_exit = any(
                graph.ports[pid].side == flow_exit_side
                for pid in section.exit_ports
                if pid in graph.ports
            )
            if has_flow_exit and layers:
                max_layer = max(layers.values())
                char_width = 7.0
                max_label_half = 0.0
                for sid_l, l in layers.items():
                    if l == max_layer:
                        station = sub.stations.get(sid_l)
                        if station and station.label.strip():
                            label_half = len(station.label) * char_width / 2
                            max_label_half = max(max_label_half, label_half)
                exit_gap = max(x_spacing * 0.4, max_label_half)
                if section.direction == "LR":
                    section.bbox_w += exit_gap
                else:
                    # Shift stations right to create clearance on the left
                    # (exit) side without moving the bbox boundary, which
                    # would misalign the section within its grid column.
                    for s in sub.stations.values():
                        s.x += exit_gap
                    section.bbox_w += exit_gap

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
                        # Skip alignment if source Y is outside entry section bbox
                        # (e.g. tall rspan source whose center is far from this section)
                        entry_station = graph.stations.get(port_id)
                        if entry_station:
                            bbox_top = entry_section.bbox_y
                            bbox_bot = entry_section.bbox_y + entry_section.bbox_h
                            if not (bbox_top <= src.y <= bbox_bot):
                                break
                        target_y = src.y

                        # Clamp for TB sections with perpendicular entry:
                        # the entry port must stay above the first internal
                        # station so the direction-change curve has room.
                        if (
                            entry_section.direction == "TB"
                            and port.side in (PortSide.LEFT, PortSide.RIGHT)
                        ):
                            internal_ids = (
                                set(entry_section.station_ids)
                                - set(entry_section.entry_ports)
                                - set(entry_section.exit_ports)
                            )
                            internal_ys = [
                                graph.stations[sid].y
                                for sid in internal_ids
                                if sid in graph.stations
                                and not graph.stations[sid].is_port
                            ]
                            if internal_ys:
                                first_y = min(internal_ys)
                                min_gap = 16.0
                                max_y = first_y - min_gap
                                if target_y > max_y:
                                    # Prefer the topmost source-side
                                    # station feeding the exit port so
                                    # that line exits horizontally.
                                    exit_pid = edge.source
                                    if edge.source in junction_ids:
                                        for e2 in graph.edges:
                                            if e2.target == edge.source:
                                                ep = graph.stations.get(
                                                    e2.source
                                                )
                                                if ep and ep.is_port:
                                                    exit_pid = e2.source
                                                    break
                                    top_src_y = None
                                    for e3 in graph.edges:
                                        if e3.target == exit_pid:
                                            s3 = graph.stations.get(
                                                e3.source
                                            )
                                            if (
                                                s3
                                                and not s3.is_port
                                                and e3.source
                                                not in junction_ids
                                            ):
                                                if (
                                                    top_src_y is None
                                                    or s3.y < top_src_y
                                                ):
                                                    top_src_y = s3.y
                                    if (
                                        top_src_y is not None
                                        and top_src_y < max_y
                                    ):
                                        target_y = top_src_y
                                    else:
                                        target_y = max_y
                                    # Pull source up to maintain straight
                                    # horizontal run
                                    src.y = target_y
                                    if (
                                        src.is_port
                                        and edge.source in graph.ports
                                    ):
                                        graph.ports[edge.source].y = target_y
                                    # If source is a junction, also pull
                                    # the exit port feeding it
                                    if edge.source in junction_ids:
                                        for e2 in graph.edges:
                                            if e2.target == edge.source:
                                                ep = graph.stations.get(
                                                    e2.source
                                                )
                                                if ep and ep.is_port:
                                                    ep.y = target_y
                                                    if (
                                                        e2.source
                                                        in graph.ports
                                                    ):
                                                        graph.ports[
                                                            e2.source
                                                        ].y = target_y

                        station = graph.stations.get(port_id)
                        if station:
                            station.y = target_y
                        port.y = target_y
                    break

        elif port.side in (PortSide.TOP, PortSide.BOTTOM):
            # Collect all incoming sources to determine alignment
            sources: list[tuple[Station, str | None]] = []  # (station, section_id)
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
                    sources.append((src, src_section_id))

            if not sources:
                continue

            # Check if any source is cross-column
            my_cols = set(range(
                entry_section.grid_col,
                entry_section.grid_col + entry_section.grid_col_span,
            ))
            is_cross_column = False
            for _, src_sid in sources:
                src_sec = graph.sections.get(src_sid) if src_sid else None
                if src_sec:
                    src_cols = set(range(
                        src_sec.grid_col,
                        src_sec.grid_col + src_sec.grid_col_span,
                    ))
                    if not (src_cols & my_cols):
                        is_cross_column = True
                        break

            if is_cross_column:
                # Cross-column: don't align X. Set Y to the closest
                # source level so lines converge there instead of
                # going all the way to the bbox boundary.
                src_ys = [s.y for s, _ in sources]
                if port.side == PortSide.TOP:
                    target_y = min(src_ys)
                else:
                    target_y = max(src_ys)
                # Clamp within bbox
                target_y = max(target_y, entry_section.bbox_y)
                target_y = min(
                    target_y, entry_section.bbox_y + entry_section.bbox_h
                )
                station = graph.stations.get(port_id)
                if station:
                    station.y = target_y
                port.y = target_y
                # Only nudge X for LR/RL sections where TOP/BOTTOM ports
                # are perpendicular.  For TB sections, TOP/BOTTOM ports
                # SHOULD share X with internal stations (flow direction).
                if entry_section.direction in ("LR", "RL"):
                    _nudge_port_from_stations(
                        port_id, entry_section, graph
                    )
            else:
                # Same-column: align X with source for vertical drop
                src, _ = sources[0]
                station = graph.stations.get(port_id)
                if station:
                    station.x = src.x
                port.x = src.x


def _nudge_port_from_stations(
    port_id: str, section: Section, graph: MetroGraph, tolerance: float = 12.0
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
            # Find the target of this exit port
            for edge in graph.edges:
                if edge.source == port_id:
                    tgt = graph.stations.get(edge.target)
                    if not tgt:
                        continue

                    # If target is a junction (fan-out), don't align --
                    # the junction routes to multiple entry ports at
                    # different Y levels; aligning with one is arbitrary.
                    if edge.target in junction_ids:
                        break

                    if tgt.is_port:
                        # Don't align with perpendicular target ports (cross-axis)
                        tgt_port_obj = graph.ports.get(tgt.id)
                        if tgt_port_obj and tgt_port_obj.side in (
                            PortSide.TOP,
                            PortSide.BOTTOM,
                        ):
                            break
                        # Don't pull exit port outside its section bbox
                        bbox_top = exit_section.bbox_y
                        bbox_bot = exit_section.bbox_y + exit_section.bbox_h
                        if not (bbox_top <= tgt.y <= bbox_bot):
                            break
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
    internal stations and exit ports (e.g. umi_tools_dedup forking to
    salmon_quant and an exit port).
    """
    from collections import defaultdict

    out_targets: dict[str, set[str]] = defaultdict(set)
    in_sources: dict[str, set[str]] = defaultdict(set)

    # Use full graph edges for fork/join detection when available,
    # so that edges to/from port stations are counted as divergences.
    if full_graph is not None and section_station_ids is not None:
        for edge in full_graph.edges:
            if edge.source in section_station_ids and edge.target in section_station_ids:
                out_targets[edge.source].add(edge.target)
                in_sources[edge.target].add(edge.source)
    else:
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
    base_gap = x_spacing * 0.4

    # Compute per-layer gap scaled by label width at fork/join stations.
    # The gap must be large enough that the diagonal transition starts
    # past the label text and still has room for the transition itself.
    char_width = 7.0
    layer_gap: dict[int, float] = {}
    for layer in fork_layers | join_layers:
        max_label_half = 0.0
        for sid, l in layers.items():
            if l == layer:
                station = sub.stations.get(sid)
                if station and station.label.strip():
                    label_half = len(station.label) * char_width / 2
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
