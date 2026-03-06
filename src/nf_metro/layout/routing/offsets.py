"""Station offset computation for per-line Y positioning within bundles."""

from __future__ import annotations

from nf_metro.layout.constants import OFFSET_STEP
from nf_metro.layout.routing.reversal import detect_reversed_sections
from nf_metro.parser.model import MetroGraph, PortSide


def compute_station_offsets(
    graph: MetroGraph,
    offset_step: float = OFFSET_STEP,
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
    compact = getattr(graph, "compact_offsets", False)

    # Pre-compute per-station inbound/outbound line sets for compact mode
    if compact:
        inbound: dict[str, set[str]] = {sid: set() for sid in graph.stations}
        outbound: dict[str, set[str]] = {sid: set() for sid in graph.stations}
        for edge in graph.edges:
            if edge.target in inbound:
                inbound[edge.target].add(edge.line_id)
            if edge.source in outbound:
                outbound[edge.source].add(edge.line_id)

    reversed_sections = detect_reversed_sections(graph)

    offsets: dict[tuple[str, str], float] = {}
    for sid in graph.stations:
        lines = graph.station_lines(sid)
        if not lines:
            continue
        station = graph.stations[sid]
        reverse = station.section_id in reversed_sections

        if compact:
            # Only allocate offset slots for the max lines on either
            # side.  When a station has one line entering and a
            # different line exiting, max_side=1 and all offsets are 0
            # (dot instead of pill).
            max_side = max(len(inbound[sid]), len(outbound[sid]), 1)
            if max_side <= 1:
                for lid in lines:
                    offsets[(sid, lid)] = 0.0
            else:
                # Use the busier side to determine index ordering
                if len(inbound[sid]) >= len(outbound[sid]):
                    ref = inbound[sid]
                else:
                    ref = outbound[sid]
                ref_sorted = sorted(ref, key=lambda lid: line_priority.get(lid, 0))
                ref_idx = {lid: i for i, lid in enumerate(ref_sorted)}
                local_max = max_side - 1
                for lid in lines:
                    idx = ref_idx.get(lid, None)
                    if idx is None:
                        # Line not on the reference (busier) side;
                        # no spreading needed.
                        offsets[(sid, lid)] = 0.0
                    elif reverse:
                        offsets[(sid, lid)] = (local_max - idx) * offset_step
                    else:
                        offsets[(sid, lid)] = idx * offset_step
        else:
            for lid in lines:
                p = line_priority.get(lid, 0)
                if reverse:
                    offsets[(sid, lid)] = (max_priority - p) * offset_step
                else:
                    offsets[(sid, lid)] = p * offset_step

    # Section-local priority re-indexing for non-compact mode.
    # Lines absent from a section should not reserve offset slots within
    # it.  Re-index priorities per-section so only present lines get
    # slots, closing gaps while preserving relative ordering.
    # Only re-index when there are actual gaps in the global priorities
    # of the present lines; otherwise the global offsets are already
    # compact and re-indexing would break cross-section continuity.
    if not compact:
        section_local: dict[str, dict[str, int]] = {}
        for sec_id in graph.sections:
            present: set[str] = set()
            for sid_s, station in graph.stations.items():
                if station.section_id == sec_id:
                    present |= set(graph.station_lines(sid_s))
            ordered = sorted(present, key=lambda lid: line_priority.get(lid, 0))
            # Check for gaps: if global priorities are already consecutive
            # (e.g. [0,1,2]), re-indexing is a no-op and we skip to
            # preserve cross-section offset continuity.
            global_pris = [line_priority.get(lid, 0) for lid in ordered]
            has_gap = any(
                global_pris[i + 1] - global_pris[i] > 1
                for i in range(len(global_pris) - 1)
            )
            if has_gap:
                section_local[sec_id] = {lid: i for i, lid in enumerate(ordered)}
        for sid_s, station in graph.stations.items():
            sec_id = station.section_id
            if sec_id not in section_local:
                continue
            local_pri = section_local[sec_id]
            local_max = max(local_pri.values()) if local_pri else 0
            reverse = sec_id in reversed_sections
            for lid in graph.station_lines(sid_s):
                p = local_pri.get(lid, 0)
                if reverse:
                    offsets[(sid_s, lid)] = (local_max - p) * offset_step
                else:
                    offsets[(sid_s, lid)] = p * offset_step

        # Reconvergence ordering: when multiple upstream sections feed
        # into one section, lines from the "primary" feeder (the one
        # contributing the most lines) keep their relative offsets at
        # the top.  Lines returning from diversions are appended below,
        # so the continuing bundle transitions without slanting.
        for sec_id, section in graph.sections.items():
            if not section.entry_ports:
                continue
            # Determine which feeder section each line comes from
            line_feeder: dict[str, str] = {}
            for pid in section.entry_ports:
                for edge in graph.edges:
                    if edge.target != pid:
                        continue
                    src = graph.stations.get(edge.source)
                    if not src:
                        continue
                    feeder_sec = None
                    if src.is_port:
                        feeder_sec = src.section_id
                    elif edge.source in graph.junctions:
                        # Trace through junction to find exit port
                        for je in graph.edges:
                            if je.target == edge.source and je.line_id == edge.line_id:
                                js = graph.stations.get(je.source)
                                if js and js.is_port:
                                    feeder_sec = js.section_id
                                    break
                    if feeder_sec is not None:
                        line_feeder[edge.line_id] = feeder_sec
            if not line_feeder:
                continue

            # Group lines by feeder section
            lines_by_feeder: dict[str, list[str]] = {}
            for lid, fid in line_feeder.items():
                lines_by_feeder.setdefault(fid, []).append(lid)
            if len(lines_by_feeder) < 2:
                continue  # Single feeder, no reconvergence

            # Find primary feeder (most lines)
            primary_fid = max(lines_by_feeder, key=lambda f: len(lines_by_feeder[f]))
            primary_lines = set(lines_by_feeder[primary_fid])
            if len(primary_lines) < 2:
                continue  # Single continuing line, no ordering to preserve

            # Get primary feeder's local ordering (re-indexed if it had gaps)
            primary_order = section_local.get(primary_fid, line_priority)
            continuing = sorted(
                primary_lines, key=lambda lid: primary_order.get(lid, 0)
            )

            # All lines present in this section
            sec_present: set[str] = set()
            for sid_s, station in graph.stations.items():
                if station.section_id == sec_id:
                    sec_present |= set(graph.station_lines(sid_s))

            returning = sorted(
                sec_present - primary_lines,
                key=lambda lid: line_priority.get(lid, 0),
            )
            new_order = continuing + returning

            # Skip if ordering matches global (no change needed)
            global_ordered = sorted(
                sec_present, key=lambda lid: line_priority.get(lid, 0)
            )
            if new_order == global_ordered:
                continue

            # Apply reconvergence ordering
            new_local = {lid: i for i, lid in enumerate(new_order)}
            local_max = len(new_order) - 1
            reverse = sec_id in reversed_sections
            for sid_s, station in graph.stations.items():
                if station.section_id != sec_id:
                    continue
                for lid in graph.station_lines(sid_s):
                    p = new_local.get(lid, 0)
                    if reverse:
                        offsets[(sid_s, lid)] = (local_max - p) * offset_step
                    else:
                        offsets[(sid_s, lid)] = p * offset_step

    # Section-wide consistency for entry lines in compact mode.
    # All lines entering a section should maintain consistent relative
    # offsets at every multi-line station, including hidden pass-throughs.
    # This prevents crossings (offset flip between entry and convergence)
    # and overlaps (offset collapse when only one line exits a station).
    if compact:
        for sec_id, section in graph.sections.items():
            sec_entry_lines: list[str] = []
            for pid in section.entry_ports:
                sec_entry_lines.extend(graph.station_lines(pid))
            # Deduplicate preserving priority order
            seen: set[str] = set()
            unique_entry: list[str] = []
            for lid in sorted(
                set(sec_entry_lines), key=lambda x: line_priority.get(x, 0)
            ):
                if lid not in seen:
                    seen.add(lid)
                    unique_entry.append(lid)
            if len(unique_entry) < 2:
                continue
            sec_reverse = sec_id in reversed_sections
            sec_offs: dict[str, float] = {}
            for i, lid in enumerate(unique_entry):
                if sec_reverse:
                    sec_offs[lid] = (len(unique_entry) - 1 - i) * offset_step
                else:
                    sec_offs[lid] = i * offset_step
            for sid_s, station in graph.stations.items():
                if station.section_id != sec_id:
                    continue
                slines = graph.station_lines(sid_s)
                present = [lid for lid in slines if lid in sec_offs]
                if len(slines) >= 2 and present:
                    # Multi-line station: apply section-wide ordering
                    for lid in present:
                        offsets[(sid_s, lid)] = sec_offs[lid]
                elif station.is_hidden and len(slines) == 1 and slines[0] in sec_offs:
                    # Hidden pass-through: match section-wide offset
                    # so routing stays flat to the convergence station
                    offsets[(sid_s, slines[0])] = sec_offs[slines[0]]

    # Set exit port offsets on TB sections with LEFT/RIGHT exits to
    # match the exit L-shape's horiz_y_off.  For non-reversed sections,
    # reverse the internal offset (concentric arc swaps ordering).
    # For reversed sections, keep the internal offset as-is (the
    # internal offsets already account for the reversal).
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

    # Set exit port offsets on LR/RL sections with LEFT/RIGHT exits
    # based on the spatial Y ordering of the feeding stations.  Lines
    # arriving from above the exit port should appear at the top of
    # the bundle and lines from below at the bottom, preventing visual
    # crossings from diagonal merges.
    lr_rl_sections = {
        sid for sid, s in graph.sections.items() if s.direction in ("LR", "RL")
    }
    for port_id, port_obj in graph.ports.items():
        if port_obj.is_entry or port_obj.section_id not in lr_rl_sections:
            continue
        if port_obj.side not in (PortSide.LEFT, PortSide.RIGHT):
            continue
        # Compute average Y of internal stations feeding each line
        line_ys: dict[str, list[float]] = {}
        for edge in graph.edges:
            if edge.target == port_id:
                src_st = graph.stations.get(edge.source)
                if src_st and not src_st.is_port:
                    line_ys.setdefault(edge.line_id, []).append(src_st.y)
        if len(line_ys) < 2:
            continue
        line_avg_y = {lid: sum(ys) / len(ys) for lid, ys in line_ys.items()}
        # Only apply spatial ordering when lines arrive at genuinely
        # different Ys.  When all feeders share the same Y, the global
        # priority ordering is already correct and re-indexing would
        # shift the bundle base offset, misaligning with neighbours.
        unique_ys = set(line_avg_y.values())
        if len(unique_ys) < 2:
            continue
        # Sort lines by average Y (top to bottom), breaking ties with
        # line priority so equal-Y lines don't get arbitrarily reordered.
        sorted_lines = sorted(
            line_avg_y,
            key=lambda lid: (line_avg_y[lid], line_priority.get(lid, 0)),
        )
        spatial_offs = {lid: i * offset_step for i, lid in enumerate(sorted_lines)}
        for lid, off in spatial_offs.items():
            offsets[(port_id, lid)] = off

        # Propagate spatial ordering to internal hub stations that
        # fan out to the exit-bound stations.  Trace back: exit port
        # ← single-line stations ← hub station.
        feeder_ids: set[str] = set()
        for edge in graph.edges:
            if edge.target == port_id:
                src_st = graph.stations.get(edge.source)
                if src_st and not src_st.is_port:
                    feeder_ids.add(edge.source)
        # Only propagate when there are multiple feeders (actual fan-out).
        # A single feeder means the hub is just a pass-through, and
        # overwriting its offsets would misalign it with neighbours.
        if len(feeder_ids) >= 2:
            # Find common predecessors of the feeders (the hub)
            hub_candidates: set[str] = set()
            for edge in graph.edges:
                if edge.target in feeder_ids:
                    hub_candidates.add(edge.source)
            for hub_id in hub_candidates:
                hub_lines = graph.station_lines(hub_id)
                overlap = [lid for lid in hub_lines if lid in spatial_offs]
                if len(overlap) >= 2:
                    for lid in overlap:
                        offsets[(hub_id, lid)] = spatial_offs[lid]

    # Junctions have section_id=None so they get default line-priority
    # ordering above, which may not match the exit port feeding them.
    # Inherit offsets from the upstream exit port instead.
    for jid in graph.junctions:
        for edge in graph.edges:
            if edge.target == jid:
                src = graph.stations.get(edge.source)
                port_obj = graph.ports.get(edge.source)
                if src and src.is_port and port_obj and not port_obj.is_entry:
                    # Copy exit port's offsets to the junction
                    for lid in graph.station_lines(jid):
                        port_off = offsets.get((edge.source, lid))
                        if port_off is not None:
                            offsets[(jid, lid)] = port_off
                    break

    # Override TOP entry port offsets to match the inter-section routing
    # from upstream TB BOTTOM exits.  The inter-section routing reverses
    # exit port offsets using the local max at the exit port, but the
    # default offsets above use the global max_priority.  This mismatch
    # causes a visible horizontal discontinuity at the section boundary.
    tb_right_entry: set[str] = set()
    for port_obj in graph.ports.values():
        if (
            port_obj.is_entry
            and port_obj.side == PortSide.RIGHT
            and port_obj.section_id in tb_sections
        ):
            tb_right_entry.add(port_obj.section_id)

    for port_id, port_obj in graph.ports.items():
        if not port_obj.is_entry or port_obj.side != PortSide.TOP:
            continue
        for edge in graph.edges:
            if edge.target != port_id:
                continue
            src = graph.stations.get(edge.source)
            if not src or not src.is_port:
                continue
            src_port = graph.ports.get(edge.source)
            if not (
                src_port
                and not src_port.is_entry
                and src_port.side == PortSide.BOTTOM
                and src.section_id in tb_sections
            ):
                continue
            # Found a TB BOTTOM exit feeding this TOP entry.
            # Compute the same reversed offsets that route_edges uses
            # in the src_is_tb_bottom path.
            exit_port_id = edge.source
            all_exit_offs = [
                offsets.get((exit_port_id, lid), 0.0)
                for lid in graph.station_lines(exit_port_id)
            ]
            max_exit_off = max(all_exit_offs) if all_exit_offs else 0.0
            if src.section_id in tb_right_entry:
                for lid in graph.station_lines(port_id):
                    offsets[(port_id, lid)] = offsets.get((exit_port_id, lid), 0.0)
            else:
                for lid in graph.station_lines(port_id):
                    exit_off = offsets.get((exit_port_id, lid), 0.0)
                    offsets[(port_id, lid)] = max_exit_off - exit_off
            break

    # Propagate LR/RL exit port spatial ordering to downstream entry
    # ports so the inter-section bundle doesn't cross between sections.
    # Only propagate when a single exit port carries ALL lines at the
    # entry (direct continuation).  When multiple exit ports converge
    # at one entry, their offsets are inconsistent and the global
    # priority ordering should be preserved.
    for port_id, port_obj in graph.ports.items():
        if not port_obj.is_entry:
            continue
        if port_obj.side not in (PortSide.LEFT, PortSide.RIGHT):
            continue
        # Count distinct exit ports feeding this entry
        feeding_exit_ports: set[str] = set()
        for edge in graph.edges:
            if edge.target != port_id:
                continue
            src = graph.stations.get(edge.source)
            if not src or not src.is_port:
                continue
            src_port = graph.ports.get(edge.source)
            if src_port and not src_port.is_entry:
                feeding_exit_ports.add(edge.source)
        if len(feeding_exit_ports) != 1:
            continue  # Multiple sources converge; keep global offsets
        exit_port_id = next(iter(feeding_exit_ports))
        src_port = graph.ports.get(exit_port_id)
        if not (src_port and src_port.section_id in lr_rl_sections):
            continue
        # Only propagate when the exit port carries the same lines
        exit_lines = set(graph.station_lines(exit_port_id))
        entry_lines = set(graph.station_lines(port_id))
        if exit_lines != entry_lines:
            continue
        # Copy exit port offsets directly to entry port
        entry_offs: dict[str, float] = {}
        for lid in graph.station_lines(port_id):
            exit_off = offsets.get((exit_port_id, lid))
            if exit_off is not None:
                offsets[(port_id, lid)] = exit_off
                entry_offs[lid] = exit_off
        # Propagate to the first internal station(s) connected
        # to this entry port so offsets don't cross immediately.
        if len(entry_offs) >= 2:
            for e2 in graph.edges:
                if e2.source == port_id:
                    tgt_st = graph.stations.get(e2.target)
                    if tgt_st and not tgt_st.is_port:
                        tgt_lines = graph.station_lines(e2.target)
                        overlap = [lid for lid in tgt_lines if lid in entry_offs]
                        if len(overlap) >= 2:
                            for lid in overlap:
                                offsets[(e2.target, lid)] = entry_offs[lid]

    # Ensure multi-line entry ports have separated offsets and propagate
    # those offsets to directly-connected upstream exit ports so the
    # inter-section segment stays flat.  Skip ports that already have
    # distinct offsets (set by the exit-port propagation above).
    if compact:
        for sec_id, section in graph.sections.items():
            entry_lines: list[str] = []
            for pid in section.entry_ports:
                entry_lines.extend(graph.station_lines(pid))
            unique = sorted(set(entry_lines), key=lambda x: line_priority.get(x, 0))
            if len(unique) < 2:
                continue
            # Check if offsets are already separated (non-zero spread)
            existing = [
                offsets.get((pid, lid), 0.0)
                for pid in section.entry_ports
                for lid in unique
                if lid in graph.station_lines(pid)
            ]
            if len(set(existing)) >= 2:
                continue
            sec_reverse = sec_id in reversed_sections
            for i, lid in enumerate(unique):
                if sec_reverse:
                    off = (len(unique) - 1 - i) * offset_step
                else:
                    off = i * offset_step
                for pid in section.entry_ports:
                    if lid in graph.station_lines(pid):
                        offsets[(pid, lid)] = off
                        for edge in graph.edges:
                            if edge.target == pid and edge.line_id == lid:
                                src_port = graph.ports.get(edge.source)
                                if src_port and not src_port.is_entry:
                                    offsets[(edge.source, lid)] = off

    return offsets
