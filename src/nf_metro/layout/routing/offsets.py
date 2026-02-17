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

    reversed_sections = detect_reversed_sections(graph)

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

    return offsets
