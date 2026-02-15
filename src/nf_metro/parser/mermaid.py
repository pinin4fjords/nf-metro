"""Parser for Mermaid graph definitions with %%metro directives.

Uses a simple line-by-line approach rather than a full grammar parser,
since the Mermaid subset we need is straightforward.

Sections are defined as Mermaid subgraphs with %%metro entry/exit directives.
"""

from __future__ import annotations

import re

from nf_metro.parser.model import (
    Edge,
    MetroGraph,
    MetroLine,
    Port,
    PortSide,
    Section,
    Station,
)


def parse_metro_mermaid(text: str) -> MetroGraph:
    """Parse a Mermaid graph definition with %%metro directives."""
    graph = MetroGraph()
    lines = text.strip().split("\n")

    current_section_id: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Subgraph end
        if stripped == "end":
            current_section_id = None
            continue

        # Subgraph start
        subgraph_m = _SUBGRAPH_PATTERN.match(stripped)
        if subgraph_m:
            section_id = subgraph_m.group(1)
            display_name = subgraph_m.group(2) or section_id
            section = Section(id=section_id, name=display_name.strip())
            graph.add_section(section)
            current_section_id = section_id
            continue

        # Metro directives
        if stripped.startswith("%%metro"):
            _parse_directive(stripped, graph, current_section_id)
            continue

        # Skip regular comments and graph declaration
        if stripped.startswith("%%") or stripped.startswith("graph "):
            continue

        # Try edge first (contains arrow)
        if "-->" in stripped or "---" in stripped or "==>" in stripped:
            _parse_edge(stripped, graph, current_section_id)
            continue

        # Try node definition
        _parse_node(stripped, graph, current_section_id)

    # Post-parse: resolve sections if we have subgraph-based sections
    if graph.sections:
        _resolve_sections(graph)

    return graph


# Subgraph pattern: subgraph id [Display Name]
_SUBGRAPH_PATTERN = re.compile(
    r"^subgraph\s+(\w+)\s*(?:\[(.+?)\])?\s*$"
)


def _parse_directive(
    line: str,
    graph: MetroGraph,
    current_section_id: str | None = None,
) -> None:
    """Parse a %%metro directive line."""
    content = line[len("%%metro"):].strip()

    if content.startswith("title:"):
        graph.title = content[len("title:"):].strip()
    elif content.startswith("style:"):
        graph.style = content[len("style:"):].strip()
    elif content.startswith("line:"):
        parts = content[len("line:"):].strip().split("|")
        if len(parts) >= 3:
            graph.add_line(MetroLine(
                id=parts[0].strip(),
                display_name=parts[1].strip(),
                color=parts[2].strip(),
            ))
    elif content.startswith("entry:"):
        if current_section_id:
            _parse_port_hint(content, graph, current_section_id, is_entry=True)
    elif content.startswith("exit:"):
        if current_section_id:
            _parse_port_hint(content, graph, current_section_id, is_entry=False)
    elif content.startswith("direction:"):
        if current_section_id and current_section_id in graph.sections:
            direction = content[len("direction:"):].strip().upper()
            if direction in ("LR", "RL", "TB"):
                graph.sections[current_section_id].direction = direction
    elif content.startswith("grid:"):
        _parse_grid_directive(content, graph)
    elif content.startswith("legend:"):
        pos = content[len("legend:"):].strip().lower()
        if pos in ("bl", "br", "tl", "tr", "bottom", "right", "none"):
            graph.legend_position = pos


def _parse_port_hint(
    content: str,
    graph: MetroGraph,
    section_id: str,
    is_entry: bool,
) -> None:
    """Parse %%metro entry:/exit: and store as a hint on the Section.

    Does NOT create Port objects - those are created later in _resolve_sections
    based on actual inter-section edges.
    """
    prefix = "entry:" if is_entry else "exit:"
    rest = content[len(prefix):].strip()
    parts = rest.split("|")
    if len(parts) < 2:
        return

    side_str = parts[0].strip().lower()
    side_map = {
        "left": PortSide.LEFT,
        "right": PortSide.RIGHT,
        "top": PortSide.TOP,
        "bottom": PortSide.BOTTOM,
    }
    side = side_map.get(side_str)
    if side is None:
        return

    line_ids = [lid.strip() for lid in parts[1].strip().split(",") if lid.strip()]

    section = graph.sections.get(section_id)
    if section:
        if is_entry:
            section.entry_hints.append((side, line_ids))
        else:
            section.exit_hints.append((side, line_ids))


def _parse_grid_directive(content: str, graph: MetroGraph) -> None:
    """Parse %%metro grid: section_id | col,row[,rowspan[,colspan]] directive."""
    rest = content[len("grid:"):].strip()
    parts = rest.split("|")
    if len(parts) < 2:
        return

    section_id = parts[0].strip()
    coords = parts[1].strip().split(",")
    if len(coords) < 2:
        return

    try:
        col = int(coords[0].strip())
        row = int(coords[1].strip())
        rowspan = int(coords[2].strip()) if len(coords) >= 3 else 1
        colspan = int(coords[3].strip()) if len(coords) >= 4 else 1
    except ValueError:
        return
    graph.grid_overrides[section_id] = (col, row, rowspan, colspan)


# Regex patterns for node shapes
_NODE_PATTERNS = [
    # stadium: node_id([label])
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(\[(.+?)\]\)$"),
    # subroutine: node_id[[label]]
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[\[(.+?)\]\]$"),
    # circle: node_id((label))
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(\((.+?)\)\)$"),
    # square bracket: node_id[label]
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(.+?)\]$"),
    # round bracket: node_id(label)
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.+?)\)$"),
    # rhombus: node_id{label}
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\{(.+?)\}$"),
    # bare id
    re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)$"),
]

# Edge pattern: source -->|label| target  or  source --> target
_EDGE_PATTERN = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*"   # source
    r"(-->|---|==>)"                     # arrow
    r"(?:\|([^|]*)\|)?\s*"              # optional |label|
    r"([a-zA-Z_][a-zA-Z0-9_]*)$"       # target
)


def _parse_node(
    line: str,
    graph: MetroGraph,
    section_id: str | None = None,
) -> None:
    """Parse a node definition line."""
    for pattern in _NODE_PATTERNS:
        m = pattern.match(line)
        if m:
            node_id = m.group(1)
            label = m.group(2).strip() if m.lastindex >= 2 else node_id
            if node_id not in graph.stations:
                station = Station(id=node_id, label=label, section_id=section_id)
                graph.add_station(station)
                if section_id and section_id in graph.sections:
                    graph.sections[section_id].station_ids.append(node_id)
            else:
                # Update label if station was auto-created from an edge
                graph.stations[node_id].label = label
                # Also set section if not yet set
                if section_id and graph.stations[node_id].section_id is None:
                    graph.stations[node_id].section_id = section_id
                    if section_id in graph.sections:
                        graph.sections[section_id].station_ids.append(node_id)
            return


def _parse_edge(
    line: str,
    graph: MetroGraph,
    section_id: str | None = None,
) -> None:
    """Parse an edge definition line.

    Supports comma-separated line IDs: a -->|line1,line2,line3| b
    Creates a separate Edge for each line ID.
    """
    m = _EDGE_PATTERN.match(line)
    if not m:
        return

    source = m.group(1)
    label = m.group(3).strip() if m.group(3) else "default"
    target = m.group(4)

    # Ensure stations exist
    if source not in graph.stations:
        station = Station(id=source, label=source, section_id=section_id)
        graph.add_station(station)
        if section_id and section_id in graph.sections:
            graph.sections[section_id].station_ids.append(source)
    if target not in graph.stations:
        station = Station(id=target, label=target, section_id=section_id)
        graph.add_station(station)
        if section_id and section_id in graph.sections:
            graph.sections[section_id].station_ids.append(target)

    # Split comma-separated line IDs
    line_ids = [lid.strip() for lid in label.split(",")]
    for line_id in line_ids:
        graph.add_edge(Edge(source=source, target=target, line_id=line_id))


def _resolve_sections(graph: MetroGraph) -> None:
    """Post-parse: classify edges, create ports, rewrite inter-section edges.

    Key design: ONE exit port per source section. All lines leaving a section
    exit together, ensuring consistent ordering. Junctions handle fan-out
    to multiple target sections. ONE entry port per target section per side
    (side from hints or LEFT default).
    """
    # Build line->side mapping from explicit entry hints
    entry_side_for_line: dict[tuple[str, str], PortSide] = {}
    for sec_id, section in graph.sections.items():
        for side, line_ids in section.entry_hints:
            for lid in line_ids:
                entry_side_for_line[(sec_id, lid)] = side

    # Classify edges as internal or inter-section
    internal_edges: list[Edge] = []
    inter_section_edges: list[Edge] = []

    for edge in graph.edges:
        src_sec = graph.section_for_station(edge.source)
        tgt_sec = graph.section_for_station(edge.target)

        if src_sec and tgt_sec and src_sec != tgt_sec:
            inter_section_edges.append(edge)
        else:
            internal_edges.append(edge)
            sec_id = src_sec or tgt_sec
            if sec_id and sec_id in graph.sections:
                graph.sections[sec_id].internal_edges.append(edge)

    if not inter_section_edges:
        for i, section in enumerate(graph.sections.values()):
            if section.number == 0:
                section.number = i + 1
        return

    # Determine section-level exit side from hints: if all exit hints
    # point to ONE side, use that side; otherwise default to RIGHT.
    # This keeps one exit port per section (no splitting), just changes
    # which boundary it sits on.
    section_exit_side: dict[str, PortSide] = {}
    for sec_id, section in graph.sections.items():
        unique_sides = {side for side, _line_ids in section.exit_hints}
        if len(unique_sides) == 1:
            section_exit_side[sec_id] = unique_sides.pop()
        else:
            section_exit_side[sec_id] = PortSide.RIGHT

    # ONE exit port per source section (all lines leave together)
    exit_group_edges: dict[str, list[Edge]] = {}
    # ONE entry port per (target_section, entry_side)
    entry_group_edges: dict[tuple[str, PortSide], list[Edge]] = {}

    for edge in inter_section_edges:
        src_sec = graph.section_for_station(edge.source)
        tgt_sec = graph.section_for_station(edge.target)
        entry_side = entry_side_for_line.get((tgt_sec, edge.line_id), PortSide.LEFT)

        exit_group_edges.setdefault(src_sec, []).append(edge)

        entry_key = (tgt_sec, entry_side)
        entry_group_edges.setdefault(entry_key, []).append(edge)

    # Create exit ports (one per source section)
    port_counter = 0
    exit_port_map: dict[str, str] = {}

    for sec_id, edges in exit_group_edges.items():
        side = section_exit_side.get(sec_id, PortSide.RIGHT)
        all_line_ids = sorted({e.line_id for e in edges})
        port_id = f"{sec_id}__exit_{side.value}_{port_counter}"
        port = Port(
            id=port_id, section_id=sec_id, side=side,
            line_ids=all_line_ids, is_entry=False,
        )
        graph.add_port(port)
        exit_port_map[sec_id] = port_id
        port_counter += 1

    # Create entry ports (one per target section per side)
    entry_port_map: dict[tuple[str, PortSide], str] = {}

    for (sec_id, side), edges in entry_group_edges.items():
        all_line_ids = sorted({e.line_id for e in edges})
        port_id = f"{sec_id}__entry_{side.value}_{port_counter}"
        port = Port(
            id=port_id, section_id=sec_id, side=side,
            line_ids=all_line_ids, is_entry=True,
        )
        graph.add_port(port)
        entry_port_map[(sec_id, side)] = port_id
        port_counter += 1

    # Rewrite inter-section edges into 3-part chains
    new_edges: list[Edge] = list(internal_edges)

    # Group inter-section edges by exit port to detect fan-outs
    # Key: exit_port_id -> dict of entry_port_id -> list of (edge, entry_port_id)
    exit_fan: dict[str, dict[str, list[Edge]]] = {}

    for edge in inter_section_edges:
        src_sec = graph.section_for_station(edge.source)
        tgt_sec = graph.section_for_station(edge.target)
        entry_side = entry_side_for_line.get((tgt_sec, edge.line_id), PortSide.LEFT)

        exit_port_id = exit_port_map[src_sec]
        entry_port_id = entry_port_map[(tgt_sec, entry_side)]

        # source -> exit_port (always needed)
        new_edges.append(Edge(source=edge.source, target=exit_port_id, line_id=edge.line_id))
        # entry_port -> target (always needed)
        new_edges.append(Edge(source=entry_port_id, target=edge.target, line_id=edge.line_id))

        # Track the middle segment (exit_port -> entry_port) for junction insertion
        exit_fan.setdefault(exit_port_id, {}).setdefault(entry_port_id, []).append(edge)

    # Insert junctions where an exit port fans out to multiple entry ports
    for exit_port_id, entry_targets in exit_fan.items():
        if len(entry_targets) <= 1:
            # Single destination - direct edge, no junction needed
            for entry_port_id, edges in entry_targets.items():
                for edge in edges:
                    new_edges.append(Edge(source=exit_port_id, target=entry_port_id, line_id=edge.line_id))
        else:
            # Multiple destinations - create a junction station
            junction_id = f"__junction_{port_counter}"
            port_counter += 1
            junction = Station(id=junction_id, label="", is_port=True, section_id=None)
            graph.add_station(junction)
            graph.junctions.append(junction_id)

            # All lines: exit_port -> junction
            all_line_ids: set[str] = set()
            for edges in entry_targets.values():
                for edge in edges:
                    all_line_ids.add(edge.line_id)
            for lid in sorted(all_line_ids):
                new_edges.append(Edge(source=exit_port_id, target=junction_id, line_id=lid))

            # Per-destination: junction -> entry_port
            for entry_port_id, edges in entry_targets.items():
                for edge in edges:
                    new_edges.append(Edge(source=junction_id, target=entry_port_id, line_id=edge.line_id))

    graph.edges = new_edges

    for i, section in enumerate(graph.sections.values()):
        if section.number == 0:
            section.number = i + 1
