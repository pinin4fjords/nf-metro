"""Crossing-free opening order for semantic fan branches."""

from __future__ import annotations

from nf_metro.layout.geometry import packed_section_visual_rank
from nf_metro.layout.route_topology import convergence_entry_port_id
from nf_metro.layout.routing.common import resolve_section, resolve_section_colrow
from nf_metro.parser.model import MetroGraph, Station
from nf_metro.parser.route_topology import RouteTopologyQuery


def _fan_branch_entry_port(
    graph: MetroGraph,
    target_id: str,
    topology: RouteTopologyQuery | None = None,
) -> str | None:
    """Return the entry port reached by one resolved divergence branch."""
    port = graph.ports.get(target_id)
    if port is not None:
        return target_id if port.is_entry else None
    return convergence_entry_port_id(graph, target_id, topology)


def _section_order_coordinate(graph: MetroGraph, station: Station) -> float | None:
    """Return settled X or a stable pre-placement packed-cell coordinate."""
    section = resolve_section(graph, station, prefer_upstream=False)
    if section is None:
        return None
    col, row = resolve_section_colrow(graph, station)
    if col is None or row is None:
        return None
    if section.bbox_w > 0:
        return station.x
    rank = packed_section_visual_rank(graph, section, col, row)
    return col * (len(graph.sections) + 1) + rank


def fanout_divergence_peel_order(
    graph: MetroGraph,
    junction_id: str,
    line_priority: dict[str, int],
    topology: RouteTopologyQuery | None = None,
) -> list[str] | None:
    """Return the crossing-free opening order for a clean divergence.

    The result runs outermost to innermost at the shared turn. Unsupported or
    ambiguous groups return ``None`` and keep declaration order.
    """
    junction = graph.stations.get(junction_id)
    if junction is None:
        return None
    source_col, source_row = resolve_section_colrow(graph, junction)
    if source_col is None or source_row is None:
        return None
    source_x = _section_order_coordinate(graph, junction)
    if source_x is None:
        return None

    reach: dict[str, int] = {}
    row_delta: dict[str, int] = {}
    target_x: dict[str, float] = {}
    claimed: dict[str, str] = {}
    converging = False
    for edge in graph.edges_from(junction_id):
        entry_id = _fan_branch_entry_port(graph, edge.target, topology)
        if entry_id is None:
            return None
        converging |= entry_id != edge.target
        entry = graph.stations[entry_id]
        target_col, target_row = resolve_section_colrow(graph, entry)
        if target_col is None or target_row is None:
            return None
        if entry_id in claimed and claimed[entry_id] != edge.line_id:
            return None
        if edge.line_id in reach:
            return None
        claimed[entry_id] = edge.line_id
        reach[edge.line_id] = target_col - source_col
        row_delta[edge.line_id] = target_row - source_row
        coordinate = _section_order_coordinate(graph, entry)
        if coordinate is None:
            return None
        target_x[edge.line_id] = coordinate

    if len(reach) < 2:
        return None

    if len(set(reach.values())) > 1 and 0 in row_delta.values():
        if len(set(row_delta.values())) < 2:
            return None
        if len({value > 0 for value in row_delta.values() if value != 0}) != 1:
            return None
        return sorted(
            reach,
            key=lambda line_id: (
                row_delta[line_id],
                (
                    abs(reach[line_id])
                    if row_delta[line_id] == 0
                    else -abs(reach[line_id])
                ),
                line_priority.get(line_id, 0),
            ),
        )

    if converging:
        return None

    if len(set(reach.values())) == 1:
        descenders = [value for value in row_delta.values() if value != 0]
        if len({value > 0 for value in descenders}) != 1:
            return None
        if len(set(row_delta.values())) < 2:
            if len(set(target_x.values())) < 2:
                return None
            drop_down = descenders[0] > 0
            return sorted(
                reach,
                key=lambda line_id: (
                    (
                        abs(target_x[line_id] - source_x)
                        if drop_down
                        else -abs(target_x[line_id] - source_x)
                    ),
                    line_priority.get(line_id, 0),
                ),
            )
        return sorted(
            reach,
            key=lambda line_id: (
                row_delta[line_id],
                line_priority.get(line_id, 0),
            ),
        )

    if len({value > 0 for value in row_delta.values()}) != 1:
        return None
    drop_down = next(iter(row_delta.values())) > 0
    return sorted(
        reach,
        key=lambda line_id: (
            -abs(reach[line_id]) if drop_down else abs(reach[line_id]),
            line_priority.get(line_id, 0),
        ),
    )
