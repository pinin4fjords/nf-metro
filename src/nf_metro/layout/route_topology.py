"""Semantic route identity for layout and routing consumers."""

from __future__ import annotations

from collections import defaultdict

from nf_metro.parser.model import MetroGraph
from nf_metro.parser.route_topology import (
    RouteTopologyQuery,
    build_route_topology_query,
)


def divergence_junction_sources(
    graph: MetroGraph,
    topology: RouteTopologyQuery | None = None,
) -> dict[str, str]:
    """Map resolved divergence junctions to their upstream exit ports."""
    query = topology if topology is not None else build_route_topology_query(graph)
    if query is not None:
        return {view.junction_id: view.exit_port_id for view in query.divergences}

    result: dict[str, str] = {}
    for junction_id in graph.junctions:
        sources = {edge.source for edge in graph.edges_to(junction_id)}
        if len(sources) == 1 and graph.edges_from(junction_id):
            result[junction_id] = next(iter(sources))
    return result


def convergence_junction_ids(
    graph: MetroGraph,
    topology: RouteTopologyQuery | None = None,
) -> tuple[str, ...]:
    """Return resolved convergence junction ids in semantic order."""
    query = topology if topology is not None else build_route_topology_query(graph)
    if query is not None:
        return tuple(view.junction_id for view in query.convergences)

    predecessors: dict[str, set[str]] = defaultdict(set)
    successors: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        predecessors[edge.target].add(edge.source)
        successors[edge.source].add(edge.target)
    result: list[str] = []
    for junction_id in graph.junctions:
        if len(predecessors[junction_id]) <= 1 or len(successors[junction_id]) != 1:
            continue
        successor = next(iter(successors[junction_id]))
        port = graph.ports.get(successor)
        if port is not None and port.is_entry:
            result.append(junction_id)
    return tuple(result)


def convergence_entry_port_id(
    graph: MetroGraph,
    junction_id: str,
    topology: RouteTopologyQuery | None = None,
) -> str | None:
    """Return the resolved entry port represented by a convergence junction."""
    query = topology if topology is not None else build_route_topology_query(graph)
    if query is not None:
        convergence = query.convergence_for_junction(junction_id)
        return convergence.entry_port_id if convergence is not None else None

    entry_ports = [
        edge.target
        for edge in graph.edges_from(junction_id)
        if (port := graph.ports.get(edge.target)) is not None and port.is_entry
    ]
    return entry_ports[0] if len(entry_ports) == 1 else None


def merge_fanout_junction_ids(
    graph: MetroGraph,
    topology: RouteTopologyQuery | None = None,
    convergence_ids: set[str] | None = None,
) -> tuple[str, ...]:
    """Return divergence junctions feeding multiple same-line convergences."""
    query = topology if topology is not None else build_route_topology_query(graph)
    if query is not None:
        return query.merge_fanout_junction_ids()

    merges = (
        convergence_ids
        if convergence_ids is not None
        else set(convergence_junction_ids(graph))
    )
    result: list[str] = []
    for source in graph.stations:
        targets_by_line: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges_from(source):
            if edge.target in merges:
                targets_by_line[edge.line_id].add(edge.target)
        if any(len(targets) >= 2 for targets in targets_by_line.values()):
            result.append(source)
    return tuple(result)
