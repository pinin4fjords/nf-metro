"""Small NetworkX views of nf-metro's internal graph representations."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from nf_metro.parser.model import MetroGraph


def directed_graph(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> nx.DiGraph[str]:
    """Build a directed graph while preserving the supplied node and edge order."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def line_graphs(graph: MetroGraph) -> dict[str, nx.DiGraph[str]]:
    """Return one directed station graph per metro line."""
    views: dict[str, nx.DiGraph[str]] = {
        line_id: nx.DiGraph() for line_id in graph.lines
    }
    for edge in graph.edges:
        views.setdefault(edge.line_id, nx.DiGraph()).add_edge(edge.source, edge.target)
    return views
