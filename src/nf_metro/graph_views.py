"""Small NetworkX views of nf-metro's internal graph representations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from nf_metro.parser.model import MetroGraph


def directed_graph(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> nx.DiGraph[str]:
    """Build a directed graph while preserving the supplied node and edge order."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def longest_path_layers(
    graph: nx.DiGraph[str], node_order: Iterable[str]
) -> dict[str, int]:
    """Return longest-path layers in the supplied semantic node order."""
    ordered_nodes = tuple(node_order)
    rank = {node: index for index, node in enumerate(ordered_nodes)}
    topological_order = nx.lexicographical_topological_sort(graph, key=rank.__getitem__)
    layers: dict[str, int] = {}
    for node in topological_order:
        layers[node] = (
            max(
                (layers[predecessor] for predecessor in graph.predecessors(node)),
                default=-1,
            )
            + 1
        )
    return layers


def line_graphs(graph: MetroGraph) -> dict[str, nx.DiGraph[str]]:
    """Return one directed station graph per metro line."""
    views: dict[str, nx.DiGraph[str]] = {
        line_id: nx.DiGraph() for line_id in graph.lines
    }
    for edge in graph.edges:
        views.setdefault(edge.line_id, nx.DiGraph()).add_edge(edge.source, edge.target)
    return views
