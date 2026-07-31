"""Layer assignment for metro map layout (X-coordinate positioning).

Uses longest-path layering on a topological sort to ensure left-to-right
monotonicity: every edge goes from a lower layer to a higher layer.
"""

from __future__ import annotations

__all__ = ["assign_layers", "build_station_digraph"]

import networkx as nx

from nf_metro.graph_views import longest_path_layers
from nf_metro.parser.model import MetroGraph


def build_station_digraph(graph: MetroGraph) -> nx.DiGraph[str]:
    """Build a station-id DiGraph from a MetroGraph's edges.

    Every edge becomes a directed edge; stations carrying no edges are added
    as isolated nodes so the graph covers all stations.
    """
    G: nx.DiGraph[str] = nx.DiGraph()
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)
    for sid in graph.stations:
        if sid not in G:
            G.add_node(sid)
    return G


def assign_layers(graph: MetroGraph) -> dict[str, int]:
    """Assign each station to a layer (integer X position).

    Uses longest-path layering: each node's layer is 1 + the maximum
    layer of its predecessors. This spreads nodes out to fill the
    available width and keeps edges pointing rightward.

    Returns a dict mapping station_id -> layer number (0-based).
    """
    G = build_station_digraph(graph)

    return longest_path_layers(G, graph.stations)
