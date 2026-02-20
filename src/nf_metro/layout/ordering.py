"""Track-per-line vertical ordering.

Each metro line gets a dedicated horizontal track (Y position). Nodes on
the main path of a line snap to its base track. Short branches (nodes
whose predecessors are far from the line's base track) stay near their
predecessors instead of jumping to a distant track.
"""

from __future__ import annotations

__all__ = ["assign_tracks"]

from collections import defaultdict

import networkx as nx

from nf_metro.layout.constants import (
    DEFAULT_LINE_PRIORITY,
    DIAMOND_COMPRESSION,
    FANOUT_SPACING,
    LINE_GAP,
    SIDE_BRANCH_NUDGE,
)
from nf_metro.parser.model import MetroGraph


def assign_tracks(
    graph: MetroGraph,
    layers: dict[str, int],
    line_gap: float = LINE_GAP,
) -> dict[str, float]:
    """Assign each station a track using the track-per-line strategy.

    Args:
        graph: The metro graph.
        layers: Layer assignment from assign_layers().
        line_gap: Fixed gap (in track units) between line base tracks.

    Returns a dict mapping station_id -> track (float).
    """
    if not graph.lines:
        return {sid: float(i) for i, sid in enumerate(graph.stations)}

    G = nx.DiGraph()
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)
    for sid in graph.stations:
        if sid not in G:
            G.add_node(sid)

    line_order = list(graph.lines.keys())
    line_priority = {lid: i for i, lid in enumerate(line_order)}

    # Step 1: Determine primary line for each node
    node_primary: dict[str, str | None] = {}
    for sid in graph.stations:
        node_lines = graph.station_lines(sid)
        if node_lines:
            node_primary[sid] = min(
                node_lines, key=lambda ln: line_priority.get(ln, DEFAULT_LINE_PRIORITY)
            )
        else:
            node_primary[sid] = None

    # Step 2: Fixed-gap base tracks per line
    line_base: dict[str, float] = {}
    for i, lid in enumerate(line_order):
        line_base[lid] = i * line_gap

    # Step 3: Group nodes by (layer, primary_line)
    layer_line_groups: dict[tuple[int, str | None], list[str]] = defaultdict(list)
    for sid in graph.stations:
        layer_line_groups[(layers.get(sid, 0), node_primary[sid])].append(sid)

    tracks: dict[str, float] = {}
    max_layer = max(layers.values()) if layers else 0
    orphan_track = len(line_order) * line_gap

    for layer_idx in range(max_layer + 1):
        for lid in line_order:
            nodes = layer_line_groups.get((layer_idx, lid), [])
            if not nodes:
                continue

            base = line_base[lid]

            if len(nodes) == 1:
                tracks[nodes[0]] = _place_single_node(
                    nodes[0],
                    base,
                    line_gap,
                    G,
                    tracks,
                    graph,
                    layers,
                )
            else:
                _place_fan_out(nodes, base, line_gap, G, tracks)

        # Orphans (no line)
        orphans = layer_line_groups.get((layer_idx, None), [])
        for node in orphans:
            tracks[node] = orphan_track
            orphan_track += 1

        # Equalize cross-line fork groups at this layer so downstream
        # placement sees corrected positions.
        _equalize_fork_groups(
            layer_idx, layers, tracks, G, graph, node_primary, line_gap
        )

    return tracks


def _is_diamond_node(
    node: str,
    layer: int,
    G: nx.DiGraph,
    layers: dict[str, int],
    graph: MetroGraph | None = None,
) -> bool:
    """Check if node is part of a diamond (fork-join) pattern.

    A diamond node shares the same predecessors with another node at
    the same layer, both converge to at least one common successor,
    AND both nodes carry the same set of metro lines (i.e. they are
    alternative paths for the same lines, like FastP/TrimGalore).
    Nodes on different lines that happen to share predecessors/successors
    (like salmon_pseudo/kallisto) are NOT diamonds.
    """
    preds = set(G.predecessors(node))
    succs = set(G.successors(node))
    if not preds or not succs:
        return False

    node_lines = set(graph.station_lines(node)) if graph else set()

    same_layer = [n for n, ly in layers.items() if ly == layer and n != node]
    for other in same_layer:
        if set(G.predecessors(other)) == preds and succs & set(G.successors(other)):
            if graph:
                other_lines = set(graph.station_lines(other))
                if node_lines == other_lines:
                    return True
            else:
                return True
    return False


def _predecessor_avg(
    node: str, G: nx.DiGraph, tracks: dict[str, float]
) -> float | None:
    """Average track position of a node's already-placed predecessors."""
    preds = [p for p in G.predecessors(node) if p in tracks]
    if not preds:
        return None
    return sum(tracks[p] for p in preds) / len(preds)


def _place_single_node(
    node: str,
    base: float,
    line_gap: float,
    G: nx.DiGraph,
    tracks: dict[str, float],
    graph: MetroGraph | None = None,
    layers: dict[str, int] | None = None,
) -> float:
    """Place a single node, choosing between line base track and predecessor proximity.

    At divergence points (predecessor has more lines than this node),
    snap to the line's base track so diverging branches fan out properly.
    Exception: diamond (fork-join) patterns stay compact near the trunk.

    Otherwise, if predecessors are close, snap to base. If far (a
    side-branch deep in the graph), stay near predecessors.
    """
    pred_avg = _predecessor_avg(node, G, tracks)
    if pred_avg is None:
        return base

    # Detect divergence: predecessor has more lines than this node
    if graph is not None:
        preds = list(G.predecessors(node))
        node_lines = set(graph.station_lines(node))
        pred_lines: set[str] = set()
        for p in preds:
            pred_lines.update(graph.station_lines(p))
        if len(pred_lines) > len(node_lines):
            # Check if this is a diamond (temporary fork-join)
            node_layer = layers.get(node, 0) if layers else 0
            if layers and _is_diamond_node(node, node_layer, G, layers, graph):
                # Diamond: compress toward trunk for compact visual
                return pred_avg + (base - pred_avg) * DIAMOND_COMPRESSION
            else:
                # Permanent divergence: snap to base track
                return base

        # Detect convergence: node has more lines than its largest
        # predecessor (lines merging from different tracks). Snap to
        # base track so the main bundle stays compact and downstream
        # stations don't zigzag between the merged and base positions.
        if len(preds) > 1:
            max_pred_lines = max(len(set(graph.station_lines(p))) for p in preds)
            if len(node_lines) > max_pred_lines:
                return base

    distance = abs(base - pred_avg)
    if distance <= line_gap:
        # Close enough - snap to base track
        return base
    else:
        # Side-branch: stay near predecessors, nudge toward base
        direction = 1.0 if base > pred_avg else -1.0
        return pred_avg + direction * SIDE_BRANCH_NUDGE


def _place_fan_out(
    nodes: list[str],
    base: float,
    line_gap: float,
    G: nx.DiGraph,
    tracks: dict[str, float],
) -> None:
    """Place multiple nodes in the same layer+line, centered around an anchor.

    The anchor is the line's base track if predecessors are nearby,
    or the predecessor average if they're far away (fan-out from a branch).
    """
    # Compute barycenters for ordering
    bary: dict[str, float] = {}
    pred_avgs: list[float] = []
    for node in nodes:
        avg = _predecessor_avg(node, G, tracks)
        if avg is not None:
            bary[node] = avg
            pred_avgs.append(avg)
        else:
            bary[node] = base

    nodes.sort(key=lambda n: bary.get(n, base))

    # Decide anchor: base track or predecessor center
    if pred_avgs:
        overall_pred_avg = sum(pred_avgs) / len(pred_avgs)
        if abs(base - overall_pred_avg) <= line_gap:
            anchor = base
        else:
            anchor = overall_pred_avg
    else:
        anchor = base

    # Center the fan-out around the anchor.
    # Use sub-linear scaling so larger fans don't grow proportionally:
    # per-item spacing = FANOUT_SPACING * (n-1)^(p-1) with p=0.8,
    # giving total spread = FANOUT_SPACING * (n-1)^0.8.
    n = len(nodes)
    if n > 2:
        fan_spacing = FANOUT_SPACING * (n - 1) ** (0.8 - 1)
    else:
        fan_spacing = FANOUT_SPACING
    for i, node in enumerate(nodes):
        offset = (i - (n - 1) / 2) * fan_spacing
        tracks[node] = anchor + offset


def _equalize_fork_groups(
    layer: int,
    layers: dict[str, int],
    tracks: dict[str, float],
    G: nx.DiGraph,
    graph: MetroGraph,
    node_primary: dict[str, str | None],
    line_gap: float,
) -> None:
    """Redistribute cross-line fork siblings to equidistant spacing.

    When multiple stations at the same layer diverge from a common
    predecessor (or are root nodes entering from the same port),
    per-line base track assignment can create uneven spacing -- especially
    when one station carries more lines than its siblings, pushing the
    next sibling further away.

    This function detects such groups and compacts them to consecutive
    positions (one *line_gap* apart), preserving their track ordering.
    Groups where all members share the same primary line (diamonds /
    fan-outs already handled by ``_place_fan_out``) are skipped.
    """
    layer_nodes = [sid for sid, lyr in layers.items() if lyr == layer and sid in tracks]
    if len(layer_nodes) < 2:
        return

    # Group stations by their predecessor set
    pred_groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for sid in layer_nodes:
        preds = frozenset(G.predecessors(sid))
        pred_groups[preds].append(sid)

    for _pred_set, group in pred_groups.items():
        if len(group) < 2:
            continue

        # Skip groups where all members share the same primary line
        # (these are diamond / fan-out groups already well-placed).
        primaries = {node_primary.get(sid) for sid in group}
        primaries.discard(None)
        if len(primaries) < 2:
            continue

        # Sort by current track position to preserve ordering
        group.sort(key=lambda sid: tracks[sid])

        # Compute current spacings between consecutive members
        spacings = [
            tracks[group[i + 1]] - tracks[group[i]] for i in range(len(group) - 1)
        ]

        # Check whether equalization is needed:
        #   2 stations  - gap exceeds line_gap (multi-line station padding)
        #   3+ stations - spacing is uneven
        if len(group) == 2:
            if spacings[0] <= line_gap + 0.01:
                continue
        else:
            if max(spacings) - min(spacings) < 0.01:
                continue

        # Compact to consecutive positions starting from the lowest track
        base_track = tracks[group[0]]
        for i, sid in enumerate(group):
            tracks[sid] = base_track + i * line_gap


def _reorder_by_span(graph: MetroGraph, line_order: list[str]) -> list[str]:
    """Reorder lines by section span (descending).

    Lines that span more sections get earlier (inner) tracks.
    Ties are broken by preserving the original definition order.
    """
    if not graph.sections:
        return line_order

    # For each line, count how many distinct sections it touches
    line_sections: dict[str, set[str]] = {lid: set() for lid in line_order}
    for edge in graph.edges:
        lid = edge.line_id
        if lid not in line_sections:
            continue
        src = graph.stations.get(edge.source)
        tgt = graph.stations.get(edge.target)
        if src and src.section_id:
            line_sections[lid].add(src.section_id)
        if tgt and tgt.section_id:
            line_sections[lid].add(tgt.section_id)

    # Stable sort: descending by section count, preserving original order for ties
    return sorted(
        line_order,
        key=lambda lid: (-len(line_sections.get(lid, set())), line_order.index(lid)),
    )
