"""A flow-axis entry whose consumer is an interior station re-anchors to the
section's leading edge instead of folding back over the trunk (#1182).

When a section's entry sits on its flow-*trailing* edge (LEFT on an RL section,
RIGHT on an LR one -- the edge the section flows out of) but its consumer is an
interior station rather than the one beside that edge, the connecting leg has
to run inward past intervening stations and the line doubles straight back -- a
same-side hairpin that covers a stretch of the trunk in opposing directions.

``_reanchor_flow_axis_ports`` moves such an entry to the section's leading edge,
where the connecting leg arrives with the flow.  The producer then reaches the
relocated port by wrapping over the top of the section (the engine's existing
entry-wrap), so the line reads as a clean loop instead of a fold.
"""

from __future__ import annotations

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.guards import iter_opposing_line_overlaps
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import apply_route_offsets
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide

CULDESAC = "examples/topologies/same_side_culdesac.mmd"


def _layout(path: str, *, validate: bool):
    graph = parse_metro_mermaid(open(path).read())
    compute_layout(graph, validate=validate)
    return graph


def test_same_side_culdesac_validates() -> None:
    """The minimal cul-de-sac map lays out without raising.

    With the entry left on the trailing edge the connecting leg covers a
    stretch of the trunk in opposing directions and ``compute_layout`` raises
    ``PhaseInvariantError`` on the opposing-overlap guard; this end-to-end check
    pins that re-anchoring keeps it clean.
    """
    _layout(CULDESAC, validate=True)


def test_same_side_culdesac_entry_reanchored_to_leading() -> None:
    """The interior-consumer entry sits on the section's leading edge.

    ``mid`` is RL, so its leading edge is RIGHT; the explicit ``entry: left``
    hint is overridden because its consumer is interior.
    """
    graph = _layout(CULDESAC, validate=False)
    mid = graph.sections["mid"]
    assert mid.entry_ports
    assert all(graph.ports[pid].side is PortSide.RIGHT for pid in mid.entry_ports), [
        (pid, graph.ports[pid].side.value) for pid in mid.entry_ports
    ]


def test_same_side_culdesac_producer_wraps_over_section_top() -> None:
    """The producer reaches the relocated entry by wrapping above the section.

    The clean idiom rises in the inter-section gap and runs above ``mid``'s box
    rather than folding along the trunk, so the producer->entry route has a
    point above the section's top edge.
    """
    graph = _layout(CULDESAC, validate=False)
    mid = graph.sections["mid"]
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    entry_ports = set(mid.entry_ports)
    wrap = next(
        (r for r in routes if r.line_id == "flow" and r.edge.target in entry_ports),
        None,
    )
    assert wrap is not None, "no producer->entry route found"
    pts = apply_route_offsets(wrap, offsets)
    assert any(y < mid.bbox_y for _, y in pts), (
        f"producer->entry route never rises above mid's top {mid.bbox_y}: {pts}"
    )


def test_same_side_culdesac_no_foldback() -> None:
    """No opposing-direction overlap remains on the cul-de-sac line."""
    graph = _layout(CULDESAC, validate=False)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    overlaps = list(iter_opposing_line_overlaps(graph, offsets=offsets, routes=routes))
    assert not overlaps, overlaps
