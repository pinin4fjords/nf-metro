"""The generic bypass keeps a far LEFT-entry feed outside every section."""

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.guards import routes_through_unrelated_sections
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.parser.mermaid import parse_metro_mermaid


def test_bypass_into_far_left_entry_clears_sections() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/topologies/bypass_left_entry_from_right.mmd"
    )
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    offenders = routes_through_unrelated_sections(graph, routes=routes, offsets=offsets)
    assert not offenders, "\n".join(
        f"line {route.line_id!r} {route.edge.source!r}->{route.edge.target!r} "
        f"passes through section {section_id!r}"
        for route, section_id in offenders
    )
