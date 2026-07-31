"""A junction fan keeps its bypass branch outside unrelated sections."""

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.guards import routes_through_unrelated_sections
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.parser.mermaid import parse_metro_mermaid


def test_fan_bypass_branch_clears_sections() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/topologies/fan_bypass_shared_band.mmd"
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
