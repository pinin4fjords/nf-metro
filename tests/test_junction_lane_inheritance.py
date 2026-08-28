"""A divergence junction rides the lanes its feeding exit port settled on.

When a bundle is shifted wholesale at an exit port so the port's own feeder run
comes out level, the divergence junction a few pixels downstream has to take the
same lanes.  Where it keeps an earlier set instead, the short run between the
two is spent climbing off the port's lane, and every branch beyond the junction
spends it again coming back: two shallow slants across runs far too short to
form a turn on, either side of a junction that is invisible in the render.

``junction_entry_lane_step`` places such a junction ten pixels past the exit
port feeding it, with one branch continuing along the row and one leaving it, so
both halves of that mismatch are drawn.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.route_topology import divergence_junction_exit_ports
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import apply_route_offsets
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = "examples/topologies/junction_entry_lane_step.mmd"

_TOLERANCE = 0.5


def _routed() -> tuple[object, dict[tuple[str, str], float], list]:
    graph = parse_metro_mermaid((ROOT / FIXTURE).read_text(), max_station_columns=15)
    compute_layout(graph)
    offsets = dict(compute_station_offsets(graph))
    return graph, offsets, route_edges(graph, station_offsets=offsets)


def _drawn(route, offsets) -> list[tuple[float, float]]:
    return apply_route_offsets(route, offsets)


def test_junction_holds_the_lanes_of_the_exit_port_feeding_it() -> None:
    """Every line the junction shares with its feeder sits on the feeder's lane."""
    graph, offsets, _routes = _routed()
    feeders = divergence_junction_exit_ports(graph)
    assert feeders, f"{FIXTURE} no longer carries a junction fed by an exit port"
    mismatched = {
        (junction_id, line_id): (offsets.get((port_id, line_id)), junction_lane)
        for junction_id, port_id in feeders.items()
        for line_id in graph.station_lines(junction_id)
        if (junction_lane := offsets.get((junction_id, line_id))) is not None
        and offsets.get((port_id, line_id)) is not None
        and abs(offsets[(port_id, line_id)] - junction_lane) > _TOLERANCE
    }
    assert not mismatched


def test_the_run_from_the_exit_port_into_the_junction_is_level() -> None:
    """The stub between a port and the junction it feeds draws flat."""
    graph, offsets, routes = _routed()
    feeders = divergence_junction_exit_ports(graph)
    slanted = [
        f"{route.edge.source}->{route.edge.target} ({route.line_id}): {points}"
        for route in routes
        if feeders.get(route.edge.target) == route.edge.source
        for points in [_drawn(route, offsets)]
        if max(y for _x, y in points) - min(y for _x, y in points) > _TOLERANCE
    ]
    assert not slanted, "\n".join(slanted)


def test_the_branch_continuing_along_the_junction_row_stays_level() -> None:
    """A branch whose target shares the junction's row never leaves it."""
    graph, offsets, routes = _routed()
    junctions = set(divergence_junction_exit_ports(graph))
    continuing = [
        (route, _drawn(route, offsets))
        for route in routes
        if route.edge.source in junctions
        and abs(
            graph.stations[route.edge.source].y - graph.stations[route.edge.target].y
        )
        <= _TOLERANCE
    ]
    assert continuing, f"{FIXTURE} no longer carries a branch along the junction row"
    slanted = [
        f"{route.edge.source}->{route.edge.target} ({route.line_id}): {points}"
        for route, points in continuing
        if max(y for _x, y in points) - min(y for _x, y in points) > _TOLERANCE
    ]
    assert not slanted, "\n".join(slanted)
