"""A TB/BT BOTTOM exit must leave its port vertically, never along the edge.

When a vertical-flow (TB/BT) section has a BOTTOM exit port that feeds a side
(LEFT/RIGHT) entry sharing the exit's Y, the exit port sits on the section's own
bottom edge. A straight horizontal run at that Y travels along the edge and out
through the corner, away from any declared port (#1052). The route must instead
leave the port downward into the inter-row corridor, clear of the box, before
turning toward the neighbour.

Encoded two ways: the strict ``compute_layout(validate=True)`` guard
(``_guard_routes_enter_sections_at_ports``) must accept the targeted fixture,
and across every topology fixture each BOTTOM-exit route's first move must be a
downward vertical step rather than a horizontal run along the edge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide

TOPOLOGIES_DIR = Path(__file__).parent.parent / "examples" / "topologies"
TOPOLOGY_FILES = sorted(TOPOLOGIES_DIR.glob("*.mmd"))
TOPOLOGY_IDS = [f.stem for f in TOPOLOGY_FILES]


def _bottom_exit_ports(graph) -> set[str]:
    return {
        sid
        for sid, st in graph.stations.items()
        if st.is_port
        and (port := graph.ports.get(sid)) is not None
        and not port.is_entry
        and port.side == PortSide.BOTTOM
        and st.section_id in graph.sections
    }


def test_tb_perp_exit_side_neighbour_enters_at_ports() -> None:
    """The strict port-entry guard accepts the down-and-over corridor route."""
    text = (TOPOLOGIES_DIR / "tb_perp_exit_side_neighbour.mmd").read_text()
    graph = parse_metro_mermaid(text)
    # Raises PhaseInvariantError on a route that grazes the section boundary.
    compute_layout(graph, validate=True)


def test_tb_perp_exit_side_entry_aligns_with_consumer() -> None:
    """The side entry sits at its consumer's Y, so the turn-in is horizontal.

    The entry is fed only by the TB BOTTOM exit, whose Y is the section's bottom
    edge below its stations.  Anchoring the entry there would force a diagonal
    into the first station; the port instead lands at the consumer's Y so the
    inter-section route rises in the column gap and turns in level.
    """
    graph = parse_metro_mermaid(
        (TOPOLOGIES_DIR / "tb_perp_exit_side_neighbour.mmd").read_text()
    )
    compute_layout(graph)
    entry = graph.stations["out_sec__entry_left_1"]
    consumer = graph.stations["o1"]
    assert abs(entry.y - consumer.y) < 1.0, (
        f"LEFT entry port y={entry.y:.1f} is off its consumer o1 y={consumer.y:.1f}; "
        "the turn-in would be a diagonal rather than level"
    )


@pytest.mark.parametrize("path", TOPOLOGY_FILES, ids=TOPOLOGY_IDS)
def test_bottom_exit_routes_leave_port_downward(path: Path) -> None:
    """Every BOTTOM-exit route descends out of its port before turning.

    A graze leaves the port horizontally along the bottom edge; a corridor or
    drop route's first move is a downward vertical step.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)

    bottom_exit_ids = _bottom_exit_ports(graph)
    for route in routes:
        if route.edge.source not in bottom_exit_ids or len(route.points) < 2:
            continue
        (sx, sy), (nx, ny) = route.points[0], route.points[1]
        assert abs(nx - sx) < 1.0, (
            f"route from {route.edge.source} leaves the BOTTOM port "
            f"horizontally (dx={nx - sx:.1f}); it grazes the section edge"
        )
        assert ny > sy + 1.0, (
            f"route from {route.edge.source} does not descend below the exit "
            f"(sy={sy:.1f}, next_y={ny:.1f})"
        )
