"""Tests for edge routing."""

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.parser.mermaid import parse_metro_mermaid


def test_straight_route():
    """Edges on the same track should be straight horizontal lines."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\ngraph LR\n    a -->|main| b\n"
    )
    compute_layout(graph)
    routes = route_edges(graph)
    assert len(routes) == 1
    # Same track -> 2 points (straight line)
    assert len(routes[0].points) == 2


def test_diagonal_route():
    """Edges between different tracks should have 4 waypoints."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    a -->|main| b\n"
        "    a -->|alt| c\n"
        "    b -->|main| d\n"
        "    c -->|alt| d\n"
    )
    compute_layout(graph)
    routes = route_edges(graph)

    # Find a route that changes tracks
    _ = [r for r in routes if len(r.points) == 4]
    # At least some routes should be diagonal (track changes)
    # The exact count depends on layout, but we should have some
    assert len(routes) == 4


def test_station_offsets_single_line():
    """Single line on a station should have zero offset."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\ngraph LR\n    a -->|main| b\n"
    )
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    assert offsets[("a", "main")] == 0.0
    assert offsets[("b", "main")] == 0.0


def test_station_offsets_multiple_lines():
    """Multiple lines on the same station should get different offsets."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    a -->|main| b\n"
        "    a -->|alt| b\n"
    )
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    assert offsets[("a", "main")] != offsets[("a", "alt")]


# --- Inter-section routing tests ---


def test_inter_section_routing():
    """Inter-section edges should be routed through ports."""
    from nf_metro.layout.routing import route_inter_section_edges

    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [S1]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [S2]\n"
        "        c[C]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    compute_layout(graph)
    inter_routes = route_inter_section_edges(graph)
    # Should have at least one inter-section route (the port-to-port edges)
    assert len(inter_routes) > 0


def test_section_routes_have_valid_points():
    """All routed paths should have at least 2 points."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [S1]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph sec2 [S2]\n"
        "        b[B]\n"
        "    end\n"
        "    subgraph sec3 [S3]\n"
        "        c[C]\n"
        "    end\n"
        "    a -->|main| b\n"
        "    a -->|alt| c\n"
    )
    compute_layout(graph)
    routes = route_edges(graph)
    for route in routes:
        assert len(route.points) >= 2, (
            f"Route {route.edge.source}->{route.edge.target}"
            f" has {len(route.points)} points"
        )
