"""Tests for the layout engine."""

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.layers import assign_layers
from nf_metro.layout.ordering import assign_tracks
from nf_metro.parser.mermaid import parse_metro_mermaid


def _make_simple_graph():
    return parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    a[A]\n"
        "    b[B]\n"
        "    c[C]\n"
        "    a -->|main| b\n"
        "    b -->|main| c\n"
    )


def _make_branching_graph():
    return parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    a[A]\n"
        "    b[B]\n"
        "    c[C]\n"
        "    d[D]\n"
        "    a -->|main| b\n"
        "    b -->|main| d\n"
        "    a -->|alt| c\n"
        "    c -->|alt| d\n"
    )


def test_layer_assignment_linear():
    graph = _make_simple_graph()
    layers = assign_layers(graph)
    assert layers["a"] == 0
    assert layers["b"] == 1
    assert layers["c"] == 2


def test_layer_assignment_branching():
    graph = _make_branching_graph()
    layers = assign_layers(graph)
    assert layers["a"] == 0
    # b and c both have a as predecessor, so both at layer 1
    assert layers["b"] == 1
    assert layers["c"] == 1
    # d has b and c as predecessors (both at layer 1), so at layer 2
    assert layers["d"] == 2


def test_track_assignment():
    graph = _make_branching_graph()
    layers = assign_layers(graph)
    tracks = assign_tracks(graph, layers)
    # a is alone in layer 0
    assert tracks["a"] == 0
    # b and c are in layer 1 - should be on different tracks
    assert tracks["b"] != tracks["c"]


def test_compute_layout_sets_coordinates():
    """Layout assigns increasing x for a linear chain within a section."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        a -->|main| b\n"
        "        b -->|main| c\n"
        "    end\n"
    )
    compute_layout(graph, x_spacing=100, y_spacing=50)
    # Stations should be in order by x
    assert graph.stations["a"].x < graph.stations["b"].x
    assert graph.stations["b"].x < graph.stations["c"].x


def test_compute_layout_branching():
    """Layout assigns correct layers for a diamond pattern within a section."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Section]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        a -->|main| b\n"
        "        b -->|main| d\n"
        "        a -->|alt| c\n"
        "        c -->|alt| d\n"
        "    end\n"
    )
    compute_layout(graph, x_spacing=100, y_spacing=50)
    # a at layer 0, d at layer 2
    assert graph.stations["a"].layer == 0
    assert graph.stations["d"].layer == 2
    # b and c at same layer but different tracks
    assert graph.stations["b"].layer == graph.stations["c"].layer == 1
    assert graph.stations["b"].track != graph.stations["c"].track


# --- Section-first layout tests ---


def _make_section_graph():
    """Two-section graph with an inter-section edge."""
    return parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        c -->|main| d\n"
        "    end\n"
        "    b -->|main| c\n"
    )


def test_section_layout_assigns_coordinates():
    """Section-first layout assigns non-zero coordinates to all real stations."""
    graph = _make_section_graph()
    compute_layout(graph)
    for sid, station in graph.stations.items():
        if not station.is_port:
            # Stations should have non-negative coordinates
            assert station.x >= 0, f"Station {sid} has x={station.x}"
            assert station.y >= 0, f"Station {sid} has y={station.y}"


def test_section_layout_sections_dont_overlap():
    """Section bounding boxes should not overlap."""
    graph = _make_section_graph()
    compute_layout(graph)

    boxes = []
    for section in graph.sections.values():
        if section.bbox_w > 0:
            boxes.append(
                (
                    section.bbox_x,
                    section.bbox_y,
                    section.bbox_x + section.bbox_w,
                    section.bbox_y + section.bbox_h,
                )
            )

    # Check pairwise non-overlap
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            overlap = not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)
            assert not overlap, (
                f"Sections {i} and {j} overlap: {boxes[i]} vs {boxes[j]}"
            )


def test_section_layout_preserves_edge_order():
    """Within a section, layering should preserve edge direction (a before b)."""
    graph = _make_section_graph()
    compute_layout(graph)
    # a should be to the left of b (earlier layer)
    assert graph.stations["a"].x < graph.stations["b"].x
    # c should be to the left of d
    assert graph.stations["c"].x < graph.stations["d"].x


def test_section_layout_sec1_left_of_sec2():
    """Section 1 (upstream) should be to the left of section 2 (downstream)."""
    graph = _make_section_graph()
    compute_layout(graph)
    sec1 = graph.sections["sec1"]
    sec2 = graph.sections["sec2"]
    assert sec1.bbox_x < sec2.bbox_x


def test_section_layout_with_grid_override():
    """Grid overrides should position sections at specified grid cells."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "%%metro grid: sec2 | 1,0\n"
        "%%metro grid: sec3 | 1,1\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        b[B]\n"
        "    end\n"
        "    subgraph sec3 [Section Three]\n"
        "        c[C]\n"
        "    end\n"
        "    a -->|main| b\n"
        "    a -->|alt| c\n"
    )
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    # sec2 and sec3 should be in the same column but different rows
    assert graph.sections["sec2"].grid_col == graph.sections["sec3"].grid_col == 1
    assert graph.sections["sec2"].grid_row != graph.sections["sec3"].grid_row
    # sec2 (row 0) above sec3 (row 1)
    assert graph.sections["sec2"].bbox_y < graph.sections["sec3"].bbox_y


def test_section_layout_ports_skip_rendering():
    """Port stations should be filtered from label placement."""
    from nf_metro.layout.labels import place_labels

    graph = _make_section_graph()
    compute_layout(graph)
    labels = place_labels(graph)
    port_labels = [lb for lb in labels if lb.station_id in graph.ports]
    assert len(port_labels) == 0
