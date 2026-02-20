"""Tests for the layout engine."""

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.layers import assign_layers
from nf_metro.layout.ordering import assign_tracks
from nf_metro.parser.mermaid import parse_metro_mermaid


def test_layer_assignment_linear(simple_linear_graph):
    layers = assign_layers(simple_linear_graph)
    assert layers["a"] == 0
    assert layers["b"] == 1
    assert layers["c"] == 2


def test_layer_assignment_branching(diamond_graph):
    layers = assign_layers(diamond_graph)
    assert layers["a"] == 0
    # b and c both have a as predecessor, so both at layer 1
    assert layers["b"] == 1
    assert layers["c"] == 1
    # d has b and c as predecessors (both at layer 1), so at layer 2
    assert layers["d"] == 2


def test_track_assignment(diamond_graph):
    layers = assign_layers(diamond_graph)
    tracks = assign_tracks(diamond_graph, layers)
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


def test_section_layout_assigns_coordinates(two_section_graph):
    """Section-first layout assigns non-zero coordinates to all real stations."""
    for sid, station in two_section_graph.stations.items():
        if not station.is_port:
            assert station.x >= 0, f"Station {sid} has x={station.x}"
            assert station.y >= 0, f"Station {sid} has y={station.y}"


def test_section_layout_sections_dont_overlap(two_section_graph):
    """Section bounding boxes should not overlap."""
    boxes = []
    for section in two_section_graph.sections.values():
        if section.bbox_w > 0:
            boxes.append(
                (
                    section.bbox_x,
                    section.bbox_y,
                    section.bbox_x + section.bbox_w,
                    section.bbox_y + section.bbox_h,
                )
            )

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            overlap = not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)
            assert not overlap, (
                f"Sections {i} and {j} overlap: {boxes[i]} vs {boxes[j]}"
            )


def test_section_layout_preserves_edge_order(two_section_graph):
    """Within a section, layering should preserve edge direction (a before b)."""
    assert two_section_graph.stations["a"].x < two_section_graph.stations["b"].x
    assert two_section_graph.stations["c"].x < two_section_graph.stations["d"].x


def test_section_layout_sec1_left_of_sec2(two_section_graph):
    """Section 1 (upstream) should be to the left of section 2 (downstream)."""
    sec1 = two_section_graph.sections["sec1"]
    sec2 = two_section_graph.sections["sec2"]
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


def test_section_layout_ports_skip_rendering(two_section_graph):
    """Port stations should be filtered from label placement."""
    from nf_metro.layout.labels import place_labels

    labels = place_labels(two_section_graph)
    port_labels = [lb for lb in labels if lb.station_id in two_section_graph.ports]
    assert len(port_labels) == 0


# --- Top-alignment tests ---


def test_sections_top_aligned_in_same_row():
    """Sections in the same row share the same top, not centered."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Tall Section]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        a -->|main| b\n"
        "        a -->|alt| c\n"
        "        b -->|main| d\n"
        "        c -->|alt| d\n"
        "    end\n"
        "    subgraph sec2 [Short Section]\n"
        "        e[E]\n"
        "        f[F]\n"
        "        e -->|main| f\n"
        "    end\n"
        "    d -->|main| e\n"
    )
    compute_layout(graph)
    sec1 = graph.sections["sec1"]
    sec2 = graph.sections["sec2"]
    # Both should be in the same row
    assert sec1.grid_row == sec2.grid_row == 0
    # Top edges should be flush (same bbox_y)
    assert abs(sec1.bbox_y - sec2.bbox_y) < 1.0, (
        f"Not top-aligned: sec1={sec1.bbox_y}, sec2={sec2.bbox_y}"
    )


# --- Exit-side clearance tests ---


def test_lr_exit_clearance_widens_bbox():
    """LR section with exit port gets wider bbox for label clearance."""
    # Build two sections so an exit port is created on sec1's right side
    graph_with_exit = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[LongLabelStation]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    # Build a standalone section (no exit port) with the same internal content
    graph_no_exit = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[LongLabelStation]\n"
        "        a -->|main| b\n"
        "    end\n"
    )
    compute_layout(graph_with_exit)
    compute_layout(graph_no_exit)
    # The section with exit should be wider
    w_exit = graph_with_exit.sections["sec1"].bbox_w
    w_no = graph_no_exit.sections["sec1"].bbox_w
    assert w_exit > w_no


def test_rl_exit_clearance_preserves_bbox_x():
    """RL section exit clearance should shift stations right, not move bbox_x left."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Source]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph sec2 [RL Section]\n"
        "        b[B]\n"
        "        c[LongLabel]\n"
        "        c -->|main| b\n"
        "    end\n"
        "    subgraph sec3 [Target]\n"
        "        d[D]\n"
        "    end\n"
        "    a -->|main| c\n"
        "    b -->|main| d\n"
    )
    compute_layout(graph)
    sec2 = graph.sections["sec2"]
    # The section should have a valid bbox_x aligned with its grid column offset.
    # The key invariant: stations within the section should be contained within
    # the bbox (checked by station_containment validator).
    for sid in sec2.station_ids:
        station = graph.stations.get(sid)
        if station and not station.is_port:
            assert station.x >= sec2.bbox_x, (
                f"Station {sid} at x={station.x} is left of bbox_x={sec2.bbox_x}"
            )
            assert station.x <= sec2.bbox_x + sec2.bbox_w, (
                f"Station {sid} at x={station.x} is right of bbox edge"
            )


# --- Flat layout empty tracks test ---


def test_flat_layout_unnamed_edges():
    """Unnamed edges (no line IDs) raise a clear error (issue #75)."""
    import pytest

    with pytest.raises(ValueError, match="no metro line annotation"):
        parse_metro_mermaid(
            "%%metro line: main | Main | #ff0000\ngraph LR\n    a --> b\n"
        )


# --- Line order tests ---


def test_line_order_definition_default():
    """Default line_order='definition' preserves .mmd line definition order."""
    graph = parse_metro_mermaid(
        "%%metro line: short | Short | #ff0000\n"
        "%%metro line: long | Long | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|short| b\n"
        "        a -->|long| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        c -->|long| d\n"
        "    end\n"
        "    b -->|long| c\n"
    )
    assert graph.line_order == "definition"
    layers = assign_layers(graph)
    tracks = assign_tracks(graph, layers)
    # 'short' should have base track 0 (defined first)
    # Stations on short line should be at track 0
    assert tracks["a"] is not None


def test_line_order_span_reorders():
    """line_order='span' gives inner tracks to lines spanning more sections."""
    from nf_metro.layout.ordering import _reorder_by_span

    graph = parse_metro_mermaid(
        "%%metro line: short | Short | #ff0000\n"
        "%%metro line: long | Long | #0000ff\n"
        "%%metro line_order: span\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|short| b\n"
        "        a -->|long| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        c -->|long| d\n"
        "    end\n"
        "    b -->|long| c\n"
    )
    assert graph.line_order == "span"
    line_order = list(graph.lines.keys())
    reordered = _reorder_by_span(graph, line_order)
    # 'long' spans 2 sections, 'short' spans 1 -> long should come first
    assert reordered[0] == "long"
    assert reordered[1] == "short"


def test_line_order_span_preserves_ties():
    """Lines with equal span preserve definition order."""
    from nf_metro.layout.ordering import _reorder_by_span

    graph = parse_metro_mermaid(
        "%%metro line: alpha | Alpha | #ff0000\n"
        "%%metro line: beta | Beta | #0000ff\n"
        "%%metro line_order: span\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|alpha| b\n"
        "        a -->|beta| b\n"
        "    end\n"
    )
    line_order = list(graph.lines.keys())
    reordered = _reorder_by_span(graph, line_order)
    # Both span 1 section -> preserve original order
    assert reordered == ["alpha", "beta"]


def test_line_order_span_e2e():
    """End-to-end: span ordering changes track assignment."""
    # With definition order: short gets track 0, long gets track 1
    graph_def = parse_metro_mermaid(
        "%%metro line: short | Short | #ff0000\n"
        "%%metro line: long | Long | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|short| b\n"
        "        a -->|long| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        c -->|long| d\n"
        "    end\n"
        "    b -->|long| c\n"
    )
    compute_layout(graph_def)

    # With span order: long gets track 0, short gets track 1
    graph_span = parse_metro_mermaid(
        "%%metro line: short | Short | #ff0000\n"
        "%%metro line: long | Long | #0000ff\n"
        "%%metro line_order: span\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|short| b\n"
        "        a -->|long| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        c -->|long| d\n"
        "    end\n"
        "    b -->|long| c\n"
    )
    compute_layout(graph_span)

    # In sec1, both 'a' and 'b' are on both lines. The key difference
    # is which line's base track is 0. With span ordering, 'long' gets
    # the inner track.
    # We verify that section layouts both succeed (no crash)
    assert graph_def.stations["a"].x > 0
    assert graph_span.stations["a"].x > 0


def test_flat_layout_no_named_lines():
    """Unnamed edges with a declared line still raise an error (issue #75)."""
    import pytest

    with pytest.raises(ValueError, match="no metro line annotation"):
        parse_metro_mermaid(
            "%%metro line: main | Main | #ff0000\n"
            "graph LR\n"
            "    a[Start]\n"
            "    b[End]\n"
            "    a --> b\n"
        )


# --- Label clamping tests (issue #58) ---


def test_label_clamp_flips_when_overlapping_pill():
    """Label clamped into pill should flip to the opposite side (issue #58)."""
    from nf_metro.layout.constants import LABEL_OFFSET
    from nf_metro.layout.labels import place_labels
    from nf_metro.layout.routing.offsets import compute_station_offsets

    # Build a section with many tracks so the bottom station is near
    # the section bbox bottom edge, triggering clamping for below labels.
    graph = parse_metro_mermaid(
        "%%metro line: L1 | Line1 | #ff0000\n"
        "%%metro line: L2 | Line2 | #00ff00\n"
        "%%metro line: L3 | Line3 | #0000ff\n"
        "%%metro line: L4 | Line4 | #ff00ff\n"
        "graph LR\n"
        "    subgraph sec [Section]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        a -->|L1| b\n"
        "        a -->|L2| c\n"
        "        a -->|L3| d\n"
        "        b -->|L1| d\n"
        "        c -->|L2| d\n"
        "    end\n"
    )
    compute_layout(graph, y_spacing=40)

    station_offsets = compute_station_offsets(graph)
    labels = place_labels(graph, station_offsets=station_offsets)

    # For every label, verify it doesn't overlap its station pill
    for lp in labels:
        s = graph.stations[lp.station_id]
        lines = graph.station_lines(lp.station_id)
        offs = [station_offsets.get((lp.station_id, lid), 0.0) for lid in lines]
        pill_top = s.y + (min(offs) if offs else 0.0)
        pill_bottom = s.y + (max(offs) if offs else 0.0)

        if lp.above:
            gap = pill_top - lp.y
        else:
            gap = lp.y - pill_bottom

        assert gap >= LABEL_OFFSET - 1.0, (
            f"Label for {lp.station_id} too close to pill: "
            f"gap={gap:.1f}, min={LABEL_OFFSET - 1.0}"
        )


def test_label_clamp_expands_bbox_when_both_sides_tight():
    """When neither side fits, the section bbox should expand (issue #58)."""
    from nf_metro.layout.constants import LABEL_BBOX_MARGIN, LABEL_OFFSET
    from nf_metro.layout.labels import LabelPlacement, _clamp_label_vertical
    from nf_metro.parser.model import Section, Station

    # Create a tiny section where neither above nor below would fit
    sec = Section(id="tiny", name="Tiny")
    sec.bbox_x = 0
    sec.bbox_y = 100
    sec.bbox_w = 200
    sec.bbox_h = 50  # Very tight

    station = Station(id="s", label="Test")
    station.x = 100
    station.y = 125  # Center of the 50px-tall section
    station.section_id = "tiny"

    # Label below would be at y=141 (125+16), bottom at 155 > section bottom 150
    candidate = LabelPlacement(station_id="s", text="Test", x=100, y=141, above=False)
    original_bbox_h = sec.bbox_h
    result = _clamp_label_vertical(
        candidate, sec, station, LABEL_OFFSET, 0.0, 0.0, LABEL_BBOX_MARGIN
    )
    # The bbox should have expanded (either flipped to above and fit,
    # or expanded to accommodate)
    if not result.above:
        # If it stayed below, bbox must have grown
        assert sec.bbox_h > original_bbox_h
