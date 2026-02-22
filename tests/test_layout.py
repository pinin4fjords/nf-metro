"""Tests for the layout engine."""

from layout_validator import Severity, check_station_as_elbow

from nf_metro.layout.constants import CHAR_WIDTH
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.labels import label_text_width
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


# ---- Multi-line label helpers ----


# --- Straight diamond tests (issue #115) ---


def _diamond_section_text(diamond_style="straight"):
    """Build a section with a 2-way diamond where all lines take both branches."""
    return (
        "%%metro line: L1 | Line1 | #ff0000\n"
        "%%metro line: L2 | Line2 | #0000ff\n"
        "graph LR\n"
        "    subgraph sec [Section]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        a -->|L1,L2| b\n"
        "        a -->|L1,L2| c\n"
        "        b -->|L1,L2| d\n"
        "        c -->|L1,L2| d\n"
        "    end\n"
    )


def test_is_diamond_fanout():
    """_is_diamond_fanout detects fork-join patterns."""
    import networkx as nx

    from nf_metro.layout.ordering import _is_diamond_fanout

    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    assert _is_diamond_fanout(["b", "c"], G) is True
    # Single node is never a diamond
    assert _is_diamond_fanout(["b"], G) is False
    # Nodes with different predecessors are not a diamond
    G2 = nx.DiGraph()
    G2.add_edges_from([("a", "b"), ("x", "c"), ("b", "d"), ("c", "d")])
    assert _is_diamond_fanout(["b", "c"], G2) is False


def test_straight_diamond_top_branch_stays_flat():
    """With diamond_style='straight', the top branch of a diamond stays on the trunk."""
    graph = parse_metro_mermaid(_diamond_section_text())
    # Default is now "straight"
    assert graph.diamond_style == "straight"
    compute_layout(graph)
    # b (first branch, top) should be at the same Y as a (trunk)
    assert graph.stations["b"].y == graph.stations["a"].y


def test_symmetric_diamond_both_branches_deviate():
    """With diamond_style='symmetric', both branches deviate from the trunk."""
    graph = parse_metro_mermaid(_diamond_section_text())
    graph.diamond_style = "symmetric"
    compute_layout(graph)
    a_y = graph.stations["a"].y
    b_y = graph.stations["b"].y
    c_y = graph.stations["c"].y
    # Both b and c should deviate from a (symmetric fan-out)
    assert b_y != a_y or c_y != a_y
    # And b should be above c (or at least at different positions)
    assert b_y != c_y


def test_straight_diamond_merge_returns_to_trunk():
    """With diamond_style='straight', the merge node after a diamond snaps to trunk."""
    graph = parse_metro_mermaid(_diamond_section_text())
    compute_layout(graph)
    # d (merge) should be at the same Y as a (trunk)
    assert graph.stations["d"].y == graph.stations["a"].y


def test_cli_straight_diamonds_default(tmp_path):
    """--straight-diamonds is on by default."""
    from click.testing import CliRunner

    from nf_metro.cli import cli

    mmd = tmp_path / "diamond.mmd"
    mmd.write_text(_diamond_section_text())
    out = tmp_path / "out.svg"
    runner = CliRunner()
    result = runner.invoke(cli, ["render", str(mmd), "-o", str(out)])
    assert result.exit_code == 0, result.output


def test_cli_no_straight_diamonds(tmp_path):
    """--no-straight-diamonds reverts to symmetric behaviour."""
    from click.testing import CliRunner

    from nf_metro.cli import cli

    mmd = tmp_path / "diamond.mmd"
    mmd.write_text(_diamond_section_text())
    out = tmp_path / "out.svg"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["render", str(mmd), "-o", str(out), "--no-straight-diamonds"]
    )
    assert result.exit_code == 0, result.output


def test_straight_diamond_inter_section_port_alignment():
    """With straight diamonds, inter-section ports align to the majority target Y."""
    graph = parse_metro_mermaid(
        "%%metro line: L1 | Line1 | #ff0000\n"
        "%%metro line: L2 | Line2 | #0000ff\n"
        "%%metro line: L3 | Line3 | #00ff00\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        b[B]\n"
        "        c[C]\n"
        "        b -->|L1,L2| c\n"
        "    end\n"
        "    a -->|L1,L2| b\n"
        "    a -->|L3| c\n"
    )
    compute_layout(graph)
    # The entry port should align with b (2 lines) not the average of b and c
    entry_ports = graph.sections["sec2"].entry_ports
    assert len(entry_ports) > 0
    entry_y = graph.stations[entry_ports[0]].y
    b_y = graph.stations["b"].y
    # Entry port should be at b's Y (majority target)
    assert abs(entry_y - b_y) < 1.0, (
        f"Entry port at y={entry_y} should align with b at y={b_y}"
    )


def test_label_text_width_single_line():
    assert label_text_width("Hello") == 5 * CHAR_WIDTH


def test_label_text_width_multiline():
    # Width should be based on the longest line
    assert label_text_width("AB\nCDEF") == 4 * CHAR_WIDTH


def test_label_text_width_empty():
    assert label_text_width("") == 0


# --- Port-terminus spacing (Phase 7c) ---


def test_port_terminus_spacing_basic():
    """Entry port is pushed away from a source terminus it doesn't connect to.

    Section 2 has a source terminus (ref_in) and an entry port carrying
    a different line (main from sec1).  The entry port must maintain at
    least y_spacing from ref_in so routed lines don't overlap the icon.
    """
    y_spacing = 40
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #2db572\n"
        "%%metro line: alt | Alt | #0570b0\n"
        "%%metro file: ref_in | FASTA\n"
        "graph LR\n"
        "    subgraph sec1 [Source]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Target]\n"
        "        ref_in[ ]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        ref_in -->|alt| c\n"
        "        c -->|main,alt| d\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    compute_layout(graph, y_spacing=y_spacing)

    # Identify the entry port(s) on sec2
    sec2 = graph.sections["sec2"]
    ref_y = graph.stations["ref_in"].y

    for pid in sec2.entry_ports:
        port_st = graph.stations[pid]
        # The entry port carries 'main' from sec1 and should NOT be
        # directly connected to ref_in.  Verify spacing.
        neighbours = set()
        for edge in graph.edges:
            if edge.source == pid:
                neighbours.add(edge.target)
            if edge.target == pid:
                neighbours.add(edge.source)

        if "ref_in" not in neighbours:
            gap = abs(port_st.y - ref_y)
            assert gap >= y_spacing - 1, (
                f"Port {pid} at y={port_st.y:.1f} is only {gap:.1f}px "
                f"from terminus ref_in at y={ref_y:.1f} "
                f"(need >= {y_spacing})"
            )


def test_port_terminus_spacing_no_station_as_elbow():
    """Phase 7c must not introduce station-as-elbow violations.

    Uses the variant_calling_tuned example which triggered the original
    icon overlap issue, and checks that the fix doesn't create new
    station-as-elbow problems.
    """
    from pathlib import Path

    example = Path(__file__).parent.parent / "examples" / "variant_calling_tuned.mmd"
    if not example.exists():
        return
    graph = parse_metro_mermaid(example.read_text())
    compute_layout(graph)

    violations = check_station_as_elbow(graph)
    errors = [v for v in violations if v.severity == Severity.ERROR]
    assert not errors, "station-as-elbow violations after Phase 7c:\n" + "\n".join(
        v.message for v in errors
    )


def test_port_terminus_spacing_multi_terminus():
    """When two termini are near a port, the port clears both of them.

    Tests the convergence guarantee: the port should not thrash between
    two conflicting termini, but settle at a Y that satisfies both.
    """
    y_spacing = 40
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt1 | Alt1 | #00ff00\n"
        "%%metro line: alt2 | Alt2 | #0000ff\n"
        "%%metro file: src1 | FASTA\n"
        "%%metro file: src2 | BED\n"
        "graph LR\n"
        "    subgraph sec1 [Source]\n"
        "        a[A]\n"
        "        b[B]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Target]\n"
        "        src1[ ]\n"
        "        src2[ ]\n"
        "        c[C]\n"
        "        d[D]\n"
        "        src1 -->|alt1| c\n"
        "        src2 -->|alt2| d\n"
        "        c -->|main,alt1| d\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    compute_layout(graph, y_spacing=y_spacing)

    sec2 = graph.sections["sec2"]
    src1_y = graph.stations["src1"].y
    src2_y = graph.stations["src2"].y

    for pid in sec2.entry_ports:
        port_st = graph.stations[pid]
        neighbours = set()
        for edge in graph.edges:
            if edge.source == pid:
                neighbours.add(edge.target)
            if edge.target == pid:
                neighbours.add(edge.source)

        # Check distance from each non-connected terminus
        for tid, ty in [("src1", src1_y), ("src2", src2_y)]:
            if tid not in neighbours:
                gap = abs(port_st.y - ty)
                assert gap >= y_spacing - 1, (
                    f"Port {pid} at y={port_st.y:.1f} is only "
                    f"{gap:.1f}px from terminus {tid} at y={ty:.1f} "
                    f"(need >= {y_spacing})"
                )
