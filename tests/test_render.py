"""Tests for SVG rendering."""

import xml.etree.ElementTree as ET

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.render.svg import render_svg
from nf_metro.themes import LIGHT_THEME, NFCORE_THEME


def _render_simple():
    graph = parse_metro_mermaid(
        "%%metro title: Test\n"
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    a[Input]\n"
        "    b[Output]\n"
        "    a -->|main| b\n"
    )
    compute_layout(graph)
    return render_svg(graph, NFCORE_THEME)


def test_render_produces_valid_svg():
    svg = _render_simple()
    # Should be valid XML
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") or "svg" in root.tag


def test_render_contains_title():
    svg = _render_simple()
    assert "Test" in svg


def test_render_contains_station_labels():
    svg = _render_simple()
    assert "Input" in svg
    assert "Output" in svg


def test_render_contains_line_color():
    svg = _render_simple()
    assert "#ff0000" in svg


def test_render_nfcore_theme_background():
    svg = _render_simple()
    assert NFCORE_THEME.background_color in svg


def test_render_light_theme():
    graph = parse_metro_mermaid(
        "%%metro title: Light Test\n"
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    a --> b\n"
    )
    compute_layout(graph)
    svg = render_svg(graph, LIGHT_THEME)
    # Light theme uses transparent background (no background rectangle)
    assert LIGHT_THEME.background_color == "none"
    assert "#333333" in svg  # label/stroke color present


def test_render_empty_graph():
    graph = parse_metro_mermaid("graph LR\n")
    svg = render_svg(graph, NFCORE_THEME)
    assert "svg" in svg


def test_render_legend():
    svg = _render_simple()
    # Legend should contain the line display name
    assert "Main" in svg


def test_render_file_size():
    """SVG output should be reasonably small."""
    graph = parse_metro_mermaid(
        "%%metro title: Size Test\n"
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    a[A]\n    b[B]\n    c[C]\n    d[D]\n    e[E]\n"
        "    a -->|main| b\n    b -->|main| c\n    c -->|main| d\n    d -->|main| e\n"
        "    a -->|alt| c\n    c -->|alt| e\n"
    )
    compute_layout(graph)
    svg = render_svg(graph, NFCORE_THEME)
    # Should be well under 50KB for a small graph
    assert len(svg) < 50000


# --- First-class section rendering tests ---


def test_render_first_class_sections():
    """First-class sections render section boxes with names."""
    graph = parse_metro_mermaid(
        "%%metro title: Section Test\n"
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Processing]\n"
        "        a[Input]\n"
        "        b[Middle]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Output]\n"
        "        c[Result]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    compute_layout(graph)
    svg = render_svg(graph, NFCORE_THEME)
    assert "Processing" in svg
    assert "Output" in svg
    assert "Input" in svg
    assert "Result" in svg
    # Should be valid XML
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") or "svg" in root.tag


def test_render_sections_no_port_labels():
    """Port stations should not appear as labels in the SVG."""
    graph = parse_metro_mermaid(
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [S1]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph sec2 [S2]\n"
        "        b[B]\n"
        "    end\n"
        "    a -->|main| b\n"
    )
    compute_layout(graph)
    svg = render_svg(graph, NFCORE_THEME)
    # Port IDs should not appear in the SVG text
    for port_id in graph.ports:
        assert port_id not in svg, f"Port {port_id} should not appear in SVG"


def test_render_rnaseq_sections_example():
    """The rnaseq_sections.mmd example should render without errors."""
    from pathlib import Path

    examples = Path(__file__).parent.parent / "examples"
    text = (examples / "rnaseq_sections.mmd").read_text()
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    svg = render_svg(graph, NFCORE_THEME)
    # Title text is replaced by embedded logo, so check section labels
    assert "Pre-processing" in svg
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") or "svg" in root.tag
