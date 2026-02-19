"""Tests for the Mermaid + metro directive parser."""

from pathlib import Path

from nf_metro.parser.mermaid import parse_metro_mermaid

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_title():
    text = "%%metro title: My Pipeline\ngraph LR\n"
    graph = parse_metro_mermaid(text)
    assert graph.title == "My Pipeline"


def test_parse_style():
    text = "%%metro style: light\ngraph LR\n"
    graph = parse_metro_mermaid(text)
    assert graph.style == "light"


def test_parse_line_order():
    text = "%%metro line_order: span\ngraph LR\n"
    graph = parse_metro_mermaid(text)
    assert graph.line_order == "span"


def test_parse_line_order_default():
    text = "graph LR\n"
    graph = parse_metro_mermaid(text)
    assert graph.line_order == "definition"


def test_parse_line_order_invalid_ignored():
    text = "%%metro line_order: invalid\ngraph LR\n"
    graph = parse_metro_mermaid(text)
    assert graph.line_order == "definition"


def test_parse_lines():
    text = (
        "%%metro line: main | Main Line | #ff0000\n"
        "%%metro line: alt | Alt Line | #0000ff\n"
        "graph LR\n"
    )
    graph = parse_metro_mermaid(text)
    assert len(graph.lines) == 2
    assert graph.lines["main"].display_name == "Main Line"
    assert graph.lines["main"].color == "#ff0000"
    assert graph.lines["alt"].color == "#0000ff"


def test_parse_nodes_square_bracket():
    text = "graph LR\n    fastqc[FastQC]\n"
    graph = parse_metro_mermaid(text)
    assert "fastqc" in graph.stations
    assert graph.stations["fastqc"].label == "FastQC"


def test_parse_nodes_bare():
    text = "graph LR\n    mynode\n"
    graph = parse_metro_mermaid(text)
    assert "mynode" in graph.stations
    assert graph.stations["mynode"].label == "mynode"


def test_parse_edges():
    text = "graph LR\n    a[Input]\n    b[Output]\n    a -->|main| b\n"
    graph = parse_metro_mermaid(text)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "a"
    assert graph.edges[0].target == "b"
    assert graph.edges[0].line_id == "main"


def test_parse_edges_no_label():
    text = "graph LR\n    a --> b\n"
    graph = parse_metro_mermaid(text)
    assert len(graph.edges) == 1
    assert graph.edges[0].line_id == "default"


def test_station_lines():
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    a -->|main| b\n"
        "    a -->|alt| c\n"
    )
    graph = parse_metro_mermaid(text)
    lines = graph.station_lines("a")
    assert "main" in lines
    assert "alt" in lines


def test_line_stations():
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    a -->|main| b\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    stations = graph.line_stations("main")
    assert stations == ["a", "b", "c"]


def test_parse_simple_fixture():
    text = (FIXTURES / "rnaseq_simple.mmd").read_text()
    graph = parse_metro_mermaid(text)
    assert graph.title == "Test Pipeline"
    assert len(graph.stations) == 4
    assert len(graph.edges) == 4
    assert len(graph.lines) == 2


def test_ignores_comments():
    text = "%% This is a regular comment\n%%metro title: Test\ngraph LR\n    a --> b\n"
    graph = parse_metro_mermaid(text)
    assert graph.title == "Test"
    assert len(graph.edges) == 1


# --- Subgraph (first-class section) tests ---


def test_parse_subgraph_basic():
    """Subgraphs create first-class sections."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[Input]\n"
        "        b[Middle]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[Output]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    assert len(graph.sections) == 2
    assert "sec1" in graph.sections
    assert "sec2" in graph.sections
    assert graph.sections["sec1"].name == "Section One"
    assert graph.sections["sec2"].name == "Section Two"


def test_subgraph_station_membership():
    """Stations inside subgraphs get section_id set."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[Input]\n"
        "        b[Middle]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[Output]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    assert graph.stations["a"].section_id == "sec1"
    assert graph.stations["b"].section_id == "sec1"
    assert graph.stations["c"].section_id == "sec2"


def test_subgraph_section_station_ids():
    """Section.station_ids lists the stations in the section."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[Input]\n"
        "        b[Middle]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[Output]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    # sec1 has a, b plus port stations
    real_stations_sec1 = [
        s for s in graph.sections["sec1"].station_ids if not graph.stations[s].is_port
    ]
    assert set(real_stations_sec1) == {"a", "b"}
    real_stations_sec2 = [
        s for s in graph.sections["sec2"].station_ids if not graph.stations[s].is_port
    ]
    assert set(real_stations_sec2) == {"c"}


def test_inter_section_edge_rewriting():
    """Inter-section edges are rewritten into 3-part chains with ports."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[Input]\n"
        "        b[Middle]\n"
        "        a -->|main| b\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        c[Output]\n"
        "    end\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    # Should have ports
    assert len(graph.ports) > 0
    # Original direct b->c edge should be gone,
    # replaced by b->exit, exit->entry, entry->c
    direct_edges = [e for e in graph.edges if e.source == "b" and e.target == "c"]
    assert len(direct_edges) == 0
    # Should have edges from b to an exit port
    b_to_port = [
        e for e in graph.edges if e.source == "b" and graph.stations[e.target].is_port
    ]
    assert len(b_to_port) >= 1


def test_port_directive_parsing():
    """%%metro entry/exit directives inside subgraphs create ports."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        %%metro exit: right | main, alt\n"
        "        a[Input]\n"
        "    end\n"
        "    subgraph sec2 [Section Two]\n"
        "        %%metro entry: left | main, alt\n"
        "        b[Output]\n"
        "    end\n"
        "    a -->|main| b\n"
        "    a -->|alt| b\n"
    )
    graph = parse_metro_mermaid(text)
    # Should have explicit ports from directives plus auto-created ports
    exit_ports = [
        p for p in graph.ports.values() if not p.is_entry and p.section_id == "sec1"
    ]
    entry_ports = [
        p for p in graph.ports.values() if p.is_entry and p.section_id == "sec2"
    ]
    assert len(exit_ports) >= 1
    assert len(entry_ports) >= 1


def test_grid_directive_parsing():
    """%%metro grid: directives set grid overrides."""
    text = "%%metro grid: sec2 | 1,0\n%%metro grid: sec3 | 1,1\ngraph LR\n    a --> b\n"
    graph = parse_metro_mermaid(text)
    assert graph.grid_overrides["sec2"] == (1, 0, 1, 1)
    assert graph.grid_overrides["sec3"] == (1, 1, 1, 1)


def test_section_numbering():
    """Sections are auto-numbered in definition order."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "graph LR\n"
        "    subgraph first [First]\n"
        "        a[A]\n"
        "    end\n"
        "    subgraph second [Second]\n"
        "        b[B]\n"
        "    end\n"
        "    subgraph third [Third]\n"
        "        c[C]\n"
        "    end\n"
        "    a -->|main| b\n"
        "    b -->|main| c\n"
    )
    graph = parse_metro_mermaid(text)
    assert graph.sections["first"].number == 1
    assert graph.sections["second"].number == 2
    assert graph.sections["third"].number == 3


def test_subgraph_without_display_name():
    """Subgraph without [display name] uses the id as name."""
    text = "graph LR\n    subgraph mysection\n        a[A]\n    end\n"
    graph = parse_metro_mermaid(text)
    assert "mysection" in graph.sections
    assert graph.sections["mysection"].name == "mysection"


# --- Hidden station tests ---


def test_hidden_station_underscore_prefix():
    """Stations with _ prefix are marked as hidden."""
    text = "graph LR\n    _hidden[Split Point]\n    visible[Visible]\n"
    graph = parse_metro_mermaid(text)
    assert graph.stations["_hidden"].is_hidden is True
    assert graph.stations["visible"].is_hidden is False


def test_hidden_station_auto_created_from_edge():
    """Stations with _ prefix are hidden even when auto-created from edges."""
    text = "graph LR\n    a -->|main| _split\n    _split -->|main| b\n"
    graph = parse_metro_mermaid(text)
    assert graph.stations["_split"].is_hidden is True
    assert graph.stations["a"].is_hidden is False
    assert graph.stations["b"].is_hidden is False


def test_hidden_station_edge_before_definition():
    """Hidden flag is set correctly when edge precedes node definition."""
    text = (
        "graph LR\n"
        "    a -->|main| _split\n"
        "    _split[Split]\n"
        "    _split -->|main| b\n"
    )
    graph = parse_metro_mermaid(text)
    assert graph.stations["_split"].is_hidden is True
    assert graph.stations["_split"].label == "Split"


def test_hidden_station_definition_before_edge():
    """Hidden flag is set correctly when node definition precedes edge."""
    text = (
        "graph LR\n"
        "    _split[Split]\n"
        "    a -->|main| _split\n"
        "    _split -->|main| b\n"
    )
    graph = parse_metro_mermaid(text)
    assert graph.stations["_split"].is_hidden is True


def test_hidden_station_in_section():
    """Hidden stations work inside sections."""
    text = (
        "%%metro line: main | Main | #ff0000\n"
        "%%metro line: alt | Alt | #0000ff\n"
        "graph LR\n"
        "    subgraph sec1 [Section One]\n"
        "        a[Input]\n"
        "        _branch\n"
        "        a -->|main,alt| _branch\n"
        "        _branch -->|main| b[Output A]\n"
        "        _branch -->|alt| c[Output B]\n"
        "    end\n"
    )
    graph = parse_metro_mermaid(text)
    assert graph.stations["_branch"].is_hidden is True
    assert graph.stations["_branch"].section_id == "sec1"
