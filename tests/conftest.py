"""Shared test fixtures and helpers for nf-metro test suite."""

from __future__ import annotations

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph

# --- Graph text constants ---

SIMPLE_LINEAR_TEXT = (
    "%%metro line: main | Main | #ff0000\n"
    "graph LR\n"
    "    a[A]\n"
    "    b[B]\n"
    "    c[C]\n"
    "    a -->|main| b\n"
    "    b -->|main| c\n"
)

DIAMOND_TEXT = (
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

TWO_SECTION_TEXT = (
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


# --- Parse/layout helpers ---


def parse_and_layout(text: str, **kwargs) -> MetroGraph:
    """Parse Mermaid text and run the full layout pipeline.

    Accepts keyword arguments passed to compute_layout (e.g. x_spacing, y_spacing).
    """
    graph = parse_metro_mermaid(text)
    compute_layout(graph, **kwargs)
    return graph


# --- Pytest fixtures ---


@pytest.fixture
def simple_linear_graph() -> MetroGraph:
    """A 3-node linear chain: a -> b -> c on one line."""
    return parse_metro_mermaid(SIMPLE_LINEAR_TEXT)


@pytest.fixture
def diamond_graph() -> MetroGraph:
    """A 4-node diamond: a -> {b, c} -> d on two lines."""
    return parse_metro_mermaid(DIAMOND_TEXT)


@pytest.fixture
def two_section_graph() -> MetroGraph:
    """Two sections with one inter-section edge, laid out."""
    return parse_and_layout(TWO_SECTION_TEXT)
