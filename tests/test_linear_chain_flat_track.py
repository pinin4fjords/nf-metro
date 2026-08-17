"""A linear chain inside a section rides one track.

Consecutive stations joined by a sole-predecessor / sole-successor link have no
branching between them, so nothing displaces the second one off the first one's
track: in an LR/RL section they share a Y, in a TB/BT section an X.  A chain
that steps between tracks reads as a staircase, and the space that step opens
is space no plan reserved.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph

SEED_DIR = Path(__file__).parent / "fixtures" / "hash_seed_determinism"
TOPOLOGIES_DIR = Path(__file__).parent.parent / "examples" / "topologies"

_HORIZONTAL = {"LR", "RL"}


def _laid_out(path: Path) -> MetroGraph:
    graph = parse_metro_mermaid(path.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_layout(graph)
    return graph


def _drawn_stations(graph: MetroGraph, section_id: str) -> set[str]:
    return {
        station.id
        for station in graph.stations.values()
        if station.section_id == section_id
        and not station.is_port
        and not station.off_track
        and not station.id.startswith("__")
    }


def _sole_links(graph: MetroGraph, members: set[str]) -> list[tuple[str, str]]:
    """Sole-predecessor / sole-successor station pairs drawn inside one section.

    Both endpoints' full in/out degree is counted over the whole graph, so a
    station that also feeds a section exit is not treated as unbranched.
    """
    out_degree: dict[str, set[str]] = {member: set() for member in members}
    in_degree: dict[str, set[str]] = {member: set() for member in members}
    for edge in graph.edges:
        if edge.source in members:
            out_degree[edge.source].add(edge.target)
        if edge.target in members:
            in_degree[edge.target].add(edge.source)
    return [
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.source in members
        and edge.target in members
        and out_degree[edge.source] == {edge.target}
        and in_degree[edge.target] == {edge.source}
    ]


def _staircase_steps(graph: MetroGraph) -> list[str]:
    steps: list[str] = []
    for section_id, section in graph.sections.items():
        members = _drawn_stations(graph, section_id)
        if len(members) < 2:
            continue
        horizontal = (section.direction or "LR").upper() in _HORIZONTAL
        for source, target in _sole_links(graph, members):
            first = graph.stations[source]
            second = graph.stations[target]
            track = (first.y, second.y) if horizontal else (first.x, second.x)
            if abs(track[0] - track[1]) > 0.5:
                steps.append(
                    f"{section_id}: {source} and {target} are one unbranched chain "
                    f"but sit on tracks {track[0]} and {track[1]}"
                )
    return steps


@pytest.mark.parametrize("seed", [15, 41, 72, 77])
def test_seed_sections_hold_linear_chains_on_one_track(seed: int) -> None:
    assert _staircase_steps(_laid_out(SEED_DIR / f"seed_{seed}.mmd")) == []


_KNOWN_STEPPING = {
    # `p1` merges a line arriving through the section's entry port with one
    # arriving in-section, and is seated on the entry lane so that entry edge
    # runs flat; the seating is never handed on to `p2`, so the chain steps.
    # A different mechanism from the track assignment this module covers.
    "compact_hidden_passthrough.mmd",
}


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True, reason="in-section merge seats on the entry lane alone"
            ),
        )
        if name in _KNOWN_STEPPING
        else name
        for name in sorted(path.name for path in TOPOLOGIES_DIR.glob("*.mmd"))
    ],
)
def test_topology_sections_hold_linear_chains_on_one_track(fixture: str) -> None:
    assert _staircase_steps(_laid_out(TOPOLOGIES_DIR / fixture)) == []
