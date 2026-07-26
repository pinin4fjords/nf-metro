"""Which remedy a folded flow-axis port gets, per flow axis.

A port declared on the edge opposite its connecting section makes the leg run
the length of the trunk and double back through every station in between.
``_reanchor_flow_axis_ports`` has two remedies: reverse the section's flow so
the declared edge becomes the right one, or move the port to the edge its
connecting station is actually on.

The choice is keyed to the flow axis, and deliberately so.  A horizontal
section is reversed; a vertical one has its port re-anchored, because reversing
it re-seats the trailing exit on the far edge and the route out then wraps
around the section and back through its target's interior.  These tests pin
that split so the asymmetry is a stated contract rather than an accident of
which directions someone happened to put in a lookup table.
"""

from __future__ import annotations

import warnings

import pytest

from nf_metro.parser.model import Edge, MetroGraph, PortSide, Section, Station
from nf_metro.parser.resolve import (
    _LEADING_SIDE,
    _TRAILING_SIDE,
    _flow_axis_is_x,
    _reanchor_flow_axis_ports,
)

_HIGH_SIDE = (PortSide.RIGHT, PortSide.BOTTOM)
_REVERSED = {"LR": "RL", "RL": "LR", "TB": "BT", "BT": "TB"}


def _folded_section_graph(direction: str) -> tuple[MetroGraph, list[Edge]]:
    """A 'mid' section flowing *direction*, whose exit folds.

    'mid' holds m1 -> m2.  Its entry sits on the trailing edge and feeds m2,
    'mid's own flow-sink, so the entry does not itself fold.  Its exit sits on
    the leading edge and is fed by that same m2 rather than by the flow-source
    m1, so the exit folds.  Neither port runs with the flow, which is the
    precondition a reversal needs.
    """
    graph = MetroGraph()
    feed = Section(id="feed", name="Feed", direction="LR")
    mid = Section(id="mid", name="Mid", direction=direction)
    sink = Section(id="sink", name="Sink", direction="LR")
    graph.sections = {"feed": feed, "mid": mid, "sink": sink}

    graph.stations = {
        "f1": Station(id="f1", label="F1", section_id="feed"),
        "m1": Station(id="m1", label="M1", section_id="mid"),
        "m2": Station(id="m2", label="M2", section_id="mid"),
        "s1": Station(id="s1", label="S1", section_id="sink"),
    }
    feed.station_ids = ["f1"]
    mid.station_ids = ["m1", "m2"]
    sink.station_ids = ["s1"]
    mid.internal_edges = [Edge(source="m1", target="m2", line_id="a")]
    entry_side = _TRAILING_SIDE[direction]
    exit_side = _LEADING_SIDE[direction]
    mid.entry_hints = [(entry_side, ["a"])]
    mid.exit_hints = [(exit_side, ["a"])]

    # Each neighbour sits on the side of 'mid' its port faces, so the fold is
    # the section's own flow being backwards rather than a misplaced port.
    axis_is_x = _flow_axis_is_x(direction)
    feed_pos = 2 if entry_side in _HIGH_SIDE else 0
    for section, pos in ((feed, feed_pos), (mid, 1), (sink, 2 - feed_pos)):
        section.grid_col, section.grid_row = (pos, 0) if axis_is_x else (0, pos)

    inter_section_edges = [
        Edge(source="f1", target="m2", line_id="a"),
        Edge(source="m2", target="s1", line_id="a"),
    ]
    return graph, inter_section_edges


def _resolve(direction: str) -> tuple[MetroGraph, list[str]]:
    graph, edges = _folded_section_graph(direction)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reanchor_flow_axis_ports(graph, edges)
    return graph, [str(w.message) for w in caught]


@pytest.mark.parametrize("direction", ["LR", "RL"])
def test_horizontal_fold_reverses_the_section(direction: str) -> None:
    """A horizontal section's flow is reversed, leaving its port hints alone."""
    graph, messages = _resolve(direction)
    mid = graph.sections["mid"]

    assert mid.direction == _REVERSED[direction]
    assert "mid" in graph._fold_reoriented_sections
    assert mid.entry_hints == [(_TRAILING_SIDE[direction], ["a"])]
    assert mid.exit_hints == [(_LEADING_SIDE[direction], ["a"])]
    assert any("flow re-oriented" in m for m in messages), messages


@pytest.mark.parametrize("direction", ["TB", "BT"])
def test_vertical_fold_reanchors_the_port(direction: str) -> None:
    """A vertical section keeps its flow and has the folded port moved.

    Reversing it instead would wrap the route out of the re-seated exit around
    the section and back through its target's interior, so the port-side
    remedy is the one that applies on this axis.
    """
    graph, messages = _resolve(direction)
    mid = graph.sections["mid"]

    assert mid.direction == direction
    assert "mid" not in graph._fold_reoriented_sections
    assert mid.exit_hints == [(_TRAILING_SIDE[direction], ["a"])]
    assert any("re-anchored" in m for m in messages), messages


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("LR", True), ("RL", True), ("TB", False), ("BT", False)],
)
def test_flow_axis_is_x_covers_every_direction(direction: str, expected: bool) -> None:
    """The axis question is answered from the frame, for all four flows.

    The same lookup once answered it by membership, tying "does this flow run
    along X" to "can this flow be reversed" -- two different questions that
    happen to have the same answer only for the horizontal pair.
    """
    assert _flow_axis_is_x(direction) is expected
