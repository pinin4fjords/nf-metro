"""Flow reversal for a folded flow-axis port must not depend on axis.

``_reanchor_flow_axis_ports`` resolves a folded flow-axis port -- one declared
on the end its connecting station is not adjacent to, so the line would route
back through the section's own stations -- with one of two remedies: reverse
the section's flow, or re-anchor the port to the opposite side.  Reversal is
only safe when no flow-axis port already runs with the flow; whether the
section's flow happens to run horizontally or vertically carries no bearing
on that safety condition, so the two remedies must be reachable identically
in both cases.
"""

from __future__ import annotations

import pytest

from nf_metro.parser.model import Edge, MetroGraph, PortSide, Section, Station
from nf_metro.parser.resolve import (
    _LEADING_SIDE,
    _TRAILING_SIDE,
    _reanchor_flow_axis_ports,
)

_REVERSED = {"LR": "RL", "RL": "LR", "TB": "BT", "BT": "TB"}

# A grid position higher along a port side's axis: right of centre on the
# horizontal axis, below it on the vertical one.  Mirrors the low/high sense
# _connecting_flow_side (resolve.py) reads off grid_col/grid_row.
_HIGH_SIDE = frozenset({PortSide.RIGHT, PortSide.BOTTOM})


def _folded_section_graph(direction: str) -> tuple[MetroGraph, list[Edge]]:
    """A 'mid' section fed by 'feed' and feeding 'sink', flowing *direction*.

    'mid' holds two stations, m1 -> m2.  Its entry sits on the trailing edge
    and feeds m2 directly (m2 is 'mid's own flow-sink), so the entry does not
    itself fold.  Its exit sits on the leading edge and is fed by that same
    m2 -- not 'mid's flow-source m1 -- so the exit folds.  Neither port runs
    with the flow (an entry on the leading edge, or an exit on the trailing
    one), so the fold is resolvable by reversing 'mid' rather than re-anchoring
    the exit.

    'feed' and 'sink' are gridded genuinely on the trailing and leading sides
    respectively -- the declared hints already face the right way, so only
    'mid's own flow assumption is backwards.  A fold whose connecting station
    is not actually on the declared side has no direct-approach route either
    way the section faces, so reversal only fires once that much is true.
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

    axis = "col" if entry_side in (PortSide.LEFT, PortSide.RIGHT) else "row"
    mid_pos, feed_pos, sink_pos = 1, (2 if entry_side in _HIGH_SIDE else 0), (
        2 if exit_side in _HIGH_SIDE else 0
    )
    for section, pos in ((feed, feed_pos), (mid, mid_pos), (sink, sink_pos)):
        col, row = (pos, 0) if axis == "col" else (0, pos)
        section.grid_col, section.grid_row = col, row

    inter_section_edges = [
        Edge(source="f1", target="m2", line_id="a"),
        Edge(source="m2", target="s1", line_id="a"),
    ]
    return graph, inter_section_edges


@pytest.mark.parametrize("direction", ["LR", "RL", "TB", "BT"])
def test_folded_exit_reverses_flow_regardless_of_axis(direction: str) -> None:
    """A vertical flow gets the same reversal remedy a horizontal one gets.

    The remedy is asserted in flow-relative terms: the section's declared
    direction becomes the reverse of what it started as, and the declared
    port hints -- which side each port sits on relative to the section's own
    edges -- are untouched (a re-anchor would instead rewrite the folded
    port's hint to the opposite side).
    """
    graph, inter_section_edges = _folded_section_graph(direction)
    original_entry_hints = list(graph.sections["mid"].entry_hints)
    original_exit_hints = list(graph.sections["mid"].exit_hints)

    with pytest.warns(UserWarning, match="flow re-oriented"):
        _reanchor_flow_axis_ports(graph, inter_section_edges)

    mid = graph.sections["mid"]
    assert mid.direction == _REVERSED[direction]
    assert mid.entry_hints == original_entry_hints
    assert mid.exit_hints == original_exit_hints
    assert "mid" in graph._fold_reoriented_sections
