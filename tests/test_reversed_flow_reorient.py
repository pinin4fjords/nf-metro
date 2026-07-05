"""Flow-axis re-orientation of a folded section against an inferred LR flow.

#1298 -- a section that declares a flow-axis entry on one side and an exit on
the opposite side reads as reversed even when one of those ports feeds (or is
fed by) the internal flow extreme, so the port does not itself double back.
``_reanchor_flow_axis_ports`` must re-orient the whole section rather than
re-anchor a single port and leave the flow pointing the wrong way.
"""

from __future__ import annotations

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid

REVERSED_FLOW = "examples/topologies/rl_entry_right_exit_left.mmd"


def _graph(path: str, *, validate: bool):
    graph = parse_metro_mermaid(open(path).read())
    compute_layout(graph, validate=validate)
    return graph


def test_entry_right_exit_left_reorients_to_rl() -> None:
    """A section fed from the right that exits left resolves to RL flow (#1298).

    The right entry feeds the section's internal flow-sink, so on the inferred
    LR reading it does not double back.  Re-orientation is gated on no flow-axis
    port running with the inferred flow (rather than on every port doubling
    back), so the section resolves to RL and lays out without raising.
    """
    graph = _graph(REVERSED_FLOW, validate=True)
    assert graph.sections["mid"].direction == "RL"
