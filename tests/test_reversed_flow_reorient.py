"""Flow-axis re-orientation and exit-port anchoring on folded RL sections.

Two defects on horizontal sections whose declared ports face against an
inferred LR flow:

* #1298 -- a section that declares a flow-axis entry on one side and an exit on
  the opposite side reads as reversed even when one of those ports feeds (or is
  fed by) the internal flow extreme, so the port does not itself double back.
  ``_reanchor_flow_axis_ports`` must re-orient the whole section rather than
  re-anchor a single port and leave the flow pointing the wrong way.

* #1300 -- a perpendicular-entry station shift on an RL section grows the
  trailing edge outward; the flow-axis exit port must travel with that edge
  instead of being stranded inside the box, where the exit leg would double
  back to reach it.
"""

from __future__ import annotations

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide

REVERSED_FLOW = "examples/topologies/rl_entry_right_exit_left.mmd"
RIBOSEQ_FOLD = "tests/fixtures/regressions/riboseq_fold_1300.mmd"


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


def test_perp_entry_shift_keeps_flow_exit_port_on_boundary() -> None:
    """An RL section's LEFT exit stays on its trailing edge after the shift (#1300).

    The P-site section takes a perpendicular top entry (which shifts its
    internal stations and grows the trailing edge) and a flow-axis left exit fed
    by its internal flow-sink.  The exit port must sit on the (moved) left
    boundary, not stranded inside the box where the exit leg bulges out and
    doubles back.
    """
    graph = _graph(RIBOSEQ_FOLD, validate=False)
    section = graph.sections["psite_id"]
    exit_left = [
        pid for pid in section.exit_ports if graph.ports[pid].side == PortSide.LEFT
    ]
    assert exit_left, "psite_id should have a LEFT exit port"
    for pid in exit_left:
        port_x = graph.stations[pid].x
        assert abs(port_x - section.bbox_x) < 1.0, (
            f"exit port {pid} at x={port_x:.1f} is not on the section's left "
            f"boundary x={section.bbox_x:.1f}"
        )
