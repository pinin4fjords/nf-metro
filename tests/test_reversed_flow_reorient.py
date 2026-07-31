"""Flow-axis re-orientation of a folded section against an inferred LR flow.

#1298 -- a section that declares a flow-axis entry on one side and an exit on
the opposite side reads as reversed even when one of those ports feeds (or is
fed by) the internal flow extreme, so the port does not itself double back.
``_reanchor_flow_axis_ports`` must re-orient the whole section rather than
re-anchor a single port and leave the flow pointing the wrong way.
"""

from __future__ import annotations

from dataclasses import replace

from orientation_transform import transformable_reason

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.provenance import DecisionReason, DecisionState

REVERSED_FLOW = "examples/topologies/rl_entry_right_exit_left.mmd"
MULTILINE_REVERSED_FLOW = "examples/topologies/rl_entry_runway.mmd"


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
    decision = graph.layout_provenance.direction_decision("mid")

    assert graph.sections["mid"].direction == "RL"
    assert decision is not None
    assert decision.state is DecisionState.INFERRED_THEN_PINNED
    assert decision.reason is DecisionReason.FLOW_REORIENTED_DIRECTION
    assert transformable_reason(graph) == (
        "inferred flow direction: ['feed', 'mid', 'sink']"
    )


def test_routing_reads_the_reorientation_reason() -> None:
    graph = parse_metro_mermaid(open(MULTILINE_REVERSED_FLOW).read())
    compute_layout(graph, validate=True)
    geometry = {
        station_id: (station.x, station.y)
        for station_id, station in graph.stations.items()
    }
    exit_port = next(
        port_id
        for port_id, port in graph.ports.items()
        if port.section_id == "src_sec" and not port.is_entry
    )
    line_ids = graph.station_lines(exit_port)
    offsets = compute_station_offsets(graph)

    assert [offsets[(exit_port, line_id)] for line_id in line_ids] == [0.0, 4.0]

    decision = graph.layout_provenance.directions["src_sec"]
    graph.layout_provenance.directions["src_sec"] = replace(
        decision, reason=DecisionReason.AUTO_DIRECTION
    )
    without_transition = compute_station_offsets(graph)

    assert [without_transition[(exit_port, line_id)] for line_id in line_ids] == [
        -4.0,
        0.0,
    ]
    assert {
        station_id: (station.x, station.y)
        for station_id, station in graph.stations.items()
    } == geometry
