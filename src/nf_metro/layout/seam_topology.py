"""Topology predicates shared by section placement and seam routing."""

from __future__ import annotations

from collections import defaultdict

from nf_metro.layout.geometry import lanes_run_along_y
from nf_metro.parser.model import MetroGraph, Port, PortSide, Section

_OUTWARD_COLUMN_SIGN = {PortSide.LEFT: -1, PortSide.RIGHT: 1}


def is_stacked_same_side_half_turn(
    exit_port: Port,
    entry_port: Port,
    feeder: Section,
    consumer: Section,
    *,
    via_junction: bool,
) -> bool:
    """Whether same-facing side ports form a cross-row outer half-turn.

    The consumer must sit no further back than the feeder's column in the
    direction the shared side faces, so the run leads out into the outer margin
    and doubles back into the same-facing port instead of carrying straight on.
    """
    outward = _OUTWARD_COLUMN_SIGN.get(exit_port.side)
    if via_junction or outward is None or entry_port.side is not exit_port.side:
        return False
    return (
        feeder.grid_row != consumer.grid_row
        and (consumer.grid_col - feeder.grid_col) * outward >= 0
    )


def entry_fan_receives_stacked_reversed_bundle(
    graph: MetroGraph,
    section: Section,
) -> bool:
    """Whether one direct stacked half-turn feeds distinct internal branches.

    Either same-facing half-turn qualifies: both hand the consumer a bundle
    whose lane order is mirrored, so the branch tracks it fans into have to be
    mirrored with it or the branches cross on their way to their stations.
    """
    if not lanes_run_along_y(section.direction) or len(section.entry_ports) != 1:
        return False
    entry_port_id = section.entry_ports[0]
    entry_port = graph.ports.get(entry_port_id)
    if entry_port is None:
        return False

    targets_by_line: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges_from(entry_port_id):
        target = graph.stations.get(edge.target)
        if (
            target is not None
            and not target.is_port
            and target.section_id == section.id
        ):
            targets_by_line[edge.line_id].add(edge.target)

    exit_ports = {
        edge.source: source
        for edge in graph.edges_to(entry_port_id)
        if (source := graph.ports.get(edge.source)) is not None and not source.is_entry
    }
    if len(exit_ports) != 1:
        return False
    incoming_lines = {
        edge.line_id
        for edge in graph.edges_to(entry_port_id)
        if edge.source in exit_ports
    }
    branch_lines = incoming_lines & targets_by_line.keys()
    if (
        len(branch_lines) < 2
        or len({target for line in branch_lines for target in targets_by_line[line]})
        < 2
    ):
        return False

    exit_port = next(iter(exit_ports.values()))
    return is_stacked_same_side_half_turn(
        exit_port,
        entry_port,
        graph.section_for_port(exit_port),
        section,
        via_junction=False,
    )
