"""A port keeps clearance from the box edges it is not anchored to.

Every port is pinned to one edge of its section's bounding box: a LEFT/RIGHT
port to a vertical edge, a TOP/BOTTOM port to a horizontal one.  Along its
*other* axis the port is free, and landing it flush on a second edge draws the
inbound run along the box border, where the route and the border read as one
stroke.  It also starves the section header: with a route on the top edge the
above-left position is never clear, so the placement chain falls through to
``nudge`` and the number badge slides right, away from the box it labels.

``_compact_row_content_to_bbox_top`` (Stage 5.4) is what pulls a perpendicular
port toward the top edge, and it reserves ``PERP_PORT_EDGE_CLEARANCE`` so the
port stops short of it.

Covers:

* Corpus: no shipped fixture seats a port flush on an unanchored box edge.
* Meaningfulness: a hand-planted flush port is caught, so the corpus check is
  not vacuous.
* Regression: ``tb_exit_terminal_on_carrier``'s carrier row keeps entry-port
  headroom and anchors its header at the box's top-left corner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import PERP_PORT_EDGE_CLEARANCE, SAME_COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.guards import (
    PhaseInvariantError,
    _guard_ports_clear_unanchored_box_edges,
)
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide
from nf_metro.render.constants import SECTION_NUM_CIRCLE_R_LARGE
from nf_metro.render.section_header import resolve_all_section_headers

REPO_ROOT = Path(__file__).resolve().parent.parent
CARRIER_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd"

EPSILON = SAME_COORD_TOLERANCE
"""Sub-pixel slack, shared with the runtime guard so the two agree on the floor."""


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    for rel in (
        "tests/fixtures/topologies",
        "tests/fixtures",
        "examples",
        "examples/topologies",
    ):
        paths.extend(sorted((REPO_ROOT / rel).glob("*.mmd")))
    return paths


def _unanchored_edge_clearances(graph) -> list[tuple[str, str, float]]:
    """``(section_id, port_id, clearance)`` for every port, measured to the
    nearer of the two box edges the port is *not* anchored to."""
    out: list[tuple[str, str, float]] = []
    for sec in graph.sections.values():
        if sec.bbox_w <= 0 or sec.bbox_h <= 0:
            continue
        for pid in (*sec.entry_ports, *sec.exit_ports):
            port = graph.ports.get(pid)
            station = graph.stations.get(pid)
            if port is None or station is None:
                continue
            if port.side in (PortSide.LEFT, PortSide.RIGHT):
                near = min(
                    station.y - sec.bbox_y,
                    sec.bbox_y + sec.bbox_h - station.y,
                )
            else:
                near = min(
                    station.x - sec.bbox_x,
                    sec.bbox_x + sec.bbox_w - station.x,
                )
            out.append((sec.id, pid, near))
    return out


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_ports_clear_unanchored_box_edges(path: Path) -> None:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    flush = [
        (sid, pid, clear)
        for sid, pid, clear in _unanchored_edge_clearances(graph)
        if clear < PERP_PORT_EDGE_CLEARANCE - EPSILON
    ]
    assert not flush, (
        "ports sit within "
        f"{PERP_PORT_EDGE_CLEARANCE}px of a box edge they are not anchored to: "
        + ", ".join(f"{sid}/{pid} clearance={clear:.1f}" for sid, pid, clear in flush)
    )


def test_guard_catches_planted_flush_port() -> None:
    """The runtime guard is not vacuous: seating a port on the top edge trips it."""
    graph = parse_metro_mermaid(CARRIER_FIXTURE.read_text())
    compute_layout(graph)
    section = graph.sections["quantification"]
    pid = next(iter(section.entry_ports))
    graph.stations[pid].y = section.bbox_y
    graph.ports[pid].y = section.bbox_y

    with pytest.raises(PhaseInvariantError, match="clearance"):
        _guard_ports_clear_unanchored_box_edges(graph, "planted")


def test_carrier_row_entry_port_keeps_headroom() -> None:
    graph = parse_metro_mermaid(CARRIER_FIXTURE.read_text())
    compute_layout(graph)
    section = graph.sections["quantification"]
    pid = next(iter(section.entry_ports))
    clearance = graph.stations[pid].y - section.bbox_y
    assert clearance >= PERP_PORT_EDGE_CLEARANCE - EPSILON, (
        f"{pid} sits {clearance:.1f}px below its box top; the inbound bundle "
        "rides the border"
    )


def test_carrier_row_header_anchors_at_box_corner() -> None:
    """With the entry port clear of the top edge the header needs no nudge."""
    graph = parse_metro_mermaid(CARRIER_FIXTURE.read_text())
    compute_layout(graph)
    polylines = [
        rp.points
        for rp in route_edges_centred(
            graph, station_offsets=compute_station_offsets(graph)
        )
    ]
    placement = resolve_all_section_headers(graph, 14.0, polylines, 24.0)[
        "quantification"
    ]
    section = graph.sections["quantification"]
    assert placement.mode == "above"
    assert placement.badge_cx == pytest.approx(
        section.bbox_x + SECTION_NUM_CIRCLE_R_LARGE, abs=EPSILON
    )
