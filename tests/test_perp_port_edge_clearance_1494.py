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
port stops short of it.  The rotated case -- a horizontal flow's TOP/BOTTOM port
approaching a vertical edge -- is held by ``_reserve_perp_port_edge_inset``
(Stage 3.5), which grows the edge instead.

Covers:

* Corpus: no shipped fixture seats a port flush on an unanchored box edge, and
  every horizontal-flow perpendicular port keeps the full designed inset from
  both vertical edges.
* Meaningfulness: a hand-planted flush port is caught, so the corpus check is
  not vacuous.
* Regression: ``tb_exit_terminal_on_carrier``'s carrier row keeps entry-port
  headroom and anchors its header at the box's top-left corner.
* Rotation: both axes read their perpendicular pair off the flow axis, and a
  pair on either axis ends up equidistant from the edges it faces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import (
    PERP_PORT_EDGE_CLEARANCE,
    PERP_PORT_EDGE_INSET,
    SAME_COORD_TOLERANCE,
    SECTION_Y_PADDING,
)
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.geometry import AxisFrame, perpendicular_port_sides
from nf_metro.layout.phases.guards import (
    PhaseInvariantError,
    _guard_ports_clear_unanchored_box_edges,
)
from nf_metro.layout.phases.row_align import _perp_port_lead_edge_reserve
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, Port, PortSide, Section, Station
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


def test_perpendicular_port_sides_follow_the_flow_axis() -> None:
    """The perpendicular pair is read off the lane axis, for every rotation.

    The corpus has no LR/RL section carrying both a TOP and a BOTTOM port and no
    BT section at all, so the X-axis and reversed-flow arms of the rule are
    pinned here rather than by a fixture.
    """
    assert perpendicular_port_sides("TB") == (PortSide.LEFT, PortSide.RIGHT)
    assert perpendicular_port_sides("BT") == (PortSide.LEFT, PortSide.RIGHT)
    assert perpendicular_port_sides("LR") == (PortSide.TOP, PortSide.BOTTOM)
    assert perpendicular_port_sides("RL") == (PortSide.TOP, PortSide.BOTTOM)


def test_carrier_row_ports_sit_symmetrically_in_their_box() -> None:
    """The two perpendicular ports end equidistant from the edges they face."""
    graph = parse_metro_mermaid(CARRIER_FIXTURE.read_text())
    compute_layout(graph)
    section = graph.sections["quantification"]
    entry = graph.stations[next(iter(section.entry_ports))]
    exit_ = graph.stations[next(iter(section.exit_ports))]
    top_clearance = entry.y - section.bbox_y
    bottom_clearance = section.bbox_y + section.bbox_h - exit_.y
    assert top_clearance == pytest.approx(bottom_clearance, abs=EPSILON), (
        f"entry sits {top_clearance:.1f}px below the box top but the exit sits "
        f"{bottom_clearance:.1f}px above its bottom"
    )


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_lone_perpendicular_port_reserves_only_the_floor(path: Path) -> None:
    """A section with no perpendicular pair falls back to the designed inset."""
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    for section in graph.sections.values():
        if section.bbox_h <= 0:
            continue
        sides = perpendicular_port_sides(section.direction or "LR")
        perp = [
            pid
            for pid in (*section.entry_ports, *section.exit_ports)
            if (port := graph.ports.get(pid)) is not None and port.side in sides
        ]
        if len(perp) >= 2:
            continue
        assert _perp_port_lead_edge_reserve(
            graph, section, SECTION_Y_PADDING
        ) == pytest.approx(PERP_PORT_EDGE_INSET, abs=EPSILON)


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_horizontal_perp_ports_keep_the_designed_inset(path: Path) -> None:
    """Every LR/RL perpendicular port keeps the full inset from both side edges.

    A corpus ratchet, not a runtime guard.  A render only aborts below the much
    smaller ``PERP_PORT_EDGE_CLEARANCE``, which leaves a novel map free to land
    a port short of the designed inset; the shipped corpus is held to it here.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    tight: list[str] = []
    for section in graph.sections.values():
        direction = section.direction or "LR"
        if section.bbox_w <= 0 or AxisFrame.axes_for_direction(direction)[0] != "x":
            continue
        for pid in (*section.entry_ports, *section.exit_ports):
            port = graph.ports.get(pid)
            station = graph.stations.get(pid)
            if (
                port is None
                or station is None
                or port.side not in perpendicular_port_sides(direction)
            ):
                continue
            near = min(
                station.x - section.bbox_x,
                section.bbox_x + section.bbox_w - station.x,
            )
            if near < PERP_PORT_EDGE_INSET - EPSILON:
                tight.append(f"{section.id}/{pid} clearance={near:.1f}")
    assert not tight, (
        f"perpendicular ports sit within {PERP_PORT_EDGE_INSET}px of a vertical "
        "box edge: " + ", ".join(tight)
    )


def _lr_section_with_perp_pair(entry_x: float, exit_x: float) -> MetroGraph:
    """One LR section spanning x in [0, 300] with a TOP entry and a BOTTOM exit.

    The rotation of the vertical-flow carrier row: both ports are perpendicular
    to the flow, so both are free along X and clear the left and right edges.
    """
    graph = MetroGraph()
    section = Section(id="s", name="s", direction="LR")
    section.bbox_x, section.bbox_y, section.bbox_w, section.bbox_h = (
        0.0,
        0.0,
        300.0,
        100.0,
    )
    section.station_ids = ["c", "pin", "pout"]
    section.entry_ports = ["pin"]
    section.exit_ports = ["pout"]
    graph.sections["s"] = section
    graph.stations["c"] = Station(id="c", label="C", section_id="s", x=150.0, y=50.0)
    for pid, px, side, is_entry in (
        ("pin", entry_x, PortSide.TOP, True),
        ("pout", exit_x, PortSide.BOTTOM, False),
    ):
        graph.stations[pid] = Station(
            id=pid, label="", section_id="s", is_port=True, x=px, y=0.0
        )
        graph.ports[pid] = Port(
            id=pid, section_id="s", side=side, is_entry=is_entry, x=px, y=0.0
        )
    return graph


def test_lead_edge_reserve_measures_on_x_for_a_horizontal_flow() -> None:
    """The rotated shape resolves through the same rule, measured along X.

    No corpus fixture carries both a TOP and a BOTTOM port on an LR/RL section,
    so this pins the X arm directly: the reserve must track the *right*-edge
    clearance of the rightmost perpendicular port, and must ignore Y entirely.
    """
    graph = _lr_section_with_perp_pair(entry_x=40.0, exit_x=260.0)
    section = graph.sections["s"]
    # Right edge lands at content_x + padding = 150 + 50 = 200, but the rightmost
    # port at x=260 pushes it out to 260, leaving that port zero right-clearance,
    # so the floor governs.
    assert _perp_port_lead_edge_reserve(
        graph, section, SECTION_Y_PADDING
    ) == pytest.approx(PERP_PORT_EDGE_INSET, abs=EPSILON)

    # Pull the rightmost port inside the padded right edge and the reserve becomes
    # that port's own right-edge clearance: 200 - 170 = 30.
    graph.stations["pout"].x = 150.0
    graph.ports["pout"].x = 150.0
    assert _perp_port_lead_edge_reserve(
        graph, section, SECTION_Y_PADDING
    ) == pytest.approx(50.0, abs=EPSILON)

    # Moving a port along Y cannot change an X-axis reserve.
    graph.stations["pout"].y = 100.0
    graph.ports["pout"].y = 100.0
    assert _perp_port_lead_edge_reserve(
        graph, section, SECTION_Y_PADDING
    ) == pytest.approx(50.0, abs=EPSILON)


def _row_group_slacks(graph, section) -> list[float]:
    """Above-content slack for each of ``section``'s contiguous row-mates.

    Mirrors what ``_compact_row_content_to_bbox_top`` computes per section before
    taking the group minimum: how far content could rise before it would breach
    ``SECTION_Y_PADDING`` under the box top.
    """
    if AxisFrame.axes_for_direction(section.direction or "LR")[0] != "y":
        return []
    slacks = []
    for mate in graph.sections.values():
        if (
            mate.bbox_h <= 0
            or mate.grid_row != section.grid_row
            or abs(mate.grid_col - section.grid_col) > 1
        ):
            continue
        content = [
            graph.stations[sid].y
            for sid in mate.station_ids
            if sid in graph.stations and not graph.stations[sid].is_port
        ]
        if content:
            slacks.append(min(content) - mate.bbox_y - SECTION_Y_PADDING)
    return slacks


def _perp_pair_clearances(graph, section, axis):
    """``(low, high)`` clearance of the outermost perpendicular ports, or None."""
    sides = perpendicular_port_sides(section.direction or "LR")
    coords = [
        (graph.stations[pid].y if axis == "y" else graph.stations[pid].x)
        for pid in (*section.entry_ports, *section.exit_ports)
        if (port := graph.ports.get(pid)) is not None
        and port.side in sides
        and pid in graph.stations
    ]
    if len(coords) < 2:
        return None
    low = section.bbox_y if axis == "y" else section.bbox_x
    high = low + (section.bbox_h if axis == "y" else section.bbox_w)
    return min(coords) - low, high - max(coords)


def _iter_perp_pairs(graph, want_axis):
    for section in graph.sections.values():
        if section.bbox_h <= 0 or section.bbox_w <= 0:
            continue
        axis = AxisFrame.axes_for_direction(section.direction or "LR")[0]
        if axis != want_axis:
            continue
        pair = _perp_pair_clearances(graph, section, axis)
        if pair is not None:
            yield section, pair


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_vertical_perp_port_pair_is_balanced_or_its_row_is_blocked(path: Path) -> None:
    """A vertical flow's perpendicular pair is equidistant unless its row is stuck.

    Compaction shifts a whole grid row by one uniform delta so the row's shared
    trunk Y survives it, so the delta is the minimum slack across the row.  A
    row-mate with no slack pins it at zero and the reserve has nothing to act on;
    balancing one section alone would drag its trunk off the row.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    for section, (low, high) in _iter_perp_pairs(graph, "y"):
        if abs(low - high) < EPSILON:
            continue
        slacks = _row_group_slacks(graph, section)
        assert slacks and min(slacks) <= EPSILON, (
            f"{section.id!r} perpendicular ports are asymmetric "
            f"({low:.1f} vs {high:.1f}) yet its row group has slack "
            f"{min(slacks) if slacks else None}; the reserve should have "
            "balanced them"
        )


_LR_PERP_PAIR_MMD = """\
%%metro title: LR perpendicular pair (top entry, bottom exit)
%%metro line: dna | DNA | #e6007e
%%metro grid: intake | 0,0
%%metro grid: mid | 0,1
%%metro grid: report | 0,2

graph LR
    subgraph intake [Intake]
        %%metro exit: bottom | dna
        samplesheet[Samplesheet]
        fastqc[FastQC]
        samplesheet -->|dna| fastqc
    end

    subgraph mid [Alignment]
        %%metro entry: top | dna
        %%metro exit: bottom | dna
        bwa[BWA-MEM2]
        sort[Sort BAM]
        bwa -->|dna| sort
    end

    subgraph report [Reporting]
        %%metro entry: top | dna
        multiqc[MultiQC]
    end

    fastqc -->|dna| bwa
    sort -->|dna| multiqc
"""
"""An LR section carrying both perpendicular ports, so both are free along X.

Held inline rather than as a corpus fixture: the shape also trips
``test_no_line_folds_back_over_its_track`` and the placement-purity suites, so
shipping it under ``examples/topologies/`` would red CI on defects unrelated to
the clearance this module is about.
"""


def test_horizontal_perp_port_pair_is_balanced() -> None:
    """The rotation of the vertical rule: an LR/RL TOP+BOTTOM pair sits alike.

    Deliberately has no row-blocked escape.  The vertical excuse mirrors a Y-axis
    compaction pass with no column-wise counterpart, so borrowing it here would
    make this assertion unfailable -- which is how the gap stayed invisible.
    """
    graph = parse_metro_mermaid(_LR_PERP_PAIR_MMD)
    compute_layout(graph)
    pairs = list(_iter_perp_pairs(graph, "x"))
    assert pairs, "the inline map no longer produces an X-axis perpendicular pair"
    offenders = [
        f"{section.id}: {low:.1f} vs {high:.1f}"
        for section, (low, high) in pairs
        if abs(low - high) >= EPSILON
    ]
    assert not offenders, "; ".join(offenders)
