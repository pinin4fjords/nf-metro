"""Invariant tests for a bundle crossing into a per-section rail-mode section.

A ``%%metro line_spread: rails | <section>`` section lays each of its lines on
its own horizontal rail, and a station several lines pass through renders as an
interchange pill spanning those rails.  A multi-line bundle arriving from an
adjacent bundled section must therefore separate on the way in so each line
meets the pill on *its own* rail, the way the lines leaving that pill each run
along their own rail to their downstream station.  Landing the whole bundle at
the pill's centre Y instead draws the incoming lines diving into the middle of
the pill body.

The oracle is the geometry the renderer published (``graph.rendered_geometry``
via :func:`layout_metrics.measured_geometry`): section bboxes grow during label
placement, so re-routing a laid-out graph yields paths the viewer never saw.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
from layout_metrics import drawn_polylines, measured_geometry

from nf_metro.layout import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph
from nf_metro.render import render_svg
from nf_metro.themes import THEMES

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOPOLOGIES = Path(__file__).resolve().parent.parent / "examples" / "topologies"

# Fixtures whose rail section is entered by an inter-section bundle.  A rail
# section with no edge crossing into it cannot exercise the boundary fan.
BOUNDARY_FIXTURES = [
    FIXTURES / "rail_marked_single_line.mmd",
    TOPOLOGIES / "rail_boundary_bundle_fan.mmd",
]


@lru_cache(maxsize=None)
def _rendered(path: Path) -> MetroGraph:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    render_svg(graph, THEMES[graph.style if graph.style in THEMES else "nfcore"])
    return graph


def _is_multi_rail_pill(graph: MetroGraph, station_id: str, line_id: str) -> bool:
    """Whether *station_id* draws a per-rail marker for *line_id*.

    An off-track feeder and a blank terminus converge their lines to a point
    rather than to one rail each, so neither offers a rail to land on.
    """
    station = graph.stations[station_id]
    if station.off_track or station.is_blank_terminus:
        return False
    if not graph.station_is_rail(station_id):
        return False
    served = graph.station_lines_ordered(station_id)
    return (
        len(served) > 1
        and line_id in served
        and len(station.rail_used_ys) == len(served)
    )


def _rail_y(graph: MetroGraph, station_id: str, line_id: str) -> float:
    """Y at which *line_id* meets the pill of rail station *station_id*.

    Reconstructed from the published span rather than by calling the router's
    own resolver, so the expectation is independent of the code under test.
    """
    station = graph.stations[station_id]
    served = graph.station_lines_ordered(station_id)
    return station.rail_used_ys[served.index(line_id)]


def _boundary_approaches(graph: MetroGraph) -> list[tuple[str, str, float, float]]:
    """``(station_id, line_id, approach_y, rail_y)`` for each entering leg.

    An "entering leg" is a drawn edge between a section port and a rail-laid
    interchange pill: the last leg a line travels before it meets the pill (or
    the first leg after it leaves).
    """
    offsets, routes = measured_geometry(graph)
    out = []
    for route, points in drawn_polylines(routes, offsets):
        source = graph.stations[route.edge.source]
        target = graph.stations[route.edge.target]
        for port_end, station_end, point in (
            (source, target, points[-1]),
            (target, source, points[0]),
        ):
            if not port_end.is_port or station_end.is_port:
                continue
            if not _is_multi_rail_pill(graph, station_end.id, route.line_id):
                continue
            out.append(
                (
                    station_end.id,
                    route.line_id,
                    point[1],
                    _rail_y(graph, station_end.id, route.line_id),
                )
            )
    return out


@pytest.mark.parametrize("path", BOUNDARY_FIXTURES, ids=lambda p: p.stem)
def test_bundle_entering_rail_section_lands_on_each_lines_rail(path: Path) -> None:
    graph = _rendered(path)

    approaches = _boundary_approaches(graph)
    assert approaches, f"{path.name} has no port-to-rail-station leg to check"

    offenders = [
        (sid, lid, got, want)
        for sid, lid, got, want in approaches
        if abs(got - want) > 1.0
    ]
    assert not offenders, (
        "line(s) meet a rail station away from their own rail: "
        + ", ".join(
            f"{sid}/{lid} at y={got:.1f} (rail y={want:.1f})"
            for sid, lid, got, want in offenders
        )
    )


@pytest.mark.parametrize("path", BOUNDARY_FIXTURES, ids=lambda p: p.stem)
def test_bundle_entering_rail_section_spans_its_rails(path: Path) -> None:
    """The entering bundle is spread across rails, not stacked in one lane.

    Complements the per-line rail check, which one member of a lane-packed
    bundle can satisfy by coincidence when its lane happens to fall on its rail.
    """
    graph = _rendered(path)

    by_station: dict[str, list[float]] = {}
    for sid, _lid, got, _want in _boundary_approaches(graph):
        by_station.setdefault(sid, []).append(got)
    assert by_station, f"{path.name} has no port-to-rail-pill leg to check"

    for sid, ys in by_station.items():
        station = graph.stations[sid]
        rail_span = max(station.rail_used_ys) - min(station.rail_used_ys)
        got_span = max(ys) - min(ys)
        assert got_span >= rail_span - 1.0, (
            f"{sid}: entering bundle spans {got_span:.1f}px but its rails "
            f"span {rail_span:.1f}px - the lines are bunched, not fanned"
        )
