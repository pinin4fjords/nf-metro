"""Foreign section keep-outs for post-routing channel reseating."""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases._common import routes_through_unrelated_sections
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.layout.routing.normalize import (
    _reseated_segment_crosses_other_section,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge, MetroGraph, Section, Station

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("path", "center_ports"),
    (
        (ROOT / "examples" / "genomeassembly.mmd", False),
        (ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd", True),
    ),
    ids=("genomeassembly", "genomeassembly-organellar"),
)
def test_chained_assemblies_carrier_preserves_foreign_section_keepouts(
    path: Path, center_ports: bool
) -> None:
    graph = parse_metro_mermaid(path.read_text())
    graph.center_ports = center_ports
    compute_layout(graph, validate=False)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    carriers = [
        route
        for route in routes
        if route.line_id == "assemblies" and route.edge.target == "__merge_5"
    ]

    violations = [
        (route.edge.source, route.edge.target, section_id)
        for route, section_id in routes_through_unrelated_sections(
            graph, routes=routes, offsets=offsets
        )
        if route.line_id == "assemblies" and route.edge.target == "__merge_5"
    ]

    assert violations == []
    carrier_ys = {
        max(
            y1
            for (x1, y1), (x2, y2) in zip(route.points, route.points[1:])
            if abs(y2 - y1) <= 1.0 and abs(x2 - x1) > 1.0
        )
        for route in carriers
    }
    assert len(carrier_ys) == 1
    assert next(iter(carrier_ys)) > (
        graph.sections["scaffolding"].bbox_y + graph.sections["scaffolding"].bbox_h
    )


@pytest.mark.parametrize(
    ("points", "axis", "coordinate", "blocker_bbox"),
    (
        (
            [(0.0, 0.0), (20.0, 0.0), (20.0, 100.0), (180.0, 100.0)],
            1,
            50.0,
            (15.0, 20.0, 10.0, 10.0),
        ),
        (
            [(0.0, 0.0), (0.0, 20.0), (100.0, 20.0), (100.0, 180.0)],
            0,
            50.0,
            (20.0, 15.0, 10.0, 10.0),
        ),
    ),
    ids=("horizontal-carrier", "vertical-carrier"),
)
def test_reseat_checks_transposed_stretched_flank_keepout(
    points: list[tuple[float, float]],
    axis: int,
    coordinate: float,
    blocker_bbox: tuple[float, float, float, float],
) -> None:
    graph = MetroGraph(
        stations={
            "source": Station("source", "Source", section_id="source-section"),
            "target": Station("target", "Target", section_id="target-section"),
        },
        sections={
            "source-section": Section("source-section", "Source"),
            "target-section": Section("target-section", "Target"),
            "blocker": Section("blocker", "Blocker"),
        },
    )
    blocker = graph.sections["blocker"]
    blocker.bbox_x, blocker.bbox_y, blocker.bbox_w, blocker.bbox_h = blocker_bbox
    route = RoutedPath(
        edge=Edge("source", "target", "line"),
        line_id="line",
        points=points,
        is_inter_section=True,
    )

    assert _reseated_segment_crosses_other_section(
        graph, route, 2, coordinate, axis=axis
    )


def test_merge_right_entry_keeps_valid_same_line_overlay() -> None:
    path = ROOT / "examples" / "topologies" / "merge_right_entry.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=False)
    routes = route_edges(graph, station_offsets=compute_station_offsets(graph))

    through = next(
        route
        for route in routes
        if route.edge.source == "step_a__exit_right_2"
        and route.edge.target == "sink__entry_right_4"
    )
    feeder = next(
        route
        for route in routes
        if route.edge.source == "__junction_7" and route.edge.target == "__merge_2"
    )

    assert feeder.points[2][1] == pytest.approx(through.points[2][1])
    assert feeder.points[3] == pytest.approx(through.points[3])


def test_distinct_traverse_bundle_excludes_other_same_line_cohorts() -> None:
    path = ROOT / "examples" / "genomeassembly.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=False)
    routes = route_edges(graph, station_offsets=compute_station_offsets(graph))
    selected = {
        (route.line_id, route.edge.target): route
        for route in routes
        if route.edge.source == "__junction_8"
        and (route.line_id, route.edge.target)
        in {
            ("hic_reads", "scaffolding__entry_left_5"),
            ("assemblies", "__merge_3"),
            ("assemblies", "__merge_4"),
            ("assemblies", "__merge_5"),
        }
    }

    assert selected[("hic_reads", "scaffolding__entry_left_5")].points[2][1] == (
        pytest.approx(242.0)
    )
    assert selected[("assemblies", "__merge_3")].points[2][1] == pytest.approx(238.0)
    assert selected[("assemblies", "__merge_4")].points[2][1] == pytest.approx(195.0)
    assert selected[("assemblies", "__merge_5")].points[2][1] == pytest.approx(285.0)
    assert selected[("assemblies", "__merge_5")].points[2][1] > (
        graph.sections["scaffolding"].bbox_y + graph.sections["scaffolding"].bbox_h
    )
