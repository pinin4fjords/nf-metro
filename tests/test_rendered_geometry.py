"""Geometry instruments consume the same RenderPlan the emitter drew."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pytest
from conftest import parse_and_layout
from layout_metrics import measured_geometry

from nf_metro.layout.routing.common import RoutedPath, apply_route_offsets
from nf_metro.render import build_render_plan, emit_render_plan
from nf_metro.render.validate import _expected_line_segments, parse_route_polylines
from nf_metro.themes import THEMES

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FIXTURES = [
    EXAMPLES / "sarek_metro.mmd",
    EXAMPLES / "diagonal_labels.mmd",
    EXAMPLES / "longread_variant_calling.mmd",
    EXAMPLES / "guide" / "04_directions.mmd",
    EXAMPLES / "topologies" / "bypass_leftward_far_side_entry.mmd",
    EXAMPLES / "simple_pipeline.mmd",
    EXAMPLES / "rnaseq_auto.mmd",
]
_PRECISION = 3


def _render(path: Path):
    graph = parse_and_layout(path.read_text())
    theme = THEMES[graph.style if graph.style in THEMES else "nfcore"]
    plan = build_render_plan(graph, theme)
    return graph, plan, emit_render_plan(plan, theme)


def _vertices_by_line(
    runs: Iterable[tuple[str, Iterable[tuple[float, float]]]],
) -> dict[str, set[tuple[float, float]]]:
    out: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for line_id, points in runs:
        out[line_id].update(
            (round(x, _PRECISION), round(y, _PRECISION)) for x, y in points
        )
    return out


def _ink_vertices(svg: str) -> dict[str, set[tuple[float, float]]]:
    return _vertices_by_line(
        (line_id, subpath)
        for line_id, subpaths in parse_route_polylines(svg)
        for subpath in subpaths
    )


def _route_vertices(
    offsets: dict[tuple[str, str], float], routes: list[RoutedPath]
) -> dict[str, set[tuple[float, float]]]:
    return _vertices_by_line(
        (route.line_id, apply_route_offsets(route, offsets)) for route in routes
    )


def _segment_vertices(segments):
    return _vertices_by_line(
        (line_id, segment) for line_id, values in segments.items() for segment in values
    )


def _assert_contained(measured, svg: str) -> None:
    ink = _ink_vertices(svg)
    stray = {
        line_id: sorted(vertices - ink[line_id])
        for line_id, vertices in measured.items()
        if vertices - ink[line_id]
    }
    assert not stray


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_scorecard_measures_emitted_plan(path: Path) -> None:
    graph, plan, svg = _render(path)
    _assert_contained(_route_vertices(*measured_geometry(graph, plan)), svg)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_offset_oracle_measures_emitted_plan(path: Path) -> None:
    _graph, plan, svg = _render(path)
    _assert_contained(_segment_vertices(_expected_line_segments(plan)), svg)


def test_plan_geometry_is_independent_of_source_graph() -> None:
    graph, plan, _svg = _render(EXAMPLES / "sarek_metro.mmd")
    before = plan.offset_polylines()
    next(iter(graph.stations.values())).x += 1000
    assert plan.offset_polylines() == before


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_reading_plan_geometry_is_pure(path: Path) -> None:
    graph, plan, _svg = _render(path)
    before = {station_id: (s.x, s.y) for station_id, s in graph.stations.items()}
    offsets, _ = measured_geometry(graph, plan)
    _expected_line_segments(plan)
    assert {
        station_id: (s.x, s.y) for station_id, s in graph.stations.items()
    } == before
    assert measured_geometry(graph, plan)[0] == offsets
