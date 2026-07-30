"""Contracts at the settled-geometry to artifact-emission boundary."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.parser.model import MetroGraph
from nf_metro.render.html import emit_render_plan_html
from nf_metro.render.manifest import read_manifest
from nf_metro.render.plan import (
    RenderPlan,
    contains_mutable_model_reference,
    thaw_render_value,
)
from nf_metro.render.svg import build_render_plan, emit_render_plan, render_svg

ROOT = Path(__file__).parents[1]


def _plan(path: str) -> tuple[MetroGraph, RenderPlan]:
    source = ROOT / path
    graph = prepare_graph(source.read_text(), source_dir=str(source.parent))
    theme = resolve_theme(None, graph)
    return graph, build_render_plan(graph, theme)


@pytest.mark.parametrize(
    ("path", "station_count", "route_count", "dimensions"),
    [
        ("examples/guide/01_minimal.mmd", 6, 7, (488, 286)),
        ("examples/topologies/divergent_fanout_split.mmd", 9, 8, (640, 422)),
        ("examples/guide/03b_fan_in_merge.mmd", 18, 25, (880, 325)),
        ("examples/topologies/fold_double.mmd", 55, 108, (1537, 716)),
        (
            "examples/topologies/lr_perp_top_entry_bottom_exit.mmd",
            9,
            8,
            (373, 628),
        ),
        ("examples/rail_mode.mmd", 11, 32, (531, 965)),
        ("examples/file_icons.mmd", 9, 8, (540, 385)),
    ],
)
def test_representative_plan_geometry_snapshot(
    path: str,
    station_count: int,
    route_count: int,
    dimensions: tuple[int, int],
) -> None:
    _, plan = _plan(path)

    assert len(plan.graph.stations) == station_count
    assert len(plan.routes) == route_count
    assert (plan.svg_width, plan.svg_height) == dimensions
    assert all(isinstance(route.points, tuple) for route in plan.routes)


def test_render_plan_is_deeply_immutable_and_model_free() -> None:
    _, plan = _plan("examples/rail_mode.mmd")

    assert not contains_mutable_model_reference(plan)
    with pytest.raises(FrozenInstanceError):
        plan.svg_width = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.graph.stations["new"] = object()  # type: ignore[index]


def test_repeated_plan_emission_is_byte_identical() -> None:
    graph, plan = _plan("examples/topologies/fold_double.mmd")
    theme = resolve_theme(None, graph)
    before = repr(plan)

    first = emit_render_plan(plan, theme)
    second = emit_render_plan(plan, theme)

    assert first == second
    assert repr(plan) == before


def test_bridge_gaps_are_settled_in_plan() -> None:
    _, plan = _plan("examples/topologies/self_crossing_bridge.mmd")

    assert any(plan.bridge_breaks)
    assert len(plan.bridge_breaks) == len(plan.edge_routes)


def test_repeated_html_emission_is_byte_identical() -> None:
    graph, plan = _plan("examples/guide/01_minimal.mmd")
    theme = resolve_theme(None, graph)

    assert emit_render_plan_html(plan, theme) == emit_render_plan_html(plan, theme)


def test_rendering_does_not_mutate_prepared_graph() -> None:
    source = ROOT / "examples" / "sarek_metro.mmd"
    graph = prepare_graph(source.read_text(), source_dir=str(source.parent))
    theme = resolve_theme(None, graph)
    before = copy.deepcopy(graph)

    render_svg(graph, theme)

    assert graph == before


def test_manifest_geometry_is_frozen_in_plan() -> None:
    graph, plan = _plan("examples/guide/01_minimal.mmd")
    theme = resolve_theme(None, graph)
    expected = thaw_render_value(plan.manifest)
    assert expected is not None

    graph.stations["fastqc"].x += 1000
    emitted = read_manifest(emit_render_plan(plan, theme))

    assert emitted == expected
    nodes = {node["id"]: node for node in emitted["nodes"]}
    assert nodes["fastqc"]["x"] == plan.graph.stations["fastqc"].x
