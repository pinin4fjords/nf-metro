"""The geometry instruments read is the geometry the renderer drew (#1590).

Routing consults section bounding boxes, and label placement grows those boxes
*after* the routes are settled.  Re-routing a rendered graph therefore routes
against grown boxes and yields paths the viewer never saw -- on the corpus the
divergence reaches 24px.  Anything that claims to measure or guard the render
(the render-diff quality scorecard, the offset-collapse oracle) must read the
geometry the renderer published rather than re-derive it.

The oracle here is the drawn SVG ink: ``parse_route_polylines`` recovers each
route's logical vertices exactly (smoothing ``Q`` control points collapse back
to their pre-smoothing corner), so every vertex of the measured geometry must
appear in the ink of its line.  Bridge hop gaps and corner smoothing only add
vertices, never move them, which is why containment is the exact relation and
not an approximation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from conftest import parse_and_layout
from layout_metrics import measured_geometry

from nf_metro.layout.routing.common import RoutedPath, apply_route_offsets
from nf_metro.render import render_svg
from nf_metro.render.validate import _expected_line_segments, parse_route_polylines
from nf_metro.themes import THEMES

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Fixtures whose render mutates a section bbox enough to move a re-routed path
# (sarek_metro by 24px, the widest on the corpus), alongside clean maps where the
# two agreed all along.
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
    """Lay out and render *path*, returning ``(graph, svg)``."""
    graph = parse_and_layout(path.read_text())
    theme = graph.style if graph.style in THEMES else "nfcore"
    return graph, render_svg(graph, THEMES[theme])


def _ink_vertices(svg: str) -> dict[str, set[tuple[float, float]]]:
    """Every drawn route vertex in the SVG, grouped by line id."""
    ink: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for line_id, subpaths in parse_route_polylines(svg):
        for subpath in subpaths:
            ink[line_id].update(
                (round(x, _PRECISION), round(y, _PRECISION)) for x, y in subpath
            )
    return ink


def _route_vertices(
    offsets: dict[tuple[str, str], float], routes: list[RoutedPath]
) -> dict[str, set[tuple[float, float]]]:
    """Every post-offset route vertex, grouped by line id."""
    out: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for route in routes:
        out[route.line_id].update(
            (round(x, _PRECISION), round(y, _PRECISION))
            for x, y in apply_route_offsets(route, offsets)
        )
    return out


def _segment_vertices(
    segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
) -> dict[str, set[tuple[float, float]]]:
    """Every endpoint of the offset-collapse oracle's segments, by line id."""
    out: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for line_id, segs in segments.items():
        for a, b in segs:
            out[line_id].update(
                (round(p[0], _PRECISION), round(p[1], _PRECISION)) for p in (a, b)
            )
    return out


def _assert_contained_in_ink(
    measured: dict[str, set[tuple[float, float]]], svg: str, what: str
) -> None:
    ink = _ink_vertices(svg)
    stray = {
        line_id: sorted(verts - ink[line_id])
        for line_id, verts in measured.items()
        if verts - ink[line_id]
    }
    assert not stray, (
        f"{what} reads {sum(len(v) for v in stray.values())} vertex/vertices the "
        f"render never drew: "
        + "; ".join(f"line {ln!r} at {pts[:3]}" for ln, pts in sorted(stray.items()))
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_scorecard_measures_the_drawn_geometry(path: Path) -> None:
    """Every route vertex the quality scorecard scores was actually drawn."""
    graph, svg = _render(path)
    _assert_contained_in_ink(
        _route_vertices(*measured_geometry(graph)), svg, "the quality scorecard"
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_offset_collapse_oracle_reads_the_drawn_geometry(path: Path) -> None:
    """The offset-collapse oracle's reference segments were actually drawn.

    It compares the drawn ink against the separation the offset regime assigned,
    so a reference derived from paths the renderer never emitted would report
    collapses (and clear real ones) against a picture that does not exist.
    """
    graph, svg = _render(path)
    _assert_contained_in_ink(
        _segment_vertices(_expected_line_segments(graph)),
        svg,
        "the offset-collapse oracle",
    )


def test_offset_collapse_needs_a_rendered_graph() -> None:
    """The oracle abstains on a graph that was laid out but never rendered.

    It has no drawn geometry to compare the ink against, and re-deriving one is
    exactly the defect; abstaining is deliberate, not an accident of the lookup.
    """
    graph = parse_and_layout((EXAMPLES / "sarek_metro.mmd").read_text())
    assert graph.rendered_geometry is None
    assert _expected_line_segments(graph) == {}

    render_svg(graph, THEMES["nfcore"])
    assert graph.rendered_geometry is not None
    assert _expected_line_segments(graph)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_reading_the_geometry_does_not_perturb_the_render(path: Path) -> None:
    """Measuring leaves station coordinates and offsets as the render left them.

    An instrument that settled bubble-centred markers onto ``graph.stations``
    would move the very geometry the next reader measures, so reads must stay
    free of side effects.
    """
    graph, _ = _render(path)
    before = {sid: (s.x, s.y) for sid, s in graph.stations.items()}

    offsets, _ = measured_geometry(graph)
    _expected_line_segments(graph)

    assert {sid: (s.x, s.y) for sid, s in graph.stations.items()} == before
    assert measured_geometry(graph)[0] == offsets
