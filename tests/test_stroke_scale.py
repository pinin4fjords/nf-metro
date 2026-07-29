"""``stroke_scale``: coarsened ink, and the pitch that has to track it."""

from __future__ import annotations

import pytest

from nf_metro.api import RenderConfig, prepare_graph, render_graph, resolve_theme
from nf_metro.layout.constants import (
    DEFAULT_LINE_WIDTH,
    OFFSET_STEP,
    graph_offset_step,
    resolve_offset_step,
)
from nf_metro.parser import parse_metro_mermaid
from nf_metro.render.svg import _scale_theme_strokes

_SRC = """%%metro title: Bundle
%%metro line: a | A | #e41a1c
%%metro line: b | B | #377eb8
%%metro line: c | C | #4daf4a

graph LR
    in[Input] -->|a,b,c| split[Split]
    split -->|a| one[One]
    split -->|b| two[Two]
    split -->|c| three[Three]
"""


def _graph(**opts: object):
    return prepare_graph(_SRC, layout_options=opts)


@pytest.mark.parametrize("track_gap", [None, 0.0, 1.0, 2.5])
def test_offset_step_unchanged_at_unit_scale(track_gap: float | None) -> None:
    """Unit scale must reproduce the unscaled pitch on both resolution paths.

    The whole corpus rendering byte-identically by default rests on this: a
    graph that never sets ``stroke_scale`` has to land on exactly the pitch
    ``resolve_offset_step`` yields for the same inputs.
    """
    graph = parse_metro_mermaid("graph LR\n")
    graph.track_gap = track_gap

    assert graph_offset_step(graph) == resolve_offset_step(track_gap)
    # The render path passes its theme's drawn width; unit scale leaves it be.
    assert graph_offset_step(graph, 4.0) == resolve_offset_step(track_gap, 4.0)


@pytest.mark.parametrize("scale", [1.3, 1.6, 2.0])
def test_pitch_scales_with_stroke(scale: float) -> None:
    """Gap and stroke coarsen together, so bundle lines stay separable.

    Scaling the stroke alone would hold the gap at its absolute default and
    close it up under the downscale the option exists to survive.
    """
    graph = parse_metro_mermaid("graph LR\n")

    graph.track_gap = None
    graph.stroke_scale = scale
    assert graph_offset_step(graph) == pytest.approx(OFFSET_STEP * scale)

    graph.track_gap = 2.0
    assert graph_offset_step(graph) == pytest.approx((2.0 + DEFAULT_LINE_WIDTH) * scale)


def test_pitch_clears_the_drawn_stroke() -> None:
    """The reserved pitch must stay wider than the stroke painted in it.

    A pitch that fell below the drawn width would paint adjacent lines of a
    bundle over each other.
    """
    graph = parse_metro_mermaid("graph LR\n")
    for scale in (1.0, 1.3, 1.6, 2.0, 3.0):
        graph.stroke_scale = scale
        for base_width in (3.0, 4.0):
            drawn = base_width * scale
            for gap in (None, 0.0, 1.0, 2.0):
                graph.track_gap = gap
                assert graph_offset_step(graph, drawn) >= drawn


def test_scaled_theme_coarsens_strokes_but_not_marker_radius() -> None:
    """Marker radius is excluded: layout reserves clearance against a fixed one."""
    graph = _graph()
    theme = resolve_theme(None, graph)
    scaled = _scale_theme_strokes(theme, 2.0)

    assert scaled.line_width == pytest.approx(theme.line_width * 2.0)
    assert scaled.station_stroke_width == pytest.approx(
        theme.station_stroke_width * 2.0
    )
    assert scaled.label_halo_width == pytest.approx(theme.label_halo_width * 2.0)
    assert scaled.station_radius == theme.station_radius


def test_unit_scale_returns_the_same_theme() -> None:
    theme = resolve_theme(None, _graph())
    assert _scale_theme_strokes(theme, 1.0) is theme


def test_render_emits_coarser_tracks() -> None:
    """The scale reaches the drawn SVG, not just the layout reservation."""

    plain = render_graph(_graph(), resolve_theme(None, _graph()), RenderConfig())
    coarse_graph = _graph(stroke_scale=2.0)
    coarse = render_graph(
        coarse_graph, resolve_theme(None, coarse_graph), RenderConfig()
    )
    assert plain != coarse
    assert 'stroke-width="6' in coarse
