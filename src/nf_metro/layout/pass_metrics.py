"""Pass-scoped scaling of the render metrics layout reserves against.

The layout engine reserves room using theme-agnostic proxies -- ``CHAR_WIDTH``
and ``FONT_HEIGHT`` for text, :data:`~nf_metro.layout.constants.
STATION_RADIUS_APPROX` and :data:`~nf_metro.layout.constants.
STATION_STROKE_APPROX` for station markers.  The ``font_scale`` and
``stroke_scale`` directives multiply what the renderer actually draws, so those
proxies have to be read through the same multiplier or a scaled render collides
with reservations made for unscaled ink.

The scale is ambient for the duration of a layout or render pass rather than a
parameter because the readers are geometry helpers taking a ``Station`` or a
bare coordinate, far from any caller holding the ``MetroGraph``.  Wrap a pass in
:func:`font_scale_context` / :func:`stroke_scale_context` and read through the
accessors here.
"""

from __future__ import annotations

__all__ = [
    "active_font_scale",
    "active_stroke_scale",
    "canvas_edge_clearance",
    "font_scale_context",
    "station_radius_approx",
    "station_stroke_approx",
    "stroke_scale_context",
]

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from nf_metro.layout.constants import (
    CANVAS_EDGE_CLEARANCE,
    STATION_RADIUS_APPROX,
    STATION_STROKE_APPROX,
    WIDEST_THEME_LINE_WIDTH,
)

_ACTIVE_FONT_SCALE: ContextVar[float] = ContextVar("active_font_scale", default=1.0)
_ACTIVE_STROKE_SCALE: ContextVar[float] = ContextVar("active_stroke_scale", default=1.0)


def active_font_scale() -> float:
    """Font-size multiplier in effect for the current layout/render pass."""
    return _ACTIVE_FONT_SCALE.get()


@contextmanager
def font_scale_context(scale: float) -> Iterator[None]:
    """Apply ``scale`` to the text metrics for the duration of the block."""
    token = _ACTIVE_FONT_SCALE.set(scale)
    try:
        yield
    finally:
        _ACTIVE_FONT_SCALE.reset(token)


def active_stroke_scale() -> float:
    """Stroke-weight multiplier in effect for the current layout/render pass."""
    return _ACTIVE_STROKE_SCALE.get()


@contextmanager
def stroke_scale_context(scale: float) -> Iterator[None]:
    """Apply ``scale`` to the station-marker metrics for the block's duration."""
    token = _ACTIVE_STROKE_SCALE.set(scale)
    try:
        yield
    finally:
        _ACTIVE_STROKE_SCALE.reset(token)


def station_radius_approx() -> float:
    """Station pill radius to reserve against, under the active stroke scale."""
    return STATION_RADIUS_APPROX * active_stroke_scale()


def station_stroke_approx() -> float:
    """Station marker stroke width to reserve against, under the active scale."""
    return STATION_STROKE_APPROX * active_stroke_scale()


def canvas_edge_clearance() -> float:
    """Canvas-margin clearance to reserve against, under the active stroke scale.

    ``stroke_scale`` multiplies the stroke a route is drawn with but not the
    direction chevron's arms, so only the half-stroke term of
    :data:`~nf_metro.layout.constants.CANVAS_EDGE_CLEARANCE` scales.  Reading the
    unscaled constant would leave a coarsened render short of the margin by half
    the extra stroke weight.
    """
    extra_stroke = WIDEST_THEME_LINE_WIDTH * (active_stroke_scale() - 1.0)
    return CANVAS_EDGE_CLEARANCE + extra_stroke / 2
