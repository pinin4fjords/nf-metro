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
    "font_scale_context",
    "station_radius_approx",
    "station_stroke_approx",
    "stroke_scale_context",
]

from contextlib import contextmanager
from typing import Iterator

from nf_metro.layout.constants import STATION_RADIUS_APPROX, STATION_STROKE_APPROX

_ACTIVE_FONT_SCALE: float = 1.0
_ACTIVE_STROKE_SCALE: float = 1.0


def active_font_scale() -> float:
    """Font-size multiplier in effect for the current layout/render pass."""
    return _ACTIVE_FONT_SCALE


@contextmanager
def font_scale_context(scale: float) -> Iterator[None]:
    """Apply ``scale`` to the text metrics for the duration of the block."""
    global _ACTIVE_FONT_SCALE
    previous = _ACTIVE_FONT_SCALE
    _ACTIVE_FONT_SCALE = scale
    try:
        yield
    finally:
        _ACTIVE_FONT_SCALE = previous


def active_stroke_scale() -> float:
    """Stroke-weight multiplier in effect for the current layout/render pass."""
    return _ACTIVE_STROKE_SCALE


@contextmanager
def stroke_scale_context(scale: float) -> Iterator[None]:
    """Apply ``scale`` to the station-marker metrics for the block's duration."""
    global _ACTIVE_STROKE_SCALE
    previous = _ACTIVE_STROKE_SCALE
    _ACTIVE_STROKE_SCALE = scale
    try:
        yield
    finally:
        _ACTIVE_STROKE_SCALE = previous


def station_radius_approx() -> float:
    """Station pill radius to reserve against, under the active stroke scale."""
    return STATION_RADIUS_APPROX * _ACTIVE_STROKE_SCALE


def station_stroke_approx() -> float:
    """Station marker stroke width to reserve against, under the active scale."""
    return STATION_STROKE_APPROX * _ACTIVE_STROKE_SCALE
