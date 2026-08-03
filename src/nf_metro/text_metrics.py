"""Deterministic text measurement for layout and SVG rendering."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterator, Protocol

from nf_metro._inter_metrics import (
    INTER_ASCENT,
    INTER_BOLD_METRICS,
    INTER_DESCENT,
    INTER_LINE_GAP,
    INTER_REGULAR_METRICS,
    INTER_REPLACEMENT_CODEPOINT,
    INTER_UNITS_PER_EM,
)

__all__ = [
    "DEFAULT_TEXT_METRICS",
    "MetricsFace",
    "TextBBox",
    "TextMetrics",
    "TextRole",
    "TextStyle",
    "active_metrics_face",
    "metrics_face_context",
    "text_style",
]


class MetricsFace(str, Enum):
    """Face whose deterministic metrics govern a measurement."""

    FALLBACK = "fallback"
    INTER = "inter"


class TextRole(str, Enum):
    """Semantic role whose clearance policy applies to measured text."""

    STATION_LABEL = "station_label"
    RAIL_LABEL = "rail_label"
    SECTION_HEADER = "section_header"
    SECTION_NUMBER = "section_number"
    LEGEND_ENTRY = "legend_entry"
    ICON_LABEL = "icon_label"
    ICON_CAPTION = "icon_caption"
    GROUP_CAPTION = "group_caption"
    TITLE = "title"
    FIGURE_CAPTION = "figure_caption"
    WATERMARK = "watermark"
    DEBUG = "debug"


@dataclass(frozen=True, slots=True)
class TextStyle:
    """Font properties that affect deterministic text geometry."""

    font_size: float
    weight: str = "400"
    face: MetricsFace = MetricsFace.FALLBACK


@dataclass(frozen=True, slots=True)
class TextBBox:
    """Ink bounds relative to the first line's left baseline origin."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class TextMetrics(Protocol):
    """Immutable interface for deterministic text geometry."""

    def advance(self, text: str, style: TextStyle, role: TextRole) -> float: ...

    def ink_bbox(self, text: str, style: TextStyle, role: TextRole) -> TextBBox: ...

    def line_height(self, style: TextStyle, role: TextRole) -> float: ...

    def reserve_width(self, text: str, style: TextStyle, role: TextRole) -> float: ...


_FALLBACK_ADVANCE_EM: dict[str, float] = {
    " ": 0.278,
    "!": 0.333,
    '"': 0.474,
    "#": 0.556,
    "$": 0.556,
    "%": 0.889,
    "&": 0.722,
    "'": 0.238,
    "(": 0.333,
    ")": 0.333,
    "*": 0.389,
    "+": 0.584,
    ",": 0.278,
    "-": 0.333,
    ".": 0.278,
    "/": 0.278,
    "0": 0.556,
    "1": 0.556,
    "2": 0.556,
    "3": 0.556,
    "4": 0.556,
    "5": 0.556,
    "6": 0.556,
    "7": 0.556,
    "8": 0.556,
    "9": 0.556,
    ":": 0.333,
    ";": 0.333,
    "<": 0.584,
    "=": 0.584,
    ">": 0.584,
    "?": 0.611,
    "@": 0.975,
    "A": 0.722,
    "B": 0.722,
    "C": 0.722,
    "D": 0.722,
    "E": 0.667,
    "F": 0.611,
    "G": 0.778,
    "H": 0.722,
    "I": 0.278,
    "J": 0.556,
    "K": 0.722,
    "L": 0.611,
    "M": 0.833,
    "N": 0.722,
    "O": 0.778,
    "P": 0.667,
    "Q": 0.778,
    "R": 0.722,
    "S": 0.667,
    "T": 0.611,
    "U": 0.722,
    "V": 0.667,
    "W": 0.944,
    "X": 0.667,
    "Y": 0.667,
    "Z": 0.611,
    "[": 0.333,
    "\\": 0.278,
    "]": 0.333,
    "^": 0.584,
    "_": 0.556,
    "`": 0.333,
    "a": 0.556,
    "b": 0.611,
    "c": 0.556,
    "d": 0.611,
    "e": 0.556,
    "f": 0.333,
    "g": 0.611,
    "h": 0.611,
    "i": 0.278,
    "j": 0.278,
    "k": 0.556,
    "l": 0.278,
    "m": 0.889,
    "n": 0.611,
    "o": 0.611,
    "p": 0.611,
    "q": 0.611,
    "r": 0.389,
    "s": 0.556,
    "t": 0.333,
    "u": 0.611,
    "v": 0.556,
    "w": 0.778,
    "x": 0.556,
    "y": 0.556,
    "z": 0.500,
    "{": 0.389,
    "|": 0.280,
    "}": 0.389,
    "~": 0.584,
}
_FALLBACK_DEFAULT_ADVANCE_EM = 0.6
_FALLBACK_ASCENT_EM = 0.8
_FALLBACK_DESCENT_EM = 0.2
_FALLBACK_LINE_HEIGHT_EM = 1.2
_LINE_HEIGHT_EM: dict[TextRole, float] = {
    TextRole.ICON_LABEL: 1.1,
}
_INK_WIDTH_RATIO: dict[TextRole, float] = {
    TextRole.STATION_LABEL: 0.75,
    TextRole.RAIL_LABEL: 0.75,
}

# These ratios preserve the distinct clearance policies that predate the
# centralized metrics layer. They are policy, not claims about glyph geometry.
_RESERVATION_EM: dict[TextRole, float] = {
    TextRole.STATION_LABEL: 9.0 / 13.0,
    TextRole.RAIL_LABEL: 9.0 / 13.0,
    TextRole.SECTION_HEADER: 0.6,
    TextRole.SECTION_NUMBER: 0.6,
    TextRole.LEGEND_ENTRY: 0.55,
    TextRole.ICON_LABEL: 0.6,
    TextRole.ICON_CAPTION: 0.55,
    TextRole.GROUP_CAPTION: 0.55,
    TextRole.TITLE: 0.6,
    TextRole.FIGURE_CAPTION: 0.55,
    TextRole.WATERMARK: 0.55,
    TextRole.DEBUG: 0.6,
}


def _is_bold(weight: str) -> bool:
    return weight.strip().lower() in {"bold", "600", "700"}


def _lines(text: str) -> tuple[str, ...]:
    return tuple(text.split("\n"))


class _DeterministicTextMetrics:
    @lru_cache(maxsize=8192)
    def advance(self, text: str, style: TextStyle, role: TextRole) -> float:
        if not text:
            return 0.0
        if style.face is MetricsFace.INTER:
            return self._inter_advance(text, style)
        return max(
            sum(
                _FALLBACK_ADVANCE_EM.get(char, _FALLBACK_DEFAULT_ADVANCE_EM)
                for char in line
            )
            * style.font_size
            for line in _lines(text)
        )

    @lru_cache(maxsize=8192)
    def ink_bbox(self, text: str, style: TextStyle, role: TextRole) -> TextBBox:
        if not text:
            return TextBBox(0.0, 0.0, 0.0, 0.0)
        if style.face is MetricsFace.INTER:
            return self._inter_ink_bbox(text, style, role)
        line_height = self.line_height(style, role)
        return TextBBox(
            0.0,
            -style.font_size * _FALLBACK_ASCENT_EM,
            self.advance(text, style, role),
            (len(_lines(text)) - 1) * line_height
            + style.font_size * _FALLBACK_DESCENT_EM,
        )

    @lru_cache(maxsize=256)
    def line_height(self, style: TextStyle, role: TextRole) -> float:
        policy = _LINE_HEIGHT_EM.get(role, _FALLBACK_LINE_HEIGHT_EM)
        if style.face is MetricsFace.INTER:
            units = INTER_ASCENT - INTER_DESCENT + INTER_LINE_GAP
            return style.font_size * max(units / INTER_UNITS_PER_EM, policy)
        return style.font_size * policy

    @lru_cache(maxsize=1024)
    def line_block_height(
        self,
        line_count: int,
        style: TextStyle,
        role: TextRole,
        first_line_height: float | None = None,
    ) -> float:
        """Height of stacked lines under the role's deterministic policy."""
        first = style.font_size if first_line_height is None else first_line_height
        if line_count <= 1:
            return first
        if style.face is MetricsFace.FALLBACK:
            return first + (line_count - 1) * style.font_size * _LINE_HEIGHT_EM.get(
                role, _FALLBACK_LINE_HEIGHT_EM
            )
        return first + (line_count - 1) * self.line_height(style, role)

    @lru_cache(maxsize=8192)
    def reserve_width(self, text: str, style: TextStyle, role: TextRole) -> float:
        if not text:
            return 0.0
        line_length = max(len(line) for line in _lines(text))
        ratio = _RESERVATION_EM[role]
        if role in (
            TextRole.STATION_LABEL,
            TextRole.RAIL_LABEL,
            TextRole.LEGEND_ENTRY,
        ):
            policy_width = line_length * (style.font_size * ratio)
        else:
            policy_width = line_length * style.font_size * ratio
        if style.face is MetricsFace.INTER:
            return max(policy_width, self.advance(text, style, role))
        return policy_width

    @lru_cache(maxsize=8192)
    def ink_half_width(self, text: str, style: TextStyle, role: TextRole) -> float:
        """Half-width of ink under the role's collision policy."""
        if style.face is MetricsFace.INTER:
            return self.ink_bbox(text, style, role).width / 2
        return (
            self.reserve_width(text, style, role) / 2 * _INK_WIDTH_RATIO.get(role, 1.0)
        )

    def inter_glyph_codepoint(self, char: str, style: TextStyle) -> int:
        table = self._inter_table(style)
        codepoint = ord(char)
        return codepoint if codepoint in table else INTER_REPLACEMENT_CODEPOINT

    def _inter_table(self, style: TextStyle) -> dict[int, tuple[int, ...]]:
        return INTER_BOLD_METRICS if _is_bold(style.weight) else INTER_REGULAR_METRICS

    def _inter_advance(self, text: str, style: TextStyle) -> float:
        table = self._inter_table(style)
        replacement = table[INTER_REPLACEMENT_CODEPOINT]
        return max(
            sum(table.get(ord(char), replacement)[0] for char in line)
            * style.font_size
            / INTER_UNITS_PER_EM
            for line in _lines(text)
        )

    def _inter_ink_bbox(self, text: str, style: TextStyle, role: TextRole) -> TextBBox:
        table = self._inter_table(style)
        replacement = table[INTER_REPLACEMENT_CODEPOINT]
        scale = style.font_size / INTER_UNITS_PER_EM
        line_height = self.line_height(style, role)
        bounds: list[tuple[float, float, float, float]] = []
        for line_index, line in enumerate(_lines(text)):
            cursor = 0
            baseline_y = line_index * line_height
            for char in line:
                metrics = table.get(ord(char), replacement)
                if len(metrics) == 5:
                    _advance, x_min, y_min, x_max, y_max = metrics
                    bounds.append(
                        (
                            (cursor + x_min) * scale,
                            baseline_y - y_max * scale,
                            (cursor + x_max) * scale,
                            baseline_y - y_min * scale,
                        )
                    )
                cursor += metrics[0]
        if not bounds:
            return TextBBox(0.0, 0.0, 0.0, 0.0)
        return TextBBox(
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )


DEFAULT_TEXT_METRICS = _DeterministicTextMetrics()

_ACTIVE_METRICS_FACE: ContextVar[MetricsFace] = ContextVar(
    "nf_metro_metrics_face", default=MetricsFace.FALLBACK
)


def active_metrics_face() -> MetricsFace:
    """Return the face selected for the current layout or render pass."""
    return _ACTIVE_METRICS_FACE.get()


@contextmanager
def metrics_face_context(face: MetricsFace) -> Iterator[None]:
    """Select deterministic metrics for one isolated layout or render pass."""
    token = _ACTIVE_METRICS_FACE.set(face)
    try:
        yield
    finally:
        _ACTIVE_METRICS_FACE.reset(token)


@lru_cache(maxsize=512)
def _cached_text_style(font_size: float, weight: str, face: MetricsFace) -> TextStyle:
    return TextStyle(font_size=font_size, weight=weight, face=face)


def text_style(font_size: float, weight: str = "400") -> TextStyle:
    """Build a style using the face active in the current measurement pass."""
    return _cached_text_style(font_size, weight, active_metrics_face())
