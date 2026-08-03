"""Deterministic text-metrics contracts for layout and rendering."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nf_metro.text_metrics import (
    DEFAULT_TEXT_METRICS,
    MetricsFace,
    TextRole,
    TextStyle,
)


@pytest.mark.parametrize(
    ("role", "font_size", "expected"),
    [
        (TextRole.STATION_LABEL, 13.0, 27.0),
        (TextRole.SECTION_HEADER, 16.0, 28.8),
        (TextRole.LEGEND_ENTRY, 14.0, 23.1),
        (TextRole.ICON_LABEL, 13.0, 23.4),
        (TextRole.ICON_CAPTION, 7.8, 12.87),
        (TextRole.GROUP_CAPTION, 12.35, 20.3775),
        (TextRole.RAIL_LABEL, 13.0, 27.0),
    ],
)
def test_fallback_reservations_characterize_current_roles(
    role: TextRole, font_size: float, expected: float
) -> None:
    style = TextStyle(font_size=font_size, weight="bold", face=MetricsFace.FALLBACK)
    assert DEFAULT_TEXT_METRICS.reserve_width("WWW", style, role) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("WWW", 36.816),
        ("Ill", 10.842),
        ("café", 26.585),
    ],
)
def test_fallback_advance_preserves_helvetica_bold_model(
    text: str, expected: float
) -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.FALLBACK)
    assert DEFAULT_TEXT_METRICS.advance(
        text, style, TextRole.STATION_LABEL
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("weight", "text", "expected"),
    [
        ("400", "WWW", 38.4287109375),
        ("bold", "WWW", 40.46630859375),
        ("600", "WWW", 40.46630859375),
        ("700", "café", 28.11376953125),
    ],
)
def test_inter_advance_comes_from_bundled_font_tables(
    weight: str, text: str, expected: float
) -> None:
    style = TextStyle(font_size=13.0, weight=weight, face=MetricsFace.INTER)
    assert DEFAULT_TEXT_METRICS.advance(
        text, style, TextRole.STATION_LABEL
    ) == pytest.approx(expected)


def test_multiline_advance_uses_widest_line() -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.INTER)
    assert DEFAULT_TEXT_METRICS.advance(
        "Ill\nWWW", style, TextRole.STATION_LABEL
    ) == pytest.approx(
        DEFAULT_TEXT_METRICS.advance("WWW", style, TextRole.STATION_LABEL)
    )


def test_inter_advance_includes_spaces() -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.INTER)
    with_space = DEFAULT_TEXT_METRICS.advance("A A", style, TextRole.STATION_LABEL)
    without_space = DEFAULT_TEXT_METRICS.advance("AA", style, TextRole.STATION_LABEL)
    space = DEFAULT_TEXT_METRICS.advance(" ", style, TextRole.STATION_LABEL)
    assert with_space == pytest.approx(without_space + space)


def test_inter_ink_bbox_tracks_glyph_bounds_and_multiline_height() -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.INTER)
    single = DEFAULT_TEXT_METRICS.ink_bbox("café", style, TextRole.STATION_LABEL)
    multiline = DEFAULT_TEXT_METRICS.ink_bbox(
        "café\nWWW", style, TextRole.STATION_LABEL
    )
    assert single.width < DEFAULT_TEXT_METRICS.advance(
        "café", style, TextRole.STATION_LABEL
    )
    assert single.y_min < 0.0 < single.y_max
    assert multiline.width >= single.width
    assert multiline.height > single.height


def test_inter_line_height_uses_bundled_vertical_metrics() -> None:
    style = TextStyle(font_size=13.0, weight="400", face=MetricsFace.INTER)
    expected = 13.0 * (1984 + 494) / 2048
    assert DEFAULT_TEXT_METRICS.line_height(
        style, TextRole.STATION_LABEL
    ) == pytest.approx(expected)


def test_font_scaling_is_linear() -> None:
    small = TextStyle(font_size=10.0, weight="bold", face=MetricsFace.INTER)
    large = TextStyle(font_size=20.0, weight="bold", face=MetricsFace.INTER)
    assert DEFAULT_TEXT_METRICS.advance(
        "café", large, TextRole.STATION_LABEL
    ) == pytest.approx(
        2 * DEFAULT_TEXT_METRICS.advance("café", small, TextRole.STATION_LABEL)
    )
    assert DEFAULT_TEXT_METRICS.ink_bbox(
        "café", large, TextRole.STATION_LABEL
    ).width == pytest.approx(
        2 * DEFAULT_TEXT_METRICS.ink_bbox("café", small, TextRole.STATION_LABEL).width
    )


def test_missing_inter_glyph_uses_visible_replacement_metrics() -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.INTER)
    assert DEFAULT_TEXT_METRICS.advance(
        "Ω", style, TextRole.STATION_LABEL
    ) == pytest.approx(DEFAULT_TEXT_METRICS.advance("?", style, TextRole.STATION_LABEL))
    assert DEFAULT_TEXT_METRICS.ink_bbox(
        "Ω", style, TextRole.STATION_LABEL
    ) == DEFAULT_TEXT_METRICS.ink_bbox("?", style, TextRole.STATION_LABEL)


def test_measurements_are_cached() -> None:
    style = TextStyle(font_size=13.0, weight="bold", face=MetricsFace.INTER)
    assert DEFAULT_TEXT_METRICS.ink_bbox(
        "cached", style, TextRole.STATION_LABEL
    ) is DEFAULT_TEXT_METRICS.ink_bbox("cached", style, TextRole.STATION_LABEL)


def test_no_ad_hoc_font_measurements_outside_metrics_module() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "nf_metro"
    findings: list[str] = []
    banned_names = {
        "GLYPH_ADVANCE_EM",
        "GLYPH_ADVANCE_DEFAULT_EM",
        "ICON_LABEL_CHAR_WIDTH_RATIO",
        "LEGEND_CHAR_WIDTH_RATIO",
        "SECTION_LABEL_CHAR_WIDTH_RATIO",
    }
    for path in root.rglob("*.py"):
        if path.name in {"text_metrics.py", "_inter_metrics.py"}:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_names:
                findings.append(f"{path.relative_to(root)}:{node.lineno}: {node.id}")
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
                continue
            calls_len = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "len"
                for child in ast.walk(node)
            )
            names = {
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            call_names = {
                child.func.attr
                if isinstance(child.func, ast.Attribute)
                else child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, (ast.Attribute, ast.Name))
            }
            centralized = {"line_height", "header_line_height", "reserve_width"}
            if (
                calls_len
                and not call_names.intersection(centralized)
                and any("font" in name or "char_width" in name for name in names)
            ):
                findings.append(
                    f"{path.relative_to(root)}:{node.lineno}: len-based width"
                )
    assert not findings, "\n".join(sorted(set(findings)))
