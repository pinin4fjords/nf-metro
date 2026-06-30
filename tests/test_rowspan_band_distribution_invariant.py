"""Invariant: single-row sections stacked beside a rowspan neighbour fill its band.

When a column holds single-row sections stacked one per grid row beside a taller
``grid_row_span > 1`` section spanning those same rows, the stack must be
distributed across that section's vertical band: the topmost section's bbox top
meets the band top and the bottommost's bbox bottom meets the band bottom.

Without this the topmost section's fan, centred on its row line, spreads upward
out of the layout into the title band, and the bottommost section floats high
with empty slack beneath it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from layout_validator import Severity, validate_layout

from nf_metro.layout.constants import SAME_COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, Section

SHOWCASE_DIR = Path(__file__).parent.parent / "examples" / "showcase"

# Fixtures with a single-row section stack beside a rowspan neighbour.
FIXTURES = [
    "single_row_rowspan_neighbor",
]

# Generous slack: a band that exceeds the stack's natural height by less than
# this is treated as "already filled" and not distributed.
_SLACK_FLOOR = 2 * SAME_COORD_TOLERANCE


def _stacks_beside_rowspan(
    graph: MetroGraph,
) -> list[tuple[list[Section], float, float]]:
    """Return (stack, band_top, band_bot) for every column whose single-row
    sections fill the row range a neighbouring rowspan section spans."""
    rowspans = [
        s
        for s in graph.sections.values()
        if s.grid_row_span > 1 and s.bbox_h > 0 and s.grid_row >= 0
    ]
    if not rowspans:
        return []

    by_col: dict[int, list[Section]] = defaultdict(list)
    for section in graph.sections.values():
        if section.grid_row_span == 1 and section.bbox_h > 0 and section.grid_row >= 0:
            by_col[section.grid_col].append(section)

    out: list[tuple[list[Section], float, float]] = []
    for col, stack in by_col.items():
        stack.sort(key=lambda s: s.grid_row)
        band = [
            r
            for r in rowspans
            if abs(r.grid_col - col) == 1
            and r.grid_row <= stack[0].grid_row
            and r.grid_row + r.grid_row_span - 1 >= stack[-1].grid_row
        ]
        if not band:
            continue
        band_top_row = min(r.grid_row for r in band)
        band_bot_row = max(r.grid_row + r.grid_row_span - 1 for r in band)
        if sorted(s.grid_row for s in stack) != list(
            range(band_top_row, band_bot_row + 1)
        ):
            continue
        band_top = min(r.bbox_y for r in band)
        band_bot = max(r.bbox_y + r.bbox_h for r in band)
        if (band_bot - band_top) - sum(s.bbox_h for s in stack) <= _SLACK_FLOOR:
            continue
        out.append((stack, band_top, band_bot))
    return out


@pytest.mark.parametrize("stem", FIXTURES)
def test_stacked_rows_fill_rowspan_band(stem: str) -> None:
    graph = parse_metro_mermaid((SHOWCASE_DIR / f"{stem}.mmd").read_text())
    compute_layout(graph, validate=True)

    stacks = _stacks_beside_rowspan(graph)
    assert stacks, (
        f"{stem}: expected a single-row stack beside a rowspan neighbour; "
        "fixture no longer exercises the invariant"
    )

    for stack, band_top, band_bot in stacks:
        top = stack[0]
        bot = stack[-1]
        assert abs(top.bbox_y - band_top) <= SAME_COORD_TOLERANCE, (
            f"{stem}: top section '{top.id}' bbox top {top.bbox_y:.1f} does not "
            f"meet band top {band_top:.1f} (rises out of the band by "
            f"{band_top - top.bbox_y:.1f}px)"
        )
        bot_edge = bot.bbox_y + bot.bbox_h
        assert abs(bot_edge - band_bot) <= SAME_COORD_TOLERANCE, (
            f"{stem}: bottom section '{bot.id}' bbox bottom {bot_edge:.1f} does "
            f"not meet band bottom {band_bot:.1f} (slack of "
            f"{band_bot - bot_edge:.1f}px below it)"
        )


@pytest.mark.parametrize("stem", FIXTURES)
def test_showcase_fixture_has_no_layout_errors(stem: str) -> None:
    """The relocated fixture skips the auto-globbed topology corpus, so run the
    error-level layout validation here (sub-pixel warnings from the center-ported
    fan are out of scope)."""
    graph = parse_metro_mermaid((SHOWCASE_DIR / f"{stem}.mmd").read_text())
    compute_layout(graph, validate=True)

    errors = [v for v in validate_layout(graph) if v.severity == Severity.ERROR]
    assert not errors, "\n".join(v.message for v in errors)
