"""Invariant: TB sections with BOTTOM-only exits keep normal padding below last station.

A TB section whose only exit is BOTTOM exits straight down into the section
below; it does not fold sideways through the inter-row gap.  The fold-span
extension in _apply_tb_fold_spans must therefore not grow its bbox_h, so the
space below the last internal station stays close to the normal SECTION_Y_PADDING
rather than doubling it (issue #1062).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import SECTION_Y_PADDING, Y_SPACING
from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide

TOPOLOGIES_DIR = Path(__file__).parent.parent / "examples" / "topologies"

# Fixtures with a TB section that has only a BOTTOM exit and a LR/RL section
# below in the same column.  The gap below the last internal station should be
# SECTION_Y_PADDING, not the doubled value produced by the fold-span extension.
FIXTURES = [
    "lr_top_entry_cross_column",
    "tb_column_continuation_two_lines",
]

# Upper bound: SECTION_Y_PADDING plus one full y_spacing to absorb any
# legitimate label or entry-shift padding without catching the bug (which
# adds a full section_y_gap ≈ SECTION_Y_PADDING on top of the natural height).
_GAP_CEILING = SECTION_Y_PADDING + Y_SPACING


@pytest.mark.parametrize("stem", FIXTURES)
def test_tb_bottom_exit_section_gap(stem: str) -> None:
    graph = parse_metro_mermaid((TOPOLOGIES_DIR / f"{stem}.mmd").read_text())
    compute_layout(graph)

    for sec in graph.sections.values():
        if sec.direction != "TB":
            continue
        exit_sides = {s for s, _ in sec.exit_hints}
        if PortSide.BOTTOM not in exit_sides:
            continue
        if exit_sides & {PortSide.LEFT, PortSide.RIGHT}:
            continue

        internal_ys = [
            graph.stations[sid].y
            for sid in sec.station_ids
            if sid in graph.stations and not graph.stations[sid].is_port
        ]
        if not internal_ys:
            continue

        last_y = max(internal_ys)
        bbox_bottom = sec.bbox_y + sec.bbox_h
        gap = bbox_bottom - last_y

        assert gap <= _GAP_CEILING, (
            f"Section '{sec.id}' in {stem}: gap below last station is {gap:.0f}px "
            f"(last_y={last_y:.0f}, bbox_bottom={bbox_bottom:.0f}), "
            f"expected ≤ {_GAP_CEILING:.0f}px"
        )
