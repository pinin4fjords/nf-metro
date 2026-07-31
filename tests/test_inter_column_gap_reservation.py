"""Placement reservations for opposing inter-column routing bundles."""

from pathlib import Path

import pytest

from nf_metro.layout.constants import (
    BUNDLE_TO_BUNDLE_CLEARANCE,
    CURVE_RADIUS,
    EDGE_TO_BUNDLE_CLEARANCE,
    graph_offset_step,
)
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing.common import column_gap_edges
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES = Path(__file__).parents[1] / "examples" / "topologies"


@pytest.mark.parametrize(
    ("stem", "gap", "row", "bundle_line_counts", "anchored_runway"),
    [
        (
            "packed_cell_right_exit_left_entry_wrap",
            (1, 2),
            0,
            (2, 2),
            0.0,
        ),
        (
            "merge_around_below_leftmost",
            (2, 3),
            0,
            (1, 1),
            CURVE_RADIUS,
        ),
    ],
)
def test_column_gap_reserves_opposing_bundle_footprints(
    stem: str,
    gap: tuple[int, int],
    row: int,
    bundle_line_counts: tuple[int, int],
    anchored_runway: float,
) -> None:
    graph = parse_metro_mermaid((TOPOLOGIES / f"{stem}.mmd").read_text())
    compute_layout(graph)
    left, right = column_gap_edges(graph, *gap, row=row)
    step = graph_offset_step(graph)
    bundle_widths = tuple((count - 1) * step for count in bundle_line_counts)
    required = (
        2 * EDGE_TO_BUNDLE_CLEARANCE
        + sum(bundle_widths)
        + BUNDLE_TO_BUNDLE_CLEARANCE
        + anchored_runway
    )

    assert right - left >= required


def test_merge_wrap_runway_is_reserved_only_for_its_source_gap() -> None:
    graph = parse_metro_mermaid(
        (TOPOLOGIES / "merge_around_below_leftmost.mmd").read_text()
    )
    compute_layout(graph)

    left, right = column_gap_edges(graph, 1, 2, row=0)

    assert right - left == pytest.approx(50.0)
