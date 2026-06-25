"""Concentric entry corner for a multi-line TB-bottom-exit -> LR-top-entry drop.

A TB section dropping from its BOTTOM exit into the TOP entry of an LR section
turns each line through one 90-degree corner (drop, then turn into the run).
When two or more lines share that entry, the per-line drop channel and the
per-line turn-in offset must nest concentrically: the line on the inside of the
vertical drop is the line on the inside of the horizontal turn-in, so the bundle
neither pinches through the bend nor crosses a bundle-mate (issue #1061).

The fixture drops into a consumer placed to the right of the drop column, so the
run turns toward larger X: the in-section line order follows the arrival order
directly rather than being mirrored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import (
    compute_station_offsets,
    route_edges_centred,
)
from nf_metro.layout.routing.invariants import (
    check_bundle_order_preserved,
    check_concentric_bundle_corners,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES_DIR = Path(__file__).parent.parent / "examples" / "topologies"

PERP_ENTRY_BUNDLE_FIXTURES = [
    "lr_top_entry_cross_column_two_line",
]


@pytest.mark.parametrize("stem", PERP_ENTRY_BUNDLE_FIXTURES)
def test_perp_entry_bundle_corner_is_concentric(stem: str) -> None:
    graph = parse_metro_mermaid((TOPOLOGIES_DIR / f"{stem}.mmd").read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)

    pinches = check_concentric_bundle_corners(graph, routes, offsets)
    assert not pinches, "\n".join(v.message() for v in pinches)

    flips = check_bundle_order_preserved(routes)
    assert not flips, "\n".join(v.message() for v in flips)
