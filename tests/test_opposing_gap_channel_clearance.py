"""Opposite-running vertical gap bundles keep separate corridors."""

from pathlib import Path

import pytest

from nf_metro.layout.constants import BUNDLE_TO_BUNDLE_CLEARANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import apply_route_offsets, gap_lo_for_x
from nf_metro.layout.routing.invariants import check_opposing_gap_channel_clearance
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES = Path(__file__).parent.parent / "examples" / "topologies"


@pytest.mark.parametrize(
    "fixture",
    [
        "dogleg_exempt_distinct.mmd",
        "dogleg_exempt_sameline.mmd",
        "exit_run_three_drop_columns.mmd",
        "merge_around_below_leftmost.mmd",
        "merge_feeder_shared_channel_gap.mmd",
        "packed_cell_right_exit_left_entry_wrap.mmd",
    ],
)
def test_opposing_gap_bundles_are_separated(fixture: str) -> None:
    graph = parse_metro_mermaid((TOPOLOGIES / fixture).read_text())
    compute_layout(graph, validate=False)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)

    violations = check_opposing_gap_channel_clearance(graph, routes, offsets)

    assert not violations, "; ".join(v.message() for v in violations)

    if fixture != "packed_cell_right_exit_left_entry_wrap.mmd":
        return

    downward_xs: list[float] = []
    upward_xs: list[float] = []
    for route in routes:
        if route.line_id not in {"assembled", "reference", "short"}:
            continue
        if not route.is_inter_section:
            continue
        points = apply_route_offsets(route, offsets)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if abs(x1 - x0) >= 0.1:
                continue
            if gap_lo_for_x(graph, x0, min(y0, y1), max(y0, y1)) == (1, 0):
                target = downward_xs if y1 > y0 else upward_xs
                target.append(x0)

    assert downward_xs
    assert upward_xs
    assert min(abs(down_x - up_x) for down_x in downward_xs for up_x in upward_xs) >= (
        BUNDLE_TO_BUNDLE_CLEARANCE - 0.1
    )
