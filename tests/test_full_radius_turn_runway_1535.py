"""Turn columns are seated a full concentric radius clear of what they leave.

`check_orthogonal_turns_form_curves` holds every inter-section
horizontal-to-vertical turn to the radius its bundle asked for, and
``test_orthogonal_turns_form_curves_corpus`` sweeps that guard over every
example.  These are the two seatings that starved the runway behind it, pinned
at the mechanism so a regression names its own cause rather than surfacing as a
clamped corner somewhere downstream:

* the inter-column gap a channel is bundled into must be bounded by real section
  edges in the row it traverses -- a gap reported for a row where one bounding
  column has no section spans several real columns, and re-centring a channel in
  that span parks it against the port it turns into;
* a bundle's lead-in must clear its source by the *outer* lane's arc, since that
  lane sweeps the base radius plus the whole bundle width.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.layout.routing.common import column_gap_edges
from nf_metro.layout.routing.corners import outer_lane_radius
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
TOPOLOGIES = ROOT / "examples" / "topologies"

_RADIUS_TOL = 0.5


def _route(path: Path) -> tuple:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    return graph, route_edges_centred(graph, station_offsets=offsets)


def test_column_gap_edges_reports_no_gap_where_lower_column_is_absent() -> None:
    """A row the lower column does not occupy bounds no gap in that row.

    ``merge_offrow_continuation`` has no section in column 1 of row 0, so the
    gap between columns 1 and 2 does not exist there.  Reporting one bounded at
    the canvas origin would span two real columns of the map, and any channel
    inside that span would match it.
    """
    graph = parse_metro_mermaid(
        (TOPOLOGIES / "merge_offrow_continuation.mmd").read_text()
    )
    compute_layout(graph)
    left, right = column_gap_edges(graph, 1, 2, row=0)
    assert right <= left, f"fabricated gap [{left:.0f}..{right:.0f}] in an empty cell"
    # The same gap over every row it does exist in is bounded by real edges.
    left_all, right_all = column_gap_edges(graph, 1, 2, row=None)
    assert right_all > left_all
    sink = graph.sections["sink"]
    assert left_all == pytest.approx(sink.bbox_x + sink.bbox_w)


def test_bypass_descent_keeps_its_runway_into_the_entry_port() -> None:
    """The off-row bypass turns into ``sink``'s port a full radius clear of it.

    Its descent column sits in the gap one column to the left; seating it in the
    span the absent row-0 cell reported would leave a stub of runway for the turn
    at the bottom.
    """
    graph, routes = _route(TOPOLOGIES / "merge_offrow_continuation.mmd")
    port_x = graph.stations["sink__entry_left_3"].x
    bypass = next(
        rp
        for rp in routes
        if rp.edge.source == "above_src__exit_right_0" and rp.line_id == "bypass"
    )
    descent_x = bypass.points[-2][0]
    assert port_x - descent_x >= CURVE_RADIUS - _RADIUS_TOL, (
        f"descent column at x={descent_x:.0f} is {port_x - descent_x:.0f}px "
        f"from the port at x={port_x:.0f}"
    )


def test_bundle_lead_in_covers_the_widest_lane_arc() -> None:
    """A two-line drop leads out far enough for its outer lane's wider arc.

    Both lines leave ``src_sec``'s RIGHT exit and turn down toward ``tgt_sec``'s
    TOP entry.  The outer lane of that turn sweeps one ``OFFSET_STEP`` wider than
    the inner one, so the lead-in must run the *outer* lane's radius clear of the
    exit -- a lead sized to the base radius clamps both lanes to it.
    """
    graph, routes = _route(TOPOLOGIES / "lr_to_tb_top_two_lines.mmd")
    exit_x = graph.stations["src_sec__exit_right_0"].x
    drops = [rp for rp in routes if rp.edge.source == "src_sec__exit_right_0"]
    assert len(drops) == 2, "fixture must route two lines out of the RIGHT exit"
    widest = max(drops, key=lambda rp: rp.curve_radii[0])
    assert widest.curve_radii[0] == pytest.approx(
        outer_lane_radius(2, CURVE_RADIUS, OFFSET_STEP)
    )
    for rp in drops:
        lead = abs(rp.points[1][0] - exit_x)
        assert lead >= rp.curve_radii[0] - _RADIUS_TOL, (
            f"line {rp.line_id!r} leads out {lead:.0f}px before a corner "
            f"asking for {rp.curve_radii[0]:.0f}px"
        )
