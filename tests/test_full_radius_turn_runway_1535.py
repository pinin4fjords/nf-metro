"""Inter-section orthogonal turns round at their full requested radius (#1535).

Every horizontal-to-vertical turn an inter-section route makes rounds at the
radius its bundle asked for.  A channel seated closer than that radius to the
port it turns into leaves the corner clamped by the stub of runway that is left,
so the arc draws tighter than the concentric geometry intended even though it is
not a hard 90.

Two seatings can starve that runway, so each is pinned here:

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
from nf_metro.layout.routing.common import (
    COORD_TOLERANCE,
    Edge,
    RoutedPath,
    column_gap_edges,
)
from nf_metro.layout.routing.corners import outer_lane_radius, resolve_curve_radii
from nf_metro.layout.routing.invariants import check_orthogonal_turns_form_curves
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
TOPOLOGIES = ROOT / "examples" / "topologies"

FIXTURES = [
    TOPOLOGIES / "merge_offrow_continuation.mmd",
    TOPOLOGIES / "cross_row_gap_wrap.mmd",
    TOPOLOGIES / "lr_to_tb_top_two_lines.mmd",
    TOPOLOGIES / "packed_cell_consumer_drop_in.mmd",
    TOPOLOGIES / "bypass_leftward_far_side_entry.mmd",
    TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
]
IDS = [p.stem for p in FIXTURES]

_RADIUS_TOL = 0.5


def _route(path: Path) -> tuple:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    return graph, route_edges_centred(graph, station_offsets=offsets)


def _clamped_turns(routes: list[RoutedPath]) -> list[str]:
    """Every inter-section H<->V turn whose radius falls short of its request."""
    short: list[str] = []
    for rp in routes:
        if not rp.is_inter_section or len(rp.points) < 3:
            continue
        effective = resolve_curve_radii(rp.points, rp.curve_radii)
        for k, eff in enumerate(effective):
            i = k + 1
            prev, curr, nxt = rp.points[i - 1], rp.points[i], rp.points[i + 1]
            in_h = abs(curr[0] - prev[0]) > COORD_TOLERANCE
            in_v = abs(curr[1] - prev[1]) > COORD_TOLERANCE
            out_h = abs(nxt[0] - curr[0]) > COORD_TOLERANCE
            out_v = abs(nxt[1] - curr[1]) > COORD_TOLERANCE
            if not (in_h != in_v and out_h != out_v and in_h != out_h):
                continue
            radii = rp.curve_radii
            requested = (
                radii[k]
                if radii and k < len(radii) and radii[k] is not None
                else CURVE_RADIUS
            )
            if eff < requested - _RADIUS_TOL:
                short.append(
                    f"{rp.edge.source}->{rp.edge.target} [{rp.line_id}] at "
                    f"({curr[0]:.0f},{curr[1]:.0f}): radius {eff:.1f} "
                    f"of requested {requested:.1f}"
                )
    return short


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_inter_section_turns_round_at_full_radius(path: Path) -> None:
    _graph, routes = _route(path)
    short = _clamped_turns(routes)
    assert not short, "turns clamped below their requested radius:\n" + "\n".join(short)


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
    # The same gap taken over every row it does exist in is bounded by real edges.
    left_all, right_all = column_gap_edges(graph, 1, 2, row=None)
    assert right_all > left_all
    assert left_all == pytest.approx(
        graph.sections["sink"].bbox_x + graph.sections["sink"].bbox_w
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
    assert len(drops) == 2, "fixture no longer carries a two-line drop"
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


def test_guard_flags_a_turn_curving_below_its_requested_radius() -> None:
    """A formed-but-tight corner is a violation, not just a hard 90."""
    edge = Edge(source="a", target="b", line_id="x")
    tight = RoutedPath(
        edge=edge,
        line_id="x",
        points=[(0.0, 0.0), (40.0, 0.0), (40.0, 100.0), (46.0, 100.0)],
        is_inter_section=True,
        curve_radii=[CURVE_RADIUS, CURVE_RADIUS],
    )
    violations = check_orthogonal_turns_form_curves(None, [tight])  # type: ignore[arg-type]
    assert violations, "guard missed a turn clamped below its requested radius"
    assert violations[0].effective == pytest.approx(6.0)
    assert violations[0].requested == pytest.approx(CURVE_RADIUS)
