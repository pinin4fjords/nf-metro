"""Same-line merge feeders converge as a single stroke.

A merge junction has N>1 feeders of one metro line converging on a single
entry port.  The farthest feeder carries a full bypass to the entry (the
"trunk"); a feeder classified as a branch descends onto the trunk's bypass
channel, so the converging line is a single stroke up to the point it genuinely
diverges.  A feeder in the column next to the merge whose short hop is clear
and whose channel sits past the merge is NOT a branch: for that one the channel
is a detour away from its target and back, which draws the second stroke the
branch rule exists to prevent.  Three invariants pin this: no two same-line
feeders run offset-parallel (which would draw as duplicate tracks, or abort the
render when their descents land an offset-step apart), a branch terminates on
the trunk's channel, and a feeder that is not a branch never visits it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from nf_metro.layout.constants import (
    COORD_TOLERANCE,
    CURVE_RADIUS,
    DIAGONAL_RUN,
    EDGE_TO_BUNDLE_CLEARANCE,
)
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import (
    initial_fanout_descent_span,
    iter_horizontal_trunks,
)
from nf_metro.layout.routing.context import _build_routing_context, _resolve_section_col
from nf_metro.layout.routing.invariants import check_no_same_line_parallel_descents
from nf_metro.layout.routing.normalize import (
    _final_port_approach,
    _initial_fanout_descent,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

_ROOT = Path(__file__).resolve().parents[1]
_TOPOLOGIES = _ROOT / "examples" / "topologies"

_FIXTURES = {
    name: (_TOPOLOGIES / f"{name}.mmd").read_text()
    for name in (
        "merge_adjacent_feeder",
        "merge_bottom_row_bypass",
        "merge_pullaway",
        "merge_right_entry",
    )
}

# Fixtures whose merge has at least one feeder classified onto the trunk
# channel.  ``merge_adjacent_feeder`` is deliberately absent: its one non-trunk
# feeder reaches the merge directly, which is what that fixture pins.
_BRANCH_FIXTURES = ("merge_bottom_row_bypass", "merge_pullaway", "merge_right_entry")


def _layout_and_route(mmd: str):
    graph = parse_metro_mermaid(mmd)
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    ctx = _build_routing_context(graph, DIAGONAL_RUN, CURVE_RADIUS, offsets)
    return graph, routes, offsets, ctx


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_no_same_line_parallel_merge_descents(name: str) -> None:
    """No two same-line feeders of a merge run offset-parallel on the V axis."""
    graph, routes, offsets, _ctx = _layout_and_route(_FIXTURES[name])
    violations = check_no_same_line_parallel_descents(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


@pytest.mark.parametrize("name", sorted(_BRANCH_FIXTURES))
def test_merge_branches_join_trunk_channel(name: str) -> None:
    """Each feeder classified a branch terminates on the trunk's bypass channel.

    A branch that ends on an emitted horizontal trunk run has dropped onto the
    trunk to travel as one stroke; one that ends elsewhere is a second
    independent stroke into the merge.
    """
    graph, routes, _offsets, ctx = _layout_and_route(_FIXTURES[name])
    by_key = {(r.edge.source, r.edge.target, r.line_id): r for r in routes}
    checked = 0
    for mjid, trunk_source in ctx.merge.trunk_source.items():
        trunk_route = next(
            route
            for route in routes
            if route.edge.source == trunk_source and route.edge.target == mjid
        )
        trunk_channels = tuple(
            segment.y for _rank, segment in iter_horizontal_trunks(trunk_route)
        )
        assert trunk_channels, f"{name}: merge trunk has no horizontal run"
        for e in graph.edges_to(mjid):
            key = (e.source, e.target, e.line_id)
            if key not in ctx.merge.branch_edges:
                continue
            rp = by_key.get(key)
            if rp is None:
                continue
            checked += 1
            end_y = rp.points[-1][1]
            assert any(
                abs(end_y - channel_y) <= COORD_TOLERANCE
                for channel_y in trunk_channels
            ), (
                f"{name}: branch feeder {e.source}->{mjid} ends at "
                f"y={end_y:.1f}, not on a trunk channel {trunk_channels}"
            )
    assert checked, f"{name}: expected at least one branch feeder"


def test_clear_adjacent_feeder_does_not_detour_to_the_trunk_channel() -> None:
    """A clear one-gap hop into the merge stays between its ends.

    ``merge_adjacent_feeder`` puts the trunk's channel well below a feeder that
    sits one column from the merge with nothing in the way.  Dropping that
    feeder onto the channel would carry it past the merge and straight back,
    which lands its descent beside the trunk's ascent and leaves a stub where
    the trunk has turned away -- so its route must stay within the band its own
    two endpoints span.
    """
    graph, routes, _offsets, ctx = _layout_and_route(_FIXTURES["merge_adjacent_feeder"])
    (mjid,) = ctx.merge.trunk_source
    trunk_src = ctx.merge.trunk_source[mjid]
    channel_y = ctx.merge.trunk_by[mjid]
    merge_y = graph.stations[mjid].y

    feeders = [e for e in graph.edges_to(mjid) if e.source != trunk_src]
    assert feeders, "fixture no longer has a non-trunk merge feeder"
    for e in feeders:
        assert (e.source, e.target, e.line_id) not in ctx.merge.branch_edges, (
            f"clear adjacent feeder {e.source}->{mjid} was classified a branch"
        )
        rp = next(
            r for r in routes if (r.edge.source, r.edge.target) == (e.source, mjid)
        )
        start_y = rp.points[0][1]
        lo_y = min(start_y, merge_y) - COORD_TOLERANCE
        hi_y = max(start_y, merge_y) + COORD_TOLERANCE
        strayed = [(x, y) for x, y in rp.points if not lo_y <= y <= hi_y]
        assert not strayed, (
            f"feeder {e.source}->{mjid} leaves the band its endpoints span "
            f"({lo_y:.1f}..{hi_y:.1f}) at {strayed} -- detoured toward the "
            f"trunk channel at y={channel_y:.1f}"
        )


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_feeder_descent_ownership_and_legacy_snapping(name: str) -> None:
    """Planned turns stay immutable; legacy descents snap only in one column.

    ``_coincide_same_line_tracks`` snaps a feeder onto the trunk's exact
    descent X only when the feeder shares the trunk's source column; a feeder in
    another column descends in its own inter-column gap and converges along the
    shared horizontal channel instead.  An exit-turn plan owns its opening
    descent, so the legacy normalizer must not expose that segment as movable.
    """
    graph, routes, _offsets, ctx = _layout_and_route(_FIXTURES[name])
    by_key = {(r.edge.source, r.edge.target, r.line_id): r for r in routes}
    seen = 0
    for mjid, trunk_src in ctx.merge.trunk_source.items():
        trunk_rp = next(
            (
                by_key[(e.source, e.target, e.line_id)]
                for e in graph.edges_to(mjid)
                if e.source == trunk_src and (e.source, e.target, e.line_id) in by_key
            ),
            None,
        )
        trunk_span = initial_fanout_descent_span(trunk_rp) if trunk_rp else None
        if trunk_span is None:
            continue
        trunk_x = trunk_span[0]
        trunk_col = _resolve_section_col(graph, graph.stations[trunk_src])
        for e in graph.edges_to(mjid):
            if e.source == trunk_src:
                continue
            rp = by_key.get((e.source, e.target, e.line_id))
            if rp is None or initial_fanout_descent_span(rp) is None:
                continue
            seen += 1
            planned = (
                rp.exit_turn_axis_id is not None and rp.exit_turn_segment_rank == 1
            ) or bool(rp.convergence_owned_segment_ranks)
            ch = _initial_fanout_descent(rp)
            if planned:
                assert ch is None
                continue
            assert ch is not None
            same_col = (
                _resolve_section_col(graph, graph.stations[e.source]) == trunk_col
            )
            coincident = abs(ch.x - trunk_x) <= COORD_TOLERANCE
            if same_col:
                assert coincident, (
                    f"{name}: same-column feeder {e.source} descends at "
                    f"x={ch.x:.1f}, not fused with trunk descent x={trunk_x:.1f}"
                )
            else:
                assert not coincident, (
                    f"{name}: cross-column feeder {e.source} was snapped onto the "
                    f"trunk descent x={trunk_x:.1f}; distinct corridors collapsed"
                )
    assert seen, f"{name}: expected at least one non-trunk merge feeder"


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_same_line_port_approaches_coincide(name: str) -> None:
    """Same-line vertical approaches converging on one entry port share an X.

    The merge trunk ends at the entry port carrying the merge junction as its
    edge target; a same-line feed arriving directly at that port (an exit-port
    source not folded into the merge) must share the trunk's final riser rather
    than running an offset apart beside it.
    """
    _graph, routes, _offsets, ctx = _layout_and_route(_FIXTURES[name])
    by_port: dict[tuple[str, str, bool], list[float]] = defaultdict(list)
    for rp in routes:
        if not rp.is_inter_section:
            continue
        ch = _final_port_approach(rp)
        if ch is None:
            continue
        target = ctx.merge.entry_port_for.get(rp.edge.target, rp.edge.target)
        by_port[(target, rp.line_id, ch.down)].append(ch.x)
    for (target, line, _down), xs in by_port.items():
        # Consecutive same-line approaches to one port must be either coincident
        # (one fused track) or genuinely distant (separate corridors beyond the
        # fuse band); a small offset between them is the duplicate-riser defect.
        for a, b in zip(sorted(xs), sorted(xs)[1:]):
            gap = b - a
            assert gap <= COORD_TOLERANCE or gap > EDGE_TO_BUNDLE_CLEARANCE, (
                f"{name}: line {line!r} approaches port {target!r} on two "
                f"near-parallel risers (x={a:.1f}, {b:.1f}; {gap:.1f}px apart) "
                "instead of one fused track"
            )
