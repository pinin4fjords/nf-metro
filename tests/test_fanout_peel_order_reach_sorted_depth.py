"""Peel order for a fan whose lanes stack on column reach, not on peel depth.

Every branch of this fan leaves the source row, and the branches hop different
column counts, so the opening order sorts on ``|reach|`` rather than on the
depth each branch peels off at.  A repeated line collapses onto one bundle slot
regardless, and that slot runs innermost: its peel-offs travel outward across
every lane outside it that has not already turned away.  Contiguity in a
reach-sorted order says nothing about depth, and ``|reach|`` is the very
quantity a shared slot shape forces equal across the merged branches, so the
merge has to be judged against the peel depth directly -- a distinct line that
descends past the block's nearest peel lays its riser straight across it.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.fan_ordering import fanout_divergence_peel_order
from nf_metro.parser.mermaid import parse_metro_mermaid

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "regressions"
    / "fanout_repeat_reach_sorted_depth_span.mmd"
)


def test_repeat_spanning_depths_under_a_deeper_line_declines_to_reorder() -> None:
    """``rep``'s two branches cannot share a slot beneath ``d0``'s riser.

    ``rep`` reaches rows 3 and 6 a column out; ``d0`` reaches row 7 two columns
    out, so the reach sort opens ``d0`` outermost and leaves ``rep``'s merged
    slot innermost.  ``rep`` then peels off at row 3 while ``d0`` has yet to
    reach row 7, so no single slot for ``rep`` clears ``d0``.
    """
    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph)
    line_priority = {lid: i for i, lid in enumerate(graph.lines)}
    junction = next(
        sid
        for sid in graph.stations
        if sid.startswith("__junction") and len(list(graph.edges_from(sid))) > 1
    )

    assert fanout_divergence_peel_order(graph, junction, line_priority) is None


def test_repeat_spanning_depths_renders_within_invariants() -> None:
    """The same fan settles without a crossing between ``rep`` and ``d0``."""
    compute_layout(parse_metro_mermaid(FIXTURE.read_text()), validate=True)
