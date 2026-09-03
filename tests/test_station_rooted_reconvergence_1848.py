"""A trunkless symmetric fan that reconverges onto a hidden merge node sits on
the fan's served-span centreline even when the fan is rooted at an internal
station rather than a section entry port.

``orf_merge`` (entry-port-rooted) and this join are the same shape: several
branches diverge and fully reconverge, and under ``diamond_style: symmetric``
the join belongs on the midpoint of the branches it merges.  A hidden merge
node carries no glyph and is skipped by every on-track placement mechanism, so
without this rule it stays on whatever grid row its topological layer occupied
(the bottom branch's row here), not the span midpoint (issue #1848).

A *visible* internal join is deliberately excluded: the general placement
pipeline already seats it on its midpoint, and re-seating it from the grid
snap's intermediate coordinates displaces it.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.fan_bundles import _symmetric_reconvergence_joins
from nf_metro.parser.mermaid import parse_metro_mermaid

FIXTURES = Path(__file__).parent / "fixtures" / "curve_invariant_repros"

_HIDDEN_FIXTURE = FIXTURES / "riboseq_inter_row_corridor.mmd"
_HIDDEN_JOIN = "__converge_te_out_1"
_HIDDEN_BRANCHES = ("anota2seq", "deltate", "dotseq")

_VISIBLE_FIXTURE = FIXTURES / "inter_row_corridor_overflow.mmd"
_VISIBLE_JOIN = "step_f5"
_VISIBLE_BRANCHES = ("step_f2", "step_f3", "step_f4")

TOL = 2.0


def _layout(fixture):
    graph = parse_metro_mermaid(fixture.read_text())
    compute_layout(graph)
    return graph


def _fan_midpoint(graph, branches):
    branch_ys = [graph.stations[b].y for b in branches]
    return (min(branch_ys) + max(branch_ys)) / 2.0


def test_hidden_station_rooted_join_is_admitted():
    graph = _layout(_HIDDEN_FIXTURE)
    joins = _symmetric_reconvergence_joins(graph)
    assert _HIDDEN_JOIN in joins, sorted(joins)
    assert graph.stations[_HIDDEN_JOIN].is_hidden
    assert set(joins[_HIDDEN_JOIN]) == set(_HIDDEN_BRANCHES)


def test_hidden_station_rooted_join_on_fan_centreline():
    graph = _layout(_HIDDEN_FIXTURE)
    midpoint = _fan_midpoint(graph, _HIDDEN_BRANCHES)
    join_y = graph.stations[_HIDDEN_JOIN].y
    assert abs(join_y - midpoint) < TOL, (
        f"hidden reconvergence join {_HIDDEN_JOIN} y={join_y} not on fan "
        f"centreline {midpoint}"
    )


def test_visible_station_rooted_join_is_excluded():
    # A station-rooted fan whose join is visible must NOT be seated here: the
    # general placement pipeline already sits it on its midpoint, and re-seating
    # it from the grid snap's intermediate coordinates would drag it off.
    graph = _layout(_VISIBLE_FIXTURE)
    joins = _symmetric_reconvergence_joins(graph)
    assert not graph.stations[_VISIBLE_JOIN].is_hidden
    assert _VISIBLE_JOIN not in joins, sorted(joins)
    midpoint = _fan_midpoint(graph, _VISIBLE_BRANCHES)
    join_y = graph.stations[_VISIBLE_JOIN].y
    assert abs(join_y - midpoint) < TOL, (join_y, midpoint)
