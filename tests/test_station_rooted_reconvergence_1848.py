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

FIXTURE = (
    Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "curve_invariant_repros"
    / "riboseq_inter_row_corridor.mmd"
)

_JOIN = "__converge_te_out_1"
_BRANCHES = ("anota2seq", "deltate", "dotseq")
TOL = 2.0


def _layout():
    graph = parse_metro_mermaid(FIXTURE.read_text())
    compute_layout(graph)
    return graph


def test_hidden_station_rooted_join_is_admitted():
    graph = _layout()
    joins = _symmetric_reconvergence_joins(graph)
    assert _JOIN in joins, sorted(joins)
    assert graph.stations[_JOIN].is_hidden
    assert set(joins[_JOIN]) == set(_BRANCHES)


def test_hidden_station_rooted_join_on_fan_centreline():
    graph = _layout()
    branch_ys = [graph.stations[b].y for b in _BRANCHES]
    midpoint = (min(branch_ys) + max(branch_ys)) / 2.0
    join_y = graph.stations[_JOIN].y
    assert abs(join_y - midpoint) < TOL, (
        f"hidden reconvergence join {_JOIN} y={join_y} not on fan centreline {midpoint}"
    )
