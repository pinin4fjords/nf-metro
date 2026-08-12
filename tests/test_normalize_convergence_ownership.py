"""Convergence ownership boundaries for post-emission normalisation."""

from __future__ import annotations

from types import SimpleNamespace

import nf_metro.layout.routing.normalize as normalize
from nf_metro.layout.routing.common import RoutedPath
from nf_metro.parser.model import Edge


def _route(
    line_id: str,
    points: list[tuple[float, float]],
    *,
    owned: tuple[int, ...] = (),
) -> RoutedPath:
    return RoutedPath(
        edge=Edge("fork", f"target_{line_id}", line_id),
        line_id=line_id,
        points=points,
        is_inter_section=True,
        curve_radii=[10.0] * (len(points) - 2),
        convergence_owned_segment_ranks=owned,
    )


def test_fanout_traverse_bundling_uses_complete_group_ownership() -> None:
    owned = _route(
        "owned",
        [(0.0, 0.0), (10.0, 0.0), (10.0, 40.0), (90.0, 40.0), (90.0, 90.0)],
        owned=(3,),
    )
    sibling = _route(
        "sibling",
        [(0.0, 0.0), (13.0, 0.0), (13.0, 55.0), (80.0, 55.0), (80.0, 90.0)],
    )
    members = normalize._fanout_traverse_legs([owned, sibling])[("fork", True)]

    assert not normalize._fanout_traverse_group_is_movable(members)
    before = [list(route.points) for route in (owned, sibling)]
    normalize._bundle_divergent_distinct_traverses(
        [owned, sibling], SimpleNamespace(offset_step=3.0)
    )
    assert [owned.points, sibling.points] == before


def test_bypass_nesting_ownership_is_per_segment_boundary(monkeypatch) -> None:
    wrap = _route("wrap", [(10.0, 40.0), (90.0, 40.0)])
    owned = _route(
        "owned",
        [(0.0, 80.0), (0.0, 60.0), (100.0, 60.0), (100.0, 120.0)],
        owned=(1,),
    )
    movable = _route(
        "movable",
        [(0.0, 80.0), (0.0, 70.0), (100.0, 70.0), (100.0, 120.0)],
    )
    monkeypatch.setattr(
        normalize,
        "iter_inter_row_gaps",
        lambda _graph: [(0, 0.0, 100.0)],
    )
    monkeypatch.setattr(
        normalize,
        "_route_gap_span",
        lambda _graph, route, _top, _bottom: (
            (True, False) if route is wrap else (False, True)
        ),
    )

    assert not normalize._bypass_nesting_leg_is_movable(owned, 1)
    assert normalize._bypass_nesting_leg_is_movable(movable, 1)
    owned_before = list(owned.points)
    normalize._nest_bypass_above_over_top_wrap(
        [wrap, owned, movable], SimpleNamespace(graph=object())
    )

    assert owned.points == owned_before
    assert movable.points == [
        (0.0, 80.0),
        (0.0, 26.0),
        (100.0, 26.0),
        (100.0, 120.0),
    ]
