"""Convergence ownership boundaries for post-emission normalisation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def _descent(
    source: str,
    target: str,
    line_id: str,
    points: list[tuple[float, float]],
    *,
    frozen: bool = False,
) -> RoutedPath:
    return RoutedPath(
        edge=Edge(source, target, line_id),
        line_id=line_id,
        points=points,
        is_inter_section=True,
        curve_radii=[10.0, 10.0],
        fan_route_emitter="test" if frozen else None,
    )


def _descent_context(
    *, junctions: list[str] = (), merge_junctions: set[str] | None = None
) -> SimpleNamespace:
    graph = SimpleNamespace(ports={}, junctions=junctions)
    merge = SimpleNamespace(junctions=merge_junctions or set())
    return SimpleNamespace(
        graph=graph,
        merge=merge,
        offset_step=3.0,
        curve_radius=10.0,
    )


@pytest.mark.parametrize(
    ("source", "junctions", "expected_moves"),
    [
        ("renamed-generated-fork", ["renamed-generated-fork"], 2),
        ("__junction_authored", [], 0),
    ],
)
def test_frozen_fan_descent_ownership_uses_graph_junction_membership(
    monkeypatch, source, junctions, expected_moves
) -> None:
    routes = [
        _descent(
            source,
            "left",
            "a",
            [(0.0, 0.0), (10.0, 0.0), (10.0, 40.0), (30.0, 40.0)],
            frozen=True,
        ),
        _descent(
            source,
            "right",
            "b",
            [(0.0, 0.0), (13.0, 0.0), (13.0, 20.0), (30.0, 20.0)],
            frozen=True,
        ),
    ]
    moves: list[tuple[str, float]] = []
    monkeypatch.setattr(normalize, "_descent_crosses_section", lambda *_args: False)
    monkeypatch.setattr(
        normalize,
        "_set_vchannel_x",
        lambda channel, target_x, *_args, **_kwargs: moves.append(
            (channel.route.line_id, target_x)
        ),
    )

    normalize._bundle_divergent_distinct_descents(
        routes,
        _descent_context(junctions=junctions),
        settle_frozen_arcs=True,
    )

    assert len(moves) == expected_moves


@pytest.mark.parametrize(
    ("merge_target", "merge_junctions", "expected_order"),
    [
        ("renamed-generated-merge", {"renamed-generated-merge"}, ["a", "b"]),
        ("__merge_authored", set(), ["b", "a"]),
    ],
)
def test_descent_order_uses_merge_junction_membership(
    monkeypatch, merge_target, merge_junctions, expected_order
) -> None:
    routes = [
        _descent(
            "fork",
            merge_target,
            "a",
            [(0.0, 0.0), (10.0, 0.0), (10.0, 40.0), (30.0, 40.0)],
        ),
        _descent(
            "fork",
            "direct-target",
            "b",
            [(0.0, 10.0), (20.0, 10.0), (20.0, 20.0), (30.0, 20.0)],
        ),
    ]
    moves: list[tuple[str, float]] = []
    monkeypatch.setattr(normalize, "_descent_crosses_section", lambda *_args: False)
    monkeypatch.setattr(
        normalize,
        "_set_vchannel_x",
        lambda channel, target_x, *_args, **_kwargs: moves.append(
            (channel.route.line_id, target_x)
        ),
    )

    normalize._bundle_divergent_distinct_descents(
        routes,
        _descent_context(merge_junctions=merge_junctions),
    )

    assert [
        line_id for line_id, _target_x in sorted(moves, key=lambda item: item[1])
    ] == expected_order
