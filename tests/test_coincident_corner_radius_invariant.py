"""Same-line legs fused onto one channel must share their turn radius.

:func:`_coincide_same_line_tracks` snaps several same-line vertical legs onto
one reference X, so they read as a single stroke.  Each leg carries the
flanking-corner radius its own handler assigned: a plain solo leg gets the base
radius, while a leg that is the outer member of a concentric multi-line bundle
gets a wider one.  Where the fused legs share a turn vertex the two arcs draw
concentrically a few pixels apart -- a doubled corner.

``check_concentric_bundle_corners`` deliberately skips this case (it tests
*offset* bundle-mates, which nest by design); ``check_coincident_corner_radii``
covers it, and :func:`_unify_coincident_corner_radii` snaps every such shared
turn to the widest coincident radius so the fused stroke is one clean arc.

Covers:

* Corpus: no shipped fixture routes a same-line turn shared by two legs with
  unequal resolved radii.
* Meaningfulness: a hand-planted radius mismatch at a coincident corner is
  caught, so the corpus check is not vacuous.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from layout_validator import shared_same_line_turn_vertices

from nf_metro.api import prepare_graph
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.envelope_settlement import settle_route_envelopes
from nf_metro.layout.routing import (
    compute_station_offsets,
    observe_route_edges,
    route_edges_centred,
)
from nf_metro.layout.routing.corners import resolve_curve_radii
from nf_metro.layout.routing.invariants import check_coincident_corner_radii
from nf_metro.layout.routing.normalize import _unify_coincident_corner_radii
from nf_metro.parser.mermaid import parse_metro_mermaid

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPOLOGIES = REPO_ROOT / "tests" / "fixtures" / "topologies"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
EXAMPLES = REPO_ROOT / "examples"

# Fixtures that route a same-line turn shared by two legs, so they genuinely
# exercise the coincident-corner unification rather than passing vacuously.
COINCIDENT_CORNER_FIXTURES = [
    "examples/longread_variant_calling.mmd",
    "examples/topologies/fanout_bundle_plus_spurs.mmd",
    "examples/topologies/merge_trunk_out_of_range_section.mmd",
    "tests/fixtures/target_entry_runway_bypass.mmd",
]


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(TOPOLOGIES.glob("*.mmd")))
    paths.extend(sorted(FIXTURES.glob("*.mmd")))
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    paths.extend(sorted((EXAMPLES / "topologies").glob("*.mmd")))
    return paths


def _route(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    return graph, routes, offsets


def _settled_route(path: Path):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    offsets = compute_station_offsets(graph)
    preflight = observe_route_edges(graph, station_offsets=offsets)
    settlement = settle_route_envelopes(graph, preflight.plan)
    offsets = compute_station_offsets(graph)
    final = observe_route_edges(
        graph,
        station_offsets=offsets,
        envelope_proofs=settlement.capacity_proofs,
        envelope_limitations=settlement.capacity_limitations,
        envelope_reservations=preflight.plan.reservations,
        envelope_bindings=preflight.plan.bindings,
        envelope_identity_projections=settlement.identity_projections,
    )
    return graph, final.routes, offsets


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_no_doubled_coincident_corner(path: Path) -> None:
    """No same-line turn is shared by two legs with unequal resolved radii."""
    graph, routes, offsets = _route(path)
    violations = check_coincident_corner_radii(graph, routes, offsets)
    assert not violations, (
        f"{path.name}: {len(violations)} doubled coincident corner(s); "
        f"first: {violations[0].message()}"
    )


@pytest.mark.parametrize("fixture", COINCIDENT_CORNER_FIXTURES)
def test_named_fixtures_have_a_coincident_turn(fixture: str) -> None:
    """The named fixtures genuinely route a shared same-line turn.

    Guards the corpus sweep against silently going vacuous if a layout change
    stops these fixtures from fusing same-line legs onto a shared corner: with
    no such corner the unification has nothing to equalise and a passing
    ``test_no_doubled_coincident_corner`` would prove nothing here.
    """
    graph, routes, offsets = _settled_route(REPO_ROOT / fixture)
    assert shared_same_line_turn_vertices(routes), (
        f"{fixture} no longer routes a coincident same-line turn"
    )
    assert not check_coincident_corner_radii(graph, routes, offsets)


def _make_route(source: str, target: str, radius: float):
    """An L-shaped same-line route turning at (100, 100) with the given radius."""
    from nf_metro.layout.routing import OffsetRegime
    from nf_metro.layout.routing.common import RoutedPath
    from nf_metro.parser.model import Edge

    return RoutedPath(
        edge=Edge(source=source, target=target, line_id="l"),
        line_id="l",
        points=[(0.0, 100.0), (100.0, 100.0), (100.0, 300.0)],
        is_inter_section=True,
        offset_regime=OffsetRegime.BAKED,
        curve_radii=[radius],
    )


def test_check_reports_unequal_radii_at_shared_turn() -> None:
    """Two same-line routes turning at one vertex with unequal radii are caught.

    Proves the corpus sweep is not vacuous: the check fires on a hand-built
    doubled corner and stays silent once the radii match.
    """
    from nf_metro.parser.model import MetroGraph

    graph = MetroGraph()
    mismatched = [_make_route("a", "x", 10.0), _make_route("b", "x", 18.0)]
    assert check_coincident_corner_radii(graph, mismatched, {})

    matched = [_make_route("a", "x", 14.0), _make_route("b", "x", 14.0)]
    assert not check_coincident_corner_radii(graph, matched, {})


def test_unify_uses_widest_radius_all_shared_legs_can_resolve() -> None:
    """A short lead limits the common radius instead of drawing a double arc."""
    from nf_metro.parser.model import MetroGraph

    short = _make_route("short", "x", 22.0)
    long = _make_route("long", "x", 22.0)
    short.points[0] = (80.0, 100.0)
    long.points[0] = (70.0, 100.0)
    routes = [short, long]
    graph = MetroGraph()

    assert [
        resolve_curve_radii(route.points, route.curve_radii)[0] for route in routes
    ] == [20.0, 22.0]
    assert check_coincident_corner_radii(graph, routes, {})

    _unify_coincident_corner_radii(routes)

    assert [
        resolve_curve_radii(route.points, route.curve_radii)[0] for route in routes
    ] == pytest.approx([20.0, 20.0])
    assert not check_coincident_corner_radii(graph, routes, {})


def test_unify_materialises_the_default_radius_for_a_shared_turn() -> None:
    """An implicit base-radius leg participates in coincident unification."""
    from nf_metro.parser.model import MetroGraph

    explicit = _make_route("explicit", "x", 14.0)
    implicit = _make_route("implicit", "x", 10.0)
    implicit.curve_radii = None
    routes = [explicit, implicit]
    graph = MetroGraph()

    assert check_coincident_corner_radii(graph, routes, {})

    _unify_coincident_corner_radii(routes)

    assert explicit.curve_radii == [14.0]
    assert implicit.curve_radii == [14.0]
    assert not check_coincident_corner_radii(graph, routes, {})


def test_unify_preserves_route_order_and_allocated_segment_ownership() -> None:
    """Radius reconciliation cannot rebind settled channel allocations."""
    first = _make_route("first", "x", 14.0)
    second = _make_route("second", "x", 18.0)
    first.envelope_allocated_segments = ((1, 0, 100.0),)
    second.envelope_allocated_segments = ((1, 0, 100.0),)
    routes = [first, second]
    order = tuple(id(route) for route in routes)
    allocations = tuple(route.envelope_allocated_segments for route in routes)

    _unify_coincident_corner_radii(routes)

    assert tuple(id(route) for route in routes) == order
    assert tuple(route.envelope_allocated_segments for route in routes) == allocations
    assert [
        resolve_curve_radii(route.points, route.curve_radii)[0] for route in routes
    ] == pytest.approx([18.0, 18.0])


def test_unify_materialises_a_missing_default_radius_in_a_sparse_list() -> None:
    """Missing radius entries retain the renderer's base-radius fallback."""
    explicit = _make_route("explicit", "x", 14.0)
    sparse = _make_route("sparse", "x", 10.0)
    for route in (explicit, sparse):
        route.points = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (200.0, 100.0),
        ]
    explicit.curve_radii = [10.0, 14.0]
    sparse.curve_radii = [10.0]

    _unify_coincident_corner_radii([explicit, sparse])

    assert explicit.curve_radii == [10.0, 14.0]
    assert sparse.curve_radii == [10.0, 14.0]


def test_unify_refreshes_a_shared_route_after_changing_one_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second coincident bucket sees an earlier radius correction."""
    import nf_metro.layout.routing.normalize as normalize

    shared = _make_route("shared", "x", 22.0)
    shared.points = [
        (0.0, 100.0),
        (100.0, 100.0),
        (100.0, 200.0),
        (200.0, 200.0),
    ]
    shared.curve_radii = [22.0, 22.0]

    short = _make_route("short", "x", 22.0)
    short.points = [(80.0, 100.0), (100.0, 100.0), (100.0, 300.0)]

    peer = _make_route("peer", "x", 22.0)
    peer.points = [(100.0, 100.0), (100.0, 200.0), (200.0, 200.0)]

    real_resolve = normalize.resolve_curve_radii
    shared_resolutions = 0

    def counting_resolve(points, radii, *args, **kwargs):
        nonlocal shared_resolutions
        if points is shared.points:
            shared_resolutions += 1
        return real_resolve(points, radii, *args, **kwargs)

    monkeypatch.setattr(normalize, "resolve_curve_radii", counting_resolve)
    _unify_coincident_corner_radii([shared, short, peer])

    assert shared.curve_radii == pytest.approx([20.0, 22.0])
    assert shared_resolutions == 3
