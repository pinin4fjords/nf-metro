"""Tests for the bundle-order-preservation invariant.

Covers:

* Happy-path: every gallery fixture and example yields zero violations
  when passed through :func:`check_bundle_order_preserved`.
* Route-level negative: a synthetic ``RoutedPath`` pair with a
  hand-crafted flipped corner correctly surfaces as a violation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import (
    OffsetRegime,
    compute_station_offsets,
    route_edges,
)
from nf_metro.layout.routing.common import (
    Direction,
    RoutedPath,
    apply_route_offsets,
)
from nf_metro.layout.routing.invariants import (
    BundleOrderViolation,
    Side,
    check_bundle_order_preserved,
    check_shared_run_turn_preserves_bundle_order,
    check_tb_exit_corner_preserves_column_order,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
TOPOLOGIES = FIXTURES / "topologies"
EXAMPLES = REPO_ROOT / "examples"
_CURVE_REPROS = FIXTURES / "curve_invariant_repros"
_HASH_SEEDS = FIXTURES / "hash_seed_determinism"

_TWO_MOVABLE_MEMBERS = (
    "one heading whose members ALL move seats as a single gap bundle, whose "
    "line order the crossing minimiser picks from the deep-end turn-offs "
    "rather than from the lateral order the members held on the shared run"
)

# Fixtures each corpus sweep below knows to carry a violation of its own rule,
# with what that fixture carries.  Naming a carrier keeps it IN the sweep under
# a strict mark, so the day its defect is fixed the sweep reds, rather than its
# whole tree sitting outside the corpus unwatched.
_KNOWN_BUNDLE_ORDER_CARRIERS: dict[str, str] = {
    "seed_15.mmd": (
        "an exit-port bundle turns L->D onto columns that contradict the "
        "incoming run's order, crossing the pair inside the corner"
    ),
    "seed_41.mmd": (
        "an exit-port bundle turns U->L onto columns that contradict the "
        "incoming run's order, crossing the pair inside the corner"
    ),
}
_KNOWN_SHARED_RUN_CARRIERS: dict[str, str] = {
    "seed_15.mmd": _TWO_MOVABLE_MEMBERS,
    "seed_41.mmd": _TWO_MOVABLE_MEMBERS,
    "seed_72.mmd": _TWO_MOVABLE_MEMBERS,
    "seed_77.mmd": _TWO_MOVABLE_MEMBERS,
}


# ---------------------------------------------------------------------------
# Happy-path: every fixture and example must pass the invariant
# ---------------------------------------------------------------------------


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(FIXTURES.glob("*.mmd")))
    paths.extend(sorted(TOPOLOGIES.glob("*.mmd")))
    paths.extend(sorted(_CURVE_REPROS.glob("*.mmd")))
    paths.extend(sorted(_HASH_SEEDS.glob("*.mmd")))
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    paths.extend(sorted((EXAMPLES / "topologies").glob("*.mmd")))
    return paths


def _sweep(carriers: dict[str, str] | None = None) -> list[ParameterSet]:
    """The corpus as parameters, strict-xfailing this sweep's known carriers."""
    known = carriers or {}
    return [
        pytest.param(
            path,
            id=path.relative_to(REPO_ROOT).as_posix(),
            marks=(
                [pytest.mark.xfail(strict=True, reason=known[path.name])]
                if path.name in known
                else []
            ),
        )
        for path in _gather_fixtures()
    ]


@pytest.mark.parametrize("path", _sweep(_KNOWN_BUNDLE_ORDER_CARRIERS))
def test_no_bundle_order_violations_in_gallery(path: Path) -> None:
    """Every shipped topology and example must route without a
    bundle-order violation.

    This is the corpus-level happy-path check.  A regression to a
    routing handler that creates a flipped concentric bundle would
    cause exactly one fixture to start failing here.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    violations = check_bundle_order_preserved(routes)
    assert violations == [], (
        f"{path.name}: {len(violations)} bundle-order violation(s); "
        f"first: {violations[0].message() if violations else ''}"
    )


# ---------------------------------------------------------------------------
# Cross-column perpendicular-exit -> perpendicular-entry bundles
# ---------------------------------------------------------------------------

_CROSS_COL_PERP_ENTRY_FIXTURES = [
    "lr_perp_top_exit_perp_entry",
    "lr_perp_bottom_exit_perp_entry",
    "lr_perp_top_exit_perp_entry_diverging",
]


@pytest.mark.parametrize("stem", _CROSS_COL_PERP_ENTRY_FIXTURES)
def test_cross_column_perp_entry_preserves_bundle_order(stem: str) -> None:
    """A co-travelling bundle taken over the corridor from a perpendicular
    exit on one LR section into the perpendicular entry of another LR
    section in a different column keeps a single left/right order through
    the whole riser -> corridor -> entry-drop chain.

    The entry-drop leg's per-line channel order must agree with the
    corridor's descent order; a disagreement flips the bundle at the
    entry -> first-station corner, which both trips the runtime guard
    and renders a crossover at the drop.
    """
    path = EXAMPLES / "topologies" / f"{stem}.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    violations = check_bundle_order_preserved(routes)
    assert violations == [], (
        f"{stem}: {len(violations)} bundle-order violation(s); "
        f"first: {violations[0].message() if violations else ''}"
    )


# ---------------------------------------------------------------------------
# Left-entry wrap: bundle order preserved in both riser directions
# ---------------------------------------------------------------------------

_LEFT_ENTRY_WRAP = """\
%%metro title: Left-entry wrap {direction}
%%metro style: dark
%%metro line: w1 | W1 | #e63946
%%metro line: w2 | W2 | #0570b0
%%metro grid: left_tgt | 0,{tgt_row}
%%metro grid: right_src | 1,{src_row}
graph LR
    subgraph left_tgt [Left Target]
        %%metro entry: left | w1, w2
        lt1[Collect]
        lt2[Output]
        lt1 -->|w1,w2| lt2
    end
    subgraph right_src [Right Source]
        rs1[Input R]
        rs2[Hub R]
        rs1 -->|w1,w2| rs2
    end
    rs2 -->|w1,w2| lt1
"""


@pytest.mark.parametrize(
    ("direction", "tgt_row", "src_row"),
    [("up", 0, 1), ("down", 1, 0)],
)
def test_left_entry_wrap_preserves_bundle_order(
    direction: str, tgt_row: int, src_row: int
) -> None:
    """A multi-line bundle wrapping into a LEFT entry keeps one left/right
    order around the wrap, whether the riser climbs (source below the
    target) or descends (source above it).

    Both wrap runs go rightward, so the order is fixed by the port-offset
    stacking rather than the riser's vertical direction.
    """
    text = _LEFT_ENTRY_WRAP.format(
        direction=direction, tgt_row=tgt_row, src_row=src_row
    )
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    violations = check_bundle_order_preserved(routes)
    assert violations == [], (
        f"left-entry {direction}-wrap: {len(violations)} bundle-order "
        f"violation(s); first: {violations[0].message() if violations else ''}"
    )


# ---------------------------------------------------------------------------
# TB exit-corner column-order preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _sweep())
def test_no_tb_exit_corner_column_flips_in_gallery(path: Path) -> None:
    """No shipped fixture turns a TB section's bundle out through a LEFT/RIGHT
    exit in an order that disagrees with its in-section vertical column.

    A disagreement crosses the two lines at the feeder station marker.  A
    regression to the exit-corner drop X, the exit-port Y order, or the
    downstream reversal flag would surface as exactly one fixture failing here.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    violations = check_tb_exit_corner_preserves_column_order(graph, routes, offsets)
    assert violations == [], (
        f"{path.name}: {len(violations)} TB exit-corner column flip(s); "
        f"first: {violations[0].message() if violations else ''}"
    )


_TB_EXIT_CORNER_FIXTURES = [
    EXAMPLES / "topologies" / "junction_entry_reversed_fold.mmd",
    EXAMPLES / "guide" / "04_directions.mmd",
]


@pytest.mark.parametrize("path", _TB_EXIT_CORNER_FIXTURES, ids=lambda p: p.stem)
def test_tb_exit_corner_continues_column_and_keeps_bundle_order(path: Path) -> None:
    """A TB-section bundle exiting LEFT/RIGHT keeps a single order through the
    reversal corner: it continues the in-section column (no crossing at the
    feeder station) and stays bundle-order consistent through every turn.

    Both fixtures route a multi-line bundle out of a TB section through a
    reversal corner into a downstream section, the case where the exit-corner
    drop X, the exit-port Y order, and the downstream reversal flag must agree.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    corner = check_tb_exit_corner_preserves_column_order(graph, routes, offsets)
    assert corner == [], (
        f"{path.name}: {len(corner)} TB exit-corner column flip(s); "
        f"first: {corner[0].message() if corner else ''}"
    )
    bundle = check_bundle_order_preserved(routes)
    assert bundle == [], (
        f"{path.name}: {len(bundle)} bundle-order violation(s); "
        f"first: {bundle[0].message() if bundle else ''}"
    )


# ---------------------------------------------------------------------------
# Shared-run turn order preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _sweep(_KNOWN_SHARED_RUN_CARRIERS))
def test_no_shared_run_turn_flips_in_gallery(path: Path) -> None:
    """No shipped fixture turns lines off one shared run onto crossing columns.

    Lines leaving a source bundled on one horizontal run are one bundle while
    they share it, so the column each turns down at must keep that bundle's
    order even where the routes head for different targets.  A regression to the
    gap-bundle seating, to a coincidence fusion's reference column, or to a
    port's lane order would surface as exactly one fixture failing here.
    """
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    violations = check_shared_run_turn_preserves_bundle_order(routes, offsets)
    assert violations == [], (
        f"{path.name}: {len(violations)} shared-run turn flip(s); "
        f"first: {violations[0].message() if violations else ''}"
    )


def test_exit_run_drop_columns_nest_across_three_handlers() -> None:
    """Three lines leaving one exit port turn onto correctly nested columns.

    The bundle out of section C's right exit carries ``main`` on top, then
    ``report``, then ``sheets``, and the three are routed onward by three
    different inter-section handlers -- a direct inter-row L-shape and two
    bypass-trunk feeders.  A rightward run turning down is concentric, so the
    line highest on the run turns at the largest x; nothing reconciling the
    three handlers' column choices lets ``main`` and ``report`` transpose and
    cross twice, once inside the exit arc and again entering the target.

    ``report`` reaches this gap twice -- once on a long bypass whose handler owns
    its own channel, once as a feeder off the exit junction -- and a coincidence
    fusion later pulls the feeder onto the bypass's column.  Seating the bundle
    on that owned column is what keeps ``main`` clear of it.
    """
    path = EXAMPLES / "topologies" / "exit_run_three_drop_columns.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)

    turns = {
        rp.line_id: (pts[0][1], pts[1][0])
        for rp in routes
        if rp.edge.source == "__junction_9"
        and len(pts := apply_route_offsets(rp, offsets)) >= 3
        and abs(pts[1][1] - pts[0][1]) <= COORD_TOLERANCE
        and abs(pts[2][0] - pts[1][0]) <= COORD_TOLERANCE
        and pts[2][1] > pts[1][1]
    }
    assert {"main", "report"} <= set(turns), turns
    run_y_main, turn_x_main = turns["main"]
    run_y_report, turn_x_report = turns["report"]
    assert run_y_main < run_y_report, turns
    assert turn_x_main > turn_x_report, (
        f"main runs above report (y={run_y_main} vs {run_y_report}) so it must "
        f"turn down at the larger x, but turned at {turn_x_main} vs {turn_x_report}"
    )

    flips = check_shared_run_turn_preserves_bundle_order(routes, offsets)
    at_exit = [v for v in flips if v.source_id == "__junction_9"]
    assert at_exit == [], (
        f"{len(at_exit)} shared-run turn flip(s) at the exit junction; first: "
        f"{at_exit[0].message() if at_exit else ''}"
    )


# Each fixture below names one heading whose ladder an exempt handler states
# and whose other member the gap seating places, so the rule is pinned to a
# named source rather than left to the whole-corpus sweeps above.
@pytest.mark.parametrize(
    ("relative_path", "source_id"),
    [
        (_CURVE_REPROS / "rl_return_row_convergence.mmd", "__junction_16"),
        (_HASH_SEEDS / "seed_77.mmd", "__junction_35"),
    ],
    ids=["rl_return_row_convergence", "seed_77"],
)
def test_return_row_gap_seating_keeps_the_headings_lateral_order(
    relative_path: Path, source_id: str
) -> None:
    """A gap-seated ladder turns onto columns running with its lateral order.

    Lines leaving one source on one horizontal run turn down onto a ladder of
    columns whose progression is the turn's ``lateral_order_sign``: rightward
    into a downturn, the line lower on the run is inside the bend and turns
    first, at the smaller x.  The member whose handler owns its column states
    where that ladder sits, and the gap carrying the other member's descent
    seats it on the column the ladder gives it rather than centring it.
    """
    graph = parse_metro_mermaid(relative_path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    flips = [
        violation
        for violation in check_shared_run_turn_preserves_bundle_order(routes, offsets)
        if violation.source_id == source_id
    ]
    assert flips == [], (
        f"{len(flips)} shared-run turn flip(s) out of {source_id!r}; first: "
        f"{flips[0].message() if flips else ''}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The descent lanes out of a bottom exit are seated by line priority, "
        "so the member continuing deepest can hold an inner column and its "
        "siblings' leftward turns cut across the descent it is still drawing."
    ),
)
def test_descending_exit_bundle_seats_the_deepest_member_outermost() -> None:
    """A descending bundle's peel-offs clear the sibling running deeper.

    Three lanes leave ``integration`` together and turn west at three depths.
    A lane turning west sweeps every column to its left, so a lane whose
    descent continues past that depth has to stand to its right: the descent
    columns read in landing-depth order, deepest outermost, and the turns nest.
    """
    path = EXAMPLES / "topologies" / "fold_stacked_branch.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)

    descents = {
        rp.line_id: (pts[0][0], pts[1][1])
        for rp in routes
        if rp.edge.source == "__junction_15"
        and len(pts := apply_route_offsets(rp, offsets)) >= 3
        and abs(pts[1][0] - pts[0][0]) <= COORD_TOLERANCE
        and pts[1][1] > pts[0][1]
        and pts[2][0] < pts[1][0]
    }
    assert {"rna", "atac", "protein"} <= set(descents), descents
    by_depth = sorted(descents, key=lambda line_id: descents[line_id][1])
    columns = [descents[line_id][0] for line_id in by_depth]
    assert columns == sorted(columns), (
        "lanes turning west earlier must stand left of the lanes still "
        f"descending, but landing-depth order {by_depth} draws columns {columns}"
    )


# ---------------------------------------------------------------------------
# Route-level negative test: a synthetic flipped corner is caught
# ---------------------------------------------------------------------------


def _synthetic_route(line_id: str, points: list[tuple[float, float]]) -> RoutedPath:
    """Build a ``RoutedPath`` from a points list for testing.

    Source/target IDs are fixed (``'__src__'``, ``'__tgt__'``) so the
    paths share a bundle key.  The ``Edge`` carries the line id; the
    rest of the routing metadata is irrelevant to
    :func:`check_bundle_order_preserved`.
    """
    return RoutedPath(
        edge=Edge(source="__src__", target="__tgt__", line_id=line_id),
        line_id=line_id,
        points=points,
        is_inter_section=True,
        offset_regime=OffsetRegime.BAKED,
    )


def test_check_skips_clean_bundle() -> None:
    """Two paths that share waypoints exactly produce zero violations:
    the COINCIDENT path-pair has nothing to compare on either side.
    """
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (200.0, 100.0)]
    routes = [_synthetic_route("A", pts), _synthetic_route("B", pts)]
    assert check_bundle_order_preserved(routes) == []


def test_check_skips_single_line_bundle() -> None:
    """A bundle with only one line has no pairs to compare; no
    violation is possible.
    """
    pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (200.0, 100.0)]
    routes = [_synthetic_route("A", pts)]
    assert check_bundle_order_preserved(routes) == []


def test_synthetic_flipped_corner_is_caught() -> None:
    """A hand-crafted bundle with a deliberate flip at a near-shared
    corner surfaces as a :class:`BundleOrderViolation`.

    Two L-shape routes whose elbows are half a pixel apart on both
    axes: A is on the LEFT of B going east, then on the RIGHT going
    south.  LEFT -> RIGHT is exactly the flip the invariant exists to
    catch.
    """
    a_pts = [
        (0.0, 100.0),
        (100.0, 100.0),
        (100.0, 200.0),
    ]
    b_pts = [
        (0.0, 100.5),
        (100.5, 100.5),
        (100.5, 200.0),
    ]
    routes = [_synthetic_route("A", a_pts), _synthetic_route("B", b_pts)]
    violations = check_bundle_order_preserved(routes)
    assert violations, "expected a synthetic bundle-order violation; got an empty list"
    v = violations[0]
    assert v.line_a == "A" and v.line_b == "B"
    assert v.in_tangent is Direction.R
    assert v.out_tangent is Direction.D
    assert {v.before, v.after} == {Side.LEFT, Side.RIGHT}, v.message()


def test_violation_message_self_describing() -> None:
    """The violation's ``message()`` includes the corner xy, line ids,
    tangent directions, and the offending before/after sides - the
    fields downstream callers (the engine guard and CI logs) rely on
    for diagnosis.
    """
    v = BundleOrderViolation(
        edge_source="src",
        edge_target="tgt",
        line_a="alpha",
        line_b="beta",
        corner_xy=(100.0, 200.0),
        in_tangent=Direction.D,
        out_tangent=Direction.L,
        before=Side.LEFT,
        after=Side.RIGHT,
    )
    msg = v.message()
    assert "100.0" in msg and "200.0" in msg
    assert "alpha" in msg and "beta" in msg
    assert "D" in msg and "L" in msg
    assert "LEFT" in msg and "RIGHT" in msg


def test_disjoint_runs_sharing_a_name_are_free_to_differ_at_the_two_ends() -> None:
    """An inherited exit slot binds a stroke that spans the section.

    In this section the entry bundle terminates inside and separate runs of the
    same lines leave it, so the two ends share names but no stroke: holding the
    exit to the entry's order would constrain geometry that is not connected.
    Chaining the two halves gives each name one stroke end to end, and the same
    offsets then are a violation.

    This fixture's entry and exit also sit on one materialized linear-entry
    frame, which threads first/second/third identically end to end for an
    unrelated, separately-tested reason (see
    ``test_linear_entry_cohort_keeps_one_lane_frame``): the offsets
    :func:`compute_station_offsets` hands back naturally agree at both ports.
    The exit port's offsets are overridden below (reversing the entry order)
    so this test exercises the disjoint-run exemption on its own, independent
    of that separate frame invariant.
    """
    from nf_metro.layout.routing.invariants import (
        check_exit_inherits_entry_bundle_order,
    )

    path = EXAMPLES / "topologies" / "target_lane_transition.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph, validate=True)
    offsets = dict(compute_station_offsets(graph))
    entry_id = "source__entry_left_2"
    exit_id = "source__exit_right_1"
    shared = ("first", "second", "third")

    entry_order = [offsets[(entry_id, line_id)] for line_id in shared]
    exit_keys = ((exit_id, line_id) for line_id in shared)
    offsets.update(zip(exit_keys, reversed(entry_order)))

    assert [offsets[(entry_id, line_id)] for line_id in shared] != [
        offsets[(exit_id, line_id)] for line_id in shared
    ]
    assert not check_exit_inherits_entry_bundle_order(graph, offsets)

    for line_id in shared:
        graph.add_edge(Edge("enter", "leave", line_id))

    violations = check_exit_inherits_entry_bundle_order(graph, offsets)
    assert [item.exit_port for item in violations] == [exit_id]
