"""Tests for the fused co-travelling distinct-line invariant.

Two DIFFERENT lines running the same way along one corridor nest a full
``OFFSET_STEP`` apart, which is what leaves a hairline of background showing
between their strokes.  Closed to less than that they paint one two-tone
stripe and one of the two lines is not there to read.

The defect can appear in planning and again on the settled re-route, because
both stages place tracks inside shared reservation bands. The reported fixtures
are exercised through the render chokepoint rather than a single
``route_edges`` call.

The checker observes plan-owned tracks as well as tracks that can be re-seated.
An immutable violation is attributed to the route system and plans that should
have separated it before geometry froze.

Covers:

* Happy-path: every shipped topology and example routes with no fused pair.
* Targeted: the three corridors a reservation band pulled together
  (``rl_return_row_convergence``, ``convergence_fold_diamond``,
  ``seed72_cross_family_fan``) keep the full step on the settled geometry.
* Meaningfulness: on the fixtures whose bands leave the pair short of the step
  the tracks fuse once both separation stages are disabled, and settlement
  lands each pair exactly on the step rather than merely clear of the check.
* Corpus ratchet: the checker has no silent fused-pair exemptions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import nf_metro.layout.routing.convergences as convergences
import nf_metro.layout.routing.core as routing_core
import nf_metro.layout.routing.invariants as invariants
import nf_metro.layout.routing.member_geometry as member_geometry
import nf_metro.layout.routing.normalize as normalize
from nf_metro.layout.constants import graph_offset_step
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases._common import routes_through_unrelated_sections
from nf_metro.layout.routing import (
    compute_station_offsets,
    observe_route_edges,
    route_edges,
)
from nf_metro.layout.routing.common import (
    OffsetRegime,
    RoutedPath,
    apply_route_offsets,
)
from nf_metro.layout.routing.invariants import (
    _routes_crossings,
    check_no_fused_cotravelling_lines,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge, MetroGraph

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPOLOGIES = REPO_ROOT / "tests" / "fixtures" / "topologies"
EXAMPLES = REPO_ROOT / "examples"
EXAMPLE_TOPOLOGIES = EXAMPLES / "topologies"
CURVE_REPROS = REPO_ROOT / "tests" / "fixtures" / "curve_invariant_repros"
REGRESSIONS = REPO_ROOT / "tests" / "fixtures" / "regressions"
THROUGH_SECTION = REPO_ROOT / "tests" / "fixtures" / "through_section"
FROZEN_FUZZ = REPO_ROOT / "tests" / "fixtures" / "hash_seed_determinism"

REPORTED = {
    CURVE_REPROS / "rl_return_row_convergence.mmd": frozenset(
        {("bam", "other", "Y"), ("bam", "snvvcf", "Y")}
    ),
    EXAMPLE_TOPOLOGIES / "convergence_fold_diamond.mmd": frozenset(
        {("left_path", "right_path", "X")}
    ),
    EXAMPLE_TOPOLOGIES / "seed72_cross_family_fan.mmd": frozenset(
        {("exempt", "normal", "X"), ("exempt", "normal", "Y")}
    ),
}

FUSED_WITHOUT_THE_PASS = {
    EXAMPLE_TOPOLOGIES / "packed_multiline_serpentine_grid.mmd": frozenset(
        {("l1", "l2", "X")}
    ),
    REGRESSIONS / "entry_trunk_row_bow.mmd": frozenset({("l1", "l2", "Y")}),
    THROUGH_SECTION / "riboseq_packed_lr.mmd": frozenset({("riboseq", "rnaseq", "X")}),
}


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(TOPOLOGIES.glob("*.mmd")))
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    return paths


def _corpus() -> list[Path]:
    paths = sorted(EXAMPLES.rglob("*.mmd"))
    paths += sorted((REPO_ROOT / "tests" / "fixtures").rglob("*.mmd"))
    paths += sorted((REPO_ROOT / "tests" / "fixtures").rglob("*.metro"))
    return paths


_CORPUS = _corpus()

_ExemptPair = tuple[str, str, str, str, float, float]

# The live checker has no semantic exemptions. The empty ledger makes any silent
# corpus pair a test failure.
EXEMPT_FUSED_PAIRS: frozenset[_ExemptPair] = frozenset()


def _route(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    return graph, routes, offsets


def _settled(path: Path, monkeypatch: pytest.MonkeyPatch):
    """The geometry the renderer draws, plus the violations the chokepoint saw.

    The check is replaced by a recording stand-in that reports nothing, so the
    render runs to completion on a fixture carrying the defect and the test can
    measure its final geometry rather than only catch the abort.
    The collinearity guard is suppressed because it detects the same induced
    bad geometry first and would prevent that measurement.
    """
    from nf_metro.api import prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan

    final: list[tuple] = []

    def spy(graph, routes, offsets):
        found = check_no_fused_cotravelling_lines(graph, routes, offsets)
        final.clear()
        final.append((graph, routes, offsets, found))
        return []

    monkeypatch.setattr(invariants, "check_no_fused_cotravelling_lines", spy)
    monkeypatch.setattr(
        invariants, "check_collinear_distinct_lines", lambda *_args, **_kwargs: []
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        build_observed_render_plan(graph, resolve_theme(None, graph))
    assert final, "the render chokepoint never ran the check"
    return final[0]


def _disable_separation_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        convergences,
        "_pack_cotravelling_corridor_runs",
        lambda plans, graph, member_runs: plans,
    )
    monkeypatch.setattr(
        convergences,
        "_separate_distinct_cotravelling_trunks",
        lambda plans, graph, member_runs: plans,
    )
    monkeypatch.setattr(
        convergences,
        "_settle_reserved_trunk_axes",
        lambda plans, graph, ctx, member_runs: plans,
    )
    monkeypatch.setattr(
        convergences,
        "_repack_crowded_gap_channels",
        lambda plans, graph, curve_radius, fixed_channels, lane_obstacles=(): plans,
    )
    monkeypatch.setattr(
        convergences,
        "_separate_distinct_terminal_gap_channels",
        lambda plans, graph, curve_radius: plans,
    )
    monkeypatch.setattr(
        routing_core,
        "_separate_fused_cotravelling_runs",
        lambda routes, ctx, **kwargs: None,
    )
    monkeypatch.setattr(
        routing_core,
        "_stagger_convergent_distinct_lines",
        lambda routes, ctx, **kwargs: set(),
    )
    monkeypatch.setattr(
        member_geometry,
        "_separate_fused_cotravelling_runs",
        lambda routes, ctx, **kwargs: None,
    )
    monkeypatch.setattr(
        member_geometry,
        "_stagger_convergent_distinct_lines",
        lambda routes, ctx, **kwargs: set(),
    )


def _pair_separations(routes, offsets) -> dict[tuple[str, str, str], float]:
    """Lateral separation of every co-travelling distinct-line track pair."""
    from nf_metro.layout.routing.common import (
        apply_route_offsets,
        corridor_lanes,
        corridor_runs,
    )

    lanes = corridor_lanes(
        run
        for rp in routes
        if rp.is_inter_section
        for run in corridor_runs(rp, apply_route_offsets(rp, offsets))
    )
    out: dict[tuple[str, str, str], float] = {}
    for i, first in enumerate(lanes):
        for second in lanes[i + 1 :]:
            if first.axis != second.axis or first.sign != second.sign:
                continue
            if first.line_id == second.line_id:
                continue
            if not any(
                max(mine.span[0], theirs.span[0]) < min(mine.span[1], theirs.span[1])
                for mine in first.runs
                for theirs in second.runs
            ):
                continue
            axis = "X" if first.axis == 0 else "Y"
            key = tuple(sorted((first.line_id, second.line_id))) + (axis,)
            separation = abs(first.coord - second.coord)
            if key not in out or separation < out[key]:
                out[key] = separation  # type: ignore[index]
    return out  # type: ignore[return-value]


def _longest_horizontal_run(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    horizontal = [
        (abs(end[0] - start[0]), start[1])
        for start, end in zip(points, points[1:], strict=False)
        if start[1] == end[1]
    ]
    assert horizontal, "route has no horizontal run"
    return max(horizontal, key=lambda item: item[0])


def _production_routes(path: Path):
    """Routes and drawn polylines from the final production render pass."""
    from nf_metro.api import prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    return graph, tuple(zip(observed.plan.routes, observed.plan.route_polylines))


def _pair_identity(
    axis: str, first: tuple[str, float], second: tuple[str, float]
) -> tuple[str, str, str, float, float]:
    """One fused pair, ordered by line id so either scan order names it alike."""
    low, high = sorted((first, second))
    return (axis, low[0], high[0], round(low[1], 4), round(high[1], 4))


def _unreported_fused_pairs(path: Path) -> set[tuple[str, str, str, float, float]]:
    """The fused pairs the checker sees at *path*'s chokepoint and does not report.

    Measured as the difference between every co-travelling distinct-line pair
    drawn within one nesting step and the pairs the checker returns, so what is
    pinned is the checker's own silence rather than a restatement of the
    predicate producing it.  Read on the geometry the chokepoint is handed, which
    a fixture reaches whether or not the render then aborts on another guard.
    """
    from nf_metro.layout.routing.common import (
        apply_route_offsets,
        corridor_lanes,
        corridor_runs,
    )

    found: set[tuple[str, str, str, float, float]] = set()
    real = invariants.check_no_fused_cotravelling_lines

    def spy(graph, routes, offsets):
        step = graph_offset_step(graph)
        lanes = corridor_lanes(
            run
            for rp in routes
            if rp.is_inter_section
            for run in corridor_runs(rp, apply_route_offsets(rp, offsets))
        )
        fused = {
            _pair_identity(
                "X" if first.axis == 0 else "Y",
                (first.line_id, first.coord),
                (second.line_id, second.coord),
            )
            for i, first in enumerate(lanes)
            for second in lanes[i + 1 :]
            if first.fused_span(second, step) is not None
        }
        violations = real(graph, routes, offsets)
        found.update(
            fused
            - {
                _pair_identity(
                    item.axis,
                    (item.first_line, item.first_coord),
                    (item.second_line, item.second_coord),
                )
                for item in violations
            }
        )
        return violations

    from nf_metro.api import prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan

    invariants.check_no_fused_cotravelling_lines = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
            build_observed_render_plan(graph, resolve_theme(None, graph))
    except Exception:  # noqa: BLE001 - the check ran before any abort
        pass
    finally:
        invariants.check_no_fused_cotravelling_lines = real
    return found


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(REPO_ROOT)) for p in _CORPUS]
)
def test_the_fused_pairs_the_checker_exempts_are_the_recorded_ones(path: Path) -> None:
    """The population the plan-owned exemption hides does not grow unnoticed.

    The checker's guarantee is "no two distinct lines are drawn as one stroke",
    minus the pairs it cannot get repaired.  Left unenumerated, that subtraction
    can absorb new fused pairs while the suite stays green and the guarantee
    reads as unconditional.
    """
    rel = str(path.relative_to(REPO_ROOT))
    found = {(rel, *item) for item in _unreported_fused_pairs(path)}
    expected = {item for item in EXEMPT_FUSED_PAIRS if item[0] == rel}
    assert found == expected, (
        "the fused pairs the checker declines to report are not the ones "
        f"recorded: unrecorded {sorted(found - expected)}, recorded but now "
        f"separated {sorted(expected - found)}. A new one is two lines drawn "
        "over each other that no guard will mention; one that separated means "
        "dropping its EXEMPT_FUSED_PAIRS entry"
    )


def test_every_recorded_exempt_pair_names_a_corpus_fixture() -> None:
    """A stale entry would silently excuse a fixture that no longer exists."""
    corpus = {str(item.relative_to(REPO_ROOT)) for item in _CORPUS}
    named = {item[0] for item in EXEMPT_FUSED_PAIRS}
    assert named <= corpus, named - corpus


def test_seed_41_plan_owned_trunks_keep_the_nesting_step() -> None:
    """Global convergence settlement separates independently planned trunks."""
    path = REPO_ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_41.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    observed = observe_route_edges(graph, station_offsets=offsets)
    separations = _pair_separations(observed.routes, offsets)

    assert separations[("l0", "l2", "Y")] >= graph_offset_step(graph)


def test_seed_77_merge_trunk_stays_in_its_inter_row_corridor() -> None:
    """A convergence trunk does not cross the destination row's stations."""
    path = FROZEN_FUZZ / "seed_77.mmd"
    _graph, routes, _offsets = _route(path)
    trunk = next(
        route
        for route in routes
        if (route.edge.source, route.edge.target, route.line_id)
        == ("__junction_42", "__merge_12", "l4")
    )

    assert trunk.points[2] == pytest.approx((2706.0, 356.0))


def test_seed_41_separated_trunk_lane_stays_out_of_the_section_boxes() -> None:
    """Distinct-line separation reseats a trunk in a corridor, not in a box.

    ``__junction_32``'s l3 trunk reaches five columns west along its own row,
    and the lane it is separated onto has to be one a run may occupy: the gap
    above the row carries it clear of every box between the two ends.
    """
    path = FROZEN_FUZZ / "seed_41.mmd"
    graph, routes, offsets = _route(path)
    trunk = next(
        route
        for route in routes
        if (route.edge.source, route.edge.target, route.line_id)
        == ("__junction_32", "__merge_17", "l3")
    )

    assert trunk.points[2] == pytest.approx((706.0, 546.0))
    assert not routes_through_unrelated_sections(graph, routes=[trunk], offsets=offsets)


def test_seed_77_settled_entry_bundle_keeps_allocation_lanes() -> None:
    """A settled member carries every destination peer's allocation lane."""
    _graph, routes, _offsets = _route(FROZEN_FUZZ / "seed_77.mmd")
    l1 = next(
        route
        for route in routes
        if (route.edge.source, route.edge.target, route.line_id)
        == ("__junction_39", "s9__entry_right_25", "l1")
    )
    l3 = next(
        route
        for route in routes
        if (route.edge.source, route.edge.target, route.line_id)
        == ("__junction_41", "s9__entry_right_25", "l3")
    )

    assert l3.points[-3][0] - l1.points[-3][0] == graph_offset_step(_graph)


def test_seed_77_turning_member_stays_off_straight_continuation_lane() -> None:
    """A planned source turn retains its lane beside a straight continuation."""
    from nf_metro.api import RenderConfig, _emit_svg_plan, prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan
    from nf_metro.render.validate import OFFSET_COLLAPSE, validate_render

    path = FROZEN_FUZZ / "seed_77.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    svg = _emit_svg_plan(graph, observed.plan, RenderConfig())
    routes = {
        (route.edge.source, route.edge.target, route.line_id): route
        for route in observed.plan.routes
    }
    straight = routes[("__junction_36", "s4__entry_right_21", "l1")]
    turning = routes[("__junction_42", "__merge_12", "l4")]

    assert turning.points[2][0] - straight.points[3][0] == 3 * graph_offset_step(graph)
    assert not [
        finding
        for finding in validate_render(svg, plan=observed.plan)
        if finding.kind == OFFSET_COLLAPSE
    ]


def test_seed_15_terminal_openings_keep_the_nesting_step() -> None:
    """Late same-line fusion preserves the lanes of distinct terminal feeds."""
    graph, routes, offsets = _route(FROZEN_FUZZ / "seed_15.mmd")
    separations = _pair_separations(routes, offsets)

    assert separations[("l0", "l2", "X")] == graph_offset_step(graph)


def test_seed_77_candidate_executor_completes() -> None:
    """The full production executor accepts the settled seed_77 route."""
    from nf_metro.candidate_executor import (
        CandidateExecutionRequest,
        CandidateStage,
        CandidateStatus,
        ExecutionLimits,
        execute_candidates,
    )

    path = FROZEN_FUZZ / "seed_77.mmd"
    baseline = execute_candidates(
        CandidateExecutionRequest(
            path.read_text(),
            source_dir=str(path.parent),
            limits=ExecutionLimits(1, 60.0, 70.0),
        )
    ).baseline

    assert baseline.status is CandidateStatus.ACCEPTED
    assert baseline.stage is CandidateStage.COMPLETE


@pytest.mark.parametrize(
    "path",
    tuple(sorted(FROZEN_FUZZ.glob("seed_*.mmd"))),
    ids=lambda path: path.stem,
)
def test_stable_generated_inputs_have_no_plan_owned_fused_pair(path: Path) -> None:
    """The frozen fuzz seeds bound the live invariant's generated-input rate."""
    graph, routes, offsets = _route(path)
    violations = check_no_fused_cotravelling_lines(graph, routes, offsets)
    plan_owned = [item for item in violations if item.plan_ids]
    assert not plan_owned, (
        "generated input contains a plan-owned fused pair:\n"
        + "\n".join(item.message() for item in plan_owned)
    )


def test_novel_plan_owned_corridor_is_settled_before_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gallery reproducer needs pre-freeze planning to remain renderable."""
    path = EXAMPLE_TOPOLOGIES / "plan_owned_distinct_lane_separation.mmd"
    _disable_separation_stages(monkeypatch)
    graph, routes, offsets = _route(path)
    violations = check_no_fused_cotravelling_lines(graph, routes, offsets)
    assert any(
        {item.first_line, item.second_line} == {"primary", "secondary"}
        and item.separation == pytest.approx(0.0)
        for item in violations
    )

    monkeypatch.undo()
    graph, routes, offsets = _route(path)
    assert not check_no_fused_cotravelling_lines(graph, routes, offsets)
    assert _pair_separations(routes, offsets)[("primary", "secondary", "Y")] == (
        pytest.approx(graph_offset_step(graph))
    )


@pytest.mark.parametrize(
    ("path", "primary_source", "secondary_sources"),
    (
        (
            EXAMPLE_TOPOLOGIES / "plan_owned_distinct_lane_separation.mmd",
            "__junction_8",
            ("secondary_near__exit_left_1", "secondary_far__exit_left_2"),
        ),
        (
            REGRESSIONS / "plan_owned_distinct_lane_separation_reordered.mmd",
            "__junction_9",
            ("secondary_near__exit_left_0", "secondary_far__exit_left_1"),
        ),
    ),
    ids=("authored-order", "reordered-edges"),
)
def test_plan_owned_distinct_lanes_minimize_crossings_independent_of_edge_order(
    path: Path,
    primary_source: str,
    secondary_sources: tuple[str, str],
) -> None:
    graph, routes, offsets = _route(path)
    primary_routes = tuple(
        route
        for route in routes
        if route.line_id == "primary"
        and route.edge.source == primary_source
        and route.edge.target in {"__merge_2", "__merge_3"}
    )
    secondary_routes = tuple(
        route
        for route in routes
        if route.line_id == "secondary" and route.edge.source in secondary_sources
    )
    assert len(primary_routes) == len(secondary_routes) == 2

    for primary in primary_routes:
        primary_points = apply_route_offsets(primary, offsets)
        primary_trunk_y = _longest_horizontal_run(primary_points)[1]
        for secondary in secondary_routes:
            secondary_points = apply_route_offsets(secondary, offsets)
            secondary_trunk_y = _longest_horizontal_run(secondary_points)[1]
            assert secondary_trunk_y < primary_trunk_y
            assert not tuple(_routes_crossings(primary_points, secondary_points))


@pytest.mark.parametrize(
    ("path", "primary_source", "secondary_sources"),
    (
        (
            EXAMPLE_TOPOLOGIES / "plan_owned_distinct_lane_separation.mmd",
            "__junction_8",
            ("secondary_near__exit_left_1", "secondary_far__exit_left_2"),
        ),
        (
            REGRESSIONS / "plan_owned_distinct_lane_separation_reordered.mmd",
            "__junction_9",
            ("secondary_near__exit_left_0", "secondary_far__exit_left_1"),
        ),
    ),
    ids=("authored-order", "reordered-edges"),
)
def test_production_plan_owned_distinct_lanes_preserve_planned_order(
    path: Path,
    primary_source: str,
    secondary_sources: tuple[str, str],
) -> None:
    graph, routed = _production_routes(path)
    primary = tuple(
        points
        for route, points in routed
        if route.line_id == "primary"
        and route.edge.source == primary_source
        and route.edge.target in {"__merge_2", "__merge_3"}
    )
    secondary = tuple(
        points
        for route, points in routed
        if route.line_id == "secondary" and route.edge.source in secondary_sources
    )
    assert len(primary) == len(secondary) == 2

    step = graph_offset_step(graph)
    for primary_points in primary:
        primary_y = _longest_horizontal_run(primary_points)[1]
        for secondary_points in secondary:
            secondary_y = _longest_horizontal_run(secondary_points)[1]
            assert primary_y - secondary_y == pytest.approx(step)
            assert not tuple(_routes_crossings(primary_points, secondary_points))


def test_production_longread_divergence_preserves_source_lane_order() -> None:
    graph, routed = _production_routes(EXAMPLES / "longread_variant_calling.mmd")
    trunks = {
        route.line_id: points
        for route, points in routed
        if route.edge.source == "__junction_16"
        and route.edge.target
        in {"tr_calling__entry_right_9", "cnv_calling__entry_right_10"}
    }
    assert set(trunks) == {"bam", "other"}

    bam_y = _longest_horizontal_run(trunks["bam"])[1]
    other_y = _longest_horizontal_run(trunks["other"])[1]
    assert other_y - bam_y == pytest.approx(graph_offset_step(graph))
    assert not tuple(_routes_crossings(trunks["bam"], trunks["other"]))


def test_reserved_trunk_order_bypasses_live_crossing_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nf_metro.api import prepare_graph

    path = EXAMPLES / "longread_variant_calling.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    offsets = compute_station_offsets(graph)
    first = observe_route_edges(
        graph,
        station_offsets=offsets,
        allow_convergence_clearance_requirements=True,
    )
    live_rank = normalize._band_order_crossings

    def reject_target_live_rank(order, features=None):
        members = {
            (trunk.route.edge.source, trunk.route.edge.target, trunk.route.line_id)
            for slot in order
            for trunk in slot
        }
        target_members = {
            ("__junction_16", "tr_calling__entry_right_9", "bam"),
            ("__junction_16", "cnv_calling__entry_right_10", "other"),
        }
        if target_members <= members:
            raise AssertionError("frozen trunk cohort consulted live crossing rank")
        return live_rank(order, features)

    monkeypatch.setattr(normalize, "_band_order_crossings", reject_target_live_rank)
    replay = routing_core.observe_route_edges_centred(
        graph,
        station_offsets=offsets,
        reservations=first.plan,
    )
    trunks = {
        route.line_id: route
        for route in replay.routes
        if route.edge.source == "__junction_16"
        and route.edge.target
        in {"tr_calling__entry_right_9", "cnv_calling__entry_right_10"}
    }
    assert (
        _longest_horizontal_run(trunks["bam"].points)[1]
        < (_longest_horizontal_run(trunks["other"].points)[1])
    )


def test_frozen_order_rejects_partial_claim_coverage() -> None:
    from types import SimpleNamespace

    from nf_metro.layout.routing.reserved_bands import ReservedCorridors

    def route(source: str, target: str, line_id: str) -> RoutedPath:
        return RoutedPath(
            edge=Edge(source=source, target=target, line_id=line_id),
            line_id=line_id,
            points=[(0.0, 0.0), (10.0, 0.0)],
        )

    first = route("hub", "first", "red")
    unclaimed_fan_member = route("hub", "first-fan", "red")
    second = route("hub", "second", "blue")
    ctx = SimpleNamespace(
        reserved_bands=ReservedCorridors(
            planned_order_coordinates={
                ("hub", "first", "red", 1): 10.0,
                ("hub", "second", "blue", 1): 14.0,
            }
        )
    )
    groups = (
        ((first, 1), (unclaimed_fan_member, 1)),
        ((second, 1),),
    )

    assert (
        normalize._frozen_order_coordinates(
            ctx, groups, require_single_common_source=True
        )
        is None
    )
    assert (
        normalize._frozen_order_coordinates(
            ctx, groups, require_single_common_source=False
        )
        is None
    )


def test_frozen_source_cohort_rejects_identical_multi_source_sets() -> None:
    from types import SimpleNamespace

    from nf_metro.layout.routing.reserved_bands import ReservedCorridors

    def route(source: str, target: str, line_id: str) -> RoutedPath:
        return RoutedPath(
            edge=Edge(source=source, target=target, line_id=line_id),
            line_id=line_id,
            points=[(0.0, 0.0), (10.0, 0.0)],
        )

    first = (route("source-a", "red-a", "red"), route("source-b", "red-b", "red"))
    second = (
        route("source-a", "blue-a", "blue"),
        route("source-b", "blue-b", "blue"),
    )
    claims = {
        (member.edge.source, member.edge.target, member.line_id, 1): coordinate
        for group, coordinate in ((first, 10.0), (second, 14.0))
        for member in group
    }
    ctx = SimpleNamespace(
        reserved_bands=ReservedCorridors(planned_order_coordinates=claims)
    )
    groups = tuple(tuple((member, 1) for member in group) for group in (first, second))

    assert (
        normalize._frozen_order_coordinates(
            ctx, groups, require_single_common_source=True
        )
        is None
    )
    assert normalize._frozen_order_coordinates(
        ctx, groups, require_single_common_source=False
    ) == (10.0, 14.0)


def test_reserved_plan_owned_lane_never_takes_reversed_live_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reseat = normalize._reseat_lane

    def reject_reversed_candidate(lane, coordinate, *, projected=False):
        if lane.line_id == "secondary" and coordinate > lane.coord:
            raise AssertionError("frozen lane took the reversed live candidate")
        return reseat(lane, coordinate, projected=projected)

    monkeypatch.setattr(normalize, "_reseat_lane", reject_reversed_candidate)
    path = EXAMPLE_TOPOLOGIES / "plan_owned_distinct_lane_separation.mmd"
    graph, routed = _production_routes(path)
    trunks = {
        route.line_id: _longest_horizontal_run(points)[1]
        for route, points in routed
        if (
            route.line_id == "primary"
            and route.edge.source == "__junction_8"
            and route.edge.target == "__merge_2"
        )
        or (
            route.line_id == "secondary"
            and route.edge.source == "secondary_near__exit_left_1"
        )
    }
    assert trunks["primary"] - trunks["secondary"] == pytest.approx(
        graph_offset_step(graph)
    )


def test_analytic_candidate_segments_match_the_plan_mover() -> None:
    """Analytic scoring prices exactly the geometry the selected move emits."""
    from nf_metro.api import prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan

    path = EXAMPLE_TOPOLOGIES / "plan_owned_distinct_lane_separation.mmd"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    plans = tuple(
        plan
        for plan in observed.route_plan.convergence_plans
        if plan.line_ids == ("primary",) and plan.trunk_axis is not None
    )
    assert len(plans) == 2
    for plan in plans:
        axis = plan.trunk_axis
        assert axis is not None
        candidates = convergences._clear_lane_candidates(
            (axis.coordinate,), graph_offset_step(graph)
        )
        assert candidates
        for candidate in candidates:
            analytic = convergences._plan_segments(plan, candidate)
            moved = convergences._plan_segments(
                convergences._move_trunk_axis(plan, candidate)
            )
            assert analytic == moved


@pytest.mark.parametrize(
    ("first_points", "second_points", "axis"),
    (
        (
            [(0.0, -20.0), (0.0, 0.0), (100.0, 0.0), (100.0, 20.0)],
            [(0.0, -16.0), (0.0, 2.0), (100.0, 2.0), (100.0, 24.0)],
            "Y",
        ),
        (
            [(100.0, -20.0), (100.0, 0.0), (0.0, 0.0), (0.0, 20.0)],
            [(100.0, -16.0), (100.0, 2.0), (0.0, 2.0), (0.0, 24.0)],
            "Y",
        ),
        (
            [(-20.0, 0.0), (0.0, 0.0), (0.0, 100.0), (20.0, 100.0)],
            [(-16.0, 0.0), (2.0, 0.0), (2.0, 100.0), (24.0, 100.0)],
            "X",
        ),
        (
            [(-20.0, 100.0), (0.0, 100.0), (0.0, 0.0), (20.0, 0.0)],
            [(-16.0, 100.0), (2.0, 100.0), (2.0, 0.0), (24.0, 0.0)],
            "X",
        ),
    ),
    ids=("right", "left", "down", "up"),
)
def test_plan_owned_fused_lanes_are_attributed_in_every_direction(
    first_points: list[tuple[float, float]],
    second_points: list[tuple[float, float]],
    axis: str,
) -> None:
    """The live check reports immutable fused lanes on either routing axis."""

    def planned_route(
        line_id: str,
        points: list[tuple[float, float]],
        plan_id: str,
    ) -> RoutedPath:
        return RoutedPath(
            Edge(f"{line_id}_source", f"{line_id}_target", line_id),
            line_id,
            points,
            is_inter_section=True,
            offset_regime=OffsetRegime.BAKED,
            route_system_id="route-system:test",
            convergence_plan_id=plan_id,
            convergence_owned_segment_ranks=(1,),
        )

    violations = check_no_fused_cotravelling_lines(
        MetroGraph(),
        [
            planned_route("first", first_points, "convergence-plan:first"),
            planned_route("second", second_points, "convergence-plan:second"),
        ],
        {},
    )

    assert len(violations) == 1
    assert violations[0].axis == axis
    message = violations[0].message()
    assert "route-system:test" in message
    assert "convergence-plan:first" in message
    assert "convergence-plan:second" in message


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_no_fused_cotravelling_lines_in_gallery(path: Path) -> None:
    """No shipped topology or example paints two distinct lines as one stroke."""
    graph, routes, offsets = _route(path)
    violations = check_no_fused_cotravelling_lines(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


@pytest.mark.parametrize("path", REPORTED, ids=lambda p: p.stem)
def test_reported_corridors_keep_the_nesting_step(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corridors a reservation band pulled together keep the full step."""
    graph, routes, offsets, _violations = _settled(path, monkeypatch)
    separations = _pair_separations(routes, offsets)
    for pair in REPORTED[path]:
        assert pair in separations, f"{pair} no longer shares a corridor"
        assert separations[pair] >= graph_offset_step(graph)


@pytest.mark.parametrize("path", FUSED_WITHOUT_THE_PASS, ids=lambda p: p.stem)
def test_tracks_fuse_without_the_separation_stages(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling both separation stages reproduces the fused pair."""
    _disable_separation_stages(monkeypatch)
    graph, routes, offsets, _violations = _settled(path, monkeypatch)
    step = graph_offset_step(graph)
    separations = _pair_separations(routes, offsets)
    for pair in FUSED_WITHOUT_THE_PASS[path]:
        assert pair in separations, f"{pair} no longer shares a corridor"
        assert separations[pair] < step


@pytest.mark.parametrize("path", FUSED_WITHOUT_THE_PASS, ids=lambda p: p.stem)
def test_separated_pairs_land_on_the_nesting_pitch(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each settled pair lands on the nesting pitch, not an accidental gap."""
    _disable_separation_stages(monkeypatch)
    graph, routes, offsets, _violations = _settled(path, monkeypatch)
    step = graph_offset_step(graph)
    separations = _pair_separations(routes, offsets)
    fused = FUSED_WITHOUT_THE_PASS[path]
    for pair in fused:
        assert pair in separations, f"{pair} no longer shares a corridor"
        assert separations[pair] < step

    monkeypatch.undo()
    graph, routes, offsets, _violations = _settled(path, monkeypatch)
    separations = _pair_separations(routes, offsets)
    step = graph_offset_step(graph)
    for pair in fused:
        assert pair in separations, f"{pair} no longer shares a corridor"
        assert separations[pair] >= step
        assert separations[pair] % step == pytest.approx(0.0, abs=1e-6)
