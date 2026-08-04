"""Row and column envelopes settle monotonically around route reservations."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

import nf_metro.layout.envelope_settlement as envelope_settlement
import nf_metro.render.svg as svg_render
from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.envelope_settlement import (
    SettlementAxis,
    settle_route_envelopes,
)
from nf_metro.layout.phases.guards import (
    LayoutInvariantError,
    assert_reservations_are_settled,
)
from nf_metro.layout.route_plan import (
    DemandAxis,
    EmissionMemberId,
    build_route_plan_query,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    ColumnGapRegion,
    ReservationCoordinateTranslation,
    RowGapRegion,
    _project_shared_coordinate,
    realise_reservation,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.render.svg import (
    _assert_settlement_decisions_frozen,
    build_observed_render_plan,
)

ROOT = Path(__file__).parents[1]
TOPOLOGIES = ROOT / "examples" / "topologies"
REPORT_HO = ROOT / "tests" / "fixtures" / "route_reservations" / "reportho.metro"

# Fixtures whose reservations carry a capacity deficit on unsettled geometry, so
# settlement has real work to do.  Spread across the four flow directions and
# both corridor axes.
DEFICIT_CORPUS = (
    TOPOLOGIES / "convergence_fold_diamond.mmd",
    TOPOLOGIES / "convergence_sink_fold.mmd",
    TOPOLOGIES / "fold_split_targets.mmd",
    TOPOLOGIES / "merge_right_entry.mmd",
    TOPOLOGIES / "off_track_input_above_consumer.mmd",
    TOPOLOGIES / "right_entry_over_top_tall_upstream.mmd",
    TOPOLOGIES / "same_line_fan_distinct_descent.mmd",
    ROOT / "examples" / "differentialabundance.mmd",
    ROOT / "tests" / "fixtures" / "da_pipeline.mmd",
    ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd",
)

# Fixtures with no positive deficit anywhere: settlement must not touch them.
SETTLED_CORPUS = (
    ROOT / "examples" / "rnaseq_sections.mmd",
    ROOT / "examples" / "rnaseq_auto.mmd",
    ROOT / "examples" / "hlatyping.mmd",
    ROOT / "examples" / "epitopeprediction.mmd",
)

# One fixture per supported flow direction, so the single axis-based
# implementation is exercised under rotation and reflection.
DIRECTION_CORPUS = {
    "LR": ROOT / "examples" / "rnaseq_sections.mmd",
    "RL": ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_41.mmd",
    "TB": ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd",
    "BT": TOPOLOGIES / "bt_perp_left_entry_right_exit.mmd",
}

LEDGER_STABILITY_CORPUS = (
    ROOT / "examples" / "longread_variant_calling.mmd",
    TOPOLOGIES / "bypass_fan_in_outer_slot.mmd",
    TOPOLOGIES / "complex_multipath.mmd",
    TOPOLOGIES / "convergence_fold_diamond.mmd",
    TOPOLOGIES / "convergence_sink_fold.mmd",
    TOPOLOGIES / "fold_split_targets.mmd",
    ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
    ROOT / "tests" / "fixtures" / "planned_compatibility_channel_collision.mmd",
    ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd",
)


def _observe(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        observation = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        )
    return graph, observation.plan


def _rendered_plan(path: Path, *, permissive: bool = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        graph.permissive = permissive
        return build_observed_render_plan(graph, resolve_theme(None, graph))


def _capacity_deficits(plan) -> dict[str, float]:
    """Reservation id -> negative capacity slack, for gap-region corridors."""
    query = build_route_plan_query(plan)
    deficits: dict[str, float] = {}
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is not None and realised.capacity_slack < -0.01:
            deficits[str(reservation.id)] = realised.capacity_slack
    return deficits


def _geometry(graph) -> dict[str, tuple[float, ...]]:
    return {
        **{
            f"section:{key}": (
                section.bbox_x,
                section.bbox_y,
                section.bbox_w,
                section.bbox_h,
            )
            for key, section in graph.sections.items()
        },
        **{
            f"station:{key}": (station.x, station.y)
            for key, station in graph.stations.items()
        },
        **{f"port:{key}": (port.x, port.y) for key, port in graph.ports.items()},
    }


def _section_local_geometry(graph) -> dict[str, tuple]:
    """Each section's size and its content's position within it.

    Rounded to a hundredth of a pixel: a rigid translation of a whole row is
    exact in intent but not bit-exact once the same offset is added to and then
    subtracted from two different coordinates.
    """
    return {
        key: (
            round(section.bbox_w, 2),
            round(section.bbox_h, 2),
            tuple(
                (
                    round(graph.stations[item].x - section.bbox_x, 2),
                    round(graph.stations[item].y - section.bbox_y, 2),
                )
                for item in sorted(section.station_ids)
                if item in graph.stations
            ),
        )
        for key, section in graph.sections.items()
    }


def _row_gaps(graph) -> dict[tuple[int, int], float]:
    """Vertical separation between each adjacent pair of grid rows."""
    return _axis_gaps(graph, SettlementAxis.ROW)


def _column_gaps(graph) -> dict[tuple[int, int], float]:
    return _axis_gaps(graph, SettlementAxis.COLUMN)


def _axis_gaps(graph, axis: SettlementAxis) -> dict[tuple[int, int], float]:
    starts: dict[int, list[float]] = {}
    ends: dict[int, list[float]] = {}
    for section in graph.sections.values():
        if section.bbox_w <= 0 or section.bbox_h <= 0:
            continue
        if axis is SettlementAxis.ROW:
            index = section.grid_row
            last = section.grid_row + section.grid_row_span - 1
            lo, hi = section.bbox_y, section.bbox_y + section.bbox_h
        else:
            index = section.grid_col
            last = section.grid_col + section.grid_col_span - 1
            lo, hi = section.bbox_x, section.bbox_x + section.bbox_w
        starts.setdefault(index, []).append(lo)
        ends.setdefault(last, []).append(hi)
    gaps: dict[tuple[int, int], float] = {}
    for index in sorted(starts):
        if index - 1 not in ends:
            continue
        gaps[(index - 1, index)] = min(starts[index]) - max(ends[index - 1])
    return gaps


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_satisfies_every_gap_region_reservation(path: Path) -> None:
    """Final geometry gives every reserved corridor its required width."""
    observed = _rendered_plan(path)
    assert _capacity_deficits(observed.route_plan) == {}


def test_reportho_report_trunk_keeps_its_authored_inter_row_corridor() -> None:
    """The 12 report feeders share one trunk lane needing 78px between rows.

    Rendered permissively because this map also puts two opposing channels in
    one column gap without separating them, which is a lane-placement defect
    with plenty of measured corridor to spare rather than a corridor width this
    stage owns.  Every reservation the settled geometry publishes is checked
    below, that one included.
    """
    observed = _rendered_plan(REPORT_HO, permissive=True)
    plan = observed.route_plan
    query = build_route_plan_query(plan)
    reservation = max(
        (item for item in plan.reservations if isinstance(item.region, RowGapRegion)),
        key=lambda item: item.minimum_width,
    )
    assert len(reservation.connector_ids) == 12
    realised = query.realised_reservation(reservation.id)
    assert realised is not None
    assert reservation.minimum_width == 78
    assert realised.available_width >= 78.0
    assert realised.capacity_slack >= 0.0
    assert _capacity_deficits(plan) == {}


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_run_twice_is_an_exact_geometry_no_op(path: Path) -> None:
    """Settlement reaches a fixpoint in one directional pass."""
    graph, plan = _observe(path)
    settle_route_envelopes(graph, plan)
    settled = _geometry(graph)
    second = settle_route_envelopes(graph, plan)
    assert second.translations == ()
    assert _geometry(graph) == settled


@pytest.mark.parametrize("path", SETTLED_CORPUS, ids=lambda item: item.name)
def test_settlement_leaves_a_deficit_free_layout_untouched(path: Path) -> None:
    graph, plan = _observe(path)
    before = _geometry(graph)
    settlement = settle_route_envelopes(graph, plan)
    assert settlement.translations == ()
    assert _geometry(graph) == before


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_never_narrows_a_row_or_column_gap(path: Path) -> None:
    graph, plan = _observe(path)
    before_rows, before_columns = _row_gaps(graph), _column_gaps(graph)
    settle_route_envelopes(graph, plan)
    for key, gap in _row_gaps(graph).items():
        assert gap >= before_rows[key] - 0.01, f"row gap {key} narrowed"
    for key, gap in _column_gaps(graph).items():
        assert gap >= before_columns[key] - 0.01, f"column gap {key} narrowed"


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_preserves_frozen_local_geometry(path: Path) -> None:
    """Only whole-row and whole-column offsets move; nothing moves inside a
    section, and no bbox is resized."""
    graph, plan = _observe(path)
    before = _section_local_geometry(graph)
    settle_route_envelopes(graph, plan)
    assert _section_local_geometry(graph) == before


@pytest.mark.parametrize(
    "path", tuple(DIRECTION_CORPUS.values()), ids=tuple(DIRECTION_CORPUS)
)
def test_one_axis_based_implementation_covers_every_flow_direction(path: Path) -> None:
    """Narrowing a satisfied corridor is recovered whatever the flow direction.

    Settlement keys on grid rows and columns, never on a section's flow
    direction, so injecting the same deficit into an LR, RL, TB, or BT layout
    must be answered by the same pass.
    """
    graph, plan = _observe(path)
    query = build_route_plan_query(plan)
    # Every corridor at one boundary shares its translation, so squeeze by the
    # tightest of them: any more and a second corridor drives the deficit.
    by_boundary: dict[tuple[SettlementAxis, int], list] = {}
    for reservation in plan.reservations:
        region = reservation.region
        if isinstance(region, RowGapRegion):
            key = (SettlementAxis.ROW, region.lower_row)
        elif isinstance(region, ColumnGapRegion):
            key = (SettlementAxis.COLUMN, region.right_column)
        else:
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is not None:
            by_boundary.setdefault(key, []).append((reservation, realised))

    (axis, boundary), residents = max(
        by_boundary.items(),
        key=lambda item: min(got.capacity_slack for _res, got in item[1]),
    )
    target, tightest = min(residents, key=lambda pair: pair[1].capacity_slack)
    shortfall = 4.0
    _narrow(graph, axis, boundary, tightest.capacity_slack + shortfall)

    settlement = settle_route_envelopes(graph, plan)
    injected = [
        item
        for item in settlement.translations
        if (item.axis, item.boundary) == (axis, boundary)
    ]
    assert len(injected) == 1
    assert injected[0].amount == pytest.approx(shortfall, abs=1.0)
    recovered = realise_reservation(graph, target)
    assert recovered is not None
    assert recovered.capacity_slack >= 0.0


def test_a_corridor_bounded_by_a_row_spanning_section_is_attributed_not_moved() -> None:
    """Row-spanning sections straddle the boundary, so no translation helps.

    Settlement declines the deficit and names the sections that bound it,
    rather than translating rows that would add whitespace without widening the
    corridor.
    """
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)

    obstructions = [
        item for item in settlement.obstructions if item.axis is SettlementAxis.ROW
    ]
    assert obstructions
    for obstruction in obstructions:
        assert obstruction.deficit > 0
        assert obstruction.blocking_section_ids
        for section_id in obstruction.blocking_section_ids:
            section = graph.sections[section_id]
            assert section.grid_row < obstruction.boundary
            assert section.grid_row + section.grid_row_span - 1 >= obstruction.boundary


def test_settlement_restores_every_coordinate_when_a_translation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, plan = _observe(TOPOLOGIES / "convergence_fold_diamond.mmd")
    before = _geometry(graph)
    shift_section = envelope_settlement.shift_section
    calls = 0

    def fail_after_one_write(graph, section, *, dx=0.0, dy=0.0):
        nonlocal calls
        shift_section(graph, section, dx=dx, dy=dy)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected translation failure")

    monkeypatch.setattr(envelope_settlement, "shift_section", fail_after_one_write)
    with pytest.raises(RuntimeError, match="injected translation failure"):
        settle_route_envelopes(graph, plan)
    assert calls == 1
    assert _geometry(graph) == before


def test_translation_names_every_deficient_claim_at_its_boundary() -> None:
    graph, plan = _observe(TOPOLOGIES / "convergent_offrow_exit_climb.mmd")
    query = build_route_plan_query(plan)
    expected: dict[tuple[SettlementAxis, int], set] = {}
    for reservation in plan.reservations:
        if isinstance(reservation.region, RowGapRegion):
            key = (SettlementAxis.ROW, reservation.region.lower_row)
        elif isinstance(reservation.region, ColumnGapRegion):
            key = (SettlementAxis.COLUMN, reservation.region.right_column)
        else:
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is not None and realised.capacity_slack < -0.01:
            expected.setdefault(key, set()).add(reservation.id)

    settlement = settle_route_envelopes(graph, plan)
    translated = {
        (item.axis, item.boundary): set(item.reservation_ids)
        for item in settlement.translations
    }
    assert any(len(reservation_ids) > 1 for reservation_ids in expected.values())
    assert translated == expected


@pytest.mark.parametrize("path", LEDGER_STABILITY_CORPUS, ids=lambda item: item.name)
def test_render_keeps_the_initial_grid_reservation_ledger(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    settle = svg_render.settle_route_envelopes

    def capture_plan(graph, plan):
        captured.append(plan)
        return settle(graph, plan)

    monkeypatch.setattr(svg_render, "settle_route_envelopes", capture_plan)
    final = _rendered_plan(path, permissive=True).route_plan
    assert len(captured) == 1
    frozen = captured[0]
    frozen_grid = tuple(
        item
        for item in frozen.reservations
        if isinstance(item.region, RowGapRegion | ColumnGapRegion)
    )
    final_grid = tuple(
        item
        for item in final.reservations
        if isinstance(item.region, RowGapRegion | ColumnGapRegion)
    )
    assert final_grid == frozen_grid

    reference_ids = {item.reference_id for item in frozen_grid}
    demand_ids = {
        demand_id for reservation in frozen_grid for demand_id in reservation.demand_ids
    }
    assert {
        item.id: item for item in final.shared_references if item.id in reference_ids
    } == {
        item.id: item for item in frozen.shared_references if item.id in reference_ids
    }
    assert {item.id: item for item in final.demands if item.id in demand_ids} == {
        item.id: item for item in frozen.demands if item.id in demand_ids
    }


def test_final_canvas_claims_are_not_projected_twice() -> None:
    plan = _rendered_plan(
        TOPOLOGIES / "convergent_offrow_exit_climb.mmd", permissive=True
    ).route_plan
    region_by_id = {item.id: item.region for item in plan.reservations}
    canvas = tuple(
        item
        for item in plan.realised_reservations
        if isinstance(region_by_id[item.reservation_id], CanvasRegion)
    )
    grid = tuple(
        item
        for item in plan.realised_reservations
        if isinstance(region_by_id[item.reservation_id], RowGapRegion | ColumnGapRegion)
    )

    assert canvas
    assert all(not item.coordinate_translations for item in canvas)
    assert any(item.coordinate_translations for item in grid)


def test_deficit_free_render_skips_ledger_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_adoption(*_args, **_kwargs):
        raise AssertionError("deficit-free render rebuilt its reservation ledger")

    monkeypatch.setattr(
        svg_render, "adopt_route_reservation_ledger", unexpected_adoption
    )
    _rendered_plan(SETTLED_CORPUS[0])


def test_authored_spanning_grid_is_a_precise_strict_failure() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)
    with pytest.raises(LayoutInvariantError) as error:
        assert_reservations_are_settled(graph, plan, settlement, strict=True)
    message = str(error.value)
    assert "system route-system|" in message
    assert "reservation route-reservation:" in message
    assert "requires" in message and "has" in message
    assert "spans columns" in message and "rows" in message
    assert "conflicting authored grid section(s)" in message


# Route systems the convergence planner puts on its compatibility disposition
# for want of "a wider shared settlement".  Each one's corridors reach their
# required width, which is the evidence that what limits them is a channel
# decision rather than envelope allocation.
COMPATIBILITY_EXIT_MATRIX = (
    (
        TOPOLOGIES / "exit_run_three_drop_columns.mmd",
        "plan-driven shared-channel emission",
        False,
    ),
    (
        TOPOLOGIES / "merge_around_below_leftmost.mmd",
        "plan-driven shared-channel emission",
        False,
    ),
    (
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
        "plan-driven shared-channel emission",
        False,
    ),
    (
        ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
        "plan-driven shared-channel emission",
        False,
    ),
    (
        TOPOLOGIES / "merge_bottom_row_bypass.mmd",
        "plan-driven opposing-opening emission",
        False,
    ),
    (
        TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
        "plan-driven opposing-opening emission",
        False,
    ),
    (
        TOPOLOGIES / "funcprofiler_upstream.mmd",
        "plan-driven whole-system emission",
        False,
    ),
    (
        TOPOLOGIES / "merge_right_entry.mmd",
        "plan-driven whole-system emission",
        False,
    ),
    (
        ROOT / "examples" / "genomeassembly.mmd",
        "plan-driven chained-convergence emission",
        False,
    ),
    (
        ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
        "plan-driven chained-convergence emission",
        True,
    ),
)


@pytest.mark.parametrize(
    ("path", "owner", "obstructed"),
    COMPATIBILITY_EXIT_MATRIX,
    ids=[item[0].name for item in COMPATIBILITY_EXIT_MATRIX],
)
def test_compatibility_systems_publish_exact_settlement_exit_evidence(
    path: Path,
    owner: str,
    obstructed: bool,
) -> None:
    _graph, initial = _observe(path)
    observed = _rendered_plan(path, permissive=True)
    final = observed.route_plan
    assert tuple(
        (item.id, item.disposition, item.legacy_reason)
        for item in final.convergence_plans
    ) == tuple(
        (item.id, item.disposition, item.legacy_reason)
        for item in initial.convergence_plans
    )
    diagnostics = tuple(
        item for item in final.diagnostics if item.code == "convergence-settlement-exit"
    )
    assert len(diagnostics) == 1
    message = diagnostics[0].message
    assert owner in message
    assert "#1658" in message
    if obstructed:
        assert _capacity_deficits(final)
        assert "bounded by spanning section(s)" in message
        assert "global row or column translation cannot supply" in message
    else:
        assert _capacity_deficits(final) == {}
        assert "row or column corridor claim(s) fit" in message


def test_decision_guard_rejects_route_turn_and_plan_changes() -> None:
    graph, observation = _observe(TOPOLOGIES / "merge_right_entry.mmd")
    routed = observe_route_edges(graph, station_offsets=compute_station_offsets(graph))
    changed_routes = list(routed.routes)
    start = changed_routes[0].points[0]
    second = changed_routes[0].points[1]
    changed_routes[0] = replace(
        changed_routes[0],
        points=[
            start,
            (
                start[0] - (second[0] - start[0]),
                start[1] - (second[1] - start[1]),
            ),
            *changed_routes[0].points[2:],
        ],
    )
    with pytest.raises(LayoutInvariantError, match="route topology"):
        _assert_settlement_decisions_frozen(
            routed.routes,
            routed.plan,
            changed_routes,
            routed.plan,
        )

    convergence = next(
        item for item in observation.convergence_plans if item.legacy_reason is not None
    )
    changed_plan = replace(
        observation,
        convergence_plans=tuple(
            replace(item, legacy_reason=f"{item.legacy_reason}: changed")
            if item.id == convergence.id
            else item
            for item in observation.convergence_plans
        ),
    )
    with pytest.raises(LayoutInvariantError, match="planning decisions"):
        _assert_settlement_decisions_frozen([], observation, [], changed_plan)


def test_shared_plan_projection_requires_one_coordinate_for_every_claimant() -> None:
    first = EmissionMemberId("first")
    second = EmissionMemberId("second")
    translation = ReservationCoordinateTranslation(
        DemandAxis.Y,
        10.0,
        4.0,
        fully_owned_member_ids=(first,),
        crossing_member_ids=(second,),
    )
    with pytest.raises(ValueError, match="separates shared exit-turn geometry"):
        _project_shared_coordinate(
            0.0,
            DemandAxis.Y,
            (first, second),
            (translation,),
        )


def _narrow(graph, axis: SettlementAxis, boundary: int, amount: float) -> None:
    """Translate everything from *boundary* onward back toward the content
    above or left of it, eating the corridor settlement then has to restore."""
    for section in graph.sections.values():
        index = section.grid_row if axis is SettlementAxis.ROW else section.grid_col
        if index < boundary:
            continue
        if axis is SettlementAxis.ROW:
            section.bbox_y -= amount
        else:
            section.bbox_x -= amount
        for station_id in section.station_ids:
            for item in (graph.stations.get(station_id), graph.ports.get(station_id)):
                if item is None:
                    continue
                if axis is SettlementAxis.ROW:
                    item.y -= amount
                else:
                    item.x -= amount
