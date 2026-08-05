"""Row and column envelopes settle monotonically around route reservations."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest import mock

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout import envelope_settlement
from nf_metro.layout.envelope_settlement import (
    EnvelopeSettlement,
    SettlementAxis,
    SettlementShortfall,
    sections_spanning_the_gap,
    settle_route_envelopes,
)
from nf_metro.layout.phases.guards import (
    LayoutInvariantError,
    assert_reservations_are_settled,
)
from nf_metro.layout.route_plan import build_route_plan_query
from nf_metro.layout.route_reservations import (
    ColumnGapRegion,
    RowGapRegion,
    realise_reservation,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.layout.routing.common import _inter_row_band_fits
from nf_metro.render.svg import build_observed_render_plan

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
def test_settlement_meets_the_demands_it_was_handed(path: Path) -> None:
    """Settlement's own contract, measured against its own input.

    The ledger a settled re-route publishes is a different set of claims --
    corridors appear and vanish across a translation -- so measuring there
    answers a different question than "did settlement satisfy what it was
    given".  This measures the input ledger on the geometry settlement left.
    """
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)
    assert settlement.translations
    unmet = [item for item in settlement.shortfalls if item.describes_a_gap]
    assert unmet == []


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_the_rerouted_ledger_is_also_free_of_deficits(path: Path) -> None:
    """A separate, weaker statement than the contract above: the claims the
    settled geometry goes on to publish are satisfied too."""
    observed = _rendered_plan(path)
    assert _capacity_deficits(observed.route_plan) == {}


def test_an_unmet_demand_is_reported_against_the_input_ledger() -> None:
    """The shortfall record names the reservation settlement could not satisfy,
    classified by why, rather than leaving it to be rediscovered downstream."""
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)
    assert settlement.shortfalls
    for shortfall in settlement.shortfalls:
        assert shortfall.available_width < shortfall.required_width
        assert shortfall.claimant_member_ids
    assert any(not item.describes_a_gap for item in settlement.shortfalls)


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


def test_a_span_across_the_boundary_leaves_no_width_to_allocate() -> None:
    """Why settlement reports one kind of obstruction rather than two.

    A far-side blocker settlement cannot translate is exactly a section
    spanning across the boundary, and the row measurement puts every spanning
    section on the near side as well.  The section's own box therefore lies
    inside the width the claim asks for and the far side is measured ahead of
    the near side, so such a claim can never be a genuine corridor that
    settlement merely fails to widen.
    """
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    checked = 0
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is None:
            continue
        spanning = sections_spanning_the_gap(graph, reservation, realised)
        if not spanning:
            continue
        checked += 1
        tallest = max(graph.sections[section_id].bbox_h for section_id in spanning)
        assert realised.available_width <= -tallest
    assert checked


# Route systems the convergence planner puts on its compatibility disposition
# for want of "a wider shared settlement".  Each one's corridors reach their
# required width, which is the evidence that what limits them is a channel
# decision rather than envelope allocation.
SHARED_SETTLEMENT_CANDIDATES = (
    TOPOLOGIES / "exit_run_three_drop_columns.mmd",
    TOPOLOGIES / "merge_around_below_leftmost.mmd",
    TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
    ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
    TOPOLOGIES / "merge_bottom_row_bypass.mmd",
    TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
    TOPOLOGIES / "funcprofiler_upstream.mmd",
    TOPOLOGIES / "merge_right_entry.mmd",
    ROOT / "examples" / "genomeassembly.mmd",
)


@pytest.mark.parametrize(
    "path", SHARED_SETTLEMENT_CANDIDATES, ids=lambda item: item.name
)
def test_compatibility_systems_are_not_short_of_corridor(path: Path) -> None:
    observed = _rendered_plan(path, permissive=True)
    assert _capacity_deficits(observed.route_plan) == {}


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_a_failed_settlement_leaves_the_graph_as_it_found_it(path: Path) -> None:
    """Settlement writes many sections in sequence; a failure rolls all of it
    back rather than leaving a part-translated graph."""
    graph, plan = _observe(path)
    before = _geometry(graph)
    boom = RuntimeError("settlement failed")

    def explode(*_args, **_kwargs):
        raise boom

    with mock.patch.object(envelope_settlement, "realise_reservation", explode):
        with pytest.raises(RuntimeError) as caught:
            settle_route_envelopes(graph, plan)
    assert caught.value is boom
    assert _geometry(graph) == before


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_rolls_back_a_failure_that_lands_mid_translation(
    path: Path,
) -> None:
    graph, plan = _observe(path)
    before = _geometry(graph)
    moved: list[float] = []
    real = envelope_settlement.shift_section

    def move_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        moved.append(1.0)
        raise RuntimeError("settlement failed")

    with mock.patch.object(envelope_settlement, "shift_section", move_then_fail):
        with pytest.raises(RuntimeError):
            settle_route_envelopes(graph, plan)
    assert moved, "the fixture never reached a translation, so nothing was rolled back"
    assert _geometry(graph) == before


@pytest.mark.parametrize("path", DEFICIT_CORPUS + SETTLED_CORPUS, ids=lambda i: i.name)
def test_one_pass_settles_everything_one_ledger_asks_for(path: Path) -> None:
    """Settlement allocates against fixed demand, so it needs no second pass.

    A second productive pass against the same ledger would mean the first left
    a deficit it owned, which is the property the single-pass sweep exists to
    rule out.
    """
    graph, plan = _observe(path)
    first = settle_route_envelopes(graph, plan)
    second = settle_route_envelopes(graph, plan)
    assert second.translations == ()
    assert {item.reservation_id for item in second.obstructions} == {
        item.reservation_id for item in first.obstructions
    }


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_settlement_does_not_chase_the_ledger_the_reroute_publishes(
    path: Path,
) -> None:
    """Re-routing settled geometry yields a different ledger; that is exactly
    why settlement must not iterate against it."""
    graph, plan = _observe(path)
    before = {item.id for item in plan.reservations}
    assert settle_route_envelopes(graph, plan).translations
    rerouted = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    after = {item.id for item in rerouted.reservations}
    assert before or after
    # The ledgers may differ; what must not happen is settlement treating the
    # new one as more work to do against the old geometry.
    assert settle_route_envelopes(graph, plan).translations == ()


COMPATIBILITY_SYSTEMS = (
    TOPOLOGIES / "exit_run_three_drop_columns.mmd",
    TOPOLOGIES / "merge_around_below_leftmost.mmd",
    TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
    ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
    TOPOLOGIES / "merge_bottom_row_bypass.mmd",
    TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
    TOPOLOGIES / "funcprofiler_upstream.mmd",
    TOPOLOGIES / "merge_right_entry.mmd",
    ROOT / "examples" / "genomeassembly.mmd",
    ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
)


@pytest.mark.parametrize("path", COMPATIBILITY_SYSTEMS, ids=lambda item: item.name)
def test_every_compatibility_system_is_attributed_in_the_published_plan(
    path: Path,
) -> None:
    """#1660 lets a system stay on compatibility only with evidence naming a
    separate owner, so that evidence has to reach the plan a consumer reads."""
    observed = _rendered_plan(path, permissive=True)
    published = [
        item
        for item in observed.route_plan.diagnostics
        if item.code == "convergence-settlement-exit"
    ]
    assert all(
        item.blocking is False
        for item in observed.route_plan.diagnostics
        if item.code == "envelope-settlement-translation"
    )
    assert published
    for item in published:
        assert item.blocking is False
        assert "#1658" in item.message


def test_an_unallocatable_claim_names_the_author_pin_that_spans_it() -> None:
    """The section swallowing the claimed gap is one the author's grid placed,
    so the diagnostic names the pin rather than reporting a missing corridor
    nobody owns."""
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)
    pinned = [item for item in settlement.obstructions if item.pinned_section_ids]
    assert pinned, "fixture no longer exercises an author-pinned span"
    for item in pinned:
        assert "the authored grid pins" in item.message
        for section_id in item.pinned_section_ids:
            assert graph.layout_provenance.author_owns_grid(section_id)
            assert section_id in item.blocking_section_ids


def test_a_corridor_narrower_than_its_reservation_fails_the_strict_path() -> None:
    """A route drawn through a violated hard clearance is rejected, and the
    rejection names everything needed to act on it: who claimed the corridor,
    the span it claimed, what bounds it, and both widths."""
    graph, plan = _observe(TOPOLOGIES / "convergence_fold_diamond.mmd")
    settlement = settle_route_envelopes(graph, plan)
    target, boundary = _widest_slack_row_reservation(graph, plan)
    realised = realise_reservation(graph, target)
    assert realised is not None
    _narrow(graph, SettlementAxis.ROW, boundary, realised.capacity_slack + 4.0)

    squeezed = realise_reservation(graph, target)
    assert squeezed is not None and squeezed.capacity_slack < 0
    with pytest.raises(LayoutInvariantError) as caught:
        assert_reservations_are_settled(graph, plan, settlement, strict=True)

    message = str(caught.value)
    for claimant in target.claimant_member_ids:
        assert claimant in message
    assert f"columns {target.span.min_column}-{target.span.max_column}" in message
    assert f"rows {target.span.min_row}-{target.span.max_row}" in message
    for blocker in squeezed.negative_blocker_ids + squeezed.positive_blocker_ids:
        assert blocker in message
    assert f"{squeezed.required_width:.1f}px" in message
    assert f"{squeezed.available_width:.1f}px" in message


def test_an_unmet_handed_demand_fails_the_strict_path() -> None:
    """Settlement's own postcondition is enforced, not just reported: a demand
    it was handed and did not meet stops the render rather than being drawn."""
    graph, plan = _observe(ROOT / "examples" / "rnaseq_sections.mmd")
    reservation = next(
        item
        for item in plan.reservations
        if isinstance(item.region, RowGapRegion | ColumnGapRegion)
    )
    shortfall = SettlementShortfall(
        reservation.id, reservation.claimant_member_ids, 64.0, 12.0, True
    )
    settlement = EnvelopeSettlement((), (), (), (shortfall,))

    with pytest.raises(LayoutInvariantError) as caught:
        assert_reservations_are_settled(graph, plan, settlement, strict=True)
    message = str(caught.value)
    for claimant in reservation.claimant_member_ids:
        assert claimant in message
    assert "64.00px" in message
    assert "12.00px" in message

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        assert_reservations_are_settled(graph, plan, settlement, strict=False)
    assert any(shortfall.message in str(item.message) for item in caught_warnings)


def test_a_claim_with_no_allocatable_gap_does_not_fail_the_strict_path() -> None:
    """The counterpart to the two rejections above.  A span across the boundary
    is a ledger coherence problem with no arrangement to reject, so it is
    reported without failing a map that is otherwise drawable."""
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph, plan = _observe(path)
    settlement = settle_route_envelopes(graph, plan)
    assert settlement.obstructions
    assert any(not item.describes_a_gap for item in settlement.shortfalls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_reservations_are_settled(graph, plan, settlement, strict=True)


def _widest_slack_row_reservation(graph, plan):
    """The row reservation with the most spare width, and its boundary.

    Every corridor at one boundary shares its translation, so squeezing the
    roomiest boundary by the least-roomy resident's slack is what makes exactly
    one claim short.
    """
    by_boundary: dict[int, list] = {}
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is not None:
            by_boundary.setdefault(reservation.region.lower_row, []).append(
                (reservation, realised)
            )
    boundary, residents = max(
        by_boundary.items(),
        key=lambda item: min(got.capacity_slack for _res, got in item[1]),
    )
    target, _ = min(residents, key=lambda pair: pair[1].capacity_slack)
    return target, boundary


def test_a_compatibility_system_with_adequate_corridors_names_its_owner() -> None:
    graph, plan = _observe(TOPOLOGIES / "merge_bottom_row_bypass.mmd")
    settlement = settle_route_envelopes(graph, plan)
    assert settlement.compatibility_ownership
    for item in settlement.compatibility_ownership:
        assert item.convergence_reason
        assert item.corridor_count > 0
        assert item.worst_capacity_slack >= 0.0
        assert "#1658" in item.owner


@pytest.mark.parametrize("path", DEFICIT_CORPUS, ids=lambda item: item.name)
def test_a_reserved_inter_row_corridor_never_forces_the_router_to_improvise(
    path: Path,
) -> None:
    """The router's narrow-band fallback biases a run against the header badge
    it cannot clear.  A corridor that owns a reservation must never leave the
    router in that position, so the band it measures has to fit outright."""
    observed = _rendered_plan(path, permissive=True)
    query = build_route_plan_query(observed.route_plan)
    checked = 0
    for reservation in observed.route_plan.reservations:
        if not isinstance(reservation.region, RowGapRegion):
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is None:
            continue
        checked += 1
        assert _inter_row_band_fits(realised.region_start, realised.region_end)
    assert checked


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
