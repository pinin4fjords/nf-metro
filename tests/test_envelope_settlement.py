"""Final route reservations drive one bounded envelope settlement."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import nf_metro.layout.envelope_settlement as settlement_module
from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import graph_offset_step
from nf_metro.layout.envelope_settlement import (
    EnvelopeAxis,
    EnvelopeSettlementError,
    assert_route_envelopes_satisfied,
    settle_route_envelopes,
)
from nf_metro.layout.route_plan import DemandAxis
from nf_metro.layout.route_reservations import RowGapRegion
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.parser.model import PermissiveGuardWarning
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]
REPORTHO = ROOT / "tests" / "fixtures" / "route_reservations" / "reportho.metro"
TOPOLOGIES = ROOT / "examples" / "topologies"


def _observe(graph):
    offsets = compute_station_offsets(graph)
    return observe_route_edges(graph, station_offsets=offsets)


def _local_geometry(graph):
    return {
        station.id: (
            station.x - graph.sections[station.section_id].bbox_x,
            station.y - graph.sections[station.section_id].bbox_y,
        )
        for station in graph.stations.values()
        if station.section_id in graph.sections
    }


def _absolute_geometry(graph):
    return (
        tuple(
            (item.id, item.bbox_x, item.bbox_y, item.bbox_w, item.bbox_h)
            for item in graph.sections.values()
        ),
        tuple((item.id, item.x, item.y) for item in graph.stations.values()),
        tuple((item.id, item.x, item.y) for item in graph.ports.values()),
    )


def test_reportho_reservation_settles_transactionally_and_idempotently() -> None:
    graph = prepare_graph(REPORTHO.read_text(), source_dir=str(REPORTHO.parent))
    before = _observe(graph)
    reservation = next(
        item
        for item in before.plan.reservations
        if item.region == RowGapRegion(0, 1)
        and len(item.connector_ids) == 12
        and item.minimum_width == 78
    )
    diagnostic = next(
        item
        for item in before.plan.reservation_diagnostics
        if item.reservation_id == reservation.id
    )
    assert diagnostic.capacity_slack == pytest.approx(-146.2)

    local_before = _local_geometry(graph)
    settled = settle_route_envelopes(graph, before.plan)
    assert settled.translations
    assert all(item.amount >= 0 for item in settled.translations)
    assert _local_geometry(graph) == local_before

    after = _observe(graph)
    realised = next(
        item
        for item in after.plan.realised_reservations
        if item.reservation_id
        == next(
            current.id
            for current in after.plan.reservations
            if current.region == RowGapRegion(0, 1)
            and len(current.connector_ids) == 12
            and current.minimum_width == 78
        )
    )
    assert realised.available_width >= realised.required_width
    assert realised.negative_side_slack >= 0
    assert realised.positive_side_slack >= 0

    geometry_once = {
        station.id: (station.x, station.y) for station in graph.stations.values()
    }
    settled_again = settle_route_envelopes(graph, after.plan)
    assert settled_again.translations == ()
    assert {
        station.id: (station.x, station.y) for station in graph.stations.values()
    } == geometry_once


def test_settlement_has_a_finite_monotone_boundary_bound_and_one_owner() -> None:
    graph = prepare_graph(REPORTHO.read_text(), source_dir=str(REPORTHO.parent))
    observed = _observe(graph)
    plan_before = observed.plan
    ledger_before = (
        plan_before.reservations,
        plan_before.shared_references,
        plan_before.demands,
    )
    sections_before = {
        item.id: (
            item.grid_col,
            item.grid_row,
            item.grid_col_span,
            item.grid_row_span,
            item.direction,
            item.bbox_x,
            item.bbox_y,
            item.bbox_w,
            item.bbox_h,
        )
        for item in graph.sections.values()
    }
    port_sides = {item.id: item.side for item in graph.ports.values()}

    settlement = settle_route_envelopes(graph, plan_before)

    assert len(settlement.translations) <= settlement.boundary_count
    quantum = graph_offset_step(graph)
    assert all(
        item.amount > 0
        and item.amount / quantum == pytest.approx(round(item.amount / quantum))
        for item in settlement.translations
    )
    for translation in settlement.translations:
        positive = translation.boundary[1]
        expected = tuple(
            section.id
            for section in sorted(graph.sections.values(), key=lambda item: item.id)
            if (
                section.grid_col
                if translation.axis is EnvelopeAxis.X
                else section.grid_row
            )
            >= positive
        )
        assert translation.section_ids == expected

    for first in graph.sections.values():
        before = sections_before[first.id]
        assert (
            first.grid_col,
            first.grid_row,
            first.grid_col_span,
            first.grid_row_span,
            first.direction,
            first.bbox_w,
            first.bbox_h,
        ) == (*before[:5], *before[7:])
        for second in graph.sections.values():
            second_before = sections_before[second.id]
            if first.grid_col < second.grid_col:
                assert second.bbox_x - first.bbox_x >= (
                    second_before[5] - before[5] - 1e-6
                )
            if first.grid_row < second.grid_row:
                assert second.bbox_y - first.bbox_y >= (
                    second_before[6] - before[6] - 1e-6
                )
    assert {item.id: item.side for item in graph.ports.values()} == port_sides
    assert (
        plan_before.reservations,
        plan_before.shared_references,
        plan_before.demands,
    ) == ledger_before
    assert plan_before.reservations is ledger_before[0]
    assert plan_before.shared_references is ledger_before[1]
    assert plan_before.demands is ledger_before[2]


def test_satisfied_layout_is_an_exact_noop() -> None:
    path = TOPOLOGIES / "merge_adjacent_feeder.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)
    before = _absolute_geometry(graph)

    settlement = settle_route_envelopes(graph, observed.plan)

    assert settlement.translations == ()
    assert _absolute_geometry(graph) == before


def test_settlement_failure_rolls_back_geometry_and_leaves_ledger_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = prepare_graph(REPORTHO.read_text(), source_dir=str(REPORTHO.parent))
    observed = _observe(graph)
    before = _absolute_geometry(graph)
    ledger = (
        observed.plan.reservations,
        observed.plan.shared_references,
        observed.plan.demands,
    )

    def reject_translated_ledger(*_args, **_kwargs):
        raise EnvelopeSettlementError("synthetic post-translation failure")

    monkeypatch.setattr(
        settlement_module,
        "_capacity_proofs",
        reject_translated_ledger,
    )

    with pytest.raises(EnvelopeSettlementError, match="post-translation failure"):
        settle_route_envelopes(graph, observed.plan)

    assert _absolute_geometry(graph) == before
    assert observed.plan.reservations is ledger[0]
    assert observed.plan.shared_references is ledger[1]
    assert observed.plan.demands is ledger[2]


def test_strict_infeasible_pinned_gap_has_complete_attribution() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    graph.strict = True

    with pytest.raises(EnvelopeSettlementError) as raised:
        assert_route_envelopes_satisfied(graph, plan)

    message = str(raised.value)
    assert "reservation route-reservation:" in message
    assert "across columns 0-4 and rows 0-3" in message
    assert "blockers section-bottom:scaffolding" in message
    assert "required 119.00px" in message
    assert "available -190.00px" in message
    assert "conflicting pin scaffolding" in message
    assert "claimant members emission-member|" in message


def test_strict_render_rejects_infeasible_pins_before_route_guards() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    graph.strict = True

    with pytest.raises(EnvelopeSettlementError) as raised:
        build_observed_render_plan(graph, resolve_theme(None, graph))

    message = str(raised.value)
    assert "reservation route-reservation:" in message
    assert "across columns 0-4 and rows 0-3" in message
    assert "blockers section-bottom:scaffolding" in message
    assert "required 119.00px" in message
    assert "available -190.00px" in message
    assert "conflicting pin" in message
    assert "claimant members emission-member|" in message


@pytest.mark.parametrize(
    ("path", "direction"),
    (
        (TOPOLOGIES / "merge_adjacent_feeder.mmd", "LR"),
        (TOPOLOGIES / "serpentine_rl_bundle.mmd", "RL"),
        (TOPOLOGIES / "tb_bottom_exit_bundle_jog.mmd", "TB"),
        (TOPOLOGIES / "bt_to_lr.mmd", "BT"),
    ),
)
def test_all_flow_directions_use_the_same_axis_settlement(
    path: Path, direction: str
) -> None:
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    assert direction in {item.direction for item in graph.sections.values()}
    observed = _observe(graph)

    settlement = settle_route_envelopes(graph, observed.plan)

    assert len(settlement.translations) <= settlement.boundary_count


def test_observed_render_activates_reportho_reservation() -> None:
    graph = prepare_graph(REPORTHO.read_text(), source_dir=str(REPORTHO.parent))
    graph.strict = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _render_plan, route_plan = build_observed_render_plan(
            graph, resolve_theme(None, graph)
        )

    report_reservations = tuple(
        item
        for item in route_plan.reservations
        if item.region == RowGapRegion(0, 1)
        and len(item.connector_ids) == 12
        and item.minimum_width == 78
    )
    assert len(report_reservations) == 1
    assert not {
        diagnostic.reservation_id for diagnostic in route_plan.reservation_diagnostics
    }.intersection(item.id for item in report_reservations)
    assert not any(item.category is PermissiveGuardWarning for item in caught)


def test_settlement_translates_a_wholly_anchored_junction_with_its_row() -> None:
    path = TOPOLOGIES / "merge_right_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    junction = graph.stations["__junction_7"]
    section = graph.sections["extra"]
    relative_before = (junction.x - section.bbox_x, junction.y - section.bbox_y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_plan, _route_plan = build_observed_render_plan(
            graph, resolve_theme(None, graph)
        )

    settled_junction = render_plan.graph.stations["__junction_7"]
    settled_section = render_plan.graph.sections["extra"]
    assert (
        settled_junction.x - settled_section.bbox_x,
        settled_junction.y - settled_section.bbox_y,
    ) == relative_before
    assert not any(item.category is PermissiveGuardWarning for item in caught)


@pytest.mark.parametrize(
    "path",
    (
        TOPOLOGIES / "exit_run_three_drop_columns.mmd",
        TOPOLOGIES / "merge_around_below_leftmost.mmd",
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
        TOPOLOGIES / "merge_bottom_row_bypass.mmd",
        TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
        TOPOLOGIES / "merge_right_entry.mmd",
        ROOT / "examples" / "genomeassembly.mmd",
        ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
        ROOT / "examples" / "guide" / "03b_fan_in_merge.mmd",
    ),
)
def test_allocation_limited_convergences_exit_compatibility(path: Path) -> None:
    preflight_graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    preflight = _observe(preflight_graph).plan
    expected_membership = {
        item.id: (
            item.system_id,
            item.convergence_ids,
            item.connector_ids,
            item.member_ids,
            item.resolved_member_paths,
            item.resolved_member_edges,
        )
        for item in preflight.convergence_plans
    }
    expected_systems = {
        item.id: (
            item.connector_ids,
            item.line_ids,
            item.bundle_ids,
            item.exit_group_ids,
            item.entry_group_ids,
            item.divergence_ids,
            item.convergence_ids,
            item.member_ids,
            item.branch_ids,
            item.feeder_ids,
            item.exit_turn_plan_ids,
            item.fan_plan_ids,
            item.convergence_plan_ids,
        )
        for item in preflight.systems
    }
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))

    _render_plan, route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )

    assert route_plan.convergence_plans
    assert all(item.owns_geometry for item in route_plan.convergence_plans)
    assert {
        item.id: (
            item.system_id,
            item.convergence_ids,
            item.connector_ids,
            item.member_ids,
            item.resolved_member_paths,
            item.resolved_member_edges,
        )
        for item in route_plan.convergence_plans
    } == expected_membership
    assert {
        item.id: (
            item.connector_ids,
            item.line_ids,
            item.bundle_ids,
            item.exit_group_ids,
            item.entry_group_ids,
            item.divergence_ids,
            item.convergence_ids,
            item.member_ids,
            item.branch_ids,
            item.feeder_ids,
            item.exit_turn_plan_ids,
            item.fan_plan_ids,
            item.convergence_plan_ids,
        )
        for item in route_plan.systems
    } == expected_systems
    assert route_plan.provenance.lane_order == preflight.provenance.lane_order
    system_ids = {item.system_id for item in route_plan.convergence_plans}
    systems = tuple(item for item in route_plan.systems if item.id in system_ids)
    systems_by_id = {item.id: item for item in systems}
    assert systems
    assert all(item.shared_reference_ids for item in systems)
    assert all(item.demand_ids for item in systems)
    assert all(item.lane_order for item in route_plan.convergence_plans)
    assert all(item.shared_reference_ids for item in route_plan.convergence_plans)
    assert all(item.demand_ids for item in route_plan.convergence_plans)
    assert all(
        set(item.shared_reference_ids).issubset(
            systems_by_id[item.system_id].shared_reference_ids
        )
        and set(item.demand_ids).issubset(systems_by_id[item.system_id].demand_ids)
        for item in route_plan.convergence_plans
    )
    assert all(
        any(reservation.system_id == item.id for reservation in route_plan.reservations)
        for item in systems
    )
    assert not any(
        item.code == "convergence-plan-legacy" for item in route_plan.diagnostics
    )


def test_funcprofiler_compatibility_is_owned_by_fan_consolidation() -> None:
    path = TOPOLOGIES / "funcprofiler_upstream.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))

    _render_plan, route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )

    convergence_diagnostics = tuple(
        item
        for item in route_plan.diagnostics
        if item.code == "convergence-plan-legacy"
    )
    assert convergence_diagnostics
    assert all(
        "overlapping fan ownership" in item.message for item in convergence_diagnostics
    )
    assert all("owner #1658" in item.message for item in convergence_diagnostics)


@pytest.mark.parametrize(
    "path",
    (
        TOPOLOGIES / "merge_around_below_leftmost.mmd",
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
    ),
)
def test_shared_openings_consume_the_settled_claim_coordinate(path: Path) -> None:
    measured_graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(measured_graph)
    settlement = settle_route_envelopes(measured_graph, observed.plan)
    allocation_coordinates: dict[tuple[object, DemandAxis], set[float]] = {}
    for proof in settlement.capacity_proofs:
        for allocation in proof.allocations:
            allocation_coordinates.setdefault(
                (allocation.member_id, allocation.axis), set()
            ).add(allocation.coordinate)

    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    _render_plan, route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )
    opening_groups: dict[tuple[str, str, int], list[object]] = {}
    for plan in route_plan.convergence_plans:
        for landing in plan.landings:
            segment = landing.opening_turn_segment
            if segment is None:
                continue
            direction = 1 if segment[1][1] > segment[0][1] else -1
            opening_groups.setdefault(
                (landing.edge.line_id, landing.source_junction_id, direction), []
            ).append(landing)
    shared_groups = tuple(
        landings for landings in opening_groups.values() if len(landings) > 1
    )

    assert shared_groups
    for landings in shared_groups:
        assert len({item.opening_turn_coordinate for item in landings}) == 1
        for landing in landings:
            assert (
                landing.opening_turn_coordinate
                in allocation_coordinates[(landing.member_id, DemandAxis.X)]
            )


def test_pinned_organellar_compatibility_has_complete_capacity_evidence() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    measured_graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(measured_graph)
    settlement = settle_route_envelopes(measured_graph, observed.plan)
    limited_system_ids = {item.system_id for item in settlement.capacity_limitations}
    assert limited_system_ids
    assert all(
        "scaffolding" in item.pinned_section_ids
        for item in settlement.capacity_limitations
    )
    assert not {item.system_id for item in settlement.capacity_proofs}.intersection(
        limited_system_ids
    )

    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    _render_plan, route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )

    assert route_plan.convergence_plans
    assert not any(item.owns_geometry for item in route_plan.convergence_plans)
    diagnostics = tuple(
        item
        for item in route_plan.diagnostics
        if item.code == "convergence-plan-legacy"
    )
    assert diagnostics
    assert all(
        "infeasible under authored grid commitments" in item.message
        for item in diagnostics
    )
    assert all("scaffolding" in item.message for item in diagnostics)
    assert all("owner #1658" in item.message for item in diagnostics)
