"""Route reservations preserve corridor ownership and signed final evidence."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
from layout_metrics import compute_metrics

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.route_plan import (
    BindingKind,
    ExitTurnDisposition,
    GridSpan,
    ReservationDecisionKind,
    SharedReferenceId,
    build_route_plan_query,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    CanvasSide,
    ColumnGapRegion,
    CorridorKind,
    CorridorMeasurementScope,
    CorridorOrientation,
    RowGapRegion,
)
from nf_metro.layout.routing import (
    compute_station_offsets,
    observe_route_edges,
    route_edges,
)
from nf_metro.layout.routing.common import Direction, apply_route_offsets
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.render.manifest import read_manifest
from nf_metro.render.plan import freeze_render_value
from nf_metro.render.svg import (
    build_observed_render_plan,
    build_render_plan,
    emit_render_plan,
)

ROOT = Path(__file__).parents[1]
TOPOLOGIES = ROOT / "examples" / "topologies"
REPORT_HO = ROOT / "tests" / "fixtures" / "route_reservations" / "reportho.metro"
RESERVATION_CORPUS = tuple(
    TOPOLOGIES / name
    for name in (
        "inter_row_wrap_clearance.mmd",
        "cross_row_gap_wrap.mmd",
        "merge_bottom_row_bypass.mmd",
        "corridor_narrow_gap_fallback.mmd",
        "fan_bypass_shared_band.mmd",
        "packed_cell_right_exit_left_entry_wrap.mmd",
        "opposing_bypass_corridor.mmd",
        "opposing_return_row_pair.mmd",
        "lr_to_tb_top_near_vertical.mmd",
    )
) + (ROOT / "tests" / "fixtures" / "regressions" / "stacked_collector_fanin.mmd",)

EXPECTED_RESERVATION_CLAIMS = {
    "inter_row_wrap_clearance.mmd": (
        (21, 1),
        (21, 2),
        (21, 3),
        (22, 1),
        (22, 2),
        (22, 3),
        (23, 1),
        (23, 2),
        (23, 3),
    ),
    "cross_row_gap_wrap.mmd": ((20, 1), (20, 2), (21, 1), (21, 2), (22, 0)),
    "merge_bottom_row_bypass.mmd": (
        (11, 1),
        (12, 1),
        (12, 2),
        (12, 3),
        (14, 1),
    ),
    "corridor_narrow_gap_fallback.mmd": (
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (13, 0),
        (13, 1),
        (13, 2),
        (13, 3),
        (14, 1),
        (14, 2),
        (14, 3),
    ),
    "fan_bypass_shared_band.mmd": ((9, 1), (9, 2), (9, 3), (9, 4)),
    "packed_cell_right_exit_left_entry_wrap.mmd": (
        (57, 1),
        (57, 2),
        (58, 1),
        (58, 2),
        (59, 1),
        (59, 2),
        (60, 1),
        (60, 2),
        (61, 0),
        (61, 1),
        (61, 2),
        (61, 3),
        (61, 4),
        (62, 0),
        (62, 1),
        (62, 2),
        (62, 3),
        (63, 0),
        (64, 0),
        (65, 0),
        (66, 0),
        (67, 0),
        (68, 0),
        (68, 2),
        (68, 3),
        (69, 0),
        (69, 2),
        (69, 3),
        (70, 1),
        (70, 2),
        (70, 3),
        (71, 0),
        (71, 1),
    ),
    "opposing_bypass_corridor.mmd": (
        (18, 0),
        (19, 1),
        (19, 2),
        (20, 0),
        (21, 0),
        (22, 0),
        (22, 1),
        (22, 2),
        (23, 1),
        (23, 2),
        (25, 1),
        (25, 2),
    ),
    "opposing_return_row_pair.mmd": (
        (8, 0),
        (9, 0),
        (10, 0),
        (10, 1),
        (10, 2),
        (11, 0),
        (11, 1),
        (11, 2),
    ),
    "lr_to_tb_top_near_vertical.mmd": ((4, 1), (4, 2)),
    "stacked_collector_fanin.mmd": (
        *((rank, 0) for rank in range(200, 212)),
        (213, 0),
        (215, 0),
        (217, 0),
        (197, 1),
        (198, 1),
        (199, 1),
        *((rank, 1) for rank in range(206, 218)),
        (197, 2),
        (198, 2),
        (199, 2),
        *((rank, 2) for rank in range(206, 218)),
        (197, 3),
        (198, 3),
        (199, 3),
        *((rank, 3) for rank in range(206, 218)),
        *((rank, 4) for rank in range(206, 218)),
    ),
}


def _observe(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        observation = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        )
    return graph, observation.routes, observation.plan


def _report_reservation(plan):
    matches = tuple(
        reservation
        for reservation in plan.reservations
        if reservation.kind is CorridorKind.DIRECT_INTER_ROW_BAND
        and reservation.route_family_ids == (RouteFamilyId.MERGE_TRUNK,)
        and isinstance(reservation.region, RowGapRegion)
    )
    assert len(matches) == 1
    return matches[0]


def _reservation_order(plan, reservation):
    system_rank = {item.id: rank for rank, item in enumerate(plan.systems)}
    member_rank = {item.id: rank for rank, item in enumerate(plan.members)}
    region = reservation.region
    if isinstance(region, RowGapRegion):
        region_key = region.kind.value, region.upper_row, region.lower_row
    elif isinstance(region, ColumnGapRegion):
        region_key = region.kind.value, region.left_column, region.right_column
    else:
        region_key = region.kind.value, list(CanvasSide).index(region.side), -1
    return (
        list(CorridorOrientation).index(reservation.orientation),
        region_key,
        reservation.span.min_column,
        reservation.span.max_column,
        reservation.span.min_row,
        reservation.span.max_row,
        system_rank[reservation.system_id],
        min(member_rank[item] for item in reservation.claimant_member_ids),
        reservation.id,
    )


def test_reportho_preserves_the_full_authored_corridor() -> None:
    graph, _routes, plan = _observe(REPORT_HO)
    reservation = _report_reservation(plan)
    query = build_route_plan_query(plan)
    realised = query.realised_reservation(reservation.id)
    assert realised is not None

    assert reservation.region == RowGapRegion(0, 1)
    assert reservation.span == GridSpan(0, 5, 0, 1)
    assert reservation.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
    assert reservation.lane_count == 1
    assert reservation.bundle_width == 0
    assert reservation.negative_side_clearance == 26
    assert reservation.positive_side_clearance == 52
    assert reservation.minimum_width == 78
    assert len(reservation.connector_ids) == 12
    assert len(reservation.claimant_member_ids) == 2
    assert len(reservation.claims) == 1
    assert set(reservation.keep_out_classes) >= {
        "section",
        "header",
        "label",
        "marker",
    }
    assert all(len(str(item)) < 64 for item in reservation.demand_ids)
    assert len(str(reservation.id)) < 64

    assert realised.coordinate == pytest.approx(492.0)
    assert realised.longitudinal_axis.value == "x"
    assert realised.longitudinal_start == pytest.approx(356.0)
    assert realised.longitudinal_end == pytest.approx(2314.0)
    assert realised.region_start == pytest.approx(466.0)
    assert realised.region_end == pytest.approx(544.0)
    assert realised.available_width == pytest.approx(78.0)
    assert realised.required_width == pytest.approx(78.0)
    assert realised.capacity_slack == pytest.approx(0.0)
    assert realised.negative_side_slack == pytest.approx(0.0)
    assert realised.positive_side_slack == pytest.approx(0.0)
    assert realised.negative_blocker_ids == ("section-bottom:fetch_ortho",)
    assert realised.positive_blocker_ids == ("section-header:report",)

    # A satisfied corridor publishes no shortfall record.
    assert not [
        item
        for item in plan.reservation_diagnostics
        if item.reservation_id == reservation.id
    ]

    assert graph.route_topology is not None
    connector_by_id = {
        connector.id: connector for connector in graph.route_topology.connectors
    }
    assert {
        connector_by_id[item].source_section for item in reservation.connector_ids
    } >= {"input", "fetch_ortho", "merge_ids", "score_orthologs"}
    assert {
        connector_by_id[item].target_section for item in reservation.connector_ids
    } == {"report"}


def test_reportho_rejects_overlapping_opposed_source_turn_axes() -> None:
    _graph, _routes, plan = _observe(REPORT_HO)
    exit_turn_plan = next(
        item for item in plan.exit_turn_plans if item.source_id == "__junction_15"
    )

    assert exit_turn_plan.disposition is ExitTurnDisposition.LEGACY
    assert exit_turn_plan.legacy_reason == "overlapping-planned-turn-axes"


def test_reportho_ownership_does_not_depend_on_resolved_section_pairs() -> None:
    graph, _routes, plan = _observe(REPORT_HO)
    reservation = _report_reservation(plan)

    physical_section_pairs = {
        (
            graph.stations[edge.source].section_id,
            graph.stations[edge.target].section_id,
        )
        for edge in graph.edges
    }
    assert ("fetch_ortho", "report") not in physical_section_pairs
    realised = build_route_plan_query(plan).realised_reservation(reservation.id)
    assert realised is not None
    assert "fetch_ortho" in {
        item.removeprefix("section-bottom:") for item in realised.negative_blocker_ids
    }


@pytest.mark.parametrize(
    ("name", "kind", "region_type"),
    (
        (
            "inter_row_wrap_clearance.mmd",
            CorridorKind.DIRECT_INTER_ROW_BAND,
            RowGapRegion,
        ),
        ("fan_bypass_shared_band.mmd", CorridorKind.BYPASS_BAND, CanvasRegion),
        ("cross_row_gap_wrap.mmd", CorridorKind.OVER_TOP_BAND, CanvasRegion),
        (
            "merge_bottom_row_bypass.mmd",
            CorridorKind.INTER_COLUMN_CHANNEL,
            ColumnGapRegion,
        ),
    ),
)
def test_supported_corridor_families_publish_complete_records(
    name: str, kind: CorridorKind, region_type: type
) -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / name)
    query = build_route_plan_query(plan)
    reservation = next(
        item
        for item in plan.reservations
        if item.kind is kind and isinstance(item.region, region_type)
    )

    assert reservation.connector_ids
    assert reservation.claimant_member_ids
    assert reservation.claims
    assert reservation.route_family_ids
    assert reservation.keep_out_classes
    assert reservation.provenance
    assert reservation.minimum_width > 0
    assert query.shared_reference(reservation.reference_id).claimant_member_ids == (
        reservation.claimant_member_ids
    )
    (demand_id,) = reservation.demand_ids
    demand = query.demand(demand_id)
    assert demand.span == reservation.span
    assert demand.lane_count == reservation.lane_count
    assert demand.minimum_size == reservation.minimum_width
    assert query.reservation(reservation.id) is reservation
    assert reservation in query.reservations_for_system(reservation.system_id)
    assert all(
        reservation in query.reservations_for_member(member_id)
        for member_id in reservation.claimant_member_ids
    )


def test_narrow_corridor_fallback_retains_the_original_demand() -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / "corridor_narrow_gap_fallback.mmd")
    reservation = next(
        item
        for item in plan.reservations
        if item.kind is CorridorKind.BYPASS_BAND
        and isinstance(item.region, RowGapRegion)
    )
    realised = build_route_plan_query(plan).realised_reservation(reservation.id)
    assert realised is not None

    assert reservation.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
    assert realised.available_width == pytest.approx(reservation.minimum_width)
    assert realised.coordinate > realised.region_end
    assert realised.positive_side_slack < 0
    assert any(
        item.reservation_id == reservation.id for item in plan.reservation_diagnostics
    )


@pytest.mark.parametrize(
    "name",
    (
        "bottom_row_climb_clear_corridor.mmd",
        "divergent_fanout_split.mmd",
    ),
)
def test_native_canvas_detours_do_not_manufacture_topology_gap_intent(
    name: str,
) -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / name)
    bypasses = tuple(
        item
        for item in plan.reservations
        if item.route_family_ids == (RouteFamilyId.BYPASS_FAMILY,)
        and isinstance(item.region, CanvasRegion)
    )

    assert bypasses
    assert all(
        item.measurement_scope is CorridorMeasurementScope.OBSERVED_RUN
        for item in bypasses
    )


@pytest.mark.parametrize(
    "name",
    (
        "merge_around_below_leftmost.mmd",
        "merge_leftmost_sink_branch.mmd",
    ),
)
def test_canvas_left_merge_risers_do_not_claim_a_nearest_column_gap(
    name: str,
) -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / name)

    assert any(
        item.route_family_ids == (RouteFamilyId.MERGE_TRUNK,)
        and item.region == CanvasRegion(CanvasSide.LEFT)
        and item.measurement_scope is CorridorMeasurementScope.OBSERVED_RUN
        for item in plan.reservations
    )
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    assert all(
        realised_by_id[item.id].negative_side_slack >= 0
        and realised_by_id[item.id].positive_side_slack >= 0
        for item in plan.reservations
        if item.route_family_ids == (RouteFamilyId.MERGE_TRUNK,)
        and isinstance(item.region, ColumnGapRegion)
        and item.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
    )


def test_plain_l_shape_does_not_publish_a_horizontal_wrap_band() -> None:
    _graph, _routes, plain_plan = _observe(
        TOPOLOGIES / "fanout_line_reused_nonadjacent_leg.mmd"
    )
    _graph, _routes, wrap_plan = _observe(TOPOLOGIES / "inter_row_wrap_clearance.mmd")

    assert all(
        not isinstance(item.region, RowGapRegion) for item in plain_plan.reservations
    )
    assert any(
        item.kind is CorridorKind.DIRECT_INTER_ROW_BAND
        and isinstance(item.region, RowGapRegion)
        for item in wrap_plan.reservations
    )


def test_coincident_concurrent_approaches_share_one_physical_lane() -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / "multi_input_convergence.mmd")
    reservation = next(
        item
        for item in plan.reservations
        if item.kind is CorridorKind.INTER_COLUMN_CHANNEL
    )
    members = {member.id: member for member in plan.members}

    assert len(reservation.claims) == 3
    assert reservation.lane_count == 1
    assert reservation.lanes[0].claim_indices == (0, 1, 2)
    assert len({claim.allocation_coordinate for claim in reservation.claims}) == 1
    assert {members[claim.member_id].line_id for claim in reservation.claims} == {
        "main"
    }


def test_stacked_collector_reuses_three_lanes_across_twelve_claims() -> None:
    path = ROOT / "tests" / "fixtures" / "regressions" / "stacked_collector_fanin.mmd"
    _graph, _routes, plan = _observe(path)
    reservation = next(
        item
        for item in plan.reservations
        if item.kind is CorridorKind.INTER_COLUMN_CHANNEL and len(item.claims) == 12
    )

    assert reservation.lane_count == 3
    assert sorted(len(lane.claim_indices) for lane in reservation.lanes) == [4, 4, 4]
    assert reservation.bundle_width == pytest.approx(8.0)


def test_asymmetric_grid_spans_select_provenance_on_the_canonical_axes() -> None:
    path = ROOT / "tests" / "fixtures" / "regressions" / "stacked_collector_fanin.mmd"
    graph, _routes, plan = _observe(path)
    asymmetric_sections = {
        section.id
        for section in graph.sections.values()
        if section.grid_row_span != section.grid_col_span
    }
    assert asymmetric_sections

    for reservation in plan.reservations:
        expected = {
            section.id
            for section in graph.sections.values()
            if reservation.span.min_column
            <= section.grid_col + section.grid_col_span - 1
            and section.grid_col <= reservation.span.max_column
            and reservation.span.min_row <= section.grid_row + section.grid_row_span - 1
            and section.grid_row <= reservation.span.max_row
        }
        actual = {
            item.subject_id
            for item in reservation.provenance
            if item.kind is ReservationDecisionKind.SECTION_GRID
        }
        assert actual == expected


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "examples" / "showcase" / "seqinspector.mmd",
        TOPOLOGIES / "bottom_row_climb_clear_corridor.mmd",
        TOPOLOGIES / "convergent_offrow_exit_climb.mmd",
        TOPOLOGIES / "exit_corner_offset_dogleg.mmd",
    ),
    ids=lambda path: path.name,
)
def test_claim_ranges_reference_the_exact_final_point_pairs(path: Path) -> None:
    graph, routes, plan = _observe(path)
    offsets = compute_station_offsets(graph)
    saw_skipped_predecessor = False

    for reservation in plan.reservations:
        for claim in reservation.claims:
            points = apply_route_offsets(routes[claim.path_rank], offsets)
            assert claim.segment_end_rank < len(points) - 1
            start = points[claim.segment_rank]
            end = points[claim.segment_end_rank + 1]
            if reservation.orientation is CorridorOrientation.HORIZONTAL:
                expected_start, expected_end = sorted((start[0], end[0]))
                expected_coordinate = start[1]
            else:
                expected_start, expected_end = sorted((start[1], end[1]))
                expected_coordinate = start[0]
            assert claim.longitudinal_start == pytest.approx(expected_start)
            assert claim.longitudinal_end == pytest.approx(expected_end)
            assert claim.allocation_coordinate == pytest.approx(expected_coordinate)
            saw_skipped_predecessor |= any(
                not (
                    abs(first[0] - second[0]) <= 1e-6
                    and abs(first[1] - second[1]) > 1e-6
                    or abs(first[1] - second[1]) <= 1e-6
                    and abs(first[0] - second[0]) > 1e-6
                )
                for first, second in zip(
                    points[: claim.segment_rank],
                    points[1 : claim.segment_rank + 1],
                    strict=True,
                )
            )

    assert saw_skipped_predecessor


def test_opposing_routes_remain_separate_directional_claims() -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / "opposing_return_row_pair.mmd")
    row_claims = tuple(
        item for item in plan.reservations if isinstance(item.region, RowGapRegion)
    )
    assert {item.direction for item in row_claims} == {Direction.R, Direction.L}
    assert all(item.lane_count == 1 for item in row_claims)
    assert len({item.id for item in row_claims}) == len(row_claims)


def test_large_valid_slack_is_not_reported_as_waste() -> None:
    _graph, _routes, plan = _observe(TOPOLOGIES / "opposing_return_row_pair.mmd")
    realised = max(plan.realised_reservations, key=lambda item: item.capacity_slack)
    assert realised.capacity_slack == pytest.approx(228.0)
    assert realised.negative_side_slack > 0
    assert realised.positive_side_slack > 0
    assert all(
        item.reservation_id != realised.reservation_id
        for item in plan.reservation_diagnostics
    )


def test_reservation_corpus_has_one_linked_record_per_observed_claim() -> None:
    for path in RESERVATION_CORPUS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
            observed = build_observed_render_plan(graph, resolve_theme(None, graph))
        routes = observed.plan.routes
        plan = observed.route_plan
        query = build_route_plan_query(plan)
        reservation_reference_ids = {item.reference_id for item in plan.reservations}
        reservation_demand_ids = {
            demand_id for item in plan.reservations for demand_id in item.demand_ids
        }
        assert sum(
            item.id in reservation_reference_ids for item in plan.shared_references
        ) == len(plan.reservations), path
        assert sum(item.id in reservation_demand_ids for item in plan.demands) == len(
            plan.reservations
        ), path
        assert plan.reservations, path
        assert tuple(plan.reservations) == tuple(
            sorted(
                plan.reservations,
                key=lambda item: _reservation_order(plan, item),
            )
        ), path
        realised_ids = {item.reservation_id for item in plan.realised_reservations}
        claims = tuple(
            (claim.path_id, claim.segment_rank)
            for reservation in plan.reservations
            for claim in reservation.claims
        )
        path_id_by_rank = {
            binding.path_rank: binding.path_id
            for binding in plan.bindings
            if binding.kind is BindingKind.EMITTED
        }
        expected_claims = {
            (path_id_by_rank[path_rank], segment_rank)
            for path_rank, segment_rank in EXPECTED_RESERVATION_CLAIMS[path.name]
        }
        assert len(claims) == len(set(claims)), path
        assert set(claims) == expected_claims, path
        for reservation in plan.reservations:
            assert reservation.lane_count == len(reservation.lanes), path
            assert sorted(
                index for lane in reservation.lanes for index in lane.claim_indices
            ) == list(range(len(reservation.claims))), path
            assert len(
                {
                    (item.path_rank, item.segment_rank, item.segment_end_rank)
                    for item in reservation.claims
                }
            ) == len(reservation.claims), path
            for claim in reservation.claims:
                assert claim.path_rank < len(routes), path
                start = routes[claim.path_rank].points[claim.segment_rank]
                end = routes[claim.path_rank].points[claim.segment_end_rank + 1]
                assert claim.longitudinal_start == pytest.approx(
                    min(start[0], end[0])
                    if reservation.orientation is CorridorOrientation.HORIZONTAL
                    else min(start[1], end[1])
                ), path
                assert claim.longitudinal_end == pytest.approx(
                    max(start[0], end[0])
                    if reservation.orientation is CorridorOrientation.HORIZONTAL
                    else max(start[1], end[1])
                ), path
                assert claim.allocation_coordinate == pytest.approx(
                    start[1]
                    if reservation.orientation is CorridorOrientation.HORIZONTAL
                    else start[0]
                ), path
                (binding,) = query.bindings_for(claim.member_id)
                assert binding.kind is BindingKind.EMITTED, path
                assert binding.path_id == claim.path_id, path
                assert binding.path_rank == claim.path_rank, path
            assert reservation.id in realised_ids, path
        assert tuple(
            item.reservation_id for item in plan.realised_reservations
        ) == tuple(item.id for item in plan.reservations if item.id in realised_ids), (
            path
        )
        assert tuple(
            item.reservation_id for item in plan.reservation_diagnostics
        ) == tuple(
            item.id
            for item in plan.reservations
            if any(
                diagnostic.reservation_id == item.id
                for diagnostic in plan.reservation_diagnostics
            )
        ), path


@pytest.mark.parametrize(
    "name",
    (
        "fan_bypass_shared_band.mmd",
        "cross_row_gap_wrap.mmd",
        "merge_bottom_row_bypass.mmd",
        "bottom_row_climb_clear_corridor.mmd",
    ),
)
def test_observed_run_blockers_intersect_the_exact_final_run(name: str) -> None:
    path = TOPOLOGIES / name
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    query = build_route_plan_query(observed.route_plan)

    assert len(observed.route_plan.realised_reservations) == len(
        observed.route_plan.reservations
    )
    for reservation in observed.route_plan.reservations:
        if reservation.measurement_scope is not CorridorMeasurementScope.OBSERVED_RUN:
            continue
        realised = query.realised_reservation(reservation.id)
        assert realised is not None
        for blocker_id in (
            *realised.negative_blocker_ids,
            *realised.positive_blocker_ids,
        ):
            if blocker_id.startswith("canvas:"):
                continue
            section = graph.sections[blocker_id.split(":", 1)[1]]
            if realised.longitudinal_axis.value == "x":
                section_start = section.bbox_x
                section_end = section.bbox_x + section.bbox_w
            else:
                section_start = section.bbox_y
                section_end = section.bbox_y + section.bbox_h
            assert min(realised.longitudinal_end, section_end) > max(
                realised.longitudinal_start, section_start
            )


@pytest.mark.parametrize(
    "path",
    RESERVATION_CORPUS + (TOPOLOGIES / "fan_in_merge.mmd", REPORT_HO),
    ids=lambda path: path.name,
)
def test_observed_render_plan_is_byte_and_metric_neutral(path: Path) -> None:
    source = path.read_text()
    source_dir = str(path.parent)
    plain_graph = prepare_graph(source, source_dir=source_dir)
    observed_graph = prepare_graph(source, source_dir=source_dir)
    if path == REPORT_HO:
        plain_graph.permissive = True
        observed_graph.permissive = True
    plain_plan = build_render_plan(plain_graph, resolve_theme(None, plain_graph))
    observed = build_observed_render_plan(
        observed_graph, resolve_theme(None, observed_graph)
    )

    assert freeze_render_value(observed.plan) == freeze_render_value(plain_plan)
    plain_svg = emit_render_plan(plain_plan)
    observed_svg = emit_render_plan(observed.plan)
    assert observed_svg == plain_svg
    assert read_manifest(observed_svg) == read_manifest(plain_svg)
    assert json.dumps(
        compute_metrics(observed_graph, plan=observed.plan), sort_keys=True
    ) == json.dumps(compute_metrics(plain_graph, plan=plain_plan), sort_keys=True)
    assert observed.route_plan.reservations
    assert len(observed.route_plan.realised_reservations) == len(
        observed.route_plan.reservations
    )


@pytest.mark.parametrize(
    "path",
    RESERVATION_CORPUS + (REPORT_HO,),
    ids=lambda path: path.name,
)
def test_observer_toggle_preserves_settled_graph_and_raw_routes(path: Path) -> None:
    source = path.read_text()
    source_dir = str(path.parent)
    observed_graph = prepare_graph(source, source_dir=source_dir)
    graph_before = copy.deepcopy(observed_graph)
    offsets = compute_station_offsets(observed_graph)
    observation = observe_route_edges(observed_graph, station_offsets=offsets)
    assert observed_graph == graph_before

    plain_graph = prepare_graph(source, source_dir=source_dir)
    plain_routes = route_edges(
        plain_graph, station_offsets=compute_station_offsets(plain_graph)
    )
    assert observed_graph == plain_graph
    assert freeze_render_value(observation.routes) == freeze_render_value(plain_routes)


def test_query_rejects_a_reservation_with_an_unknown_reference() -> None:
    _graph, _routes, plan = _observe(REPORT_HO)
    reservation = _report_reservation(plan)
    malformed = replace(
        reservation, reference_id=SharedReferenceId("missing-reference")
    )
    malformed_plan = replace(
        plan,
        reservations=tuple(
            malformed if item.id == reservation.id else item
            for item in plan.reservations
        ),
    )

    with pytest.raises(ValueError, match="unknown shared reference"):
        build_route_plan_query(malformed_plan)
