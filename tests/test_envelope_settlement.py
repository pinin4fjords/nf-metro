"""Final route reservations drive one bounded envelope settlement."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

import nf_metro.layout.envelope_settlement as settlement_module
import nf_metro.layout.route_reservations as reservations_module
import nf_metro.layout.routing.core as routing_core
from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import (
    BUNDLE_TO_BUNDLE_CLEARANCE,
    COORD_TOLERANCE,
    TITLE_BAND_ROUTE_FLOOR,
    graph_offset_step,
)
from nf_metro.layout.envelope_settlement import (
    EnvelopeAxis,
    EnvelopeSettlementError,
    assert_route_envelopes_satisfied,
    settle_route_envelopes,
)
from nf_metro.layout.phases.guards import (
    LayoutInvariantError,
    assert_render_title_route_clearance,
)
from nf_metro.layout.route_plan import (
    DemandAxis,
    EmissionMemberId,
    build_route_plan_query,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    CanvasSide,
    CorridorMeasurementScope,
    RowGapRegion,
    realise_route_reservations,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.parser.model import PermissiveGuardWarning
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]
REPORTHO = ROOT / "tests" / "fixtures" / "route_reservations" / "reportho.metro"
TOPOLOGIES = ROOT / "examples" / "topologies"
CROSS_COLUMN = (
    ROOT / "tests" / "fixtures" / "regressions" / "cross_column_perp_entry_overflow.mmd"
)
SEED_41 = ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_41.mmd"
STRAIGHT_DROP = TOPOLOGIES / "straight_drop_below.mmd"
TITLE_BAND_TOP_ROUTES = tuple(
    TOPOLOGIES / name
    for name in (
        "bt_to_lr.mmd",
        "cross_col_top_entry.mmd",
        "cross_column_perp_drop.mmd",
        "cross_column_perp_drop_far_exit.mmd",
        "left_exit_sink_below.mmd",
        "lr_perp_top_exit_perp_entry.mmd",
        "lr_perp_top_exit_perp_entry_diverging.mmd",
        "lr_perp_top_exit_side_entry.mmd",
        "orbit_perp_exit_back_row_entry.mmd",
        "orbit_perp_exit_flow_entry.mmd",
        "orbit_perp_exit_perp_entry.mmd",
        "orbit_perp_exit_turning_entry.mmd",
        "tb_bottom_entry_flow_start.mmd",
        "tb_internal_diagonal.mmd",
        "tb_lr_exit_left.mmd",
        "tb_lr_exit_right.mmd",
    )
)


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


@pytest.mark.parametrize("fixture", TITLE_BAND_TOP_ROUTES, ids=lambda path: path.stem)
def test_settled_top_canvas_routes_clear_the_rendered_title(fixture: Path):
    graph = prepare_graph(fixture.read_text(), source_dir=str(fixture.parent))
    observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    top_route_coordinate = min(
        y for route in observed.plan.routes for _x, y in route.points
    )
    assert top_route_coordinate >= TITLE_BAND_ROUTE_FLOOR - COORD_TOLERANCE


def test_title_route_floor_is_not_reserved_for_bare_output():
    fixture = TOPOLOGIES / "cross_col_top_entry.mmd"
    graph = prepare_graph(
        fixture.read_text(), source_dir=str(fixture.parent), bare=True
    )
    observed = build_observed_render_plan(graph, resolve_theme(None, graph), bare=True)

    assert not graph.reserve_title_band
    assert min(y for route in observed.plan.routes for _x, y in route.points) < (
        TITLE_BAND_ROUTE_FLOOR
    )


def test_final_title_route_guard_rejects_an_underfloor_route():
    fixture = TOPOLOGIES / "cross_col_top_entry.mmd"
    graph = prepare_graph(fixture.read_text(), source_dir=str(fixture.parent))
    observation = _observe(graph)
    observation.routes[0].points[0] = (
        0.0,
        TITLE_BAND_ROUTE_FLOOR - COORD_TOLERANCE - 1.0,
    )

    with pytest.raises(LayoutInvariantError, match="inside the rendered title band"):
        assert_render_title_route_clearance(graph, observation.routes, strict=True)


def _straight_drop_canvas_component():
    graph = prepare_graph(
        STRAIGHT_DROP.read_text(), source_dir=str(STRAIGHT_DROP.parent)
    )
    plan = _observe(graph).plan
    component = next(
        component
        for component in settlement_module._boundary_components(
            settlement_module._canvas_reservations(plan, graph, plan)[CanvasSide.TOP]
        )
        if len(component) == 2
        and all(
            claim.endpoint_anchor_ids == ("__junction_3",)
            for item in component
            for claim in item.reservation.claims
        )
    )
    return graph, plan, component


def test_touching_endpoint_canvas_claims_preserve_authored_lane_separation() -> None:
    graph, plan, _component = _straight_drop_canvas_component()
    geometry = _absolute_geometry(graph)
    graph.strict = True

    first = settle_route_envelopes(graph, plan)
    second = settle_route_envelopes(graph, plan)
    canvas = next(
        proof
        for proof in first.capacity_proofs
        if proof.region == CanvasRegion(CanvasSide.TOP)
    )

    assert first.translations == ()
    assert first.capacity_limitations == ()
    assert [
        allocation.coordinate
        for reservation in canvas.reservations
        for allocation in reservation.allocations
    ] == [120.0, 124.0]
    assert second.translations == ()
    assert second.capacity_limitations == ()
    assert _absolute_geometry(graph) == geometry


@pytest.mark.parametrize("coordinate_sign", (-1, 1))
def test_touching_endpoint_separation_is_canvas_side_symmetric(
    coordinate_sign: int,
) -> None:
    graph, _plan, component = _straight_drop_canvas_component()

    packed = settlement_module._pack_component_claims(
        graph,
        component,
        coordinate_sign=coordinate_sign,
    )

    assert packed == ((120.0,), (124.0,))


@pytest.mark.parametrize(
    ("orientation", "direction"),
    (
        (
            reservations_module.CorridorOrientation.HORIZONTAL,
            reservations_module.Direction.R,
        ),
        (
            reservations_module.CorridorOrientation.HORIZONTAL,
            reservations_module.Direction.L,
        ),
        (
            reservations_module.CorridorOrientation.VERTICAL,
            reservations_module.Direction.D,
        ),
        (
            reservations_module.CorridorOrientation.VERTICAL,
            reservations_module.Direction.U,
        ),
    ),
)
def test_touching_endpoint_separation_is_axis_generic(
    orientation,
    direction,
) -> None:
    _graph, _plan, component = _straight_drop_canvas_component()
    kind = (
        component[0].reservation.kind
        if orientation is reservations_module.CorridorOrientation.HORIZONTAL
        else reservations_module.CorridorKind.INTER_COLUMN_CHANNEL
    )
    region = (
        component[0].reservation.region
        if orientation is reservations_module.CorridorOrientation.HORIZONTAL
        else CanvasRegion(CanvasSide.LEFT)
    )
    first, second = tuple(
        replace(
            item,
            reservation=replace(
                item.reservation,
                kind=kind,
                orientation=orientation,
                direction=direction,
                region=region,
            ),
        )
        for item in component
    )

    assert settlement_module._touching_endpoint_separation(first, 0, second, 0) == (
        pytest.approx(4.0)
    )


@pytest.mark.parametrize(
    "invalid_relationship",
    (
        "overlapping-interiors",
        "different-anchor",
        "missing-anchor",
        "different-system",
        "opposing-direction",
    ),
)
def test_touching_endpoint_separation_requires_exact_shared_ownership(
    invalid_relationship: str,
) -> None:
    _graph, _plan, component = _straight_drop_canvas_component()
    first, second = component
    if invalid_relationship == "overlapping-interiors":
        claim = first.reservation.claims[0]
        first = replace(
            first,
            reservation=replace(
                first.reservation,
                claims=(replace(claim, longitudinal_start=169.0),),
            ),
        )
    elif invalid_relationship in {"different-anchor", "missing-anchor"}:
        claim = second.reservation.claims[0]
        second = replace(
            second,
            reservation=replace(
                second.reservation,
                claims=(
                    replace(
                        claim,
                        endpoint_anchor_ids=("unrelated",)
                        if invalid_relationship == "different-anchor"
                        else (),
                    ),
                ),
            ),
        )
    elif invalid_relationship == "different-system":
        second = replace(
            second,
            reservation=replace(second.reservation, system_id="unrelated-system"),
        )
    else:
        second = replace(
            second,
            reservation=replace(
                second.reservation,
                direction=reservations_module.Direction.L,
            ),
        )

    assert settlement_module._touching_endpoint_separation(first, 0, second, 0) is None
    assert settlement_module._distinct_claim_separation(first, 0, second, 0) == (
        pytest.approx(12.0)
    )


def test_endpoint_touching_section_boundary_is_not_a_canvas_claim() -> None:
    graph = prepare_graph(SEED_41.read_text(), source_dir=str(SEED_41.parent))
    port = graph.stations["s3__entry_left_16"]
    section = graph.sections[port.section_id]

    assert not reservations_module._endpoint_segment_owns_canvas(
        graph, CanvasRegion(CanvasSide.TOP), port.y, (port.id,)
    )
    assert not reservations_module._endpoint_segment_owns_canvas(
        graph, CanvasRegion(CanvasSide.LEFT), section.bbox_x + 20, (port.id,)
    )
    assert reservations_module._endpoint_segment_owns_canvas(
        graph, CanvasRegion(CanvasSide.TOP), section.bbox_y - 20, (port.id,)
    )
    assert reservations_module._endpoint_segment_owns_canvas(
        graph, CanvasRegion(CanvasSide.LEFT), section.bbox_x - 20, (port.id,)
    )
    assert reservations_module._endpoint_segment_owns_canvas(
        graph, CanvasRegion(CanvasSide.LEFT), section.bbox_x, (port.id,)
    )

    plan = _observe(graph).plan
    member = next(
        item
        for item in plan.members
        if (item.edge.source, item.edge.target, item.line_id)
        == ("__junction_27", "s3__entry_left_16", "l0")
    )
    assert not any(
        isinstance(reservation.region, CanvasRegion)
        for reservation in plan.reservations
        for claim in reservation.claims
        if claim.member_id == member.id
    )


def test_synthetic_junction_exterior_canvas_run_is_retained() -> None:
    path = TOPOLOGIES / "exit_lane_settlement_without_crossings.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    member_by_id = {member.id: member for member in plan.members}
    reservation, claim = next(
        (reservation, claim)
        for reservation in plan.reservations
        if isinstance(reservation.region, CanvasRegion)
        for claim in reservation.claims
        if claim.endpoint_anchor_ids
        and all(
            graph.stations[anchor_id].section_id is None
            for anchor_id in claim.endpoint_anchor_ids
        )
    )
    inner, _blockers = reservations_module.canvas_inner_boundary(graph, reservation)
    member = member_by_id[claim.member_id]

    assert reservation.region.side is CanvasSide.BOTTOM
    assert claim.allocation_coordinate > inner
    assert claim.endpoint_anchor_ids == ("__junction_6",)
    assert (
        member.edge.source,
        member.edge.target,
        member.line_id,
    ) == ("__junction_6", "drop_target__entry_bottom_3", "drop")


@pytest.mark.parametrize(
    ("orientation", "start", "end", "before", "after", "expected"),
    (
        (
            reservations_module.CorridorOrientation.HORIZONTAL,
            (10.0, 30.0),
            (90.0, 30.0),
            (10.0, 10.0),
            (90.0, 20.0),
            RowGapRegion(7, 8),
        ),
        (
            reservations_module.CorridorOrientation.HORIZONTAL,
            (10.0, 30.0),
            (90.0, 30.0),
            (10.0, 50.0),
            (90.0, 40.0),
            RowGapRegion(2, 3),
        ),
        (
            reservations_module.CorridorOrientation.VERTICAL,
            (30.0, 10.0),
            (30.0, 90.0),
            (10.0, 10.0),
            (20.0, 90.0),
            reservations_module.ColumnGapRegion(5, 6),
        ),
        (
            reservations_module.CorridorOrientation.VERTICAL,
            (30.0, 10.0),
            (30.0, 90.0),
            (50.0, 10.0),
            (40.0, 90.0),
            reservations_module.ColumnGapRegion(1, 2),
        ),
    ),
)
def test_outer_turn_region_is_axis_generic_and_side_symmetric(
    orientation,
    start,
    end,
    before,
    after,
    expected,
) -> None:
    segment = reservations_module._AxisSegment(
        1,
        1,
        orientation,
        (
            reservations_module.Direction.R
            if orientation is reservations_module.CorridorOrientation.HORIZONTAL
            else reservations_module.Direction.D
        ),
        start,
        end,
        before,
        after,
    )
    span = reservations_module.GridSpan(2, 5, 3, 7)

    assert reservations_module._outer_turn_region(segment, (segment,), span) == expected


def test_target_side_outer_return_keeps_its_exterior_canvas_region() -> None:
    path = ROOT / "tests" / "fixtures" / "regressions" / "entry_trunk_row_bow.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    member_ids = {
        member.id
        for member in plan.members
        if member.edge.source == "src2__exit_right_1"
        and member.edge.target == "tgt__entry_left_2"
        and member.line_id in {"l1", "l2"}
    }
    regions_by_segment = {
        claim.segment_rank: reservation.region
        for reservation in plan.reservations
        for claim in reservation.claims
        if claim.member_id in member_ids
    }

    assert regions_by_segment[1] == CanvasRegion(CanvasSide.RIGHT)
    assert regions_by_segment[2] == RowGapRegion(0, 1)
    assert regions_by_segment[3] == CanvasRegion(CanvasSide.LEFT)


@pytest.mark.parametrize(
    ("points", "expected_straight_run", "corridor"),
    (
        (
            [(-100.0, 0.0), (0.0, 0.0), (0.0, 24.2), (100.0, 24.2)],
            False,
            (RowGapRegion(1, 2), CorridorMeasurementScope.TOPOLOGY_SPAN),
        ),
        (
            [(-100.0, 0.0), (0.0, 0.0), (0.0, 25.2), (100.0, 25.2)],
            True,
            (RowGapRegion(1, 2), CorridorMeasurementScope.TOPOLOGY_SPAN),
        ),
        (
            [(0.0, -100.0), (0.0, 0.0), (24.2, 0.0), (24.2, 100.0)],
            False,
            (
                reservations_module.ColumnGapRegion(1, 2),
                CorridorMeasurementScope.TOPOLOGY_SPAN,
            ),
        ),
    ),
)
def test_symbolic_lane_transition_is_stable_across_tangent_threshold(
    points,
    expected_straight_run,
    corridor,
) -> None:
    previous, transition, following = reservations_module._maximal_axis_segments(points)

    assert (
        reservations_module.axis_segment_has_straight_run(
            points, [15.0, 10.0], transition.rank, transition.end_rank + 1
        )
        is expected_straight_run
    )
    assert reservations_module._segment_is_allocatable_lane_transition(
        transition,
        previous,
        following,
        corridor,
        corridor,
    )


def test_symbolic_lane_transition_requires_one_proven_corridor() -> None:
    points = [(-100.0, 0.0), (0.0, 0.0), (0.0, 24.2), (100.0, 24.2)]
    previous, transition, following = reservations_module._maximal_axis_segments(points)

    assert not reservations_module._segment_is_allocatable_lane_transition(
        transition,
        previous,
        following,
        (RowGapRegion(1, 2), CorridorMeasurementScope.TOPOLOGY_SPAN),
        (RowGapRegion(2, 3), CorridorMeasurementScope.TOPOLOGY_SPAN),
    )


@pytest.mark.parametrize(
    ("points", "endpoint_index"),
    (
        ([(-100.0, 0.0), (0.0, 0.0), (0.0, 10.0)], 1),
        ([(0.0, -100.0), (0.0, 0.0), (10.0, 0.0)], 1),
        ([(0.0, 10.0), (0.0, 0.0), (100.0, 0.0)], 0),
        ([(10.0, 0.0), (0.0, 0.0), (0.0, -100.0)], 0),
    ),
)
def test_symbolic_endpoint_transition_is_stable_under_axis_allocation(
    points,
    endpoint_index,
) -> None:
    segments = reservations_module._maximal_axis_segments(points)
    endpoint = segments[endpoint_index]
    previous = segments[endpoint_index - 1] if endpoint_index else None
    following = segments[endpoint_index + 1] if endpoint_index == 0 else None

    assert not reservations_module.axis_segment_has_straight_run(
        points, [10.0], endpoint.rank, endpoint.end_rank + 1
    )
    assert reservations_module._segment_is_allocatable_endpoint_transition(
        endpoint,
        previous,
        following,
        previous is not None,
        following is not None,
        ("source" if endpoint_index == 0 else "target",),
    )


def test_symbolic_endpoint_transition_requires_exact_adjacent_ownership() -> None:
    points = [(-100.0, 0.0), (0.0, 0.0), (0.0, 10.0)]
    previous, endpoint = reservations_module._maximal_axis_segments(points)

    assert not reservations_module._segment_is_allocatable_endpoint_transition(
        endpoint,
        previous,
        None,
        False,
        False,
        ("target",),
    )
    assert not reservations_module._segment_is_allocatable_endpoint_transition(
        endpoint,
        previous,
        None,
        True,
        False,
        (),
    )
    assert not reservations_module._segment_is_allocatable_endpoint_transition(
        replace(endpoint, orientation=previous.orientation),
        previous,
        None,
        True,
        False,
        ("target",),
    )


def test_packed_cell_lane_transition_has_immutable_gap_ownership() -> None:
    path = TOPOLOGIES / "packed_cell_right_exit_left_entry_wrap.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    members = {member.id: member for member in plan.members}
    owner = next(
        reservation
        for reservation in plan.reservations
        if reservation.region == reservations_module.ColumnGapRegion(1, 2)
        and any(
            claim.segment_rank == 3
            and (
                members[claim.member_id].edge.source,
                members[claim.member_id].edge.target,
                members[claim.member_id].line_id,
            )
            == ("__junction_10", "annot__entry_left_9", "reference")
            for claim in reservation.claims
        )
    )

    settlement = settle_route_envelopes(graph, plan)
    settled_plan = realise_route_reservations(plan, graph)

    assert settlement.capacity_limitations == ()
    assert any(reservation.id == owner.id for reservation in settled_plan.reservations)


def test_synthetic_endpoint_requires_complete_axis_continuity() -> None:
    graph = prepare_graph(SEED_41.read_text(), source_dir=str(SEED_41.parent))
    endpoint_anchor_ids = (min(graph.junction_ids),)
    same_axis = frozenset(
        (
            EmissionMemberId("member:a"),
            EmissionMemberId("member:b"),
        )
    )

    assert not settlement_module._endpoint_anchor_requires_fixed(
        graph, endpoint_anchor_ids, (same_axis,), set(same_axis)
    )
    assert settlement_module._endpoint_anchor_requires_fixed(
        graph,
        endpoint_anchor_ids,
        (same_axis,),
        {next(iter(same_axis))},
    )
    assert settlement_module._endpoint_anchor_requires_fixed(
        graph,
        (next(item for item in graph.stations if item not in graph.junction_ids),),
        (same_axis,),
        set(same_axis),
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

    proof = next(
        item
        for item in settled.capacity_proofs
        if reservation.id in item.id.reservation_ids
    )
    assert proof.available_width >= proof.required_width
    assert all(
        proof.region_start <= claim.coordinate <= proof.region_end
        for allocation in proof.reservations
        for claim in allocation.allocations
    )

    geometry_once = {
        station.id: (station.x, station.y) for station in graph.stations.values()
    }
    after_plan = realise_route_reservations(before.plan, graph)
    settled_again = settle_route_envelopes(graph, after_plan)
    assert settled_again.translations == ()
    assert {
        station.id: (station.x, station.y) for station in graph.stations.values()
    } == geometry_once


def test_cross_column_translation_preserves_exact_row_blocker_ownership() -> None:
    graph = prepare_graph(CROSS_COLUMN.read_text(), source_dir=str(CROSS_COLUMN.parent))
    observed = _observe(graph)
    reservation = next(
        item
        for item in observed.plan.reservations
        if item.region == RowGapRegion(0, 1)
        and item.measurement_scope is CorridorMeasurementScope.OBSERVED_RUN
        and item.span.min_row == 1
        and item.span.max_row == 3
    )
    realised_before = next(
        item
        for item in observed.plan.realised_reservations
        if item.reservation_id == reservation.id
    )
    ledger_before = observed.plan.reservations
    snapshot = settlement_module._snapshot(graph)
    local_before = _local_geometry(graph)

    owners = tuple(item.id for item in graph.sections.values() if item.grid_col >= 1)
    blocker = graph.sections["preprocessing"]
    target = graph.sections["post_vc"]
    amount = blocker.bbox_x + blocker.bbox_w - target.bbox_x + graph_offset_step(graph)
    settlement_module._translate_sections(
        graph,
        observed.plan,
        owners,
        dx=amount,
        dy=0.0,
    )
    projected = settlement_module._project_ledger_translations(
        graph, observed.plan, snapshot
    )
    projected_reservation = next(
        item for item in projected.reservations if item.id == reservation.id
    )
    realised_after = next(
        item
        for item in realise_route_reservations(
            projected, graph, blocker_plan=observed.plan
        ).realised_reservations
        if item.reservation_id == reservation.id
    )

    assert amount > 0
    assert min(item.longitudinal_start for item in projected_reservation.claims) == (
        pytest.approx(graph.stations["variant_calling__exit_right_1"].x)
    )
    assert max(item.longitudinal_end for item in projected_reservation.claims) == (
        pytest.approx(graph.stations["post_vc__entry_left_5"].x)
    )
    assert realised_after.negative_blocker_ids == realised_before.negative_blocker_ids
    assert realised_after.positive_blocker_ids == realised_before.positive_blocker_ids
    assert realised_after.region_start == pytest.approx(realised_before.region_start)
    assert realised_after.region_end > realised_after.region_start
    assert _local_geometry(graph) == local_before
    assert observed.plan.reservations is ledger_before
    assert reservation in ledger_before


def test_claim_packer_applies_every_pairwise_constraint_to_its_second_claim() -> None:
    graph = prepare_graph(SEED_41.read_text(), source_dir=str(SEED_41.parent))
    plan = _observe(graph).plan
    component = max(
        (
            component
            for items in settlement_module._boundary_reservations(
                plan, graph, plan
            ).values()
            for component in settlement_module._boundary_components(items)
        ),
        key=lambda items: sum(len(item.reservation.claims) for item in items),
    )
    component = tuple(
        replace(
            item,
            realised=replace(
                item.realised,
                region_start=-10_000.0,
                region_end=10_000.0,
                available_width=20_000.0,
            ),
        )
        for item in component
    )

    packed = settlement_module._pack_component_claims(graph, component)
    assert isinstance(packed, tuple)
    nodes = settlement_module._claim_nodes(component)
    checked_pairs = 0
    for first_rank, first in enumerate(nodes):
        for second in nodes[first_rank + 1 :]:
            first_item = component[first.reservation_rank]
            second_item = component[second.reservation_rank]
            if settlement_module._claim_nodes_share_channel(
                component, first, second
            ) or not settlement_module._claim_interval_overlaps(
                first_item,
                first.claim_rank,
                second_item,
                second.claim_rank,
            ):
                continue
            if first.reservation_rank == second.reservation_rank:
                separation = max(
                    abs(second.immutable_coordinate - first.immutable_coordinate),
                    graph_offset_step(graph),
                )
            else:
                separation = max(
                    first_item.reservation.peer_clearance,
                    second_item.reservation.peer_clearance,
                )
            first_coordinate = packed[first.reservation_rank][first.claim_rank]
            second_coordinate = packed[second.reservation_rank][second.claim_rank]
            assert (
                abs(second_coordinate - first_coordinate) + COORD_TOLERANCE
                >= separation
            )
            checked_pairs += 1
    assert checked_pairs > 1


def test_merge_opening_siblings_publish_one_immutable_physical_lane() -> None:
    path = TOPOLOGIES / "exit_run_three_drop_columns.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    members = {item.id: item for item in plan.members}
    reservation, lane = next(
        (reservation, lane)
        for reservation in plan.reservations
        for lane in reservation.lanes
        if len(lane.claim_indices) > 1
        and len(
            {
                (
                    members[reservation.claims[claim_rank].member_id].target.station_id,
                    members[reservation.claims[claim_rank].member_id].line_id,
                )
                for claim_rank in lane.claim_indices
            }
        )
        == 1
    )
    claims = tuple(reservation.claims[rank] for rank in lane.claim_indices)

    settlement = settle_route_envelopes(graph, plan)

    assert len(claims) == 2
    assert len({claim.allocation_coordinate for claim in claims}) == 1
    assert all(members[claim.member_id].convergence_ids for claim in claims)
    assert reservation.system_id not in {
        item.system_id for item in settlement.capacity_limitations
    }
    proof = next(
        proof
        for proof in settlement.capacity_proofs
        if reservation.id in proof.id.reservation_ids
    )
    allocation = next(
        item for item in proof.reservations if item.reservation_id == reservation.id
    )
    assert any(item.claim_indices == lane.claim_indices for item in allocation.lanes)


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


def test_organellar_outer_gap_deficit_settles_without_limitation() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    settlement = settle_route_envelopes(graph, plan)

    assert [
        (
            diagnostic.capacity_slack,
            diagnostic.negative_side_slack,
            diagnostic.positive_side_slack,
        )
        for diagnostic in plan.reservation_diagnostics
    ] == [
        (44.0, -44.0, 88.0),
        (-3.0, -48.0, 4.0),
    ]
    assert {
        (translation.axis, translation.boundary, translation.amount)
        for translation in settlement.translations
    } == {
        (EnvelopeAxis.X, (2, 3), 4.0),
        (EnvelopeAxis.Y, (3, 4), 4.0),
    }
    assert settlement.capacity_limitations == ()
    assert {
        reservation.reservation_id
        for proof in settlement.capacity_proofs
        for reservation in proof.reservations
    } == {reservation.id for reservation in plan.reservations}


def test_strict_organellar_settlement_has_complete_capacity_evidence() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    graph.strict = True

    settlement = settle_route_envelopes(graph, plan)
    settled_plan = realise_route_reservations(plan, graph)

    assert settlement.capacity_limitations == ()
    assert_route_envelopes_satisfied(graph, settled_plan)


def test_genome_external_convergence_anchor_owns_minimal_suffix_growth() -> None:
    path = ROOT / "examples" / "genomeassembly.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    snapshot = settlement_module._snapshot(graph)
    external_anchor_ids = settlement_module._external_route_anchor_ids(graph, plan)
    anchor = graph.stations["__merge_5"]
    fixed_axis = next(
        axis
        for exit_plan in plan.exit_turn_plans
        for axis in exit_plan.axes
        if axis.fixed_anchor_id == anchor.id
    )
    local_before = {
        station.id: (
            station.x - graph.sections[station.section_id].bbox_x,
            station.y - graph.sections[station.section_id].bbox_y,
        )
        for station in graph.stations.values()
        if station.section_id in graph.sections
        and station.id not in external_anchor_ids
    }
    anchor_before = (anchor.x, anchor.y)

    settlement = settle_route_envelopes(graph, plan)
    projected = settlement_module._project_ledger_translations(graph, plan, snapshot)
    measured = realise_route_reservations(projected, graph, blocker_plan=plan)
    boundary = settlement_module._AxisBoundary(EnvelopeAxis.X, 2, 3)
    component = settlement_module._boundary_components(
        settlement_module._boundary_reservations(measured, graph, plan)[boundary]
    )[0]
    packed = settlement_module._pack_component_claims(graph, component)

    assert settlement.translations == (
        settlement_module.EnvelopeTranslation(
            EnvelopeAxis.X,
            (2, 3),
            4.0,
            ("genome_statistics", "scaffolding"),
            tuple(sorted(str(item.reservation.id) for item in component)),
        ),
    )
    assert settlement.capacity_limitations == ()
    assert anchor.id in external_anchor_ids
    assert (anchor.x, anchor.y) == anchor_before
    assert fixed_axis.coordinate == anchor.x
    assert packed == ((770.0,), (786.0, 782.0))
    assert {
        station.id: (
            station.x - graph.sections[station.section_id].bbox_x,
            station.y - graph.sections[station.section_id].bbox_y,
        )
        for station in graph.stations.values()
        if station.id in local_before
    } == local_before

    second = settle_route_envelopes(graph, projected)

    assert second.translations == ()
    assert second.capacity_limitations == ()
    assert (anchor.x, anchor.y) == anchor_before
    assert fixed_axis.coordinate == anchor.x


def test_external_convergence_anchor_translation_ownership_transposes() -> None:
    path = ROOT / "examples" / "genomeassembly.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    anchor = graph.stations["__merge_5"]
    section = graph.sections[anchor.section_id]
    anchor_before = (anchor.x, anchor.y)
    anchor.x = section.bbox_x + section.bbox_w / 2
    anchor.y = section.bbox_y + section.bbox_h / 2
    assert anchor.id not in settlement_module._external_route_anchor_ids(graph, plan)
    anchor.x, anchor.y = anchor_before
    real_station = graph.stations["asmstats"]
    port = next(
        item for item in graph.ports.values() if item.section_id == "genome_statistics"
    )
    for station in (real_station, port):
        station_before = (station.x, station.y)
        station.x = section.bbox_x - 100.0
        station.y = section.bbox_y - 100.0
        assert station.id not in settlement_module._external_route_anchor_ids(
            graph, plan
        )
        station.x, station.y = station_before

    for section in graph.sections.values():
        section.bbox_x, section.bbox_y = section.bbox_y, section.bbox_x
        section.bbox_w, section.bbox_h = section.bbox_h, section.bbox_w
        section.grid_col, section.grid_row = section.grid_row, section.grid_col
        section.grid_col_span, section.grid_row_span = (
            section.grid_row_span,
            section.grid_col_span,
        )
    for station in graph.stations.values():
        station.x, station.y = station.y, station.x
    for station_id, port in graph.ports.items():
        if graph.stations.get(station_id) is not port:
            port.x, port.y = port.y, port.x

    anchor_before = (anchor.x, anchor.y)
    assert anchor.id in settlement_module._external_route_anchor_ids(graph, plan)
    boundary = settlement_module._AxisBoundary(EnvelopeAxis.Y, 2, 3)
    assert not settlement_module._external_anchor_is_positive(
        graph, anchor.id, boundary
    )
    owners = tuple(
        section.id for section in graph.sections.values() if section.grid_row >= 3
    )

    settlement_module._translate_sections(
        graph,
        plan,
        owners,
        dx=0.0,
        dy=4.0,
    )

    assert (anchor.x, anchor.y) == anchor_before

    earlier = settlement_module._AxisBoundary(EnvelopeAxis.Y, 1, 2)
    assert settlement_module._external_anchor_is_positive(graph, anchor.id, earlier)
    earlier_owners = tuple(
        section.id for section in graph.sections.values() if section.grid_row >= 2
    )
    settlement_module._translate_sections(
        graph,
        plan,
        earlier_owners,
        dx=0.0,
        dy=4.0,
    )

    assert (anchor.x, anchor.y) == (anchor_before[0], anchor_before[1] + 4.0)


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


def test_fixed_exit_anchor_remains_on_its_structural_coordinate() -> None:
    path = TOPOLOGIES / "rail_boundary_bundle_fan.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)
    fixed_member_ids = {
        assignment.member_id
        for plan in observed.plan.exit_turn_plans
        for assignment in plan.assignments
        if assignment.axis_id is not None
    }
    settlement = settle_route_envelopes(graph, observed.plan)

    assert settlement.translations == ()
    proof = next(
        item
        for item in settlement.capacity_proofs
        if item.axis is DemandAxis.X and item.boundary == (0, 1)
    )
    fixed = next(
        item
        for item in proof.reservations
        if fixed_member_ids.intersection(item.claimant_member_ids)
    )
    assert fixed.coordinate == fixed.original_coordinate
    assert proof.available_width >= proof.required_width


def test_inherited_boundary_translation_projects_the_corridor_frame() -> None:
    path = TOPOLOGIES / "target_lane_transition.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)
    snapshot = settlement_module._snapshot(graph)
    boundary = settlement_module._AxisBoundary(EnvelopeAxis.X, 0, 1)
    owners = tuple(
        section.id
        for section in graph.sections.values()
        if boundary.starts_after(section)
    )

    settlement_module._translate_sections(
        graph,
        observed.plan,
        owners,
        dx=8.0,
        dy=0.0,
    )
    projected = settlement_module._project_ledger_translations(
        graph, observed.plan, snapshot
    )
    original_claim = next(
        claim
        for reservation in observed.plan.reservations
        for claim in reservation.claims
        if claim.allocation_coordinate == pytest.approx(412.0)
    )
    projected_claim = next(
        claim
        for reservation in projected.reservations
        for claim in reservation.claims
        if claim.member_id == original_claim.member_id
        and claim.segment_rank == original_claim.segment_rank
    )
    assert projected_claim.allocation_coordinate == pytest.approx(420.0)


def test_joint_allocation_preserves_each_reservations_own_corridor() -> None:
    path = TOPOLOGIES / "complex_multipath.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)
    snapshot = settlement_module._snapshot(graph)

    settlement = settle_route_envelopes(graph, observed.plan)
    projected = settlement_module._project_ledger_translations(
        graph, observed.plan, snapshot
    )
    realised = realise_route_reservations(projected, graph)
    boundary = settlement_module._AxisBoundary(EnvelopeAxis.X, 1, 2)
    component = next(
        item
        for item in settlement_module._boundary_components(
            settlement_module._boundary_reservations(realised, graph)[boundary]
        )
        if len(item) > 1
    )
    proof = next(
        item
        for item in settlement.capacity_proofs
        if item.axis is DemandAxis.X
        and item.boundary == (1, 2)
        and set(item.id.reservation_ids)
        == {current.reservation.id for current in component}
    )
    coordinate_by_id = {
        item.reservation_id: item.coordinate for item in proof.reservations
    }

    assert proof.region_start == min(item.realised.region_start for item in component)
    assert proof.region_end == max(item.realised.region_end for item in component)
    for item in component:
        coordinate = coordinate_by_id[item.reservation.id]
        assert (
            coordinate
            >= item.realised.region_start + item.negative_extent - COORD_TOLERANCE
        )
        assert (
            coordinate
            <= item.realised.region_end - item.positive_extent + COORD_TOLERANCE
        )


@pytest.mark.parametrize(
    "path",
    (
        TOPOLOGIES / "exit_lane_settlement_without_crossings.mmd",
        TOPOLOGIES / "peeloff_straight_drop_near_wall.mmd",
    ),
)
def test_fixed_local_endpoint_corridors_exit_compatibility(
    path: Path,
) -> None:
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)

    settlement = settle_route_envelopes(graph, observed.plan)

    assert settlement.capacity_limitations == ()
    proved_ids = {
        reservation.reservation_id
        for proof in settlement.capacity_proofs
        for reservation in proof.reservations
    }
    assert proved_ids == {reservation.id for reservation in observed.plan.reservations}


def test_seed_77_fixed_corridors_exit_compatibility_after_settlement() -> None:
    path = ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_77.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)

    settlement = settle_route_envelopes(graph, observed.plan)

    assert settlement.translations
    assert settlement.capacity_limitations == ()
    assert settlement.identity_projections == ()
    assert {
        reservation.reservation_id
        for proof in settlement.capacity_proofs
        for reservation in proof.reservations
    } == {reservation.id for reservation in observed.plan.reservations}


def test_opposing_resource_does_not_inherit_ordered_exit_axis_clearance() -> None:
    path = ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_77.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = _observe(graph).plan
    snapshot = settlement_module._snapshot(graph)
    settle_route_envelopes(graph, plan)
    projected = settlement_module._project_ledger_translations(graph, plan, snapshot)
    measured = realise_route_reservations(projected, graph, blocker_plan=plan)
    boundary = settlement_module._AxisBoundary(EnvelopeAxis.X, 8, 9)
    component = settlement_module._boundary_components(
        settlement_module._boundary_reservations(measured, graph, plan)[boundary]
    )[0]
    first = next(
        item
        for item in component
        if item.fixed
        and sum(identity is not None for identity in item.exit_turn_axis_identities)
        >= 2
    )

    assert settlement_module._sibling_exit_axis_separation(
        first,
        0,
        first,
        1,
    ) == pytest.approx(graph_offset_step(graph))
    opposing = replace(
        first,
        reservation=replace(
            first.reservation,
            direction=reservations_module.Direction.U,
        ),
    )
    assert (
        settlement_module._sibling_exit_axis_separation(first, 0, opposing, 1) is None
    )
    assert max(
        first.reservation.peer_clearance,
        opposing.reservation.peer_clearance,
    ) == pytest.approx(3 * graph_offset_step(graph))


def test_strict_fixed_local_endpoint_corridor_settles_without_mutation() -> None:
    path = TOPOLOGIES / "exit_lane_settlement_without_crossings.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(graph)
    geometry_before = {item.id: (item.x, item.y) for item in graph.stations.values()}
    graph.strict = True

    settlement = settle_route_envelopes(graph, observed.plan)

    assert settlement.capacity_limitations == ()
    assert {
        item.id: (item.x, item.y) for item in graph.stations.values()
    } == geometry_before


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


def test_final_allocation_emission_preserves_frozen_local_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = TOPOLOGIES / "bypass_fan_in_outer_slot.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    settlement_started = False
    real_settle = settlement_module.settle_route_envelopes
    real_apply_moves = routing_core._settle_station_moves

    def observe_settlement(graph, plan):
        nonlocal settlement_started
        settlement_started = True
        return real_settle(graph, plan)

    def reject_late_moves(graph, moves):
        assert not settlement_started
        real_apply_moves(graph, moves)

    monkeypatch.setattr(settlement_module, "settle_route_envelopes", observe_settlement)
    monkeypatch.setattr(routing_core, "_settle_station_moves", reject_late_moves)

    build_observed_render_plan(graph, resolve_theme(None, graph))


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
    build_route_plan_query(route_plan)

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
        for reservation in proof.reservations:
            for allocation in reservation.allocations:
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


def test_canvas_trunk_clears_opposing_row_gap_trunk() -> None:
    path = TOPOLOGIES / "merge_around_below_leftmost.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))

    render_plan, _route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )

    def longest_horizontal_y(source: str, target: str) -> float:
        route = next(
            item
            for item in render_plan.routes
            if item.edge.source == source and item.edge.target == target
        )
        horizontal = tuple(
            (abs(x1 - x0), y0)
            for (x0, y0), (x1, y1) in zip(route.points, route.points[1:])
            if abs(y1 - y0) <= COORD_TOLERANCE
        )
        return max(horizontal)[1]

    canvas_y = longest_horizontal_y("__junction_4", "__merge_2")
    row_gap_y = longest_horizontal_y("__junction_5", "__merge_3")

    assert abs(canvas_y - row_gap_y) >= (BUNDLE_TO_BUNDLE_CLEARANCE - COORD_TOLERANCE)


def test_genomeassembly_canvas_bundle_keeps_physical_lane_pitch() -> None:
    path = ROOT / "examples" / "genomeassembly.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))

    render_plan, _route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )
    routes = {
        (route.edge.source, route.edge.target, route.line_id): route
        for route in render_plan.routes
    }

    def longest_horizontal_y(route) -> float:
        return max(
            (
                (abs(x1 - x0), y0)
                for (x0, y0), (x1, y1) in zip(route.points, route.points[1:])
                if abs(y1 - y0) <= COORD_TOLERANCE
            )
        )[1]

    assemblies_y = longest_horizontal_y(
        routes[("__junction_8", "__merge_3", "assemblies")]
    )
    hic_reads_y = longest_horizontal_y(
        routes[("__junction_8", "scaffolding__entry_left_5", "hic_reads")]
    )

    assert abs(assemblies_y - hic_reads_y) == pytest.approx(graph_offset_step(graph))


def test_organellar_compatibility_exits_with_complete_capacity_evidence() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    measured_graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = _observe(measured_graph)
    settlement = settle_route_envelopes(measured_graph, observed.plan)
    members = {member.id: member for member in observed.plan.members}
    outer_owner = next(
        reservation
        for reservation in observed.plan.reservations
        if reservation.region == RowGapRegion(3, 4)
        and any(
            (
                members[claim.member_id].edge.source,
                members[claim.member_id].edge.target,
                members[claim.member_id].line_id,
            )
            == ("__junction_9", "__merge_5", "assemblies")
            for claim in reservation.claims
        )
    )
    assert reservations_module._adjacent_outer_turn_region_is_proven(
        outer_owner, members
    )
    assert not reservations_module._adjacent_outer_turn_region_is_proven(
        replace(
            outer_owner,
            orientation=reservations_module.CorridorOrientation.VERTICAL,
            region=reservations_module.ColumnGapRegion(3, 4),
            kind=reservations_module.CorridorKind.INTER_COLUMN_CHANNEL,
            direction=reservations_module.Direction.D,
        ),
        members,
    )
    assert not reservations_module._adjacent_outer_turn_region_is_proven(
        replace(outer_owner, region=RowGapRegion(4, 5)), members
    )
    assert settlement.capacity_limitations == ()
    assert {
        reservation.reservation_id
        for proof in settlement.capacity_proofs
        for reservation in proof.reservations
    } == {reservation.id for reservation in observed.plan.reservations}
    outer_proof = next(
        proof
        for proof in settlement.capacity_proofs
        if outer_owner.id in proof.id.reservation_ids
    )
    assert outer_proof.boundary == (3, 4)
    assert outer_proof.available_width >= outer_proof.required_width

    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    _render_plan, route_plan = build_observed_render_plan(
        graph, resolve_theme(None, graph)
    )

    assert route_plan.convergence_plans
    assert all(item.owns_geometry for item in route_plan.convergence_plans)
    assert any(item.id == outer_owner.id for item in route_plan.reservations)
    assert not any(
        item.code == "convergence-plan-legacy" for item in route_plan.diagnostics
    )
