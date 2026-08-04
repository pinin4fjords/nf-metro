"""Immutable convergence plans own merge trunks before route emission."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import nf_metro.layout.routing.convergences as convergence_routing
import nf_metro.layout.routing.core as routing_core
from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.envelope_settlement import settle_route_envelopes
from nf_metro.layout.geometry import point_to_polyline_distance
from nf_metro.layout.phases._common import (
    routes_through_own_section_interior,
    routes_through_unrelated_sections,
)
from nf_metro.layout.route_plan import (
    BindingKind,
    ConvergenceDisposition,
    ConvergenceEndpointRole,
    ConvergencePlanId,
    ConvergenceTrunkAxis,
    ConvergenceTrunkReason,
    CoordinateRegime,
    DemandAxis,
    DemandKind,
    KeepOutClass,
    RouteFamilyId,
    RoutePlan,
    RouteSystemId,
    SharedReferenceId,
    SharedReferenceKind,
    build_route_plan_query,
    convergence_resource_ids,
)
from nf_metro.layout.route_reservations import (
    expected_convergence_foreign_references,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.layout.routing.common import Direction, OffsetRegime, RoutedPath
from nf_metro.layout.routing.convergences import (
    ConvergenceInvariantError,
    ConvergencePlanningError,
    UnsupportedConvergenceError,
    _direct_axis,
    _extend_axis_segment_to_coordinates,
    _seat_route_on_trunk_flanks,
    validate_convergence_plans,
)
from nf_metro.layout.routing.corners import concentric_corner_radius_at
from nf_metro.layout.routing.inter_section_handlers import (
    _merge_entry_cross_axis_order,
)
from nf_metro.layout.routing.invariants import (
    check_merge_branches_meet_trunk,
    check_merge_feeders_land_on_trunk,
)
from nf_metro.parser.model import Edge, MetroGraph, PortSide, Station
from nf_metro.parser.route_topology import build_route_topology_query
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]
TOPOLOGIES = ROOT / "examples" / "topologies"
GUIDE = ROOT / "examples" / "guide"
FROZEN = ROOT / "tests" / "fixtures" / "hash_seed_determinism"


def _observe(path: Path):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    offsets = compute_station_offsets(graph)
    observed = observe_route_edges(graph, station_offsets=offsets)
    return graph, offsets, observed


def _observe_text(text: str):
    graph = prepare_graph(text)
    offsets = compute_station_offsets(graph)
    observed = observe_route_edges(graph, station_offsets=offsets)
    return graph, offsets, observed


def _observe_after_settlement(path: Path, *, offset_step: float | None = None):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    offsets = compute_station_offsets(graph, offset_step=offset_step)
    preflight = observe_route_edges(
        graph,
        station_offsets=offsets,
        offset_step=offset_step,
    )
    settlement = settle_route_envelopes(graph, preflight.plan)
    offsets = compute_station_offsets(graph, offset_step=offset_step)
    final = observe_route_edges(
        graph,
        station_offsets=offsets,
        offset_step=offset_step,
        envelope_proofs=settlement.capacity_proofs,
        envelope_limitations=settlement.capacity_limitations,
        envelope_reservations=preflight.plan.reservations,
        envelope_bindings=preflight.plan.bindings,
        envelope_identity_projections=settlement.identity_projections,
    )
    return graph, offsets, preflight, final


@pytest.fixture(scope="module")
def three_column_route_plan() -> RoutePlan:
    return _observe(TOPOLOGIES / "merge_feeders_three_columns.mmd")[2].plan


@pytest.fixture(scope="module")
def right_entry_route_plan() -> RoutePlan:
    return _observe(TOPOLOGIES / "merge_right_entry.mmd")[2].plan


def test_three_column_merge_has_one_complete_planned_convergence() -> None:
    graph, _offsets, observed = _observe(TOPOLOGIES / "merge_feeders_three_columns.mmd")
    (plan,) = observed.plan.convergence_plans

    assert plan.disposition is ConvergenceDisposition.PLANNED
    assert plan.primary_trunk_reason is ConvergenceTrunkReason.LONGEST_BYPASS
    assert plan.primary_trunk_member_id in plan.member_ids
    assert plan.trunk_axis is not None
    assert plan.trunk_axis.axis is DemandAxis.X
    assert plan.trunk_axis.extent_start < plan.trunk_axis.extent_end
    topology = build_route_topology_query(graph)
    assert topology is not None
    assert plan.merge_junction_ids == tuple(
        item.junction_id for item in topology.convergences
    )
    assert len(plan.landings) == 3
    assert tuple(item.order for item in plan.landings) == (0, 1, 2)
    expected_lane_rank = plan.lane_order.index(plan.line_ids[0])
    assert tuple(item.lane_rank for item in plan.landings) == (
        expected_lane_rank,
        expected_lane_rank,
        expected_lane_rank,
    )
    assert len(plan.lane_order) > 1
    assert any(item.bypass for item in plan.landings)
    assert any(item.long_haul for item in plan.landings)
    assert plan.outgoing_continuations
    assert set(plan.member_ids) == {
        ownership.member_id for ownership in plan.endpoint_ownership
    }
    assert {ownership.role for ownership in plan.endpoint_ownership} >= {
        ConvergenceEndpointRole.FEEDER,
        ConvergenceEndpointRole.COVERED_CONTINUATION,
    }


@pytest.mark.parametrize(
    ("fixture", "reason"),
    (
        (
            "exit_run_three_drop_columns.mmd",
            "planned convergence trunks require one shared channel decision",
        ),
        (
            "funcprofiler_upstream.mmd",
            "planned convergence corridor depends on unresolved overlapping fan "
            "ownership (owner #1658)",
        ),
        (
            "merge_around_below_leftmost.mmd",
            "planned convergence trunks require one shared channel decision",
        ),
    ),
)
def test_conflicting_route_systems_use_whole_system_compatibility(
    fixture: str, reason: str
) -> None:
    _graph, _offsets, observed = _observe(TOPOLOGIES / fixture)

    assert observed.plan.convergence_plans
    assert {item.disposition for item in observed.plan.convergence_plans} == {
        ConvergenceDisposition.LEGACY
    }
    assert len({item.system_id for item in observed.plan.convergence_plans}) == 1
    assert {item.legacy_reason for item in observed.plan.convergence_plans} == {reason}


@pytest.mark.parametrize(
    ("path", "reason"),
    (
        (
            TOPOLOGIES / "merge_bottom_row_bypass.mmd",
            "planned fan arms require opposing opening channels",
        ),
        (
            ROOT / "examples" / "genomeassembly.mmd",
            "chained same-line convergences require one shared system settlement",
        ),
        (
            ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
            "chained same-line convergences require one shared system settlement",
        ),
        (
            ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
            "planned convergence feeder approaches require one shared channel decision",
        ),
    ),
)
def test_reviewed_conflicts_keep_the_complete_system_on_compatibility(
    path: Path, reason: str
) -> None:
    _graph, _offsets, observed = _observe(path)

    assert observed.plan.convergence_plans
    assert all(
        item.disposition is ConvergenceDisposition.LEGACY
        for item in observed.plan.convergence_plans
    )
    assert {item.legacy_reason for item in observed.plan.convergence_plans} == {reason}


@pytest.mark.parametrize(
    "path",
    (
        TOPOLOGIES / "merge_adjacent_feeder.mmd",
        TOPOLOGIES / "merge_right_entry.mmd",
        TOPOLOGIES / "merge_trunk_over_low_section.mmd",
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
    ),
)
def test_non_conflicting_reviewed_systems_remain_planned(path: Path) -> None:
    _graph, _offsets, observed = _observe(path)

    assert observed.plan.convergence_plans
    assert all(item.owns_geometry for item in observed.plan.convergence_plans)
    assert all(item.legacy_reason is None for item in observed.plan.convergence_plans)
    assert all(
        set(item.member_ids)
        == {ownership.member_id for ownership in item.endpoint_ownership}
        for item in observed.plan.convergence_plans
    )


def test_shared_terminal_convergence_preserves_unowned_concentric_radii() -> None:
    _graph, _offsets, observed = _observe(ROOT / "examples" / "genomic_pipeline.mmd")
    routes = {
        route.line_id: route
        for route in observed.routes
        if route.edge.source == "__junction_8"
        and route.edge.target in {"__merge_2", "__merge_3", "__merge_4"}
    }

    assert set(routes) == {"germline", "tumor_only", "somatic"}
    assert {line_id: route.curve_radii for line_id, route in routes.items()} == {
        "germline": [18.0, 18.0, 10.0, 10.0],
        "tumor_only": [14.0, 14.0, 14.0, 14.0],
        "somatic": [10.0, 10.0, 18.0, 18.0],
    }
    for route in routes.values():
        assert route.convergence_plan_id is not None
        assert route.convergence_owned_segment_ranks == (4, 1)
        assert not {2, 3}.intersection(route.convergence_owned_segment_ranks)


def test_convergence_plan_is_queryable_through_every_semantic_identity() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    (plan,) = observed.plan.convergence_plans
    query = build_route_plan_query(observed.plan)

    assert query.convergence_plan(plan.id) is plan
    assert query.convergence_plans_for_system(plan.system_id) == (plan,)
    for convergence_id in plan.convergence_ids:
        assert query.convergence_plans_for_convergence(convergence_id) == (plan,)
    for connector_id in plan.connector_ids:
        assert query.convergence_plans_for_connector(connector_id) == (plan,)
    for member_id in plan.member_ids:
        assert query.convergence_plans_for_member(member_id) == (plan,)
    for path in plan.resolved_member_paths:
        assert query.convergence_plans_for_resolved_path(path) == (plan,)


def test_planned_merge_does_not_depend_on_late_feeder_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing_core,
        "_land_merge_feeders_on_trunk",
        lambda _routes, _ctx: None,
    )
    graph, offsets, observed = _observe(TOPOLOGIES / "merge_feeders_three_columns.mmd")

    assert not check_merge_feeders_land_on_trunk(graph, observed.routes, offsets)


def test_planned_coverage_does_not_depend_on_late_hop_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing_core,
        "_drop_covered_merge_entry_hops",
        lambda _routes, _ctx, **_kwargs: (),
    )
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    continuation = plan.outgoing_continuations[0]
    query = build_route_plan_query(observed.plan)

    assert continuation.covered_by_member_id is not None
    assert query.bindings_for(continuation.member_id)[0].kind is BindingKind.MERGE_SKIP


def test_planned_convergence_publishes_trunk_and_landing_resources() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    (plan,) = observed.plan.convergence_plans
    references = {
        reference.id: reference for reference in observed.plan.shared_references
    }
    demands = {demand.id: demand for demand in observed.plan.demands}

    assert tuple(references[item].kind for item in plan.shared_reference_ids) == (
        SharedReferenceKind.TRUNK,
        SharedReferenceKind.LANDING_SEQUENCE,
    )
    assert all(demands[item].system_id == plan.system_id for item in plan.demand_ids)


def test_overlapping_foreign_convergence_axes_are_indexed() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    first = observed.plan.convergence_plans[0]
    second = replace(
        first,
        id=ConvergencePlanId("foreign-convergence"),
        system_id=RouteSystemId("foreign-system"),
        shared_reference_ids=(
            SharedReferenceId("foreign-trunk"),
            SharedReferenceId("foreign-landings"),
        ),
    )
    synthetic = replace(observed.plan, convergence_plans=(first, second))
    conflicts = expected_convergence_foreign_references(synthetic)

    assert conflicts[first.id] == (second.shared_reference_ids[0],)
    assert conflicts[second.id][0] == first.shared_reference_ids[0]
    assert any(
        reference_id.startswith("corridor-reference:")
        for reference_id in conflicts[second.id]
    )


@pytest.mark.parametrize(
    ("source_axis", "coordinate", "expected"),
    (
        (DemandAxis.X, 515.0, True),
        (DemandAxis.X, 1_000.0, False),
        (DemandAxis.Y, 198.0, True),
        (DemandAxis.Y, 1_000.0, False),
    ),
)
def test_foreign_exit_turn_conflicts_use_physical_trunk_geometry(
    source_axis: DemandAxis,
    coordinate: float,
    expected: bool,
) -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    convergence = observed.plan.convergence_plans[0]
    exit_turn = next(item for item in observed.plan.exit_turn_plans if item.axes)
    foreign_reference = SharedReferenceId("foreign-exit-turn")
    foreign_exit = replace(
        exit_turn,
        system_id=RouteSystemId("foreign-exit-system"),
        source_run_direction=(
            Direction.R if source_axis is DemandAxis.X else Direction.D
        ),
        source_axis=source_axis,
        axes=tuple(
            replace(axis, axis=source_axis, coordinate=coordinate)
            for axis in exit_turn.axes
        ),
        reference_id=foreign_reference,
    )
    synthetic = replace(observed.plan, exit_turn_plans=(foreign_exit,))

    conflicts = expected_convergence_foreign_references(synthetic)

    assert (foreign_reference in conflicts[convergence.id]) is expected


def test_convergence_endpoint_ownership_matches_final_bindings() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    (plan,) = observed.plan.convergence_plans
    query = build_route_plan_query(observed.plan)

    for ownership in plan.endpoint_ownership:
        (binding,) = query.bindings_for(ownership.member_id)
        if ownership.role is ConvergenceEndpointRole.COVERED_CONTINUATION:
            assert binding.kind in {
                BindingKind.MERGE_SKIP,
                BindingKind.COVERED_MERGE_HOP,
            }
        else:
            assert binding.kind is BindingKind.EMITTED


def test_every_feeder_join_connects_to_the_target_entry() -> None:
    graph, _offsets, observed = _observe(TOPOLOGIES / "merge_feeders_three_columns.mmd")
    (plan,) = observed.plan.convergence_plans
    routes = [
        route for route in observed.routes if route.convergence_plan_id == str(plan.id)
    ]
    entry_point = plan.outgoing_continuations[0].end_point
    neighbours: dict[int, set[int]] = {rank: set() for rank in range(len(routes))}
    for rank, route in enumerate(routes):
        for other_rank, other in enumerate(routes):
            if rank == other_rank:
                continue
            if any(
                point_to_polyline_distance(point, other.points) <= 1e-6
                for point in (route.points[0], route.points[-1])
            ):
                neighbours[rank].add(other_rank)
    entry_routes = {
        rank
        for rank, route in enumerate(routes)
        if point_to_polyline_distance(entry_point, route.points) <= 1e-6
    }

    assert entry_routes
    for landing in plan.landings:
        start = next(
            rank
            for rank, route in enumerate(routes)
            if route.convergence_member_id == str(landing.member_id)
        )
        reachable = {start}
        pending = [start]
        while pending:
            rank = pending.pop()
            for neighbour in neighbours[rank] - reachable:
                reachable.add(neighbour)
                pending.append(neighbour)
        assert reachable & entry_routes, landing


def test_multiple_lines_share_the_target_entry_bundle_order() -> None:
    _graph, _offsets, _preflight, observed = _observe_after_settlement(
        FROZEN / "seed_15.mmd"
    )
    plans_by_entry: dict[object, list] = {}
    for plan in observed.plan.convergence_plans:
        plans_by_entry.setdefault(plan.entry_group_ids[0], []).append(plan)
    plans = next(
        items
        for items in plans_by_entry.values()
        if len({plan.line_ids[0] for plan in items}) >= 3
    )
    lane_order = plans[0].lane_order

    assert all(plan.lane_order == lane_order for plan in plans)
    assert {plan.line_ids[0] for plan in plans}.issubset(lane_order)
    for plan in plans:
        expected_rank = lane_order.index(plan.line_ids[0])
        assert all(item.lane_rank == expected_rank for item in plan.landings)
        assert all(
            item.lane_rank == expected_rank for item in plan.outgoing_continuations
        )


def test_seed15_chained_convergences_exit_compatibility_after_settlement() -> None:
    graph, _offsets, preflight = _observe(FROZEN / "seed_15.mmd")
    reason = "chained same-line convergences require one shared system settlement"

    assert preflight.plan.convergence_plans
    assert {plan.legacy_reason for plan in preflight.plan.convergence_plans} == {reason}

    settlement = settle_route_envelopes(graph, preflight.plan)
    final = observe_route_edges(
        graph,
        station_offsets=compute_station_offsets(graph),
        envelope_proofs=settlement.capacity_proofs,
        envelope_limitations=settlement.capacity_limitations,
        envelope_reservations=preflight.plan.reservations,
        envelope_bindings=preflight.plan.bindings,
        envelope_identity_projections=settlement.identity_projections,
    )

    assert all(plan.owns_geometry for plan in final.plan.convergence_plans)
    assert tuple(plan.member_ids for plan in final.plan.convergence_plans) == tuple(
        plan.member_ids for plan in preflight.plan.convergence_plans
    )
    assert final.plan.bindings == preflight.plan.bindings
    assert not check_merge_branches_meet_trunk(
        graph, final.routes, compute_station_offsets(graph)
    )
    assert not check_merge_feeders_land_on_trunk(
        graph, final.routes, compute_station_offsets(graph)
    )


def test_seed41_has_complete_planned_convergences() -> None:
    graph, offsets, observed = _observe(FROZEN / "seed_41.mmd")

    assert observed.plan.convergence_plans
    assert all(plan.owns_geometry for plan in observed.plan.convergence_plans)
    assert not check_merge_branches_meet_trunk(graph, observed.routes, offsets)
    assert not check_merge_feeders_land_on_trunk(graph, observed.routes, offsets)


def test_immutable_covered_continuation_uses_its_named_carrier_terminal() -> None:
    path = ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd"
    graph, _offsets, preflight = _observe(path)
    continuation_edge = next(
        member.edge
        for member in preflight.plan.members
        if (member.edge.source, member.edge.target, member.line_id)
        == ("__merge_3", "side__entry_left_3", "a")
    )
    continuation_member = next(
        member for member in preflight.plan.members if member.edge == continuation_edge
    )
    binding = next(
        item
        for item in preflight.plan.bindings
        if item.member_id == continuation_member.id
    )

    assert binding.kind is BindingKind.COVERED_MERGE_HOP
    assert binding.covering_member_id is not None

    settlement = settle_route_envelopes(graph, preflight.plan)
    final = observe_route_edges(
        graph,
        station_offsets=compute_station_offsets(graph),
        envelope_proofs=settlement.capacity_proofs,
        envelope_limitations=settlement.capacity_limitations,
        envelope_reservations=preflight.plan.reservations,
        envelope_bindings=preflight.plan.bindings,
        envelope_identity_projections=settlement.identity_projections,
    )
    plan = next(
        item
        for item in final.plan.convergence_plans
        if continuation_member.id in item.member_ids
    )
    continuation = next(
        item
        for item in plan.outgoing_continuations
        if item.member_id == continuation_member.id
    )
    routes_by_member = {
        route.convergence_member_id: route
        for route in final.routes
        if route.convergence_member_id is not None
    }
    carrier = routes_by_member[str(binding.covering_member_id)]
    primary = routes_by_member[str(plan.primary_trunk_member_id)]

    assert plan.owns_geometry
    assert continuation.covered_by_member_id == binding.covering_member_id
    assert continuation.start_point == carrier.points[-2]
    assert continuation.start_point != primary.points[-2]
    assert point_to_polyline_distance(continuation.start_point, carrier.points) <= 1e-6
    assert point_to_polyline_distance(continuation.end_point, carrier.points) <= 1e-6


@pytest.mark.parametrize(
    ("side", "source", "target", "expected"),
    (
        (PortSide.LEFT, (9, 1), (3, 2), -1),
        (PortSide.RIGHT, (9, 3), (3, 2), 1),
        (PortSide.TOP, (1, 9), (2, 3), -1),
        (PortSide.BOTTOM, (3, 9), (2, 3), 1),
    ),
)
def test_merge_entry_cross_axis_order_transposes_rows_and_columns(
    side: PortSide,
    source: tuple[int, int],
    target: tuple[int, int],
    expected: int,
) -> None:
    facts = SimpleNamespace(
        src_col=source[0],
        src_row=source[1],
        tgt_col=target[0],
        tgt_row=target[1],
    )

    assert _merge_entry_cross_axis_order(facts, side) == expected


def test_seed41_right_entry_convergences_clear_sections_and_keep_endpoints() -> None:
    graph, offsets, observed = _observe(FROZEN / "seed_41.mmd")
    routes_by_edge = {
        (route.edge.source, route.edge.target, route.edge.line_id): route
        for route in observed.routes
        if route.convergence_member_id is not None
    }
    plans = tuple(
        plan
        for plan in observed.plan.convergence_plans
        if graph.ports[plan.target_entry_port_ids[0]].side is PortSide.RIGHT
    )

    assert plans
    opposing_merge_feeder = next(
        member
        for member in observed.plan.members
        if (member.edge.source, member.edge.target, member.line_id)
        == ("__junction_33", "__merge_11", "l0")
    )
    assert opposing_merge_feeder.family_id is RouteFamilyId.MERGE_ENTRY
    assert not routes_through_own_section_interior(
        graph, routes=observed.routes, offsets=offsets
    )
    assert not routes_through_unrelated_sections(
        graph, routes=observed.routes, offsets=offsets
    )
    for plan in plans:
        for landing in plan.landings:
            route = routes_by_edge[
                (landing.edge.source, landing.edge.target, landing.edge.line_id)
            ]
            assert point_to_polyline_distance(landing.join_point, route.points) <= 1e-6
        edge_by_member = dict(
            zip(plan.member_ids, plan.resolved_member_edges, strict=True)
        )
        for continuation in plan.outgoing_continuations:
            carrier_id = continuation.covered_by_member_id or continuation.member_id
            carrier_edge = edge_by_member[carrier_id]
            carrier = routes_by_edge[
                (carrier_edge.source, carrier_edge.target, carrier_edge.line_id)
            ]
            assert (
                point_to_polyline_distance(continuation.end_point, carrier.points)
                <= 1e-6
            )


def test_mixed_direct_bypass_and_multirow_approaches_are_frozen() -> None:
    _graph, _offsets, _preflight, observed = _observe_after_settlement(
        FROZEN / "seed_15.mmd"
    )
    landings = [
        landing for plan in observed.plan.convergence_plans for landing in plan.landings
    ]

    assert {landing.bypass for landing in landings} == {False, True}
    assert any(landing.long_haul for landing in landings)
    assert any(landing.multiple_row for landing in landings)


def test_packed_adjacency_convergences_are_planned() -> None:
    graph, offsets, observed = _observe(TOPOLOGIES / "merge_adjacent_feeder.mmd")

    assert observed.plan.convergence_plans
    assert all(plan.owns_geometry for plan in observed.plan.convergence_plans)
    assert not check_merge_branches_meet_trunk(graph, observed.routes, offsets)
    assert not check_merge_feeders_land_on_trunk(graph, observed.routes, offsets)


def test_fan_in_merge_settles_the_complete_system_before_emission() -> None:
    graph, offsets, observed = _observe(TOPOLOGIES / "fan_in_merge.mmd")

    assert observed.plan.convergence_plans
    assert all(plan.owns_geometry for plan in observed.plan.convergence_plans)
    assert all(plan.legacy_reason is None for plan in observed.plan.convergence_plans)
    assert not check_merge_branches_meet_trunk(graph, observed.routes, offsets)
    assert not check_merge_feeders_land_on_trunk(graph, observed.routes, offsets)


@pytest.mark.parametrize(
    ("direction", "side"),
    (
        ("LR", PortSide.LEFT),
        ("RL", PortSide.RIGHT),
        ("TB", PortSide.TOP),
        ("BT", PortSide.BOTTOM),
    ),
)
def test_target_section_orientations_use_one_convergence_model(
    direction: str,
    side: PortSide,
) -> None:
    source = (GUIDE / "03b_fan_in_merge.mmd").read_text()
    text = source.replace(
        "subgraph sink [Sink]",
        "subgraph sink [Sink]\n"
        f"        %%metro direction: {direction}\n"
        f"        %%metro entry: {side.value} | main, aux",
    )
    graph, offsets, observed = _observe_text(text)
    sink_plans = [
        plan
        for plan in observed.plan.convergence_plans
        if graph.ports[plan.target_entry_port_ids[0]].section_id == "sink"
    ]

    assert sink_plans
    assert graph.sections["sink"].direction == direction
    assert all(
        graph.ports[plan.target_entry_port_ids[0]].side is side for plan in sink_plans
    )
    assert all(plan.owns_geometry for plan in sink_plans)
    assert all(plan.legacy_reason is None for plan in sink_plans)
    assert not check_merge_feeders_land_on_trunk(graph, observed.routes, offsets)


@pytest.mark.parametrize(
    "path",
    (
        FROZEN / "seed_15.mmd",
        ROOT / "tests/fixtures/regressions/cross_column_perp_entry_overflow.mmd",
    ),
)
def test_planned_opening_turns_match_realised_allocations(path: Path) -> None:
    from nf_metro.layout.routing.normalize import _opening_fanout_descent

    _graph, _offsets, _preflight, observed = _observe_after_settlement(path)
    planned = {
        landing.member_id: landing
        for plan in observed.plan.convergence_plans
        if plan.owns_geometry
        for landing in plan.landings
        if landing.opening_turn_coordinate is not None
    }

    assert planned
    for route in observed.routes:
        if route.convergence_member_id not in {str(item) for item in planned}:
            continue
        landing = planned[
            next(
                member_id
                for member_id in planned
                if str(member_id) == route.convergence_member_id
            )
        ]
        opening = _opening_fanout_descent(route)
        assert opening is not None
        assert landing.opening_turn_segment is not None
        expected = [list(point) for point in landing.opening_turn_segment]
        for local_rank in range(2):
            point_rank = opening.idx + local_rank
            for (
                segment_rank,
                coordinate_rank,
                coordinate,
            ) in route.envelope_allocated_segments:
                if segment_rank <= point_rank <= segment_rank + 1:
                    expected[local_rank][coordinate_rank] = coordinate
        assert tuple(route.points[opening.idx : opening.idx + 2]) == tuple(
            tuple(point) for point in expected
        )
        assert opening.idx in route.convergence_owned_segment_ranks


def test_runtime_guard_rejects_a_mutated_planned_opening_segment() -> None:
    _graph, _offsets, _preflight, observed = _observe_after_settlement(
        FROZEN / "seed_15.mmd"
    )
    from nf_metro.layout.routing.normalize import _opening_fanout_descent

    candidates = []
    for plan in observed.plan.convergence_plans:
        for landing in plan.landings:
            if (
                landing.opening_turn_segment is None
                or landing.member_id == plan.primary_trunk_member_id
            ):
                continue
            route = next(
                item
                for item in observed.routes
                if item.convergence_member_id == str(landing.member_id)
            )
            opening = _opening_fanout_descent(route)
            if opening is not None and opening.idx + 2 < len(route.points):
                candidates.append((plan, landing, route, opening))
    plan, _landing, route, opening = candidates[0]
    x, y = route.points[opening.idx + 1]
    route.points[opening.idx + 1] = (x, y + 3.0)
    execution = replace(
        convergence_routing.empty_convergence_plan_execution(),
        plans=(plan,),
        query=convergence_routing._query((plan,)),
    )

    with pytest.raises(ConvergenceInvariantError, match="planned opening"):
        validate_convergence_plans(observed.routes, execution)


@pytest.mark.parametrize(
    ("axis", "route_points", "segment_rank"),
    (
        (
            ConvergenceTrunkAxis(
                DemandAxis.X,
                20.0,
                10.0,
                30.0,
                Direction.R,
                0.0,
                40.0,
            ),
            [
                (0.0, 0.0),
                (12.0, 0.0),
                (12.0, 20.0),
                (30.0, 20.0),
                (30.0, 40.0),
                (40.0, 40.0),
            ],
            1,
        ),
        (
            ConvergenceTrunkAxis(
                DemandAxis.Y,
                20.0,
                10.0,
                30.0,
                Direction.D,
                0.0,
                40.0,
            ),
            [
                (0.0, 0.0),
                (0.0, 12.0),
                (20.0, 12.0),
                (20.0, 30.0),
                (40.0, 30.0),
                (40.0, 40.0),
            ],
            1,
        ),
    ),
)
def test_trunk_flank_settlement_rederives_curve_radii(
    axis: ConvergenceTrunkAxis,
    route_points: list[tuple[float, float]],
    segment_rank: int,
) -> None:
    route = RoutedPath(
        Edge("source", "target", "line"),
        "line",
        route_points,
        is_inter_section=True,
        curve_radii=[99.0] * (len(route_points) - 2),
        offset_regime=OffsetRegime.BAKED,
    )

    _seat_route_on_trunk_flanks(route, axis, MetroGraph(), lane_offset=2.0)

    assert route.curve_radii is not None
    radii_and_offsets = (
        (segment_rank - 1, 2.0),
        (segment_rank, 0.0),
        (segment_rank + 1, 0.0),
        (segment_rank + 2, -2.0),
    )
    for radius_rank, offset in radii_and_offsets:
        assert route.curve_radii[radius_rank] == pytest.approx(
            concentric_corner_radius_at(
                *route.points[radius_rank : radius_rank + 3],
                offset,
                10.0,
            )
        )
        assert route.curve_radii[radius_rank] != 99.0


def test_perpendicular_top_entry_convergences_travel_from_exterior_to_entry() -> None:
    path = (
        ROOT
        / "tests"
        / "fixtures"
        / "regressions"
        / "cross_column_perp_entry_overflow.mmd"
    )
    graph, _offsets, observed = _observe(path)
    vertical = [
        plan
        for plan in observed.plan.convergence_plans
        if plan.trunk_axis is not None and plan.trunk_axis.axis is DemandAxis.Y
    ]

    assert vertical
    assert {graph.ports[plan.target_entry_port_ids[0]].side for plan in vertical} == {
        PortSide.TOP
    }
    assert {plan.trunk_axis.direction for plan in vertical} == {Direction.D}
    for plan in vertical:
        axis = plan.trunk_axis
        assert axis.source_endpoint_coordinate is not None
        assert axis.target_endpoint_coordinate is not None
        assert axis.source_endpoint_coordinate < axis.target_endpoint_coordinate
        continuation = plan.outgoing_continuations[0]
        assert continuation.covered_by_member_id is not None
        assert continuation.start_point != continuation.end_point
        carrier = next(
            route
            for route in observed.routes
            if route.convergence_member_id == str(continuation.covered_by_member_id)
        )
        assert point_to_polyline_distance(continuation.start_point, carrier.points) == 0
        assert point_to_polyline_distance(continuation.end_point, carrier.points) == 0


def test_planned_landing_facts_match_emitted_terminal_geometry() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]

    for landing in plan.landings:
        route = next(
            item
            for item in observed.routes
            if item.convergence_member_id == str(landing.member_id)
        )
        approach = convergence_routing._landing_approach(route, landing.join_point)
        assert approach is not None
        direction, handedness, runway = approach
        assert direction is landing.approach_direction
        assert handedness is landing.corner_handedness
        assert runway >= landing.minimum_runway
        if landing.member_id != plan.primary_trunk_member_id:
            assert not route.normalize_exempt


def test_runtime_guard_rejects_reduced_planned_landing_runway() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    landing = next(
        item
        for item in plan.landings
        if item.member_id != plan.primary_trunk_member_id and item.minimum_runway > 2.0
    )
    route = next(
        item
        for item in observed.routes
        if item.convergence_member_id == str(landing.member_id)
    )
    join_x, join_y = landing.join_point
    if landing.approach_axis is DemandAxis.X:
        sign = 1 if landing.approach_direction is Direction.R else -1
        route.points[-2] = (join_x - sign * 2.0, join_y)
    else:
        sign = 1 if landing.approach_direction is Direction.D else -1
        route.points[-2] = (join_x, join_y - sign * 2.0)
    execution = replace(
        convergence_routing.empty_convergence_plan_execution(),
        plans=(plan,),
        query=convergence_routing._query((plan,)),
    )

    with pytest.raises(ConvergenceInvariantError, match="runway"):
        validate_convergence_plans(observed.routes, execution)


@pytest.mark.parametrize(
    ("entry_point", "axis", "direction"),
    (
        ((20.0, 10.0), DemandAxis.X, "R"),
        ((0.0, 10.0), DemandAxis.X, "L"),
        ((10.0, 20.0), DemandAxis.Y, "D"),
        ((10.0, 0.0), DemandAxis.Y, "U"),
    ),
)
def test_direct_trunk_axis_rotates_and_reverses(
    entry_point: tuple[float, float],
    axis: DemandAxis,
    direction: str,
) -> None:
    merge = Station("merge", "", x=10.0, y=10.0)
    entry = Station("entry", "", x=entry_point[0], y=entry_point[1])
    trunk = _direct_axis(merge, entry)

    assert trunk.axis is axis
    assert trunk.direction.value == direction


def test_one_planning_failure_rolls_back_the_whole_route_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise UnsupportedConvergenceError("unsupported convergence shape")

    monkeypatch.setattr(convergence_routing, "_build_planned_convergence", reject)
    _graph, _offsets, observed = _observe(FROZEN / "seed_15.mmd")
    plans = observed.plan.convergence_plans

    assert len(plans) > 1
    assert {plan.disposition for plan in plans} == {ConvergenceDisposition.LEGACY}
    assert all(plan.legacy_reason == "unsupported convergence shape" for plan in plans)
    assert all(not plan.shared_reference_ids and not plan.demand_ids for plan in plans)
    assert sum(
        diagnostic.code == "convergence-plan-legacy"
        for diagnostic in observed.plan.diagnostics
    ) == len(plans)


@pytest.mark.parametrize("error", (AssertionError("bug"), TypeError("bug")))
def test_programming_errors_do_not_silently_fall_back(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def reject(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(convergence_routing, "_build_planned_convergence", reject)

    with pytest.raises(type(error), match="bug"):
        _observe(FROZEN / "seed_15.mmd")


def test_incomplete_semantic_membership_is_a_planning_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise ConvergencePlanningError("missing member")

    monkeypatch.setattr(convergence_routing, "_plan_membership", reject)

    with pytest.raises(ConvergencePlanningError, match="missing member"):
        _observe(FROZEN / "seed_15.mmd")


def test_exit_turn_shared_channel_has_exact_whole_system_ownership() -> None:
    path = TOPOLOGIES / "exit_run_three_drop_columns.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    _render_plan, observed = build_observed_render_plan(
        graph,
        resolve_theme(None, graph),
    )
    plans = observed.convergence_plans

    assert plans
    assert all(
        plan.disposition is ConvergenceDisposition.PLANNED
        and plan.legacy_reason is None
        and plan.primary_trunk_member_id in plan.member_ids
        and set(plan.member_ids)
        == {ownership.member_id for ownership in plan.endpoint_ownership}
        for plan in plans
    )


@pytest.mark.parametrize(
    ("axis", "points", "expected"),
    (
        (
            DemandAxis.X,
            [
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 20.0),
                (20.0, 20.0),
                (20.0, 40.0),
                (30.0, 40.0),
            ],
            [
                (0.0, 0.0),
                (5.0, 0.0),
                (5.0, 20.0),
                (25.0, 20.0),
                (25.0, 40.0),
                (30.0, 40.0),
            ],
        ),
        (
            DemandAxis.Y,
            [
                (0.0, 0.0),
                (0.0, 10.0),
                (20.0, 10.0),
                (20.0, 20.0),
                (40.0, 20.0),
                (40.0, 30.0),
            ],
            [
                (0.0, 0.0),
                (0.0, 5.0),
                (20.0, 5.0),
                (20.0, 25.0),
                (40.0, 25.0),
                (40.0, 30.0),
            ],
        ),
    ),
)
def test_trunk_flank_extension_is_axis_generic(
    axis: DemandAxis,
    points: list[tuple[float, float]],
    expected: list[tuple[float, float]],
) -> None:
    route = RoutedPath(Edge("source", "target", "line"), "line", points)

    _extend_axis_segment_to_coordinates(route, 2, axis, (5.0, 25.0))

    assert route.points == expected


def test_runtime_guard_names_the_plan_member_and_broken_join() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    ownership = next(
        item
        for item in plan.endpoint_ownership
        if item.role is ConvergenceEndpointRole.FEEDER
        and plan.connector_ids[0] not in item.connector_ids
    )
    landing = next(
        item for item in plan.landings if item.member_id == ownership.member_id
    )
    route = next(
        item
        for item in observed.routes
        if item.convergence_member_id == str(landing.member_id)
    )
    route.points[-1] = (route.points[-1][0], route.points[-1][1] + 100.0)
    execution = replace(
        convergence_routing.empty_convergence_plan_execution(),
        plans=(plan,),
        query=convergence_routing._query((plan,)),
    )

    with pytest.raises(ConvergenceInvariantError) as error:
        validate_convergence_plans(observed.routes, execution)

    message = str(error.value)
    assert str(plan.system_id) in message
    connector_set = ", ".join(
        str(connector_id) for connector_id in ownership.connector_ids
    )
    assert f"connectors {connector_set} member" in message
    assert str(landing.member_id) in message
    assert "planned join" in message
    assert "emitted endpoint" in message


def test_runtime_guard_rejects_a_disconnected_diagonal_trunk() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    route = next(
        item
        for item in observed.routes
        if item.convergence_member_id == str(plan.primary_trunk_member_id)
    )
    axis = plan.trunk_axis
    assert axis is not None
    segment_rank = next(
        rank
        for rank, (start, end) in enumerate(zip(route.points, route.points[1:]))
        if abs(start[1] - axis.coordinate) < 1e-6
        and abs(end[1] - axis.coordinate) < 1e-6
        and min(start[0], end[0]) <= axis.extent_start
        and max(start[0], end[0]) >= axis.extent_end
    )
    end = route.points[segment_rank + 1]
    route.points[segment_rank + 1] = (end[0], end[1] + 20.0)
    execution = replace(
        convergence_routing.empty_convergence_plan_execution(),
        plans=(plan,),
        query=convergence_routing._query((plan,)),
    )

    with pytest.raises(ConvergenceInvariantError, match="does not emit planned"):
        validate_convergence_plans(observed.routes, execution)


def test_runtime_guard_rejects_a_missing_terminal_trunk_cap() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    route = next(
        item
        for item in observed.routes
        if item.convergence_member_id == str(plan.primary_trunk_member_id)
    )
    axis = plan.trunk_axis
    assert axis is not None
    assert axis.source_endpoint_coordinate is not None
    source_longitudinal = axis.extent_start
    segment_rank = next(
        rank
        for rank, (start, end) in enumerate(zip(route.points, route.points[1:]))
        if abs(start[1] - axis.source_flank_coordinate) < 1e-6
        and abs(end[1] - axis.source_flank_coordinate) < 1e-6
        and min(start[0], end[0]) <= axis.source_endpoint_coordinate
        and max(start[0], end[0]) >= source_longitudinal
    )
    route.points[segment_rank + 1] = route.points[segment_rank]
    execution = replace(
        convergence_routing.empty_convergence_plan_execution(),
        plans=(plan,),
        query=convergence_routing._query((plan,)),
    )

    with pytest.raises(ConvergenceInvariantError, match="does not emit planned"):
        validate_convergence_plans(observed.routes, execution)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("entry_group_ids", ("unknown-entry",)),
        ("merge_junction_ids", ("unknown-merge",)),
        ("target_entry_port_ids", ("unknown-port",)),
    ),
)
def test_route_plan_rejects_mutated_convergence_semantic_identity(
    field: str, value: tuple[str, ...]
) -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    mutated = replace(plan, **{field: value})
    route_plan = replace(observed.plan, convergence_plans=(mutated,))

    with pytest.raises(ValueError, match="semantic fields"):
        build_route_plan_query(route_plan)


def test_route_plan_rejects_coverage_that_disagrees_with_binding() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    covered = next(
        item
        for item in plan.endpoint_ownership
        if item.role is ConvergenceEndpointRole.COVERED_CONTINUATION
    )
    ownership = tuple(
        replace(
            item,
            role=ConvergenceEndpointRole.CONTINUATION,
            covered_by_member_id=None,
        )
        if item.member_id == covered.member_id
        else item
        for item in plan.endpoint_ownership
    )
    mutated = replace(plan, endpoint_ownership=ownership)
    route_plan = replace(observed.plan, convergence_plans=(mutated,))

    with pytest.raises(ValueError, match="endpoint owner"):
        build_route_plan_query(route_plan)


def test_route_plan_rejects_endpoint_connectors_from_another_member() -> None:
    _graph, _offsets, observed = _observe(
        TOPOLOGIES / "merge_feeders_three_columns.mmd"
    )
    plan = observed.plan.convergence_plans[0]
    first, second = plan.endpoint_ownership[:2]
    assert first.connector_ids != second.connector_ids
    ownership = (
        replace(first, connector_ids=second.connector_ids),
        *plan.endpoint_ownership[1:],
    )
    mutated = replace(plan, endpoint_ownership=ownership)
    route_plan = replace(observed.plan, convergence_plans=(mutated,))

    with pytest.raises(ValueError, match="connectors disagree with member"):
        build_route_plan_query(route_plan)


@pytest.mark.parametrize(
    "changes",
    (
        {"kind": SharedReferenceKind.BAND},
        {"coordinate_regime": CoordinateRegime.RELATIVE_FRAME},
        {"claimant_member_ids": ()},
    ),
)
def test_route_plan_rejects_mutated_convergence_references(
    changes: dict[str, object],
    three_column_route_plan: RoutePlan,
) -> None:
    plan = three_column_route_plan.convergence_plans[0]
    reference_id = plan.shared_reference_ids[0]
    malformed = next(
        replace(item, **changes)
        for item in three_column_route_plan.shared_references
        if item.id == reference_id
    )
    references = tuple(
        malformed if item.id == reference_id else item
        for item in three_column_route_plan.shared_references
    )

    with pytest.raises(ValueError, match="shared references"):
        build_route_plan_query(
            replace(three_column_route_plan, shared_references=references)
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"kind": DemandKind.KEEP_OUT},
        {"axis": DemandAxis.BOTH},
        {"lane_count": 999},
        {"ordered_reference_ids": ()},
        {"keep_out_classes": (KeepOutClass.SECTION,)},
        {"claimant_member_ids": ()},
    ),
)
def test_route_plan_rejects_mutated_convergence_lane_demands(
    changes: dict[str, object],
    three_column_route_plan: RoutePlan,
) -> None:
    plan = three_column_route_plan.convergence_plans[0]
    demand_id = plan.demand_ids[0]
    malformed = next(
        replace(item, **changes)
        for item in three_column_route_plan.demands
        if item.id == demand_id
    )
    demands = tuple(
        malformed if item.id == demand_id else item
        for item in three_column_route_plan.demands
    )

    with pytest.raises(ValueError, match="symbolic demands"):
        build_route_plan_query(replace(three_column_route_plan, demands=demands))


def test_route_plan_rejects_mutated_convergence_runway_demand(
    three_column_route_plan: RoutePlan,
) -> None:
    plan = three_column_route_plan.convergence_plans[0]
    demand_id = plan.demand_ids[1]
    malformed = next(
        replace(item, minimum_size=item.minimum_size + 1.0)
        for item in three_column_route_plan.demands
        if item.id == demand_id and item.minimum_size is not None
    )
    demands = tuple(
        malformed if item.id == demand_id else item
        for item in three_column_route_plan.demands
    )

    with pytest.raises(ValueError, match="symbolic demands"):
        build_route_plan_query(replace(three_column_route_plan, demands=demands))


def test_route_plan_rejects_duplicate_semantic_convergence_coverage(
    right_entry_route_plan: RoutePlan,
) -> None:
    (plan,) = right_entry_route_plan.convergence_plans
    duplicate_id = ConvergencePlanId("duplicate-convergence-plan")
    duplicate_reference_ids, duplicate_demand_ids = convergence_resource_ids(
        duplicate_id
    )
    reference_id_map = dict(
        zip(plan.shared_reference_ids, duplicate_reference_ids, strict=True)
    )
    duplicate = replace(
        plan,
        id=duplicate_id,
        shared_reference_ids=duplicate_reference_ids,
        demand_ids=duplicate_demand_ids,
    )
    references_by_id = {
        item.id: item for item in right_entry_route_plan.shared_references
    }
    demands_by_id = {item.id: item for item in right_entry_route_plan.demands}
    duplicate_references = tuple(
        replace(references_by_id[source_id], id=target_id)
        for source_id, target_id in zip(
            plan.shared_reference_ids, duplicate_reference_ids, strict=True
        )
    )
    duplicate_demands = tuple(
        replace(
            demands_by_id[source_id],
            id=target_id,
            ordered_reference_ids=tuple(
                reference_id_map[reference_id]
                for reference_id in demands_by_id[source_id].ordered_reference_ids
            ),
        )
        for source_id, target_id in zip(
            plan.demand_ids, duplicate_demand_ids, strict=True
        )
    )
    systems = tuple(
        replace(
            item,
            convergence_plan_ids=(*item.convergence_plan_ids, duplicate.id),
        )
        if item.id == plan.system_id
        else item
        for item in right_entry_route_plan.systems
    )
    route_plan = replace(
        right_entry_route_plan,
        systems=systems,
        convergence_plans=(plan, duplicate),
        shared_references=(
            *right_entry_route_plan.shared_references,
            *duplicate_references,
        ),
        demands=(*right_entry_route_plan.demands, *duplicate_demands),
    )

    with pytest.raises(ValueError, match="coverage"):
        build_route_plan_query(route_plan)


def test_route_plan_rejects_missing_semantic_convergence_coverage(
    right_entry_route_plan: RoutePlan,
) -> None:
    (plan,) = right_entry_route_plan.convergence_plans
    systems = tuple(
        replace(item, convergence_plan_ids=()) if item.id == plan.system_id else item
        for item in right_entry_route_plan.systems
    )

    with pytest.raises(ValueError, match="coverage"):
        build_route_plan_query(
            replace(right_entry_route_plan, systems=systems, convergence_plans=())
        )


def test_route_plan_rejects_incomplete_convergence_emission_membership(
    right_entry_route_plan: RoutePlan,
) -> None:
    (plan,) = right_entry_route_plan.convergence_plans
    remaining_paths = plan.resolved_member_paths[1:]
    remaining_edges = tuple(
        dict.fromkeys(edge for path in remaining_paths for edge in path)
    )
    member_by_edge = {item.edge: item.id for item in right_entry_route_plan.members}
    remaining_member_ids = tuple(member_by_edge[edge] for edge in remaining_edges)
    remaining_member_id_set = set(remaining_member_ids)
    remaining_landings = tuple(
        replace(item, order=rank)
        for rank, item in enumerate(
            item for item in plan.landings if item.member_id in remaining_member_id_set
        )
    )
    mutated = replace(
        plan,
        member_ids=remaining_member_ids,
        resolved_member_paths=remaining_paths,
        resolved_member_edges=remaining_edges,
        landings=remaining_landings,
        outgoing_continuations=tuple(
            item
            for item in plan.outgoing_continuations
            if item.member_id in remaining_member_id_set
        ),
        endpoint_ownership=tuple(
            item
            for item in plan.endpoint_ownership
            if item.member_id in remaining_member_id_set
        ),
    )

    with pytest.raises(ValueError, match="membership is incomplete"):
        build_route_plan_query(
            replace(right_entry_route_plan, convergence_plans=(mutated,))
        )
