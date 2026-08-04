"""Settled envelope allocations retain exact immutable ownership."""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

import nf_metro.layout.envelope_settlement as settlement_module
from nf_metro.api import prepare_graph, render_string, resolve_theme
from nf_metro.layout.constants import SECTION_Y_GAP, graph_offset_step
from nf_metro.layout.envelope_settlement import settle_route_envelopes
from nf_metro.layout.route_plan import (
    BindingKind,
    DemandId,
    ResolvedEdge,
    RouteSystemId,
    SharedReferenceId,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    CanvasSide,
    CorridorOrientation,
    RouteReservationDiagnostic,
    canvas_inner_boundary,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.layout.routing.common import (
    RoutedPath,
    iter_port_peeloff_bundles,
    peeloff_target_slots,
    projected_perp_entry_lane_coordinate,
    tail_on_slot,
)
from nf_metro.layout.routing.convergences import (
    ConvergenceAllocationConflict,
    ConvergenceAllocationNeed,
    _capacity_proofs_for_conflict,
)
from nf_metro.layout.routing.envelope_allocations import (
    EnvelopeAllocationError,
    build_envelope_allocation_query,
)
from nf_metro.layout.routing.normalize import _Coincidence, _snap_group, _VChannel
from nf_metro.parser.model import (
    Edge,
    LayoutGeometryWarning,
    MetroGraph,
    PermissiveGuardWarning,
    Port,
    PortSide,
    Section,
    Station,
)
from nf_metro.render.svg import (
    _settle_render_geometry,
    build_observed_render_plan,
    emit_render_plan,
)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "examples" / "topologies" / "merge_around_below_leftmost.mmd"


def _settled_records():
    graph = prepare_graph(FIXTURE.read_text(), source_dir=str(FIXTURE.parent))
    plan = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    settlement = settle_route_envelopes(graph, plan)
    member_by_edge = {item.edge: item.id for item in plan.members}
    return plan, settlement.capacity_proofs, member_by_edge


def test_query_authenticates_every_proof_against_immutable_reservation() -> None:
    plan, proofs, member_by_edge = _settled_records()

    query = build_envelope_allocation_query(
        proofs, member_by_edge, plan.reservations, plan.bindings
    )

    assert any(
        query.allocations_for_member(member_id) for member_id in member_by_edge.values()
    )


@pytest.mark.parametrize("resource", ("reference", "demand"))
def test_query_rejects_forged_proof_resources(resource: str) -> None:
    plan, proofs, member_by_edge = _settled_records()
    proof = proofs[0]
    allocation = proof.reservations[0]
    forged = (
        replace(allocation, reference_id=SharedReferenceId("forged-reference"))
        if resource == "reference"
        else replace(allocation, demand_ids=(DemandId("forged-demand"),))
    )
    malformed = replace(proof, reservations=(forged, *proof.reservations[1:]))

    with pytest.raises(
        EnvelopeAllocationError,
        match="disagrees with its immutable reservation",
    ):
        build_envelope_allocation_query(
            (malformed, *proofs[1:]),
            member_by_edge,
            plan.reservations,
            plan.bindings,
        )


def test_query_rejects_proofs_without_the_immutable_ledger() -> None:
    _plan, proofs, member_by_edge = _settled_records()

    with pytest.raises(EnvelopeAllocationError, match="immutable reservation ledger"):
        build_envelope_allocation_query(proofs, member_by_edge)


def test_disjoint_proof_groups_do_not_jointly_authorise_one_conflict() -> None:
    _plan, proofs, _member_by_edge = _settled_records()
    first = proofs[0]
    second = next(
        proof
        for proof in proofs[1:]
        if proof.system_ids == first.system_ids
        and {
            allocation.member_id
            for reservation in proof.reservations
            for allocation in reservation.allocations
        }.isdisjoint(
            allocation.member_id
            for reservation in first.reservations
            for allocation in reservation.allocations
        )
    )
    first_member = first.reservations[0].allocations[0].member_id
    second_member = second.reservations[0].allocations[0].member_id
    conflict = ConvergenceAllocationConflict(
        ConvergenceAllocationNeed.SHARED_CHANNEL,
        "requires one jointly owned channel",
        (first_member, second_member),
    )

    assert not _capacity_proofs_for_conflict(
        first.system_ids[0], conflict, (first, second)
    )


def test_query_rejects_forged_direct_claim_identity() -> None:
    plan, proofs, member_by_edge = _settled_records()
    proof = proofs[0]
    reservation = proof.reservations[0]
    claim = reservation.allocations[0]
    forged_claim = replace(
        claim,
        segment_rank=claim.segment_rank + 1,
        segment_end_rank=claim.segment_end_rank + 1,
    )
    forged_reservation = replace(reservation, allocations=(forged_claim,))
    forged_proof = replace(
        proof, reservations=(forged_reservation, *proof.reservations[1:])
    )

    with pytest.raises(
        EnvelopeAllocationError,
        match="changed its immutable claim projection",
    ):
        build_envelope_allocation_query(
            (forged_proof, *proofs[1:]),
            member_by_edge,
            plan.reservations,
            plan.bindings,
        )


def test_query_rejects_forged_aggregate_system_membership() -> None:
    plan, proofs, member_by_edge = _settled_records()
    proof = proofs[0]
    forged = replace(
        proof,
        system_ids=(*proof.system_ids, RouteSystemId("forged-system")),
    )

    with pytest.raises(
        EnvelopeAllocationError,
        match="inconsistent aggregate ownership",
    ):
        build_envelope_allocation_query(
            (forged, *proofs[1:]),
            member_by_edge,
            plan.reservations,
            plan.bindings,
        )


def test_canvas_identity_allocation_consumes_unowned_member_exactly() -> None:
    path = ROOT / "examples" / "topologies" / "merge_right_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    plan = observed.plan
    member_by_edge = {item.edge: item.id for item in plan.members}
    convergence_members = {
        member_id
        for convergence in plan.convergence_plans
        for member_id in convergence.member_ids
    }
    unowned_claim = next(
        claim
        for reservation in plan.reservations
        if isinstance(reservation.region, CanvasRegion)
        for claim in reservation.claims
        if claim.member_id not in convergence_members
    )
    settlement = settle_route_envelopes(graph, plan)
    query = build_envelope_allocation_query(
        settlement.capacity_proofs,
        member_by_edge,
        plan.reservations,
        plan.bindings,
        settlement.capacity_limitations,
        settlement.identity_projections,
    )

    assert query.directly_allocates((unowned_claim.member_id,))
    for route in observed.routes:
        query.consume(route)
    query.assert_complete(observed.routes)
    allocation = next(
        item
        for item in query.allocations_for_member(unowned_claim.member_id)
        if item.path_rank == unowned_claim.path_rank
        and item.segment_rank == unowned_claim.segment_rank
        and item.segment_end_rank == unowned_claim.segment_end_rank
    )
    assert allocation.original_coordinate == unowned_claim.allocation_coordinate
    route = observed.routes[unowned_claim.path_rank]
    axis_rank = 0 if allocation.axis.value == "x" else 1
    assert all(
        route.points[rank][axis_rank] == pytest.approx(allocation.coordinate)
        and route.points[rank + 1][axis_rank] == pytest.approx(allocation.coordinate)
        for rank in range(allocation.segment_rank, allocation.segment_end_rank + 1)
    )


def test_organellar_reservations_all_exit_with_exact_capacity_proofs() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    plan = observed.plan
    settlement = settle_route_envelopes(graph, plan)
    proof_reservation_ids = {
        reservation.reservation_id
        for proof in settlement.capacity_proofs
        for reservation in proof.reservations
    }

    assert not settlement.capacity_limitations
    assert not settlement.identity_projections
    assert proof_reservation_ids == {item.id for item in plan.reservations}


def test_projected_identity_keeps_shared_canvas_channels_coincident() -> None:
    """A global row move translates a shared channel without unpacking it."""
    path = ROOT / "examples" / "longread_variant_calling.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    theme = resolve_theme(None, graph)
    _offsets, routes, _labels, _final, ledger, _limitations = _settle_render_geometry(
        graph,
        theme,
        graph_offset_step(graph, theme.line_width),
        graph.section_y_gap or SECTION_Y_GAP,
    )
    assert ledger is not None

    by_original: defaultdict[tuple[object, float], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    for reservation in ledger.reservations:
        if not isinstance(reservation.region, CanvasRegion):
            continue
        axis_rank = 0 if reservation.orientation is CorridorOrientation.VERTICAL else 1
        for claim in reservation.claims:
            route = routes[claim.path_rank]
            consumed = next(
                coordinate
                for rank, rank_axis, coordinate in route.envelope_allocated_segments
                if rank == claim.segment_rank and rank_axis == axis_rank
            )
            by_original[(reservation.id, claim.allocation_coordinate)].append(
                (claim.allocation_coordinate, consumed)
            )

    translated_shared = [
        values
        for values in by_original.values()
        if len(values) > 1 and abs(values[0][1] - values[0][0]) > 1e-6
    ]
    assert translated_shared
    assert all(
        len({round(consumed, 6) for _original, consumed in values}) == 1
        for values in translated_shared
    )
    staircase_member = next(
        member
        for member in ledger.members
        if (member.edge.source, member.edge.target, member.line_id)
        == ("__junction_19", "annotation__entry_right_14", "svvcf")
    )
    staircase_binding = next(
        binding
        for binding in ledger.bindings
        if binding.member_id == staircase_member.id
    )
    staircase = routes[staircase_binding.path_rank]
    assert staircase.points[0][1] != pytest.approx(staircase.points[2][1])
    assert staircase.points[1][1] != pytest.approx(staircase.points[2][1])
    assert staircase.points[1][0] == pytest.approx(staircase.points[2][0])
    bundle = next(
        item
        for item in iter_port_peeloff_bundles(
            routes, graph, graph_offset_step(graph, theme.line_width)
        )
        if item.port_id == "jointcalling__entry_right_13"
    )
    slots = peeloff_target_slots(bundle)
    assert all(
        tail_on_slot(tail, slots[route.line_id]) for route, tail in bundle.entries
    )
    for route, tail in bundle.entries:
        path_rank = routes.index(route)
        claim = next(
            claim
            for reservation in ledger.reservations
            for claim in reservation.claims
            if claim.path_rank == path_rank
            and claim.segment_rank == len(route.points) - 3
        )
        assert claim.allocation_coordinate == pytest.approx(slots[route.line_id].peel_x)
        assert tail.peel_x == pytest.approx(slots[route.line_id].peel_x)


def test_query_rejects_claim_path_rank_that_disagrees_with_binding() -> None:
    plan, proofs, member_by_edge = _settled_records()
    claimed_member = proofs[0].reservations[0].allocations[0].member_id
    forged_bindings = tuple(
        replace(binding, path_rank=binding.path_rank + 1)
        if binding.member_id == claimed_member
        else binding
        for binding in plan.bindings
    )

    with pytest.raises(
        EnvelopeAllocationError,
        match="immutable claim projection",
    ):
        build_envelope_allocation_query(
            proofs,
            member_by_edge,
            plan.reservations,
            forged_bindings,
        )


def test_final_consumption_rejects_route_path_rank_mismatch() -> None:
    path = ROOT / "examples" / "topologies" / "merge_right_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observed = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    plan = observed.plan
    member_by_edge = {item.edge: item.id for item in plan.members}
    settlement = settle_route_envelopes(graph, plan)
    query = build_envelope_allocation_query(
        settlement.capacity_proofs,
        member_by_edge,
        plan.reservations,
        plan.bindings,
        settlement.capacity_limitations,
        settlement.identity_projections,
    )
    routes = list(observed.routes)
    for route in routes:
        query.consume(route)
    query.assert_complete(routes)
    allocated_rank = next(
        rank
        for rank, route in enumerate(routes)
        if (member_id := query.member_for_route(route)) is not None
        and query.allocations_for_member(member_id)
    )
    swap_rank = allocated_rank - 1 if allocated_rank else allocated_rank + 1
    routes[allocated_rank], routes[swap_rank] = (
        routes[swap_rank],
        routes[allocated_rank],
    )

    with pytest.raises(EnvelopeAllocationError, match="final path rank"):
        query.assert_complete(routes)


def test_coincidence_does_not_override_distinct_planned_axes() -> None:
    def route(source: str, x: float, *, planned: bool) -> RoutedPath:
        item = RoutedPath(
            edge=Edge(source=source, target="target", line_id="shared"),
            line_id="shared",
            points=[(0.0, 0.0), (x, 0.0), (x, 20.0), (30.0, 20.0)],
            is_inter_section=True,
        )
        if planned:
            item.envelope_allocated_segments = ((1, 0, x),)
        return item

    routes = [
        route("fixed-left", 10.0, planned=True),
        route("fixed-right", 20.0, planned=True),
        route("unplanned", 15.0, planned=False),
    ]
    channels = [
        _VChannel(item, 1, item.points[1][0], 0.0, 20.0, True) for item in routes
    ]

    _snap_group(_Coincidence(channels, 10.0), MetroGraph())

    assert [item.points[1][0] for item in routes] == [10.0, 20.0, 15.0]


def test_final_render_preserves_allocated_dogleg_trunk() -> None:
    path = ROOT / "examples" / "topologies" / "dogleg_exempt_sameline.mmd"

    assert "<svg " in render_string(path.read_text(), chrome_css=False)


def test_non_convergence_perp_entry_preserves_its_bundled_boundary_seam() -> None:
    path = ROOT / "examples" / "topologies" / "cross_row_gap_wrap.mmd"
    with warnings.catch_warnings():
        warnings.simplefilter("error", LayoutGeometryWarning)
        warnings.simplefilter("error", PermissiveGuardWarning)
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        preflight = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        )
        final = build_observed_render_plan(graph, resolve_theme(None, graph))

    seam_by_line: dict[str, list[float]] = defaultdict(list)
    for route, points in zip(
        final.plan.routes, final.plan.route_polylines, strict=True
    ):
        edge = route.edge
        if (
            edge.source == "__junction_8" and edge.target == "merge_pt__entry_top_6"
        ) or (edge.source == "merge_pt__entry_top_6" and edge.target == "g1"):
            coordinate = (
                points[-1][0] if edge.source == "__junction_8" else points[0][0]
            )
            seam_by_line[route.line_id].append(coordinate)

    assert seam_by_line == {"main": [600.0, 600.0], "feed": [596.0, 596.0]}
    assert final.route_plan.bindings == preflight.plan.bindings
    assert final.route_plan.convergence_plans == preflight.plan.convergence_plans


def test_final_convergence_emission_preserves_immutable_route_membership() -> None:
    path = (
        ROOT
        / "tests"
        / "fixtures"
        / "regressions"
        / ("cross_column_perp_entry_overflow.mmd")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", LayoutGeometryWarning)
        warnings.simplefilter("error", PermissiveGuardWarning)
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        preflight = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        )
        final = build_observed_render_plan(graph, resolve_theme(None, graph))

    preflight_edges = tuple(
        ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        for route in preflight.routes
    )
    final_edges = tuple(
        ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        for route in final.plan.routes
    )
    assert final_edges == preflight_edges
    assert final.route_plan.bindings == preflight.plan.bindings
    assert all(plan.owns_geometry for plan in final.route_plan.convergence_plans)


@pytest.mark.parametrize(
    ("direction", "side", "consumer", "expected"),
    (
        ("LR", PortSide.TOP, (140.0, 200.0), 96.0),
        ("LR", PortSide.BOTTOM, (140.0, 200.0), 104.0),
        ("RL", PortSide.TOP, (60.0, 200.0), 104.0),
        ("RL", PortSide.BOTTOM, (60.0, 200.0), 96.0),
        ("TB", PortSide.LEFT, (100.0, 240.0), 204.0),
        ("TB", PortSide.RIGHT, (100.0, 240.0), 196.0),
        ("BT", PortSide.LEFT, (100.0, 160.0), 196.0),
        ("BT", PortSide.RIGHT, (100.0, 160.0), 204.0),
    ),
)
def test_perpendicular_convergence_lane_projection_transposes_by_axis(
    direction: str,
    side: PortSide,
    consumer: tuple[float, float],
    expected: float,
) -> None:
    entry = Station("entry", "", section_id="section", is_port=True, x=100.0, y=200.0)
    graph = MetroGraph(
        stations={
            "entry": entry,
            "consumer": Station(
                "consumer", "", section_id="section", x=consumer[0], y=consumer[1]
            ),
            "junction_a": Station("junction_a", "", x=20.0, y=20.0),
            "junction_b": Station("junction_b", "", x=40.0, y=40.0),
        },
        edges=[
            Edge("junction_a", "entry", "a"),
            Edge("junction_b", "entry", "b"),
            Edge("entry", "consumer", "a"),
            Edge("entry", "consumer", "b"),
        ],
        sections={
            "section": Section(
                "section",
                "Section",
                station_ids=["entry", "consumer"],
                entry_ports=["entry"],
                direction=direction,
            )
        },
        ports={
            "entry": Port("entry", "section", side, is_entry=True, x=100.0, y=200.0)
        },
        junctions=["junction_a", "junction_b"],
    )

    assert (
        projected_perp_entry_lane_coordinate(graph, "entry", "b", {("entry", "b"): 4.0})
        == expected
    )


def test_planned_convergence_owns_an_immutable_suppressed_continuation() -> None:
    path = ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_15.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    preflight = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
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

    edge = ResolvedEdge("__merge_9", "s6__entry_right_14", "l0")
    member_id = next(
        member.id for member in preflight.plan.members if member.edge == edge
    )
    immutable = next(
        binding for binding in preflight.plan.bindings if binding.member_id == member_id
    )
    emitted = next(
        binding for binding in final.plan.bindings if binding.member_id == member_id
    )
    immutable_membership = tuple(
        plan.member_ids for plan in preflight.plan.convergence_plans
    )
    final_membership = tuple(plan.member_ids for plan in final.plan.convergence_plans)

    assert immutable.kind in {
        BindingKind.MERGE_SKIP,
        BindingKind.COVERED_MERGE_HOP,
    }
    assert immutable.covering_member_id is not None
    assert emitted == immutable
    assert final_membership == immutable_membership
    assert all(plan.owns_geometry for plan in final.plan.convergence_plans)


def test_canvas_proofs_are_exact_finite_outward_and_idempotent() -> None:
    fixture_sides = (
        (
            ROOT / "examples" / "topologies" / "samerow_left_exit_far_left_entry.mmd",
            {CanvasSide.TOP, CanvasSide.LEFT},
        ),
        (
            ROOT
            / "examples"
            / "topologies"
            / "bottom_exit_stacked_right_entry_fan.mmd",
            {CanvasSide.RIGHT},
        ),
        (
            ROOT / "examples" / "topologies" / "bottom_entry_same_row_boundary.mmd",
            {CanvasSide.BOTTOM},
        ),
    )
    seen_sides: set[CanvasSide] = set()
    for path, expected_sides in fixture_sides:
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        plan = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        ).plan
        section_geometry = {
            section.id: (section.bbox_x, section.bbox_y)
            for section in graph.sections.values()
        }

        first = settle_route_envelopes(graph, plan)
        second = settle_route_envelopes(graph, plan)

        assert first == second
        assert first.translations == ()
        assert {
            section.id: (section.bbox_x, section.bbox_y)
            for section in graph.sections.values()
        } == section_geometry
        canvas_proofs = tuple(
            proof
            for proof in first.capacity_proofs
            if isinstance(proof.region, CanvasRegion)
        )
        assert {proof.region.side for proof in canvas_proofs} == expected_sides
        seen_sides.update(expected_sides)
        immutable_by_id = {item.id: item for item in plan.reservations}
        for proof in canvas_proofs:
            side = proof.region.side
            outward_sign = -1 if side in {CanvasSide.TOP, CanvasSide.LEFT} else 1
            assert proof.id.region == proof.region
            assert proof.id.axis is proof.axis
            assert proof.id.reservation_ids == tuple(
                item.reservation_id for item in proof.reservations
            )
            assert math.isfinite(proof.required_width)
            for allocation in proof.reservations:
                immutable = immutable_by_id[allocation.reservation_id]
                inner, _blockers = canvas_inner_boundary(graph, immutable)
                lane_claims = tuple(
                    rank for lane in allocation.lanes for rank in lane.claim_indices
                )
                assert sorted(lane_claims) == list(range(len(immutable.claims)))
                for lane in allocation.lanes:
                    assert lane.minimum_coordinate <= lane.coordinate
                    assert lane.coordinate <= lane.maximum_coordinate
                    assert math.isfinite(lane.coordinate)
                for claim in allocation.allocations:
                    assert math.isfinite(claim.coordinate)
                    assert outward_sign * (claim.coordinate - inner) >= 0
                    assert (
                        outward_sign * (claim.coordinate - claim.original_coordinate)
                        >= 0
                    )
    assert seen_sides == set(CanvasSide)


@pytest.mark.parametrize(
    ("fixture", "dx", "dy"),
    (
        ("bt_to_lr.mmd", 0.0, -600.0),
        ("dogleg_exempt_sameline.mmd", -600.0, 0.0),
    ),
    ids=("top", "left"),
)
def test_origin_shift_preserves_routing_identity_and_zero_viewbox(
    fixture: str, dx: float, dy: float
) -> None:
    path = ROOT / "examples" / "topologies" / fixture

    def prepared() -> MetroGraph:
        return prepare_graph(path.read_text(), source_dir=str(path.parent))

    baseline_graph = prepared()
    baseline = build_observed_render_plan(
        baseline_graph, resolve_theme(None, baseline_graph)
    )
    shifted_graph = prepared()
    for station in shifted_graph.stations.values():
        station.x += dx
        station.y += dy
    for section in shifted_graph.sections.values():
        section.bbox_x += dx
        section.bbox_y += dy
    for port in shifted_graph.ports.values():
        port.x += dx
        port.y += dy
    shifted_graph.bypass_label_obstacles = {
        station_id: (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        for station_id, (x0, y0, x1, y1) in (
            shifted_graph.bypass_label_obstacles.items()
        )
    }

    shifted = build_observed_render_plan(
        shifted_graph, resolve_theme(None, shifted_graph)
    )

    assert shifted.route_plan.bindings == baseline.route_plan.bindings
    assert [
        (member.id, member.edge, member.family_id, member.connector_ids)
        for member in shifted.route_plan.members
    ] == [
        (member.id, member.edge, member.family_id, member.connector_ids)
        for member in baseline.route_plan.members
    ]
    assert [
        (route.edge.source, route.edge.target, route.line_id)
        for route in shifted.plan.routes
    ] == [
        (route.edge.source, route.edge.target, route.line_id)
        for route in baseline.plan.routes
    ]
    assert min(x for route in shifted.plan.routes for x, _y in route.points) >= 0.0
    assert min(y for route in shifted.plan.routes for _x, y in route.points) >= 0.0
    root = ET.fromstring(emit_render_plan(shifted.plan))
    assert root.attrib["viewBox"].split()[:2] == ["0", "0"]


def test_canvas_proof_identity_and_identity_projection_have_single_ownership() -> None:
    path = ROOT / "examples" / "topologies" / "samerow_left_exit_far_left_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    settlement = settle_route_envelopes(graph, plan)
    proof = next(
        item
        for item in settlement.capacity_proofs
        if isinstance(item.region, CanvasRegion) and len(item.reservations) > 1
    )
    member_by_edge = {item.edge: item.id for item in plan.members}
    reversed_id = replace(
        proof.id,
        reservation_ids=tuple(reversed(proof.id.reservation_ids)),
    )

    with pytest.raises(
        EnvelopeAllocationError,
        match="inconsistent reservation membership",
    ):
        build_envelope_allocation_query(
            (replace(proof, id=reversed_id),),
            member_by_edge,
            plan.reservations,
            plan.bindings,
        )

    reservation = proof.reservations[0]
    projection = settlement_module.EnvelopeIdentityProjection(
        reservation.reservation_id,
        reservation.allocations,
    )
    with pytest.raises(
        EnvelopeAllocationError,
        match="identity projections disagree with immutable reservations",
    ):
        build_envelope_allocation_query(
            (proof,),
            member_by_edge,
            plan.reservations,
            plan.bindings,
            identity_projections=(projection,),
        )


def test_one_limited_reservation_suppresses_its_whole_system_proofs() -> None:
    path = ROOT / "examples" / "topologies" / "samerow_left_exit_far_left_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    reservation = next(
        item for item in plan.reservations if isinstance(item.region, CanvasRegion)
    )
    assert (
        sum(item.system_id == reservation.system_id for item in plan.reservations) > 1
    )
    limited = replace(
        plan,
        reservation_diagnostics=(
            *plan.reservation_diagnostics,
            RouteReservationDiagnostic(
                reservation.id,
                reservation.claimant_member_ids,
                "synthetic-capacity-limit",
                "one reservation in the system is allocation-limited",
                -2.0,
                -2.0,
                0.0,
            ),
        ),
    )

    proofs = settlement_module._capacity_proofs(graph, limited, plan)

    assert not any(reservation.system_id in proof.system_ids for proof in proofs)


def test_canvas_packer_separates_distinct_lines_with_coincident_preferences() -> None:
    path = ROOT / "examples" / "topologies" / "around_below_ep_col_gt0.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    item = settlement_module._canvas_reservations(plan, graph, plan)[CanvasSide.LEFT][0]
    members = {member.id: member for member in plan.members}
    assert (
        len({members[claim.member_id].line_id for claim in item.reservation.claims}) > 1
    )
    synthetic = replace(
        item,
        fixed=False,
        fixed_claims=tuple(False for _claim in item.reservation.claims),
        lane_coordinates=tuple(200.0 for _claim in item.reservation.claims),
    )

    packed = settlement_module._pack_component_claims(
        graph, (synthetic,), coordinate_sign=-1
    )

    assert isinstance(packed, tuple)
    first, second = packed[0]
    immutable_separation = abs(
        item.sharing_coordinates[1] - item.sharing_coordinates[0]
    )
    assert abs(second - first) >= immutable_separation
    assert all(math.isfinite(coordinate) for coordinate in packed[0])


def test_canvas_packer_exactly_shares_a_same_line_witnessed_channel() -> None:
    path = ROOT / "examples" / "topologies" / "samerow_left_exit_far_left_entry.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    plan = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    items = settlement_module._canvas_reservations(plan, graph, plan)[CanvasSide.TOP]
    members = {member.id: member for member in plan.members}
    witnessed = next(
        (first_rank, first_claim, second_rank, second_claim)
        for first_rank, first in enumerate(items)
        for second_rank, second in enumerate(items[first_rank + 1 :], first_rank + 1)
        for first_claim, first_key in enumerate(first.sharing_keys)
        for second_claim, second_key in enumerate(second.sharing_keys)
        if first_key.intersection(second_key)
        and first.sharing_coordinates[first_claim]
        == second.sharing_coordinates[second_claim]
        and members[first.reservation.claims[first_claim].member_id].line_id
        == members[second.reservation.claims[second_claim].member_id].line_id
    )
    first_rank, first_claim, second_rank, second_claim = witnessed
    component = tuple(
        replace(
            item,
            fixed=False,
            fixed_claims=tuple(False for _claim in item.reservation.claims),
            lane_coordinates=tuple(
                180.0 + 12.0 * rank for _claim in item.reservation.claims
            ),
        )
        for rank, item in enumerate((items[first_rank], items[second_rank]))
    )

    packed = settlement_module._pack_component_claims(
        graph, component, coordinate_sign=-1
    )

    assert isinstance(packed, tuple)
    assert packed[0][first_claim] == packed[1][second_claim]
