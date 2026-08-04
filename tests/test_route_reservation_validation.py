"""Route-plan queries reject contradictory reservation ledger records."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout import compute_layout
from nf_metro.layout.constants import SECTION_Y_GAP, graph_offset_step
from nf_metro.layout.envelope_settlement import (
    EnvelopeCapacityLimitation,
    EnvelopeCapacityOwnerKind,
)
from nf_metro.layout.route_plan import (
    CoordinateRegime,
    DemandAxis,
    EmittedPathId,
    ExitTurnPlanId,
    GridSpan,
    ReservationDecisionKind,
    RouteSystemId,
    SharedReferenceKind,
    build_route_plan_query,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    CanvasSide,
    ColumnGapRegion,
    CorridorKind,
    CorridorMeasurementScope,
    CorridorOrientation,
    RouteReservationId,
    RouteReservationLane,
    RowGapRegion,
    _adjacent_outer_turn_region_is_proven,
    _exact_pinned_exit_fan_owner,
    _materialise_immutable_reservations,
    _reservation_claim_witness,
    _reservation_semantic_witness,
    _validate_materialised_realisation,
    _validate_reemitted_reservation,
    apply_route_reservation_ledger,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.layout.routing.common import Direction, apply_route_offsets
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.parser import parse_metro_mermaid
from nf_metro.parser.route_topology import semantic_route_id
from nf_metro.render.svg import _settle_render_geometry

ROOT = Path(__file__).parents[1]
REPORT_HO = ROOT / "tests" / "fixtures" / "route_reservations" / "reportho.metro"
TOPOLOGIES = ROOT / "examples" / "topologies"
ORGANELLAR = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
SEED_77 = ROOT / "tests" / "fixtures" / "hash_seed_determinism" / "seed_77.mmd"
PERPENDICULAR_PORTS = (
    ROOT / "tests" / "fixtures" / "regressions" / "lr_perpendicular_ports_overflow.mmd"
)


def _plan(path: Path = REPORT_HO):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        return observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        ).plan


def _replace_record(records, replacement, *, id_field: str = "id"):
    replacement_id = getattr(replacement, id_field)
    return tuple(
        replacement if getattr(item, id_field) == replacement_id else item
        for item in records
    )


@pytest.mark.parametrize(
    ("record_kind", "message"),
    (
        ("reservation", "ordered exit-axis identities"),
        ("reference", "shared reference is inconsistent"),
        ("demand", "symbolic demand is inconsistent"),
    ),
)
def test_ordered_exit_axis_identities_reject_wrong_plan_rank_or_order(
    record_kind: str,
    message: str,
) -> None:
    plan = _plan(SEED_77)
    reservation = next(
        item
        for item in plan.reservations
        if sum(identity is not None for identity in item.exit_turn_axis_identities) >= 2
    )
    identities = reservation.exit_turn_axis_identities
    first_identity = identities[0]
    assert first_identity is not None
    if record_kind == "reservation":
        malformed_identities = (
            replace(first_identity, plan_id=ExitTurnPlanId("wrong-exit-turn-plan")),
            *identities[1:],
        )
        malformed = replace(
            plan,
            reservations=_replace_record(
                plan.reservations,
                replace(
                    reservation,
                    exit_turn_axis_identities=malformed_identities,
                ),
            ),
        )
    elif record_kind == "reference":
        reference = next(
            item
            for item in plan.shared_references
            if item.id == reservation.reference_id
        )
        malformed_identities = (
            replace(first_identity, axis_rank=99),
            *identities[1:],
        )
        malformed = replace(
            plan,
            shared_references=_replace_record(
                plan.shared_references,
                replace(
                    reference,
                    exit_turn_axis_identities=malformed_identities,
                ),
            ),
        )
    else:
        demand = next(
            item for item in plan.demands if item.id in reservation.demand_ids
        )
        malformed_identities = tuple(reversed(identities))
        malformed = replace(
            plan,
            demands=_replace_record(
                plan.demands,
                replace(
                    demand,
                    exit_turn_axis_identities=malformed_identities,
                ),
            ),
        )

    with pytest.raises(ValueError, match=message):
        build_route_plan_query(malformed)


def _settled(path: Path):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    theme = resolve_theme(None, graph)
    station_offsets, routes, _labels, final, ledger, _limitations = (
        _settle_render_geometry(
            graph,
            theme,
            graph_offset_step(graph, theme.line_width),
            graph.section_y_gap or SECTION_Y_GAP,
        )
    )
    assert ledger is not None
    return graph, station_offsets, routes, final, ledger


def _organellar_settled():
    graph = prepare_graph(ORGANELLAR.read_text(), source_dir=str(ORGANELLAR.parent))
    theme = resolve_theme(None, graph)
    station_offsets, routes, _labels, final, ledger, limitations = (
        _settle_render_geometry(
            graph,
            theme,
            graph_offset_step(graph, theme.line_width),
            graph.section_y_gap or SECTION_Y_GAP,
        )
    )
    assert ledger is not None
    return graph, station_offsets, routes, final, ledger, limitations


def _limited_settled(path: Path):
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    theme = resolve_theme(None, graph)
    station_offsets, routes, _labels, final, ledger, limitations = (
        _settle_render_geometry(
            graph,
            theme,
            graph_offset_step(graph, theme.line_width),
            graph.section_y_gap or SECTION_Y_GAP,
        )
    )
    assert ledger is not None
    assert len(limitations) == 1
    return graph, station_offsets, routes, final, ledger, limitations[0]


def test_endpoint_transition_keeps_exact_symbolic_membership_after_allocation() -> None:
    graph = parse_metro_mermaid(PERPENDICULAR_PORTS.read_text())
    graph.center_ports = True
    compute_layout(graph)
    theme = resolve_theme(None, graph)
    _offsets, _routes, _labels, final, ledger, limitations = _settle_render_geometry(
        graph,
        theme,
        graph_offset_step(graph, theme.line_width),
        graph.section_y_gap or SECTION_Y_GAP,
    )
    assert ledger is not None
    system_id = next(
        system.id
        for system in ledger.systems
        if any("snpeff" in str(connector_id) for connector_id in system.connector_ids)
    )

    def ids(records):
        return tuple(item.id for item in records if item.system_id == system_id)

    assert ids(final.reservations) == ids(ledger.reservations)
    assert ids(final.demands) == ids(ledger.demands)
    assert ids(final.shared_references) == ids(ledger.shared_references)
    assert {
        reservation.orientation
        for reservation in ledger.reservations
        if reservation.system_id == system_id
    } == {CorridorOrientation.HORIZONTAL, CorridorOrientation.VERTICAL}
    assert limitations == ()


def test_endpoint_transition_rejects_clearance_on_the_section_side() -> None:
    graph = parse_metro_mermaid(PERPENDICULAR_PORTS.read_text())
    graph.center_ports = True
    compute_layout(graph)
    plan = observe_route_edges(
        graph,
        station_offsets=compute_station_offsets(graph),
    ).plan
    reservation = next(
        item
        for item in plan.reservations
        if item.orientation is CorridorOrientation.HORIZONTAL
        and item.region == RowGapRegion(0, 1)
        and item.claims[0].endpoint_anchor_ids
    )
    assert reservation.negative_side_clearance > 0.0
    assert reservation.positive_side_clearance == 0.0
    malformed = replace(
        reservation,
        negative_side_clearance=0.0,
        positive_side_clearance=reservation.negative_side_clearance,
    )

    with pytest.raises(ValueError, match="clearance policy is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                reservations=_replace_record(plan.reservations, malformed),
            )
        )

    malformed_claim = replace(
        reservation.claims[0],
        endpoint_anchor_ids=("not-a-member-endpoint",),
    )
    with pytest.raises(ValueError, match="unknown endpoint anchor"):
        build_route_plan_query(
            replace(
                plan,
                reservations=_replace_record(
                    plan.reservations,
                    replace(reservation, claims=(malformed_claim,)),
                ),
            )
        )


@pytest.fixture(scope="module")
def exact_convergence_reservations():
    path = TOPOLOGIES / "exit_run_three_drop_columns.mmd"
    graph, station_offsets, routes, final, ledger = _settled(path)
    final_by_witness = {
        _reservation_semantic_witness(item): item for item in final.reservations
    }
    immutable = next(item for item in ledger.reservations if len(item.claims) > 1)
    reemitted = final_by_witness[_reservation_semantic_witness(immutable)]
    assert {_reservation_claim_witness(item) for item in reemitted.claims} == {
        _reservation_claim_witness(item) for item in immutable.claims
    }
    return graph, station_offsets, routes, final, ledger, reemitted, immutable


@pytest.mark.parametrize("field", ("path_id", "path_rank"))
def test_query_rejects_frozen_claim_paths_inconsistent_with_binding(field: str) -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    claim = reservation.claims[0]
    value = "wrong-path" if field == "path_id" else claim.path_rank + 1
    malformed_claim = replace(claim, **{field: value})
    malformed = replace(reservation, claims=(malformed_claim, *reservation.claims[1:]))

    with pytest.raises(ValueError, match="disagrees with its emitted binding"):
        build_route_plan_query(
            replace(plan, reservations=_replace_record(plan.reservations, malformed))
        )


def test_delayed_ledger_graft_preserves_symbols_and_uses_final_projection() -> None:
    path = TOPOLOGIES / "complex_multipath.mmd"
    graph, station_offsets, routes, final, ledger = _settled(path)
    reservation, claim, points, longitudinal_rank = next(
        (reservation, claim, points, longitudinal_rank)
        for reservation in ledger.reservations
        for claim in reservation.claims
        for points in (apply_route_offsets(routes[claim.path_rank], station_offsets),)
        for longitudinal_rank in (
            0 if reservation.orientation is CorridorOrientation.HORIZONTAL else 1,
        )
        if not (
            min(
                points[claim.segment_rank][longitudinal_rank],
                points[claim.segment_end_rank + 1][longitudinal_rank],
            )
            == pytest.approx(claim.longitudinal_start)
            and max(
                points[claim.segment_rank][longitudinal_rank],
                points[claim.segment_end_rank + 1][longitudinal_rank],
            )
            == pytest.approx(claim.longitudinal_end)
        )
    )

    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
    )
    query = build_route_plan_query(combined)

    assert combined.reservations is ledger.reservations
    assert combined.bindings == ledger.bindings
    assert combined.reservations == ledger.reservations
    assert tuple(
        (
            item.lane_count,
            item.lanes,
            tuple(_reservation_claim_witness(claim) for claim in item.claims),
        )
        for item in combined.reservations
    ) == tuple(
        (
            item.lane_count,
            item.lanes,
            tuple(_reservation_claim_witness(claim) for claim in item.claims),
        )
        for item in ledger.reservations
    )
    realised = query.realised_reservation(reservation.id)
    assert realised.longitudinal_start == pytest.approx(
        min(
            points[claim.segment_rank][longitudinal_rank],
            points[claim.segment_end_rank + 1][longitudinal_rank],
        )
    )
    assert realised.longitudinal_end == pytest.approx(
        max(
            points[claim.segment_rank][longitudinal_rank],
            points[claim.segment_end_rank + 1][longitudinal_rank],
        )
    )
    corridor_reference_ids = {item.reference_id for item in ledger.reservations}
    ledger_references = {
        item.id: item
        for item in ledger.shared_references
        if item.id in corridor_reference_ids
    }
    assert all(
        next(item for item in combined.shared_references if item.id == reference_id)
        is reference
        for reference_id, reference in ledger_references.items()
    )


def test_organellar_exits_compatibility_with_exact_immutable_membership() -> None:
    graph, station_offsets, routes, final, ledger, limitations = _organellar_settled()

    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
        compatibility_limitations=limitations,
    )

    assert limitations == ()
    assert combined.reservations is ledger.reservations
    assert tuple(
        tuple(_reservation_claim_witness(claim) for claim in reservation.claims)
        for reservation in combined.reservations
    ) == tuple(
        tuple(_reservation_claim_witness(claim) for claim in reservation.claims)
        for reservation in ledger.reservations
    )


def _obsolete_compatibility_limitation(graph, plan, ledger):
    system_ids_with_convergence = {item.system_id for item in plan.convergence_plans}
    reservation = next(
        item
        for item in ledger.reservations
        if item.system_id in system_ids_with_convergence
    )
    pinned_section_id = next(
        section_id
        for section_id in graph.sections
        if (decision := graph.layout_provenance.grid_decision(section_id)) is not None
        and decision.is_reinference_locked
    )
    return EnvelopeCapacityLimitation(
        reservation.system_id,
        (reservation.id,),
        (f"section-bottom:{pinned_section_id}",),
        (pinned_section_id,),
    )


def test_delayed_ledger_graft_rejects_an_obsolete_compatibility_owner() -> None:
    graph, station_offsets, routes, final, ledger, _limitations = _organellar_settled()
    limitation = _obsolete_compatibility_limitation(graph, final, ledger)

    with pytest.raises(ValueError, match="not the final convergence owner"):
        apply_route_reservation_ledger(
            final,
            ledger,
            graph,
            routes,
            station_offsets,
            canvas_width=2000.0,
            canvas_height=2000.0,
            compatibility_limitations=(limitation,),
        )


def test_pinned_convergence_limitation_is_the_exact_final_owner() -> None:
    path = TOPOLOGIES / "merge_pullaway.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    theme = resolve_theme(None, graph)
    station_offsets, routes, _labels, final, ledger, limitations = (
        _settle_render_geometry(
            graph,
            theme,
            graph_offset_step(graph, theme.line_width),
            graph.section_y_gap or SECTION_Y_GAP,
        )
    )
    assert ledger is not None
    assert len(limitations) == 1
    limitation = limitations[0]
    owners = tuple(
        item
        for item in final.convergence_plans
        if item.system_id == limitation.system_id
    )

    assert limitation.owner_kind is EnvelopeCapacityOwnerKind.CONVERGENCE
    assert limitation.owner_issue == 1658
    assert limitation.owner_plan_ids == tuple(item.id for item in owners)
    assert owners
    assert all(not item.owns_geometry for item in owners)
    assert all(
        item.legacy_reason and "owner #1658" in item.legacy_reason for item in owners
    )
    apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
        compatibility_limitations=limitations,
    )


def test_pinned_exit_fan_limitation_has_an_exact_final_system_owner() -> None:
    graph, station_offsets, routes, final, ledger, limitation = _limited_settled(
        TOPOLOGIES / "multicarrier_offrow_exit_climb.mmd"
    )

    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
        compatibility_limitations=(limitation,),
    )

    assert limitation.owner_issue == 1658
    assert limitation.owner_kind is EnvelopeCapacityOwnerKind.PINNED_EXIT_FAN
    assert limitation.owner_plan_ids
    assert _exact_pinned_exit_fan_owner(final, ledger, limitation)
    assert combined.reservations == ledger.reservations
    assert combined.shared_references == ledger.shared_references
    assert combined.demands == ledger.demands


def test_pinned_exit_fan_owner_rejects_changed_classification_or_plan_ids() -> None:
    graph, station_offsets, routes, final, ledger, limitation = _limited_settled(
        TOPOLOGIES / "multicarrier_offrow_exit_climb.mmd"
    )

    for malformed in (
        replace(limitation, owner_kind=EnvelopeCapacityOwnerKind.CONVERGENCE),
        replace(limitation, owner_plan_ids=limitation.owner_plan_ids[:-1]),
    ):
        with pytest.raises(ValueError, match="capacity limitation is not"):
            apply_route_reservation_ledger(
                final,
                ledger,
                graph,
                routes,
                station_offsets,
                canvas_width=2000.0,
                canvas_height=2000.0,
                compatibility_limitations=(malformed,),
            )


def test_pinned_exit_fan_owner_rejects_changed_system_membership() -> None:
    _graph, _offsets, _routes, final, ledger, limitation = _limited_settled(
        TOPOLOGIES / "multicarrier_offrow_exit_climb.mmd"
    )
    system = next(item for item in final.systems if item.id == limitation.system_id)
    malformed_system = replace(system, member_ids=system.member_ids[:-1])
    malformed = replace(
        final,
        systems=_replace_record(final.systems, malformed_system),
    )

    assert not _exact_pinned_exit_fan_owner(malformed, ledger, limitation)


@pytest.mark.parametrize("resource_name", ("shared_references", "demands"))
def test_pinned_exit_fan_owner_rejects_changed_reservation_resources(
    resource_name: str,
) -> None:
    _graph, _offsets, _routes, final, ledger, limitation = _limited_settled(
        TOPOLOGIES / "multicarrier_offrow_exit_climb.mmd"
    )
    system = next(item for item in final.systems if item.id == limitation.system_id)
    resource_ids = (
        system.shared_reference_ids
        if resource_name == "shared_references"
        else system.demand_ids
    )
    resources = getattr(final, resource_name)
    resource = next(item for item in resources if item.id in resource_ids)
    malformed_resource = replace(
        resource, system_id=RouteSystemId("changed-resource-system")
    )
    malformed = replace(
        final,
        **{resource_name: _replace_record(resources, malformed_resource)},
    )

    assert not _exact_pinned_exit_fan_owner(malformed, ledger, limitation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("owner_issue", 9999, "unexpected owner"),
        ("system_id", "unlisted-route-system", "unknown route system"),
        (
            "reservation_ids",
            (RouteReservationId("unlisted-reservation"),),
            "reservation ownership is inconsistent",
        ),
        ("blocker_ids", (), "evidence is incomplete"),
        ("pinned_section_ids", ("unlisted-section",), "pin evidence"),
    ),
)
def test_delayed_ledger_graft_rejects_mismatched_capacity_limitation(
    field: str,
    value: object,
    message: str,
) -> None:
    graph, station_offsets, routes, final, ledger, _limitations = _organellar_settled()
    limitation = _obsolete_compatibility_limitation(graph, final, ledger)
    malformed = (replace(limitation, **{field: value}),)

    with pytest.raises(ValueError, match=message):
        apply_route_reservation_ledger(
            final,
            ledger,
            graph,
            routes,
            station_offsets,
            canvas_width=2000.0,
            canvas_height=2000.0,
            compatibility_limitations=malformed,
        )


def test_delayed_ledger_graft_rejects_an_unconsumed_immutable_segment() -> None:
    path = TOPOLOGIES / "complex_multipath.mmd"
    graph, station_offsets, routes, final, ledger = _settled(path)
    claim = next(
        claim
        for reservation in ledger.reservations
        for claim in reservation.claims
        if claim.segment_rank > 0
    )
    route = routes[claim.path_rank]
    route.envelope_allocated_segments = tuple(
        item
        for item in route.envelope_allocated_segments
        if item[0] != claim.segment_rank
    )

    with pytest.raises(ValueError, match="did not consume an immutable"):
        apply_route_reservation_ledger(
            final,
            ledger,
            graph,
            routes,
            station_offsets,
            canvas_width=2000.0,
            canvas_height=2000.0,
        )


def test_delayed_ledger_graft_rejects_a_shrunk_lane_envelope() -> None:
    path = TOPOLOGIES / "complex_multipath.mmd"
    graph, station_offsets, routes, final, ledger = _settled(path)
    reservation = next(
        item
        for item in ledger.reservations
        if item.bundle_width > 0 and len(item.claims) > 1
    )
    allocated_coordinates = {
        claim: next(
            coordinate
            for rank, coordinate_rank, coordinate in routes[
                claim.path_rank
            ].envelope_allocated_segments
            if rank == claim.segment_rank
            and coordinate_rank
            == (1 if reservation.orientation is CorridorOrientation.HORIZONTAL else 0)
        )
        for claim in reservation.claims
    }
    maximum = max(allocated_coordinates.values())
    replacement_coordinate = min(allocated_coordinates.values())
    claim = next(
        item
        for item, coordinate in allocated_coordinates.items()
        if coordinate == maximum
    )
    route = routes[claim.path_rank]
    allocation_rank = (
        1 if reservation.orientation is CorridorOrientation.HORIZONTAL else 0
    )
    for point_rank in range(claim.segment_rank, claim.segment_end_rank + 2):
        point = list(route.points[point_rank])
        point[allocation_rank] = replacement_coordinate
        route.points[point_rank] = (point[0], point[1])
    route.envelope_allocated_segments = tuple(
        (rank, coordinate_rank, replacement_coordinate)
        if claim.segment_rank <= rank <= claim.segment_end_rank
        and coordinate_rank == allocation_rank
        else (rank, coordinate_rank, coordinate)
        for rank, coordinate_rank, coordinate in route.envelope_allocated_segments
    )

    with pytest.raises(ValueError, match="monotone occupied projection"):
        apply_route_reservation_ledger(
            final,
            ledger,
            graph,
            routes,
            station_offsets,
            canvas_width=2000.0,
            canvas_height=2000.0,
        )


def test_materialised_realisation_rejects_inconsistent_capacity_evidence() -> None:
    path = TOPOLOGIES / "complex_multipath.mmd"
    graph, station_offsets, routes, final, ledger = _settled(path)
    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
    )
    reservation = combined.reservations[0]
    realised = build_route_plan_query(combined).realised_reservation(reservation.id)

    with pytest.raises(ValueError, match="capacity evidence"):
        _validate_materialised_realisation(
            reservation,
            replace(realised, capacity_slack=realised.capacity_slack + 10.0),
        )


@pytest.mark.parametrize(
    "path",
    (ROOT / "examples" / "longread_variant_calling.mmd",),
    ids=lambda path: path.stem,
)
def test_delayed_ledger_graft_disambiguates_semantic_peers_by_direct_claims(
    path: Path,
) -> None:
    graph, station_offsets, routes, final, ledger = _settled(path)
    semantic_counts: dict[tuple[object, ...], int] = {}
    for reservation in ledger.reservations:
        witness = _reservation_semantic_witness(reservation)
        semantic_counts[witness] = semantic_counts.get(witness, 0) + 1
    assert any(count > 1 for count in semantic_counts.values())

    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
    )

    assert combined.reservations is ledger.reservations
    assert {item.reservation_id for item in combined.realised_reservations} == {
        item.id for item in ledger.reservations
    }


def test_delayed_ledger_graft_rejects_changed_claim_path_identity() -> None:
    graph = prepare_graph(REPORT_HO.read_text(), source_dir=str(REPORT_HO.parent))
    ledger = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    ).plan
    member_id = ledger.reservations[0].claims[0].member_id
    path_rank = max(item.path_rank or 0 for item in ledger.bindings) + 1
    path_id = EmittedPathId(semantic_route_id("emitted-path", member_id, path_rank))
    bindings = tuple(
        replace(item, path_id=path_id, path_rank=path_rank)
        if item.member_id == member_id
        else item
        for item in ledger.bindings
    )
    reservations = tuple(
        replace(
            item,
            claims=tuple(
                replace(claim, path_id=path_id, path_rank=path_rank)
                if claim.member_id == member_id
                else claim
                for claim in item.claims
            ),
        )
        if any(claim.member_id == member_id for claim in item.claims)
        else item
        for item in ledger.reservations
    )
    reemitted = replace(ledger, bindings=bindings, reservations=reservations)
    build_route_plan_query(reemitted)

    with pytest.raises(ValueError, match="changed immutable claim path identity"):
        _validate_reemitted_reservation(
            reemitted,
            ledger.reservations[0],
            reemitted.reservations[0],
        )


def test_delayed_ledger_graft_accepts_exact_reemission(
    exact_convergence_reservations,
) -> None:
    (
        graph,
        station_offsets,
        routes,
        final,
        ledger,
        reemitted,
        immutable,
    ) = exact_convergence_reservations

    combined = apply_route_reservation_ledger(
        final,
        ledger,
        graph,
        routes,
        station_offsets,
        canvas_width=2000.0,
        canvas_height=2000.0,
    )
    query = build_route_plan_query(combined)

    assert combined.reservations is ledger.reservations
    realised = query.realised_reservation(immutable.id)
    assert realised is not None
    assert {_reservation_claim_witness(item) for item in reemitted.claims} == {
        _reservation_claim_witness(item) for item in immutable.claims
    }


def _synthetic_owned_corridor_extension(exact_convergence_reservations):
    (
        station_offsets,
        routes,
        final,
        ledger,
        *_unused,
    ) = exact_convergence_reservations[1:]

    def claimed_without_donor(donor):
        return {
            (claim.member_id, claim.path_rank, rank)
            for reservation in ledger.reservations
            if reservation is not donor
            for claim in reservation.claims
            for rank in range(claim.segment_rank, claim.segment_end_rank + 1)
        }

    owner, donor, extra = next(
        (owner, donor, extra)
        for owner in ledger.reservations
        for donor in ledger.reservations
        if owner is not donor
        and owner.orientation is CorridorOrientation.HORIZONTAL
        and donor.orientation is CorridorOrientation.HORIZONTAL
        and donor.system_id == owner.system_id
        and donor.span.min_column <= owner.span.min_column
        and owner.span.max_column <= donor.span.max_column
        and donor.span.min_row <= owner.span.min_row
        and owner.span.max_row <= donor.span.max_row
        and donor.span != owner.span
        for extra in donor.claims
        if extra.segment_rank > 0
        and (
            extra.member_id,
            extra.path_rank,
            extra.segment_rank - 1,
        )
        in claimed_without_donor(donor)
        and (
            extra.member_id,
            extra.path_rank,
            extra.segment_end_rank + 1,
        )
        in claimed_without_donor(donor)
    )
    bracket_ranks = {extra.segment_rank - 1, extra.segment_end_rank + 1}
    brackets = tuple(
        reservation
        for reservation in ledger.reservations
        if reservation is not donor
        and any(
            claim.member_id == extra.member_id
            and claim.path_rank == extra.path_rank
            and claim.segment_rank in bracket_ranks
            for claim in reservation.claims
        )
    )
    synthetic_ledger = replace(ledger, reservations=(owner, *brackets))
    connector_ids = tuple(dict.fromkeys((*owner.connector_ids, *donor.connector_ids)))
    reemitted = replace(
        owner,
        connector_ids=connector_ids,
        claims=(*owner.claims, extra),
        span=donor.span,
        lanes=(RouteReservationLane((0, 1)),),
        lane_count=1,
    )
    synthetic_final = replace(final, reservations=(reemitted,))
    return station_offsets, routes, synthetic_final, synthetic_ledger, owner, extra


def test_delayed_ledger_graft_accepts_bounded_convergence_corridor_extension(
    exact_convergence_reservations,
) -> None:
    station_offsets, routes, final, ledger, immutable, extra = (
        _synthetic_owned_corridor_extension(exact_convergence_reservations)
    )

    materialised = _materialise_immutable_reservations(
        final,
        ledger,
        routes,
        station_offsets,
    )
    grafted = next(item for item in materialised if item.id == immutable.id)

    assert {_reservation_claim_witness(claim) for claim in grafted.claims} == {
        *(_reservation_claim_witness(claim) for claim in immutable.claims),
        _reservation_claim_witness(extra),
    }


def test_delayed_ledger_graft_rejects_unowned_corridor_extension(
    exact_convergence_reservations,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station_offsets, routes, final, ledger, _immutable, extra = (
        _synthetic_owned_corridor_extension(exact_convergence_reservations)
    )
    monkeypatch.setattr(routes[extra.path_rank], "convergence_plan_id", None)

    with pytest.raises(ValueError, match="unattributed additional direct claim"):
        _materialise_immutable_reservations(
            final,
            ledger,
            routes,
            station_offsets,
        )


def test_delayed_ledger_graft_rejects_unowned_additional_claim(
    exact_convergence_reservations,
) -> None:
    *_context, final, _ledger, reemitted, immutable = exact_convergence_reservations
    unowned = replace(final, convergence_plans=())
    extra = replace(
        reemitted.claims[-1],
        segment_rank=reemitted.claims[-1].segment_end_rank + 1,
        segment_end_rank=reemitted.claims[-1].segment_end_rank + 1,
    )
    fabricated_claims = (*reemitted.claims, extra)
    fabricated = replace(
        reemitted,
        claims=fabricated_claims,
        lanes=(RouteReservationLane(tuple(range(len(fabricated_claims)))),),
        lane_count=1,
    )

    with pytest.raises(ValueError, match="unowned additional direct claim"):
        _validate_reemitted_reservation(unowned, immutable, fabricated)


def test_delayed_ledger_graft_rejects_a_missing_immutable_claim(
    exact_convergence_reservations,
) -> None:
    *_context, final, _ledger, reemitted, immutable = exact_convergence_reservations
    missing = replace(
        reemitted,
        claims=reemitted.claims[1:],
        lanes=(RouteReservationLane(tuple(range(len(reemitted.claims) - 1))),),
        lane_count=1,
    )

    with pytest.raises(ValueError, match="lost or changed direct claims"):
        _validate_reemitted_reservation(final, immutable, missing)


def test_delayed_ledger_graft_rejects_changed_immutable_claim_membership(
    exact_convergence_reservations,
) -> None:
    *_context, final, _ledger, reemitted, immutable = exact_convergence_reservations
    immutable_claims = {_reservation_claim_witness(item) for item in immutable.claims}
    changed = replace(
        reemitted,
        claims=tuple(
            replace(
                item,
                segment_rank=item.segment_rank + 1,
                segment_end_rank=item.segment_end_rank + 1,
            )
            if _reservation_claim_witness(item) in immutable_claims
            else item
            for item in reemitted.claims
        ),
    )

    with pytest.raises(ValueError, match="lost or changed direct claims"):
        _validate_reemitted_reservation(final, immutable, changed)


def test_query_rejects_loss_of_a_covered_reservation_claimant() -> None:
    plan = _plan(TOPOLOGIES / "merge_right_entry.mmd")
    reservation = next(
        item
        for item in plan.reservations
        if len(item.claimant_member_ids)
        > len({claim.member_id for claim in item.claims})
    )
    direct_claimants = tuple(
        member.id
        for member in plan.members
        if member.id in {claim.member_id for claim in reservation.claims}
    )
    malformed = replace(reservation, claimant_member_ids=direct_claimants)

    with pytest.raises(ValueError, match="claimant list disagrees"):
        build_route_plan_query(
            replace(plan, reservations=_replace_record(plan.reservations, malformed))
        )


def test_query_rejects_incomplete_connector_attribution() -> None:
    plan = _plan()
    reservation = next(
        item for item in plan.reservations if len(item.connector_ids) > 1
    )
    malformed = replace(reservation, connector_ids=reservation.connector_ids[:-1])

    with pytest.raises(ValueError, match="connector attribution is incomplete"):
        build_route_plan_query(
            replace(plan, reservations=_replace_record(plan.reservations, malformed))
        )


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        (
            "axis",
            lambda demand: (
                DemandAxis.X if demand.axis is DemandAxis.Y else DemandAxis.Y
            ),
        ),
        ("minimum_size", lambda demand: demand.minimum_size + 1.0),
        ("minimum_size_regime", lambda _demand: CoordinateRegime.SETTLED_GRID),
        ("keep_out_classes", lambda demand: tuple(reversed(demand.keep_out_classes))),
        ("provenance", lambda _demand: ()),
    ),
)
def test_query_rejects_inconsistent_symbolic_demands(field, mutate) -> None:
    plan = _plan()
    demand_id = plan.reservations[0].demand_ids[0]
    demand = next(item for item in plan.demands if item.id == demand_id)
    malformed = replace(demand, **{field: mutate(demand)})

    with pytest.raises(ValueError, match="symbolic demand is inconsistent"):
        build_route_plan_query(
            replace(plan, demands=_replace_record(plan.demands, malformed))
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", SharedReferenceKind.TRUNK),
        ("coordinate_regime", CoordinateRegime.LAYOUT_CANVAS),
        ("provenance", ()),
    ),
)
def test_query_rejects_inconsistent_shared_references(field, value) -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    reference = next(
        item for item in plan.shared_references if item.id == reservation.reference_id
    )
    malformed = replace(reference, **{field: value})

    with pytest.raises(ValueError, match="shared reference is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                shared_references=_replace_record(plan.shared_references, malformed),
            )
        )


def test_query_rejects_consistently_fabricated_reservation_provenance() -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    reference = next(
        item for item in plan.shared_references if item.id == reservation.reference_id
    )
    demand = next(item for item in plan.demands if item.id in reservation.demand_ids)
    malformed_reservation = replace(reservation, provenance=())
    malformed_reference = replace(reference, provenance=())
    malformed_demand = replace(demand, provenance=())

    with pytest.raises(
        ValueError, match="provenance is inconsistent with the route plan"
    ):
        build_route_plan_query(
            replace(
                plan,
                reservations=_replace_record(plan.reservations, malformed_reservation),
                shared_references=_replace_record(
                    plan.shared_references, malformed_reference
                ),
                demands=_replace_record(plan.demands, malformed_demand),
            )
        )


def test_query_rejects_a_consistently_fabricated_complete_span() -> None:
    plan = _plan()
    reservation = next(
        item
        for item in plan.reservations
        if item.span.min_column < item.span.max_column
    )
    fabricated_span = GridSpan(
        reservation.span.min_column,
        reservation.span.min_column,
        reservation.span.min_row,
        reservation.span.min_row,
    )
    included_sections = {
        item.section_id
        for item in plan.provenance.sections
        if item.grid is not None
        and fabricated_span.min_column <= item.grid.value[0] + item.grid.value[3] - 1
        and item.grid.value[0] <= fabricated_span.max_column
        and fabricated_span.min_row <= item.grid.value[1] + item.grid.value[2] - 1
        and item.grid.value[1] <= fabricated_span.max_row
    }
    fabricated_provenance = tuple(
        item
        for item in reservation.provenance
        if item.kind
        not in {
            ReservationDecisionKind.SECTION_GRID,
            ReservationDecisionKind.SECTION_DIRECTION,
        }
        or item.subject_id in included_sections
    )
    reference = next(
        item for item in plan.shared_references if item.id == reservation.reference_id
    )
    demand = next(item for item in plan.demands if item.id in reservation.demand_ids)
    malformed_reservation = replace(
        reservation, span=fabricated_span, provenance=fabricated_provenance
    )
    malformed_reference = replace(reference, provenance=fabricated_provenance)
    malformed_demand = replace(
        demand, span=fabricated_span, provenance=fabricated_provenance
    )

    with pytest.raises(ValueError, match="span is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                reservations=_replace_record(plan.reservations, malformed_reservation),
                shared_references=_replace_record(
                    plan.shared_references, malformed_reference
                ),
                demands=_replace_record(plan.demands, malformed_demand),
            )
        )


def test_query_rejects_a_consistently_fabricated_clearance_policy() -> None:
    plan = _plan()
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    reservation = next(item for item in plan.reservations if item.id in realised_by_id)
    demand = next(item for item in plan.demands if item.id in reservation.demand_ids)
    realised = realised_by_id[reservation.id]
    keepouts = tuple(reversed(reservation.keep_out_classes))
    malformed_reservation = replace(
        reservation,
        negative_side_clearance=reservation.negative_side_clearance + 10.0,
        minimum_width=reservation.minimum_width + 10.0,
        keep_out_classes=keepouts,
    )
    malformed_demand = replace(
        demand,
        minimum_size=demand.minimum_size + 10.0,
        keep_out_classes=keepouts,
    )
    malformed_realised = replace(
        realised,
        required_width=realised.required_width + 10.0,
        capacity_slack=realised.capacity_slack - 10.0,
        negative_side_slack=realised.negative_side_slack - 10.0,
    )

    with pytest.raises(ValueError, match="clearance policy is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                reservations=_replace_record(plan.reservations, malformed_reservation),
                demands=_replace_record(plan.demands, malformed_demand),
                realised_reservations=_replace_record(
                    plan.realised_reservations,
                    malformed_realised,
                    id_field="reservation_id",
                ),
            )
        )


def test_query_rejects_fabricated_route_family_attribution() -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    fabricated = next(
        family for family in RouteFamilyId if family not in reservation.route_family_ids
    )
    malformed = replace(reservation, route_family_ids=(fabricated,))

    with pytest.raises(ValueError, match="route-family attribution is inconsistent"):
        build_route_plan_query(
            replace(plan, reservations=_replace_record(plan.reservations, malformed))
        )


def test_query_rejects_a_noncanonical_reservation_identity() -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    malformed = replace(
        reservation, id=RouteReservationId("route-reservation:fabricated")
    )

    with pytest.raises(ValueError, match="canonical identity is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                reservations=tuple(
                    malformed if item.id == reservation.id else item
                    for item in plan.reservations
                ),
            )
        )


def test_topology_reservation_rejects_a_gap_outside_connector_crossings() -> None:
    plan = _plan()
    reservation = next(
        item
        for item in plan.reservations
        if item.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
    )
    if reservation.orientation is CorridorOrientation.HORIZONTAL:
        region = RowGapRegion(
            reservation.span.max_row + 10, reservation.span.max_row + 11
        )
    else:
        region = ColumnGapRegion(
            reservation.span.max_column + 10,
            reservation.span.max_column + 11,
        )
    malformed = replace(reservation, region=region)

    with pytest.raises(ValueError, match="region is not crossed"):
        build_route_plan_query(
            replace(plan, reservations=_replace_record(plan.reservations, malformed))
        )


def test_adjacent_outer_turn_proof_is_axis_generic() -> None:
    plan = _plan(ORGANELLAR)
    members = {item.id: item for item in plan.members}
    reservation = next(
        item
        for item in plan.reservations
        if item.kind is CorridorKind.BYPASS_BAND
        and item.route_family_ids == (RouteFamilyId.MERGE_TRUNK,)
        and isinstance(item.region, RowGapRegion)
        and item.region.upper_row == item.span.max_row
    )

    assert _adjacent_outer_turn_region_is_proven(reservation, members)
    vertical = replace(
        reservation,
        kind=CorridorKind.INTER_COLUMN_CHANNEL,
        orientation=CorridorOrientation.VERTICAL,
        direction=Direction.D,
        region=ColumnGapRegion(
            reservation.span.max_column,
            reservation.span.max_column + 1,
        ),
    )
    assert _adjacent_outer_turn_region_is_proven(vertical, members)
    assert not _adjacent_outer_turn_region_is_proven(
        replace(
            vertical,
            region=ColumnGapRegion(
                reservation.span.max_column + 1,
                reservation.span.max_column + 2,
            ),
        ),
        members,
    )


def test_reservation_rejects_direction_on_the_wrong_axis() -> None:
    plan = _plan()
    reservation = plan.reservations[0]
    direction = (
        Direction.D
        if reservation.orientation is CorridorOrientation.HORIZONTAL
        else Direction.R
    )

    with pytest.raises(ValueError, match="direction and orientation disagree"):
        replace(reservation, direction=direction)


def test_canvas_reservation_kind_follows_its_side() -> None:
    plan = _plan()
    reservation = next(
        item
        for item in plan.reservations
        if isinstance(item.region, CanvasRegion)
        and item.region.side in {CanvasSide.TOP, CanvasSide.BOTTOM}
    )
    wrong_kind = (
        CorridorKind.BYPASS_BAND
        if reservation.region.side is CanvasSide.TOP
        else CorridorKind.OVER_TOP_BAND
    )

    with pytest.raises(ValueError, match="canvas side and corridor kind disagree"):
        replace(reservation, kind=wrong_kind)


def test_query_rejects_fabricated_boundary_blockers() -> None:
    plan = _plan()
    realised = plan.realised_reservations[0]
    malformed = replace(realised, negative_blocker_ids=("invented:negative",))

    with pytest.raises(ValueError, match="invalid boundary blocker ids"):
        build_route_plan_query(
            replace(
                plan,
                realised_reservations=_replace_record(
                    plan.realised_reservations,
                    malformed,
                    id_field="reservation_id",
                ),
            )
        )


def test_query_requires_every_non_canvas_realisation() -> None:
    plan = _plan()
    reservation = next(
        item for item in plan.reservations if not isinstance(item.region, CanvasRegion)
    )
    remaining = tuple(
        item
        for item in plan.realised_reservations
        if item.reservation_id != reservation.id
    )

    with pytest.raises(ValueError, match="missing its realisation"):
        build_route_plan_query(replace(plan, realised_reservations=remaining))


def test_reservation_claim_rejects_a_zero_length_interval() -> None:
    plan = _plan()
    claim = plan.reservations[0].claims[0]

    with pytest.raises(ValueError, match="positive travel interval"):
        replace(claim, longitudinal_end=claim.longitudinal_start)


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        ("required_width", lambda realised: realised.required_width + 1.0),
        ("capacity_slack", lambda realised: realised.capacity_slack + 1.0),
        ("negative_side_slack", lambda realised: realised.negative_side_slack + 1.0),
        ("positive_side_slack", lambda realised: realised.positive_side_slack + 1.0),
        ("coordinate", lambda realised: realised.coordinate + 1.0),
        (
            "occupied_end_translation",
            lambda realised: realised.occupied_end_translation + 10.0,
        ),
    ),
)
def test_query_rejects_inconsistent_realisation_arithmetic(field, mutate) -> None:
    plan = _plan()
    realised = plan.realised_reservations[0]
    malformed = replace(realised, **{field: mutate(realised)})

    with pytest.raises(ValueError, match="realised reservation is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                realised_reservations=_replace_record(
                    plan.realised_reservations,
                    malformed,
                    id_field="reservation_id",
                ),
            )
        )


def test_query_rejects_realisation_axes_that_disagree_with_orientation() -> None:
    plan = _plan()
    realised = plan.realised_reservations[0]
    malformed = replace(
        realised,
        allocation_axis=realised.longitudinal_axis,
        longitudinal_axis=realised.allocation_axis,
    )

    with pytest.raises(ValueError, match="realised reservation is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                realised_reservations=_replace_record(
                    plan.realised_reservations,
                    malformed,
                    id_field="reservation_id",
                ),
            )
        )


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        ("claimant_member_ids", lambda _diagnostic: ()),
        ("capacity_slack", lambda diagnostic: diagnostic.capacity_slack + 1.0),
        (
            "negative_side_slack",
            lambda diagnostic: diagnostic.negative_side_slack + 1.0,
        ),
        (
            "positive_side_slack",
            lambda diagnostic: diagnostic.positive_side_slack + 1.0,
        ),
        ("message", lambda diagnostic: f"{diagnostic.message} fabricated"),
    ),
)
def test_query_rejects_inconsistent_diagnostic_values(field, mutate) -> None:
    plan = _plan(TOPOLOGIES / "convergence_sink_fold.mmd")
    diagnostic = plan.reservation_diagnostics[0]
    malformed = replace(diagnostic, **{field: mutate(diagnostic)})

    with pytest.raises(ValueError, match="reservation diagnostic is inconsistent"):
        build_route_plan_query(
            replace(
                plan,
                reservation_diagnostics=_replace_record(
                    plan.reservation_diagnostics,
                    malformed,
                    id_field="reservation_id",
                ),
            )
        )


def test_query_rejects_duplicate_reservation_diagnostics() -> None:
    plan = _plan(TOPOLOGIES / "convergence_sink_fold.mmd")
    diagnostic = plan.reservation_diagnostics[0]

    with pytest.raises(ValueError, match="duplicate reservation diagnostics"):
        build_route_plan_query(
            replace(
                plan,
                reservation_diagnostics=(diagnostic, *plan.reservation_diagnostics),
            )
        )


def test_query_rejects_reservation_diagnostics_out_of_order() -> None:
    plan = _plan(TOPOLOGIES / "convergence_sink_fold.mmd")
    assert len(plan.reservation_diagnostics) > 1

    with pytest.raises(ValueError, match="not in reservation order"):
        build_route_plan_query(
            replace(
                plan,
                reservation_diagnostics=tuple(reversed(plan.reservation_diagnostics)),
            )
        )
