from pathlib import Path

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.route_plan import BindingKind
from nf_metro.layout.route_reservations import CanvasRegion, CanvasSide, ColumnGapRegion
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.render.svg import build_observed_render_plan

FIXTURES = Path(__file__).parent / "fixtures"


def test_settlement_preserves_a_latent_destination_flank_claim() -> None:
    fixture = FIXTURES / "gate_1660_destination_tail_collision.mmd"
    graph = prepare_graph(fixture.read_text(), source_dir=str(fixture.parent))

    observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    assert any(
        isinstance(reservation.region, ColumnGapRegion)
        and reservation.region.left_column == 0
        and any(claim.segment_rank == 1 for claim in reservation.claims)
        for reservation in observed.route_plan.reservations
    )


def test_settlement_owns_a_shared_entry_convergence_corridor() -> None:
    fixture = FIXTURES / "gate_1660_unowned_corridor_convergence.mmd"
    graph = prepare_graph(fixture.read_text(), source_dir=str(fixture.parent))
    graph.strict = True

    observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    (convergence,) = observed.route_plan.convergence_plans
    assert convergence.owns_geometry
    assert convergence.legacy_reason is None

    members = {member.id: member for member in observed.route_plan.members}
    (bypass_member,) = (
        member
        for member in members.values()
        if member.family_id is RouteFamilyId.BYPASS_FAMILY
    )
    assert convergence.primary_trunk_member_id is not None
    assert (
        members[convergence.primary_trunk_member_id].family_id
        is RouteFamilyId.MERGE_TRUNK
    )
    assert bypass_member.id not in convergence.member_ids

    (canvas_reservation,) = (
        reservation
        for reservation in observed.route_plan.reservations
        if isinstance(reservation.region, CanvasRegion)
        and reservation.region.side is CanvasSide.BOTTOM
    )
    expected_claimants = {
        bypass_member.id,
        convergence.primary_trunk_member_id,
    }
    assert set(canvas_reservation.claimant_member_ids) == expected_claimants
    assert {
        claim.member_id for claim in canvas_reservation.claims
    } == expected_claimants
    assert canvas_reservation.reference_id in {
        reference.id
        for reference in observed.route_plan.shared_references
        if convergence.primary_trunk_member_id in reference.claimant_member_ids
    }

    planned_route_edges = {
        (route.edge.source, route.edge.target, route.edge.line_id)
        for route in observed.plan.routes
        if route.convergence_plan_id == convergence.id
    }
    emitted_member_ids = {
        binding.member_id
        for binding in observed.route_plan.bindings
        if binding.kind is BindingKind.EMITTED
    }
    assert planned_route_edges == {
        (
            members[member_id].edge.source,
            members[member_id].edge.target,
            members[member_id].edge.line_id,
        )
        for member_id in convergence.member_ids
        if member_id in emitted_member_ids
    }
