"""Exact consumption of settled route-envelope allocations."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.route_plan import (
    BindingKind,
    DemandAxis,
    EmissionBinding,
    EmissionMemberId,
    ResolvedEdge,
)
from nf_metro.layout.routing.common import Direction

if TYPE_CHECKING:
    from nf_metro.layout.envelope_settlement import (
        EnvelopeCapacityLimitation,
        EnvelopeCapacityProof,
        EnvelopeClaimAllocation,
        EnvelopeIdentityProjection,
    )
    from nf_metro.layout.route_reservations import RouteReservation, RouteReservationId
    from nf_metro.layout.routing.common import RoutedPath


class EnvelopeAllocationError(ValueError):
    """A settled allocation cannot be consumed by its immutable member."""


@dataclass(frozen=True, slots=True)
class EnvelopeAllocationQuery:
    """Complete immutable allocation index for one final routing pass."""

    _member_by_edge: Mapping[ResolvedEdge, EmissionMemberId]
    _by_member: Mapping[EmissionMemberId, tuple[EnvelopeClaimAllocation, ...]]
    _binding_by_member: Mapping[EmissionMemberId, EmissionBinding]
    _boundary_by_claim: Mapping[
        tuple[EmissionMemberId, int, int, int, DemandAxis], tuple[int, int] | None
    ]

    def allocations_for_member(
        self, member_id: EmissionMemberId
    ) -> tuple[EnvelopeClaimAllocation, ...]:
        return self._by_member.get(member_id, ())

    def project_point(
        self,
        member_id: EmissionMemberId,
        point_rank: int,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        """Project one immutable waypoint through its exact settled claims."""
        projected = list(point)
        assigned: dict[int, float] = {}
        for allocation in self.allocations_for_member(member_id):
            if not (
                allocation.segment_rank <= point_rank <= allocation.segment_end_rank + 1
            ):
                continue
            coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
            previous = assigned.get(coordinate_rank)
            if (
                previous is not None
                and abs(previous - allocation.coordinate) > COORD_TOLERANCE
            ):
                raise EnvelopeAllocationError(
                    "settled claims project one waypoint to conflicting coordinates"
                )
            assigned[coordinate_rank] = allocation.coordinate
            projected[coordinate_rank] = allocation.coordinate
        return projected[0], projected[1]

    def project_segment(
        self,
        member_id: EmissionMemberId,
        segment_rank: int,
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Project one immutable segment through its exact settled claims."""
        return (
            self.project_point(member_id, segment_rank, segment[0]),
            self.project_point(member_id, segment_rank + 1, segment[1]),
        )

    def directly_allocates(self, member_ids: tuple[EmissionMemberId, ...]) -> bool:
        return bool(member_ids) and all(
            self._by_member.get(item) for item in member_ids
        )

    def immutable_binding_for(
        self, member_id: EmissionMemberId
    ) -> EmissionBinding | None:
        return self._binding_by_member.get(member_id)

    def immutable_binding_for_edge(self, edge: ResolvedEdge) -> EmissionBinding | None:
        member_id = self._member_by_edge.get(edge)
        return self._binding_by_member.get(member_id) if member_id is not None else None

    def assert_complete(self, routes: list[RoutedPath]) -> None:
        """Require every published allocation on exactly one final route."""
        routes_by_member: defaultdict[
            EmissionMemberId, list[tuple[int, RoutedPath]]
        ] = defaultdict(list)
        for path_rank, route in enumerate(routes):
            member_id = self.member_for_route(route)
            if member_id is not None:
                routes_by_member[member_id].append((path_rank, route))
        for member_id, allocations in self._by_member.items():
            member_routes = routes_by_member[member_id]
            if len(member_routes) != 1:
                raise EnvelopeAllocationError(
                    "settled envelope member does not map to one final route"
                )
            path_rank, route = member_routes[0]
            if any(allocation.path_rank != path_rank for allocation in allocations):
                raise EnvelopeAllocationError(
                    "settled envelope claim disagrees with its final path rank"
                )
            expected = {
                (
                    rank,
                    0 if allocation.axis is DemandAxis.X else 1,
                    allocation.coordinate,
                )
                for allocation in allocations
                for rank in range(
                    allocation.segment_rank, allocation.segment_end_rank + 1
                )
            }
            if set(route.envelope_allocated_segments) != expected:
                raise EnvelopeAllocationError(
                    "final route did not consume its complete settled allocation"
                )
        assert_route_allocations(routes, "final allocation validation")

    def member_for_route(self, route: RoutedPath) -> EmissionMemberId | None:
        return self._member_by_edge.get(
            ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        )

    def consume(self, route: RoutedPath) -> None:
        member_id = self.member_for_route(route)
        if member_id is None:
            return
        allocations = self.allocations_for_member(member_id)
        if not allocations:
            return
        projected_points = [
            self.project_point(member_id, rank, point)
            for rank, point in enumerate(route.points)
        ]
        consumed: list[tuple[int, int, float]] = []
        gap_claims: list[tuple[tuple[int, int], int, int]] = []
        for allocation in allocations:
            start = allocation.segment_rank
            end = allocation.segment_end_rank + 1
            if start < 0 or end >= len(route.points) or start >= end:
                raise EnvelopeAllocationError(
                    "settled envelope claim is outside its emitted route"
                )
            coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
            claim_key = (
                allocation.member_id,
                allocation.path_rank,
                allocation.segment_rank,
                allocation.segment_end_rank,
                allocation.axis,
            )
            coordinate = allocation.coordinate
            changed = any(
                abs(point[coordinate_rank] - coordinate) > COORD_TOLERANCE
                for point in route.points[start : end + 1]
            )
            if changed and (
                route.exit_lane_transition_plan_id is not None
                or route.exit_turn_segment_rank is not None
                and start <= route.exit_turn_segment_rank <= allocation.segment_end_rank
            ):
                raise EnvelopeAllocationError(
                    "settled envelope allocation intersects frozen exit geometry"
                )
            boundary = self._boundary_by_claim.get(claim_key)
            if boundary is not None and allocation.axis is DemandAxis.X:
                gap_claims.append((boundary, start, end))
            consumed.extend(
                (rank, coordinate_rank, coordinate)
                for rank in range(start, allocation.segment_end_rank + 1)
            )
        for allocation in allocations:
            start = allocation.segment_rank
            end = allocation.segment_end_rank + 1
            coordinate_rank = 0 if allocation.axis is DemandAxis.X else 1
            travel_rank = 1 - coordinate_rank
            points = projected_points[start : end + 1]
            if any(
                abs(first[coordinate_rank] - second[coordinate_rank]) > COORD_TOLERANCE
                or abs(first[travel_rank] - second[travel_rank]) <= COORD_TOLERANCE
                for first, second in zip(points, points[1:])
            ):
                raise EnvelopeAllocationError(
                    "settled envelope claim disagrees with its emitted segment axis"
                )
        expected = tuple(dict.fromkeys(consumed))
        if route.envelope_allocated_segments and (
            route.envelope_allocated_segments != expected
        ):
            raise EnvelopeAllocationError(
                "route consumed inconsistent settled envelope allocations"
            )
        projected_route = replace(
            route,
            points=projected_points,
            envelope_allocated_segments=expected,
        )
        assert_route_allocations((projected_route,), "allocation consumption")
        route.points = projected_points
        route.envelope_allocated_segments = expected
        for boundary, start, end in gap_claims:
            direction = (
                Direction.D
                if projected_points[end][1] > projected_points[start][1]
                else Direction.U
            )
            if not any(
                slot.gap_lo_col == boundary[0]
                and slot.gap_hi_col == boundary[1]
                and slot.direction is direction
                for slot in route.gap_slots
            ):
                route.declare_gap_slot(
                    lo_col=boundary[0],
                    hi_col=boundary[1],
                    row=None,
                    direction=direction,
                    slot_index=0,
                    n_slots=1,
                )


_ClaimKey = tuple[EmissionMemberId, int, int, int, DemandAxis]


def _binding_index(
    bindings: tuple[EmissionBinding, ...],
) -> dict[EmissionMemberId, EmissionBinding]:
    by_member: dict[EmissionMemberId, EmissionBinding] = {}
    for binding in bindings:
        if binding.member_id in by_member:
            raise EnvelopeAllocationError(
                "immutable envelope bindings contain duplicate membership"
            )
        by_member[binding.member_id] = binding
    return by_member


def _index_capacity_proofs(
    proofs: tuple[EnvelopeCapacityProof, ...],
    member_by_edge: Mapping[ResolvedEdge, EmissionMemberId],
    immutable_by_id: Mapping[RouteReservationId, RouteReservation],
    binding_by_member: Mapping[EmissionMemberId, EmissionBinding],
) -> tuple[
    defaultdict[EmissionMemberId, list[EnvelopeClaimAllocation]],
    dict[_ClaimKey, tuple[int, int] | None],
    set[_ClaimKey],
]:
    from nf_metro.layout.route_reservations import (
        CanvasRegion,
        CanvasSide,
        ColumnGapRegion,
    )

    known_members = set(member_by_edge.values())
    by_member: defaultdict[EmissionMemberId, list[EnvelopeClaimAllocation]] = (
        defaultdict(list)
    )
    boundary_by_claim: dict[_ClaimKey, tuple[int, int] | None] = {}
    seen: set[_ClaimKey] = set()
    for proof in proofs:
        if proof.available_width + COORD_TOLERANCE < proof.required_width:
            raise EnvelopeAllocationError("settled allocation proof is infeasible")
        if proof.id.axis is not proof.axis or proof.id.region != proof.region:
            raise EnvelopeAllocationError("settled allocation proof identity disagrees")
        if proof.id.reservation_ids != tuple(
            item.reservation_id for item in proof.reservations
        ):
            raise EnvelopeAllocationError(
                "settled allocation proof has inconsistent reservation membership"
            )
        expected_system_ids = tuple(
            dict.fromkeys(item.system_id for item in proof.reservations)
        )
        claimant_set = {
            member_id
            for item in proof.reservations
            for member_id in item.claimant_member_ids
        }
        expected_claimant_ids = tuple(
            dict.fromkeys(
                member_id
                for member_id in member_by_edge.values()
                if member_id in claimant_set
            )
        )
        if (
            proof.system_ids != expected_system_ids
            or proof.claimant_member_ids != expected_claimant_ids
        ):
            raise EnvelopeAllocationError(
                "settled allocation proof has inconsistent aggregate ownership"
            )
        for reservation in proof.reservations:
            immutable = immutable_by_id.get(reservation.reservation_id)
            if (
                immutable is None
                or reservation.system_id not in proof.system_ids
                or not reservation.reference_id
                or not reservation.demand_ids
            ):
                raise EnvelopeAllocationError(
                    "settled allocation lacks exact symbolic ownership"
                )
            if (
                reservation.system_id != immutable.system_id
                or reservation.reference_id != immutable.reference_id
                or reservation.demand_ids != immutable.demand_ids
                or reservation.direction is not immutable.direction
                or reservation.claimant_member_ids != immutable.claimant_member_ids
                or len(reservation.allocations) != len(immutable.claims)
            ):
                raise EnvelopeAllocationError(
                    "settled allocation disagrees with its immutable reservation"
                )
            expected_axis = (
                DemandAxis.X
                if isinstance(immutable.region, ColumnGapRegion)
                or isinstance(immutable.region, CanvasRegion)
                and immutable.region.side in {CanvasSide.LEFT, CanvasSide.RIGHT}
                else DemandAxis.Y
            )
            if immutable.region != proof.region or expected_axis is not proof.axis:
                raise EnvelopeAllocationError(
                    "settled allocation proof disagrees with its immutable region"
                )
            immutable_claims = {
                (
                    claim.member_id,
                    claim.path_rank,
                    claim.segment_rank,
                    claim.segment_end_rank,
                ): claim
                for claim in immutable.claims
            }
            allocation_by_claim = {
                (
                    allocation.member_id,
                    allocation.path_rank,
                    allocation.segment_rank,
                    allocation.segment_end_rank,
                ): allocation
                for allocation in reservation.allocations
            }
            if set(allocation_by_claim) != set(immutable_claims):
                raise EnvelopeAllocationError(
                    "settled allocation changed its immutable claim projection"
                )
            for key, allocation in allocation_by_claim.items():
                immutable_claim = immutable_claims[key]
                claim_binding = binding_by_member.get(allocation.member_id)
                if (
                    claim_binding is None
                    or claim_binding.kind is not BindingKind.EMITTED
                    or claim_binding.path_rank != allocation.path_rank
                    or abs(
                        allocation.original_coordinate
                        - immutable_claim.allocation_coordinate
                    )
                    > COORD_TOLERANCE
                ):
                    raise EnvelopeAllocationError(
                        "settled allocation changed its immutable claim projection"
                    )
            lane_coordinate_by_claim: dict[int, float] = {}
            for lane_allocation in reservation.lanes:
                if not 0 <= lane_allocation.lane_rank < len(immutable.lanes):
                    raise EnvelopeAllocationError(
                        "settled allocation names an unknown immutable lane"
                    )
                immutable_lane = immutable.lanes[lane_allocation.lane_rank]
                if not lane_allocation.claim_indices or any(
                    claim_rank not in immutable_lane.claim_indices
                    or claim_rank in lane_coordinate_by_claim
                    for claim_rank in lane_allocation.claim_indices
                ):
                    raise EnvelopeAllocationError(
                        "settled allocation changed immutable lane membership"
                    )
                lane_claims = tuple(
                    immutable.claims[claim_rank]
                    for claim_rank in lane_allocation.claim_indices
                )
                expected_witnesses = tuple(
                    dict.fromkeys(claim.member_id for claim in lane_claims)
                )
                if (
                    lane_allocation.claimant_member_ids != expected_witnesses
                    or not math.isfinite(lane_allocation.minimum_coordinate)
                    or not math.isfinite(lane_allocation.maximum_coordinate)
                    or lane_allocation.coordinate
                    < lane_allocation.minimum_coordinate - COORD_TOLERANCE
                    or lane_allocation.coordinate
                    > lane_allocation.maximum_coordinate + COORD_TOLERANCE
                    or any(
                        abs(
                            claim.allocation_coordinate
                            - lane_allocation.original_coordinate
                        )
                        > COORD_TOLERANCE
                        for claim in lane_claims
                    )
                ):
                    raise EnvelopeAllocationError(
                        "settled allocation has inconsistent immutable lane witnesses"
                    )
                for claim_rank, claim in zip(
                    lane_allocation.claim_indices, lane_claims, strict=True
                ):
                    lane_claim_allocation = allocation_by_claim.get(
                        (
                            claim.member_id,
                            claim.path_rank,
                            claim.segment_rank,
                            claim.segment_end_rank,
                        )
                    )
                    if (
                        lane_claim_allocation is None
                        or abs(
                            lane_claim_allocation.coordinate
                            - lane_allocation.coordinate
                        )
                        > COORD_TOLERANCE
                    ):
                        raise EnvelopeAllocationError(
                            "settled claim disagrees with its physical lane allocation"
                        )
                    lane_coordinate_by_claim[claim_rank] = lane_allocation.coordinate
            if set(lane_coordinate_by_claim) != set(range(len(immutable.claims))):
                raise EnvelopeAllocationError(
                    "settled allocation does not cover every immutable lane claim"
                )
            for allocation in reservation.allocations:
                direct_key = (
                    allocation.member_id,
                    allocation.path_rank,
                    allocation.segment_rank,
                    allocation.segment_end_rank,
                    allocation.axis,
                )
                if (
                    allocation.member_id not in known_members
                    or allocation.member_id not in reservation.claimant_member_ids
                    or allocation.member_id not in proof.claimant_member_ids
                    or allocation.axis is not proof.axis
                    or not math.isfinite(allocation.coordinate)
                    or direct_key in seen
                ):
                    raise EnvelopeAllocationError(
                        "settled allocation has inconsistent direct claim ownership"
                    )
                direct_immutable_claim = immutable_claims.get(direct_key[:4])
                claim_binding = binding_by_member.get(allocation.member_id)
                if (
                    direct_immutable_claim is None
                    or claim_binding is None
                    or claim_binding.kind is not BindingKind.EMITTED
                    or claim_binding.path_rank != allocation.path_rank
                    or abs(
                        allocation.original_coordinate
                        - direct_immutable_claim.allocation_coordinate
                    )
                    > COORD_TOLERANCE
                ):
                    raise EnvelopeAllocationError(
                        "settled allocation changed its immutable claim projection"
                    )
                seen.add(direct_key)
                by_member[allocation.member_id].append(allocation)
                boundary_by_claim[direct_key] = proof.boundary
    return by_member, boundary_by_claim, seen


def _projection_index(
    projections: tuple[EnvelopeIdentityProjection, ...],
    immutable_by_id: Mapping[RouteReservationId, RouteReservation],
    binding_by_member: Mapping[EmissionMemberId, EmissionBinding],
    proofed_reservation_ids: frozenset[RouteReservationId],
) -> dict[RouteReservationId, EnvelopeIdentityProjection]:
    from nf_metro.layout.route_reservations import CorridorOrientation

    by_id = {item.reservation_id: item for item in projections}
    expected_reservation_ids = set(immutable_by_id).difference(proofed_reservation_ids)
    if len(by_id) != len(projections) or set(by_id) != expected_reservation_ids:
        raise EnvelopeAllocationError(
            "identity projections disagree with immutable reservations"
        )
    for reservation_id, projection in by_id.items():
        immutable = immutable_by_id[reservation_id]
        axis = (
            DemandAxis.X
            if immutable.orientation is CorridorOrientation.VERTICAL
            else DemandAxis.Y
        )
        immutable_claims = {
            (
                claim.member_id,
                claim.path_rank,
                claim.segment_rank,
                claim.segment_end_rank,
            ): claim
            for claim in immutable.claims
        }
        projected_claims = {
            (
                allocation.member_id,
                allocation.path_rank,
                allocation.segment_rank,
                allocation.segment_end_rank,
            ): allocation
            for allocation in projection.allocations
        }
        if len(projected_claims) != len(projection.allocations) or set(
            projected_claims
        ) != set(immutable_claims):
            raise EnvelopeAllocationError(
                "identity projection changed immutable claim membership"
            )
        for key, allocation in projected_claims.items():
            claim = immutable_claims[key]
            binding = binding_by_member.get(claim.member_id)
            if (
                allocation.axis is not axis
                or allocation.member_id not in immutable.claimant_member_ids
                or binding is None
                or binding.kind is not BindingKind.EMITTED
                or binding.path_rank != claim.path_rank
                or abs(allocation.original_coordinate - claim.allocation_coordinate)
                > COORD_TOLERANCE
                or not math.isfinite(allocation.coordinate)
            ):
                raise EnvelopeAllocationError(
                    "identity projection changed immutable claim identity"
                )
    return by_id


def build_envelope_allocation_query(
    proofs: tuple[EnvelopeCapacityProof, ...],
    member_by_edge: Mapping[ResolvedEdge, EmissionMemberId],
    reservations: tuple[RouteReservation, ...] = (),
    bindings: tuple[EmissionBinding, ...] = (),
    limitations: tuple[EnvelopeCapacityLimitation, ...] = (),
    identity_projections: tuple[EnvelopeIdentityProjection, ...] = (),
) -> EnvelopeAllocationQuery:
    """Validate and index every direct claim in the settled proof ledger."""
    immutable_by_id = {item.id: item for item in reservations}
    if len(immutable_by_id) != len(reservations):
        raise EnvelopeAllocationError(
            "immutable envelope ledger contains duplicate reservations"
        )
    if proofs and not immutable_by_id:
        raise EnvelopeAllocationError(
            "settled allocation proofs require their immutable reservation ledger"
        )
    binding_by_member = _binding_index(bindings)
    proofed_system_ids = {
        system_id for proof in proofs for system_id in proof.system_ids
    }
    limited_system_ids = {item.system_id for item in limitations}
    if proofed_system_ids.intersection(limited_system_ids):
        raise EnvelopeAllocationError(
            "one route system has both capacity proof and limitation ownership"
        )
    for limitation in limitations:
        if not limitation.reservation_ids or any(
            (reservation := immutable_by_id.get(reservation_id)) is None
            or reservation.system_id != limitation.system_id
            for reservation_id in limitation.reservation_ids
        ):
            raise EnvelopeAllocationError(
                "capacity limitation disagrees with immutable system ownership"
            )
    by_member, boundary_by_claim, seen = _index_capacity_proofs(
        proofs,
        member_by_edge,
        immutable_by_id,
        binding_by_member,
    )
    if reservations:
        proofed_reservation_ids = frozenset(
            reservation.reservation_id
            for proof in proofs
            for reservation in proof.reservations
        )
        projections_by_id = _projection_index(
            identity_projections,
            immutable_by_id,
            binding_by_member,
            proofed_reservation_ids,
        )
        for projection in projections_by_id.values():
            for allocation in projection.allocations:
                key = (
                    allocation.member_id,
                    allocation.path_rank,
                    allocation.segment_rank,
                    allocation.segment_end_rank,
                    allocation.axis,
                )
                if key in seen:
                    raise EnvelopeAllocationError(
                        "settled envelope contains duplicate claim allocations"
                    )
                seen.add(key)
                by_member[allocation.member_id].append(allocation)
                boundary_by_claim[key] = None
    return EnvelopeAllocationQuery(
        MappingProxyType(dict(member_by_edge)),
        MappingProxyType(
            {
                member_id: tuple(
                    sorted(
                        allocations,
                        key=lambda item: (
                            item.path_rank,
                            item.segment_rank,
                            item.segment_end_rank,
                            item.axis.value,
                        ),
                    )
                )
                for member_id, allocations in by_member.items()
            }
        ),
        MappingProxyType(binding_by_member),
        MappingProxyType(boundary_by_claim),
    )


def assert_route_allocations(
    routes: tuple[RoutedPath, ...] | list[RoutedPath], stage: str
) -> None:
    """Require every consumed segment to retain its settled coordinate."""
    for route in routes:
        for rank, coordinate_rank, coordinate in route.envelope_allocated_segments:
            if rank + 1 >= len(route.points):
                raise EnvelopeAllocationError(
                    f"{stage} removed a settled envelope segment"
                )
            if any(
                abs(route.points[point_rank][coordinate_rank] - coordinate)
                > COORD_TOLERANCE
                for point_rank in (rank, rank + 1)
            ):
                raise EnvelopeAllocationError(
                    f"{stage} moved settled segment {rank} on "
                    f"{route.edge.source}->{route.edge.target} line "
                    f"{route.line_id} from axis coordinate {coordinate:.1f}"
                )
