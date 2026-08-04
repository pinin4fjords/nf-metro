"""Bounded monotone settlement of final row and column route envelopes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from nf_metro.layout.constants import COORD_TOLERANCE, CURVE_RADIUS, graph_offset_step
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import (
    DemandAxis,
    DemandId,
    EmissionMemberId,
    RoutePlan,
    RouteSystemId,
    SharedReferenceId,
)
from nf_metro.layout.route_reservations import (
    CanvasRegion,
    CanvasSide,
    ColumnGapRegion,
    CorridorOrientation,
    CorridorRegion,
    RealisedRouteReservation,
    RouteReservation,
    RouteReservationClaim,
    RouteReservationId,
    RowGapRegion,
    canvas_inner_boundary,
    realise_route_reservations,
    reservation_claim_lane_coordinates,
)
from nf_metro.layout.routing.common import Direction
from nf_metro.parser.model import MetroGraph, Section


class EnvelopeAxis(str, Enum):
    X = "x"
    Y = "y"


@dataclass(frozen=True, slots=True)
class EnvelopeTranslation:
    """One canonical downstream translation owned by a grid boundary."""

    axis: EnvelopeAxis
    boundary: tuple[int, int]
    amount: float
    section_ids: tuple[str, ...]
    reservation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvelopeSettlement:
    """Immutable evidence produced by one finite directional pass."""

    translations: tuple[EnvelopeTranslation, ...]
    origin_adjustment: tuple[float, float]
    boundary_count: int
    capacity_proofs: tuple[EnvelopeCapacityProof, ...]
    capacity_limitations: tuple[EnvelopeCapacityLimitation, ...]
    identity_projections: tuple[EnvelopeIdentityProjection, ...]


@dataclass(frozen=True, slots=True)
class EnvelopeClaimAllocation:
    """One immutable reservation claim projected into its settled band."""

    member_id: EmissionMemberId
    path_rank: int
    segment_rank: int
    segment_end_rank: int
    axis: DemandAxis
    original_coordinate: float
    coordinate: float


@dataclass(frozen=True, slots=True)
class EnvelopeIdentityProjection:
    """One reservation's claims projected through global settlement moves."""

    reservation_id: RouteReservationId
    allocations: tuple[EnvelopeClaimAllocation, ...]


@dataclass(frozen=True, slots=True)
class EnvelopeAllocationGroupId:
    """Stable identity for one jointly allocated boundary component."""

    axis: DemandAxis
    region: CorridorRegion
    reservation_ids: tuple[RouteReservationId, ...]

    @property
    def boundary(self) -> tuple[int, int] | None:
        if isinstance(self.region, ColumnGapRegion):
            return self.region.left_column, self.region.right_column
        if isinstance(self.region, RowGapRegion):
            return self.region.upper_row, self.region.lower_row
        return None


@dataclass(frozen=True, slots=True)
class EnvelopeLaneAllocation:
    """Exact immutable claim group allocated as one physical lane coordinate."""

    lane_rank: int
    claim_indices: tuple[int, ...]
    claimant_member_ids: tuple[EmissionMemberId, ...]
    original_coordinate: float
    coordinate: float
    minimum_coordinate: float
    maximum_coordinate: float


@dataclass(frozen=True, slots=True)
class EnvelopeReservationAllocation:
    """One reservation's materialised coordinate within a joint allocation."""

    reservation_id: RouteReservationId
    system_id: RouteSystemId
    reference_id: SharedReferenceId
    demand_ids: tuple[DemandId, ...]
    direction: Direction
    claimant_member_ids: tuple[EmissionMemberId, ...]
    original_coordinate: float
    coordinate: float
    lanes: tuple[EnvelopeLaneAllocation, ...]
    allocations: tuple[EnvelopeClaimAllocation, ...]


@dataclass(frozen=True, slots=True)
class EnvelopeCapacityProof:
    """Measured evidence for one complete joint boundary allocation."""

    id: EnvelopeAllocationGroupId
    system_ids: tuple[RouteSystemId, ...]
    region: CorridorRegion
    axis: DemandAxis
    claimant_member_ids: tuple[EmissionMemberId, ...]
    region_start: float
    region_end: float
    available_width: float
    required_width: float
    reservations: tuple[EnvelopeReservationAllocation, ...]

    @property
    def boundary(self) -> tuple[int, int] | None:
        return self.id.boundary


@dataclass(frozen=True, slots=True)
class EnvelopeCapacityLimitation:
    """Final evidence that authored commitments prevent system settlement."""

    system_id: RouteSystemId
    reservation_ids: tuple[RouteReservationId, ...]
    blocker_ids: tuple[str, ...]
    pinned_section_ids: tuple[str, ...]
    owner_issue: int = 1658


class EnvelopeSettlementError(ValueError):
    """A hard reservation cannot be satisfied by an owned translation."""


@dataclass(frozen=True, slots=True)
class _AxisBoundary:
    axis: EnvelopeAxis
    negative: int
    positive: int

    def starts_after(self, section: Section) -> bool:
        start = section.grid_col if self.axis is EnvelopeAxis.X else section.grid_row
        return start >= self.positive


@dataclass(frozen=True, slots=True)
class _GeometrySnapshot:
    sections: tuple[tuple[str, float, float], ...]
    stations: tuple[tuple[str, float, float], ...]
    ports: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class _CanvasHalfLine:
    """Internal one-sided canvas envelope used only by the finite lane packer."""

    allocation_axis: DemandAxis
    coordinate: float
    region_start: float
    region_end: float
    negative_blocker_ids: tuple[str, ...]
    positive_blocker_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BoundaryReservation:
    reservation: RouteReservation
    realised: RealisedRouteReservation | _CanvasHalfLine
    member_rank: int
    negative_footprint: float
    positive_footprint: float
    negative_extent: float
    positive_extent: float
    fixed: bool
    fixed_movable: bool
    fixed_claims: tuple[bool, ...]
    fixed_movable_claims: tuple[bool, ...]
    endpoint_continuity_members: tuple[tuple[frozenset[EmissionMemberId], ...], ...]
    sharing_coordinates: tuple[float, ...]
    lane_coordinates: tuple[float, ...]
    sharing_keys: tuple[frozenset[tuple[str, str]], ...]


def _snapshot(graph: MetroGraph) -> _GeometrySnapshot:
    return _GeometrySnapshot(
        tuple((item.id, item.bbox_x, item.bbox_y) for item in graph.sections.values()),
        tuple((item.id, item.x, item.y) for item in graph.stations.values()),
        tuple((item.id, item.x, item.y) for item in graph.ports.values()),
    )


def _restore(graph: MetroGraph, snapshot: _GeometrySnapshot) -> None:
    for item_id, x, y in snapshot.sections:
        section = graph.sections[item_id]
        section.bbox_x = x
        section.bbox_y = y
    for item_id, x, y in snapshot.stations:
        station = graph.stations[item_id]
        station.x = x
        station.y = y
    for item_id, x, y in snapshot.ports:
        port = graph.ports[item_id]
        port.x = x
        port.y = y


def _endpoint_allocation_shift(
    graph: MetroGraph,
    origins: dict[str, tuple[float, float]],
    claim: RouteReservationClaim,
    allocation_rank: int,
) -> float | None:
    shifts = tuple(
        (
            graph.stations[endpoint_id].x - origins[endpoint_id][0]
            if allocation_rank == 0
            else graph.stations[endpoint_id].y - origins[endpoint_id][1]
        )
        for endpoint_id in claim.endpoint_anchor_ids
    )
    if shifts and any(abs(shift - shifts[0]) > COORD_TOLERANCE for shift in shifts[1:]):
        raise EnvelopeSettlementError(
            "one reservation claim has divergent endpoint-anchor translations"
        )
    return shifts[0] if shifts else None


def _translated_claim_allocation_shift(
    claim: RouteReservationClaim,
    maximum_claim_rank: int,
    allocation_rank: int,
    source_shift: tuple[float, float],
    target_shift: tuple[float, float],
    rigid_shift: float | None,
    corridor_shift: float | None,
) -> float:
    if rigid_shift is not None:
        return rigid_shift
    if corridor_shift is not None:
        return corridor_shift
    if claim.segment_end_rank >= maximum_claim_rank - 1:
        return target_shift[allocation_rank]
    if claim.segment_rank <= 1:
        return source_shift[allocation_rank]
    if (
        abs(source_shift[allocation_rank] - target_shift[allocation_rank])
        <= COORD_TOLERANCE
    ):
        return source_shift[allocation_rank]
    return 0.0


def _project_ledger_translations(
    graph: MetroGraph,
    plan: RoutePlan,
    snapshot: _GeometrySnapshot,
) -> RoutePlan:
    origins = {item_id: (x, y) for item_id, x, y in snapshot.stations}
    member_by_id = {item.id: item for item in plan.members}
    max_claim_rank_by_member: dict[EmissionMemberId, int] = {}
    for reservation in plan.reservations:
        for claim in reservation.claims:
            max_claim_rank_by_member[claim.member_id] = max(
                max_claim_rank_by_member.get(claim.member_id, -1),
                claim.segment_end_rank,
            )
    original_realised_by_id = {
        item.reservation_id: item for item in plan.realised_reservations
    }
    fixed_axis_by_member = {
        assignment.member_id: axis
        for exit_plan in plan.exit_turn_plans
        for assignment in exit_plan.assignments
        if assignment.axis_id is not None
        for axis in exit_plan.axes
        if axis.id == assignment.axis_id
    }

    def projected_longitudinal_coordinate(
        coordinate: float,
        source_origin: float,
        source_coordinate: float,
        target_origin: float,
        target_coordinate: float,
    ) -> float:
        endpoint_coordinates = tuple(
            translated
            for original, translated in (
                (source_origin, source_coordinate),
                (target_origin, target_coordinate),
            )
            if abs(coordinate - original) <= COORD_TOLERANCE
        )
        if endpoint_coordinates and all(
            abs(item - endpoint_coordinates[0]) <= COORD_TOLERANCE
            for item in endpoint_coordinates[1:]
        ):
            return endpoint_coordinates[0]
        source_shift = source_coordinate - source_origin
        target_shift = target_coordinate - target_origin
        if abs(source_shift - target_shift) <= COORD_TOLERANCE:
            return coordinate + source_shift
        return coordinate

    longitudinal_projected = []
    for reservation in plan.reservations:
        allocation_rank = (
            0 if reservation.orientation is CorridorOrientation.VERTICAL else 1
        )
        travel_rank = 1 - allocation_rank
        claims = []
        for claim in reservation.claims:
            member = member_by_id[claim.member_id]
            source = graph.stations[member.source.station_id]
            target = graph.stations[member.target.station_id]
            source_origin = origins[source.id]
            target_origin = origins[target.id]
            source_shift = (
                source.x - source_origin[0],
                source.y - source_origin[1],
            )
            target_shift = (
                target.x - target_origin[0],
                target.y - target_origin[1],
            )
            projected_start = projected_longitudinal_coordinate(
                claim.longitudinal_start,
                source_origin[travel_rank],
                source_origin[travel_rank] + source_shift[travel_rank],
                target_origin[travel_rank],
                target_origin[travel_rank] + target_shift[travel_rank],
            )
            projected_end = projected_longitudinal_coordinate(
                claim.longitudinal_end,
                source_origin[travel_rank],
                source_origin[travel_rank] + source_shift[travel_rank],
                target_origin[travel_rank],
                target_origin[travel_rank] + target_shift[travel_rank],
            )
            claims.append(
                replace(
                    claim,
                    longitudinal_start=min(projected_start, projected_end),
                    longitudinal_end=max(projected_start, projected_end),
                )
            )
        longitudinal_projected.append(replace(reservation, claims=tuple(claims)))
    longitudinal_plan = replace(plan, reservations=tuple(longitudinal_projected))
    translated_realised_by_id = {
        item.reservation_id: item
        for item in realise_route_reservations(
            longitudinal_plan, graph, blocker_plan=plan
        ).realised_reservations
    }

    projected = []
    for original_reservation, longitudinal_reservation in zip(
        plan.reservations, longitudinal_plan.reservations, strict=True
    ):
        allocation_rank = (
            0 if original_reservation.orientation is CorridorOrientation.VERTICAL else 1
        )
        original_realised = original_realised_by_id.get(original_reservation.id)
        translated_realised = translated_realised_by_id.get(original_reservation.id)
        corridor_shift: float | None = None
        if original_realised is not None and translated_realised is not None:
            start_shift = (
                translated_realised.region_start - original_realised.region_start
            )
            end_shift = translated_realised.region_end - original_realised.region_end
            corridor_shift = min(start_shift, end_shift)
        allocation_axis = (
            DemandAxis.X
            if original_reservation.orientation is CorridorOrientation.VERTICAL
            else DemandAxis.Y
        )
        fixed_shifts = tuple(
            (
                graph.stations[axis.fixed_anchor_id or member.source.station_id].x
                - origins[axis.fixed_anchor_id or member.source.station_id][0]
                if allocation_axis is DemandAxis.X
                else graph.stations[axis.fixed_anchor_id or member.source.station_id].y
                - origins[axis.fixed_anchor_id or member.source.station_id][1]
            )
            for claim in original_reservation.claims
            for axis in (fixed_axis_by_member.get(claim.member_id),)
            if axis is not None
            and axis.axis is allocation_axis
            and claim.segment_rank <= 1 <= claim.segment_end_rank
            for member in (member_by_id[claim.member_id],)
        )
        if fixed_shifts and any(
            abs(item - fixed_shifts[0]) > COORD_TOLERANCE for item in fixed_shifts[1:]
        ):
            raise EnvelopeSettlementError(
                "one reservation has conflicting fixed-axis translation owners"
            )
        rigid_shift = fixed_shifts[0] if fixed_shifts else None
        claims = []
        for original_claim, longitudinal_claim in zip(
            original_reservation.claims,
            longitudinal_reservation.claims,
            strict=True,
        ):
            member = member_by_id[original_claim.member_id]
            source = graph.stations[member.source.station_id]
            target = graph.stations[member.target.station_id]
            source_origin = origins[source.id]
            target_origin = origins[target.id]
            source_shift = (
                source.x - source_origin[0],
                source.y - source_origin[1],
            )
            target_shift = (
                target.x - target_origin[0],
                target.y - target_origin[1],
            )
            endpoint_shift = _endpoint_allocation_shift(
                graph, origins, original_claim, allocation_rank
            )
            allocation_shift = (
                endpoint_shift
                if endpoint_shift is not None
                else _translated_claim_allocation_shift(
                    original_claim,
                    max_claim_rank_by_member[original_claim.member_id],
                    allocation_rank,
                    source_shift,
                    target_shift,
                    rigid_shift,
                    corridor_shift,
                )
            )
            claims.append(
                replace(
                    longitudinal_claim,
                    allocation_coordinate=original_claim.allocation_coordinate
                    + allocation_shift,
                )
            )
        projected.append(replace(original_reservation, claims=tuple(claims)))
    preliminary = replace(plan, reservations=tuple(projected))
    realised_preliminary = realise_route_reservations(
        preliminary, graph, blocker_plan=plan
    )
    shared_delta_by_id: dict[RouteReservationId, float] = {}
    for reservations in _boundary_reservations(
        realised_preliminary, graph, plan
    ).values():
        parent = list(range(len(reservations)))

        def root(rank: int) -> int:
            while parent[rank] != rank:
                parent[rank] = parent[parent[rank]]
                rank = parent[rank]
            return rank

        for first_rank, first in enumerate(reservations):
            for second_rank in range(first_rank + 1, len(reservations)):
                second = reservations[second_rank]
                if _claims_overlap(first, second) and _shares_channel(first, second):
                    first_root, second_root = root(first_rank), root(second_rank)
                    if first_root != second_root:
                        parent[second_root] = first_root
        groups: dict[int, list[_BoundaryReservation]] = {}
        for rank, boundary_item in enumerate(reservations):
            groups.setdefault(root(rank), []).append(boundary_item)
        for group in groups.values():
            if len(group) < 2:
                continue
            fixed_deltas = {
                item.realised.coordinate
                - original_realised_by_id[item.reservation.id].coordinate
                for item in group
                if item.fixed
            }
            if not fixed_deltas:
                continue
            delta = min(fixed_deltas)
            if any(abs(item - delta) > COORD_TOLERANCE for item in fixed_deltas):
                raise EnvelopeSettlementError(
                    "shared channel has conflicting projected fixed coordinates"
                )
            for item in group:
                shared_delta_by_id[item.reservation.id] = delta
    if not shared_delta_by_id:
        return preliminary
    return replace(
        preliminary,
        reservations=tuple(
            replace(
                reservation,
                claims=tuple(
                    replace(
                        claim,
                        allocation_coordinate=original_claim.allocation_coordinate
                        + shared_delta_by_id[reservation.id],
                    )
                    for original_claim, claim in zip(
                        next(
                            item
                            for item in plan.reservations
                            if item.id == reservation.id
                        ).claims,
                        reservation.claims,
                        strict=True,
                    )
                ),
            )
            if reservation.id in shared_delta_by_id
            else reservation
            for reservation in preliminary.reservations
        ),
    )


def _identity_projections(
    original: RoutePlan,
    projected: RoutePlan,
) -> tuple[EnvelopeIdentityProjection, ...]:
    """Publish claim coordinates derived from the final global translations."""
    projected_by_id = {item.id: item for item in projected.reservations}
    records: list[EnvelopeIdentityProjection] = []
    for reservation in original.reservations:
        projected_reservation = projected_by_id[reservation.id]
        projected_claims = {
            (
                claim.member_id,
                claim.path_rank,
                claim.segment_rank,
                claim.segment_end_rank,
            ): claim
            for claim in projected_reservation.claims
        }
        axis = (
            DemandAxis.X
            if reservation.orientation is CorridorOrientation.VERTICAL
            else DemandAxis.Y
        )
        allocations = tuple(
            EnvelopeClaimAllocation(
                claim.member_id,
                claim.path_rank,
                claim.segment_rank,
                claim.segment_end_rank,
                axis,
                claim.allocation_coordinate,
                projected_claims[
                    (
                        claim.member_id,
                        claim.path_rank,
                        claim.segment_rank,
                        claim.segment_end_rank,
                    )
                ].allocation_coordinate,
            )
            for claim in reservation.claims
        )
        records.append(EnvelopeIdentityProjection(reservation.id, allocations))
    return tuple(records)


def _quantised_growth(shortfall: float, quantum: float) -> float:
    if shortfall <= COORD_TOLERANCE:
        return 0.0
    return math.ceil(shortfall / quantum) * quantum


def _boundary_for_region(region: object) -> _AxisBoundary | None:
    if isinstance(region, RowGapRegion):
        return _AxisBoundary(EnvelopeAxis.Y, region.upper_row, region.lower_row)
    if isinstance(region, ColumnGapRegion):
        return _AxisBoundary(EnvelopeAxis.X, region.left_column, region.right_column)
    return None


def _locked_crossing_blockers(
    graph: MetroGraph,
    boundary: _AxisBoundary,
    realised: RealisedRouteReservation | _CanvasHalfLine,
) -> tuple[str, ...]:
    locked: list[str] = []
    negative_prefix = (
        "section-right" if boundary.axis is EnvelopeAxis.X else "section-bottom"
    )
    positive_prefix = (
        "section-left" if boundary.axis is EnvelopeAxis.X else "section-header"
    )
    for section in graph.sections.values():
        start = (
            section.grid_col if boundary.axis is EnvelopeAxis.X else section.grid_row
        )
        span = (
            section.grid_col_span
            if boundary.axis is EnvelopeAxis.X
            else section.grid_row_span
        )
        if not (start <= boundary.negative and start + span - 1 >= boundary.positive):
            continue
        decision = graph.layout_provenance.grid_decision(section.id)
        if decision is None or not decision.is_reinference_locked:
            continue
        if (
            f"{negative_prefix}:{section.id}" in realised.negative_blocker_ids
            and f"{positive_prefix}:{section.id}" in realised.positive_blocker_ids
        ):
            locked.append(section.id)
    return tuple(locked)


def _claims_overlap(first: _BoundaryReservation, second: _BoundaryReservation) -> bool:
    return any(
        min(first_claim.longitudinal_end, second_claim.longitudinal_end)
        - max(first_claim.longitudinal_start, second_claim.longitudinal_start)
        > -CURVE_RADIUS + COORD_TOLERANCE
        for first_claim in first.reservation.claims
        for second_claim in second.reservation.claims
    )


def _endpoint_incident_members(
    plan: RoutePlan,
) -> dict[
    tuple[CorridorOrientation, str, str],
    tuple[tuple[EmissionMemberId, float], ...],
]:
    members = {member.id: member for member in plan.members}
    coordinates: dict[
        tuple[CorridorOrientation, str, str],
        set[tuple[EmissionMemberId, float]],
    ] = {}
    for reservation in plan.reservations:
        for claim in reservation.claims:
            member = members[claim.member_id]
            for endpoint in (
                member.source.station_id,
                member.target.station_id,
            ):
                coordinates.setdefault(
                    (reservation.orientation, member.line_id, endpoint), set()
                ).add((member.id, claim.allocation_coordinate))
    return {
        key: tuple(sorted(items, key=lambda item: (str(item[0]), item[1])))
        for key, items in coordinates.items()
    }


def _boundary_reservations(
    plan: RoutePlan,
    graph: MetroGraph | None = None,
    sharing_plan: RoutePlan | None = None,
) -> dict[_AxisBoundary, tuple[_BoundaryReservation, ...]]:
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    sharing_reservation_by_id = {
        item.id: item for item in (sharing_plan or plan).reservations
    }
    member_rank = {item.id: rank for rank, item in enumerate(plan.members)}
    fixed_axis_by_member = {
        assignment.member_id: axis
        for exit_plan in plan.exit_turn_plans
        for assignment in exit_plan.assignments
        if assignment.axis_id is not None
        for axis in exit_plan.axes
        if axis.id == assignment.axis_id
    }
    fixed_axis_member_ids = set(fixed_axis_by_member)
    transition_member_ids = {
        member_id
        for exit_plan in plan.exit_turn_plans
        for transition in exit_plan.lane_transitions
        for member_id in transition.claimant_member_ids
    }
    fixed_member_ids = fixed_axis_member_ids | transition_member_ids
    member_by_id = {item.id: item for item in plan.members}
    incident_members = _endpoint_incident_members(plan)
    junction_anchors = _junction_anchors(plan)

    def endpoint_is_positive(station_id: str, boundary: _AxisBoundary) -> bool:
        if graph is None:
            return False
        station = graph.stations[station_id]
        section_ids = (
            (station.section_id,)
            if station.section_id is not None
            else tuple(junction_anchors.get(station_id, ()))
        )
        return bool(section_ids) and all(
            (
                graph.sections[section_id].grid_col
                if boundary.axis is EnvelopeAxis.X
                else graph.sections[section_id].grid_row
            )
            >= boundary.positive
            for section_id in section_ids
        )

    grouped: dict[_AxisBoundary, list[_BoundaryReservation]] = {}
    for reservation in plan.reservations:
        boundary = _boundary_for_region(reservation.region)
        realised = realised_by_id.get(reservation.id)
        if boundary is None or realised is None:
            continue
        reservation_fixed_members = tuple(
            member_id
            for member_id in reservation.claimant_member_ids
            if member_id in fixed_member_ids
        )
        lane_coordinates = (
            reservation_claim_lane_coordinates(graph, reservation, member_by_id)
            if graph is not None
            else tuple(claim.allocation_coordinate for claim in reservation.claims)
        )

        def fixed_member_moves(member_id: EmissionMemberId) -> bool:
            member = member_by_id[member_id]
            axis = fixed_axis_by_member.get(member_id)
            if axis is not None:
                return endpoint_is_positive(
                    axis.fixed_anchor_id or member.source.station_id,
                    boundary,
                )
            return all(
                endpoint_is_positive(endpoint, boundary)
                for endpoint in (
                    member.source.station_id,
                    member.target.station_id,
                )
            )

        grouped.setdefault(boundary, []).append(
            _BoundaryReservation(
                reservation,
                realised,
                min(member_rank[item] for item in reservation.claimant_member_ids),
                realised.coordinate - realised.occupied_start,
                realised.occupied_end - realised.coordinate,
                realised.coordinate
                - realised.occupied_start
                + reservation.negative_side_clearance,
                realised.occupied_end
                - realised.coordinate
                + reservation.positive_side_clearance,
                bool(reservation_fixed_members),
                bool(reservation_fixed_members)
                and all(fixed_member_moves(item) for item in reservation_fixed_members),
                tuple(
                    claim.member_id in transition_member_ids
                    or claim.member_id in fixed_axis_member_ids
                    and claim.segment_rank <= 1 <= claim.segment_end_rank
                    for claim in reservation.claims
                ),
                tuple(
                    not claim.endpoint_anchor_ids
                    or all(
                        endpoint_is_positive(endpoint, boundary)
                        for endpoint in claim.endpoint_anchor_ids
                    )
                    for claim in reservation.claims
                ),
                tuple(
                    tuple(
                        frozenset(
                            member_id
                            for member_id, coordinate in incident_members[
                                (
                                    reservation.orientation,
                                    member_by_id[claim.member_id].line_id,
                                    endpoint,
                                )
                            ]
                            if abs(coordinate - claim.allocation_coordinate)
                            <= COORD_TOLERANCE
                        )
                        for endpoint in claim.endpoint_anchor_ids
                    )
                    for claim in reservation.claims
                ),
                tuple(
                    claim.allocation_coordinate
                    for claim in sharing_reservation_by_id[reservation.id].claims
                ),
                lane_coordinates,
                tuple(
                    frozenset(
                        (member.line_id, endpoint)
                        for member in (member_by_id[claim.member_id],)
                        for endpoint in (
                            member.source.station_id,
                            member.target.station_id,
                        )
                    )
                    for claim in reservation.claims
                ),
            )
        )
    return {
        boundary: tuple(
            sorted(
                items,
                key=lambda item: (
                    item.realised.coordinate,
                    item.member_rank,
                    str(item.reservation.id),
                ),
            )
        )
        for boundary, items in grouped.items()
    }


def _canvas_reservations(
    plan: RoutePlan,
    graph: MetroGraph,
    sharing_plan: RoutePlan,
) -> dict[CanvasSide, tuple[_BoundaryReservation, ...]]:
    """Materialise canvas-side records without assigning translation ownership."""
    sharing_by_id = {item.id: item for item in sharing_plan.reservations}
    member_rank = {item.id: rank for rank, item in enumerate(plan.members)}
    member_by_id = {item.id: item for item in plan.members}
    incident_members = _endpoint_incident_members(plan)
    fixed_axis_member_ids = {
        assignment.member_id
        for exit_plan in plan.exit_turn_plans
        for assignment in exit_plan.assignments
        if assignment.axis_id is not None
    }
    transition_member_ids = {
        member_id
        for exit_plan in plan.exit_turn_plans
        for transition in exit_plan.lane_transitions
        for member_id in transition.claimant_member_ids
    }
    fixed_member_ids = fixed_axis_member_ids | transition_member_ids
    grouped: dict[CanvasSide, list[_BoundaryReservation]] = {}
    for reservation in plan.reservations:
        if not isinstance(reservation.region, CanvasRegion):
            continue
        fixed_members = tuple(
            member_id
            for member_id in reservation.claimant_member_ids
            if member_id in fixed_member_ids
        )
        lane_coordinates = reservation_claim_lane_coordinates(
            graph, reservation, member_by_id
        )
        occupied_start = min(lane_coordinates)
        occupied_end = max(lane_coordinates)
        inner, inner_blockers = canvas_inner_boundary(graph, reservation)
        outward_positive = reservation.region.side in {
            CanvasSide.BOTTOM,
            CanvasSide.RIGHT,
        }
        region_start = inner if outward_positive else float("-inf")
        region_end = float("inf") if outward_positive else inner
        allocation_axis = (
            DemandAxis.X
            if reservation.orientation is CorridorOrientation.VERTICAL
            else DemandAxis.Y
        )
        realised = _CanvasHalfLine(
            allocation_axis,
            (occupied_start + occupied_end) / 2,
            region_start,
            region_end,
            inner_blockers if outward_positive else ("canvas:auto-extent",),
            ("canvas:auto-extent",) if outward_positive else inner_blockers,
        )
        sharing = sharing_by_id[reservation.id]
        grouped.setdefault(reservation.region.side, []).append(
            _BoundaryReservation(
                reservation,
                realised,
                min(member_rank[item] for item in reservation.claimant_member_ids),
                realised.coordinate - occupied_start,
                occupied_end - realised.coordinate,
                realised.coordinate
                - occupied_start
                + reservation.negative_side_clearance,
                occupied_end
                - realised.coordinate
                + reservation.positive_side_clearance,
                bool(fixed_members),
                False,
                tuple(
                    claim.member_id in transition_member_ids
                    or claim.member_id in fixed_axis_member_ids
                    and claim.segment_rank <= 1 <= claim.segment_end_rank
                    for claim in reservation.claims
                ),
                tuple(not claim.endpoint_anchor_ids for claim in reservation.claims),
                tuple(
                    tuple(
                        frozenset(
                            member_id
                            for member_id, coordinate in incident_members[
                                (
                                    reservation.orientation,
                                    member_by_id[claim.member_id].line_id,
                                    endpoint,
                                )
                            ]
                            if abs(coordinate - claim.allocation_coordinate)
                            <= COORD_TOLERANCE
                        )
                        for endpoint in claim.endpoint_anchor_ids
                    )
                    for claim in reservation.claims
                ),
                tuple(claim.allocation_coordinate for claim in sharing.claims),
                lane_coordinates,
                tuple(
                    frozenset(
                        (member.line_id, endpoint)
                        for member in (member_by_id[claim.member_id],)
                        for endpoint in (
                            member.source.station_id,
                            member.target.station_id,
                        )
                    )
                    for claim in reservation.claims
                ),
            )
        )
    return {
        side: tuple(
            sorted(
                items,
                key=lambda item: (
                    item.realised.coordinate,
                    item.member_rank,
                    str(item.reservation.id),
                ),
            )
        )
        for side, items in grouped.items()
    }


def _boundary_components(
    items: tuple[_BoundaryReservation, ...],
) -> tuple[tuple[_BoundaryReservation, ...], ...]:
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first_rank, first in enumerate(items):
        for second_rank in range(first_rank + 1, len(items)):
            second = items[second_rank]
            if (
                first.reservation.system_id == second.reservation.system_id
                or _claims_overlap(first, second)
            ):
                first_root, second_root = root(first_rank), root(second_rank)
                if first_root != second_root:
                    parent[second_root] = first_root
    components: dict[int, list[_BoundaryReservation]] = {}
    for rank, item in enumerate(items):
        components.setdefault(root(rank), []).append(item)
    return tuple(tuple(component) for component in components.values())


def _shares_channel(first: _BoundaryReservation, second: _BoundaryReservation) -> bool:
    return bool(_shared_channel_offsets(first, second))


def _shared_channel_offsets(
    first: _BoundaryReservation,
    second: _BoundaryReservation,
) -> tuple[float, ...]:
    if first.reservation.direction is not second.reservation.direction:
        return ()
    offsets = {
        (
            first.lane_coordinates[first_rank]
            - first.realised.coordinate
            - second.lane_coordinates[second_rank]
            + second.realised.coordinate
        )
        for first_rank, first_claim in enumerate(first.reservation.claims)
        for second_rank, second_claim in enumerate(second.reservation.claims)
        if min(first_claim.longitudinal_end, second_claim.longitudinal_end)
        - max(first_claim.longitudinal_start, second_claim.longitudinal_start)
        > -CURVE_RADIUS + COORD_TOLERANCE
        and abs(
            first.sharing_coordinates[first_rank]
            - second.sharing_coordinates[second_rank]
        )
        <= COORD_TOLERANCE
        and first.sharing_keys[first_rank].intersection(
            second.sharing_keys[second_rank]
        )
    }
    return tuple(sorted(offsets))


def _separation(first: _BoundaryReservation, second: _BoundaryReservation) -> float:
    if _shares_channel(first, second):
        return 0.0
    return (
        first.positive_footprint
        + max(
            first.reservation.peer_clearance,
            second.reservation.peer_clearance,
        )
        + second.negative_footprint
    )


def _component_region(
    component: tuple[_BoundaryReservation, ...],
) -> tuple[float, float]:
    return (
        min(item.realised.region_start for item in component),
        max(item.realised.region_end for item in component),
    )


def _component_required_width(
    component: tuple[_BoundaryReservation, ...],
    coordinates: tuple[float, ...],
) -> float:
    return max(
        coordinate + item.positive_extent
        for coordinate, item in zip(coordinates, component, strict=True)
    ) - min(
        coordinate - item.negative_extent
        for coordinate, item in zip(coordinates, component, strict=True)
    )


def _component_claim_required_width(
    component: tuple[_BoundaryReservation, ...],
    coordinates: tuple[tuple[float, ...], ...],
) -> float:
    claims = tuple(
        (
            claim.longitudinal_start - CURVE_RADIUS / 2,
            claim.longitudinal_end + CURVE_RADIUS / 2,
            coordinate,
            item.reservation.negative_side_clearance,
            item.reservation.positive_side_clearance,
        )
        for item, reservation_coordinates in zip(component, coordinates, strict=True)
        for claim, coordinate in zip(
            item.reservation.claims, reservation_coordinates, strict=True
        )
    )
    critical_coordinates = tuple(
        sorted({value for start, end, *_rest in claims for value in (start, end)})
    )
    return max(
        (
            max(
                coordinate + positive
                for _start, _end, coordinate, _negative, positive in active
            )
            - min(
                coordinate - negative
                for _start, _end, coordinate, negative, _positive in active
            )
        )
        for longitudinal in critical_coordinates
        for active in (
            tuple(claim for claim in claims if claim[0] <= longitudinal <= claim[1]),
        )
        if active
    )


def _component_lower_coordinates(
    component: tuple[_BoundaryReservation, ...],
) -> tuple[float, ...]:
    coordinates: list[float] = []
    for rank, item in enumerate(component):
        coordinate = max(
            (
                item.realised.region_start + item.negative_extent,
                *(
                    coordinates[prior_rank] + _separation(prior, item)
                    for prior_rank, prior in enumerate(component[:rank])
                    if _claims_overlap(prior, item)
                ),
            ),
        )
        if item.fixed:
            if item.realised.coordinate < coordinate - COORD_TOLERANCE:
                raise EnvelopeSettlementError(
                    "fixed planner geometry cannot satisfy the settled allocation"
                )
            coordinate = item.realised.coordinate
        coordinates.append(coordinate)
    return tuple(coordinates)


def _component_positive_shortfall(
    component: tuple[_BoundaryReservation, ...],
) -> float:
    negative_infinity = float("-inf")
    lower_forms: list[tuple[float, float]] = []
    minimum_translation = 0.0
    maximum_translation = float("inf")
    for rank, item in enumerate(component):
        constant = item.realised.region_start + item.negative_extent
        translated = negative_infinity
        for prior_rank, prior in enumerate(component[:rank]):
            if not _claims_overlap(prior, item):
                continue
            separation = _separation(prior, item)
            prior_constant, prior_translated = lower_forms[prior_rank]
            constant = max(constant, prior_constant + separation)
            translated = max(translated, prior_translated + separation)

        upper = item.realised.region_end - item.positive_extent
        if item.fixed:
            fixed_slope = 1 if item.fixed_movable else 0
            for slope, intercept in enumerate((constant, translated)):
                if intercept == negative_infinity:
                    continue
                if slope == fixed_slope:
                    if intercept > item.realised.coordinate + COORD_TOLERANCE:
                        raise EnvelopeSettlementError(
                            "fixed planner geometry cannot satisfy the settled "
                            "allocation"
                        )
                elif slope < fixed_slope:
                    minimum_translation = max(
                        minimum_translation,
                        intercept - item.realised.coordinate,
                    )
                else:
                    maximum_translation = min(
                        maximum_translation,
                        item.realised.coordinate - intercept,
                    )
            if item.fixed_movable:
                if item.realised.coordinate > upper + COORD_TOLERANCE:
                    raise EnvelopeSettlementError(
                        "fixed planner geometry exceeds its positive keep-out"
                    )
                lower_forms.append((negative_infinity, item.realised.coordinate))
            else:
                minimum_translation = max(
                    minimum_translation,
                    item.realised.coordinate - upper,
                )
                lower_forms.append((item.realised.coordinate, negative_infinity))
            continue

        minimum_translation = max(minimum_translation, constant - upper)
        if translated > upper + COORD_TOLERANCE:
            raise EnvelopeSettlementError(
                "translated planner geometry cannot satisfy the settled allocation"
            )
        lower_forms.append((constant, translated))

    minimum_translation = max(0.0, minimum_translation)
    if minimum_translation > maximum_translation + COORD_TOLERANCE:
        raise EnvelopeSettlementError(
            "positive boundary translation cannot preserve fixed planner geometry"
        )
    return minimum_translation


def _pack_component(
    component: tuple[_BoundaryReservation, ...],
) -> tuple[float, ...]:
    adjacency: dict[int, list[tuple[int, float]]] = {
        rank: [] for rank in range(len(component))
    }
    for first_rank, first in enumerate(component):
        for second_rank in range(first_rank + 1, len(component)):
            second = component[second_rank]
            offsets = _shared_channel_offsets(first, second)
            if not offsets:
                continue
            if any(
                abs(offset - offsets[0]) > COORD_TOLERANCE for offset in offsets[1:]
            ):
                raise EnvelopeSettlementError(
                    "shared reservation lanes require incompatible rigid translations"
                )
            adjacency[first_rank].append((second_rank, offsets[0]))
            adjacency[second_rank].append((first_rank, -offsets[0]))

    groups_list: list[list[int]] = []
    bases = [0.0] * len(component)
    seen: set[int] = set()
    for start in range(len(component)):
        if start in seen:
            continue
        group: list[int] = []
        stack = [start]
        seen.add(start)
        while stack:
            rank = stack.pop()
            group.append(rank)
            for peer, offset in adjacency[rank]:
                expected = bases[rank] + offset
                if peer in seen:
                    if abs(bases[peer] - expected) > COORD_TOLERANCE:
                        raise EnvelopeSettlementError(
                            "shared channel translations are inconsistent"
                        )
                    continue
                seen.add(peer)
                bases[peer] = expected
                stack.append(peer)
        groups_list.append(sorted(group))
    groups = tuple(groups_list)
    group_by_rank = {
        rank: group_rank for group_rank, ranks in enumerate(groups) for rank in ranks
    }
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    preferred: list[float] = []
    for ranks in groups:
        lower = max(
            component[rank].realised.region_start
            + component[rank].negative_extent
            - bases[rank]
            for rank in ranks
        )
        upper = min(
            component[rank].realised.region_end
            - component[rank].positive_extent
            - bases[rank]
            for rank in ranks
        )
        fixed_values = tuple(
            component[rank].realised.coordinate - bases[rank]
            for rank in ranks
            if component[rank].fixed
        )
        if fixed_values and any(
            abs(value - fixed_values[0]) > COORD_TOLERANCE for value in fixed_values[1:]
        ):
            raise EnvelopeSettlementError(
                "shared channel has conflicting fixed planner coordinates"
            )
        if fixed_values:
            lower = max(lower, fixed_values[0])
            upper = min(upper, fixed_values[0])
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        first_rank_in_group = ranks[0]
        preferred.append(
            component[first_rank_in_group].realised.coordinate
            - bases[first_rank_in_group]
        )

    constraints: dict[tuple[int, int], float] = {}
    for first_group, first_ranks in enumerate(groups):
        for second_group in range(first_group + 1, len(groups)):
            second_ranks = groups[second_group]
            requirements = tuple(
                bases[first_rank]
                + _separation(component[first_rank], component[second_rank])
                - bases[second_rank]
                for first_rank in first_ranks
                for second_rank in second_ranks
                if _claims_overlap(component[first_rank], component[second_rank])
            )
            if requirements:
                constraints[first_group, second_group] = max(requirements)
    for group_rank in range(len(groups) - 1, -1, -1):
        upper_bounds[group_rank] = min(
            (
                upper_bounds[group_rank],
                *(
                    upper_bounds[later_group] - requirement
                    for (earlier_group, later_group), requirement in constraints.items()
                    if earlier_group == group_rank
                ),
            )
        )
    group_coordinates: list[float] = []
    for group_rank in range(len(groups)):
        lower = max(
            (
                lower_bounds[group_rank],
                *(
                    group_coordinates[earlier_group] + requirement
                    for (earlier_group, later_group), requirement in constraints.items()
                    if later_group == group_rank
                ),
            )
        )
        coordinate = min(max(preferred[group_rank], lower), upper_bounds[group_rank])
        if coordinate < lower - COORD_TOLERANCE:
            raise EnvelopeSettlementError(
                "settled boundary allocation cannot preserve fixed planner geometry"
            )
        group_coordinates.append(coordinate)

    coordinates = tuple(
        bases[rank] + group_coordinates[group_by_rank[rank]]
        for rank in range(len(component))
    )
    return coordinates


@dataclass(frozen=True, slots=True)
class _BoundaryClaim:
    reservation_rank: int
    claim_rank: int
    lane_rank: int
    coordinate: float
    immutable_coordinate: float
    fixed: bool
    fixed_movable: bool


def _claim_interval_overlaps(
    first: _BoundaryReservation,
    first_rank: int,
    second: _BoundaryReservation,
    second_rank: int,
) -> bool:
    first_claim = first.reservation.claims[first_rank]
    second_claim = second.reservation.claims[second_rank]
    return (
        min(first_claim.longitudinal_end, second_claim.longitudinal_end)
        - max(first_claim.longitudinal_start, second_claim.longitudinal_start)
        > -CURVE_RADIUS + COORD_TOLERANCE
    )


def _claim_nodes(
    component: tuple[_BoundaryReservation, ...],
    coordinate_sign: int = 1,
) -> tuple[_BoundaryClaim, ...]:
    nodes: list[_BoundaryClaim] = []
    for reservation_rank, item in enumerate(component):
        lane_by_claim = {
            claim_rank: lane_rank
            for lane_rank, lane in enumerate(item.reservation.lanes)
            for claim_rank in lane.claim_indices
        }
        nodes.extend(
            _BoundaryClaim(
                reservation_rank,
                claim_rank,
                lane_by_claim[claim_rank],
                coordinate_sign * item.lane_coordinates[claim_rank],
                coordinate_sign * item.sharing_coordinates[claim_rank],
                item.fixed or item.fixed_claims[claim_rank],
                (
                    item.fixed_movable
                    and (
                        not item.fixed_claims[claim_rank]
                        or item.fixed_movable_claims[claim_rank]
                    )
                    if item.fixed
                    else item.fixed_movable_claims[claim_rank]
                ),
            )
            for claim_rank in range(len(item.reservation.claims))
        )
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                node.coordinate,
                component[node.reservation_rank].member_rank,
                node.reservation_rank,
                node.lane_rank,
                node.claim_rank,
            ),
        )
    )


def _claim_nodes_share_channel(
    component: tuple[_BoundaryReservation, ...],
    first: _BoundaryClaim,
    second: _BoundaryClaim,
) -> bool:
    first_item = component[first.reservation_rank]
    second_item = component[second.reservation_rank]
    if (
        not _claim_interval_overlaps(
            first_item, first.claim_rank, second_item, second.claim_rank
        )
        and first.reservation_rank != second.reservation_rank
    ):
        return False
    if first.reservation_rank == second.reservation_rank:
        first_claim = first_item.reservation.claims[first.claim_rank]
        second_claim = second_item.reservation.claims[second.claim_rank]
        overlaps = (
            min(first_claim.longitudinal_end, second_claim.longitudinal_end)
            - max(first_claim.longitudinal_start, second_claim.longitudinal_start)
            > COORD_TOLERANCE
        )
        return first.lane_rank == second.lane_rank and (
            overlaps
            or abs(first.immutable_coordinate - second.immutable_coordinate)
            <= COORD_TOLERANCE
        )
    return (
        first_item.reservation.direction is second_item.reservation.direction
        and abs(first.immutable_coordinate - second.immutable_coordinate)
        <= COORD_TOLERANCE
        and bool(
            first_item.sharing_keys[first.claim_rank].intersection(
                second_item.sharing_keys[second.claim_rank]
            )
        )
    )


def _endpoint_anchor_requires_fixed(
    graph: MetroGraph,
    endpoint_anchor_ids: tuple[str, ...],
    endpoint_continuity_members: tuple[frozenset[EmissionMemberId], ...],
    group_member_ids: set[EmissionMemberId],
) -> bool:
    return any(
        endpoint not in graph.junction_ids
        or len(incident) < 2
        or not incident <= group_member_ids
        for endpoint, incident in zip(
            endpoint_anchor_ids,
            endpoint_continuity_members,
            strict=True,
        )
    )


def _pack_component_claims(  # noqa: C901
    graph: MetroGraph,
    component: tuple[_BoundaryReservation, ...],
    *,
    measure_shortfall: bool = False,
    coordinate_sign: int = 1,
) -> tuple[tuple[float, ...], ...] | float:
    if coordinate_sign not in {-1, 1}:
        raise ValueError("claim packing coordinate sign must be -1 or 1")
    nodes = list(_claim_nodes(component, coordinate_sign))
    parent = list(range(len(nodes)))

    def root(rank: int) -> int:
        while parent[rank] != rank:
            parent[rank] = parent[parent[rank]]
            rank = parent[rank]
        return rank

    for first_rank, first_node in enumerate(nodes):
        for second_rank in range(first_rank + 1, len(nodes)):
            second_node = nodes[second_rank]
            if not _claim_nodes_share_channel(component, first_node, second_node):
                continue
            first_root = root(first_rank)
            second_root = root(second_rank)
            if first_root != second_root:
                parent[second_root] = first_root

    grouped: dict[int, list[int]] = {}
    for rank in range(len(nodes)):
        grouped.setdefault(root(rank), []).append(rank)
    for node_ranks in grouped.values():
        group_member_ids = {
            component[nodes[rank].reservation_rank]
            .reservation.claims[nodes[rank].claim_rank]
            .member_id
            for rank in node_ranks
        }
        for rank in node_ranks:
            node = nodes[rank]
            item = component[node.reservation_rank]
            claim = item.reservation.claims[node.claim_rank]
            incomplete_anchor = _endpoint_anchor_requires_fixed(
                graph,
                claim.endpoint_anchor_ids,
                item.endpoint_continuity_members[node.claim_rank],
                group_member_ids,
            )
            if not incomplete_anchor:
                continue
            nodes[rank] = replace(
                node,
                fixed=True,
                fixed_movable=(
                    node.fixed_movable and item.fixed_movable_claims[node.claim_rank]
                    if node.fixed
                    else item.fixed_movable_claims[node.claim_rank]
                ),
            )
    successors: dict[int, set[int]] = {group: set() for group in grouped}
    indegree = {group: 0 for group in grouped}
    for reservation_rank in range(len(component)):
        lane_nodes: dict[int, list[int]] = {}
        for node_rank, node in enumerate(nodes):
            if node.reservation_rank == reservation_rank:
                lane_nodes.setdefault(node.lane_rank, []).append(node_rank)
        ordered_groups = []
        for _coordinate, _lane_rank, group in sorted(
            (
                sum(nodes[rank].coordinate for rank in node_ranks) / len(node_ranks),
                lane_rank,
                root(node_ranks[0]),
            )
            for lane_rank, node_ranks in lane_nodes.items()
        ):
            if group not in ordered_groups:
                ordered_groups.append(group)
        for earlier_group_id, later_group_id in zip(ordered_groups, ordered_groups[1:]):
            if later_group_id in successors[earlier_group_id]:
                continue
            successors[earlier_group_id].add(later_group_id)
            indegree[later_group_id] += 1

    group_reservations = {
        group: frozenset(nodes[rank].reservation_rank for rank in node_ranks)
        for group, node_ranks in grouped.items()
    }
    group_directions = {
        group: frozenset(
            component[reservation_rank].reservation.direction
            for reservation_rank in reservation_ranks
        )
        for group, reservation_ranks in group_reservations.items()
    }
    ordering_step = graph_offset_step(graph)

    def group_region_lower(group: int) -> float:
        if coordinate_sign > 0:
            return max(
                component[nodes[rank].reservation_rank].realised.region_start
                + component[
                    nodes[rank].reservation_rank
                ].reservation.negative_side_clearance
                for rank in grouped[group]
            )
        return max(
            -component[nodes[rank].reservation_rank].realised.region_end
            + component[
                nodes[rank].reservation_rank
            ].reservation.positive_side_clearance
            for rank in grouped[group]
        )

    def immovable_fixed_coordinate(group: int) -> float | None:
        coordinates = tuple(
            nodes[rank].coordinate
            for rank in grouped[group]
            if nodes[rank].fixed and not nodes[rank].fixed_movable
        )
        if coordinates and any(
            abs(coordinate - coordinates[0]) > COORD_TOLERANCE
            for coordinate in coordinates[1:]
        ):
            raise EnvelopeSettlementError(
                "one physical lane has conflicting fixed planner coordinates"
            )
        return coordinates[0] if coordinates else None

    def ordered_group_separation(first_group: int, second_group: int) -> float | None:
        separations: list[float] = []
        for first_rank in grouped[first_group]:
            for second_rank in grouped[second_group]:
                first = nodes[first_rank]
                second = nodes[second_rank]
                first_item = component[first.reservation_rank]
                second_item = component[second.reservation_rank]
                if not _claim_interval_overlaps(
                    first_item, first.claim_rank, second_item, second.claim_rank
                ):
                    continue
                if first.reservation_rank == second.reservation_rank:
                    separations.append(
                        max(
                            abs(
                                second.immutable_coordinate - first.immutable_coordinate
                            ),
                            ordering_step,
                        )
                    )
                else:
                    separations.append(
                        max(
                            first_item.reservation.peer_clearance,
                            second_item.reservation.peer_clearance,
                        )
                    )
        return max(separations) if separations else None

    def cannot_precede_fixed(candidate: int, fixed_group: int) -> bool:
        fixed_coordinate = immovable_fixed_coordinate(fixed_group)
        separation = ordered_group_separation(candidate, fixed_group)
        return (
            fixed_coordinate is not None
            and separation is not None
            and group_region_lower(candidate) + separation
            > fixed_coordinate + COORD_TOLERANCE
        )

    def available_order(group: int) -> tuple[float, ...]:
        node_ranks = grouped[group]
        cohort = min(
            (
                coordinate_sign * component[reservation_rank].realised.coordinate,
                component[reservation_rank].member_rank,
                reservation_rank,
            )
            for reservation_rank in group_reservations[group]
        )
        return (
            min(nodes[rank].coordinate for rank in node_ranks),
            float(cohort[0]),
            float(cohort[1]),
            float(cohort[2]),
            float(min(nodes[rank].lane_rank for rank in node_ranks)),
            float(node_ranks[0]),
        )

    available = {group for group, degree in indegree.items() if degree == 0}
    ordered_group_ids: list[int] = []
    active_reservations: frozenset[int] = frozenset()
    prior_coordinate: float | None = None
    while available:
        feasible = {
            group
            for group in available
            if not any(
                other != group and cannot_precede_fixed(group, other)
                for other in available
            )
        }
        candidates = feasible or available
        continuing = tuple(
            group
            for group in candidates
            if group_reservations[group].intersection(active_reservations)
            and prior_coordinate is not None
            and min(nodes[rank].coordinate for rank in grouped[group])
            - prior_coordinate
            <= ordering_step + COORD_TOLERANCE
            and not any(
                other != group
                and min(nodes[rank].coordinate for rank in grouped[other])
                <= min(nodes[rank].coordinate for rank in grouped[group])
                + COORD_TOLERANCE
                and group_directions[other].intersection(group_directions[group])
                for other in candidates
            )
        )
        group = min(continuing or tuple(candidates), key=available_order)
        available.remove(group)
        ordered_group_ids.append(group)
        active_reservations = group_reservations[group]
        prior_coordinate = max(nodes[rank].coordinate for rank in grouped[group])
        for successor in successors[group]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                available.add(successor)
    if len(ordered_group_ids) != len(grouped):
        owners = ", ".join(
            str(component[rank].reservation.id)
            for rank in sorted(
                {
                    reservation_rank
                    for group, degree in indegree.items()
                    if degree > 0
                    for reservation_rank in group_reservations[group]
                }
            )
        )
        raise EnvelopeSettlementError(
            "immutable reservation lane order cycles after shared-channel "
            f"union; owners {owners}"
        )
    groups = tuple(grouped[group] for group in ordered_group_ids)
    group_by_node = {
        node_rank: group_rank
        for group_rank, node_ranks in enumerate(groups)
        for node_rank in node_ranks
    }

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    region_lower_bounds: list[float] = []
    region_upper_bounds: list[float] = []
    preferred: list[float] = []
    fixed_forms: list[tuple[int, float] | None] = []
    for node_ranks in groups:
        if coordinate_sign > 0:
            lower = max(
                component[nodes[rank].reservation_rank].realised.region_start
                + component[
                    nodes[rank].reservation_rank
                ].reservation.negative_side_clearance
                for rank in node_ranks
            )
            upper = min(
                component[nodes[rank].reservation_rank].realised.region_end
                - component[
                    nodes[rank].reservation_rank
                ].reservation.positive_side_clearance
                for rank in node_ranks
            )
        else:
            lower = max(
                -component[nodes[rank].reservation_rank].realised.region_end
                + component[
                    nodes[rank].reservation_rank
                ].reservation.positive_side_clearance
                for rank in node_ranks
            )
            upper = min(
                -component[nodes[rank].reservation_rank].realised.region_start
                - component[
                    nodes[rank].reservation_rank
                ].reservation.negative_side_clearance
                for rank in node_ranks
            )
        region_lower_bounds.append(lower)
        region_upper_bounds.append(upper)
        fixed_coordinates = tuple(
            nodes[rank].coordinate for rank in node_ranks if nodes[rank].fixed
        )
        fixed_slopes = {
            int(nodes[rank].fixed_movable) for rank in node_ranks if nodes[rank].fixed
        }
        if fixed_coordinates and any(
            abs(coordinate - fixed_coordinates[0]) > COORD_TOLERANCE
            for coordinate in fixed_coordinates[1:]
        ):
            raise EnvelopeSettlementError(
                "one physical lane has conflicting fixed planner coordinates"
            )
        if len(fixed_slopes) > 1:
            raise EnvelopeSettlementError(
                "one physical lane has conflicting translation ownership"
            )
        fixed_forms.append(
            (next(iter(fixed_slopes)), fixed_coordinates[0])
            if fixed_coordinates
            else None
        )
        if fixed_coordinates:
            lower = max(lower, fixed_coordinates[0])
            upper = min(upper, fixed_coordinates[0])
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        preferred.append(nodes[node_ranks[0]].coordinate)

    constraints: dict[tuple[int, int], float] = {}
    offset_step = graph_offset_step(graph)
    for first_rank, first in enumerate(nodes):
        for second_rank in range(first_rank + 1, len(nodes)):
            second = nodes[second_rank]
            first_group = group_by_node[first_rank]
            second_group = group_by_node[second_rank]
            if first_group == second_group:
                continue
            earlier_node = first
            later_node = second
            if first_group > second_group:
                first_group, second_group = second_group, first_group
                earlier_node, later_node = later_node, earlier_node
            first_item = component[earlier_node.reservation_rank]
            second_item = component[later_node.reservation_rank]
            if not _claim_interval_overlaps(
                first_item,
                earlier_node.claim_rank,
                second_item,
                later_node.claim_rank,
            ):
                continue
            if earlier_node.reservation_rank == later_node.reservation_rank:
                separation = max(
                    abs(
                        later_node.immutable_coordinate
                        - earlier_node.immutable_coordinate
                    ),
                    offset_step,
                )
            else:
                separation = max(
                    first_item.reservation.peer_clearance,
                    second_item.reservation.peer_clearance,
                )
            key = (first_group, second_group)
            constraints[key] = max(constraints.get(key, 0.0), separation)

    if measure_shortfall:
        lower_forms: list[dict[int, float]] = []
        for group_rank in range(len(groups)):
            forms = {0: region_lower_bounds[group_rank]}
            fixed_form = fixed_forms[group_rank]
            if fixed_form is not None:
                slope, intercept = fixed_form
                forms[slope] = max(forms.get(slope, float("-inf")), intercept)
            for (earlier_group, later_group), separation in constraints.items():
                if later_group != group_rank:
                    continue
                for slope, intercept in lower_forms[earlier_group].items():
                    forms[slope] = max(
                        forms.get(slope, float("-inf")), intercept + separation
                    )
            lower_forms.append(forms)

        upper_forms: list[dict[int, float]] = [{} for _group in groups]
        for group_rank in range(len(groups) - 1, -1, -1):
            forms = {1: region_upper_bounds[group_rank]}
            fixed_form = fixed_forms[group_rank]
            if fixed_form is not None:
                slope, intercept = fixed_form
                forms[slope] = min(forms.get(slope, float("inf")), intercept)
            for (earlier_group, later_group), separation in constraints.items():
                if earlier_group != group_rank:
                    continue
                for slope, intercept in upper_forms[later_group].items():
                    forms[slope] = min(
                        forms.get(slope, float("inf")), intercept - separation
                    )
            upper_forms[group_rank] = forms

        minimum_translation = 0.0
        maximum_translation = float("inf")
        for lower_form, upper_form in zip(lower_forms, upper_forms, strict=True):
            for lower_slope, lower_intercept in lower_form.items():
                for upper_slope, upper_intercept in upper_form.items():
                    slope = lower_slope - upper_slope
                    distance = upper_intercept - lower_intercept
                    if slope == 0:
                        if distance < -COORD_TOLERANCE:
                            raise EnvelopeSettlementError(
                                "fixed physical lanes cannot satisfy authored spacing"
                            )
                    elif slope < 0:
                        minimum_translation = max(minimum_translation, distance / slope)
                    else:
                        maximum_translation = min(maximum_translation, distance / slope)
        if minimum_translation > maximum_translation + COORD_TOLERANCE:
            raise EnvelopeSettlementError(
                "positive boundary growth conflicts with fixed physical lanes"
            )
        return max(0.0, minimum_translation)

    for group_rank in range(len(groups) - 1, -1, -1):
        upper_bounds[group_rank] = min(
            (
                upper_bounds[group_rank],
                *(
                    upper_bounds[later_group] - separation
                    for (earlier_group, later_group), separation in constraints.items()
                    if earlier_group == group_rank
                ),
            )
        )
    group_coordinates: list[float] = []
    for group_rank in range(len(groups)):
        lower = max(
            (
                lower_bounds[group_rank],
                *(
                    group_coordinates[earlier_group] + separation
                    for (earlier_group, later_group), separation in constraints.items()
                    if later_group == group_rank
                ),
            )
        )
        coordinate = min(max(preferred[group_rank], lower), upper_bounds[group_rank])
        if coordinate < lower - COORD_TOLERANCE:
            raise EnvelopeSettlementError(
                "settled boundary cannot fit immutable physical lane group "
                f"{group_rank}: requires {lower:.1f}px at or before "
                f"{upper_bounds[group_rank]:.1f}px"
            )
        group_coordinates.append(coordinate)

    coordinates = [[0.0 for _claim in item.reservation.claims] for item in component]
    for node_rank, node in enumerate(nodes):
        coordinates[node.reservation_rank][node.claim_rank] = (
            coordinate_sign * group_coordinates[group_by_node[node_rank]]
        )
    return tuple(tuple(items) for items in coordinates)


def _lane_allocation_evidence(
    item: _BoundaryReservation,
    immutable_reservation: RouteReservation,
    claim_coordinates: tuple[float, ...],
    *,
    minimum_coordinate: float | None = None,
    maximum_coordinate: float | None = None,
) -> tuple[EnvelopeLaneAllocation, ...]:
    evidence: list[EnvelopeLaneAllocation] = []
    for lane_rank, lane in enumerate(immutable_reservation.lanes):
        groups: list[list[int]] = []
        for claim_rank in lane.claim_indices:
            claim = immutable_reservation.claims[claim_rank]
            group = next(
                (
                    ranks
                    for ranks in groups
                    if abs(
                        immutable_reservation.claims[ranks[0]].allocation_coordinate
                        - claim.allocation_coordinate
                    )
                    <= COORD_TOLERANCE
                    and abs(claim_coordinates[ranks[0]] - claim_coordinates[claim_rank])
                    <= COORD_TOLERANCE
                ),
                None,
            )
            if group is None:
                groups.append([claim_rank])
            else:
                group.append(claim_rank)
        for claim_ranks in groups:
            first = immutable_reservation.claims[claim_ranks[0]]
            evidence.append(
                EnvelopeLaneAllocation(
                    lane_rank,
                    tuple(claim_ranks),
                    tuple(
                        dict.fromkeys(
                            immutable_reservation.claims[rank].member_id
                            for rank in claim_ranks
                        )
                    ),
                    first.allocation_coordinate,
                    claim_coordinates[claim_ranks[0]],
                    (
                        item.realised.region_start
                        + item.reservation.negative_side_clearance
                        if minimum_coordinate is None
                        else minimum_coordinate
                    ),
                    (
                        item.realised.region_end
                        - item.reservation.positive_side_clearance
                        if maximum_coordinate is None
                        else maximum_coordinate
                    ),
                )
            )
    return tuple(evidence)


def _boundary_shortfalls(
    graph: MetroGraph, plan: RoutePlan
) -> dict[_AxisBoundary, tuple[float, set[str]]]:
    claims: dict[_AxisBoundary, tuple[float, set[str]]] = {}
    for boundary, reservations in _boundary_reservations(plan, graph).items():
        shortfall = _boundary_shortfall(graph, boundary, reservations)
        if shortfall <= COORD_TOLERANCE:
            continue
        claims[boundary] = (
            shortfall,
            {str(item.reservation.id) for item in reservations},
        )
    return claims


def _boundary_shortfall(
    graph: MetroGraph,
    boundary: _AxisBoundary,
    reservations: tuple[_BoundaryReservation, ...],
) -> float:
    shortfall = 0.0
    for component in _boundary_components(reservations):
        if any(
            _locked_crossing_blockers(graph, boundary, item.realised)
            for item in component
        ):
            continue
        try:
            component_shortfall = _pack_component_claims(
                graph, component, measure_shortfall=True
            )
            assert isinstance(component_shortfall, float)
        except EnvelopeSettlementError:
            if not _component_has_compatibility_owner(graph, component):
                raise
            continue
        shortfall = max(shortfall, component_shortfall)
    return max(0.0, shortfall)


def _component_pinned_section_ids(
    graph: MetroGraph,
    component: tuple[_BoundaryReservation, ...],
) -> tuple[str, ...]:
    blocker_section_ids = {
        blocker_id.split(":", 1)[1]
        for item in component
        for blocker_id in (
            *item.realised.negative_blocker_ids,
            *item.realised.positive_blocker_ids,
        )
        if blocker_id.startswith("section-")
    }
    return tuple(
        sorted(
            section_id
            for section_id in blocker_section_ids
            if (decision := graph.layout_provenance.grid_decision(section_id))
            is not None
            and decision.is_reinference_locked
        )
    )


def _component_has_compatibility_owner(
    graph: MetroGraph,
    component: tuple[_BoundaryReservation, ...],
) -> bool:
    return any(item.fixed for item in component) or bool(
        _component_pinned_section_ids(graph, component)
    )


def _junction_anchors(plan: RoutePlan) -> dict[str, frozenset[str]]:
    section_by_group = {item.id: item.section_id for item in plan.endpoint_groups}
    anchors: dict[str, set[str]] = {}
    for divergence in plan.divergences:
        anchors.setdefault(divergence.junction_id, set()).add(
            section_by_group[divergence.exit_group_id]
        )
    for convergence in plan.convergences:
        anchors.setdefault(convergence.junction_id, set()).add(
            section_by_group[convergence.entry_group_id]
        )
    return {
        junction_id: frozenset(section_ids)
        for junction_id, section_ids in anchors.items()
    }


def _translate_sections(
    graph: MetroGraph,
    plan: RoutePlan,
    section_ids: tuple[str, ...],
    *,
    dx: float,
    dy: float,
) -> None:
    owner_ids = frozenset(section_ids)
    junction_anchors = _junction_anchors(plan)
    shifted_station_ids: set[str] = set()
    for section_id in section_ids:
        section = graph.sections[section_id]
        shifted_station_ids.update(section.station_ids)
        shift_section(graph, section, dx=dx, dy=dy)
    for station in graph.stations.values():
        if station.id in shifted_station_ids:
            continue
        belongs_to_owner = station.section_id in owner_ids
        anchored_to_owners = (
            station.section_id is None
            and (anchors := junction_anchors.get(station.id))
            and anchors.issubset(owner_ids)
        )
        if belongs_to_owner or anchored_to_owners:
            station.x += dx
            station.y += dy


def project_route_plan_origin(plan: RoutePlan, *, dx: float, dy: float) -> RoutePlan:
    """Project absolute reservation evidence through one uniform origin shift."""
    reservations = tuple(
        replace(
            reservation,
            claims=tuple(
                replace(
                    claim,
                    longitudinal_start=claim.longitudinal_start
                    + (
                        dy
                        if reservation.orientation is CorridorOrientation.VERTICAL
                        else dx
                    ),
                    longitudinal_end=claim.longitudinal_end
                    + (
                        dy
                        if reservation.orientation is CorridorOrientation.VERTICAL
                        else dx
                    ),
                    allocation_coordinate=claim.allocation_coordinate
                    + (
                        dx
                        if reservation.orientation is CorridorOrientation.VERTICAL
                        else dy
                    ),
                )
                for claim in reservation.claims
            ),
        )
        for reservation in plan.reservations
    )
    realised = tuple(
        replace(
            item,
            coordinate=item.coordinate
            + (dx if item.allocation_axis is DemandAxis.X else dy),
            longitudinal_start=item.longitudinal_start
            + (dx if item.longitudinal_axis is DemandAxis.X else dy),
            longitudinal_end=item.longitudinal_end
            + (dx if item.longitudinal_axis is DemandAxis.X else dy),
            region_start=item.region_start
            + (dx if item.allocation_axis is DemandAxis.X else dy),
            region_end=item.region_end
            + (dx if item.allocation_axis is DemandAxis.X else dy),
            occupied_start=item.occupied_start
            + (dx if item.allocation_axis is DemandAxis.X else dy),
            occupied_end=item.occupied_end
            + (dx if item.allocation_axis is DemandAxis.X else dy),
        )
        for item in plan.realised_reservations
    )
    return replace(
        plan,
        reservations=reservations,
        realised_reservations=realised,
    )


def _canvas_origin_adjustment(
    proofs: tuple[EnvelopeCapacityProof, ...],
) -> tuple[float, float]:
    """Return the unique positive origin shift required by TOP/LEFT packs."""
    left_edge = min(
        (
            proof.region_start
            for proof in proofs
            if isinstance(proof.region, CanvasRegion)
            and proof.region.side is CanvasSide.LEFT
        ),
        default=0.0,
    )
    top_edge = min(
        (
            proof.region_start
            for proof in proofs
            if isinstance(proof.region, CanvasRegion)
            and proof.region.side is CanvasSide.TOP
        ),
        default=0.0,
    )
    return max(0.0, -left_edge), max(0.0, -top_edge)


def _settle_axis(
    graph: MetroGraph,
    plan: RoutePlan,
    snapshot: _GeometrySnapshot,
    axis: EnvelopeAxis,
    quantum: float,
) -> tuple[EnvelopeTranslation, ...]:
    translations: list[EnvelopeTranslation] = []
    boundaries = sorted(
        {
            boundary
            for reservation in plan.reservations
            if (boundary := _boundary_for_region(reservation.region)) is not None
            and boundary.axis is axis
        },
        key=lambda item: (item.negative, item.positive),
    )
    for boundary in boundaries:
        projected_plan = _project_ledger_translations(graph, plan, snapshot)
        measured_plan = realise_route_reservations(
            projected_plan, graph, blocker_plan=plan
        )
        reservations = _boundary_reservations(measured_plan, graph, plan).get(
            boundary, ()
        )
        shortfall = _boundary_shortfall(graph, boundary, reservations)
        if shortfall <= COORD_TOLERANCE:
            continue
        reservation_ids = {str(item.reservation.id) for item in reservations}
        amount = _quantised_growth(shortfall, quantum)
        owners = tuple(
            section.id
            for section in sorted(graph.sections.values(), key=lambda item: item.id)
            if boundary.starts_after(section)
        )
        if not owners:
            claimants = ", ".join(sorted(reservation_ids))
            raise EnvelopeSettlementError(
                f"reservation boundary {boundary.negative}/{boundary.positive} "
                f"on axis {axis.value} has {shortfall:.2f}px deficit and no "
                f"downstream translation owner; claimants {claimants}"
            )
        _translate_sections(
            graph,
            plan,
            owners,
            dx=amount if axis is EnvelopeAxis.X else 0.0,
            dy=amount if axis is EnvelopeAxis.Y else 0.0,
        )
        translations.append(
            EnvelopeTranslation(
                axis,
                (boundary.negative, boundary.positive),
                amount,
                owners,
                tuple(sorted(reservation_ids)),
            )
        )
    return tuple(translations)


def _component_capacity_proof(
    graph: MetroGraph,
    component: tuple[_BoundaryReservation, ...],
    immutable_plan: RoutePlan,
    *,
    region: CorridorRegion,
) -> EnvelopeCapacityProof:
    immutable_reservation_by_id = {
        item.id: item for item in immutable_plan.reservations
    }
    immutable_realised_by_id = {
        item.reservation_id: item for item in immutable_plan.realised_reservations
    }
    canvas_side = region.side if isinstance(region, CanvasRegion) else None
    coordinate_sign = -1 if canvas_side in {CanvasSide.TOP, CanvasSide.LEFT} else 1
    coordinates = _pack_component_claims(
        graph, component, coordinate_sign=coordinate_sign
    )
    assert isinstance(coordinates, tuple)
    if canvas_side in {CanvasSide.BOTTOM, CanvasSide.RIGHT}:
        region_start = min(item.realised.region_start for item in component)
        region_end = max(
            coordinate + item.reservation.positive_side_clearance
            for item, claim_coordinates in zip(component, coordinates, strict=True)
            for coordinate in claim_coordinates
        )
    elif canvas_side in {CanvasSide.TOP, CanvasSide.LEFT}:
        region_start = min(
            coordinate - item.reservation.negative_side_clearance
            for item, claim_coordinates in zip(component, coordinates, strict=True)
            for coordinate in claim_coordinates
        )
        region_end = max(item.realised.region_end for item in component)
    else:
        region_start, region_end = _component_region(component)
    reservation_allocations = []
    for item, claim_coordinates in zip(component, coordinates, strict=True):
        immutable_reservation = immutable_reservation_by_id[item.reservation.id]
        immutable_claim_by_key = {
            (
                claim.member_id,
                claim.path_rank,
                claim.segment_rank,
                claim.segment_end_rank,
            ): claim
            for claim in immutable_reservation.claims
        }
        immutable_realised = immutable_realised_by_id.get(item.reservation.id)
        original_coordinate = (
            immutable_realised.coordinate
            if immutable_realised is not None
            else (
                min(
                    claim.allocation_coordinate
                    for claim in immutable_reservation.claims
                )
                + max(
                    claim.allocation_coordinate
                    for claim in immutable_reservation.claims
                )
            )
            / 2
        )
        coordinate = (min(claim_coordinates) + max(claim_coordinates)) / 2
        reservation_allocations.append(
            EnvelopeReservationAllocation(
                item.reservation.id,
                item.reservation.system_id,
                item.reservation.reference_id,
                item.reservation.demand_ids,
                item.reservation.direction,
                item.reservation.claimant_member_ids,
                original_coordinate,
                coordinate,
                _lane_allocation_evidence(
                    item,
                    immutable_reservation,
                    claim_coordinates,
                    minimum_coordinate=(
                        item.realised.region_start
                        + item.reservation.negative_side_clearance
                        if canvas_side in {CanvasSide.BOTTOM, CanvasSide.RIGHT}
                        else region_start + item.reservation.negative_side_clearance
                        if canvas_side is not None
                        else None
                    ),
                    maximum_coordinate=(
                        region_end - item.reservation.positive_side_clearance
                        if canvas_side in {CanvasSide.BOTTOM, CanvasSide.RIGHT}
                        else item.realised.region_end
                        - item.reservation.positive_side_clearance
                        if canvas_side is not None
                        else None
                    ),
                ),
                tuple(
                    EnvelopeClaimAllocation(
                        claim.member_id,
                        claim.path_rank,
                        claim.segment_rank,
                        claim.segment_end_rank,
                        item.realised.allocation_axis,
                        immutable_claim_by_key[
                            (
                                claim.member_id,
                                claim.path_rank,
                                claim.segment_rank,
                                claim.segment_end_rank,
                            )
                        ].allocation_coordinate,
                        claim_coordinates[claim_rank],
                    )
                    for claim_rank, claim in enumerate(item.reservation.claims)
                ),
            )
        )
    claimant_set = {
        member_id
        for item in component
        for member_id in item.reservation.claimant_member_ids
    }
    reservation_ids = tuple(item.reservation.id for item in component)
    return EnvelopeCapacityProof(
        EnvelopeAllocationGroupId(
            component[0].realised.allocation_axis,
            region,
            reservation_ids,
        ),
        tuple(dict.fromkeys(item.reservation.system_id for item in component)),
        region,
        component[0].realised.allocation_axis,
        tuple(item.id for item in immutable_plan.members if item.id in claimant_set),
        region_start,
        region_end,
        region_end - region_start,
        _component_claim_required_width(component, coordinates),
        tuple(reservation_allocations),
    )


def _capacity_proofs(
    graph: MetroGraph,
    plan: RoutePlan,
    immutable_plan: RoutePlan,
    excluded_system_ids: frozenset[RouteSystemId] = frozenset(),
) -> tuple[EnvelopeCapacityProof, ...]:
    reservation_by_id = {item.id: item for item in plan.reservations}
    incomplete_system_ids = {
        reservation_by_id[item.reservation_id].system_id
        for item in plan.reservation_diagnostics
        if item.capacity_slack < -COORD_TOLERANCE
    }
    incomplete_system_ids.update(excluded_system_ids)
    proofs: list[EnvelopeCapacityProof] = []
    for boundary, reservations in _boundary_reservations(
        plan, graph, immutable_plan
    ).items():
        for component in _boundary_components(reservations):
            if any(
                item.reservation.system_id in incomplete_system_ids
                for item in component
            ):
                continue
            try:
                proof = _component_capacity_proof(
                    graph,
                    component,
                    immutable_plan,
                    region=component[0].reservation.region,
                )
            except EnvelopeSettlementError:
                if not _component_has_compatibility_owner(graph, component):
                    raise
                continue
            proofs.append(proof)
    for _side, reservations in _canvas_reservations(
        plan, graph, immutable_plan
    ).items():
        for component in _boundary_components(reservations):
            if any(
                item.reservation.system_id in incomplete_system_ids
                for item in component
            ):
                continue
            proofs.append(
                _component_capacity_proof(
                    graph,
                    component,
                    immutable_plan,
                    region=component[0].reservation.region,
                )
            )
    return tuple(proofs)


def _capacity_limitations(
    graph: MetroGraph, plan: RoutePlan
) -> tuple[EnvelopeCapacityLimitation, ...]:
    reservation_by_id = {item.id: item for item in plan.reservations}
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    fixed_reservation_ids = {
        item.reservation.id
        for reservations in _boundary_reservations(plan, graph).values()
        for item in reservations
        if item.fixed
    }
    deficient_by_system: dict[RouteSystemId, list[RouteReservationId]] = {}
    for diagnostic in plan.reservation_diagnostics:
        if diagnostic.capacity_slack >= -COORD_TOLERANCE:
            continue
        reservation = reservation_by_id[diagnostic.reservation_id]
        if _boundary_for_region(reservation.region) is None:
            continue
        deficient_by_system.setdefault(reservation.system_id, []).append(reservation.id)
    for _boundary, reservations in _boundary_reservations(plan, graph).items():
        for component in _boundary_components(reservations):
            try:
                _component_positive_shortfall(component)
                _pack_component_claims(graph, component)
            except EnvelopeSettlementError:
                if not _component_has_compatibility_owner(graph, component):
                    raise
                for item in component:
                    system_reservations = deficient_by_system.setdefault(
                        item.reservation.system_id, []
                    )
                    if item.reservation.id not in system_reservations:
                        system_reservations.append(item.reservation.id)
    limitations: list[EnvelopeCapacityLimitation] = []
    for system_id, reservation_ids in deficient_by_system.items():
        blocker_ids = tuple(
            dict.fromkeys(
                blocker_id
                for reservation_id in reservation_ids
                for blocker_id in (
                    *realised_by_id[reservation_id].negative_blocker_ids,
                    *realised_by_id[reservation_id].positive_blocker_ids,
                )
            )
        )
        blocker_sections = tuple(
            dict.fromkeys(
                blocker_id.split(":", 1)[1]
                for blocker_id in blocker_ids
                if blocker_id.startswith("section-")
            )
        )
        pinned_section_ids = tuple(
            section_id
            for section_id in blocker_sections
            if (decision := graph.layout_provenance.grid_decision(section_id))
            is not None
            and decision.is_reinference_locked
        )
        if not pinned_section_ids and not set(reservation_ids).intersection(
            fixed_reservation_ids
        ):
            claimant_reservations = ", ".join(str(item) for item in reservation_ids)
            raise EnvelopeSettlementError(
                "envelope settlement left an unowned capacity deficit for "
                f"system {system_id}; reservations {claimant_reservations}; blockers "
                f"{', '.join(blocker_ids)}"
            )
        limitations.append(
            EnvelopeCapacityLimitation(
                system_id,
                tuple(reservation_ids),
                blocker_ids,
                pinned_section_ids,
            )
        )
    return tuple(limitations)


def _assert_deficient_boundaries_have_owners(
    graph: MetroGraph, plan: RoutePlan
) -> None:
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    for reservation in plan.reservations:
        boundary = _boundary_for_region(reservation.region)
        realised = realised_by_id.get(reservation.id)
        if (
            boundary is None
            or realised is None
            or realised.capacity_slack >= -COORD_TOLERANCE
        ):
            continue
        if _locked_crossing_blockers(graph, boundary, realised):
            continue
        if any(boundary.starts_after(section) for section in graph.sections.values()):
            continue
        raise EnvelopeSettlementError(
            f"reservation boundary {boundary.negative}/{boundary.positive} "
            f"on axis {boundary.axis.value} has "
            f"{-realised.capacity_slack:.2f}px deficit and no downstream "
            f"translation owner; claimant {reservation.id}"
        )


def settle_route_envelopes(
    graph: MetroGraph,
    plan: RoutePlan,
) -> EnvelopeSettlement:
    """Satisfy final gap claims by one finite pass over each allocation axis.

    Boundaries are visited from negative to positive. A translation at one
    boundary moves every downstream owner, so it cannot shrink an earlier gap;
    a later translation leaves both sides of every later-independent boundary
    unchanged or moves only its positive side. The pass therefore terminates
    after at most one event per claimed row or column boundary.
    """
    snapshot = _snapshot(graph)
    _assert_deficient_boundaries_have_owners(graph, plan)
    measured_plan = realise_route_reservations(plan, graph)
    quantum = graph_offset_step(graph)
    try:
        column_translations = _settle_axis(
            graph, plan, snapshot, EnvelopeAxis.X, quantum
        )
        row_translations = _settle_axis(graph, plan, snapshot, EnvelopeAxis.Y, quantum)
        translations = (*column_translations, *row_translations)
        settled_projected_plan = _project_ledger_translations(graph, plan, snapshot)
        settled_plan = realise_route_reservations(
            settled_projected_plan, graph, blocker_plan=plan
        )
        limitations = _capacity_limitations(graph, settled_plan)
        proofs = _capacity_proofs(
            graph,
            settled_plan,
            measured_plan,
            frozenset(item.system_id for item in limitations),
        )
        origin_adjustment = _canvas_origin_adjustment(proofs)
        if graph.strict and not graph.permissive and limitations:
            limitation = limitations[0]
            reservation = next(
                item
                for item in settled_plan.reservations
                if item.id == limitation.reservation_ids[0]
            )
            realised = next(
                item
                for item in settled_plan.realised_reservations
                if item.reservation_id == reservation.id
            )
            span = reservation.span
            blocker_owner = (
                ", ".join(limitation.pinned_section_ids) or "fixed planner geometry"
            )
            raise EnvelopeSettlementError(
                f"reservation {reservation.id} across columns "
                f"{span.min_column}-{span.max_column} and rows "
                f"{span.min_row}-{span.max_row} is infeasible: blockers "
                f"{', '.join(limitation.blocker_ids)}, required "
                f"{realised.required_width:.2f}px, available "
                f"{realised.available_width:.2f}px, conflicting pin "
                f"{blocker_owner}, "
                "owner #1658, claimant members "
                f"{', '.join(str(item) for item in reservation.claimant_member_ids)}"
            )
    except ValueError as error:
        _restore(graph, snapshot)
        if isinstance(error, EnvelopeSettlementError):
            raise
        raise EnvelopeSettlementError(
            "the final reservation ledger could not be remeasured after its "
            "owned boundary translations"
        ) from error
    except Exception:
        _restore(graph, snapshot)
        raise
    claimed_boundaries = {
        boundary
        for reservation in plan.reservations
        if (boundary := _boundary_for_region(reservation.region)) is not None
    }
    proofed_reservation_ids = {
        reservation.reservation_id
        for proof in proofs
        for reservation in proof.reservations
    }
    return EnvelopeSettlement(
        tuple(translations),
        origin_adjustment,
        len(claimed_boundaries),
        proofs,
        limitations,
        tuple(
            projection
            for projection in _identity_projections(plan, settled_projected_plan)
            if projection.reservation_id not in proofed_reservation_ids
        ),
    )


def route_envelopes_need_settlement(graph: MetroGraph, plan: RoutePlan) -> bool:
    """Whether the immutable ledger publishes an allocation event."""
    measured_plan = realise_route_reservations(plan, graph)
    return bool(
        _boundary_shortfalls(graph, measured_plan)
        or any(
            diagnostic.capacity_slack < -COORD_TOLERANCE
            for diagnostic in measured_plan.reservation_diagnostics
        )
        or any(
            isinstance(reservation.region, CanvasRegion)
            for reservation in plan.reservations
        )
    )


def assert_route_envelopes_satisfied(graph: MetroGraph, plan: RoutePlan) -> None:
    """Reject a strict render whose final reservation ledger has a deficit."""
    if not graph.strict or graph.permissive:
        return
    reservation_by_id = {item.id: item for item in plan.reservations}
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    diagnostic = next(
        (
            item
            for item in plan.reservation_diagnostics
            if item.capacity_slack < -COORD_TOLERANCE
        ),
        None,
    )
    if diagnostic is None:
        return
    reservation = reservation_by_id[diagnostic.reservation_id]
    realised = realised_by_id[diagnostic.reservation_id]
    blocker_ids = tuple(
        dict.fromkeys((*realised.negative_blocker_ids, *realised.positive_blocker_ids))
    )
    blocker_sections = tuple(
        dict.fromkeys(
            blocker_id.split(":", 1)[1]
            for blocker_id in blocker_ids
            if blocker_id.startswith("section-")
        )
    )
    pinned = tuple(
        section_id
        for section_id in blocker_sections
        if (decision := graph.layout_provenance.grid_decision(section_id)) is not None
        and decision.is_reinference_locked
    )
    span = reservation.span
    raise EnvelopeSettlementError(
        f"reservation {reservation.id} across columns "
        f"{span.min_column}-{span.max_column} and rows "
        f"{span.min_row}-{span.max_row} is infeasible: blockers "
        f"{', '.join(blocker_ids)}, required {realised.required_width:.2f}px, "
        f"available {realised.available_width:.2f}px, conflicting pin "
        f"{', '.join(pinned) if pinned else 'none'}, claimant members "
        f"{', '.join(str(item) for item in reservation.claimant_member_ids)}"
    )
