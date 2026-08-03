"""Bounded monotone settlement of final row and column route envelopes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nf_metro.layout.constants import (
    BUNDLE_TO_BUNDLE_CLEARANCE,
    COORD_TOLERANCE,
    graph_offset_step,
)
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import (
    DemandAxis,
    EmissionMemberId,
    RoutePlan,
    RouteSystemId,
)
from nf_metro.layout.route_reservations import (
    ColumnGapRegion,
    RealisedRouteReservation,
    RouteReservation,
    RouteReservationId,
    RowGapRegion,
    realise_route_reservations,
)
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
    boundary_count: int
    capacity_proofs: tuple[EnvelopeCapacityProof, ...]
    capacity_limitations: tuple[EnvelopeCapacityLimitation, ...]


@dataclass(frozen=True, slots=True)
class EnvelopeClaimAllocation:
    """One immutable reservation claim projected into its settled band."""

    member_id: EmissionMemberId
    path_rank: int
    segment_rank: int
    segment_end_rank: int
    axis: DemandAxis
    coordinate: float


@dataclass(frozen=True, slots=True)
class EnvelopeCapacityProof:
    """Measured evidence that one reservation owns a feasible final band."""

    reservation_id: RouteReservationId
    system_id: RouteSystemId
    boundary: tuple[int, int]
    axis: DemandAxis
    claimant_member_ids: tuple[EmissionMemberId, ...]
    region_start: float
    region_end: float
    available_width: float
    required_width: float
    coordinate: float
    translation_amount: float
    allocations: tuple[EnvelopeClaimAllocation, ...]


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


def _quantised_growth(shortfall: float, quantum: float) -> float:
    if shortfall <= COORD_TOLERANCE:
        return 0.0
    # One re-emission can transfer a route across one lane slot. Include that
    # bounded displacement in the event instead of reopening the boundary.
    return math.ceil((shortfall + quantum) / quantum) * quantum


def _boundary_for_region(region: object) -> _AxisBoundary | None:
    if isinstance(region, RowGapRegion):
        return _AxisBoundary(EnvelopeAxis.Y, region.upper_row, region.lower_row)
    if isinstance(region, ColumnGapRegion):
        return _AxisBoundary(EnvelopeAxis.X, region.left_column, region.right_column)
    return None


def _boundary_shortfalls(
    graph: MetroGraph, plan: RoutePlan
) -> dict[_AxisBoundary, tuple[float, set[str]]]:
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    compatibility_systems = {
        item.system_id for item in plan.convergence_plans if not item.owns_geometry
    }
    member_by_id = {item.id: item for item in plan.members}
    reservations_by_boundary: dict[_AxisBoundary, list[RouteReservation]] = {}
    for reservation in plan.reservations:
        boundary = _boundary_for_region(reservation.region)
        realised = realised_by_id.get(reservation.id)
        if boundary is None or realised is None:
            continue
        reservations_by_boundary.setdefault(boundary, []).append(reservation)
    claims: dict[_AxisBoundary, tuple[float, set[str]]] = {}
    for boundary, reservations in reservations_by_boundary.items():
        reservation_ids: set[str] = set()
        shortfall = 0.0
        for reservation in reservations:
            realised = realised_by_id[reservation.id]
            reservation_ids.add(str(reservation.id))
            shortfall = max(shortfall, -realised.capacity_slack)
        for rank, first in enumerate(reservations):
            for second in reservations[rank + 1 :]:
                opposite = first.direction.value != second.direction.value
                if not opposite:
                    continue
                overlapping = any(
                    min(first_claim.longitudinal_end, second_claim.longitudinal_end)
                    - max(
                        first_claim.longitudinal_start,
                        second_claim.longitudinal_start,
                    )
                    > COORD_TOLERANCE
                    for first_claim in first.claims
                    for second_claim in second.claims
                )
                if not overlapping:
                    continue
                compatibility_pair = bool(
                    {
                        first.system_id,
                        second.system_id,
                    }.intersection(compatibility_systems)
                )
                if compatibility_pair:
                    available_width = min(
                        realised_by_id[first.id].available_width,
                        realised_by_id[second.id].available_width,
                    )
                    combined_required = (
                        first.minimum_width
                        + second.minimum_width
                        + BUNDLE_TO_BUNDLE_CLEARANCE
                    )
                    shortfall = max(
                        shortfall,
                        combined_required - available_width,
                    )
                first_coordinate = sum(
                    item.allocation_coordinate for item in first.claims
                ) / len(first.claims)
                second_coordinate = sum(
                    item.allocation_coordinate for item in second.claims
                ) / len(second.claims)
                separation = abs(first_coordinate - second_coordinate)
                shares_line = bool(
                    {
                        member_by_id[item.member_id].edge.line_id
                        for item in first.claims
                    }.intersection(
                        member_by_id[item.member_id].edge.line_id
                        for item in second.claims
                    )
                )
                if (
                    not (first.system_id == second.system_id and shares_line)
                    and separation < BUNDLE_TO_BUNDLE_CLEARANCE - COORD_TOLERANCE
                ):
                    shortfall = max(
                        shortfall,
                        2 * (BUNDLE_TO_BUNDLE_CLEARANCE - separation),
                    )
        shortfall = max(
            0.0,
            shortfall,
        )
        if shortfall <= COORD_TOLERANCE:
            continue
        claims[boundary] = (shortfall, reservation_ids)
    return claims


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


def _settle_axis(
    graph: MetroGraph,
    plan: RoutePlan,
    axis: EnvelopeAxis,
    claims: dict[_AxisBoundary, tuple[float, set[str]]],
    quantum: float,
) -> tuple[EnvelopeTranslation, ...]:
    translations: list[EnvelopeTranslation] = []
    boundaries = sorted(
        (item for item in claims if item.axis is axis),
        key=lambda item: (item.negative, item.positive),
    )
    for boundary in boundaries:
        shortfall, reservation_ids = claims[boundary]
        amount = _quantised_growth(shortfall, quantum)
        owners = tuple(
            section.id
            for section in sorted(graph.sections.values(), key=lambda item: item.id)
            if boundary.starts_after(section)
        )
        if not owners:
            reservations = ", ".join(sorted(reservation_ids))
            raise EnvelopeSettlementError(
                f"reservation boundary {boundary.negative}/{boundary.positive} "
                f"on axis {axis.value} has {shortfall:.2f}px deficit and no "
                f"downstream translation owner; claimants {reservations}"
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


def _settled_coordinate(
    reservation: RouteReservation,
    realised: RealisedRouteReservation,
) -> float:
    half_bundle = reservation.bundle_width / 2
    lower = realised.region_start + reservation.negative_side_clearance + half_bundle
    upper = realised.region_end - reservation.positive_side_clearance - half_bundle
    if lower > upper + COORD_TOLERANCE:
        raise EnvelopeSettlementError(
            f"reservation {reservation.id} has no coordinate in its settled band: "
            f"required {realised.required_width:.2f}px, available "
            f"{realised.available_width:.2f}px"
        )
    return min(max(realised.coordinate, lower), upper)


def _capacity_proofs(
    plan: RoutePlan,
    translations: tuple[EnvelopeTranslation, ...],
) -> tuple[EnvelopeCapacityProof, ...]:
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    translated_by_boundary = {
        (item.axis, item.boundary): item.amount for item in translations
    }
    reservation_by_id = {item.id: item for item in plan.reservations}
    incomplete_system_ids = {
        reservation_by_id[item.reservation_id].system_id
        for item in plan.reservation_diagnostics
        if item.capacity_slack < -COORD_TOLERANCE
    }
    proofs: list[EnvelopeCapacityProof] = []
    for reservation in plan.reservations:
        boundary = _boundary_for_region(reservation.region)
        realised = realised_by_id.get(reservation.id)
        if (
            boundary is None
            or realised is None
            or reservation.system_id in incomplete_system_ids
        ):
            continue
        if realised.capacity_slack < -COORD_TOLERANCE:
            continue
        coordinate = _settled_coordinate(reservation, realised)
        delta = coordinate - realised.coordinate
        proofs.append(
            EnvelopeCapacityProof(
                reservation.id,
                reservation.system_id,
                (boundary.negative, boundary.positive),
                realised.allocation_axis,
                reservation.claimant_member_ids,
                realised.region_start,
                realised.region_end,
                realised.available_width,
                realised.required_width,
                coordinate,
                translated_by_boundary.get(
                    (boundary.axis, (boundary.negative, boundary.positive)), 0.0
                ),
                tuple(
                    EnvelopeClaimAllocation(
                        claim.member_id,
                        claim.path_rank,
                        claim.segment_rank,
                        claim.segment_end_rank,
                        realised.allocation_axis,
                        claim.allocation_coordinate + delta,
                    )
                    for claim in reservation.claims
                ),
            )
        )
    return tuple(proofs)


def _capacity_limitations(
    graph: MetroGraph, plan: RoutePlan
) -> tuple[EnvelopeCapacityLimitation, ...]:
    reservation_by_id = {item.id: item for item in plan.reservations}
    realised_by_id = {item.reservation_id: item for item in plan.realised_reservations}
    deficient_by_system: dict[RouteSystemId, list[RouteReservationId]] = {}
    for diagnostic in plan.reservation_diagnostics:
        if diagnostic.capacity_slack >= -COORD_TOLERANCE:
            continue
        reservation = reservation_by_id[diagnostic.reservation_id]
        if _boundary_for_region(reservation.region) is None:
            continue
        deficient_by_system.setdefault(reservation.system_id, []).append(reservation.id)
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
        if not pinned_section_ids:
            reservations = ", ".join(str(item) for item in reservation_ids)
            raise EnvelopeSettlementError(
                "envelope settlement left an unowned capacity deficit for "
                f"system {system_id}; reservations {reservations}; blockers "
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
    initial_claims = _boundary_shortfalls(graph, measured_plan)
    quantum = graph_offset_step(graph)
    try:
        column_translations = _settle_axis(
            graph, measured_plan, EnvelopeAxis.X, initial_claims, quantum
        )
        column_settled_plan = realise_route_reservations(measured_plan, graph)
        row_claims = _boundary_shortfalls(graph, column_settled_plan)
        row_translations = _settle_axis(
            graph, column_settled_plan, EnvelopeAxis.Y, row_claims, quantum
        )
        translations = (*column_translations, *row_translations)
        settled_plan = realise_route_reservations(measured_plan, graph)
        proofs = _capacity_proofs(settled_plan, tuple(translations))
        limitations = _capacity_limitations(graph, settled_plan)
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
        item for item in initial_claims if item.axis is EnvelopeAxis.X
    } | {item for item in row_claims if item.axis is EnvelopeAxis.Y}
    return EnvelopeSettlement(
        tuple(translations), len(claimed_boundaries), proofs, limitations
    )


def route_envelopes_need_settlement(graph: MetroGraph, plan: RoutePlan) -> bool:
    """Whether the immutable ledger publishes a row or column boundary event."""
    measured_plan = realise_route_reservations(plan, graph)
    return bool(_boundary_shortfalls(graph, measured_plan))


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
