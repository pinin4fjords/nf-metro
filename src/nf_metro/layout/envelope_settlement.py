"""Monotone row and column envelope settlement around route reservations.

Local station geometry, section bboxes, header keep-outs, and the immutable
``RouteReservation`` ledger are all final before this runs.  Settlement owns one
thing only: the global row and column offsets needed to give every reserved
corridor the width its ledger entry requires.  It never resizes a box, moves a
station inside its section, or revisits a route decision.

Termination is structural rather than iterative.  Each adjacent-index boundary
(the gap between row ``b-1`` and row ``b``, or between column ``b-1`` and column
``b``) is visited exactly once in ascending order, and translating everything
from ``b`` onward has three effects and no others:

* boundaries before ``b`` keep both blockers stationary, so they are unchanged;
* boundary ``b`` widens by exactly the translated amount;
* boundaries after ``b`` move both blockers together, so they are unchanged.

A section spanning across ``b`` stays where it is, which can only increase its
distance to the content below.  No separation therefore ever decreases, the
sweep is a single directional pass over a finite set of boundaries, and a second
run finds no deficit and writes nothing.

A boundary whose far-side blocker cannot be translated -- a row-spanning section
straddling the corridor, or a blocker pinned above the boundary -- is not an
envelope-allocation problem.  Settlement records an attributed obstruction
naming the blocker instead of translating geometry that would not help.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from nf_metro.layout.constants import COORD_TOLERANCE, SETTLEMENT_QUANTUM
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import (
    CONVERGENCE_COMPAT_CHAINED_SYSTEM,
    CONVERGENCE_COMPAT_OPPOSING_OPENINGS,
    CONVERGENCE_COMPAT_SHARED_FEEDERS,
    CONVERGENCE_COMPAT_SHARED_TRUNK,
    CONVERGENCE_COMPAT_UNOWNED_MEMBER,
    CONVERGENCE_COMPAT_UNOWNED_MEMBERS,
    ConvergenceDisposition,
    ConvergencePlan,
    ConvergencePlanId,
    DemandAxis,
    EmissionMemberId,
    RoutePlan,
    RoutePlanDiagnostic,
    RouteSystemId,
)
from nf_metro.layout.route_reservations import (
    SECTION_HEADER_BLOCKER,
    SECTION_LEFT_BLOCKER,
    ColumnGapRegion,
    ReservationCoordinateTranslation,
    RouteReservation,
    RouteReservationId,
    RowGapRegion,
    realise_reservation,
)
from nf_metro.parser.model import MetroGraph, Section


class SettlementAxis(Enum):
    """The grid axis a boundary separates, and the coordinate it translates."""

    ROW = "row"
    COLUMN = "column"


@dataclass(frozen=True, slots=True)
class SettlementTranslation:
    """One applied global translation of everything from *boundary* onward."""

    axis: SettlementAxis
    boundary: int
    coordinate: float
    amount: float
    section_ids: tuple[str, ...]
    reservation_ids: tuple[RouteReservationId, ...]
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(sorted(self.blocker_ids))
        return (
            f"{self.axis.value} boundary {self.boundary} widened by "
            f"{self.amount:.2f}px for {len(self.reservation_ids)} corridor "
            f"claim(s) owned by {claimants}, "
            f"held from below by {blockers}"
        )


@dataclass(frozen=True, slots=True)
class SettlementObstruction:
    """A deficit no translation this stage owns is able to close."""

    axis: SettlementAxis
    boundary: int
    deficit: float
    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]
    blocking_section_ids: tuple[str, ...]

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(self.blocking_section_ids)
        return (
            f"{self.axis.value} boundary {self.boundary} is short of the "
            f"corridor claimed by {claimants} by {self.deficit:.2f}px, but "
            f"section(s) {blockers} bound the far side without belonging to "
            f"the translated {self.axis.value}s, so widening the boundary "
            f"cannot supply it"
        )


@dataclass(frozen=True, slots=True)
class CompatibilityExitEvidence:
    """Why one compatible convergence system is outside settlement ownership."""

    system_id: RouteSystemId
    convergence_plan_ids: tuple[ConvergencePlanId, ...]
    compatibility_reasons: tuple[str, ...]
    reservation_ids: tuple[RouteReservationId, ...]
    minimum_capacity_slack: float | None
    obstruction_reservation_ids: tuple[RouteReservationId, ...]
    blocking_section_ids: tuple[str, ...]
    owner_kinds: tuple[str, ...]
    owner_issue: str = "#1658"

    @property
    def message(self) -> str:
        reasons = "; ".join(self.compatibility_reasons)
        owners = ", ".join(self.owner_kinds)
        if self.obstruction_reservation_ids:
            blockers = ", ".join(self.blocking_section_ids)
            outcome = (
                f"{len(self.obstruction_reservation_ids)} corridor claim(s) remain "
                f"bounded by spanning section(s) {blockers}, so global row or "
                "column translation cannot supply their separation"
            )
        elif self.minimum_capacity_slack is None:
            outcome = "the system publishes no row or column corridor claim"
        else:
            outcome = (
                f"all {len(self.reservation_ids)} row or column corridor claim(s) "
                f"fit, with minimum capacity slack "
                f"{self.minimum_capacity_slack:.2f}px"
            )
        return (
            f"convergence system {self.system_id}: {outcome}. Compatibility "
            f"reason: {reasons}. The remaining limitation belongs to {owners} "
            f"in {self.owner_issue}, not envelope allocation"
        )


@dataclass(frozen=True, slots=True)
class EnvelopeSettlement:
    """What one settlement pass moved, and what it could not."""

    translations: tuple[SettlementTranslation, ...]
    obstructions: tuple[SettlementObstruction, ...]
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = ()
    compatibility_exits: tuple[CompatibilityExitEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class _GeometrySnapshot:
    """Mutable coordinates owned by envelope settlement."""

    sections: dict[str, tuple[float, float]]
    stations: dict[str, tuple[float, float]]
    ports: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class _Axis:
    """The per-axis differences between row and column settlement."""

    axis: SettlementAxis
    boundary_of: Callable[[RouteReservation], int]
    start_index: Callable[[Section], int]
    coordinate_of: Callable[[Section], float]
    blocker_prefix: str
    translate: Callable[[MetroGraph, int, float], None]


def _translate_rows(graph: MetroGraph, boundary: int, amount: float) -> None:
    for section in graph.sections.values():
        if section.grid_row >= boundary:
            shift_section(graph, section, dy=amount)


def _translate_columns(graph: MetroGraph, boundary: int, amount: float) -> None:
    for section in graph.sections.values():
        if section.grid_col >= boundary:
            shift_section(graph, section, dx=amount)


_ROW_AXIS = _Axis(
    SettlementAxis.ROW,
    lambda reservation: _row_region(reservation).lower_row,
    lambda section: section.grid_row,
    lambda section: section.bbox_y,
    SECTION_HEADER_BLOCKER,
    _translate_rows,
)

_COLUMN_AXIS = _Axis(
    SettlementAxis.COLUMN,
    lambda reservation: _column_region(reservation).right_column,
    lambda section: section.grid_col,
    lambda section: section.bbox_x,
    SECTION_LEFT_BLOCKER,
    _translate_columns,
)


_COMPATIBILITY_OWNER_BY_REASON = {
    CONVERGENCE_COMPAT_SHARED_TRUNK: ("plan-driven shared-channel emission",),
    CONVERGENCE_COMPAT_SHARED_FEEDERS: ("plan-driven shared-channel emission",),
    CONVERGENCE_COMPAT_OPPOSING_OPENINGS: ("plan-driven opposing-opening emission",),
    CONVERGENCE_COMPAT_UNOWNED_MEMBER: ("plan-driven whole-system emission",),
    CONVERGENCE_COMPAT_UNOWNED_MEMBERS: ("plan-driven whole-system emission",),
    CONVERGENCE_COMPAT_CHAINED_SYSTEM: ("plan-driven chained-convergence emission",),
}


def _row_region(reservation: RouteReservation) -> RowGapRegion:
    assert isinstance(reservation.region, RowGapRegion)
    return reservation.region


def _column_region(reservation: RouteReservation) -> ColumnGapRegion:
    assert isinstance(reservation.region, ColumnGapRegion)
    return reservation.region


def _reservations_on(
    plan: RoutePlan, region_type: type
) -> tuple[RouteReservation, ...]:
    return tuple(
        reservation
        for reservation in plan.reservations
        if isinstance(reservation.region, region_type)
    )


def _obstructing_sections(
    graph: MetroGraph, axis: _Axis, boundary: int, blocker_ids: Iterable[str]
) -> tuple[str, ...]:
    """Far-side blockers that stay put when *boundary* onward translates.

    A blocker outside the translated band still bounds the corridor after the
    translation, so the boundary gains nothing.  Row-spanning sections that
    straddle the boundary are the usual case.
    """
    stuck: list[str] = []
    for blocker_id in blocker_ids:
        section_id = blocker_id.removeprefix(f"{axis.blocker_prefix}:")
        section = graph.sections.get(section_id)
        if section is None or axis.start_index(section) < boundary:
            stuck.append(section_id)
    return tuple(sorted(set(stuck)))


def _settle_axis(
    graph: MetroGraph,
    plan: RoutePlan,
    reservations: tuple[RouteReservation, ...],
    axis: _Axis,
    prior_translations: tuple[SettlementTranslation, ...] = (),
) -> tuple[list[SettlementTranslation], list[SettlementObstruction]]:
    by_boundary: dict[int, list[RouteReservation]] = {}
    for reservation in reservations:
        by_boundary.setdefault(axis.boundary_of(reservation), []).append(reservation)

    translations: list[SettlementTranslation] = []
    obstructions: list[SettlementObstruction] = []
    coordinate_translations = list(
        _reservation_coordinate_translations(prior_translations, plan)
    )
    for boundary in sorted(by_boundary):
        projected_prefix = tuple(coordinate_translations)
        claims: list[tuple[float, RouteReservation, tuple[str, ...]]] = []
        for reservation in sorted(by_boundary[boundary], key=lambda item: item.id):
            realised = realise_reservation(
                graph,
                reservation,
                coordinate_translations=projected_prefix,
            )
            if realised is None:
                continue
            deficit = -realised.capacity_slack
            if deficit <= COORD_TOLERANCE:
                continue
            stuck = _obstructing_sections(
                graph, axis, boundary, realised.positive_blocker_ids
            )
            if stuck:
                obstructions.append(
                    SettlementObstruction(
                        axis.axis,
                        boundary,
                        deficit,
                        reservation.id,
                        reservation.claimant_member_ids,
                        realised.positive_blocker_ids,
                        stuck,
                    )
                )
                continue
            claims.append((deficit, reservation, realised.negative_blocker_ids))
        if not claims:
            continue
        deficit, _reservation, _blockers = max(claims, key=lambda item: item[0])
        amount = math.ceil(deficit / SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM
        owners = tuple(
            section.id
            for section in sorted(graph.sections.values(), key=lambda item: item.id)
            if axis.start_index(section) >= boundary
        )
        if not owners:
            raise ValueError(
                f"{axis.axis.value} boundary {boundary} has no translation owner"
            )
        coordinate = min(
            axis.coordinate_of(graph.sections[section_id]) for section_id in owners
        )
        axis.translate(graph, boundary, amount)
        reservations = tuple(item[1] for item in claims)
        translations.append(
            SettlementTranslation(
                axis.axis,
                boundary,
                coordinate,
                amount,
                owners,
                tuple(item.id for item in reservations),
                tuple(
                    sorted(
                        {
                            member_id
                            for item in reservations
                            for member_id in item.claimant_member_ids
                        }
                    )
                ),
                tuple(sorted({blocker for _d, _r, got in claims for blocker in got})),
            )
        )
        coordinate_translations.append(
            _reservation_coordinate_translation(translations[-1], plan)
        )
    return translations, obstructions


def _reservation_coordinate_translation(
    translation: SettlementTranslation,
    plan: RoutePlan,
) -> ReservationCoordinateTranslation:
    section_ids = frozenset(translation.section_ids)
    fully_owned: list[EmissionMemberId] = []
    crossing: list[EmissionMemberId] = []
    for member in plan.members:
        source_owned = member.source.section_id in section_ids
        target_owned = member.target.section_id in section_ids
        if source_owned and target_owned:
            fully_owned.append(member.id)
        elif source_owned != target_owned:
            crossing.append(member.id)
    return ReservationCoordinateTranslation(
        DemandAxis.Y if translation.axis is SettlementAxis.ROW else DemandAxis.X,
        translation.coordinate,
        translation.amount,
        tuple(fully_owned),
        tuple(crossing),
    )


def _reservation_coordinate_translations(
    translations: tuple[SettlementTranslation, ...],
    plan: RoutePlan,
) -> tuple[ReservationCoordinateTranslation, ...]:
    return tuple(
        _reservation_coordinate_translation(translation, plan)
        for translation in translations
    )


def _snapshot_geometry(graph: MetroGraph) -> _GeometrySnapshot:
    return _GeometrySnapshot(
        {
            section_id: (section.bbox_x, section.bbox_y)
            for section_id, section in graph.sections.items()
        },
        {
            station_id: (station.x, station.y)
            for station_id, station in graph.stations.items()
        },
        {port_id: (port.x, port.y) for port_id, port in graph.ports.items()},
    )


def _restore_geometry(graph: MetroGraph, snapshot: _GeometrySnapshot) -> None:
    for section_id, (x, y) in snapshot.sections.items():
        section = graph.sections[section_id]
        section.bbox_x = x
        section.bbox_y = y
    for station_id, (x, y) in snapshot.stations.items():
        station = graph.stations[station_id]
        station.x = x
        station.y = y
    for port_id, (x, y) in snapshot.ports.items():
        port = graph.ports[port_id]
        port.x = x
        port.y = y


def _compatibility_exit_evidence(
    graph: MetroGraph,
    plan: RoutePlan,
    obstructions: tuple[SettlementObstruction, ...],
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...],
) -> tuple[CompatibilityExitEvidence, ...]:
    plans_by_system: dict[RouteSystemId, list[ConvergencePlan]] = {}
    for convergence in plan.convergence_plans:
        if (
            convergence.disposition is ConvergenceDisposition.LEGACY
            and convergence.legacy_reason in _COMPATIBILITY_OWNER_BY_REASON
        ):
            plans_by_system.setdefault(convergence.system_id, []).append(convergence)

    obstruction_by_reservation = {item.reservation_id: item for item in obstructions}
    evidence: list[CompatibilityExitEvidence] = []
    for system_id in sorted(plans_by_system, key=str):
        convergence_plans = plans_by_system[system_id]
        reservations = tuple(
            item
            for item in plan.reservations
            if item.system_id == system_id
            and isinstance(item.region, RowGapRegion | ColumnGapRegion)
        )
        realised = tuple(
            (item, result)
            for item in reservations
            if (
                result := realise_reservation(
                    graph,
                    item,
                    coordinate_translations=coordinate_translations,
                )
            )
            is not None
        )
        deficits = tuple(
            (item, result)
            for item, result in realised
            if result.capacity_slack < -COORD_TOLERANCE
        )
        if any(item.id not in obstruction_by_reservation for item, _got in deficits):
            continue
        relevant_obstructions = tuple(
            obstruction_by_reservation[item.id] for item, _got in deficits
        )
        reasons = tuple(
            dict.fromkeys(
                item.legacy_reason
                for item in convergence_plans
                if item.legacy_reason is not None
            )
        )
        evidence.append(
            CompatibilityExitEvidence(
                system_id,
                tuple(item.id for item in convergence_plans),
                reasons,
                tuple(item.id for item in reservations),
                min(
                    (result.capacity_slack for _item, result in realised),
                    default=None,
                ),
                tuple(item.id for item, _result in deficits),
                tuple(
                    sorted(
                        {
                            section_id
                            for item in relevant_obstructions
                            for section_id in item.blocking_section_ids
                        }
                    )
                ),
                tuple(
                    dict.fromkeys(
                        owner
                        for reason in reasons
                        for owner in _COMPATIBILITY_OWNER_BY_REASON[reason]
                    )
                ),
            )
        )
    return tuple(evidence)


def attach_compatibility_exit_diagnostics(
    plan: RoutePlan, settlement: EnvelopeSettlement
) -> RoutePlan:
    """Publish the final ownership boundary without changing route decisions."""
    diagnostics = tuple(
        item for item in plan.diagnostics if item.code != "convergence-settlement-exit"
    ) + tuple(
        RoutePlanDiagnostic(
            None,
            "convergence-settlement-exit",
            item.message,
            blocking=False,
        )
        for item in settlement.compatibility_exits
    )
    return replace(plan, diagnostics=diagnostics)


def settle_route_envelopes(graph: MetroGraph, plan: RoutePlan) -> EnvelopeSettlement:
    """Widen row and column boundaries until every reserved corridor fits.

    Mutates *graph* in place, translating whole rows and whole columns only.
    Returns what moved and any deficit outside this stage's ownership.
    """
    snapshot = _snapshot_geometry(graph)
    try:
        row_translations, row_obstructions = _settle_axis(
            graph,
            plan,
            _reservations_on(plan, RowGapRegion),
            _ROW_AXIS,
        )
        column_translations, column_obstructions = _settle_axis(
            graph,
            plan,
            _reservations_on(plan, ColumnGapRegion),
            _COLUMN_AXIS,
            tuple(row_translations),
        )
        translations = tuple(row_translations + column_translations)
        obstructions = tuple(row_obstructions + column_obstructions)
        coordinate_translations = _reservation_coordinate_translations(
            translations, plan
        )
        compatibility_exits = _compatibility_exit_evidence(
            graph,
            plan,
            obstructions,
            coordinate_translations,
        )
        return EnvelopeSettlement(
            translations,
            obstructions,
            coordinate_translations,
            compatibility_exits,
        )
    except Exception:
        _restore_geometry(graph, snapshot)
        raise
