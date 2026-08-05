"""Monotone row and column envelope settlement around route reservations.

Local station geometry, section bboxes, header keep-outs, and the immutable
``RouteReservation`` ledger are all final before this runs.  Settlement owns one
thing only: the global row and column offsets needed to give every reserved
corridor the width its ledger entry requires.  It never resizes a box, moves a
station inside its section, or revisits a route decision.

Termination is structural, and it depends on settling against ONE ledger.  Each
adjacent-index boundary (the gap between row ``b-1`` and row ``b``, or between
column ``b-1`` and column ``b``) is visited exactly once in ascending order, and
translating everything from ``b`` onward has three effects and no others:

* boundaries before ``b`` keep both blockers stationary, so they are unchanged;
* boundary ``b`` widens by exactly the translated amount;
* boundaries after ``b`` move both blockers together, so they are unchanged.

A section spanning across ``b`` stays where it is, which can only increase its
distance to the content below.  No separation therefore ever decreases, and the
sweep is a single directional pass over a finite set of boundaries.

Re-routing the settled geometry produces a *different* ledger -- corridors
appear, vanish, and change their required width -- so iterating settlement
against successive ledgers would be a fixpoint search over a moving constraint
set, with no convergence argument behind it.  Settlement therefore runs once,
against the ledger it was handed.  A demand that only the re-routed geometry
reveals is reported, not chased.

A claim whose far side is bounded by a section spanning across the boundary is
not a corridor at all.  The measurement puts a spanning section on the near side
as well, so the section's own box lies inside the width the claim asks for, and
the far side lands ahead of the near side.  Settlement records an attributed
obstruction naming that section instead of translating geometry that cannot
help.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from nf_metro.layout.constants import COORD_TOLERANCE, SETTLEMENT_QUANTUM
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import (
    EmissionMemberId,
    RoutePlan,
    RoutePlanDiagnostic,
    RouteSystemId,
)
from nf_metro.layout.route_reservations import (
    SECTION_HEADER_BLOCKER,
    SECTION_LEFT_BLOCKER,
    ColumnGapRegion,
    RealisedRouteReservation,
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
    amount: float
    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]
    section_ids: tuple[str, ...]
    reservation_ids: tuple[RouteReservationId, ...]

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(sorted(self.blocker_ids))
        return (
            f"{self.axis.value} boundary {self.boundary} widened by "
            f"{self.amount:.2f}px for the corridor claimed by {claimants}, "
            f"held from below by {blockers}; it moved "
            f"{len(self.section_ids)} section(s) and settled "
            f"{len(self.reservation_ids)} claim(s)"
        )


@dataclass(frozen=True, slots=True)
class SettlementObstruction:
    """A claim that does not describe an allocatable gap.

    The far side is bounded by a section spanning across the boundary, which
    the measurement also puts on the near side.  There is no width between a
    box and itself, so no offset opens one.
    """

    axis: SettlementAxis
    boundary: int
    deficit: float
    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]
    blocking_section_ids: tuple[str, ...]
    pinned_section_ids: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(self.blocking_section_ids)
        pins = (
            f"; the authored grid pins {', '.join(self.pinned_section_ids)} across it"
            if self.pinned_section_ids
            else ""
        )
        return (
            f"the corridor claimed by {claimants} at {self.axis.value} "
            f"boundary {self.boundary} measures its far side ahead of its "
            f"near side, because section(s) {blockers} span across the boundary "
            f"instead of bounding it; there is no gap here to allocate{pins}"
        )


@dataclass(frozen=True, slots=True)
class CompatibilityOwnership:
    """A route system on the compatibility path whose corridors all fit.

    #1660 may only leave a system on compatibility with evidence that what
    limits it is not envelope allocation.  Every corridor the system claims
    reaching its required width is that evidence, and it points at whoever owns
    the decision the system is actually short of.
    """

    system_id: RouteSystemId
    convergence_reason: str
    corridor_count: int
    worst_capacity_slack: float
    owner: str

    @property
    def message(self) -> str:
        if self.worst_capacity_slack < -COORD_TOLERANCE:
            fit = (
                f"its tightest of {self.corridor_count} reserved corridor(s) is "
                f"still {-self.worst_capacity_slack:.2f}px short, which "
                f"settlement has attributed separately"
            )
        else:
            fit = (
                f"its {self.corridor_count} reserved corridor(s) all fit, the "
                f"tightest with {self.worst_capacity_slack:.2f}px to spare, so "
                f"envelope allocation is not what limits it"
            )
        return (
            f"route system {self.system_id} stays on compatibility for "
            f"{self.convergence_reason!r}: {fit}.  {self.owner} owns the "
            f"decision it needs."
        )


@dataclass(frozen=True, slots=True)
class SettlementShortfall:
    """A demand settlement was handed and did not meet.

    Measured against the ledger settlement was given, on the geometry it left
    behind.  Re-routing publishes a different ledger, so a deficit found there
    says something about the new constraint set, not about whether settlement
    honoured its own.
    """

    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    required_width: float
    available_width: float
    describes_a_gap: bool

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        return (
            f"the corridor claimed by {claimants} still has "
            f"{self.available_width:.2f}px against the {self.required_width:.2f}px "
            f"its reservation requires"
        )


@dataclass(frozen=True, slots=True)
class EnvelopeSettlement:
    """What one settlement pass moved, and what it could not."""

    translations: tuple[SettlementTranslation, ...]
    obstructions: tuple[SettlementObstruction, ...]
    compatibility_ownership: tuple[CompatibilityOwnership, ...] = ()
    shortfalls: tuple[SettlementShortfall, ...] = ()


@dataclass(frozen=True, slots=True)
class _Axis:
    """The per-axis differences between row and column settlement."""

    axis: SettlementAxis
    boundary_of: Callable[[RouteReservation], int]
    start_index: Callable[[Section], int]
    blocker_prefix: str
    translate: Callable[[MetroGraph, int, float], tuple[str, ...]]


def _translate_rows(graph: MetroGraph, boundary: int, amount: float) -> tuple[str, ...]:
    moved = []
    for key, section in graph.sections.items():
        if section.grid_row >= boundary:
            shift_section(graph, section, dy=amount)
            moved.append(key)
    return tuple(sorted(moved))


def _translate_columns(
    graph: MetroGraph, boundary: int, amount: float
) -> tuple[str, ...]:
    moved = []
    for key, section in graph.sections.items():
        if section.grid_col >= boundary:
            shift_section(graph, section, dx=amount)
            moved.append(key)
    return tuple(sorted(moved))


_ROW_AXIS = _Axis(
    SettlementAxis.ROW,
    lambda reservation: _row_region(reservation).lower_row,
    lambda section: section.grid_row,
    SECTION_HEADER_BLOCKER,
    _translate_rows,
)

_COLUMN_AXIS = _Axis(
    SettlementAxis.COLUMN,
    lambda reservation: _column_region(reservation).right_column,
    lambda section: section.grid_col,
    SECTION_LEFT_BLOCKER,
    _translate_columns,
)


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


def _axis_of(reservation: RouteReservation) -> _Axis | None:
    if isinstance(reservation.region, RowGapRegion):
        return _ROW_AXIS
    if isinstance(reservation.region, ColumnGapRegion):
        return _COLUMN_AXIS
    return None


def sections_spanning_the_gap(
    graph: MetroGraph,
    reservation: RouteReservation,
    realised: RealisedRouteReservation,
) -> tuple[str, ...]:
    """Far-side blockers that span across the boundary instead of bounding it.

    A section reaches the far side of a boundary either by starting at or after
    it, or by spanning across it; only the first kind moves when settlement
    translates the boundary onward.  The second kind is also measured on the
    near side, so its own box lies inside the width the claim asks for and the
    far side is measured ahead of the near side.  Naming it says both things at
    once: nothing here is allocatable, and no translation would help.

    Recomputed from the graph and the measurement rather than remembered
    against a reservation id, so it stays valid across a re-route, whose ledger
    need not carry the ids settlement saw.
    """
    axis = _axis_of(reservation)
    if axis is None:
        return ()
    boundary = axis.boundary_of(reservation)
    spanning = set()
    for blocker_id in realised.positive_blocker_ids:
        section = graph.sections.get(blocker_id.removeprefix(f"{axis.blocker_prefix}:"))
        if section is not None and axis.start_index(section) < boundary:
            spanning.add(section.id)
    return tuple(sorted(spanning))


def _author_pinned(graph: MetroGraph, section_ids: Iterable[str]) -> tuple[str, ...]:
    provenance = graph.layout_provenance
    return tuple(
        section_id
        for section_id in section_ids
        if provenance.author_owns_grid(section_id)
    )


def _settle_axis(
    graph: MetroGraph,
    reservations: tuple[RouteReservation, ...],
    axis: _Axis,
) -> tuple[list[SettlementTranslation], list[SettlementObstruction]]:
    by_boundary: dict[int, list[RouteReservation]] = {}
    for reservation in reservations:
        by_boundary.setdefault(axis.boundary_of(reservation), []).append(reservation)

    translations: list[SettlementTranslation] = []
    obstructions: list[SettlementObstruction] = []
    for boundary in sorted(by_boundary):
        claims: list[tuple[float, RouteReservation, tuple[str, ...]]] = []
        for reservation in sorted(by_boundary[boundary], key=lambda item: item.id):
            realised = realise_reservation(graph, reservation)
            if realised is None:
                continue
            deficit = -realised.capacity_slack
            if deficit <= COORD_TOLERANCE:
                continue
            spanning = sections_spanning_the_gap(graph, reservation, realised)
            if spanning:
                obstructions.append(
                    SettlementObstruction(
                        axis.axis,
                        boundary,
                        deficit,
                        reservation.id,
                        reservation.claimant_member_ids,
                        realised.positive_blocker_ids,
                        spanning,
                        _author_pinned(graph, spanning),
                    )
                )
                continue
            claims.append((deficit, reservation, realised.negative_blocker_ids))
        if not claims:
            continue
        deficit, reservation, blockers = max(claims, key=lambda item: item[0])
        amount = math.ceil(deficit / SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM
        moved = axis.translate(graph, boundary, amount)
        if not moved:
            raise ValueError(
                f"{axis.axis.value} boundary {boundary} has no translation "
                "owner: nothing sits at or beyond it to move"
            )
        translations.append(
            SettlementTranslation(
                axis.axis,
                boundary,
                amount,
                reservation.id,
                reservation.claimant_member_ids,
                blockers,
                moved,
                tuple(item[1].id for item in claims),
            )
        )
    return translations, obstructions


# Whoever owns the decision a compatibility system is short of, once its
# corridors are shown to fit.  Plan-driven emission is the programme stage that
# consolidates those channel and lane decisions.
_COMPATIBILITY_OWNER = "plan-driven inter-section emission (#1658)"

# Each compatibility reason the convergence planner can record, mapped to the
# emission decision the system is actually short of.  An unmapped reason falls
# back to the generic owner rather than being dropped.
_COMPATIBILITY_OWNER_BY_REASON = {
    "planned convergence trunks require one shared channel decision": (
        "plan-driven shared-channel emission (#1658)"
    ),
    "planned convergence feeder approaches require one shared channel decision": (
        "plan-driven shared-channel emission (#1658)"
    ),
    "planned fan arms require opposing opening channels": (
        "plan-driven opposing-opening emission (#1658)"
    ),
    "planned convergence corridor conflicts with unowned route-system member": (
        "plan-driven whole-system emission (#1658)"
    ),
    "planned convergence corridor conflicts with unowned route-system members": (
        "plan-driven whole-system emission (#1658)"
    ),
    "chained same-line convergences require one shared system settlement": (
        "plan-driven chained-convergence emission (#1658)"
    ),
    "planned convergence approaches and trunks have no settlement room": (
        "plan-driven chained-convergence emission (#1658)"
    ),
}


def _compatibility_ownership(
    graph: MetroGraph, plan: RoutePlan
) -> tuple[CompatibilityOwnership, ...]:
    """Attribute every compatibility system whose corridors are all adequate."""
    slack_by_system: dict[RouteSystemId, list[float]] = {}
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is not None:
            slack_by_system.setdefault(reservation.system_id, []).append(
                realised.capacity_slack
            )

    found: dict[RouteSystemId, CompatibilityOwnership] = {}
    for convergence in plan.convergence_plans:
        reason = convergence.legacy_reason
        if reason is None:
            continue
        system_id = convergence.system_id
        slacks = slack_by_system.get(system_id)
        if not slacks:
            continue
        found[system_id] = CompatibilityOwnership(
            system_id,
            reason,
            len(slacks),
            min(slacks),
            _COMPATIBILITY_OWNER_BY_REASON.get(reason, _COMPATIBILITY_OWNER),
        )
    return tuple(found[key] for key in sorted(found))


def attach_settlement_diagnostics(
    plan: RoutePlan, settlement: EnvelopeSettlement
) -> RoutePlan:
    """Publish what settlement moved and what it attributed, into *plan*.

    A record nobody can read is not evidence.  Emitting these as non-blocking
    plan diagnostics gives every translated row and column a named owner in the
    published plan, and lets the emission stage that owns a compatibility
    system's remaining decision find that attribution without re-deriving the
    measurement.
    """
    added = tuple(
        RoutePlanDiagnostic(
            None, "envelope-settlement-translation", item.message, blocking=False
        )
        for item in settlement.translations
    ) + tuple(
        RoutePlanDiagnostic(
            None, "convergence-settlement-exit", item.message, blocking=False
        )
        for item in settlement.compatibility_ownership
    )
    if not added:
        return plan
    return replace(plan, diagnostics=plan.diagnostics + added)


def _verify_against_input_ledger(
    graph: MetroGraph,
    plan: RoutePlan,
) -> tuple[SettlementShortfall, ...]:
    """Re-measure the demands settlement was handed, on the geometry it left.

    This is settlement's own postcondition, and it is the only measurement that
    can state it: the ledger a later re-route publishes is a different set of
    claims, so a deficit found there answers a different question.
    """
    shortfalls: list[SettlementShortfall] = []
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is None or realised.capacity_slack >= -COORD_TOLERANCE:
            continue
        shortfalls.append(
            SettlementShortfall(
                reservation.id,
                reservation.claimant_member_ids,
                realised.required_width,
                realised.available_width,
                not sections_spanning_the_gap(graph, reservation, realised),
            )
        )
    return tuple(shortfalls)


def _coordinate_state(
    graph: MetroGraph,
) -> tuple[tuple[str, float, float], ...]:
    """Every coordinate settlement is allowed to write, as plain values."""
    return (
        *(
            (f"section:{key}", section.bbox_x, section.bbox_y)
            for key, section in graph.sections.items()
        ),
        *(
            (f"station:{key}", station.x, station.y)
            for key, station in graph.stations.items()
        ),
        *((f"port:{key}", port.x, port.y) for key, port in graph.ports.items()),
    )


def _restore_coordinate_state(
    graph: MetroGraph, state: tuple[tuple[str, float, float], ...]
) -> None:
    for key, x, y in state:
        kind, _, item_id = key.partition(":")
        if kind == "section":
            section = graph.sections[item_id]
            section.bbox_x, section.bbox_y = x, y
        elif kind == "station":
            station = graph.stations[item_id]
            station.x, station.y = x, y
        else:
            port = graph.ports[item_id]
            port.x, port.y = x, y


def settle_route_envelopes(graph: MetroGraph, plan: RoutePlan) -> EnvelopeSettlement:
    """Widen row and column boundaries until every reserved corridor fits.

    Mutates *graph* in place, translating whole rows and whole columns only.
    Returns what moved and any deficit outside this stage's ownership.

    The write is transactional.  A pass touches many sections in sequence, so a
    failure part-way through would otherwise leave the graph in a state that is
    neither the one measured nor the one intended; the pre-settlement
    coordinates are restored before the error propagates.  The reservation
    ledger needs no such care: settlement only reads it.
    """
    restore_point = _coordinate_state(graph)
    try:
        row_translations, row_obstructions = _settle_axis(
            graph, _reservations_on(plan, RowGapRegion), _ROW_AXIS
        )
        column_translations, column_obstructions = _settle_axis(
            graph, _reservations_on(plan, ColumnGapRegion), _COLUMN_AXIS
        )
        ownership = _compatibility_ownership(graph, plan)
        shortfalls = _verify_against_input_ledger(graph, plan)
    except Exception:
        _restore_coordinate_state(graph, restore_point)
        raise
    return EnvelopeSettlement(
        tuple(row_translations + column_translations),
        tuple(row_obstructions + column_obstructions),
        ownership,
        shortfalls,
    )
