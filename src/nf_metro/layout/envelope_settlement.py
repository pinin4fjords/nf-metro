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

A boundary whose far-side blocker cannot be translated is not an
envelope-allocation problem.  Settlement records an attributed obstruction
naming the blocker instead of translating geometry that would not help, and
distinguishes a claim it cannot act on from a corridor it merely cannot widen.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from nf_metro.layout.constants import COORD_TOLERANCE, SETTLEMENT_QUANTUM
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import (
    ConflictRelief,
    ConvergenceConflict,
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
    RealisedRouteReservation,
    RouteReservation,
    RouteReservationId,
    RowGapRegion,
    realise_reservation,
)
from nf_metro.parser.model import MetroGraph, Section


class ObstructionKind(Enum):
    """Why a deficit survived a settlement pass."""

    PINNED_BLOCKER = "pinned-blocker"
    """The corridor is real, but a section outside the translated band bounds
    it, so no offset this stage owns can widen it."""

    INCOHERENT_CLAIM = "incoherent-claim"
    """The claim does not describe a corridor: one section bounds both of its
    sides, because it spans across the boundary instead of sitting either side
    of it.  There is no gap between a box and itself to allocate."""


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
    """A deficit no translation this stage owns is able to close."""

    axis: SettlementAxis
    boundary: int
    deficit: float
    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]
    blocking_section_ids: tuple[str, ...]
    kind: ObstructionKind

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(self.blocking_section_ids)
        if self.kind is ObstructionKind.INCOHERENT_CLAIM:
            return (
                f"the corridor claimed by {claimants} at {self.axis.value} "
                f"boundary {self.boundary} measures its far side ahead of its "
                f"near side, because section(s) {blockers} cross the boundary "
                f"instead of bounding it; there is no gap here to allocate"
            )
        return (
            f"{self.axis.value} boundary {self.boundary} is short of the "
            f"corridor claimed by {claimants} by {self.deficit:.2f}px, but "
            f"section(s) {blockers} bound the far side without belonging to "
            f"the translated {self.axis.value}s, so widening the boundary "
            f"cannot supply it"
        )


class SettlementReach(Enum):
    """Whether settlement's own translations can reach a compatibility limit."""

    SEPARATION_FIXED = "separation-fixed"
    """Both conflicting runs sit in one translated band, so every offset this
    stage owns moves them together and the distance between them never
    changes."""

    SEPARATION_ONLY_GROWS = "separation-only-grows"
    """The runs sit in different bands, so an offset does move them apart --
    which is the wrong direction for a conflict that needs one shared channel,
    and settlement is forbidden from moving them together."""

    WITHIN_REACH = "within-reach"
    """An offset this stage owns changes the separation in the direction the
    conflict needs.  The limit is not attributed away from settlement."""


@dataclass(frozen=True, slots=True)
class CompatibilityOwnership:
    """A route system on the compatibility path, measured against settlement.

    #1660 may only leave a system on compatibility with evidence that what
    limits it is not envelope allocation.  The evidence is two measurements on
    the settled geometry: every corridor the system reserved reaching its
    required width, and the conflict that still holds it standing where no
    global row or column offset can move it.
    """

    system_id: RouteSystemId
    conflict: ConvergenceConflict | None
    reach: SettlementReach | None
    bands: tuple[int, int] | None
    corridor_count: int
    worst_capacity_slack: float

    @property
    def owner(self) -> str:
        if self.conflict is None:
            return _UNATTRIBUTED_OWNER
        return self.conflict.kind.owner

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
                f"tightest with {self.worst_capacity_slack:.2f}px to spare"
            )
        if self.conflict is None or self.bands is None:
            return (
                f"route system {self.system_id} stays on compatibility: {fit}, but "
                f"its planner recorded no measured conflict, so what limits it is "
                f"unattributed"
            )
        axis = _settlement_axis(self.conflict).axis.value
        first, second = self.bands
        if self.reach is SettlementReach.SEPARATION_FIXED:
            proof = (
                f"both runs sit in {axis} band {first}, which every offset this "
                f"stage owns moves together, so that {self.conflict.separation:.2f}px "
                f"is not an envelope allocation"
            )
        elif self.reach is SettlementReach.SEPARATION_ONLY_GROWS:
            proof = (
                f"the runs sit in {axis} bands {first} and {second}, and settlement "
                f"never shrinks a separation, so widening moves them further from "
                f"the one shared channel this conflict needs"
            )
        else:
            proof = (
                f"the runs sit in {axis} bands {first} and {second}, so an offset "
                f"this stage owns does change the {self.conflict.separation:.2f}px "
                f"between them; the limit is not attributed away from settlement"
            )
        return (
            f"route system {self.system_id} stays on compatibility: {fit}; what "
            f"holds it is {self.conflict.measurement}, and {proof}.  "
            f"{self.owner} owns the decision it needs."
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
    kind: ObstructionKind | None
    pinned_section_ids: tuple[str, ...] = ()

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
    shortfalls: tuple[SettlementShortfall, ...] = ()


@dataclass(frozen=True, slots=True)
class _Axis:
    """The per-axis differences between row and column settlement."""

    axis: SettlementAxis
    boundary_of: Callable[[RouteReservation], int]
    start_index: Callable[[Section], int]
    origin: Callable[[Section], float]
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


def _blocker_sections(blocker_ids: Iterable[str]) -> set[str]:
    return {blocker_id.partition(":")[2] for blocker_id in blocker_ids}


def sections_bounding_both_sides(
    realised: RealisedRouteReservation,
) -> tuple[str, ...]:
    """Sections named as the blocker on both sides of the same corridor.

    A section can only bound a gap from above and below at once by spanning
    across it, which means the measurement found no gap there at all.  Derived
    from the measurement itself rather than from a reservation id, so it stays
    valid across the re-routed ledger, whose ids need not be the ones
    settlement saw.
    """
    both = _blocker_sections(realised.negative_blocker_ids) & _blocker_sections(
        realised.positive_blocker_ids
    )
    return tuple(sorted(both))


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
            stuck = _obstructing_sections(
                graph, axis, boundary, realised.positive_blocker_ids
            )
            shared = sections_bounding_both_sides(realised)
            if shared:
                obstructions.append(
                    SettlementObstruction(
                        axis.axis,
                        boundary,
                        deficit,
                        reservation.id,
                        reservation.claimant_member_ids,
                        realised.positive_blocker_ids,
                        shared,
                        ObstructionKind.INCOHERENT_CLAIM,
                    )
                )
                continue
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
                        ObstructionKind.PINNED_BLOCKER,
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


# A compatibility system whose planner recorded no measurement has nothing to
# attribute it by, so it is named as unattributed rather than assigned an owner
# the evidence does not support.
_UNATTRIBUTED_OWNER = "unattributed"


def _settlement_axis(conflict: ConvergenceConflict) -> _Axis:
    """The one axis whose translations can change *conflict*'s separation.

    A row translation writes y and a column translation writes x, so a distance
    measured along one of those axes is out of the other's reach entirely.
    """
    return _ROW_AXIS if conflict.axis is DemandAxis.Y else _COLUMN_AXIS


def _translated_bands(graph: MetroGraph, axis: _Axis) -> dict[int, float]:
    """The first coordinate each boundary's translation would carry with it.

    Widening boundary ``b`` translates every section from index ``b`` onward, so
    the topmost (or leftmost) of those boxes is where its effect starts.  Taken
    as a running minimum from the last boundary backwards, those starts are
    non-decreasing in ``b``, which makes the boundaries that move a given
    coordinate a prefix, and their count the band that coordinate sits in.
    """
    origin_by_index: dict[int, float] = {}
    for section in graph.sections.values():
        index = axis.start_index(section)
        origin = axis.origin(section)
        origin_by_index[index] = min(origin_by_index.get(index, origin), origin)
    bands: dict[int, float] = {}
    onward = math.inf
    for boundary in sorted(origin_by_index, reverse=True):
        onward = min(onward, origin_by_index[boundary])
        bands[boundary] = onward
    return bands


def _band_of(bands: dict[int, float], coordinate: float) -> int:
    moved = [
        boundary
        for boundary, start in bands.items()
        if coordinate >= start - COORD_TOLERANCE
    ]
    return max(moved, default=min(bands, default=0) - 1)


def _settlement_reach(
    graph: MetroGraph, conflict: ConvergenceConflict
) -> tuple[SettlementReach, tuple[int, int]]:
    """Whether any offset this stage owns changes a measured conflict.

    A separation along y moves only when whole rows move, and one along x only
    when whole columns do, so exactly one axis is capable of touching it.  Two
    runs the same axis translation carries together keep the distance between
    them whatever settlement does; runs it carries apart only ever get further
    apart, because no step may shrink a separation.
    """
    axis = _settlement_axis(conflict)
    index = 1 if conflict.axis is DemandAxis.Y else 0
    bands = _translated_bands(graph, axis)
    first, second = (_band_of(bands, site[0][index]) for site in conflict.sites)
    if first == second:
        return SettlementReach.SEPARATION_FIXED, (first, second)
    if conflict.kind.relief is ConflictRelief.SHARED_CHANNEL:
        return SettlementReach.SEPARATION_ONLY_GROWS, (first, second)
    return SettlementReach.WITHIN_REACH, (first, second)


def attribute_compatibility_systems(
    graph: MetroGraph, plan: RoutePlan
) -> tuple[CompatibilityOwnership, ...]:
    """Measure every compatibility system in *plan* against what settlement moves.

    Both measurements have to come from the geometry the map actually draws, so
    this reads the plan that survived the re-route rather than the one
    settlement was handed: a translated layout publishes new corridors and new
    conflict coordinates, and an exit quoting the pre-translation frame would be
    describing a map nobody sees.
    """
    compatibility = tuple(
        item for item in plan.convergence_plans if item.legacy_reason is not None
    )
    if not compatibility:
        return ()
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
    for convergence in compatibility:
        system_id = convergence.system_id
        slacks = slack_by_system.get(system_id)
        if not slacks:
            continue
        conflict = convergence.conflict
        reach, bands = (
            (None, None) if conflict is None else _settlement_reach(graph, conflict)
        )
        found[system_id] = CompatibilityOwnership(
            system_id, conflict, reach, bands, len(slacks), min(slacks)
        )
    return tuple(found[key] for key in sorted(found))


def attach_settlement_diagnostics(
    graph: MetroGraph, plan: RoutePlan, settlement: EnvelopeSettlement
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
        for item in attribute_compatibility_systems(graph, plan)
    )
    if not added:
        return plan
    return replace(plan, diagnostics=plan.diagnostics + added)


def _verify_against_input_ledger(
    graph: MetroGraph,
    plan: RoutePlan,
    obstructions: list[SettlementObstruction],
) -> tuple[SettlementShortfall, ...]:
    """Re-measure the demands settlement was handed, on the geometry it left.

    This is settlement's own postcondition, and it is the only measurement that
    can state it: the ledger a later re-route publishes is a different set of
    claims, so a deficit found there answers a different question.
    """
    kind_by_id = {item.reservation_id: item.kind for item in obstructions}
    blockers_by_id = {
        item.reservation_id: item.blocking_section_ids for item in obstructions
    }
    provenance = graph.layout_provenance
    shortfalls: list[SettlementShortfall] = []
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is None or realised.capacity_slack >= -COORD_TOLERANCE:
            continue
        pinned = tuple(
            section_id
            for section_id in blockers_by_id.get(reservation.id, ())
            if provenance.author_owns_grid(section_id)
        )
        shortfalls.append(
            SettlementShortfall(
                reservation.id,
                reservation.claimant_member_ids,
                realised.required_width,
                realised.available_width,
                kind_by_id.get(reservation.id),
                pinned,
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
        shortfalls = _verify_against_input_ledger(
            graph, plan, row_obstructions + column_obstructions
        )
    except Exception:
        _restore_coordinate_state(graph, restore_point)
        raise
    return EnvelopeSettlement(
        tuple(row_translations + column_translations),
        tuple(row_obstructions + column_obstructions),
        shortfalls,
    )
