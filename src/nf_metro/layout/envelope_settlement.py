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

A section belongs to the band holding its grid start, so boundary ``b`` owns
every section starting at or beyond it.  A section straddling ``b`` starts above
it and stays: carrying it would take its upper portion into the gap above and
narrow that separation, and no step here may shrink a satisfied gap.  Holding it
is sound exactly when it does not bound a corridor the translation widened -- if
it did, the widening never reached that corridor.  Both halves of that are
asserted on the settled geometry rather than argued: every facing pair is
re-measured for monotonicity, and every straddling section is checked against the
blockers of each corridor its boundary settled.  The sweep is therefore a single
directional pass over a finite set of boundaries.

The two axes settle in sequence -- every row boundary, then every column
boundary -- and the row phase is never revisited.  That is sound because a
column translation writes only x, so it cannot move an edge a row corridor is
measured between; it can reach a row corridor only by changing which sections
the corridor's run overlaps.  A corridor scoped to its topology span selects by
grid index, which no translation moves.  A run-scoped corridor selects by
x-overlap, and sections at or beyond the translated boundary move together with
whatever part of the run reaches them, so a section clear of the run cannot be
drawn into it -- unless it spans across the translated boundary, in which case it
stays put while a crossing run lengthens.  That single exception is checked
rather than assumed: every row corridor is re-measured after the column phase
and a narrowed one fails.  Nothing is retried.

Re-routing the settled geometry produces a *different* ledger -- corridors
appear, vanish, and change their required width -- so iterating settlement
against successive ledgers would be a fixpoint search over a moving constraint
set, with no convergence argument behind it.  Settlement therefore runs once,
against the ledger it was handed.  A demand that only the re-routed geometry
reveals is reported, not chased.

Every row- and column-gap claim this stage is handed is therefore allocatable:
the measurement bounds a boundary by the sections that lie wholly on each side
of it, so a boundary every relevant section straddles has no side to measure and
is never chosen as a corridor's region.  A deficit that survives the sweep is an
infeasible arrangement, not a claim outside this stage's reach, and the closing
guard refuses it on the strict path.
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
    """One applied global translation of everything from *boundary* onward.

    ``coordinate`` is where the translated band started before the move -- the
    smallest origin among the moved sections -- so a frozen claim coordinate can
    be told apart as sitting inside or ahead of the band.
    """

    axis: SettlementAxis
    boundary: int
    coordinate: float
    amount: float
    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    blocker_ids: tuple[str, ...]
    section_ids: tuple[str, ...]
    reservation_ids: tuple[RouteReservationId, ...]
    spanning_section_ids: tuple[str, ...] = ()
    """Sections straddling the boundary, which this translation cannot own."""

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(sorted(self.blocker_ids))
        held = (
            f", holding {', '.join(self.spanning_section_ids)} in place across it"
            if self.spanning_section_ids
            else ""
        )
        return (
            f"{self.axis.value} boundary {self.boundary} widened by "
            f"{self.amount:.2f}px for the corridor claimed by {claimants}, "
            f"held from below by {blockers}; it moved "
            f"{len(self.section_ids)} section(s){held} and settled "
            f"{len(self.reservation_ids)} claim(s)"
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
    """What one settlement pass moved, and what it could not.

    ``coordinate_translations`` is the same set of moves expressed as they act
    on frozen claim coordinates, for every later measurement of the ledger
    settlement consumed.
    """

    translations: tuple[SettlementTranslation, ...]
    shortfalls: tuple[SettlementShortfall, ...] = ()
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = ()


@dataclass(frozen=True, slots=True)
class _Axis:
    """The per-axis differences between row and column settlement."""

    axis: SettlementAxis
    boundary_of: Callable[[RouteReservation], int]
    start_index: Callable[[Section], int]
    span: Callable[[Section], int]
    origin: Callable[[Section], float]
    blocker_prefix: str
    shift: Callable[[MetroGraph, Section, float], None]


@dataclass(frozen=True, slots=True)
class _TranslationOwnership:
    """Which sections one boundary translation owns, and which it cannot.

    A section belongs to the band holding its grid start, so a boundary
    translation owns every section starting at or beyond it.  A section
    straddling the boundary starts above it and stays: carrying it would take
    its upper portion into the gap above and narrow that separation, which no
    settlement step may do.  Naming both sets makes the ownership a computed
    fact rather than a side effect of the comparison that applies it.
    """

    moved_section_ids: tuple[str, ...]
    spanning_section_ids: tuple[str, ...]


def _translation_ownership(
    graph: MetroGraph, axis: _Axis, boundary: int
) -> _TranslationOwnership:
    moved: list[str] = []
    spanning: list[str] = []
    for key, section in graph.sections.items():
        start = axis.start_index(section)
        if start >= boundary:
            moved.append(key)
        elif start + axis.span(section) > boundary:
            spanning.append(key)
    return _TranslationOwnership(tuple(sorted(moved)), tuple(sorted(spanning)))


def _apply_translation(
    graph: MetroGraph,
    axis: _Axis,
    ownership: _TranslationOwnership,
    amount: float,
) -> None:
    for section_id in ownership.moved_section_ids:
        axis.shift(graph, graph.sections[section_id], amount)


_ROW_AXIS = _Axis(
    SettlementAxis.ROW,
    lambda reservation: _row_region(reservation).lower_row,
    lambda section: section.grid_row,
    lambda section: section.grid_row_span,
    lambda section: section.bbox_y,
    SECTION_HEADER_BLOCKER,
    lambda graph, section, amount: shift_section(graph, section, dy=amount),
)

_COLUMN_AXIS = _Axis(
    SettlementAxis.COLUMN,
    lambda reservation: _column_region(reservation).right_column,
    lambda section: section.grid_col,
    lambda section: section.grid_col_span,
    lambda section: section.bbox_x,
    SECTION_LEFT_BLOCKER,
    lambda graph, section, amount: shift_section(graph, section, dx=amount),
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


def _reservation_coordinate_translation(
    translation: SettlementTranslation,
    plan: RoutePlan,
) -> ReservationCoordinateTranslation:
    """Express one applied settlement move as it acts on frozen claim coordinates.

    A member whose endpoints both sit in moved sections had its whole run
    carried, so every coordinate it claimed moves; a member with one endpoint
    in the moved band was stretched across the boundary, so only the
    coordinates at or beyond the band's start move.
    """
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


def _settle_axis(
    graph: MetroGraph,
    plan: RoutePlan,
    reservations: tuple[RouteReservation, ...],
    axis: _Axis,
    prior_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> tuple[
    list[SettlementTranslation],
    list[ReservationCoordinateTranslation],
]:
    by_boundary: dict[int, list[RouteReservation]] = {}
    for reservation in reservations:
        by_boundary.setdefault(axis.boundary_of(reservation), []).append(reservation)

    translations: list[SettlementTranslation] = []
    coordinate_translations = list(prior_translations)
    for boundary in sorted(by_boundary):
        projected_prefix = tuple(coordinate_translations)
        claims: list[tuple[float, RouteReservation, tuple[str, ...]]] = []
        for reservation in sorted(by_boundary[boundary], key=lambda item: item.id):
            realised = realise_reservation(
                graph, reservation, coordinate_translations=projected_prefix
            )
            if realised is None:
                continue
            deficit = -realised.capacity_slack
            if deficit <= COORD_TOLERANCE:
                continue
            claims.append((deficit, reservation, realised.negative_blocker_ids))
        if not claims:
            continue
        deficit, reservation, blockers = max(claims, key=lambda item: item[0])
        amount = math.ceil(deficit / SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM
        ownership = _translation_ownership(graph, axis, boundary)
        if not ownership.moved_section_ids:
            raise ValueError(
                f"{axis.axis.value} boundary {boundary} has no translation "
                "owner: nothing sits at or beyond it to move"
            )
        band_start = min(
            axis.origin(graph.sections[section_id])
            for section_id in ownership.moved_section_ids
        )
        _apply_translation(graph, axis, ownership, amount)
        translations.append(
            SettlementTranslation(
                axis.axis,
                boundary,
                band_start,
                amount,
                reservation.id,
                reservation.claimant_member_ids,
                blockers,
                ownership.moved_section_ids,
                tuple(item[1].id for item in claims),
                ownership.spanning_section_ids,
            )
        )
        coordinate_translations.append(
            _reservation_coordinate_translation(translations[-1], plan)
        )
    return (
        translations,
        coordinate_translations[len(prior_translations) :],
    )


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

    Both measurements come from the geometry the map actually draws: the
    conflict coordinates are read from the published frames, and each corridor
    slack is measured against the live graph with the projection its published
    realisation records.
    """
    compatibility = tuple(
        item for item in plan.convergence_plans if item.legacy_reason is not None
    )
    if not compatibility:
        return ()
    held_translations = {
        item.reservation_id: item.coordinate_translations
        for item in plan.realised_reservations
    }
    slack_by_system: dict[RouteSystemId, list[float]] = {}
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(
            graph,
            reservation,
            coordinate_translations=held_translations.get(reservation.id, ()),
        )
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


def _measured_widths(
    graph: MetroGraph,
    reservations: tuple[RouteReservation, ...],
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...],
) -> dict[RouteReservationId, float]:
    widths: dict[RouteReservationId, float] = {}
    for reservation in reservations:
        realised = realise_reservation(
            graph, reservation, coordinate_translations=coordinate_translations
        )
        if realised is not None:
            widths[reservation.id] = realised.available_width
    return widths


def _assert_the_column_phase_left_the_row_phase_standing(
    before: dict[RouteReservationId, float],
    after: dict[RouteReservationId, float],
) -> None:
    """No row corridor is narrower after the column phase than before it.

    Rows settle first and columns second, with no row recheck, which is sound
    because a column translation writes only x: the edges a row corridor is
    measured between are y values it never touches.  It can reach a row corridor
    only by changing which sections the corridor's own run overlaps, and for a
    corridor scoped to the topology span there is nothing to change -- grid
    indices do not move.  For a run-scoped corridor, sections at or beyond the
    translated column boundary move with the part of the run that reaches them,
    and sections before it move with neither, so a section outside the run
    cannot be drawn into it.

    The exception is a section spanning across the translated column boundary:
    it stays put while a run crossing that boundary lengthens, so such a section
    can end up inside a run that clears it before the translation.  That is a
    real narrowing, which is why the conclusion is checked here on every settled
    layout rather than taken from the argument alone.  A layout that hits it
    fails inside the transactional scope rather than being re-settled: a second
    pass would be settling against a constraint set the first pass moved.
    """
    for reservation_id, width in after.items():
        held = before.get(reservation_id)
        if held is not None and width < held - COORD_TOLERANCE:
            raise ValueError(
                f"the column phase narrowed the row corridor {reservation_id} "
                f"from {held:.2f}px to {width:.2f}px: a section spanning across a "
                "translated column boundary was drawn into the corridor's run, "
                "so the two axes are not independent for this layout"
            )


def _verify_against_input_ledger(
    graph: MetroGraph,
    plan: RoutePlan,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...],
) -> tuple[SettlementShortfall, ...]:
    """Re-measure the demands settlement was handed, on the geometry it left.

    This is settlement's own postcondition, and it is the only measurement that
    can state it: the ledger a later re-route publishes is a different set of
    claims, so a deficit found there answers a different question.  Every
    reservation on both axes is measured here with the complete translation
    set, so a row demand whose blockers a later column translation changed is
    caught rather than assumed independent.
    """
    shortfalls: list[SettlementShortfall] = []
    for reservation in plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = realise_reservation(
            graph, reservation, coordinate_translations=coordinate_translations
        )
        if realised is None or realised.capacity_slack >= -COORD_TOLERANCE:
            continue
        shortfalls.append(
            SettlementShortfall(
                reservation.id,
                reservation.claimant_member_ids,
                realised.required_width,
                realised.available_width,
            )
        )
    return tuple(shortfalls)


def _box_extents(section: Section) -> tuple[tuple[float, float], tuple[float, float]]:
    """One section's box as its horizontal and vertical intervals, in that order."""
    return (
        (section.bbox_x, section.bbox_x + section.bbox_w),
        (section.bbox_y, section.bbox_y + section.bbox_h),
    )


def _axis_gaps(graph: MetroGraph, axis: _Axis) -> dict[tuple[str, str], float]:
    """The signed clearance between every pair of boxes that face each other.

    Keyed by the ordered pair, so the same pair is comparable before and after a
    translation.  Boxes that do not overlap across the axis never face each
    other, so the distance between them is not a separation this stage owes
    anything to.
    """
    along_index = 1 if axis.axis is SettlementAxis.ROW else 0
    across_index = 1 - along_index
    gaps: dict[tuple[str, str], float] = {}
    sections = sorted(graph.sections.items())
    for first_key, first in sections:
        for second_key, second in sections:
            if first_key >= second_key:
                continue
            first_extents = _box_extents(first)
            second_extents = _box_extents(second)
            first_lo, first_hi = first_extents[across_index]
            second_lo, second_hi = second_extents[across_index]
            if min(first_hi, second_hi) <= max(first_lo, second_lo):
                continue
            first_start, first_end = first_extents[along_index]
            second_start, second_end = second_extents[along_index]
            if first_end <= second_start:
                gaps[first_key, second_key] = second_start - first_end
            elif second_end <= first_start:
                gaps[second_key, first_key] = first_start - second_end
    return gaps


def _assert_no_separation_decreased(
    before: dict[tuple[str, str], float],
    after: dict[tuple[str, str], float],
    axis: _Axis,
) -> None:
    """Every pair facing each other in both states is at least as far apart.

    The monotone claim this stage rests on.  A pair facing each other in only
    one of the two states has no separation to compare.
    """
    for pair, gap in after.items():
        held = before.get(pair)
        if held is not None and gap < held - COORD_TOLERANCE:
            first, second = pair
            raise ValueError(
                f"envelope settlement narrowed the {axis.axis.value} separation "
                f"between sections {first!r} and {second!r} from {held:.2f}px to "
                f"{gap:.2f}px; no settlement step may shrink a satisfied gap"
            )


def _assert_spanning_sections_bound_nothing_settled(
    graph: MetroGraph,
    plan: RoutePlan,
    translations: Iterable[SettlementTranslation],
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...],
) -> None:
    """A section a translation could not move must not bound what it settled.

    This is what makes holding a straddling section in place sound.  If such a
    section bounds one of the corridors the translation widened, the widening
    never reached that corridor and the translation's record of settling it is
    false.  Measured on the final geometry, after both axes, so a row
    translation invalidated by a later column translation -- which moves the
    horizontal intervals a row corridor's blockers are selected by -- is caught
    here rather than assumed away.
    """
    reservation_by_id = {item.id: item for item in plan.reservations}
    for translation in translations:
        if not translation.spanning_section_ids:
            continue
        held = frozenset(translation.spanning_section_ids)
        for reservation_id in translation.reservation_ids:
            reservation = reservation_by_id.get(reservation_id)
            if reservation is None:
                continue
            realised = realise_reservation(
                graph, reservation, coordinate_translations=coordinate_translations
            )
            if realised is None:
                continue
            blockers = {
                blocker_id.partition(":")[2]
                for blocker_id in (
                    realised.negative_blocker_ids + realised.positive_blocker_ids
                )
            }
            trespassing = sorted(held & blockers)
            if trespassing:
                raise ValueError(
                    f"{translation.axis.value} boundary {translation.boundary} was "
                    f"widened by {translation.amount:.2f}px for the corridor "
                    f"{reservation.description!r}, but section(s) "
                    f"{', '.join(trespassing)} straddle that boundary and bound "
                    "the corridor, so the translation could not have widened it"
                )


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
    row_gaps_before = _axis_gaps(graph, _ROW_AXIS)
    column_gaps_before = _axis_gaps(graph, _COLUMN_AXIS)
    row_reservations = _reservations_on(plan, RowGapRegion)
    try:
        row_translations, row_coordinate = _settle_axis(
            graph, plan, row_reservations, _ROW_AXIS
        )
        row_widths = _measured_widths(graph, row_reservations, tuple(row_coordinate))
        column_translations, column_coordinate = _settle_axis(
            graph,
            plan,
            _reservations_on(plan, ColumnGapRegion),
            _COLUMN_AXIS,
            tuple(row_coordinate),
        )
        coordinate_translations = tuple(row_coordinate + column_coordinate)
        translations = tuple(row_translations + column_translations)
        _assert_the_column_phase_left_the_row_phase_standing(
            row_widths,
            _measured_widths(graph, row_reservations, coordinate_translations),
        )
        _assert_no_separation_decreased(
            row_gaps_before, _axis_gaps(graph, _ROW_AXIS), _ROW_AXIS
        )
        _assert_no_separation_decreased(
            column_gaps_before, _axis_gaps(graph, _COLUMN_AXIS), _COLUMN_AXIS
        )
        _assert_spanning_sections_bound_nothing_settled(
            graph, plan, translations, coordinate_translations
        )
        shortfalls = _verify_against_input_ledger(graph, plan, coordinate_translations)
    except Exception:
        _restore_coordinate_state(graph, restore_point)
        raise
    return EnvelopeSettlement(
        translations,
        shortfalls,
        coordinate_translations,
    )
