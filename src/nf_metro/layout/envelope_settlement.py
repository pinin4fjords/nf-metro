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
from dataclasses import dataclass
from enum import Enum

from nf_metro.layout.constants import COORD_TOLERANCE, SETTLEMENT_QUANTUM
from nf_metro.layout.geometry import shift_section
from nf_metro.layout.route_plan import EmissionMemberId, RoutePlan
from nf_metro.layout.route_reservations import (
    SECTION_HEADER_BLOCKER,
    SECTION_LEFT_BLOCKER,
    ColumnGapRegion,
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

    @property
    def message(self) -> str:
        claimants = ", ".join(sorted(self.claimant_member_ids))
        blockers = ", ".join(sorted(self.blocker_ids))
        return (
            f"{self.axis.value} boundary {self.boundary} widened by "
            f"{self.amount:.2f}px for the corridor claimed by {claimants}, "
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
class EnvelopeSettlement:
    """What one settlement pass moved, and what it could not."""

    translations: tuple[SettlementTranslation, ...]
    obstructions: tuple[SettlementObstruction, ...]


@dataclass(frozen=True, slots=True)
class _Axis:
    """The per-axis differences between row and column settlement."""

    axis: SettlementAxis
    boundary_of: Callable[[RouteReservation], int]
    start_index: Callable[[Section], int]
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
        deficit, reservation, blockers = max(claims, key=lambda item: item[0])
        amount = math.ceil(deficit / SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM
        axis.translate(graph, boundary, amount)
        translations.append(
            SettlementTranslation(
                axis.axis,
                boundary,
                amount,
                reservation.id,
                reservation.claimant_member_ids,
                blockers,
            )
        )
    return translations, obstructions


def settlement_pass_bound(graph: MetroGraph) -> int:
    """How many settle-and-reroute passes a graph can possibly need.

    One pass settles every boundary against the demand the routed members
    declared.  A second pass is only reachable when widening a boundary let the
    router admit a further line into a bundle crossing it, raising that
    corridor's required width.  A bundle cannot hold more lines than the graph
    defines, so demand can escalate at most that many times, after which a pass
    finds no deficit and writes nothing.
    """
    return len(graph.lines) + 1


def settle_route_envelopes(graph: MetroGraph, plan: RoutePlan) -> EnvelopeSettlement:
    """Widen row and column boundaries until every reserved corridor fits.

    Mutates *graph* in place, translating whole rows and whole columns only.
    Returns what moved and any deficit outside this stage's ownership.
    """
    row_translations, row_obstructions = _settle_axis(
        graph, _reservations_on(plan, RowGapRegion), _ROW_AXIS
    )
    column_translations, column_obstructions = _settle_axis(
        graph, _reservations_on(plan, ColumnGapRegion), _COLUMN_AXIS
    )
    return EnvelopeSettlement(
        tuple(row_translations + column_translations),
        tuple(row_obstructions + column_obstructions),
    )
