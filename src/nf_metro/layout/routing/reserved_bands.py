"""Row-gap corridors the reservation ledger allocates, as the router reads them.

A ``RouteReservation`` in a :class:`RowGapRegion` names the grid boundary its
corridor crosses, the blockers that bound it over the corridor's own declared
span, and the clearance it must keep from each of them.  Measuring that record
against live geometry yields the clear span a channel in that gap may occupy.
The router places its channel inside that span rather than deriving one from
the row edges it happens to have in hand: the row edges are a proxy that
over-states the obstruction wherever a section spans the boundary or sits
outside the corridor's run, and a proxy narrow enough to hold no channel at all
is what drives the header-biased fallback in ``_center_inter_row_channel``.

A boundary is only readable this way once a ledger exists, which is after the
first routing pass has published one.  The settled re-route is where that
matters: envelope settlement has just translated rows to make these corridors
fit, and re-deriving the band would discard the allocation it was given.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nf_metro.layout.constants import COORD_TOLERANCE

if TYPE_CHECKING:
    from nf_metro.layout.route_plan import RoutePlan
    from nf_metro.parser.model import MetroGraph


@dataclass(frozen=True, slots=True)
class ReservedBand:
    """The clear span a row boundary's reservations leave for a channel."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo - COORD_TOLERANCE:
            raise ValueError("a reserved band cannot be narrower than nothing")

    def place(self, offset: float) -> float:
        """The channel Y at *offset* from the band centre, held inside it."""
        centre = (self.lo + self.hi) / 2
        return min(max(centre + offset, self.lo), self.hi)


@dataclass(frozen=True, slots=True)
class ReservedRowBands:
    """Realised row-gap corridors, keyed by the lower row of their boundary."""

    bands: Mapping[int, ReservedBand] = field(default_factory=dict)

    def at(self, lower_row: int | None) -> ReservedBand | None:
        """The band reserved below *lower_row - 1*, or ``None`` when unclaimed."""
        if lower_row is None:
            return None
        return self.bands.get(lower_row)


def build_reserved_row_bands(graph: MetroGraph, plan: RoutePlan) -> ReservedRowBands:
    """Measure *plan*'s row-gap reservations against the live *graph* geometry.

    Several corridors can claim one boundary over different spans, so a
    boundary's band is the intersection of what each of them leaves clear: a
    channel there has to satisfy every claim, not the most generous one.  An
    empty intersection describes no single corridor, so that boundary is left
    unclaimed and the router derives its own band as it does for any gap the
    ledger never reached.
    """
    from nf_metro.layout.route_reservations import RowGapRegion, realise_reservation

    spans: dict[int, tuple[float, float]] = {}
    for reservation in plan.reservations:
        region = reservation.region
        if not isinstance(region, RowGapRegion):
            continue
        realised = realise_reservation(graph, reservation)
        if realised is None:
            continue
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        held = spans.get(region.lower_row)
        spans[region.lower_row] = (
            (lo, hi) if held is None else (max(held[0], lo), min(held[1], hi))
        )
    return ReservedRowBands(
        {
            row: ReservedBand(lo, hi)
            for row, (lo, hi) in sorted(spans.items())
            if hi >= lo - COORD_TOLERANCE
        }
    )
