"""Gap corridors the reservation ledger allocates, as the router reads them.

A ``RouteReservation`` in a :class:`~nf_metro.layout.route_reservations.RowGapRegion`
or :class:`~nf_metro.layout.route_reservations.ColumnGapRegion` names the grid
boundary its corridor crosses, the blockers that bound it over the corridor's own
declared span, and the clearance it must keep from each of them.  Measuring that
record against live geometry yields the clear span a channel in that gap may
occupy.  The router places its channel inside that span rather than deriving one
from the row or column edges it happens to have in hand: those edges are a proxy
that over-states the obstruction wherever a section spans the boundary or sits
outside the corridor's run, and a proxy narrow enough to hold no channel at all
is what drives the header-biased fallback in ``_center_inter_row_channel``.

Both axes read the same way -- a boundary index, a clear span, and the same
clamp -- so one measurement serves rows and columns, and
:class:`ReservedCorridors` is only the pair of axis results.

A boundary is only readable this way once a ledger exists, which is after the
first routing pass has published one, so only the re-route
(``_settle_render_geometry``) consumes bands.  Where envelope settlement has
translated rows or columns to make a corridor fit, re-deriving the band would
discard the allocation it was just given; where the corridor already fitted, the
band remains a measurement over the corridor's own span rather than over the two
rows or columns as a whole.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nf_metro.layout.constants import COORD_TOLERANCE

if TYPE_CHECKING:
    from nf_metro.layout.route_plan import RoutePlan
    from nf_metro.layout.route_reservations import (
        CorridorRegion,
        ReservationCoordinateTranslation,
    )
    from nf_metro.parser.model import MetroGraph


@dataclass(frozen=True, slots=True)
class ReservedBand:
    """The clear span a boundary's reservations leave for a channel."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo - COORD_TOLERANCE:
            raise ValueError("a reserved band cannot be narrower than nothing")

    def hold(self, coordinate: float) -> float:
        """*coordinate* itself when it is inside the band, else its nearer edge."""
        return min(max(coordinate, self.lo), self.hi)

    def place(self, offset: float) -> float:
        """The channel coordinate at *offset* from the band centre, held inside it."""
        return self.hold((self.lo + self.hi) / 2 + offset)


def held_in_reserved_band(coordinate: float, band: ReservedBand | None) -> float:
    """*coordinate* held inside *band*; *coordinate* itself where *band* is absent.

    The router derives a channel's clearance from the section edges it has to
    hand, which is a proxy for the blockers a reservation measured over the
    corridor's own span.  Where the two disagree the reservation wins, so a
    proxy-derived clearance floor is applied through this rather than to the
    bare coordinate.
    """
    return coordinate if band is None else band.hold(coordinate)


@dataclass(frozen=True, slots=True)
class ReservedBands:
    """Realised gap corridors on one axis, keyed by the boundary they cross."""

    bands: Mapping[int, ReservedBand] = field(default_factory=dict)

    def at(self, boundary: int | None) -> ReservedBand | None:
        """The band reserved at *boundary*, or ``None`` when unclaimed.

        A boundary is named by the higher of the two grid indices it separates:
        the lower row of a row gap, the right column of a column gap.
        """
        if boundary is None:
            return None
        return self.bands.get(boundary)


@dataclass(frozen=True, slots=True)
class ReservedCorridors:
    """Both axes' realised gap corridors."""

    rows: ReservedBands = field(default_factory=ReservedBands)
    columns: ReservedBands = field(default_factory=ReservedBands)


def _axis_bands(
    graph: MetroGraph,
    plan: RoutePlan,
    boundary_of: Callable[[CorridorRegion], int | None],
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> ReservedBands:
    """Measure the reservations *boundary_of* recognises against live geometry.

    Several corridors can claim one boundary over different spans, so a
    boundary's band is the intersection of what each of them leaves clear: a
    channel there has to satisfy every claim, not the most generous one.  An
    empty intersection describes no single corridor, so that boundary is left
    unclaimed and the router derives its own band as it does for any gap the
    ledger never reached.
    """
    from nf_metro.layout.route_reservations import realise_reservation

    spans: dict[int, tuple[float, float]] = {}
    for reservation in plan.reservations:
        boundary = boundary_of(reservation.region)
        if boundary is None:
            continue
        realised = realise_reservation(
            graph, reservation, coordinate_translations=translations
        )
        if realised is None:
            continue
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        held = spans.get(boundary)
        spans[boundary] = (
            (lo, hi) if held is None else (max(held[0], lo), min(held[1], hi))
        )
    return ReservedBands(
        {
            boundary: ReservedBand(lo, hi)
            for boundary, (lo, hi) in sorted(spans.items())
            if hi >= lo - COORD_TOLERANCE
        }
    )


def build_reserved_corridors(
    graph: MetroGraph,
    plan: RoutePlan,
    translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> ReservedCorridors:
    """Measure *plan*'s row- and column-gap reservations against live *graph*.

    *plan* is the frozen ledger settlement consumed, so its claim coordinates
    are projected through *translations* before measurement.
    """
    from nf_metro.layout.route_reservations import ColumnGapRegion, RowGapRegion

    return ReservedCorridors(
        _axis_bands(
            graph,
            plan,
            lambda region: (
                region.lower_row if isinstance(region, RowGapRegion) else None
            ),
            translations,
        ),
        _axis_bands(
            graph,
            plan,
            lambda region: (
                region.right_column if isinstance(region, ColumnGapRegion) else None
            ),
            translations,
        ),
    )
