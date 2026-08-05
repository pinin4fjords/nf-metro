"""Row-gap corridor bands read from an existing reservation ledger."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nf_metro.layout.constants import COORD_TOLERANCE

if TYPE_CHECKING:
    from nf_metro.layout.route_plan import RoutePlan
    from nf_metro.layout.route_reservations import ReservationCoordinateTranslation
    from nf_metro.parser.model import MetroGraph


@dataclass(frozen=True, slots=True)
class ReservedBand:
    """The clear span reserved for channels crossing one row boundary."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo - COORD_TOLERANCE:
            raise ValueError("a reserved band cannot be narrower than nothing")

    def place(self, offset: float) -> float:
        """Place a channel at *offset* from centre, held inside the band."""
        centre = (self.lo + self.hi) / 2
        return min(max(centre + offset, self.lo), self.hi)


@dataclass(frozen=True, slots=True)
class ReservedRowBands:
    """Realised row-gap bands keyed by the lower row of each boundary."""

    bands: Mapping[int, ReservedBand] = field(default_factory=dict)

    def at(self, lower_row: int | None) -> ReservedBand | None:
        if lower_row is None:
            return None
        return self.bands.get(lower_row)


def build_reserved_row_bands(
    graph: MetroGraph,
    plan: RoutePlan,
    translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> ReservedRowBands:
    """Measure the clear band left by every row-gap claim in *plan*."""
    from nf_metro.layout.route_reservations import RowGapRegion, realise_reservation

    spans: dict[int, tuple[float, float]] = {}
    for reservation in plan.reservations:
        region = reservation.region
        if not isinstance(region, RowGapRegion):
            continue
        realised = realise_reservation(
            graph,
            reservation,
            coordinate_translations=translations,
        )
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
