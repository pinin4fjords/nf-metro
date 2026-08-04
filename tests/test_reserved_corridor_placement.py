"""The router places a reserved inter-row corridor from its reservation.

A row-gap ``RouteReservation`` measures the blockers that bound its corridor
over the corridor's own declared span.  Once envelope settlement has widened a
boundary to hold that corridor, the settled re-route has to land the channel in
the band the reservation realises rather than re-deriving one from the row
edges it happens to have in hand -- those edges name whichever sections sit in
the two grid rows, which is a different, and here a wrong, set of blockers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import (
    INTER_ROW_EDGE_CLEARANCE,
    INTER_ROW_HEADER_CLEARANCE,
)
from nf_metro.layout.route_plan import build_route_plan_query
from nf_metro.layout.route_reservations import RowGapRegion
from nf_metro.layout.routing import common
from nf_metro.layout.routing.common import _center_inter_row_channel
from nf_metro.layout.routing.reserved_bands import (
    ReservedBand,
    ReservedRowBands,
    build_reserved_row_bands,
)
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]

# A row-spanning section bounds the corridor between grid rows 1 and 2 from
# above, while the row-1 box that the raw row edges name stops 63.6px higher.
# Settlement widens that boundary for the corridor, so the re-route consumes
# the ledger.
SPANNING_BLOCKER_FIXTURE = (
    ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd"
)
SPANNING_BLOCKER_BOUNDARY = 2


def _rendered(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        return build_observed_render_plan(graph, resolve_theme(None, graph))


def _row_gap_realisations(route_plan, lower_row: int):
    query = build_route_plan_query(route_plan)
    for reservation in route_plan.reservations:
        region = reservation.region
        if not isinstance(region, RowGapRegion) or region.lower_row != lower_row:
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is not None:
            yield reservation, realised


def test_reserved_row_corridor_lands_on_the_band_its_reservation_realises() -> None:
    observed = _rendered(SPANNING_BLOCKER_FIXTURE)
    found = list(_row_gap_realisations(observed.route_plan, SPANNING_BLOCKER_BOUNDARY))
    assert found, "fixture no longer reserves the corridor under test"
    for reservation, realised in found:
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        assert realised.coordinate == pytest.approx((lo + hi) / 2, abs=0.01)


def test_reserved_row_corridor_keeps_both_of_its_declared_clearances() -> None:
    """The consequence the raw row edges could not deliver.

    Deriving the band from the row edges leaves the run inside the clearance it
    owes the section that actually bounds it, which shows up as a negative
    side slack even while the corridor's total capacity is ample.
    """
    observed = _rendered(SPANNING_BLOCKER_FIXTURE)
    for _reservation, realised in _row_gap_realisations(
        observed.route_plan, SPANNING_BLOCKER_BOUNDARY
    ):
        assert realised.negative_side_slack >= -0.01
        assert realised.positive_side_slack >= -0.01


def test_a_reserved_band_is_used_without_consulting_the_raw_gap(monkeypatch) -> None:
    """The narrow-gap fallback is unreachable for a corridor that owns a band.

    The fallback is guarded by ``_inter_row_band_fits`` on the raw edges, so a
    reserved channel that never asks that question can never take it.
    """

    def _refuse(*_args: float) -> bool:
        raise AssertionError("a reserved corridor consulted the raw gap")

    monkeypatch.setattr(common, "_inter_row_band_fits", _refuse)
    # Raw edges far too close together for either clearance: without the
    # reservation this is exactly the case that biases the run to the header.
    placed = _center_inter_row_channel(
        100.0, 110.0, reserved=ReservedBand(200.0, 260.0)
    )
    assert placed == pytest.approx(230.0)


def test_a_reserved_band_holds_an_oversized_stagger_inside_itself() -> None:
    band = ReservedBand(200.0, 260.0)
    assert _center_inter_row_channel(0.0, 0.0, 400.0, reserved=band) == band.hi
    assert _center_inter_row_channel(0.0, 0.0, -400.0, reserved=band) == band.lo


def test_a_band_narrower_than_nothing_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="narrower than nothing"):
        ReservedBand(260.0, 200.0)


def test_an_unclaimed_boundary_reports_no_band() -> None:
    bands = ReservedRowBands({2: ReservedBand(200.0, 260.0)})
    assert bands.at(2) == ReservedBand(200.0, 260.0)
    assert bands.at(3) is None
    assert bands.at(None) is None


def test_published_bands_are_the_reservation_clearances_at_the_boundary() -> None:
    """What the router reads back is the ledger's own measurement.

    Several corridors can claim one boundary, so the band is the intersection
    of what each leaves clear, and a published band always holds a channel.
    """
    observed = _rendered(SPANNING_BLOCKER_FIXTURE)
    graph = observed.plan.graph
    bands = build_reserved_row_bands(graph, observed.route_plan)
    assert bands.bands
    for lower_row, band in bands.bands.items():
        assert band.hi >= band.lo
        claims = list(_row_gap_realisations(observed.route_plan, lower_row))
        assert band.lo == pytest.approx(
            max(
                item[1].region_start + item[0].negative_side_clearance
                for item in claims
            )
        )
        assert band.hi == pytest.approx(
            min(item[1].region_end - item[0].positive_side_clearance for item in claims)
        )


def test_row_gap_clearances_are_the_ones_the_raw_derivation_uses() -> None:
    """The reservation and the raw derivation differ only in their blockers."""
    observed = _rendered(SPANNING_BLOCKER_FIXTURE)
    for reservation, _realised in _row_gap_realisations(
        observed.route_plan, SPANNING_BLOCKER_BOUNDARY
    ):
        assert reservation.negative_side_clearance == INTER_ROW_EDGE_CLEARANCE
        assert reservation.positive_side_clearance == INTER_ROW_HEADER_CLEARANCE
