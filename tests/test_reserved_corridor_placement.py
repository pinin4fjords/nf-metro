"""The router places a reserved corridor from its reservation, on either axis.

A ``RouteReservation`` measures the blockers that bound its corridor over the
corridor's own declared span.  The re-route has to land the channel in the band
that reservation realises rather than re-deriving one from the row or column
edges it happens to have in hand -- those edges name whichever sections sit in
the two grid rows or columns, which is a different, and here a wrong, set of
blockers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from conftest import drawn_claim_coordinates

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import (
    INTER_ROW_EDGE_CLEARANCE,
    INTER_ROW_HEADER_CLEARANCE,
)
from nf_metro.layout.route_plan import build_route_plan_query
from nf_metro.layout.route_reservations import ColumnGapRegion, RowGapRegion
from nf_metro.layout.routing import common
from nf_metro.layout.routing.common import (
    _center_inter_row_channel,
    centre_inter_column_channel,
    column_gap_midpoint,
)
from nf_metro.layout.routing.reserved_bands import (
    ReservedBand,
    ReservedBands,
    build_reserved_corridors,
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

# A same-row bypass trunk whose gap the ledger sizes to exactly the bundle it
# carries, so its band is one coordinate wide, and the section bottoms the trunk
# would size its own depth from sit an OFFSET_STEP shallower than that.
OFF_BAND_TRUNK_FIXTURE = ROOT / "examples" / "topologies" / "merge_pullaway.mmd"
OFF_BAND_TRUNK_BOUNDARY = 1

# A fold whose branch column is entered from a section spanning the boundary,
# so the corridor's own blockers sit 7.5px inboard of the raw column midpoint.
RESERVED_COLUMN_FIXTURE = (
    ROOT / "examples" / "topologies" / "convergence_fold_diamond.mmd"
)
RESERVED_COLUMN_BOUNDARY = 1


def _rendered(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        return build_observed_render_plan(graph, resolve_theme(None, graph))


def _drawn_claim_coordinates(observed, reservation):
    """Every drawn allocation-axis coordinate across all of *reservation*'s claims."""
    for claim in reservation.claims:
        yield from drawn_claim_coordinates(observed, reservation, claim)


def _column_gap_realisations(route_plan, right_column: int):
    query = build_route_plan_query(route_plan)
    for reservation in route_plan.reservations:
        region = reservation.region
        if (
            not isinstance(region, ColumnGapRegion)
            or region.right_column != right_column
        ):
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is not None:
            yield reservation, realised


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
        drawn = list(_drawn_claim_coordinates(observed, reservation))
        assert drawn
        assert (min(drawn) + max(drawn)) / 2 == pytest.approx((lo + hi) / 2, abs=0.01)


def test_reserved_row_corridor_keeps_both_of_its_declared_clearances() -> None:
    """The consequence the raw row edges could not deliver.

    Deriving the band from the row edges leaves the run inside the clearance it
    owes the section that actually bounds it, so the drawn run would sit closer
    to that section than the reservation permits even while the corridor's
    total capacity is ample.
    """
    observed = _rendered(SPANNING_BLOCKER_FIXTURE)
    for reservation, realised in _row_gap_realisations(
        observed.route_plan, SPANNING_BLOCKER_BOUNDARY
    ):
        drawn = list(_drawn_claim_coordinates(observed, reservation))
        assert drawn
        assert min(drawn) >= (
            realised.region_start + reservation.negative_side_clearance - 0.01
        )
        assert max(drawn) <= (
            realised.region_end - reservation.positive_side_clearance + 0.01
        )


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


def test_a_reserved_band_keeps_an_oversized_stagger_distinct() -> None:
    """Two lanes of one bundle never resolve onto a single coordinate.

    A boundary band is the intersection of every claim crossing it, so it can be
    narrower than a bundle claiming it, and clamping each lane into it in turn
    would seat them all on the same coordinate: the two lines would draw as one
    stroke and one of them would be invisible.  Distinctness is kept and the
    overrun is what ``assert_reservations_are_settled`` reports.
    """
    band = ReservedBand(200.0, 260.0)
    centre = _center_inter_row_channel(0.0, 0.0, 0.0, reserved=band)
    assert centre == pytest.approx(230.0)
    lanes = [
        _center_inter_row_channel(0.0, 0.0, offset, reserved=band)
        for offset in (-400.0, -4.0, 4.0, 400.0)
    ]
    assert lanes == sorted(lanes)
    assert len(set(lanes)) == len(lanes)
    assert lanes == pytest.approx([-170.0, 226.0, 234.0, 630.0])


def test_a_lone_reserved_run_is_held_inside_its_band() -> None:
    """Containment applies to a run with no stagger to keep distinct."""
    band = ReservedBand(200.0, 260.0)
    assert band.hold(400.0) == band.hi
    assert band.hold(0.0) == band.lo
    assert band.hold(230.0) == pytest.approx(230.0)


def test_a_band_narrower_than_nothing_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="narrower than nothing"):
        ReservedBand(260.0, 200.0)


def test_an_unclaimed_boundary_reports_no_band() -> None:
    bands = ReservedBands({2: ReservedBand(200.0, 260.0)})
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
    bands = build_reserved_corridors(graph, observed.route_plan).rows
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


def test_a_reserved_column_corridor_lands_on_the_band_its_reservation_realises() -> (
    None
):
    """The column-axis twin of the row corridor above.

    The raw column midpoint is bounded by whichever sections occupy the two
    columns; the reservation is bounded by the blockers over the corridor's own
    run, and here the two disagree.
    """
    observed = _rendered(RESERVED_COLUMN_FIXTURE)
    found = list(
        _column_gap_realisations(observed.route_plan, RESERVED_COLUMN_BOUNDARY)
    )
    assert found, "fixture no longer reserves the corridor under test"
    raw = column_gap_midpoint(
        observed.plan.graph, RESERVED_COLUMN_BOUNDARY - 1, RESERVED_COLUMN_BOUNDARY
    )
    for reservation, realised in found:
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        assert raw != pytest.approx((lo + hi) / 2, abs=0.01), (
            "fixture no longer distinguishes the reservation from the raw gap"
        )
        drawn = list(_drawn_claim_coordinates(observed, reservation))
        assert drawn
        assert (min(drawn) + max(drawn)) / 2 == pytest.approx((lo + hi) / 2, abs=0.01)


def test_a_reserved_column_band_is_used_without_consulting_the_raw_gap() -> None:
    observed = _rendered(RESERVED_COLUMN_FIXTURE)
    graph = observed.plan.graph
    band = ReservedBand(500.0, 560.0)
    reserved = ReservedBands({RESERVED_COLUMN_BOUNDARY: band})
    placed = centre_inter_column_channel(
        graph,
        RESERVED_COLUMN_BOUNDARY - 1,
        RESERVED_COLUMN_BOUNDARY,
        reserved=reserved,
    )
    assert placed == pytest.approx(530.0)


def test_a_column_corridor_spanning_further_than_one_boundary_keeps_the_raw_gap() -> (
    None
):
    """Only adjacent columns name one boundary, so only they claim a band."""
    observed = _rendered(RESERVED_COLUMN_FIXTURE)
    graph = observed.plan.graph
    reserved = ReservedBands({2: ReservedBand(500.0, 560.0)})
    assert centre_inter_column_channel(graph, 0, 2, reserved=reserved) == pytest.approx(
        column_gap_midpoint(graph, 0, 2)
    )


def test_a_bypass_trunk_is_held_on_the_band_its_reservation_realises() -> None:
    """A trunk depth derived from section bottoms is held inside its band.

    The trunk sizes its own depth from the boxes it has to clear plus a
    clearance, which is a proxy for the blockers the reservation measured; the
    reservation's answer is the one the corridor is allocated.
    """
    observed = _rendered(OFF_BAND_TRUNK_FIXTURE)
    found = list(_row_gap_realisations(observed.route_plan, OFF_BAND_TRUNK_BOUNDARY))
    assert found, "fixture no longer reserves the corridor under test"
    for reservation, realised in found:
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        drawn = list(_drawn_claim_coordinates(observed, reservation))
        assert drawn
        assert lo - 0.01 <= min(drawn) and max(drawn) <= hi + 0.01
