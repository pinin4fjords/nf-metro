"""The settled reroute consumes row-gap allocations from its input ledger."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.route_plan import DemandAxis, build_route_plan_query
from nf_metro.layout.route_reservations import (
    ReservationCoordinateTranslation,
    RowGapRegion,
)
from nf_metro.layout.routing import common
from nf_metro.layout.routing.common import _center_inter_row_channel
from nf_metro.layout.routing.reserved_bands import (
    ReservedBand,
    ReservedRowBands,
    build_reserved_row_bands,
)
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tb_exit_terminal_on_carrier.mmd"
BOUNDARY = 2


def _rendered():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(FIXTURE.read_text(), source_dir=str(FIXTURE.parent))
        return build_observed_render_plan(graph, resolve_theme(None, graph))


def _row_claims(plan, lower_row: int):
    return tuple(
        item
        for item in plan.reservations
        if isinstance(item.region, RowGapRegion) and item.region.lower_row == lower_row
    )


def test_settled_corridor_honours_both_side_clearances() -> None:
    observed = _rendered()
    query = build_route_plan_query(observed.route_plan)
    claims = _row_claims(observed.route_plan, BOUNDARY)
    assert claims
    for reservation in claims:
        realised = query.realised_reservation(reservation.id)
        assert realised is not None
        assert realised.capacity_slack >= -COORD_TOLERANCE
        assert realised.negative_side_slack >= -COORD_TOLERANCE
        assert realised.positive_side_slack >= -COORD_TOLERANCE


def test_reserved_band_wins_over_raw_row_edges(monkeypatch) -> None:
    def refuse_raw_gap(*_args: float) -> bool:
        raise AssertionError("reserved placement consulted the raw row edges")

    monkeypatch.setattr(common, "_inter_row_band_fits", refuse_raw_gap)
    assert _center_inter_row_channel(
        100.0,
        110.0,
        reserved=ReservedBand(200.0, 260.0),
    ) == pytest.approx(230.0)


def test_reserved_band_clamps_lane_offsets() -> None:
    band = ReservedBand(200.0, 260.0)
    assert band.place(400.0) == band.hi
    assert band.place(-400.0) == band.lo


def test_reserved_band_rejects_an_empty_span() -> None:
    with pytest.raises(ValueError, match="narrower than nothing"):
        ReservedBand(260.0, 200.0)


def test_unclaimed_boundary_has_no_reserved_band() -> None:
    bands = ReservedRowBands({BOUNDARY: ReservedBand(200.0, 260.0)})
    assert bands.at(BOUNDARY) == ReservedBand(200.0, 260.0)
    assert bands.at(BOUNDARY + 1) is None
    assert bands.at(None) is None


def test_published_band_is_the_intersection_of_its_claims() -> None:
    observed = _rendered()
    graph = observed.plan.graph
    query = build_route_plan_query(observed.route_plan)
    bands = build_reserved_row_bands(graph, observed.route_plan)
    for lower_row, band in bands.bands.items():
        claims = _row_claims(observed.route_plan, lower_row)
        realisations = tuple(query.realised_reservation(item.id) for item in claims)
        assert all(item is not None for item in realisations)
        assert band.lo == pytest.approx(
            max(
                realised.region_start + reservation.negative_side_clearance
                for reservation, realised in zip(claims, realisations, strict=True)
                if realised is not None
            )
        )
        assert band.hi == pytest.approx(
            min(
                realised.region_end - reservation.positive_side_clearance
                for reservation, realised in zip(claims, realisations, strict=True)
                if realised is not None
            )
        )


def test_band_measurement_receives_settlement_translations(monkeypatch) -> None:
    observed = _rendered()
    translations = (
        ReservationCoordinateTranslation(
            axis=DemandAxis.Y,
            coordinate=100.0,
            amount=20.0,
        ),
    )
    received = []

    def capture(_graph, _reservation, **kwargs):
        received.append(kwargs.get("coordinate_translations"))
        return None

    monkeypatch.setattr(
        "nf_metro.layout.route_reservations.realise_reservation",
        capture,
    )
    build_reserved_row_bands(observed.plan.graph, observed.route_plan, translations)
    assert received and all(item == translations for item in received)
