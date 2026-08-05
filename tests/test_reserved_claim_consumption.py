"""Every realised gap claim is drawn inside the band its reservation realises.

A row-gap or column-gap ``RouteReservation`` allocates a corridor band for the
specific emitted path segments its claims name.  The drawn geometry has to
consume that allocation: each claim's own polyline points, read through the
claim's ``(path_rank, segment_rank .. segment_end_rank + 1)`` identity, must
lie inside ``[region_start + negative_side_clearance, region_end -
positive_side_clearance]``.  A claimed corridor drawn outside that band was
positioned by a geometry-derived fallback instead of its reservation, which is
exactly what the reservation ledger exists to forbid.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.route_plan import build_route_plan_query
from nf_metro.layout.route_reservations import ColumnGapRegion, RowGapRegion
from nf_metro.render.svg import build_observed_render_plan

_ROOT = Path(__file__).parents[1]


def _corpus() -> list[Path]:
    paths = sorted((_ROOT / "examples").rglob("*.mmd"))
    paths += sorted((_ROOT / "tests" / "fixtures").rglob("*.mmd"))
    paths += sorted((_ROOT / "tests" / "fixtures").rglob("*.metro"))
    return paths


_CORPUS = _corpus()


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(_ROOT)) for p in _CORPUS]
)
def test_realised_gap_claims_are_drawn_in_their_reserved_band(path: Path) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
            observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    except Exception as exc:  # noqa: BLE001 - erroring fixtures have their own tests
        pytest.skip(f"fixture does not render: {type(exc).__name__}")

    route_plan = observed.route_plan
    if route_plan is None:
        return
    query = build_route_plan_query(route_plan)
    polylines = observed.plan.route_polylines
    violations: list[str] = []
    for reservation in route_plan.reservations:
        region = reservation.region
        if isinstance(region, RowGapRegion):
            axis = 1
        elif isinstance(region, ColumnGapRegion):
            axis = 0
        else:
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is None:
            continue
        lo = realised.region_start + reservation.negative_side_clearance
        hi = realised.region_end - reservation.positive_side_clearance
        for claim in reservation.claims:
            drawn = [
                polylines[claim.path_rank][rank][axis]
                for rank in range(claim.segment_rank, claim.segment_end_rank + 2)
            ]
            if all(lo - COORD_TOLERANCE <= v <= hi + COORD_TOLERANCE for v in drawn):
                continue
            violations.append(
                f"{reservation.id} claim {claim.member_id} "
                f"(path {claim.path_rank}, segments {claim.segment_rank}.."
                f"{claim.segment_end_rank}): drawn "
                f"[{min(drawn):.2f}, {max(drawn):.2f}] outside band "
                f"[{lo:.2f}, {hi:.2f}]"
            )
    assert not violations, (
        f"{len(violations)} claimed corridor(s) drawn outside their reserved "
        "band:\n" + "\n".join(violations)
    )
