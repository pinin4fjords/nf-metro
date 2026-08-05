"""Every realised gap claim is drawn inside the band its reservation realises.

A row-gap or column-gap ``RouteReservation`` allocates a corridor band for the
specific emitted path segments its claims name.  The drawn geometry has to
consume that allocation: each claim's own polyline points, read through the
claim's ``(path_rank, segment_rank .. segment_end_rank + 1)`` identity, must
lie inside ``[region_start + negative_side_clearance, region_end -
positive_side_clearance]``.  A claimed corridor drawn outside that band was
positioned by a geometry-derived fallback instead of its reservation, which is
exactly what the reservation ledger exists to forbid.

The whole corpus satisfies that, and every fixture is held to it with no
exceptions.  All but two claims are drawn inside their band at exact precision,
rather than merely within the tolerance this bound allows.

Those two are out by exactly one pixel: the ``hic_reads`` lane turning up into
``scaffolding`` in each of the two ``genomeassembly`` maps.  Its column gap is
50px wide and carries three lanes.  The lowest is a planned exit turn's descent,
whose coordinate is the ``ExitTurnAxis``, so no pass may reseat it; the two above
it are a bundle seated from that descent by :func:`cotravelling_lane_clearance`
and one ``OFFSET_STEP``.  The descent stands 4px above the band floor and the
stack from it takes 15px of the band's 18, so the upper lane ends 1px past the
far clearance.

That shortfall is a position rather than a width -- the reservation's own
``minimum_width`` is met with 14px to spare -- so nothing the boundary is asked
for states it, and reaching it needs the pinned descent to bound the boundary the
way ``launch_anchors`` makes a launch station bound it.  Stating it that way was
built and measured, and settlement cannot pay it: ``SETTLEMENT_QUANTUM`` is
``COORD_TOLERANCE``, ``_settle_axis`` acts only above ``COORD_TOLERANCE``, and
``ReservationCoordinateTranslation`` refuses an amount that small, so the least
translation settlement can express is 2px and a 1px deficit is below the
resolution the ledger works at.  Lowering all three floors does close both claims
for +1px of map width each, and costs
``examples/topologies/exit_run_three_drop_columns.mmd`` its render -- settlement
then changes route topology under ``_assert_settlement_decisions_frozen`` -- and
puts an ``ambiguous_exit_continuation`` lane 4.5px out of band between two pinned
lanes 17.5px apart that owe each other 22px.  The coordinate is the exit-turn
plan's, and so is the room the lanes beside it need.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.route_plan import build_route_plan_query
from nf_metro.layout.route_reservations import (
    ColumnGapRegion,
    RowGapRegion,
    drawn_corridor_containment,
)
from nf_metro.render.svg import build_observed_render_plan

_ROOT = Path(__file__).parents[1]


def _corpus() -> list[Path]:
    paths = sorted((_ROOT / "examples").rglob("*.mmd"))
    paths += sorted((_ROOT / "tests" / "fixtures").rglob("*.mmd"))
    paths += sorted((_ROOT / "tests" / "fixtures").rglob("*.metro"))
    return paths


_CORPUS = _corpus()

# Fixtures that never reach a route plan at all: fixtures under `invalid/`
# and `nextflow/` are exercised by their own tests for the error they raise,
# and the frozen determinism/topology fixtures abort on a routing invariant
# tracked by other tests. None of them can be held to a claim-consumption
# bound, so their failure to render is not itself a finding here.
KNOWN_NOT_RENDERING = frozenset(
    {
        "tests/fixtures/hash_seed_determinism/seed_15.mmd",
        "tests/fixtures/hash_seed_determinism/seed_41.mmd",
        "tests/fixtures/hash_seed_determinism/seed_77.mmd",
        "tests/fixtures/invalid/backward_feed_rl.mmd",
        "tests/fixtures/invalid/merge_trunk_rightward_source.mmd",
        "tests/fixtures/invalid/mixed_entry_opposing.mmd",
        "tests/fixtures/invalid/mixed_entry_perpendicular.mmd",
        "tests/fixtures/nextflow/duplicate_processes.mmd",
        "tests/fixtures/nextflow/flat_pipeline.mmd",
        "tests/fixtures/nextflow/unquoted_labels.mmd",
        "tests/fixtures/nextflow/variant_calling.mmd",
        "tests/fixtures/nextflow/with_subworkflows.mmd",
        "tests/fixtures/topologies/twoline_fanout_up.mmd",
    }
)


def _out_of_band_claims(path: Path) -> dict[tuple[int, int], str] | None:
    """*path*'s claims drawn outside their band, or ``None`` if it cannot render.

    Keyed by the claim's own ``(path_rank, segment_rank)`` so the bound names the
    leg, with the measured band and drawn interval as the value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
            observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    except Exception:  # noqa: BLE001 - erroring fixtures have their own tests
        return None

    route_plan = observed.route_plan
    if route_plan is None:
        return {}
    query = build_route_plan_query(route_plan)
    polylines = observed.plan.route_polylines
    violations: dict[tuple[int, int], str] = {}
    for reservation in route_plan.reservations:
        if not isinstance(reservation.region, RowGapRegion | ColumnGapRegion):
            continue
        realised = query.realised_reservation(reservation.id)
        if realised is None:
            continue
        for claim in reservation.claims:
            drawn = drawn_corridor_containment(
                reservation, realised, polylines, (claim,)
            )
            if (
                min(drawn.negative_side_slack, drawn.positive_side_slack)
                >= -COORD_TOLERANCE
            ):
                continue
            violations[claim.path_rank, claim.segment_rank] = (
                f"{reservation.id} claim {claim.member_id} "
                f"(path {claim.path_rank}, segments {claim.segment_rank}.."
                f"{claim.segment_end_rank}): drawn "
                f"[{drawn.drawn_start:.2f}, {drawn.drawn_end:.2f}] outside band "
                f"[{drawn.band_start:.2f}, {drawn.band_end:.2f}]"
            )
    return violations


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(_ROOT)) for p in _CORPUS]
)
def test_realised_gap_claims_are_drawn_in_their_reserved_band(path: Path) -> None:
    rel = str(path.relative_to(_ROOT))
    violations = _out_of_band_claims(path)
    if violations is None:
        if rel in KNOWN_NOT_RENDERING:
            pytest.skip("fixture does not render")
        pytest.fail(
            f"{rel} raised while building its render plan. A fixture that stops "
            "rendering cannot be held to a claim-consumption bound; either fix "
            "the regression or add it to KNOWN_NOT_RENDERING with the reason it "
            "cannot render."
        )
    assert not violations, (
        "a claim drawn outside its reserved band was positioned by a "
        "geometry-derived fallback rather than by its reservation:\n"
        + "\n".join(violations[key] for key in sorted(violations))
    )
