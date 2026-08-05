"""Every realised gap claim is drawn inside the band its reservation realises.

A row-gap or column-gap ``RouteReservation`` allocates a corridor band for the
specific emitted path segments its claims name.  The drawn geometry has to
consume that allocation: each claim's own polyline points, read through the
claim's ``(path_rank, segment_rank .. segment_end_rank + 1)`` identity, must
lie inside ``[region_start + negative_side_clearance, region_end -
positive_side_clearance]``.  A claimed corridor drawn outside that band was
positioned by a geometry-derived fallback instead of its reservation, which is
exactly what the reservation ledger exists to forbid.

Most of the corpus satisfies that outright, and those fixtures are held to it.
The rest are enumerated in ``KNOWN_UNCONSUMED`` by the ``(path_rank,
segment_rank)`` of the claim itself, so the bound names which leg is out rather
than how many are: an unrecorded claim fails, a recorded one that comes into
band fails until its entry goes, and swapping one leg for another fails too.

What those remaining claims are is measured, not assumed.  All nine are legs
whose coordinate a pre-routing plan fixes and validates the emitted geometry
against, so no post-pass may write them: the fan traverse of a planned
bottom-exit landing (5 claims), a planned exit turn's column (2), and a
convergence trunk (2).  Each plan has to choose that coordinate inside the band
its own reservation realises, which is work at the plan, not a repair after it.
Every other claim in the corpus is drawn inside its band at exact precision,
rather than merely within the tolerance this bound allows.
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

# Fixture -> the ``(path_rank, segment_rank)`` of each realised gap claim drawn
# outside its own reservation's band by more than ``COORD_TOLERANCE``.
# Regenerate by running this module's ``_out_of_band_claims`` over ``_CORPUS``;
# closing a fixture means deleting its entry.
KNOWN_UNCONSUMED: dict[str, frozenset[tuple[int, int]]] = {
    "examples/topologies/bottom_exit_stacked_right_entry_fan.mmd": frozenset(
        {(10, 1), (11, 1)}
    ),
    "examples/topologies/bottom_exit_stacked_right_entry_multiline_branch.mmd": (
        frozenset({(15, 1), (16, 1), (17, 1)})
    ),
    "examples/topologies/exit_lane_settlement_without_crossings.mmd": frozenset(
        {(25, 1)}
    ),
    "examples/topologies/peeloff_straight_drop_near_wall.mmd": frozenset({(12, 1)}),
    "tests/fixtures/regressions/cross_column_perp_entry_overflow.mmd": frozenset(
        {(216, 2), (217, 2)}
    ),
}


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
    recorded = KNOWN_UNCONSUMED.get(rel, frozenset())
    assert set(violations) == recorded, (
        "the claims drawn outside their reserved band are not the ones recorded: "
        f"unrecorded {sorted(set(violations) - recorded)}, recorded but in band "
        f"{sorted(recorded - set(violations))}. An unrecorded claim is a "
        "regression; one that comes into band means dropping its "
        "KNOWN_UNCONSUMED entry, and the fixture's key once the set empties:\n"
        + "\n".join(violations[key] for key in sorted(violations))
    )


def test_the_unconsumed_ledger_names_only_fixtures_that_render() -> None:
    """A stale entry would silently loosen the bound for a live fixture.

    ``test_realised_gap_claims_are_drawn_in_their_reserved_band`` fails outright
    for any ``KNOWN_UNCONSUMED`` entry that stops rendering, so the disjointness
    checked here is what makes that guarantee actually apply to every entry:
    a fixture cannot hide behind ``KNOWN_NOT_RENDERING`` while also carrying a
    claim-consumption bound.
    """
    known = set(KNOWN_UNCONSUMED)
    corpus = {str(path.relative_to(_ROOT)) for path in _CORPUS}
    assert known <= corpus, known - corpus
    assert not (known & KNOWN_NOT_RENDERING), known & KNOWN_NOT_RENDERING
