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
The rest are enumerated in ``KNOWN_UNCONSUMED`` with the count each one carries,
which ratchets two ways: a fixture that gains an out-of-band claim fails, and so
does one that loses every one of them without the entry being removed.

What those remaining claims are is measured, not assumed.  Each one is a leg
:func:`~nf_metro.layout.routing.normalize._hold_runs_in_corridor_clearance`
cannot reseat, and there are exactly five reasons for that:

* A pre-routing plan owns the segment's coordinates and validates the emitted
  geometry against them, so no post-pass may write it (9 claims).
* The leg is a route's end leg, whose coordinate is the port marker it lands on
  (3 claims).
* A diagonal flanks the leg, and moving its coordinate would change that
  diagonal's angle (2 claims).
* Two legs need one coordinate between them, and their bands are measured at
  two *different* boundaries, so no one boundary's width states the room the
  pair takes (1 claim).
* The band the pass measures from the leg's own endpoint sections is wider than
  the band its reservation realises from the corridor's topology span, so the
  pass holds the leg inside a band the ledger scores it outside of (1 claim).

None of the five is a router that fails to read the ledger, and none is closed
by moving a leg the pass already reaches.  Two corridors confined at one
boundary, each needing a coordinate the other cannot leave it, are absent from
that list because ``RouteReservation.peer_width`` states the room the pair takes
and settlement widens the boundary for both.
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

# Fixture -> how many of its realised gap claims are drawn outside their own
# reservation's band by more than ``COORD_TOLERANCE``.  Regenerate by running
# this module's ``_out_of_band_claims`` over ``_CORPUS``; closing a fixture means
# deleting its entry.  Two further claims sit within one tolerance of their band,
# which is the width this codebase treats two coordinates as equal within, so
# they are not counted here.
KNOWN_UNCONSUMED = {
    "examples/topologies/bottom_exit_stacked_right_entry_fan.mmd": 2,
    "examples/topologies/bottom_exit_stacked_right_entry_multiline_branch.mmd": 3,
    "examples/topologies/convergence_stacked_sink.mmd": 1,
    "examples/topologies/exit_lane_settlement_without_crossings.mmd": 1,
    "examples/topologies/fan_bypass_shared_band.mmd": 1,
    "examples/topologies/peeloff_straight_drop_near_wall.mmd": 1,
    "examples/topologies/top_entry_bundle_offset_seam.mmd": 1,
    "examples/variantbenchmarking_auto.mmd": 1,
    "tests/fixtures/regressions/cross_column_perp_entry_overflow.mmd": 2,
    "tests/fixtures/regressions/lr_perpendicular_ports_overflow.mmd": 1,
    "tests/fixtures/tb_exit_terminal_on_carrier.mmd": 2,
}


def _out_of_band_claims(path: Path) -> list[str] | None:
    """Each claim of *path* drawn outside its band, or ``None`` if it cannot render."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
            observed = build_observed_render_plan(graph, resolve_theme(None, graph))
    except Exception:  # noqa: BLE001 - erroring fixtures have their own tests
        return None

    route_plan = observed.route_plan
    if route_plan is None:
        return []
    query = build_route_plan_query(route_plan)
    polylines = observed.plan.route_polylines
    violations: list[str] = []
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
            violations.append(
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
    allowed = KNOWN_UNCONSUMED.get(rel, 0)
    assert len(violations) == allowed, (
        f"{len(violations)} claimed corridor(s) drawn outside their reserved band, "
        f"{allowed} recorded. Gaining one is a regression; losing one means "
        "tightening this fixture's KNOWN_UNCONSUMED count, or deleting the entry "
        "once it reaches zero:\n" + "\n".join(violations)
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
