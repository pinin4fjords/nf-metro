"""Every realised gap claim is drawn inside the band its reservation realises.

A row-gap or column-gap ``RouteReservation`` allocates a corridor band for the
specific emitted path segments its claims name.  The drawn geometry has to
consume that allocation: each claim's own polyline points, read through the
claim's ``(path_rank, segment_rank .. segment_end_rank + 1)`` identity, must
lie inside ``[region_start + negative_side_clearance, region_end -
positive_side_clearance]``.  A claimed corridor drawn outside that band was
positioned by a geometry-derived fallback instead of its reservation, which is
exactly what the reservation ledger exists to forbid.

Most of the corpus already satisfies that outright, and those fixtures are held
to it.  The rest are enumerated in ``KNOWN_UNCONSUMED`` with the count each one
carries, which ratchets two ways: a fixture that gains an out-of-band claim
fails, and so does one that loses every one of them without the entry being
removed.

What those remaining claims are is measured, not assumed.  In every one of them
the drawn coordinate equals the coordinate the claim itself recorded, and the
claim's own coordinate is what lies outside the band -- there is no case of an
allocation sitting inside its band with the route drawn elsewhere.  So the
router consumes the ledger faithfully; what is short is the demand's side
clearance.  Settlement guarantees capacity, and does: these reservations carry
non-negative ``capacity_slack``, the region is wide enough for the bundle.  The
bundle is seated off-centre within it, so one ``*_side_slack`` goes negative
while the other absorbs the whole surplus.  Closing one therefore means moving
where a handler seats its bundle inside an allocation it already honours, which
changes drawn geometry, rather than teaching the router to read the ledger.
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

# Fixture -> how many of its realised gap claims are drawn outside their own
# reservation's band.  Regenerate by running this module's ``_out_of_band_claims``
# over ``_CORPUS``; closing a fixture means deleting its entry.
KNOWN_UNCONSUMED = {
    "examples/longread_variant_calling.mmd": 2,
    "examples/topologies/bottom_exit_stacked_right_entry_fan.mmd": 2,
    "examples/topologies/bottom_exit_stacked_right_entry_multiline_branch.mmd": 3,
    "examples/topologies/bypass_left_entry_from_right.mmd": 1,
    "examples/topologies/bypass_leftward_far_side_entry.mmd": 1,
    "examples/topologies/convergence_fold_diamond.mmd": 2,
    "examples/topologies/convergence_sink_fold.mmd": 4,
    "examples/topologies/convergence_stacked_sink.mmd": 2,
    "examples/topologies/cross_row_gap_wrap.mmd": 1,
    "examples/topologies/dogleg_exempt_distinct.mmd": 1,
    "examples/topologies/dogleg_exempt_sameline.mmd": 1,
    "examples/topologies/exit_lane_settlement_without_crossings.mmd": 1,
    "examples/topologies/fan_bypass_shared_band.mmd": 1,
    "examples/topologies/fold_split_targets.mmd": 2,
    "examples/topologies/leftward_up_exit_turn_order.mmd": 2,
    "examples/topologies/merge_right_entry.mmd": 2,
    "examples/topologies/merge_trunk_out_of_range_section.mmd": 1,
    "examples/topologies/opposing_bypass_corridor.mmd": 2,
    "examples/topologies/opposing_return_row_pair.mmd": 1,
    "examples/topologies/packed_cell_right_exit_left_entry_wrap.mmd": 1,
    "examples/topologies/packed_multiline_serpentine_grid.mmd": 1,
    "examples/topologies/peeloff_straight_drop_near_wall.mmd": 1,
    "examples/topologies/right_entry_over_top_tall_upstream.mmd": 2,
    "examples/topologies/right_entry_wrap_no_fan.mmd": 1,
    "examples/topologies/same_line_fan_distinct_descent.mmd": 1,
    "examples/topologies/same_side_culdesac.mmd": 1,
    "examples/topologies/samerow_left_exit_far_left_entry.mmd": 1,
    "examples/topologies/straddling_fanout_junction.mmd": 2,
    "examples/topologies/tb_bottom_exit_fork_diamond.mmd": 1,
    "examples/topologies/tb_left_exit_step.mmd": 2,
    "examples/topologies/tb_right_entry_stack.mmd": 1,
    "examples/topologies/top_entry_bundle_offset_seam.mmd": 1,
    "examples/variantbenchmarking.mmd": 1,
    "examples/variantbenchmarking_auto.mmd": 2,
    "tests/fixtures/opposing_entry_partial_port_line.mmd": 2,
    "tests/fixtures/planned_compatibility_channel_collision.mmd": 8,
    "tests/fixtures/regressions/away_exit_wrap_interior_left.mmd": 1,
    "tests/fixtures/regressions/cross_column_perp_entry_overflow.mmd": 2,
    "tests/fixtures/regressions/lr_perpendicular_ports_overflow.mmd": 1,
    "tests/fixtures/target_entry_runway_bypass.mmd": 3,
    "tests/fixtures/tb_exit_terminal_on_carrier.mmd": 2,
    "tests/fixtures/through_section/riboseq_packed_lr.mmd": 1,
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
    return violations


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(_ROOT)) for p in _CORPUS]
)
def test_realised_gap_claims_are_drawn_in_their_reserved_band(path: Path) -> None:
    violations = _out_of_band_claims(path)
    if violations is None:
        pytest.skip("fixture does not render")
    allowed = KNOWN_UNCONSUMED.get(str(path.relative_to(_ROOT)), 0)
    assert len(violations) <= allowed, (
        f"{len(violations)} claimed corridor(s) drawn outside their reserved band, "
        f"{allowed} recorded:\n" + "\n".join(violations)
    )
    if allowed:
        assert violations, (
            "every claimed corridor here is now consumed; delete this fixture's "
            "KNOWN_UNCONSUMED entry"
        )


def test_the_unconsumed_ledger_names_only_fixtures_that_render() -> None:
    """A stale entry would silently loosen the bound for a live fixture."""
    known = set(KNOWN_UNCONSUMED)
    corpus = {str(path.relative_to(_ROOT)) for path in _CORPUS}
    assert known <= corpus, known - corpus
    assert sum(KNOWN_UNCONSUMED.values()) == 71
