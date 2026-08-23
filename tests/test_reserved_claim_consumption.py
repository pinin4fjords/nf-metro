"""Every realised gap claim is drawn inside the band its reservation realises.

A row-gap or column-gap ``RouteReservation`` allocates a corridor band for the
specific emitted path segments its claims name.  The drawn geometry has to
consume that allocation: each claim's own polyline points, read through the
claim's ``(path_rank, segment_rank .. segment_end_rank + 1)`` identity, must
lie inside ``[region_start + negative_side_clearance, region_end -
positive_side_clearance]``.  A claimed corridor drawn outside that band was
positioned by a geometry-derived fallback instead of its reservation, which is
exactly what the reservation ledger exists to forbid.

Every fixture is held to that, minus the corridors the ledger reserves for
fewer lanes than they carry: where two independently-raised reservations
realise one band and neither sizes it for the other's lanes, no arrangement
puts every claim inside, and the separation stage draws one of them outside
rather than paint two distinct lines as a single stroke.  Those are enumerated
by identity in ``UNDERSIZED_CORRIDORS``, so the population cannot grow without
the ratchet naming the new member.  The slack is a
:func:`~nf_metro.layout.route_reservations.measured_distance`, so a run drawn
flush against a band edge scores as flush rather than as overrunning it by the
floating-point residue from subtracting two canvas coordinates.
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


# Claims drawn outside their band by no more than one ``COORD_TOLERANCE``, which
# is this codebase's definition of two coordinates being equal, so they satisfy
# the bound below.  They are enumerated by identity all the same: "zero beyond
# tolerance" is only worth something if the population sitting just inside the
# tolerance cannot grow without anyone noticing.  Adding one here is a decision
# to be argued for, not a side effect.
WITHIN_TOLERANCE_OVERHANGS: frozenset[tuple[str, int, int]] = frozenset(
    {
        ("examples/genomeassembly.mmd", 39, 3),
        ("tests/fixtures/regressions/entry_trunk_row_bow.mmd", 29, 2),
    }
)


# Claims whose corridor is reserved for fewer lanes than it carries, so no
# placement satisfies every claim realising that band.  Two reservations raised
# independently over one row gap each size it for their own lanes alone; the
# lane the separation stage moves to keep the two apart therefore leaves the
# band, which is the lesser of the two outcomes -- the alternative is two
# distinct lines drawn as one stroke.  Enumerated by ``(fixture, path_rank,
# segment_rank)`` and the width of the shortfall so the corridor that grows an
# extra lane, or the one that gets sized correctly, both show up as a change.
UNDERSIZED_CORRIDORS: frozenset[tuple[str, int, int, float]] = frozenset(
    {
        (
            "tests/fixtures/curve_invariant_repros/inter_row_corridor_overflow.mmd",
            111,
            2,
            4.0,
        ),
    }
)


def _claim_overhangs(path: Path) -> dict[tuple[int, int], tuple[float, str]] | None:
    """*path*'s claims drawn outside their band, or ``None`` if it cannot render.

    Keyed by the claim's own ``(path_rank, segment_rank)`` so a bound names the
    leg, valued by how far outside it is drawn and the geometry that was
    measured.  Every overhang the ledger's own measurement resolves is reported,
    however small, so a caller can hold the ones within tolerance separately from
    the ones beyond it.
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
    overhangs: dict[tuple[int, int], tuple[float, str]] = {}
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
            short = -min(drawn.negative_side_slack, drawn.positive_side_slack)
            if short <= 0.0:
                continue
            overhangs[claim.path_rank, claim.segment_rank] = (
                short,
                f"{reservation.id} claim {claim.member_id} "
                f"(path {claim.path_rank}, segments {claim.segment_rank}.."
                f"{claim.segment_end_rank}): drawn "
                f"[{drawn.drawn_start:.2f}, {drawn.drawn_end:.2f}] outside band "
                f"[{drawn.band_start:.2f}, {drawn.band_end:.2f}] by {short:.2f}px",
            )
    return overhangs


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(_ROOT)) for p in _CORPUS]
)
def test_realised_gap_claims_are_drawn_in_their_reserved_band(path: Path) -> None:
    rel = str(path.relative_to(_ROOT))
    overhangs = _claim_overhangs(path)
    if overhangs is None:
        if rel in KNOWN_NOT_RENDERING:
            pytest.skip("fixture does not render")
        pytest.fail(
            f"{rel} raised while building its render plan. A fixture that stops "
            "rendering cannot be held to a claim-consumption bound; either fix "
            "the regression or add it to KNOWN_NOT_RENDERING with the reason it "
            "cannot render."
        )
    beyond = {
        key: (short, message)
        for key, (short, message) in overhangs.items()
        if short > COORD_TOLERANCE
    }
    found = {key: round(short, 2) for key, (short, _message) in beyond.items()}
    recorded = {
        (path_rank, segment_rank): shortfall
        for name, path_rank, segment_rank, shortfall in UNDERSIZED_CORRIDORS
        if name == rel
    }
    assert found == recorded, (
        "a claim drawn outside its reserved band was positioned by a "
        "geometry-derived fallback rather than by its reservation, unless its "
        "corridor is one UNDERSIZED_CORRIDORS records as reserved for fewer "
        f"lanes than it carries: unrecorded {sorted(set(found) - set(recorded))}, "
        f"recorded but now consuming its band {sorted(set(recorded) - set(found))}"
        "\n" + "\n".join(beyond[key][1] for key in sorted(beyond))
    )


@pytest.mark.parametrize(
    "path", _CORPUS, ids=[str(p.relative_to(_ROOT)) for p in _CORPUS]
)
def test_claims_drawn_within_one_tolerance_of_their_band_are_the_recorded_ones(
    path: Path,
) -> None:
    """The population sitting just inside the tolerance does not grow unnoticed.

    The bound above is "no claim is drawn more than one ``COORD_TOLERANCE``
    outside its band".  On its own that lets claims accumulate at 0.99 of a
    tolerance without any test reddening, and the guarantee would erode while
    still reading as clean.  This pins the ones that are there by identity.
    """
    rel = str(path.relative_to(_ROOT))
    overhangs = _claim_overhangs(path)
    if overhangs is None:
        pytest.skip("fixture does not render")
    found = {
        (rel, path_rank, segment_rank)
        for (path_rank, segment_rank), (short, _message) in overhangs.items()
        if short <= COORD_TOLERANCE
    }
    expected = {item for item in WITHIN_TOLERANCE_OVERHANGS if item[0] == rel}
    assert found == expected, (
        "the claims drawn within one tolerance of their band are not the ones "
        f"recorded: unrecorded {sorted(found - expected)}, recorded but now "
        f"clean {sorted(expected - found)}. A new one is a claim that stopped "
        "consuming its reservation exactly and got away with it; one that "
        "cleaned up means dropping its WITHIN_TOLERANCE_OVERHANGS entry:\n"
        + "\n".join(
            message
            for (_path_rank, _segment_rank), (short, message) in sorted(
                overhangs.items()
            )
            if short <= COORD_TOLERANCE
        )
    )


def test_every_recorded_overhang_names_a_corpus_fixture() -> None:
    """A stale entry would silently excuse a fixture that no longer exists."""
    corpus = {str(item.relative_to(_ROOT)) for item in _CORPUS}
    named = {item[0] for item in WITHIN_TOLERANCE_OVERHANGS}
    named |= {item[0] for item in UNDERSIZED_CORRIDORS}
    assert named <= corpus, named - corpus
