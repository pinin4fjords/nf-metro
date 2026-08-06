"""A compatibility system's limit is probed against capacity, not read off geometry."""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout import capacity_probe
from nf_metro.layout.capacity_probe import (
    CAPACITY_MULTIPLES,
    CapacityGrant,
    CapacityProbe,
    CapacityScope,
    CapacityVerdict,
    GrantOutcome,
    _replan,
    claimed_boundaries,
    probe_settlement_capacity,
    translate_boundaries,
)
from nf_metro.layout.route_plan import RouteSystemId
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.render.svg import _settled_render_graph, build_observed_render_plan

ROOT = Path(__file__).parents[1]
TOPOLOGIES = ROOT / "examples" / "topologies"
REGRESSIONS = ROOT / "tests" / "fixtures" / "regressions"

PLANNED_OUTRIGHT = None
"""The expected result for a fixture whose route systems the planner all owns.

Such a fixture is listed and probed as a control: the probe reports on
compatibility systems, so an empty result is the assertion that the fixture has
none, and the row fails loudly the moment one appears.
"""

# Every fixture whose convergence planning has been probed, with the result:
# a `CapacityVerdict` for a system on the compatibility path, or
# `PLANNED_OUTRIGHT` where the fixture leaves none.  #1657 lets a system stay
# compatible only where its limit is not an envelope allocation, so a system
# listed here as anything other than BEYOND_ALLOCATION is one whose exit that
# criterion does not license.
COMPATIBILITY_CORPUS: tuple[tuple[Path, CapacityVerdict | None], ...] = (
    (ROOT / "examples" / "genomeassembly.mmd", CapacityVerdict.BEYOND_ALLOCATION),
    (
        ROOT / "examples" / "genomeassembly_staggered.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (ROOT / "examples" / "genomic_pipeline.mmd", CapacityVerdict.BEYOND_ALLOCATION),
    (
        TOPOLOGIES / "exit_run_three_drop_columns.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (TOPOLOGIES / "funcprofiler_upstream.mmd", CapacityVerdict.BEYOND_ALLOCATION),
    (TOPOLOGIES / "merge_around_below_leftmost.mmd", PLANNED_OUTRIGHT),
    (TOPOLOGIES / "merge_bottom_row_bypass.mmd", CapacityVerdict.BEYOND_ALLOCATION),
    (
        TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (TOPOLOGIES / "merge_right_entry.mmd", CapacityVerdict.BEYOND_ALLOCATION),
    (
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (
        ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (
        ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (REGRESSIONS / "cross_column_perp_entry_overflow.mmd", PLANNED_OUTRIGHT),
    (REGRESSIONS / "stacked_collector_fanin.mmd", CapacityVerdict.BEYOND_ALLOCATION),
)

# A system the planner owns on its own geometry, whose reserved boundaries are
# narrow enough that ``STARVATION`` drops it onto the compatibility path.  It is
# what makes a positive probe result reachable by construction rather than only
# observed.
STARVABLE = TOPOLOGIES / "fan_in_merge.mmd"
STARVATION = -10.0


def _settled(path: Path):
    """The geometry a render of *path* draws, and the plan drawn on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        graph.permissive = True
        theme = resolve_theme(None, graph)
        route_plan = build_observed_render_plan(graph, theme).route_plan
        return _settled_render_graph(graph, theme), route_plan


def _sole(items: tuple):
    assert len(items) == 1, f"expected one probed system, found {len(items)}"
    return items[0]


def _starved(path: Path, amount: float):
    """*path*'s settled map with its planned system's boundaries taken in.

    The plan is re-observed on the narrowed geometry rather than carried over,
    so what the probe is handed is a real compatibility disposition the planner
    reached, not a record edited to look like one.  The narrowed map carries the
    junction placement its own sections imply, so the disposition is one the
    pipeline could draw its way into rather than one a stranded junction
    manufactures.
    """
    graph, plan = _settled(path)
    planned = sorted(
        {
            item.system_id
            for item in plan.convergence_plans
            if item.legacy_reason is None
        }
    )
    system_id = _sole(tuple(planned))
    rows, columns, _widths = claimed_boundaries(plan, system_id)
    graph = copy.deepcopy(graph)
    translate_boundaries(graph, rows, columns, amount)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starved_plan = observe_route_edges(
            graph, station_offsets=compute_station_offsets(graph)
        ).plan
    return graph, starved_plan, system_id


@pytest.mark.parametrize(
    ("path", "expected"),
    COMPATIBILITY_CORPUS,
    ids=lambda item: getattr(item, "name", item),
)
def test_every_compatibility_system_is_probed_against_boundary_capacity(
    path: Path, expected: CapacityVerdict | None
) -> None:
    """#1657 lets a system stay compatible only where its limit is not an
    envelope allocation, and the only way to establish that is to give the
    planner the room and see what it decides.

    The published sentence is what a reader acts on, so the grants behind it are
    checked too: which capacities were planned at, and that the message quotes
    the one the verdict rests on.
    """
    graph, plan = _settled(path)
    if expected is PLANNED_OUTRIGHT:
        assert probe_settlement_capacity(graph, plan) == ()
        assert plan is not None
        assert plan.convergence_plans
        assert all(item.legacy_reason is None for item in plan.convergence_plans)
        return
    probe = _sole(probe_settlement_capacity(graph, plan))
    assert probe.verdict is expected
    assert len(probe.grants) == 2 * len(CAPACITY_MULTIPLES)
    assert not probe.diverged_grants, (
        "a grant whose re-plan describes neither disposition says nothing about "
        "allocation, so a verdict resting on one would be reading an "
        "uninterpretable counterfactual: "
        + str([item.capacity for item in probe.diverged_grants])
    )
    assert probe.capacity > 0.0
    assert probe.control_conflict is not None
    assert probe.control_conflict.reason in probe.message

    planned = [item for item in probe.grants if item.planned]
    if expected is CapacityVerdict.BEYOND_ALLOCATION:
        assert not planned
        assert probe.quoted is None
        assert f"{max(item.capacity for item in probe.grants):.2f}px" in probe.message
        return
    assert planned
    assert probe.quoted is not None
    quoted_scope, quoted_capacity = probe.quoted
    assert f"{quoted_capacity:.2f}px" in probe.message
    tail = [
        item
        for item in probe.grants
        if item.scope is quoted_scope and item.capacity >= quoted_capacity
    ]
    reaches = expected is CapacityVerdict.ALLOCATION_REACHES
    assert all(item.planned for item in tail) is reaches


def test_a_starved_system_is_handed_back_the_capacity_that_starved_it() -> None:
    """A probe that could only ever report an unreachable limit would be
    indistinguishable from one that does nothing, so a system whose limitation
    is capacity by construction has to come back as one it reaches."""
    graph, plan, system_id = _starved(STARVABLE, STARVATION)
    on_compatibility = [
        item
        for item in plan.convergence_plans
        if item.system_id == system_id and item.legacy_reason is not None
    ]
    assert on_compatibility, "starvation did not put the planner on compatibility"

    probe = _sole(probe_settlement_capacity(graph, plan))
    assert probe.system_id == system_id
    assert probe.verdict is CapacityVerdict.ALLOCATION_REACHES
    assert probe.quoted is not None
    quoted_scope, quoted_capacity = probe.quoted
    assert quoted_scope is CapacityScope.CLAIMED_BOUNDARIES
    assert quoted_capacity >= -STARVATION


def test_the_probe_never_writes_to_the_map_it_measures() -> None:
    """Nothing a counterfactual moves may reach the geometry that gets drawn.

    Read on a system the probe plans, so the grants behind the check are ones
    that moved the copy far enough to change what the planner decided.
    """
    graph, plan, _system_id = _starved(STARVABLE, STARVATION)
    before = copy.deepcopy(graph)
    probe = _sole(probe_settlement_capacity(graph, plan))
    assert probe.verdict is CapacityVerdict.ALLOCATION_REACHES
    assert {
        key: (section.bbox_x, section.bbox_y, section.bbox_w, section.bbox_h)
        for key, section in graph.sections.items()
    } == {
        key: (section.bbox_x, section.bbox_y, section.bbox_w, section.bbox_h)
        for key, section in before.sections.items()
    }
    assert {key: (item.x, item.y) for key, item in graph.stations.items()} == {
        key: (item.x, item.y) for key, item in before.stations.items()
    }


def test_capacity_carries_a_shared_opening_turn_instead_of_opening_it() -> None:
    """Two fan arms leaving one junction turn on a coordinate that junction sets,
    and a boundary translation carries the junction with the sections it joins,
    so the distance between the two turns is the same at every capacity.

    Reading the separation rather than only the verdict is what tells a limit no
    allocation reaches apart from one a grant merely stopped measuring: a
    translation that stranded the junction would report the pair drifting apart
    and the planner falling silent about them.
    """
    graph, plan = _settled(TOPOLOGIES / "merge_bottom_row_bypass.mmd")
    probe = _sole(probe_settlement_capacity(graph, plan))
    rows, columns, _widths = claimed_boundaries(plan, probe.system_id)
    for multiple in CAPACITY_MULTIPLES:
        amount = probe.capacity * multiple
        granted = copy.deepcopy(graph)
        translate_boundaries(granted, rows, columns, amount)
        replanned, _offset_step = _replan(granted)
        separations = {
            round(item.conflict.separation, 6)
            for item in replanned[probe.system_id]
            if item.conflict is not None
        }
        assert separations == {0.0}, f"at {amount}px: {separations}"


def test_the_probe_answers_the_same_way_twice() -> None:
    """Evidence that changes between two readings of one map is not evidence."""
    graph, plan = _settled(TOPOLOGIES / "merge_right_entry.mmd")
    first = probe_settlement_capacity(graph, plan)
    second = probe_settlement_capacity(graph, plan)
    assert first == second


def test_a_grant_the_replan_loses_the_system_on_is_not_a_negative_result() -> None:
    """A re-plan that neither owns the system nor keeps it whole is uninterpretable.

    ``_is_planned`` returns ``None`` for a system a re-plan lost or split, which
    is the same disposition-mismatch hazard the control check refuses one step
    further in: at a grant instead of at the baseline.  Folded into "not planned"
    it would count as evidence *for* ``BEYOND_ALLOCATION``, so a probe that had
    stopped describing its system at every capacity would publish the conclusion
    the exit criteria are read off.
    """
    graph, plan = _settled(TOPOLOGIES / "merge_right_entry.mmd")

    def lose_the_system(_graph):
        return {}, 4.0

    with mock.patch.object(capacity_probe, "_replan", side_effect=lose_the_system):
        outcome = capacity_probe._grant_outcome(
            graph, RouteSystemId("any"), (), (), 0.0
        )
    assert outcome is GrantOutcome.DIVERGED

    probe = _sole(probe_settlement_capacity(graph, plan))
    diverged = replace(
        probe,
        grants=tuple(
            replace(item, outcome=GrantOutcome.DIVERGED) for item in probe.grants
        ),
    )
    assert probe.verdict is CapacityVerdict.BEYOND_ALLOCATION
    assert diverged.verdict is CapacityVerdict.GRANTS_DIVERGED
    assert diverged.measured_grants == ()
    assert "could not be probed" in diverged.message


def test_a_diverged_grant_does_not_break_a_planned_tail() -> None:
    """The tail is read over the grants that describe the system.

    A grant excluded from the verdict cannot count against it either, or the
    exclusion would be a third way of saying "not planned".
    """
    grants = (
        CapacityGrant(CapacityScope.CLAIMED_BOUNDARIES, 10.0, GrantOutcome.COMPATIBLE),
        CapacityGrant(CapacityScope.CLAIMED_BOUNDARIES, 20.0, GrantOutcome.PLANNED),
        CapacityGrant(CapacityScope.CLAIMED_BOUNDARIES, 40.0, GrantOutcome.DIVERGED),
        CapacityGrant(CapacityScope.CLAIMED_BOUNDARIES, 80.0, GrantOutcome.PLANNED),
    )
    probe = CapacityProbe(RouteSystemId("s"), 1.0, 10.0, grants, True, None)
    assert probe.verdict is CapacityVerdict.ALLOCATION_REACHES
    assert probe.sufficient == (CapacityScope.CLAIMED_BOUNDARIES, 20.0)


def test_a_control_that_does_not_reproduce_the_map_is_reported_unmeasured() -> None:
    """A grant means something only as a difference from a reproduced baseline,
    so a plan the graph in hand does not agree with is refused rather than
    measured against an unknown.

    The pairing here is the starved map's plan against the geometry it was
    starved from, which is a compatibility record the planner reaches nowhere on
    the graph being probed.
    """
    graph, _plan = _settled(STARVABLE)
    _starved_graph, starved_plan, system_id = _starved(STARVABLE, STARVATION)
    probe = _sole(probe_settlement_capacity(graph, starved_plan))
    assert probe.system_id == system_id
    assert probe.verdict is CapacityVerdict.CONTROL_DIVERGED
    assert probe.grants == ()
    assert "did not reproduce" in probe.message
