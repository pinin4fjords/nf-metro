"""A compatibility system's limit is probed against capacity, not read off geometry."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.capacity_probe import (
    _COLUMN_AXIS,
    _ROW_AXIS,
    CAPACITY_MULTIPLES,
    CapacityScope,
    CapacityVerdict,
    _widen,
    claimed_boundaries,
    probe_settlement_capacity,
)
from nf_metro.layout.routing import compute_station_offsets, observe_route_edges
from nf_metro.render.svg import _settled_render_graph, build_observed_render_plan

ROOT = Path(__file__).parents[1]
TOPOLOGIES = ROOT / "examples" / "topologies"
REGRESSIONS = ROOT / "tests" / "fixtures" / "regressions"

# Every convergence system the corpus leaves on the compatibility path, with the
# verdict granting it boundary capacity produces.  #1657 lets a system stay
# compatible only where its limit is not an envelope allocation, so a fixture
# listed here as anything other than BEYOND_ALLOCATION is one whose exit that
# criterion does not license.
COMPATIBILITY_CORPUS: tuple[tuple[Path, CapacityVerdict], ...] = (
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
    (
        TOPOLOGIES / "merge_around_below_leftmost.mmd",
        CapacityVerdict.ALLOCATION_REACHES,
    ),
    (TOPOLOGIES / "merge_bottom_row_bypass.mmd", CapacityVerdict.ALLOCATION_REACHES),
    (
        TOPOLOGIES / "merge_feeder_shared_channel_gap.mmd",
        CapacityVerdict.ALLOCATION_REACHES,
    ),
    (TOPOLOGIES / "merge_right_entry.mmd", CapacityVerdict.ALLOCATION_REACHES),
    (
        TOPOLOGIES / "merge_trunk_out_of_range_section.mmd",
        CapacityVerdict.ALLOCATION_UNSTABLE,
    ),
    (
        ROOT / "tests" / "fixtures" / "ambiguous_exit_continuation.mmd",
        CapacityVerdict.ALLOCATION_REACHES,
    ),
    (
        ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (
        REGRESSIONS / "cross_column_perp_entry_overflow.mmd",
        CapacityVerdict.BEYOND_ALLOCATION,
    ),
    (REGRESSIONS / "stacked_collector_fanin.mmd", CapacityVerdict.BEYOND_ALLOCATION),
)

# A system the planner owns on its own geometry, whose reserved boundaries are
# narrow enough that taking one offset step out of them starves it onto the
# compatibility path.  It is what makes a positive probe result reachable by
# construction rather than only observed.
STARVABLE = TOPOLOGIES / "fan_in_merge.mmd"
STARVATION = -8.0


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
    reached, not a record edited to look like one.
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
    _widen(graph, _ROW_AXIS, rows, amount)
    _widen(graph, _COLUMN_AXIS, columns, amount)
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
    path: Path, expected: CapacityVerdict
) -> None:
    """#1657 lets a system stay compatible only where its limit is not an
    envelope allocation, and the only way to establish that is to give the
    planner the room and see what it decides.

    The published sentence is what a reader acts on, so the grants behind it are
    checked too: which capacities were planned at, and that the message quotes
    the one the verdict rests on.
    """
    graph, plan = _settled(path)
    probe = _sole(probe_settlement_capacity(graph, plan))
    assert probe.verdict is expected
    assert len(probe.grants) == 2 * len(CAPACITY_MULTIPLES)
    assert probe.capacity > 0.0
    assert probe.control_conflict is not None
    assert probe.control_conflict.reason in probe.message

    planned = [item for item in probe.grants if item.planned]
    if expected is CapacityVerdict.BEYOND_ALLOCATION:
        assert not planned
        assert probe.sufficient_capacity is None
        assert f"{max(item.capacity for item in probe.grants):.2f}px" in probe.message
        return
    assert planned
    assert probe.sufficient_capacity is not None
    assert probe.sufficient_scope is not None
    assert f"{probe.sufficient_capacity:.2f}px" in probe.message
    tail = [
        item
        for item in probe.grants
        if item.scope is probe.sufficient_scope
        and item.capacity >= probe.sufficient_capacity
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
    assert probe.sufficient_scope is CapacityScope.CLAIMED_BOUNDARIES
    assert probe.sufficient_capacity is not None
    assert probe.sufficient_capacity >= -STARVATION


def test_the_probe_never_writes_to_the_map_it_measures() -> None:
    """Nothing a counterfactual moves may reach the geometry that gets drawn."""
    graph, plan = _settled(TOPOLOGIES / "merge_bottom_row_bypass.mmd")
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


def test_the_probe_answers_the_same_way_twice() -> None:
    """Evidence that changes between two readings of one map is not evidence."""
    graph, plan = _settled(TOPOLOGIES / "merge_right_entry.mmd")
    first = probe_settlement_capacity(graph, plan)
    second = probe_settlement_capacity(graph, plan)
    assert first == second


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
