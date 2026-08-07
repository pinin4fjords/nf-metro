"""Final convergence settlement cannot publish infeasible planned geometry."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import nf_metro.layout.routing.convergences as convergences
from nf_metro.api import prepare_graph
from nf_metro.layout.route_plan import RouteSystemId
from nf_metro.layout.routing.convergences import (
    ConvergencePlanExecution,
    FinalConvergenceFeasibilityError,
    empty_convergence_plan_execution,
    settle_global_convergence_execution,
)
from nf_metro.layout.routing.core import observe_route_edges
from nf_metro.layout.routing.member_geometry import empty_member_geometry_execution
from nf_metro.layout.routing.offsets import compute_station_offsets
from nf_metro.parser.route_topology import ResolvedEdge

ROOT = Path(__file__).parents[1]


def test_fan_in_coupled_convergence_flanks_move_as_one_allocation() -> None:
    path = ROOT / "examples" / "topologies" / "fan_in_merge.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    member = next(
        plan
        for plan in observation.plan.member_geometry_plans
        if plan.edge == ResolvedEdge("__junction_6", "sink__entry_left_5", "aux")
    )
    member_channel = next(
        channel
        for channel in member.gap_channels
        if (channel.gap_lo_col, channel.row) == (0, 0)
    )
    lookup = convergences.gap_lookup_geometry(graph)
    convergence_columns = {
        channel.coordinate
        for plan in observation.plan.convergence_plans
        for channel in convergences._plan_gap_channels(plan, graph, lookup)
        if channel.gap == (0, 0) and channel.line_ids == frozenset({"main"})
    }

    assert member_channel.start[0] == member_channel.end[0] == 210.0
    assert convergence_columns == {222.0}
    assert (
        min(abs(column - member_channel.start[0]) for column in convergence_columns)
        == 12.0
    )
    assert all(plan.owns_geometry for plan in observation.plan.convergence_plans)


def test_genomic_pipeline_mirrored_bundles_keep_exact_line_channels() -> None:
    path = ROOT / "examples" / "genomic_pipeline.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    lookup = convergences.gap_lookup_geometry(graph)
    convergence_channels = tuple(
        (plan, channel)
        for plan in observation.plan.convergence_plans
        for channel in convergences._plan_gap_channels(plan, graph, lookup)
        if channel.gap == (0, 1)
    )
    member_channels = tuple(
        (plan, channel)
        for plan in observation.plan.member_geometry_plans
        for channel in plan.gap_channels
        if (channel.gap_lo_col, channel.row) == (0, 1)
    )

    assert convergence_channels
    assert all(plan.owns_geometry for plan, _channel in convergence_channels)
    for line_id in ("germline", "tumor_only", "somatic"):
        convergence_columns = {
            channel.coordinate
            for _plan, channel in convergence_channels
            if channel.line_ids == frozenset({line_id})
        }
        member_columns = {
            channel.start[0]
            for plan, channel in member_channels
            if plan.edge.line_id == line_id
        }
        assert convergence_columns == member_columns


def test_organellar_joint_allocation_freezes_member_before_emission() -> None:
    path = ROOT / "tests" / "fixtures" / "genomeassembly_organellar.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    observation = observe_route_edges(
        graph, station_offsets=compute_station_offsets(graph)
    )
    member = next(
        plan
        for plan in observation.plan.member_geometry_plans
        if plan.edge
        == ResolvedEdge("__junction_9", "scaffolding__entry_left_5", "hic_reads")
    )
    (member_channel,) = tuple(
        channel for channel in member.gap_channels if channel.gap_lo_col == 0
    )
    route = next(
        route
        for route in observation.routes
        if ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        == member.edge
    )
    lookup = convergences.gap_lookup_geometry(graph)
    convergence_columns = {
        channel.coordinate
        for plan in observation.plan.convergence_plans
        for channel in convergences._plan_gap_channels(plan, graph, lookup)
        if channel.gap == (0, 0)
        and channel.line_ids == frozenset({"assemblies"})
        and convergences.spans_share_corridor(
            channel.y_lo,
            channel.y_hi,
            min(member_channel.start[1], member_channel.end[1]),
            max(member_channel.start[1], member_channel.end[1]),
        )
    }

    assert member_channel.start[0] == member_channel.end[0] == 246.0
    assert convergence_columns == {258.0}
    assert tuple(route.points) == member.points
    assert tuple(
        route.points[member_channel.segment_rank : member_channel.segment_rank + 2]
    ) == (member_channel.start, member_channel.end)


def test_starved_final_settlement_does_not_publish_crowded_plan(monkeypatch) -> None:
    system_id = RouteSystemId("system")
    plan = SimpleNamespace(
        id="convergence-plan",
        system_id=system_id,
        owns_geometry=True,
        line_ids=("convergence",),
        trunk_axis=None,
        endpoint_ownership=(),
    )
    query = empty_convergence_plan_execution().query
    execution = ConvergencePlanExecution((plan,), (), (), (), query)
    convergence_channel = convergences._PlanGapChannel(
        1,
        100.0,
        0.0,
        100.0,
        True,
        (0, 0),
        frozenset({"convergence"}),
        frozenset({"convergence-member"}),
    )
    member_channel = convergences._PlanGapChannel(
        None,
        102.0,
        0.0,
        100.0,
        True,
        (0, 0),
        frozenset({"member"}),
        frozenset({"member"}),
    )

    monkeypatch.setattr(
        convergences,
        "_planned_member_gap_channels",
        lambda plans, member_geometry: (member_channel,),
    )
    monkeypatch.setattr(
        convergences,
        "_plan_gap_channels",
        lambda candidate, graph, lookup: (convergence_channel,),
    )
    monkeypatch.setattr(convergences, "gap_lookup_geometry", lambda graph: None)
    monkeypatch.setattr(convergences, "_system_conflict", lambda plans, ctx: None)
    for name in (
        "_settle_shared_trunk_channels",
        "_settle_shared_opening_pivots",
        "_settle_landing_trunk_flanks",
        "_settle_same_line_gap_flanks",
    ):
        monkeypatch.setattr(convergences, name, lambda plans, *args, **kwargs: plans)
    monkeypatch.setattr(
        convergences,
        "_settle_opposing_landing_channels",
        lambda plans, *args, **kwargs: plans,
    )
    monkeypatch.setattr(
        convergences,
        "_settle_opposing_gap_flanks",
        lambda plans, *args, **kwargs: plans,
    )

    with pytest.raises(
        FinalConvergenceFeasibilityError,
        match="crowds a planned member channel",
    ):
        settle_global_convergence_execution(
            execution,
            SimpleNamespace(),
            SimpleNamespace(curve_radius=8.0),
            exit_turn_plans=(),
            member_geometry=empty_member_geometry_execution(),
            planned_system_ids=frozenset({system_id}),
            include_resources=False,
        )
