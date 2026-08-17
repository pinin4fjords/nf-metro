"""Convergence consumes the cohort compiler's complete grant exactly once."""

from dataclasses import replace
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import CURVE_RADIUS, DIAGONAL_RUN, OFFSET_STEP
from nf_metro.layout.routing import convergences
from nf_metro.layout.routing.context import _build_routing_context
from nf_metro.layout.routing.convergences import (
    ConvergenceInvariantError,
    ConvergencePlanExecution,
    apply_convergence_corridor_grants,
    convergence_corridor_requests,
)
from nf_metro.layout.routing.core import observe_route_edges
from nf_metro.layout.routing.corridor_cohort_integration import (
    CorridorScalarGrant,
    CorridorScalarOwnerKind,
)
from nf_metro.layout.routing.offsets import compute_station_offsets

ROOT = Path(__file__).parents[1]


def _corridor_fixture():
    path = ROOT / "examples" / "topologies" / "fan_in_merge.mmd"
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    station_offsets = compute_station_offsets(graph)
    observation = observe_route_edges(graph, station_offsets=station_offsets)
    ctx = _build_routing_context(
        graph,
        DIAGONAL_RUN,
        CURVE_RADIUS,
        station_offsets,
    )
    plans = observation.plan.convergence_plans
    edge_order = tuple(member.edge for member in observation.plan.members)
    execution = ConvergencePlanExecution(
        plans,
        (),
        (),
        (),
        convergences._query(plans, edge_order),
    )
    targets, requests = convergence_corridor_requests(plans, graph, ctx)
    assert len(requests) >= 2
    return execution, targets, requests


def _shifted_grants(requests):
    grants = []
    for rank, request in enumerate(requests):
        coordinate_delta = OFFSET_STEP * (rank + 1)
        grants.append(
            CorridorScalarGrant(
                variable_id=request.variable.variable_id,
                owner_kind=CorridorScalarOwnerKind.CONVERGENCE_TRUNK,
                owner_id=request.variable.owner_id,
                coordinate=request.variable.coordinate + coordinate_delta,
                coordinate_delta=coordinate_delta,
                control_recipe=request.control_recipe,
            )
        )
    return tuple(grants)


def test_convergence_consumes_one_complete_grant_set_atomically() -> None:
    execution, _targets, requests = _corridor_fixture()
    grants = _shifted_grants(requests)
    original_coordinates = {
        str(plan.id): plan.trunk_axis.coordinate
        for plan in execution.plans
        if plan.trunk_axis is not None
    }

    granted = apply_convergence_corridor_grants(
        execution,
        tuple(reversed(requests)),
        tuple(reversed(grants)),
    )

    granted_coordinates = {
        str(plan.id): plan.trunk_axis.coordinate
        for plan in granted.plans
        if plan.trunk_axis is not None
    }
    assert granted_coordinates == {grant.owner_id: grant.coordinate for grant in grants}
    assert granted.query.plans == granted.plans
    assert {
        str(plan.id): plan.trunk_axis.coordinate
        for plan in execution.plans
        if plan.trunk_axis is not None
    } == original_coordinates


def test_convergence_refuses_partial_or_extra_grant_sets() -> None:
    execution, _targets, requests = _corridor_fixture()
    grants = _shifted_grants(requests)

    with pytest.raises(
        ConvergenceInvariantError,
        match="does not complete its request set",
    ):
        apply_convergence_corridor_grants(execution, requests, grants[:-1])

    extra = replace(
        grants[0],
        variable_id=f"{grants[0].variable_id}|extra",
    )
    with pytest.raises(
        ConvergenceInvariantError,
        match="does not complete its request set",
    ):
        apply_convergence_corridor_grants(
            execution,
            requests,
            (*grants, extra),
        )


def test_convergence_grant_is_not_a_repeatable_repair_pass() -> None:
    execution, _targets, requests = _corridor_fixture()
    grants = _shifted_grants(requests)
    granted = apply_convergence_corridor_grants(execution, requests, grants)

    with pytest.raises(
        ConvergenceInvariantError,
        match="does not match its source plan",
    ):
        apply_convergence_corridor_grants(granted, requests, grants)


def test_convergence_refuses_a_grant_from_another_geometry_owner() -> None:
    execution, _targets, requests = _corridor_fixture()
    grants = _shifted_grants(requests)
    mismatched = replace(
        grants[0],
        owner_kind=CorridorScalarOwnerKind.MEMBER_CARRIER,
    )

    with pytest.raises(
        ConvergenceInvariantError,
        match="wrong semantic owner",
    ):
        apply_convergence_corridor_grants(
            execution,
            requests,
            (mismatched, *grants[1:]),
        )


def test_adapter_exposes_every_planned_trunk_without_replacing_members() -> None:
    execution, targets, requests = _corridor_fixture()
    eligible = {
        str(plan.id): plan
        for plan in execution.plans
        if plan.owns_geometry and plan.trunk_axis is not None
    }

    assert {request.variable.owner_id for request in requests} == set(eligible)
    assert len(targets) == len(requests) == len(eligible)
    assert len({target.member_id for target in targets}) == len(targets)
    for target, request in zip(targets, requests, strict=True):
        plan = eligible[request.variable.owner_id]
        assert target.mutable
        assert target.member_id == request.variable.member_id
        assert target.member_id != plan.primary_trunk_member_id
        assert request.variable.coordinate == plan.trunk_axis.coordinate
        assert request.domain.member_id == request.variable.variable_id
