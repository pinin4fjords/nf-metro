"""A route system decides the lanes of a channel two of its trunks would share.

Each convergence plan reads its trunk coordinate from a trial route produced with
no knowledge of its siblings, so two plans of one system taking the same channel
derive the same coordinate.  The decision belongs to the system: it lanes them by
``cotravelling_lane_clearance``, which separates a line from its own return leg by
a turn radius and fuses two trunks running the same way onto one stroke.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.layout.constants import COORD_TOLERANCE, CURVE_RADIUS
from nf_metro.layout.geometry import cotravelling_lane_clearance
from nf_metro.layout.route_plan import (
    ConvergenceContinuation,
    ConvergenceDisposition,
    ConvergenceEndpointOwnership,
    ConvergenceEndpointRole,
    ConvergenceLanding,
    ConvergencePlan,
    ConvergencePlanId,
    ConvergenceTrunkAxis,
    ConvergenceTrunkReason,
    DemandAxis,
    DemandId,
    EmissionMemberId,
    RouteSystemId,
    SharedReferenceId,
)
from nf_metro.layout.routing.common import Direction
from nf_metro.layout.routing.convergences import _settle_shared_trunk_channels
from nf_metro.parser.route_topology import (
    ConnectorId,
    ConvergenceId,
    EndpointGroupId,
    ResolvedEdge,
)
from nf_metro.render.svg import build_observed_render_plan

ROOT = Path(__file__).parents[1]

# Fixtures whose route systems carry more than one convergence trunk: one pair
# that counter-runs and has to be laned, and two that co-travel and have to stay
# fused.
SHARED_CHANNEL_FIXTURES = (
    ROOT / "examples" / "topologies" / "merge_around_below_leftmost.mmd",
    ROOT / "examples" / "topologies" / "fan_in_merge.mmd",
    ROOT / "examples" / "guide" / "03b_fan_in_merge.mmd",
)

LANE_CLEARANCE = cotravelling_lane_clearance(
    same_line=True, counter_running=True, curve_radius=CURVE_RADIUS
)


def _systems_with_shared_trunks(
    path: Path,
) -> dict[str, list[ConvergencePlan]]:
    """*path*'s route systems that converge more than once, by system."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        plan = build_observed_render_plan(graph, resolve_theme(None, graph)).route_plan
    assert plan is not None
    by_system: dict[str, list[ConvergencePlan]] = {}
    for item in plan.convergence_plans:
        by_system.setdefault(str(item.system_id), []).append(item)
    return {key: items for key, items in by_system.items() if len(items) > 1}


def _overlap(first: ConvergenceTrunkAxis, second: ConvergenceTrunkAxis) -> float:
    return min(first.extent_end, second.extent_end) - max(
        first.extent_start, second.extent_start
    )


@pytest.mark.parametrize(
    "path", SHARED_CHANNEL_FIXTURES, ids=lambda item: Path(item).stem
)
def test_one_system_lanes_the_trunk_channel_its_plans_share(path: Path) -> None:
    systems = _systems_with_shared_trunks(path)
    assert systems, f"{path.name} publishes no system that converges twice"
    for plans in systems.values():
        assert {item.disposition for item in plans} == {
            ConvergenceDisposition.PLANNED
        }, "a shared trunk channel is a decision the system makes, not one it declines"
        axes = [item.trunk_axis for item in plans if item.trunk_axis is not None]
        for first, second in combinations(axes, 2):
            if first.axis is not second.axis or _overlap(first, second) <= (
                COORD_TOLERANCE
            ):
                continue
            separation = abs(first.coordinate - second.coordinate)
            if first.direction is second.direction:
                assert separation <= COORD_TOLERANCE, (
                    "trunks running one way along one channel draw one stroke"
                )
            else:
                assert separation >= LANE_CLEARANCE, (
                    f"a line and its return leg need {LANE_CLEARANCE}px between "
                    f"their lanes, not {separation}px"
                )


def _planned_plan(
    tag: str, axis: DemandAxis, direction: Direction, flank: float
) -> ConvergencePlan:
    """A minimal planned convergence whose only geometry is one trunk."""
    feeder = ResolvedEdge(f"{tag}_source", f"{tag}_merge", "a")
    outgoing = ResolvedEdge(f"{tag}_merge", f"{tag}_entry", "a")
    feeder_member = EmissionMemberId(f"{tag}_feeder")
    outgoing_member = EmissionMemberId(f"{tag}_outgoing")
    connector = ConnectorId(f"{tag}_connector")
    travelling_x = axis is DemandAxis.X
    join = (0.0, 50.0) if travelling_x else (50.0, 0.0)
    return ConvergencePlan(
        id=ConvergencePlanId(f"{tag}_plan"),
        system_id=RouteSystemId("system"),
        convergence_ids=(ConvergenceId(f"{tag}_convergence"),),
        entry_group_ids=(EndpointGroupId(f"{tag}_entry_group"),),
        merge_junction_ids=(f"{tag}_merge",),
        target_entry_port_ids=(f"{tag}_entry",),
        connector_ids=(connector,),
        member_ids=(feeder_member, outgoing_member),
        resolved_member_paths=((feeder, outgoing),),
        resolved_member_edges=(feeder, outgoing),
        line_ids=("a",),
        upstream_exit_turn_plan_ids=(),
        upstream_fan_plan_ids=(),
        primary_trunk_member_id=feeder_member,
        primary_trunk_reason=ConvergenceTrunkReason.LONGEST_BYPASS,
        trunk_axis=ConvergenceTrunkAxis(
            axis=axis,
            coordinate=50.0,
            extent_start=0.0,
            extent_end=100.0,
            direction=direction,
            source_flank_coordinate=flank,
            target_flank_coordinate=flank,
            claimant_member_ids=(feeder_member,),
        ),
        landings=(
            ConvergenceLanding(
                member_id=feeder_member,
                edge=feeder,
                source_junction_id=f"{tag}_source",
                approach_axis=DemandAxis.Y if travelling_x else DemandAxis.X,
                approach_direction=Direction.D if travelling_x else Direction.R,
                source_column=0,
                source_row=0,
                lane_rank=0,
                order=0,
                join_point=join,
                corner_handedness=None,
                minimum_runway=CURVE_RADIUS,
                opening_turn_coordinate=None,
                opening_turn_segment=None,
                bypass=True,
                long_haul=False,
                multiple_row=False,
            ),
        ),
        outgoing_continuations=(
            ConvergenceContinuation(
                member_id=outgoing_member,
                edge=outgoing,
                entry_port_id=f"{tag}_entry",
                lane_rank=0,
                start_point=(100.0, 50.0) if travelling_x else (50.0, 100.0),
                end_point=(110.0, 50.0) if travelling_x else (50.0, 110.0),
                covered_by_member_id=None,
            ),
        ),
        lane_order=("a",),
        endpoint_ownership=(
            ConvergenceEndpointOwnership(
                member_id=feeder_member,
                edge=feeder,
                connector_ids=(connector,),
                role=ConvergenceEndpointRole.TRUNK,
                endpoint=join,
            ),
            ConvergenceEndpointOwnership(
                member_id=outgoing_member,
                edge=outgoing,
                connector_ids=(connector,),
                role=ConvergenceEndpointRole.COVERED_CONTINUATION,
                endpoint=(110.0, 50.0) if travelling_x else (50.0, 110.0),
                covered_by_member_id=feeder_member,
            ),
        ),
        shared_reference_ids=(
            SharedReferenceId(f"{tag}_reference_0"),
            SharedReferenceId(f"{tag}_reference_1"),
        ),
        demand_ids=(DemandId(f"{tag}_demand"),),
        foreign_reference_ids=(),
        disposition=ConvergenceDisposition.PLANNED,
        legacy_reason=None,
    )


@pytest.mark.parametrize(
    ("axis", "forward", "backward"),
    (
        (DemandAxis.X, Direction.R, Direction.L),
        (DemandAxis.Y, Direction.D, Direction.U),
    ),
    ids=("travelling-x", "travelling-y"),
)
def test_the_lane_decision_reads_the_same_on_either_travel_axis(
    axis: DemandAxis, forward: Direction, backward: Direction
) -> None:
    """The channel is named by the axis its trunks travel, so a system rotated a
    quarter turn lanes by the same rule rather than by a copy of it.

    Both trunks arrive on one coordinate, as two trial routes through one channel
    do, and the second yields toward the side its own flanks lead.
    """
    plans = (
        _planned_plan("first", axis, forward, 10.0),
        _planned_plan("second", axis, backward, 90.0),
    )

    settled = _settle_shared_trunk_channels(plans, CURVE_RADIUS)

    axes = [item.trunk_axis for item in settled]
    assert axes[0] is not None and axes[1] is not None
    assert axes[0].coordinate == 50.0
    assert axes[1].coordinate == pytest.approx(50.0 + LANE_CLEARANCE)
    landing = settled[1].landings[0]
    across = 1 if axis is DemandAxis.X else 0
    assert landing.join_point[across] == pytest.approx(50.0 + LANE_CLEARANCE)
