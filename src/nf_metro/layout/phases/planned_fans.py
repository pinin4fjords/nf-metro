"""Materialise immutable fan plans in settled section-local frames."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from nf_metro.layout.constants import ICON_HALF_HEIGHT
from nf_metro.layout.pass_metrics import station_radius_approx
from nf_metro.layout.phases._common import (
    grow_section_bbox_max_edge,
    grow_section_bbox_min_edge,
)
from nf_metro.parser.model import MetroGraph, Section, Station

if TYPE_CHECKING:
    from nf_metro.layout.route_plan import FanPlan, FanPlanId


def _planned_fans(graph: MetroGraph) -> tuple[FanPlan, ...]:
    return tuple(plan for plan in graph.fan_plans if plan.owns_geometry)


def planned_fan_layout_station_ids(graph: MetroGraph) -> set[str]:
    """Return the stations whose secondary coordinate has one semantic owner."""
    return {
        station_id
        for plan in _planned_fans(graph)
        for station_id in plan.layout_station_ids
    }


def planned_fan_layout_section_ids(graph: MetroGraph) -> set[str]:
    """Return sections containing coordinates owned by a semantic fan plan."""
    return {
        station.section_id
        for station_id in planned_fan_layout_station_ids(graph)
        if (station := graph.stations.get(station_id)) is not None
        and station.section_id is not None
    }


def planned_fan_port_ids(graph: MetroGraph) -> set[str]:
    """Return ports participating in a planned fan's complete membership."""
    return {
        port_id
        for plan in _planned_fans(graph)
        for port_id in (*plan.entry_port_ids, *plan.exit_port_ids)
    }


def _centreline_coordinate(graph: MetroGraph, plan: FanPlan) -> float | None:
    frame = plan.frame
    anchor = plan.centreline_anchor
    if frame is None or anchor is None:
        return None
    station = graph.stations.get(anchor.station_id)
    if station is None:
        from nf_metro.layout.phases.guards import PhaseInvariantError

        raise PhaseInvariantError(
            f"planned fan {plan.id!r} centreline anchor "
            f"{anchor.station_id!r} is missing"
        )
    return plan.appearance_centreline_coordinate(anchor, station)


def _apply_planned_fan_port_geometry(graph: MetroGraph) -> None:
    """Continue each settled local fan frame through same-axis boundary ports."""
    for plan in _planned_fans(graph):
        frame = plan.frame
        centreline = _centreline_coordinate(graph, plan)
        if frame is None or centreline is None:
            continue
        for port_id in plan.centreline_port_ids:
            port = graph.ports.get(port_id)
            station = graph.stations.get(port_id)
            if port is None or station is None:
                continue
            frame.secondary.set(port, centreline)
            frame.secondary.set(station, centreline)


def _snapshot_planned_fan_centrelines(
    graph: MetroGraph,
) -> Mapping[FanPlanId, float]:
    """Freeze each complete plan's centreline at a structural boundary."""
    centrelines: dict[FanPlanId, float] = {}
    for plan in _planned_fans(graph):
        if not plan.layout_station_ids:
            continue
        centreline = _centreline_coordinate(graph, plan)
        if centreline is None:
            from nf_metro.layout.phases.guards import PhaseInvariantError

            raise PhaseInvariantError(
                f"planned fan {plan.id!r} has no settled centreline"
            )
        centrelines[plan.id] = centreline
    return MappingProxyType(centrelines)


def _apply_planned_fan_geometry(
    graph: MetroGraph,
    centrelines: Mapping[FanPlanId, float],
) -> None:
    """Place every plan-owned station from its one frozen relative frame."""
    for plan in _planned_fans(graph):
        frame = plan.frame
        if frame is None or not plan.layout_station_ids:
            continue
        try:
            centreline = centrelines[plan.id]
        except KeyError as error:
            from nf_metro.layout.phases.guards import PhaseInvariantError

            raise PhaseInvariantError(
                f"planned fan {plan.id!r} has no frozen placement centreline"
            ) from error
        _materialise_plan_stations(graph.stations, plan, centreline)

        if frame.secondary.name == "y":
            graph.symfan_trunk_station_ids.update(plan.centreline_station_ids)
            graph.half_grid_station_ids.update(
                station_id
                for branch in plan.branches
                if branch.lane_offset is not None
                and abs(
                    branch.lane_offset / frame.secondary.step
                    - round(branch.lane_offset / frame.secondary.step)
                )
                > 1e-9
                for station_id in branch.lane_station_ids
            )


def _materialise_plan_stations(
    stations: Mapping[str, Station],
    plan: FanPlan,
    centreline: float,
    *,
    section_id: str | None = None,
) -> None:
    """Place one plan into a station mapping from its settled centreline."""
    frame = plan.frame
    if frame is None:
        return

    def eligible(station: Station) -> bool:
        return section_id is None or station.section_id == section_id

    for station_id in plan.centreline_station_ids:
        station = stations.get(station_id)
        if station is not None and eligible(station):
            frame.secondary.set(station, centreline)
    for branch in plan.branches:
        if branch.lane_offset is None:
            continue
        coordinate = plan.appearance_coordinate(centreline, branch.lane_offset)
        for station_id in branch.lane_station_ids:
            station = stations.get(station_id)
            if station is not None and eligible(station):
                frame.secondary.set(station, coordinate)


def _fit_planned_fan_bboxes(
    graph: MetroGraph,
    section_x_padding: float,
    section_y_padding: float,
) -> bool:
    """Fit section extents to plan-owned coordinates after frame translation."""
    x_changed = False
    for plan in _planned_fans(graph):
        frame = plan.frame
        if frame is None:
            continue
        by_section: dict[str, list[str]] = {}
        for station_id in plan.layout_station_ids:
            station = graph.stations.get(station_id)
            if station is not None and station.section_id is not None:
                by_section.setdefault(station.section_id, []).append(station_id)
        for section_id, station_ids in by_section.items():
            section = graph.sections.get(section_id)
            if section is None:
                continue
            stations = [graph.stations[station_id] for station_id in station_ids]
            if frame.secondary.name == "y":
                desired_top = min(
                    station.y
                    - max(
                        section_y_padding,
                        ICON_HALF_HEIGHT
                        if station.off_track or station.is_terminus
                        else station_radius_approx(),
                    )
                    for station in stations
                )
                desired_bottom = max(
                    station.y
                    + max(
                        section_y_padding,
                        ICON_HALF_HEIGHT
                        if station.off_track or station.is_terminus
                        else station_radius_approx(),
                    )
                    for station in stations
                )
                grow_section_bbox_min_edge(graph, section, "y", desired_top)
                grow_section_bbox_max_edge(graph, section, "y", desired_bottom)
            else:
                desired_left = (
                    min(station.x for station in stations) - section_x_padding
                )
                desired_right = (
                    max(station.x for station in stations) + section_x_padding
                )
                old_left = section.bbox_x
                old_right = section.bbox_x + section.bbox_w
                grow_section_bbox_min_edge(graph, section, "x", desired_left)
                grow_section_bbox_max_edge(graph, section, "x", desired_right)
                x_changed |= (
                    section.bbox_x != old_left
                    or section.bbox_x + section.bbox_w != old_right
                )
    return x_changed


def apply_planned_fans_to_section_subgraph(
    graph: MetroGraph, subgraph: MetroGraph, section: Section
) -> None:
    """Seat plan-owned local stations before their section bbox is measured."""
    for plan in _planned_fans(graph):
        frame = plan.frame
        if frame is None or not set(plan.layout_station_ids).intersection(
            subgraph.stations
        ):
            continue
        origin = subgraph.stations.get(plan.fork_station_id)
        if origin is None:
            origin = next(
                (
                    subgraph.stations[station_id]
                    for station_id in plan.centreline_station_ids
                    if station_id in subgraph.stations
                ),
                None,
            )
        axis = frame.secondary.name
        if origin is not None:
            centreline = getattr(origin, axis)
        else:
            local_anchor = plan.local_frame_anchor
            anchor_station = subgraph.stations.get(
                local_anchor.station_id if local_anchor is not None else ""
            )
            if anchor_station is None or local_anchor is None:
                continue
            centreline = plan.appearance_centreline_coordinate(
                local_anchor, anchor_station
            )
        _materialise_plan_stations(
            subgraph.stations, plan, centreline, section_id=section.id
        )
