"""Materialise immutable fan plans in settled section-local frames."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from nf_metro.layout.constants import ICON_HALF_HEIGHT
from nf_metro.layout.geometry import flow_port_sides, lanes_run_along_x
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


def _layout_section_id(graph: MetroGraph, plan: FanPlan) -> str | None:
    station = graph.stations.get(plan.fork_station_id)
    if station is not None and station.section_id is not None:
        return station.section_id
    port = graph.ports.get(plan.fork_station_id)
    return port.section_id if port is not None else None


def _centreline_ports(
    graph: MetroGraph, plan: FanPlan, section_id: str
) -> Iterable[str]:
    if plan.frame is None or plan.direction is None:
        return ()
    sides = flow_port_sides(plan.direction)
    candidates = []
    for port_id in (*plan.entry_port_ids, *plan.exit_port_ids):
        port = graph.ports.get(port_id)
        if (
            port is not None
            and port.section_id == section_id
            and port.side in sides
            and port_id not in candidates
        ):
            candidates.append(port_id)
    candidates.sort(key=lambda port_id: not graph.ports[port_id].is_entry)
    return candidates


def _centreline_coordinate(graph: MetroGraph, plan: FanPlan) -> float | None:
    frame = plan.frame
    if frame is None:
        return None
    section_id = _layout_section_id(graph, plan)
    if section_id is None:
        return None
    axis = frame.secondary.name
    local_trunks = tuple(
        branch
        for branch in plan.branches
        if branch.is_trunk_continuation
        and branch.lane_station_ids
        and not branch.landing_port_ids
    )
    if len(local_trunks) == 1:
        origin = graph.stations.get(plan.fork_station_id)
        if origin is not None:
            return getattr(origin, axis)
    for port_id in _centreline_ports(graph, plan, section_id):
        station = graph.stations.get(port_id)
        if station is not None:
            return getattr(station, axis)
    origin = graph.stations.get(plan.fork_station_id)
    return getattr(origin, axis) if origin is not None else None


def _local_frame_centreline(graph: MetroGraph, plan: FanPlan) -> float | None:
    frame = plan.frame
    if frame is None or plan.local_frame_anchor_station_id is None:
        return None
    anchor = graph.stations.get(plan.local_frame_anchor_station_id)
    if anchor is None or plan.local_frame_anchor_offset is None:
        return None
    return (
        frame.secondary.get(anchor)
        - frame.secondary_sign * plan.local_frame_anchor_offset
    )


def _upstream_port_centreline(graph: MetroGraph, plan: FanPlan) -> float | None:
    frame = plan.frame
    layout_section_id = _layout_section_id(graph, plan)
    layout_section = graph.sections.get(layout_section_id or "")
    if frame is None or plan.direction is None or layout_section is None:
        return None
    horizontal = not lanes_run_along_x(plan.direction)
    candidates: list[tuple[float, str]] = []
    for port_id in (*plan.entry_port_ids, *plan.exit_port_ids):
        port = graph.ports.get(port_id)
        section = graph.sections.get(port.section_id) if port is not None else None
        if (
            port is None
            or port.is_entry
            or section is None
            or section.id == layout_section.id
            or (not lanes_run_along_x(section.direction)) != horizontal
            or port.side not in flow_port_sides(section.direction)
        ):
            continue
        if horizontal:
            same_strip = section.grid_row == layout_section.grid_row
            distance = (layout_section.grid_col - section.grid_col) * frame.primary_sign
        else:
            same_strip = section.grid_col == layout_section.grid_col
            distance = (layout_section.grid_row - section.grid_row) * frame.primary_sign
        if same_strip and distance > 0:
            candidates.append((distance, port_id))
    if not candidates:
        return None
    _, port_id = min(candidates)
    station = graph.stations.get(port_id)
    return frame.secondary.get(station) if station is not None else None


def planned_fan_centreline_port_ids(
    graph: MetroGraph, plan: FanPlan
) -> tuple[str, ...]:
    """Ports whose settled boundary anchor continues the fan centreline."""
    del graph
    return plan.centreline_port_ids


def _apply_planned_fan_port_geometry(graph: MetroGraph) -> None:
    """Continue each settled local fan frame through same-axis boundary ports."""
    for plan in _planned_fans(graph):
        frame = plan.frame
        centreline = _upstream_port_centreline(graph, plan)
        if centreline is None:
            centreline = _centreline_coordinate(graph, plan)
        if centreline is None:
            centreline = _local_frame_centreline(graph, plan)
        if frame is None or centreline is None:
            continue
        layout_section_id = _layout_section_id(graph, plan)
        layout_section = graph.sections.get(layout_section_id or "")
        if layout_section is None:
            continue
        for port_id in planned_fan_centreline_port_ids(graph, plan):
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
        coordinate = centreline + frame.secondary_sign * branch.lane_offset
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
            anchor = subgraph.stations.get(plan.local_frame_anchor_station_id or "")
            if anchor is None or plan.local_frame_anchor_offset is None:
                continue
            centreline = (
                getattr(anchor, axis)
                - frame.secondary_sign * plan.local_frame_anchor_offset
            )
        _materialise_plan_stations(
            subgraph.stations, plan, centreline, section_id=section.id
        )
