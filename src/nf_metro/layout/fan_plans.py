"""Structural fan recognition and immutable relative geometry plans.

This module is intentionally independent of layout phase order and routing
dispatch.  It reads authored edge identity plus resolver lineage, recognises a
complete fan, and either gives that whole object one owner or records one
deterministic legacy disposition.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeVar, cast, runtime_checkable

from nf_metro.graph_views import directed_graph, longest_path_layers
from nf_metro.layout.constants import graph_offset_step
from nf_metro.layout.fan_ordering import fanout_divergence_peel_order
from nf_metro.layout.geometry import (
    AxisFrame,
    flow_port_sides,
    lanes_run_along_x,
    lanes_run_along_y,
)
from nf_metro.layout.labels import tb_left_label_marker_pitch
from nf_metro.layout.route_plan import (
    DemandId,
    FanAppearancePolicy,
    FanBranchPlan,
    FanBranchPlanId,
    FanCentrelineAnchor,
    FanOffsetAssignment,
    FanOffsetCarrier,
    FanPlan,
    FanPlanDisposition,
    FanPlanId,
    FanRouteEmission,
    FanRouteEmitter,
    SharedReferenceId,
)
from nf_metro.parser.commitments import FlowDirection, is_flow_direction
from nf_metro.parser.model import MetroGraph, PortSide
from nf_metro.parser.route_topology import (
    AuthoredEdgeFact,
    BundleId,
    ConnectorId,
    ConvergenceId,
    ResolvedConvergenceView,
    ResolvedEdge,
    RouteConnector,
    RouteTopologyQuery,
    semantic_route_id,
)

if TYPE_CHECKING:
    from nf_metro.layout.routing.common import RoutedPath


@runtime_checkable
class FanTopologyQuery(Protocol):
    """Route-topology surface required by the fan planner."""

    @property
    def authored_edges(self) -> tuple[AuthoredEdgeFact, ...]: ...

    @property
    def convergences(self) -> tuple[ResolvedConvergenceView, ...]: ...

    def resolved_paths(
        self, edge_id: ConnectorId
    ) -> tuple[tuple[ResolvedEdge, ...], ...]: ...

    def connector(self, edge_id: ConnectorId) -> RouteConnector: ...


def symmetric_lane_offsets(branch_count: int, lane_pitch: float) -> tuple[float, ...]:
    """Return centreline-relative lane offsets in canonical branch order."""
    if branch_count < 2:
        raise ValueError("a fan requires at least two branches")
    if not math.isfinite(lane_pitch) or lane_pitch <= 0:
        raise ValueError("fan lane pitch must be finite and positive")
    midpoint = (branch_count - 1) / 2.0
    return tuple((rank - midpoint) * lane_pitch for rank in range(branch_count))


def fan_lane_offsets(
    branches: Sequence[FanBranchPlan],
    appearance_policy: FanAppearancePolicy,
    lane_pitch: float,
    appearance_centreline_branch_id: FanBranchPlanId | None = None,
) -> tuple[float, ...]:
    """Return canonical lane offsets for an authored fan appearance."""
    if not isinstance(appearance_policy, FanAppearancePolicy):
        raise ValueError("fan appearance policy is not canonical")
    if (
        appearance_policy is FanAppearancePolicy.SYMMETRIC
        or appearance_centreline_branch_id is None
    ):
        return symmetric_lane_offsets(len(branches), lane_pitch)
    if not branches:
        raise ValueError("fan requires at least one branch")
    if appearance_centreline_branch_id not in {branch.id for branch in branches}:
        raise ValueError("fan appearance centreline names an unknown branch")
    offsets = {appearance_centreline_branch_id: 0.0}
    offsets.update(
        (branch.id, slot * lane_pitch)
        for slot, branch in enumerate(
            (
                branch
                for branch in branches
                if branch.id != appearance_centreline_branch_id
            ),
            start=1,
        )
    )
    return tuple(offsets[branch.id] for branch in branches)


def _appearance_centreline_branch_id(
    branches: Sequence[FanBranchPlan],
    appearance_policy: FanAppearancePolicy,
    structural_trunk_rank: int | None,
) -> FanBranchPlanId | None:
    """Choose the branch that a straight local fan keeps on its main track."""
    if appearance_policy is not FanAppearancePolicy.STRAIGHT or not any(
        branch.lane_station_ids for branch in branches
    ):
        return None
    trunk_branches = tuple(
        branch for branch in branches if branch.is_trunk_continuation
    )
    if len(trunk_branches) == 1:
        return trunk_branches[0].id
    if structural_trunk_rank is not None:
        structural_trunk = next(
            (branch for branch in branches if branch.rank == structural_trunk_rank),
            None,
        )
        if structural_trunk is not None:
            return structural_trunk.id
    return min(branches, key=lambda branch: (branch.opening_rank, branch.rank)).id


def vertical_fan_label_lane_pitch(
    graph: MetroGraph,
    branches: Sequence[FanBranchPlan],
    frame: AxisFrame,
    floor: float = 0.0,
) -> float:
    """Return the uniform X pitch needed by same-layer vertical fan labels."""
    if frame.secondary.name != "x":
        return floor
    section_ids = {
        section_id
        for branch in branches
        for station_id in branch.lane_station_ids
        if (section_id := graph.section_for_station(station_id)) is not None
    }
    if len(section_ids) != 1:
        return floor
    section_id = next(iter(section_ids))
    section = graph.sections[section_id]
    node_ids = tuple(
        station_id
        for station_id in section.station_ids
        if station_id in graph.stations and station_id not in graph.ports
    )
    node_set = set(node_ids)
    layers = longest_path_layers(
        directed_graph(
            node_ids,
            (
                (edge.source, edge.target)
                for edge in graph.edges
                if edge.source in node_set and edge.target in node_set
            ),
        ),
        node_ids,
    )
    from nf_metro.layout.routing.reversal import tb_positive_fan_sections

    lane_sign = (
        1.0 if section_id in tb_positive_fan_sections(graph) else frame.secondary_sign
    )
    screen_order = sorted(
        (branch for branch in branches if branch.lane_offset is not None),
        key=lambda branch: frame.secondary_sign * cast(float, branch.lane_offset),
    )
    pitch = floor
    for left_branch, right_branch in pairwise(screen_order):
        left_by_layer = {
            layers[station_id]: station_id
            for station_id in left_branch.lane_station_ids
            if station_id in layers
        }
        for right_id in right_branch.lane_station_ids:
            layer = layers.get(right_id)
            left_id = left_by_layer.get(layer) if layer is not None else None
            right = graph.stations.get(right_id)
            if left_id is None or right is None or not right.label:
                continue
            pitch = max(
                pitch,
                tb_left_label_marker_pitch(
                    right.label,
                    left_line_count=len(graph.station_lines(left_id)),
                    right_line_count=len(graph.station_lines(right_id)),
                    lane_sign=lane_sign,
                    offset_step=graph_offset_step(graph),
                ),
            )
    return pitch


def _fan_branch_solo_station_ids(
    graph: MetroGraph, branch: FanBranchPlan
) -> tuple[str, ...]:
    """Branch stations whose only present line may return to its trunk."""
    if len(branch.line_ids) != 1:
        return ()
    return cast(
        tuple[str, ...],
        _ordered_unique(
            station_id
            for path in branch.resolved_paths
            for edge in path
            for station_id in (edge.source, edge.target)
            if station_id not in graph.junction_ids
            and graph.station_lines(station_id) == list(branch.line_ids)
        ),
    )


@dataclass(frozen=True, slots=True)
class FanPlanQuery:
    """Read-only ownership indexes over one complete fan-plan build."""

    plans: tuple[FanPlan, ...]
    _by_id: Mapping[FanPlanId, FanPlan]
    _by_fork: Mapping[str, FanPlan]
    _by_authored_edge: Mapping[ConnectorId, FanPlan]
    _structural_by_resolved_edge: Mapping[ResolvedEdge, FanPlan]
    _structural_branch_by_resolved_edge: Mapping[ResolvedEdge, FanBranchPlan]
    _route_emission_by_resolved_edge: Mapping[
        ResolvedEdge, tuple[FanPlan, FanBranchPlan, FanRouteEmission]
    ]
    _by_station: Mapping[str, FanPlan]

    @classmethod
    def build(cls, plans: tuple[FanPlan, ...]) -> FanPlanQuery:
        by_id: dict[FanPlanId, FanPlan] = {}
        by_fork: dict[str, FanPlan] = {}
        by_authored_edge: dict[ConnectorId, FanPlan] = {}
        structural_by_resolved_edge: dict[ResolvedEdge, FanPlan] = {}
        structural_branch_by_resolved_edge: dict[ResolvedEdge, FanBranchPlan] = {}
        route_emission_by_resolved_edge: dict[
            ResolvedEdge, tuple[FanPlan, FanBranchPlan, FanRouteEmission]
        ] = {}
        shared_branch_edges: set[ResolvedEdge] = set()
        by_station: dict[str, FanPlan] = {}
        for plan in plans:
            if plan.id in by_id:
                raise ValueError(f"duplicate fan plan id {plan.id!r}")
            by_id[plan.id] = plan
            if plan.disposition is not FanPlanDisposition.PLANNED:
                continue
            if plan.fork_station_id in by_fork:
                raise ValueError("two planned fans own one fork")
            by_fork[plan.fork_station_id] = plan
            for edge_id in plan.authored_edge_ids:
                if edge_id in by_authored_edge:
                    raise ValueError("two planned fans own one authored edge")
                by_authored_edge[edge_id] = plan
            for edge in plan.resolved_member_edges:
                if edge in structural_by_resolved_edge:
                    raise ValueError("two planned fans own one resolved edge")
                structural_by_resolved_edge[edge] = plan
            for branch in plan.branches:
                for path in branch.continuation_resolved_paths:
                    for edge in path:
                        if edge in shared_branch_edges:
                            continue
                        existing = structural_branch_by_resolved_edge.get(edge)
                        if existing is not None and existing is not branch:
                            del structural_branch_by_resolved_edge[edge]
                            shared_branch_edges.add(edge)
                        else:
                            structural_branch_by_resolved_edge[edge] = branch
            branches_by_id = {branch.id: branch for branch in plan.branches}
            for emission in plan.route_emissions:
                if emission.edge in route_emission_by_resolved_edge:
                    raise ValueError("two planned fan emitters own one resolved edge")
                route_emission_by_resolved_edge[emission.edge] = (
                    plan,
                    branches_by_id[emission.branch_id],
                    emission,
                )
            for station_id in plan.owned_station_ids:
                if station_id in by_station:
                    raise ValueError("two planned fans own one station")
                by_station[station_id] = plan
        return cls(
            plans=plans,
            _by_id=MappingProxyType(by_id),
            _by_fork=MappingProxyType(by_fork),
            _by_authored_edge=MappingProxyType(by_authored_edge),
            _structural_by_resolved_edge=MappingProxyType(structural_by_resolved_edge),
            _structural_branch_by_resolved_edge=MappingProxyType(
                structural_branch_by_resolved_edge
            ),
            _route_emission_by_resolved_edge=MappingProxyType(
                route_emission_by_resolved_edge
            ),
            _by_station=MappingProxyType(by_station),
        )

    def plan(self, plan_id: FanPlanId) -> FanPlan:
        return self._by_id[plan_id]

    def __deepcopy__(self, memo: dict[int, object]) -> FanPlanQuery:
        del memo
        return self

    def planned_for_fork(self, station_id: str) -> FanPlan | None:
        return self._by_fork.get(station_id)

    def owner_for_authored_edge(self, edge_id: ConnectorId) -> FanPlan | None:
        return self._by_authored_edge.get(edge_id)

    def structural_owner_for_resolved_edge(self, edge: ResolvedEdge) -> FanPlan | None:
        return self._structural_by_resolved_edge.get(edge)

    def structural_branch_for_resolved_edge(
        self, edge: ResolvedEdge
    ) -> FanBranchPlan | None:
        return self._structural_branch_by_resolved_edge.get(edge)

    def route_emission_for_resolved_edge(
        self, edge: ResolvedEdge
    ) -> tuple[FanPlan, FanBranchPlan, FanRouteEmission] | None:
        return self._route_emission_by_resolved_edge.get(edge)

    def owner_for_station(self, station_id: str) -> FanPlan | None:
        return self._by_station.get(station_id)


@dataclass(frozen=True, slots=True)
class FanPlanExecution:
    """Context-local result installed for later layout and routing consumers."""

    plans: tuple[FanPlan, ...]
    query: FanPlanQuery

    def __deepcopy__(self, memo: dict[int, object]) -> FanPlanExecution:
        del memo
        return self


def _authored_edges(topology: FanTopologyQuery) -> tuple[AuthoredEdgeFact, ...]:
    return tuple(sorted(topology.authored_edges, key=lambda fact: fact.rank))


_T = TypeVar("_T")


def _ordered_unique(values: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(dict.fromkeys(values))


def _node_rank(facts: Sequence[AuthoredEdgeFact]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fact in facts:
        result.setdefault(fact.key.source, fact.rank)
        result.setdefault(fact.key.target, fact.rank)
    return result


def _adjacency(
    facts: Sequence[AuthoredEdgeFact],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[tuple[str, str], tuple[AuthoredEdgeFact, ...]],
]:
    targets: dict[str, list[str]] = defaultdict(list)
    sources: dict[str, list[str]] = defaultdict(list)
    bundles: dict[tuple[str, str], list[AuthoredEdgeFact]] = defaultdict(list)
    for fact in facts:
        key = (fact.key.source, fact.key.target)
        bundles[key].append(fact)
        if fact.key.target not in targets[fact.key.source]:
            targets[fact.key.source].append(fact.key.target)
        if fact.key.source not in sources[fact.key.target]:
            sources[fact.key.target].append(fact.key.source)
    return (
        {source: tuple(values) for source, values in targets.items()},
        {target: tuple(values) for target, values in sources.items()},
        {key: tuple(values) for key, values in bundles.items()},
    )


def _distances(adjacency: Mapping[str, tuple[str, ...]], root: str) -> dict[str, int]:
    result = {root: 0}
    pending = deque([root])
    while pending:
        source = pending.popleft()
        for target in adjacency.get(source, ()):
            if target not in result:
                result[target] = result[source] + 1
                pending.append(target)
    return result


def _nearest_common_join(
    adjacency: Mapping[str, tuple[str, ...]],
    branch_roots: tuple[str, ...],
    ranks: Mapping[str, int],
) -> str | None:
    distances = tuple(_distances(adjacency, root) for root in branch_roots)
    common = set(distances[0]).intersection(*(set(item) for item in distances[1:]))
    candidates = [
        station_id
        for station_id in common
        if all(item[station_id] > 0 for item in distances)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda station_id: (
            max(item[station_id] for item in distances),
            sum(item[station_id] for item in distances),
            ranks.get(station_id, len(ranks)),
            station_id,
        ),
    )


def _can_reach(
    adjacency: Mapping[str, tuple[str, ...]], source: str, target: str
) -> bool:
    return target in _distances(adjacency, source)


def _unique_path_to_join(
    adjacency: Mapping[str, tuple[str, ...]], root: str, join: str
) -> tuple[str, ...] | None:
    path = [root]
    current = root
    visited = {root}
    while current != join:
        continuations = tuple(
            candidate
            for candidate in adjacency.get(current, ())
            if _can_reach(adjacency, candidate, join)
        )
        if len(continuations) != 1:
            return None
        current = continuations[0]
        if current in visited:
            return None
        visited.add(current)
        path.append(current)
    return tuple(path)


def _linear_path(
    adjacency: Mapping[str, tuple[str, ...]], root: str
) -> tuple[str, ...]:
    path = [root]
    visited = {root}
    current = root
    while len(adjacency.get(current, ())) == 1:
        target = adjacency[current][0]
        if target in visited:
            break
        visited.add(target)
        path.append(target)
        current = target
    return tuple(path)


def _paths_for(
    topology: FanTopologyQuery, facts: Iterable[AuthoredEdgeFact]
) -> tuple[tuple[ResolvedEdge, ...], ...]:
    result: list[tuple[ResolvedEdge, ...]] = []
    for fact in facts:
        result.extend(topology.resolved_paths(fact.id))
    return tuple(result)


def _path_nodes(path: tuple[ResolvedEdge, ...]) -> tuple[str, ...]:
    if not path:
        return ()
    return (path[0].source, *(edge.target for edge in path))


def _common_prefix_nodes(paths: Sequence[tuple[ResolvedEdge, ...]]) -> tuple[str, ...]:
    nodes = tuple(_path_nodes(path) for path in paths)
    if not nodes or any(not item for item in nodes):
        return ()
    prefix: list[str] = []
    for values in zip(*nodes, strict=False):
        if len(set(values)) != 1:
            break
        prefix.append(values[0])
    return tuple(prefix)


def _common_suffix_nodes(paths: Sequence[tuple[ResolvedEdge, ...]]) -> tuple[str, ...]:
    reversed_nodes = tuple(tuple(reversed(_path_nodes(path))) for path in paths)
    if not reversed_nodes or any(not item for item in reversed_nodes):
        return ()
    suffix: list[str] = []
    for values in zip(*reversed_nodes, strict=False):
        if len(set(values)) != 1:
            break
        suffix.append(values[0])
    return tuple(reversed(suffix))


def _trim_member_path(
    path: tuple[ResolvedEdge, ...], fork_id: str, join_id: str | None
) -> tuple[ResolvedEdge, ...]:
    nodes = _path_nodes(path)
    start = nodes.index(fork_id) if fork_id in nodes else 0
    end = (
        nodes.index(join_id) if join_id is not None and join_id in nodes else len(path)
    )
    if end < start:
        return ()
    return path[start:end]


def _facts_for_node_path(
    path: tuple[str, ...],
    bundles: Mapping[tuple[str, str], tuple[AuthoredEdgeFact, ...]],
    line_ids: frozenset[str] | None = None,
) -> tuple[AuthoredEdgeFact, ...] | None:
    result: list[AuthoredEdgeFact] = []
    for source, target in zip(path, path[1:]):
        matching = tuple(
            fact
            for fact in bundles[(source, target)]
            if line_ids is None or fact.key.line_id in line_ids
        )
        if not matching:
            return None
        result.extend(matching)
    return tuple(result)


def _extra_output_facts(
    path: tuple[str, ...],
    adjacency: Mapping[str, tuple[str, ...]],
    bundles: Mapping[tuple[str, str], tuple[AuthoredEdgeFact, ...]],
) -> tuple[AuthoredEdgeFact, ...]:
    result: list[AuthoredEdgeFact] = []
    for index, source in enumerate(path[:-1]):
        continuation = path[index + 1]
        for target in adjacency.get(source, ()):
            if target != continuation:
                result.extend(bundles[(source, target)])
    return tuple(result)


def _direction_for_fork(
    graph: MetroGraph,
    fork_id: str,
    source_id: str,
    lead_facts: Sequence[AuthoredEdgeFact],
) -> FlowDirection | None:
    section_id = graph.section_for_station(fork_id)
    if section_id is None and fork_id in graph.ports:
        section_id = graph.ports[fork_id].section_id
    if section_id is None:
        section_id = next(
            (
                fact.source_section
                for fact in lead_facts
                if fact.source_section is not None
            ),
            None,
        )
    if section_id is None:
        section_id = graph.section_for_station(source_id)
    section = graph.sections.get(section_id or "")
    if section is None or not is_flow_direction(section.direction):
        return None
    return section.direction


def _port_ids(
    graph: MetroGraph, paths: Iterable[tuple[ResolvedEdge, ...]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    entry: list[str] = []
    exit_: list[str] = []
    for path in paths:
        for station_id in _path_nodes(path):
            port = graph.ports.get(station_id)
            if port is None:
                continue
            target = entry if port.is_entry else exit_
            if station_id not in target:
                target.append(station_id)
    return tuple(entry), tuple(exit_)


def _grid_position(graph: MetroGraph, section_id: str) -> tuple[int, int]:
    section = graph.sections[section_id]
    override = graph.grid_overrides.get(section_id)
    if override is not None:
        return override[0], override[1]
    return section.grid_col, section.grid_row


def _trunk_followers(
    graph: MetroGraph,
    fork_id: str,
    join_id: str | None,
    approach_paths: Iterable[tuple[ResolvedEdge, ...]],
    departure_paths: Iterable[tuple[ResolvedEdge, ...]],
) -> tuple[str, ...]:
    result: list[str] = []
    for path in approach_paths:
        nodes = _path_nodes(path)
        if fork_id not in nodes:
            continue
        for station_id in reversed(nodes[: nodes.index(fork_id)]):
            if station_id in graph.ports or station_id in graph.junction_ids:
                continue
            if station_id not in result:
                result.append(station_id)
            break
    if join_id is None:
        return tuple(result)
    for path in departure_paths:
        nodes = _path_nodes(path)
        if join_id not in nodes:
            continue
        for station_id in nodes[nodes.index(join_id) + 1 :]:
            if station_id in graph.ports or station_id in graph.junction_ids:
                continue
            if station_id not in result:
                result.append(station_id)
            break
    return tuple(result)


def _entry_offset_carriers(
    graph: MetroGraph,
    entry_handoff_paths: tuple[tuple[ResolvedEdge, ...], ...],
    offset_line_order: tuple[str, ...],
    offset_sign: int,
) -> tuple[FanOffsetCarrier, ...]:
    """Return the exact flat, full-bundle chain feeding a fan handoff."""
    if not entry_handoff_paths or not offset_line_order:
        return ()
    fan_line_ids = frozenset(offset_line_order)
    carried_to_station: dict[str, set[str]] = defaultdict(set)
    for path in entry_handoff_paths:
        if path:
            carried_to_station[path[0].source].update(
                edge.line_id for edge in path if edge.line_id in fan_line_ids
            )
    path_station_ids = {
        station_id
        for path in entry_handoff_paths
        for edge in path
        for station_id in (edge.source, edge.target)
    }
    carriers: dict[str, set[str]] = {}
    queue = deque(carried_to_station)
    while queue:
        current_id = queue.popleft()
        current_lines = carried_to_station[current_id]
        section_id = graph.section_for_station(current_id)
        section = graph.sections.get(section_id or "")
        if section is None or lanes_run_along_x(section.direction):
            continue
        incoming_by_source: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges_to(current_id):
            if graph.section_for_station(edge.source) == section_id:
                incoming_by_source[edge.source].add(edge.line_id)
        predecessors = [
            (source_id, current_lines.intersection(carried_lines))
            for source_id, carried_lines in incoming_by_source.items()
            if current_lines.intersection(carried_lines)
        ]
        if len(predecessors) != 1:
            continue
        source_id, propagated = predecessors[0]
        if propagated != current_lines:
            continue
        if source_id not in path_station_ids:
            carriers.setdefault(source_id, set()).update(propagated)
        known = carried_to_station.setdefault(source_id, set())
        unseen = propagated.difference(known)
        if unseen:
            known.update(unseen)
            queue.append(source_id)
    return tuple(
        FanOffsetCarrier(
            station_id=station_id,
            assignments=tuple(
                FanOffsetAssignment(line_id, rank * offset_sign)
                for rank, line_id in enumerate(offset_line_order)
                if line_id in carried_lines
            ),
        )
        for station_id, carried_lines in carriers.items()
    )


def _offset_carriers(
    graph: MetroGraph,
    *,
    branches: Sequence[FanBranchPlan],
    offset_line_order: tuple[str, ...],
    shared_paths: Sequence[tuple[ResolvedEdge, ...]],
    shared_station_ids: Iterable[str | None],
    upstream_carriers: Sequence[FanOffsetCarrier],
    offset_sign: int,
) -> tuple[FanOffsetCarrier, ...]:
    """Freeze stations whose fan-line permutation is structurally shared."""
    if not offset_line_order:
        return ()

    fan_lines = frozenset(offset_line_order)
    carrier_lines: dict[str, set[str]] = {}

    def add_station(station_id: str | None, lines: Iterable[str]) -> None:
        if station_id is None or station_id not in graph.stations:
            return
        present = fan_lines.intersection(lines, graph.station_lines(station_id))
        if len(present) >= 2:
            carrier_lines.setdefault(station_id, set()).update(present)

    shared_path_lines: dict[str, set[str]] = defaultdict(set)
    for path in shared_paths:
        for edge in path:
            shared_path_lines[edge.source].add(edge.line_id)
            shared_path_lines[edge.target].add(edge.line_id)
    for station_id, lines in shared_path_lines.items():
        add_station(station_id, lines)
    for shared_station_id in shared_station_ids:
        add_station(shared_station_id, fan_lines)
    for carrier in upstream_carriers:
        add_station(carrier.station_id, carrier.line_ids)

    branch_incidence: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for branch in branches:
        for path in branch.resolved_paths:
            for edge in path:
                branch_incidence[edge.source][branch.rank].add(edge.line_id)
                branch_incidence[edge.target][branch.rank].add(edge.line_id)
    for station_id, by_branch in branch_incidence.items():
        if len(by_branch) < 2:
            continue
        add_station(
            station_id,
            (line_id for lines in by_branch.values() for line_id in lines),
        )

    return tuple(
        FanOffsetCarrier(
            station_id=station_id,
            assignments=tuple(
                FanOffsetAssignment(line_id, rank * offset_sign)
                for rank, line_id in enumerate(offset_line_order)
                if line_id in lines
            ),
        )
        for station_id, lines in carrier_lines.items()
    )


def _bottom_exit_source_port_id(
    graph: MetroGraph,
    exit_port_ids: Sequence[str],
) -> str | None:
    candidates = tuple(
        port_id
        for port_id in exit_port_ids
        if (port := graph.ports.get(port_id)) is not None
        and not port.is_entry
        and port.side is PortSide.BOTTOM
        and (section := graph.sections.get(port.section_id)) is not None
        and lanes_run_along_x(section.direction)
        and AxisFrame.flow_sign(section.direction) > 0
    )
    return candidates[0] if len(candidates) == 1 else None


def _route_emissions(
    graph: MetroGraph,
    fork_id: str,
    branches: Sequence[FanBranchPlan],
    exit_port_ids: Sequence[str],
    offset_line_order: Sequence[str],
) -> tuple[FanRouteEmission, ...]:
    """Freeze edges handled by the stacked RIGHT-landing fan emitter."""
    if (
        fork_id not in graph.junction_ids
        or _bottom_exit_source_port_id(graph, exit_port_ids) is None
    ):
        return ()
    landing_section_ids: list[str] = []
    for branch in branches:
        if len(branch.landing_port_ids) != 1:
            return ()
        port = graph.ports.get(branch.landing_port_ids[0])
        section = graph.sections.get(port.section_id) if port is not None else None
        if (
            port is None
            or port.side is not PortSide.RIGHT
            or section is None
            or not lanes_run_along_y(section.direction)
        ):
            return ()
        if port.section_id not in landing_section_ids:
            landing_section_ids.append(port.section_id)
    if len(landing_section_ids) != len(branches):
        return ()

    result = tuple(
        FanRouteEmission(
            edge=edge,
            branch_id=branch.id,
            emitter=FanRouteEmitter.BOTTOM_EXIT_RIGHT_LANDINGS,
        )
        for branch in branches
        for path in branch.continuation_resolved_paths
        for edge in path
        if edge.source == fork_id and edge.target in branch.landing_port_ids
    )
    if {item.branch_id for item in result} != {branch.id for branch in branches}:
        return ()
    emitted_by_branch: dict[FanBranchPlanId, set[str]] = defaultdict(set)
    for item in result:
        emitted_by_branch[item.branch_id].add(item.edge.line_id)
    if any(
        emitted_by_branch.get(branch.id, set()) != set(branch.line_ids)
        for branch in branches
    ):
        return ()
    emitted_lines = tuple(item.edge.line_id for item in result)
    if len(emitted_lines) != len(set(emitted_lines)) or set(emitted_lines) != set(
        offset_line_order
    ):
        return ()
    return result


def _apply_screen_offset_assignments(
    graph: MetroGraph,
    branches: Sequence[FanBranchPlan],
    route_emissions: Sequence[FanRouteEmission],
    exit_port_ids: Sequence[str],
    owned_station_ids: Sequence[str],
    carriers: Sequence[FanOffsetCarrier],
    line_priority: Mapping[str, int],
) -> tuple[FanOffsetCarrier, ...]:
    """Freeze exact source-side slots for the stacked RIGHT-landing emitter."""
    exit_port_id = _bottom_exit_source_port_id(graph, exit_port_ids)
    if not route_emissions or exit_port_id is None:
        return tuple(carriers)
    # A BOTTOM-exit fold into a RIGHT entry stores the receiving horizontal
    # section's lanes reflected. Earlier landing branches take the leftmost
    # descent block; lines within each block follow that reflected seam order.
    ordered_lines = tuple(
        line_id
        for branch in sorted(branches, key=lambda item: item.landing_rank)
        for line_id in sorted(
            branch.line_ids,
            key=lambda item: line_priority.get(item, len(line_priority)),
            reverse=True,
        )
    )
    if len(set(ordered_lines)) != len(ordered_lines):
        return tuple(carriers)
    screen_slots = {
        line_id: len(ordered_lines) - rank - 1
        for rank, line_id in enumerate(ordered_lines)
    }

    assignments: dict[str, dict[str, int]] = {
        carrier.station_id: {
            assignment.line_id: assignment.slot for assignment in carrier.assignments
        }
        for carrier in carriers
    }
    source_section_id = graph.ports[exit_port_id].section_id
    for station_id in owned_station_ids:
        station = graph.stations.get(station_id)
        if station is None:
            continue
        if station.section_id != source_section_id and station_id not in {
            exit_port_id,
            route_emissions[0].edge.source,
        }:
            continue
        present_lines = set(graph.station_lines(station_id))
        station_assignments = assignments.setdefault(station_id, {})
        for line_id in ordered_lines:
            if line_id in present_lines:
                station_assignments[line_id] = screen_slots[line_id]

    return tuple(
        FanOffsetCarrier(
            station_id=station_id,
            assignments=tuple(
                FanOffsetAssignment(line_id, slot)
                for line_id, slot in line_assignments.items()
            ),
        )
        for station_id, line_assignments in assignments.items()
        if line_assignments
    )


def _apply_solo_branch_offset_assignments(
    graph: MetroGraph,
    branches: Sequence[FanBranchPlan],
    fork_id: str,
    carriers: Sequence[FanOffsetCarrier],
) -> tuple[FanOffsetCarrier, ...]:
    """Freeze trunk-slot assignments for single-line branch stations."""
    assignments: dict[str, dict[str, int]] = {
        carrier.station_id: {
            assignment.line_id: assignment.slot for assignment in carrier.assignments
        }
        for carrier in carriers
    }
    fork_assignments = assignments.get(fork_id, {})
    for branch in branches:
        if len(branch.line_ids) != 1:
            continue
        line_id = branch.line_ids[0]
        if fork_assignments.get(line_id) != 0:
            continue
        for station_id in _fan_branch_solo_station_ids(graph, branch):
            assignments.setdefault(station_id, {})[line_id] = 0

    return tuple(
        FanOffsetCarrier(
            station_id=station_id,
            assignments=tuple(
                FanOffsetAssignment(line_id, slot)
                for line_id, slot in line_assignments.items()
            ),
        )
        for station_id, line_assignments in assignments.items()
    )


def _layout_section_id(graph: MetroGraph, fork_id: str) -> str | None:
    port = graph.ports.get(fork_id)
    if port is not None:
        section = graph.sections.get(port.section_id)
        if section is None:
            return None
        if port.side not in flow_port_sides(section.direction):
            return None
        return port.section_id
    return graph.section_for_station(fork_id)


def _centreline_port_ids(
    graph: MetroGraph,
    direction: FlowDirection | None,
    layout_section_id: str | None,
    port_ids: Sequence[str],
) -> tuple[str, ...]:
    """Freeze boundary ports that continue one fan's local centreline."""
    layout_section = graph.sections.get(layout_section_id or "")
    if direction is None or layout_section is None:
        return ()
    fan_is_horizontal = not lanes_run_along_x(direction)
    layout_column, layout_row = _grid_position(graph, layout_section.id)
    result: list[str] = []
    for port_id in port_ids:
        port = graph.ports.get(port_id)
        section = graph.sections.get(port.section_id) if port is not None else None
        if port is None or section is None:
            continue
        neighbour_section_ids = {
            neighbour.section_id
            for edge in (*graph.edges_to(port_id), *graph.edges_from(port_id))
            for neighbour_id in (
                edge.source if edge.target == port_id else edge.target,
            )
            if (neighbour := graph.stations.get(neighbour_id)) is not None
            and neighbour.section_id is not None
            and neighbour.section_id != section.id
        }
        has_perpendicular_neighbour = any(
            (not lanes_run_along_x(neighbour_section.direction)) != fan_is_horizontal
            for neighbour_id in neighbour_section_ids
            if (neighbour_section := graph.sections.get(neighbour_id)) is not None
        )
        if (
            (not lanes_run_along_x(section.direction)) != fan_is_horizontal
            or port.side not in flow_port_sides(section.direction)
            or has_perpendicular_neighbour
            or (
                _grid_position(graph, section.id)[1] != layout_row
                if fan_is_horizontal
                else _grid_position(graph, section.id)[0] != layout_column
            )
        ):
            continue
        result.append(port_id)
    return tuple(dict.fromkeys(result))


def _centreline_anchor(
    graph: MetroGraph,
    *,
    direction: FlowDirection | None,
    frame: AxisFrame | None,
    fork_id: str,
    layout_section_id: str | None,
    branches: Sequence[FanBranchPlan],
    entry_port_ids: Sequence[str],
    exit_port_ids: Sequence[str],
    local_frame_anchor: tuple[str, float | None] | None,
) -> FanCentrelineAnchor | None:
    """Freeze the source of one fan's settled absolute centreline."""
    layout_section = graph.sections.get(layout_section_id or "")
    if frame is not None and direction is not None and layout_section is not None:
        horizontal = not lanes_run_along_x(direction)
        candidates: list[tuple[float, str]] = []
        for port_id in (*entry_port_ids, *exit_port_ids):
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
                distance = (
                    layout_section.grid_col - section.grid_col
                ) * frame.primary_sign
            else:
                same_strip = section.grid_col == layout_section.grid_col
                distance = (
                    layout_section.grid_row - section.grid_row
                ) * frame.primary_sign
            if same_strip and distance > 0:
                candidates.append((distance, port_id))
        if candidates:
            return FanCentrelineAnchor(min(candidates)[1])

        local_trunks = tuple(
            branch
            for branch in branches
            if branch.is_trunk_continuation
            and branch.lane_station_ids
            and not branch.landing_port_ids
        )
        if len(local_trunks) == 1 and fork_id in graph.stations:
            return FanCentrelineAnchor(fork_id)

        flow_sides = flow_port_sides(direction)
        local_ports = [
            port_id
            for port_id in (*entry_port_ids, *exit_port_ids)
            if (port := graph.ports.get(port_id)) is not None
            and port.section_id == layout_section.id
            and port.side in flow_sides
        ]
        local_ports = list(dict.fromkeys(local_ports))
        local_ports.sort(key=lambda port_id: not graph.ports[port_id].is_entry)
        if local_ports:
            return FanCentrelineAnchor(local_ports[0])
        if fork_id in graph.stations:
            return FanCentrelineAnchor(fork_id)

    if local_frame_anchor is None or local_frame_anchor[1] is None:
        return None
    return FanCentrelineAnchor(
        station_id=local_frame_anchor[0],
        lane_offset=local_frame_anchor[1],
    )


def _lane_station_ids(
    graph: MetroGraph,
    paths: Iterable[tuple[ResolvedEdge, ...]],
    *,
    section_id: str | None,
    fork_id: str,
    join_id: str | None,
) -> tuple[str, ...]:
    if section_id is None:
        return ()
    station_ids: list[str] = []
    for path in paths:
        nodes = _path_nodes(path)
        for index, station_id in enumerate(nodes):
            if station_id == join_id:
                break
            if index > 0:
                predecessor_id = nodes[index - 1]
                incoming_sources = {edge.source for edge in graph.edges_to(station_id)}
                if incoming_sources.difference({predecessor_id}):
                    break
            if (
                station_id != fork_id
                and station_id not in graph.ports
                and station_id not in graph.junction_ids
                and graph.section_for_station(station_id) == section_id
                and station_id not in station_ids
            ):
                station_ids.append(station_id)
    return tuple(station_ids)


def _uncontested_local_terminal_branch_ids(
    graph: MetroGraph,
    node_paths: Sequence[tuple[str, ...]],
    branches: Sequence[FanBranchPlan],
    incoming: Mapping[str, tuple[str, ...]],
    layout_section_id: str | None,
) -> tuple[FanBranchPlanId, ...]:
    """Return local terminal branches that do not enter another merge frame."""
    if layout_section_id is None:
        return ()
    result: list[FanBranchPlanId] = []
    for path, branch in zip(node_paths, branches, strict=True):
        if (
            not branch.terminal
            or branch.landing_port_ids
            or not branch.lane_station_ids
        ):
            continue
        if any(
            graph.section_for_station(station_id) != layout_section_id
            for station_id in path[1:]
        ):
            continue
        if any(
            set(incoming.get(station_id, ())).difference({predecessor_id})
            for predecessor_id, station_id in zip(path, path[1:])
        ):
            continue
        result.append(branch.id)
    return tuple(result)


def _handoff_ids(
    topology: FanTopologyQuery, edge_ids: tuple[ConnectorId, ...]
) -> tuple[tuple[BundleId, ...], tuple[ConvergenceId, ...]]:
    bundles: list[BundleId] = []
    for edge_id in edge_ids:
        try:
            bundle_id = topology.connector(edge_id).bundle_id
        except KeyError:
            continue
        if bundle_id not in bundles:
            bundles.append(bundle_id)
    convergences: list[ConvergenceId] = []
    for view in topology.convergences:
        group = view.group
        if set(group.connector_ids).intersection(edge_ids):
            convergences.append(group.id)
    return tuple(bundles), tuple(convergences)


def _legacy(plan: FanPlan, reason: str) -> FanPlan:
    branches = tuple(
        replace(
            branch,
            lane_station_ids=(),
            lane_offset=None,
            diagonal_runway=None,
        )
        for branch in plan.branches
    )
    return replace(
        plan,
        branches=branches,
        frame=None,
        entry_runway=None,
        exit_runway=None,
        centreline_reference_id=None,
        demand_ids=(),
        offset_carriers=(),
        route_emissions=(),
        centreline_port_ids=(),
        centreline_station_ids=(),
        centreline_anchor=None,
        local_frame_anchor_station_id=None,
        local_frame_anchor_offset=None,
        appearance_centreline_branch_id=None,
        appearance_lane_pitch=None,
        disposition=FanPlanDisposition.LEGACY,
        legacy_reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _FanPlanningContext:
    graph: MetroGraph
    topology: FanTopologyQuery
    adjacency: Mapping[str, tuple[str, ...]]
    incoming: Mapping[str, tuple[str, ...]]
    bundles: Mapping[tuple[str, str], tuple[AuthoredEdgeFact, ...]]
    ranks: Mapping[str, int]
    x_spacing: float
    y_spacing: float
    minimum_runway: float


@dataclass(frozen=True, slots=True)
class _RecognisedFan:
    source_id: str
    branch_targets: tuple[str, ...]
    lead_fact_groups: tuple[tuple[AuthoredEdgeFact, ...], ...]
    lead_paths: tuple[tuple[tuple[ResolvedEdge, ...], ...], ...]
    all_lead_paths: tuple[tuple[ResolvedEdge, ...], ...]
    prefix: tuple[str, ...]
    fork_id: str
    reason: str | None
    authored_join: str | None
    node_paths: tuple[tuple[str, ...], ...]
    structural_trunk_rank: int | None
    continuation_facts: tuple[tuple[AuthoredEdgeFact, ...], ...]
    extra_facts: tuple[tuple[AuthoredEdgeFact, ...], ...]
    final_paths: tuple[tuple[ResolvedEdge, ...], ...]
    suffix: tuple[str, ...]
    join_id: str | None


def _recognise_fan(
    ctx: _FanPlanningContext,
    source_id: str,
    branch_targets: tuple[str, ...],
) -> _RecognisedFan:
    """Recognise complete authored and resolved membership without geometry."""
    topology = ctx.topology
    adjacency = ctx.adjacency
    bundles = ctx.bundles
    lead_fact_groups = tuple(bundles[(source_id, target)] for target in branch_targets)
    lead_paths = tuple(_paths_for(topology, facts) for facts in lead_fact_groups)
    all_lead_paths = tuple(path for paths in lead_paths for path in paths)
    complete_leads = all(paths and all(path for path in paths) for paths in lead_paths)
    prefix = _common_prefix_nodes(all_lead_paths) if complete_leads else (source_id,)
    fork_id = prefix[-1] if prefix else source_id
    reason = (
        "missing-resolved-member-path"
        if not complete_leads
        else None
        if prefix
        else "ambiguous-resolved-fork"
    )

    authored_join = _nearest_common_join(adjacency, branch_targets, ctx.ranks)
    node_paths: list[tuple[str, ...]] = []
    if authored_join is not None:
        for target in branch_targets:
            path = _unique_path_to_join(adjacency, target, authored_join)
            if path is None:
                reason = reason or "ambiguous-branch-to-join"
                path = _linear_path(adjacency, target)
            node_paths.append((source_id, *path))
    else:
        node_paths = [
            (source_id, *_linear_path(adjacency, target)) for target in branch_targets
        ]
    extended_branch_ranks = tuple(
        rank for rank, path in enumerate(node_paths) if len(path) > 2
    )
    structural_trunk_rank = (
        extended_branch_ranks[0]
        if authored_join is None and len(extended_branch_ranks) == 1
        else None
    )

    selected_continuations = tuple(
        _facts_for_node_path(
            path,
            bundles,
            frozenset(fact.key.line_id for fact in lead_facts),
        )
        for path, lead_facts in zip(node_paths, lead_fact_groups, strict=True)
    )
    if any(facts is None for facts in selected_continuations):
        reason = reason or "unsupported-branch-line-transition"
    continuation_facts = tuple(
        facts if facts is not None else _facts_for_node_path(path, bundles) or ()
        for path, facts in zip(node_paths, selected_continuations, strict=True)
    )
    extra_facts = tuple(
        _extra_output_facts(path[1:], adjacency, bundles)
        if authored_join is not None
        else ()
        for path in node_paths
    )
    final_fact_groups = tuple(bundles[(path[-2], path[-1])] for path in node_paths)
    final_paths = tuple(
        path for facts in final_fact_groups for path in _paths_for(topology, facts)
    )
    suffix = _common_suffix_nodes(final_paths) if authored_join is not None else ()
    join_id = suffix[0] if suffix else None
    if authored_join is not None and join_id is None:
        reason = reason or "ambiguous-resolved-join"

    return _RecognisedFan(
        source_id=source_id,
        branch_targets=branch_targets,
        lead_fact_groups=lead_fact_groups,
        lead_paths=lead_paths,
        all_lead_paths=all_lead_paths,
        prefix=prefix,
        fork_id=fork_id,
        reason=reason,
        authored_join=authored_join,
        node_paths=tuple(node_paths),
        structural_trunk_rank=structural_trunk_rank,
        continuation_facts=continuation_facts,
        extra_facts=extra_facts,
        final_paths=final_paths,
        suffix=suffix,
        join_id=join_id,
    )


def _build_candidate(
    ctx: _FanPlanningContext,
    source_id: str,
    branch_targets: tuple[str, ...],
) -> FanPlan:
    recognised = _recognise_fan(ctx, source_id, branch_targets)
    graph = ctx.graph
    topology = ctx.topology
    adjacency = ctx.adjacency
    incoming = ctx.incoming
    bundles = ctx.bundles
    minimum_runway = ctx.minimum_runway
    lead_fact_groups = recognised.lead_fact_groups
    lead_paths = recognised.lead_paths
    all_lead_paths = recognised.all_lead_paths
    prefix = recognised.prefix
    fork_id = recognised.fork_id
    reason = recognised.reason
    authored_join = recognised.authored_join
    node_paths = recognised.node_paths
    structural_trunk_rank = recognised.structural_trunk_rank
    continuation_facts = recognised.continuation_facts
    extra_facts = recognised.extra_facts
    final_paths = recognised.final_paths
    suffix = recognised.suffix
    join_id = recognised.join_id

    appearance_policy = FanAppearancePolicy(graph.diamond_style)

    direction = _direction_for_fork(graph, fork_id, source_id, lead_fact_groups[0])
    if direction is None:
        reason = reason or "unsupported-fan-direction"
    lane_pitch = (
        AxisFrame.for_direction(direction, ctx.x_spacing, ctx.y_spacing).secondary.step
        if direction is not None
        else ctx.y_spacing
    )
    offsets = symmetric_lane_offsets(len(branch_targets), lane_pitch)
    layout_section_id = _layout_section_id(graph, fork_id)
    if layout_section_id is not None and any(
        station_id != authored_join
        and graph.section_for_station(station_id) == layout_section_id
        and set(incoming.get(station_id, ())).difference({predecessor_id})
        for path in node_paths
        for predecessor_id, station_id in zip(path, path[1:])
    ):
        reason = reason or "local-layout-has-foreign-owner"
    branch_plans: list[FanBranchPlan] = []
    all_member_facts: list[AuthoredEdgeFact] = []
    all_raw_paths: list[tuple[ResolvedEdge, ...]] = []
    for rank, (node_path, facts, outputs, branch_lead_paths) in enumerate(
        zip(node_paths, continuation_facts, extra_facts, lead_paths, strict=True)
    ):
        raw_continuation = _paths_for(topology, facts)
        raw_outputs = _paths_for(topology, outputs)
        if not raw_continuation or any(not path for path in raw_continuation):
            reason = reason or "missing-resolved-member-path"
        if outputs and (not raw_outputs or any(not path for path in raw_outputs)):
            reason = reason or "missing-resolved-extra-output-path"
        branch_prefix = _common_prefix_nodes(branch_lead_paths)
        root_id = (
            branch_prefix[len(prefix)]
            if prefix and len(branch_prefix) > len(prefix)
            else node_path[1]
        )
        if authored_join is not None:
            tail_id = join_id or node_path[-1]
        else:
            tail_paths = _paths_for(topology, bundles[(node_path[-2], node_path[-1])])
            tails = {path[-1].target for path in tail_paths if path}
            tail_id = next(iter(tails)) if len(tails) == 1 else node_path[-1]
            if len(tails) != 1:
                reason = reason or "ambiguous-resolved-branch-tail"
        trimmed = tuple(
            _trim_member_path(path, fork_id, join_id) for path in raw_continuation
        )
        if any(not path for path in trimmed):
            reason = reason or "empty-resolved-member-path"
        lines = cast(
            tuple[str, ...],
            _ordered_unique(fact.key.line_id for fact in (*facts, *outputs)),
        )
        branch_id = FanBranchPlanId(
            semantic_route_id("fan-branch", source_id, *(fact.id for fact in facts))
        )
        terminal = authored_join is None and not adjacency.get(node_path[-1], ())
        branch_plans.append(
            FanBranchPlan(
                id=branch_id,
                rank=rank,
                landing_rank=rank,
                opening_rank=rank,
                root_station_id=root_id,
                tail_station_id=tail_id,
                continuation_edge_ids=tuple(fact.id for fact in facts),
                continuation_resolved_paths=trimmed,
                line_ids=lines,
                extra_output_edge_ids=tuple(fact.id for fact in outputs),
                extra_output_resolved_paths=raw_outputs,
                landing_port_ids=_port_ids(graph, trimmed)[0],
                lane_station_ids=_lane_station_ids(
                    graph,
                    (*trimmed, *raw_outputs),
                    section_id=layout_section_id,
                    fork_id=fork_id,
                    join_id=join_id,
                ),
                is_trunk_continuation=any(
                    graph.ports[port_id].section_id == layout_section_id
                    for port_id in _port_ids(graph, raw_continuation)[1]
                )
                or rank == structural_trunk_rank,
                terminal=terminal,
                lane_offset=offsets[rank],
                diagonal_runway=max(minimum_runway, abs(offsets[rank])),
            )
        )
        all_member_facts.extend((*facts, *outputs))
        all_raw_paths.extend((*raw_continuation, *raw_outputs))

    def landing_key(branch: FanBranchPlan) -> tuple[int, int, int]:
        positions = [
            (row, column)
            for port_id in branch.landing_port_ids
            if (port := graph.ports.get(port_id)) is not None
            and (section := graph.sections.get(port.section_id)) is not None
            for column, row in (_grid_position(graph, section.id),)
            if row >= 0 and column >= 0
        ]
        if not positions:
            return len(graph.sections), len(graph.sections), branch.rank
        row, column = min(positions)
        return row, column, branch.rank

    landing_order = {
        branch.id: rank
        for rank, branch in enumerate(sorted(branch_plans, key=landing_key))
    }
    branch_plans = [
        replace(
            branch,
            landing_rank=landing_order[branch.id],
            diagonal_runway=max(
                branch.diagonal_runway or minimum_runway,
                minimum_runway + landing_order[branch.id] * lane_pitch,
            ),
        )
        for branch in branch_plans
    ]
    if fork_id in graph.junction_ids:
        peel_order = fanout_divergence_peel_order(
            graph,
            fork_id,
            {line_id: rank for rank, line_id in enumerate(graph.lines)},
            cast(RouteTopologyQuery, topology),
        )
        branch_by_line = {
            branch.line_ids[0]: branch
            for branch in branch_plans
            if len(branch.line_ids) == 1
        }
        if (
            peel_order is not None
            and len(branch_by_line) == len(branch_plans) == len(peel_order)
            and set(peel_order) == set(branch_by_line)
        ):
            opening_order = {
                branch_by_line[line_id].id: rank
                for rank, line_id in enumerate(peel_order)
            }
            branch_plans = [
                replace(branch, opening_rank=opening_order[branch.id])
                for branch in branch_plans
            ]
    local_terminal_ids = _uncontested_local_terminal_branch_ids(
        graph,
        node_paths,
        branch_plans,
        incoming,
        layout_section_id,
    )
    if len(local_terminal_ids) == 1:
        local_terminal_id = local_terminal_ids[0]
        branch_plans = [
            replace(
                branch,
                is_trunk_continuation=branch.id == local_terminal_id,
            )
            for branch in branch_plans
        ]
    appearance_centreline_branch_id = _appearance_centreline_branch_id(
        branch_plans,
        appearance_policy,
        structural_trunk_rank,
    )
    lane_offsets = fan_lane_offsets(
        branch_plans,
        appearance_policy,
        lane_pitch,
        appearance_centreline_branch_id,
    )
    branch_plans = [
        replace(
            branch,
            lane_offset=lane_offset,
            diagonal_runway=max(
                minimum_runway,
                branch.diagonal_runway or 0.0,
                abs(lane_offset),
            ),
        )
        for branch, lane_offset in zip(branch_plans, lane_offsets, strict=True)
    ]

    frame = (
        AxisFrame.for_direction(direction, ctx.x_spacing, ctx.y_spacing)
        if direction is not None
        else None
    )
    if frame is not None:
        required_pitch = vertical_fan_label_lane_pitch(
            graph, branch_plans, frame, lane_pitch
        )
        if required_pitch > lane_pitch:
            scale = required_pitch / lane_pitch
            lane_pitch = required_pitch
            branch_plans = [
                replace(
                    branch,
                    lane_offset=(
                        branch.lane_offset * scale
                        if branch.lane_offset is not None
                        else None
                    ),
                    diagonal_runway=max(
                        minimum_runway + branch.landing_rank * lane_pitch,
                        abs(branch.lane_offset * scale)
                        if branch.lane_offset is not None
                        else 0.0,
                    ),
                )
                for branch in branch_plans
            ]

    branch_line_sets = [set(branch.line_ids) for branch in branch_plans]
    all_shared_lines = set.intersection(*branch_line_sets)
    has_line_divergence = bool(set.union(*branch_line_sets) - all_shared_lines)
    has_layout_lanes = any(branch.lane_station_ids for branch in branch_plans)
    line_priority = {line_id: rank for rank, line_id in enumerate(graph.lines)}
    offset_line_order = (
        cast(
            tuple[str, ...],
            _ordered_unique(
                line_id
                for branch in sorted(
                    branch_plans,
                    key=lambda item: (
                        item.lane_offset
                        if has_layout_lanes and item.lane_offset is not None
                        else item.opening_rank
                    ),
                )
                for line_id in sorted(
                    branch.line_ids, key=lambda item: line_priority.get(item, 0)
                )
            ),
        )
        if has_line_divergence
        else ()
    )

    member_facts = tuple(dict.fromkeys(all_member_facts))
    member_ids = tuple(fact.id for fact in member_facts)
    branch_member_paths = tuple(
        path for branch in branch_plans for path in branch.resolved_paths
    )
    entry_seam_paths = (
        cast(
            tuple[tuple[ResolvedEdge, ...], ...],
            _ordered_unique(tuple(path[: len(prefix) - 1]) for path in all_lead_paths),
        )
        if len(prefix) > 1
        else ()
    )
    exit_seam_paths = (
        cast(
            tuple[tuple[ResolvedEdge, ...], ...],
            _ordered_unique(tuple(path[-(len(suffix) - 1) :]) for path in final_paths),
        )
        if len(suffix) > 1
        else ()
    )
    seam_edges = cast(
        tuple[ResolvedEdge, ...],
        _ordered_unique(
            edge for path in (*entry_seam_paths, *exit_seam_paths) for edge in path
        ),
    )
    member_paths = (*entry_seam_paths, *branch_member_paths, *exit_seam_paths)
    member_edges = cast(
        tuple[ResolvedEdge, ...],
        _ordered_unique(edge for path in member_paths for edge in path),
    )
    incoming_facts = tuple(
        fact
        for predecessor in incoming.get(source_id, ())
        for fact in bundles[(predecessor, source_id)]
        if fact.id not in set(member_ids)
    )
    exit_facts = (
        tuple(
            fact
            for target in adjacency.get(authored_join, ())
            for fact in bundles[(authored_join, target)]
            if fact.id not in set(member_ids)
        )
        if authored_join is not None
        else ()
    )
    entry_handoff_ids = tuple(fact.id for fact in incoming_facts)
    exit_handoff_ids = tuple(fact.id for fact in exit_facts)
    entry_handoff_paths = _paths_for(topology, incoming_facts)
    exit_handoff_paths = _paths_for(topology, exit_facts)
    offset_sign = 1
    entry_offset_carriers = _entry_offset_carriers(
        graph,
        entry_handoff_paths,
        offset_line_order,
        offset_sign,
    )
    handoff_paths = (*entry_handoff_paths, *exit_handoff_paths)
    entry_ports, exit_ports = _port_ids(graph, (*all_raw_paths, *handoff_paths))
    owned_stations = cast(
        tuple[str, ...],
        _ordered_unique(
            station_id
            for edge in member_edges
            for station_id in (edge.source, edge.target)
        ),
    )
    if fork_id not in owned_stations:
        owned_stations = (fork_id, *owned_stations)
    if join_id is not None and join_id not in owned_stations:
        owned_stations = (*owned_stations, join_id)
    plan_id = FanPlanId(semantic_route_id("fan-plan", source_id, *member_ids))
    reference_id = SharedReferenceId(semantic_route_id("fan-centreline", plan_id))
    demand_ids = (
        DemandId(semantic_route_id("fan-entry-runway", plan_id)),
        DemandId(semantic_route_id("fan-exit-runway", plan_id)),
        *(
            DemandId(semantic_route_id("fan-branch-runway", plan_id, branch.id))
            for branch in branch_plans
        ),
    )
    bundle_handoffs, convergence_handoffs = _handoff_ids(
        topology, (*member_ids, *entry_handoff_ids, *exit_handoff_ids)
    )
    trunk_follower_ids = _trunk_followers(
        graph,
        fork_id,
        join_id,
        (*all_lead_paths, *entry_handoff_paths),
        exit_handoff_paths,
    )
    fork_section_id = graph.section_for_station(fork_id)
    frame_port_ids = tuple(
        port_id
        for port_id in (*entry_ports, *exit_ports)
        if (port := graph.ports.get(port_id)) is not None
        and port.section_id == fork_section_id
    )
    offset_carriers = _offset_carriers(
        graph,
        branches=branch_plans,
        offset_line_order=offset_line_order,
        shared_paths=(
            *entry_seam_paths,
            *exit_seam_paths,
            *entry_handoff_paths,
            *exit_handoff_paths,
        ),
        shared_station_ids=(
            fork_id,
            join_id,
            *trunk_follower_ids,
            *frame_port_ids,
        ),
        upstream_carriers=entry_offset_carriers,
        offset_sign=offset_sign,
    )
    owned_stations = cast(
        tuple[str, ...],
        _ordered_unique(
            (
                *owned_stations,
                *trunk_follower_ids,
                *(carrier.station_id for carrier in offset_carriers),
            )
        ),
    )
    centreline_station_ids = (
        cast(
            tuple[str, ...],
            _ordered_unique(
                station_id
                for station_id in (fork_id, join_id, *trunk_follower_ids)
                if station_id is not None
                and station_id not in graph.ports
                and station_id not in graph.junction_ids
                and graph.section_for_station(station_id) == layout_section_id
            ),
        )
        if layout_section_id is not None
        else ()
    )
    layout_station_ids = (
        *centreline_station_ids,
        *(
            station_id
            for branch in branch_plans
            for station_id in branch.lane_station_ids
        ),
    )
    if len(set(layout_station_ids)) != len(layout_station_ids):
        reason = reason or "overlapping-branch-lane-ownership"
    if layout_station_ids and frame is not None and frame.secondary_sign < 0:
        offset_carriers = tuple(
            replace(
                carrier,
                assignments=tuple(
                    replace(assignment, slot=-assignment.slot)
                    for assignment in carrier.assignments
                ),
            )
            for carrier in offset_carriers
        )
    if any(graph.station_is_rail(station_id) for station_id in owned_stations):
        reason = reason or "rail-layout-owns-fan-geometry"
    if any(
        station is not None and station.off_track
        for station_id in owned_stations
        if (station := graph.stations.get(station_id)) is not None
    ):
        reason = reason or "off-track-layout-owns-fan-geometry"
    route_emissions = (
        _route_emissions(
            graph,
            fork_id,
            branch_plans,
            exit_ports,
            offset_line_order,
        )
        if reason is None
        else ()
    )
    offset_carriers = _apply_screen_offset_assignments(
        graph,
        branch_plans,
        route_emissions,
        exit_ports,
        owned_stations,
        offset_carriers,
        line_priority,
    )
    offset_carriers = _apply_solo_branch_offset_assignments(
        graph,
        branch_plans,
        fork_id,
        offset_carriers,
    )
    if any(
        set(graph.station_lines(carrier.station_id)) != set(carrier.line_ids)
        for carrier in offset_carriers
    ):
        reason = reason or "offset-carrier-has-unowned-line"
    if (
        reason is None
        and authored_join is not None
        and appearance_policy is FanAppearancePolicy.STRAIGHT
    ):
        reason = "straight-diamond-layout-owns-geometry"
    # Same-line terminal and boundary arms have no semantic trunk identity.
    # The section allocator must choose their tracks before it sizes the box.
    if (
        reason is None
        and authored_join is None
        and appearance_policy is FanAppearancePolicy.STRAIGHT
        and len(local_terminal_ids) == 1
        and any(branch.landing_port_ids for branch in branch_plans)
        and len({frozenset(branch.line_ids) for branch in branch_plans}) == 1
    ):
        reason = "same-line-open-fan-layout-owns-geometry"
    local_anchor = next(
        ((station_id, 0.0) for station_id in centreline_station_ids), None
    )
    if local_anchor is None:
        local_anchor = next(
            (
                (branch.lane_station_ids[0], branch.lane_offset)
                for branch in sorted(
                    branch_plans,
                    key=lambda branch: (
                        abs(branch.lane_offset)
                        if branch.lane_offset is not None
                        else math.inf,
                        branch.rank,
                    ),
                )
                if branch.lane_station_ids and branch.lane_offset is not None
            ),
            None,
        )
    candidate_centreline_port_ids = (
        _centreline_port_ids(
            graph,
            direction,
            layout_section_id,
            (*entry_ports, *exit_ports),
        )
        if reason is None
        else ()
    )
    needs_centreline_anchor = bool(layout_station_ids or candidate_centreline_port_ids)
    candidate_centreline_anchor = (
        _centreline_anchor(
            graph,
            direction=direction,
            frame=frame,
            fork_id=fork_id,
            layout_section_id=layout_section_id,
            branches=branch_plans,
            entry_port_ids=entry_ports,
            exit_port_ids=exit_ports,
            local_frame_anchor=local_anchor,
        )
        if reason is None and needs_centreline_anchor
        else None
    )
    if (
        reason is None
        and needs_centreline_anchor
        and candidate_centreline_anchor is None
    ):
        reason = "missing-centreline-anchor"
    planned = reason is None
    if not planned:
        route_emissions = ()
    centreline_port_ids = candidate_centreline_port_ids if planned else ()
    owned_stations = cast(
        tuple[str, ...],
        _ordered_unique((*owned_stations, *centreline_port_ids)),
    )
    plan = FanPlan(
        id=plan_id,
        authored_source_id=source_id,
        authored_join_station_id=authored_join,
        fork_station_id=fork_id,
        direction=direction,
        join_station_id=join_id,
        appearance_policy=appearance_policy,
        appearance_centreline_branch_id=(
            appearance_centreline_branch_id if planned else None
        ),
        appearance_lane_pitch=lane_pitch if planned else None,
        branches=(
            tuple(branch_plans)
            if planned
            else tuple(
                replace(
                    branch,
                    lane_station_ids=(),
                    lane_offset=None,
                    diagonal_runway=None,
                )
                for branch in branch_plans
            )
        ),
        offset_line_order=offset_line_order,
        authored_edge_ids=member_ids,
        resolved_member_paths=member_paths,
        resolved_member_edges=member_edges,
        entry_seam_paths=entry_seam_paths,
        exit_seam_paths=exit_seam_paths,
        resolved_seam_edges=seam_edges,
        entry_handoff_edge_ids=entry_handoff_ids,
        exit_handoff_edge_ids=exit_handoff_ids,
        entry_handoff_paths=entry_handoff_paths,
        exit_handoff_paths=exit_handoff_paths,
        offset_carriers=offset_carriers if planned else (),
        route_emissions=route_emissions,
        centreline_port_ids=centreline_port_ids,
        entry_port_ids=entry_ports,
        exit_port_ids=exit_ports,
        trunk_follower_ids=trunk_follower_ids,
        entry_runway=minimum_runway if planned else None,
        exit_runway=minimum_runway if planned else None,
        centreline_reference_id=reference_id if planned else None,
        demand_ids=demand_ids if planned else (),
        bundle_handoff_ids=bundle_handoffs,
        convergence_handoff_ids=convergence_handoffs,
        owned_station_ids=owned_stations,
        centreline_station_ids=centreline_station_ids if planned else (),
        centreline_anchor=candidate_centreline_anchor if planned else None,
        local_frame_anchor_station_id=(
            local_anchor[0] if planned and local_anchor is not None else None
        ),
        local_frame_anchor_offset=(
            local_anchor[1] if planned and local_anchor is not None else None
        ),
        frame=frame if planned else None,
        disposition=(
            FanPlanDisposition.PLANNED if planned else FanPlanDisposition.LEGACY
        ),
        legacy_reason=reason,
    )
    return plan


def _reject_overlaps(
    plans: tuple[FanPlan, ...], facts_by_id: Mapping[ConnectorId, AuthoredEdgeFact]
) -> tuple[FanPlan, ...]:
    subsumed: set[FanPlanId] = set()
    for inner in plans:
        lead_ids = {
            edge_id
            for edge_id in inner.authored_edge_ids
            if facts_by_id[edge_id].key.source == inner.authored_source_id
        }
        if any(
            inner.authored_source_id in outer.owned_station_ids
            and lead_ids.issubset(outer.authored_edge_ids)
            for outer in plans
            if outer.id != inner.id
        ):
            subsumed.add(inner.id)
    plans = tuple(plan for plan in plans if plan.id not in subsumed)
    conflicts: set[FanPlanId] = set()
    for index, left in enumerate(plans):
        left_authored = set(left.authored_edge_ids)
        left_resolved = set(left.resolved_member_edges)
        left_stations = set(left.owned_station_ids)
        for right in plans[index + 1 :]:
            if (
                left_authored.intersection(right.authored_edge_ids)
                or left_resolved.intersection(right.resolved_member_edges)
                or left_stations.intersection(right.owned_station_ids)
            ):
                conflicts.update((left.id, right.id))
    return tuple(
        _legacy(plan, "overlapping-fan-ownership") if plan.id in conflicts else plan
        for plan in plans
    )


def build_fan_plan_execution(
    graph: MetroGraph,
    topology: FanTopologyQuery,
    *,
    x_spacing: float,
    y_spacing: float,
    minimum_runway: float,
) -> FanPlanExecution:
    """Recognise every authored fan and plan each complete object atomically."""
    for name, spacing in (("x", x_spacing), ("y", y_spacing)):
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"fan {name}-spacing must be finite and positive")
    if not math.isfinite(minimum_runway) or minimum_runway <= 0:
        raise ValueError("fan minimum runway must be finite and positive")
    facts = _authored_edges(topology)
    adjacency, incoming, bundles = _adjacency(facts)
    ranks = _node_rank(facts)
    context = _FanPlanningContext(
        graph=graph,
        topology=topology,
        adjacency=adjacency,
        incoming=incoming,
        bundles=bundles,
        ranks=ranks,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        minimum_runway=minimum_runway,
    )
    plans = tuple(
        _build_candidate(context, source_id, targets)
        for source_id, targets in adjacency.items()
        if len(targets) >= 2
    )
    plans = _reject_overlaps(plans, {fact.id: fact for fact in facts})
    return FanPlanExecution(plans=plans, query=FanPlanQuery.build(plans))


def install_fan_plan_execution(graph: MetroGraph, execution: FanPlanExecution) -> None:
    """Publish one complete build for later layout and routing consumers."""
    graph.fan_plan_execution = execution


def validate_fan_route_emissions(
    graph: MetroGraph, routes: Sequence[RoutedPath]
) -> None:
    """Require every exclusive fan emission to tag exactly one final route."""
    expected = {
        emission.edge: (plan, emission)
        for plan in graph.fan_plans
        if plan.owns_geometry
        for emission in plan.route_emissions
    }
    consumed: dict[ResolvedEdge, int] = defaultdict(int)
    for route in routes:
        edge = ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        tagged = route.fan_plan_id is not None or route.fan_route_emitter is not None
        binding = expected.get(edge)
        if not tagged:
            continue
        if binding is None:
            raise RuntimeError(f"unclaimed fan route emission tagged {edge!r}")
        plan, emission = binding
        if (
            route.fan_plan_id != plan.id
            or route.fan_route_emitter != emission.emitter.value
        ):
            raise RuntimeError(
                f"planned fan {plan.id!s} route tag drifted for {edge!r}"
            )
        consumed[edge] += 1
    for edge, (plan, _emission) in expected.items():
        if consumed.get(edge, 0) != 1:
            raise RuntimeError(
                f"planned fan {plan.id!s} expected one consumed route for {edge!r}; "
                f"found {consumed.get(edge, 0)}"
            )
