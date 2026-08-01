"""Immutable semantic route systems observed at the routing boundary.

The records in this module describe ownership and emission coverage. They do
not select geometry. :class:`RoutePlanObserver` is a transient companion to the
production dispatcher: it copies scalar facts from the settled graph, records
the family selected for each resolved inter-section leg, and binds the final
route set without retaining graph objects.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, NewType, TypeAlias, TypeVar

from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.options import LineOrder
from nf_metro.parser.model import MetroGraph, PortSide, is_bypass_v
from nf_metro.parser.provenance import (
    ConnectorEndpointRole,
    DecisionOrigin,
    DecisionReason,
    EffectiveDecision,
    FoldThresholdSource,
    GridCell,
    LineOrderSource,
)
from nf_metro.parser.route_topology import (
    BundleId,
    ConnectorId,
    ConvergenceId,
    DivergenceId,
    EndpointGroupId,
    ResolvedEdge,
    RouteConnector,
    RouteTopology,
    RouteTopologyQuery,
    build_route_topology_query,
    semantic_route_id,
)

if TYPE_CHECKING:
    from nf_metro.layout.route_reservations import (
        RealisedRouteReservation,
        RouteReservation,
        RouteReservationDiagnostic,
        RouteReservationId,
    )
    from nf_metro.layout.routing.common import RoutedPath
    from nf_metro.layout.routing.context import _EdgeKey, _RoutingCtx


RouteSystemId = NewType("RouteSystemId", str)
EmissionMemberId = NewType("EmissionMemberId", str)
EmittedPathId = NewType("EmittedPathId", str)
RouteBranchId = NewType("RouteBranchId", str)
RouteFeederId = NewType("RouteFeederId", str)
SharedReferenceId = NewType("SharedReferenceId", str)
DemandId = NewType("DemandId", str)
_T = TypeVar("_T")


class CoordinateRegime(str, Enum):
    """Coordinate system used by a coordinate-bearing record."""

    SETTLED_GRID = "settled-grid"
    LAYOUT_CANVAS = "layout-canvas"


class EmissionRole(str, Enum):
    """Semantic role played by a physical resolved leg."""

    CONTINUATION = "continuation"
    PEEL_OFF = "peel-off"
    BYPASS = "bypass"
    TERMINAL = "terminal"


class BindingKind(str, Enum):
    """How an emission member is represented in the final route set."""

    EMITTED = "emitted"
    MERGE_SKIP = "merge-skip"
    COVERED_MERGE_HOP = "covered-merge-hop"
    UNROUTED = "unrouted"


class CoverageReason(str, Enum):
    """Why another emitted member completely represents a resolved leg."""

    MERGE_TRUNK_COVERS_ENTRY_HOP = "merge-trunk-covers-entry-hop"


class SharedReferenceKind(str, Enum):
    """Vocabulary for geometry shared by members of one route system."""

    CENTRELINE = "centreline"
    TRUNK = "trunk"
    BAND = "band"
    RUNWAY = "runway"
    ORDERED_TURNS = "ordered-turns"
    LANDING_SEQUENCE = "landing-sequence"


class DemandKind(str, Enum):
    """Kinds of symbolic space a later planning stage may reserve."""

    SPAN = "span"
    LANES = "lanes"
    RUNWAY = "runway"
    ORDERED_TURNS = "ordered-turns"
    KEEP_OUT = "keep-out"


class DemandAxis(str, Enum):
    X = "x"
    Y = "y"
    BOTH = "both"


class KeepOutClass(str, Enum):
    """Obstacle classes a symbolic allocation must clear."""

    SECTION = "section"
    HEADER = "header"
    LABEL = "label"
    MARKER = "marker"
    CANVAS = "canvas"


class ReservationDecisionKind(str, Enum):
    """Layout decision referenced by a reservation or symbolic demand."""

    SECTION_GRID = "section-grid"
    SECTION_DIRECTION = "section-direction"
    CONNECTOR_SIDE = "connector-side"
    FOLD_THRESHOLD = "fold-threshold"
    LANE_ORDER = "lane-order"


class ReservationDecisionSource(str, Enum):
    """Who supplied a reservation-affecting layout decision."""

    AUTHOR = "author"
    CALLER = "caller"
    INFERENCE = "inference"


@dataclass(frozen=True, slots=True)
class ReservationDecisionRef:
    """Typed reference to one existing effective layout decision."""

    kind: ReservationDecisionKind
    subject_id: str
    decision: ReservationEffectiveDecision
    role: ConnectorEndpointRole | None = None

    def __post_init__(self) -> None:
        endpoint = self.kind is ReservationDecisionKind.CONNECTOR_SIDE
        if endpoint != (self.role is not None):
            raise ValueError("only connector-side decisions have an endpoint role")
        value = self.decision.value
        if self.kind is ReservationDecisionKind.SECTION_GRID:
            valid_grid = (
                isinstance(value, tuple)
                and len(value) == 4
                and all(isinstance(item, int) for item in value)
            )
            if not valid_grid:
                raise ValueError("section-grid decision requires a four-integer value")
        elif self.kind is ReservationDecisionKind.FOLD_THRESHOLD:
            if not isinstance(value, int):
                raise ValueError("fold-threshold decision requires an integer value")
        elif self.kind is ReservationDecisionKind.CONNECTOR_SIDE:
            if not isinstance(value, PortSide):
                raise ValueError("connector-side decision requires a PortSide value")
        elif not isinstance(value, str):
            raise ValueError(f"{self.kind.value} decision requires a string value")

    @property
    def source(self) -> ReservationDecisionSource:
        if self.decision.reason in {
            DecisionReason.CALLER_FOLD_THRESHOLD,
            DecisionReason.CALLER_LINE_ORDER,
            DecisionReason.CALLER_COMMITMENT,
        }:
            return ReservationDecisionSource.CALLER
        if self.decision.origin is DecisionOrigin.AUTHORED:
            return ReservationDecisionSource.AUTHOR
        return ReservationDecisionSource.INFERENCE


ReservationEffectiveDecision: TypeAlias = (
    EffectiveDecision[GridCell]
    | EffectiveDecision[str]
    | EffectiveDecision[int]
    | EffectiveDecision[PortSide]
    | EffectiveDecision[LineOrder]
)


@dataclass(frozen=True, slots=True)
class GridSpan:
    """Inclusive complete grid extent for a symbolic claim."""

    min_column: int
    max_column: int
    min_row: int
    max_row: int
    coordinate_regime: CoordinateRegime = CoordinateRegime.SETTLED_GRID


@dataclass(frozen=True, slots=True)
class EndpointFact:
    """Settled scalar facts for one physical leg endpoint."""

    station_id: str
    section_id: str | None
    port_id: str | None
    side: PortSide | None
    column: int | None
    row: int | None
    coordinate_regime: CoordinateRegime


@dataclass(frozen=True, slots=True)
class ConnectorLegRef:
    """One connector path occurrence attributed to a physical resolved leg."""

    connector_id: ConnectorId
    path_rank: int
    leg_rank: int


@dataclass(frozen=True, slots=True)
class SectionDecisionFacts:
    section_id: str
    grid: EffectiveDecision[GridCell] | None
    direction: EffectiveDecision[str] | None


@dataclass(frozen=True, slots=True)
class ConnectorDecisionFacts:
    connector_id: ConnectorId
    exit_side: EffectiveDecision[PortSide] | None
    entry_side: EffectiveDecision[PortSide] | None


@dataclass(frozen=True, slots=True)
class LaneOrderFacts:
    policy: EffectiveDecision[LineOrder]
    source: LineOrderSource
    realised_line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutePlanProvenance:
    sections: tuple[SectionDecisionFacts, ...]
    connectors: tuple[ConnectorDecisionFacts, ...]
    fold_threshold: EffectiveDecision[int] | None
    fold_threshold_source: FoldThresholdSource
    lane_order: LaneOrderFacts


@dataclass(frozen=True, slots=True)
class ResolvedEndpointGroup:
    """One topology endpoint group and its resolved boundary port."""

    id: EndpointGroupId
    system_id: RouteSystemId
    role: ConnectorEndpointRole
    section_id: str
    side: PortSide
    port_id: str
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class RouteDivergence:
    """One topology divergence and its resolved fan-out junction."""

    id: DivergenceId
    system_id: RouteSystemId
    junction_id: str
    exit_group_id: EndpointGroupId
    entry_group_ids: tuple[EndpointGroupId, ...]
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class RouteConvergence:
    """One topology convergence and its resolved merge junction."""

    id: ConvergenceId
    system_id: RouteSystemId
    junction_id: str
    entry_group_id: EndpointGroupId
    source_junction_ids: tuple[str, ...]
    divergence_ids: tuple[DivergenceId, ...]
    connector_ids: tuple[ConnectorId, ...]
    line_id: str


@dataclass(frozen=True, slots=True)
class EmissionMember:
    """One unique physical resolved inter-section leg."""

    id: EmissionMemberId
    system_id: RouteSystemId
    source: EndpointFact
    target: EndpointFact
    line_id: str
    line_rank: int
    connector_ids: tuple[ConnectorId, ...]
    leg_refs: tuple[ConnectorLegRef, ...]
    bundle_ids: tuple[BundleId, ...]
    exit_group_ids: tuple[EndpointGroupId, ...]
    entry_group_ids: tuple[EndpointGroupId, ...]
    divergence_ids: tuple[DivergenceId, ...]
    convergence_ids: tuple[ConvergenceId, ...]
    roles: tuple[EmissionRole, ...]
    family_id: RouteFamilyId | None

    @property
    def edge(self) -> ResolvedEdge:
        """Return the scalar final-edge key represented by this member."""
        return ResolvedEdge(
            self.source.station_id, self.target.station_id, self.line_id
        )


@dataclass(frozen=True, slots=True)
class RouteBranch:
    id: RouteBranchId
    system_id: RouteSystemId
    divergence_id: DivergenceId
    entry_group_id: EndpointGroupId
    connector_ids: tuple[ConnectorId, ...]
    line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteFeeder:
    id: RouteFeederId
    system_id: RouteSystemId
    convergence_id: ConvergenceId
    divergence_id: DivergenceId
    connector_ids: tuple[ConnectorId, ...]
    line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SharedReference:
    """A shared geometry identity populated by its owning child planner."""

    id: SharedReferenceId
    system_id: RouteSystemId
    kind: SharedReferenceKind
    claimant_member_ids: tuple[EmissionMemberId, ...]
    coordinate_regime: CoordinateRegime
    provenance: tuple[ReservationDecisionRef, ...]


@dataclass(frozen=True, slots=True)
class SymbolicDemand:
    """A complete symbolic allocation claim with no absolute geometry."""

    id: DemandId
    system_id: RouteSystemId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    kind: DemandKind
    axis: DemandAxis
    span: GridSpan
    lane_count: int
    minimum_size: float | None
    minimum_size_regime: CoordinateRegime | None
    ordered_reference_ids: tuple[SharedReferenceId, ...]
    keep_out_classes: tuple[KeepOutClass, ...]
    provenance: tuple[ReservationDecisionRef, ...]

    def __post_init__(self) -> None:
        if (self.minimum_size is None) is not (self.minimum_size_regime is None):
            raise ValueError(
                "minimum_size and minimum_size_regime must be provided together"
            )


@dataclass(frozen=True, slots=True)
class RouteSystem:
    """One maximal semantically coupled authored connector component."""

    id: RouteSystemId
    connector_ids: tuple[ConnectorId, ...]
    line_ids: tuple[str, ...]
    bundle_ids: tuple[BundleId, ...]
    exit_group_ids: tuple[EndpointGroupId, ...]
    entry_group_ids: tuple[EndpointGroupId, ...]
    divergence_ids: tuple[DivergenceId, ...]
    convergence_ids: tuple[ConvergenceId, ...]
    member_ids: tuple[EmissionMemberId, ...]
    branch_ids: tuple[RouteBranchId, ...]
    feeder_ids: tuple[RouteFeederId, ...]
    shared_reference_ids: tuple[SharedReferenceId, ...]
    demand_ids: tuple[DemandId, ...]
    reservation_ids: tuple[RouteReservationId, ...]


@dataclass(frozen=True, slots=True)
class EmissionBinding:
    """Final observational binding for one emission member."""

    member_id: EmissionMemberId
    kind: BindingKind
    path_id: EmittedPathId | None = None
    path_rank: int | None = None
    covering_member_id: EmissionMemberId | None = None
    coverage_reason: CoverageReason | None = None

    def __post_init__(self) -> None:
        emitted = self.kind is BindingKind.EMITTED
        covered = self.kind in {
            BindingKind.MERGE_SKIP,
            BindingKind.COVERED_MERGE_HOP,
        }
        if emitted:
            valid = (
                self.path_id is not None
                and self.path_rank is not None
                and self.path_rank >= 0
                and self.covering_member_id is None
                and self.coverage_reason is None
            )
        elif covered:
            valid = (
                self.path_id is None
                and self.path_rank is None
                and self.covering_member_id is not None
                and self.coverage_reason is not None
            )
        else:
            valid = (
                self.path_id is None
                and self.path_rank is None
                and self.covering_member_id is None
                and self.coverage_reason is None
            )
        if not valid:
            raise ValueError(f"invalid {self.kind.value} emission binding")


@dataclass(frozen=True, slots=True)
class RoutePlanDiagnostic:
    member_id: EmissionMemberId | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RoutePlan:
    systems: tuple[RouteSystem, ...]
    endpoint_groups: tuple[ResolvedEndpointGroup, ...]
    divergences: tuple[RouteDivergence, ...]
    convergences: tuple[RouteConvergence, ...]
    members: tuple[EmissionMember, ...]
    branches: tuple[RouteBranch, ...]
    feeders: tuple[RouteFeeder, ...]
    shared_references: tuple[SharedReference, ...]
    demands: tuple[SymbolicDemand, ...]
    reservations: tuple[RouteReservation, ...]
    realised_reservations: tuple[RealisedRouteReservation, ...]
    reservation_diagnostics: tuple[RouteReservationDiagnostic, ...]
    bindings: tuple[EmissionBinding, ...]
    provenance: RoutePlanProvenance
    diagnostics: tuple[RoutePlanDiagnostic, ...] = ()


@dataclass(slots=True)
class RouteObservation:
    """Mutable route output paired with an immutable context-local plan."""

    routes: list[RoutedPath]
    plan: RoutePlan


def _ordered_unique(values: Iterable[_T]) -> tuple[_T, ...]:
    seen: set[_T] = set()
    result: list[_T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _inter_section_leg(graph: MetroGraph, edge: ResolvedEdge) -> bool:
    source = graph.stations.get(edge.source)
    target = graph.stations.get(edge.target)
    if source is None or target is None:
        return False
    junction_ids = graph.junction_ids
    return (source.is_port or edge.source in junction_ids) and (
        target.is_port or edge.target in junction_ids
    )


def _resolved_member_refs(
    graph: MetroGraph,
    topology: RouteTopology,
    query: RouteTopologyQuery,
) -> tuple[
    dict[ResolvedEdge, list[ConnectorLegRef]],
    tuple[ResolvedEdge, ...],
]:
    refs_by_edge: dict[ResolvedEdge, list[ConnectorLegRef]] = defaultdict(list)
    edge_order: list[ResolvedEdge] = []
    for connector in topology.connectors:
        for path_rank, path in enumerate(query.resolved_paths(connector.id)):
            for leg_rank, edge in enumerate(path):
                if not _inter_section_leg(graph, edge):
                    continue
                if edge not in refs_by_edge:
                    edge_order.append(edge)
                refs_by_edge[edge].append(
                    ConnectorLegRef(connector.id, path_rank, leg_rank)
                )
    return refs_by_edge, tuple(edge_order)


def _semantic_components(
    topology: RouteTopology,
    refs_by_edge: Mapping[ResolvedEdge, list[ConnectorLegRef]],
) -> tuple[tuple[ConnectorId, ...], ...]:
    ordered_ids = tuple(connector.id for connector in topology.connectors)
    parent = {connector_id: connector_id for connector_id in ordered_ids}
    rank = {connector_id: index for index, connector_id in enumerate(ordered_ids)}

    def root(connector_id: ConnectorId) -> ConnectorId:
        while parent[connector_id] != connector_id:
            parent[connector_id] = parent[parent[connector_id]]
            connector_id = parent[connector_id]
        return connector_id

    def join(connector_ids: tuple[ConnectorId, ...]) -> None:
        if not connector_ids:
            return
        winner = min((root(item) for item in connector_ids), key=rank.__getitem__)
        for connector_id in connector_ids:
            parent[root(connector_id)] = winner

    for records in (
        topology.bundles,
        topology.exit_groups,
        topology.entry_groups,
        topology.divergences,
        topology.convergences,
    ):
        for record in records:
            join(record.connector_ids)

    for refs in refs_by_edge.values():
        join(_ordered_unique(ref.connector_id for ref in refs))

    members: dict[ConnectorId, list[ConnectorId]] = defaultdict(list)
    for connector_id in ordered_ids:
        members[root(connector_id)].append(connector_id)
    return tuple(tuple(values) for values in members.values())


def _endpoint_fact(graph: MetroGraph, station_id: str) -> EndpointFact:
    station = graph.stations[station_id]
    port = graph.ports.get(station_id)
    section_id = port.section_id if port is not None else station.section_id
    section = graph.sections.get(section_id) if section_id is not None else None
    column = section.grid_col if section is not None and section.grid_col >= 0 else None
    row = section.grid_row if section is not None and section.grid_row >= 0 else None
    return EndpointFact(
        station_id=station_id,
        section_id=section_id,
        port_id=station_id if port is not None else None,
        side=port.side if port is not None else None,
        column=column,
        row=row,
        coordinate_regime=CoordinateRegime.SETTLED_GRID,
    )


def _plan_provenance(
    graph: MetroGraph, connectors: tuple[RouteConnector, ...]
) -> RoutePlanProvenance:
    provenance = graph.layout_provenance
    sections = tuple(
        SectionDecisionFacts(
            section_id,
            provenance.grid_decision(section_id),
            provenance.direction_decision(section_id),
        )
        for section_id in graph.sections
    )
    connector_facts = tuple(
        ConnectorDecisionFacts(
            connector.id,
            provenance.endpoint_decision(
                provenance.endpoint_key(connector.id, ConnectorEndpointRole.EXIT)
            ),
            provenance.endpoint_decision(
                provenance.endpoint_key(connector.id, ConnectorEndpointRole.ENTRY)
            ),
        )
        for connector in connectors
    )
    fold_source = (
        provenance.authored.fold_threshold.selected_source
        if provenance.authored is not None
        else FoldThresholdSource.DEFAULT
    )
    line_order = provenance.line_order_decision
    if line_order is None:
        raise ValueError("line-order provenance was not captured")
    line_source = (
        provenance.authored.line_order.selected_source
        if provenance.authored is not None
        else LineOrderSource.DEFAULT
    )
    return RoutePlanProvenance(
        sections,
        connector_facts,
        provenance.fold_threshold_decision,
        fold_source,
        LaneOrderFacts(
            line_order,
            line_source,
            tuple(graph.lines),
        ),
    )


@dataclass(slots=True)
class RoutePlanObserver:
    """Transient route-plan collector attached to one routing invocation."""

    graph: MetroGraph
    context: _RoutingCtx | None
    _family_by_edge: dict[_EdgeKey, RouteFamilyId] = field(default_factory=dict)
    _merge_skips: dict[_EdgeKey, _EdgeKey | None] = field(default_factory=dict)
    _covered_hops: dict[_EdgeKey, _EdgeKey | None] = field(default_factory=dict)

    def record_dispatch(self, edge: _EdgeKey, family_id: RouteFamilyId) -> None:
        self._family_by_edge[edge] = family_id

    def record_rail_routes(self, routes: Iterable[RoutedPath]) -> None:
        for route in routes:
            self._family_by_edge[
                (route.edge.source, route.edge.target, route.line_id)
            ] = RouteFamilyId.RAIL_INTER_SECTION

    def record_merge_skip(self, edge: _EdgeKey, covering_edge: _EdgeKey | None) -> None:
        self._merge_skips[edge] = covering_edge

    def covering_edge(self, edge: _EdgeKey) -> _EdgeKey | None:
        """Return the merge-trunk member that covers one entry hop."""
        if self.context is None:
            return None
        return _covering_edge(self.context, edge)

    def record_covered_merge_hops(
        self, records: tuple[tuple[_EdgeKey, _EdgeKey | None], ...]
    ) -> None:
        self._covered_hops.update(records)

    def finish(self, routes: list[RoutedPath]) -> RoutePlan:
        return _build_route_plan(self, routes)


def _covering_edge(context: _RoutingCtx, edge: _EdgeKey) -> _EdgeKey | None:
    source, _target, line_id = edge
    trunk_source = context.merge.trunk_source.get(source)
    if trunk_source is None:
        return None
    return trunk_source, source, line_id


def build_route_plan_observer(
    graph: MetroGraph, context: _RoutingCtx | None
) -> RoutePlanObserver:
    """Create one transient observer after settled routing context construction."""
    return RoutePlanObserver(graph, context)


def _member_roles(
    graph: MetroGraph,
    edge: ResolvedEdge,
    family: RouteFamilyId | None,
) -> tuple[EmissionRole, ...]:
    roles: set[EmissionRole] = set()
    if (
        family
        in {
            RouteFamilyId.BYPASS_FAMILY,
            RouteFamilyId.RIGHT_ENTRY_PLOUGH_BYPASS,
        }
        or is_bypass_v(edge.source)
        or is_bypass_v(edge.target)
    ):
        roles.add(EmissionRole.BYPASS)
    target_port = graph.ports.get(edge.target)
    if target_port is not None and target_port.is_entry:
        roles.add(EmissionRole.TERMINAL)
    return tuple(role for role in EmissionRole if role in roles)


@dataclass(slots=True)
class _ResolutionRecords:
    endpoint_groups: list[ResolvedEndpointGroup]
    divergences: list[RouteDivergence]
    convergences: list[RouteConvergence]
    exit_group_ids_by_system: dict[RouteSystemId, list[EndpointGroupId]]
    entry_group_ids_by_system: dict[RouteSystemId, list[EndpointGroupId]]
    divergence_ids_by_system: dict[RouteSystemId, list[DivergenceId]]
    convergence_ids_by_system: dict[RouteSystemId, list[ConvergenceId]]
    divergence_ids_by_connector: dict[ConnectorId, list[DivergenceId]]
    convergence_ids_by_connector: dict[ConnectorId, list[ConvergenceId]]


def _build_resolution_records(
    topology: RouteTopology,
    query: RouteTopologyQuery,
    system_for: Callable[[tuple[ConnectorId, ...]], RouteSystemId],
) -> _ResolutionRecords:
    endpoint_groups: list[ResolvedEndpointGroup] = []
    exit_group_ids_by_system: dict[RouteSystemId, list[EndpointGroupId]] = defaultdict(
        list
    )
    entry_group_ids_by_system: dict[RouteSystemId, list[EndpointGroupId]] = defaultdict(
        list
    )
    for role, groups in (
        (ConnectorEndpointRole.EXIT, topology.exit_groups),
        (ConnectorEndpointRole.ENTRY, topology.entry_groups),
    ):
        for endpoint_group in groups:
            system_id = system_for(endpoint_group.connector_ids)
            port_id = (
                query.exit_port(endpoint_group.id)
                if role is ConnectorEndpointRole.EXIT
                else query.entry_port(endpoint_group.id)
            )
            endpoint_groups.append(
                ResolvedEndpointGroup(
                    id=endpoint_group.id,
                    system_id=system_id,
                    role=role,
                    section_id=endpoint_group.section_id,
                    side=endpoint_group.side,
                    port_id=port_id,
                    connector_ids=endpoint_group.connector_ids,
                )
            )
            target = (
                exit_group_ids_by_system
                if role is ConnectorEndpointRole.EXIT
                else entry_group_ids_by_system
            )
            target[system_id].append(endpoint_group.id)

    divergences: list[RouteDivergence] = []
    divergence_ids_by_system: dict[RouteSystemId, list[DivergenceId]] = defaultdict(
        list
    )
    divergence_ids_by_connector: dict[ConnectorId, list[DivergenceId]] = defaultdict(
        list
    )
    for divergence_view in query.divergences:
        divergence_group = divergence_view.group
        system_id = system_for(divergence_group.connector_ids)
        divergences.append(
            RouteDivergence(
                id=divergence_group.id,
                system_id=system_id,
                junction_id=divergence_view.junction_id,
                exit_group_id=divergence_group.exit_group_id,
                entry_group_ids=divergence_group.entry_group_ids,
                connector_ids=divergence_group.connector_ids,
            )
        )
        divergence_ids_by_system[system_id].append(divergence_group.id)
        for connector_id in divergence_group.connector_ids:
            divergence_ids_by_connector[connector_id].append(divergence_group.id)

    convergences: list[RouteConvergence] = []
    convergence_ids_by_system: dict[RouteSystemId, list[ConvergenceId]] = defaultdict(
        list
    )
    convergence_ids_by_connector: dict[ConnectorId, list[ConvergenceId]] = defaultdict(
        list
    )
    for convergence_view in query.convergences:
        convergence_group = convergence_view.group
        system_id = system_for(convergence_group.connector_ids)
        convergences.append(
            RouteConvergence(
                id=convergence_group.id,
                system_id=system_id,
                junction_id=convergence_view.junction_id,
                entry_group_id=convergence_group.entry_group_id,
                source_junction_ids=convergence_view.source_junction_ids,
                divergence_ids=convergence_group.divergence_ids,
                connector_ids=convergence_group.connector_ids,
                line_id=convergence_group.line_id,
            )
        )
        convergence_ids_by_system[system_id].append(convergence_group.id)
        for connector_id in convergence_group.connector_ids:
            convergence_ids_by_connector[connector_id].append(convergence_group.id)

    return _ResolutionRecords(
        endpoint_groups,
        divergences,
        convergences,
        exit_group_ids_by_system,
        entry_group_ids_by_system,
        divergence_ids_by_system,
        convergence_ids_by_system,
        divergence_ids_by_connector,
        convergence_ids_by_connector,
    )


def _bind_member(
    observer: RoutePlanObserver,
    edge: ResolvedEdge,
    member_id: EmissionMemberId,
    route_ranks: list[int],
    member_id_by_edge: Mapping[ResolvedEdge, EmissionMemberId],
    family: RouteFamilyId | None,
) -> tuple[EmissionBinding, tuple[RoutePlanDiagnostic, ...]]:
    if len(route_ranks) == 1:
        rank = route_ranks[0]
        binding = EmissionBinding(
            member_id,
            BindingKind.EMITTED,
            EmittedPathId(semantic_route_id("emitted-path", member_id, rank)),
            rank,
        )
        if family is not None:
            return binding, ()
        return binding, (
            RoutePlanDiagnostic(
                member_id,
                "production-family",
                f"{edge.source}->{edge.target} ({edge.line_id}) emitted without "
                "an observed production family",
            ),
        )

    edge_key = (edge.source, edge.target, edge.line_id)
    suppression = None
    if not route_ranks and edge_key in observer._merge_skips:
        suppression = BindingKind.MERGE_SKIP, observer._merge_skips[edge_key]
    elif not route_ranks and edge_key in observer._covered_hops:
        suppression = BindingKind.COVERED_MERGE_HOP, observer._covered_hops[edge_key]
    if suppression is not None:
        kind, covering_edge = suppression
        covering_member_id = (
            member_id_by_edge.get(ResolvedEdge(*covering_edge))
            if covering_edge is not None
            else None
        )
        if covering_member_id is not None and covering_member_id != member_id:
            binding = EmissionBinding(
                member_id,
                kind,
                covering_member_id=covering_member_id,
                coverage_reason=CoverageReason.MERGE_TRUNK_COVERS_ENTRY_HOP,
            )
            if kind is not BindingKind.COVERED_MERGE_HOP or family is not None:
                return binding, ()
            return binding, (
                RoutePlanDiagnostic(
                    member_id,
                    "production-family",
                    f"{edge.source}->{edge.target} ({edge.line_id}) was removed "
                    "after dispatch without a recorded family",
                ),
            )
        return EmissionBinding(member_id, BindingKind.UNROUTED), (
            RoutePlanDiagnostic(
                member_id,
                "coverage-carrier",
                f"{edge.source}->{edge.target} ({edge.line_id}) has no resolved "
                "carrying emission member",
            ),
        )

    detail = "no final route" if not route_ranks else f"{len(route_ranks)} final routes"
    return EmissionBinding(member_id, BindingKind.UNROUTED), (
        RoutePlanDiagnostic(
            member_id,
            "emission-coverage",
            f"{edge.source}->{edge.target} ({edge.line_id}) has {detail}",
        ),
    )


def _build_route_plan(
    observer: RoutePlanObserver, routes: list[RoutedPath]
) -> RoutePlan:
    graph = observer.graph
    topology = graph.route_topology
    query = observer.context.topology if observer.context is not None else None
    if query is None:
        query = build_route_topology_query(graph)
    if topology is None or query is None:
        return RoutePlan(
            systems=(),
            endpoint_groups=(),
            divergences=(),
            convergences=(),
            members=(),
            branches=(),
            feeders=(),
            shared_references=(),
            demands=(),
            reservations=(),
            realised_reservations=(),
            reservation_diagnostics=(),
            bindings=(),
            provenance=_plan_provenance(graph, ()),
        )

    refs_by_edge, edge_order = _resolved_member_refs(graph, topology, query)
    components = _semantic_components(topology, refs_by_edge)
    system_by_connector: dict[ConnectorId, RouteSystemId] = {}
    ordered_system_ids: list[RouteSystemId] = []
    for connector_ids in components:
        system_id = RouteSystemId(semantic_route_id("route-system", *connector_ids))
        ordered_system_ids.append(system_id)
        for connector_id in connector_ids:
            system_by_connector[connector_id] = system_id

    def system_for(connector_ids: tuple[ConnectorId, ...]) -> RouteSystemId:
        if not connector_ids:
            raise ValueError("route-plan ownership record has no connectors")
        system_id = system_by_connector[connector_ids[0]]
        if any(system_by_connector[item] != system_id for item in connector_ids[1:]):
            raise ValueError("one topology record spans multiple route systems")
        return system_id

    bundle_ids_by_system: dict[RouteSystemId, list[BundleId]] = defaultdict(list)
    for bundle in topology.bundles:
        bundle_ids_by_system[system_for(bundle.connector_ids)].append(bundle.id)
    resolution = _build_resolution_records(topology, query, system_for)

    member_id_by_edge: dict[ResolvedEdge, EmissionMemberId] = {}
    for edge in edge_order:
        connector_ids = _ordered_unique(ref.connector_id for ref in refs_by_edge[edge])
        system_id = system_for(connector_ids)
        member_id_by_edge[edge] = EmissionMemberId(
            semantic_route_id(
                "emission-member", system_id, edge.source, edge.target, edge.line_id
            )
        )

    route_ranks: dict[ResolvedEdge, list[int]] = defaultdict(list)
    for path_rank, route in enumerate(routes):
        edge = ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        if edge in member_id_by_edge:
            route_ranks[edge].append(path_rank)

    line_rank = {line_id: rank for rank, line_id in enumerate(graph.lines)}
    diagnostics: list[RoutePlanDiagnostic] = []
    members: list[EmissionMember] = []
    member_ids_by_system: dict[RouteSystemId, list[EmissionMemberId]] = defaultdict(
        list
    )
    bindings: list[EmissionBinding] = []
    endpoint_facts: dict[str, EndpointFact] = {}
    for edge in edge_order:
        leg_refs = tuple(refs_by_edge[edge])
        connector_ids = _ordered_unique(ref.connector_id for ref in leg_refs)
        connectors = tuple(query.connector(item) for item in connector_ids)
        system_id = system_for(connector_ids)
        member_id = member_id_by_edge[edge]
        family = observer._family_by_edge.get(edge)
        ranks = route_ranks.get(edge, [])

        for station_id in (edge.source, edge.target):
            if station_id not in endpoint_facts:
                endpoint_facts[station_id] = _endpoint_fact(graph, station_id)

        members.append(
            EmissionMember(
                id=member_id,
                system_id=system_id,
                source=endpoint_facts[edge.source],
                target=endpoint_facts[edge.target],
                line_id=edge.line_id,
                line_rank=line_rank.get(edge.line_id, len(line_rank)),
                connector_ids=connector_ids,
                leg_refs=leg_refs,
                bundle_ids=_ordered_unique(item.bundle_id for item in connectors),
                exit_group_ids=_ordered_unique(
                    item.exit_group_id for item in connectors
                ),
                entry_group_ids=_ordered_unique(
                    item.entry_group_id for item in connectors
                ),
                divergence_ids=_ordered_unique(
                    item
                    for connector_id in connector_ids
                    for item in resolution.divergence_ids_by_connector[connector_id]
                ),
                convergence_ids=_ordered_unique(
                    item
                    for connector_id in connector_ids
                    for item in resolution.convergence_ids_by_connector[connector_id]
                ),
                roles=_member_roles(graph, edge, family),
                family_id=family,
            )
        )
        member_ids_by_system[system_id].append(member_id)
        binding, binding_diagnostics = _bind_member(
            observer,
            edge,
            member_id,
            ranks,
            member_id_by_edge,
            family,
        )
        bindings.append(binding)
        diagnostics.extend(binding_diagnostics)

    branches: list[RouteBranch] = []
    branch_ids_by_system: dict[RouteSystemId, list[RouteBranchId]] = defaultdict(list)
    for divergence in topology.divergences:
        connectors_by_entry: dict[EndpointGroupId, list[ConnectorId]] = defaultdict(
            list
        )
        for connector_id in divergence.connector_ids:
            entry_group_id = query.connector(connector_id).entry_group_id
            connectors_by_entry[entry_group_id].append(connector_id)
        for entry_group_id in divergence.entry_group_ids:
            connector_ids = tuple(connectors_by_entry[entry_group_id])
            system_id = system_for(connector_ids)
            branch_id = RouteBranchId(
                semantic_route_id(
                    "route-branch", system_id, divergence.id, entry_group_id
                )
            )
            branches.append(
                RouteBranch(
                    branch_id,
                    system_id,
                    divergence.id,
                    entry_group_id,
                    connector_ids,
                    _ordered_unique(
                        query.connector(item).line_id for item in connector_ids
                    ),
                )
            )
            branch_ids_by_system[system_id].append(branch_id)
    feeders: list[RouteFeeder] = []
    feeder_ids_by_system: dict[RouteSystemId, list[RouteFeederId]] = defaultdict(list)
    for convergence in topology.convergences:
        connectors_by_divergence: dict[DivergenceId, list[ConnectorId]] = defaultdict(
            list
        )
        convergence_divergences = set(convergence.divergence_ids)
        for connector_id in convergence.connector_ids:
            for divergence_id in resolution.divergence_ids_by_connector[connector_id]:
                if divergence_id in convergence_divergences:
                    connectors_by_divergence[divergence_id].append(connector_id)
        for divergence_id in convergence.divergence_ids:
            connector_ids = tuple(connectors_by_divergence[divergence_id])
            system_id = system_for(connector_ids)
            feeder_id = RouteFeederId(
                semantic_route_id(
                    "route-feeder",
                    system_id,
                    convergence.id,
                    divergence_id,
                )
            )
            feeders.append(
                RouteFeeder(
                    feeder_id,
                    system_id,
                    convergence.id,
                    divergence_id,
                    connector_ids,
                    _ordered_unique(
                        query.connector(item).line_id for item in connector_ids
                    ),
                )
            )
            feeder_ids_by_system[system_id].append(feeder_id)

    systems: list[RouteSystem] = []
    for system_id, connector_ids in zip(ordered_system_ids, components, strict=True):
        systems.append(
            RouteSystem(
                system_id,
                connector_ids,
                _ordered_unique(
                    query.connector(connector_id).line_id
                    for connector_id in connector_ids
                ),
                tuple(bundle_ids_by_system[system_id]),
                tuple(resolution.exit_group_ids_by_system[system_id]),
                tuple(resolution.entry_group_ids_by_system[system_id]),
                tuple(resolution.divergence_ids_by_system[system_id]),
                tuple(resolution.convergence_ids_by_system[system_id]),
                tuple(member_ids_by_system[system_id]),
                tuple(branch_ids_by_system[system_id]),
                tuple(feeder_ids_by_system[system_id]),
                (),
                (),
                (),
            )
        )

    plan = RoutePlan(
        systems=tuple(systems),
        endpoint_groups=tuple(resolution.endpoint_groups),
        divergences=tuple(resolution.divergences),
        convergences=tuple(resolution.convergences),
        members=tuple(members),
        branches=tuple(branches),
        feeders=tuple(feeders),
        shared_references=(),
        demands=(),
        reservations=(),
        realised_reservations=(),
        reservation_diagnostics=(),
        bindings=tuple(bindings),
        provenance=_plan_provenance(graph, topology.connectors),
        diagnostics=tuple(diagnostics),
    )
    from nf_metro.layout.route_reservations import attach_route_reservations

    return attach_route_reservations(
        plan,
        graph,
        routes,
        observer.context.station_offsets if observer.context is not None else None,
    )


@dataclass(frozen=True, slots=True)
class RoutePlanQuery:
    """Transient read-only indexes over canonical route-plan tuples."""

    plan: RoutePlan
    _endpoint_groups: Mapping[EndpointGroupId, ResolvedEndpointGroup]
    _divergences: Mapping[DivergenceId, RouteDivergence]
    _convergences: Mapping[ConvergenceId, RouteConvergence]
    _members: Mapping[EmissionMemberId, EmissionMember]
    _bindings: Mapping[EmissionMemberId, tuple[EmissionBinding, ...]]
    _shared_references: Mapping[SharedReferenceId, SharedReference]
    _demands: Mapping[DemandId, SymbolicDemand]
    _reservations: Mapping[RouteReservationId, RouteReservation]
    _realisations: Mapping[RouteReservationId, RealisedRouteReservation]
    _reservations_by_system: Mapping[RouteSystemId, tuple[RouteReservation, ...]]
    _reservations_by_member: Mapping[EmissionMemberId, tuple[RouteReservation, ...]]

    def endpoint_group(self, group_id: EndpointGroupId) -> ResolvedEndpointGroup:
        return self._endpoint_groups[group_id]

    def divergence(self, divergence_id: DivergenceId) -> RouteDivergence:
        return self._divergences[divergence_id]

    def convergence(self, convergence_id: ConvergenceId) -> RouteConvergence:
        return self._convergences[convergence_id]

    def member(self, member_id: EmissionMemberId) -> EmissionMember:
        return self._members[member_id]

    def bindings_for(self, member_id: EmissionMemberId) -> tuple[EmissionBinding, ...]:
        return self._bindings.get(member_id, ())

    def shared_reference(self, reference_id: SharedReferenceId) -> SharedReference:
        return self._shared_references[reference_id]

    def demand(self, demand_id: DemandId) -> SymbolicDemand:
        return self._demands[demand_id]

    def reservation(self, reservation_id: RouteReservationId) -> RouteReservation:
        return self._reservations[reservation_id]

    def realised_reservation(
        self, reservation_id: RouteReservationId
    ) -> RealisedRouteReservation | None:
        return self._realisations.get(reservation_id)

    def reservations_for_system(
        self, system_id: RouteSystemId
    ) -> tuple[RouteReservation, ...]:
        return self._reservations_by_system.get(system_id, ())

    def reservations_for_member(
        self, member_id: EmissionMemberId
    ) -> tuple[RouteReservation, ...]:
        return self._reservations_by_member.get(member_id, ())


def build_route_plan_query(plan: RoutePlan) -> RoutePlanQuery:
    endpoint_groups = {item.id: item for item in plan.endpoint_groups}
    divergences = {item.id: item for item in plan.divergences}
    convergences = {item.id: item for item in plan.convergences}
    members = {member.id: member for member in plan.members}
    for label, index, records in (
        ("endpoint group", endpoint_groups, plan.endpoint_groups),
        ("divergence", divergences, plan.divergences),
        ("convergence", convergences, plan.convergences),
    ):
        if len(index) != len(records):
            raise ValueError(f"route plan contains duplicate {label} ids")
    if len(members) != len(plan.members):
        raise ValueError("route plan contains duplicate emission member ids")
    bindings: dict[EmissionMemberId, list[EmissionBinding]] = defaultdict(list)
    for binding in plan.bindings:
        if binding.member_id not in members:
            raise ValueError(f"binding has unknown member {binding.member_id!r}")
        if (
            binding.covering_member_id is not None
            and binding.covering_member_id not in members
        ):
            raise ValueError(
                f"binding has unknown carrier {binding.covering_member_id!r}"
            )
        member = members[binding.member_id]
        family_required = binding.kind in {
            BindingKind.EMITTED,
            BindingKind.COVERED_MERGE_HOP,
        }
        if family_required != (member.family_id is not None):
            raise ValueError(
                f"{binding.kind.value} member has inconsistent production family"
            )
        bindings[binding.member_id].append(binding)
    if set(bindings) != set(members) or any(
        len(member_bindings) != 1 for member_bindings in bindings.values()
    ):
        raise ValueError("every emission member must have exactly one binding")
    for binding in plan.bindings:
        if binding.covering_member_id is None:
            continue
        member = members[binding.member_id]
        carrier = members[binding.covering_member_id]
        if carrier.id == member.id or carrier.system_id != member.system_id:
            raise ValueError("covered members require a distinct same-system carrier")
        (carrier_binding,) = bindings[carrier.id]
        if carrier_binding.kind is not BindingKind.EMITTED:
            raise ValueError("covered members require an emitted carrier")

    from nf_metro.layout.route_reservations import build_reservation_query_indexes

    reservation_indexes = build_reservation_query_indexes(plan, members, bindings)
    return RoutePlanQuery(
        plan,
        MappingProxyType(endpoint_groups),
        MappingProxyType(divergences),
        MappingProxyType(convergences),
        MappingProxyType(members),
        MappingProxyType({key: tuple(value) for key, value in bindings.items()}),
        MappingProxyType(reservation_indexes.references),
        MappingProxyType(reservation_indexes.demands),
        MappingProxyType(reservation_indexes.reservations),
        MappingProxyType(reservation_indexes.realisations),
        MappingProxyType(
            {key: tuple(value) for key, value in reservation_indexes.by_system.items()}
        ),
        MappingProxyType(
            {key: tuple(value) for key, value in reservation_indexes.by_member.items()}
        ),
    )


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"route plan contains unsupported {type(value).__name__}")


def serialize_route_plan(plan: RoutePlan) -> str:
    """Return the canonical JSON representation of one immutable plan."""
    return json.dumps(_json_value(plan), sort_keys=True, separators=(",", ":"))
