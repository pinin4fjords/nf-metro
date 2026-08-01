"""Immutable semantic route decisions and observations at the routing boundary.

The records in this module describe ownership, pre-routing decisions, and final
emission coverage. :class:`RoutePlanObserver` is a transient companion to the
production dispatcher: it copies scalar facts from the settled graph, carries
the complete exit-turn decisions consumed by routing, records the family
selected for each resolved inter-section leg, and binds the final route set
without retaining graph objects.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, NewType, TypeAlias, TypeVar

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.routing.common import Direction, right_normal_axis_sign
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
ExitTurnPlanId = NewType("ExitTurnPlanId", str)
ExitTurnAxisId = NewType("ExitTurnAxisId", str)
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


class ExitTurnDisposition(str, Enum):
    """Whether one complete exit group uses planned or legacy geometry."""

    PLANNED = "planned"
    LEGACY = "legacy"


class TurnHandedness(str, Enum):
    """Screen-space handedness of one perpendicular cardinal turn."""

    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


def turn_handedness(run: Direction, turn: Direction) -> TurnHandedness:
    """Return the screen-space handedness of a perpendicular cardinal turn."""
    vectors = {
        Direction.R: (1, 0),
        Direction.L: (-1, 0),
        Direction.U: (0, -1),
        Direction.D: (0, 1),
    }
    run_x, run_y = vectors[run]
    turn_x, turn_y = vectors[turn]
    cross_product = run_x * turn_y - run_y * turn_x
    if cross_product == 0:
        raise ValueError("run and turn directions must be perpendicular")
    return (
        TurnHandedness.CLOCKWISE
        if cross_product > 0
        else TurnHandedness.COUNTERCLOCKWISE
    )


class ExitLaneTransitionPlacement(str, Enum):
    """Which end of a lane hand-off keeps only its minimum runway."""

    SOURCE = "source"
    TARGET = "target"


class ExitLaneOrderSource(str, Enum):
    """Evidence used to order one exit group's active source lanes."""

    STATION_OFFSETS = "station-offsets"
    FRAME_CONSTRAINTS = "frame-constraints"
    GRAPH_LINE_ORDER_FALLBACK = "graph-line-order-fallback"


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

    def overlaps(self, other: GridSpan) -> bool:
        """Whether two inclusive grid extents intersect."""
        return not (
            self.max_column < other.min_column
            or other.max_column < self.min_column
            or self.max_row < other.min_row
            or other.max_row < self.min_row
        )


def grid_span_for_sections(
    graph: MetroGraph,
    section_ids: Iterable[str],
) -> GridSpan:
    """Return the inclusive grid extent of the named sections."""
    sections = tuple(graph.sections[section_id] for section_id in section_ids)
    if not sections:
        raise ValueError("grid span has no sections")
    return GridSpan(
        min(section.grid_col for section in sections),
        max(section.grid_col + section.grid_col_span - 1 for section in sections),
        min(section.grid_row for section in sections),
        max(section.grid_row + section.grid_row_span - 1 for section in sections),
    )


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
class ExitSourceLane:
    """One active source lane in compact visual order."""

    line_id: str
    rank: int
    member_ids: tuple[EmissionMemberId, ...]
    station_ids: tuple[str, ...]
    input_offset: float
    planned_offset: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_offset) or not math.isfinite(
            self.planned_offset
        ):
            raise ValueError("exit source-lane offsets must be finite")


@dataclass(frozen=True, slots=True)
class ExitLaneTransition:
    """Template input for one 45-degree compact-lane hand-off."""

    edge: ResolvedEdge
    claimant_member_ids: tuple[EmissionMemberId, ...]
    source_point: tuple[float, float]
    target_point: tuple[float, float]
    source_offset: float
    target_offset: float
    source_lane_offset: float
    target_lane_offset: float
    run_direction: Direction
    placement: ExitLaneTransitionPlacement
    diagonal_run: float
    source_runway: float
    target_runway: float
    coordinate_regime: CoordinateRegime = CoordinateRegime.LAYOUT_CANVAS

    def __post_init__(self) -> None:
        if not self.claimant_member_ids or len(set(self.claimant_member_ids)) != len(
            self.claimant_member_ids
        ):
            raise ValueError(
                "exit lane-transition claimants must be unique and nonempty"
            )
        values = (
            *self.source_point,
            *self.target_point,
            self.source_offset,
            self.target_offset,
            self.source_lane_offset,
            self.target_lane_offset,
            self.diagonal_run,
            self.source_runway,
            self.target_runway,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("exit lane-transition geometry must be finite")


@dataclass(frozen=True, slots=True)
class ExitTurnAxis:
    """One shared turn axis assigned to every arm of one source lane."""

    id: ExitTurnAxisId
    line_id: str
    axis: DemandAxis
    coordinate: float
    rank: int
    claimant_member_ids: tuple[EmissionMemberId, ...]
    fixed_anchor_id: str | None
    fixed_anchor_coordinate: float | None
    fixed_anchor_offset: float | None
    coordinate_regime: CoordinateRegime = CoordinateRegime.LAYOUT_CANVAS

    def __post_init__(self) -> None:
        if not math.isfinite(self.coordinate):
            raise ValueError("exit-turn axis coordinate must be finite")
        anchor_values = (
            self.fixed_anchor_id,
            self.fixed_anchor_coordinate,
            self.fixed_anchor_offset,
        )
        if any(item is None for item in anchor_values) and any(
            item is not None for item in anchor_values
        ):
            raise ValueError("exit-turn fixed-axis anchor is incomplete")
        if self.fixed_anchor_coordinate is not None and not math.isfinite(
            self.fixed_anchor_coordinate
        ):
            raise ValueError("exit-turn fixed-axis anchor must be finite")
        if self.fixed_anchor_offset is not None and not math.isfinite(
            self.fixed_anchor_offset
        ):
            raise ValueError("exit-turn fixed-axis offset must be finite")


@dataclass(frozen=True, slots=True)
class ExitTurnAssignment:
    """Planned source-side geometry for one outbound emission member."""

    member_id: EmissionMemberId
    entry_group_id: EndpointGroupId
    destination_section_id: str
    destination_column: int
    destination_row: int
    destination_side: PortSide
    source_lane_rank: int
    planned_family_id: RouteFamilyId
    roles: tuple[EmissionRole, ...]
    run_direction: Direction | None
    turn_direction: Direction | None
    launch_coordinate: float | None
    minimum_runway: float | None
    handedness: TurnHandedness | None
    axis_id: ExitTurnAxisId | None


@dataclass(frozen=True, slots=True)
class ExitTurnPlan:
    """Complete immutable source-bundle decision made before route emission."""

    id: ExitTurnPlanId
    system_id: RouteSystemId
    exit_group_id: EndpointGroupId
    exit_port_id: str
    divergence_id: DivergenceId | None
    source_id: str
    source_run_direction: Direction | None
    source_axis: DemandAxis
    connector_ids: tuple[ConnectorId, ...]
    system_member_ids: tuple[EmissionMemberId, ...]
    member_ids: tuple[EmissionMemberId, ...]
    source_lanes: tuple[ExitSourceLane, ...]
    lane_order_source: ExitLaneOrderSource
    lane_transitions: tuple[ExitLaneTransition, ...]
    axes: tuple[ExitTurnAxis, ...]
    assignments: tuple[ExitTurnAssignment, ...]
    unclassified_member_ids: tuple[EmissionMemberId, ...]
    spacing: float
    minimum_runway: float
    reference_id: SharedReferenceId | None
    demand_ids: tuple[DemandId, ...]
    foreign_reference_ids: tuple[SharedReferenceId, ...]
    disposition: ExitTurnDisposition
    legacy_reason: str | None
    provenance: tuple[ReservationDecisionRef, ...]

    def __post_init__(self) -> None:
        planned = self.disposition is ExitTurnDisposition.PLANNED
        if not isinstance(self.lane_order_source, ExitLaneOrderSource):
            raise ValueError("exit-turn lane-order provenance must be typed")
        if (
            not math.isfinite(self.spacing)
            or not math.isfinite(self.minimum_runway)
            or self.spacing <= 0
            or self.minimum_runway <= 0
        ):
            raise ValueError(
                "exit-turn geometry requirements must be finite and positive"
            )
        if planned != (self.legacy_reason is None):
            raise ValueError("exit-turn disposition and legacy reason disagree")
        if (self.reference_id is not None) != (planned and bool(self.axes)):
            raise ValueError("only planned turn axes own a shared reference")
        expected_axis = (
            DemandAxis.X
            if self.source_run_direction in {Direction.R, Direction.L}
            else DemandAxis.Y
            if self.source_run_direction in {Direction.U, Direction.D}
            else None
        )
        if self.source_axis not in {DemandAxis.X, DemandAxis.Y}:
            raise ValueError("exit turn has no source axis")
        if planned and self.source_axis is not expected_axis:
            raise ValueError("planned exit turn has inconsistent source orientation")


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
    exit_turn_plan_ids: tuple[ExitTurnPlanId, ...]
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
    blocking: bool = True


@dataclass(frozen=True, slots=True)
class RoutePlan:
    systems: tuple[RouteSystem, ...]
    endpoint_groups: tuple[ResolvedEndpointGroup, ...]
    divergences: tuple[RouteDivergence, ...]
    convergences: tuple[RouteConvergence, ...]
    members: tuple[EmissionMember, ...]
    branches: tuple[RouteBranch, ...]
    feeders: tuple[RouteFeeder, ...]
    exit_turn_plans: tuple[ExitTurnPlan, ...]
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


def reservation_decision_refs(
    provenance: RoutePlanProvenance,
    connector_ids: tuple[ConnectorId, ...],
    span: GridSpan,
) -> tuple[ReservationDecisionRef, ...]:
    """Return the settled decisions governing one complete geometry claim."""
    records: list[ReservationDecisionRef] = []
    for section_fact in provenance.sections:
        grid = section_fact.grid
        if grid is None:
            continue
        column, row, row_span, column_span = grid.value
        if not (
            span.min_column <= column + column_span - 1
            and column <= span.max_column
            and span.min_row <= row + row_span - 1
            and row <= span.max_row
        ):
            continue
        for kind, decision in (
            (ReservationDecisionKind.SECTION_GRID, section_fact.grid),
            (ReservationDecisionKind.SECTION_DIRECTION, section_fact.direction),
        ):
            if decision is not None:
                records.append(
                    ReservationDecisionRef(kind, section_fact.section_id, decision)
                )
    connector_set = set(connector_ids)
    for connector_fact in provenance.connectors:
        if connector_fact.connector_id not in connector_set:
            continue
        for role, side_decision in (
            (ConnectorEndpointRole.EXIT, connector_fact.exit_side),
            (ConnectorEndpointRole.ENTRY, connector_fact.entry_side),
        ):
            if side_decision is not None:
                records.append(
                    ReservationDecisionRef(
                        ReservationDecisionKind.CONNECTOR_SIDE,
                        str(connector_fact.connector_id),
                        side_decision,
                        role,
                    )
                )
    if provenance.fold_threshold is not None:
        records.append(
            ReservationDecisionRef(
                ReservationDecisionKind.FOLD_THRESHOLD,
                "layout",
                provenance.fold_threshold,
            )
        )
    records.append(
        ReservationDecisionRef(
            ReservationDecisionKind.LANE_ORDER,
            "line-order",
            provenance.lane_order.policy,
        )
    )
    return tuple(records)


@dataclass(slots=True)
class RoutePlanObserver:
    """Transient route-plan collector attached to one routing invocation."""

    graph: MetroGraph
    context: _RoutingCtx | None
    scaffold: RouteSemanticScaffold | None = None
    exit_turn_plans: tuple[ExitTurnPlan, ...] = ()
    exit_turn_references: tuple[SharedReference, ...] = ()
    exit_turn_demands: tuple[SymbolicDemand, ...] = ()
    exit_turn_diagnostics: tuple[RoutePlanDiagnostic, ...] = ()
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
    graph: MetroGraph,
    context: _RoutingCtx | None,
    *,
    scaffold: RouteSemanticScaffold | None = None,
    exit_turn_plans: tuple[ExitTurnPlan, ...] = (),
    exit_turn_references: tuple[SharedReference, ...] = (),
    exit_turn_demands: tuple[SymbolicDemand, ...] = (),
    exit_turn_diagnostics: tuple[RoutePlanDiagnostic, ...] = (),
) -> RoutePlanObserver:
    """Create one transient observer after settled routing context construction."""
    return RoutePlanObserver(
        graph,
        context,
        scaffold,
        exit_turn_plans,
        exit_turn_references,
        exit_turn_demands,
        exit_turn_diagnostics,
    )


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


@dataclass(frozen=True, slots=True)
class RouteSemanticScaffold:
    """Canonical semantic identities shared by planning and final observation."""

    topology: RouteTopology
    query: RouteTopologyQuery
    refs_by_edge: Mapping[ResolvedEdge, tuple[ConnectorLegRef, ...]]
    edge_order: tuple[ResolvedEdge, ...]
    components: tuple[tuple[ConnectorId, ...], ...]
    ordered_system_ids: tuple[RouteSystemId, ...]
    system_by_connector: Mapping[ConnectorId, RouteSystemId]
    resolution: _ResolutionRecords
    member_id_by_edge: Mapping[ResolvedEdge, EmissionMemberId]

    def system_for(self, connector_ids: tuple[ConnectorId, ...]) -> RouteSystemId:
        if not connector_ids:
            raise ValueError("route-plan ownership record has no connectors")
        system_id = self.system_by_connector[connector_ids[0]]
        if any(
            self.system_by_connector[item] != system_id for item in connector_ids[1:]
        ):
            raise ValueError("one topology record spans multiple route systems")
        return system_id


def build_route_semantic_scaffold(
    graph: MetroGraph,
    query: RouteTopologyQuery | None = None,
) -> RouteSemanticScaffold | None:
    """Build stable route-system and member identities before route emission."""
    topology = graph.route_topology
    if query is None:
        query = build_route_topology_query(graph)
    if topology is None or query is None:
        return None

    mutable_refs, edge_order = _resolved_member_refs(graph, topology, query)
    refs_by_edge = {edge: tuple(refs) for edge, refs in mutable_refs.items()}
    components = _semantic_components(topology, mutable_refs)
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

    return RouteSemanticScaffold(
        topology,
        query,
        MappingProxyType(refs_by_edge),
        edge_order,
        components,
        tuple(ordered_system_ids),
        MappingProxyType(system_by_connector),
        resolution,
        MappingProxyType(member_id_by_edge),
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
    context_query = observer.context.topology if observer.context is not None else None
    scaffold = observer.scaffold or build_route_semantic_scaffold(graph, context_query)
    if scaffold is None:
        return RoutePlan(
            systems=(),
            endpoint_groups=(),
            divergences=(),
            convergences=(),
            members=(),
            branches=(),
            feeders=(),
            exit_turn_plans=(),
            shared_references=(),
            demands=(),
            reservations=(),
            realised_reservations=(),
            reservation_diagnostics=(),
            bindings=(),
            provenance=_plan_provenance(graph, ()),
        )

    topology = scaffold.topology
    query = scaffold.query
    refs_by_edge = scaffold.refs_by_edge
    edge_order = scaffold.edge_order
    components = scaffold.components
    ordered_system_ids = scaffold.ordered_system_ids
    system_for = scaffold.system_for

    bundle_ids_by_system: dict[RouteSystemId, list[BundleId]] = defaultdict(list)
    for bundle in topology.bundles:
        bundle_ids_by_system[system_for(bundle.connector_ids)].append(bundle.id)
    resolution = scaffold.resolution
    member_id_by_edge = scaffold.member_id_by_edge

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

    exit_turn_ids_by_system: dict[RouteSystemId, list[ExitTurnPlanId]] = defaultdict(
        list
    )
    for exit_turn_plan in observer.exit_turn_plans:
        exit_turn_ids_by_system[exit_turn_plan.system_id].append(exit_turn_plan.id)

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
                tuple(exit_turn_ids_by_system[system_id]),
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
        exit_turn_plans=observer.exit_turn_plans,
        shared_references=observer.exit_turn_references,
        demands=observer.exit_turn_demands,
        reservations=(),
        realised_reservations=(),
        reservation_diagnostics=(),
        bindings=tuple(bindings),
        provenance=_plan_provenance(graph, topology.connectors),
        diagnostics=tuple(diagnostics) + observer.exit_turn_diagnostics,
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
    _exit_turn_plans: Mapping[ExitTurnPlanId, ExitTurnPlan]
    _exit_turns_by_source: Mapping[str, tuple[ExitTurnPlan, ...]]
    _exit_turns_by_member: Mapping[EmissionMemberId, tuple[ExitTurnPlan, ...]]
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

    def exit_turn_plan(self, plan_id: ExitTurnPlanId) -> ExitTurnPlan:
        return self._exit_turn_plans[plan_id]

    def exit_turn_plans_for_source(self, source_id: str) -> tuple[ExitTurnPlan, ...]:
        return self._exit_turns_by_source.get(source_id, ())

    def exit_turn_plans_for_member(
        self, member_id: EmissionMemberId
    ) -> tuple[ExitTurnPlan, ...]:
        return self._exit_turns_by_member.get(member_id, ())

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


def _validate_exit_turn_assignment(
    exit_turn_plan: ExitTurnPlan,
    assignment: ExitTurnAssignment,
    members: Mapping[EmissionMemberId, EmissionMember],
    lanes: Mapping[int, ExitSourceLane],
    axes: Mapping[ExitTurnAxisId, ExitTurnAxis],
    endpoint_groups: Mapping[EndpointGroupId, ResolvedEndpointGroup],
    section_grids: Mapping[str, GridCell],
) -> None:
    member = members[assignment.member_id]
    lane = lanes.get(assignment.source_lane_rank)
    entry_group = endpoint_groups.get(assignment.entry_group_id)
    destination_grid = (
        section_grids.get(entry_group.section_id) if entry_group is not None else None
    )
    if (
        assignment.member_id not in exit_turn_plan.member_ids
        or lane is None
        or assignment.member_id not in lane.member_ids
    ):
        raise ValueError("exit-turn assignment has inconsistent lane membership")
    if (
        entry_group is None
        or entry_group.system_id != exit_turn_plan.system_id
        or entry_group.role is not ConnectorEndpointRole.ENTRY
        or assignment.entry_group_id not in member.entry_group_ids
        or destination_grid is None
        or assignment.destination_section_id != entry_group.section_id
        or assignment.destination_column != destination_grid[0]
        or assignment.destination_row != destination_grid[1]
        or assignment.destination_side is not entry_group.side
    ):
        raise ValueError("exit-turn assignment has inconsistent destination")
    semantic_roles = set(assignment.roles) - {
        EmissionRole.CONTINUATION,
        EmissionRole.PEEL_OFF,
    }
    seam_roles = set(assignment.roles) & {
        EmissionRole.CONTINUATION,
        EmissionRole.PEEL_OFF,
    }
    expected_seam_role = (
        EmissionRole.CONTINUATION
        if assignment.turn_direction is None
        else EmissionRole.PEEL_OFF
    )
    seam_roles_are_consistent = (
        seam_roles == {expected_seam_role}
        if assignment.run_direction is not None
        else exit_turn_plan.disposition is ExitTurnDisposition.LEGACY
        and len(seam_roles) == 1
    )
    canonical_roles = tuple(
        role for role in EmissionRole if role in set(assignment.roles)
    )
    expected_handedness = (
        turn_handedness(assignment.run_direction, assignment.turn_direction)
        if assignment.turn_direction is not None
        and assignment.run_direction is not None
        else None
    )
    has_turn_requirement = (
        assignment.launch_coordinate is not None
        and assignment.minimum_runway is not None
        and math.isfinite(assignment.launch_coordinate)
        and math.isfinite(assignment.minimum_runway)
        and assignment.minimum_runway > 0
    )
    if (
        assignment.planned_family_id is not member.family_id
        or semantic_roles != set(member.roles)
        or not seam_roles_are_consistent
        or assignment.roles != canonical_roles
        or (
            assignment.run_direction not in set(Direction)
            and not (
                exit_turn_plan.disposition is ExitTurnDisposition.LEGACY
                and assignment.run_direction is None
            )
        )
        or assignment.handedness is not expected_handedness
        or (assignment.turn_direction is not None) != has_turn_requirement
    ):
        raise ValueError("exit-turn assignment has inconsistent semantics")
    axis = axes.get(assignment.axis_id) if assignment.axis_id is not None else None
    if exit_turn_plan.disposition is ExitTurnDisposition.PLANNED and (
        assignment.turn_direction is None
    ) != (axis is None):
        raise ValueError("exit-turn assignment has incomplete turn geometry")
    if (
        exit_turn_plan.disposition is ExitTurnDisposition.LEGACY
        and assignment.axis_id is not None
    ):
        raise ValueError("legacy exit-turn assignment cannot own an axis")
    if axis is not None and (
        assignment.member_id not in axis.claimant_member_ids
        or axis.line_id != lane.line_id
        or axis.rank != lane.rank
        or axis.axis is not exit_turn_plan.source_axis
        or axis.coordinate_regime is not CoordinateRegime.LAYOUT_CANVAS
    ):
        raise ValueError("exit-turn axis has inconsistent assignment geometry")


def _validate_exit_turn_demands(
    exit_turn_plan: ExitTurnPlan,
    expected_span: GridSpan,
    turning_assignment_ids: tuple[EmissionMemberId, ...],
    ordered_turn_span: float,
    demands: Mapping[DemandId, SymbolicDemand],
) -> None:
    if any(demand_id not in demands for demand_id in exit_turn_plan.demand_ids):
        raise ValueError("exit-turn plan has an unknown symbolic demand")
    owned_demands = tuple(demands[item] for item in exit_turn_plan.demand_ids)
    turn_demand_count = 2 if exit_turn_plan.axes else 0
    if len(owned_demands) != turn_demand_count + len(exit_turn_plan.lane_transitions):
        raise ValueError("exit-turn symbolic demand is inconsistent")
    if exit_turn_plan.axes:
        ordered_demand, runway_demand = owned_demands[:2]
        common_facts = (
            exit_turn_plan.system_id,
            exit_turn_plan.source_axis,
            expected_span,
            CoordinateRegime.LAYOUT_CANVAS,
            (exit_turn_plan.reference_id,),
            (KeepOutClass.SECTION, KeepOutClass.MARKER),
            exit_turn_plan.provenance,
        )

        def demand_facts(demand: SymbolicDemand) -> tuple[object, ...]:
            return (
                demand.system_id,
                demand.axis,
                demand.span,
                demand.minimum_size_regime,
                demand.ordered_reference_ids,
                demand.keep_out_classes,
                demand.provenance,
            )

        if (
            ordered_demand.kind is not DemandKind.ORDERED_TURNS
            or ordered_demand.claimant_member_ids != turning_assignment_ids
            or ordered_demand.lane_count != len(exit_turn_plan.axes)
            or ordered_demand.minimum_size != ordered_turn_span
            or demand_facts(ordered_demand) != common_facts
            or runway_demand.kind is not DemandKind.RUNWAY
            or runway_demand.claimant_member_ids != turning_assignment_ids
            or runway_demand.lane_count != len(exit_turn_plan.axes)
            or runway_demand.minimum_size != exit_turn_plan.minimum_runway
            or demand_facts(runway_demand) != common_facts
        ):
            raise ValueError("exit-turn symbolic demand is inconsistent")
    for transition, demand in zip(
        exit_turn_plan.lane_transitions,
        owned_demands[turn_demand_count:],
        strict=True,
    ):
        if (
            demand.system_id != exit_turn_plan.system_id
            or demand.claimant_member_ids != transition.claimant_member_ids
            or demand.kind is not DemandKind.RUNWAY
            or demand.axis
            is not (
                DemandAxis.X
                if transition.run_direction in {Direction.R, Direction.L}
                else DemandAxis.Y
            )
            or demand.span != expected_span
            or demand.lane_count != 1
            or demand.minimum_size
            != transition.source_runway
            + transition.diagonal_run
            + transition.target_runway
            or demand.minimum_size_regime is not CoordinateRegime.LAYOUT_CANVAS
            or demand.ordered_reference_ids
            or demand.keep_out_classes != (KeepOutClass.SECTION, KeepOutClass.MARKER)
            or demand.provenance != exit_turn_plan.provenance
        ):
            raise ValueError("exit-turn lane-transition demand is inconsistent")


def _validate_planned_exit_turn_resources(
    plan: RoutePlan,
    exit_turn_plan: ExitTurnPlan,
    exit_group: ResolvedEndpointGroup,
    members: Mapping[EmissionMemberId, EmissionMember],
    endpoint_groups: Mapping[EndpointGroupId, ResolvedEndpointGroup],
    section_grids: Mapping[str, GridCell],
    references: Mapping[SharedReferenceId, SharedReference],
    demands: Mapping[DemandId, SymbolicDemand],
) -> None:
    source_run_direction = exit_turn_plan.source_run_direction
    if source_run_direction not in set(Direction):
        raise ValueError("planned exit-turn plan has no source direction")
    claimed_sections = {exit_group.section_id} | {
        endpoint_groups[item.entry_group_id].section_id
        for item in exit_turn_plan.assignments
    }
    if any(item not in section_grids for item in claimed_sections):
        raise ValueError("exit-turn plan has an unknown section grid")
    claimed_cells = [section_grids[item] for item in claimed_sections]
    expected_span = GridSpan(
        min(item[0] for item in claimed_cells),
        max(item[0] + item[3] - 1 for item in claimed_cells),
        min(item[1] for item in claimed_cells),
        max(item[1] + item[2] - 1 for item in claimed_cells),
    )
    if exit_turn_plan.provenance != reservation_decision_refs(
        plan.provenance,
        exit_turn_plan.connector_ids,
        expected_span,
    ):
        raise ValueError("exit-turn plan has inconsistent provenance")
    offsets = tuple(lane.planned_offset for lane in exit_turn_plan.source_lanes)
    if any(
        abs(abs(right - left) - exit_turn_plan.spacing) > 1e-6
        for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("planned exit-turn source lanes are not compact")
    turning_assignment_ids = tuple(
        item.member_id
        for item in exit_turn_plan.assignments
        if item.axis_id is not None
    )
    reference = (
        references.get(exit_turn_plan.reference_id)
        if exit_turn_plan.reference_id is not None
        else None
    )
    if exit_turn_plan.axes:
        if (
            reference is None
            or reference.system_id != exit_turn_plan.system_id
            or reference.kind is not SharedReferenceKind.ORDERED_TURNS
            or reference.claimant_member_ids != turning_assignment_ids
            or reference.coordinate_regime is not CoordinateRegime.LAYOUT_CANVAS
            or reference.provenance != exit_turn_plan.provenance
        ):
            raise ValueError("exit-turn shared reference is inconsistent")
    elif exit_turn_plan.reference_id is not None:
        raise ValueError("axis-free exit-turn plan has a shared reference")
    turning_assignments = tuple(
        item for item in exit_turn_plan.assignments if item.axis_id is not None
    )
    cohort_ranks: dict[tuple[Direction, Direction], set[int]] = defaultdict(set)
    for assignment in turning_assignments:
        if assignment.run_direction is None or assignment.turn_direction is None:
            raise ValueError("exit-turn assignment has incomplete directions")
        cohort_ranks[assignment.run_direction, assignment.turn_direction].add(
            assignment.source_lane_rank
        )
    ordered_turn_span = max(
        ((len(ranks) - 1) * exit_turn_plan.spacing for ranks in cohort_ranks.values()),
        default=0.0,
    )
    if any(
        member_id not in exit_turn_plan.member_ids
        or members[member_id].system_id != exit_turn_plan.system_id
        or members[member_id].line_id != transition.edge.line_id
        for transition in exit_turn_plan.lane_transitions
        for member_id in transition.claimant_member_ids
    ):
        raise ValueError("exit-turn lane transition has inconsistent claimants")
    _validate_exit_turn_demands(
        exit_turn_plan,
        expected_span,
        turning_assignment_ids,
        ordered_turn_span,
        demands,
    )
    axis_claimants = {
        axis.id: tuple(
            item.member_id
            for item in exit_turn_plan.assignments
            if item.axis_id == axis.id
        )
        for axis in exit_turn_plan.axes
    }
    if any(
        axis.claimant_member_ids != axis_claimants[axis.id]
        for axis in exit_turn_plan.axes
    ):
        raise ValueError("exit-turn axis has inconsistent claimants")
    if len(exit_turn_plan.axes) != len(
        {
            (
                assignment.run_direction,
                assignment.turn_direction,
                assignment.source_lane_rank,
            )
            for assignment in turning_assignments
        }
    ):
        raise ValueError("exit-turn axes have inconsistent cohort membership")
    if any(
        (right.input_offset - left.input_offset)
        * (right.planned_offset - left.planned_offset)
        <= 0
        or abs(abs(right.planned_offset - left.planned_offset) - exit_turn_plan.spacing)
        > 1e-6
        for left, right in zip(
            exit_turn_plan.source_lanes,
            exit_turn_plan.source_lanes[1:],
        )
    ):
        raise ValueError("exit-turn source lanes do not preserve travel order")
    axis_by_id = {axis.id: axis for axis in exit_turn_plan.axes}
    for (run_direction, turn_direction), _ranks in cohort_ranks.items():
        cohort_axes = tuple(
            axis_by_id[assignment.axis_id]
            for assignment in turning_assignments
            if assignment.run_direction is run_direction
            and assignment.turn_direction is turn_direction
            and assignment.axis_id is not None
        )
        unique_axes = tuple(dict.fromkeys(cohort_axes))
        unique_axes = tuple(sorted(unique_axes, key=lambda axis: axis.rank))
        progression = right_normal_axis_sign(turn_direction)
        if any(
            abs(
                right.coordinate
                - left.coordinate
                - progression * exit_turn_plan.spacing
            )
            > 1e-6
            for left, right in zip(unique_axes, unique_axes[1:])
        ):
            raise ValueError("exit-turn axes do not preserve planned lane spacing")
    if any(
        assignment.axis_id is not None
        and (
            assignment.run_direction is None
            or assignment.launch_coordinate is None
            or assignment.minimum_runway is None
            or (
                axis_by_id[assignment.axis_id].coordinate - assignment.launch_coordinate
            )
            * assignment.run_direction.sign
            < assignment.minimum_runway - 1e-6
        )
        for assignment in turning_assignments
    ):
        raise ValueError("exit-turn axis does not satisfy its source runway")
    if any(
        axis.fixed_anchor_id is not None
        and axis.fixed_anchor_id
        not in {
            station_id
            for member_id in axis.claimant_member_ids
            for station_id in (
                members[member_id].source.station_id,
                members[member_id].target.station_id,
            )
        }
        for axis in exit_turn_plan.axes
    ):
        raise ValueError("exit-turn axis has an inconsistent fixed anchor")


def _validate_exit_turn_identity(
    exit_turn_plan: ExitTurnPlan,
    systems: Mapping[RouteSystemId, RouteSystem],
    endpoint_groups: Mapping[EndpointGroupId, ResolvedEndpointGroup],
    divergences: Mapping[DivergenceId, RouteDivergence],
    members: Mapping[EmissionMemberId, EmissionMember],
) -> ResolvedEndpointGroup:
    system = systems.get(exit_turn_plan.system_id)
    if system is None:
        raise ValueError("exit-turn plan has an unknown route system")
    if exit_turn_plan.spacing <= 0 or exit_turn_plan.minimum_runway <= 0:
        raise ValueError("exit-turn plan has invalid geometry requirements")
    exit_group = endpoint_groups.get(exit_turn_plan.exit_group_id)
    if (
        exit_group is None
        or exit_group.system_id != exit_turn_plan.system_id
        or exit_group.role is not ConnectorEndpointRole.EXIT
        or exit_group.port_id != exit_turn_plan.exit_port_id
        or exit_group.connector_ids != exit_turn_plan.connector_ids
    ):
        raise ValueError("exit-turn plan has inconsistent exit-group ownership")
    expected_source_run_direction = (
        Direction.R
        if exit_group.side is PortSide.RIGHT
        else Direction.L
        if exit_group.side is PortSide.LEFT
        else Direction.U
        if exit_group.side is PortSide.TOP
        else Direction.D
        if exit_group.side is PortSide.BOTTOM
        else None
    )
    expected_source_axis = (
        DemandAxis.X
        if expected_source_run_direction in {Direction.R, Direction.L}
        else DemandAxis.Y
    )
    if (
        exit_turn_plan.source_run_direction is not expected_source_run_direction
        or exit_turn_plan.source_axis is not expected_source_axis
    ):
        raise ValueError("exit-turn plan has inconsistent source-run direction")
    divergence = None
    if exit_turn_plan.divergence_id is None:
        if exit_turn_plan.source_id != exit_turn_plan.exit_port_id:
            raise ValueError("exit-turn plan has an inconsistent source")
    else:
        divergence = divergences.get(exit_turn_plan.divergence_id)
        if (
            divergence is None
            or divergence.system_id != exit_turn_plan.system_id
            or divergence.exit_group_id != exit_turn_plan.exit_group_id
            or divergence.junction_id != exit_turn_plan.source_id
        ):
            raise ValueError("exit-turn plan has an inconsistent divergence")
    if not exit_turn_plan.connector_ids or any(
        item not in system.connector_ids for item in exit_turn_plan.connector_ids
    ):
        raise ValueError("exit-turn plan has inconsistent connector ownership")
    if not exit_turn_plan.member_ids or any(
        item not in members or members[item].system_id != exit_turn_plan.system_id
        for item in exit_turn_plan.member_ids
    ):
        raise ValueError("exit-turn plan has inconsistent member ownership")
    expected_member_ids = tuple(
        member.id
        for member in members.values()
        if exit_turn_plan.exit_group_id in member.exit_group_ids
        and (
            member.source.station_id == exit_turn_plan.source_id
            or (
                divergence is not None
                and member.source.station_id == exit_turn_plan.exit_port_id
                and member.target.station_id == exit_turn_plan.source_id
            )
        )
    )
    if exit_turn_plan.member_ids != expected_member_ids:
        raise ValueError("exit-turn plan does not cover its complete exit group")
    if exit_turn_plan.system_member_ids != system.member_ids:
        raise ValueError("exit-turn plan does not cover its complete route system")
    return exit_group


def _validate_exit_source_lanes(
    exit_turn_plan: ExitTurnPlan,
    members: Mapping[EmissionMemberId, EmissionMember],
    station_owners: dict[tuple[str, str], ExitTurnPlanId],
) -> tuple[dict[int, ExitSourceLane], dict[str, ExitSourceLane]]:
    if tuple(lane.rank for lane in exit_turn_plan.source_lanes) != tuple(
        range(len(exit_turn_plan.source_lanes))
    ):
        raise ValueError("exit-turn source lanes are not compactly ranked")
    if (
        exit_turn_plan.disposition is ExitTurnDisposition.PLANNED
        and exit_turn_plan.lane_order_source
        is ExitLaneOrderSource.GRAPH_LINE_ORDER_FALLBACK
    ):
        raise ValueError("planned exit-turn source order has fallback provenance")
    if len({lane.line_id for lane in exit_turn_plan.source_lanes}) != len(
        exit_turn_plan.source_lanes
    ):
        raise ValueError("exit-turn plan contains duplicate source lanes")
    lane_by_rank = {lane.rank: lane for lane in exit_turn_plan.source_lanes}
    lane_members = tuple(
        member_id
        for lane in exit_turn_plan.source_lanes
        for member_id in lane.member_ids
    )
    if Counter(lane_members) != Counter(exit_turn_plan.member_ids) or any(
        count != 1 for count in Counter(lane_members).values()
    ):
        raise ValueError("exit-turn source lanes do not partition all members")
    if any(
        members[member_id].line_id != lane.line_id
        for lane in exit_turn_plan.source_lanes
        for member_id in lane.member_ids
    ):
        raise ValueError("exit-turn source lane has inconsistent line ownership")
    for lane in exit_turn_plan.source_lanes:
        if len(set(lane.station_ids)) != len(lane.station_ids):
            raise ValueError("exit-turn source lane repeats a station owner")
        if exit_turn_plan.disposition is ExitTurnDisposition.PLANNED:
            if not lane.station_ids:
                raise ValueError("planned exit-turn source lane has no station owner")
            if (
                exit_turn_plan.exit_port_id not in lane.station_ids
                or exit_turn_plan.source_id not in lane.station_ids
            ):
                raise ValueError("planned exit-turn source lane misses its boundary")
            for station_id in lane.station_ids:
                key = (station_id, lane.line_id)
                owner = station_owners.setdefault(key, exit_turn_plan.id)
                if owner != exit_turn_plan.id:
                    raise ValueError("exit-turn station lane has more than one owner")
        elif lane.station_ids or lane.planned_offset != lane.input_offset:
            raise ValueError("legacy exit-turn source lane cannot own offsets")
    lane_by_line = {lane.line_id: lane for lane in exit_turn_plan.source_lanes}
    return lane_by_rank, lane_by_line


def _validate_exit_turn_assignments(
    exit_turn_plan: ExitTurnPlan,
    members: Mapping[EmissionMemberId, EmissionMember],
    lane_by_rank: Mapping[int, ExitSourceLane],
    endpoint_groups: Mapping[EndpointGroupId, ResolvedEndpointGroup],
    section_grids: Mapping[str, GridCell],
) -> dict[ExitTurnAxisId, ExitTurnAxis]:
    assignment_ids = {item.member_id for item in exit_turn_plan.assignments}
    if len(assignment_ids) != len(exit_turn_plan.assignments):
        raise ValueError("exit-turn plan contains duplicate assignments")
    expected_assignment_ids = {
        member_id
        for member_id in exit_turn_plan.member_ids
        if members[member_id].source.station_id == exit_turn_plan.source_id
    }
    unclassified_ids = set(exit_turn_plan.unclassified_member_ids)
    if (
        len(unclassified_ids) != len(exit_turn_plan.unclassified_member_ids)
        or assignment_ids & unclassified_ids
        or assignment_ids | unclassified_ids != expected_assignment_ids
    ):
        raise ValueError("exit-turn assignments do not cover every outbound member")
    if exit_turn_plan.disposition is ExitTurnDisposition.PLANNED and unclassified_ids:
        raise ValueError("planned exit-turn plan contains unclassified members")
    axes = {axis.id: axis for axis in exit_turn_plan.axes}
    if len(axes) != len(exit_turn_plan.axes):
        raise ValueError("exit-turn plan contains duplicate axes")
    for assignment in exit_turn_plan.assignments:
        _validate_exit_turn_assignment(
            exit_turn_plan,
            assignment,
            members,
            lane_by_rank,
            axes,
            endpoint_groups,
            section_grids,
        )
    return axes


def _validate_exit_lane_transitions(
    exit_turn_plan: ExitTurnPlan,
    lane_by_line: Mapping[str, ExitSourceLane],
    transition_owners: dict[ResolvedEdge, ExitTurnPlanId],
) -> None:
    for transition in exit_turn_plan.lane_transitions:
        lane = lane_by_line.get(transition.edge.line_id)
        source_lateral = (
            transition.source_point[1]
            if transition.run_direction in {Direction.R, Direction.L}
            else transition.source_point[0]
        ) + transition.source_offset
        target_lateral = (
            transition.target_point[1]
            if transition.run_direction in {Direction.R, Direction.L}
            else transition.target_point[0]
        ) + transition.target_offset
        source_owned = lane is not None and transition.edge.source in lane.station_ids
        target_owned = lane is not None and transition.edge.target in lane.station_ids
        expected_placement = (
            ExitLaneTransitionPlacement.SOURCE
            if source_owned and not target_owned
            else ExitLaneTransitionPlacement.TARGET
            if target_owned and not source_owned
            else None
        )
        if (
            exit_turn_plan.disposition is not ExitTurnDisposition.PLANNED
            or lane is None
            or expected_placement is None
            or transition.placement is not expected_placement
            or transition.coordinate_regime is not CoordinateRegime.LAYOUT_CANVAS
            or abs(transition.diagonal_run - abs(target_lateral - source_lateral))
            > COORD_TOLERANCE
            or transition.diagonal_run <= 0
            or transition.source_runway <= 0
            or transition.target_runway <= 0
            or (source_owned and transition.source_lane_offset != lane.planned_offset)
            or (target_owned and transition.target_lane_offset != lane.planned_offset)
        ):
            raise ValueError("exit-turn lane transition is inconsistent")
        owner = transition_owners.setdefault(transition.edge, exit_turn_plan.id)
        if owner != exit_turn_plan.id:
            raise ValueError("exit-turn lane transition has more than one owner")


def _validate_exit_turn_diagnostics(plan: RoutePlan) -> None:
    actual = Counter(
        item for item in plan.diagnostics if item.code == "exit-turn-legacy"
    )
    expected = Counter(
        RoutePlanDiagnostic(
            item.member_ids[0] if item.member_ids else None,
            "exit-turn-legacy",
            f"exit group {item.exit_group_id} uses legacy routing: "
            f"{item.legacy_reason}",
            blocking=False,
        )
        for item in plan.exit_turn_plans
        if item.disposition is ExitTurnDisposition.LEGACY
    )
    if actual != expected:
        raise ValueError("exit-turn legacy diagnostics are inconsistent")


def _validate_exit_turn_records(
    plan: RoutePlan,
    members: Mapping[EmissionMemberId, EmissionMember],
) -> tuple[
    dict[ExitTurnPlanId, ExitTurnPlan],
    dict[str, list[ExitTurnPlan]],
    dict[EmissionMemberId, list[ExitTurnPlan]],
]:
    systems = {system.id: system for system in plan.systems}
    endpoint_groups = {item.id: item for item in plan.endpoint_groups}
    divergences = {item.id: item for item in plan.divergences}
    section_grids = {
        item.section_id: item.grid.value
        for item in plan.provenance.sections
        if item.grid is not None
    }
    references = {item.id: item for item in plan.shared_references}
    demands = {item.id: item for item in plan.demands}
    exit_turn_plans = {item.id: item for item in plan.exit_turn_plans}
    if len(exit_turn_plans) != len(plan.exit_turn_plans):
        raise ValueError("route plan contains duplicate exit-turn plan ids")
    by_source: dict[str, list[ExitTurnPlan]] = defaultdict(list)
    by_member: dict[EmissionMemberId, list[ExitTurnPlan]] = defaultdict(list)
    station_owners: dict[tuple[str, str], ExitTurnPlanId] = {}
    transition_owners: dict[ResolvedEdge, ExitTurnPlanId] = {}
    for exit_turn_plan in plan.exit_turn_plans:
        exit_group = _validate_exit_turn_identity(
            exit_turn_plan,
            systems,
            endpoint_groups,
            divergences,
            members,
        )
        lane_by_rank, lane_by_line = _validate_exit_source_lanes(
            exit_turn_plan,
            members,
            station_owners,
        )
        _validate_exit_turn_assignments(
            exit_turn_plan,
            members,
            lane_by_rank,
            endpoint_groups,
            section_grids,
        )
        _validate_exit_lane_transitions(
            exit_turn_plan,
            lane_by_line,
            transition_owners,
        )
        if exit_turn_plan.disposition is ExitTurnDisposition.PLANNED:
            _validate_planned_exit_turn_resources(
                plan,
                exit_turn_plan,
                exit_group,
                members,
                endpoint_groups,
                section_grids,
                references,
                demands,
            )
        elif (
            exit_turn_plan.axes
            or exit_turn_plan.demand_ids
            or exit_turn_plan.lane_transitions
        ):
            raise ValueError("legacy exit-turn plans cannot own geometry")
        by_source[exit_turn_plan.source_id].append(exit_turn_plan)
        for member_id in exit_turn_plan.member_ids:
            by_member[member_id].append(exit_turn_plan)

    for system in plan.systems:
        expected = tuple(
            item.id for item in plan.exit_turn_plans if item.system_id == system.id
        )
        if system.exit_turn_plan_ids != expected:
            raise ValueError("route system exit-turn index is inconsistent")
    if any(len(owners) != 1 for owners in by_member.values()):
        raise ValueError("exit-turn member has more than one owning plan")
    _validate_exit_turn_diagnostics(plan)
    from nf_metro.layout.route_reservations import (
        expected_exit_turn_foreign_references,
    )

    expected_foreign = expected_exit_turn_foreign_references(plan)
    if any(
        item.foreign_reference_ids != expected_foreign[item.id]
        for item in plan.exit_turn_plans
    ):
        raise ValueError("exit-turn foreign-reference index is inconsistent")
    return exit_turn_plans, by_source, by_member


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
    exit_turn_plans, exit_turns_by_source, exit_turns_by_member = (
        _validate_exit_turn_records(plan, members)
    )
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
        MappingProxyType(exit_turn_plans),
        MappingProxyType(
            {key: tuple(value) for key, value in exit_turns_by_source.items()}
        ),
        MappingProxyType(
            {key: tuple(value) for key, value in exit_turns_by_member.items()}
        ),
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
