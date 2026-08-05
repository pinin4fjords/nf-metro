"""Immutable shared-corridor claims observed from final routed members."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import NewType, TypeAlias

from nf_metro.layout.constants import (
    BUNDLE_TO_BUNDLE_CLEARANCE,
    COORD_TOLERANCE,
    CURVE_RADIUS,
    EDGE_TO_BUNDLE_CLEARANCE,
    INTER_ROW_EDGE_CLEARANCE,
    INTER_ROW_HEADER_CLEARANCE,
    SAME_COORD_TOLERANCE,
)
from nf_metro.layout.route_plan import (
    BindingKind,
    ConvergenceDisposition,
    ConvergencePlan,
    ConvergencePlanId,
    CoordinateRegime,
    DemandAxis,
    DemandId,
    DemandKind,
    EmissionBinding,
    EmissionMember,
    EmissionMemberId,
    EmissionRole,
    EmittedPathId,
    ExitTurnDisposition,
    ExitTurnPlan,
    ExitTurnPlanId,
    FanPlanDisposition,
    GridSpan,
    KeepOutClass,
    ReservationDecisionRef,
    RoutePlan,
    RouteSystem,
    RouteSystemId,
    SharedReference,
    SharedReferenceId,
    SharedReferenceKind,
    SymbolicDemand,
    grid_span_for_sections,
    reservation_decision_refs,
)
from nf_metro.layout.routing.common import Direction, RoutedPath, apply_route_offsets
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.parser.model import MetroGraph, Section
from nf_metro.parser.provenance import (
    ConnectorEndpointRole,
    GridCell,
)
from nf_metro.parser.route_topology import ConnectorId

RouteReservationId = NewType("RouteReservationId", str)

# Blocker ids name the section edge that bounds a corridor.  The prefix is the
# link between the measurement that publishes a blocker and the settlement that
# has to decide whether that edge is one it can translate.
SECTION_BOTTOM_BLOCKER = "section-bottom"
SECTION_HEADER_BLOCKER = "section-header"
SECTION_LEFT_BLOCKER = "section-left"
SECTION_RIGHT_BLOCKER = "section-right"


class CorridorKind(str, Enum):
    """Structurally distinct shared routing spaces."""

    DIRECT_INTER_ROW_BAND = "direct-inter-row-band"
    BYPASS_BAND = "bypass-band"
    OVER_TOP_BAND = "over-top-band"
    INTER_COLUMN_CHANNEL = "inter-column-channel"


class CorridorOrientation(str, Enum):
    """Axis along which the route travels through the reservation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class CorridorMeasurementScope(str, Enum):
    """Evidence used to select the enclosing corridor blockers."""

    TOPOLOGY_SPAN = "topology-span"
    OBSERVED_RUN = "observed-run"


class CorridorRegionKind(str, Enum):
    ROW_GAP = "row-gap"
    COLUMN_GAP = "column-gap"
    CANVAS = "canvas"


class CanvasSide(str, Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


@dataclass(frozen=True, slots=True)
class RowGapRegion:
    """The adjacent grid-row boundary allocated to a horizontal band."""

    upper_row: int
    lower_row: int
    kind: CorridorRegionKind = CorridorRegionKind.ROW_GAP
    coordinate_regime: CoordinateRegime = CoordinateRegime.SETTLED_GRID

    def __post_init__(self) -> None:
        if self.lower_row != self.upper_row + 1:
            raise ValueError("row-gap boundaries must be adjacent")


@dataclass(frozen=True, slots=True)
class ColumnGapRegion:
    """The adjacent grid-column boundary allocated to a vertical channel."""

    left_column: int
    right_column: int
    kind: CorridorRegionKind = CorridorRegionKind.COLUMN_GAP
    coordinate_regime: CoordinateRegime = CoordinateRegime.SETTLED_GRID

    def __post_init__(self) -> None:
        if self.right_column != self.left_column + 1:
            raise ValueError("column-gap boundaries must be adjacent")


@dataclass(frozen=True, slots=True)
class CanvasRegion:
    """A corridor outside every occupied grid row or column."""

    side: CanvasSide
    kind: CorridorRegionKind = CorridorRegionKind.CANVAS
    coordinate_regime: CoordinateRegime = CoordinateRegime.LAYOUT_CANVAS


CorridorRegion: TypeAlias = RowGapRegion | ColumnGapRegion | CanvasRegion


@dataclass(frozen=True, slots=True)
class RouteReservationClaim:
    """One bound emitted member segment claiming a corridor."""

    member_id: EmissionMemberId
    path_id: EmittedPathId
    path_rank: int
    segment_rank: int
    segment_end_rank: int
    longitudinal_start: float
    longitudinal_end: float
    allocation_coordinate: float

    def __post_init__(self) -> None:
        if (
            self.path_rank < 0
            or self.segment_rank < 0
            or self.segment_end_rank < self.segment_rank
        ):
            raise ValueError("reservation claim requires an ordered point-pair range")
        if not all(
            math.isfinite(value)
            for value in (
                self.longitudinal_start,
                self.longitudinal_end,
                self.allocation_coordinate,
            )
        ):
            raise ValueError("reservation claim coordinates must be finite")
        if self.longitudinal_end - self.longitudinal_start <= COORD_TOLERANCE:
            raise ValueError("reservation claim requires a positive travel interval")


@dataclass(frozen=True, slots=True)
class RouteReservationLane:
    """Claims that reuse one simultaneous physical corridor lane."""

    claim_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.claim_indices:
            raise ValueError("reservation lane requires at least one claim")
        if any(item < 0 for item in self.claim_indices):
            raise ValueError("reservation lane claim indices must be non-negative")
        if tuple(sorted(set(self.claim_indices))) != self.claim_indices:
            raise ValueError(
                "reservation lane claim indices must be unique and ordered"
            )


@dataclass(frozen=True, slots=True)
class RouteReservation:
    """One complete symbolic allocation claim for a shared corridor."""

    id: RouteReservationId
    system_id: RouteSystemId
    connector_ids: tuple[ConnectorId, ...]
    claimant_member_ids: tuple[EmissionMemberId, ...]
    claims: tuple[RouteReservationClaim, ...]
    kind: CorridorKind
    orientation: CorridorOrientation
    direction: Direction
    region: CorridorRegion
    span: GridSpan
    measurement_scope: CorridorMeasurementScope
    lanes: tuple[RouteReservationLane, ...]
    lane_count: int
    bundle_width: float
    peer_clearance: float
    negative_side_clearance: float
    positive_side_clearance: float
    minimum_width: float
    preferred_width: float | None
    keep_out_classes: tuple[KeepOutClass, ...]
    route_family_ids: tuple[RouteFamilyId, ...]
    reference_id: SharedReferenceId
    demand_ids: tuple[DemandId, ...]
    provenance: tuple[ReservationDecisionRef, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.connector_ids:
            raise ValueError("reservation requires authored connector attribution")
        if len(set(self.connector_ids)) != len(self.connector_ids):
            raise ValueError("reservation connector attribution contains duplicates")
        if not self.claimant_member_ids or not self.claims:
            raise ValueError("reservation requires bound member claimants")
        if self.lane_count != len(self.lanes):
            raise ValueError("reservation lane count must match its physical lanes")
        lane_claim_indices = tuple(
            index for lane in self.lanes for index in lane.claim_indices
        )
        if sorted(lane_claim_indices) != list(range(len(self.claims))):
            raise ValueError("reservation lanes must partition its emitted claims")
        if len(set(lane_claim_indices)) != len(lane_claim_indices):
            raise ValueError("reservation claim belongs to multiple physical lanes")
        if (
            self.bundle_width < 0
            or self.peer_clearance < 0
            or self.negative_side_clearance < 0
            or self.positive_side_clearance < 0
            or self.minimum_width < 0
        ):
            raise ValueError("reservation widths and clearances cannot be negative")
        expected = (
            self.negative_side_clearance
            + self.bundle_width
            + self.positive_side_clearance
        )
        if abs(self.minimum_width - expected) > COORD_TOLERANCE:
            raise ValueError("reservation minimum width disagrees with its clearances")
        if (
            self.preferred_width is not None
            and self.preferred_width < self.minimum_width
        ):
            raise ValueError("reservation preferred width is below its minimum")
        if not self.keep_out_classes:
            raise ValueError("reservation requires keep-out classes")
        if not self.route_family_ids:
            raise ValueError("reservation requires a production route family")
        if len(self.demand_ids) != 1:
            raise ValueError("observed corridor requires exactly one symbolic demand")
        expected_orientation = (
            CorridorOrientation.VERTICAL
            if self.kind is CorridorKind.INTER_COLUMN_CHANNEL
            else CorridorOrientation.HORIZONTAL
        )
        if self.orientation is not expected_orientation:
            raise ValueError("reservation kind and orientation disagree")
        valid_directions = (
            {Direction.L, Direction.R}
            if self.orientation is CorridorOrientation.HORIZONTAL
            else {Direction.U, Direction.D}
        )
        if self.direction not in valid_directions:
            raise ValueError("reservation direction and orientation disagree")
        region_matches_orientation = (
            self.orientation is CorridorOrientation.HORIZONTAL
            and (
                isinstance(self.region, RowGapRegion)
                or isinstance(self.region, CanvasRegion)
                and self.region.side in {CanvasSide.TOP, CanvasSide.BOTTOM}
            )
        ) or (
            self.orientation is CorridorOrientation.VERTICAL
            and (
                isinstance(self.region, ColumnGapRegion)
                or isinstance(self.region, CanvasRegion)
                and self.region.side in {CanvasSide.LEFT, CanvasSide.RIGHT}
            )
        )
        if not region_matches_orientation:
            raise ValueError("reservation region and orientation disagree")
        if isinstance(self.region, CanvasRegion):
            expected_kind = {
                CanvasSide.TOP: CorridorKind.OVER_TOP_BAND,
                CanvasSide.BOTTOM: CorridorKind.BYPASS_BAND,
                CanvasSide.LEFT: CorridorKind.INTER_COLUMN_CHANNEL,
                CanvasSide.RIGHT: CorridorKind.INTER_COLUMN_CHANNEL,
            }[self.region.side]
            if self.kind is not expected_kind:
                raise ValueError("canvas side and corridor kind disagree")
        if (
            isinstance(self.region, CanvasRegion)
            and self.measurement_scope is not CorridorMeasurementScope.OBSERVED_RUN
        ):
            raise ValueError("canvas corridors require observed-run blocker scope")


@dataclass(frozen=True, slots=True)
class RealisedRouteReservation:
    """Final canvas measurement of one symbolic corridor reservation."""

    reservation_id: RouteReservationId
    allocation_axis: DemandAxis
    longitudinal_axis: DemandAxis
    coordinate: float
    longitudinal_start: float
    longitudinal_end: float
    region_start: float
    region_end: float
    occupied_start: float
    occupied_end: float
    available_width: float
    required_width: float
    capacity_slack: float
    negative_side_slack: float
    positive_side_slack: float
    negative_blocker_ids: tuple[str, ...]
    positive_blocker_ids: tuple[str, ...]
    coordinate_regime: CoordinateRegime = CoordinateRegime.LAYOUT_CANVAS
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = ()

    def __post_init__(self) -> None:
        if self.allocation_axis is DemandAxis.BOTH:
            raise ValueError("a corridor realisation allocates one canvas axis")
        if self.longitudinal_axis is DemandAxis.BOTH:
            raise ValueError("a corridor run occupies one canvas axis")
        if self.longitudinal_axis is self.allocation_axis:
            raise ValueError("corridor travel and allocation axes must be orthogonal")
        if self.longitudinal_end < self.longitudinal_start:
            raise ValueError("realised longitudinal interval is reversed")
        if self.occupied_end < self.occupied_start:
            raise ValueError("realised occupied interval is reversed")
        if (
            abs(self.available_width - (self.region_end - self.region_start))
            > COORD_TOLERANCE
        ):
            raise ValueError("realised available width disagrees with its edges")
        if not self.negative_blocker_ids or not self.positive_blocker_ids:
            raise ValueError("realised corridor requires both boundary blockers")


@dataclass(frozen=True, slots=True)
class ReservationCoordinateTranslation:
    """One global row or column translation, as it acts on frozen claims.

    Settlement translates whole sections, so a claim coordinate observed
    before the translation stays meaningful afterwards only when projected the
    same way the geometry moved.  A member whose endpoints both sit in moved
    sections travels wholesale; a member crossing the boundary keeps its near
    end and carries its far end, so only coordinates at or beyond the
    translated band's start move.
    """

    axis: DemandAxis
    coordinate: float
    amount: float
    fully_owned_member_ids: tuple[EmissionMemberId, ...] = ()
    crossing_member_ids: tuple[EmissionMemberId, ...] = ()

    def __post_init__(self) -> None:
        if self.axis is DemandAxis.BOTH:
            raise ValueError("reservation translation requires one canvas axis")
        if not math.isfinite(self.coordinate) or not math.isfinite(self.amount):
            raise ValueError("reservation translation must be finite")
        if self.amount <= COORD_TOLERANCE:
            raise ValueError("reservation translation must be positive")
        if set(self.fully_owned_member_ids).intersection(self.crossing_member_ids):
            raise ValueError("reservation translation member ownership overlaps")


def project_reservation_coordinate(
    value: float,
    axis: DemandAxis,
    member_id: EmissionMemberId,
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> float:
    """Project one frozen claim coordinate through global settlement translations."""
    projected = value
    for translation in translations:
        if translation.axis is not axis:
            continue
        if member_id in translation.fully_owned_member_ids:
            projected += translation.amount
        elif (
            member_id in translation.crossing_member_ids
            and projected >= translation.coordinate - SAME_COORD_TOLERANCE
        ):
            projected += translation.amount
    return projected


@dataclass(frozen=True, slots=True)
class RouteReservationDiagnostic:
    """Attributed evidence that final geometry violates a reservation."""

    reservation_id: RouteReservationId
    claimant_member_ids: tuple[EmissionMemberId, ...]
    code: str
    message: str
    capacity_slack: float
    negative_side_slack: float
    positive_side_slack: float


@dataclass(frozen=True, slots=True)
class _AxisSegment:
    rank: int
    end_rank: int
    orientation: CorridorOrientation
    direction: Direction
    start: tuple[float, float]
    end: tuple[float, float]
    before: tuple[float, float] | None
    after: tuple[float, float] | None

    @property
    def span_start(self) -> float:
        return min(self.travel_start, self.travel_end)

    @property
    def span_end(self) -> float:
        return max(self.travel_start, self.travel_end)

    @property
    def travel_start(self) -> float:
        return (
            self.start[0]
            if self.orientation is CorridorOrientation.HORIZONTAL
            else self.start[1]
        )

    @property
    def travel_end(self) -> float:
        return (
            self.end[0]
            if self.orientation is CorridorOrientation.HORIZONTAL
            else self.end[1]
        )

    @property
    def coordinate(self) -> float:
        return (
            self.start[1]
            if self.orientation is CorridorOrientation.HORIZONTAL
            else self.start[0]
        )

    @property
    def length(self) -> float:
        return self.span_end - self.span_start


@dataclass(frozen=True, slots=True)
class _ObservedClaim:
    system_id: RouteSystemId
    member: EmissionMember
    connector_ids: tuple[ConnectorId, ...]
    sharing_ids: tuple[str, ...]
    claim: RouteReservationClaim
    kind: CorridorKind
    orientation: CorridorOrientation
    direction: Direction
    region: CorridorRegion
    span: GridSpan
    measurement_scope: CorridorMeasurementScope
    travel_start: float
    travel_end: float
    coordinate: float


@dataclass(frozen=True, slots=True)
class _RegionMeasurement:
    start: float
    end: float
    negative_blocker_ids: tuple[str, ...]
    positive_blocker_ids: tuple[str, ...]


def _stable_id(prefix: str, *parts: object) -> str:
    """Return a compact content-derived identity with unambiguous framing."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode()
        digest.update(str(len(encoded)).encode())
        digest.update(b":")
        digest.update(encoded)
    return f"{prefix}:{digest.hexdigest()[:24]}"


def _is_horizontal(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return (
        abs(first[1] - second[1]) <= COORD_TOLERANCE
        and abs(first[0] - second[0]) > COORD_TOLERANCE
    )


def _is_vertical(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return (
        abs(first[0] - second[0]) <= COORD_TOLERANCE
        and abs(first[1] - second[1]) > COORD_TOLERANCE
    )


def _maximal_axis_segments(
    points: Sequence[tuple[float, float]],
) -> tuple[_AxisSegment, ...]:
    segments: list[_AxisSegment] = []
    index = 0
    while index < len(points) - 1:
        first, second = points[index], points[index + 1]
        if _is_horizontal(first, second):
            orientation = CorridorOrientation.HORIZONTAL
            direction = Direction.R if second[0] > first[0] else Direction.L
            coordinate = first[1]
        elif _is_vertical(first, second):
            orientation = CorridorOrientation.VERTICAL
            direction = Direction.D if second[1] > first[1] else Direction.U
            coordinate = first[0]
        else:
            index += 1
            continue
        end_index = index + 1
        while end_index < len(points) - 1:
            current, following = points[end_index], points[end_index + 1]
            same_axis = (
                orientation is CorridorOrientation.HORIZONTAL
                and _is_horizontal(current, following)
                and abs(current[1] - coordinate) <= COORD_TOLERANCE
                and (following[0] - current[0]) * direction.sign > 0
            ) or (
                orientation is CorridorOrientation.VERTICAL
                and _is_vertical(current, following)
                and abs(current[0] - coordinate) <= COORD_TOLERANCE
                and (following[1] - current[1]) * direction.sign > 0
            )
            if not same_axis:
                break
            end_index += 1
        segments.append(
            _AxisSegment(
                index,
                end_index - 1,
                orientation,
                direction,
                first,
                points[end_index],
                points[index - 1] if index > 0 else None,
                points[end_index + 1] if end_index + 1 < len(points) else None,
            )
        )
        index = end_index
    return tuple(segments)


def _overlaps(
    first_lo: float, first_hi: float, second_lo: float, second_hi: float
) -> bool:
    return min(first_hi, second_hi) > max(first_lo, second_lo) + COORD_TOLERANCE


def _section_x_overlaps(section: Section, lo: float, hi: float) -> bool:
    return _overlaps(section.bbox_x, section.bbox_x + section.bbox_w, lo, hi)


def _section_y_overlaps(section: Section, lo: float, hi: float) -> bool:
    return _overlaps(section.bbox_y, section.bbox_y + section.bbox_h, lo, hi)


def _row_end(section: Section) -> int:
    return section.grid_row + section.grid_row_span - 1


def _column_end(section: Section) -> int:
    return section.grid_col + section.grid_col_span - 1


def _span_overlaps_section_columns(span: GridSpan, section: Section) -> bool:
    return (
        span.min_column <= _column_end(section) and section.grid_col <= span.max_column
    )


def _span_overlaps_section_rows(span: GridSpan, section: Section) -> bool:
    return span.min_row <= _row_end(section) and section.grid_row <= span.max_row


def _edge_blockers(
    items: Iterable[tuple[str, float]], *, maximum: bool
) -> tuple[float, tuple[str, ...]]:
    ordered = tuple(items)
    if not ordered:
        raise ValueError("corridor boundary has no settled blocker")
    coordinate = (max if maximum else min)(value for _item_id, value in ordered)
    blockers = tuple(
        item_id
        for item_id, value in ordered
        if abs(value - coordinate) <= COORD_TOLERANCE
    )
    return coordinate, blockers


def _row_region_measurement(
    graph: MetroGraph,
    region: RowGapRegion,
    measurement_scope: CorridorMeasurementScope,
    span: GridSpan,
    longitudinal_start: float,
    longitudinal_end: float,
) -> _RegionMeasurement:
    """Measure the clear width between the two rows *region* separates.

    A section spanning across the boundary bounds neither side of it: its
    bottom edge lies below the boundary and its header above, so it occupies
    the boundary rather than bounding it.  Where every relevant section spans
    across, the boundary has no side to measure and this raises, which is what
    tells the region search that this corridor does not run in a row gap here.
    """
    relevant = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_w > 0
        and (
            _span_overlaps_section_columns(span, section)
            if measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
            else _section_x_overlaps(section, longitudinal_start, longitudinal_end)
        )
    )
    upper = tuple(
        section for section in relevant if _row_end(section) <= region.upper_row
    )
    lower = tuple(
        section for section in relevant if section.grid_row >= region.lower_row
    )
    start, negative = _edge_blockers(
        (
            (f"{SECTION_BOTTOM_BLOCKER}:{section.id}", section.bbox_y + section.bbox_h)
            for section in upper
        ),
        maximum=True,
    )
    end, positive = _edge_blockers(
        (
            (f"{SECTION_HEADER_BLOCKER}:{section.id}", section.bbox_y)
            for section in lower
        ),
        maximum=False,
    )
    return _RegionMeasurement(start, end, negative, positive)


def _column_region_measurement(
    graph: MetroGraph,
    region: ColumnGapRegion,
    measurement_scope: CorridorMeasurementScope,
    span: GridSpan,
    longitudinal_start: float,
    longitudinal_end: float,
) -> _RegionMeasurement:
    relevant = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_h > 0
        and (
            _span_overlaps_section_rows(span, section)
            if measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
            else _section_y_overlaps(section, longitudinal_start, longitudinal_end)
        )
    )
    left = tuple(
        section for section in relevant if _column_end(section) <= region.left_column
    )
    right = tuple(
        section for section in relevant if section.grid_col >= region.right_column
    )
    start, negative = _edge_blockers(
        (
            (f"{SECTION_RIGHT_BLOCKER}:{section.id}", section.bbox_x + section.bbox_w)
            for section in left
        ),
        maximum=True,
    )
    end, positive = _edge_blockers(
        ((f"{SECTION_LEFT_BLOCKER}:{section.id}", section.bbox_x) for section in right),
        maximum=False,
    )
    return _RegionMeasurement(start, end, negative, positive)


def _candidate_row_gaps(
    graph: MetroGraph, connector_ids: tuple[ConnectorId, ...]
) -> tuple[RowGapRegion, ...]:
    assert graph.route_topology is not None
    connector_by_id = {
        connector.id: connector for connector in graph.route_topology.connectors
    }
    rows: set[int] = set()
    for connector_id in connector_ids:
        connector = connector_by_id[connector_id]
        source = graph.sections[connector.source_section]
        target = graph.sections[connector.target_section]
        lo = min(source.grid_row, target.grid_row)
        hi = max(_row_end(source), _row_end(target))
        if source.grid_row != target.grid_row or _row_end(source) != _row_end(target):
            rows.update(range(lo, hi))
    return tuple(RowGapRegion(row, row + 1) for row in sorted(rows))


def _candidate_column_gaps(
    graph: MetroGraph, connector_ids: tuple[ConnectorId, ...]
) -> tuple[ColumnGapRegion, ...]:
    assert graph.route_topology is not None
    connector_by_id = {
        connector.id: connector for connector in graph.route_topology.connectors
    }
    columns: set[int] = set()
    for connector_id in connector_ids:
        connector = connector_by_id[connector_id]
        source = graph.sections[connector.source_section]
        target = graph.sections[connector.target_section]
        lo = min(source.grid_col, target.grid_col)
        hi = max(_column_end(source), _column_end(target))
        if source.grid_col != target.grid_col or _column_end(source) != _column_end(
            target
        ):
            columns.update(range(lo, hi))
    return tuple(ColumnGapRegion(column, column + 1) for column in sorted(columns))


def _region_containing_coordinate(
    graph: MetroGraph,
    segment: _AxisSegment,
    span: GridSpan,
    measurement_scope: CorridorMeasurementScope,
    candidates: Sequence[RowGapRegion | ColumnGapRegion],
) -> RowGapRegion | ColumnGapRegion | None:
    matches: list[tuple[float, RowGapRegion | ColumnGapRegion]] = []
    for region in candidates:
        try:
            measurement = (
                _row_region_measurement(
                    graph,
                    region,
                    measurement_scope,
                    span,
                    segment.span_start,
                    segment.span_end,
                )
                if isinstance(region, RowGapRegion)
                else _column_region_measurement(
                    graph,
                    region,
                    measurement_scope,
                    span,
                    segment.span_start,
                    segment.span_end,
                )
            )
        except ValueError:
            continue
        if (
            measurement.start - COORD_TOLERANCE
            <= segment.coordinate
            <= measurement.end + COORD_TOLERANCE
        ):
            matches.append((measurement.end - measurement.start, region))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _dominant_interior_segment(
    segment: _AxisSegment, segments: tuple[_AxisSegment, ...]
) -> bool:
    peers = tuple(item for item in segments if item.orientation is segment.orientation)
    has_two_turns = segment.before is not None and segment.after is not None
    return (
        has_two_turns
        and segment.length >= max(item.length for item in peers) - COORD_TOLERANCE
    )


def _topology_gap_fallback_is_proven(
    member: EmissionMember,
    segment: _AxisSegment,
    segments: tuple[_AxisSegment, ...],
) -> bool:
    return (
        segment.orientation is CorridorOrientation.HORIZONTAL
        and member.family_id
        in {RouteFamilyId.MERGE_TRUNK, RouteFamilyId.LEFT_ENTRY_WRAP}
        and _dominant_interior_segment(segment, segments)
    )


def _nearest_topology_region(
    graph: MetroGraph,
    segment: _AxisSegment,
    span: GridSpan,
    candidates: Sequence[RowGapRegion | ColumnGapRegion],
) -> RowGapRegion | ColumnGapRegion | None:
    measured: list[tuple[float, RowGapRegion | ColumnGapRegion]] = []
    for region in candidates:
        try:
            value = (
                _row_region_measurement(
                    graph,
                    region,
                    CorridorMeasurementScope.TOPOLOGY_SPAN,
                    span,
                    segment.span_start,
                    segment.span_end,
                )
                if isinstance(region, RowGapRegion)
                else _column_region_measurement(
                    graph,
                    region,
                    CorridorMeasurementScope.TOPOLOGY_SPAN,
                    span,
                    segment.span_start,
                    segment.span_end,
                )
            )
        except ValueError:
            continue
        centre = (value.start + value.end) / 2
        measured.append((abs(segment.coordinate - centre), region))
    return min(measured, key=lambda item: item[0])[1] if measured else None


def _geometric_row_gap(
    graph: MetroGraph, segment: _AxisSegment, span: GridSpan
) -> RowGapRegion | None:
    sections = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_w > 0
        and _section_x_overlaps(section, segment.span_start, segment.span_end)
    )
    if not sections:
        return None
    candidates = tuple(
        RowGapRegion(row, row + 1)
        for row in range(
            min(section.grid_row for section in sections),
            max(_row_end(section) for section in sections),
        )
    )
    region = _region_containing_coordinate(
        graph,
        segment,
        span,
        CorridorMeasurementScope.OBSERVED_RUN,
        candidates,
    )
    return region if isinstance(region, RowGapRegion) else None


def _geometric_column_gap(
    graph: MetroGraph, segment: _AxisSegment, span: GridSpan
) -> ColumnGapRegion | None:
    sections = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_h > 0
        and _section_y_overlaps(section, segment.span_start, segment.span_end)
    )
    if not sections:
        return None
    candidates = tuple(
        ColumnGapRegion(column, column + 1)
        for column in range(
            min(section.grid_col for section in sections),
            max(_column_end(section) for section in sections),
        )
    )
    region = _region_containing_coordinate(
        graph,
        segment,
        span,
        CorridorMeasurementScope.OBSERVED_RUN,
        candidates,
    )
    return region if isinstance(region, ColumnGapRegion) else None


def _canvas_region_for_segment(
    graph: MetroGraph, segment: _AxisSegment
) -> CanvasRegion | None:
    """The canvas margin *segment* runs in, or ``None`` for an interior run.

    A canvas corridor lies between the map's content and the canvas edge, so
    the coordinate is compared against the extreme of EVERY placed section,
    not only the sections the run passes beside: an inter-row or inter-column
    run overlaps neither of its own bounding sections longitudinally, and
    judging it against whatever unrelated section pokes into its window would
    call an interior corridor a margin one.  The overlap precondition keeps a
    run outside all content longitudinally (an empty map edge) unclassified.
    """
    placed = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_w > 0 and section.bbox_h > 0
    )
    if segment.orientation is CorridorOrientation.HORIZONTAL:
        if not any(
            _section_x_overlaps(section, segment.span_start, segment.span_end)
            for section in placed
        ):
            return None
        if segment.coordinate < min(section.bbox_y for section in placed):
            return CanvasRegion(CanvasSide.TOP)
        if segment.coordinate > max(
            section.bbox_y + section.bbox_h for section in placed
        ):
            return CanvasRegion(CanvasSide.BOTTOM)
        return None
    if not any(
        _section_y_overlaps(section, segment.span_start, segment.span_end)
        for section in placed
    ):
        return None
    if segment.coordinate < min(section.bbox_x for section in placed):
        return CanvasRegion(CanvasSide.LEFT)
    if segment.coordinate > max(section.bbox_x + section.bbox_w for section in placed):
        return CanvasRegion(CanvasSide.RIGHT)
    return None


def _corridor_region(
    graph: MetroGraph,
    segment: _AxisSegment,
    segments: tuple[_AxisSegment, ...],
    span: GridSpan,
    connector_ids: tuple[ConnectorId, ...],
    member: EmissionMember,
) -> tuple[CorridorRegion, CorridorMeasurementScope] | None:
    if segment.orientation is CorridorOrientation.HORIZONTAL:
        candidates = _candidate_row_gaps(graph, connector_ids)
        region = _region_containing_coordinate(
            graph,
            segment,
            span,
            CorridorMeasurementScope.TOPOLOGY_SPAN,
            candidates,
        )
        if region is not None:
            return region, CorridorMeasurementScope.TOPOLOGY_SPAN
        # A run drawn outside every section beside it is a canvas corridor, and
        # the router consumes a gap claim where its band lies.  Assigning such a
        # run the nearest topology gap would demand a corridor the frozen route
        # shape cannot reach -- the drawn dip cannot become a between-rows run
        # by translation -- so the canvas classification precedes the fallback.
        canvas_region = _canvas_region_for_segment(graph, segment)
        if canvas_region is not None:
            return canvas_region, CorridorMeasurementScope.OBSERVED_RUN
        if candidates and _topology_gap_fallback_is_proven(member, segment, segments):
            region = _nearest_topology_region(graph, segment, span, candidates)
            if region is not None:
                return region, CorridorMeasurementScope.TOPOLOGY_SPAN
        observed_region: CorridorRegion | None = _geometric_row_gap(
            graph, segment, span
        )
        return (
            (observed_region, CorridorMeasurementScope.OBSERVED_RUN)
            if observed_region is not None
            else None
        )
    column_candidates = _candidate_column_gaps(graph, connector_ids)
    region = _region_containing_coordinate(
        graph,
        segment,
        span,
        CorridorMeasurementScope.TOPOLOGY_SPAN,
        column_candidates,
    )
    if region is not None:
        return region, CorridorMeasurementScope.TOPOLOGY_SPAN
    canvas_column_region = _canvas_region_for_segment(graph, segment)
    if canvas_column_region is not None:
        return canvas_column_region, CorridorMeasurementScope.OBSERVED_RUN
    if column_candidates and _topology_gap_fallback_is_proven(
        member, segment, segments
    ):
        region = _nearest_topology_region(graph, segment, span, column_candidates)
        if region is not None:
            return region, CorridorMeasurementScope.TOPOLOGY_SPAN
    observed_region = _geometric_column_gap(graph, segment, span)
    return (
        (observed_region, CorridorMeasurementScope.OBSERVED_RUN)
        if observed_region is not None
        else None
    )


def _horizontal_kind(
    segment: _AxisSegment, region: CorridorRegion, member: EmissionMember
) -> CorridorKind:
    y = segment.coordinate
    neighbours = tuple(
        point[1]
        for point in (segment.before, segment.after)
        if point is not None and abs(point[1] - y) > COORD_TOLERANCE
    )
    if isinstance(region, CanvasRegion):
        return (
            CorridorKind.OVER_TOP_BAND
            if region.side is CanvasSide.TOP
            else CorridorKind.BYPASS_BAND
        )
    if len(neighbours) == 2 and all(value > y for value in neighbours):
        return CorridorKind.OVER_TOP_BAND
    if len(neighbours) == 2 and all(value < y for value in neighbours):
        return CorridorKind.BYPASS_BAND
    if EmissionRole.BYPASS in member.roles:
        return CorridorKind.BYPASS_BAND
    return CorridorKind.DIRECT_INTER_ROW_BAND


def _connector_span(
    graph: MetroGraph, connector_ids: tuple[ConnectorId, ...]
) -> GridSpan:
    assert graph.route_topology is not None
    connector_set = set(connector_ids)
    section_ids = tuple(
        section_id
        for connector in graph.route_topology.connectors
        if connector.id in connector_set
        for section_id in (connector.source_section, connector.target_section)
    )
    if not section_ids:
        raise ValueError("corridor claim has no authored section span")
    return grid_span_for_sections(graph, section_ids)


def _covered_members(
    plan: RoutePlan, emitted_member_id: EmissionMemberId
) -> tuple[EmissionMemberId, ...]:
    return tuple(
        binding.member_id
        for binding in plan.bindings
        if binding.covering_member_id == emitted_member_id
    )


def _sharing_ids(member: EmissionMember) -> tuple[str, ...]:
    """Semantic groups that can prove two member lanes share one allocation."""
    return tuple(
        str(item)
        for records in (
            member.bundle_ids,
            member.exit_group_ids,
            member.entry_group_ids,
            member.divergence_ids,
            member.convergence_ids,
        )
        for item in records
    )


def _observed_claims(
    graph: MetroGraph,
    routes: list[RoutedPath],
    plan: RoutePlan,
    station_offsets: dict[tuple[str, str], float],
) -> tuple[_ObservedClaim, ...]:
    member_by_id = {member.id: member for member in plan.members}
    binding_by_member = {binding.member_id: binding for binding in plan.bindings}
    system_by_id = {system.id: system for system in plan.systems}
    claims: list[_ObservedClaim] = []
    for member in plan.members:
        binding = binding_by_member[member.id]
        if binding.kind is not BindingKind.EMITTED:
            continue
        assert binding.path_id is not None and binding.path_rank is not None
        attributed_member_ids = (member.id, *_covered_members(plan, member.id))
        connector_membership = {
            connector_id
            for member_id in attributed_member_ids
            for connector_id in member_by_id[member_id].connector_ids
        }
        connector_ids = tuple(
            connector_id
            for connector_id in system_by_id[member.system_id].connector_ids
            if connector_id in connector_membership
        )
        span = _connector_span(graph, connector_ids)
        segments = _maximal_axis_segments(
            apply_route_offsets(routes[binding.path_rank], station_offsets)
        )
        for segment in segments:
            if (
                segment.orientation is CorridorOrientation.HORIZONTAL
                and member.family_id is RouteFamilyId.MERGE_BRANCH
            ):
                continue
            corridor = _corridor_region(
                graph, segment, segments, span, connector_ids, member
            )
            if corridor is None:
                continue
            region, measurement_scope = corridor
            kind = (
                _horizontal_kind(segment, region, member)
                if segment.orientation is CorridorOrientation.HORIZONTAL
                else CorridorKind.INTER_COLUMN_CHANNEL
            )
            claims.append(
                _ObservedClaim(
                    member.system_id,
                    member,
                    connector_ids,
                    _sharing_ids(member),
                    RouteReservationClaim(
                        member.id,
                        binding.path_id,
                        binding.path_rank,
                        segment.rank,
                        segment.end_rank,
                        segment.span_start,
                        segment.span_end,
                        segment.coordinate,
                    ),
                    kind,
                    segment.orientation,
                    segment.direction,
                    region,
                    span,
                    measurement_scope,
                    segment.span_start,
                    segment.span_end,
                    segment.coordinate,
                )
            )
    return tuple(claims)


def _region_key(region: CorridorRegion) -> tuple[str, int, int]:
    if isinstance(region, RowGapRegion):
        return region.kind.value, region.upper_row, region.lower_row
    if isinstance(region, ColumnGapRegion):
        return region.kind.value, region.left_column, region.right_column
    return region.kind.value, list(CanvasSide).index(region.side), -1


_ClaimBaseKey: TypeAlias = tuple[
    RouteSystemId, str, str, str, tuple[str, int, int], str
]


def _claim_base_key(claim: _ObservedClaim) -> _ClaimBaseKey:
    return (
        claim.system_id,
        claim.orientation.value,
        claim.kind.value,
        claim.direction.value,
        _region_key(claim.region),
        claim.measurement_scope.value,
    )


def _group_claims(
    claims: tuple[_ObservedClaim, ...],
    system_rank: dict[RouteSystemId, int],
    member_rank: dict[EmissionMemberId, int],
) -> tuple[tuple[_ObservedClaim, ...], ...]:
    by_key: defaultdict[_ClaimBaseKey, list[_ObservedClaim]] = defaultdict(list)
    for claim in claims:
        by_key[_claim_base_key(claim)].append(claim)
    groups: list[tuple[_ObservedClaim, ...]] = []
    for key in sorted(by_key, key=lambda item: (system_rank[item[0]], item[1:])):
        ordered = sorted(
            by_key[key],
            key=lambda item: (
                item.travel_start,
                item.travel_end,
                member_rank[item.member.id],
                item.claim.segment_rank,
            ),
        )
        parent = list(range(len(ordered)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def join(first: int, second: int) -> None:
            first_root, second_root = root(first), root(second)
            if first_root != second_root:
                parent[second_root] = first_root

        sharing_sets = tuple(set(item.sharing_ids) for item in ordered)
        for first_rank, first in enumerate(ordered):
            for second_rank in range(first_rank + 1, len(ordered)):
                second = ordered[second_rank]
                if second.travel_start >= first.travel_end - COORD_TOLERANCE:
                    break
                if sharing_sets[first_rank].intersection(sharing_sets[second_rank]):
                    join(first_rank, second_rank)
        components: defaultdict[int, list[_ObservedClaim]] = defaultdict(list)
        for rank, claim in enumerate(ordered):
            components[root(rank)].append(claim)
        groups.extend(tuple(value) for value in components.values())
    return tuple(groups)


def _provenance(
    plan: RoutePlan,
    connector_ids: tuple[ConnectorId, ...],
    span: GridSpan,
) -> tuple[ReservationDecisionRef, ...]:
    return reservation_decision_refs(plan.provenance, connector_ids, span)


def _connector_endpoint_cells(
    plan: RoutePlan, connector_ids: tuple[ConnectorId, ...]
) -> dict[tuple[ConnectorId, ConnectorEndpointRole], GridCell]:
    connector_set = set(connector_ids)
    endpoint_sections = {
        (connector_id, group.role): group.section_id
        for group in plan.endpoint_groups
        for connector_id in group.connector_ids
        if connector_id in connector_set
    }
    grid_by_section = {item.section_id: item.grid for item in plan.provenance.sections}
    cells: dict[tuple[ConnectorId, ConnectorEndpointRole], GridCell] = {}
    for connector_id in connector_ids:
        for role in (ConnectorEndpointRole.EXIT, ConnectorEndpointRole.ENTRY):
            section_id = endpoint_sections.get((connector_id, role))
            if section_id is None:
                raise ValueError("reservation connector has incomplete endpoint facts")
            decision = grid_by_section.get(section_id)
            if decision is None:
                raise ValueError(
                    "reservation endpoint section has no settled grid fact"
                )
            cells[connector_id, role] = decision.value
    return cells


def _connector_span_from_plan(
    plan: RoutePlan, connector_ids: tuple[ConnectorId, ...]
) -> GridSpan:
    cells = tuple(_connector_endpoint_cells(plan, connector_ids).values())
    return GridSpan(
        min(column for column, _row, _row_span, _column_span in cells),
        max(column + column_span - 1 for column, _row, _row_span, column_span in cells),
        min(row for _column, row, _row_span, _column_span in cells),
        max(row + row_span - 1 for _column, row, row_span, _column_span in cells),
    )


def _topology_gap_regions_from_plan(
    plan: RoutePlan,
    connector_ids: tuple[ConnectorId, ...],
    orientation: CorridorOrientation,
) -> frozenset[RowGapRegion | ColumnGapRegion]:
    cells = _connector_endpoint_cells(plan, connector_ids)
    regions: set[RowGapRegion | ColumnGapRegion] = set()
    for connector_id in connector_ids:
        source = cells[connector_id, ConnectorEndpointRole.EXIT]
        target = cells[connector_id, ConnectorEndpointRole.ENTRY]
        if orientation is CorridorOrientation.HORIZONTAL:
            source_start, source_end = source[1], source[1] + source[2] - 1
            target_start, target_end = target[1], target[1] + target[2] - 1
            if source_start != target_start or source_end != target_end:
                lo, hi = min(source_start, target_start), max(source_end, target_end)
                regions.update(RowGapRegion(row, row + 1) for row in range(lo, hi))
        else:
            source_start, source_end = source[0], source[0] + source[3] - 1
            target_start, target_end = target[0], target[0] + target[3] - 1
            if source_start != target_start or source_end != target_end:
                lo, hi = min(source_start, target_start), max(source_end, target_end)
                regions.update(
                    ColumnGapRegion(column, column + 1) for column in range(lo, hi)
                )
    return frozenset(regions)


def _clearances(
    orientation: CorridorOrientation, region: CorridorRegion
) -> tuple[float, float, tuple[KeepOutClass, ...]]:
    if orientation is CorridorOrientation.HORIZONTAL:
        if isinstance(region, CanvasRegion) and region.side is CanvasSide.TOP:
            return (
                CURVE_RADIUS,
                INTER_ROW_HEADER_CLEARANCE,
                (
                    KeepOutClass.CANVAS,
                    KeepOutClass.HEADER,
                    KeepOutClass.SECTION,
                    KeepOutClass.LABEL,
                    KeepOutClass.MARKER,
                ),
            )
        if isinstance(region, CanvasRegion):
            return (
                INTER_ROW_EDGE_CLEARANCE,
                CURVE_RADIUS,
                (
                    KeepOutClass.SECTION,
                    KeepOutClass.HEADER,
                    KeepOutClass.LABEL,
                    KeepOutClass.MARKER,
                    KeepOutClass.CANVAS,
                ),
            )
        return (
            INTER_ROW_EDGE_CLEARANCE,
            INTER_ROW_HEADER_CLEARANCE,
            (
                KeepOutClass.SECTION,
                KeepOutClass.HEADER,
                KeepOutClass.LABEL,
                KeepOutClass.MARKER,
            ),
        )
    if isinstance(region, CanvasRegion) and region.side is CanvasSide.LEFT:
        return (
            CURVE_RADIUS,
            EDGE_TO_BUNDLE_CLEARANCE,
            (
                KeepOutClass.CANVAS,
                KeepOutClass.SECTION,
                KeepOutClass.LABEL,
                KeepOutClass.MARKER,
            ),
        )
    if isinstance(region, CanvasRegion):
        return (
            EDGE_TO_BUNDLE_CLEARANCE,
            CURVE_RADIUS,
            (
                KeepOutClass.SECTION,
                KeepOutClass.LABEL,
                KeepOutClass.MARKER,
                KeepOutClass.CANVAS,
            ),
        )
    return (
        EDGE_TO_BUNDLE_CLEARANCE,
        EDGE_TO_BUNDLE_CLEARANCE,
        (
            KeepOutClass.SECTION,
            KeepOutClass.LABEL,
            KeepOutClass.MARKER,
        ),
    )


def _union_span(claims: tuple[_ObservedClaim, ...]) -> GridSpan:
    return GridSpan(
        min(item.span.min_column for item in claims),
        max(item.span.max_column for item in claims),
        min(item.span.min_row for item in claims),
        max(item.span.max_row for item in claims),
    )


def _reservation_content_id(
    system_id: RouteSystemId,
    kind: CorridorKind,
    direction: Direction,
    region: CorridorRegion,
    measurement_scope: CorridorMeasurementScope,
    span: GridSpan,
    claimant_ids: tuple[EmissionMemberId, ...],
    claims: tuple[RouteReservationClaim, ...],
) -> RouteReservationId:
    return RouteReservationId(
        _stable_id(
            "route-reservation",
            system_id,
            kind.value,
            direction.value,
            *_region_key(region),
            measurement_scope.value,
            span.min_column,
            span.max_column,
            span.min_row,
            span.max_row,
            *claimant_ids,
            *(
                f"{claim.member_id}:{claim.segment_rank}:{claim.segment_end_rank}"
                for claim in claims
            ),
        )
    )


def _description(
    kind: CorridorKind, region: CorridorRegion, span: GridSpan, lane_count: int
) -> str:
    if isinstance(region, RowGapRegion):
        location = f"row gap {region.upper_row}/{region.lower_row}"
    elif isinstance(region, ColumnGapRegion):
        location = f"column gap {region.left_column}/{region.right_column}"
    else:
        location = f"{region.side.value} canvas corridor"
    return (
        f"{kind.value} in {location}, columns {span.min_column}-{span.max_column}, "
        f"rows {span.min_row}-{span.max_row}, {lane_count} lane"
        f"{'s' if lane_count != 1 else ''}"
    )


def _reservation_order_key(
    reservation: RouteReservation,
    system_rank: dict[RouteSystemId, int],
    member_rank: dict[EmissionMemberId, int],
) -> tuple[object, ...]:
    return (
        list(CorridorOrientation).index(reservation.orientation),
        _region_key(reservation.region),
        reservation.span.min_column,
        reservation.span.max_column,
        reservation.span.min_row,
        reservation.span.max_row,
        system_rank[reservation.system_id],
        min(member_rank[item] for item in reservation.claimant_member_ids),
        reservation.id,
    )


def _allocate_physical_lanes(
    claims: tuple[RouteReservationClaim, ...],
) -> tuple[tuple[RouteReservationLane, ...], float]:
    active_by_lane: list[list[RouteReservationClaim]] = []
    claim_indices_by_lane: list[list[int]] = []
    maximum_bundle_width = 0.0
    sweep = sorted(
        enumerate(claims),
        key=lambda item: (
            item[1].longitudinal_start,
            item[1].longitudinal_end,
            item[1].allocation_coordinate,
            item[0],
        ),
    )
    for claim_index, claim in sweep:
        for active in active_by_lane:
            active[:] = [
                item
                for item in active
                if item.longitudinal_end > claim.longitudinal_start + COORD_TOLERANCE
            ]
        lane_index = next(
            (
                index
                for index, active in enumerate(active_by_lane)
                if active
                and abs(active[0].allocation_coordinate - claim.allocation_coordinate)
                <= COORD_TOLERANCE
            ),
            None,
        )
        if lane_index is None:
            lane_index = next(
                (index for index, active in enumerate(active_by_lane) if not active),
                None,
            )
        if lane_index is None:
            lane_index = len(active_by_lane)
            active_by_lane.append([])
            claim_indices_by_lane.append([])
        active_by_lane[lane_index].append(claim)
        claim_indices_by_lane[lane_index].append(claim_index)
        occupied_coordinates = tuple(
            active[0].allocation_coordinate for active in active_by_lane if active
        )
        maximum_bundle_width = max(
            maximum_bundle_width,
            max(occupied_coordinates) - min(occupied_coordinates),
        )
    lanes = tuple(
        RouteReservationLane(tuple(sorted(indices)))
        for indices in claim_indices_by_lane
    )
    return lanes, maximum_bundle_width


def _build_symbolic_records(
    graph: MetroGraph,
    plan: RoutePlan,
    groups: tuple[tuple[_ObservedClaim, ...], ...],
) -> tuple[
    tuple[SharedReference, ...],
    tuple[SymbolicDemand, ...],
    tuple[RouteReservation, ...],
]:
    member_rank = {member.id: rank for rank, member in enumerate(plan.members)}
    system_rank = {system.id: rank for rank, system in enumerate(plan.systems)}
    system_by_id = {system.id: system for system in plan.systems}
    records: list[tuple[SharedReference, SymbolicDemand, RouteReservation]] = []
    for group in groups:
        first = group[0]
        if any(item.measurement_scope is not first.measurement_scope for item in group):
            raise ValueError("one reservation cannot mix blocker measurement scopes")
        system = system_by_id[first.system_id]
        claimant_set = {
            member_id
            for item in group
            for member_id in (item.member.id, *_covered_members(plan, item.member.id))
        }
        claimant_ids = tuple(
            member.id for member in plan.members if member.id in claimant_set
        )
        ordered_group = tuple(
            sorted(
                group,
                key=lambda item: (
                    member_rank[item.member.id],
                    item.claim.path_rank,
                    item.claim.segment_rank,
                ),
            )
        )
        claims = tuple(item.claim for item in ordered_group)
        lanes, bundle_width = _allocate_physical_lanes(claims)
        connector_set = {
            connector_id for item in group for connector_id in item.connector_ids
        }
        connector_ids = tuple(
            item for item in system.connector_ids if item in connector_set
        )
        span = _union_span(group)
        provenance = _provenance(plan, connector_ids, span)
        reservation_id = _reservation_content_id(
            first.system_id,
            first.kind,
            first.direction,
            first.region,
            first.measurement_scope,
            span,
            claimant_ids,
            claims,
        )
        reference_id = SharedReferenceId(
            _stable_id("corridor-reference", reservation_id)
        )
        demand_id = DemandId(_stable_id("corridor-demand", reservation_id))
        lane_count = len(lanes)
        negative, positive, keepouts = _clearances(first.orientation, first.region)
        families = tuple(
            family
            for family in RouteFamilyId
            if family
            in {
                item.member.family_id
                for item in group
                if item.member.family_id is not None
            }
        )
        reservation = RouteReservation(
            reservation_id,
            first.system_id,
            connector_ids,
            claimant_ids,
            claims,
            first.kind,
            first.orientation,
            first.direction,
            first.region,
            span,
            first.measurement_scope,
            lanes,
            lane_count,
            bundle_width,
            BUNDLE_TO_BUNDLE_CLEARANCE,
            negative,
            positive,
            negative + bundle_width + positive,
            None,
            keepouts,
            families,
            reference_id,
            (demand_id,),
            provenance,
            _description(first.kind, first.region, span, lane_count),
        )
        reference = SharedReference(
            reference_id,
            first.system_id,
            SharedReferenceKind.BAND,
            claimant_ids,
            CoordinateRegime.SETTLED_GRID,
            provenance,
        )
        demand = SymbolicDemand(
            demand_id,
            first.system_id,
            claimant_ids,
            DemandKind.LANES,
            DemandAxis.Y
            if first.orientation is CorridorOrientation.HORIZONTAL
            else DemandAxis.X,
            span,
            lane_count,
            reservation.minimum_width,
            CoordinateRegime.LAYOUT_CANVAS,
            (reference_id,),
            keepouts,
            provenance,
        )
        records.append((reference, demand, reservation))
    records.sort(
        key=lambda item: _reservation_order_key(item[2], system_rank, member_rank)
    )
    return (
        tuple(item[0] for item in records),
        tuple(item[1] for item in records),
        tuple(item[2] for item in records),
    )


def _canvas_region_measurement(
    graph: MetroGraph,
    region: CanvasRegion,
    longitudinal_start: float,
    longitudinal_end: float,
    canvas_width: float,
    canvas_height: float,
) -> _RegionMeasurement:
    if region.side in {CanvasSide.TOP, CanvasSide.BOTTOM}:
        sections = tuple(
            section
            for section in graph.sections.values()
            if section.bbox_w > 0
            and _section_x_overlaps(section, longitudinal_start, longitudinal_end)
        )
        if region.side is CanvasSide.TOP:
            end, positive = _edge_blockers(
                (
                    (f"{SECTION_HEADER_BLOCKER}:{section.id}", section.bbox_y)
                    for section in sections
                ),
                maximum=False,
            )
            return _RegionMeasurement(0.0, end, ("canvas:top",), positive)
        start, negative = _edge_blockers(
            (
                (
                    f"{SECTION_BOTTOM_BLOCKER}:{section.id}",
                    section.bbox_y + section.bbox_h,
                )
                for section in sections
            ),
            maximum=True,
        )
        return _RegionMeasurement(start, canvas_height, negative, ("canvas:bottom",))
    sections = tuple(
        section
        for section in graph.sections.values()
        if section.bbox_h > 0
        and _section_y_overlaps(section, longitudinal_start, longitudinal_end)
    )
    if region.side is CanvasSide.LEFT:
        end, positive = _edge_blockers(
            (
                (f"{SECTION_LEFT_BLOCKER}:{section.id}", section.bbox_x)
                for section in sections
            ),
            maximum=False,
        )
        return _RegionMeasurement(0.0, end, ("canvas:left",), positive)
    start, negative = _edge_blockers(
        (
            (f"{SECTION_RIGHT_BLOCKER}:{section.id}", section.bbox_x + section.bbox_w)
            for section in sections
        ),
        maximum=True,
    )
    return _RegionMeasurement(start, canvas_width, negative, ("canvas:right",))


@dataclass(frozen=True, slots=True)
class _ProjectedClaimBounds:
    allocation_axis: DemandAxis
    longitudinal_axis: DemandAxis
    longitudinal_start: float
    longitudinal_end: float
    occupied_start: float
    occupied_end: float


def _projected_claim_bounds(
    reservation: RouteReservation,
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> _ProjectedClaimBounds:
    allocation_axis, longitudinal_axis = _reservation_axes(reservation)
    projected_claims = tuple(
        (
            project_reservation_coordinate(
                item.longitudinal_start,
                longitudinal_axis,
                item.member_id,
                translations,
            ),
            project_reservation_coordinate(
                item.longitudinal_end,
                longitudinal_axis,
                item.member_id,
                translations,
            ),
            project_reservation_coordinate(
                item.allocation_coordinate,
                allocation_axis,
                item.member_id,
                translations,
            ),
        )
        for item in reservation.claims
    )
    return _ProjectedClaimBounds(
        allocation_axis,
        longitudinal_axis,
        min(item[0] for item in projected_claims),
        max(item[1] for item in projected_claims),
        min(item[2] for item in projected_claims),
        max(item[2] for item in projected_claims),
    )


def _realise_one(
    graph: MetroGraph,
    reservation: RouteReservation,
    canvas_width: float | None,
    canvas_height: float | None,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> RealisedRouteReservation | None:
    if isinstance(reservation.region, CanvasRegion) and (
        canvas_width is None or canvas_height is None
    ):
        return None
    projected = _projected_claim_bounds(reservation, coordinate_translations)
    longitudinal_start = projected.longitudinal_start
    longitudinal_end = projected.longitudinal_end
    occupied_start = projected.occupied_start
    occupied_end = projected.occupied_end
    if isinstance(reservation.region, RowGapRegion):
        measurement = _row_region_measurement(
            graph,
            reservation.region,
            reservation.measurement_scope,
            reservation.span,
            longitudinal_start,
            longitudinal_end,
        )
    elif isinstance(reservation.region, ColumnGapRegion):
        measurement = _column_region_measurement(
            graph,
            reservation.region,
            reservation.measurement_scope,
            reservation.span,
            longitudinal_start,
            longitudinal_end,
        )
    else:
        assert canvas_width is not None and canvas_height is not None
        measurement = _canvas_region_measurement(
            graph,
            reservation.region,
            longitudinal_start,
            longitudinal_end,
            canvas_width,
            canvas_height,
        )
    available = measurement.end - measurement.start
    return RealisedRouteReservation(
        reservation.id,
        projected.allocation_axis,
        projected.longitudinal_axis,
        (occupied_start + occupied_end) / 2,
        longitudinal_start,
        longitudinal_end,
        measurement.start,
        measurement.end,
        occupied_start,
        occupied_end,
        available,
        reservation.minimum_width,
        available - reservation.minimum_width,
        occupied_start - (measurement.start + reservation.negative_side_clearance),
        measurement.end - reservation.positive_side_clearance - occupied_end,
        measurement.negative_blocker_ids,
        measurement.positive_blocker_ids,
        CoordinateRegime.LAYOUT_CANVAS,
        coordinate_translations,
    )


def realise_reservation(
    graph: MetroGraph,
    reservation: RouteReservation,
    *,
    canvas_width: float | None = None,
    canvas_height: float | None = None,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> RealisedRouteReservation | None:
    """Measure one reservation against *graph* as it currently stands.

    Callers that settle geometry re-measure between writes, so the measurement
    has to read live section envelopes rather than a frozen realisation.
    A claim observed before a settlement translation is projected through
    *coordinate_translations* so it measures the geometry it now describes.
    Returns ``None`` for a canvas-side corridor when no canvas bounds are known.
    """
    return _realise_one(
        graph,
        reservation,
        canvas_width,
        canvas_height,
        coordinate_translations,
    )


def _realise_all(
    graph: MetroGraph,
    reservations: tuple[RouteReservation, ...],
    canvas_width: float | None,
    canvas_height: float | None,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> tuple[RealisedRouteReservation, ...]:
    realised = (
        _realise_one(
            graph,
            reservation,
            canvas_width,
            canvas_height,
            ()
            if isinstance(reservation.region, CanvasRegion)
            else coordinate_translations,
        )
        for reservation in reservations
    )
    return tuple(item for item in realised if item is not None)


def _diagnostics(
    reservations: tuple[RouteReservation, ...],
    realised: tuple[RealisedRouteReservation, ...],
) -> tuple[RouteReservationDiagnostic, ...]:
    reservation_by_id = {item.id: item for item in reservations}
    diagnostics: list[RouteReservationDiagnostic] = []
    for item in realised:
        if (
            min(item.capacity_slack, item.negative_side_slack, item.positive_side_slack)
            >= -COORD_TOLERANCE
        ):
            continue
        reservation = reservation_by_id[item.reservation_id]
        diagnostics.append(
            RouteReservationDiagnostic(
                item.reservation_id,
                reservation.claimant_member_ids,
                "reservation-deficit",
                _diagnostic_message(reservation, item),
                item.capacity_slack,
                item.negative_side_slack,
                item.positive_side_slack,
            )
        )
    return tuple(diagnostics)


def _diagnostic_message(
    reservation: RouteReservation, realised: RealisedRouteReservation
) -> str:
    return (
        f"{reservation.description}: available {realised.available_width:.2f}px, "
        f"required {realised.required_width:.2f}px, capacity slack "
        f"{realised.capacity_slack:.2f}px, side slacks "
        f"{realised.negative_side_slack:.2f}px and "
        f"{realised.positive_side_slack:.2f}px"
    )


@dataclass(frozen=True, slots=True)
class ReservationQueryIndexes:
    references: dict[SharedReferenceId, SharedReference]
    demands: dict[DemandId, SymbolicDemand]
    reservations: dict[RouteReservationId, RouteReservation]
    realisations: dict[RouteReservationId, RealisedRouteReservation]
    by_system: dict[RouteSystemId, list[RouteReservation]]
    by_member: dict[EmissionMemberId, list[RouteReservation]]


def _same_measurement(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-9)


def _reservation_axes(
    reservation: RouteReservation,
) -> tuple[DemandAxis, DemandAxis]:
    if reservation.orientation is CorridorOrientation.HORIZONTAL:
        return DemandAxis.Y, DemandAxis.X
    return DemandAxis.X, DemandAxis.Y


def _validate_reservation_record(
    plan: RoutePlan,
    reservation: RouteReservation,
    systems: Mapping[RouteSystemId, RouteSystem],
    members: Mapping[EmissionMemberId, EmissionMember],
    bindings: Mapping[EmissionMemberId, Sequence[EmissionBinding]],
    references: Mapping[SharedReferenceId, SharedReference],
    demands: Mapping[DemandId, SymbolicDemand],
) -> None:
    system = systems.get(reservation.system_id)
    if system is None:
        raise ValueError("reservation has an unknown route system")
    if any(item not in system.connector_ids for item in reservation.connector_ids):
        raise ValueError("reservation has a connector outside its route system")
    if any(item not in members for item in reservation.claimant_member_ids):
        raise ValueError("reservation has an unknown claimant member")
    if any(
        members[item].system_id != reservation.system_id
        for item in reservation.claimant_member_ids
    ):
        raise ValueError("reservation claimant belongs to another route system")
    if any(
        claim.member_id not in reservation.claimant_member_ids
        for claim in reservation.claims
    ):
        raise ValueError("reservation claim is absent from its claimant list")
    if len(
        {
            (item.path_id, item.segment_rank, item.segment_end_rank)
            for item in reservation.claims
        }
    ) != len(reservation.claims):
        raise ValueError("reservation contains duplicate emitted claims")
    for claim in reservation.claims:
        member_bindings = bindings.get(claim.member_id, ())
        if len(member_bindings) != 1:
            raise ValueError("reservation claim has inconsistent emission coverage")
        binding = member_bindings[0]
        if (
            binding.kind is not BindingKind.EMITTED
            or binding.path_id != claim.path_id
            or binding.path_rank != claim.path_rank
        ):
            raise ValueError("reservation claim disagrees with its emitted binding")

    expected_lanes, expected_bundle_width = _allocate_physical_lanes(reservation.claims)
    if reservation.lanes != expected_lanes or not _same_measurement(
        reservation.bundle_width, expected_bundle_width
    ):
        raise ValueError("reservation physical lanes are inconsistent")

    member_rank = {member_id: rank for rank, member_id in enumerate(members)}
    expected_claims = tuple(
        sorted(
            reservation.claims,
            key=lambda item: (
                member_rank[item.member_id],
                item.path_rank,
                item.segment_rank,
            ),
        )
    )
    if reservation.claims != expected_claims:
        raise ValueError("reservation claims are not in canonical member order")

    expected_span = _connector_span_from_plan(plan, reservation.connector_ids)
    if reservation.span != expected_span:
        raise ValueError(
            "reservation span is inconsistent with its connector endpoints"
        )
    if (
        reservation.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
        and reservation.region
        not in _topology_gap_regions_from_plan(
            plan, reservation.connector_ids, reservation.orientation
        )
    ):
        raise ValueError("reservation region is not crossed by its connectors")

    expected_negative, expected_positive, expected_keepouts = _clearances(
        reservation.orientation, reservation.region
    )
    expected_minimum = expected_negative + expected_bundle_width + expected_positive
    if (
        not _same_measurement(reservation.negative_side_clearance, expected_negative)
        or not _same_measurement(reservation.positive_side_clearance, expected_positive)
        or not _same_measurement(reservation.peer_clearance, BUNDLE_TO_BUNDLE_CLEARANCE)
        or not _same_measurement(reservation.minimum_width, expected_minimum)
        or reservation.preferred_width is not None
        or reservation.keep_out_classes != expected_keepouts
    ):
        raise ValueError("reservation clearance policy is inconsistent")

    expected_families = tuple(
        family
        for family in RouteFamilyId
        if family
        in {
            members[claim.member_id].family_id
            for claim in reservation.claims
            if members[claim.member_id].family_id is not None
        }
    )
    if reservation.route_family_ids != expected_families:
        raise ValueError("reservation route-family attribution is inconsistent")

    direct_claimants = {item.member_id for item in reservation.claims}
    expected_claimant_set = direct_claimants | {
        member_id
        for member_id, member_bindings in bindings.items()
        if len(member_bindings) == 1
        and member_bindings[0].covering_member_id in direct_claimants
    }
    expected_claimants = tuple(
        member_id for member_id in members if member_id in expected_claimant_set
    )
    if reservation.claimant_member_ids != expected_claimants:
        raise ValueError("reservation claimant list disagrees with emission coverage")
    expected_connector_set = {
        connector_id
        for member_id in expected_claimants
        for connector_id in members[member_id].connector_ids
    }
    expected_connectors = tuple(
        item for item in system.connector_ids if item in expected_connector_set
    )
    if reservation.connector_ids != expected_connectors:
        raise ValueError("reservation connector attribution is incomplete")

    reference = references.get(reservation.reference_id)
    if reference is None:
        raise ValueError("reservation has an unknown shared reference")
    if any(demand_id not in demands for demand_id in reservation.demand_ids):
        raise ValueError("reservation has an unknown symbolic demand")

    expected_reservation_id = _reservation_content_id(
        reservation.system_id,
        reservation.kind,
        reservation.direction,
        reservation.region,
        reservation.measurement_scope,
        reservation.span,
        reservation.claimant_member_ids,
        reservation.claims,
    )
    expected_reference_id = SharedReferenceId(
        _stable_id("corridor-reference", expected_reservation_id)
    )
    expected_demand_id = DemandId(
        _stable_id("corridor-demand", expected_reservation_id)
    )
    if (
        reservation.id != expected_reservation_id
        or reservation.reference_id != expected_reference_id
        or reservation.demand_ids != (expected_demand_id,)
        or reservation.description
        != _description(
            reservation.kind,
            reservation.region,
            reservation.span,
            reservation.lane_count,
        )
    ):
        raise ValueError("reservation canonical identity is inconsistent")

    expected_provenance = _provenance(plan, reservation.connector_ids, reservation.span)
    if reservation.provenance != expected_provenance:
        raise ValueError("reservation provenance is inconsistent with the route plan")
    if (
        reference.system_id != reservation.system_id
        or reference.claimant_member_ids != reservation.claimant_member_ids
        or reference.kind is not SharedReferenceKind.BAND
        or reference.coordinate_regime is not CoordinateRegime.SETTLED_GRID
        or reference.provenance != reservation.provenance
    ):
        raise ValueError("reservation shared reference is inconsistent")

    allocation_axis, _longitudinal_axis = _reservation_axes(reservation)
    for demand_id in reservation.demand_ids:
        demand = demands[demand_id]
        if (
            demand.system_id != reservation.system_id
            or demand.claimant_member_ids != reservation.claimant_member_ids
            or demand.span != reservation.span
            or demand.ordered_reference_ids != (reservation.reference_id,)
            or demand.kind is not DemandKind.LANES
            or demand.axis is not allocation_axis
            or demand.lane_count != reservation.lane_count
            or demand.minimum_size is None
            or not _same_measurement(demand.minimum_size, reservation.minimum_width)
            or demand.minimum_size_regime is not CoordinateRegime.LAYOUT_CANVAS
            or demand.keep_out_classes != reservation.keep_out_classes
            or demand.provenance != reservation.provenance
        ):
            raise ValueError("reservation symbolic demand is inconsistent")


def _section_grid_cells(plan: RoutePlan) -> dict[str, GridCell]:
    return {
        item.section_id: item.grid.value
        for item in plan.provenance.sections
        if item.grid is not None
    }


def _valid_section_blockers(
    plan: RoutePlan,
    reservation: RouteReservation,
    blocker_ids: tuple[str, ...],
    *,
    prefix: str,
    negative_side: bool,
) -> bool:
    cells = _section_grid_cells(plan)
    if len(set(blocker_ids)) != len(blocker_ids):
        return False
    for blocker_id in blocker_ids:
        marker = f"{prefix}:"
        if not blocker_id.startswith(marker):
            return False
        cell = cells.get(blocker_id[len(marker) :])
        if cell is None:
            return False
        column, row, row_span, column_span = cell
        row_end = row + row_span - 1
        column_end = column + column_span - 1
        if isinstance(reservation.region, RowGapRegion):
            correct_side = (
                row <= reservation.region.upper_row
                if negative_side
                else row_end >= reservation.region.lower_row
            )
            relevant_span = (
                reservation.span.min_column <= column_end
                and column <= reservation.span.max_column
            )
        elif isinstance(reservation.region, ColumnGapRegion):
            correct_side = (
                column <= reservation.region.left_column
                if negative_side
                else column_end >= reservation.region.right_column
            )
            relevant_span = (
                reservation.span.min_row <= row_end and row <= reservation.span.max_row
            )
        else:
            correct_side = True
            relevant_span = True
        if not correct_side or (
            reservation.measurement_scope is CorridorMeasurementScope.TOPOLOGY_SPAN
            and not relevant_span
        ):
            return False
    return True


def _validate_blocker_ids(
    plan: RoutePlan,
    reservation: RouteReservation,
    realised: RealisedRouteReservation,
) -> None:
    region = reservation.region
    if isinstance(region, RowGapRegion):
        valid = _valid_section_blockers(
            plan,
            reservation,
            realised.negative_blocker_ids,
            prefix=SECTION_BOTTOM_BLOCKER,
            negative_side=True,
        ) and _valid_section_blockers(
            plan,
            reservation,
            realised.positive_blocker_ids,
            prefix=SECTION_HEADER_BLOCKER,
            negative_side=False,
        )
    elif isinstance(region, ColumnGapRegion):
        valid = _valid_section_blockers(
            plan,
            reservation,
            realised.negative_blocker_ids,
            prefix=SECTION_RIGHT_BLOCKER,
            negative_side=True,
        ) and _valid_section_blockers(
            plan,
            reservation,
            realised.positive_blocker_ids,
            prefix=SECTION_LEFT_BLOCKER,
            negative_side=False,
        )
    elif region.side is CanvasSide.TOP:
        valid = realised.negative_blocker_ids == (
            "canvas:top",
        ) and _valid_section_blockers(
            plan,
            reservation,
            realised.positive_blocker_ids,
            prefix=SECTION_HEADER_BLOCKER,
            negative_side=False,
        )
    elif region.side is CanvasSide.BOTTOM:
        valid = _valid_section_blockers(
            plan,
            reservation,
            realised.negative_blocker_ids,
            prefix=SECTION_BOTTOM_BLOCKER,
            negative_side=True,
        ) and realised.positive_blocker_ids == ("canvas:bottom",)
    elif region.side is CanvasSide.LEFT:
        valid = realised.negative_blocker_ids == (
            "canvas:left",
        ) and _valid_section_blockers(
            plan,
            reservation,
            realised.positive_blocker_ids,
            prefix=SECTION_LEFT_BLOCKER,
            negative_side=False,
        )
    else:
        valid = _valid_section_blockers(
            plan,
            reservation,
            realised.negative_blocker_ids,
            prefix=SECTION_RIGHT_BLOCKER,
            negative_side=True,
        ) and realised.positive_blocker_ids == ("canvas:right",)
    if not valid:
        raise ValueError("realised reservation has invalid boundary blocker ids")


def _validate_reservation_realisation(
    plan: RoutePlan,
    reservation: RouteReservation,
    realised: RealisedRouteReservation,
) -> None:
    allocation_axis, longitudinal_axis = _reservation_axes(reservation)
    expected_bounds = _projected_claim_bounds(
        reservation, realised.coordinate_translations
    )
    expected_longitudinal_start = expected_bounds.longitudinal_start
    expected_longitudinal_end = expected_bounds.longitudinal_end
    expected_occupied_start = expected_bounds.occupied_start
    expected_occupied_end = expected_bounds.occupied_end
    expected_capacity = realised.available_width - reservation.minimum_width
    expected_negative = realised.occupied_start - (
        realised.region_start + reservation.negative_side_clearance
    )
    expected_positive = (
        realised.region_end
        - reservation.positive_side_clearance
        - realised.occupied_end
    )
    expected_coordinate = (realised.occupied_start + realised.occupied_end) / 2
    if (
        realised.allocation_axis is not allocation_axis
        or realised.longitudinal_axis is not longitudinal_axis
        or realised.coordinate_regime is not CoordinateRegime.LAYOUT_CANVAS
        or not _same_measurement(realised.required_width, reservation.minimum_width)
        or not _same_measurement(realised.capacity_slack, expected_capacity)
        or not _same_measurement(realised.negative_side_slack, expected_negative)
        or not _same_measurement(realised.positive_side_slack, expected_positive)
        or not _same_measurement(realised.coordinate, expected_coordinate)
        or not _same_measurement(
            realised.longitudinal_start, expected_longitudinal_start
        )
        or not _same_measurement(realised.longitudinal_end, expected_longitudinal_end)
        or not _same_measurement(realised.occupied_start, expected_occupied_start)
        or not _same_measurement(realised.occupied_end, expected_occupied_end)
    ):
        raise ValueError("realised reservation is inconsistent with its reservation")
    _validate_blocker_ids(plan, reservation, realised)


def _validate_reservation_diagnostics(
    plan: RoutePlan,
    reservations: Mapping[RouteReservationId, RouteReservation],
    realisations: Mapping[RouteReservationId, RealisedRouteReservation],
) -> None:
    diagnostic_ids = tuple(item.reservation_id for item in plan.reservation_diagnostics)
    if len(set(diagnostic_ids)) != len(diagnostic_ids):
        raise ValueError("route plan contains duplicate reservation diagnostics")
    if any(item not in reservations for item in diagnostic_ids):
        raise ValueError("reservation diagnostic has an unknown reservation")
    expected_ids = tuple(
        reservation.id
        for reservation in plan.reservations
        if (
            (realised := realisations.get(reservation.id)) is not None
            and min(
                realised.capacity_slack,
                realised.negative_side_slack,
                realised.positive_side_slack,
            )
            < -COORD_TOLERANCE
        )
    )
    if diagnostic_ids != expected_ids:
        raise ValueError("reservation diagnostics are not in reservation order")
    for diagnostic in plan.reservation_diagnostics:
        reservation = reservations[diagnostic.reservation_id]
        realised = realisations[diagnostic.reservation_id]
        if (
            diagnostic.claimant_member_ids != reservation.claimant_member_ids
            or diagnostic.code != "reservation-deficit"
            or diagnostic.message != _diagnostic_message(reservation, realised)
            or not _same_measurement(diagnostic.capacity_slack, realised.capacity_slack)
            or not _same_measurement(
                diagnostic.negative_side_slack, realised.negative_side_slack
            )
            or not _same_measurement(
                diagnostic.positive_side_slack, realised.positive_side_slack
            )
        ):
            raise ValueError("reservation diagnostic is inconsistent")


def _validate_system_reservation_indexes(plan: RoutePlan) -> None:
    for system in plan.systems:
        expected_references = tuple(
            item.id for item in plan.shared_references if item.system_id == system.id
        )
        expected_demands = tuple(
            item.id for item in plan.demands if item.system_id == system.id
        )
        expected_reservations = tuple(
            item.id for item in plan.reservations if item.system_id == system.id
        )
        if system.shared_reference_ids != expected_references:
            raise ValueError("route system shared-reference index is inconsistent")
        if system.demand_ids != expected_demands:
            raise ValueError("route system demand index is inconsistent")
        if system.reservation_ids != expected_reservations:
            raise ValueError("route system reservation index is inconsistent")


def build_reservation_query_indexes(
    plan: RoutePlan,
    members: Mapping[EmissionMemberId, EmissionMember],
    bindings: Mapping[EmissionMemberId, Sequence[EmissionBinding]],
) -> ReservationQueryIndexes:
    systems = {system.id: system for system in plan.systems}
    references = {item.id: item for item in plan.shared_references}
    demands = {item.id: item for item in plan.demands}
    reservations = {item.id: item for item in plan.reservations}
    realisations = {item.reservation_id: item for item in plan.realised_reservations}
    for label, index, records in (
        ("route system", systems, plan.systems),
        ("shared reference", references, plan.shared_references),
        ("symbolic demand", demands, plan.demands),
        ("route reservation", reservations, plan.reservations),
        ("realised reservation", realisations, plan.realised_reservations),
    ):
        if len(index) != len(records):
            raise ValueError(f"route plan contains duplicate {label} ids")
    by_system: dict[RouteSystemId, list[RouteReservation]] = defaultdict(list)
    by_member: dict[EmissionMemberId, list[RouteReservation]] = defaultdict(list)
    for reservation in plan.reservations:
        _validate_reservation_record(
            plan,
            reservation,
            systems,
            members,
            bindings,
            references,
            demands,
        )
        by_system[reservation.system_id].append(reservation)
        for member_id in reservation.claimant_member_ids:
            by_member[member_id].append(reservation)

    system_rank = {system.id: rank for rank, system in enumerate(plan.systems)}
    member_rank = {member.id: rank for rank, member in enumerate(plan.members)}
    expected_reservation_order = tuple(
        sorted(
            plan.reservations,
            key=lambda item: _reservation_order_key(item, system_rank, member_rank),
        )
    )
    if plan.reservations != expected_reservation_order:
        raise ValueError("route reservations are not in canonical order")

    exit_turn_reference_ids = tuple(
        item.reference_id
        for item in plan.exit_turn_plans
        if item.reference_id is not None
    )
    reservation_reference_ids = tuple(item.reference_id for item in plan.reservations)
    reservation_reference_set = set(reservation_reference_ids)
    planner_references = tuple(
        item
        for item in plan.shared_references
        if item.id not in reservation_reference_set
    )
    expected_reference_ids = (
        tuple(item.id for item in planner_references) + reservation_reference_ids
    )
    if tuple(references) != expected_reference_ids:
        raise ValueError("route plan contains unlinked shared references")
    exit_turn_demand_ids = tuple(
        demand_id for item in plan.exit_turn_plans for demand_id in item.demand_ids
    )
    reservation_demand_ids = tuple(
        demand_id for item in plan.reservations for demand_id in item.demand_ids
    )
    reservation_demand_set = set(reservation_demand_ids)
    planner_demands = tuple(
        item for item in plan.demands if item.id not in reservation_demand_set
    )
    expected_demand_ids = (
        tuple(item.id for item in planner_demands) + reservation_demand_ids
    )
    if tuple(demands) != expected_demand_ids:
        raise ValueError("route plan contains unlinked symbolic demands")

    exit_turn_reference_set = set(exit_turn_reference_ids)
    exit_turn_demand_set = set(exit_turn_demand_ids)
    fan_reference_ids = tuple(
        item.centreline_reference_id
        for item in plan.fan_plans
        if item.system_id is not None and item.centreline_reference_id is not None
    )
    fan_demand_ids = tuple(
        demand_id
        for item in plan.fan_plans
        if item.system_id is not None
        for demand_id in item.demand_ids
    )
    convergence_reference_ids = tuple(
        reference_id
        for item in plan.convergence_plans
        for reference_id in item.shared_reference_ids
    )
    convergence_demand_ids = tuple(
        demand_id for item in plan.convergence_plans for demand_id in item.demand_ids
    )
    if tuple(item.id for item in planner_references) != (
        *exit_turn_reference_ids,
        *fan_reference_ids,
        *convergence_reference_ids,
    ):
        raise ValueError("planner shared-reference ownership is inconsistent")
    if tuple(item.id for item in planner_demands) != (
        *exit_turn_demand_ids,
        *fan_demand_ids,
        *convergence_demand_ids,
    ):
        raise ValueError("planner symbolic-demand ownership is inconsistent")
    if exit_turn_reference_set.intersection(fan_reference_ids):
        raise ValueError("exit-turn and fan plans share a reference id")
    if exit_turn_demand_set.intersection(fan_demand_ids):
        raise ValueError("exit-turn and fan plans share a demand id")
    if set(convergence_reference_ids).intersection(
        (*exit_turn_reference_ids, *fan_reference_ids)
    ):
        raise ValueError("convergence plans share a reference id with another planner")
    if set(convergence_demand_ids).intersection(
        (*exit_turn_demand_ids, *fan_demand_ids)
    ):
        raise ValueError("convergence plans share a demand id with another planner")
    fan_references = tuple(references[item] for item in fan_reference_ids)
    fan_demands = tuple(demands[item] for item in fan_demand_ids)
    fan_reference_id_set = set(fan_reference_ids)
    linked_fan_reference_ids: set[SharedReferenceId] = set()
    for reference in fan_references:
        if (
            reference.system_id not in systems
            or reference.kind is not SharedReferenceKind.CENTRELINE
            or reference.coordinate_regime is not CoordinateRegime.RELATIVE_FRAME
            or any(
                member_id not in members
                or members[member_id].system_id != reference.system_id
                for member_id in reference.claimant_member_ids
            )
        ):
            raise ValueError("fan shared reference is inconsistent")
    for demand in fan_demands:
        if len(demand.ordered_reference_ids) == 1:
            linked_fan_reference_ids.add(demand.ordered_reference_ids[0])
        demand_reference = (
            references.get(demand.ordered_reference_ids[0])
            if len(demand.ordered_reference_ids) == 1
            else None
        )
        if (
            demand_reference is None
            or demand_reference.id not in fan_reference_id_set
            or demand.system_id != demand_reference.system_id
            or demand.kind is not DemandKind.RUNWAY
            or demand.axis not in {DemandAxis.X, DemandAxis.Y}
            or demand.lane_count <= 0
            or demand.minimum_size is None
            or not math.isfinite(demand.minimum_size)
            or demand.minimum_size <= 0
            or demand.minimum_size_regime is not CoordinateRegime.RELATIVE_FRAME
            or demand.keep_out_classes != (KeepOutClass.SECTION, KeepOutClass.MARKER)
            or demand.provenance != demand_reference.provenance
            or any(
                member_id not in demand_reference.claimant_member_ids
                for member_id in demand.claimant_member_ids
            )
        ):
            raise ValueError("fan symbolic demand is inconsistent")
    if linked_fan_reference_ids != fan_reference_id_set:
        raise ValueError("route plan contains unlinked fan shared references")

    _validate_system_reservation_indexes(plan)
    if set(realisations).difference(reservations):
        raise ValueError("realisation has an unknown route reservation")
    if any(
        not isinstance(reservation.region, CanvasRegion)
        and reservation.id not in realisations
        for reservation in plan.reservations
    ):
        raise ValueError("non-canvas reservation is missing its realisation")
    realised_order = tuple(item.reservation_id for item in plan.realised_reservations)
    expected_realised_order = tuple(
        item.id for item in plan.reservations if item.id in realisations
    )
    if realised_order != expected_realised_order:
        raise ValueError("realised reservations are not in reservation order")
    for reservation_id, realised in realisations.items():
        _validate_reservation_realisation(plan, reservations[reservation_id], realised)
    _validate_reservation_diagnostics(plan, reservations, realisations)
    return ReservationQueryIndexes(
        references, demands, reservations, realisations, by_system, by_member
    )


def expected_exit_turn_foreign_references(
    plan: RoutePlan,
) -> dict[ExitTurnPlanId, tuple[SharedReferenceId, ...]]:
    """Return canonical cross-system axis and corridor-band conflicts."""
    planned = tuple(
        item
        for item in plan.exit_turn_plans
        if item.disposition is ExitTurnDisposition.PLANNED
        and item.axes
        and item.reference_id is not None
        and item.demand_ids
    )
    demands = {item.id: item for item in plan.demands}
    spans = {item.id: demands[item.demand_ids[0]].span for item in planned}
    foreign: defaultdict[ExitTurnPlanId, list[SharedReferenceId]] = defaultdict(list)
    for rank, first in enumerate(planned):
        assert first.reference_id is not None
        for second in planned[rank + 1 :]:
            assert second.reference_id is not None
            if (
                first.system_id != second.system_id
                and first.source_axis is second.source_axis
                and spans[first.id].overlaps(spans[second.id])
                and any(
                    abs(left.coordinate - right.coordinate)
                    < max(first.spacing, second.spacing)
                    for left in first.axes
                    for right in second.axes
                )
            ):
                foreign[first.id].append(second.reference_id)
                foreign[second.id].append(first.reference_id)
    for exit_turn_plan in planned:
        expected_orientation = (
            CorridorOrientation.VERTICAL
            if exit_turn_plan.source_axis is DemandAxis.X
            else CorridorOrientation.HORIZONTAL
        )
        for reservation in plan.reservations:
            if (
                reservation.system_id == exit_turn_plan.system_id
                or reservation.orientation is not expected_orientation
                or not spans[exit_turn_plan.id].overlaps(reservation.span)
                or not any(
                    abs(axis.coordinate - claim.allocation_coordinate)
                    < max(exit_turn_plan.spacing, reservation.peer_clearance)
                    for axis in exit_turn_plan.axes
                    for claim in reservation.claims
                )
            ):
                continue
            if reservation.reference_id not in foreign[exit_turn_plan.id]:
                foreign[exit_turn_plan.id].append(reservation.reference_id)
    return {item.id: tuple(foreign[item.id]) for item in plan.exit_turn_plans}


def expected_convergence_foreign_references(
    plan: RoutePlan,
) -> dict[ConvergencePlanId, tuple[SharedReferenceId, ...]]:
    """Return canonical planner and corridor conflicts for each convergence."""
    planned = tuple(
        item
        for item in plan.convergence_plans
        if item.disposition is ConvergenceDisposition.PLANNED
        and item.trunk_axis is not None
        and item.demand_ids
        and item.shared_reference_ids
    )
    demands = {item.id: item for item in plan.demands}
    spans = {item.id: demands[item.demand_ids[0]].span for item in planned}
    foreign: defaultdict[ConvergencePlanId, dict[SharedReferenceId, None]] = (
        defaultdict(dict)
    )

    def add(plan_id: ConvergencePlanId, reference_id: SharedReferenceId) -> None:
        foreign[plan_id].setdefault(reference_id, None)

    for rank, first in enumerate(planned):
        assert first.trunk_axis is not None
        for second in planned[rank + 1 :]:
            assert second.trunk_axis is not None
            if (
                first.system_id != second.system_id
                and first.trunk_axis.axis is second.trunk_axis.axis
                and spans[first.id].overlaps(spans[second.id])
                and abs(first.trunk_axis.coordinate - second.trunk_axis.coordinate)
                < CURVE_RADIUS
            ):
                add(first.id, second.shared_reference_ids[0])
                add(second.id, first.shared_reference_ids[0])

    for convergence in planned:
        assert convergence.trunk_axis is not None
        span = spans[convergence.id]
        trunk = convergence.trunk_axis
        endpoint_coordinates = tuple(
            value
            for value in (
                trunk.source_endpoint_coordinate,
                trunk.target_endpoint_coordinate,
            )
            if value is not None
        )
        longitudinal_start = min((trunk.extent_start, *endpoint_coordinates))
        longitudinal_end = max((trunk.extent_end, *endpoint_coordinates))
        for exit_turn in plan.exit_turn_plans:
            if (
                exit_turn.disposition is not ExitTurnDisposition.PLANNED
                or exit_turn.system_id == convergence.system_id
                or exit_turn.reference_id is None
                or not exit_turn.demand_ids
                or not span.overlaps(demands[exit_turn.demand_ids[0]].span)
            ):
                continue
            clearance = max(exit_turn.spacing, CURVE_RADIUS)
            if exit_turn.source_axis is trunk.axis:
                conflicts = any(
                    longitudinal_start - clearance
                    < axis.coordinate
                    < longitudinal_end + clearance
                    for axis in exit_turn.axes
                )
            else:
                conflicts = any(
                    abs(axis.coordinate - trunk.coordinate) < clearance
                    for axis in exit_turn.axes
                )
            if not conflicts:
                continue
            add(convergence.id, exit_turn.reference_id)
        for fan in plan.fan_plans:
            if (
                fan.disposition is not FanPlanDisposition.PLANNED
                or fan.system_id is None
                or fan.system_id == convergence.system_id
                or fan.centreline_reference_id is None
                or not fan.demand_ids
                or fan.frame is None
                or DemandAxis(fan.frame.primary.name) is not convergence.trunk_axis.axis
                or not span.overlaps(demands[fan.demand_ids[0]].span)
            ):
                continue
            add(convergence.id, fan.centreline_reference_id)
        expected_orientation = (
            CorridorOrientation.HORIZONTAL
            if convergence.trunk_axis.axis is DemandAxis.X
            else CorridorOrientation.VERTICAL
        )
        for reservation in plan.reservations:
            if (
                reservation.system_id == convergence.system_id
                or reservation.orientation is not expected_orientation
                or not span.overlaps(reservation.span)
                or not any(
                    abs(convergence.trunk_axis.coordinate - claim.allocation_coordinate)
                    < max(CURVE_RADIUS, reservation.peer_clearance)
                    for claim in reservation.claims
                )
            ):
                continue
            add(convergence.id, reservation.reference_id)
    return {item.id: tuple(foreign[item.id]) for item in plan.convergence_plans}


def _finalise_reservation_ledger(
    plan: RoutePlan,
    graph: MetroGraph,
    *,
    canvas_width: float | None = None,
    canvas_height: float | None = None,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> RoutePlan:
    """Realise *plan*'s ledger and rebuild every index derived from it."""
    exit_foreign = expected_exit_turn_foreign_references(plan)
    exit_turn_plans = tuple(
        replace(item, foreign_reference_ids=exit_foreign[item.id])
        for item in plan.exit_turn_plans
    )
    convergence_foreign = expected_convergence_foreign_references(plan)
    convergence_plans = tuple(
        replace(item, foreign_reference_ids=convergence_foreign[item.id])
        for item in plan.convergence_plans
    )
    reference_ids_by_system: defaultdict[RouteSystemId, list[SharedReferenceId]] = (
        defaultdict(list)
    )
    demand_ids_by_system: defaultdict[RouteSystemId, list[DemandId]] = defaultdict(list)
    reservation_ids_by_system: defaultdict[RouteSystemId, list[RouteReservationId]] = (
        defaultdict(list)
    )
    for reference in plan.shared_references:
        reference_ids_by_system[reference.system_id].append(reference.id)
    for demand in plan.demands:
        demand_ids_by_system[demand.system_id].append(demand.id)
    for reservation in plan.reservations:
        reservation_ids_by_system[reservation.system_id].append(reservation.id)
    systems = tuple(
        replace(
            system,
            shared_reference_ids=tuple(reference_ids_by_system[system.id]),
            demand_ids=tuple(demand_ids_by_system[system.id]),
            reservation_ids=tuple(reservation_ids_by_system[system.id]),
        )
        for system in plan.systems
    )
    realised = _realise_all(
        graph,
        plan.reservations,
        canvas_width,
        canvas_height,
        coordinate_translations,
    )
    return replace(
        plan,
        systems=systems,
        exit_turn_plans=exit_turn_plans,
        convergence_plans=convergence_plans,
        realised_reservations=realised,
        reservation_diagnostics=_diagnostics(plan.reservations, realised),
    )


def attach_route_reservations(
    plan: RoutePlan,
    graph: MetroGraph,
    routes: list[RoutedPath],
    station_offsets: dict[tuple[str, str], float] | None,
    *,
    canvas_width: float | None = None,
    canvas_height: float | None = None,
) -> RoutePlan:
    """Return *plan* with canonical symbolic and realised corridor ledgers."""
    if not plan.systems:
        return plan
    offsets = station_offsets or {}
    claims = _observed_claims(graph, routes, plan, offsets)
    groups = _group_claims(
        claims,
        {system.id: rank for rank, system in enumerate(plan.systems)},
        {member.id: rank for rank, member in enumerate(plan.members)},
    )
    observed_references, observed_demands, reservations = _build_symbolic_records(
        graph, plan, groups
    )
    plan_with_corridors = replace(
        plan,
        shared_references=plan.shared_references + observed_references,
        demands=plan.demands + observed_demands,
        reservations=reservations,
    )
    return _finalise_reservation_ledger(
        plan_with_corridors,
        graph,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _project_shared_coordinate(
    value: float,
    axis: DemandAxis,
    member_ids: tuple[EmissionMemberId, ...],
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> float:
    """Project a coordinate every listed member shares, requiring agreement.

    Shared plan geometry has one position, so every claimant must project it
    identically; a disagreement means a translation tore shared geometry
    apart, which settlement is forbidden from doing.
    """
    projected = tuple(
        project_reservation_coordinate(value, axis, member_id, translations)
        for member_id in member_ids
    )
    if not projected or any(
        abs(item - projected[0]) > SAME_COORD_TOLERANCE for item in projected[1:]
    ):
        raise ValueError("settlement separates shared plan geometry")
    return projected[0]


def _project_shared_point(
    point: tuple[float, float],
    member_ids: tuple[EmissionMemberId, ...],
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> tuple[float, float]:
    return (
        _project_shared_coordinate(point[0], DemandAxis.X, member_ids, translations),
        _project_shared_coordinate(point[1], DemandAxis.Y, member_ids, translations),
    )


def _project_exit_turn_plan(
    plan: ExitTurnPlan,
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> ExitTurnPlan:
    """Carry an exit-turn frame's absolute coordinates through the translations.

    Lengths -- runways, spacing, diagonal runs -- are frozen demands the drawn
    geometry must meet or exceed, so only positions move.
    """
    axis_axis = {axis.id: axis.axis for axis in plan.axes}
    axes = tuple(
        replace(
            axis,
            coordinate=_project_shared_coordinate(
                axis.coordinate, axis.axis, axis.claimant_member_ids, translations
            ),
            fixed_anchor_coordinate=(
                None
                if axis.fixed_anchor_coordinate is None
                else _project_shared_coordinate(
                    axis.fixed_anchor_coordinate,
                    axis.axis,
                    axis.claimant_member_ids,
                    translations,
                )
            ),
        )
        for axis in plan.axes
    )
    lane_transitions = tuple(
        replace(
            item,
            source_point=_project_shared_point(
                item.source_point, item.claimant_member_ids, translations
            ),
            target_point=_project_shared_point(
                item.target_point, item.claimant_member_ids, translations
            ),
        )
        for item in plan.lane_transitions
    )
    assignments = tuple(
        item
        if item.launch_coordinate is None or item.axis_id is None
        else replace(
            item,
            launch_coordinate=project_reservation_coordinate(
                item.launch_coordinate,
                axis_axis[item.axis_id],
                item.member_id,
                translations,
            ),
        )
        for item in plan.assignments
    )
    return replace(
        plan, axes=axes, lane_transitions=lane_transitions, assignments=assignments
    )


def _project_convergence_plan(
    plan: ConvergencePlan,
    translations: tuple[ReservationCoordinateTranslation, ...],
) -> ConvergencePlan:
    """Carry a convergence frame's absolute coordinates through the translations."""
    trunk = plan.trunk_axis
    if trunk is not None:
        travel = trunk.axis
        across = DemandAxis.Y if travel is DemandAxis.X else DemandAxis.X
        members = plan.member_ids
        trunk = replace(
            trunk,
            coordinate=_project_shared_coordinate(
                trunk.coordinate, across, members, translations
            ),
            extent_start=_project_shared_coordinate(
                trunk.extent_start, travel, members, translations
            ),
            extent_end=_project_shared_coordinate(
                trunk.extent_end, travel, members, translations
            ),
            source_flank_coordinate=_project_shared_coordinate(
                trunk.source_flank_coordinate, across, members, translations
            ),
            target_flank_coordinate=_project_shared_coordinate(
                trunk.target_flank_coordinate, across, members, translations
            ),
            source_endpoint_coordinate=(
                None
                if trunk.source_endpoint_coordinate is None
                else _project_shared_coordinate(
                    trunk.source_endpoint_coordinate, travel, members, translations
                )
            ),
            target_endpoint_coordinate=(
                None
                if trunk.target_endpoint_coordinate is None
                else _project_shared_coordinate(
                    trunk.target_endpoint_coordinate, travel, members, translations
                )
            ),
        )
    landings = tuple(
        replace(
            item,
            join_point=_project_shared_point(
                item.join_point, (item.member_id,), translations
            ),
            opening_turn_coordinate=(
                None
                if item.opening_turn_coordinate is None
                else project_reservation_coordinate(
                    item.opening_turn_coordinate,
                    DemandAxis.X,
                    item.member_id,
                    translations,
                )
            ),
            opening_turn_segment=(
                None
                if item.opening_turn_segment is None
                else (
                    _project_shared_point(
                        item.opening_turn_segment[0], (item.member_id,), translations
                    ),
                    _project_shared_point(
                        item.opening_turn_segment[1], (item.member_id,), translations
                    ),
                )
            ),
        )
        for item in plan.landings
    )
    return replace(plan, trunk_axis=trunk, landings=landings)


def adopt_route_reservation_ledger(
    frozen_plan: RoutePlan,
    graph: MetroGraph,
    *,
    coordinate_translations: tuple[ReservationCoordinateTranslation, ...] = (),
) -> RoutePlan:
    """Republish the plan settlement consumed, on the translated geometry.

    Envelope settlement consumes exactly one immutable ledger, so the plan the
    render publishes is that plan rather than the one the settled re-route
    would observe: re-observation lets corridors appear, vanish, and change
    their required width, which turns settlement's fixed demand set into a
    moving one.  Memberships, dispositions, references, demands, and
    reservations are the frozen records; the exit-turn and convergence frames'
    absolute coordinates are projected through the translations; and the
    realised ledger is re-measured the same way.  The drawn corridor positions
    live in the routes, which may place a corridor anywhere inside its
    reservation's band -- the published ledger records the demand and its
    measured band, not the drawn outcome.  Canvas claims are the exception:
    they are measured against live canvas bounds, so they refresh rather than
    project.
    """
    projected = replace(
        frozen_plan,
        exit_turn_plans=tuple(
            _project_exit_turn_plan(item, coordinate_translations)
            for item in frozen_plan.exit_turn_plans
        ),
        convergence_plans=tuple(
            _project_convergence_plan(item, coordinate_translations)
            for item in frozen_plan.convergence_plans
        ),
    )
    return _finalise_reservation_ledger(
        projected,
        graph,
        coordinate_translations=coordinate_translations,
    )


def realise_route_reservations(
    plan: RoutePlan,
    graph: MetroGraph,
    *,
    canvas_width: float,
    canvas_height: float,
) -> RoutePlan:
    """Refresh the realised ledger against final render canvas bounds.

    Each reservation keeps the projection its realisation was measured with:
    re-realising a frozen claim without its settlement translations would
    measure geometry the translations moved out from under it.
    """
    held_translations = {
        item.reservation_id: item.coordinate_translations
        for item in plan.realised_reservations
    }
    realised = tuple(
        item
        for reservation in plan.reservations
        if (
            item := _realise_one(
                graph,
                reservation,
                canvas_width,
                canvas_height,
                held_translations.get(reservation.id, ()),
            )
        )
        is not None
    )
    return replace(
        plan,
        realised_reservations=realised,
        reservation_diagnostics=_diagnostics(plan.reservations, realised),
    )
