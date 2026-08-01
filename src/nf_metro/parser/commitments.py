"""Typed pre-inference layout commitments for isolated candidate execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

from nf_metro.options import LineOrder, is_line_order
from nf_metro.parser.model import FLOW_DIRECTIONS, MetroGraph, PortSide
from nf_metro.parser.provenance import (
    ConnectorEndpointKey,
    ConnectorEndpointRole,
    DecisionOrigin,
    DecisionReason,
    EndpointSideSelection,
    FoldThresholdSource,
    GridCell,
    LineOrderSource,
)
from nf_metro.parser.route_topology import AuthoredRouteCapture, ConnectorId

FlowDirection = Literal["LR", "RL", "TB", "BT"]


def is_flow_direction(value: object) -> TypeGuard[FlowDirection]:
    return isinstance(value, str) and value in FLOW_DIRECTIONS


@dataclass(frozen=True, slots=True)
class GridCommitment:
    section_id: str
    cell: GridCell


@dataclass(frozen=True, slots=True)
class DirectionCommitment:
    section_id: str
    direction: FlowDirection


@dataclass(frozen=True, slots=True)
class EndpointCommitment:
    connector_id: ConnectorId
    role: ConnectorEndpointRole
    side: PortSide


@dataclass(frozen=True, slots=True)
class LayoutCommitments:
    grids: tuple[GridCommitment, ...] = ()
    fold_threshold: int | None = None
    directions: tuple[DirectionCommitment, ...] = ()
    endpoints: tuple[EndpointCommitment, ...] = ()
    line_order: LineOrder | None = None


@dataclass(frozen=True, slots=True)
class LayoutCommitmentOverlay:
    """Caller pins plus one candidate proposal, applied as one transaction."""

    caller: LayoutCommitments = LayoutCommitments()
    candidate: LayoutCommitments = LayoutCommitments()


@dataclass(frozen=True, slots=True)
class EndpointCommitmentSelection:
    endpoint: ConnectorEndpointKey
    selection: EndpointSideSelection


@dataclass(frozen=True, slots=True)
class AppliedLayoutCommitments:
    """Resolver input that cannot mutate or broaden the submitted overlay."""

    endpoints: tuple[EndpointCommitmentSelection, ...] = ()

    def selection_for(
        self,
        connector_ids: tuple[ConnectorId, ...],
        role: ConnectorEndpointRole,
        index: dict[ConnectorEndpointKey, EndpointSideSelection] | None = None,
    ) -> EndpointSideSelection | None:
        if index is None:
            index = {item.endpoint: item.selection for item in self.endpoints}
        selections = tuple(
            index[ConnectorEndpointKey(connector_id, role)]
            for connector_id in connector_ids
            if ConnectorEndpointKey(connector_id, role) in index
        )
        if not selections:
            return None
        sides = {item.side for item in selections}
        if len(sides) != 1:
            raise CommitmentConflictError(
                f"{role.value} commitments disagree for one resolved connector group"
            )
        if len(selections) != len(connector_ids):
            raise CommitmentConflictError(
                f"{role.value} commitment covers only part of one resolved "
                "connector group"
            )
        origins = {item.origin for item in selections}
        reasons = {item.reason for item in selections}
        if len(origins) != 1 or len(reasons) != 1:
            raise CommitmentConflictError(
                f"{role.value} commitments mix caller and candidate ownership"
            )
        return selections[0]


class CommitmentValidationError(ValueError):
    """A commitment record is malformed, duplicated, or names no target."""


class CommitmentConflictError(ValueError):
    """A candidate attempts to replace author or caller-owned layout input."""


class CommitmentSettlementError(ValueError):
    """Inference or resolution did not preserve an accepted commitment."""


def _validate_grid_cell(cell: object) -> TypeGuard[GridCell]:
    return (
        isinstance(cell, tuple)
        and len(cell) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in cell)
        and cell[0] >= 0
        and cell[1] >= 0
        and cell[2] >= 1
        and cell[3] >= 1
    )


def _validate_commitments(
    label: str,
    commitments: LayoutCommitments,
    section_ids: set[str],
    connector_ids: set[ConnectorId],
) -> None:
    if not isinstance(commitments, LayoutCommitments):
        raise CommitmentValidationError(f"{label} commitments are malformed")
    for field_name in ("grids", "directions", "endpoints"):
        if not isinstance(getattr(commitments, field_name), tuple):
            raise CommitmentValidationError(
                f"{label} {field_name} commitments must be an immutable tuple"
            )

    seen_grids: set[str] = set()
    for grid_commitment in commitments.grids:
        if not isinstance(grid_commitment, GridCommitment):
            raise CommitmentValidationError(f"{label} grid record is malformed")
        if (
            not isinstance(grid_commitment.section_id, str)
            or not grid_commitment.section_id
            or grid_commitment.section_id not in section_ids
        ):
            raise CommitmentValidationError(
                f"{label} grid names unknown section {grid_commitment.section_id!r}"
            )
        if grid_commitment.section_id in seen_grids:
            raise CommitmentValidationError(
                f"{label} repeats grid commitment for {grid_commitment.section_id!r}"
            )
        seen_grids.add(grid_commitment.section_id)
        if not _validate_grid_cell(grid_commitment.cell):
            raise CommitmentValidationError(
                f"{label} grid for {grid_commitment.section_id!r} must be "
                "(nonnegative col, nonnegative row, positive rowspan, "
                "positive colspan)"
            )

    seen_directions: set[str] = set()
    for direction_commitment in commitments.directions:
        if not isinstance(direction_commitment, DirectionCommitment):
            raise CommitmentValidationError(f"{label} direction record is malformed")
        if (
            not isinstance(direction_commitment.section_id, str)
            or not direction_commitment.section_id
            or direction_commitment.section_id not in section_ids
        ):
            raise CommitmentValidationError(
                f"{label} direction names unknown section "
                f"{direction_commitment.section_id!r}"
            )
        if direction_commitment.section_id in seen_directions:
            raise CommitmentValidationError(
                f"{label} repeats direction commitment for "
                f"{direction_commitment.section_id!r}"
            )
        seen_directions.add(direction_commitment.section_id)
        if not is_flow_direction(direction_commitment.direction):
            raise CommitmentValidationError(
                f"{label} direction for {direction_commitment.section_id!r} "
                "must be LR/RL/TB/BT"
            )

    seen_endpoints: set[ConnectorEndpointKey] = set()
    for endpoint_commitment in commitments.endpoints:
        if not isinstance(endpoint_commitment, EndpointCommitment):
            raise CommitmentValidationError(f"{label} endpoint record is malformed")
        if (
            not isinstance(endpoint_commitment.connector_id, str)
            or endpoint_commitment.connector_id not in connector_ids
        ):
            raise CommitmentValidationError(
                f"{label} endpoint names unknown connector "
                f"{endpoint_commitment.connector_id!r}"
            )
        if not isinstance(endpoint_commitment.role, ConnectorEndpointRole):
            raise CommitmentValidationError(f"{label} endpoint role is invalid")
        if not isinstance(endpoint_commitment.side, PortSide):
            raise CommitmentValidationError(f"{label} endpoint side is invalid")
        endpoint = ConnectorEndpointKey(
            endpoint_commitment.connector_id, endpoint_commitment.role
        )
        if endpoint in seen_endpoints:
            raise CommitmentValidationError(
                f"{label} repeats {endpoint_commitment.role.value} commitment for "
                f"{endpoint_commitment.connector_id!r}"
            )
        seen_endpoints.add(endpoint)

    fold = commitments.fold_threshold
    if fold is not None and (
        not isinstance(fold, int) or isinstance(fold, bool) or fold <= 0
    ):
        raise CommitmentValidationError(f"{label} fold threshold must be positive")
    if commitments.line_order is not None and not is_line_order(commitments.line_order):
        raise CommitmentValidationError(
            f"{label} line order must be 'definition' or 'span'"
        )


def _grid_index(commitments: LayoutCommitments) -> dict[str, GridCell]:
    return {item.section_id: item.cell for item in commitments.grids}


def _direction_index(commitments: LayoutCommitments) -> dict[str, FlowDirection]:
    return {item.section_id: item.direction for item in commitments.directions}


def _endpoint_index(
    commitments: LayoutCommitments,
) -> dict[ConnectorEndpointKey, PortSide]:
    return {
        ConnectorEndpointKey(item.connector_id, item.role): item.side
        for item in commitments.endpoints
    }


def _candidate_conflicts(
    graph: MetroGraph, overlay: LayoutCommitmentOverlay
) -> tuple[str, ...]:
    provenance = graph.layout_provenance
    authored = provenance.authored
    if authored is None:
        raise RuntimeError("layout commitments require captured provenance")

    caller_grids = _grid_index(overlay.caller)
    caller_directions = _direction_index(overlay.caller)
    caller_endpoints = _endpoint_index(overlay.caller)
    authored_grids = {item.section_id: item.value for item in authored.grids}
    authored_directions = {item.section_id: item.value for item in authored.directions}
    authored_endpoints = authored.endpoint_values_index()
    conflicts: list[str] = []

    for grid_commitment in overlay.candidate.grids:
        owned_grid = caller_grids.get(
            grid_commitment.section_id,
            authored_grids.get(grid_commitment.section_id),
        )
        if owned_grid is not None and owned_grid != grid_commitment.cell:
            conflicts.append(f"grid:{grid_commitment.section_id}")
    for direction_commitment in overlay.candidate.directions:
        owned_direction = caller_directions.get(
            direction_commitment.section_id,
            authored_directions.get(direction_commitment.section_id),
        )
        if (
            owned_direction is not None
            and owned_direction != direction_commitment.direction
        ):
            conflicts.append(f"direction:{direction_commitment.section_id}")

    fold = overlay.candidate.fold_threshold
    fold_intent = authored.fold_threshold
    if fold is not None:
        owned_fold = overlay.caller.fold_threshold
        if (
            owned_fold is None
            and fold_intent.selected_source is not FoldThresholdSource.DEFAULT
        ):
            owned_fold = fold_intent.selected_value
        if owned_fold is not None and fold != owned_fold:
            conflicts.append("fold-threshold")

    line_order = overlay.candidate.line_order
    line_intent = authored.line_order
    if line_order is not None:
        owned_order = overlay.caller.line_order
        if (
            owned_order is None
            and line_intent.selected_source is not LineOrderSource.DEFAULT
        ):
            owned_order = line_intent.selected_value
        if owned_order is not None and line_order != owned_order:
            conflicts.append("line-order")

    for endpoint, side in _endpoint_index(overlay.candidate).items():
        caller_side = caller_endpoints.get(endpoint)
        authored_sides = authored_endpoints.get(endpoint, ())
        if caller_side is not None:
            if caller_side is not side:
                conflicts.append(f"{endpoint.role.value}:{endpoint.connector_id}")
        elif authored_sides and not all(value is side for value in authored_sides):
            conflicts.append(f"{endpoint.role.value}:{endpoint.connector_id}")
    return tuple(conflicts)


def apply_layout_commitment_overlay(
    graph: MetroGraph,
    routes: AuthoredRouteCapture,
    overlay: LayoutCommitmentOverlay,
) -> AppliedLayoutCommitments:
    """Validate the complete overlay, then apply it atomically before inference."""
    section_ids = set(graph.sections)
    connector_ids = {item.key.id for item in routes.edges}
    _validate_commitments("caller", overlay.caller, section_ids, connector_ids)
    _validate_commitments("candidate", overlay.candidate, section_ids, connector_ids)
    conflicts = _candidate_conflicts(graph, overlay)
    if conflicts:
        raise CommitmentConflictError(
            "candidate conflicts with author or caller: " + ", ".join(conflicts)
        )

    provenance = graph.layout_provenance
    caller_grids = _grid_index(overlay.caller)
    candidate_grids = _grid_index(overlay.candidate)
    for section_id, grid_value in caller_grids.items():
        graph.grid_overrides[section_id] = grid_value
        provenance.record_committed_grid(
            section_id,
            grid_value,
            DecisionOrigin.AUTHORED,
            DecisionReason.CALLER_COMMITMENT,
        )
    for section_id, grid_value in candidate_grids.items():
        if section_id in caller_grids:
            continue
        grid_decision = provenance.grid_decision(section_id)
        if grid_decision is not None and grid_decision.is_author_owned:
            continue
        graph.grid_overrides[section_id] = grid_value
        provenance.record_committed_grid(
            section_id,
            grid_value,
            DecisionOrigin.INFERRED,
            DecisionReason.CANDIDATE_COMMITMENT,
        )

    caller_directions = _direction_index(overlay.caller)
    candidate_directions = _direction_index(overlay.candidate)
    for section_id, direction_value in caller_directions.items():
        graph.sections[section_id].direction = direction_value
        provenance.record_committed_direction(
            section_id,
            direction_value,
            DecisionOrigin.AUTHORED,
            DecisionReason.CALLER_COMMITMENT,
        )
    for section_id, direction_value in candidate_directions.items():
        if section_id in caller_directions:
            continue
        direction_decision = provenance.direction_decision(section_id)
        if direction_decision is not None and direction_decision.is_author_owned:
            continue
        graph.sections[section_id].direction = direction_value
        provenance.record_committed_direction(
            section_id,
            direction_value,
            DecisionOrigin.INFERRED,
            DecisionReason.CANDIDATE_COMMITMENT,
        )

    caller_fold = overlay.caller.fold_threshold
    if caller_fold is not None:
        graph.fold_threshold = caller_fold
    fold = overlay.candidate.fold_threshold
    if (
        fold is not None
        and caller_fold is None
        and graph.layout_provenance.authored is not None
    ):
        if (
            graph.layout_provenance.authored.fold_threshold.selected_source
            is FoldThresholdSource.DEFAULT
        ):
            graph.fold_threshold = fold
            provenance.record_candidate_fold_threshold(fold)

    caller_line_order = overlay.caller.line_order
    if caller_line_order is not None:
        graph.line_order = caller_line_order
    line_order = overlay.candidate.line_order
    if (
        line_order is not None
        and caller_line_order is None
        and graph.layout_provenance.authored is not None
    ):
        if (
            graph.layout_provenance.authored.line_order.selected_source
            is LineOrderSource.DEFAULT
        ):
            graph.line_order = line_order
            provenance.record_candidate_line_order(line_order)

    caller_endpoints = _endpoint_index(overlay.caller)
    candidate_endpoints = _endpoint_index(overlay.candidate)
    authored_endpoints = (
        provenance.authored.endpoint_values_index()
        if provenance.authored is not None
        else {}
    )
    selections: list[EndpointCommitmentSelection] = []
    for endpoint, side in caller_endpoints.items():
        selections.append(
            EndpointCommitmentSelection(
                endpoint,
                EndpointSideSelection(
                    side,
                    DecisionOrigin.AUTHORED,
                    True,
                    DecisionReason.CALLER_COMMITMENT,
                ),
            )
        )
    for endpoint, side in candidate_endpoints.items():
        if endpoint in caller_endpoints:
            continue
        authored_sides = authored_endpoints.get(endpoint, ())
        if authored_sides and all(value is side for value in authored_sides):
            continue
        selections.append(
            EndpointCommitmentSelection(
                endpoint,
                EndpointSideSelection(
                    side,
                    DecisionOrigin.INFERRED,
                    True,
                    DecisionReason.CANDIDATE_COMMITMENT,
                ),
            )
        )
    return AppliedLayoutCommitments(tuple(selections))


def verify_settled_commitments(
    graph: MetroGraph, overlay: LayoutCommitmentOverlay
) -> None:
    """Prove that every caller and candidate pin survived inference exactly."""
    connectors = (
        {item.id: item for item in graph.route_topology.connectors}
        if graph.route_topology is not None
        else {}
    )
    for commitments in (overlay.caller, overlay.candidate):
        for grid_commitment in commitments.grids:
            grid_decision = graph.layout_provenance.grid_decision(
                grid_commitment.section_id
            )
            grid_section = graph.sections.get(grid_commitment.section_id)
            settled_grid = (
                (
                    grid_section.grid_col,
                    grid_section.grid_row,
                    grid_section.grid_row_span,
                    grid_section.grid_col_span,
                )
                if grid_section is not None
                else None
            )
            if (
                grid_decision is None
                or grid_decision.value != grid_commitment.cell
                or not grid_decision.locked
                or settled_grid != grid_commitment.cell
            ):
                raise CommitmentSettlementError(
                    f"grid commitment for {grid_commitment.section_id!r} "
                    f"settled as {settled_grid!r}"
                )
        for direction_commitment in commitments.directions:
            direction_decision = graph.layout_provenance.direction_decision(
                direction_commitment.section_id
            )
            direction_section = graph.sections.get(direction_commitment.section_id)
            if (
                direction_decision is None
                or direction_decision.value != direction_commitment.direction
                or not direction_decision.locked
                or direction_section is None
                or direction_section.direction != direction_commitment.direction
            ):
                raise CommitmentSettlementError(
                    f"direction commitment for {direction_commitment.section_id!r} "
                    "was not preserved"
                )
        for endpoint_commitment in commitments.endpoints:
            endpoint = ConnectorEndpointKey(
                endpoint_commitment.connector_id, endpoint_commitment.role
            )
            endpoint_decision = graph.layout_provenance.endpoint_decision(endpoint)
            connector = connectors.get(endpoint_commitment.connector_id)
            settled_side = (
                connector.exit_side
                if connector is not None
                and endpoint_commitment.role is ConnectorEndpointRole.EXIT
                else connector.entry_side
                if connector is not None
                else None
            )
            if (
                endpoint_decision is None
                or endpoint_decision.value is not endpoint_commitment.side
                or not endpoint_decision.locked
                or settled_side is not endpoint_commitment.side
            ):
                raise CommitmentSettlementError(
                    f"{endpoint_commitment.role.value} commitment for "
                    f"{endpoint_commitment.connector_id!r} was not preserved"
                )
        if commitments.fold_threshold is not None:
            fold_decision = graph.layout_provenance.fold_threshold_decision
            if (
                fold_decision is None
                or fold_decision.value != commitments.fold_threshold
                or not fold_decision.locked
                or graph.fold_threshold != commitments.fold_threshold
            ):
                raise CommitmentSettlementError(
                    "fold-threshold commitment was not preserved"
                )
        if commitments.line_order is not None:
            line_order_decision = graph.layout_provenance.line_order_decision
            if (
                line_order_decision is None
                or line_order_decision.value != commitments.line_order
                or not line_order_decision.locked
                or graph.line_order != commitments.line_order
            ):
                raise CommitmentSettlementError(
                    "line-order commitment was not preserved"
                )
