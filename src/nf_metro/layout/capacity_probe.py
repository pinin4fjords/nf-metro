"""Counterfactual boundary-capacity probe for compatibility route systems.

``attribute_compatibility_systems`` measures whether a *translation* changes the
distance between the two coordinates a convergence planner recorded as its
conflict.  That is a statement about two points already drawn.  #1657's exit
criteria ask a different question: whether a boundary with enough room would let
the planner allocate every member of the system, which no measurement of drawn
geometry can answer because the planner never ran against that geometry.

This module answers it directly.  For one compatibility system it copies the
settled graph, translates whole rows and columns to widen the boundaries the
system is measured at, re-runs convergence planning on the copy, and reads the
disposition that comes back.  A system the planner plans once it has room was
held by an envelope allocation; a system that stays on the compatibility path
across every capacity granted was held by something no allocation supplies.

Three properties make the answer usable as evidence.

*Read-only.*  Every grant runs on ``copy.deepcopy`` of the graph and the plan is
only read, so no probe geometry, plan, reservation, or offset can reach the map
that gets drawn.  Nothing here is called from the render path.

*Controlled.*  Re-planning is only meaningful if it reproduces the disposition
the map already has.  Each system is first re-planned on an untouched copy, and
a system whose control does not come back on the compatibility path with the
conflict it published is reported as ``CONTROL_DIVERGED`` rather than measured:
its grants would be differences against an unknown baseline.

*Falsifiable.*  A probe that can only ever report "no allocation reaches this"
would be indistinguishable from a probe that does nothing, so the result has to
be reachable in both directions.  It is: the corpus contains systems the probe
reports planned, and ``tests/test_capacity_probe.py`` also starves a system that
the planner plans on its own geometry and watches the probe hand its capacity
back.

The planner's answer is not monotone in capacity -- a system can be planned at
one capacity and compatible at a larger one, because moving whole rows and
columns changes which runs overlap as well as how much room they have.  A single
grant is therefore not evidence.  The verdict is taken from a *tail*: a system
counts as reached only when it is planned at some granted capacity and at every
larger one, which no isolated coincidence of alignment satisfies.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum

from nf_metro.layout.constants import CURVE_RADIUS, SETTLEMENT_QUANTUM
from nf_metro.layout.envelope_settlement import (
    _COLUMN_AXIS,
    _ROW_AXIS,
    _Axis,
    _translation_ownership,
)
from nf_metro.layout.route_plan import (
    ConvergenceConflictKind,
    ConvergenceDisposition,
    ConvergencePlan,
    RoutePlan,
    RouteSystemId,
)
from nf_metro.layout.route_reservations import ColumnGapRegion, RowGapRegion
from nf_metro.parser.model import MetroGraph

CAPACITY_MULTIPLES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
"""Multiples of a system's own derived capacity unit that each get granted.

The unit is what one competing pair of runs costs, so the top of the ladder is
sixteen of them stacked into a boundary that holds a handful.  A system still on
the compatibility path there is not short of room.
"""


class CapacityScope(Enum):
    """Which boundaries one grant widens."""

    CLAIMED_BOUNDARIES = "claimed-boundaries"
    """Only the row and column boundaries the system's own gap reservations are
    filed against, which is where settlement would ever allocate for it."""

    EVERY_BOUNDARY = "every-boundary"
    """Every row and column boundary in the grid, so a conflict whose relief
    lies outside the system's own claims is still offered the room."""


class CapacityVerdict(Enum):
    """What granting boundary capacity did to one compatibility system."""

    ALLOCATION_REACHES = "allocation-reaches"
    """The planner returns a planned convergence at some granted capacity and at
    every larger one.  The limitation is an envelope allocation."""

    ALLOCATION_UNSTABLE = "allocation-unstable"
    """The planner returns a planned convergence at some granted capacity but
    not at the largest, so capacity changes the answer without a threshold above
    which it holds."""

    BEYOND_ALLOCATION = "beyond-allocation"
    """No granted capacity makes the planner plan the system.  This is the
    evidence #1657's exit criteria ask for."""

    CONTROL_DIVERGED = "control-diverged"
    """Re-planning the untouched copy did not reproduce the disposition the map
    publishes, so nothing measured against it would mean anything."""


@dataclass(frozen=True, slots=True)
class CapacityGrant:
    """One counterfactual: this much room at these boundaries, and the answer."""

    scope: CapacityScope
    capacity: float
    planned: bool


@dataclass(frozen=True, slots=True)
class CapacityProbe:
    """What boundary capacity does to one system on the compatibility path."""

    system_id: RouteSystemId
    verdict: CapacityVerdict
    unit: float
    """The largest single distance the system's own limit is measured against:
    the widest corridor it reserves, the offset step between lanes, or the turn
    radius two runs need between them."""
    capacity: float
    """``unit`` plus the separation the conflict recorded, quantised to a
    translation settlement could express.  Each grant is a multiple of this."""
    grants: tuple[CapacityGrant, ...]
    sufficient_capacity: float | None
    sufficient_scope: CapacityScope | None
    control_conflict: ConvergenceConflictKind | None

    @property
    def message(self) -> str:
        if self.verdict is CapacityVerdict.CONTROL_DIVERGED:
            return (
                f"route system {self.system_id} could not be probed: re-planning "
                f"its settled geometry unchanged did not reproduce the "
                f"compatibility disposition the map publishes"
            )
        held = (
            "nothing it recorded"
            if self.control_conflict is None
            else self.control_conflict.reason
        )
        granted = max((item.capacity for item in self.grants), default=0.0)
        if self.verdict is CapacityVerdict.BEYOND_ALLOCATION:
            return (
                f"route system {self.system_id} stays on compatibility under "
                f"every capacity this probe granted, up to {granted:.2f}px at "
                f"{len(self.grants)} boundary allocations; what holds it is "
                f"{held}, and no envelope allocation supplies it"
            )
        assert self.sufficient_capacity is not None
        assert self.sufficient_scope is not None
        if self.verdict is CapacityVerdict.ALLOCATION_REACHES:
            return (
                f"route system {self.system_id} is planned once its "
                f"{self.sufficient_scope.value} carry {self.sufficient_capacity:.2f}px "
                f"more, and at every larger capacity granted, so what holds it "
                f"({held}) is an envelope allocation and not a decision to "
                f"attribute elsewhere"
            )
        return (
            f"route system {self.system_id} is planned at "
            f"{self.sufficient_capacity:.2f}px more across its "
            f"{self.sufficient_scope.value} but not at the largest capacity "
            f"granted, so capacity changes what the planner decides about it "
            f"({held}) without a threshold above which the decision holds"
        )


def probe_settlement_capacity(
    graph: MetroGraph, plan: RoutePlan
) -> tuple[CapacityProbe, ...]:
    """Ask what boundary capacity would do to every compatibility system in *plan*.

    *graph* is the settled geometry the map draws and is never written to: each
    counterfactual runs on its own deep copy.
    """
    compatibility = _ordered_compatibility_systems(plan)
    if not compatibility:
        return ()
    control, offset_step = _replan(copy.deepcopy(graph))
    return tuple(
        _probe_system(graph, plan, system_id, conflict, control, offset_step)
        for system_id, conflict in compatibility
    )


def _ordered_compatibility_systems(
    plan: RoutePlan,
) -> tuple[tuple[RouteSystemId, ConvergenceConflictKind | None], ...]:
    """Each compatibility system once, in the plan's own system order."""
    found: dict[RouteSystemId, ConvergenceConflictKind | None] = {}
    for convergence in plan.convergence_plans:
        if convergence.legacy_reason is None:
            continue
        if convergence.system_id in found:
            continue
        found[convergence.system_id] = (
            None if convergence.conflict is None else convergence.conflict.kind
        )
    return tuple(sorted(found.items(), key=lambda item: item[0]))


def _probe_system(
    graph: MetroGraph,
    plan: RoutePlan,
    system_id: RouteSystemId,
    conflict: ConvergenceConflictKind | None,
    control: dict[RouteSystemId, tuple[ConvergencePlan, ...]],
    offset_step: float,
) -> CapacityProbe:
    if _is_planned(control, system_id) is not False:
        return CapacityProbe(
            system_id, CapacityVerdict.CONTROL_DIVERGED, 0.0, 0.0, (), None, None, None
        )
    rows, columns, unit = _demand(plan, system_id, offset_step)
    separation = max(
        (
            item.conflict.separation
            for item in plan.convergence_plans
            if item.system_id == system_id and item.conflict is not None
        ),
        default=0.0,
    )
    capacity = math.ceil((separation + unit) / SETTLEMENT_QUANTUM) * SETTLEMENT_QUANTUM
    every_row = tuple(sorted({item.grid_row for item in graph.sections.values()}))
    every_column = tuple(sorted({item.grid_col for item in graph.sections.values()}))

    grants: list[CapacityGrant] = []
    for scope, at_rows, at_columns in (
        (CapacityScope.CLAIMED_BOUNDARIES, rows, columns),
        (CapacityScope.EVERY_BOUNDARY, every_row, every_column),
    ):
        for multiple in CAPACITY_MULTIPLES:
            amount = round(capacity * multiple, 6)
            grants.append(
                CapacityGrant(
                    scope,
                    amount,
                    _plans_with_capacity(graph, system_id, at_rows, at_columns, amount),
                )
            )
    sufficient = _least_sufficient(grants)
    sufficient_scope: CapacityScope | None = None
    sufficient_capacity: float | None = None
    if sufficient is not None:
        verdict = CapacityVerdict.ALLOCATION_REACHES
        sufficient_scope, sufficient_capacity = sufficient
    elif any(item.planned for item in grants):
        verdict = CapacityVerdict.ALLOCATION_UNSTABLE
        cheapest = min(
            (item for item in grants if item.planned),
            key=lambda item: (item.capacity, item.scope.value),
        )
        sufficient_scope, sufficient_capacity = cheapest.scope, cheapest.capacity
    else:
        verdict = CapacityVerdict.BEYOND_ALLOCATION
    return CapacityProbe(
        system_id,
        verdict,
        unit,
        capacity,
        tuple(grants),
        sufficient_capacity,
        sufficient_scope,
        conflict,
    )


def _demand(
    plan: RoutePlan, system_id: RouteSystemId, offset_step: float
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """The boundaries one system is measured at, and what one pair of runs costs.

    A conflict is always a distance two runs need between them, and the planner
    compares against the turn radius or the offset step; a corridor the system
    reserves states a width the ledger sizes a boundary by.  The largest of
    those is the unit no allocation smaller than could be called generous.
    """
    rows: set[int] = set()
    columns: set[int] = set()
    widths: list[float] = []
    for reservation in plan.reservations:
        if reservation.system_id != system_id:
            continue
        region = reservation.region
        if isinstance(region, RowGapRegion):
            rows.add(region.lower_row)
        elif isinstance(region, ColumnGapRegion):
            columns.add(region.right_column)
        else:
            continue
        widths.append(reservation.minimum_width)
    unit = max(CURVE_RADIUS, offset_step, max(widths, default=0.0))
    return tuple(sorted(rows)), tuple(sorted(columns)), unit


def _least_sufficient(
    grants: tuple[CapacityGrant, ...] | list[CapacityGrant],
) -> tuple[CapacityScope, float] | None:
    """The smallest capacity planned at, and at every larger capacity granted.

    Read per scope, because a grant's meaning depends on which boundaries it
    widened; the answer is the cheapest such capacity across the scopes.
    """
    found: list[tuple[float, CapacityScope]] = []
    for scope in CapacityScope:
        ladder = sorted(
            (item for item in grants if item.scope is scope),
            key=lambda item: item.capacity,
            reverse=True,
        )
        tail: float | None = None
        for item in ladder:
            if not item.planned:
                break
            tail = item.capacity
        if tail is not None:
            found.append((tail, scope))
    if not found:
        return None
    capacity, scope = min(found, key=lambda item: (item[0], item[1].value))
    return scope, capacity


def _plans_with_capacity(
    graph: MetroGraph,
    system_id: RouteSystemId,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    amount: float,
) -> bool:
    """Whether the planner plans *system_id* once those boundaries carry *amount*."""
    probe_graph = copy.deepcopy(graph)
    for axis, boundaries in ((_ROW_AXIS, rows), (_COLUMN_AXIS, columns)):
        _widen(probe_graph, axis, boundaries, amount)
    try:
        replanned, _offset_step = _replan(probe_graph)
    except Exception:  # noqa: BLE001
        return False
    return _is_planned(replanned, system_id) is True


def _widen(
    graph: MetroGraph, axis: _Axis, boundaries: tuple[int, ...], amount: float
) -> None:
    """Translate everything at or beyond each boundary, exactly as settlement does.

    Applied in ascending boundary order so a section beyond several of them
    accumulates every one, which is what makes each named boundary wider by
    ``amount`` rather than the last one alone.
    """
    for boundary in sorted(boundaries):
        ownership = _translation_ownership(graph, axis, boundary)
        for section_id in ownership.moved_section_ids:
            axis.shift(graph, graph.sections[section_id], amount)


def _replan(
    graph: MetroGraph,
) -> tuple[dict[RouteSystemId, tuple[ConvergencePlan, ...]], float]:
    """Run convergence planning over *graph* and group the result by system.

    Imported here because the planner reaches back into layout: the probe is a
    consumer of routing, not a dependency of it.
    """
    from nf_metro.layout.constants import DIAGONAL_RUN
    from nf_metro.layout.routing import compute_station_offsets
    from nf_metro.layout.routing.context import _build_routing_context
    from nf_metro.layout.routing.convergences import build_convergence_plan_execution
    from nf_metro.layout.routing.exit_turns import build_exit_turn_execution

    context = _build_routing_context(
        graph, DIAGONAL_RUN, CURVE_RADIUS, compute_station_offsets(graph)
    )
    exit_turns = build_exit_turn_execution(graph, context)
    context.exit_turns = exit_turns.query
    if exit_turns.scaffold is None:
        return {}, context.offset_step
    execution = build_convergence_plan_execution(
        graph,
        context,
        exit_turns.scaffold,
        exit_turn_plans=exit_turns.plans,
        fan_plans=graph.fan_plans,
        include_resources=False,
    )
    by_system: dict[RouteSystemId, tuple[ConvergencePlan, ...]] = {}
    for item in execution.plans:
        by_system[item.system_id] = (*by_system.get(item.system_id, ()), item)
    return by_system, context.offset_step


def _is_planned(
    by_system: dict[RouteSystemId, tuple[ConvergencePlan, ...]],
    system_id: RouteSystemId,
) -> bool | None:
    """Whether a re-plan owns the whole system, or ``None`` if it lost it.

    A system is planned or compatible as a whole, so a mixed result is a
    disagreement with that rule rather than a capacity answer, and is reported
    as an absent baseline the same way a vanished system is.
    """
    plans = by_system.get(system_id)
    if not plans:
        return None
    dispositions = {item.disposition for item in plans}
    if dispositions == {ConvergenceDisposition.PLANNED}:
        return True
    if dispositions == {ConvergenceDisposition.LEGACY}:
        return False
    return None
