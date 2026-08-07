"""Atomic route-system ownership for inter-section emission."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from nf_metro.layout.route_plan import (
    ConvergenceDisposition,
    ConvergencePlan,
    EmissionMemberId,
    ExitTurnDisposition,
    ExitTurnPlan,
    FanPlan,
    FanPlanDisposition,
    RouteMemberGeometryPlan,
    RouteSemanticScaffold,
    RouteSystemCompatibilityReason,
    RouteSystemDisposition,
    RouteSystemId,
)
from nf_metro.layout.route_reservations import reservation_ids_by_claimant_member
from nf_metro.layout.routing.families import RouteFamilyId
from nf_metro.parser.model import Edge
from nf_metro.parser.route_topology import ResolvedEdge

if TYPE_CHECKING:
    from nf_metro.layout.route_plan import RoutePlan
    from nf_metro.layout.routing.common import RoutedPath


_COMPATIBILITY_REASONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "exit-turn-plan": frozenset(
            {
                "ambiguous-continuation",
                "ambiguous-source-lane-boundary",
                "continuation-transition-has-no-runway",
                "family-changed-after-lane-compaction",
                "fixed-anchor-owned-by-another-plan",
                "fixed-axis-conflict",
                "insufficient-fixed-runway",
                "insufficient-structural-runway",
                "invalid-source-turn-requirement",
                "lane-transition-order-inversion",
                "linear-entry-frame-ownership-conflict",
                "missing-or-ambiguous-source-order",
                "missing-outbound-member",
                "missing-production-family",
                "missing-source-turn",
                "multiple-destinations",
                "opposed-source-run",
                "overlapping-planned-turn-axes",
                "planned-axis-overlaps-compatibility-channel",
                "shared-source-ownership-conflict",
                "shared-station-lane-collision",
                "single-member-group",
                "source-lane-transition-has-no-runway",
                "unresolved-perpendicular-entry-seam",
                "unsupported-family:bottom-exit-junction",
                "unsupported-family:bypass-family",
                "unsupported-family:left-entry-wrap-family",
                "unsupported-family:merge-branch",
                "unsupported-family:merge-trunk",
                "unsupported-family:near-vertical-same-col-junction",
                "unsupported-family:perp-exit",
                "unsupported-family:perp-exit-far-side-entry-wrap",
                "unsupported-family:right-entry-plough-bypass",
                "unsupported-family:right-entry-wrap",
                "unsupported-family:tb-perp-exit-over",
                "unsupported-subshape:degenerate-horizontal-straight",
                "unsupported-subshape:left-exit-right-entry-step",
                "unsupported-subshape:merge-entry-straight",
                "unsupported-subshape:nonvertical-tb-exit",
                "unsupported-subshape:opposed-horizontal-straight",
                "unsupported-subshape:unaligned-perpendicular-entry",
                "unsupported-subshape:vertical-source-horizontal-straight",
            }
        ),
        "fan-plan": frozenset(
            {
                "ambiguous-branch-to-join",
                "ambiguous-resolved-branch-tail",
                "ambiguous-resolved-fork",
                "ambiguous-resolved-join",
                "empty-resolved-member-path",
                "fan-route-system-has-no-emission-member",
                "local-layout-has-foreign-owner",
                "missing-centreline-anchor",
                "missing-resolved-extra-output-path",
                "missing-resolved-member-path",
                "off-track-layout-owns-fan-geometry",
                "offset-carrier-has-unowned-line",
                "overlapping-branch-lane-ownership",
                "overlapping-fan-ownership",
                "rail-layout-owns-fan-geometry",
                "same-line-open-fan-layout-owns-geometry",
                "unsupported-branch-line-transition",
                "unsupported-fan-direction",
            }
        ),
        "convergence-plan": frozenset(
            {
                "convergence alignment conflicts with an upstream exit turn",
                "convergence landing conflicts with an upstream exit turn",
                "convergence landing has no approach",
                "convergence template declined its member",
                "covered continuation is absent from its carrier",
                "direct convergence has no emitted terminal approach",
                "feeder template declined its member",
                "planned convergence approaches and trunks have no settlement room",
                "planned convergence member has no routing family",
                "planned trunk has no drawable segment",
                "primary trunk template declined its member",
                "primary trunk template emitted no shared run",
                "unsupported convergence shape",
            }
        ),
        "member-geometry-plan": frozenset(
            {
                "canonical-template-declined-member",
                "missing-emission-edge",
                "missing-production-family",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RouteSystemEmissionMember:
    member_id: EmissionMemberId
    edge: ResolvedEdge
    family_id: RouteFamilyId | None = None
    geometry_plan: RouteMemberGeometryPlan | None = None
    reservation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteSystemEmission:
    system_id: RouteSystemId
    connector_ids: tuple[str, ...]
    members: tuple[RouteSystemEmissionMember, ...]
    disposition: RouteSystemDisposition
    compatibility_reasons: tuple[RouteSystemCompatibilityReason, ...]
    plan_ids: tuple[str, ...]
    reservation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        compatible = self.disposition is RouteSystemDisposition.COMPATIBILITY
        if compatible != bool(self.compatibility_reasons):
            raise ValueError("route-system disposition and compatibility disagree")
        member_reservation_ids = tuple(
            dict.fromkeys(
                reservation_id
                for member in self.members
                for reservation_id in member.reservation_ids
            )
        )
        if member_reservation_ids != self.reservation_ids:
            raise ValueError(
                "route-system reservation union disagrees with member claimants"
            )


@dataclass(frozen=True, slots=True)
class RouteSystemEmissionExecution:
    systems: tuple[RouteSystemEmission, ...]
    _by_edge: Mapping[ResolvedEdge, RouteSystemEmission]
    _member_by_edge: Mapping[ResolvedEdge, RouteSystemEmissionMember]
    _covered_by_member: Mapping[EmissionMemberId, EmissionMemberId]

    def system_for_edge(self, edge: Edge | ResolvedEdge) -> RouteSystemEmission | None:
        resolved = (
            edge
            if isinstance(edge, ResolvedEdge)
            else ResolvedEdge(edge.source, edge.target, edge.line_id)
        )
        return self._by_edge.get(resolved)

    def member_for_edge(
        self, edge: Edge | ResolvedEdge
    ) -> RouteSystemEmissionMember | None:
        resolved = (
            edge
            if isinstance(edge, ResolvedEdge)
            else ResolvedEdge(edge.source, edge.target, edge.line_id)
        )
        return self._member_by_edge.get(resolved)

    def covering_member_for(
        self, member_id: EmissionMemberId
    ) -> EmissionMemberId | None:
        return self._covered_by_member.get(member_id)

    def attribute_route(self, route: RoutedPath) -> None:
        system = self.system_for_edge(route.edge)
        member = self.member_for_edge(route.edge)
        if system is None or member is None:
            return
        route.route_system_id = str(system.system_id)
        route.emission_member_id = str(member.member_id)
        route.route_system_disposition = system.disposition.value
        route.route_plan_ids = system.plan_ids
        route.route_reservation_ids = member.reservation_ids


def _compatibility_reason(owner: str, reason: str) -> RouteSystemCompatibilityReason:
    if reason not in _COMPATIBILITY_REASONS.get(owner, ()):
        raise ValueError(f"unregistered compatibility reason {owner}:{reason}")
    return RouteSystemCompatibilityReason(
        owner=owner,
        reason=reason,
        justification=(
            "The owning planner has no complete geometry for this named family; "
            "the established compatibility templates remain its explicit emitter."
        ),
        follow_up=f"Teach {owner} to own {reason!r}, then remove this registry entry.",
    )


@dataclass(frozen=True, slots=True)
class RouteSystemDispositionDecision:
    """One lightweight whole-system disposition in canonical system order."""

    system_id: RouteSystemId
    disposition: RouteSystemDisposition
    compatibility_reasons: tuple[RouteSystemCompatibilityReason, ...]


def classify_route_system_dispositions(
    scaffold: RouteSemanticScaffold,
    *,
    exit_turn_plans: tuple[ExitTurnPlan, ...],
    fan_plans: tuple[FanPlan, ...],
    convergence_plans: tuple[ConvergencePlan, ...],
    member_geometry_failures: Mapping[RouteSystemId, str] | None = None,
) -> tuple[RouteSystemDispositionDecision, ...]:
    """Classify each canonical system without constructing emission members."""
    exit_by_system: dict[RouteSystemId, list[ExitTurnPlan]] = defaultdict(list)
    fan_by_system: dict[RouteSystemId, list[FanPlan]] = defaultdict(list)
    convergence_by_system: dict[RouteSystemId, list[ConvergencePlan]] = defaultdict(
        list
    )
    for exit_plan in exit_turn_plans:
        exit_by_system[exit_plan.system_id].append(exit_plan)
    for fan_plan in fan_plans:
        if fan_plan.system_id is not None:
            fan_by_system[fan_plan.system_id].append(fan_plan)
    for convergence_plan in convergence_plans:
        convergence_by_system[convergence_plan.system_id].append(convergence_plan)

    decisions: list[RouteSystemDispositionDecision] = []
    for system_id in scaffold.ordered_system_ids:
        exit_plans = tuple(exit_by_system.get(system_id, ()))
        system_fans = tuple(fan_by_system.get(system_id, ()))
        convergences = tuple(convergence_by_system.get(system_id, ()))
        geometry_failure = (
            None
            if member_geometry_failures is None
            else member_geometry_failures.get(system_id)
        )
        decisive: tuple[tuple[str, tuple[str, ...]], ...]
        if geometry_failure is not None:
            decisive = (("member-geometry-plan", (geometry_failure,)),)
        elif convergences:
            decisive = (
                (
                    "convergence-plan",
                    tuple(
                        convergence_plan.legacy_reason
                        for convergence_plan in convergences
                        if convergence_plan.disposition is ConvergenceDisposition.LEGACY
                        and convergence_plan.legacy_reason is not None
                    ),
                ),
            )
        elif system_fans and all(
            fan_plan.disposition is FanPlanDisposition.PLANNED
            for fan_plan in system_fans
        ):
            decisive = ()
        else:
            decisive = (
                (
                    "exit-turn-plan",
                    tuple(
                        exit_plan.legacy_reason
                        for exit_plan in exit_plans
                        if exit_plan.disposition is ExitTurnDisposition.LEGACY
                        and exit_plan.legacy_reason is not None
                    ),
                ),
                (
                    "fan-plan",
                    tuple(
                        fan_plan.legacy_reason
                        for fan_plan in system_fans
                        if fan_plan.disposition is FanPlanDisposition.LEGACY
                        and fan_plan.legacy_reason is not None
                    ),
                ),
            )
        reasons = tuple(
            dict.fromkeys(
                _compatibility_reason(owner, reason)
                for owner, owner_reasons in decisive
                for reason in owner_reasons
            )
        )
        decisions.append(
            RouteSystemDispositionDecision(
                system_id,
                (
                    RouteSystemDisposition.COMPATIBILITY
                    if reasons
                    else RouteSystemDisposition.PLANNED
                ),
                reasons,
            )
        )
    return tuple(decisions)


def build_route_system_emission_execution(
    scaffold: RouteSemanticScaffold,
    *,
    exit_turn_plans: tuple[ExitTurnPlan, ...],
    fan_plans: tuple[FanPlan, ...],
    convergence_plans: tuple[ConvergencePlan, ...],
    reservation_ids_by_member: Mapping[EmissionMemberId, tuple[str, ...]] | None = None,
    family_by_edge: Mapping[ResolvedEdge, RouteFamilyId] | None = None,
    member_geometry_plans: tuple[RouteMemberGeometryPlan, ...] = (),
    member_geometry_failures: Mapping[RouteSystemId, str] | None = None,
    require_member_geometry: bool = False,
) -> RouteSystemEmissionExecution:
    """Freeze canonical system order and one emission disposition per system."""
    exit_by_system: dict[RouteSystemId, list[ExitTurnPlan]] = defaultdict(list)
    fan_by_system: dict[RouteSystemId, list[FanPlan]] = defaultdict(list)
    convergence_by_system: dict[RouteSystemId, list[ConvergencePlan]] = defaultdict(
        list
    )
    covered_by_member: dict[EmissionMemberId, EmissionMemberId] = {}
    geometry_by_edge = {plan.edge: plan for plan in member_geometry_plans}
    for exit_plan in exit_turn_plans:
        exit_by_system[exit_plan.system_id].append(exit_plan)
    for fan_plan in fan_plans:
        if fan_plan.system_id is not None:
            fan_by_system[fan_plan.system_id].append(fan_plan)
    for convergence_plan in convergence_plans:
        convergence_by_system[convergence_plan.system_id].append(convergence_plan)
        for ownership in convergence_plan.endpoint_ownership:
            if ownership.covered_by_member_id is None:
                continue
            prior = covered_by_member.setdefault(
                ownership.member_id, ownership.covered_by_member_id
            )
            if prior != ownership.covered_by_member_id:
                raise ValueError(
                    f"emission member {ownership.member_id} has conflicting carriers"
                )

    decisions = {
        decision.system_id: decision
        for decision in classify_route_system_dispositions(
            scaffold,
            exit_turn_plans=exit_turn_plans,
            fan_plans=fan_plans,
            convergence_plans=convergence_plans,
            member_geometry_failures=member_geometry_failures,
        )
    }

    members_by_system: dict[RouteSystemId, list[RouteSystemEmissionMember]] = (
        defaultdict(list)
    )
    for edge in scaffold.edge_order:
        system_id = scaffold.system_for_edge(edge)
        members_by_system[system_id].append(
            RouteSystemEmissionMember(
                scaffold.member_id_by_edge[edge],
                edge,
                None if family_by_edge is None else family_by_edge.get(edge),
                geometry_by_edge.get(edge),
                (
                    ()
                    if reservation_ids_by_member is None
                    else reservation_ids_by_member.get(
                        scaffold.member_id_by_edge[edge], ()
                    )
                ),
            )
        )

    systems: list[RouteSystemEmission] = []
    for system_id, connector_ids in zip(
        scaffold.ordered_system_ids, scaffold.components, strict=True
    ):
        exit_plans = tuple(exit_by_system.get(system_id, ()))
        system_fans = tuple(fan_by_system.get(system_id, ()))
        convergences = tuple(convergence_by_system.get(system_id, ()))
        decision = decisions[system_id]
        compatibility_reasons = decision.compatibility_reasons
        disposition = decision.disposition
        system_members = tuple(members_by_system.get(system_id, ()))
        if disposition is RouteSystemDisposition.COMPATIBILITY:
            system_members = tuple(
                replace(member, geometry_plan=None, reservation_ids=())
                for member in system_members
            )
        system_reservation_ids = (
            tuple(
                dict.fromkeys(
                    reservation_id
                    for member in system_members
                    for reservation_id in member.reservation_ids
                )
            )
            if disposition is RouteSystemDisposition.PLANNED
            else ()
        )
        systems.append(
            RouteSystemEmission(
                system_id,
                tuple(str(item) for item in connector_ids),
                system_members,
                disposition,
                compatibility_reasons,
                (
                    (
                        *tuple(
                            str(exit_plan.id)
                            for exit_plan in exit_plans
                            if exit_plan.disposition is ExitTurnDisposition.PLANNED
                        ),
                        *tuple(
                            str(fan_plan.id)
                            for fan_plan in system_fans
                            if fan_plan.disposition is FanPlanDisposition.PLANNED
                        ),
                        *tuple(
                            str(convergence_plan.id)
                            for convergence_plan in convergences
                            if convergence_plan.disposition
                            is ConvergenceDisposition.PLANNED
                        ),
                        *tuple(
                            str(item.geometry_plan.id)
                            for item in members_by_system.get(system_id, ())
                            if item.geometry_plan is not None
                        ),
                    )
                    if disposition is RouteSystemDisposition.PLANNED
                    else ()
                ),
                system_reservation_ids,
            )
        )

    frozen = tuple(systems)
    for system in frozen if require_member_geometry else ():
        if system.disposition is not RouteSystemDisposition.PLANNED:
            continue
        convergence_edges = {
            edge
            for plan in convergence_by_system.get(system.system_id, ())
            if plan.disposition is ConvergenceDisposition.PLANNED
            for edge in plan.resolved_member_edges
        }
        for member in system.members:
            owners = int(member.geometry_plan is not None) + int(
                member.edge in convergence_edges
            )
            if owners != 1:
                raise ValueError(
                    f"planned route system {system.system_id} member "
                    f"{member.member_id} has {owners} geometry decisions"
                )
    by_edge = {member.edge: system for system in frozen for member in system.members}
    member_by_edge = {
        member.edge: member for system in frozen for member in system.members
    }
    if len(by_edge) != sum(len(system.members) for system in frozen):
        raise ValueError("one emission member belongs to multiple route systems")
    planned_member_ids = {
        member.member_id
        for system in frozen
        if system.disposition is RouteSystemDisposition.PLANNED
        for member in system.members
    }
    final_covered_by_member = {
        member_id: carrier_id
        for member_id, carrier_id in covered_by_member.items()
        if member_id in planned_member_ids
    }
    return RouteSystemEmissionExecution(
        frozen,
        MappingProxyType(by_edge),
        MappingProxyType(member_by_edge),
        MappingProxyType(final_covered_by_member),
    )


def validate_route_system_emission(
    routes: list[RoutedPath],
    execution: RouteSystemEmissionExecution,
    covered_routes: tuple[tuple[tuple[str, str, str], tuple[str, str, str]], ...] = (),
) -> None:
    """Require one emitted or explicitly covered binding per canonical member."""
    covering_by_member = dict(execution._covered_by_member)
    for covered_edge_key, carrier_edge_key in covered_routes:
        covered_member = execution.member_for_edge(ResolvedEdge(*covered_edge_key))
        carrier_member = execution.member_for_edge(ResolvedEdge(*carrier_edge_key))
        if covered_member is None or carrier_member is None:
            raise RuntimeError("covered route has no canonical emission member")
        prior = covering_by_member.setdefault(
            covered_member.member_id, carrier_member.member_id
        )
        if prior != carrier_member.member_id:
            raise RuntimeError(
                f"emission member {covered_member.member_id} has conflicting carriers"
            )
    emitted: dict[EmissionMemberId, int] = defaultdict(int)
    for route in routes:
        system = execution.system_for_edge(route.edge)
        if system is None:
            if (
                route.route_system_id is not None
                or route.emission_member_id is not None
            ):
                raise RuntimeError("attributed route has no canonical emission member")
            continue
        edge = ResolvedEdge(route.edge.source, route.edge.target, route.line_id)
        member = execution.member_for_edge(edge)
        if member is None:
            raise RuntimeError(
                f"route system {system.system_id} has no emission member"
            )
        context = (
            f"route system {system.system_id} connectors "
            f"{', '.join(system.connector_ids)} member {member.member_id} plans "
            f"{', '.join(system.plan_ids) or 'none'} reservations "
            f"{', '.join(member.reservation_ids) or 'none'}"
        )
        if (
            route.route_system_id != str(system.system_id)
            or route.emission_member_id != str(member.member_id)
            or route.route_system_disposition != system.disposition.value
            or route.route_plan_ids != system.plan_ids
            or route.route_reservation_ids != member.reservation_ids
        ):
            raise RuntimeError(f"{context} has inconsistent emission attribution")
        emitted[member.member_id] += 1

    canonical = {
        member.member_id: (system, member)
        for system in execution.systems
        for member in system.members
    }
    for member_id, (system, _member) in canonical.items():
        count = emitted.get(member_id, 0)
        covering_member_id = covering_by_member.get(member_id)
        if covering_member_id is not None:
            if count:
                raise RuntimeError(
                    f"route system {system.system_id} covered member {member_id} "
                    "also emitted a route"
                )
            covering_binding = canonical.get(covering_member_id)
            if covering_binding is None:
                raise RuntimeError(
                    f"route system {system.system_id} member {member_id} has an "
                    "unknown covering member"
                )
            covering_system, _covering_member = covering_binding
            if covering_system.system_id != system.system_id:
                raise RuntimeError(
                    f"route system {system.system_id} member {member_id} has "
                    f"covering member {covering_member_id} from route system "
                    f"{covering_system.system_id}"
                )
            if emitted.get(covering_member_id, 0) != 1:
                raise RuntimeError(
                    f"route system {system.system_id} member {member_id} has no "
                    "emitted covering member"
                )
            continue
        if count != 1:
            raise RuntimeError(
                f"route system {system.system_id} member {member_id} has "
                f"{count} emitted routes"
            )


def validate_published_route_attribution(
    routes: list[RoutedPath], plan: RoutePlan
) -> None:
    """Check final paths against reservations claimed by their emission member."""
    systems = {str(system.id): system for system in plan.systems}
    reservations_by_member = reservation_ids_by_claimant_member(plan.reservations)
    for route in routes:
        if route.route_system_id is None:
            continue
        system = systems[route.route_system_id]
        member_id = EmissionMemberId(route.emission_member_id or "")
        expected = reservations_by_member.get(member_id, ())
        if route.route_reservation_ids != expected:
            raise RuntimeError(
                f"route system {system.id} connectors "
                f"{', '.join(system.connector_ids)} member "
                f"{route.emission_member_id or 'unknown'} plans "
                f"{', '.join(route.route_plan_ids) or 'none'} reservations "
                f"{', '.join(expected) or 'none'} has inconsistent published "
                "reservation attribution"
            )
