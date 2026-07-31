"""Immutable authored-route facts captured before resolver rewrites."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, NewType

import networkx as nx

from nf_metro.graph_views import directed_graph
from nf_metro.parser.model import Edge, MetroGraph, PortSide

if TYPE_CHECKING:
    from nf_metro.parser.resolve import SectionEndpointResolution


ConnectorId = NewType("ConnectorId", str)
NetworkId = NewType("NetworkId", str)
BundleId = NewType("BundleId", str)
EndpointGroupId = NewType("EndpointGroupId", str)
DivergenceId = NewType("DivergenceId", str)
ConvergenceId = NewType("ConvergenceId", str)


class ResolvedEdge(NamedTuple):
    """One final graph edge in an authored connector's resolved path."""

    source: str
    target: str
    line_id: str


def _route_id(kind: str, *parts: object) -> str:
    """Build an unambiguous content-derived identifier from typed parts."""
    encoded = "|".join(f"{len(str(part))}:{part}" for part in parts)
    return f"{kind}|{encoded}"


@dataclass(frozen=True, slots=True)
class AuthoredEdgeKey:
    """Stable identity for one authored per-line edge."""

    source: str
    target: str
    line_id: str
    duplicate_ordinal: int

    @property
    def id(self) -> ConnectorId:
        """Identity unaffected by authored edges with different content."""
        return ConnectorId(
            _route_id(
                "connector",
                self.source,
                self.target,
                self.line_id,
                self.duplicate_ordinal,
            )
        )


@dataclass(frozen=True, slots=True)
class AuthoredEdgeFact:
    """Source facts and ordering for one authored per-line edge."""

    key: AuthoredEdgeKey
    rank: int
    source_line: int | None
    source_section: str | None
    target_section: str | None


@dataclass(frozen=True, slots=True)
class AuthoredRouteCapture:
    """Authored route facts and definition ranks before synthetic rewrites."""

    edges: tuple[AuthoredEdgeFact, ...]
    line_ids: tuple[str, ...]
    station_ids: tuple[str, ...]
    section_ids: tuple[str, ...]


@dataclass(slots=True)
class AuthoredEdgeLineage:
    """Parser-local mapping from current edges to their authored origins."""

    _origins_by_edge_id: dict[int, tuple[AuthoredEdgeKey, ...]]
    _rank_by_key: dict[AuthoredEdgeKey, int]

    @classmethod
    def from_capture(
        cls,
        edges: list[Edge],
        capture: AuthoredRouteCapture,
    ) -> AuthoredEdgeLineage:
        """Associate each captured edge with its current graph edge object."""
        if len(edges) != len(capture.edges):
            raise RouteTopologyLineageError(
                "authored capture and graph edge counts differ"
            )
        return cls(
            {
                id(edge): (fact.key,)
                for edge, fact in zip(edges, capture.edges, strict=True)
            },
            {fact.key: fact.rank for fact in capture.edges},
        )

    def origins(self, edge: Edge) -> tuple[AuthoredEdgeKey, ...]:
        """Return the authored origins carried by a current edge."""
        return self._origins_by_edge_id.get(id(edge), ())

    def replace(self, old_edge: Edge, new_edge: Edge) -> None:
        """Transfer one edge's origins to its replacement."""
        origins = self._origins_by_edge_id.pop(id(old_edge), ())
        self._origins_by_edge_id[id(new_edge)] = origins

    def discard(self, edge: Edge) -> None:
        """Remove an edge that has been replaced or deleted."""
        self._origins_by_edge_id.pop(id(edge), None)

    def bind(
        self, edge: Edge, origins: tuple[AuthoredEdgeKey, ...] | list[AuthoredEdgeKey]
    ) -> None:
        """Associate a synthetic edge with an authored-order origin union."""
        self._origins_by_edge_id[id(edge)] = self.ordered_union(origins)

    def ordered_union(
        self, origins: tuple[AuthoredEdgeKey, ...] | list[AuthoredEdgeKey]
    ) -> tuple[AuthoredEdgeKey, ...]:
        """Deduplicate origins and order them by their authored rank."""
        unique = set(origins)
        return tuple(sorted(unique, key=self._rank_by_key.__getitem__))


class RouteTopologyLineageError(RuntimeError):
    """The synthetic edge lineage cannot map authored connectors exactly."""


def capture_authored_routes(graph: MetroGraph) -> AuthoredRouteCapture:
    """Capture authored per-line edges and definition order without mutation."""
    occurrences: dict[tuple[str, str, str], int] = defaultdict(int)
    facts: list[AuthoredEdgeFact] = []
    for rank, edge in enumerate(graph.edges):
        content = (edge.source, edge.target, edge.line_id)
        duplicate_ordinal = occurrences[content]
        occurrences[content] += 1
        facts.append(
            AuthoredEdgeFact(
                key=AuthoredEdgeKey(*content, duplicate_ordinal),
                rank=rank,
                source_line=edge.source_line,
                source_section=graph.section_for_station(edge.source),
                target_section=graph.section_for_station(edge.target),
            )
        )
    return AuthoredRouteCapture(
        edges=tuple(facts),
        line_ids=tuple(graph.lines),
        station_ids=tuple(graph.stations),
        section_ids=tuple(graph.sections),
    )


@dataclass(frozen=True, slots=True)
class LineNetwork:
    """One connected component of one authored metro line."""

    id: NetworkId
    line_id: str
    station_ids: tuple[str, ...]
    edge_ids: tuple[ConnectorId, ...]
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class RouteConnector:
    """One authored per-line edge crossing a section boundary."""

    id: ConnectorId
    duplicate_ordinal: int
    source_line: int | None
    source: str
    target: str
    line_id: str
    source_section: str
    target_section: str
    exit_side: PortSide
    entry_side: PortSide
    network_id: NetworkId
    bundle_id: BundleId
    exit_group_id: EndpointGroupId
    entry_group_id: EndpointGroupId


@dataclass(frozen=True, slots=True)
class BundleRun:
    """Authored lines sharing one exact source-target endpoint pair."""

    id: BundleId
    source: str
    target: str
    connector_ids: tuple[ConnectorId, ...]
    line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointGroup:
    """Connectors sharing one resolved section-boundary port requirement."""

    id: EndpointGroupId
    section_id: str
    side: PortSide
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class DivergenceGroup:
    """One exit group that feeds more than one entry group."""

    id: DivergenceId
    exit_group_id: EndpointGroupId
    entry_group_ids: tuple[EndpointGroupId, ...]
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceGroup:
    """Divergences that feed one entry group on the same line."""

    id: ConvergenceId
    entry_group_id: EndpointGroupId
    line_id: str
    divergence_ids: tuple[DivergenceId, ...]
    connector_ids: tuple[ConnectorId, ...]


@dataclass(frozen=True, slots=True)
class RouteTopology:
    """Canonical authored route topology, independent of mutable graph models."""

    line_networks: tuple[LineNetwork, ...]
    connectors: tuple[RouteConnector, ...]
    bundles: tuple[BundleRun, ...]
    exit_groups: tuple[EndpointGroup, ...]
    entry_groups: tuple[EndpointGroup, ...]
    divergences: tuple[DivergenceGroup, ...]
    convergences: tuple[ConvergenceGroup, ...]


@dataclass(frozen=True, slots=True)
class ResolvedEndpointPort:
    """Synthetic port created for one topology endpoint group."""

    group_id: EndpointGroupId
    port_id: str


@dataclass(frozen=True, slots=True)
class ResolvedDivergence:
    """Synthetic junction created for one topology divergence."""

    group_id: DivergenceId
    junction_id: str


@dataclass(frozen=True, slots=True)
class ResolvedConvergence:
    """Synthetic junction created for one topology convergence."""

    group_id: ConvergenceId
    junction_id: str


@dataclass(frozen=True, slots=True)
class ResolvedConnector:
    """Final ordered graph-edge paths for one authored connector."""

    connector_id: ConnectorId
    edge_paths: tuple[tuple[ResolvedEdge, ...], ...]


@dataclass(frozen=True, slots=True)
class RouteResolutionTrace:
    """Immutable mapping from authored topology to resolved synthetic ids."""

    connectors: tuple[ResolvedConnector, ...] = ()
    exit_ports: tuple[ResolvedEndpointPort, ...] = ()
    entry_ports: tuple[ResolvedEndpointPort, ...] = ()
    divergences: tuple[ResolvedDivergence, ...] = ()
    convergences: tuple[ResolvedConvergence, ...] = ()


@dataclass(frozen=True, slots=True)
class _ResolvedAuthoredConnector:
    fact: AuthoredEdgeFact
    source_section: str
    target_section: str
    exit_side: PortSide
    entry_side: PortSide


def _definition_rank(values: tuple[str, ...]) -> dict[str, int]:
    return {value: rank for rank, value in enumerate(values)}


def _rank_key(rank: dict[str, int], value: str) -> tuple[int, str]:
    return rank.get(value, len(rank)), value


def _line_networks(
    capture: AuthoredRouteCapture,
    connector_ids: set[ConnectorId],
) -> tuple[tuple[LineNetwork, ...], dict[ConnectorId, NetworkId]]:
    edges_by_line: dict[str, list[AuthoredEdgeFact]] = defaultdict(list)
    for fact in capture.edges:
        edges_by_line[fact.key.line_id].append(fact)

    line_rank = _definition_rank(capture.line_ids)
    station_rank = _definition_rank(capture.station_ids)
    records: list[LineNetwork] = []
    network_by_edge: dict[ConnectorId, NetworkId] = {}

    for line_id in sorted(edges_by_line, key=lambda item: _rank_key(line_rank, item)):
        line_edges = edges_by_line[line_id]
        graph = directed_graph(
            {
                endpoint
                for fact in line_edges
                for endpoint in (fact.key.source, fact.key.target)
            },
            [(fact.key.source, fact.key.target) for fact in line_edges],
        )
        components = [
            tuple(sorted(component, key=lambda item: _rank_key(station_rank, item)))
            for component in nx.weakly_connected_components(graph)
        ]
        components.sort(
            key=lambda stations: min(
                _rank_key(station_rank, station) for station in stations
            )
        )
        component_by_station = {
            station: component_rank
            for component_rank, stations in enumerate(components)
            for station in stations
        }
        edges_by_component: list[list[AuthoredEdgeFact]] = [
            [] for _component in components
        ]
        for fact in line_edges:
            component_rank = component_by_station[fact.key.source]
            edges_by_component[component_rank].append(fact)

        for station_ids, component_edges in zip(
            components, edges_by_component, strict=True
        ):
            edge_ids = tuple(fact.key.id for fact in component_edges)
            network_id = NetworkId(_route_id("network", line_id, *edge_ids))
            records.append(
                LineNetwork(
                    id=network_id,
                    line_id=line_id,
                    station_ids=station_ids,
                    edge_ids=edge_ids,
                    connector_ids=tuple(
                        edge_id for edge_id in edge_ids if edge_id in connector_ids
                    ),
                )
            )
            for edge_id in edge_ids:
                network_by_edge[edge_id] = network_id

    return tuple(records), network_by_edge


def _endpoint_group_id(kind: str, section_id: str, side: PortSide) -> EndpointGroupId:
    return EndpointGroupId(_route_id(kind, section_id, side.value))


def _resolved_connectors(
    capture: AuthoredRouteCapture,
    lineage: AuthoredEdgeLineage,
    boundary_plan: SectionEndpointResolution | None,
) -> tuple[_ResolvedAuthoredConnector, ...]:
    if boundary_plan is None:
        return ()

    fact_by_key = {fact.key: fact for fact in capture.edges}
    assignments: dict[AuthoredEdgeKey, _ResolvedAuthoredConnector] = {}
    assignment_counts: dict[AuthoredEdgeKey, int] = defaultdict(int)
    for endpoint in boundary_plan.connectors:
        for key in lineage.origins(endpoint.edge):
            fact = fact_by_key[key]
            if fact.source_section == fact.target_section:
                continue
            assignment_counts[key] += 1
            assignments[key] = _ResolvedAuthoredConnector(
                fact=fact,
                source_section=endpoint.source_section,
                target_section=endpoint.target_section,
                exit_side=endpoint.exit_side,
                entry_side=endpoint.entry_side,
            )

    expected = [
        fact
        for fact in capture.edges
        if fact.source_section is not None
        and fact.target_section is not None
        and fact.source_section != fact.target_section
    ]
    invalid = [
        (fact.key.id, assignment_counts[fact.key])
        for fact in expected
        if assignment_counts[fact.key] != 1
    ]
    if invalid:
        detail = ", ".join(f"{connector_id}={count}" for connector_id, count in invalid)
        raise RouteTopologyLineageError(
            "authored cross-section connectors require one boundary assignment: "
            f"{detail}"
        )

    return tuple(assignments[fact.key] for fact in expected)


def build_route_topology(
    capture: AuthoredRouteCapture,
    lineage: AuthoredEdgeLineage,
    boundary_plan: SectionEndpointResolution | None = None,
) -> RouteTopology:
    """Project authored topology through the resolver's authoritative boundaries."""
    resolved = _resolved_connectors(capture, lineage, boundary_plan)
    connector_ids = {item.fact.key.id for item in resolved}
    networks, network_by_edge = _line_networks(capture, connector_ids)

    line_rank = _definition_rank(capture.line_ids)
    bundles_by_endpoint: dict[tuple[str, str], list[ConnectorId]] = defaultdict(list)
    lines_by_bundle: dict[tuple[str, str], set[str]] = defaultdict(set)
    exit_members: dict[tuple[str, PortSide], list[ConnectorId]] = defaultdict(list)
    entry_members: dict[tuple[str, PortSide], list[ConnectorId]] = defaultdict(list)
    for item in resolved:
        fact = item.fact
        endpoint = (fact.key.source, fact.key.target)
        bundles_by_endpoint[endpoint].append(fact.key.id)
        lines_by_bundle[endpoint].add(fact.key.line_id)
        exit_members[(item.source_section, item.exit_side)].append(fact.key.id)
        entry_members[(item.target_section, item.entry_side)].append(fact.key.id)

    bundle_id_by_endpoint = {
        endpoint: BundleId(_route_id("bundle", *endpoint))
        for endpoint in bundles_by_endpoint
    }
    bundles = tuple(
        BundleRun(
            id=bundle_id_by_endpoint[(source, target)],
            source=source,
            target=target,
            connector_ids=tuple(bundles_by_endpoint[(source, target)]),
            line_ids=tuple(
                sorted(
                    lines_by_bundle[(source, target)],
                    key=lambda item: _rank_key(line_rank, item),
                )
            ),
        )
        for source, target in bundles_by_endpoint
    )

    exit_groups = tuple(
        EndpointGroup(
            id=_endpoint_group_id("exit", section_id, side),
            section_id=section_id,
            side=side,
            connector_ids=tuple(members),
        )
        for (section_id, side), members in exit_members.items()
    )
    entry_groups = tuple(
        EndpointGroup(
            id=_endpoint_group_id("entry", section_id, side),
            section_id=section_id,
            side=side,
            connector_ids=tuple(members),
        )
        for (section_id, side), members in entry_members.items()
    )

    connectors = tuple(
        RouteConnector(
            id=item.fact.key.id,
            duplicate_ordinal=item.fact.key.duplicate_ordinal,
            source_line=item.fact.source_line,
            source=item.fact.key.source,
            target=item.fact.key.target,
            line_id=item.fact.key.line_id,
            source_section=item.source_section,
            target_section=item.target_section,
            exit_side=item.exit_side,
            entry_side=item.entry_side,
            network_id=network_by_edge[item.fact.key.id],
            bundle_id=bundle_id_by_endpoint[
                (item.fact.key.source, item.fact.key.target)
            ],
            exit_group_id=_endpoint_group_id(
                "exit", item.source_section, item.exit_side
            ),
            entry_group_id=_endpoint_group_id(
                "entry", item.target_section, item.entry_side
            ),
        )
        for item in resolved
    )

    entry_targets_by_exit: dict[EndpointGroupId, list[EndpointGroupId]] = defaultdict(
        list
    )
    connector_ids_by_exit: dict[EndpointGroupId, list[ConnectorId]] = defaultdict(list)
    for connector in connectors:
        targets = entry_targets_by_exit[connector.exit_group_id]
        if connector.entry_group_id not in targets:
            targets.append(connector.entry_group_id)
        connector_ids_by_exit[connector.exit_group_id].append(connector.id)

    divergences = tuple(
        DivergenceGroup(
            id=DivergenceId(_route_id("divergence", exit_group_id, *entry_group_ids)),
            exit_group_id=exit_group_id,
            entry_group_ids=tuple(entry_group_ids),
            connector_ids=tuple(connector_ids_by_exit[exit_group_id]),
        )
        for exit_group_id, entry_group_ids in entry_targets_by_exit.items()
        if len(entry_group_ids) > 1
    )

    divergence_by_exit = {
        divergence.exit_group_id: divergence for divergence in divergences
    }
    divergence_ids_by_entry_line: dict[
        tuple[EndpointGroupId, str], list[DivergenceId]
    ] = defaultdict(list)
    connector_ids_by_entry_line: dict[
        tuple[EndpointGroupId, str], list[ConnectorId]
    ] = defaultdict(list)
    for connector in connectors:
        divergence = divergence_by_exit.get(connector.exit_group_id)
        if divergence is None:
            continue
        key = (connector.entry_group_id, connector.line_id)
        if divergence.id not in divergence_ids_by_entry_line[key]:
            divergence_ids_by_entry_line[key].append(divergence.id)
        connector_ids_by_entry_line[key].append(connector.id)

    convergences = tuple(
        ConvergenceGroup(
            id=ConvergenceId(
                _route_id("convergence", entry_group_id, line_id, *divergence_ids)
            ),
            entry_group_id=entry_group_id,
            line_id=line_id,
            divergence_ids=tuple(divergence_ids),
            connector_ids=tuple(connector_ids_by_entry_line[(entry_group_id, line_id)]),
        )
        for (
            entry_group_id,
            line_id,
        ), divergence_ids in divergence_ids_by_entry_line.items()
        if len(divergence_ids) > 1
    )

    return RouteTopology(
        line_networks=networks,
        connectors=connectors,
        bundles=bundles,
        exit_groups=exit_groups,
        entry_groups=entry_groups,
        divergences=divergences,
        convergences=convergences,
    )
