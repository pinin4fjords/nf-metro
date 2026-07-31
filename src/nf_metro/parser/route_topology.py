"""Immutable authored-route facts captured before resolver rewrites."""

from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from nf_metro.graph_views import directed_graph
from nf_metro.parser.model import Edge, MetroGraph, PortSide
from nf_metro.parser.resolve import (
    _build_entry_side_mapping,
    _build_exit_side_mapping,
    _classify_edges,
    _exit_side_for_edge,
    _reanchor_flow_axis_ports,
    _reside_folded_flow_ports_to_grid,
)


@dataclass(frozen=True, slots=True)
class LineNetwork:
    """One connected component of one authored metro line."""

    id: str
    line_id: str
    station_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    connector_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteConnector:
    """One authored per-line edge crossing a section boundary."""

    id: str
    authored_edge_ordinal: int
    line_ordinal: int
    source_line: int | None
    source: str
    target: str
    line_id: str
    source_section: str
    target_section: str
    exit_side: PortSide
    entry_side: PortSide
    network_id: str
    bundle_id: str
    exit_group_id: str
    entry_group_id: str


@dataclass(frozen=True, slots=True)
class BundleRun:
    """Authored lines sharing one exact source-target endpoint pair."""

    id: str
    source: str
    target: str
    connector_ids: tuple[str, ...]
    line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointGroup:
    """Connectors sharing one resolved section-boundary port requirement."""

    id: str
    section_id: str
    side: PortSide
    connector_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DivergenceGroup:
    """One exit group that feeds more than one entry group."""

    id: str
    exit_group_id: str
    entry_group_ids: tuple[str, ...]
    connector_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceGroup:
    """Divergences that feed one entry group on the same line."""

    id: str
    entry_group_id: str
    line_id: str
    divergence_ids: tuple[str, ...]
    connector_ids: tuple[str, ...]


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
class _EdgeIdentity:
    edge_ordinal: int
    line_ordinal: int

    @property
    def id(self) -> str:
        return f"edge:{self.edge_ordinal}:line:{self.line_ordinal}"


def _edge_identities(edges: list[Edge]) -> list[_EdgeIdentity]:
    """Assign unique deterministic identities, including for programmatic edges."""
    next_fallback = (
        max(
            (
                edge.authored_edge_ordinal
                for edge in edges
                if edge.authored_edge_ordinal is not None
            ),
            default=-1,
        )
        + 1
    )
    identities: list[_EdgeIdentity] = []
    used: set[tuple[int, int]] = set()
    for edge in edges:
        if edge.authored_edge_ordinal is None:
            identity = _EdgeIdentity(next_fallback, 0)
            next_fallback += 1
        else:
            identity = _EdgeIdentity(
                edge.authored_edge_ordinal,
                edge.authored_line_ordinal or 0,
            )
        if (identity.edge_ordinal, identity.line_ordinal) in used:
            identity = _EdgeIdentity(next_fallback, 0)
            next_fallback += 1
        used.add((identity.edge_ordinal, identity.line_ordinal))
        identities.append(identity)
    return identities


def _line_networks(
    edges: list[Edge], identities: list[_EdgeIdentity], connector_ids: set[str]
) -> tuple[tuple[LineNetwork, ...], dict[str, str]]:
    edges_by_line: dict[str, list[tuple[Edge, _EdgeIdentity]]] = defaultdict(list)
    for edge, identity in zip(edges, identities, strict=True):
        edges_by_line[edge.line_id].append((edge, identity))

    records: list[LineNetwork] = []
    network_by_edge: dict[str, str] = {}
    for line_id in sorted(edges_by_line):
        line_edges = edges_by_line[line_id]
        endpoint_edges = [
            (
                edge.authored_source or edge.source,
                edge.authored_target or edge.target,
            )
            for edge, _identity in line_edges
        ]
        graph = directed_graph(
            sorted({endpoint for edge in endpoint_edges for endpoint in edge}),
            endpoint_edges,
        )
        components = sorted(
            (
                tuple(sorted(component))
                for component in nx.weakly_connected_components(graph)
            ),
            key=lambda stations: stations,
        )
        for component_ordinal, station_ids in enumerate(components):
            station_set = set(station_ids)
            edge_ids = tuple(
                identity.id
                for (source, target), (_edge, identity) in zip(
                    endpoint_edges, line_edges, strict=True
                )
                if source in station_set and target in station_set
            )
            network_id = f"network:{line_id}:{component_ordinal}"
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


def _endpoint_group_id(kind: str, section_id: str, side: PortSide) -> str:
    return f"{kind}:{section_id}:{side.value}"


def build_route_topology(graph: MetroGraph) -> RouteTopology:
    """Observe authored topology without mutating or retaining the input graph.

    The caller invokes this after section layout inference and before any
    terminus-convergence, port, junction, or bypass rewrite. Final port sides
    are derived on a private copy because the authoritative resolver owns the
    same normalisation on the production graph.
    """
    working = copy.deepcopy(graph)
    working.route_topology = None
    identities = _edge_identities(working.edges)
    identity_by_edge = {
        id(edge): identity
        for edge, identity in zip(working.edges, identities, strict=True)
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _internal_edges, inter_section_edges = _classify_edges(working)
        _reside_folded_flow_ports_to_grid(working, inter_section_edges)
        _reanchor_flow_axis_ports(working, inter_section_edges)
        entry_sides = _build_entry_side_mapping(working, inter_section_edges)
    exit_sides = _build_exit_side_mapping(working)

    connector_facts: list[tuple[Edge, _EdgeIdentity, str, str, PortSide, PortSide]] = []
    for edge in inter_section_edges:
        source_section = working.section_for_station(edge.source)
        target_section = working.section_for_station(edge.target)
        assert source_section is not None and target_section is not None
        entry_side = entry_sides.get((target_section, edge.line_id), PortSide.LEFT)
        exit_side = _exit_side_for_edge(
            working,
            edge,
            source_section,
            target_section,
            exit_sides,
            entry_sides,
        )
        connector_facts.append(
            (
                edge,
                identity_by_edge[id(edge)],
                source_section,
                target_section,
                exit_side,
                entry_side,
            )
        )

    connector_ids = {identity.id for _, identity, *_ in connector_facts}
    networks, network_by_edge = _line_networks(working.edges, identities, connector_ids)

    bundles_by_endpoint: dict[tuple[str, str], list[str]] = defaultdict(list)
    lines_by_bundle: dict[tuple[str, str], set[str]] = defaultdict(set)
    exit_members: dict[tuple[str, PortSide], list[str]] = defaultdict(list)
    entry_members: dict[tuple[str, PortSide], list[str]] = defaultdict(list)
    for (
        edge,
        identity,
        source_section,
        target_section,
        exit_side,
        entry_side,
    ) in connector_facts:
        source = edge.authored_source or edge.source
        target = edge.authored_target or edge.target
        bundles_by_endpoint[(source, target)].append(identity.id)
        lines_by_bundle[(source, target)].add(edge.line_id)
        exit_members[(source_section, exit_side)].append(identity.id)
        entry_members[(target_section, entry_side)].append(identity.id)

    bundle_keys = sorted(bundles_by_endpoint)
    bundle_id_by_endpoint = {
        endpoint: f"bundle:{ordinal}" for ordinal, endpoint in enumerate(bundle_keys)
    }
    bundles = tuple(
        BundleRun(
            id=bundle_id_by_endpoint[(source, target)],
            source=source,
            target=target,
            connector_ids=tuple(bundles_by_endpoint[(source, target)]),
            line_ids=tuple(sorted(lines_by_bundle[(source, target)])),
        )
        for source, target in bundle_keys
    )

    exit_keys = sorted(exit_members, key=lambda item: (item[0], item[1].value))
    entry_keys = sorted(entry_members, key=lambda item: (item[0], item[1].value))
    exit_groups = tuple(
        EndpointGroup(
            id=_endpoint_group_id("exit", section_id, side),
            section_id=section_id,
            side=side,
            connector_ids=tuple(exit_members[(section_id, side)]),
        )
        for section_id, side in exit_keys
    )
    entry_groups = tuple(
        EndpointGroup(
            id=_endpoint_group_id("entry", section_id, side),
            section_id=section_id,
            side=side,
            connector_ids=tuple(entry_members[(section_id, side)]),
        )
        for section_id, side in entry_keys
    )

    connectors = tuple(
        RouteConnector(
            id=identity.id,
            authored_edge_ordinal=identity.edge_ordinal,
            line_ordinal=identity.line_ordinal,
            source_line=edge.source_line,
            source=edge.authored_source or edge.source,
            target=edge.authored_target or edge.target,
            line_id=edge.line_id,
            source_section=source_section,
            target_section=target_section,
            exit_side=exit_side,
            entry_side=entry_side,
            network_id=network_by_edge[identity.id],
            bundle_id=bundle_id_by_endpoint[
                (
                    edge.authored_source or edge.source,
                    edge.authored_target or edge.target,
                )
            ],
            exit_group_id=_endpoint_group_id("exit", source_section, exit_side),
            entry_group_id=_endpoint_group_id("entry", target_section, entry_side),
        )
        for (
            edge,
            identity,
            source_section,
            target_section,
            exit_side,
            entry_side,
        ) in connector_facts
    )

    entry_targets_by_exit: dict[str, set[str]] = defaultdict(set)
    for connector in connectors:
        entry_targets_by_exit[connector.exit_group_id].add(connector.entry_group_id)
    divergent_exit_ids = sorted(
        exit_group_id
        for exit_group_id, entry_group_ids in entry_targets_by_exit.items()
        if len(entry_group_ids) > 1
    )
    divergences = tuple(
        DivergenceGroup(
            id=f"divergence:{ordinal}",
            exit_group_id=exit_group_id,
            entry_group_ids=tuple(sorted(entry_targets_by_exit[exit_group_id])),
            connector_ids=tuple(
                connector.id
                for connector in connectors
                if connector.exit_group_id == exit_group_id
            ),
        )
        for ordinal, exit_group_id in enumerate(divergent_exit_ids)
    )

    convergence_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    divergence_by_exit = {
        divergence.exit_group_id: divergence for divergence in divergences
    }
    for connector in connectors:
        divergence = divergence_by_exit.get(connector.exit_group_id)
        if divergence is not None:
            convergence_sources[(connector.entry_group_id, connector.line_id)].add(
                divergence.id
            )
    convergence_keys = sorted(
        key
        for key, divergence_ids in convergence_sources.items()
        if len(divergence_ids) > 1
    )
    convergences = tuple(
        ConvergenceGroup(
            id=f"convergence:{ordinal}",
            entry_group_id=entry_group_id,
            line_id=line_id,
            divergence_ids=tuple(
                sorted(convergence_sources[(entry_group_id, line_id)])
            ),
            connector_ids=tuple(
                connector.id
                for connector in connectors
                if connector.entry_group_id == entry_group_id
                and connector.line_id == line_id
                and connector.exit_group_id in divergence_by_exit
            ),
        )
        for ordinal, (entry_group_id, line_id) in enumerate(convergence_keys)
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
