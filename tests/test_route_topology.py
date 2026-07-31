"""Authored route topology contracts across parser resolution rewrites."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from nf_metro.layout.routing.common import merge_junction_ids
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import (
    Edge,
    MetroGraph,
    PortSide,
    Section,
    Station,
    is_bypass_v,
    is_converge_junction,
)
from nf_metro.parser.resolve import resolve_section_endpoints
from nf_metro.parser.route_topology import (
    AuthoredEdgeLineage,
    RouteTopology,
    build_route_topology,
    capture_authored_routes,
)

ROOT = Path(__file__).parents[1]
CORPUS = sorted(
    path
    for root in (ROOT / "examples", ROOT / "tests" / "fixtures")
    for path in root.rglob("*.mmd")
    if "nextflow" not in path.parts
)


def _parse(text: str) -> MetroGraph:
    graph = parse_metro_mermaid(text)
    assert graph.route_topology is not None
    return graph


def _two_section_text(edges: str, *, extra: str = "") -> str:
    return f"""\
%%metro line: red | Red | #f00
%%metro line: blue | Blue | #00f
{extra}graph LR
    subgraph source [Source]
        a[A]
    end
    subgraph target [Target]
        b[B]
    end
{edges}
"""


def test_connector_identity_preserves_multiline_order_and_exact_duplicates() -> None:
    topology = _parse(
        _two_section_text("    a -->|red,blue| b\n    a -->|red| b\n")
    ).route_topology
    assert topology is not None

    assert [
        (connector.source, connector.target, connector.line_id)
        for connector in topology.connectors
    ] == [("a", "b", "red"), ("a", "b", "blue"), ("a", "b", "red")]
    assert [connector.duplicate_ordinal for connector in topology.connectors] == [
        0,
        0,
        1,
    ]
    assert len({connector.id for connector in topology.connectors}) == 3
    assert topology.bundles[0].connector_ids == tuple(
        connector.id for connector in topology.connectors
    )
    assert topology.bundles[0].line_ids == ("red", "blue")


def test_line_networks_are_stable_connected_components() -> None:
    base = """\
%%metro line: red | Red | #f00
graph LR
    subgraph one [One]
        a[A]
        b[B]
        a -->|red| b
    end
    subgraph two [Two]
        c[C]
        d[D]
        c -->|red| d
    end
"""
    expanded = (
        base.replace(
            "graph LR",
            "%%metro line: green | Green | #0f0\ngraph LR",
        )
        + """
    subgraph unrelated [Unrelated]
        u[U]
        v[V]
        u -->|green| v
    end
"""
    )

    original = _parse(base).route_topology
    observed = _parse(expanded).route_topology
    assert original is not None and observed is not None

    original_red = {
        network.station_ids: network.id
        for network in original.line_networks
        if network.line_id == "red"
    }
    observed_red = {
        network.station_ids: network.id
        for network in observed.line_networks
        if network.line_id == "red"
    }
    assert original_red == observed_red
    assert tuple(original_red) == (("a", "b"), ("c", "d"))


def test_programmatic_duplicate_edges_get_stable_unique_identities() -> None:
    graph = MetroGraph(
        stations={
            "a": Station(id="a", label="A", section_id="source"),
            "b": Station(id="b", label="B", section_id="target"),
        },
        edges=[
            Edge(source="a", target="b", line_id="red"),
            Edge(source="a", target="b", line_id="red"),
        ],
        sections={
            "source": Section(id="source", name="Source", station_ids=["a"]),
            "target": Section(id="target", name="Target", station_ids=["b"]),
        },
    )
    capture = capture_authored_routes(graph)
    lineage = AuthoredEdgeLineage.from_capture(graph.edges, capture)
    resolution = resolve_section_endpoints(graph)

    topology = build_route_topology(capture, lineage, resolution)

    assert [connector.duplicate_ordinal for connector in topology.connectors] == [
        0,
        1,
    ]
    assert len({connector.id for connector in topology.connectors}) == 2
    assert all(connector.source_line is None for connector in topology.connectors)


def test_terminus_topology_uses_the_resolvers_reanchored_entry_side() -> None:
    graph = _parse(
        """\
%%metro line: red | Red | #f00
%%metro file: output | Results
graph LR
    subgraph upstream [Upstream]
        a[A]
    end
    subgraph output_section [Output]
        %%metro direction: LR
        %%metro entry: right | red
        b[B]
        output[Output]
        b -->|red| output
    end
    a -->|red| output
"""
    )
    topology = graph.route_topology
    assert topology is not None

    connector = next(
        connector for connector in topology.connectors if connector.source == "a"
    )
    assert connector.target == "output"
    assert connector.entry_side is PortSide.LEFT
    assert {
        port.side
        for port in graph.ports.values()
        if port.section_id == "output_section" and port.is_entry
    } == {PortSide.LEFT}


def test_duplicate_terminus_connectors_survive_many_to_one_lineage() -> None:
    topology = _parse(
        """\
%%metro line: red | Red | #f00
%%metro file: output | Results
graph LR
    subgraph upstream [Upstream]
        a[A]
    end
    subgraph output_section [Output]
        b[B]
        output[Output]
        b -->|red| output
    end
    a -->|red| output
    a -->|red| output
"""
    ).route_topology
    assert topology is not None

    connectors = [
        connector
        for connector in topology.connectors
        if (connector.source, connector.target, connector.line_id)
        == ("a", "output", "red")
    ]
    assert [connector.duplicate_ordinal for connector in connectors] == [0, 1]
    assert len({connector.id for connector in connectors}) == 2
    assert connectors[0].entry_group_id == connectors[1].entry_group_id


def test_topology_records_are_deeply_immutable_and_detached() -> None:
    graph = _parse(_two_section_text("    a -->|red| b\n"))
    topology = graph.route_topology
    assert topology is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.connectors = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.connectors[0].source = "other"  # type: ignore[misc]

    mutable_models = (MetroGraph, Edge, Section, Station, list, dict, set)

    def visit(value: object) -> None:
        assert not isinstance(value, mutable_models)
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                visit(item)

    visit(topology)
    assert [field.name for field in dataclasses.fields(Edge)] == [
        "source",
        "target",
        "line_id",
        "source_line",
    ]


def _identity_maps(topology: RouteTopology) -> dict[str, dict[tuple, str]]:
    return {
        "connectors": {
            (
                item.source,
                item.target,
                item.line_id,
                item.duplicate_ordinal,
            ): item.id
            for item in topology.connectors
        },
        "bundles": {(item.source, item.target): item.id for item in topology.bundles},
        "networks": {
            (item.line_id, item.edge_ids): item.id for item in topology.line_networks
        },
        "divergences": {
            (item.exit_group_id, item.entry_group_ids): item.id
            for item in topology.divergences
        },
        "convergences": {
            (item.entry_group_id, item.line_id, item.divergence_ids): item.id
            for item in topology.convergences
        },
    }


def test_ids_do_not_shift_when_an_unrelated_component_is_added() -> None:
    base = (ROOT / "examples" / "guide" / "03b_fan_in_merge.mmd").read_text()
    expanded = (
        base.replace(
            "graph LR",
            "%%metro line: unrelated | Unrelated | #0f0\ngraph LR",
        )
        + """
    subgraph unrelated [Unrelated]
        unrelated_a[A]
        unrelated_b[B]
        unrelated_a -->|unrelated| unrelated_b
    end
"""
    )
    original = _parse(base).route_topology
    observed = _parse(expanded).route_topology
    assert original is not None and observed is not None

    for kind, original_ids in _identity_maps(original).items():
        observed_ids = _identity_maps(observed)[kind]
        assert observed_ids.items() >= original_ids.items()


def _actual_shapes(
    graph: MetroGraph,
) -> tuple[set[tuple], set[tuple], set[tuple], set[tuple]]:
    port_key = {
        port_id: (port.section_id, port.side.value)
        for port_id, port in graph.ports.items()
    }
    exit_members: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    entry_members: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    junction_exit: dict[str, tuple[str, str]] = {}
    junction_targets: dict[str, set[tuple[tuple[str, str], str]]] = defaultdict(set)

    def authored_source(station_id: str, line_id: str) -> str:
        while is_bypass_v(station_id):
            incoming = [
                edge for edge in graph.edges_to(station_id) if edge.line_id == line_id
            ]
            assert len(incoming) == 1
            station_id = incoming[0].source
        return station_id

    def authored_target(station_id: str, line_id: str) -> str:
        while is_converge_junction(station_id):
            outgoing = [
                edge for edge in graph.edges_from(station_id) if edge.line_id == line_id
            ]
            assert len(outgoing) == 1
            station_id = outgoing[0].target
        return station_id

    for edge in graph.edges:
        target_port = graph.ports.get(edge.target)
        source_port = graph.ports.get(edge.source)
        if target_port is not None and not target_port.is_entry:
            exit_members[port_key[edge.target]].add(
                (authored_source(edge.source, edge.line_id), edge.line_id)
            )
        if source_port is not None and source_port.is_entry:
            entry_members[port_key[edge.source]].add(
                (authored_target(edge.target, edge.line_id), edge.line_id)
            )
        if (
            source_port is not None
            and not source_port.is_entry
            and graph.is_fanout_junction(edge.target)
        ):
            junction_exit[edge.target] = port_key[edge.source]
        if (
            graph.is_fanout_junction(edge.source)
            and target_port is not None
            and target_port.is_entry
        ):
            junction_targets[edge.source].add((port_key[edge.target], edge.line_id))

    merge_ids = merge_junction_ids(graph)
    fan_ids = {
        junction for junction in graph.junctions if graph.is_fanout_junction(junction)
    }
    for fan_id in fan_ids:
        for edge in graph.edges_from(fan_id):
            if edge.target not in merge_ids:
                continue
            for outgoing in graph.edges_from(edge.target):
                target_port = graph.ports.get(outgoing.target)
                if target_port is not None and target_port.is_entry:
                    junction_targets[fan_id].add(
                        (port_key[outgoing.target], outgoing.line_id)
                    )
    fan_groups = {
        (junction_exit[junction], tuple(sorted(junction_targets[junction])))
        for junction in fan_ids
        if junction in junction_exit
    }

    merge_groups: set[tuple] = set()
    for merge_id in merge_ids:
        outgoing = [
            edge for edge in graph.edges_from(merge_id) if edge.target in graph.ports
        ]
        assert len(outgoing) == 1
        target = outgoing[0]
        source_exit_groups = {
            junction_exit[edge.source]
            for edge in graph.edges_to(merge_id)
            if edge.source in junction_exit
        }
        merge_groups.add(
            (
                port_key[target.target],
                target.line_id,
                tuple(sorted(source_exit_groups)),
            )
        )

    return (
        {(key, tuple(sorted(members))) for key, members in exit_members.items()},
        {(key, tuple(sorted(members))) for key, members in entry_members.items()},
        fan_groups,
        merge_groups,
    )


def _predicted_shapes(
    topology: RouteTopology,
) -> tuple[set[tuple], set[tuple], set[tuple], set[tuple]]:
    connectors = {connector.id: connector for connector in topology.connectors}
    exit_groups = {
        (
            (group.section_id, group.side.value),
            tuple(
                sorted(
                    {
                        (
                            connectors[connector_id].source,
                            connectors[connector_id].line_id,
                        )
                        for connector_id in group.connector_ids
                    }
                )
            ),
        )
        for group in topology.exit_groups
    }
    entry_groups = {
        (
            (group.section_id, group.side.value),
            tuple(
                sorted(
                    {
                        (
                            connectors[connector_id].target,
                            connectors[connector_id].line_id,
                        )
                        for connector_id in group.connector_ids
                    }
                )
            ),
        )
        for group in topology.entry_groups
    }
    entry_by_id = {group.id: group for group in topology.entry_groups}
    exit_by_id = {group.id: group for group in topology.exit_groups}
    divergence_by_id = {group.id: group for group in topology.divergences}
    fan_groups = {
        (
            (
                exit_by_id[group.exit_group_id].section_id,
                exit_by_id[group.exit_group_id].side.value,
            ),
            tuple(
                sorted(
                    {
                        (
                            (
                                entry_by_id[
                                    connectors[connector_id].entry_group_id
                                ].section_id,
                                entry_by_id[
                                    connectors[connector_id].entry_group_id
                                ].side.value,
                            ),
                            connectors[connector_id].line_id,
                        )
                        for connector_id in group.connector_ids
                    }
                )
            ),
        )
        for group in topology.divergences
    }
    merge_groups = {
        (
            (
                entry_by_id[group.entry_group_id].section_id,
                entry_by_id[group.entry_group_id].side.value,
            ),
            group.line_id,
            tuple(
                sorted(
                    (
                        exit_by_id[
                            divergence_by_id[divergence_id].exit_group_id
                        ].section_id,
                        exit_by_id[
                            divergence_by_id[divergence_id].exit_group_id
                        ].side.value,
                    )
                    for divergence_id in group.divergence_ids
                )
            ),
        )
        for group in topology.convergences
    }
    return exit_groups, entry_groups, fan_groups, merge_groups


@pytest.mark.parametrize(
    "path", CORPUS, ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_corpus_topology_shapes_match_final_resolver(path: Path) -> None:
    graph = parse_metro_mermaid(path.read_text())
    topology = graph.route_topology
    assert topology is not None
    assert _predicted_shapes(topology) == _actual_shapes(graph)


def test_route_topology_is_hash_seed_deterministic() -> None:
    fixture = ROOT / "examples" / "topologies" / "fanout_bundle_plus_spurs.mmd"
    script = (
        "from pathlib import Path; "
        "from nf_metro.parser.mermaid import parse_metro_mermaid; "
        f"p=Path({str(fixture)!r}); "
        "print(repr(parse_metro_mermaid(p.read_text()).route_topology))"
    )
    outputs = []
    for seed in ("1", "7", "41"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout)
    assert len(set(outputs)) == 1
