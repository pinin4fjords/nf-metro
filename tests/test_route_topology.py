"""The pre-resolution route topology is immutable and matches resolver rewrites."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

import nf_metro.parser.mermaid as mermaid
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge, MetroGraph, Section, Station
from nf_metro.parser.route_topology import (
    DivergenceGroup,
    RouteTopology,
    build_route_topology,
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


def test_connector_identity_preserves_authored_multiline_and_duplicate_edges() -> None:
    graph = _parse(_two_section_text("    a -->|red,blue| b\n    a -->|red| b\n"))
    topology = graph.route_topology
    assert topology is not None

    assert [connector.id for connector in topology.connectors] == [
        "edge:0:line:0",
        "edge:0:line:1",
        "edge:1:line:0",
    ]
    assert [connector.authored_edge_ordinal for connector in topology.connectors] == [
        0,
        0,
        1,
    ]
    assert [connector.line_ordinal for connector in topology.connectors] == [0, 1, 0]
    assert {
        (connector.source, connector.target) for connector in topology.connectors
    } == {("a", "b")}
    assert len(topology.bundles) == 1
    assert topology.bundles[0].connector_ids == (
        "edge:0:line:0",
        "edge:0:line:1",
        "edge:1:line:0",
    )
    assert topology.bundles[0].line_ids == ("blue", "red")


def test_line_networks_are_connected_components() -> None:
    graph = _parse(
        """\
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
    )
    topology = graph.route_topology
    assert topology is not None

    networks = [
        network for network in topology.line_networks if network.line_id == "red"
    ]
    assert [network.station_ids for network in networks] == [("a", "b"), ("c", "d")]
    assert [network.edge_ids for network in networks] == [
        ("edge:0:line:0",),
        ("edge:1:line:0",),
    ]


def test_programmatic_duplicate_edges_get_unique_fallback_identities() -> None:
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

    topology = build_route_topology(graph)

    assert [connector.id for connector in topology.connectors] == [
        "edge:0:line:0",
        "edge:1:line:0",
    ]
    assert all(connector.source_line is None for connector in topology.connectors)


def test_topology_keeps_authored_terminus_endpoint() -> None:
    graph = _parse(
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
"""
    )
    topology = graph.route_topology
    assert topology is not None

    connector = next(c for c in topology.connectors if c.source == "a")
    assert connector.target == "output"
    assert not connector.target.startswith("__converge_")
    assert any(station_id.startswith("__converge_") for station_id in graph.stations)


def test_divergence_and_convergence_membership_matches_resolver_shape() -> None:
    graph = _parse((ROOT / "examples" / "guide" / "03b_fan_in_merge.mmd").read_text())
    topology = graph.route_topology
    assert topology is not None

    assert topology.divergences == (
        DivergenceGroup(
            id="divergence:0",
            exit_group_id="exit:source:right",
            entry_group_ids=(
                "entry:sink:left",
                "entry:step_a:left",
                "entry:step_b:left",
            ),
            connector_ids=(
                "edge:4:line:0",
                "edge:5:line:0",
                "edge:6:line:0",
                "edge:7:line:0",
            ),
        ),
        DivergenceGroup(
            id="divergence:1",
            exit_group_id="exit:step_a:right",
            entry_group_ids=("entry:sink:left", "entry:step_b:left"),
            connector_ids=("edge:8:line:0", "edge:9:line:0"),
        ),
    )
    assert [
        (
            group.entry_group_id,
            group.line_id,
            group.divergence_ids,
            group.connector_ids,
        )
        for group in topology.convergences
    ] == [
        (
            "entry:sink:left",
            "main",
            ("divergence:0", "divergence:1"),
            ("edge:6:line:0", "edge:9:line:0"),
        ),
        (
            "entry:step_b:left",
            "main",
            ("divergence:0", "divergence:1"),
            ("edge:5:line:0", "edge:8:line:0"),
        ),
    ]


def test_topology_records_are_deeply_immutable() -> None:
    topology = _parse(_two_section_text("    a -->|red| b\n")).route_topology
    assert topology is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.connectors = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.connectors[0].source = "other"  # type: ignore[misc]
    assert isinstance(topology.connectors, tuple)
    assert isinstance(topology.connectors[0].network_id, str)
    assert isinstance(topology.bundles[0].connector_ids, tuple)

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


def _actual_groups(
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
        while station_id.startswith("__bypass_"):
            incoming = [
                edge for edge in graph.edges_to(station_id) if edge.line_id == line_id
            ]
            assert len(incoming) == 1
            station_id = incoming[0].source
        return station_id

    def authored_target(station_id: str, line_id: str) -> str:
        while station_id.startswith("__converge_"):
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
            and edge.target in graph.junctions
        ):
            junction_exit[edge.target] = port_key[edge.source]
        if (
            edge.source in graph.junctions
            and target_port is not None
            and target_port.is_entry
        ):
            junction_targets[edge.source].add((port_key[edge.target], edge.line_id))

    merge_ids = {jid for jid in graph.junctions if jid.startswith("__merge_")}
    fan_ids = {jid for jid in graph.junctions if jid.startswith("__junction_")}
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
        (junction_exit[jid], tuple(sorted(junction_targets[jid])))
        for jid in fan_ids
        if jid in junction_exit
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


def _predicted_groups(
    topology: RouteTopology,
) -> tuple[set[tuple], set[tuple], set[tuple], set[tuple]]:
    connectors = {connector.id: connector for connector in topology.connectors}
    exit_groups = {
        (
            (group.section_id, group.side.value),
            tuple(
                sorted(
                    {
                        (connectors[cid].source, connectors[cid].line_id)
                        for cid in group.connector_ids
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
                        (connectors[cid].target, connectors[cid].line_id)
                        for cid in group.connector_ids
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
                                entry_by_id[connectors[cid].entry_group_id].section_id,
                                entry_by_id[connectors[cid].entry_group_id].side.value,
                            ),
                            connectors[cid].line_id,
                        )
                        for cid in group.connector_ids
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
                        exit_by_id[divergence_by_id[did].exit_group_id].section_id,
                        exit_by_id[divergence_by_id[did].exit_group_id].side.value,
                    )
                    for did in group.divergence_ids
                )
            ),
        )
        for group in topology.convergences
    }
    return exit_groups, entry_groups, fan_groups, merge_groups


@pytest.mark.parametrize(
    "path", CORPUS, ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_corpus_topology_exactly_predicts_resolver_groups(path: Path) -> None:
    graph = parse_metro_mermaid(path.read_text())
    topology = graph.route_topology
    assert topology is not None
    assert _predicted_groups(topology) == _actual_groups(graph)


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


@pytest.mark.parametrize(
    "path", CORPUS, ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_topology_construction_does_not_change_resolved_graph(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = path.read_text()
    observed = parse_metro_mermaid(text)

    monkeypatch.setattr(mermaid, "build_route_topology", lambda graph: None)
    baseline = parse_metro_mermaid(text)

    observed.route_topology = None
    assert observed == baseline
