"""Authored layout intent and effective engine decisions stay distinct."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, PortSide
from nf_metro.parser.provenance import (
    ConnectorEndpointKey,
    ConnectorEndpointRole,
    DecisionOrigin,
    DecisionReason,
    DecisionState,
    FoldThresholdSource,
)
from nf_metro.render.svg import _fold_threshold_error

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

AUTHORED_SOURCE = """\
%%metro title: Authored provenance
%%metro line: a | A | #ff0000
%%metro fold_threshold: 9
%%metro grid: source | 2,3,2,1
%%metro grid: target | 3,3
graph LR
    subgraph source [Source]
        %%metro direction: TB
        %%metro exit: bottom | a
        s1[S1]
    end
    subgraph target [Target]
        %%metro direction: LR
        %%metro entry: top | a
        t1[T1]
    end
    s1 -->|a| t1
"""


def _connector(graph, source: str, target: str, line_id: str):
    assert graph.route_topology is not None
    return next(
        connector
        for connector in graph.route_topology.connectors
        if connector.source == source
        and connector.target == target
        and connector.line_id == line_id
    )


def test_authored_snapshot_captures_every_layout_directive_before_inference() -> None:
    graph = parse_metro_mermaid(AUTHORED_SOURCE)
    provenance = graph.layout_provenance
    authored = provenance.authored

    assert authored is not None
    assert [(item.section_id, item.value) for item in authored.grids] == [
        ("source", (2, 3, 2, 1)),
        ("target", (3, 3, 1, 1)),
    ]
    assert [(item.section_id, item.value) for item in authored.directions] == [
        ("source", "TB"),
        ("target", "LR"),
    ]
    assert authored.fold_threshold.directive_value == 9
    assert authored.fold_threshold.caller_value is None
    assert authored.fold_threshold.selected_source is FoldThresholdSource.DIRECTIVE
    assert [
        (hint.section_id, hint.role, hint.side, hint.line_ids)
        for hint in authored.port_hints
    ] == [
        (
            "source",
            ConnectorEndpointRole.EXIT,
            PortSide.BOTTOM,
            ("a",),
        ),
        (
            "target",
            ConnectorEndpointRole.ENTRY,
            PortSide.TOP,
            ("a",),
        ),
    ]
    with pytest.raises(FrozenInstanceError):
        authored.fold_threshold.selected_value = 1  # type: ignore[misc]

    connector = _connector(graph, "s1", "t1", "a")
    exit_key = provenance.endpoint_key(connector.id, ConnectorEndpointRole.EXIT)
    entry_key = provenance.endpoint_key(connector.id, ConnectorEndpointRole.ENTRY)
    assert authored.endpoint_values(exit_key) == (PortSide.BOTTOM,)
    assert authored.endpoint_values(entry_key) == (PortSide.TOP,)

    for decision in (
        provenance.grid_decision("source"),
        provenance.direction_decision("source"),
        provenance.endpoint_decision(exit_key),
        provenance.endpoint_decision(entry_key),
        provenance.fold_threshold_decision,
    ):
        assert decision is not None
        assert decision.origin is DecisionOrigin.AUTHORED
        assert decision.is_author_owned
        assert decision.is_reinference_locked
        assert decision.state is DecisionState.AUTHORED


def test_caller_fold_threshold_is_preserved_separately_from_directive() -> None:
    graph = parse_metro_mermaid(AUTHORED_SOURCE, max_station_columns=12)
    authored = graph.layout_provenance.authored
    decision = graph.layout_provenance.fold_threshold_decision

    assert authored is not None
    assert authored.fold_threshold.directive_value == 9
    assert authored.fold_threshold.caller_value == 12
    assert authored.fold_threshold.selected_source is FoldThresholdSource.CALLER
    assert decision is not None
    assert decision.value == 12
    assert decision.origin is DecisionOrigin.AUTHORED
    assert decision.is_reinference_locked
    assert decision.state is DecisionState.AUTHORED
    assert decision.reason is DecisionReason.CALLER_FOLD_THRESHOLD


def test_default_fold_threshold_is_inferred_and_unlocked() -> None:
    source = AUTHORED_SOURCE.replace("%%metro fold_threshold: 9\n", "")
    graph = parse_metro_mermaid(source)
    decision = graph.layout_provenance.fold_threshold_decision

    assert decision is not None
    assert decision.value == 15
    assert decision.origin is DecisionOrigin.INFERRED
    assert not decision.is_reinference_locked
    assert decision.reason is DecisionReason.DEFAULT_FOLD_THRESHOLD


def test_partially_hinted_entry_tracks_each_semantic_connector() -> None:
    graph = parse_metro_mermaid(
        """\
%%metro line: a | A | #ff0000
%%metro line: b | B | #0000ff
%%metro grid: source | 0,0
%%metro grid: target | 1,0
graph LR
    subgraph source [Source]
        s1[S1]
    end
    subgraph target [Target]
        %%metro entry: top | a
        t1[T1]
    end
    s1 -->|a,b| t1
"""
    )
    provenance = graph.layout_provenance
    authored_connector = _connector(graph, "s1", "t1", "a")
    inferred_connector = _connector(graph, "s1", "t1", "b")
    authored_key = provenance.endpoint_key(
        authored_connector.id, ConnectorEndpointRole.ENTRY
    )
    inferred_key = provenance.endpoint_key(
        inferred_connector.id, ConnectorEndpointRole.ENTRY
    )

    authored = provenance.endpoint_decision(authored_key)
    inferred = provenance.endpoint_decision(inferred_key)
    assert authored is not None and inferred is not None
    assert authored.value is PortSide.TOP
    assert authored.is_author_owned
    assert inferred.value is PortSide.TOP
    assert not inferred.is_author_owned
    assert inferred.state is DecisionState.INFERRED
    assert inferred.reason is DecisionReason.SHARED_CONNECTOR_ENTRY_SIDE
    assert authored_key != inferred_key


def test_repeated_identical_hints_remain_author_owned() -> None:
    source = AUTHORED_SOURCE.replace(
        "        %%metro entry: top | a\n",
        "        %%metro entry: top | a\n        %%metro entry: top | a\n",
    )
    graph = parse_metro_mermaid(source)
    connector = _connector(graph, "s1", "t1", "a")
    endpoint = graph.layout_provenance.endpoint_key(
        connector.id, ConnectorEndpointRole.ENTRY
    )
    decision = graph.layout_provenance.endpoint_decision(endpoint)

    assert decision is not None
    assert decision.is_author_owned
    assert decision.authored_values == (PortSide.TOP, PortSide.TOP)


def test_tall_anchor_directions_are_inferred_then_pinned() -> None:
    graph = parse_metro_mermaid((EXAMPLES / "genomic_pipeline.mmd").read_text())

    for section_id in ("variant_calling", "post_vc", "annotation", "reporting"):
        decision = graph.layout_provenance.direction_decision(section_id)
        assert decision is not None
        assert decision.value == "LR"
        assert decision.origin is DecisionOrigin.INFERRED
        assert not decision.is_author_owned
        assert decision.is_reinference_locked
        assert decision.state is DecisionState.INFERRED_THEN_PINNED
        assert decision.reason is DecisionReason.TALL_ANCHOR_DIRECTION


def test_fold_relocation_keeps_authored_and_effective_port_sides_distinct() -> None:
    graph = parse_metro_mermaid(
        (EXAMPLES / "genomeassembly.mmd").read_text(), max_station_columns=5
    )
    provenance = graph.layout_provenance

    exit_connector = _connector(graph, "yahs", "asmstats", "assemblies")
    exit_key = provenance.endpoint_key(exit_connector.id, ConnectorEndpointRole.EXIT)
    exit_decision = provenance.endpoint_decision(exit_key)
    assert exit_decision is not None
    assert exit_decision.value is PortSide.LEFT
    assert exit_decision.authored_values == (PortSide.RIGHT,)
    assert exit_decision.origin is DecisionOrigin.INFERRED
    assert not exit_decision.is_author_owned
    assert exit_decision.is_reinference_locked
    assert exit_decision.reason is DecisionReason.FOLD_RELOCATED_SIDE

    entry_connector = _connector(graph, "yahs", "asmstats", "assemblies")
    entry_key = provenance.endpoint_key(entry_connector.id, ConnectorEndpointRole.ENTRY)
    entry_decision = provenance.endpoint_decision(entry_key)
    assert entry_decision is not None
    assert entry_decision.value is PortSide.RIGHT
    assert entry_decision.authored_values == (PortSide.LEFT,)
    assert entry_decision.reason is DecisionReason.FOLD_RELOCATED_SIDE


def test_fold_error_reads_the_typed_caller_threshold() -> None:
    graph = parse_metro_mermaid(
        (EXAMPLES / "genomeassembly.mmd").read_text(), max_station_columns=5
    )
    decision = graph.layout_provenance.fold_threshold_decision
    error = _fold_threshold_error(graph)

    assert decision is not None
    assert decision.value == 5
    assert decision.reason is DecisionReason.CALLER_FOLD_THRESHOLD
    assert error is not None
    assert "fold_threshold=5" in str(error)


def test_every_effective_corpus_commitment_has_valid_provenance() -> None:
    for path in sorted(EXAMPLES.rglob("*.mmd")):
        graph = parse_metro_mermaid(path.read_text())
        graph.layout_provenance.validate_complete(graph)
        if graph.route_topology is not None:
            assert list(graph.layout_provenance.connector_sides) == [
                ConnectorEndpointKey(connector.id, role)
                for connector in graph.route_topology.connectors
                for role in ConnectorEndpointRole
            ]


def test_legacy_provenance_fields_are_absent() -> None:
    names = {item.name for item in fields(MetroGraph)}
    assert (
        not {
            "_explicit_grid",
            "_explicit_directions",
            "_explicit_entry",
            "_explicit_exit",
            "_fold_reoriented_sections",
            "_fold_threshold_effective",
        }
        & names
    )
