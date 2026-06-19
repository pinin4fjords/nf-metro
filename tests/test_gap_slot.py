"""The symbolic gap-slot data structure (#845).

``GapSlot`` lets a handler declare *where* a vertical channel run sits in a gap
bundle -- which line of which bundle, in which inter-column corridor, travelling
which way -- without committing to a concrete X.  No handler emits one and no
pass consumes one: the field is a foundation the vertical-channel migration
builds on, so every route the corpus produces carries ``gap_slot is None``.
"""

from __future__ import annotations

import pytest
from conftest import content_corpus

from nf_metro.convert import convert_nextflow_dag
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import Direction, GapSlot, RoutedPath
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import Edge

CORPUS = content_corpus()


def test_gap_slot_carries_documented_fields():
    slot = GapSlot(
        gap_lo_col=2,
        gap_hi_col=3,
        row=1,
        direction=Direction.D,
        slot_index=1,
        n_slots=3,
    )
    assert (slot.gap_lo_col, slot.gap_hi_col) == (2, 3)
    assert slot.row == 1
    assert slot.direction is Direction.D
    assert (slot.slot_index, slot.n_slots) == (1, 3)


def test_routed_path_gap_slot_defaults_to_none():
    rp = RoutedPath(
        edge=Edge(source="a", target="b", line_id="l1"),
        line_id="l1",
        points=[(0.0, 0.0), (10.0, 0.0)],
    )
    assert rp.gap_slot is None


@pytest.mark.parametrize("fixture", CORPUS, ids=[fid for fid, _, _ in CORPUS])
def test_no_handler_emits_a_gap_slot_yet(fixture):
    fid, path, is_nextflow = fixture
    text = path.read_text()
    if is_nextflow:
        text = convert_nextflow_dag(text)
    graph = parse_metro_mermaid(text)
    compute_layout(graph, validate=False)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)

    with_slot = [r for r in routes if r.gap_slot is not None]
    assert not with_slot, (
        f"{fid}: {len(with_slot)} route(s) carry a gap_slot; this PR is purely "
        f"additive -- no handler should populate it yet"
    )
