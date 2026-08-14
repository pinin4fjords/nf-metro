"""A line keeps one lane along a flat in-section run.

Per-station offset phases may reorder or compact a station's slots, but a
line drawn across a flat edge (same base Y in an LR/RL section, same base X
in a TB/BT one) must hold one offset across it: a changed lane there draws a
near-flat slope the routing cannot absorb into a formed curve.  The frozen
hash-seed fixtures exercise the allocator's full phase stack, including the
section-order reversal and the post-settlement port frames.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph
from nf_metro.layout.routing.offsets import compute_station_offsets

ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "tests" / "fixtures" / "hash_seed_determinism"


def _graph_and_offsets(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        return graph, compute_station_offsets(graph)


def _flat_run_slopes(graph, offsets) -> list[str]:
    slopes = []
    for edge in graph.edges:
        source = graph.stations[edge.source]
        target = graph.stations[edge.target]
        if not source.section_id or source.section_id != target.section_id:
            continue
        section = graph.sections[source.section_id]
        flat = (
            abs(source.x - target.x) < 0.5
            if section.direction in ("TB", "BT")
            else abs(source.y - target.y) < 0.5
        )
        source_off = offsets.get((edge.source, edge.line_id), 0.0)
        target_off = offsets.get((edge.target, edge.line_id), 0.0)
        if flat and abs(source_off - target_off) > 0.01:
            slopes.append(
                f"{edge.source}->{edge.target} {edge.line_id} "
                f"{source_off} -> {target_off}"
            )
    return slopes


@pytest.mark.parametrize(
    "name", ["seed_15.mmd", "seed_41.mmd", "seed_72.mmd", "seed_77.mmd"]
)
def test_flat_in_section_runs_keep_their_lane(name: str) -> None:
    graph, offsets = _graph_and_offsets(FROZEN / name)
    assert not _flat_run_slopes(graph, offsets)


def test_tb_station_slots_are_contiguous_for_present_lines() -> None:
    # seed_41's s5 is a TB section whose first station carries a terminating
    # line: the gap compaction covers TB frames, so the station's occupied
    # slots run consecutively instead of leaving reserved holes that widen
    # the marker.
    graph, offsets = _graph_and_offsets(FROZEN / "seed_41.mmd")
    station_offsets = sorted(
        off for (sid, _line), off in offsets.items() if sid == "n5_0"
    )
    assert station_offsets == [0.0, 4.0, 8.0]


def test_port_seam_alignment_respects_disagreeing_run_mates() -> None:
    # rail_boundary_bundle_fan's hub rides lane 4 with its run-mate haplo:
    # a settled port frame on the other side must not pull hub alone onto
    # another lane, trading the port seam for a station seam.
    hits = list(ROOT.rglob("rail_boundary_bundle_fan.mmd"))
    assert hits
    graph, offsets = _graph_and_offsets(hits[0])
    assert offsets.get(("hub", "wes"), 0.0) == offsets.get(("haplo", "wes"), 0.0)
