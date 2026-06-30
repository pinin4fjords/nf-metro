"""A convergence sink placed above one of its feeder sections must slot its
entry bundle by feeder approach, not by line-declaration order (#1204).

When a section is placed above its convergence-sink target in the grid but its
line is declared last, the base offsets give it the bottom-most lane at the
shared entry port.  Its exit then sits above that lane, so the line runs down
through the inter-column gap to reach it -- a run that crosses its bundle-mates
and, in compact mode, lands in a gap with no declared channel and aborts the
render.

The crossing-free lane order at a LEFT-entry convergence is by feeder approach:
the feeder whose source sits highest takes the topmost lane.  These tests pin
that order (deterministic, independent of the exact section heights) and that
the shipped fixture renders without the gap-channel abort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.layout.routing.invariants import assert_render_curve_invariants
from nf_metro.parser.mermaid import parse_metro_mermaid

TOPOLOGIES_DIR = Path(__file__).parent.parent / "examples" / "topologies"
FIXTURE = "convergence_sink_above"


def _vary_fastq_tools(text: str, n: int) -> str:
    """Drop the fastq QC tools after the first *n* to vary the row-0 height.

    The bundle order is independent of section height, so the invariant must
    hold whatever the fastq column's depth -- not only at the one height that
    happens to trip the render abort.
    """
    tools = [
        "bbmap",
        "fastp",
        "fastqc",
        "fastqe",
        "fastqscreen",
        "fq_lint",
        "kraken2",
        "krona",
        "seqfu",
    ]
    kept = "\n".join(
        ln for ln in text.splitlines() if not any(f"{t}" in ln for t in tools[n:])
    )
    return kept


def _convergence_lane_order(text: str) -> list[tuple[str, float, float]]:
    """Return ``[(line_id, lane_offset, source_y)]`` at the sink's LEFT port.

    One entry per line converging into ``multiqc`` from a distinct source,
    keyed for the topmost source row of each line.
    """
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    port = next(s for s in graph.stations if s.startswith("multiqc__entry"))
    source_y: dict[str, float] = {}
    for edge in graph.edges_to(port):
        y = graph.stations[edge.source].y
        source_y[edge.line_id] = min(source_y.get(edge.line_id, y), y)
    return [(lid, offsets.get((port, lid), 0.0), source_y[lid]) for lid in source_y]


@pytest.mark.parametrize("n_tools", [9, 6, 4])
def test_convergence_lane_order_follows_source_position(n_tools: int) -> None:
    """Lanes at the sink's entry port are ordered by feeder source Y: the
    feeder whose source sits highest takes the smallest (topmost) offset."""
    text = _vary_fastq_tools((TOPOLOGIES_DIR / f"{FIXTURE}.mmd").read_text(), n_tools)
    lanes = _convergence_lane_order(text)
    assert len(lanes) >= 2, "expected a multi-feeder convergence"
    by_offset = sorted(lanes, key=lambda r: r[1])
    source_ys = [src_y for _lid, _off, src_y in by_offset]
    assert source_ys == sorted(source_ys), (
        "lane order is not monotonic in feeder source Y: "
        f"{[(lid, round(off, 1), round(sy, 1)) for lid, off, sy in by_offset]}"
    )


def test_convergence_sink_above_renders() -> None:
    """The shipped fixture lays out without the gap-channel curve abort."""
    graph = parse_metro_mermaid((TOPOLOGIES_DIR / f"{FIXTURE}.mmd").read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    assert_render_curve_invariants(graph, routes, offsets)
