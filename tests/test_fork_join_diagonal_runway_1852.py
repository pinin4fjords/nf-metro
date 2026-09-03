"""Regression locks for the fork/join diagonal runway.

A fork/join leg whose drop exceeds ``DIAGONAL_RUN`` fills its runway so the
diagonal flattens toward 45 degrees rather than reading near-vertical; its
fan-mates seat at one divergence; the widened diagonal clears every interior
sibling's name label; and the flat-equalising centring pass reaches an
asymmetric one-sided hub while leaving a symmetric straddle on its trunk.
These tests encode those invariants against committed fixtures.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import pytest

from nf_metro.layout.constants import DIAGONAL_RUN
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.guards import iter_line_label_strikes
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.postprocess import (
    _build_bubble_ctx,
    _fork_join_centerable,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

_ROOT = Path(__file__).resolve().parents[1]


def _layout(fixture: str):
    graph = parse_metro_mermaid((_ROOT / fixture).read_text())
    compute_layout(graph)
    return graph


def _diagonal_segment(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The (run, drop) of a route's slanted segment, or None if it has none."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        run, drop = abs(x1 - x0), abs(y1 - y0)
        if run > 2 and drop > 2:
            return run, drop
    return None


# Fork/join legs whose drop exceeds DIAGONAL_RUN: each fills its runway so the
# diagonal run reaches the drop and the leg lands at 45 degrees.
_FORTY_FIVE_LEGS = [
    ("examples/variant_calling_tuned.mmd", "fastq_in", "fastqc"),
    ("examples/sarek_metro.mmd", "samtools_vc", "finalise"),
]


@pytest.mark.parametrize("fixture,source,target", _FORTY_FIVE_LEGS)
def test_fork_join_diagonal_fills_runway_to_45(fixture, source, target):
    """A fork/join diagonal whose drop exceeds the old run flattens toward 45."""
    graph = _layout(fixture)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    leg = next(
        (r for r in routes if r.edge.source == source and r.edge.target == target),
        None,
    )
    assert leg is not None, f"{fixture}: no {source}->{target} route"
    seg = _diagonal_segment(leg.points)
    assert seg is not None, f"{fixture}: {source}->{target} has no diagonal segment"
    run, drop = seg
    assert drop > DIAGONAL_RUN + 2, (
        f"{fixture}: {source}->{target} drop {drop:.1f} does not exceed the old "
        f"fixed run {DIAGONAL_RUN}; pick a steeper leg to lock the widening"
    )
    angle = math.degrees(math.atan2(drop, run))
    assert angle <= 47.0, (
        f"{fixture}: {source}->{target} routes at {angle:.1f} degrees "
        f"(run {run:.1f} vs drop {drop:.1f}); the runway did not fill toward 45"
    )


_CORPUS = sorted(glob.glob(str(_ROOT / "examples" / "**" / "*.mmd"), recursive=True))


def test_widened_fork_join_diagonals_strike_no_labels():
    """The widened diagonals and their fan-mates clear every station label.

    Flattening a fork/join leg lengthens its diagonal across the fan's columns,
    where an interior sibling's name label sits; the widening caps each leg to
    clear those labels.  This locks that no example map gains a label strike.
    """
    offenders: list[str] = []
    for path in _CORPUS:
        graph = parse_metro_mermaid(Path(path).read_text())
        compute_layout(graph)
        strikes = list(iter_line_label_strikes(graph))
        if strikes:
            rel = Path(path).relative_to(_ROOT)
            offenders.append(
                f"{rel}: " + ", ".join(f"{s.line_id}->{s.station_id}" for s in strikes)
            )
    assert not offenders, "label strikes in the example corpus:\n  " + "\n  ".join(
        offenders
    )


_SYMMETRIC_DIAMOND = """\
%%metro line: x | X | #ff0000
graph LR
    subgraph s [S]
        a[A]
        b[B]
        c[C]
        d[D]
        a -->|x| b
        a -->|x| c
        b -->|x| d
        c -->|x| d
    end
"""


def _centerable_map(graph) -> dict[str, bool]:
    routes = route_edges(graph, station_offsets=None)
    ctx = _build_bubble_ctx(routes, graph)
    return {
        sid: _fork_join_centerable(graph, ctx, sid, st)
        for sid, st in graph.stations.items()
        if not st.is_port
    }


def test_symmetric_straddle_hub_is_not_centerable():
    """A fork/join hub whose branches straddle it stays on the trunk.

    A symmetric diamond is centred on its branches' midpoint by placement; the
    flat-equalising pass must leave it there rather than tilt the divergence.
    """
    graph = parse_metro_mermaid(_SYMMETRIC_DIAMOND)
    compute_layout(graph)
    centerable = _centerable_map(graph)
    assert centerable["a"] is False, "symmetric fork hub must not be centerable"
    assert centerable["d"] is False, "symmetric join hub must not be centerable"


@pytest.mark.parametrize(
    "fixture,hub",
    [
        ("examples/genomic_pipeline.mmd", "merge_index"),
        ("examples/differentialabundance_default.mmd", "annotate"),
    ],
)
def test_asymmetric_one_sided_hub_is_centerable(fixture, hub):
    """An asymmetric one-sided fork/join hub is eligible for centring.

    Its branches all peel one way at unequal drops, so it sits off-centre in its
    own run with dead flat on one side and has no branch midpoint to protect --
    the one fork/join shape the flat-equalising pass should reach.
    """
    graph = _layout(fixture)
    centerable = _centerable_map(graph)
    assert centerable.get(hub) is True, (
        f"{fixture}: {hub} is an asymmetric one-sided hub but was not marked centerable"
    )
