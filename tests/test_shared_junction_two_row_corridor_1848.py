"""A shared exit junction that fans into two downstream rows keeps its lanes.

When ``center_ports`` pulls the entry ports of two stacked downstream sections
below the incoming trunk, every line leaving a shared exit junction has to turn
down into one inter-column gap.  The exit-turn plan nests them into one ladder,
four apart, with the bundle bound for each target on its own lanes.

Settlement re-routes that ladder against a frozen reservation whose clearance
band -- an intersection measured over every row the corridor crosses -- can be
tighter than the ladder the plan seated.  Seating each target's bundle to that
band independently slides one onto the other's lanes, fusing two lines bound for
different sections onto one descent column.  The turn column a member's exit-turn
plan pins is left where the plan placed it, so the ladder survives the re-route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.api import prepare_graph, resolve_theme
from nf_metro.render.svg import build_observed_render_plan

_FIXTURES = (
    "examples/rnaseq_sections.mmd",
    "examples/rnaseq_sections_manual.mmd",
    "examples/topologies/variant_calling.mmd",
)

# Two lanes closer than this along the descent axis read as one drawn stroke.
_MIN_LANE_GAP = 3.0
# Vertical runs overlapping by less than this do not co-travel.
_MIN_SPAN_OVERLAP = 5.0


def _root() -> Path:
    return Path(__file__).parent.parent


def _descent_legs(observed):
    """Longest vertical segment (x, y_lo, y_hi) per routed inter-section line."""
    legs: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for record in observed.plan.routes:
        values = dict(record.values.entries)
        edge = dict(values["edge"].values.entries)
        points = [tuple(pt) for pt in values["points"]]
        best: tuple[float, float, float] | None = None
        best_len = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if abs(x1 - x0) < 0.5 and abs(y1 - y0) > best_len:
                best_len = abs(y1 - y0)
                best = (x0, min(y0, y1), max(y0, y1))
        if best is not None and best_len > _MIN_SPAN_OVERLAP:
            legs[(edge["source"], edge["target"], values["line_id"])] = best
    return legs


@pytest.mark.parametrize("fixture", _FIXTURES)
def test_shared_junction_fan_keeps_distinct_descent_lanes(fixture):
    path = _root() / fixture
    text = "%%metro center_ports: true\n" + path.read_text()
    graph = prepare_graph(text, source_dir=str(path.parent))

    # The bug aborts here on the unfixed tree; reaching the assertions at all is
    # already the regression the fix repairs.
    observed = build_observed_render_plan(graph, resolve_theme(None, graph))

    legs = _descent_legs(observed)
    by_source: dict[str, list[tuple[str, str, tuple[float, float, float]]]] = {}
    for (source, target, line_id), leg in legs.items():
        if "junction" in source:
            by_source.setdefault(source, []).append((target, line_id, leg))

    fused: list[str] = []
    for source, members in by_source.items():
        for i in range(len(members)):
            t_i, line_i, (x_i, lo_i, hi_i) = members[i]
            for j in range(i + 1, len(members)):
                t_j, line_j, (x_j, lo_j, hi_j) = members[j]
                if line_i == line_j:
                    continue
                overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
                if overlap > _MIN_SPAN_OVERLAP and abs(x_i - x_j) < _MIN_LANE_GAP:
                    fused.append(
                        f"{source}: {line_i}->{t_i} and {line_j}->{t_j} "
                        f"share descent x={x_i:.1f}/{x_j:.1f}"
                    )
    assert not fused, "fused co-travelling descents:\n" + "\n".join(fused)
