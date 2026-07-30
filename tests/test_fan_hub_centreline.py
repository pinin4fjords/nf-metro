"""Tests for fork-hub / join-hub centreline agreement (#1595).

Under ``diamond_style: symmetric`` the join hub of a fan is explicitly
recentred on its (post-grid-snap) branch midpoint, but the fork hub is not -
it is grid-snapped like any other station. For an odd branch count the
midpoint lands on a grid line anyway, so the two agree by coincidence. For
an even branch count the midpoint sits at a half-pitch offset, and only the
join hub gets pulled back onto it: the fork hub is left wherever the grid
snap nearest it happened to land, which can be a full pitch away from the
join hub.

A port-fed variant of this fan surfaces an overlapping but distinct defect
(#1272: ``diamond_style: symmetric`` never applies to a fork that begins at
a section's entry port) and is out of scope here.

Rail-laid sections are the one place where the two hubs legitimately differ,
because a rail station's Y is the centre of the rail span it carries rather
than a marker centreline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid


def _bare_fan(n: int) -> str:
    stations = "\n".join(f"        b{i}[B{i}]" for i in range(n))
    forks = "\n".join(f"        hub -->|a| b{i}" for i in range(n))
    joins = "\n".join(f"        b{i} -->|a| j" for i in range(n))
    return f"""%%metro title: fan
%%metro diamond_style: symmetric
%%metro line: a | A | #24b064
graph LR
    subgraph s [S]
        hub[Hub]
{stations}
        j[Join]
{forks}
{joins}
    end
"""


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_fork_and_join_hub_share_centreline(n: int) -> None:
    graph = parse_metro_mermaid(_bare_fan(n))
    compute_layout(graph)
    hub = graph.stations["hub"]
    join = graph.stations["j"]
    branch_ys = sorted(graph.stations[f"b{i}"].y for i in range(n))
    mean = (branch_ys[0] + branch_ys[-1]) / 2.0
    assert join.y == pytest.approx(mean, abs=1.0), "join hub off the branch mean"
    assert hub.y == pytest.approx(mean, abs=1.0), "fork hub off the branch mean"
    assert hub.y == pytest.approx(join.y, abs=1.0), "fork/join hubs disagree"


_RAIL_FAN = (
    Path(__file__).parent.parent
    / "examples"
    / "topologies"
    / "rail_symmetric_fork_join_spans.mmd"
)


def test_rail_fork_and_join_centre_on_their_own_rail_spans() -> None:
    """A rail-laid station's Y is the centre of the rail span it carries, so a
    fork and join carrying different line sets have different centres.

    ``bqsr`` also carries ``core`` (the line arriving from the upstream
    section), spanning four rails; ``merge`` carries only the three caller
    lines. Both are drawn as pills capping their own span, and forcing them
    onto one shared Y would leave one pill off the rails it caps.
    """
    graph = parse_metro_mermaid(_RAIL_FAN.read_text())
    compute_layout(graph, validate=True)

    bqsr = graph.stations["bqsr"]
    merge = graph.stations["merge"]
    for station in (bqsr, merge):
        span = station.rail_used_ys
        assert len(span) > 1
        assert station.y == pytest.approx((min(span) + max(span)) / 2.0, abs=0.01)

    assert len(bqsr.rail_used_ys) == 4
    assert len(merge.rail_used_ys) == 3
    assert bqsr.y != pytest.approx(merge.y, abs=1.0)
