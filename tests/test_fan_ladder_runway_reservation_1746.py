"""A settled turn ladder reserves the runway each of its members declares.

A fan's opening turn column is seated by the ladder origin, measured out from
the members' launch coordinates.  Members already sharing one axis carry no
mutual displacement, so the concentric radius that displacement implies is only
the base radius -- while the member's plan may declare a wider minimum runway
for the corner it has to form.  Seating the ladder on the narrower of the two
leaves the emitted turn inside the frame its plan declared.
"""

from __future__ import annotations

import pytest

from nf_metro.layout.constants import CURVE_RADIUS, OFFSET_STEP
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.parser.mermaid import parse_metro_mermaid

SOURCE = """%%metro title: fan
%%metro line: d0 | D0 | #6ef362
%%metro line: hub | HUB | #6ef362
%%metro line: rep | REP | #6ef362
%%metro grid: src | 0,4
%%metro grid: t0 | 1,5
%%metro grid: t1 | 1,7
%%metro grid: t2 | 1,4

graph LR
    subgraph src [Source]
        s_in[In]
        s_hub[Hub]
        s_in -->|hub| s_hub
    end
    subgraph t0 [t0]
        %%metro entry: right | d0
        t0_a[t0 a]
        t0_b[t0 b]
        t0_a -->|d0| t0_b
    end
    subgraph t1 [t1]
        t1_a[t1 a]
        t1_b[t1 b]
        t1_a -->|rep| t1_b
    end
    subgraph t2 [t2]
        %%metro entry: right | rep
        t2_a[t2 a]
        t2_b[t2 b]
        t2_a -->|rep| t2_b
    end
    s_hub -->|d0| t0_a
    s_hub -->|rep| t1_a
    s_hub -->|rep| t2_a
"""


@pytest.fixture(name="graph")
def _graph():
    graph = parse_metro_mermaid(SOURCE)
    compute_layout(graph, validate=False)
    return graph


def test_fan_emission_holds_its_planned_frame() -> None:
    """The planned fan's emitted geometry stays inside its declared frame."""
    compute_layout(parse_metro_mermaid(SOURCE), validate=True)


def test_reused_line_wrap_branch_turns_a_full_lane_past_its_radius(graph) -> None:
    """The wrap branch's opening turn clears the lane its bundle-mate holds.

    ``rep`` reaches ``t2`` by wrapping to a RIGHT entry, so its opening corner
    is one lane outside the base radius: the ladder must stand it off
    ``CURVE_RADIUS + OFFSET_STEP`` from its launch, not the base radius the
    zero mutual displacement of a single-member ladder would imply.
    """
    routes = route_edges(graph, station_offsets=compute_station_offsets(graph))
    wrap = next(
        route
        for route in routes
        if route.edge.source.startswith("__junction_")
        and route.line_id == "rep"
        and route.edge.target == "t2__entry_right_3"
    )
    runway = abs(wrap.points[1][0] - wrap.points[0][0])
    assert runway >= CURVE_RADIUS + OFFSET_STEP
