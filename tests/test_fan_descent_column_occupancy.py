"""A re-seated fan descent bundle keeps off columns other lines travel.

Ranking a fan's opening descents onto adjacent tracks reads only the group being
ranked, so a target column can already carry a line outside it.  Two distinct
lines sharing one column over a common span draw as a single stroke with one of
them hidden.
"""

from __future__ import annotations

from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.common import iter_vertical_segments
from nf_metro.layout.routing.invariants import check_collinear_distinct_lines
from nf_metro.parser.mermaid import parse_metro_mermaid

SOURCE = """%%metro title: fan descent column occupancy
%%metro line: a | A | #8453d7
%%metro line: b | B | #6ef362
%%metro line: d | D | #dde6c4
%%metro line: e | E | #156075
%%metro grid: src | 0,2
%%metro grid: t0 | 1,2
%%metro grid: t1 | 0,1
%%metro grid: t2 | 1,1
%%metro grid: t3 | 2,0

graph LR
    subgraph src [Source]
        s_in[Source input]
        s_hub[Source hub]
        s_in -->|a| s_hub
    end
    subgraph t0 [t0]
        %%metro entry: right | b
        t0_a[t0 a]
        t0_b[t0 b]
        t0_a -->|b| t0_b
    end
    subgraph t1 [t1]
        %%metro entry: right | d
        t1_a[t1 a]
        t1_b[t1 b]
        t1_a -->|d| t1_b
    end
    subgraph t2 [t2]
        t2_a[t2 a]
        t2_b[t2 b]
        t2_a -->|d| t2_b
    end
    subgraph t3 [t3]
        t3_a[t3 a]
        t3_b[t3 b]
        t3_a -->|e| t3_b
    end

    s_hub -->|b| t0_a
    s_hub -->|d| t1_a
    s_hub -->|d| t2_a
    s_hub -->|e| t3_a
"""


def test_reseated_fan_descents_stay_off_other_lines_columns() -> None:
    """No two distinct lines share a vertical column over a common span."""
    graph = parse_metro_mermaid(SOURCE)
    compute_layout(graph, validate=True)

    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    assert check_collinear_distinct_lines(graph, routes, offsets) == []

    columns = [
        (route.line_id, x, y_lo, y_hi)
        for route in routes
        if route.is_inter_section
        for _idx, x, y_lo, y_hi, _down in iter_vertical_segments(route)
    ]
    overlaps = [
        (first, second)
        for index, first in enumerate(columns)
        for second in columns[index + 1 :]
        if first[0] != second[0]
        and abs(first[1] - second[1]) <= COORD_TOLERANCE
        and min(first[3], second[3]) - max(first[2], second[2]) > COORD_TOLERANCE
    ]
    assert overlaps == []
