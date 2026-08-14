"""Level seams into single-line ports draw level.

A straight inter-section connector has no vertical leg to absorb an endpoint
lane mismatch, so the deferred-offset renderer distributes any difference
linearly and the whole seam reads as an almost-horizontal slope.  A
single-line port has no bundle constraining its lane, so its seam offset can
always agree with its partner's; these seams are held level across the frozen
seed corpus, where the slopes were first observed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.routing.common import apply_route_offsets
from nf_metro.layout.routing.core import observe_route_edges
from nf_metro.layout.routing.offsets import compute_station_offsets

ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "tests" / "fixtures" / "hash_seed_determinism"


@pytest.mark.parametrize(
    "name", ("seed_15.mmd", "seed_41.mmd", "seed_72.mmd", "seed_77.mmd")
)
def test_single_line_port_level_seams_draw_level(name: str) -> None:
    path = FROZEN / name
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offsets = compute_station_offsets(graph)
        observation = observe_route_edges(
            graph,
            station_offsets=offsets,
            allow_convergence_clearance_requirements=True,
        )

    def single_line_port(station_id: str) -> bool:
        return station_id in graph.ports and len(graph.station_lines(station_id)) == 1

    sloped = []
    for route in observation.routes:
        if not route.is_inter_section or len(route.points) != 2:
            continue
        if not (
            single_line_port(route.edge.source) or single_line_port(route.edge.target)
        ):
            continue
        (xa, ya), (xb, yb) = route.points
        if abs(ya - yb) > COORD_TOLERANCE or abs(xa - xb) <= COORD_TOLERANCE:
            continue
        drawn = apply_route_offsets(route, offsets)
        if abs(drawn[0][1] - drawn[-1][1]) > COORD_TOLERANCE:
            sloped.append(
                f"{route.edge.source}->{route.edge.target} [{route.line_id}]: "
                f"{drawn[0]} -> {drawn[-1]}"
            )
    assert not sloped, "level seams drawn sloped:\n" + "\n".join(sloped)
