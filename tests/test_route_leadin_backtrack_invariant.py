"""A route's lead-in runs from its source towards its turn, never past it.

An inter-section route that opens with a straight run into a corner takes that
run as the corner's runway: it starts back from the turn on the source's side
and travels towards it.  A lead-in laid out on the far side of the turn instead
doubles back over the corner - the drawn stroke leaves its source, overshoots,
and reverses - and the emitter offsets its members against a segment travelling
the opposite way, so the bundle's lanes come out mirrored at the turn as well.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.routing.core import observe_route_edges
from nf_metro.layout.routing.offsets import compute_station_offsets

ROOT = Path(__file__).parents[1]
CORPUS = sorted(
    (
        *(ROOT / "examples").glob("*.mmd"),
        *(ROOT / "examples" / "topologies").glob("*.mmd"),
        *(ROOT / "tests" / "fixtures" / "hash_seed_determinism").glob("*.mmd"),
    ),
    key=lambda p: (p.parent.name, p.name),
)


def _observe(path: Path) -> tuple[Any, Any]:
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offsets = compute_station_offsets(graph)
        observation = observe_route_edges(
            graph,
            station_offsets=offsets,
            allow_convergence_clearance_requirements=True,
        )
    return graph, observation


def _backtracking_leadins(graph: Any, observation: Any) -> list[str]:
    found = []
    for route in observation.routes:
        if not route.is_inter_section or len(route.points) < 3:
            continue
        (x0, y0), (x1, y1), (x2, y2) = route.points[:3]
        source = graph.stations[route.edge.source]
        for lead, turn, here, cross_lead, cross_turn in (
            (x0, x1, source.x, y0, y1),
            (y0, y1, source.y, x0, x1),
        ):
            if abs(cross_lead - cross_turn) > COORD_TOLERANCE:
                continue
            if abs(turn - lead) <= COORD_TOLERANCE:
                continue
            straight_on = (x2 - x1, y2 - y1) == (x1 - x0, y1 - y0)
            if straight_on:
                continue
            if (turn - lead) * (turn - here) < -COORD_TOLERANCE:
                found.append(
                    f"{route.edge.source}->{route.edge.target} [{route.line_id}]: "
                    f"leads in from {lead} to a turn at {turn}, "
                    f"past its source at {here}"
                )
    return found


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_lead_in_runs_towards_its_turn(path: Path) -> None:
    graph, observation = _observe(path)
    backtracking = _backtracking_leadins(graph, observation)
    assert not backtracking, "lead-ins past their own turn:\n" + "\n".join(backtracking)
