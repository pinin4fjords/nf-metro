"""A route heading the way its exit port faces stays off the section's far side.

An exit port states the side of its section a line leaves from.  When the
route's destination also lies on that side, everything the route needs is
already ahead of it, so any run beyond the section's opposite edge is a
detour that wraps the box it just left: the stroke leaves, doubles back
across the whole section, and returns.  Reaching around the far side is
reserved for a destination that actually sits there.
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
from nf_metro.parser.model import PortSide

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


def _source_exit_port(graph: Any, edge: Any) -> Any:
    """The exit port the route's source hangs off, through any junctions."""
    seen: set[str] = set()
    frontier = [edge.source]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        port = graph.ports.get(node)
        if port is not None and not port.is_entry:
            return port
        frontier.extend(upstream.source for upstream in graph.edges_to(node))
    return None


def _wraps_source_section(graph: Any, observation: Any) -> list[str]:
    found = []
    for route in observation.routes:
        if not route.is_inter_section:
            continue
        port = _source_exit_port(graph, route.edge)
        section = None if port is None else graph.sections.get(port.section_id)
        target = graph.stations.get(route.edge.target)
        if port is None or section is None or target is None:
            continue
        near, far, destination = {
            PortSide.LEFT: (
                section.bbox_x,
                section.bbox_x + section.bbox_w,
                target.x,
            ),
            PortSide.RIGHT: (
                section.bbox_x + section.bbox_w,
                section.bbox_x,
                target.x,
            ),
            PortSide.TOP: (
                section.bbox_y,
                section.bbox_y + section.bbox_h,
                target.y,
            ),
            PortSide.BOTTOM: (
                section.bbox_y + section.bbox_h,
                section.bbox_y,
                target.y,
            ),
        }[port.side]
        outward = 1.0 if near > far else -1.0
        if (destination - near) * outward < COORD_TOLERANCE:
            continue
        reach = min(
            (point[0] if port.side in (PortSide.LEFT, PortSide.RIGHT) else point[1])
            * outward
            for point in route.points
        )
        if reach < far * outward - COORD_TOLERANCE:
            found.append(
                f"{route.edge.source}->{route.edge.target} [{route.line_id}]: "
                f"leaves '{section.id}' on its {port.side.value} side towards "
                f"{destination} yet reaches {reach * outward} past the "
                f"opposite edge at {far}"
            )
    return found


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_route_does_not_wrap_the_section_it_leaves(path: Path) -> None:
    graph, observation = _observe(path)
    wrapping = _wraps_source_section(graph, observation)
    assert not wrapping, "routes wrapping their own source section:\n" + "\n".join(
        wrapping
    )
