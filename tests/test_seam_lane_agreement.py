"""Level seams into single-line ports draw level.

A straight inter-section connector has no vertical leg to absorb an endpoint
lane mismatch, so the deferred-offset renderer distributes any difference
linearly and the whole seam reads as an almost-horizontal slope.  A
single-line port has no bundle constraining its lane, so its seam offset can
always agree with its partner's; these seams are held level across the frozen
seed corpus, where the slopes were first observed.

The agreement only counts when it is not bought at the station row's expense:
a port either takes its whole flat in-section run to the seam partner's lane
or stays where its run-mates ride, never parting from them.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

import nf_metro.layout.routing.offsets as routing_offsets
from nf_metro.api import prepare_graph
from nf_metro.layout.constants import COORD_TOLERANCE
from nf_metro.layout.routing.common import apply_route_offsets
from nf_metro.layout.routing.core import observe_route_edges
from nf_metro.layout.routing.offsets import compute_station_offsets

ROOT = Path(__file__).parents[1]
FROZEN = ROOT / "tests" / "fixtures" / "hash_seed_determinism"
CORPUS = sorted(
    (
        *(ROOT / "examples").glob("*.mmd"),
        *(ROOT / "examples" / "topologies").glob("*.mmd"),
        *FROZEN.glob("*.mmd"),
    ),
    key=lambda p: (p.parent.name, p.name),
)


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


def _station_row_disagreements(ctx: Any) -> list[str]:
    """Single-line ports whose lane differs from a flat in-section run-mate."""
    graph = ctx.graph
    found = []
    for edge in graph.edges:
        src, tgt = graph.stations[edge.source], graph.stations[edge.target]
        if not src.section_id or src.section_id != tgt.section_id:
            continue
        if abs(src.y - tgt.y) > COORD_TOLERANCE:
            continue
        lid = edge.line_id
        for port_id, mate_id in (
            (edge.source, edge.target),
            (edge.target, edge.source),
        ):
            if port_id not in graph.ports or len(graph.station_lines(port_id)) != 1:
                continue
            own = ctx.offsets.get((port_id, lid), 0.0)
            mate = ctx.offsets.get((mate_id, lid), 0.0)
            if abs(own - mate) > COORD_TOLERANCE:
                found.append(
                    f"{port_id} [{lid}] on lane {own}, run-mate {mate_id} on {mate}"
                )
    return found


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_seam_snap_leaves_ports_on_their_station_row(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Levelling a seam must not slope the port's own station row instead.

    A single-line port may move to meet its seam partner's lane only with
    its whole flat in-section run behind it.  A port that moves alone
    relocates the mismatch onto the much shorter station row, where it
    reads as a steep localized diagonal rather than a seam the routed
    vertical legs or the mid-seam ramp can absorb.
    """
    original = routing_offsets._reconcile_horizontal_offsets
    disagreements: list[str] = []

    def observed(ctx: Any, *args: Any, **kwargs: Any) -> None:
        original(ctx, *args, **kwargs)
        disagreements.extend(_station_row_disagreements(ctx))

    monkeypatch.setattr(routing_offsets, "_reconcile_horizontal_offsets", observed)

    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        routing_offsets.compute_station_offsets(graph)

    assert not disagreements, "seam snap sloped a station row:\n" + "\n".join(
        dict.fromkeys(disagreements)
    )
