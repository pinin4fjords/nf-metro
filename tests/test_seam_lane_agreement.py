"""A bundle keeps one lane order across a section seam.

A straight inter-section connector has no vertical leg to absorb an endpoint
lane mismatch, so the deferred-offset renderer distributes any difference
linearly and the whole seam reads as an almost-horizontal slope; over a 10-px
exit or junction stub the same mismatch reads as a jog.  An approach that ends
off the lane its line rides inside the section breaks the drawn line at the
port instead, the bundle jumping lanes over zero width.

A single-line port has no bundle constraining its lane, so its seam offset can
always agree with its partner's - but the agreement only counts when it is not
bought at the station row's expense: such a port either takes its whole flat
in-section run to the seam partner's lane or stays where its run-mates ride,
never parting from them.
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
from nf_metro.parser.model import PortSide

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

_STUB_SLOPE_XFAILS = {
    "seed_41": "s4__exit_left_4 hands its pair to __junction_29 transposed",
}
_APPROACH_LANE_XFAILS = {
    "seed_15": "s8__exit_left_8 and __junction_26 land off s10__entry_right_19",
    "seed_77": "the two U-detour feeders land on s9__entry_right_25 transposed",
}


def _corpus_params(xfails: dict[str, str]) -> list[Any]:
    return [
        pytest.param(
            path,
            id=path.stem,
            marks=(
                [pytest.mark.xfail(strict=True, reason=xfails[path.stem])]
                if path.stem in xfails
                else []
            ),
        )
        for path in CORPUS
    ]


def _observe(path: Path) -> tuple[Any, dict[tuple[str, str], float], Any]:
    """Route *path* far enough to read every edge's drawn geometry."""
    graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        offsets = compute_station_offsets(graph)
        observation = observe_route_edges(
            graph,
            station_offsets=offsets,
            allow_convergence_clearance_requirements=True,
        )
    return graph, offsets, observation


@pytest.mark.parametrize("path", _corpus_params(_STUB_SLOPE_XFAILS))
def test_level_seam_stubs_draw_level(path: Path) -> None:
    """A seam whose endpoints share a base Y draws on one lane end to end.

    The exit-port stub, the junction stub and the connector between them are
    each one straight run with no vertical leg to absorb a lane mismatch, so
    endpoints that disagree read as a jog on a 10-px stub.
    """
    _graph, offsets, observation = _observe(path)
    sloped = []
    for route in observation.routes:
        if not route.is_inter_section or len(route.points) != 2:
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


@pytest.mark.parametrize("path", _corpus_params(_APPROACH_LANE_XFAILS))
def test_seam_approach_lands_on_the_port_lane(path: Path) -> None:
    """An approach ends on the lane its line rides inside the section.

    An approach that lands anywhere else leaves the drawn line broken at the
    port: the bundle re-orders across the seam and each mismatched line jumps
    lanes over zero width.
    """
    graph, offsets, observation = _observe(path)
    misplaced = []
    for route in observation.routes:
        if not route.is_inter_section:
            continue
        port = graph.ports.get(route.edge.target)
        if port is None or not port.is_entry:
            continue
        if port.side not in (PortSide.LEFT, PortSide.RIGHT):
            continue
        station = graph.stations[route.edge.target]
        if abs(station.y - graph.stations[route.edge.source].y) > COORD_TOLERANCE:
            continue
        lane_y = station.y + offsets.get((route.edge.target, route.line_id), 0.0)
        drawn = apply_route_offsets(route, offsets)
        if abs(drawn[-1][1] - lane_y) > COORD_TOLERANCE:
            misplaced.append(
                f"{route.edge.source}->{route.edge.target} [{route.line_id}]: "
                f"ends {drawn[-1][1]}, port lane {lane_y}"
            )
    assert not misplaced, "approaches off the port's lane:\n" + "\n".join(misplaced)


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


def test_a_line_leaving_a_station_frees_the_lane_an_arriving_line_needs() -> None:
    """Lines meeting end to end at a station share one lane.

    ``s14`` is a chain of single strokes: l3 arrives at ``n14_0`` from the
    entry port and l2 leaves it for ``n14_1``.  The two draw on opposite sides
    of the marker, so l2 does not hold l3 off the lane ``__junction_42``
    delivers it on, and the seam runs level instead of ramping into the port.
    """
    graph, offsets, observation = _observe(FROZEN / "seed_77.mmd")
    assert offsets[("n14_0", "l3")] == offsets[("n14_0", "l2")]
    seam = next(
        route
        for route in observation.routes
        if (route.edge.source, route.edge.target, route.line_id)
        == ("__junction_42", "s14__entry_left_32", "l3")
    )
    drawn = apply_route_offsets(seam, offsets)
    assert abs(drawn[0][1] - drawn[-1][1]) <= COORD_TOLERANCE


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
