"""Lane frame of a divergence junction that carries a fused turn.

A line leaving a fan-out junction on several branches turns them at one shared
vertex, so the render draws a single fused stroke whose radius is the widest of
the legs.  Pinned that way, the lane cannot follow its bundle-mates onto a
neighbouring slot: the distinct mate it would land beside reads as one
wholesale-translated corner with it, and the concentric reference sized for the
pair loses to the fusion.  Such a junction therefore keeps the frame its own
phases settled rather than re-inheriting a late recompaction at its feeder.
"""

from __future__ import annotations

from collections.abc import Callable

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets
from nf_metro.layout.routing import offsets as offsets_module
from nf_metro.parser.mermaid import parse_metro_mermaid

# ``rep`` leaves the hub for two rows while ``d0`` reaches a third, so the exit
# port's own bundle has an interior hole that the late recompaction closes.
FUSED_TURN_FAN = """%%metro title: fan
%%metro line: d0 | D0 | #6ef362
%%metro line: hub | HUB | #6ef362
%%metro line: rep | REP | #6ef362
%%metro grid: src | 0,4
%%metro grid: t0 | 1,3
%%metro grid: t1 | 1,6
%%metro grid: t2 | 2,7

graph LR
    subgraph src [Source]
        s_in[In]
        s_hub[Hub]
        s_in -->|hub| s_hub
    end
    subgraph t0 [t0]
        t0_a[t0 a]
        t0_b[t0 b]
        t0_a -->|rep| t0_b
    end
    subgraph t1 [t1]
        t1_a[t1 a]
        t1_b[t1 b]
        t1_a -->|rep| t1_b
    end
    subgraph t2 [t2]
        %%metro entry: right | d0
        t2_a[t2 a]
        t2_b[t2 b]
        t2_a -->|d0| t2_b
    end
    s_hub -->|rep| t0_a
    s_hub -->|rep| t1_a
    s_hub -->|d0| t2_a
"""

JUNCTION = "__junction_4"


def _junction_lanes(
    neutered: str | None = None,
) -> dict[tuple[str, str], float]:
    """Every settled lane the fan's junction holds, one pass optionally inert."""
    original: Callable[..., None] | None = None
    if neutered is not None:
        original = getattr(offsets_module, neutered)
        setattr(offsets_module, neutered, lambda *args, **kwargs: None)
    try:
        graph = parse_metro_mermaid(FUSED_TURN_FAN)
        compute_layout(graph)
        return {
            key: value
            for key, value in compute_station_offsets(graph).items()
            if key[0] == JUNCTION
        }
    finally:
        if neutered is not None and original is not None:
            setattr(offsets_module, neutered, original)


def test_fused_turn_junction_keeps_the_frame_its_own_phases_settled() -> None:
    """Recompacting the feeding exit port leaves the junction's lanes alone."""
    assert _junction_lanes() == _junction_lanes(
        neutered="_recompact_fan_port_bordering_stations"
    )


def test_fused_turn_fan_settles_within_the_closing_guards() -> None:
    """The same fan renders without a pinched bundle corner at the junction."""
    graph = parse_metro_mermaid(FUSED_TURN_FAN)

    compute_layout(graph, validate=True)
