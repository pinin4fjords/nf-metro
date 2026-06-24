"""Direction-agnostic section lane arranger.

A section arranges its internal lanes from its *boundary configuration*: the
order in which lines cross its edges.  When a line crosses the determining edge
at slot ``k`` and rides lane ``k``, the lines run parallel and never cross by
construction.

This module owns the reduction at the heart of that idea -- mapping a boundary
edge's crossing order to a lane order -- and nothing else.  The reduction is
axis-free: a line's position *along* an edge (an X coordinate on a TOP/BOTTOM
edge, a Y coordinate on a LEFT/RIGHT edge) is resolved by the caller before it
reaches here, so the same reduction serves any flow direction.

Callers supply the determining edge's crossing order: fan-out divergence
supplies its exit edge's peel order, and reconvergence supplies its entry
edge's primary-feeder order.  The edge derivation -- which lines cross, in what
order -- stays with each caller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryConfig:
    """A section's lane-determining boundary configuration.

    :param present: every line on the section, in the section's default
        (priority) lane order.
    :param determining: the lines that cross the determining edge, in the order
        they cross it along that edge.  Lines absent from *present* are ignored;
        lines in *present* but absent here are unconstrained.
    """

    present: tuple[str, ...]
    determining: tuple[str, ...]


def lane_order(
    config: BoundaryConfig, line_priority: dict[str, int]
) -> tuple[str, ...] | None:
    """The section's lane order, or ``None`` when it already matches priority.

    Lane *k* carries the *k*-th line crossing the determining edge, so a line
    crossing at edge-slot *k* rides lane *k*; the lines the edge does not
    constrain fall to the back of the bundle in priority order.  ``None`` means
    the resulting order is the plain priority order, so no re-slot is needed.
    """
    present = set(config.present)
    determining = tuple(lid for lid in config.determining if lid in present)
    rest = tuple(
        sorted(present - set(determining), key=lambda lid: line_priority.get(lid, 0))
    )
    order = determining + rest
    priority_order = tuple(
        sorted(config.present, key=lambda lid: line_priority.get(lid, 0))
    )
    if order == priority_order:
        return None
    return order
