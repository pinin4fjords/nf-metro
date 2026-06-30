"""Regression: a non-consumer pass-through line in a manual ``direction: RL`` row.

A manual-grid serpentine lays its return row out right to left via
``direction: RL``.  A line leaves the row's rightmost section bundled with a
consumed line, but is not consumed by the intervening RL section between source
and target.  It must bypass *around* that section, looping below the row, rather
than running through the section's interior and peeling out mid-edge.

The defect was a grid "hole": the intervening section spans wider than its grid
cell, leaving the cell to its right empty.  ``column_gap_edges`` resolved the
bypass descent gap against that empty cell, defaulting the gap's left boundary
to the canvas origin, so the descent landed deep inside the spilled-over
section.  The fix falls the gap's left boundary back to the nearest occupied
column, placing the descent in the clear gap beside the box.

See issue #1211.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases._common import routes_through_unrelated_sections
from nf_metro.layout.routing.core import route_edges_centred
from nf_metro.layout.routing.offsets import compute_station_offsets
from nf_metro.parser import parse_metro_mermaid

TOPOLOGIES = Path(__file__).parent.parent / "examples" / "topologies"
FIXTURE = "manual_rl_row_nonconsumer_bypass"


def _layout():
    graph = parse_metro_mermaid((TOPOLOGIES / f"{FIXTURE}.mmd").read_text())
    compute_layout(graph)
    return graph


def test_no_line_routes_through_intervening_section() -> None:
    """No routed line crosses the interior of a section it never touches."""
    graph = _layout()
    offenders = routes_through_unrelated_sections(graph)
    assert not offenders, "; ".join(
        f"{rp.line_id} {rp.edge.source}->{rp.edge.target} through {sid!r}"
        for rp, sid in offenders
    )


def test_rna_bypass_descends_clear_of_realign() -> None:
    """The ``rna`` bypass drops in the clear gap right of the intervening box.

    ``rna`` leaves ``consensus`` (rightmost RL section) and must reach
    ``reporting`` (leftmost) without entering ``realign`` between them.  Its
    descent leg must sit at or beyond ``realign``'s right edge so the leftward
    traverse happens below the row, not through the box's realignment trunk.
    """
    graph = _layout()
    realign = graph.sections["realign"]
    realign_right = realign.bbox_x + realign.bbox_w
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)

    descents = [
        rp
        for rp in routes
        if rp.line_id == "rna"
        and rp.is_inter_section
        and rp.edge.target == "reporting__entry_right_11"
    ]
    assert descents, "expected an inter-section rna route into reporting"
    for rp in descents:
        for (x0, y0), (x1, y1) in zip(rp.points, rp.points[1:]):
            inside_y = realign.bbox_y < y0 < realign.bbox_y + realign.bbox_h
            if abs(x1 - x0) < 1e-6 and inside_y:  # vertical descent leg
                assert x0 >= realign_right - 1.0, (
                    f"rna descent at x={x0:.1f} is left of realign right edge "
                    f"{realign_right:.1f} (inside the box)"
                )
