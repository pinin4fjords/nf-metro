"""A TB off-track output never lands on a same-row fork-branch station (#1390).

When a fork's producer emits an ``off_track`` output offset along the section's
cross axis, the grid snap can pull that output's flow coordinate onto the exact
row of a sibling fork branch.  The branch then shares the output icon's row, and
if it also shares its cross column the two stations occupy one coordinate.

A same-row, same-column station is a direct overlap and must count as an
obstacle for the collision-avoidance bump even though it isn't downstream.
"""

from __future__ import annotations

from pathlib import Path

from nf_metro.layout import engine
from nf_metro.parser.mermaid import parse_metro_mermaid_file

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "regressions"
    / "tb_offtrack_fork_branch_collision_1390.mmd"
)

OFF_TRACK = "mapped_out"
BRANCH = "markduplicates"


def test_tb_offtrack_output_clears_fork_branch_station():
    """The off-track output and the sibling fork branch hold distinct cells."""
    graph = parse_metro_mermaid_file(FIXTURE)
    engine.compute_layout(graph, validate=True)

    off = graph.stations[OFF_TRACK]
    branch = graph.stations[BRANCH]
    separation = abs(off.x - branch.x) + abs(off.y - branch.y)
    assert separation > 1.0, (
        f"off-track output {OFF_TRACK!r} at ({off.x},{off.y}) collides with fork "
        f"branch {BRANCH!r} at ({branch.x},{branch.y})"
    )
