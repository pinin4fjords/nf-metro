"""Sections sharing a grid column share a bbox left edge.

A grid column stacks its sections one above another, so their left edges read as
one vertical line -- the X counterpart of the row top-align that levels a grid
row's bbox tops.  Where the edges disagree, the space left of each section's
first station differs for no structural reason, and the column reads ragged.

``_left_align_column_bboxes_only`` (Stage 3.6) levels them by growing the
narrower boxes leftward, so interior stations and the right edge stay put.  Two
predicates hold a box short of the column line, and this module states both as
predicates rather than naming fixtures, so a new fixture is swept in
automatically:

* a left neighbour in the box's own row band, which the growth would close the
  inter-column routing corridor against;
* a LEFT port on a line channel the box shares with two or more other sections,
  whose fanned legs turn at one corner that this port's X helps seat.

Covers:

* Corpus: every column's contiguous row run shares a left edge, or one of the two
  predicates explains each member that falls short.
* Meaningfulness: the corpus holds columns the stage actually levelled and boxes
  the shared-channel carve-out held back, so neither is vacuous.  No shipped
  fixture stacks a wide left-column section under a narrow one, so the
  neighbour-corridor bound is exercised on a hand-built grid instead.
* Regression: ``convergence_fold_diamond``'s two mirror branches end on one left
  edge with equal entry runways.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.constants import MIN_INTER_SECTION_GAP, SAME_COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases._common import _column_contiguous_row_groups
from nf_metro.layout.phases.bbox import (
    _column_left_neighbour_limit,
    _left_align_column_bboxes_only,
    _left_port_shares_a_line_channel,
)
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, Section

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    for rel in (
        "tests/fixtures/topologies",
        "tests/fixtures",
        "examples",
        "examples/topologies",
    ):
        paths.extend(sorted((REPO_ROOT / rel).glob("*.mmd")))
    return paths


def _unexplained_short_edges(graph: MetroGraph) -> list[tuple[str, float]]:
    """``(section_id, shortfall)`` for every box right of its column line whose
    position neither carve-out accounts for."""
    out: list[tuple[str, float]] = []
    for group in _column_contiguous_row_groups(graph):
        target = min(s.bbox_x for s in group)
        for section in group:
            shortfall = section.bbox_x - target
            if shortfall <= SAME_COORD_TOLERANCE:
                continue
            if _left_port_shares_a_line_channel(graph, section):
                continue
            limit = _column_left_neighbour_limit(graph, section)
            if section.bbox_x <= limit + SAME_COORD_TOLERANCE:
                continue
            out.append((section.id, shortfall))
    return out


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_column_row_runs_share_a_left_edge(path: Path) -> None:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    unexplained = _unexplained_short_edges(graph)
    assert not unexplained, (
        f"{path.name}: sections sit right of their grid column's left edge with "
        f"neither a left neighbour nor a shared line channel to explain it: "
        f"{unexplained}"
    )


def _levelled_columns(graph: MetroGraph) -> list[list[Section]]:
    return [
        group
        for group in _column_contiguous_row_groups(graph)
        if len({round(s.bbox_x, 1) for s in group}) == 1
    ]


def test_corpus_levels_columns_and_takes_the_channel_carve_out() -> None:
    """The property and the shared-channel carve-out are both exercised."""
    levelled = 0
    channel_exempt = 0
    for path in _gather_fixtures():
        graph = parse_metro_mermaid(path.read_text())
        try:
            compute_layout(graph)
        except Exception:
            continue
        levelled += len(_levelled_columns(graph))
        for group in _column_contiguous_row_groups(graph):
            target = min(s.bbox_x for s in group)
            channel_exempt += sum(
                1
                for section in group
                if section.bbox_x - target > SAME_COORD_TOLERANCE
                and _left_port_shares_a_line_channel(graph, section)
            )
    assert levelled > 0, "no shipped fixture has a levelled grid column"
    assert channel_exempt > 0, "the shared-line-channel carve-out is never taken"


def _stacked_column_graph() -> MetroGraph:
    """A column whose lower cell is narrower than its upper, with a wide section
    in the previous column beside the lower cell only."""
    graph = MetroGraph()
    specs = [
        ("upper", 0, 1, 300.0, 0.0, 200.0),
        ("lower", 1, 1, 380.0, 200.0, 120.0),
        ("blocker", 1, 0, 0.0, 200.0, 330.0),
    ]
    for sid, row, col, x, y, w in specs:
        graph.sections[sid] = Section(
            id=sid,
            name=sid,
            grid_row=row,
            grid_col=col,
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=100.0,
        )
    return graph


def test_neighbour_limit_keeps_the_inter_column_corridor() -> None:
    """A left neighbour sharing the row band reserves the routing corridor."""
    graph = _stacked_column_graph()
    assert _column_left_neighbour_limit(graph, graph.sections["lower"]) == (
        330.0 + MIN_INTER_SECTION_GAP
    )
    assert _column_left_neighbour_limit(graph, graph.sections["upper"]) == float("-inf")


def test_alignment_stops_at_the_neighbour_corridor() -> None:
    """The levelling stops short of a left neighbour rather than growing over it."""
    graph = _stacked_column_graph()
    _left_align_column_bboxes_only(graph)
    lower = graph.sections["lower"]
    assert lower.bbox_x == 330.0 + MIN_INTER_SECTION_GAP
    assert lower.bbox_x + lower.bbox_w == 500.0


def test_mirror_branches_share_a_left_edge_and_runway() -> None:
    """``convergence_fold_diamond``'s two branches level onto one edge."""
    path = REPO_ROOT / "examples" / "topologies" / "convergence_fold_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    left = graph.sections["branch_left"]
    right = graph.sections["branch_right"]
    assert abs(left.bbox_x - right.bbox_x) <= SAME_COORD_TOLERANCE
    runways = []
    for section in (left, right):
        xs = [
            graph.stations[sid].x
            for sid in section.station_ids
            if sid in graph.stations and not graph.stations[sid].is_port
        ]
        runways.append(min(xs) - section.bbox_x)
    assert abs(runways[0] - runways[1]) <= SAME_COORD_TOLERANCE
