"""Column mates that share a left runway share a bbox left edge.

A grid row's sections share a trunk Y, so levelling their bbox tops always lines
up something a viewer reads.  A grid column's sections share no trunk X: the
space left of a box's first station is that section's entry runway, and two
column mates only mean the same thing by it when their first stations stand at
one X.  ``_left_align_column_bboxes_only`` (Stage 3.6) levels exactly those,
growing the narrower boxes leftward so interior stations and the right edge stay
put.  Column mates whose content starts at different X keep their own edges --
levelling those would buy an aligned edge at the price of an empty band the width
of the difference.

Covers:

* Corpus: within every shared-runway run, each member either sits on the run's
  left edge or a left neighbour in its own row band explains the shortfall.
* Corpus: the runway spread within a grid column never exceeds what the stage's
  own inputs already had, so levelling can only equalise interior space.
* Meaningfulness: shipped fixtures hold runs the stage actually levelled, and
  column mates it declined because their content starts at different X, so
  neither the property nor the restriction is vacuous.
* Regression: ``convergence_fold_diamond``'s two mirror branches end on one left
  edge with equal runways, while the ``finish`` box below them -- whose content
  starts 90px further left relative to its box -- keeps its own edge.
* Regression: ``foldback_exit_peeloff``'s ``reporting`` box, whose content starts
  197.5px right of its column mate's, is left alone.
* The neighbour-corridor bound, on a hand-built grid: no shipped fixture stacks a
  wide left-column section beside a narrow column mate.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from nf_metro.layout.constants import MIN_INTER_SECTION_GAP, SAME_COORD_TOLERANCE
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases._common import (
    _column_contiguous_row_groups,
    _content_station_ids,
)
from nf_metro.layout.phases.bbox import (
    _column_left_neighbour_limit,
    _left_align_column_bboxes_only,
    _shared_left_runway_runs,
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


def _runway(graph: MetroGraph, section: Section) -> float | None:
    xs = [graph.stations[sid].x for sid in _content_station_ids(graph, section)]
    return min(xs) - section.bbox_x if xs else None


def _unexplained_short_edges(graph: MetroGraph) -> list[tuple[str, float]]:
    """``(section_id, shortfall)`` for every box right of its run's left edge
    that the neighbour-corridor bound does not account for."""
    out: list[tuple[str, float]] = []
    for group in _column_contiguous_row_groups(graph):
        for run in _shared_left_runway_runs(graph, group):
            target = min(s.bbox_x for s in run)
            for section in run:
                shortfall = section.bbox_x - target
                if shortfall <= SAME_COORD_TOLERANCE:
                    continue
                if section.bbox_x <= _column_left_neighbour_limit(graph, section) + (
                    SAME_COORD_TOLERANCE
                ):
                    continue
                out.append((section.id, shortfall))
    return out


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_shared_runway_runs_share_a_left_edge(path: Path) -> None:
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    unexplained = _unexplained_short_edges(graph)
    assert not unexplained, (
        f"{path.name}: sections sit right of the left edge of the run they share a "
        f"runway with, with no left neighbour to explain it: {unexplained}"
    )


def _column_runway_spreads(graph: MetroGraph) -> dict[int, float]:
    """Widest minus narrowest content runway, per grid column."""
    by_col: dict[int, list[float]] = defaultdict(list)
    for section in graph.sections.values():
        if section.bbox_w <= 0 or section.grid_col < 0:
            continue
        runway = _runway(graph, section)
        if runway is not None:
            by_col[section.grid_col].append(runway)
    return {
        col: max(runways) - min(runways)
        for col, runways in by_col.items()
        if len(runways) >= 2
    }


def test_levelling_never_spreads_a_column_runway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No grid column comes out of Stage 3.6 with its runways further apart.

    The stage exists to equalise the space in front of a column's content, so
    running it must not leave any column's runways more unequal than they were
    when it started -- the failure mode of levelling boxes whose content starts
    at different X.  Compares every corpus fixture against the same layout with
    the stage stubbed out.
    """
    worse: list[tuple[str, int, float, float]] = []
    for path in _gather_fixtures():
        text = path.read_text()
        try:
            with monkeypatch.context() as patched:
                patched.setattr(
                    "nf_metro.layout.engine._left_align_column_bboxes_only",
                    lambda graph: None,
                )
                without = parse_metro_mermaid(text)
                compute_layout(without)
            graph = parse_metro_mermaid(text)
            compute_layout(graph)
        except Exception:
            continue
        before = _column_runway_spreads(without)
        after = _column_runway_spreads(graph)
        for col, spread in after.items():
            if spread > before.get(col, spread) + SAME_COORD_TOLERANCE:
                worse.append((path.name, col, before[col], spread))
    assert not worse, f"columns left with wider-spread runways: {worse}"


def test_corpus_levels_shared_runways_and_leaves_the_rest() -> None:
    """The property and the shared-runway restriction are both exercised."""
    levelled = 0
    declined = 0
    for path in _gather_fixtures():
        graph = parse_metro_mermaid(path.read_text())
        try:
            compute_layout(graph)
        except Exception:
            continue
        for group in _column_contiguous_row_groups(graph):
            runs = _shared_left_runway_runs(graph, group)
            levelled += sum(
                1 for run in runs if len({round(s.bbox_x, 1) for s in run}) == 1
            )
            declined += len(group) - sum(len(run) for run in runs)
    assert levelled > 0, "no shipped fixture has a levelled grid column"
    assert declined > 0, "no shipped column mate is held out of the levelling"


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


def _seat_one_station(graph: MetroGraph, section_id: str, x: float) -> None:
    """Give ``section_id`` a single content station at ``x``, so the section can
    join a shared-runway run."""
    from nf_metro.parser.model import Station

    sid = f"{section_id}_station"
    section = graph.sections[section_id]
    graph.stations[sid] = Station(
        id=sid, label=sid, section_id=section_id, x=x, y=section.bbox_y + 50.0
    )
    section.station_ids.append(sid)


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
    for section_id in ("upper", "lower"):
        _seat_one_station(graph, section_id, 450.0)
    _left_align_column_bboxes_only(graph)
    lower = graph.sections["lower"]
    assert lower.bbox_x == 330.0 + MIN_INTER_SECTION_GAP
    assert lower.bbox_x + lower.bbox_w == 500.0


def test_column_mates_with_different_content_columns_keep_their_edges() -> None:
    """Content starting at different X splits the column into separate runs."""
    graph = _stacked_column_graph()
    _seat_one_station(graph, "upper", 450.0)
    _seat_one_station(graph, "lower", 500.0)
    upper, lower = graph.sections["upper"], graph.sections["lower"]
    assert _shared_left_runway_runs(graph, [upper, lower]) == []
    _left_align_column_bboxes_only(graph)
    assert lower.bbox_x == 380.0


def test_mirror_branches_share_a_left_edge_and_runway() -> None:
    """``convergence_fold_diamond``'s two branches level onto one edge."""
    path = REPO_ROOT / "examples" / "topologies" / "convergence_fold_diamond.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    left = graph.sections["branch_left"]
    right = graph.sections["branch_right"]
    assert abs(left.bbox_x - right.bbox_x) <= SAME_COORD_TOLERANCE
    assert abs(_runway(graph, left) - _runway(graph, right)) <= SAME_COORD_TOLERANCE
    finish = graph.sections["finish"]
    assert finish.bbox_x < left.bbox_x - SAME_COORD_TOLERANCE, (
        "the branches were levelled onto the box below them, whose content starts "
        "further left relative to its edge"
    )


def test_peeloff_reporting_box_keeps_its_own_edge() -> None:
    """A column mate whose content starts far right is not grown out to the edge."""
    path = REPO_ROOT / "examples" / "topologies" / "foldback_exit_peeloff.mmd"
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    preprocessing = graph.sections["preprocessing"]
    reporting = graph.sections["reporting"]
    assert reporting.bbox_x > preprocessing.bbox_x + SAME_COORD_TOLERANCE
    assert (
        abs(_runway(graph, reporting) - _runway(graph, preprocessing))
        <= SAME_COORD_TOLERANCE
    )
