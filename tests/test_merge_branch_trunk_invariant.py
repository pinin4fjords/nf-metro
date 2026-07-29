"""Tests for the merge-branch-meets-trunk invariant.

At a reconvergence merge (several feeders converging on one entry port), the
non-trunk feeders ("branches") descend to the trunk's bypass channel and turn
into it.  The branch drop level is published by the routing context
(``trunk_by``); the trunk route computes its channel Y independently.  When the
two disagree -- notably when the trunk forces ``cross_row`` to route below
every section but the context did not -- the branches land at a different Y
from where the trunk actually runs and end as stubs hanging in open space.

A feeder that lands a few pixels off the trunk clears that hang report and
draws a stub anyway, so :func:`check_merge_feeders_land_on_trunk` asserts the
exact property at ``COORD_TOLERANCE``: the terminus lies ON the trunk, at or
before the trunk's own corner.

Covers:

* Happy-path: every shipped example and topology fixture routes with every
  merge feeder connected to its trunk, and terminating on it exactly.
* Targeted: ``genomeassembly_organellar`` (the reported defect) routes its
  ``assemblies`` line as connected strokes reaching ``asmstats``.
* Targeted: ``merge_feeders_three_columns`` converges three ``report`` feeders
  from three different columns onto one trunk, the shape that lands a feeder on
  a parallel lane and carries another past the trunk's corner.
* Meaningfulness: with ``_land_merge_feeders_on_trunk`` out of the way, each
  feeder terminates where its handler aimed it, and each invariant fires -- the
  hang check once the context's ``cross_row`` decision is desynced too, the
  exact check on the three-column fixture alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nf_metro.layout.routing.context as routing_context
import nf_metro.layout.routing.core as routing_core
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.invariants import (
    check_merge_branches_meet_trunk,
    check_merge_feeders_land_on_trunk,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    paths.extend(sorted((EXAMPLES / "topologies").glob("*.mmd")))
    paths.extend(sorted((EXAMPLES / "guide").glob("*.mmd")))
    paths.extend(sorted(FIXTURES.glob("*.mmd")))
    return paths


def _route(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    return graph, routes, offsets


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_merge_branches_meet_trunk_in_gallery(path: Path) -> None:
    """Every shipped fixture routes with no merge feeder hanging in open space."""
    graph, routes, offsets = _route(path)
    violations = check_merge_branches_meet_trunk(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_merge_feeders_land_on_trunk_in_gallery(path: Path) -> None:
    """Every shipped fixture terminates each merge feeder exactly on its trunk."""
    graph, routes, offsets = _route(path)
    violations = check_merge_feeders_land_on_trunk(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


def test_three_column_feeders_terminate_on_the_trunk() -> None:
    """Three ``report`` feeders converging from three columns each end on the
    trunk: none lands on a parallel lane beside it, and none is carried past the
    trunk's own corner into open space."""
    path = EXAMPLES / "topologies" / "merge_feeders_three_columns.mmd"
    graph, routes, offsets = _route(path)
    violations = check_merge_feeders_land_on_trunk(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


def test_genomeassembly_organellar_assemblies_connected() -> None:
    """The reported fixture's converging ``assemblies`` feeders all join the
    trunk rather than ending as stubs short of it."""
    graph, routes, offsets = _route(FIXTURES / "genomeassembly_organellar.mmd")
    violations = check_merge_branches_meet_trunk(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


def _without_feeder_landing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route without the pass that lands each feeder on its trunk.

    Leaves every feeder terminating wherever its handler aimed it -- at the
    channel level the context published, one tail length past its own descent
    column -- which is what both invariants exist to reject.
    """
    monkeypatch.setattr(
        routing_core, "_land_merge_feeders_on_trunk", lambda routes, ctx: None
    )


def test_checker_fires_when_context_disagrees_with_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Desyncing the context's channel decision from the trunk route's
    reproduces the hanging stubs the invariant is meant to catch: the branches
    drop to the context's level while the trunk runs elsewhere.  Patching the
    context's reference (the trunk route keeps its own) forces the disagreement,
    and the landing pass has to be out of the way or it repairs it.  Proves the
    check is not vacuous."""
    _without_feeder_landing(monkeypatch)
    monkeypatch.setattr(
        routing_context,
        "merge_trunk_force_cross_row",
        lambda *args, **kwargs: True,
    )
    graph, routes, offsets = _route(FIXTURES / "genomeassembly_organellar.mmd")
    violations = check_merge_branches_meet_trunk(graph, routes, offsets)
    assert violations, "expected hanging branches when context disagrees with trunk"


def test_landing_pass_is_what_puts_three_column_feeders_on_the_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the landing pass the three-column fixture reproduces both fault
    shapes, so the exact invariant is not vacuous either: the middle feeder ends
    an offset step below the trunk's centreline, and the near one is carried a
    tail length past the column the trunk turns down on."""
    _without_feeder_landing(monkeypatch)
    path = EXAMPLES / "topologies" / "merge_feeders_three_columns.mmd"
    graph, routes, offsets = _route(path)
    violations = check_merge_feeders_land_on_trunk(graph, routes, offsets)
    assert {v.source: round(v.gap, 1) for v in violations} == {
        "__junction_8": 2.0,
        "__junction_9": 16.1,
    }, [v.message() for v in violations]
