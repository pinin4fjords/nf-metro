"""Tests for the fused co-travelling distinct-line invariant.

Two DIFFERENT lines running the same way along one corridor nest a full
``OFFSET_STEP`` apart, which is what leaves a hairline of background showing
between their strokes.  Closed to less than that they paint one two-tone
stripe and one of the two lines is not there to read.

The defect only appears on the settled re-route, because that is the pass the
reservation ledger reaches, so the reported fixtures are exercised through the
render chokepoint rather than a single ``route_edges`` call.

Covers:

* Happy-path: every shipped topology and example routes with no fused pair.
* Targeted: the three corridors a reservation band pulled together
  (``rl_return_row_convergence``, ``convergence_fold_diamond``,
  ``seed72_cross_family_fan``) keep the full step on the settled geometry, and
  land on it exactly rather than merely clear of the check.
* Meaningfulness: with the separation pass disabled the checker fires on those
  fixtures, so the invariant genuinely encodes the defect.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import nf_metro.layout.routing.core as routing_core
import nf_metro.layout.routing.invariants as invariants
from nf_metro.layout.constants import graph_offset_step
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges
from nf_metro.layout.routing.invariants import check_no_fused_cotravelling_lines
from nf_metro.parser.mermaid import parse_metro_mermaid

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPOLOGIES = REPO_ROOT / "tests" / "fixtures" / "topologies"
EXAMPLES = REPO_ROOT / "examples"
EXAMPLE_TOPOLOGIES = EXAMPLES / "topologies"
CURVE_REPROS = REPO_ROOT / "tests" / "fixtures" / "curve_invariant_repros"

REPORTED = [
    CURVE_REPROS / "rl_return_row_convergence.mmd",
    EXAMPLE_TOPOLOGIES / "convergence_fold_diamond.mmd",
    EXAMPLE_TOPOLOGIES / "seed72_cross_family_fan.mmd",
]


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(TOPOLOGIES.glob("*.mmd")))
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    return paths


def _route(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    return graph, routes, offsets


def _settled(path: Path, monkeypatch: pytest.MonkeyPatch):
    """The geometry the renderer draws, plus the violations the chokepoint saw.

    The check is replaced by a recording stand-in that reports nothing, so the
    render runs to completion on a fixture carrying the defect and the test can
    measure its final geometry rather than only catch the abort.
    """
    from nf_metro.api import prepare_graph, resolve_theme
    from nf_metro.render.svg import build_observed_render_plan

    final: list[tuple] = []

    def spy(graph, routes, offsets):
        found = check_no_fused_cotravelling_lines(graph, routes, offsets)
        final.clear()
        final.append((graph, routes, offsets, found))
        return []

    monkeypatch.setattr(invariants, "check_no_fused_cotravelling_lines", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(path.read_text(), source_dir=str(path.parent))
        build_observed_render_plan(graph, resolve_theme(None, graph))
    assert final, "the render chokepoint never ran the check"
    return final[0]


def _pair_separations(routes, offsets) -> dict[tuple[str, str, str], float]:
    """Lateral separation of every co-travelling distinct-line track pair."""
    from nf_metro.layout.routing.common import (
        apply_route_offsets,
        corridor_lanes,
        corridor_runs,
    )

    lanes = corridor_lanes(
        run
        for rp in routes
        if rp.is_inter_section
        for run in corridor_runs(rp, apply_route_offsets(rp, offsets))
    )
    out: dict[tuple[str, str, str], float] = {}
    for i, first in enumerate(lanes):
        for second in lanes[i + 1 :]:
            if first.axis != second.axis or first.sign != second.sign:
                continue
            if first.line_id == second.line_id:
                continue
            axis = "X" if first.axis == 0 else "Y"
            key = tuple(sorted((first.line_id, second.line_id))) + (axis,)
            separation = abs(first.coord - second.coord)
            if key not in out or separation < out[key]:
                out[key] = separation  # type: ignore[index]
    return out  # type: ignore[return-value]


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_no_fused_cotravelling_lines_in_gallery(path: Path) -> None:
    """No shipped topology or example paints two distinct lines as one stroke."""
    graph, routes, offsets = _route(path)
    violations = check_no_fused_cotravelling_lines(graph, routes, offsets)
    assert not violations, "\n".join(v.message() for v in violations)


@pytest.mark.parametrize("path", REPORTED, ids=lambda p: p.stem)
def test_reported_corridors_keep_the_nesting_step(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corridors a reservation band pulled together keep the full step."""
    _graph, _routes, _offsets, violations = _settled(path, monkeypatch)
    assert not violations, "\n".join(v.message() for v in violations)


@pytest.mark.parametrize("path", REPORTED, ids=lambda p: p.stem)
def test_checker_fires_without_the_separation_pass(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling the separation pass reproduces the fused pairs the check catches."""
    monkeypatch.setattr(
        routing_core, "_separate_fused_cotravelling_runs", lambda routes, ctx: None
    )
    graph, _routes, _offsets, violations = _settled(path, monkeypatch)
    assert violations, "expected a fused pair with the separation pass off"
    step = graph_offset_step(graph)
    assert all(v.separation < step for v in violations)


@pytest.mark.parametrize("path", REPORTED, ids=lambda p: p.stem)
def test_separated_pairs_land_on_the_step(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pair the pass moves ends exactly one step apart, not merely wider.

    The pass restores the pitch a bundle is drawn at; nudging the two lanes only
    far enough to satisfy the check would read as an accidental gap rather than a
    nested pair.
    """
    monkeypatch.setattr(
        routing_core, "_separate_fused_cotravelling_runs", lambda routes, ctx: None
    )
    graph, _routes, _offsets, violations = _settled(path, monkeypatch)
    fused = {
        tuple(sorted((v.first_line, v.second_line))) + (v.axis,) for v in violations
    }
    assert fused, "expected a fused pair with the separation pass off"
    monkeypatch.undo()
    graph, routes, offsets, _violations = _settled(path, monkeypatch)
    separations = _pair_separations(routes, offsets)
    step = graph_offset_step(graph)
    for pair in fused:
        assert pair in separations, f"{pair} no longer shares a corridor"
        assert separations[pair] == pytest.approx(step, abs=1e-6)
