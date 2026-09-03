"""A symmetric two-way entry fork stays compacted onto half-pitch.

Issue #1848 (alignment-section fork): under ``diamond_style: symmetric`` a
non-reconverging two-way entry fork (``umi_dedup -> {genomecov, salmon_quant}``
on the riboseq map) is compacted by ``_recenter_full_bundle_columns`` onto
half-pitch -- the two branches sit one grid unit apart straddling the trunk,
leaving no empty trunk row between them, and both are registered in
``graph.half_grid_station_ids``.

A layout pass that re-runs after a branch becomes the root of a downstream fan
plan (e.g. an off-track file lift triggering a second placement pass) must not
drop that compaction: excluding the plan-owned branch from its own half-pitch
column collapses the pair back to full pitch (two units apart, an empty trunk
row between them) while leaving the now-stale half-grid marks on the
full-pitch coordinates.  These tests lock the compaction and the invariant that
a half-grid-marked symmetric fork branch is never left on a full-pitch row.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nf_metro.api import render_string
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.phases.fan_bundles import (
    _section_row_pitch,
    _symmetric_entry_fork_pairs,
)
from nf_metro.parser.mermaid import parse_metro_mermaid

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIRS = [
    ROOT / "examples",
    ROOT / "examples" / "topologies",
    ROOT / "tests" / "fixtures",
    ROOT / "tests" / "fixtures" / "curve_invariant_repros",
]

RIBOSEQ = (
    ROOT
    / "tests"
    / "fixtures"
    / "curve_invariant_repros"
    / ("riboseq_inter_row_corridor.mmd")
)


def _load(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    return graph


def _symmetric_fork_fixtures():
    seen: set[Path] = set()
    for directory in FIXTURE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mmd")):
            if path in seen or "diamond_style: symmetric" not in path.read_text():
                continue
            seen.add(path)
            yield path


def test_riboseq_alignment_fork_is_half_pitch():
    graph = _load(RIBOSEQ)
    trunk = graph.stations["umi_dedup"].y
    top = graph.stations["genomecov"].y
    bottom = graph.stations["salmon_quant"].y
    pitch = _section_row_pitch(graph, "alignment", graph._resolved_y_spacing)

    # Declared order preserved: genomecov above the trunk, salmon below it.
    assert top < trunk < bottom
    # Half-pitch: each branch half a grid unit off the trunk, symmetric, with no
    # empty trunk row between them (full pitch would be a whole unit each side).
    assert bottom - top == pytest.approx(pitch, abs=2.0)
    assert (trunk - top) == pytest.approx(bottom - trunk, abs=2.0)
    for sid in ("genomecov", "salmon_quant"):
        assert sid in graph.half_grid_station_ids


@pytest.mark.parametrize("path", list(_symmetric_fork_fixtures()), ids=lambda p: p.stem)
def test_half_grid_symmetric_fork_never_on_full_pitch_row(path: Path):
    graph = _load(path)
    for section in graph.sections.values():
        for a, b in _symmetric_entry_fork_pairs(graph, section):
            if a not in graph.half_grid_station_ids:
                continue
            assert b in graph.half_grid_station_ids
            pitch = _section_row_pitch(graph, section.id, graph._resolved_y_spacing)
            separation = abs(graph.stations[a].y - graph.stations[b].y)
            # Both branches carry the half-grid mark, so they must sit one grid
            # unit apart (compacted half-pitch).  A stale mark left after a
            # collapse to full pitch would place them two units apart with an
            # empty trunk row between.
            assert separation == pytest.approx(pitch, abs=2.0), (
                f"{path.stem}/{section.id}: half-grid fork separation "
                f"{separation:.1f} != one grid unit {pitch:.1f} (full-pitch)"
            )


def _station_json_y(svg: str, station_id: str) -> float:
    match = re.search(rf'"id":"{station_id}"[^}}]*"y":([-\d.]+)', svg)
    assert match, f"station {station_id} missing from rendered geometry"
    return float(match.group(1))


def _label_y(svg: str, text: str) -> float:
    match = re.search(
        rf'<text x="[-\d.]+" y="([-\d.]+)"[^>]*class="[^"]*station-label[^"]*"[^>]*>'
        rf"{re.escape(text)}</text>",
        svg,
    )
    assert match, f"label {text!r} missing from rendered SVG"
    return float(match.group(1))


def test_riboseq_fork_labels_clear_the_trunk_bundle():
    """Salmon's label and the Coverage caption clear the trunk bypass bundle.

    The alignment fork straddles a trunk row carrying the riboseq+rnaseq lines
    that bypass both branches.  Salmon's station label (below its lower branch)
    and the Coverage caption (above its upper-branch file icon) both point out of
    the fork bubble, away from that trunk row -- not into it.
    """
    svg = render_string(RIBOSEQ.read_text(), chrome_css=False)
    trunk_y = _station_json_y(svg, "umi_dedup")
    salmon_y = _station_json_y(svg, "salmon_quant")
    bigwig_y = _station_json_y(svg, "bigwig_out")

    salmon_label_y = _label_y(svg, "Salmon")
    coverage_caption_y = _label_y(svg, "Coverage")

    # Salmon rides the lower branch: its label hangs below the station, on the
    # far side from the trunk it would otherwise strike.
    assert salmon_label_y > salmon_y > trunk_y
    # Coverage's icon rides the upper branch: its caption flips above the icon,
    # away from the trunk beneath.
    assert coverage_caption_y < bigwig_y < trunk_y
