"""Section headers must never be drawn across a routed metro line (issue #774).

A line entering a section through an edge under its top-left header would cross
the title text.  The placement chain relocates the header (below, rotated onto a
side, or nudged past the trunk) instead of routing the line around the title.

Covers:

* Happy-path: every shipped example and topology fixture places every section
  header clear of every route.
* Meaningfulness: with header relocation disabled (the resolver pinned to its
  default above-left position) the new fixtures clash, proving the chain - not
  coincidence - is what keeps them clear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nf_metro.render.section_header as section_header
from nf_metro.api import resolve_theme
from nf_metro.layout.constants import SECTION_HEADER_PROTRUSION
from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import MetroGraph, Section
from nf_metro.render.constants import (
    SECTION_HEADER_ROUTE_PAD,
    SECTION_NUM_CIRCLE_R_LARGE,
)
from nf_metro.render.section_header import (
    check_section_headers_clear_routes,
    check_section_headers_fit_box_width,
    check_section_headers_hold_the_reserved_band,
    resolve_all_section_headers,
    resolve_section_header_placement,
)
from nf_metro.render.svg import apply_route_offsets

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
EXAMPLE_TOPOLOGIES = EXAMPLES / "topologies"
FIXTURE_TOPOLOGIES = REPO_ROOT / "tests" / "fixtures" / "topologies"

RELOCATION_FIXTURES = [
    EXAMPLE_TOPOLOGIES / "top_entry_header_clash.mmd",
    EXAMPLE_TOPOLOGIES / "header_side_rotated.mmd",
    EXAMPLE_TOPOLOGIES / "header_nudge.mmd",
]


def _gather_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    paths.extend(sorted(EXAMPLE_TOPOLOGIES.glob("*.mmd")))
    paths.extend(sorted(FIXTURE_TOPOLOGIES.glob("*.mmd")))
    return paths


def _polylines_and_font(path: Path):
    graph = parse_metro_mermaid(path.read_text())
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    polylines = [apply_route_offsets(route, offsets) for route in routes]
    theme = resolve_theme(None, graph)
    return graph, polylines, theme.section_label_font_size, theme.title_font_size


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_no_section_header_route_clashes_in_gallery(path: Path) -> None:
    """Every section header clears every route across the shipped corpus."""
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    clashes = check_section_headers_clear_routes(placements, polylines)
    assert not clashes, "\n".join(c.message() for c in clashes)


@pytest.mark.parametrize("path", RELOCATION_FIXTURES, ids=lambda p: p.stem)
def test_default_above_placement_would_clash(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning every header to its default above-left position reintroduces the
    clash on the relocation fixtures, so the chain is doing real work."""
    monkeypatch.setattr(section_header, "_placement_clear", lambda *a, **k: True)
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    clashes = check_section_headers_clear_routes(placements, polylines)
    assert clashes, "expected an above-left header to clash with the route"


def test_nudge_clears_a_route_to_the_right_of_the_box() -> None:
    """The nudge fallback must clear routes crossing to the right of the box.

    A header nudged right occupies ``[start, start + length]``; a route crossing
    its vertical band anywhere in that span must be stepped past, so the nudge
    consults the full width to its right rather than only the box-width extent.
    Leaving a route inside the nudged keepout hard-aborts a slightly crowded map
    on the render-time guard.

    ``above``/``below`` are blocked by a full-height trunk and the side columns
    do not fit the short box, so the resolver falls through to ``nudge``.  The
    trunk fixes the nudge origin; a second route sits to the right of the box,
    past the un-nudged header's right edge, inside the nudged header's span."""
    graph = MetroGraph()
    section = Section(id="s", name="Alignment")
    section.bbox_x, section.bbox_y = 0.0, 100.0
    section.bbox_w, section.bbox_h = 97.0, 18.0
    graph.sections["s"] = section

    trunk = [(90.0, 70.0), (90.0, 150.0)]
    right_route = [(150.0, 70.0), (150.0, 105.0)]
    polylines = [trunk, right_route]

    placement = resolve_section_header_placement(
        graph, section, label_font_size=13.0, polylines=polylines, title_font_size=13.0
    )
    assert placement.mode == "nudge"
    clashes = check_section_headers_clear_routes({"s": placement}, polylines)
    assert not clashes, "\n".join(c.message() for c in clashes)


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_section_header_fits_box_width_in_gallery(path: Path) -> None:
    """Every horizontal section header stays within its box width.

    A title wider than its box must wrap onto extra lines rather than
    overhang the box's right edge."""
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    overflowing = check_section_headers_fit_box_width(graph, placements)
    assert not overflowing, f"headers overhanging their box: {overflowing}"


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_section_header_never_crosses_box_border_in_gallery(path: Path) -> None:
    """A horizontal header's extra wrapped lines never draw across its own
    section box's border - they grow away from the box, not into it."""
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    crossings = []
    for section_id, placement in placements.items():
        section = graph.sections.get(section_id)
        if section is None or placement.label_rotation:
            continue
        if placement.mode in ("above", "nudge"):
            if placement.keepout[3] > section.bbox_y + 0.01:
                crossings.append(section_id)
        elif placement.mode == "below":
            box_bottom = section.bbox_y + section.bbox_h
            if placement.keepout[1] < box_bottom - 0.01:
                crossings.append(section_id)
    assert not crossings, f"headers crossing their box border: {crossings}"


def _rightmost_route_in_band(
    polylines: list, band_top: float, band_bottom: float
) -> float | None:
    """How far right a routed segment reaches inside a horizontal band.

    Clipped to the band rather than tested for a corner hit, so a diagonal
    crossing it contributes the x it actually occupies there.
    """
    reach: float | None = None
    for poly in polylines:
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            if max(y1, y2) < band_top or min(y1, y2) > band_bottom:
                continue
            if y1 == y2:
                inside = (x1, x2)
            else:
                inside = tuple(
                    x1 + (x2 - x1) * (edge - y1) / (y2 - y1)
                    for edge in (
                        max(band_top, min(y1, y2)),
                        min(band_bottom, max(y1, y2)),
                    )
                )
            reach = max(inside) if reach is None else max(reach, *inside)
    return reach


@pytest.mark.parametrize(
    "path", _gather_fixtures(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_relocated_headers_had_no_room_in_the_reserved_band(path: Path) -> None:
    """A header only leaves its reserved band once a route has taken the room.

    ``SECTION_HEADER_PROTRUSION`` above a box top is the only room reserved for
    that box's header, and it runs the width of the box.  A header drawn below or
    beside the box therefore has to be accounted for by a route standing in that
    band far enough right that no start within the box width clears it.  Both
    quantities come off the drawn map here - the header's own extent
    and the routed polylines - so this is a measurement of the geometry rather
    than a second run of the placement search the resolver already made.
    """
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    pad = SECTION_HEADER_ROUTE_PAD
    for section_id, placement in placements.items():
        if placement.mode in ("above", "nudge"):
            continue
        section = graph.sections[section_id]
        # A rotated header runs down a side column, so its drawn footprint says
        # nothing about the width the same title needs along the band.
        rotated = bool(placement.label_rotation)
        length = (
            section_header._header_length(section.name, font_size)
            if rotated
            else placement.keepout[2] - placement.keepout[0]
        )
        depth = (
            SECTION_HEADER_PROTRUSION
            if rotated
            else placement.keepout[3] - placement.keepout[1]
        )
        reach = _rightmost_route_in_band(
            polylines, section.bbox_y - depth - pad, section.bbox_y + pad
        )
        latest_start = section.bbox_x + section.bbox_w - length
        assert reach is not None and reach > latest_start - pad, (
            f"{section_id} left its reserved band with room in it: the band "
            f"holds a header starting up to {latest_start:.2f} and the "
            f"rightmost route in it reaches {reach}"
        )


def test_a_header_that_could_stay_in_the_band_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Denying the resolver its in-band shift strands headers the check reports,
    so the check is what holds the ranking rather than coincidence."""
    path = EXAMPLE_TOPOLOGIES / "cross_col_top_entry.mmd"
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)

    real = section_header._reserved_band_placements

    def _no_room(*args, **kwargs):
        above, nudged, _ = real(*args, **kwargs)
        return above, nudged, False

    monkeypatch.setattr(section_header, "_reserved_band_placements", _no_room)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    monkeypatch.undo()

    assert placements["consumer"].mode == "below"
    stranded = check_section_headers_hold_the_reserved_band(
        graph, placements, font_size, polylines, title_font_size
    )
    assert "consumer" in stranded


def test_in_band_shift_outranks_leaving_the_band() -> None:
    """A route crossing the header's default position shifts it along the band
    rather than dropping it below the box, while the shift fits the box width."""
    graph = MetroGraph()
    section = Section(id="s", name="Work")
    section.bbox_x, section.bbox_y = 0.0, 100.0
    section.bbox_w, section.bbox_h = 400.0, 80.0
    graph.sections["s"] = section

    # Descends through the default header's footprint into the box top.
    riser = [(10.0, 60.0), (10.0, 140.0)]
    placement = resolve_section_header_placement(
        graph, section, label_font_size=13.0, polylines=[riser], title_font_size=13.0
    )

    assert placement.mode == "nudge"
    assert placement.badge_cy - SECTION_NUM_CIRCLE_R_LARGE >= (
        section.bbox_y - SECTION_HEADER_PROTRUSION - 0.01
    )
    assert placement.keepout[3] <= section.bbox_y + 0.01
    assert placement.keepout[2] <= section.bbox_x + section.bbox_w + 0.5


def test_narrow_section_header_wraps_onto_multiple_lines() -> None:
    """A title wider than its box splits onto multiple lines."""
    path = EXAMPLE_TOPOLOGIES / "narrow_section_header_wrap.mmd"
    graph, polylines, font_size, title_font_size = _polylines_and_font(path)
    placements = resolve_all_section_headers(
        graph, font_size, polylines, title_font_size
    )
    placement = placements["wide_name"]
    assert len(placement.label_lines) > 1
    for line in placement.label_lines:
        assert section_header.estimate_section_label_width(line, font_size) <= (
            graph.sections["wide_name"].bbox_w
        )
