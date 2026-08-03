"""Ratchet on layout functions that encode only one coordinate axis.

A function that reads at least two X-family geometry fields and no Y-family
fields, or vice versa, can be a missing rotation counterpart without naming a
flow direction. The bound keeps that otherwise invisible class countable and
non-increasing. Explicitly axis-scoped primitives are exempt only with a stable
site key and a reason.
"""

from __future__ import annotations

from pathlib import Path

from single_axis_ratchet import single_axis_sites, single_axis_sites_from_source

_LAYOUT_DIR = Path(__file__).resolve().parents[1] / "src" / "nf_metro" / "layout"

_BASELINE = {"x": 27, "y": 48}

_EXEMPTIONS = {
    "labels.py::_clamp_label_to_section": (
        "horizontal label clamping has a separately named vertical primitive"
    ),
    "labels.py::_clamp_label_vertical": (
        "vertical label clamping has a separately named horizontal primitive"
    ),
    "phases/_common.py::_bbox_cols_overlap": (
        "column overlap is an explicitly X-scoped interval primitive"
    ),
    "phases/_common.py::_canvas_width": ("canvas width is intrinsically an X extent"),
    "phases/_common.py::_expand_bbox_for_y": (
        "the name and contract explicitly expand one Y extent"
    ),
    "route_reservations.py::_section_x_overlaps": (
        "X overlap has a paired Y overlap primitive"
    ),
    "route_reservations.py::_section_y_overlaps": (
        "Y overlap has a paired X overlap primitive"
    ),
    "route_reservations.py::_row_end": ("row end has a paired column end primitive"),
    "route_reservations.py::_column_end": ("column end has a paired row end primitive"),
    "routing/inter_section_handlers.py::_section_right_edge": (
        "right edge has a paired left edge primitive"
    ),
    "routing/inter_section_handlers.py::_section_left_edge": (
        "left edge has a paired right edge primitive"
    ),
    "section_placement.py::_effective_section_width": (
        "the helper computes an explicitly horizontal extent"
    ),
    "section_placement.py::_compute_row_heights": (
        "the helper computes explicitly vertical row extents"
    ),
    "section_placement.py::_rows_overlap": (
        "row overlap has a paired column overlap primitive"
    ),
    "section_placement.py::_cols_overlap": (
        "column overlap has a paired row overlap primitive"
    ),
    "section_placement.py::_enforce_min_column_gaps": (
        "column-gap enforcement has a paired row-gap pass"
    ),
    "section_placement.py::_enforce_min_column_gaps.<locals>.<lambda>#1": (
        "the callback computes the right edge for the X-scoped column-gap pass"
    ),
    "section_placement.py::_enforce_min_column_gaps.<locals>.<lambda>#2": (
        "the callback computes the left edge for the X-scoped column-gap pass"
    ),
    "section_placement.py::_enforce_min_row_gaps": (
        "row-gap enforcement has a paired column-gap pass"
    ),
    "section_placement.py::reenforce_column_gaps.<locals>.<lambda>#1": (
        "the callback computes the right edge for the X-scoped column-gap pass"
    ),
}


def _unexempted_sites():
    return {
        key: site
        for key, site in single_axis_sites(_LAYOUT_DIR).items()
        if key not in _EXEMPTIONS
    }


def _breakdown(axis: str) -> str:
    return "\n  ".join(
        f"{key}:{site.line} reads {sorted(site.fields)}"
        for key, site in _unexempted_sites().items()
        if site.axis == axis
    )


def test_no_new_single_axis_geometry_functions() -> None:
    sites = _unexempted_sites()
    for axis, baseline in _BASELINE.items():
        total = sum(site.axis == axis for site in sites.values())
        assert total <= baseline, (
            f"{axis.upper()}-only geometry function count rose to {total} "
            f"(baseline {baseline}). A pass that reads one geometry-field family "
            "needs an orientation-neutral implementation or a reasoned exemption; "
            "otherwise one rotation can silently lack a counterpart.\n  "
            f"{_breakdown(axis)}"
        )


def test_single_axis_baseline_is_current() -> None:
    sites = _unexempted_sites()
    current = {
        axis: sum(site.axis == axis for site in sites.values()) for axis in _BASELINE
    }
    assert current == _BASELINE, (
        f"single-axis baseline is {_BASELINE}, live counts are {current}; lower the "
        "baseline after removing a site, or add a reasoned exemption for a "
        "legitimate axis-scoped primitive"
    )


def test_single_axis_exemptions_are_live_and_reasoned() -> None:
    sites = single_axis_sites(_LAYOUT_DIR)
    stale = sorted(set(_EXEMPTIONS) - set(sites))
    assert not stale, f"single-axis exemptions no longer detected: {stale}"
    unreasoned = sorted(
        key for key, reason in _EXEMPTIONS.items() if not reason.strip()
    )
    assert not unreasoned, f"single-axis exemptions without reasons: {unreasoned}"


def test_reads_two_y_fields_without_x_fields() -> None:
    source = """
def row_end(section):
    return section.bbox_y + section.bbox_h
"""

    assert single_axis_sites_from_source(source) == {
        "row_end": ("y", frozenset({"bbox_y", "bbox_h"}))
    }


def test_reads_two_x_fields_without_y_fields() -> None:
    source = """
def column_end(section):
    return section.grid_col + section.grid_col_span
"""

    assert single_axis_sites_from_source(source) == {
        "column_end": ("x", frozenset({"grid_col", "grid_col_span"}))
    }


def test_ignores_functions_that_read_both_axes() -> None:
    source = """
def box_area(section):
    return section.bbox_w * section.bbox_h
"""

    assert single_axis_sites_from_source(source) == {}


def test_ignores_one_field_repeated() -> None:
    source = """
def repeated(section):
    return section.bbox_y + section.bbox_y
"""

    assert single_axis_sites_from_source(source) == {}


def test_nested_functions_are_classified_independently() -> None:
    source = """
def outer(section):
    def inner(item):
        return item.bbox_x + item.bbox_w
    return section.bbox_y + section.bbox_h
"""

    assert single_axis_sites_from_source(source) == {
        "outer": ("y", frozenset({"bbox_y", "bbox_h"})),
        "outer.<locals>.inner": ("x", frozenset({"bbox_x", "bbox_w"})),
    }


def test_lambdas_are_classified_with_stable_per_scope_keys() -> None:
    source = """
def column_edges():
    left = lambda section: section.bbox_x + section.offset_x
    right = lambda section: section.bbox_x + section.bbox_w
    return left, right
"""

    assert single_axis_sites_from_source(source) == {
        "column_edges": (
            "x",
            frozenset({"bbox_x", "bbox_w", "offset_x"}),
        ),
        "column_edges.<locals>.<lambda>#1": (
            "x",
            frozenset({"bbox_x", "offset_x"}),
        ),
        "column_edges.<locals>.<lambda>#2": (
            "x",
            frozenset({"bbox_x", "bbox_w"}),
        ),
    }
