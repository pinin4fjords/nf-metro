"""Whole-graph canvas translation into view and section renumbering by reading order."""

from __future__ import annotations

from nf_metro.layout.constants import (
    TITLE_BAND_CLEARANCE,
    TITLE_BAND_OVERLAP_FLOOR,
)
from nf_metro.layout.phases.bbox import (
    _min_drawn_section_bbox_top,
    _min_section_bbox_top,
)
from nf_metro.parser.model import MetroGraph


def _renumber_sections_by_grid(graph: MetroGraph) -> None:
    """Renumber sections by visual reading order.

    Rows are numbered top-to-bottom.  Within each row, numbering follows
    the row's horizontal flow, so ordinary rows read left-to-right and
    folded return rows read right-to-left.  Authored numbers are reserved;
    automatic sections receive the lowest unused positive numbers.
    """
    section_rank = {sid: rank for rank, sid in enumerate(graph.sections)}
    section_edges = graph.section_dag.section_edges if graph.section_dag else set()
    rows: dict[int, list[str]] = {}
    for sid, section in graph.sections.items():
        rows.setdefault(section.grid_row, []).append(sid)

    ordered_ids: list[str] = []
    for row in sorted(rows):
        row_ids = rows[row]
        row_set = set(row_ids)
        flow_score = sum(
            graph.sections[tgt].grid_col - graph.sections[src].grid_col
            for src, tgt in section_edges
            if src in row_set
            and tgt in row_set
            and graph.sections[src].grid_col != graph.sections[tgt].grid_col
        )
        if flow_score == 0:
            flow_score = sum(
                1 if graph.sections[sid].direction == "LR" else -1
                for sid in row_ids
                if graph.sections[sid].direction in ("LR", "RL")
            )

        right_to_left = flow_score < 0
        ordered_ids.extend(
            sorted(
                row_ids,
                key=lambda sid: (
                    -graph.sections[sid].grid_col
                    if right_to_left
                    else graph.sections[sid].grid_col,
                    section_rank[sid],
                ),
            )
        )

    reserved = {
        section.number_override
        for section in graph.sections.values()
        if section.number_override is not None
    }
    next_number = 1
    for sid in ordered_ids:
        section = graph.sections[sid]
        if section.number_override is not None:
            section.number = section.number_override
            continue
        while next_number in reserved:
            next_number += 1
        section.number = next_number
        next_number += 1


def _translate_graph_y(graph: MetroGraph, shift: float) -> None:
    """Shift every station, section bbox, and port down by ``shift``."""
    for st in graph.stations.values():
        st.y += shift
    for section in graph.sections.values():
        section.bbox_y += shift
    for port in graph.ports.values():
        port.y += shift


def _canvas_top_shortfall(graph: MetroGraph, section_y_padding: float) -> float:
    """Downward shift needed so the graph clears both top floors (0 if none).

    Containment: every section must sit ``section_y_padding`` below the canvas
    top.  Title: when the map is titled *and* ``graph.reserve_title_band`` (set
    by :func:`~nf_metro.api.prepare_graph` -- false for ``--bare`` or a logo
    folded into the legend, cases where render draws nothing up there), a
    *drawn* section whose header badge overlaps the title band (box top above
    ``TITLE_BAND_OVERLAP_FLOOR``) is lifted clear to ``TITLE_BAND_CLEARANCE``
    -- a map already clearing the title, and implicit holders (which draw no
    badge), are left untouched.
    """
    min_all = _min_section_bbox_top(graph, section_y_padding)
    shortfall = max(0.0, section_y_padding - min_all)
    if graph.title and graph.reserve_title_band:
        min_drawn = _min_drawn_section_bbox_top(graph)
        if min_drawn is not None and min_drawn < TITLE_BAND_OVERLAP_FLOOR:
            shortfall = max(shortfall, TITLE_BAND_CLEARANCE - min_drawn)
    return shortfall


def _canvas_top_preserved(
    graph: MetroGraph, section_y_padding: float, shift: float
) -> bool:
    """True if ``shift`` keeps every section within its top floor.

    The containment floor (``section_y_padding``) bounds all sections; the
    no-overlap floor (``TITLE_BAND_OVERLAP_FLOOR``) bounds drawn sections on a
    titled map whose title band will actually be drawn (see
    ``graph.reserve_title_band``).  Lets the grid snap reject a candidate that
    would lift the top above a floor.
    """
    min_all = _min_section_bbox_top(graph, section_y_padding)
    if min_all + shift < section_y_padding - 1e-6:
        return False
    if graph.title and graph.reserve_title_band:
        min_drawn = _min_drawn_section_bbox_top(graph)
        if (
            min_drawn is not None
            and min_drawn + shift < TITLE_BAND_OVERLAP_FLOOR - 1e-6
        ):
            return False
    return True


def _shift_graph_into_canvas(graph: MetroGraph, section_y_padding: float) -> None:
    """Shift the whole graph down if the topmost section is above the canvas.

    Lifts the graph by ``_canvas_top_shortfall`` so it clears the canvas-top
    margin and, on a titled map, the title band.  No-op when it already does.
    """
    shortfall = _canvas_top_shortfall(graph, section_y_padding)
    if shortfall > 0:
        _translate_graph_y(graph, shortfall)
