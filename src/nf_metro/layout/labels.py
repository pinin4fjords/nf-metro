"""Label placement for station names.

Uses horizontal labels (like the reference nf-core metro maps) with
above/below alternation and collision avoidance.
"""

from __future__ import annotations

__all__ = ["LabelPlacement", "place_labels"]

from dataclasses import dataclass

from nf_metro.layout.constants import (
    CHAR_WIDTH,
    COLLISION_MULTIPLIER,
    FONT_HEIGHT,
    LABEL_BBOX_MARGIN,
    LABEL_MARGIN,
    LABEL_OFFSET,
    TB_LABEL_H_SPACING,
    TB_LINE_Y_OFFSET,
    TB_PILL_EDGE_OFFSET,
)
from nf_metro.parser.model import MetroGraph


@dataclass
class LabelPlacement:
    """Placement information for a station label."""

    station_id: str
    text: str
    x: float
    y: float
    above: bool
    angle: float = 0.0  # Horizontal by default
    text_anchor: str = "middle"
    dominant_baseline: str = ""  # Empty means use above/below logic


def _label_bbox(
    placement: LabelPlacement,
    char_width: float = CHAR_WIDTH,
    font_height: float = FONT_HEIGHT,
) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) bounding box for a label."""
    text_width = len(placement.text) * char_width
    half_w = text_width / 2

    if placement.above:
        return (
            placement.x - half_w,
            placement.y - font_height,
            placement.x + half_w,
            placement.y,
        )
    else:
        return (
            placement.x - half_w,
            placement.y,
            placement.x + half_w,
            placement.y + font_height,
        )


def _boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    margin: float = LABEL_MARGIN,
) -> bool:
    """Check if two bounding boxes overlap."""
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def place_labels(
    graph: MetroGraph,
    label_offset: float = LABEL_OFFSET,
    station_offsets: dict[tuple[str, str], float] | None = None,
) -> list[LabelPlacement]:
    """Place horizontal labels alternating above/below stations.

    Strategy:
    1. Default: alternate above/below based on layer index.
    2. If it collides with an existing label, try the other side.
    3. If still colliding, push further away.
    """
    sorted_stations = sorted(
        (
            s
            for s in graph.stations.values()
            if not s.is_port and not s.is_hidden and s.label.strip()
        ),
        key=lambda s: (s.layer, s.track),
    )

    # Pre-compute per-section Y extremes for LR/RL sections so edge
    # stations prefer outward-facing labels, centering visual content.
    section_y_range: dict[str, tuple[float, float]] = {}
    for s in sorted_stations:
        if not s.section_id:
            continue
        sec = graph.sections.get(s.section_id)
        if not sec or sec.direction not in ("LR", "RL"):
            continue
        if s.section_id not in section_y_range:
            section_y_range[s.section_id] = (s.y, s.y)
        else:
            lo, hi = section_y_range[s.section_id]
            section_y_range[s.section_id] = (min(lo, s.y), max(hi, s.y))

    placements: list[LabelPlacement] = []

    for i, station in enumerate(sorted_stations):
        # Compute the vertical extent of the station pill so labels
        # are offset from the pill edge, not from station.y.
        if station_offsets:
            line_offs = [
                station_offsets.get((station.id, lid), 0.0)
                for lid in graph.station_lines(station.id)
            ]
            min_off = min(line_offs) if line_offs else 0.0
            max_off = max(line_offs) if line_offs else 0.0
        else:
            min_off = max_off = 0.0

        # Check if this is a TB section station (horizontal pill)
        is_tb_vert = False
        if station.section_id:
            sec = graph.sections.get(station.section_id)
            if sec and sec.direction == "TB":
                is_tb_vert = True

        if is_tb_vert:
            # Place label to the left of the horizontal pill
            n_lines = len(graph.station_lines(station.id))
            offset_span = (n_lines - 1) * TB_LINE_Y_OFFSET
            pill_left = station.x - offset_span / 2 - TB_PILL_EDGE_OFFSET
            candidate = LabelPlacement(
                station_id=station.id,
                text=station.label,
                x=pill_left - TB_LABEL_H_SPACING,
                y=station.y,
                above=True,
                text_anchor="end",
                dominant_baseline="central",
            )
            placements.append(candidate)
            continue

        # Alternate by layer (column): even layers below, odd layers above
        start_above = station.layer % 2 == 1

        # For edge stations in LR/RL sections, prefer labels extending
        # outward (away from center) to keep visual content centered
        # within the section bbox.
        if station.section_id and station.section_id in section_y_range:
            y_lo, y_hi = section_y_range[station.section_id]
            if y_lo < y_hi:
                if station.y == y_lo:
                    start_above = True
                elif station.y == y_hi:
                    start_above = False

        candidate = _try_place(
            station, label_offset, start_above, placements, min_off, max_off
        )

        if _has_collision(candidate, placements):
            # Try the other side
            candidate = _try_place(
                station, label_offset, not start_above, placements, min_off, max_off
            )

            if _has_collision(candidate, placements):
                # Push further in the non-default direction
                direction = -1 if not start_above else 1
                if direction < 0:
                    y = station.y + min_off - label_offset * COLLISION_MULTIPLIER
                else:
                    y = station.y + max_off + label_offset * COLLISION_MULTIPLIER
                candidate = LabelPlacement(
                    station_id=station.id,
                    text=station.label,
                    x=station.x,
                    y=y,
                    above=(direction < 0),
                )

        # Clamp labels so they stay within section bbox
        if station.section_id:
            sec = graph.sections.get(station.section_id)
            if sec and sec.bbox_w > 0:
                text_half_w = len(candidate.text) * CHAR_WIDTH / 2
                margin = LABEL_BBOX_MARGIN
                # Horizontal clamping
                min_x = sec.bbox_x + text_half_w + margin
                max_x = sec.bbox_x + sec.bbox_w - text_half_w - margin
                candidate.x = max(min_x, min(candidate.x, max_x))
                # Vertical clamping (with flip/expand on overlap)
                candidate = _clamp_label_vertical(
                    candidate,
                    sec,
                    station,
                    label_offset,
                    min_off,
                    max_off,
                    margin,
                    placements,
                )

        placements.append(candidate)

    return placements


def _clamp_label_vertical(
    candidate: LabelPlacement,
    sec,
    station,
    label_offset: float,
    min_off: float,
    max_off: float,
    margin: float,
    existing: list[LabelPlacement] | None = None,
) -> LabelPlacement:
    """Clamp label vertically within section bbox.

    If clamping would push the label into the station pill, flip it to the
    opposite side (provided the flipped position doesn't collide with an
    existing label).  If both sides would overlap, expand the section bbox
    so the label fits at its ideal position.
    """
    pill_top = station.y + min_off
    pill_bottom = station.y + max_off
    sec_top = sec.bbox_y
    sec_bottom = sec.bbox_y + sec.bbox_h

    if candidate.above:
        # Label text occupies [candidate.y - FONT_HEIGHT, candidate.y].
        min_y = sec_top + FONT_HEIGHT + margin
        if candidate.y >= min_y:
            return candidate  # fits without clamping

        # Clamping needed - would the clamped position overlap the pill?
        if min_y <= pill_top - label_offset:
            # Still enough gap after clamping
            candidate.y = min_y
            return candidate

        # Clamped position too close to pill - try flipping to below
        below_y = pill_bottom + label_offset
        max_y = sec_bottom - FONT_HEIGHT - margin
        if below_y <= max_y:
            flipped = LabelPlacement(
                station_id=candidate.station_id,
                text=candidate.text,
                x=candidate.x,
                y=below_y,
                above=False,
            )
            if existing is None or not _has_collision(flipped, existing):
                candidate.y = below_y
                candidate.above = False
                return candidate

        # Neither side fits (or flip collides) - expand bbox upward
        expand = min_y - candidate.y + margin
        sec.bbox_y -= expand
        sec.bbox_h += expand
        return candidate

    else:
        # Label text occupies [candidate.y, candidate.y + FONT_HEIGHT].
        max_y = sec_bottom - FONT_HEIGHT - margin
        if candidate.y <= max_y:
            return candidate  # fits without clamping

        # Clamping needed - would the clamped position overlap the pill?
        if max_y >= pill_bottom + label_offset:
            # Still enough gap after clamping
            candidate.y = max_y
            return candidate

        # Clamped position too close to pill - try flipping to above
        above_y = pill_top - label_offset
        min_y = sec_top + FONT_HEIGHT + margin
        if above_y >= min_y:
            flipped = LabelPlacement(
                station_id=candidate.station_id,
                text=candidate.text,
                x=candidate.x,
                y=above_y,
                above=True,
            )
            if existing is None or not _has_collision(flipped, existing):
                candidate.y = above_y
                candidate.above = True
                return candidate

        # Neither side fits (or flip collides) - expand bbox downward
        expand = candidate.y - max_y + margin
        sec.bbox_h += expand
        return candidate


def _try_place(
    station,
    label_offset: float,
    above: bool,
    existing: list[LabelPlacement],
    min_off: float = 0.0,
    max_off: float = 0.0,
) -> LabelPlacement:
    """Create a label placement above or below a station.

    Offsets are measured from the pill edge: above labels use min_off
    (top of the pill) and below labels use max_off (bottom of the pill).
    """
    if above:
        return LabelPlacement(
            station_id=station.id,
            text=station.label,
            x=station.x,
            y=station.y + min_off - label_offset,
            above=True,
        )
    else:
        return LabelPlacement(
            station_id=station.id,
            text=station.label,
            x=station.x,
            y=station.y + max_off + label_offset,
            above=False,
        )


def _has_collision(
    candidate: LabelPlacement,
    existing: list[LabelPlacement],
) -> bool:
    """Check if a candidate label collides with any existing placement."""
    cbox = _label_bbox(candidate)
    for placed in existing:
        if _boxes_overlap(cbox, _label_bbox(placed)):
            return True
    return False
