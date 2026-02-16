"""Legend generation for metro map SVGs."""

from __future__ import annotations

import drawsvg as draw

from nf_metro.parser.model import MetroGraph
from nf_metro.render.style import Theme


def compute_legend_dimensions(graph: MetroGraph, theme: Theme) -> tuple[float, float]:
    """Compute the width and height of the legend without rendering it.

    Returns (width, height). Returns (0, 0) if there are no lines.
    """
    if not graph.lines:
        return (0.0, 0.0)

    line_height = 24.0
    padding = 12.0
    swatch_width = 24.0
    text_offset = swatch_width + 12.0

    max_name_len = max(len(ml.display_name) for ml in graph.lines.values())
    width = text_offset + max_name_len * 7.5 + padding * 2
    height = padding * 2 + len(graph.lines) * line_height
    return (width, height)


def render_legend(
    drawing: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
    x: float,
    y: float,
) -> None:
    """Render a legend showing all metro lines and their colors.

    Positioned at (x, y), drawing downward.
    """
    if not graph.lines:
        return

    line_height = 24.0
    padding = 12.0
    swatch_width = 24.0
    text_offset = swatch_width + 12.0

    legend_width, legend_height = compute_legend_dimensions(graph, theme)

    # Background
    drawing.append(
        draw.Rectangle(
            x,
            y,
            legend_width,
            legend_height,
            rx=6,
            ry=6,
            fill=theme.legend_background,
        )
    )

    # Line entries
    for i, metro_line in enumerate(graph.lines.values()):
        entry_y = y + padding + i * line_height + line_height / 2

        # Color swatch (line segment)
        drawing.append(
            draw.Line(
                x + padding,
                entry_y,
                x + padding + swatch_width,
                entry_y,
                stroke=metro_line.color,
                stroke_width=theme.line_width,
                stroke_linecap="round",
            )
        )

        # Label
        drawing.append(
            draw.Text(
                metro_line.display_name,
                theme.legend_font_size,
                x + padding + text_offset,
                entry_y,
                fill=theme.legend_text_color,
                font_family=theme.label_font_family,
                dominant_baseline="central",
            )
        )
