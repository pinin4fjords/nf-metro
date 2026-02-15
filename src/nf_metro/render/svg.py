"""SVG generation for metro maps using drawsvg."""

from __future__ import annotations

import drawsvg as draw

from nf_metro.layout.labels import LabelPlacement, place_labels
from nf_metro.layout.routing import RoutedPath, compute_station_offsets, route_edges
from nf_metro.parser.model import MetroGraph
from nf_metro.render.legend import compute_legend_dimensions, render_legend
from nf_metro.render.style import Theme


def render_svg(
    graph: MetroGraph,
    theme: Theme,
    width: int | None = None,
    height: int | None = None,
    padding: float = 60.0,
) -> str:
    """Render a metro map graph to an SVG string."""
    if not graph.stations:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    # Filter out port stations for dimension calculation
    visible_stations = [s for s in graph.stations.values() if not s.is_port]
    all_stations_for_bounds = visible_stations if visible_stations else list(graph.stations.values())

    max_x = max(s.x for s in all_stations_for_bounds)
    max_y = max(s.y for s in all_stations_for_bounds)

    # Also consider section bounding boxes
    for section in graph.sections.values():
        if section.bbox_w > 0:
            max_x = max(max_x, section.bbox_x + section.bbox_w)
            max_y = max(max_y, section.bbox_y + section.bbox_h)

    # Compute legend position based on legend_position directive
    legend_x = 0.0
    legend_y = 0.0
    legend_w, legend_h = compute_legend_dimensions(graph, theme)
    show_legend = graph.legend_position != "none" and legend_w > 0

    if show_legend:
        pos = graph.legend_position
        gap = 30.0
        inset = 10.0
        # Section content bounds (or station bounds if no sections)
        content_left = min((s.bbox_x for s in graph.sections.values() if s.bbox_w > 0), default=padding)
        content_right = max_x
        content_top = min((s.bbox_y for s in graph.sections.values() if s.bbox_w > 0), default=padding)
        content_bottom = max_y

        if pos == "bl":
            legend_x = content_left
            legend_y = content_bottom - legend_h
        elif pos == "br":
            legend_x = content_right - legend_w - inset
            legend_y = content_bottom - legend_h - inset
        elif pos == "tl":
            legend_x = content_left + inset
            legend_y = content_top + inset
        elif pos == "tr":
            legend_x = content_right - legend_w - inset
            legend_y = content_top + inset
        elif pos == "bottom":
            legend_x = content_left
            legend_y = content_bottom + gap
        elif pos == "right":
            legend_x = content_right + gap
            legend_y = content_top

        max_x = max(max_x, legend_x + legend_w)
        max_y = max(max_y, legend_y + legend_h)

    auto_width = max_x + padding * 2
    auto_height = max_y + padding * 2

    svg_width = width or int(auto_width)
    svg_height = height or int(auto_height)

    d = draw.Drawing(svg_width, svg_height)

    # Background
    d.append(draw.Rectangle(0, 0, svg_width, svg_height, fill=theme.background_color))

    # Title
    if graph.title:
        d.append(draw.Text(
            graph.title,
            theme.title_font_size,
            padding, 30,
            fill=theme.title_color,
            font_family=theme.label_font_family,
            font_weight="bold",
        ))

    # Sections
    if graph.sections:
        _render_first_class_sections(d, graph, theme)

    # Route edges (compute offsets first so TB routes can pre-apply them)
    station_offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=station_offsets)

    # Draw edges (lines) behind stations
    _render_edges(d, graph, routes, station_offsets, theme)

    # Draw stations (all circles, skip ports)
    _render_stations(d, graph, theme, station_offsets)

    # Draw labels (horizontal, skip ports)
    labels = place_labels(graph)
    _render_labels(d, labels, theme)

    # Legend
    if show_legend:
        render_legend(d, graph, theme, legend_x, legend_y)

    return d.as_svg()


def _render_first_class_sections(
    d: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
) -> None:
    """Render first-class sections using pre-computed bounding boxes."""
    for section in graph.sections.values():
        if section.bbox_w <= 0 or section.bbox_h <= 0:
            continue

        d.append(draw.Rectangle(
            section.bbox_x, section.bbox_y,
            section.bbox_w, section.bbox_h,
            rx=8, ry=8,
            fill=theme.section_fill,
            stroke=theme.section_stroke,
            stroke_width=1.0,
        ))

        # Numbered circle above the box, left-aligned
        circle_r = 9
        cx = section.bbox_x + circle_r
        cy = section.bbox_y - circle_r - 4

        d.append(draw.Circle(
            cx, cy, circle_r,
            fill=theme.station_stroke,
        ))
        d.append(draw.Text(
            str(section.number),
            9,
            cx, cy,
            fill=theme.station_fill,
            font_family=theme.label_font_family,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
        ))

        # Section name to the right of the circle
        d.append(draw.Text(
            section.name,
            theme.section_label_font_size,
            cx + circle_r + 5, cy,
            fill=theme.section_label_color,
            font_family=theme.label_font_family,
            dominant_baseline="central",
        ))


def _render_edges(
    d: draw.Drawing,
    graph: MetroGraph,
    routes: list[RoutedPath],
    station_offsets: dict[tuple[str, str], float],
    theme: Theme,
    curve_radius: float = 10.0,
) -> None:
    """Render metro line edges with smooth curves at direction changes."""
    for route in routes:
        line = graph.lines.get(route.line_id)
        color = line.color if line else "#888888"

        if route.offsets_applied:
            # TB section routes have offsets pre-applied in the routing code
            pts = list(route.points)
        else:
            src_off = station_offsets.get((route.edge.source, route.line_id), 0.0)
            tgt_off = station_offsets.get((route.edge.target, route.line_id), 0.0)

            orig_sy = route.points[0][1]
            orig_ty = route.points[-1][1]
            pts = []
            for i, (x, y) in enumerate(route.points):
                if i == 0:
                    pts.append((x, y + src_off))
                elif i == len(route.points) - 1:
                    pts.append((x, y + tgt_off))
                elif abs(y - orig_sy) <= abs(y - orig_ty):
                    pts.append((x, y + src_off))
                else:
                    pts.append((x, y + tgt_off))

        if len(pts) == 2:
            d.append(draw.Line(
                pts[0][0], pts[0][1],
                pts[1][0], pts[1][1],
                stroke=color,
                stroke_width=theme.line_width,
                stroke_linecap="round",
            ))
        elif len(pts) >= 3:
            path = draw.Path(
                stroke=color,
                stroke_width=theme.line_width,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            )
            path.M(*pts[0])

            for i in range(1, len(pts) - 1):
                prev = pts[i - 1]
                curr = pts[i]
                nxt = pts[i + 1]

                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                len1 = (dx1**2 + dy1**2) ** 0.5

                dx2 = nxt[0] - curr[0]
                dy2 = nxt[1] - curr[1]
                len2 = (dx2**2 + dy2**2) ** 0.5

                # Only halve segment length when the adjacent point also
                # has a curve; endpoints (first/last points) never do.
                max_len1 = len1 / 2 if i > 1 else len1
                max_len2 = len2 / 2 if i < len(pts) - 2 else len2
                corner_idx = i - 1
                if route.curve_radii and corner_idx < len(route.curve_radii):
                    effective_r = route.curve_radii[corner_idx]
                else:
                    effective_r = curve_radius
                r = min(effective_r, max_len1, max_len2)

                if len1 > 0 and len2 > 0:
                    before_x = curr[0] - (dx1 / len1) * r
                    before_y = curr[1] - (dy1 / len1) * r
                    after_x = curr[0] + (dx2 / len2) * r
                    after_y = curr[1] + (dy2 / len2) * r

                    path.L(before_x, before_y)
                    path.Q(curr[0], curr[1], after_x, after_y)
                else:
                    path.L(*curr)

            path.L(*pts[-1])
            d.append(path)


def _render_stations(
    d: draw.Drawing,
    graph: MetroGraph,
    theme: Theme,
    station_offsets: dict[tuple[str, str], float] | None = None,
) -> None:
    """Render stations as pill shapes.

    Normal stations get vertical pills (tall, narrow). Stations in a TB
    section's vertical part (layer > 0) get horizontal pills (wide, short)
    since the lines run vertically through them.

    Skips port stations (is_port=True).
    """
    for station in graph.stations.values():
        if station.is_port:
            continue

        r = theme.station_radius

        # Determine if this is a TB vertical station (rotated pill)
        is_tb_vert = False
        if station.section_id:
            sec = graph.sections.get(station.section_id)
            if sec and sec.direction == "TB" and station.layer > 0:
                is_tb_vert = True

        if station_offsets:
            line_offsets = [
                station_offsets.get((station.id, lid), 0.0)
                for lid in graph.station_lines(station.id)
            ]
            if line_offsets:
                min_off = min(line_offsets)
                max_off = max(line_offsets)
            else:
                min_off = max_off = 0.0
        else:
            min_off = max_off = 0.0

        span = max_off - min_off

        if is_tb_vert:
            # Horizontal pill: lines spread along X axis
            w = span + r * 2
            h = r * 2
            cx = station.x + (min_off + max_off) / 2
            d.append(draw.Rectangle(
                cx - w / 2, station.y - h / 2,
                w, h,
                rx=r, ry=r,
                fill=theme.station_fill,
                stroke=theme.station_stroke,
                stroke_width=theme.station_stroke_width,
            ))
        else:
            # Vertical pill: lines spread along Y axis
            w = r * 2
            h = span + r * 2
            cy = station.y + (min_off + max_off) / 2
            d.append(draw.Rectangle(
                station.x - w / 2, cy - h / 2,
                w, h,
                rx=r, ry=r,
                fill=theme.station_fill,
                stroke=theme.station_stroke,
                stroke_width=theme.station_stroke_width,
            ))


def _render_labels(
    d: draw.Drawing,
    labels: list[LabelPlacement],
    theme: Theme,
) -> None:
    """Render station name labels."""
    for label in labels:
        if label.dominant_baseline:
            # Custom placement (e.g. TB vertical stations: right-side labels)
            d.append(draw.Text(
                label.text,
                theme.label_font_size,
                label.x, label.y,
                fill=theme.label_color,
                font_family=theme.label_font_family,
                text_anchor=label.text_anchor,
                dominant_baseline=label.dominant_baseline,
            ))
        else:
            baseline = "auto" if label.above else "hanging"
            d.append(draw.Text(
                label.text,
                theme.label_font_size,
                label.x, label.y,
                fill=theme.label_color,
                font_family=theme.label_font_family,
                text_anchor="middle",
                dominant_baseline=baseline,
            ))
